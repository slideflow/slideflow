import imgui
import numpy as np
import slideflow as sf
import slideflow.mil
import threading

from tkinter.filedialog import askdirectory
from os.path import join, exists, dirname, abspath
from typing import Dict, Optional
from slideflow.model.extractors import rebuild_extractor
from slideflow.mil._params import ModelConfigCLAM, TrainerConfigCLAM
from slideflow.mil.eval import _predict_clam, _predict_mil

from ._utils import Widget
from ..gui import imgui_utils

# -----------------------------------------------------------------------------

def _is_mil_model(path: str) -> bool:
    """Check if a given path is a valid MIL model."""
    return (exists(join(path, 'mil_params.json'))
            or (path.endswith('.pth')
                and dirname(path).endswith('models'))
                and exists(join(dirname(path), '../mil_params.json')))


def _get_mil_params(path: str) -> Dict:
    return sf.util.load_json(join(path, 'mil_params.json'))


def _draw_imgui_info(rows, viz):
    for y, cols in enumerate(rows):
        for x, col in enumerate(cols):
            col = str(col)
            if x != 0:
                imgui.same_line(viz.font_size * (6 + (x - 1) * 6))
            if x == 0:
                imgui.text_colored(col, *viz.theme.dim)
            else:
                with imgui_utils.clipped_with_tooltip(col, 22):
                    imgui.text(imgui_utils.ellipsis_clip(col, 22))

class _AttentionHeatmapWrapper:

    def __init__(self, attention: np.ndarray, slide: "sf.WSI"):
        self.attention = attention
        self.slide = slide

    def save_npz(self, path: Optional[str] = None) -> str:
        """Save heatmap predictions and uncertainty in .npz format.

        Saves heatmap predictions to ``'predictions'`` in the .npz file. If uncertainty
        was calculated, this is saved to ``'uncertainty'``. A Heatmap instance can
        load a saved .npz file with :meth:`slideflow.Heatmap.load()`.

        Args:
            path (str, optional): Destination filename for .npz file. Defaults
                to {slidename}.npz

        Returns:
            str: Path to .npz file.
        """
        if path is None:
            path = f'{self.slide.name}.npz'
        np.savez(path, predictions=self.attention)
        return path

    def load(self):
        raise NotImplementedError("Not yet implemented.")

# -----------------------------------------------------------------------------

class MILWidget(Widget):

    tag = 'mil'
    description = 'Multiple-Instance Learning'
    icon = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_mil.png')
    icon_highlighted = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_mil_highlighted.png')

    def __init__(self, viz):
        self.viz = viz
        self._clicking      = None
        self._initialize_variables()

    # --- Hooks, triggers, and internal functions -----------------------------

    def _initialize_variables(self):
        # Extractor, model, and config.
        self.model = None
        self.mil_config = None
        self.extractor = None
        self.mil_params = None
        self.extractor_params = None
        self.normalizer = None
        self.calculate_attention = True

        # Predictions and attention.
        self.predictions = None
        self.attention = None

        # Internals.
        self._show_mil_params = None
        self._rendering_message = "Generating whole-slide prediction..."
        self._generating = False
        self._triggered = False
        self._thread = None
        self._toast = None
        self._show_popup = False

    def _reload_wsi(self):
        """Reload a slide."""
        viz = self.viz
        if viz.wsi:
            viz.tile_px = self.extractor_params['tile_px']
            viz.tile_um = self.extractor_params['tile_um']
            viz.slide_widget.load(viz.wsi.path, mpp=viz.slide_widget.manual_mpp)

    def _refresh_generating_prediction(self):
        """Refresh render of asynchronous MIL prediction / attention heatmap."""
        if self._thread is not None and not self._thread.is_alive():
            self._generating = False
            self._triggered = False
            self._thread = None
            self.viz.clear_message(self._rendering_message)
            if self._toast is not None:
                self._toast.done()
                self._toast = None
            self.viz.create_toast("Prediction complete.", icon='success')

    def _on_model_load(self):
        """Trigger for when the user loads a tile-based model."""
        self.close()

    def drag_and_drop_hook(self, path: str) -> bool:
        """Drag-and-drop hook for loading an MIL model."""
        if _is_mil_model(path):
            return self.load(path)
        return False

    def open_menu_options(self) -> None:
        """Show a 'Load MIL Model' option in the File menu."""
        if imgui.menu_item('Load MIL Model...')[1]:
            self.ask_load_model()

    # --- Public API ----------------------------------------------------------

    def close(self):
        """Close the loaded MIL model."""
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
        del self.model
        del self.extractor
        del self.normalizer
        self.viz.heatmap_widget.reset()
        self._initialize_variables()

    def ask_load_model(self) -> None:
        """Prompt the user to open an MIL model."""
        mil_path = askdirectory(title="Load MIL Model (directory)...")
        if mil_path:
            self.load(mil_path)

    def load(self, path: str, allow_errors: bool = True) -> bool:
        try:
            self.close()
            self.extractor, self.normalizer = rebuild_extractor(path)
            self.mil_params = _get_mil_params(path)
            self.extractor_params = self.mil_params['bags_extractor']
            self._reload_wsi()
            self.model, self.mil_config = sf.mil.utils.load_model_weights(path)
            self.viz.close_model(True)  # Close a tile-based model, if one is loaded
            self.viz.tile_um = self.extractor_params['tile_um']
            self.viz.tile_px = self.extractor_params['tile_px']
            self.viz.create_toast('MIL model loaded', icon='success')
        except Exception as e:
            if allow_errors:
                self.viz.create_toast('Error loading MIL model', icon='error')
                sf.log.error(e)
                return False
            raise e
        return True

    def _predict_slide(self):
        viz = self.viz

        self._generating = True
        self._triggered = True

        # Generate features with the loaded extractor.
        masked_bags = self.extractor(
            viz.wsi,
            normalizer=self.normalizer,
            **viz.slide_widget.get_tile_filter_params(),
        )
        original_shape = masked_bags.shape
        masked_bags = masked_bags.reshape((-1, masked_bags.shape[-1]))
        mask = masked_bags.mask.any(axis=1)
        valid_indices = np.where(~mask)
        bags = masked_bags[valid_indices]
        bags = np.expand_dims(bags, axis=0).astype(np.float32)

        sf.log.info("Generated feature bags for {} tiles".format(bags.shape[1]))

        # Generate predictions.
        if (isinstance(self.mil_config, TrainerConfigCLAM)
        or isinstance(self.mil_config.model_config, ModelConfigCLAM)):
            self.predictions, self.attention = _predict_clam(
                self.model,
                bags,
                attention=self.calculate_attention
            )
        else:
            self.predictions, self.attention = _predict_mil(
                self.model,
                bags,
                attention=self.calculate_attention,
                use_lens=self.mil_config.model_config.use_lens
            )
        if self.attention:
            self.attention = self.attention[0]
        else:
            self.attention = None

        # Create a heatmap from the attention values
        if self.attention is not None:

            # Create a fully masked array of shape (X, Y)
            att_heatmap = np.ma.masked_all(masked_bags.shape[0], dtype=self.attention.dtype)

            # Unmask and fill the transformed data into the corresponding positions
            att_heatmap[valid_indices] = self.attention
            att_heatmap = np.reshape(att_heatmap, original_shape[0:2])

            # Normalize the heatmap
            att_heatmap = (att_heatmap - att_heatmap.min()) / (att_heatmap.max() - att_heatmap.min())

            # Render the heatmap
            self.render_attention_heatmap(att_heatmap)

    def predict_slide(self):
        """Initiate a whole-slide prediction."""
        if not self.verify_tile_size():
            return
        self.viz.set_message(self._rendering_message)
        self._toast = self.viz.create_toast(
            title="Generating prediction",
            sticky=True,
            spinner=True,
            icon='info'
        )
        self._thread = threading.Thread(target=self._predict_slide)
        self._thread.start()

    def verify_tile_size(self) -> bool:
        """Verify that the current slide matches the MIL model's tile size."""
        viz = self.viz
        mil_tile_um = self.extractor_params['tile_um']
        mil_tile_px = self.extractor_params['tile_px']
        if viz.wsi.tile_px != mil_tile_px or viz.wsi.tile_um != mil_tile_um:
            viz.create_toast(
                "MIL model tile size (tile_px={}, tile_um={}) does not match "
                "the currently loaded slide (tile_px={}, tile_um={}).".format(
                    mil_tile_px, mil_tile_um, viz.wsi.tile_px, viz.wsi.tile_um
                ),
                icon='error'
            )
            return False
        return True

    def render_attention_heatmap(self, array):
        self.viz.heatmap = _AttentionHeatmapWrapper(array, self.viz.wsi)
        self.viz.heatmap_widget.predictions = array[:, :, np.newaxis]
        self.viz.heatmap_widget.render_heatmap()

    def draw_extractor_info(self):
        """Draw a description of the extractor information."""

        viz = self.viz
        if self.extractor_params is None:
            imgui.text("No extractor loaded.")
            return
        c = self.extractor_params

        if 'normalizer' in c and c['normalizer']:
            normalizer = c['normalizer']['method']
        else:
            normalizer = "-"

        rows = [
            ['Extractor',         c['extractor']['class'].split('.')[-1]],
            ['Extractor Args',    c['extractor']['kwargs']],
            ['Normalizer',      normalizer],
            ['Num features',    c['num_features']],
            ['Tile size (px)',  c['tile_px']],
            ['Tile size (um)',  c['tile_um']],
        ]
        _draw_imgui_info(rows, viz)
        imgui_utils.vertical_break()

    def draw_mil_info(self):
        """Draw a description of the MIL model."""

        viz = self.viz
        if self.mil_params is None:
            imgui.text("No MIL model loaded.")
            return
        c = self.mil_params

        rows = [
            ['Outcomes',      c['outcomes']],
            ['Input size',    c['input_shape']],
            ['Output size',   c['output_shape']],
            ['Trainer',       c['trainer']],
        ]
        _draw_imgui_info(rows, viz)

        # MIL model params button and popup.
        with imgui_utils.grayed_out('params' not in c):
            imgui.same_line(imgui.get_content_region_max()[0] - viz.font_size - viz.spacing * 2)
            if imgui.button("HP") and 'params' in c:
                self._show_mil_params = not self._show_mil_params

    def draw_mil_params_popup(self):
        """Draw popup showing MIL model hyperparameters."""

        viz = self.viz
        hp = self.mil_params['params']
        rows = list(zip(list(map(str, hp.keys())), list(map(str, hp.values()))))

        _, self._show_mil_params = imgui.begin("MIL parameters", closable=True, flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_SCROLLBAR)
        for y, cols in enumerate(rows):
            for x, col in enumerate(cols):
                if x != 0:
                    imgui.same_line(viz.font_size * 10)
                if x == 0:
                    imgui.text_colored(col, *viz.theme.dim)
                else:
                    imgui.text(col)
        imgui.end()

    def draw_prediction(self):
        """Draw the final prediction."""
        if self.predictions is None:
            return
        assert len(self.predictions) == 1
        prediction = self.predictions[0]

        # Assemble outcome category labels.
        outcome_labels = [
            f"Outcome {i}" if 'outcome_labels' not in self.mil_params or str(i) not in self.mil_params['outcome_labels']
                           else self.mil_params['outcome_labels'][str(i)]
            for i in range(len(prediction))
        ]

        # Show prediction for each category.
        imgui.text(self.mil_params['outcomes'])
        imgui.separator()
        for i, pred_val in enumerate(prediction):
            imgui.text_colored(outcome_labels[i], *self.viz.theme.dim)
            imgui.same_line(self.viz.font_size * 12)
            imgui_utils.right_aligned_text(f"{pred_val:.3f}")
        imgui.separator()
        # Show final prediction based on which category has the highest probability.
        imgui.text("Final prediction")
        imgui.same_line(self.viz.font_size * 12)
        imgui_utils.right_aligned_text(f"{outcome_labels[np.argmax(prediction)]}")

    def draw_config_popup(self):
        viz = self.viz

        if self._show_popup:
            cx, cy = imgui.get_cursor_pos()
            imgui.set_next_window_position(viz.sidebar.full_width, cy)
            imgui.begin(
                '##mil_popup',
                flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
            )
            if imgui.menu_item('Load MIL model')[0]:
                self.ask_load_model()
            if imgui.menu_item('Close MIL model')[0]:
                self.close()

            # Hide menu if we click elsewhere
            if imgui.is_mouse_down(0) and not imgui.is_window_hovered():
                self._clicking = True
            if self._clicking and imgui.is_mouse_released(0):
                self._clicking = False
                self._show_popup = False

            imgui.end()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if self._generating:
            self._refresh_generating_prediction()

        if show:
            with viz.header_with_buttons("Multiple-Instance Learning"):
                imgui.same_line(imgui.get_content_region_max()[0] - viz.font_size*1.5)
                cx, cy = imgui.get_cursor_pos()
                imgui.set_cursor_position((cx, cy-int(viz.font_size*0.25)))
                if viz.sidebar.small_button('gear'):
                    self._clicking = False
                    self._show_popup = not self._show_popup
                self.draw_config_popup()

        if show and self.model:
            if viz.collapsing_header('Feature Extractor', default=True):
                self.draw_extractor_info()
            if viz.collapsing_header('MIL Model', default=True):
                self.draw_mil_info()
            if viz.collapsing_header('Prediction', default=True):
                self.draw_prediction()
                predict_enabled = (viz.wsi is not None
                                   and self.model is not None
                                   and not self._triggered)
                predict_text = "Predict Slide" if not self._triggered else f"Calculating{imgui_utils.spinner_text()}"
                if viz.sidebar.full_button(predict_text, enabled=predict_enabled):
                    self.predict_slide()
        elif show:
            imgui_utils.padded_text('No MIL model has been loaded.', vpad=[int(viz.font_size/2), int(viz.font_size)])
            if viz.sidebar.full_button("Load an MIL Model"):
                self.ask_load_model()

        if self._show_mil_params and self.mil_params:
            self.draw_mil_params_popup()
