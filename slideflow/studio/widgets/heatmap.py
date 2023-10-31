import numpy as np
import imgui
import threading
from typing import Union, Optional
from ..gui import imgui_utils

import slideflow as sf

#----------------------------------------------------------------------------

def _apply_cmap(img, cmap):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap)
    return (cmap(img) * 255).astype(np.uint8)


def process_grid(heatmap, grid):
    if heatmap.uq:
        predictions = grid[:, :, :-(heatmap.num_uncertainty)]
        uncertainty = grid[:, :, -(heatmap.num_uncertainty):]
        return predictions, uncertainty
    else:
        return grid, None


def convert_to_overlays(arr):
    if arr is None:
        return arr
    elif len(arr.shape) == 2:
        return [HeatmapOverlay(arr)]
    else:
        return [HeatmapOverlay(arr[:, :, i]) for i in range(arr.shape[2])]

#----------------------------------------------------------------------------

class HeatmapOverlay:
    def __init__(
        self,
        grid: np.ndarray,
        tile_um: Optional[int] = None,
        stride_div: Optional[int] = None,
        name: Optional[str] = None
    ) -> None:
        self.grid = grid
        self.tile_um = tile_um
        self.stride_div = stride_div
        self.name = name

    @property
    def overlay_kwargs(self):
        return dict(
            tile_um=self.tile_um,
            stride_div=self.stride_div,
            original=self.grid
        )


#----------------------------------------------------------------------------

class HeatmapWidget:
    def __init__(self, viz):
        import matplotlib.pyplot as plt

        self.viz                    = viz
        self.alpha                  = 0.5
        self.gain                   = 1.0
        self.show                   = True
        self.heatmap_predictions    = 0
        self.heatmap_uncertainty    = 0
        self.use_predictions        = True
        self.use_uncertainty        = False
        self.cmap_idx               = 0
        self.predictions            = None
        self.uncertainty            = None
        self._generating            = False
        self._triggered             = False
        self._old_predictions       = 0
        self._old_uncertainty       = 0
        self._predictions_gain      = 1.0
        self._uq_gain               = 1.0
        self._heatmap_grid          = None
        self._heatmap_thread        = None
        self._heatmap_toast         = None
        self._colormaps             = plt.colormaps()
        self._rendering_message     = "Calculating whole-slide prediction..."
        self._outcome_names         = None
        self._active_overlay        = None

    @property
    def cmap(self):
        return self._colormaps[self.cmap_idx]

    def _create_heatmap(self):
        viz = self.viz
        self.reset()
        if viz.low_memory:
            mp_kw = dict(num_threads=1, batch_size=4)
        else:
            mp_kw = dict()
        if sf.util.model_backend(self.viz.model) == 'torch':
            mp_kw['apply_softmax'] = self.is_categorical()
        viz.heatmap = sf.heatmap.ModelHeatmap(
            viz.wsi,
            viz.model,
            img_format=viz._model_config['img_format'],
            generate=False,
            normalizer=viz._normalizer,
            uq=viz.has_uq(),
            **mp_kw
        )

    def drag_and_drop(self, path, ignore_errors=True):
        if path.endswith('npz'):
            try:
                self.load(path)
            except Exception:
                sf.log.debug(f"Unable to load {path} as heatmap.")
                if not ignore_errors:
                    raise

    def is_categorical(self):
        """Check if model is a categorical model."""
        return self.viz.model_widget.is_categorical()

    def _generate(self):
        """Create and generate a heatmap asynchronously."""
        sw = self.viz.slide_widget
        self._create_heatmap()
        self._triggered = True
        self._generating = True
        self._heatmap_grid, self._heatmap_thread = self.viz.heatmap.generate(
            asynchronous=True,
            lazy_iter=self.viz.low_memory,
            callback=self.refresh_heatmap_grid,
            **sw.get_tile_filter_params(),
        )

    def load(self, obj: Union[str, "sf.Heatmap"]):
        """Load a heatmap from a saved *.npz file."""
        if isinstance(obj, str) and self.viz._model_config:
            if self.viz.heatmap is None:
                self._create_heatmap()
            self.viz.heatmap.load(obj)
        elif isinstance(obj, str):
            self.viz.create_toast(
                "Unable to load heatmap; model must also be loaded.",
                icon="fail"
            )
        else:
            self.viz.heatmap = obj
        self.predictions = convert_to_overlays(self.viz.heatmap.predictions)
        self.uncertainty = convert_to_overlays(self.viz.heatmap.uncertainty)
        self.render_heatmap()

    def refresh_heatmap_grid(self, grid_idx=None):
        if self.viz.heatmap is not None and self._heatmap_grid is not None:
            predictions, uncertainty = process_grid(self.viz.heatmap,
                                                    self._heatmap_grid)
            self.predictions = convert_to_overlays(predictions)
            self.uncertainty = convert_to_overlays(uncertainty)
            self.render_heatmap()
            self.viz.model_widget._update_slide_preds()
            prog_grid = predictions if len(predictions.shape) == 2 else predictions[:, :, 0]
            percent = (prog_grid != sf.heatmap.MASK).sum() / self.viz.wsi.grid.sum()
            self._heatmap_toast.set_progress(min(percent, 1.))


    def refresh_generating_heatmap(self):
        """Refresh render of the asynchronously generating heatmap."""
        if self._heatmap_thread is not None and not self._heatmap_thread.is_alive():
            self._generating = False
            self._triggered = False
            self._heatmap_thread = None
            self.viz.clear_message(self._rendering_message)
            if self._heatmap_toast is not None:
                self._heatmap_toast.done()
                self._heatmap_toast = None
            self.viz.create_toast("Heatmap complete.", icon="success")

    def render_heatmap(self, outcome_names=None):
        """Render the current heatmap."""
        if self.predictions is None:
            return
        if outcome_names and len(outcome_names) != len(self.predictions):
            raise ValueError("Number of outcome names ({}) must match number of outcomes ({}).".format(
                len(outcome_names), len(self.predictions)
            ))
        if outcome_names:
            self._outcome_names = outcome_names
        elif self._outcome_names is None:
            self._outcome_names = self.get_outcome_names()

        self._old_predictions = self.heatmap_predictions
        self._old_uncertainty = self.heatmap_uncertainty
        if self.viz.heatmap is None:
            return
        if self.use_predictions:
            self._active_overlay = self.predictions[self.heatmap_predictions]
            gain = self._predictions_gain
        else:
            self._active_overlay = self.uncertainty[self.heatmap_uncertainty]
            gain = self._uq_gain
        self.viz.rendered_heatmap = _apply_cmap(self._active_overlay.grid * gain, self.cmap)[:, :, 0:3]
        self.update_transparency()

    def reset(self):
        """Reset the heatmap display."""

        self.viz.rendered_heatmap   = None
        self.viz.overlay            = None
        self.viz.heatmap            = None
        self._heatmap_grid          = None
        self._heatmap_thread        = None
        self._old_predictions       = 0
        self._old_uncertainty       = 0
        self.predictions            = None
        self.uncertainty            = None
        self._generating            = False
        self.heatmap_predictions    = 0
        self.heatmap_uncertainty    = 0
        self._outcome_names         = None
        self.use_predictions        = True

    def update_transparency(self):
        """Update transparency of the heatmap overlay."""

        if self.viz.rendered_heatmap is not None:
            alpha_channel = np.full(self.viz.rendered_heatmap.shape[0:2],
                                    int(self.alpha * 255),
                                    dtype=np.uint8)
            overlay = np.dstack((self.viz.rendered_heatmap[:, :, 0:3], alpha_channel))
            self.viz.set_grid_overlay(
                overlay,
                **self._active_overlay.overlay_kwargs
            )

    def generate(self):
        self.viz.set_message(self._rendering_message)
        self._heatmap_toast = self.viz.create_toast(
            title="Calculating heatmap",
            icon='info',
            sticky=True,
            progress=True,
            spinner=True
        )
        _thread = threading.Thread(target=self._generate)
        _thread.start()
        self.show = True

    def get_outcome_names(self, config=None):
        if self._outcome_names is not None and config is None:
            return self._outcome_names
        if config is None:
            config = self.viz._model_config
        if config is None:
            if (self.predictions is not None
                and len(self.predictions)
                and all(p.name for p in self.predictions)):
                return [p.name for p in self.predictions]
            raise ValueError("Model is not loaded.")

        outcomes = config['outcomes']
        outcomes = [outcomes] if isinstance(outcomes, str) else outcomes
        if 'model_type' in config and config['model_type'] != 'categorical':
            return outcomes
        if len(outcomes) > 1:
            return [config['outcome_labels'][outcome][o] for outcome in outcomes for o in config['outcome_labels'][outcome]]
        else:
            return [config['outcome_labels'][str(oidx)] for oidx in range(len(config['outcome_labels']))]

    def draw_heatmap_thumb(self):
        viz = self.viz
        width = imgui.get_content_region_max()[0] - viz.spacing
        height = imgui.get_text_line_height_with_spacing() * 7 + viz.spacing
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, [0, 0])
        imgui.push_style_color(imgui.COLOR_BORDER, *viz.theme.popup_border)
        imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0, 0, 0, 1)
        imgui.push_style_color(imgui.COLOR_HEADER, 0, 0, 0, 0)
        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, 0.16, 0.29, 0.48, 0.5)
        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, 0.16, 0.29, 0.48, 0.9)
        imgui.begin_child('##heatmap_display', width=width, height=height, border=True)

        if viz.rendered_heatmap is not None:
            hw_ratio = (viz.rendered_heatmap.shape[0] / viz.rendered_heatmap.shape[1])
            max_width = min(width - viz.spacing*2, (height - viz.spacing*2) / hw_ratio)
            max_height = max_width * hw_ratio

            if viz._heatmap_tex_obj is not None:
                imgui.same_line(int((width - max_width)/2))
                imgui.image(viz._heatmap_tex_obj.gl_id, max_width, max_height)

        imgui.end_child()
        imgui.pop_style_color(5)
        imgui.pop_style_var(1)

    def draw_outcome_selection(self, radio: bool = True) -> bool:
        viz = self.viz
        changed = False

        heatmap_predictions_max = 0 if self.predictions is None else len(self.predictions)-1
        self.heatmap_predictions = min(max(self.heatmap_predictions, 0), heatmap_predictions_max)
        narrow_w = imgui.get_text_line_height_with_spacing()
        with imgui_utils.grayed_out(heatmap_predictions_max == 0):
            if radio and imgui.radio_button('##predictions_radio', self.use_predictions):
                if viz.has_uq():
                    changed = True
                    self.use_predictions = True
            if radio:
                imgui.same_line()

            with imgui_utils.item_width(-1 - narrow_w * 2 - viz.spacing*2):

                # Determine outcome name
                outcome_names = self.get_outcome_names()
                _, hpred = imgui.drag_int('##heatmap_predictions',
                                          self.heatmap_predictions+1,
                                          change_speed=0.05,
                                          min_value=1,
                                          max_value=heatmap_predictions_max+1,
                                          format=f'{outcome_names[self.heatmap_predictions]} (%d/{heatmap_predictions_max+1})')
                self.heatmap_predictions = hpred - 1
            imgui.same_line()
            if imgui_utils.button('-##heatmap_predictions', width=narrow_w):
                self.heatmap_predictions -= 1
            imgui.same_line()
            if imgui_utils.button('+##heatmap_predictions', width=narrow_w):
                self.heatmap_predictions += 1
            self.heatmap_predictions = min(max(self.heatmap_predictions, 0), heatmap_predictions_max)

        # Uncertainty.
        if viz.heatmap is None or self.uncertainty is None:
            heatmap_uncertainty_max = 0
        else:
            heatmap_uncertainty_max = len(self.uncertainty)-1

        self.heatmap_uncertainty = min(max(self.heatmap_uncertainty, 0), heatmap_uncertainty_max)
        narrow_w = imgui.get_text_line_height_with_spacing()
        with imgui_utils.grayed_out(viz.heatmap is None or not viz.has_uq()):
            if radio and imgui.radio_button('##uncertainty_radio', not self.use_predictions):
                if viz.has_uq():
                    changed = True
                    self.use_predictions = False

            if radio:
                imgui.same_line()
            with imgui_utils.item_width(-1 - narrow_w * 2 - viz.spacing*2):
                _, huq = imgui.drag_int('##heatmap_uncertainty',
                                        self.heatmap_uncertainty+1,
                                        change_speed=0.05,
                                        min_value=1,
                                        max_value=heatmap_uncertainty_max+1,
                                        format=f'Uncertainty %d/{heatmap_uncertainty_max+1}')
                self.heatmap_uncertainty = huq - 1
            imgui.same_line()
            if imgui_utils.button('-##heatmap_uncertainty', width=narrow_w):
                self.heatmap_uncertainty -= 1
            imgui.same_line()
            if imgui_utils.button('+##heatmap_uncertainty', width=narrow_w):
                self.heatmap_uncertainty += 1
            self.heatmap_uncertainty = min(max(self.heatmap_uncertainty, 0), heatmap_uncertainty_max)

        return changed

    def draw_display_options(self):
        viz = self.viz
        _cmap_changed = False
        _alpha_changed = False
        _gain_changed = False
        _uq_predictions_switched = False

        # Predictions and UQ.
        if self.predictions is not None:
            imgui_utils.vertical_break()
            _uq_predictions_switched = self.draw_outcome_selection()
            imgui_utils.vertical_break()

        # Display options (colormap, opacity, etc).
        if viz.collapsing_header('Display', default=True):
            with imgui_utils.item_width(viz.font_size * 5):
                _clicked, self.show = imgui.checkbox('##show_heatmap', self.show)
                if _clicked:
                    self.render_heatmap()
                    if self.show:
                        self.viz.slide_widget.show_slide_filter  = False
                        self.viz.slide_widget.show_tile_filter   = False

            # Colormap.
            imgui.same_line()
            with imgui_utils.item_width(imgui.get_content_region_max()[0] - viz.font_size*1.8):
                _clicked, self.cmap_idx = imgui.combo("##cmap", self.cmap_idx, self._colormaps)
            if _clicked:
                _cmap_changed = True


            with imgui_utils.grayed_out(viz.heatmap is None):

                # Alpha.
                with imgui_utils.item_width(-1 - viz.button_w - viz.spacing):
                    _alpha_changed, self.alpha = imgui.slider_float('##alpha',
                                                                    self.alpha,
                                                                    min_value=0,
                                                                    max_value=1,
                                                                    format='Alpha %.2f')

                imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.button_w)
                if imgui_utils.button('Reset##alpha', width=-1, enabled=True):
                    self.alpha = 0.5
                    _alpha_changed = True

                # Gain.
                with imgui_utils.item_width(-1 - viz.button_w - viz.spacing):
                    _gain_changed, self.gain = imgui.slider_float('##gain',
                                                                    self.gain,
                                                                    min_value=0,
                                                                    max_value=10,
                                                                    format='Gain %.2f')

                imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.button_w)
                if imgui_utils.button('Reset##gain', width=-1, enabled=True):
                    if self.use_predictions:
                        self.gain = 1.0
                    else:
                        self.gain = 1.0
                    _gain_changed = True

        # Render heatmap
        if _alpha_changed:
            self.update_transparency()
        if _gain_changed:
            if self.use_predictions:
                self._predictions_gain = self.gain
            else:
                self._uq_gain = self.gain
            self.render_heatmap()
        if _uq_predictions_switched:
            self.gain = self._predictions_gain if self.use_predictions else self._uq_gain
            self.render_heatmap()
        if (_cmap_changed
            or (self.heatmap_predictions != self._old_predictions and self.use_predictions)
            or (self.heatmap_uncertainty != self._old_uncertainty and self.use_uncertainty)
            or _uq_predictions_switched):
            self.render_heatmap()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if self._generating:
            self.refresh_generating_heatmap()

        if show:
            viz.header("Heatmap")
            self.draw_heatmap_thumb()
            txt = "Generate" if not self._triggered else "Generating..."
            if viz.sidebar.full_button(txt, enabled=(not self._triggered and viz.wsi and viz._model_config is not None)):
                self.generate()
            if viz._model_config is None and imgui.is_item_hovered():
                imgui.set_tooltip("No model loaded.")
            self.draw_display_options()
