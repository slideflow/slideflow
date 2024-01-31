import slideflow as sf
import imgui
import numpy as np
from os.path import join, dirname, abspath, exists
from threading import Thread
from tkinter.filedialog import askopenfilename

from ._utils import Widget
from ..gui import imgui_utils


class TissueSegWidget(Widget):

    tag = 'segment'
    description = 'Tissue Segmentation'
    #icon = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_segment.png')
    #icon_highlighted = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_segment_highlighted.png')

    def __init__(self, viz):
        self.viz                    = viz
        self._segment               = None
        self._thread                = None
        self._load_toast            = None
        self._working_toast         = None
        self._show_params           = False
        self._rois_at_start         = 0
        self._need_to_refresh_rois  = False

    # --- Properties ---

    @property
    def cfg(self):
        seg = self._segment
        return None if seg is None else seg.cfg

    # --- Internal ---

    def close(self):
        pass

    def is_thread_running(self):
        return self._thread is not None and self._thread.is_alive()

    def drag_and_drop_hook(self, path, ignore_errors=False) -> bool:
        """Handle file paths provided via drag-and-drop."""
        if (sf.util.path_to_ext(path).lower() == 'pth'):
            if exists(join(dirname(path), 'segment_params.json')):
                self.load(path, ignore_errors=ignore_errors)
                return True
        return False

    # --- Model loading ---

    def ask_load_model(self) -> str:
        model_path = askopenfilename(
            title="Load model...",
            filetypes=[("pth", ".pth"), ("All files", ".*")]
        )
        if model_path:
            self.load(model_path)

    def load(self, path, ignore_errors=False):
        """Load a tissue segmentation model."""
        if self.is_thread_running():
            self._thread.join()
        self._load_toast = self.viz.create_toast(
            title=f"Loading segmentation model",
            icon='info',
            sticky=True,
            spinner=True)
        self._thread = Thread(target=self._load_model, args=(path, ignore_errors))
        self._thread.start()

    def _load_model(self, path, ignore_errors=False):
        try:
            self._segment = sf.slide.qc.Segment(path)
        except Exception as e:
            if self._load_toast is not None:
                self._load_toast.done()
            sf.log.error(f"Error loading segment model: {e}")
            self.viz.create_toast(f"Error loading segment model: {e}", icon="error")
            self._segment = None
        else:
            if self._load_toast is not None:
                self._load_toast.done()
            self.viz.create_toast(
                f"Loaded model at {path}.",
                icon="success"
            )

    def generate_rois(self):
        """Generate ROIs from the loaded segmentation model."""
        if self.is_thread_running():
            self.viz.create_toast("Failed to start thread.", icon="error")
            return
        self._rois_at_start = len(self.viz.wsi.rois)
        self._working_toast = self.viz.create_toast(
            title=f"Generating ROIs from segmentation model",
            icon='info',
            sticky=True,
            spinner=True)
        self._thread = Thread(target=self._generate_rois)
        self._thread.start()

    def _generate_rois(self):
        viz = self.viz
        self._segment.generate_rois(viz.wsi)
        self._need_to_refresh_rois = True
        if self._working_toast is not None:
            self._working_toast.done()
        viz.create_toast(
            "Generated {} ROIs.".format(
                len(self.viz.wsi.rois) - self._rois_at_start
            ),
            icon="success"
        )

    # --- Drawing ---

    def draw_info(self):
        """Draw information about the loaded model."""
        viz = self.viz

        rows = [
            ['Architecture', self.cfg.arch],
            ['Encoder',      self.cfg.encoder_name],
            ['Mode',         self.cfg.mode],
            ['Classes',      self.cfg.out_classes],
            ['MPP',          self.cfg.mpp, 'Microns per pixel (optical resolution)']
        ]
        imgui.text_colored('Model', *viz.theme.dim)
        imgui.same_line(viz.font_size * 6)
        with imgui_utils.clipped_with_tooltip(self._segment.model_path, 22):
            imgui.text(imgui_utils.ellipsis_clip(self._segment.model_path, 22))
        for y, cols in enumerate(rows):
            for x, col in enumerate(cols):
                if x != 0:
                    imgui.same_line(viz.font_size * (6 + (x - 1) * 6))
                if x == 0:
                    imgui.text_colored(str(col), *viz.theme.dim)
                    if len(cols) == 3 and imgui.is_item_hovered():
                        imgui.set_tooltip(cols[2])
                elif x == 1:
                    imgui.text(str(col))

        imgui.same_line(imgui.get_content_region_max()[0] - viz.font_size - viz.spacing * 2)
        if imgui.button("HP"):
            self._show_params = not self._show_params


    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            viz.header("Tissue Segmentation")

        if show and self._segment is None:
            imgui_utils.padded_text(
                'No model has been loaded.',
                vpad=[int(viz.font_size/2), int(viz.font_size)]
            )
            if viz.sidebar.full_button("Load a Model"):
                self.ask_load_model()

        elif show:
            if viz.collapsing_header('Model Info', default=True):
                self.draw_info()

            if viz.sidebar.full_button(
                'Generate ROIs',
                enabled=(
                    not self.is_thread_running()
                    and (self._segment is not None)
                    and (viz.wsi is not None)
                )
            ):
                self.generate_rois()

        # Refresh ROIs if necessary.
        # Must be in the main loop.
        if self._need_to_refresh_rois:
            self._need_to_refresh_rois = False
            viz.slide_widget.roi_widget.refresh_rois()