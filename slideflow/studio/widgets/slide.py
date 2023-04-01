import os
import cv2
import imgui
import numpy as np
import threading

from .._renderer import CapturedException
from ..utils import EasyDict
from ..gui import imgui_utils

import slideflow as sf

#----------------------------------------------------------------------------

class SlideWidget:
    def __init__(self, viz):
        self.viz                    = viz
        self.cur_slide              = None
        self.user_slide             = ''
        self.normalize_wsi          = False
        self.norm_idx               = 0
        self.qc_idx                 = 0
        self.qc_mask                = None
        self.alpha                  = 1.0
        self.stride                 = 1
        self.show_slide_filter      = False
        self.show_tile_filter       = False
        self.gs_fraction            = sf.slide.DEFAULT_GRAYSPACE_FRACTION
        self.gs_threshold           = sf.slide.DEFAULT_GRAYSPACE_THRESHOLD
        self.ws_fraction            = sf.slide.DEFAULT_WHITESPACE_FRACTION
        self.ws_threshold           = sf.slide.DEFAULT_WHITESPACE_THRESHOLD
        self.num_total_rois         = 0
        self._filter_grid           = None
        self._filter_thread         = None
        self._capturing_ws_thresh   = None
        self._capturing_gs_thresh   = None
        self._capturing_stride      = None
        self._use_rois              = True
        self._rendering_message     = "Calculating tile filter..."
        self._show_filter_controls  = False
        self._all_normalizer_methods = [
            'reinhard',
            'reinhard_fast',
            'reinhard_mask',
            'reinhard_fast_mask',
            'macenko',
            'vahadane_spams',
            'vahadane_sklearn',
            'augment']
        self._all_normalizer_methods_str = [
            'Reinhard',
            'Reinhard (Fast)',
            'Reinhard Mask',
            'Reinhard Mask (Fast)',
            'Macenko',
            'Vahadane (SPAMS)',
            'Vahadane (Sklearn)',
            'Augment']
        self._normalizer_methods = self._all_normalizer_methods
        self._normalizer_methods_str = self._all_normalizer_methods_str
        self._qc_methods_str    = ["Otsu threshold", "Blur filter", "Blur + Otsu"]
        self._qc_methods        = ['otsu', 'blur', 'both']
        self.load('', ignore_errors=True)

    @property
    def show_overlay(self):
        return self.show_slide_filter or self.show_tile_filter

    @property
    def _thread_is_running(self):
        return self._filter_thread is not None and self._filter_thread.is_alive()

    # --- Internal ------------------------------------------------------------

    def _filter_thread_worker(self):
        if self.viz.wsi is not None:
            self.viz.set_message(self._rendering_message)
            if self.viz.low_memory or sf.slide_backend() == 'cucim':
                mp_kw = dict(num_threads=os.cpu_count(), lazy_iter=True)
            else:
                mp_kw = dict(num_processes=os.cpu_count())
            generator = self.viz.wsi.build_generator(
                img_format='numpy',
                grayspace_fraction=sf.slide.FORCE_CALCULATE_GRAYSPACE,
                grayspace_threshold=self.gs_threshold,
                whitespace_fraction=sf.slide.FORCE_CALCULATE_WHITESPACE,
                whitespace_threshold=self.ws_threshold,
                shuffle=False,
                dry_run=True,
                **mp_kw)
            if not generator:
                self.viz.clear_message(self._rendering_message)
                return
            # Returns boolean grid, where:
            #   True = tile will be extracted
            #   False = tile will be discarded (failed QC)
            self._filter_grid = np.transpose(self.viz.wsi.grid).astype(bool)
            self._ws_grid = np.zeros_like(self._filter_grid, dtype=np.float)
            self._gs_grid = np.zeros_like(self._filter_grid, dtype=np.float)
            self.render_overlay(self._filter_grid, correct_wsi_dim=True)
            for tile in generator():
                x = tile['grid'][0]
                y = tile['grid'][1]
                gs = tile['gs_fraction']
                ws = tile['ws_fraction']
                try:
                    self._ws_grid[y][x] = ws
                    self._gs_grid[y][x] = gs
                    if gs > self.gs_fraction or ws > self.ws_fraction:
                        self._filter_grid[y][x] = False
                        self.render_overlay(self._filter_grid, correct_wsi_dim=True)
                except TypeError:
                    # Occurs when the _ws_grid is reset, e.g. the slide was re-loaded.
                    sf.log.debug("Aborting tile filter calculation")
                    self.viz.clear_message(self._rendering_message)
                    return
            self.viz.clear_message(self._rendering_message)

    def _join_filter_thread(self):
        if self._filter_thread is not None:
            self._filter_thread.join()
        self._filter_thread = None

    def _reset_tile_filter_and_join_thread(self):
        self._join_filter_thread()
        if self.viz.viewer is not None:
            self.viz.viewer.clear_overlay_object()
        self._filter_grid = None
        self._filter_thread = None
        self._ws_grid = None
        self._gs_grid = None

    def _start_filter_thread(self):
        self._join_filter_thread()
        self._filter_thread = threading.Thread(target=self._filter_thread_worker)
        self._filter_thread.start()

    def _refresh_gs_ws(self):
        self._join_filter_thread()
        if self._ws_grid is not None:
            # Returns boolean grid, where:
            #   True = tile will be extracted
            #   False = tile will be discarded (failed QC)
            self._filter_grid = np.transpose(self.viz.wsi.grid).astype(bool)
            for y in range(self._ws_grid.shape[0]):
                for x in range(self._ws_grid.shape[1]):
                    ws = self._ws_grid[y][x]
                    gs = self._gs_grid[y][x]
                    if gs > self.gs_fraction or ws > self.ws_fraction:
                        self._filter_grid[y][x] = False
            if self.show_tile_filter:
                self.render_overlay(self._filter_grid, correct_wsi_dim=True)

    # --- Public interface ----------------------------------------------------

    def load(self, slide, stride=None, ignore_errors=False):
        """Load a slide."""

        viz = self.viz
        if slide == '':
            viz.result = EasyDict(message='No slide loaded')
            return

        # Wait until current ops are complete
        self._reset_tile_filter_and_join_thread()
        viz.clear_result()
        viz.skip_frame() # The input field will change on next frame.
        viz.x = None
        viz.y = None

        # Wrap the entire slide loading function in a try-catch block
        # to gracefully handle errors without crashing the application
        try:
            if hasattr(viz, 'close_gan'):
                viz.close_gan()
            name = slide.replace('\\', '/').split('/')[-1]
            self.cur_slide = slide
            self.user_slide = slide
            self._use_rois = True
            viz.set_message(f'Loading {name}...')
            sf.log.debug(f"Loading slide {slide}...")
            viz.defer_rendering()
            if stride is not None:
                self.stride = stride
            viz._reload_wsi(slide, stride=self.stride, use_rois=self._use_rois)
            viz.heatmap_widget.reset()
            self.num_total_rois = len(viz.wsi.rois)

            # Generate WSI thumbnail.
            hw_ratio = (viz.wsi.dimensions[0] / viz.wsi.dimensions[1])
            max_width = int(min(800 - viz.spacing*2, (800 - viz.spacing*2) / hw_ratio))
            viz.wsi_thumb = np.asarray(viz.wsi.thumb(width=max_width, low_res=True))
            viz.clear_message(f'Loading {name}...')
            if not viz.sidebar.expanded:
                viz.sidebar.selected = 'slide'
                viz.sidebar.expanded = True

        except Exception as e:
            self.cur_slide = None
            self.user_slide = slide
            viz.clear_message(f'Loading {name}...')
            viz.result = EasyDict(error=CapturedException())
            sf.log.warn(f"Error loading slide {slide}: {e}")
            viz.create_toast(f"Error loading slide {slide}", icon="error")
            if not ignore_errors:
                raise

    def preview_qc_mask(self, mask):
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask.shape) == 2
        self.qc_mask = ~mask
        self.show_slide_filter = True
        self.update_slide_filter()

    def render_slide_filter(self):
        """Render the slide filter (QC) to screen."""
        self.viz.heatmap_widget.show = False
        if self.viz.viewer is not None:
            self.viz.viewer.clear_overlay_object()
        self.viz._overlay_wsi_dim = None
        self.render_overlay(self.qc_mask, correct_wsi_dim=False)

    def render_overlay(self, mask, correct_wsi_dim=False):
        """Renders boolean mask as an overlay, where:

            True = show tile from slide
            False = show black box
        """
        assert mask.dtype == bool
        alpha = (~mask).astype(np.uint8) * 255
        black = np.zeros(list(mask.shape) + [3], dtype=np.uint8)
        overlay = np.dstack((black, alpha))
        if correct_wsi_dim:
            self.viz.overlay = overlay
            full_extract = int(self.viz.wsi.tile_um / self.viz.wsi.mpp)
            wsi_stride = int(full_extract / self.viz.wsi.stride_div)
            self.viz._overlay_wsi_dim = (wsi_stride * (self.viz.overlay.shape[1]),
                                         wsi_stride * (self.viz.overlay.shape[0]))
            self.viz._overlay_offset_wsi_dim = (full_extract/2 - wsi_stride/2, full_extract/2 - wsi_stride/2)

        else:
            # Cap the maximum size, to fit in GPU memory of smaller devices (e.g. Raspberry Pi)
            if (overlay.shape[1] > overlay.shape[0]) and overlay.shape[1] > 2000:
                target_shape = (2000, int((2000 / overlay.shape[1]) * overlay.shape[0]))
                overlay = cv2.resize(overlay, target_shape)
            elif (overlay.shape[1] < overlay.shape[0]) and overlay.shape[0] > 2000:
                target_shape = (int((2000 / overlay.shape[0]) * overlay.shape[1]), 2000)
                overlay = cv2.resize(overlay, target_shape)

            self.viz.overlay = overlay
            self.viz._overlay_wsi_dim = None
            self.viz._overlay_offset_wsi_dim = (0, 0)

    def show_model_normalizer(self):
        self._normalizer_methods = self._all_normalizer_methods + ['model']
        self._normalizer_methods_str = self._all_normalizer_methods_str + ['<Model>']

    def update_slide_filter(self, method=None):
        if not self.viz.wsi:
            return
        self._join_filter_thread()

        # Update the slide QC
        if self.show_slide_filter and self.viz.wsi is not None:
            self.viz.heatmap_widget.show = False
            if method is not None:
                self.viz.wsi.remove_qc()
                self.qc_mask = ~np.asarray(self.viz.wsi.qc(method), dtype=bool)
        else:
            self.qc_mask = None

        # Update the tile filter since the QC method has changed
        self._reset_tile_filter_and_join_thread()
        if self.show_tile_filter:
            self.update_tile_filter()

        # Render the slide filter
        if self.show_slide_filter and not self.show_tile_filter:
            self.render_slide_filter()

    def update_tile_filter(self):
        if self.show_tile_filter:
            self.viz.heatmap_widget.show = False
            self._join_filter_thread()
            if self.viz.viewer is not None:
                self.viz.viewer.clear_overlay_object()
            if not self.show_slide_filter:
                self.viz.overlay = None
            if self._filter_grid is None and self.viz.wsi is not None:
                self._start_filter_thread()
            elif self._filter_grid is not None:
                # Render tile filter
                self.viz.heatmap_widget.show = False
                self._join_filter_thread()
                self.render_overlay(self._filter_grid, correct_wsi_dim=True)
        else:
            if self.viz.viewer is not None:
                self.viz.viewer.clear_overlay_object()
            if self.show_slide_filter:
                self.render_slide_filter()

    def draw_info(self):
        viz = self.viz
        height = imgui.get_text_line_height_with_spacing() * 12 + viz.spacing
        if viz.wsi is not None:
            width, height = viz.wsi.dimensions
            if self._filter_grid is not None and self.show_tile_filter:
                est_tiles = int(self._filter_grid.sum())
            elif self.show_slide_filter:
                est_tiles = viz.wsi.estimated_num_tiles
            else:
                est_tiles = viz.wsi.grid.shape[0] * viz.wsi.grid.shape[1]
            vals = [
                f"{width} x {height}",
                f'{viz.wsi.mpp:.4f} ({int(10 / (viz.wsi.slide.level_downsamples[0] * viz.wsi.mpp)):d}x)',
                viz.wsi.vendor if viz.wsi.vendor is not None else '-',
                str(est_tiles),
            ]
        else:
            vals = ["-" for _ in range(8)]
        rows = [
            ['Dimensions (w x h)',  vals[0]],
            ['MPP (Magnification)', vals[1]],
            ['Scanner',             vals[2]],
            ['Est. tiles',          vals[3]],
        ]
        height = imgui.get_text_line_height_with_spacing() * len(rows) + viz.spacing
        for y, cols in enumerate(rows):
            for x, col in enumerate(cols):
                if x != 0:
                    imgui.same_line(viz.font_size * (8 + (x - 1) * 6))
                if x == 0:
                    imgui.text_colored(col, *viz.theme.dim)
                else:
                    imgui.text(col)

    def draw_filtering_popup(self):
        viz = self.viz
        cx, cy = imgui.get_cursor_pos()
        imgui.set_next_window_position(viz.sidebar.full_width, cy - viz.font_size)
        imgui.set_next_window_size(viz.font_size*17, viz.font_size*3 + viz.spacing*1.5)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *viz.theme.popup_background)
        imgui.push_style_color(imgui.COLOR_BORDER, *viz.theme.popup_border)
        imgui.begin(
            '##tile_filter_popup',
            flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        )
        with imgui_utils.grayed_out(self._thread_is_running):
            imgui.text_colored('Grayspace', *viz.theme.dim)
            imgui.same_line(viz.font_size * 5)
            slider_w = (imgui.get_content_region_max()[0] - (viz.spacing + viz.label_w)) / 2
            with imgui_utils.item_width(slider_w):
                _gsf_changed, _gs_frac = imgui.slider_float('##gs_fraction',
                                                            self.gs_fraction,
                                                            min_value=0,
                                                            max_value=1,
                                                            format='Fraction %.2f')
            imgui.same_line()
            with imgui_utils.item_width(slider_w):
                _gst_changed, _gs_thresh = imgui.slider_float('##gs_threshold',
                                                            self.gs_threshold,
                                                            min_value=0,
                                                            max_value=1,
                                                            format='Thresh %.2f')

            imgui.text_colored('Whitespace', *viz.theme.dim)
            imgui.same_line(viz.font_size * 5)
            with imgui_utils.item_width(slider_w):
                _wsf_changed, _ws_frac = imgui.slider_float('##ws_fraction',
                                                            self.ws_fraction,
                                                            min_value=0,
                                                            max_value=1,
                                                            format='Fraction %.2f')
            imgui.same_line()
            with imgui_utils.item_width(slider_w):
                _wst_changed, _ws_thresh = imgui.slider_float('##ws_threshold',
                                                            self.ws_threshold,
                                                            min_value=0,
                                                            max_value=255,
                                                            format='Thresh %.0f')

            if not self._thread_is_running:
                if _gsf_changed or _wsf_changed:
                    self.gs_fraction = _gs_frac
                    self.ws_fraction = _ws_frac
                    self._refresh_gs_ws()
                if _gst_changed or _wst_changed:
                    self._capturing_ws_thresh = _ws_thresh
                    self._capturing_gs_thresh = _gs_thresh
            if imgui.is_mouse_released() and self._capturing_gs_thresh:
                self.gs_threshold = self._capturing_gs_thresh
                self.ws_threshold = self._capturing_ws_thresh
                self._capturing_ws_thresh = None
                self._capturing_gs_thresh = None
                self._reset_tile_filter_and_join_thread()
                self.update_tile_filter()
        imgui.end()
        imgui.pop_style_color(2)

    def draw_slide_processing(self):
        viz = self.viz

        # Stride
        imgui.text_colored('Stride', *viz.theme.dim)
        imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.font_size*7)
        with imgui_utils.item_width(viz.font_size * 7):
            _stride_changed, _stride = imgui.slider_int('##stride',
                                                        self.stride,
                                                        min_value=1,
                                                        max_value=16,
                                                        format='Stride %d')
            if _stride_changed:
                self._capturing_stride = _stride
            if imgui.is_mouse_released() and self._capturing_stride:
                # Refresh stride
                self.stride = self._capturing_stride
                self._capturing_stride = None
                self.show_tile_filter = False
                self.show_slide_filter = False
                self._reset_tile_filter_and_join_thread()
                self.viz.clear_overlay()
                self.viz._reload_wsi(stride=self.stride, use_rois=self._use_rois)

        # Tile filtering
        with viz.sidebar.dim_text():
            _filter_clicked, self.show_tile_filter = imgui.checkbox('Tile filter', self.show_tile_filter)
            if _filter_clicked:
                self.update_tile_filter()
        imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.font_size*7)
        if viz.sidebar.small_button('ellipsis'):
            self._show_filter_controls = not self._show_filter_controls
        if self._show_filter_controls:
            self.draw_filtering_popup()

        # Slide filtering
        with viz.sidebar.dim_text():
            _qc_clicked, self.show_slide_filter = imgui.checkbox('Slide filter', self.show_slide_filter)
        imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.font_size*7)
        with imgui_utils.item_width(viz.font_size * 7):
            _qc_method_clicked, self.qc_idx = imgui.combo("##qc_method", self.qc_idx, self._qc_methods_str)
        if _qc_clicked or _qc_method_clicked:
            self.update_slide_filter(method=self._qc_methods[self.qc_idx])

        # Normalizing
        with viz.sidebar.dim_text():
            _norm_clicked, self.normalize_wsi = imgui.checkbox('Normalize', self.normalize_wsi)
        viz._normalize_wsi = self.normalize_wsi
        if self.normalize_wsi and viz.viewer:
            viz.viewer.set_normalizer(viz._normalizer)
        elif viz.viewer:
            viz.viewer.clear_normalizer()

        imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.font_size*7)
        with imgui_utils.item_width(viz.font_size * 7):
            _norm_method_clicked, self.norm_idx = imgui.combo("##norm_method", self.norm_idx, self._normalizer_methods_str)
        if _norm_clicked or _norm_method_clicked:
            # Update the normalizer
            method = self._normalizer_methods[self.norm_idx]
            if method == 'model':
                self.viz._normalizer = sf.util.get_model_normalizer(self.viz._model_path)
            else:
                self.viz._normalizer = sf.norm.autoselect(method, source='v3')
            viz._refresh_view = True

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            viz.sidebar.header("Slide")

        if show and viz.wsi is None:
            imgui_utils.padded_text('No slide has been loaded.', vpad=[int(viz.font_size/2), int(viz.font_size)])
            if viz.sidebar.full_button("Load a Slide"):
                viz.ask_load_slide()

        elif show:
            if viz.sidebar.collapsing_header('Info', default=True):
                self.draw_info()
            if viz.sidebar.collapsing_header('ROIs', default=True):
                imgui.text_colored('ROIs', *viz.theme.dim)
                imgui.same_line(viz.font_size * 8)
                imgui.text(str(self.num_total_rois))
            if viz.sidebar.collapsing_header('Slide Processing', default=True):
                self.draw_slide_processing()
