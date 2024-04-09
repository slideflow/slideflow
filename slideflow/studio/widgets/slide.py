import os
import cv2
import imgui
import numpy as np
import threading
import pyperclip
from typing import Optional, Tuple, Union

from .roi import ROIWidget
from .._renderer import CapturedException
from ..utils import EasyDict, LEFT_MOUSE_BUTTON, RIGHT_MOUSE_BUTTON
from ..gui import imgui_utils, gl_utils

import slideflow as sf

#----------------------------------------------------------------------------

def stride_capture(
    viz: "sf.studio.Studio",
    current_val: int,
    capturing: Optional[int],
    label: str = '0',
    *,
    draw_label: bool = True,
    offset: Union[float, str] = 'auto',
    width: Union[float, str] = 'auto',
    min_value: int = 1,
    max_value: int = 8
) -> Tuple[int, Optional[int], bool]:
    """Draw a stride capture widget."""
    if draw_label:
        imgui.text('Stride')
    if offset == 'auto':
        offset = imgui.get_content_region_max()[0] - 1 - viz.font_size*7
    imgui.same_line(offset)
    if width == 'auto':
        width = viz.font_size * 7
    with imgui_utils.item_width(width):
        _stride_changed, _stride = imgui.slider_int(f'##stride_{label}',
                                                    current_val,
                                                    min_value=min_value,
                                                    max_value=max_value,
                                                    format=f'Stride %d')

        # If the stride was changed by the user, update the capturing value
        if _stride_changed:
            capturing = _stride

        # If the user released the mouse and they were capturing a value, finalize the capture
        capture_success = imgui.is_mouse_released() and capturing is not None

        if capture_success:
            current_val = capturing  # type: ignore
            capturing = None

    return current_val, capturing, capture_success


#----------------------------------------------------------------------------

class SlideWidget:
    def __init__(self, viz: "sf.studio.Studio") -> None:
        """Widget for slide processing control and information display.

        Args:
            viz (:class:`slideflow.studio.Studio`): The parent Slideflow Studio
                object.

        """
        self.viz                    = viz
        self.roi_widget             = ROIWidget(viz)
        self.cur_slide              = None
        self.user_slide             = ''
        self.normalize_wsi          = False
        self.norm_idx               = 0
        self.qc_mask                = None
        self.alpha                  = 1.0
        self.stride                 = 1
        self.enable_stride_capture  = True
        self._filter_grid           = None
        self._filter_thread         = None
        self._capturing_ws_thresh   = None
        self._capturing_gs_thresh   = None
        self._capturing_stride      = None  # type: Optional[int]
        self._rendering_message     = "Calculating tile filter..."
        self._show_filter_controls  = False
        self._show_mpp_popup        = False
        self._input_mpp             = 1.0
        self._mpp_reload_kwargs     = dict()

        # Tile & slide filtering
        self.apply_tile_filter      = False
        self.show_tile_filter       = False
        self.show_slide_filter      = False
        self.gs_fraction            = sf.slide.DEFAULT_GRAYSPACE_FRACTION
        self.gs_threshold           = sf.slide.DEFAULT_GRAYSPACE_THRESHOLD
        self.ws_fraction            = sf.slide.DEFAULT_WHITESPACE_FRACTION
        self.ws_threshold           = sf.slide.DEFAULT_WHITESPACE_THRESHOLD

        # Tile extraction preview
        self.preview_tiles          = False
        self.tile_color             = 0
        self._tile_box_coords       = []
        self._vbo                   = None
        self._scaled_boxes          = None
        self._last_processing_params = None
        self._last_rois             = None
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
        self._tile_colors       = ['Black', 'White', 'Red', 'Green', 'Blue']
        self._tile_colors_rgb   = [0, 1, (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        self._gaussian          = sf.slide.qc.GaussianV2()
        self._otsu              = sf.slide.qc.Otsu()
        self._use_otsu          = True
        self._use_gaussian      = False
        self._use_segment       = False
        self._qc_segment        = None
        self._clicking_path     = False

        self.load('', ignore_errors=True)

    @property
    def show_overlay(self) -> bool:
        """Whether any overlay is currently being shown."""
        return self.show_slide_filter or self.show_tile_filter

    @property
    def _thread_is_running(self) -> bool:
        """Whether a thread is currently running."""
        return self._filter_thread is not None and self._filter_thread.is_alive()

    @property
    def editing_rois(self) -> bool:
        return self.roi_widget.capturing or self.roi_widget.editing

    @property
    def apply_slide_filter(self):
        return any([self._use_otsu, self._use_gaussian, self._use_segment])

    @property
    def _segment(self):
        seg_widget = self.viz.get_extension('segment')
        if seg_widget:
            return seg_widget._segment
        else:
            return None


    # --- Internal ------------------------------------------------------------

    def _segmentation_enabled(self) -> bool:
        return self.viz.get_extension('segment') is not None

    def _get_qc(self):
        _qc = []
        if self._use_segment:
            _qc.append(self._segment)
        if self._use_gaussian:
            _qc.append(self._gaussian)
        if self._use_otsu:
            _qc.append(self._otsu)
        return _qc or None

    def _filter_thread_worker(self) -> None:
        """Worker thread for calculating tile filter."""
        if self.viz.wsi is not None:
            self.viz.set_message(self._rendering_message)

            # Optimize the multiprocessing/multithreading method
            # based on the slide reading backend and whether we are operating
            # in low memory mode.
            if self.viz.low_memory and sf.slide_backend() == 'cucim':
                mp_kw = dict(lazy_iter=True, num_threads=os.cpu_count())
            elif self.viz.low_memory:
                mp_kw = dict(lazy_iter=True, num_processes=1)
            elif sf.slide_backend() == 'cucim':
                mp_kw = dict(num_processes=os.cpu_count())
            else:
                mp_kw = dict(num_processes=min(32, os.cpu_count()))

            # Build a tile generator that will yield tiles along with their
            # whitespace and grayspace fractions.
            generator = self.viz.wsi.build_generator(
                img_format='numpy',
                grayspace_fraction=sf.slide.FORCE_CALCULATE_GRAYSPACE,
                grayspace_threshold=self.gs_threshold,
                whitespace_fraction=sf.slide.FORCE_CALCULATE_WHITESPACE,
                whitespace_threshold=self.ws_threshold,
                shuffle=False,
                dry_run=True,
                **mp_kw)
            # If the generator is None, then the slide has no tiles.
            if not generator:
                self.viz.clear_message(self._rendering_message)
                return

            # Returns boolean grid, where:
            #   True = tile will be extracted
            #   False = tile will be discarded (failed QC)
            self._filter_grid = np.transpose(self.viz.wsi.grid).astype(bool)
            self._ws_grid = np.zeros_like(self._filter_grid, dtype=np.float32)
            self._gs_grid = np.zeros_like(self._filter_grid, dtype=np.float32)

            # Render the tile filter grid as an overlay.
            if self.show_tile_filter:
                self.render_overlay(self._filter_grid, correct_wsi_dim=True)

            # Iterate over the tiles and update the filter grid.
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
                        if self.show_tile_filter:
                            self.render_overlay(self._filter_grid, correct_wsi_dim=True)
                        self._update_tile_coords()
                except TypeError:
                    # Occurs when the _ws_grid is reset, e.g. the slide was re-loaded.
                    sf.log.debug("Aborting tile filter calculation")
                    self.viz.clear_message(self._rendering_message)
                    return
            self.viz.clear_message(self._rendering_message)

    def _join_filter_thread(self) -> None:
        """Join the filter thread if it is running."""
        if self._filter_thread is not None:
            self._filter_thread.join()
        self._filter_thread = None

    def _reset_tile_filter_and_join_thread(self) -> None:
        """Reset the tile filter and join the filter thread if it is running."""
        self._join_filter_thread()
        if self.viz.viewer is not None:
            self.viz.viewer.clear_overlay_object()
        self._filter_grid = None
        self._filter_thread = None
        self._ws_grid = None
        self._gs_grid = None

    def _start_filter_thread(self) -> None:
        """Start the filter thread."""
        self._join_filter_thread()
        self._filter_thread = threading.Thread(target=self._filter_thread_worker)
        self._filter_thread.start()

    def _refresh_gs_ws(self) -> None:
        """Refresh the grayspace and whitespace grids."""
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
            self.update_tile_filter()
            self.update_tile_filter_display()
            self._update_tile_coords()
            self.update_params()

    def _render_tile_boxes(self) -> None:
        """Render boxes around where tiles would be extracted."""
        if self.viz.wsi is None:
            return
        if not len(self._tile_box_coords):
            return
        scaled_boxes = self.viz.viewer._scale_rois_to_view(self._tile_box_coords).astype(np.float32)
        if (self._scaled_boxes is None 
            or (not self._scaled_boxes.shape == scaled_boxes.shape) 
            or (not np.all(self._scaled_boxes == scaled_boxes))):
            self._scaled_boxes = scaled_boxes
            self._vbo = gl_utils.create_buffer(scaled_boxes.flatten())
        c = self._tile_colors_rgb[self.tile_color]
        gl_utils.draw_boxes(scaled_boxes, color=c, linewidth=2, alpha=1, vbo=self._vbo)

    def _update_tile_coords(self) -> None:
        """Update the expected coordinates for tiles that will be extracted.

        Expected tile coordinates are based on the slide grid (which is filtered
        by ROIs and slide-level filters / QC) and the current tile filter grid
        (which may be asynchronously updating). Bounding box coordinates for
        each tile are calculated and stored in self._tile_box_coords.

        """
        viz = self.viz
        width = viz.wsi.full_extract_px

        # If there is no slide grid, there are no tiles to extract.
        if not len(viz.wsi.coord):
            self._tile_box_coords = np.array([])
            return

        indices = viz.wsi.coord[:, 2:4]
        mask = viz.wsi.grid[indices[:, 0], indices[:, 1]]
        if self._filter_grid is not None:
            mask = (mask & self._filter_grid[indices[:, 1], indices[:, 0]])
        filtered_coords = viz.wsi.coord[mask]
        _coords = np.zeros((filtered_coords.shape[0], 4, 2))  # Preallocate space for coordinates
        _coords[:, :, 0] = filtered_coords[:, np.newaxis, 0] + np.array([0, width, width, 0])
        _coords[:, :, 1] = filtered_coords[:, np.newaxis, 1] + np.array([0, 0, width, width])
        if _coords.size:
            self._tile_box_coords = _coords
        else:
            self._tile_box_coords = np.array([])

    # --- Callbacks and render triggers ---------------------------------------

    def early_render(self) -> None:
        """Render elements with OpenGL (before other UI elements are drawn).

        Triggers after the slide has been rendered, but before other UI elements are drawn.

        """
        if self.preview_tiles:
            self._render_tile_boxes()
        self.roi_widget.early_render()

    def keyboard_callback(self, key: int, action: int) -> None:
        """Handle keyboard events.

        Args:
            key (int): The key that was pressed. See ``glfw.KEY_*``.
            action (int): The action that was performed (e.g. ``glfw.PRESS``,
                ``glfw.RELEASE``, ``glfw.REPEAT``).

        """
        self.roi_widget.keyboard_callback(key, action)

    def late_render(self) -> None:
        self.roi_widget.late_render()

    # --- Public interface ----------------------------------------------------

    def is_moving(self) -> bool:
        """Check if the current view is moving (within the last 0.5 sec)."""
        if self.viz.viewer is None:
            return False
        return self.viz.viewer.is_moving()

    def get_tile_filter_params(self) -> dict:
        """Return the current tile filter (whitespace/grayspace) parameters.

        Returns:
            dict: A dictionary containing the tile extraction parameters,
                including the whitespace/grayspace thresholds and fractions.

        """
        return dict(
            grayspace_fraction=(self.gs_fraction if self.apply_tile_filter else 1),
            grayspace_threshold=self.gs_threshold,
            whitespace_fraction=(self.ws_fraction if self.apply_tile_filter else 1),
            whitespace_threshold=self.ws_threshold,
        )

    def get_slide_processing_params(self) -> dict:
        """Return the current slide processing parameters.

        Returns:
            dict: A dictionary containing the slide processing parameters,
                including the tile extraction parameters, the QC method, and
                the stride.

        """
        return dict(
            grayspace_fraction=(self.gs_fraction if self.apply_tile_filter else 1),
            grayspace_threshold=self.gs_threshold,
            whitespace_fraction=(self.ws_fraction if self.apply_tile_filter else 1),
            whitespace_threshold=self.ws_threshold,
            qc=self._get_qc(),
            stride=self.stride,
            roi_filter_method=self.roi_widget.roi_filter_method
        )

    def update_params(self) -> None:
        """Log the current slide processing parameters."""
        self._last_processing_params = self.get_slide_processing_params()
        if self.viz.wsi is not None:
            self._last_rois = len(self.viz.wsi.rois)
        else:
            self._last_rois = None

    def params_changed(self) -> bool:
        """Check if the slide processing parameters have changed.

        Returns:
            bool: True if the slide processing parameters have changed,
                False otherwise.

        """
        return (self.get_slide_processing_params() != self._last_processing_params
                or (self.viz.wsi is not None and len(self.viz.wsi.rois) != self._last_rois))

    def load(
        self,
        slide: str,
        stride: Optional[int] = None,
        ignore_errors: bool = False,
        mpp: Optional[float] = None,
        **kwargs: Optional[dict]
    ) -> None:
        """Load a slide.

        Args:
            slide (str): The path to the slide to load.
            stride (int, optional): The stride to use when extracting tiles.
                If a slide is currently loaded and this value is not None, this
                will override the current stride. Defaults to None.
            ignore_errors (bool, optional): Whether to ignore errors when
                loading the slide. Defaults to False.
            mpp (float, optional): The microns per pixel of the slide. Used
                if the slide does not contain microns per pixel metadata
                (e.g. JPG/PNG images). Defaults to None.
            **kwargs: Additional keyword arguments to pass to the slide loader.

        """
        viz = self.viz
        if slide == '':
            return

        # Wait until current ops are complete
        self._reset_tile_filter_and_join_thread()
        viz.clear_result()
        viz.skip_frame() # The input field will change on next frame.
        viz.x = None
        viz.y = None

        # Reset the ROI widget.
        self.roi_widget.reset()

        # Wrap the entire slide loading function in a try-catch block
        # to gracefully handle errors without crashing the application
        try:
            if hasattr(viz, 'close_gan'):
                viz.close_gan()
            name = slide.replace('\\', '/').split('/')[-1]
            self.cur_slide = slide
            self.user_slide = slide
            self.manual_mpp = mpp
            viz.set_message(f'Loading {name}...')
            sf.log.debug(f"Loading slide {slide}...")
            viz.defer_rendering()
            if stride is not None:
                self.stride = stride
            try:
                success = viz.reload_wsi(
                    slide,
                    stride=self.stride,
                    use_rois=self.roi_widget.use_rois,
                    ignore_missing_mpp=False,
                    **kwargs
                )
                if not success:
                    return
            except sf.errors.SlideMissingMPPError:
                self.cur_slide = None
                self.user_slide = slide
                self._show_mpp_popup = True
                self._mpp_reload_kwargs = dict(
                    slide=slide,
                    stride=stride,
                    ignore_errors=ignore_errors,
                    **kwargs
                )
                return
            viz.heatmap_widget.reset()

            # Generate WSI thumbnail.
            hw_ratio = (viz.wsi.dimensions[0] / viz.wsi.dimensions[1])
            max_width = int(min(800 - viz.spacing*2, (800 - viz.spacing*2) / hw_ratio))
            viz.wsi_thumb = np.asarray(viz.wsi.thumb(width=max_width, low_res=True))
            viz.clear_message(f'Loading {name}...')
            if not viz.sidebar.expanded:
                viz.sidebar.selected = 'slide'
                viz.sidebar.expanded = True

            # Load tile coordinates.
            self._update_tile_coords()

            # Update the slide filter.
            if self.apply_slide_filter:
                self.update_slide_filter(method=self._get_qc())

            # Update ROI colors.
            self.roi_widget.refresh_labels()

        except Exception as e:
            self.cur_slide = None
            self.user_slide = slide
            viz.clear_message()
            viz.result = EasyDict(error=CapturedException())
            sf.log.warn(f"Error loading slide {slide}: {e}")
            viz.create_toast(f"Error loading slide {slide}", icon="error")
            if not ignore_errors:
                raise

    def preview_qc_mask(self, mask: np.ndarray) -> None:
        """Preview a slide filter (QC) mask.

        Args:
            mask (np.ndarray): The slide filter mask.

        """
        if not isinstance(mask, np.ndarray):
            raise ValueError("mask must be a numpy array")
        if not mask.dtype == bool:
            raise ValueError("mask must have dtype bool")
        if not len(mask.shape) == 2:
            raise ValueError("mask must be 2D")
        self.qc_mask = ~mask
        self.show_slide_filter = True
        self.update_slide_filter()
        self.update_slide_filter_display()

    def render_slide_filter(self) -> None:
        """Render the slide filter (QC) to screen."""
        if self.qc_mask is None:
            return
        self.viz.heatmap_widget.show = False
        if self.viz.viewer is not None:
            self.viz.viewer.clear_overlay_object()
        self.viz._overlay_wsi_dim = None
        self.render_overlay(self.qc_mask, correct_wsi_dim=False)

    def render_overlay(
        self,
        mask: np.ndarray,
        correct_wsi_dim: bool = False
    ) -> None:
        """Renders boolean mask as an overlay, where:

            True = show tile from slide
            False = show black box

        Args:
            mask (np.ndarray): The boolean mask to render.
            correct_wsi_dim (bool, optional): Whether to correct the overlay
                dimensions to match the WSI dimensions. Defaults to False.

        """
        if not isinstance(mask, np.ndarray):
            raise ValueError("mask must be a numpy array")
        if not mask.dtype == bool:
            raise ValueError("mask must have dtype bool")
        alpha = (~mask).astype(np.uint8) * 255
        black = np.zeros(list(mask.shape) + [3], dtype=np.uint8)
        overlay = np.dstack((black, alpha))
        if correct_wsi_dim:
            self.viz.set_grid_overlay(overlay)

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

    def add_model_normalizer_option(self) -> None:
        """Add the model normalizer option to the dropdown."""
        self._normalizer_methods = self._all_normalizer_methods + ['model']
        self._normalizer_methods_str = self._all_normalizer_methods_str + ['<Model>']

    def update_slide_filter(self, method: Optional[str] = None) -> None:
        """Update the slide filter (QC) mask.

        This will update the slide filter mask and the tile filter mask (if
        applicable), but will not render the slide filter to screen.

        Args:
            method (str, optional): The slide filter method to use.
                Defaults to None.

        """
        if not self.viz.wsi:
            return
        self._join_filter_thread()

        # Update the slide QC
        if self.apply_slide_filter and self.viz.wsi is not None:
            if method is not None:
                self.viz.wsi.remove_qc()
                self.qc_mask = ~np.asarray(self.viz.wsi.qc(method), dtype=bool)
        else:
            self.qc_mask = None

        # Update the tile filter since the QC method has changed
        self._reset_tile_filter_and_join_thread()
        if self.apply_tile_filter:
            self.update_tile_filter()

        self._update_tile_coords()

    def update_slide_filter_display(self) -> None:
        """Update the slide filter (QC) display.

        This will render the slide filter to screen.

        """
        if not self.viz.wsi:
            return

        if self.show_slide_filter and self.viz.wsi is not None:
            self.viz.heatmap_widget.show = False

        # Render the slide filter
        if self.show_slide_filter and not self.show_tile_filter:
            self.render_slide_filter()

    def update_tile_filter(self) -> None:
        """Update the tile filter mask.

        This will update the tile filter mask, but will not render the tile
        filter to screen.

        """
        # If there is an existing tile filter update thread,
        # let that finish before executing a new tile filter update.
        self._join_filter_thread()
        if self.apply_tile_filter:
            # If this is the first request, start the tile filter thread.
            if self._filter_grid is None and self.viz.wsi is not None:
                self._start_filter_thread()
        else:
            self._filter_grid = None

    def update_tile_filter_display(self) -> None:
        """Update the tile filter display.

        This will render the tile filter to screen.

        """
        if self.show_tile_filter:
            # Hide the heatmap overlay, if one exists.
            self.viz.heatmap_widget.show = False
            # Remove any other existing slide overlays.
            if self.viz.viewer is not None:
                self.viz.viewer.clear_overlay_object()
            if not self.show_slide_filter:
                self.viz.overlay = None
            # Otherwise, render the existing tile filter.
            if self._filter_grid is not None:
                self.render_overlay(self._filter_grid, correct_wsi_dim=True)
        else:
            if self.viz.viewer is not None:
                self.viz.viewer.clear_overlay_object()
            if self.show_slide_filter:
                self.render_slide_filter()

    # --- Widget --------------------------------------------------------------

    def draw_info(self) -> None:
        """Draw the info section."""
        viz = self.viz
        height = imgui.get_text_line_height_with_spacing() * 12 + viz.spacing
        if viz.wsi is not None:
            width, height = viz.wsi.dimensions
            if self._filter_grid is not None and self.apply_tile_filter:
                est_tiles = int(self._filter_grid.sum())
            elif self.apply_slide_filter or (viz.wsi.has_rois() and viz.wsi.roi_method != 'ignore'):
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
        imgui.text_colored('Filename', *viz.theme.dim)
        imgui.same_line(viz.font_size * 8)
        with imgui_utils.clipped_with_tooltip(viz.wsi.path, 17):
            imgui.text(imgui_utils.ellipsis_clip(viz.wsi.name, 17))
        if imgui.is_item_hovered() and imgui.is_mouse_down(LEFT_MOUSE_BUTTON):
            self._clicking_path = True
        if self._clicking_path and imgui.is_mouse_released(LEFT_MOUSE_BUTTON):
            self._clicking_path = False
            pyperclip.copy(viz.wsi.path)
            viz.create_toast("Copied slide path to clipboard", icon="info")

        for y, cols in enumerate(rows):
            for x, col in enumerate(cols):
                if x != 0:
                    imgui.same_line(viz.font_size * (8 + (x - 1) * 6))
                if x == 0:
                    imgui.text_colored(col, *viz.theme.dim)
                else:
                    imgui.text(col)

        # Show the loaded tile_px and tile_um
        imgui.text_colored("Tile size (current)", *viz.theme.dim)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Slide has been loaded at this tile size, in pixels (px) and microns (um).")
        imgui.same_line(viz.font_size * (8 + (x - 1) * 6))
        imgui.text("{} px, {} um".format(viz.wsi.tile_px, viz.wsi.tile_um))
        if imgui.is_item_hovered():
            imgui.set_tooltip("Loaded tile size: {}, {}".format(
                f'{viz.wsi.tile_px} x {viz.wsi.tile_px} pixels',
                f'{viz.wsi.tile_um} x {viz.wsi.tile_um} microns'
            ))

        imgui_utils.vertical_break()

    def draw_filtering_popup(self) -> None:
        """Draw the tile filtering popup.

        This will render the tile filtering popup to screen, which allows the
        user to select the tile filtering method and parameters (grayspace
        and whitespace fraction/thresholds).

        """
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
                self.update_tile_filter_display()
                self._update_tile_coords()
                self.update_params()
        imgui.end()
        imgui.pop_style_color(2)

    def draw_slide_processing(self) -> None:
        """Draw the slide processing section.

        This will render the slide processing section to screen, which allows
        the user to select the tile-level processing and slide-level processing
        (QC) methods. It also allows the user to select the stride, which
        controls the overlap between tiles during extraction.

        """
        viz = self.viz

        # Stride
        with imgui_utils.grayed_out(not self.enable_stride_capture):
            stride, _capturing, capture_success = stride_capture(
                viz, self.stride, self._capturing_stride, label='Stride'
            )
            if self.enable_stride_capture:
                self.stride, self._capturing_stride = stride, _capturing
        if self.enable_stride_capture and capture_success:
            self.apply_tile_filter = False
            self.show_tile_filter = False
            self.show_slide_filter = False
            self._reset_tile_filter_and_join_thread()
            self.viz.clear_overlay()
            self.viz.reload_wsi(stride=self.stride, use_rois=self.roi_widget.use_rois)
            self._update_tile_coords()

        # ROI Filter Method
        _roi_filter_method = self.roi_widget.draw_roi_filter_capture()
        if _roi_filter_method is not None:
            self.viz.wsi.roi_filter_method = _roi_filter_method
            self.viz.wsi.process_rois()
            self.update_tile_filter()
            self.update_tile_filter_display()
            self._update_tile_coords()
            self.update_params()
            if self.apply_slide_filter:
                self.update_slide_filter(method=self._get_qc())
                self.update_slide_filter_display()

        # Tile filtering
        _filter_clicked, self.apply_tile_filter = imgui.checkbox('Tile filter', self.apply_tile_filter)
        if _filter_clicked and not self.apply_tile_filter and self.show_tile_filter:
            self.viz.viewer.clear_overlay_object()
            self.viz.overlay = None
        if imgui.is_item_hovered():
            imgui.set_tooltip("Set tile-level filtering strategy")
        imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.font_size*7)
        if viz.sidebar.small_button('ellipsis'):
            self._show_filter_controls = not self._show_filter_controls
        if self._show_filter_controls:
            self.draw_filtering_popup()

        # Slide filtering
        _otsu_clicked, self._use_otsu = imgui.checkbox('Otsu threshold', self._use_otsu)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Apply Otsu's thresholding algorithm")
        _gaussian_clicked, self._use_gaussian = imgui.checkbox('Gaussian filter', self._use_gaussian)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Apply Gaussian filter (GaussianV2)")

        # Segmentation
        ## Checkbox
        with imgui_utils.grayed_out(not self._segment):
            _segment_clicked, self._use_segment = imgui.checkbox('Segment', self._use_segment)
            if not self._segment:
                self._use_segment = False
                _segment_clicked = False
        ## Tooltip
        if imgui.is_item_hovered():
            if not self._segmentation_enabled():
                imgui.set_tooltip("Extension not loaded")
            elif not self._segment:
                imgui.set_tooltip("No segment model loaded")
            else:
                imgui.set_tooltip("Apply tissue segmentation model")

        # Apply slide filtering changes.
        _qc_clicked = any([_otsu_clicked, _gaussian_clicked, _segment_clicked])
        if _qc_clicked:
            if self.apply_slide_filter:
                self.update_slide_filter(method=self._get_qc())
                self.update_slide_filter_display()
            else:
                self.viz.wsi.remove_qc()

        imgui_utils.vertical_break()

    def draw_display_options(self) -> None:
        """Draw the display options section.

        This will render the display options section to screen, which allows
        the user to preivew tile extraction (as bounding box outlines), stain
        normalization, tile filtering, and slide filtering (QC) masks.

        """
        viz = self.viz

        # Show tile outlines
        _preview_clicked, self.preview_tiles = imgui.checkbox("Tile outlines", self.preview_tiles)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Show tile outlines")
        imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.font_size*7)
        with imgui_utils.item_width(viz.font_size * 7), imgui_utils.grayed_out(not self.preview_tiles):
            _color_clicked, self.tile_color = imgui.combo("##tile_color", self.tile_color, self._tile_colors)

        # Normalizing
        _norm_clicked, self.normalize_wsi = imgui.checkbox('Normalizer', self.normalize_wsi)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Preview stain normalization (does not affect model predictions)")
        viz._normalize_wsi = self.normalize_wsi
        if self.normalize_wsi and viz.viewer:
            viz.viewer.set_normalizer(viz._normalizer)
        elif viz.viewer:
            viz.viewer.clear_normalizer()

        imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.font_size*7)
        with imgui_utils.item_width(viz.font_size * 7), imgui_utils.grayed_out(not self.normalize_wsi):
            _norm_method_clicked, self.norm_idx = imgui.combo("##norm_method", self.norm_idx, self._normalizer_methods_str)
        if _norm_clicked or (_norm_method_clicked and self.normalize_wsi):
            # Update the normalizer
            method = self._normalizer_methods[self.norm_idx]
            if method == 'model':
                self.viz._normalizer = sf.util.get_model_normalizer(self.viz._model_path)
            else:
                self.viz._normalizer = sf.norm.autoselect(method, source='v3')
            viz._refresh_view = True

        # Show slide-level filtering
        with imgui_utils.grayed_out(not self.apply_slide_filter):
            _show_qc_clicked, self.show_slide_filter = imgui.checkbox("QC Mask", self.show_slide_filter)
        if _show_qc_clicked and self.show_slide_filter and self.apply_slide_filter:
            self.show_tile_filter = False
            self.update_slide_filter_display()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Show slide filter (quality control) mask")

        # Show ROI outlines
        with imgui_utils.grayed_out(not viz.wsi.has_rois()):
            _roi_clicked, _show_rois = imgui.checkbox("Show ROIs", viz.viewer.show_rois)
            if _roi_clicked and viz.wsi.has_rois():
                viz.viewer.show_rois = _show_rois
            if imgui.is_item_hovered():
                imgui.set_tooltip("Show ROI outlines")

        # Fill ROIs
        imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.font_size*7)
        with imgui_utils.grayed_out((not viz.wsi.has_rois()) or (not viz.viewer.show_rois)):
            _roi_clicked, _fill_rois = imgui.checkbox("Fill ROIs", self.roi_widget._fill_rois)
            if _roi_clicked and (viz.wsi.has_rois() and viz.viewer.show_rois):
                self.roi_widget.set_fill_rois(_fill_rois)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Show ROIs as filled polygons")

        # Show only extracted tiles
        with imgui_utils.grayed_out(not self.apply_tile_filter):
            _s, self.show_tile_filter = imgui.checkbox("Show tile-level filter", self.show_tile_filter)
        if _s and not self.show_tile_filter and self.apply_tile_filter:
            self.viz.viewer.clear_overlay_object()
            self.viz.overlay = None
        elif _s and self.apply_tile_filter:
            self.show_slide_filter = False
            self.update_tile_filter_display()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Only show extracted tiles, hiding other tiles with a black mask.")

        preview_button_text = "Preview tile extraction" if not self._thread_is_running else f"Calculating{imgui_utils.spinner_text()}"
        _params_changed = self.params_changed()
        if (viz.sidebar.full_button(preview_button_text, enabled=(not self._thread_is_running and _params_changed))
            or (_preview_clicked and self.preview_tiles and _params_changed)):
            self.preview_tiles = True
            self.update_tile_filter()
            self.update_tile_filter_display()
            self._update_tile_coords()
            self.update_params()

    def draw_mpp_popup(self) -> None:
        """Prompt the user to specify microns-per-pixel for a slide."""
        window_size = (self.viz.font_size * 18, self.viz.font_size * 8.25)
        self.viz.center_next_window(*window_size)
        imgui.set_next_window_size(*window_size)
        _, opened = imgui.begin('Microns-per-pixel (MPP) Not Found', closable=True, flags=imgui.WINDOW_NO_RESIZE)
        if not opened:
            self._show_mpp_popup = False

        imgui.text("Could not read microns-per-pixel (MPP) value.")
        imgui.text("Set a MPP to continue loading this slide.")
        imgui.separator()
        imgui.text('')
        imgui.same_line(self.viz.font_size*4)
        with imgui_utils.item_width(self.viz.font_size*4):
            _changed, self._input_mpp = imgui.input_float('MPP##input_mpp', self._input_mpp, format='%.3f')
        imgui.same_line()
        if self._input_mpp:
            mag = f'{10/self._input_mpp:.1f}x'
        else:
            mag = '-'
        imgui.text(mag)
        if self.viz.sidebar.full_button("Use MPP", width=-1):
            self.load(mpp=self._input_mpp, **self._mpp_reload_kwargs)
            self._show_mpp_popup = False
        imgui.end()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        """Draw the widget in a sidebar.

        Args:
            show (bool): Whether to draw the widget. Defaults to True.

        """
        viz = self.viz
        self.roi_widget.update(show=show)

        if show:
            viz.header("Slide")

        if show and viz.wsi is None:
            imgui_utils.padded_text('No slide has been loaded.', vpad=[int(viz.font_size/2), int(viz.font_size)])
            if viz.sidebar.full_button("Load a Slide"):
                viz.ask_load_slide()

        elif show:
            if viz.collapsing_header('Info', default=True):
                self.draw_info()
            if viz.collapsing_header('ROIs', default=True):
                self.roi_widget.draw()
            if viz.collapsing_header('Slide Processing', default=False):
                self.draw_slide_processing()
            if viz.collapsing_header('Display', default=False):
                self.draw_display_options()

        if self._show_mpp_popup:
            self.draw_mpp_popup()
