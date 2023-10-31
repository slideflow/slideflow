"""Utility for an efficient, tiled Whole-slide image viewer."""

import time
import imgui
import numpy as np

import shapely.affinity as sa
from shapely.geometry import Polygon
from rasterio.features import rasterize
from contextlib import contextmanager
from typing import Tuple, Optional, TYPE_CHECKING, Union, List
from ._viewer import Viewer
from .. import gl_utils, text_utils
from ...utils import EasyDict

import slideflow as sf

if TYPE_CHECKING:
    import pyvips

# -----------------------------------------------------------------------------

class SlideViewer(Viewer):

    movable = True
    live    = False

    def __init__(self, wsi: sf.WSI, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        # WSI parameters.
        self.rois           = []
        self.selected_rois  = {}
        self.wsi            = wsi
        self._tile_px       = wsi.tile_px
        self._tile_um       = wsi.tile_um
        self._max_w         = None  # Used for late rendering
        self._max_h         = None  # Used for late rendering
        self._last_update   = time.time()  # Used for tracking movement
        self.show_scale     = True
        self.show_thumbnail = True
        self.show_rois      = True

        self.thumb_max_width = 12
        self.thumb_max_height = 8

        # Create initial display
        wsi_ratio = self.dimensions[0] / self.dimensions[1]
        max_w, max_h = self.width, self.height
        if wsi_ratio < self.width / self.height:
            max_w = int(wsi_ratio * max_h)
        else:
            max_h = int(max_w / wsi_ratio)
        self.view_zoom = max(self.dimensions[0] / max_w,
                             self.dimensions[1] / max_h)
        self.view_params = self._calculate_view_params()
        self._refresh_view_full()
        self._refresh_rois()

        # Calculate scales
        self._um_steps = (1000, 500, 400, 250, 200, 100, 50, 30, 20, 10, 5, 3, 2, 1)
        max_scale_w = (120 * self.viz.pixel_ratio)
        self._mpp_cutoffs = np.array([um / max_scale_w for um in self._um_steps])

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self.wsi.dimensions

    @property
    def full_extract_px(self) -> int:
        return self.wsi.full_extract_px

    @property
    def mpp(self) -> float:
        return self.view_zoom * self.wsi.mpp  # type: ignore

    @property
    def scale_um(self) -> float:
        if self.mpp < self._mpp_cutoffs.min():
            return self._um_steps[-1]
        elif self.mpp > self._mpp_cutoffs.max():
            return self._um_steps[0]
        else:
            return self._um_steps[np.where(self._mpp_cutoffs < self.mpp)[0][0]]

    @staticmethod
    def _process_vips(region: "pyvips.Image") -> np.ndarray:
        """Process a vips image and conver to numpy.

        Args:
            region (pyvips.Image): Libvips image.

        Returns:
            Numpy image (uint8)
        """
        if region.bands == 4:
            region = region.flatten()
        return sf.slide.backends.vips.vips2numpy(region)

    def _calculate_view_params(
        self,
        origin: Tuple[float, float] = None
    ) -> EasyDict:
        """Calculate parameters for extracting an image view from the slide.

        Args:
            origin (Tuple[float, float]): WSI origin (x, y) for the viewer.

        Returns:
            A dictionary containing the keys

                'top_left' (Tuple[float, float]): Top-left coordinates of view.

                'window_size' (Tuple[int, int]): Size of window, in full
                magnification space.

                'target_size' (Tuple[int, int]): Target size for window to be
                resized to.
        """
        if origin is None:
            origin = self.origin

        # Refresh whole-slide view.
        # Enforce boundary limits.
        origin = [max(origin[0], 0), max(origin[1], 0)]
        origin = [min(origin[0], self.dimensions[0] - self.wsi_window_size[0]),
                            min(origin[1], self.dimensions[1] - self.wsi_window_size[1])]

        max_w = self.width
        max_h = self.height
        wsi_ratio = self.wsi_window_size[0] / self.wsi_window_size[1]
        if wsi_ratio < (max_w / max_h):
            # Image is taller than wide
            max_w = int(np.round(self.wsi_window_size[0] / (self.wsi_window_size[1] / max_h)))
        else:
            # Image is wider than tall
            max_h = int(np.round(self.wsi_window_size[1] / (self.wsi_window_size[0] / max_w)))
        self.origin = tuple(origin)

        # Calculate region to extract from image
        target_size = (max_w, max_h)
        window_size = (int(self.wsi_window_size[0]), int(self.wsi_window_size[1]))
        return EasyDict(
            top_left=origin,
            window_size=window_size,
            target_size=target_size,
        )

    def _draw_scale(self, max_w: int, max_h: int):

        r = max(self.viz.pixel_ratio, 1)
        origin_x = self.x_offset + (30 * r)
        origin_y = self.y_offset + max_h - (50 * r) - (self.viz.font_size + self.viz.spacing)
        scale_w = self.scale_um / self.mpp

        main_pos = np.array([origin_x, origin_y])
        left_pos = np.array([origin_x, origin_y])
        right_pos = np.array([origin_x+scale_w, origin_y])
        text_pos = np.array([origin_x+(scale_w/2), origin_y+int(20*r)])
        main_verts = np.array([[0, 0], [scale_w, 0]])
        edge_verts = np.array([[0, 0], [0, int(-5*r)]])

        gl_utils.draw_shadowed_line(main_verts, pos=main_pos, linewidth=int(3*r), color=0)
        gl_utils.draw_shadowed_line(main_verts, pos=main_pos, linewidth=int(1*r), color=1)
        gl_utils.draw_shadowed_line(edge_verts, pos=left_pos, linewidth=int(3*r), color=0)
        gl_utils.draw_shadowed_line(edge_verts, pos=left_pos, linewidth=int(1*r), color=1)
        gl_utils.draw_shadowed_line(edge_verts, pos=right_pos, linewidth=int(3*r), color=0)
        gl_utils.draw_shadowed_line(edge_verts, pos=right_pos, linewidth=int(1*r), color=1)

        tex = text_utils.get_texture(f"{self.scale_um:.0f} µm", size=int(18*r), max_width=max_w, max_height=max_h, outline=2)
        tex.draw(pos=text_pos, align=0.5, rint=True, color=1)

    def _draw_thumbnail(self):
        viz = self.viz

        width = viz.font_size * self.thumb_max_width
        height = imgui.get_text_line_height_with_spacing() * self.thumb_max_height + viz.spacing

        if viz.wsi_thumb is not None:
            hw_ratio = (viz.wsi_thumb.shape[0] / viz.wsi_thumb.shape[1])
            max_width = min(width - viz.spacing*2, (height - viz.spacing*2) / hw_ratio)
            max_height = max_width * hw_ratio

            imgui.set_next_window_position(viz.content_frame_width - max_width - viz.spacing*3, viz.menu_bar_height + viz.spacing)
            imgui.set_next_window_size(max_width + viz.spacing*2, max_height + viz.spacing*2)

            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, [0, 0])
            imgui.push_style_color(imgui.COLOR_HEADER, 0, 0, 0, 0)
            imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, 0.16, 0.29, 0.48, 0.5)
            imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, 0.16, 0.29, 0.48, 0.9)
            imgui.begin(
                '##slide_thumb',
                flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
            )

            if viz._wsi_tex_obj is not None:
                imgui.image(viz._wsi_tex_obj.gl_id, max_width, max_height)

                # Show location overlay
                if viz.viewer.wsi_window_size:
                    # Convert from wsi coords to thumbnail coords
                    t_x, t_y = imgui.get_window_position()
                    t_x = t_x + viz.spacing
                    t_w_ratio = max_width / viz.wsi.dimensions[0]
                    t_h_ratio = max_height / viz.wsi.dimensions[1]
                    t_x += viz.viewer.origin[0] * t_w_ratio
                    t_y += viz.viewer.origin[1] * t_h_ratio
                    t_y += viz.spacing

                    draw_list = imgui.get_window_draw_list()
                    draw_list.add_rect(
                        t_x,
                        t_y,
                        t_x + (viz.viewer.wsi_window_size[0] * t_w_ratio),
                        t_y + (viz.viewer.wsi_window_size[1] * t_h_ratio),
                        imgui.get_color_u32_rgba(0, 0, 0, 1),
                        thickness=2)

            imgui.end()
            imgui.pop_style_color(3)
            imgui.pop_style_var(1)

    def _fast_refresh_cucim(self, new_view, p, view_params):
        # Fill in parts of the missing image
        if p.moved_right and p.full_dx:
            new_horizontal = self.wsi.slide.read_from_pyramid(
                top_left=(int(p.tl_new[0]), int(p.tl_new[1])),
                window_size=(p.full_dx, view_params.window_size[1]),
                target_size=(p.dx, view_params.target_size[1]),
                convert='numpy',
                flatten=True
            )
            new_view[:, None:p.dx, :] = new_horizontal
        if p.moved_down and p.full_dy:
            new_vertical = self.wsi.slide.read_from_pyramid(
                top_left=(int(p.tl_new[0]), int(p.tl_new[1])),
                window_size=(view_params.window_size[0], p.full_dy),
                target_size=(view_params.target_size[0], p.dy),
                convert='numpy',
                flatten=True
            )
            new_view[None:p.dy, :, :] = new_vertical
        if p.moved_left and p.full_dx:
            new_horizontal = self.wsi.slide.read_from_pyramid(
                top_left=(int(p.tl_new[0] + view_params.window_size[0] + p.full_dx), int(p.tl_new[1])),
                window_size=(-p.full_dx, view_params.window_size[1]),
                target_size=(-p.dx, view_params.target_size[1]),
                convert='numpy',
                flatten=True
            )
            new_view[:, view_params.target_size[0]+p.dx:None, :] = new_horizontal
        if p.moved_up and p.full_dy:
            new_vertical = self.wsi.slide.read_from_pyramid(
                top_left=(int(p.tl_new[0]), int(p.tl_new[1] + view_params.window_size[1] + p.full_dy)),
                window_size=(view_params.window_size[0], -p.full_dy),
                target_size=(view_params.target_size[0], -p.dy),
                convert='numpy',
                flatten=True
            )
            new_view[view_params.target_size[1]+p.dy:None, :, :] = new_vertical
        return new_view

    def _fast_refresh_libvips(self, new_view, p, view_params):
        # Libvips processing
        ds_level = self.wsi.slide.best_level_for_downsample(p.target_ds)
        region = self.wsi.slide.get_downsampled_image(ds_level)
        resize_factor = self.wsi.slide.level_downsamples[ds_level] / p.target_ds
        region = region.resize(resize_factor)

        # Fill in parts of the missing image
        if p.moved_right and p.full_dx:
            left_edge = int(p.tl_new[0] / p.target_ds)
            top_edge = int(p.tl_new[1] / p.target_ds)
            extract_w = p.dx
            extract_h = min(view_params.target_size[1], region.height)
            with log_vips_error(left_edge, top_edge, extract_w, extract_h):
                new_horizontal = region.crop(left_edge, top_edge, extract_w, extract_h)
            new_horizontal = self._process_vips(new_horizontal)
            new_view[:, None:p.dx, :] = new_horizontal
        if p.moved_down and p.full_dy:
            left_edge = int(p.tl_new[0] / p.target_ds)
            top_edge = int(p.tl_new[1] / p.target_ds)
            extract_w = min(view_params.target_size[0], region.width)
            extract_h = p.dy
            with log_vips_error(left_edge, top_edge, extract_w, extract_h):
                new_vertical = region.crop(left_edge, top_edge, extract_w, extract_h)
            new_vertical = self._process_vips(new_vertical)
            new_view[None:p.dy, :, :] = new_vertical
        if p.moved_left and p.full_dx:
            left_edge = int((p.tl_new[0] + view_params.window_size[0] + p.full_dx) / p.target_ds)
            top_edge = int(p.tl_new[1] / p.target_ds)
            extract_w = -p.dx
            extract_h = min(view_params.target_size[1], region.height)
            with log_vips_error(left_edge, top_edge, extract_w, extract_h):
                new_horizontal = region.crop(left_edge, top_edge, extract_w, extract_h)
            new_horizontal = self._process_vips(new_horizontal)
            new_view[:, view_params.target_size[0]+p.dx:None, :] = new_horizontal
        if p.moved_up and p.full_dy:
            left_edge = int(p.tl_new[0] / p.target_ds)
            top_edge = int((p.tl_new[1] + view_params.window_size[1] + p.full_dy) / p.target_ds)
            extract_w = min(view_params.target_size[0], region.width)
            extract_h = -p.dy
            with log_vips_error(left_edge, top_edge, extract_w, extract_h):
                new_vertical = region.crop(left_edge, top_edge, extract_w, extract_h)
            new_vertical = self._process_vips(new_vertical)
            new_view[view_params.target_size[1]+p.dy:None, :, :] = new_vertical

        return new_view

    def _read_from_pyramid(self, **kwargs) -> np.ndarray:
        """Read from the whole-slide image pyramid and convert to numpy array.

        Keyword args:
            top_left (Tuple[int, int]): Top-left location of the region to
                extract, using base layer coordinates (x, y).
            window_size (Tuple[int, int]): Size of the region to read (width,
                height) using base layer coordinates.
            target_size (Tuple[int, int]): Resize the region to this target
                size (width, height).

        Returns:
            Numpy image (uint8)
        """
        return self.wsi.slide.read_from_pyramid(convert='numpy', flatten=True, **kwargs)

    def _refresh_view_fast(self, view_params: EasyDict) -> None:
        """Refresh the slide viewer with the given view parameters.

        Performs a fast refresh, where only edge pixels previously out of view
        are refreshed.

        Args:
            view_params (EasyDict): Dictionary containing the keys 'top_left',
                'window_size', and 'target_size'.
        """
        if (view_params.window_size != self.view_params.window_size
           or view_params.target_size != self.view_params.target_size
           or self._normalizer
           or self.mpp < self.wsi.mpp):
            self._refresh_view_full(view_params)
        else:
            new_view = np.zeros_like(self.view)
            p = EasyDict() # Parameters for view
            p.tl_old = self.view_params.top_left
            p.tl_new = view_params.top_left
            p.target_ds = view_params.window_size[0] / view_params.target_size[0]

            def end(x):
                return -x if x else None

            p.full_dx = p.tl_old[0] - p.tl_new[0]
            p.full_dy = p.tl_old[1] - p.tl_new[1]
            p.dx = int(p.full_dx / p.target_ds)
            p.dy = int(p.full_dy / p.target_ds)
            p.full_dx = int(np.round(p.dx * p.target_ds))
            p.full_dy = int(np.round(p.dy * p.target_ds))

            # Check for movement
            p.moved_right = p.dx > 0
            p.moved_down = p.dy > 0
            p.moved_left = p.dx < 0
            p.moved_up = p.dy < 0
            if p.moved_down:
                old_y_start, old_y_end = None, end(p.dy)
                new_y_start, new_y_end = p.dy, None
            elif p.moved_up:
                old_y_start, old_y_end = -p.dy, None
                new_y_start, new_y_end = None, end(-p.dy)
            else:
                old_y_start, old_y_end, new_y_start, new_y_end = None, None, None, None
            if p.moved_right:
                old_x_start, old_x_end = None, end(p.dx)
                new_x_start, new_x_end = p.dx, None
            elif p.moved_left:
                old_x_start, old_x_end = -p.dx, None
                new_x_start, new_x_end = None, end(-p.dx)
            else:
                old_x_start, old_x_end, new_x_start, new_x_end = None, None, None, None

            # Keep what we can from the old image
            old_slice = self.view[old_y_start:old_y_end, old_x_start:old_x_end, :]
            new_view[new_y_start:new_y_end, new_x_start:new_x_end, :] = old_slice
            if sf.slide_backend() == 'libvips':
                new_view = self._fast_refresh_libvips(new_view, p=p, view_params=view_params)
            elif sf.slide_backend() == 'cucim':
                new_view = self._fast_refresh_cucim(new_view, p=p, view_params=view_params)
            else:
                raise ValueError("Unrecognized slide backend {}".format(
                    sf.slide_backend()
                ))

            # Finalize
            self.view = new_view
            self.view_params = view_params
            self.origin = tuple(view_params.top_left)
            self._refresh_rois()
            self._update_view_timer()

    def _refresh_view_full(self, view_params: Optional[EasyDict] = None):
        """Refresh the slide viewer with the given view parameters.

        Performs a full refresh, where all pixels are regenerated by extracting
        a thumbnail image from the slide.

        Args:
            view_params (EasyDict): Dictionary containing the keys 'top_left',
                'window_size', and 'target_size'.
        """
        if view_params is None:
            view_params = self.view_params
            self.origin = tuple(view_params.top_left)
        else:
            self.view_params = view_params

        self.view = self._read_from_pyramid(**view_params)

        # Normalize and finalize
        if self._normalizer:
            self.view = self._normalizer.transform(self.view)

        if (self._tex_obj is not None
           and ((abs(self._tex_obj.width - self.width) > 1)
                or (abs(self._tex_obj.height - self.height) > 1))):
            self.clear()

        # Refresh ROIs
        self._refresh_rois()
        self._update_view_timer()

    def _update_view_timer(self) -> None:
        """Update the view timer."""
        self._last_update = time.time()

    def _refresh_rois(self) -> None:
        """Refresh the ROIs for the given location and zoom."""
        self.rois = []
        for roi_idx, roi in enumerate(self.wsi.rois):
            c = self._scale_roi_to_view(roi.coordinates)
            if c is not None:
                self.rois += [(roi_idx, c)]

    def rasterize_rois_in_view(self) -> Optional[np.ndarray]:
        """Rasterize the ROIs in the current view."""
        if not len(self.rois):
            return None
        return np.stack([
            rasterize(
                [sa.translate(Polygon(roi), -self.x_offset, -self.y_offset)],
                out_shape=(self.height, self.width),
                all_touched=False).astype(bool).astype(int).T * (roi_idx + 1)
            for roi_idx, roi in self.rois
            if len(roi) > 2
        ], axis=-1)

    def _render_rois(self) -> None:
        """Render the ROIs with OpenGL."""
        for roi_idx, roi in self.rois:
            if roi_idx in self.selected_rois:
                outline = self.selected_rois[roi_idx]['outline']
                if len(outline) == 4:
                    outline = (outline[0], outline[1], outline[2])
            else:
                outline = 0
            gl_utils.draw_roi(roi, color=1, alpha=0.7, linewidth=5)
            gl_utils.draw_roi(roi, color=outline, alpha=1, linewidth=3)

    def _scale_roi_to_view(self, roi: np.ndarray) -> Optional[np.ndarray]:
        roi = np.copy(roi)
        roi[:, 0] = roi[:, 0] - int(self.origin[0])
        roi[:, 0] = roi[:, 0] / self.view_zoom
        roi[:, 0] = roi[:, 0] + self.view_offset[0] + self.x_offset
        roi[:, 1] = roi[:, 1] - int(self.origin[1])
        roi[:, 1] = roi[:, 1] / self.view_zoom
        roi[:, 1] = roi[:, 1] + self.view_offset[1] + self.y_offset
        out_of_view_max = np.any(np.amax(roi, axis=0) < 0)
        out_of_view_min = np.any(np.amin(roi, axis=0) > np.array([self.width+self.x_offset, self.height+self.y_offset]))
        if not (out_of_view_min or out_of_view_max):
            return roi
        else:
            return None

    def _scale_rois_to_view(self, rois):
        rois = np.copy(rois)
        rois[:, :, 0] = rois[:, :, 0] - int(self.origin[0])
        rois[:, :, 0] = rois[:, :, 0] / self.view_zoom
        rois[:, :, 0] = rois[:, :, 0] + self.view_offset[0] + self.x_offset
        rois[:, :, 1] = rois[:, :, 1] - int(self.origin[1])
        rois[:, :, 1] = rois[:, :, 1] / self.view_zoom
        rois[:, :, 1] = rois[:, :, 1] + self.view_offset[1] + self.y_offset
        out_of_view_max = np.any(np.amax(rois, axis=1) < 0, axis=1)
        out_of_view_min = np.any(np.amin(rois, axis=1) > np.array([self.width+self.x_offset, self.height+self.y_offset]), axis=1)
        return rois[~(out_of_view_min | out_of_view_max)]

    def is_moving(self, thresh: float = 0.2):
        return (time.time() - self._last_update) < thresh

    def grid_in_view(self, wsi=None):
        """Returns coordinates of WSI grid currently in view."""
        if wsi is None:
            wsi = self.wsi
        wsi_stride = int(wsi.full_extract_px / wsi.stride_div)
        xi_start = int(self.origin[0] / wsi_stride)
        yi_start = int(self.origin[1] / wsi_stride)
        xi_end = int((self.origin[0] + self.view_params.window_size[0]) / wsi_stride)
        yi_end = int((self.origin[1] + self.view_params.window_size[1]) / wsi_stride)
        xi_start = max(xi_start-1, 0)
        yi_start = max(yi_start-1, 0)
        xi_end = min(xi_end+1, wsi.shape[0]-1)
        yi_end = min(yi_end+1, wsi.shape[1]-1)
        return (xi_start, xi_end), (yi_start, yi_end)

    def read_tile(
        self,
        x: int,
        y: int,
        img_format: Optional[str] = None,
        allow_errors: bool = True
    ) -> np.ndarray:

        # Determine destination format
        if img_format and img_format.lower() not in ('jpg', 'jpeg', 'png'):
            raise ValueError(f"Unknown image format {img_format}")
        elif img_format is None or img_format.lower() == 'png':
            convert = 'numpy'
        else:
            convert = img_format

        # Calculate resizing
        if int(self.wsi.tile_px) != int(self.wsi.extract_px):
            resize_factor = self.wsi.tile_px/self.wsi.extract_px
        else:
            resize_factor = None

        # Read region from the slide
        try:
            return self.wsi.slide.read_region(
                (x, y),
                self.wsi.downsample_level,
                (self.wsi.extract_px, self.wsi.extract_px),
                convert=convert,
                flatten=True,
                resize_factor=resize_factor
            )
        except Exception as e:
            if allow_errors:
                print(f"Tile coordinates {x}, {y} are out of bounds, skipping: {e}")
                return None
            else:
                raise

    def late_render(self):
        if self.show_rois:
            self._render_rois()
        if self.show_scale:
            self._draw_scale(self._max_w, self._max_h)
        if self.show_thumbnail:
            self._draw_thumbnail()

    def move(self, dx: float, dy: float) -> None:
        """Move the view in the given directions.

        Args:
            dx (float): Move the view this many pixels right.
            dy (float): Move the view this many pixels down.
        """
        new_origin = [self.origin[0] - (dx * self.view_zoom),
                      self.origin[1] - (dy * self.view_zoom)]

        view_params = self._calculate_view_params(new_origin)
        if view_params != self.view_params:
            self._refresh_view_fast(view_params=view_params)

    def refresh_view(self, view_params: Optional[EasyDict] = None) -> None:
        self._refresh_view_full(view_params)

    def render(self, max_w: int, max_h: int) -> None:
        """Render the Slide view display with OpenGL."""
        super().render()
        if self._tex_img is not self.view:
            self._update_texture()
        if self._tex_obj is not None:
            pos = np.array([self.x_offset + max_w / 2, self.y_offset + max_h / 2])
            zoom = min(max_w / self._tex_obj.width, max_h / self._tex_obj.height)
            zoom = np.floor(zoom) if zoom >= 1 else zoom
            self._tex_obj.draw(pos=pos, zoom=zoom, align=0.5, rint=True)
        self._max_w, self._max_h = max_w, max_h

    def select_roi(
        self,
        idx: Union[int, List[int]],
        outline: Optional[Tuple[float, float, float]] = None
    ) -> None:
        if not isinstance(idx, list):
            idx = [idx]
        if outline is None:
            outline = (1, 0, 0)
        for i in idx:
            if i not in self.selected_rois:
                self.selected_rois[i] = {'outline': outline}

    def deselect_roi(self, idx: Optional[int] = None):
        if idx is None:
            self.selected_rois = {}
        elif idx in self.selected_rois:
            del self.selected_rois[idx]
        else:
            raise IndexError(f"ROI {idx} is not selected.")

    def roi_is_selected(self, idx: int) -> bool:
        return idx in self.selected_rois

    def set_tile_px(self, tile_px: int):
        if tile_px != self.tile_px:
            sf.log.error("Attempted to set tile_px={}, existing={}".format(tile_px, self.tile_px))
            raise NotImplementedError

    def set_tile_um(self, tile_um: int):
        if tile_um != self.tile_um:
            sf.log.error("Attempted to set tile_um={}, existing={}".format(tile_um, self.tile_um))
            raise NotImplementedError

    def update(self, width: int, height: int, x_offset: int, y_offset: int, **kwargs) -> None:
        """Update the viewer with a new width, height, and offset."""
        should_refresh = ((width, height, x_offset, y_offset)
                          != (self.width, self.height, self.x_offset, self.y_offset))
        if should_refresh:
            new_origin = self.display_coords_to_wsi_coords(x_offset, y_offset)
        self.width = width
        self.height = height
        self.x_offset = x_offset
        self.y_offset = y_offset

        # Update current zoom (affected by window resizing)
        #self.view_zoom = self.wsi_window_size[0] / self.width
        wsi_width = self.wsi_window_size[0]  # self.dimensions[0]
        wsi_height = self.wsi_window_size[1]  # self.dimensions[1]
        wsi_ratio = wsi_width / wsi_height
        max_w, max_h = self.width, self.height
        if wsi_ratio < self.width / self.height:
            max_w = int(wsi_ratio * max_h)
        else:
            max_h = int(max_w / wsi_ratio)
        self.view_zoom = max(wsi_width / max_w,
                             wsi_height / max_h)

        if should_refresh:
            # Keep the current WSI view stable if the offset changes
            # (e.g. showing/hiding the control pane)
            view_params = self._calculate_view_params(new_origin)
            self.refresh_view(view_params)

    def zoom(self, cx: int, cy: int, dz: float) -> None:
        """Zoom the slide display.

        Args:
            cx (int): Zoom focus location, X coordinate, without offset.
            cy (int): Zoom focus location, Y coordinate, without offset.
            dz (float): Amount to zoom (relative).
        """
        new_zoom = min(self.view_zoom * dz,
                        max(self.dimensions[0] / self.width,
                            self.dimensions[1] / self.height))

        self.zoom_to(cx, cy, new_zoom)

    def zoom_to(self, cx: int, cy: int, z: float) -> None:
        """Zoom the slide display.

        Args:
            cx (int): Zoom focus location, X coordinate, without offset.
            cy (int): Zoom focus location, Y coordinate, without offset.
            z (float): Amount to zoom (absolute).
        """
        wsi_x, wsi_y = self.display_coords_to_wsi_coords(cx, cy, offset=False)
        new_zoom = min(z,
                       max(self.dimensions[0] / self.width,
                           self.dimensions[1] / self.height))

        # Limit maximum zoom level
        if new_zoom * self.wsi.mpp < 0.01:
            return

        self.view_zoom = new_zoom
        new_origin = [wsi_x - (cx * self.wsi_window_size[0] / self.width),
                      wsi_y - (cy * self.wsi_window_size[1] / self.height)]

        view_params = self._calculate_view_params(new_origin)
        if view_params != self.view_params:
            self._refresh_view_full(view_params=view_params)

    def zoom_to_mpp(self, cx: int, cy: int, mpp: float) -> None:
        self.zoom_to(cx, cy, mpp / self.wsi.mpp)

# -----------------------------------------------------------------------------

@contextmanager
def log_vips_error(left_edge, top_edge, extract_w, extract_h):
    try:
        yield
    except Exception:
        sf.log.error(
            "Error attempting to crop pyvips image, with "
            "top/left (x,y) = ({}, {}) and width/height = ({}, {})".format(
                left_edge,
                top_edge,
                extract_w,
                extract_h
            ))
        raise