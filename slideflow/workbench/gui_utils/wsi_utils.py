"""Utility for an efficient, tiled Whole-slide image viewer."""

import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING
from . import gl_utils, text_utils
from .viewer import Viewer
from ..utils import EasyDict

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
        self.wsi            = wsi
        self._tile_px       = wsi.tile_px
        self._tile_um       = wsi.tile_um
        self.show_scale = True

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
        max_scale_w = 120
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
            max_w = int(self.wsi_window_size[0] / (self.wsi_window_size[1] / max_h))
        else:
            # Image is wider than tall
            max_h = int(self.wsi_window_size[1] / (self.wsi_window_size[0] / max_w))
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

        origin_x = self.x_offset + 30
        origin_y = self.y_offset + max_h - 50
        scale_w = self.scale_um / self.mpp

        main_pos = np.array([origin_x, origin_y])
        left_pos = np.array([origin_x, origin_y])
        right_pos = np.array([origin_x+scale_w, origin_y])
        text_pos = np.array([origin_x+(scale_w/2), origin_y+20])
        main_verts = np.array([[0, 0], [scale_w, 0]])
        edge_verts = np.array([[0, 0], [0, -5]])

        gl_utils.draw_shadowed_line(main_verts, pos=main_pos, linewidth=3, color=0)
        gl_utils.draw_shadowed_line(main_verts, pos=main_pos, linewidth=1, color=1)
        gl_utils.draw_shadowed_line(edge_verts, pos=left_pos, linewidth=3, color=0)
        gl_utils.draw_shadowed_line(edge_verts, pos=left_pos, linewidth=1, color=1)
        gl_utils.draw_shadowed_line(edge_verts, pos=right_pos, linewidth=3, color=0)
        gl_utils.draw_shadowed_line(edge_verts, pos=right_pos, linewidth=1, color=1)

        tex = text_utils.get_texture(f"{self.scale_um:.0f} µm", size=18, max_width=max_w, max_height=max_h, outline=2)
        tex.draw(pos=text_pos, align=0.5, rint=True, color=1)

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
            new_horizontal = region.crop(
                int(p.tl_new[0] / p.target_ds),
                int(p.tl_new[1] / p.target_ds),
                p.dx,
                view_params.target_size[1])
            new_horizontal = self._process_vips(new_horizontal)
            new_view[:, None:p.dx, :] = new_horizontal
        if p.moved_down and p.full_dy:
            new_vertical = region.crop(
                int(p.tl_new[0] / p.target_ds),
                int(p.tl_new[1] / p.target_ds),
                view_params.target_size[0],
                p.dy)
            new_vertical = self._process_vips(new_vertical)
            new_view[None:p.dy, :, :] = new_vertical
        if p.moved_left and p.full_dx:
            new_horizontal = region.crop(
                int((p.tl_new[0] + view_params.window_size[0] + p.full_dx) / p.target_ds),
                int(p.tl_new[1] / p.target_ds),
                -p.dx,
                view_params.target_size[1])
            new_horizontal = self._process_vips(new_horizontal)
            new_view[:, view_params.target_size[0]+p.dx:None, :] = new_horizontal
        if p.moved_up and p.full_dy:
            new_vertical = region.crop(
                int(p.tl_new[0] / p.target_ds),
                int((p.tl_new[1] + view_params.window_size[1] + p.full_dy) / p.target_ds),
                view_params.target_size[0],
                -p.dy)
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

    def _refresh_rois(self) -> None:
        """Refresh the ROIs for the given location and zoom."""
        self.rois = []
        for roi in self.wsi.rois:
            c = np.copy(roi.coordinates)
            c[:, 0] = c[:, 0] - int(self.origin[0])
            c[:, 0] = c[:, 0] / self.view_zoom
            c[:, 0] = c[:, 0] + self.view_offset[0] + self.x_offset
            c[:, 1] = c[:, 1] - int(self.origin[1])
            c[:, 1] = c[:, 1] / self.view_zoom
            c[:, 1] = c[:, 1] + self.view_offset[1] + self.y_offset
            self.rois += [c]

    def _render_rois(self) -> None:
        """Render the ROIs with OpenGL."""
        for roi in self.rois:
            gl_utils.draw_roi(roi, color=1, alpha=0.7, linewidth=5)
            gl_utils.draw_roi(roi, color=0, alpha=1, linewidth=3)

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
        self._render_rois()

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
        if self.show_scale:
            self._draw_scale(max_w, max_h)


    def set_tile_px(self, tile_px: int):
        if tile_px != self.tile_px:
            sf.log.error("Attempted to set tile_px={}, existing={}".format(tile_px, self.tile_px))
            raise NotImplementedError

    def set_tile_um(self, tile_um: int):
        if tile_um != self.tile_um:
            sf.log.error("Attempted to set tile_um={}, existing={}".format(tile_um, self.tile_um))
            raise NotImplementedError

    def zoom(self, cx: int, cy: int, dz: float) -> None:
        """Zoom the slide display.

        Args:
            cx (int): Zoom focus location, X coordinate, without offset.
            cy (int): Zoom focus location, Y coordinate, without offset.
            dz (float): Amount to zoom.
        """
        wsi_x, wsi_y = self.display_coords_to_wsi_coords(cx, cy, offset=False)
        new_zoom = min(self.view_zoom * dz,
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
