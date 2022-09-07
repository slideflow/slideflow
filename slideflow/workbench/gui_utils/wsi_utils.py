"""Utility for an efficient, tiled Whole-slide image viewer."""

import os
import numpy as np
import OpenGL.GL as gl
import threading
import multiprocessing as mp
from queue import Queue
from functools import partial
from typing import Tuple, TYPE_CHECKING
from . import gl_utils
from ..utils import EasyDict

import slideflow as sf
from slideflow.util import log

if TYPE_CHECKING:
    import pyvips

# -----------------------------------------------------------------------------

class SlideViewer:

    def __init__(
        self,
        wsi: sf.WSI,
        width: int,
        height: int,
        bilinear: bool = True,
        mipmap: bool = True,
        x_offset: int = 0,
        y_offset: int = 0,
        normalizer: sf.norm.StainNormalizer = None
    ) -> None:
        self._tex_img       = None
        self._tex_obj       = None
        self._normalizer    = normalizer
        self.origin         = (0, 0)  # WSI origin for the current view.
        self.view           = None    # Numpy image of current view.
        self.view_zoom      = None    # Zoom level for the current view.
        self.rois           = []
        self.wsi            = wsi
        self.width          = width
        self.height         = height
        self.bilinear       = bilinear
        self.mipmap         = mipmap

        # Window offset for the display
        self.x_offset       = x_offset
        self.y_offset       = y_offset

        # Create initial display
        wsi_ratio = self.wsi.dimensions[0] / self.wsi.dimensions[1]
        max_w, max_h = width, height
        if wsi_ratio < width / height:
            max_w = int(wsi_ratio * max_h)
        else:
            max_h = int(max_w / wsi_ratio)
        self.view_zoom = max(self.wsi.dimensions[0] / max_w,
                             self.wsi.dimensions[1] / max_h)
        self.view_params = self.calculate_view_params()
        self.refresh_view_full()
        self.refresh_rois()

    @property
    def wsi_window_size(self) -> Tuple[float, float]:
        """Size of the displayed window, in WSI coordinates."""
        return (min(self.width * self.view_zoom, self.wsi.dimensions[0]),
                min(self.height * self.view_zoom, self.wsi.dimensions[1]))

    @property
    def view_offset(self) -> Tuple[int, int]:
        """Offset for the displayed thumbnail in the viewer."""
        if self.view is not None:
            return ((self.width - self.view.shape[1]) / 2,
                    (self.height - self.view.shape[0]) / 2)
        else:
            return (0, 0)

    @staticmethod
    def process_vips(region: "pyvips.Image") -> np.ndarray:
        """Process a vips image and conver to numpy.

        Args:
            region (pyvips.Image): Libvips image.

        Returns:
            Numpy image (uint8)
        """
        if region.bands == 4:
            region = region.flatten()
        return sf.slide.vips2numpy(region)

    def _update_texture(self) -> None:
        """Update the internal Texture object to match a given numpy image."""
        self._tex_img = self.view
        if (self._tex_obj is None
           or not self._tex_obj.is_compatible(image=self._tex_img)):
            if self._tex_obj is not None:
                self._tex_obj.delete()
            self._tex_obj = gl_utils.Texture(
                image=self._tex_img,
                bilinear=self.bilinear,
                mipmap=self.mipmap)
        else:
            self._tex_obj.update(self._tex_img)

    def calculate_view_params(
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
        origin = [min(origin[0], self.wsi.dimensions[0] - self.wsi_window_size[0]),
                            min(origin[1], self.wsi.dimensions[1] - self.wsi_window_size[1])]

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

    def clear(self):
        """Remove the displayed image."""
        self._tex_img = None

    def clear_normalizer(self) -> None:
        """Clear the internal normalizer, if one exists."""
        self._normalizer = None

    def display_coords_to_wsi_coords(
        self,
        x: int,
        y: int,
        offset: bool = True
    ) -> Tuple[float, float]:
        """Convert the given coordinates from screen space (with offsets)
        to WSI space (highest magnification level).

        Args:
            x (int): X coordinate in display space.
            y (int): Y coordinate in display space.

        Returns:
            A tuple containing

                float: x coordinate in WSI space (highest magnification level).

                float: y coordinate in WSI space (highest magnification level).
        """
        all_x_offset = self.view_offset[0]
        all_y_offset = self.view_offset[1]
        if offset:
            all_x_offset += self.x_offset
            all_y_offset += self.y_offset
        return (
            (x - all_x_offset) * self.view_zoom + self.origin[0],
            (y - all_y_offset) * self.view_zoom + self.origin[1]
        )

    def is_in_view(self, cx: int, cy: int) -> bool:
        """Checks if the given coordinates (in screen space) are in the active
        Slide viewer.

        Args:
            cx (int): X coordinate (without offset).
            cy (int): Y coordinate (without offset).

        Returns:
            bool
        """
        x_in_view = (self.view_offset[0] <= cx <= (self.view_offset[0] + self.view.shape[1]))
        y_in_view = (self.view_offset[1] <= cy <= (self.view_offset[1] + self.view.shape[0]))
        return x_in_view and y_in_view

    def move(self, dx: float, dy: float) -> None:
        """Move the view in the given directions.

        Args:
            dx (float): Move the view this many pixels right.
            dy (float): Move the view this many pixels down.
        """
        new_origin = [self.origin[0] - (dx * self.view_zoom),
                      self.origin[1] - (dy * self.view_zoom)]

        view_params = self.calculate_view_params(new_origin)
        if view_params != self.view_params:
            self.refresh_view_fast(view_params=view_params)

    def read_from_pyramid(self, **kwargs) -> np.ndarray:
        """Read from the Libvips slide pyramid and convert to numpy array.

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
        region = self.wsi.slide.read_from_pyramid(**kwargs)
        return self.process_vips(region)

    def refresh_view_fast(self, view_params: EasyDict) -> None:
        """Refresh the slide viewer with the given view parameters.

        Performs a fast refresh, where only edge pixels previously out of view
        are refreshed.

        Args:
            view_params (EasyDict): Dictionary containing the keys 'top_left',
                'window_size', and 'target_size'.
        """
        if (view_params.window_size != self.view_params.window_size
           or view_params.target_size != self.view_params.target_size):
            self.refresh_view_full(view_params)
        else:

            tl_old = self.view_params.top_left
            tl_new = view_params.top_left
            target_ds = view_params.window_size[0] / view_params.target_size[0]
            new_view = np.zeros_like(self.view)

            def end(x):
                return -x if x else None

            full_dx = tl_old[0] - tl_new[0]
            full_dy = tl_old[1] - tl_new[1]
            dx = int(full_dx / target_ds)
            dy = int(full_dy / target_ds)

            moved_right = dx > 0
            moved_down = dy > 0
            moved_left = dx < 0
            moved_up = dy < 0

            # Check for movement
            if moved_down:
                old_y_start, old_y_end = None, end(dy)
                new_y_start, new_y_end = dy, None
            elif moved_up:
                old_y_start, old_y_end = -dy, None
                new_y_start, new_y_end = None, end(-dy)
            else:
                old_y_start, old_y_end, new_y_start, new_y_end = None, None, None, None
            if moved_right:
                old_x_start, old_x_end = None, end(dx)
                new_x_start, new_x_end = dx, None
            elif moved_left:
                old_x_start, old_x_end = -dx, None
                new_x_start, new_x_end = None, end(-dx)
            else:
                old_x_start, old_x_end, new_x_start, new_x_end = None, None, None, None

            # Keep what we can from the old image
            old_slice = self.view[old_y_start:old_y_end, old_x_start:old_x_end, :]
            new_view[new_y_start:new_y_end, new_x_start:new_x_end, :] = old_slice

            # Libvips processing
            ds_level = self.wsi.slide.best_level_for_downsample(target_ds)
            region = self.wsi.slide.get_downsampled_image(ds_level)
            resize_factor = self.wsi.slide.level_downsamples[ds_level] / target_ds
            region = region.resize(resize_factor)

            # Fill in parts of the missing image
            if moved_right:
                new_horizontal = region.crop(
                    int(tl_new[0] / target_ds),
                    int(tl_new[1] / target_ds),
                    dx,
                    view_params.target_size[1])
                new_horizontal = self.process_vips(new_horizontal)
                new_view[:, None:dx, :] = new_horizontal
            if moved_down:
                new_vertical = region.crop(
                    int(tl_new[0] / target_ds),
                    int(tl_new[1] / target_ds),
                    view_params.target_size[0],
                    dy)
                new_vertical = self.process_vips(new_vertical)
                new_view[None:dy, :, :] = new_vertical
            if moved_left:
                new_horizontal = region.crop(
                    int((tl_new[0] + view_params.window_size[0] + full_dx) / target_ds),
                    int(tl_new[1] / target_ds),
                    -dx,
                    view_params.target_size[1])
                new_horizontal = self.process_vips(new_horizontal)
                new_view[:, view_params.target_size[0]+dx:None, :] = new_horizontal
            if moved_up:
                new_vertical = region.crop(
                    int(tl_new[0] / target_ds),
                    int((tl_new[1] + view_params.window_size[1] + full_dy) / target_ds),
                    view_params.target_size[0],
                    -dy)
                new_vertical = self.process_vips(new_vertical)
                new_view[view_params.target_size[1]+dy:None, :, :] = new_vertical

            # Finalize
            self.view = new_view
            self.view_params = view_params
            self.origin = tuple(view_params.top_left)
            self.refresh_rois()

    def refresh_view_full(self, view_params=None):
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

        self.view = self.read_from_pyramid(**view_params)

        # Normalize and finalize
        if self._normalizer:
            self.view = self._normalizer.transform(self.view)

        if (self._tex_obj is not None
           and ((abs(self._tex_obj.width - self.width) > 1)
                or (abs(self._tex_obj.height - self.height) > 1))):
            self.clear()

        # Refresh ROIs
        self.refresh_rois()

    def refresh_rois(self) -> None:
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

    def render(self, max_w: int, max_h: int) -> None:
        """Render the Slide view display with OpenGL."""
        if self._tex_img is not self.view:
            self._update_texture()
        if self._tex_obj is not None:
            pos = np.array([self.x_offset + max_w / 2, self.y_offset + max_h / 2])
            zoom = min(max_w / self._tex_obj.width, max_h / self._tex_obj.height)
            zoom = np.floor(zoom) if zoom >= 1 else zoom
            self._tex_obj.draw(pos=pos, zoom=zoom, align=0.5, rint=True)

    def render_rois(self) -> None:
        """Render the ROIs with OpenGL."""
        for roi in self.rois:
            gl_utils.draw_roi(roi, color=1, alpha=0.7, linewidth=5)
            gl_utils.draw_roi(roi, color=0, alpha=1, linewidth=3)

    def set_normalizer(self, normalizer: sf.norm.StainNormalizer) -> None:
        """Set the internal WSI normalizer.

        Args:
            normalizer (sf.norm.StainNormalizer): Stain normalizer.
        """
        self._normalizer = normalizer

    def update_offset(self, x_offset: int, y_offset: int) -> None:
        """Update the window offset.

        Args:
            x_offset (int): X offset for this Slide viewer in the parent
                OpenGL frame.
            y_offset (int): Y offset for this Slide viewer in the parent
                OpenGL frame.
        """
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.refresh_view_full()

    def wsi_coords_to_display_coords(
        self,
        x: int,
        y: int,
        offset: bool = True
    ) -> Tuple[float, float]:
        """Convert the given coordinates from WSI (highest magnification level)
        to screen space (with offsets).

        Args:
            x (int): X coordinate in WSI space (highest magnification level).
            y (int): Y coordinate in WSI space (highest magnification level).

        Returns:
            A tuple containing

                float: x coordinate in display space

                float: y coordinate in display space
        """
        all_x_offset = self.view_offset[0]
        all_y_offset = self.view_offset[1]
        if offset:
            all_x_offset += self.x_offset
            all_y_offset += self.y_offset
        return (
            ((x - self.origin[0]) / self.view_zoom) + all_x_offset,
            ((y - self.origin[1]) / self.view_zoom) + all_y_offset
        )

    def zoom(self, cx: int, cy: int, dz: float) -> None:
        """Zoom the slide display.

        Args:
            cx (int): Zoom focus location, X coordinate, without offset.
            cy (int): Zoom focus location, Y coordinate, without offset.
            dz (float): Amount to zoom.
        """
        wsi_x, wsi_y = self.display_coords_to_wsi_coords(cx, cy, offset=False)
        self.view_zoom = min(self.view_zoom * dz,
                                max(self.wsi.dimensions[0] / self.width,
                                    self.wsi.dimensions[1] / self.height))
        new_origin = [wsi_x - (cx * self.wsi_window_size[0] / self.width),
                      wsi_y - (cy * self.wsi_window_size[1] / self.height)]

        view_params = self.calculate_view_params(new_origin)
        if view_params != self.view_params:
            self.refresh_view_full(view_params=view_params)
