"""Whole-slide imaging viewer for graphical interfaces."""

from typing import Tuple
from . import gl_utils

import slideflow as sf

# -----------------------------------------------------------------------------

class Viewer:

    movable = False
    live    = False

    def __init__(
        self,
        width: int,
        height: int,
        x_offset: int = 0,
        y_offset: int = 0,
        bilinear: bool = False,
        mipmap: bool = False,
        normalizer: sf.norm.StainNormalizer = None
    ):
        self._tex_img       = None
        self._tex_obj       = None
        self._tex_to_delete = []
        self._normalizer    = normalizer
        self._tile_um       = 0
        self._tile_px       = 0
        self.origin         = (0, 0)  # WSI origin for the current view.
        self.view           = None    # Numpy image of current view.
        self.view_zoom      = None    # Zoom level for the current view.
        self.width          = width
        self.height         = height
        self.bilinear       = bilinear
        self.mipmap         = mipmap
        self.view_zoom      = 1

        # Window offset for the display
        self.x_offset       = x_offset
        self.y_offset       = y_offset

    @property
    def dimensions(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @property
    def wsi_window_size(self) -> Tuple[float, float]:
        """Size of the displayed window, in WSI coordinates."""
        return (min(self.width * self.view_zoom, self.dimensions[0]),
                min(self.height * self.view_zoom, self.dimensions[1]))

    @property
    def view_offset(self) -> Tuple[int, int]:
        """Offset for the displayed thumbnail in the viewer."""
        if self.view is not None:
            return ((self.width - self.view.shape[1]) / 2,
                    (self.height - self.view.shape[0]) / 2)
        else:
            return (0, 0)

    @property
    def tile_px(self) -> int:
        return self._tile_px

    @property
    def tile_um(self) -> int:
        return self._tile_um

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

    def clear(self):
        """Remove the displayed image."""
        self._tex_img = None

    def clear_normalizer(self) -> None:
        """Clear the internal normalizer, if one exists."""
        self._normalizer = None

    def close(self):
        pass

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

    def late_render(self):
        pass

    def refresh_view(self):
        pass

    def reload(self, **kwargs):
        return self.__class__(**kwargs)

    def render(self):
        for tex in self._tex_to_delete:
            tex.delete()
        self._tex_to_delete = []

    def set_normalizer(self, normalizer: sf.norm.StainNormalizer) -> None:
        """Set the internal WSI normalizer.

        Args:
            normalizer (sf.norm.StainNormalizer): Stain normalizer.
        """
        self._normalizer = normalizer

    def set_tile_px(self, tile_px):
        self._tile_px = tile_px

    def set_tile_um(self, tile_um):
        self._tile_um = tile_um

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
        self.refresh_view()

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
        if sf.slide_backend() == 'cucim':
            x, y = int(x), int(y)
            origin_x, origin_y = int(self.origin[0]), int(self.origin[1])
        else:
            origin_x, origin_y = self.origin
        wsi_x = ((x - origin_x) / self.view_zoom) + all_x_offset
        wsi_y = ((y - origin_y) / self.view_zoom) + all_y_offset
        return wsi_x, wsi_y
