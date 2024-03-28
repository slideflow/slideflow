"""Whole-slide imaging viewer for graphical interfaces."""

import cv2
import time
import numpy as np
import imgui
import slideflow as sf
from typing import Tuple, Optional, Union, Callable
from .. import gl_utils
from ...utils import EasyDict

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
        normalizer: sf.norm.StainNormalizer = None,
        viz = None
    ):
        self._tex_img           = None
        self._tex_obj           = None
        self._tex_to_delete     = []
        self._normalizer        = normalizer
        self._tile_um           = 0
        self._tile_px           = 0
        self._overlay           = None    # type: Optional[np.ndarray]
        self._overlay_params    = EasyDict()
        self._overlay_alpha_fn  = None
        self._alpha             = None    # type: Optional[Union[Callable, float, int]]
        self._last_alpha        = None    # type: Optional[Union[Callable, float, int]]
        self.origin             = (0, 0)  # WSI origin for the current view.
        self.view               = None    # Numpy image of current view.
        self.h_zoom             = None
        self.overlay_pos        = None
        self.width              = width
        self.height             = height
        self.bilinear           = bilinear
        self.mipmap             = mipmap
        self.view_zoom          = 1
        self.viz                = viz
        self._capture_start     = None

        # Window offset for the display
        self.x_offset           = x_offset
        self.y_offset           = y_offset

        # Overlay
        self._overlay_tex_img   = None  # type: Optional[np.ndarray]
        self._overlay_tex_obj   = None

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
        """Offset for the image being displayed in the viewer."""
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

    def apply_args(self, args):
        pass

    def capture_animation(self):
        self._capture_start = time.time()

    def clear(self):
        """Remove the displayed image."""
        self._tex_img = None
        self.clear_overlay()

    def clear_overlay(self) -> None:
        self._overlay_tex_img = None

    def clear_overlay_object(self):
        self._overlay_tex_obj = None

    def clear_normalizer(self) -> None:
        """Clear the internal normalizer, if one exists."""
        self._normalizer = None

    def close(self):
        pass

    def display_coords_to_wsi_coords(
        self,
        x: Union[int, np.ndarray],
        y: Union[int, np.ndarray],
        offset: bool = True
    ) -> Tuple[float, float]:
        """Convert the given coordinates from screen space (with offsets)
        to WSI space (highest magnification level).

        Args:
            x (int): X coordinate in display space.
            y (int): Y coordinate in display space.
            offset (bool): Include GUI offsets.

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
        x = (x - all_x_offset) * self.view_zoom + self.origin[0]
        y = (y - all_y_offset) * self.view_zoom + self.origin[1]
        return x, y

    def in_view(
        self,
        image: np.ndarray,
        dim: Tuple[int, int],
        offset: Tuple[int, int],
        resize: bool = True
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Gets the section of the overaly currently in view.

        Args:
            image (np.ndarray): Overlay image.
            dim ((int, int), optional): Dimensions of the image in the full
                WSI coordinate space (level=0). Defaults to the full WSI size.
            offset ((int, int)): Offset for the image in the full WSI
                coordinate space (level=0). Defaults to 0, 0 (no offset).
            resize (bool): Resize the region of the image currently in view
                to the size of the preview pane.

        Returns:
            np.ndarray: Crop of image currently in view.

            float: Zoom factor of cropped image view.

            Tuple[int, int]: x, y offset of cropped image in view.
        """
        overlay_zoom = dim[0] / image.shape[1]
        h_zoom = overlay_zoom / self.view_zoom  # type: ignore
        #h_pos = self.wsi_coords_to_display_coords(*offset)

        # Get the slice of overlay currently in view.
        pad = 3 # Padding
        top_left = self.display_coords_to_wsi_coords(0, 0, offset=False)
        slice_min_x = max(int((top_left[0] - offset[0]) / overlay_zoom) - pad, 0)
        slice_max_x = min(int(slice_min_x + (self.width / h_zoom)) + pad, image.shape[1])
        slice_min_y = max(int((top_left[1] - offset[1]) / overlay_zoom) - pad, 0)
        slice_max_y = min(int(slice_min_y + (self.height / h_zoom)) + pad, image.shape[0])
        if len(image.shape) == 2:
            in_view = image[slice_min_y: slice_max_y, slice_min_x: slice_max_x]
        else:
            in_view = image[slice_min_y: slice_max_y, slice_min_x: slice_max_x, :]
        in_view_offset = (int(slice_min_x * overlay_zoom),
                          int(slice_min_y * overlay_zoom))

        # --- Note: the below may not be pixel-perfect ------------------------
        # Resize the overlay if the image is very large
        # To fix: will not resize large & tall images
        if resize and in_view.shape[0] > 2 * self.width:
            shrink_factor = in_view.shape[1] / self.width
            target_shape = (int(in_view.shape[1] / shrink_factor),
                            int(in_view.shape[0] / shrink_factor))
            h_zoom *= shrink_factor
            in_view = cv2.resize(in_view, target_shape, interpolation=cv2.INTER_NEAREST)
        return in_view, h_zoom, in_view_offset

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
        # Draw the capturing animation (flash on screen)
        if self._capture_start:
            now = time.time()
            duration = 1
            if (now - self._capture_start) > duration:
                self._capture_start = None
            else:
                f = (now-self._capture_start) / duration
                alpha = (1-f) ** 3
                gl_utils.draw_rect(
                    pos=(self.x_offset, self.y_offset),
                    size=[self.width, self.height],
                    color=1,
                    alpha=alpha
                )

    def refresh_view(self):
        pass

    def reload(self, **kwargs):
        return self.__class__(**kwargs)

    def render(self):
        for tex in self._tex_to_delete:
            tex.delete()
        self._tex_to_delete = []

    def render_overlay(
        self,
        overlay: np.ndarray,
        dim: Optional[Tuple[int, int]],
        offset: Tuple[int, int] = (0, 0)
    ) -> int:
        """Render an image as an overlay on the WSI.

        Args:
            overlay (np.ndarray): Overlay image to render on the WSI.
            dim ((int, int), optional): Dimensions of the overlay in the full
                WSI coordinate space (level=0). Defaults to the full WSI size.
            offset ((int, int)): Offset for the overlay in the full WSI
                coordinate space (level=0). Defaults to 0, 0 (no offset).

        Returns:
            int: ID of the texture created for the overlay.
        """
        if dim is None:
            dim = self.dimensions

        overlay_params = EasyDict(
            dim=tuple(dim),
            offset=tuple(offset),
            view_zoom=self.view_zoom,
            width=self.width,
            height=self.height,
            origin=tuple(self.origin))
        params_different = (overlay_params != self._overlay_params)
        overlay_different = (overlay is not self._overlay)
        overlay_updated = False

        # If the overlay image is reasonably sized, display as-is.
        if max(overlay.shape) < 5000:
            if self._overlay_tex_img is None or self._overlay is not overlay:
                self._overlay_tex_img = self._overlay = overlay
                overlay_updated = True
            self._overlay_params = overlay_params
            self.h_zoom = (dim[0] / overlay.shape[1]) / self.view_zoom  # type: ignore
            self.overlay_pos = self.wsi_coords_to_display_coords(*offset)

        # Otherwise, display on the portion of the overlay in view.
        # This avoids OpenGL errors when displaying very large overlays.
        elif params_different or overlay_different:
            # Recalculate the overlay
            self._overlay = overlay
            self._overlay_params = overlay_params
            self._overlay_tex_img, self.h_zoom, in_view_offset = self.in_view(
                self._overlay,
                dim=dim,
                offset=self._overlay_params.offset
            )
            self.overlay_pos = self.wsi_coords_to_display_coords(
                int(offset[0] + in_view_offset[0]),
                int(offset[1] + in_view_offset[1]))

        # Update transparency, if function has been specified.
        if (self._alpha is not None
            and (self._alpha is not self._last_alpha
                 or overlay_updated
                 or (max(overlay.shape) >= 5000 and (params_different or overlay_different)))):
            assert self._overlay_tex_img is not None
            self._last_alpha = self._alpha
            img = self._overlay_tex_img
            if isinstance(self._alpha, float):
                alpha_channel = np.full(img.shape[0:2], int(self._alpha * 255), dtype=np.uint8)
                self._overlay_tex_img = np.dstack((img[:, :, 0:3], alpha_channel))
            elif isinstance(self._alpha, int):
                alpha_channel = np.full(img.shape[0:2], self._alpha, dtype=np.uint8)
                self._overlay_tex_img = np.dstack((img[:, :, 0:3], alpha_channel))
            else:
                self._overlay_tex_img = self._alpha(self._overlay_tex_img)

        # Draw the image texture.
        if self._overlay_tex_obj is None or not self._overlay_tex_obj.is_compatible(image=self._overlay_tex_img):
            if self._overlay_tex_obj is not None:
                self._tex_to_delete += [self._overlay_tex_obj]
            self._overlay_tex_obj = gl_utils.Texture(image=self._overlay_tex_img, bilinear=False, mipmap=False)  # type: ignore
        else:
            self._overlay_tex_obj.update(self._overlay_tex_img)
        assert self._overlay_tex_obj is not None
        self._overlay_tex_obj.draw(pos=self.overlay_pos, zoom=self.h_zoom, align=0.5, rint=True, anchor='topleft')

        return self._overlay_tex_obj.gl_id

    def render_overlay_tooltip(self, overlay: np.ndarray) -> Optional[str]:
        # If hovering with ALT key, draw a crosshair and pixel value.
        if self.viz is not None and self.viz._alt_down:
            mx, my = self.viz.get_mouse_pos()
            # Draw crosshair
            gl_utils.draw_line(
                pos=(mx, self.y_offset),
                size=(0, self.height),
                color=1,
                alpha=0.5
            )
            gl_utils.draw_line(
                pos=(self.x_offset, my),
                size=(self.width, 0),
                color=1,
                alpha=0.5
            )

            # Draw pixel value
            x, y = self.display_coords_to_wsi_coords(mx, my)

            # Dimensions of the overlay: overlay_params.dim
            # Offset over the overlay: overlay_params.offset
            # Now, adjust the x, y coordinates to be relative to the overlay
            x -= self._overlay_params.offset[0]
            y -= self._overlay_params.offset[1]

            # Scale the x, y coordinates to be relative to the overlay
            x = int(x * (overlay.shape[1] / self._overlay_params.dim[0]))
            y = int(y * (overlay.shape[0] / self._overlay_params.dim[1]))

            if (0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]):
                text = str(overlay[y, x])
                text += "\nIndex: ({}, {})".format(x, y)
                imgui.set_tooltip(text)
                return text
        return None

    def set_normalizer(self, normalizer: sf.norm.StainNormalizer) -> None:
        """Set the internal WSI normalizer.

        Args:
            normalizer (sf.norm.StainNormalizer): Stain normalizer.
        """
        self._normalizer = normalizer

    def set_overlay_alpha(self, alpha: Optional[Union[Callable, float, int]]):
        self._alpha = alpha

    def set_tile_px(self, tile_px):
        self._tile_px = tile_px

    def set_tile_um(self, tile_um):
        self._tile_um = tile_um

    def update(self, width: int, height: int, x_offset: int, y_offset: int, **kwargs) -> None:
        should_refresh = ((width, height, x_offset, y_offset)
                          != (self.width, self.height, self.x_offset, self.y_offset))
        self.width = width
        self.height = height
        self.x_offset = x_offset
        self.y_offset = y_offset
        if should_refresh:
            self.refresh_view()

    def update_offset(self, x_offset: int, y_offset: int, refresh: bool = False) -> None:
        """Update the window offset.

        Args:
            x_offset (int): X offset for this Slide viewer in the parent
                OpenGL frame.
            y_offset (int): Y offset for this Slide viewer in the parent
                OpenGL frame.
        """
        self.x_offset = x_offset
        self.y_offset = y_offset
        if refresh:
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
