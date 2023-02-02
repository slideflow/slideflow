import time
import numpy as np
import imgui
import slideflow as sf
from slideflow import log
from typing import Optional, Tuple
from slideflow.slide import wsi_reader

from . import gl_utils, viewer
from ..utils import EasyDict

# -----------------------------------------------------------------------------

class OpenGLMosaic(sf.mosaic.Mosaic):
    def __init__(self, *args, **kwargs):
        """Mosaic map designed for display using OpenGL."""
        super().__init__(*args, **kwargs)

    def _initialize_figure(self, figsize, background):
        pass

    def _add_patch(self, loc, size, **kwargs):
        pass

    def _plot_tile_image(self, image, extent, alpha=1):
        return image

    def _finalize_figure(self):
        pass

# -----------------------------------------------------------------------------

class MosaicViewer(viewer.Viewer):

    movable = True
    live    = False

    def __init__(self, mosaic, width, height, slides=None, **kwargs):
        """Workbench Viewer for displaying and interacting with a mosaic map."""
        super().__init__(width, height, **kwargs)
        self.mosaic = mosaic
        self.mosaic_x_offset = 0
        self.mosaic_y_offset = 0
        self.preview_width = 400
        self.preview_height = 400
        self.preview_texture = None
        self.zoomed = False
        self.size = 0
        self.slides = {sf.util.path_to_name(s): s for s in slides}
        self._hovering_index = None
        self._hovering_time = None
        self._wsi_preview = None

    @property
    def view_offset(self):
        return (self.mosaic_x_offset, self.mosaic_y_offset)

    def get_slide_path(self, slide: str) -> Optional[str]:
        if self.viz.P is not None:
            return self.viz.P.dataset(filters={'slide': slide}).slide_paths()[0]
        elif slide in self.slides:
            return self.slides[slide]
        else:
            return None

    def get_mouse_pos(self) -> Tuple[int, int]:
        x, y = imgui.get_mouse_pos()
        return x - self.x_offset, y - self.y_offset

    def move(self, dx: float, dy: float) -> None:
        if not self.zoomed:
            pass
        else:
            log.debug("Move: dx={}, dy={}".format(dx, dy))
            self.mosaic_x_offset += dx
            self.mosaic_y_offset += dy

    def refresh_view(self, view_params: Optional[EasyDict] = None) -> None:
        pass

    def reset_view(self, max_w: int, max_h: int) -> None:
        pass

    def render_tooltip(self, grid_x: int, grid_y: int) -> None:
        if self._hovering_index != (grid_x, grid_y):
            self._hovering_index = (grid_x, grid_y)
            self._hovering_time = time.time()
            self._wsi_preview = None
        elif time.time() > (self._hovering_time + 0.5):
            # Create a tooltip for the mosaic grid.
            # First, start by finding the associated tile.
            sel = self.mosaic.selected_points()
            point = sel.loc[((sel.grid_x == grid_x) & (sel.grid_y == grid_y))]
            slide = point.slide.values[0]
            location = point.location.values[0]
            slide_path = self.get_slide_path(slide)
            if slide_path is None:
                imgui.set_tooltip(f"Mosaic grid: ({grid_x}, {grid_y})\n{slide}: {location}")
                return

            # Get WSI preview at the tile location.
            if self._wsi_preview is None:
                reader = wsi_reader(slide_path)
                self._wsi_preview = reader.read_from_pyramid(
                    (location[0] - self.preview_width/2, location[1] - self.preview_height/2),
                    (self.preview_width, self.preview_height),
                    (self.preview_width, self.preview_height),
                    convert='numpy'
                )
                if self.preview_texture is None:
                    self.preview_texture = gl_utils.Texture(image=self._wsi_preview, bilinear=True, mipmap=False)
                else:
                    self.preview_texture.update(self._wsi_preview)

            # Create the WSI preview window.
            flags = (imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_SCROLLBAR)
            mouse_x, mouse_y = imgui.get_mouse_pos()
            imgui.set_next_window_position(mouse_x+10, mouse_y+10)
            imgui.set_next_window_size(
                self.preview_width + self.viz.spacing*2,
                self.preview_height + self.viz.spacing + imgui.get_text_line_height_with_spacing()*2
            )
            imgui.begin("##mosaic_tooltip", flags=flags)
            imgui.text(f"Mosaic grid: ({grid_x}, {grid_y})\n{slide}: {location}")
            imgui.image(self.preview_texture.gl_id, self.preview_width, self.preview_height)
            imgui.end()

    def render(self, max_w: int, max_h: int) -> None:
        """Render the mosaic map."""
        if self.size < min(max_w, max_h):
            self.zoomed = False
        if not self.zoomed:
            self.size = min(max_w, max_h)
            if max_w > self.size:
                self.mosaic_x_offset = (max_w - self.size) / 2
                self.mosaic_y_offset = 0
            else:
                self.mosaic_x_offset = 0
                self.mosaic_y_offset = (max_h - self.size) / 2
        self.width = max_w
        self.height = max_h

        image_size = int(self.size / self.mosaic.num_tiles_x)
        imgui.set_next_window_bg_alpha(0)
        mouse_x, mouse_y = self.get_mouse_pos()
        _hov_x, _hov_y = None, None

        # Set the window position by the offset, in points (not pixels)
        imgui.set_next_window_position(self.viz.offset_x, self.viz.offset_y)
        imgui.set_next_window_size(max_w, max_h)
        imgui.begin("Mosaic", flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
        for (x, y), image in self.mosaic.grid_images.items():
            pos = (
                self.mosaic_x_offset + x * image_size,
                self.mosaic_y_offset + self.size - (y * image_size)
            )
            out_of_view = ((pos[0] + image_size < 0)
                           or (pos[1] + image_size < 0)
                           or pos[0] > max_w
                           or pos[1] > max_h)
            if out_of_view:
                continue
            if isinstance(image, np.ndarray):
                self.mosaic.grid_images[(x, y)] = gl_utils.Texture(
                    image=image, bilinear=True, mipmap=False
                )
            gl_id = self.mosaic.grid_images[(x, y)].gl_id
            imgui.set_cursor_pos(pos)
            imgui.image(gl_id, image_size, image_size)
            if ((mouse_x > pos[0] and mouse_x < pos[0] + image_size)
               and (mouse_y > pos[1] and mouse_y < pos[1] + image_size)):
                _hov_x, _hov_y = x, y
        imgui.end()
        if _hov_x is not None:
            self.render_tooltip(_hov_x, _hov_y)
        if 'message' in self.viz.result:
            del self.viz.result.message

    def zoom(self, cx: int, cy: int, dz: float) -> None:
        log.debug("Zoom at ({}, {}): dz={}".format(cx, cy, dz))
        self.zoomed = True
        self.size /= dz
        self.mosaic_x_offset -= (1./dz - 1) * (cx - self.mosaic_x_offset)
        self.mosaic_y_offset -= (1./dz - 1) * (cy - self.mosaic_y_offset)

    def reload(self, **kwargs):
        pass

