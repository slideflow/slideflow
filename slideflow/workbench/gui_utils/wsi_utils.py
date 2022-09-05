"""Utility for an efficient, tiled Whole-slide image viewer."""

import os
import numpy as np
import OpenGL.GL as gl
import threading
import multiprocessing as mp
from queue import Queue
from functools import partial
from . import gl_utils
from ..utils import EasyDict

import slideflow as sf
from slideflow.util import log


# -----------------------------------------------------------------------------

class SlideViewer:

    def __init__(self, wsi, width, height, bilinear=True, mipmap=True, x_offset=0, y_offset=0, normalizer=None):
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
        self.refresh_view()
        self.refresh_rois()

    @property
    def wsi_window_size(self):
        return (min(self.width * self.view_zoom, self.wsi.dimensions[0]),
                min(self.height * self.view_zoom, self.wsi.dimensions[1]))

    @property
    def view_offset(self):
        '''Offset for the displayed thumbnail in the viewer.'''
        if self.view is not None:
            return ((self.width - self.view.shape[1]) / 2,
                    (self.height - self.view.shape[0]) / 2)
        else:
            return (0, 0)

    def _update_texture(self):
        self._tex_img = self.view
        if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
            if self._tex_obj is not None:
                self._tex_obj.delete()
            self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=self.bilinear, mipmap=self.mipmap)
        else:
            self._tex_obj.update(self._tex_img)

    def set_normalizer(self, normalizer):
        self._normalizer = normalizer

    def clear_normalizer(self):
        self._normalizer = None

    def update_offset(self, x_offset, y_offset):
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.refresh_view()

    def is_in_view(self, cx, cy):
        return ((self.view_offset[0] <= cx <= (self.view_offset[0] + self.view.shape[1]))
                and (self.view_offset[1] <= cy <= (self.view_offset[1] + self.view.shape[0])))

    def wsi_coords_to_display_coords(self, x, y):
        return (
            ((x - self.origin[0]) / self.view_zoom) + self.view_offset[0],
            ((y - self.origin[1]) / self.view_zoom) + self.view_offset[1]
        )

    def display_coords_to_wsi_coords(self, x, y):
        return (
            (x - self.view_offset[0]) * self.view_zoom + self.origin[0],
            (y - self.view_offset[1]) * self.view_zoom + self.origin[1]
        )

    def clear(self):
        self._tex_img = None

    def calculate_view_params(self, origin=None):
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

    def move(self, dx, dy):
        new_origin = [self.origin[0] - (dx * self.view_zoom),
                      self.origin[1] - (dy * self.view_zoom)]

        view_params = self.calculate_view_params(new_origin)
        if view_params != self.view_params:
            self.refresh_view(view_params=view_params)


    def zoom(self, cx, cy, dz):
        wsi_x, wsi_y = self.display_coords_to_wsi_coords(cx, cy)
        self.view_zoom = min(self.view_zoom * dz,
                                max(self.wsi.dimensions[0] / self.width,
                                    self.wsi.dimensions[1] / self.height))
        new_origin = [wsi_x - (cx * self.wsi_window_size[0] / self.width),
                      wsi_y - (cy * self.wsi_window_size[1] / self.height)]

        view_params = self.calculate_view_params(new_origin)
        if view_params != self.view_params:
            self.refresh_view(view_params=view_params)

    def refresh_view(self, view_params=None):

        if view_params is None:
            view_params = self.view_params
        else:
            self.view_params = view_params

        self.origin = tuple(view_params.top_left)
        region = self.wsi.slide.read_from_pyramid(**view_params)
        if region.bands == 4:
            region = region.flatten()  # removes alpha
        self.view = sf.slide.vips2numpy(region)

        # Normalize and finalize
        if self._normalizer:
            self.view = self._normalizer.transform(self.view)

        if (self._tex_obj is not None
           and ((abs(self._tex_obj.width - self.width) > 1)
                or (abs(self._tex_obj.height - self.height) > 1))):
            self.clear()

        # Refresh ROIs
        self.refresh_rois()

    def draw(self, max_w, max_h):
        if self._tex_img is not self.view:
            self._update_texture()
        if self._tex_obj is not None:
            pos = np.array([self.x_offset + max_w / 2, self.y_offset + max_h / 2])
            zoom = min(max_w / self._tex_obj.width, max_h / self._tex_obj.height)
            zoom = np.floor(zoom) if zoom >= 1 else zoom
            self._tex_obj.draw(pos=pos, zoom=zoom, align=0.5, rint=True)

    def refresh_rois(self):
        self.rois = []
        for roi in self.wsi.rois:
            c = np.copy(roi.coordinates)
            c[:, 0] = c[:, 0] - self.origin[0]
            c[:, 0] = c[:, 0] / self.view_zoom
            c[:, 0] = c[:, 0] + self.view_offset[0] + self.x_offset
            c[:, 1] = c[:, 1] - self.origin[1]
            c[:, 1] = c[:, 1] / self.view_zoom
            c[:, 1] = c[:, 1] + self.view_offset[1] + self.y_offset
            self.rois += [c]

    def render_rois(self):
        for roi in self.rois:
            gl_utils.draw_roi(roi, color=1, alpha=0.7, linewidth=5)
            gl_utils.draw_roi(roi, color=0, alpha=1, linewidth=3)


class TiledSlideViewer:

    def __init__(self, wsi, bilinear=True, mipmap=True):
        self._tex_img       = None
        self._tex_obj       = EasyDict(width=0, height=0)

        self.wsi            = wsi
        self.bilinear       = bilinear
        self.mipmap         = mipmap
        self.z              = 4 # Zoom (1 = full resolution, higher = zoomed out)
        self.x              = 0 # Top-left corner
        self.y              = 0 # Top-left corner
        self.max_w          = 0
        self.max_h          = 0
        self.tile_width     = 512
        self.grid           = None
        self._n_grid_x      = 0
        self._n_grid_y      = 0
        self.dim            = wsi.dimensions
        self.finished_q     = Queue()
        self.request_q      = Queue()
        self._abort         = False
        self._thread = threading.Thread(target=self._async_loader)
        self._thread.start()

    @property
    def extract_width(self):
        return self.tile_width * self.z

    def clear(self):
        self._tex_img = None

    def update(self, img):
        pass

    def zoom(self, zoom):
        self.z *= zoom
        log.debug("Zooming to: ", self.z)
        self._refresh_grid()

    def move(self, x, y):
        x *= self.z
        y *= self.z
        self.x = min(max(self.x + x, 0), self.dim[0] - self.max_w)
        self.y = min(max(self.y + y, 0), self.dim[1] - self.max_h)

    def set_position(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def load_from_slide(wsi_coord, window_size, target_size, wsi):
        region = wsi.slide.read_from_pyramid(
            top_left=wsi_coord,
            window_size=window_size,
            target_size=target_size
        )
        if region.bands == 4:
            region = region.flatten()
        return sf.slide.vips2numpy(region)

    def _empty_queues(self):
        log.debug("Emptying queues")
        self._abort = True
        self.request_q.put(None)
        self.finished_q.put(None)
        while self.request_q.qsize():
            log.debug("emptying from request q")
            self.request_q.get()
        while self.finished_q.qsize():
            log.debug("emptying finished from request q")
            if self.finished_q.get() is None:
                break
        self._abort = False
        log.debug("Empty q finished")

    def _refresh_grid(self):
        # Empty queues
        self._empty_queues()

        # Populate with images
        self._n_grid_x = int(self.dim[0] // self.extract_width)
        self._n_grid_y = int(self.dim[1] // self.extract_width)
        self.grid = np.zeros((self._n_grid_x, self._n_grid_y)).tolist()
        log.debug("Refresh grid complete")

    def _async_loader(self):
        #ctx = mp.get_context('spawn')
        #pool = ctx.Pool(os.cpu_count())
        pool = mp.dummy.Pool(os.cpu_count())
        while not self._abort:
            requests = []
            log.debug("inside async loop")
            if self.request_q.qsize() > 1:
                while self.request_q.qsize():
                    request = self.request_q.get()
                    if request is not None:
                        requests += [request]
            else:
                request = self.request_q.get()
                if request is not None:
                    requests += [request]
            if not len(requests):
                continue

            dims = [r[1] for r in requests]
            for idx, image in enumerate(pool.imap(partial(self.load_from_slide,
                                                          window_size=(self.extract_width, self.extract_width),
                                                          target_size=(self.tile_width, self.tile_width),
                                                          wsi=self.wsi),
                                                  dims)):
                i, j = requests[idx][0], requests[idx][1]
                log.debug(f"putting {(i, j)} into finished queue")
                self.finished_q.put((requests[idx][0], requests[idx][1], image))
        pool.close()

    def _load_from_queue(self):
        while self.finished_q.qsize():
            log.debug("inside finished Q loader")

            (i, j), (wsi_x, wsi_y), image = self.finished_q.get()
            log.debug("LOADED ({}) from finished q".format((i, j)))
            self.grid[i][j] = Tile(image, wsi_x, wsi_y)

    def draw(self, max_w, max_h, x_offset=0, y_offset=0):
        if max_w != self.max_w or max_h != self.max_h:
            self.max_w = max_w
            self.max_h = max_h
            self._refresh_grid()

        log.info("pre-queue")
        self._load_from_queue()
        log.info("POST-queue")

        if self.grid is not None:
            draw_size = np.array([self.tile_width, self.tile_width])

            # For now, show entire grid
            min_grid_x = int(self.x // self.extract_width) - 1
            min_grid_y = int(self.y // self.extract_width) - 1
            max_grid_x = int((self.x / self.z + self.max_w + self.tile_width) // self.tile_width) + 1
            max_grid_y = int((self.y / self.z + self.max_h + self.tile_width) // self.tile_width) + 1

            # Enforce limits
            min_grid_x = max(min_grid_x, 0)
            min_grid_y = max(min_grid_y, 0)
            max_grid_x = min(max_grid_x, self._n_grid_x)
            max_grid_y = min(max_grid_y, self._n_grid_y)

            log.info("pre-GRID")
            for i in range(min_grid_x-1, max_grid_x+1):
                for j in range(min_grid_y-1, max_grid_y+1):
                    try:
                        tile = self.grid[i][j]
                    except IndexError:
                        continue
                    in_view = ((min_grid_x <= i < max_grid_x)
                               and (min_grid_y <= j < max_grid_y))
                    #if tile and not in_view:
                    #    #self.grid[i][j] = None
                    #    pass
                    if in_view:
                        grid_wsi_x = (i * self.extract_width)
                        grid_wsi_y = (j * self.extract_width)
                        screen_x = x_offset + (grid_wsi_x - self.x) / self.z
                        screen_y = y_offset + (grid_wsi_y - self.y) / self.z
                        if isinstance(tile, Tile):
                            #log.info("drawing")
                            pos = np.array([screen_x, screen_y])
                            tile.draw(pos, draw_size)
                        elif self.grid[i][j] == 0:
                            log.info(f"putting {(i, j)} into queue")
                            self.request_q.put(((i, j), (grid_wsi_x, grid_wsi_y)))
                            self.grid[i][j] = 1 # Signify that async request has started, do not request again

    def draw_box(self, pos, size):

        gl_utils.draw_rect(pos=pos, size=size, color=[1, 0, 0], mode=gl.GL_LINE_LOOP)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glLineWidth(1)

class Tile:
    def __init__(self, image, wsi_x, wsi_y, bilinear=False, mipmap=False):
        self.x = wsi_x
        self.y = wsi_y
        self._tex_obj = gl_utils.Texture(image=image,
                                         bilinear=bilinear,
                                         mipmap=mipmap)

    def draw(self, pos, size):
        self._tex_obj.draw(pos=pos, zoom=1, align=0.5, rint=True, anchor='topleft')

        #gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        #gl.glLineWidth(3)
        #gl_utils.draw_rect(pos=pos, size=size, color=[1, 0, 0], mode=gl.GL_LINE_LOOP)
        #gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        #gl.glLineWidth(1)