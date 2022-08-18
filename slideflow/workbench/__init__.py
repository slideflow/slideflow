import multiprocessing
import pyvips
import numpy as np
import imgui
import OpenGL.GL as gl
from slideflow.workbench.gui_utils import imgui_window
from slideflow.workbench.gui_utils import imgui_utils
from slideflow.workbench.gui_utils import gl_utils
from slideflow.workbench.gui_utils import text_utils
from slideflow.workbench import slide_renderer as renderer
from slideflow.workbench import project_widget
from slideflow.workbench import slide_widget
from slideflow.workbench import model_widget
from slideflow.workbench import heatmap_widget
from slideflow.workbench import performance_widget
from slideflow.workbench import capture_widget

import slideflow as sf
import slideflow.grad
from slideflow.workbench.utils import EasyDict

if sf.backend() == 'tensorflow':
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

#----------------------------------------------------------------------------

class Workbench(imgui_window.ImguiWindow):
    def __init__(self, capture_dir=None):
        super().__init__(title=f'Slideflow Workbench ({sf.__version__})', window_width=3840, window_height=2160)

        # Internals.
        self._last_error_print  = None
        self._async_renderer    = AsyncRenderer(self)
        self._defer_rendering   = 0
        self._tex_img           = None
        self._tex_obj           = None
        self._norm_tex_img      = None
        self._norm_tex_obj      = None
        self._thumb_tex_img     = None
        self._thumb_tex_obj     = None
        self._heatmap_tex_img   = None
        self._heatmap_tex_obj   = None
        self._wsi_tex_img       = None
        self._wsi_tex_obj       = None
        self._heatmap_overlay_tex_img   = None
        self._heatmap_overlay_tex_obj   = None
        self._predictions       = None
        self._model             = None
        self._model_config      = None
        self._normalizer        = None
        self._normalize_wsi     = False
        self._use_model         = False
        self._use_uncertainty   = False
        self._gan_config        = None
        self._uncertainty       = None
        self._content_width     = None
        self._slide_path        = None
        self._refresh_thumb_flag= False
        self._show_overlay      = False

        # Widget interface.
        self.wsi                = None
        self.wsi_thumb          = None
        self.wsi_window_size    = None
        self.thumb              = None
        self.thumb_zoom         = None
        self.thumb_origin       = None
        self.thumb_offset       = None
        self.box_x              = None
        self.box_y              = None
        self.tile_px            = None
        self.tile_um            = None
        self.heatmap            = None
        self.rendered_heatmap   = None
        self.overlay_heatmap    = None
        self.rendered_qc        = None
        self.overlay_qc         = None
        self.args               = EasyDict()
        self.result             = EasyDict()
        self.pane_w             = 0
        self.label_w            = 0
        self.button_w           = 0
        self.x                  = None
        self.y                  = None

        # Widgets.
        self.project_widget     = project_widget.ProjectWidget(self)
        self.slide_widget       = slide_widget.SlideWidget(self)
        self.model_widget       = model_widget.ModelWidget(self)
        self.heatmap_widget     = heatmap_widget.HeatmapWidget(self)
        self.perf_widget        = performance_widget.PerformanceWidget(self)
        self.capture_widget     = capture_widget.CaptureWidget(self)

        if capture_dir is not None:
            self.capture_widget.path = capture_dir

        # Initialize window.
        self.set_position(0, 0)
        self._adjust_font_size()
        self.skip_frame() # Layout may change after first frame.
        self.load_slide('')

    def close(self):
        super().close()
        if self._async_renderer is not None:
            self._async_renderer.close()
            self._async_renderer = None

    def add_recent_slide(self, slide, ignore_errors=False):
        self.slide_widget.add_recent(slide, ignore_errors=ignore_errors)

    def load_project(self, project, ignore_errors=False):
        self.project_widget.load(project, ignore_errors=ignore_errors)

    def load_slide(self, slide, ignore_errors=False):
        self.slide_widget.load(slide, ignore_errors=ignore_errors)

    def load_model(self, model, ignore_errors=False):
        self.model_widget.load(model)

    def print_error(self, error):
        error = str(error)
        if error != self._last_error_print:
            print('\n' + error + '\n')
            self._last_error_print = error

    def defer_rendering(self, num_frames=1):
        self._defer_rendering = max(self._defer_rendering, num_frames)

    def clear_result(self):
        self._async_renderer.clear_result()

    def set_async(self, is_async):
        if is_async != self._async_renderer.is_async:
            self._async_renderer.set_async(is_async)
            self.clear_result()
            if 'image' in self.result:
                self.result.message = 'Switching rendering process...'
                self.defer_rendering()

    def _adjust_font_size(self):
        old = self.font_size
        self.set_font_size(min(self.content_width / 120, self.content_height / 60))
        if self.font_size != old:
            self.skip_frame() # Layout changed.

    def has_uq(self):
        return (self._model is not None
                and self._model_config is not None
                and 'uq' in self._model_config['hp']
                and self._model_config['hp']['uq'])

    def reset_thumb(self, width):
        if self.wsi:
            max_w = (self.content_width - self.pane_w)
            max_h = self.content_height
            slide_hw_ratio = (self.wsi.dimensions[1] / self.wsi.dimensions[0])

            if (max_h / max_w) < slide_hw_ratio:
                _width = int(max_h // slide_hw_ratio)
            else:
                _width = int(width)

            self.thumb = np.asarray(self.wsi.thumb(width=_width))
            if self.thumb.shape[-1] == 4:
                self.thumb = self.thumb[:, :, 0:3]
            if self._normalizer and self._normalize_wsi:
                self.thumb = self._normalizer.transform(self.thumb)
            self.thumb_zoom = self.wsi.dimensions[0] / self.thumb.shape[1]
            self.thumb_offset = (
                (max_w - self.thumb.shape[1]) / 2,
                (max_h - self.thumb.shape[0]) / 2
            )
            self.thumb_origin = [0, 0]
            self.wsi_window_size = None

    def wsi_coords_to_display_coords(self, x, y):
        return (
            int(((x - self.thumb_origin[0]) / self.thumb_zoom) + self.thumb_offset[0]),
            int(((y - self.thumb_origin[1]) / self.thumb_zoom) + self.thumb_offset[1])
        )

    def display_coords_to_wsi_coords(self, x, y):
        return (
            int((x - self.thumb_offset[0]) * self.thumb_zoom + self.thumb_origin[0]),
            int((y - self.thumb_offset[1]) * self.thumb_zoom + self.thumb_origin[1])
        )

    def draw_frame(self):
        self.begin_frame()
        self.args = EasyDict()
        self.pane_w = self.font_size * 45
        self.button_w = self.font_size * 5
        self.label_w = round(self.font_size * 4.5)
        max_w = self.content_width - self.pane_w
        max_h = self.content_height

        # Begin control pane.
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.pane_w, self.content_height)
        imgui.begin('##control_pane', closable=False, flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))

        # Widgets.
        expanded, _visible = imgui_utils.collapsing_header('Slideflow project', default=True)
        self.project_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Whole-slide image', default=True)
        self.slide_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Prediction & saliency', default=True)
        self.model_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Heatmap', default=True)
        self.heatmap_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header('Performance & capture', default=True)
        self.perf_widget(expanded)
        self.capture_widget(expanded)

        # Detect mouse dragging in the thumbnail display.
        clicking, cx, cy, wheel = imgui_utils.click_hidden_window('##result_area', x=self.pane_w, y=0, width=self.content_width-self.pane_w, height=self.content_height, mouse_idx=1)
        dragging, dx, dy = imgui_utils.drag_hidden_window('##result_area', x=self.pane_w, y=0, width=self.content_width-self.pane_w, height=self.content_height)

        if self.thumb is not None:
            wsi_x, wsi_y = self.display_coords_to_wsi_coords(cx, cy)

            # Update thumb focus location & zoom values
            # If shift-dragging or scrolling.
            if dragging:
                self.thumb_origin[0] -= (dx * self.thumb_zoom)
                self.thumb_origin[1] -= (dy * self.thumb_zoom)
            if wheel > 0:
                self.thumb_zoom /= 1.5
            if wheel < 0:
                self.thumb_zoom *= 1.5

            window_size = (int(max_w * self.thumb_zoom),
                           int(max_h * self.thumb_zoom))

            if wheel:
                self.thumb_origin[0] = wsi_x - (cx * window_size[0] / max_w)
                self.thumb_origin[1] = wsi_y - (cy * window_size[1] / max_h)
            if wheel or dragging or self._refresh_thumb_flag:
                self._refresh_thumb_flag = False
                try:
                    # Enforce boundary limits.
                    self.thumb_origin = [max(self.thumb_origin[0], 0), max(self.thumb_origin[1], 0)]
                    self.thumb_origin = [min(self.thumb_origin[0], self.wsi.dimensions[0]-window_size[0]), min(self.thumb_origin[1], self.wsi.dimensions[1]-window_size[1])]

                    target_size = (max_w, max_h)
                    region = self.wsi.slide.read_from_pyramid(
                        top_left=self.thumb_origin,
                        window_size=window_size,
                        target_size=target_size)
                    if region.bands == 4:
                        region = region.flatten()  # removes alpha
                    self.thumb = sf.slide.vips2numpy(region)
                    if self._normalizer and self._normalize_wsi:
                        self.thumb = self._normalizer.transform(self.thumb)
                    self.wsi_window_size = window_size
                except pyvips.error.Error as e:
                    self.reset_thumb(max_w)

        # Re-generate thumbnail if the window size changed.
        if self._content_width != self.content_width:
            self.reset_thumb(max_w)
            self._content_width = self.content_width

        # Display thumbnail.
        if self.thumb is not None:

            # Render thumbnail.
            t_pos = np.array([self.pane_w + max_w / 2, max_h / 2])
            if self._thumb_tex_img is not self.thumb:
                self._thumb_tex_img = self.thumb
                if self._thumb_tex_obj is None or not self._thumb_tex_obj.is_compatible(image=self._thumb_tex_img):
                    self._thumb_tex_obj = gl_utils.Texture(image=self._thumb_tex_img, bilinear=False, mipmap=False)
                else:
                    self._thumb_tex_obj.update(self._thumb_tex_img)
            t_zoom = min(max_w / self._thumb_tex_obj.width, max_h / self._thumb_tex_obj.height)
            t_zoom = np.floor(t_zoom) if t_zoom >= 1 else t_zoom
            self._thumb_tex_obj.draw(pos=t_pos, zoom=t_zoom, align=0.5, rint=True)

            # Render overlay heatmap.
            if self.overlay_heatmap is not None and self._show_overlay:
                if self._heatmap_overlay_tex_img is not self.overlay_heatmap:
                    self._heatmap_overlay_tex_img = self.overlay_heatmap
                    if self._heatmap_overlay_tex_obj is None or not self._heatmap_overlay_tex_obj.is_compatible(image=self._heatmap_overlay_tex_img):
                        self._heatmap_overlay_tex_obj = gl_utils.Texture(image=self._heatmap_overlay_tex_img, bilinear=False, mipmap=False)
                    else:
                        self._heatmap_overlay_tex_obj.update(self._heatmap_overlay_tex_img)
                h_pos_x, h_pos_y = self.wsi_coords_to_display_coords(0, 0)
                h_pos = (h_pos_x + self.pane_w + (self.wsi.dimensions[0] / self.thumb_zoom) / 2, h_pos_y + (self.wsi.dimensions[1] / self.thumb_zoom) / 2)
                h_zoom = (self.wsi.dimensions[0] / self._heatmap_overlay_tex_obj.width) / self.thumb_zoom
                self._heatmap_overlay_tex_obj.draw(pos=h_pos, zoom=h_zoom, align=0.5, rint=True)

            # Calculate thumbnail zoom and offset.
            self.thumb_offset = ((max_w - self.thumb.shape[1]) / 2, (max_h - self.thumb.shape[0]) / 2)
            thumb_max_x = self.thumb_offset[0] + self.thumb.shape[1]
            thumb_max_y = self.thumb_offset[1] + self.thumb.shape[0]

            # Calculate location for model display.
            if self._model and clicking and not dragging and (self.thumb_offset[0] <= cx <= thumb_max_x) and (self.thumb_offset[1] <= cy <= thumb_max_y):
                self.x = wsi_x - (self.wsi.full_extract_px/2)
                self.y = wsi_y - (self.wsi.full_extract_px/2)

            # Update box location.
            if self.x is not None and self.y is not None:
                if clicking or dragging or wheel:
                    self.box_x, self.box_y = self.wsi_coords_to_display_coords(self.x, self.y)
                    self.box_x += self.pane_w
                tw = self.wsi.full_extract_px // self.thumb_zoom

                # Draw box on main display.
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
                gl.glLineWidth(3)
                box_pos = np.array([self.box_x, self.box_y])
                gl_utils.draw_rect(pos=box_pos, size=np.array([tw, tw]), color=[1, 0, 0], mode=gl.GL_LINE_LOOP)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
                gl.glLineWidth(3)

        # Render WSI thumbnail in the widget.
        if self.wsi_thumb is not None:
            if self._wsi_tex_img is not self.wsi_thumb:
                self._wsi_tex_img = self.wsi_thumb
                if self._wsi_tex_obj is None or not self._wsi_tex_obj.is_compatible(image=self._wsi_tex_img):
                    self._wsi_tex_obj = gl_utils.Texture(image=self._wsi_tex_img, bilinear=False, mipmap=False)
                else:
                    self._wsi_tex_obj.update(self._wsi_tex_img)

        # Render.
        self.args.x = self.x
        self.args.y = self.y
        if self._model_config is not None and self._use_model and 'img_format' in self._model_config:
            self.args.img_format = self._model_config['img_format']
        if self.is_skipping_frames():
            pass
        elif self._defer_rendering > 0:
            self._defer_rendering -= 1
        #elif self.args.pkl is not None:
        else:
            self._async_renderer.set_args(**self.args)
            result = self._async_renderer.get_result()
            if result is not None:
                self.result = result

        # Display input image.
        middle_pos = np.array([self.pane_w + max_w/2, max_h/2])
        if 'image' in self.result:
            if self._tex_img is not self.result.image:
                self._tex_img = self.result.image
                if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                    self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=False, mipmap=False)
                else:
                    self._tex_obj.update(self._tex_img)
        if 'normalized' in self.result:
            if self._norm_tex_img is not self.result.normalized:
                self._norm_tex_img = self.result.normalized
                if self._norm_tex_obj is None or not self._norm_tex_obj.is_compatible(image=self._norm_tex_img):
                    self._norm_tex_obj = gl_utils.Texture(image=self._norm_tex_img, bilinear=False, mipmap=False)
                else:
                    self._norm_tex_obj.update(self._norm_tex_img)
        if 'error' in self.result:
            self.print_error(self.result.error)
            if 'message' not in self.result:
                self.result.message = str(self.result.error)
        if 'message' in self.result:
            tex = text_utils.get_texture(self.result.message, size=self.font_size, max_width=max_w, max_height=max_h, outline=2)
            tex.draw(pos=middle_pos, align=0.5, rint=True, color=1)

        # Display rendered (non-transparent) heatmap in widget.
        # Render overlay heatmap.
        if self.heatmap:
            if self._heatmap_tex_img is not self.rendered_heatmap:
                self._heatmap_tex_img = self.rendered_heatmap
                if self._heatmap_tex_obj is None or not self._heatmap_tex_obj.is_compatible(image=self._heatmap_tex_img):
                    self._heatmap_tex_obj = gl_utils.Texture(image=self._heatmap_tex_img, bilinear=False, mipmap=False)
                else:
                    self._heatmap_tex_obj.update(self._heatmap_tex_img)

        # End frame.
        self._adjust_font_size()
        imgui.end()
        self.end_frame()

#----------------------------------------------------------------------------

class AsyncRenderer:
    def __init__(self, visualizer):
        self._visualizer    = visualizer
        self._closed        = False
        self._is_async      = False
        self._cur_args      = None
        self._cur_result    = None
        self._cur_stamp     = 0
        self._renderer_obj  = None
        self._args_queue    = None
        self._result_queue  = None
        self._process       = None

    def close(self):
        self._closed = True
        self._renderer_obj = None
        if self._process is not None:
            self._process.terminate()
        self._process = None
        self._args_queue = None
        self._result_queue = None

    @property
    def is_async(self):
        return self._is_async

    def set_async(self, is_async):
        self._is_async = is_async

    def set_args(self, **args):
        assert not self._closed
        if args != self._cur_args:
            if self._is_async:
                self._set_args_async(**args)
            else:
                self._set_args_sync(**args)
            self._cur_args = args

    def _set_args_async(self, **args):
        if self._process is None:
            self._args_queue = multiprocessing.Queue()
            self._result_queue = multiprocessing.Queue()
            try:
                multiprocessing.set_start_method('spawn')
            except RuntimeError:
                pass
            self._process = multiprocessing.Process(target=self._process_fn, args=(self._args_queue, self._result_queue), daemon=True)
            self._process.start()
        self._args_queue.put([args, self._cur_stamp])

    def _set_args_sync(self, **args):
        if self._renderer_obj is None:
            self._renderer_obj = renderer.Renderer(self._visualizer)
        self._cur_result = self._renderer_obj.render(**args)

    def get_result(self):
        assert not self._closed
        if self._result_queue is not None:
            while self._result_queue.qsize() > 0:
                result, stamp = self._result_queue.get()
                if stamp == self._cur_stamp:
                    self._cur_result = result
        return self._cur_result

    def clear_result(self):
        assert not self._closed
        self._cur_args = None
        self._cur_result = None
        self._cur_stamp += 1

    @staticmethod
    def _process_fn(args_queue, result_queue):
        renderer_obj = renderer.Renderer()
        cur_args = None
        cur_stamp = None
        while True:
            args, stamp = args_queue.get()
            while args_queue.qsize() > 0:
                args, stamp = args_queue.get()
            if args != cur_args or stamp != cur_stamp:
                result = renderer_obj.render(**args)
                if 'error' in result:
                    result.error = renderer.CapturedException(result.error)
                result_queue.put([result, stamp])
                cur_args = args
                cur_stamp = stamp