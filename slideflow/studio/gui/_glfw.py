import time
import glfw
import imgui
import imgui.integrations.glfw
from typing import Tuple, Optional
from imgui.integrations import compute_fb_scale
from imgui.integrations.opengl import FixedPipelineRenderer
import OpenGL.GL as gl
from . import gl_utils

#----------------------------------------------------------------------------

class GlfwWindow:
    """GLFW window manager for OpenGL-enabled display."""

    def __init__(
        self,
        *,
        title: str = 'GlfwWindow',
        window_width: int = 1920,
        window_height: int = 1080,
        deferred_show: bool = True,
        background: Optional[Tuple[float, float, float, float]] = None
    ) -> None:

        if background is None:
            background = (0, 0, 0, 1)
        self._glfw_window           = None
        self._drawing_frame         = False
        self._frame_start_time      = None
        self._frame_delta           = 0
        self._fps_limit             = None
        self._vsync                 = None
        self._skip_frames           = 0
        self._deferred_show         = deferred_show
        self._exit_trigger          = False
        self._drag_and_drop_paths   = None
        self._capture_next_frame    = False
        self._captured_frame        = None
        self._shift_down            = False
        self._control_down          = False
        self._is_fullscreen         = False
        self._background_color      = background

        # Create window.
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, False)
        self._glfw_window = glfw.create_window(width=window_width, height=window_height, title=title, monitor=None, share=None)
        self._attach_glfw_callbacks()
        self.make_context_current()
        print(f"Using OpenGL version {gl.glGetString(gl.GL_VERSION)}")

        # Adjust window.
        self.set_vsync(True)
        self.update_window_size()
        self.set_window_size(window_width, window_height)
        if not self._deferred_show:
            glfw.show_window(self._glfw_window)

    def close(self):
        if self._drawing_frame:
            self.end_frame()
        if self._glfw_window is not None:
            glfw.destroy_window(self._glfw_window)
            self._glfw_window = None

    def __del__(self):
        try:
            self.close()
        except:
            pass

    @property
    def window_width(self):
        return self.content_width

    @property
    def window_height(self):
        return self.content_height + self.title_bar_height

    @property
    def pixel_ratio(self):
        return self.content_frame_width / self.content_width

    @property
    def monitor_width(self):
        _, _, width, _height = glfw.get_monitor_workarea(glfw.get_primary_monitor())
        return width

    @property
    def monitor_height(self):
        _, _, _width, height = glfw.get_monitor_workarea(glfw.get_primary_monitor())
        return height

    @property
    def frame_delta(self):
        return self._frame_delta

    def set_window_icon(self, image):
        glfw.set_window_icon(self._glfw_window, 1, image)

    def update_window_size(self):
        ws = glfw.get_window_size(self._glfw_window)
        if not ws[0]:
            # Do not update if the window size is 0.
            return
        fs = glfw.get_framebuffer_size(self._glfw_window)
        _l, top, _r, _bottom = glfw.get_window_frame_size(self._glfw_window)
        self.title_bar_height = top
        self.content_width, self.content_height = ws[0], ws[1]
        self.content_frame_width, self.content_frame_height = fs[0], fs[1]

    def set_fullscreen(self):
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        glfw.set_window_monitor(self._glfw_window, glfw.get_primary_monitor(), width=mode.size.width, height=mode.size.height, xpos=0, ypos=0, refresh_rate=60)
        self._is_fullscreen = True

    def set_windowed(self):
        glfw.set_window_monitor(self._glfw_window, monitor=None, width=1600, height=900, xpos=0, ypos=0, refresh_rate=60)
        self._is_fullscreen = False

    def toggle_fullscreen(self):
        if not self._is_fullscreen:
            self.set_fullscreen()
        else:
            self.set_windowed()

    def set_title(self, title):
        glfw.set_window_title(self._glfw_window, title)

    def set_window_size(self, width, height):
        mw, mh = self.monitor_width, self.monitor_height
        if mw and mh:
            width = min(width, mw)
            height = min(height, mh)
        glfw.set_window_size(self._glfw_window, width, max(height - self.title_bar_height, 0))
        if width == mw and height == mh:
            self.maximize()

    def set_content_size(self, width, height):
        self.set_window_size(width, height + self.title_bar_height)

    def maximize(self):
        glfw.maximize_window(self._glfw_window)

    def set_position(self, x, y):
        glfw.set_window_pos(self._glfw_window, x, y + self.title_bar_height)

    def center(self):
        self.set_position((self.monitor_width - self.window_width) // 2, (self.monitor_height - self.window_height) // 2)

    def set_vsync(self, vsync):
        vsync = bool(vsync)
        if vsync != self._vsync:
            glfw.swap_interval(1 if vsync else 0)
            self._vsync = vsync

    def set_fps_limit(self, fps_limit):
        self._fps_limit = int(fps_limit)

    def should_close(self):
        return glfw.window_should_close(self._glfw_window) or self._exit_trigger

    def run(self):
        while not self.should_close():
            self.draw_frame()
        self.close()

    def skip_frame(self):
        self.skip_frames(1)

    def skip_frames(self, num): # Do not update window for the next N frames.
        self._skip_frames = max(self._skip_frames, int(num))

    def is_skipping_frames(self):
        return self._skip_frames > 0

    def capture_next_frame(self):
        self._capture_next_frame = True

    def pop_captured_frame(self):
        frame = self._captured_frame
        self._captured_frame = None
        return frame

    def pop_drag_and_drop_paths(self):
        paths = self._drag_and_drop_paths
        self._drag_and_drop_paths = None
        return paths

    def draw_frame(self): # To be overridden by subclass.
        self.begin_frame()
        # Rendering code goes here.
        self.end_frame()

    def make_context_current(self):
        if self._glfw_window is not None:
            glfw.make_context_current(self._glfw_window)

    def begin_frame(self):
        # End previous frame.
        if self._drawing_frame:
            self.end_frame()

        # Update window size measurements
        self.update_window_size()

        # Apply FPS limit.
        if self._frame_start_time is not None and self._fps_limit is not None:
            delay = self._frame_start_time - time.perf_counter() + 1 / self._fps_limit
            if delay > 0:
                time.sleep(delay)
        cur_time = time.perf_counter()
        if self._frame_start_time is not None:
            self._frame_delta = cur_time - self._frame_start_time
        self._frame_start_time = cur_time

        # Process events.
        glfw.poll_events()

        # Begin frame.
        self._drawing_frame = True
        self.make_context_current()

        # Initialize GL state.
        gl.glViewport(0, 0, self.content_frame_width, self.content_frame_height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glTranslate(-1, 1, 0)
        gl.glScale(2 / max(self.content_frame_width, 1), -2 / max(self.content_frame_height, 1), 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA) # Pre-multiplied alpha.

        # Clear.
        gl.glClearColor(*self._background_color)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    def end_frame(self):
        assert self._drawing_frame
        self._drawing_frame = False

        # Skip frames if requested.
        if self._skip_frames > 0:
            self._skip_frames -= 1
            return

        # Capture frame if requested.
        if self._capture_next_frame:
            self._captured_frame = gl_utils.pixel_capture(self.content_frame_width, self.content_frame_height)
            self._capture_next_frame = False

        # Update window.
        if self._deferred_show:
            glfw.show_window(self._glfw_window)
            self._deferred_show = False
        glfw.swap_buffers(self._glfw_window)

    def _attach_glfw_callbacks(self):
        glfw.set_key_callback(self._glfw_window, self._glfw_key_callback)
        glfw.set_drop_callback(self._glfw_window, self._glfw_drop_callback)

    def _glfw_key_callback(self, _window, key, _scancode, action, _mods):

        # Key modifiers
        if action == glfw.PRESS and key in (glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL):
            self._control_down = True
        if action == glfw.RELEASE and key in (glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL):
            self._control_down = False
        if action == glfw.PRESS and key in (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT):
            self._shift_down = True
        if action == glfw.RELEASE and key in (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT):
            self._shift_down = False

        # Key combinations
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            self.set_windowed()
        if self._control_down and action == glfw.PRESS and key == glfw.KEY_F:
            self.toggle_fullscreen()
        if self._control_down and action == glfw.PRESS and key == glfw.KEY_EQUAL:
            self.increase_font_size()
        if self._control_down and action == glfw.PRESS and key == glfw.KEY_MINUS:
            self.decrease_font_size()


    def _glfw_drop_callback(self, _window, paths):
        self._drag_and_drop_paths = paths

#----------------------------------------------------------------------------

class GlfwRenderer(FixedPipelineRenderer):
    """Wrapper class for GlfwRenderer to add support for OpenGL 2.0"""

    def __init__(self, window, attach_callbacks=True):
        super().__init__()
        self.window = window

        if attach_callbacks:
            glfw.set_key_callback(self.window, self.keyboard_callback)
            glfw.set_cursor_pos_callback(self.window, self.mouse_callback)
            glfw.set_window_size_callback(self.window, self.resize_callback)
            glfw.set_char_callback(self.window, self.char_callback)
            glfw.set_scroll_callback(self.window, self.scroll_callback)

        self.io.display_size = glfw.get_framebuffer_size(self.window)
        self.io.get_clipboard_text_fn = self._get_clipboard_text
        self.io.set_clipboard_text_fn = self._set_clipboard_text

        self._map_keys()
        self._gui_time = None
        self.mouse_wheel_multiplier = 1

    def _get_clipboard_text(self):
        return glfw.get_clipboard_string(self.window)

    def _set_clipboard_text(self, text):
        glfw.set_clipboard_string(self.window, text)

    def _map_keys(self):
        key_map = self.io.key_map

        key_map[imgui.KEY_TAB] = glfw.KEY_TAB
        key_map[imgui.KEY_LEFT_ARROW] = glfw.KEY_LEFT
        key_map[imgui.KEY_RIGHT_ARROW] = glfw.KEY_RIGHT
        key_map[imgui.KEY_UP_ARROW] = glfw.KEY_UP
        key_map[imgui.KEY_DOWN_ARROW] = glfw.KEY_DOWN
        key_map[imgui.KEY_PAGE_UP] = glfw.KEY_PAGE_UP
        key_map[imgui.KEY_PAGE_DOWN] = glfw.KEY_PAGE_DOWN
        key_map[imgui.KEY_HOME] = glfw.KEY_HOME
        key_map[imgui.KEY_END] = glfw.KEY_END
        key_map[imgui.KEY_DELETE] = glfw.KEY_DELETE
        key_map[imgui.KEY_BACKSPACE] = glfw.KEY_BACKSPACE
        key_map[imgui.KEY_ENTER] = glfw.KEY_ENTER
        key_map[imgui.KEY_ESCAPE] = glfw.KEY_ESCAPE
        key_map[imgui.KEY_A] = glfw.KEY_A
        key_map[imgui.KEY_C] = glfw.KEY_C
        key_map[imgui.KEY_V] = glfw.KEY_V
        key_map[imgui.KEY_X] = glfw.KEY_X
        key_map[imgui.KEY_Y] = glfw.KEY_Y
        key_map[imgui.KEY_Z] = glfw.KEY_Z

    def keyboard_callback(self, window, key, scancode, action, mods):
        # perf: local for faster access
        io = self.io

        if action == glfw.PRESS:
            io.keys_down[key] = True
        elif action == glfw.RELEASE:
            io.keys_down[key] = False

        io.key_ctrl = (
            io.keys_down[glfw.KEY_LEFT_CONTROL] or
            io.keys_down[glfw.KEY_RIGHT_CONTROL]
        )

        io.key_alt = (
            io.keys_down[glfw.KEY_LEFT_ALT] or
            io.keys_down[glfw.KEY_RIGHT_ALT]
        )

        io.key_shift = (
            io.keys_down[glfw.KEY_LEFT_SHIFT] or
            io.keys_down[glfw.KEY_RIGHT_SHIFT]
        )

        io.key_super = (
            io.keys_down[glfw.KEY_LEFT_SUPER] or
            io.keys_down[glfw.KEY_RIGHT_SUPER]
        )

    def char_callback(self, window, char):
        io = imgui.get_io()

        if 0 < char < 0x10000:
            io.add_input_character(char)

    def resize_callback(self, window, width, height):
        self.io.display_size = width, height

    def mouse_callback(self, *args, **kwargs):
        pass

    def scroll_callback(self, window, x_offset, y_offset):
        self.io.mouse_wheel_horizontal = x_offset
        self.io.mouse_wheel = y_offset
        self.io.mouse_wheel += y_offset * self.mouse_wheel_multiplier

    def process_inputs(self):
        io = imgui.get_io()

        window_size = glfw.get_window_size(self.window)
        fb_size = glfw.get_framebuffer_size(self.window)

        io.display_size = window_size
        io.display_fb_scale = compute_fb_scale(window_size, fb_size)
        io.delta_time = 1.0/60

        if glfw.get_window_attrib(self.window, glfw.FOCUSED):
            io.mouse_pos = glfw.get_cursor_pos(self.window)
        else:
            io.mouse_pos = -1, -1

        io.mouse_down[0] = glfw.get_mouse_button(self.window, 0)
        io.mouse_down[1] = glfw.get_mouse_button(self.window, 1)
        io.mouse_down[2] = glfw.get_mouse_button(self.window, 2)

        current_time = glfw.get_time()

        if self._gui_time:
            self.io.delta_time = current_time - self._gui_time
        else:
            self.io.delta_time = 1. / 60.

        self._gui_time = current_time
