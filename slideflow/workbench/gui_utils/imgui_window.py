# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import glfw
import time
import contextlib
import imgui
import imgui.integrations.glfw
from imgui.integrations import compute_fb_scale
from imgui.integrations.opengl import FixedPipelineRenderer

from . import glfw_window
from . import imgui_utils
from . import text_utils
from . import gl_utils

#----------------------------------------------------------------------------

class ImguiWindow(glfw_window.GlfwWindow):
    def __init__(
        self,
        *,
        title='ImguiWindow',
        font=None,
        font_sizes=range(14,36),
        **glfw_kwargs
    ):
        if font is None:
            font = text_utils.get_default_font()
            font_bold = text_utils.get_default_font_bold()
        font_sizes = {int(size) for size in font_sizes}
        super().__init__(title=title, **glfw_kwargs)

        # Init fields.
        self._imgui_context  = None
        self._imgui_renderer = None
        self._imgui_fonts    = None
        self._imgui_fonts_bold = None
        self._cur_font_size  = max(font_sizes)
        self._font_scaling   = self.pixel_ratio
        self._toasts         = []

        # Delete leftover imgui.ini to avoid unexpected behavior.
        if os.path.isfile('imgui.ini'):
            os.remove('imgui.ini')

        # Init ImGui.
        self._imgui_context = imgui.create_context()
        self._imgui_renderer = _GlfwRenderer(self._glfw_window)
        self._attach_glfw_callbacks()
        imgui.get_io().ini_saving_rate = 0 # Disable creating imgui.ini at runtime.
        imgui.get_io().mouse_drag_threshold = 0 # Improve behavior with imgui_utils.drag_custom().
        self._imgui_fonts = {size: imgui.get_io().fonts.add_font_from_file_ttf(font, int(size * self._font_scaling)) for size in font_sizes}
        self._imgui_fonts_bold = {size: imgui.get_io().fonts.add_font_from_file_ttf(font_bold, int(size * self._font_scaling)) for size in font_sizes}
        self._imgui_renderer.refresh_font_texture()
        imgui.get_io().font_global_scale = 1 / self._font_scaling

        # Init icons.
        self._icon_textures = {
            name: gl_utils.Texture(image=icon)
            for name, icon in text_utils.icons().items()
        }

    @property
    def font_size(self):
        return self._cur_font_size + 2  # Adjustment for DroidSans

    @property
    def gl_font_size(self):
        return int(self.font_size * self.pixel_ratio)

    @property
    def spacing(self):
        return round(self._cur_font_size * 0.4)

    def _glfw_key_callback(self, _window, key, _scancode, action, _mods):
        super()._glfw_key_callback(_window, key, _scancode, action, _mods)
        self._imgui_renderer.keyboard_callback(_window, key, _scancode, action, _mods)
        if self._control_down and action == glfw.PRESS and key == glfw.KEY_EQUAL:
            self.increase_font_size()
        if self._control_down and action == glfw.PRESS and key == glfw.KEY_MINUS:
            self.decrease_font_size()

    def _render_toasts(self, padding=20):
        _to_del = []
        _cur_height = 0

        for _id, toast in enumerate(self._toasts):
            if toast.expired:
                _to_del.append(toast)
                continue
            imgui.push_style_var(imgui.STYLE_ALPHA, toast.alpha)
            _old_rounding = imgui.get_style().window_rounding
            imgui.get_style().window_rounding = 5

            _cur_height += toast.height + padding
            imgui.set_next_window_position(
                self.content_width - (toast.width + padding),
                self.content_height - _cur_height,
            )
            imgui.set_next_window_size(toast.width, 0)

            # Render with imgui
            imgui.begin(f'toast{_id}', flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_SCROLLBAR)
            if toast.icon:
                self.icon(toast.icon, sameline=True)
            if toast.title:
                imgui.text(toast.title)
                if toast.message:
                    imgui.separator()
            if toast.message:
                imgui.push_text_wrap_pos()
                imgui.text(toast.message)
                imgui.pop_text_wrap_pos()
            toast._height = imgui.get_window_height()
            imgui.end()
            imgui.pop_style_var()
            imgui.get_style().window_rounding = _old_rounding

        # Remove expired toasts
        for _expired in _to_del:
            self._toasts.remove(_expired)

    def begin_frame(self):
        # Begin glfw frame.
        super().begin_frame()

        # Process imgui events.
        self._imgui_renderer.mouse_wheel_multiplier = self._cur_font_size / 10
        if self.content_width > 0 and self.content_height > 0:
            self._imgui_renderer.process_inputs()

        # Begin imgui frame.
        imgui.new_frame()
        imgui.push_font(self._imgui_fonts[self._cur_font_size])
        imgui_utils.set_default_style(spacing=self.spacing, indent=self.font_size, scrollbar=self.font_size+4)

        # Render toasts.
        self._render_toasts()

    @contextlib.contextmanager
    def bold_font(self):
        imgui.pop_font()
        imgui.push_font(self._imgui_fonts_bold[self._cur_font_size])
        yield
        imgui.pop_font()
        imgui.push_font(self._imgui_fonts[self._cur_font_size])

    def center_text(self, text):
        size = imgui.calc_text_size(text)
        imgui.text('')
        imgui.same_line(imgui.get_content_region_max()[0]/2 - size.x/2 + self.spacing)
        imgui.text(text)

    def close(self):
        self.make_context_current()
        self._imgui_fonts = None
        self._imgui_fonts_bold = None
        if self._imgui_renderer is not None:
            self._imgui_renderer.shutdown()
            self._imgui_renderer = None
        if self._imgui_context is not None:
            #imgui.destroy_context(self._imgui_context) # Commented out to avoid creating imgui.ini at the end.
            self._imgui_context = None
        super().close()

    def create_toast(self, message=None, title=None, icon=None):
        if message is None and title is None and icon is None:
            raise ValueError("Must supply either message, title, or icon to "
                             "create_toast()")
        self._toasts.append(Toast(
            message=message,
            title=title,
            icon=icon,
        ))

    def icon(self, name, sameline=False):
        imgui.image(self._icon_textures[name].gl_id, self.font_size, self.font_size)
        if sameline:
            imgui.same_line(self.font_size + self.spacing * 2)

    def end_frame(self):
        imgui.pop_font()
        imgui.render()
        imgui.end_frame()
        self._imgui_renderer.render(imgui.get_draw_data())
        super().end_frame()

    def set_font_size(self, target): # Applied on next frame.
        self._cur_font_size = min((abs(key - target), key) for key in self._imgui_fonts.keys())[1]

    def increase_font_size(self):
        available_sizes = sorted(list(self._imgui_fonts.keys()))
        cur_idx = available_sizes.index(self._cur_font_size)
        if cur_idx == len(available_sizes) - 1:
            pass
        else:
            self.set_font_size(available_sizes[cur_idx + 1])

    def decrease_font_size(self):
        available_sizes = sorted(list(self._imgui_fonts.keys()))
        cur_idx = available_sizes.index(self._cur_font_size)
        if cur_idx == 0:
            pass
        else:
            self.set_font_size(available_sizes[cur_idx - 1])

#----------------------------------------------------------------------------

class Toast:

    msg_duration = 4
    fade_duration = 0.25

    def __init__(self, message, title, icon):
        if icon and title is None:
            title = icon.capitalize()
        self._alpha = 0
        self._height = None
        self._default_message_height = 75
        self._create_time = time.time()
        self.message = message
        self.title = title
        self.icon = icon

    def __str__(self):
        return "<Toast message={!r}, title={!r}, icon={!r}, alpha={!r}".format(
            self.message,
            self.title,
            self.icon,
            self.alpha
        )

    @property
    def alpha(self):
        elapsed = time.time() - self._create_time
        if elapsed < self.fade_duration:
            return (elapsed / self.fade_duration)
        elif elapsed < (self.fade_duration + self.msg_duration):
            return 1
        elif elapsed < (self.fade_duration * 2 + self.msg_duration):
            return 1 - ((elapsed - (self.fade_duration + self.msg_duration)) / self.fade_duration)
        else:
            return 0

    @property
    def expired(self):
        return (time.time() - self._create_time) > (self.msg_duration + self.fade_duration * 2)

    @property
    def height(self):
        if self._height:
            return self._height
        else:
            line_height = imgui.get_text_line_height_with_spacing()
            if self.title and self.message is None:
                return line_height
            elif self.title and self.message:
                return line_height * 1.5 + self._default_message_height
            else:
                return self._default_message_height

    @property
    def width(self):
        return 400

#----------------------------------------------------------------------------
# Wrapper class for GlfwRenderer to fix a mouse wheel bug on Linux,
# and support OpenGL 2.0

class _GlfwRenderer(FixedPipelineRenderer):
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
