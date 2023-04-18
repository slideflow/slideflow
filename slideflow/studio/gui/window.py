import os
import glfw
import contextlib
import imgui
import imgui.integrations.glfw


from . import imgui_utils
from . import text_utils
from . import gl_utils
from ._glfw import GlfwWindow, GlfwRenderer
from .toast import Toast

#----------------------------------------------------------------------------

class ImguiWindow(GlfwWindow):
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

        self._imgui_context  = None
        self._imgui_renderer = None
        self._imgui_fonts    = None
        self._imgui_fonts_bold = None
        self._cur_font_size  = max(font_sizes)
        self._font_scaling   = self.pixel_ratio
        self._toasts         = []
        self.widgets         = []

        # Delete leftover imgui.ini to avoid unexpected behavior.
        if os.path.isfile('imgui.ini'):
            os.remove('imgui.ini')

        # Initialize imgui.
        self._imgui_context = imgui.create_context()
        self._imgui_renderer = GlfwRenderer(self._glfw_window)
        self._attach_glfw_callbacks()
        imgui.get_io().ini_saving_rate = 0 # Disable creating imgui.ini at runtime.
        imgui.get_io().mouse_drag_threshold = 0 # Improve behavior with imgui_utils.drag_custom().
        self._imgui_fonts = {size: imgui.get_io().fonts.add_font_from_file_ttf(font, int(size * self._font_scaling)) for size in font_sizes}
        self._imgui_fonts_bold = {size: imgui.get_io().fonts.add_font_from_file_ttf(font_bold, int(size * self._font_scaling)) for size in font_sizes}
        self._imgui_renderer.refresh_font_texture()
        imgui.get_io().font_global_scale = 1 / self._font_scaling

        # Load icons.
        self._icon_textures = {
            name: gl_utils.Texture(image=icon)
            for name, icon in imgui_utils.icons().items()
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
                if toast.spinner:
                    imgui.same_line()
                    imgui_utils.spinner()
                if toast.message:
                    imgui.separator()
            if toast.message:
                imgui.push_text_wrap_pos()
                imgui.text(toast.message)
                if toast.spinner and not toast.title:
                    imgui.same_line()
                    imgui_utils.spinner()
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

    def create_toast(self, message=None, title=None, icon=None, **kwargs):
        if message is None and title is None and icon is None:
            raise ValueError("Must supply either message, title, or icon to "
                             "create_toast()")
        toast = Toast(message=message, title=title, icon=icon, **kwargs)
        self._toasts.append(toast)
        return toast

    def icon(self, name, sameline=False):
        imgui.image(self._icon_textures[name].gl_id, self.font_size, self.font_size)
        if sameline:
            imgui.same_line(self.font_size + self.spacing * 2)

    def end_frame(self):
        imgui.pop_font()
        imgui.render()
        imgui.end_frame()
        self._imgui_renderer.render(imgui.get_draw_data())
        self.slide_widget.late_render()
        for widget in self.widgets:
            if hasattr(widget, 'late_render'):
                widget.late_render()
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
