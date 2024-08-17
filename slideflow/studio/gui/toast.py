import time
import imgui
import imgui.integrations.glfw
from . import imgui_utils

#----------------------------------------------------------------------------

class Toast:

    msg_duration = 4
    fade_duration = 0.25

    def __init__(
        self,
        message,
        title,
        icon,
        viz,
        sticky=False,
        spinner=False,
        progress=False,
    ):
        if icon and title is None:
            title = icon.capitalize()
        self._alpha = 0
        self._height = None
        self._default_message_height = 75
        self._create_time = time.time()
        self._start_fade_time = None
        self._progress_vals = [0]
        self.spinner = spinner
        self.message = message
        self.title = title
        self.icon = icon
        self.sticky = sticky
        self.viz = viz  # Needed to track the font size.
        self.progress = self._parse_progress(progress)

    def __str__(self):
        return "<Toast message={!r}, title={!r}, icon={!r}, alpha={!r}, sticky={!r}, spinner={!r}, progress={!r}".format(
            self.message,
            self.title,
            self.icon,
            self.alpha,
            self.sticky,
            self.spinner,
            self.progress
        )

    @property
    def alpha(self):
        elapsed = time.time() - self._create_time

        # Fading in
        if elapsed < self.fade_duration:
            return (elapsed / self.fade_duration)

        # Waiting
        elif self.sticky or (elapsed < (self.fade_duration + self.msg_duration)):
            return 1

        # Fading out
        elif elapsed < (self.fade_duration * 2 + self.msg_duration):
            if self._start_fade_time is None:
                self._start_fade_time = time.time()
            return 1 - ((time.time() - self._start_fade_time) / self.fade_duration)

        # Removed
        else:
            return 0

    @property
    def expired(self):
        return not self.sticky and (time.time() - self._create_time) > (self.msg_duration + self.fade_duration * 2)

    @property
    def height(self):
        if self._height:
            return self._height
        else:
            line_height = imgui.get_text_line_height_with_spacing()
            running_height = 0
            if self.title and self.message is None:
                running_height = line_height
            elif self.title and self.message:
                running_height = line_height * 1.5 + self._default_message_height
            else:
                running_height = self._default_message_height
            if self.progress:
                running_height += line_height
            return running_height

    @property
    def width(self):
        return 16 * self.viz.font_size

    def _parse_progress(self, val):
        if isinstance(val, bool):
            return val
        elif isinstance(val, (int, float)):
            self._progress_vals[0] = val
            return True
        elif isinstance(val, list):
            if not all(isinstance(x, (float, int)) for x in val):
                raise ValueError("Progress must be a float or list of floats.")
            self._progress_vals = val
            return True
        else:
            return False

    def done(self):
        self.sticky = False
        self.msg_duration = 0

    def set_progress(self, val, bar_id=0):
        self._progress_vals[bar_id] = val

    def render(self, viz, toast_id=0, height_offset=0, padding=20):
        """Render a toast to the given window."""

        imgui.push_style_var(imgui.STYLE_ALPHA, self.alpha)
        _old_rounding = imgui.get_style().window_rounding
        imgui.get_style().window_rounding = 5

        imgui.set_next_window_position(
            viz.content_width - (self.width + padding),
            viz.content_height - height_offset - viz.status_bar_height,
        )
        imgui.set_next_window_size(self.width, 0)
        imgui.begin(
            f'toast{toast_id}',
            flags=(imgui.WINDOW_NO_TITLE_BAR
                   | imgui.WINDOW_NO_RESIZE
                   | imgui.WINDOW_NO_SCROLLBAR)
        )

        # Icon.
        if self.icon:
            viz.icon(self.icon, sameline=True)

        # Title and spinner.
        if self.title:
            if self.spinner:
                imgui.text(f"{self.title}{imgui_utils.spinner_text()}")
            else:
                imgui.text(self.title)
            if self.message:
                imgui.separator()

        # Message.
        if self.message:
            imgui.push_text_wrap_pos()
            imgui.text(self.message)
            if self.spinner and not self.title:
                imgui.same_line()
                imgui_utils.spinner()
            imgui.pop_text_wrap_pos()

        # Progress bar.
        if self.progress:
            for val in self._progress_vals:
                imgui_utils.progress_bar(
                    val,
                    y_pad=2,
                    color=(0.55, 1, 0.47, 1)
                )

        # Cleanup.
        self._height = imgui.get_window_height()
        imgui.end()
        imgui.pop_style_var()
        imgui.get_style().window_rounding = _old_rounding
