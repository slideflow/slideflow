import time
import imgui
import imgui.integrations.glfw

#----------------------------------------------------------------------------

class Toast:

    msg_duration = 4
    fade_duration = 0.25

    def __init__(self, message, title, icon, sticky=False, spinner=False):
        if icon and title is None:
            title = icon.capitalize()
        self._alpha = 0
        self._height = None
        self._default_message_height = 75
        self._create_time = time.time()
        self._start_fade_time = None
        self.spinner = spinner
        self.message = message
        self.title = title
        self.icon = icon
        self.sticky = sticky

    def __str__(self):
        return "<Toast message={!r}, title={!r}, icon={!r}, alpha={!r}, sticky={!r}, spinner={!r}".format(
            self.message,
            self.title,
            self.icon,
            self.alpha,
            self.sticky,
            self.spinner
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
            if self.title and self.message is None:
                return line_height
            elif self.title and self.message:
                return line_height * 1.5 + self._default_message_height
            else:
                return self._default_message_height

    @property
    def width(self):
        return 400

    def done(self):
        self.sticky = False
        self.msg_duration = 0
