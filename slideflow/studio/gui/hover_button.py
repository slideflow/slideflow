import time
import imgui
import imgui.integrations.glfw
from . import imgui_utils
from typing import Optional, Tuple, List

#----------------------------------------------------------------------------

class HoverButton:

    fade_duration = 0.25

    def __init__(self, viz):
        self.viz = viz
        self._alpha = 0
        self._start_fade_in_time = None
        self._start_fade_out_time = None
        self._expanded = False

    @property
    def fading_in(self) -> bool:
        return (self._start_fade_in_time is not None and (time.time() - self._start_fade_in_time) < self.fade_duration)

    @property
    def fading_out(self) -> bool:
        return (self._start_fade_out_time is not None and (time.time() - self._start_fade_out_time) < self.fade_duration)

    @property
    def alpha(self):
        if self._start_fade_in_time is not None:
            # Fading in
            elapsed = time.time() - self._start_fade_in_time
            if elapsed < self.fade_duration:
                return (elapsed / self.fade_duration)
            else:
                return 1
        elif self._start_fade_out_time is not None:
            # Fading out
            elapsed = time.time() - self._start_fade_out_time
            if elapsed < self.fade_duration:
                return 1 - (elapsed / self.fade_duration)
            else:
                return 0
        return 1

    @property
    def button_size(self) -> int:
        return self.viz.font_size * 3

    def fade_out(self):
        """Start fading out the button."""
        self._start_fade_in_time = None
        self._start_fade_out_time = time.time()

    def reset(self):
        """Reset the button back to its initial state."""
        self._expanded = False
        self._start_fade_in_time = None
        self._start_fade_out_time = None

    def __call__(
        self,
        main_icon: str,
        menu_icons: List[str],
        menu_labels: Optional[List[str]] = None,
        highlighted: Optional[List[bool]] = None,
        name: Optional[str] = None,
    ) -> Tuple[bool, int]:
        """Main draw function.

        Returns:
            bool: True if the button was clicked.
            int: Index of the clicked button.

        """
        viz = self.viz
        _clicked = False
        _button_index = None

        if menu_labels is not None:
            assert len(menu_icons) == len(menu_labels)
        if highlighted is not None:
            assert len(menu_icons) == len(highlighted)
        assert len(menu_icons) > 0

        if highlighted is None:
            highlighted = [False] * len(menu_icons)
        if menu_labels is None:
            menu_labels = [None] * len(menu_icons)

        # Draw main button (first icon).
        # If the button is hovered, start fading in the expanded window.
        # Disable hover theme for the main button.
        main_button_color = (0.2, 0.3, .5, 1)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *main_button_color)
        imgui.push_style_color(imgui.COLOR_BUTTON, *main_button_color)
        if viz.sidebar.large_image_button(main_icon, size=self.button_size):
            _clicked = True
            _button_index = 0
        imgui.pop_style_color(2)
        if imgui.is_item_hovered() and not self._expanded:
            self._start_fade_in_time = time.time()
            self._expanded = True

        if self._expanded:
            # === Draw expanded button window ===
            # Prepare alpha and window rounding.
            imgui.push_style_var(imgui.STYLE_ALPHA, self.alpha)
            imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *viz.theme.item_background)
            _old_rounding = imgui.get_style().window_rounding
            imgui.get_style().window_rounding = 5

            # Set the next window position.
            x, y = imgui.get_item_rect_min()
            x -= viz.spacing
            y -= viz.spacing
            imgui.set_next_window_position(x, y)
            imgui.set_next_window_size(
                self.button_size + viz.spacing*3,
                self.button_size*len(menu_icons) + viz.spacing*3 + viz.spacing*(len(menu_icons)-1)
            )
            imgui.begin(
                '##expanded_button_list{}'.format(name),
                flags=(imgui.WINDOW_NO_TITLE_BAR
                    | imgui.WINDOW_NO_MOVE
                    | imgui.WINDOW_NO_RESIZE
                    | imgui.WINDOW_NO_COLLAPSE
                    | imgui.WINDOW_NO_SCROLLBAR)
            )

            # Draw buttons.
            with imgui.begin_group():
                for i, (icon, label, highlight) in enumerate(zip(menu_icons, menu_labels, highlighted)):
                    with viz.highlighted(highlight):
                        if i == 0:
                            imgui.push_style_var(imgui.STYLE_ALPHA, self.alpha)
                        if viz.sidebar.large_image_button(icon, size=self.button_size):
                            _clicked = True
                            _button_index = i
                        if i == 0:
                            imgui.pop_style_var()
                    if imgui.is_item_hovered() and label:
                        imgui.set_tooltip(label)


            # Reset after fade out.
            if not imgui.is_item_hovered() and self.alpha == 0:
                self.reset()
            elif not imgui.is_item_hovered() and not self.fading_in and not self.fading_out:
                self.fade_out()

            # End window.
            imgui.end()
            imgui.pop_style_var()
            imgui.pop_style_color()
            imgui.get_style().window_rounding = _old_rounding


        return _clicked, _button_index
