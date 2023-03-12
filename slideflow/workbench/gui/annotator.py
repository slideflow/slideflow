import numpy as np
import imgui
from typing import Optional, List, Tuple, Union
from .imgui_utils import input_text

# -----------------------------------------------------------------------------

class AnnotationCapture:

    def __init__(self, mouse_idx=1, named=False):
        """Capture an annotation on screen with clicking and dragging.

        Args:
            mouse_idx (int): Mouse index to trigger click-and-drag.
                Defaults to 1 (right click).
            named (bool): Prompt the user to name the annotation.
                Defaults to False.
        """
        self.named = named
        self.mouse_idx = mouse_idx
        self.annotation_points = []
        self.clicking = False
        self._name_prompting = False
        self._prompt_pos = None
        self._keyboard_focus = False

    def capture(
        self,
        x_range: Tuple[int, int],
        y_range: Tuple[int, int]
    ) -> Tuple[Optional[List[Tuple[int, int]]], Union[bool, str]]:
        """Captures a mouse annotation in the given range.

        Args:
            x_range (tuple(int, int)): Range of pixels to capture an annotation,
                in the horizontal dimension.
            y_range (tuple(int, int)): Range of pixels to capture an annotation,
                in the horizontal dimension.

        Returns:
            A list of tuple with (x, y) coordinates for the annotation.

            A boolean indicating whether the annotation is finished (True)
            or still being drawn (False). If ``AnnotationCapture`` was
            initialized with `named=True`, this will instead be the name
            of the annotation given by the user.
        """
        min_x, max_x = x_range[0], x_range[1]
        min_y, max_y = y_range[0], y_range[1]
        mouse_down = imgui.is_mouse_down(self.mouse_idx)
        mouse_x, mouse_y = imgui.get_mouse_pos()
        in_range = (max_x >= mouse_x) and (mouse_x >= min_x) and (max_y >= mouse_y) and (mouse_y >= min_y)

        # First, check if the annotation is finished and we are simply
        # waiting for a name prompt.
        if self._name_prompting:
            imgui.set_cursor_pos(self._prompt_pos)
            imgui.push_style_var(imgui.STYLE_ALPHA, 255)
            enter_pressed, name = input_text('##annotation_name', '', 1024,
                flags=(imgui.INPUT_TEXT_ENTER_RETURNS_TRUE | imgui.INPUT_TEXT_AUTO_SELECT_ALL),
                width=100,
                help_text='')
            imgui.pop_style_var(1)
            if self._keyboard_focus:
                imgui.set_keyboard_focus_here()
                self._keyboard_focus = False
            if enter_pressed:
                to_return = self.annotation_points
                self.annotation_points = []
                self._name_prompting = False
                return to_return, name
            else:
                return self.annotation_points, False

        # Process mouse inputs
        if not mouse_down and not self.clicking:
            return None, False
        elif not mouse_down:
            self.clicking = False
            if self.named:
                # Show text prompt at bottom center of annotation
                ann_array = np.array(self.annotation_points)
                wx, wy = imgui.get_window_position()
                x = ann_array[:, 0].mean() + x_range[0]
                y = ann_array[:, 1].max() + y_range[0]
                self._prompt_pos = (x - 50 - wx, y - wy)
                self._keyboard_focus = True
                self._name_prompting = True
                return self.annotation_points, False
            else:
                to_return = self.annotation_points
                self.annotation_points = []
                return to_return, True
        elif not self.clicking and not in_range:
            return None, False
        else:
            self.clicking = True
            adj_x = min(max(mouse_x - min_x, 0), max_x - min_x)
            adj_y = min(max(mouse_y - min_y, 0), max_y - min_y)
            self.annotation_points.append((adj_x, adj_y))
            return self.annotation_points, False