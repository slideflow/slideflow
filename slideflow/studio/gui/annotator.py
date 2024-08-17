import numpy as np
import imgui
import glfw
import OpenGL.GL as gl
from typing import Optional, List, Tuple, Union
from .imgui_utils import input_text

from ..gui import gl_utils
from ..gui.viewer import SlideViewer

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

    def reset(self):
        """Resets the annotation capture."""
        self.annotation_points = []
        self.clicking = False
        self._name_prompting = False
        self._prompt_pos = None
        self._keyboard_focus = False

    def clip_to_range(self, x, y, x_range, y_range):
        """Adjusts a point to be within the given range."""
        min_x, max_x = x_range[0], x_range[1]
        min_y, max_y = y_range[0], y_range[1]
        adj_x = min(max(x - min_x, 0), max_x - min_x)
        adj_y = min(max(y - min_y, 0), max_y - min_y)
        return adj_x, adj_y

    def capture(
        self,
        x_range: Tuple[int, int],
        y_range: Tuple[int, int],
        pixel_ratio: float = 1
    ) -> Tuple[Optional[List[Tuple[int, int]]], Union[bool, str]]:
        """Captures a mouse annotation in the given range.

        Args:
            x_range (tuple(int, int)): Range of pixels to capture an annotation,
                in the horizontal dimension.
            y_range (tuple(int, int)): Range of pixels to capture an annotation,
                in the horizontal dimension.
            pixel_ratio (float, optional): Ratio of points to pixels.
                Defaults to 1.

        Returns:
            A list of tuple with (x, y) coordinates for the annotation.

            A boolean indicating whether the annotation is finished (True)
            or still being drawn (False). If ``AnnotationCapture`` was
            initialized with `named=True`, this will instead be the name
            of the annotation given by the user.
        """
        min_x, max_x = x_range[0], x_range[1]
        min_y, max_y = y_range[0], y_range[1]
        mouse_x, mouse_y = imgui.get_mouse_pos()
        if pixel_ratio != 1:
            mouse_x *= pixel_ratio
            mouse_y *= pixel_ratio
        in_range = (max_x >= mouse_x) and (mouse_x >= min_x) and (max_y >= mouse_y) and (mouse_y >= min_y)
        mouse_down = imgui.is_mouse_down(self.mouse_idx)

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
            elif len(self.annotation_points) >= 3:
                to_return = self.annotation_points
                self.annotation_points = []
                return to_return, True
            else:
                # Discard capture if there are less than 3 points
                return None, False
        elif not self.clicking and not in_range:
            return None, False
        else:
            self.clicking = True
            adj_x, adj_y = self.clip_to_range(mouse_x, mouse_y, x_range, y_range)
            self.annotation_points.append((adj_x, adj_y))
            if len(self.annotation_points) >= 3:
                return self.annotation_points, False
            else:
                return None, False


# -----------------------------------------------------------------------------

class SlideAnnotationCapture(AnnotationCapture):

    def __init__(self, viz, mouse_idx=1, named=False):
        """Capture an annotation on screen with clicking and dragging.

        Args:
            viz (Studio): Slideflow Studio visualization object.
                Used for tracking the current slide in view, to ensure
                coordinates are scaled correctly.
            mouse_idx (int): Mouse index to trigger click-and-drag.
                Defaults to 1 (right click).
            named (bool): Prompt the user to name the annotation.
                Defaults to False.
        """
        super().__init__(mouse_idx, named)
        self.viz = viz
        self.scaled_coords = []
        self.scaled_box_vertices = []
        self._last_view_params = None
        self._last_n_points = 0
        self._box_vertex_width = 5
        self.vbo = None
        self.box_vbo = None

    def reset(self):
        """Resets the annotation capture."""
        super().reset()
        self.scaled_coords = []
        self.scaled_box_vertices = []
        self._last_view_params = None
        self._last_n_points = 0
        self.vbo = None
        self.box_vbo = None
        self.viz.resume_mouse_input_handling()

    @property
    def x_range(self):
        return (self.viz.viewer.x_offset, self.viz.viewer.x_offset + self.viz.viewer.width)

    @property
    def y_range(self):
        return (self.viz.viewer.y_offset, self.viz.viewer.y_offset + self.viz.viewer.height)

    @property
    def pixel_ratio(self):
        return self.viz.pixel_ratio

    def to_wsi_coords(self, x: int, y: int) -> Tuple[int, int]:
        """Converts a point from screen coordinates to WSI coordinates."""
        return self.viz.viewer.display_coords_to_wsi_coords(x, y, offset=False)

    def clip_to_view(self, x: int, y: int) -> Tuple[int, int]:
        """Adjusts a point to be within the current view."""
        return super().clip_to_range(x, y, self.x_range, self.y_range)

    def process_drawn_annotation(self, annotation: List[Tuple[int, int]]) -> np.ndarray:
        """Processes the annotation after it has been drawn, scaling to WSI coordinates."""
        viz = self.viz
        ann = np.array(annotation)
        x, y = ann[:, 0], ann[:, 1]

        # Convert the annotation to WSI coordinates.
        x, y = viz.viewer.display_coords_to_wsi_coords(x, y, offset=False)
        wsi_coords = np.column_stack((x, y)).astype(int)

        # Remove duplicate coordinates.
        unique_coords, ind = np.unique(wsi_coords, axis=0, return_index=True)
        argsort_ind = np.argsort(ind)
        wsi_coords = unique_coords[argsort_ind]
        return wsi_coords

    def capture(self) -> Tuple[Optional[List[Tuple[int, int]]], Union[bool, str]]:
        """Captures a mouse annotation in the given range.

        Returns:
            A list of tuple with (x, y) coordinates for the annotation.

            A boolean indicating whether the annotation is finished (True)
            or still being drawn (False). If ``AnnotationCapture`` was
            initialized with `named=True`, this will instead be the name
            of the annotation given by the user.
        """
        new_annotation, annotation_name = super().capture(self.x_range, self.y_range, self.pixel_ratio)
        if new_annotation is not None:
            self.viz.suspend_mouse_input_handling()
        if annotation_name:
            self.viz.resume_mouse_input_handling()
            new_annotation = self.process_drawn_annotation(new_annotation)
        return new_annotation, annotation_name

    def capture_polygon(self) -> Tuple[Optional[List[Tuple[int, int]]], bool]:
        """Capture a polygon annotation in the given range."""
        min_x, max_x = self.x_range[0], self.x_range[1]
        min_y, max_y = self.y_range[0], self.y_range[1]
        mouse_x, mouse_y = imgui.get_mouse_pos()
        if self.pixel_ratio != 1:
            mouse_x *= self.pixel_ratio
            mouse_y *= self.pixel_ratio
        in_range = (max_x >= mouse_x) and (mouse_x >= min_x) and (max_y >= mouse_y) and (mouse_y >= min_y)
        mouse_clicked = imgui.is_mouse_clicked(self.mouse_idx)

        # If the user presses enter, the annotation is finished.
        if imgui.is_key_pressed(glfw.KEY_ENTER):
            annotation = self.annotation_points
            self.reset()
            return annotation, True

        # If the user has right clicked, add a new point.
        if mouse_clicked and in_range:

            # Check if we clicked on the origin.
            # If so, close the polygon and complete the annotation.
            if len(self.scaled_coords) > 2:
                first_point = self.scaled_coords[0]
                if (abs(first_point[0] - mouse_x) < 5) and (abs(first_point[1] - mouse_y) < 5):
                    annotation = self.annotation_points
                    self.reset()
                    return annotation, True

            # Otherwise, add a new point.
            adj_x, adj_y = self.clip_to_view(mouse_x, mouse_y)
            wsi_x, wsi_y = self.to_wsi_coords(adj_x, adj_y)
            self.annotation_points.append((wsi_x, wsi_y))

        return None, False

    def capture_point(self) -> Tuple[Optional[List[Tuple[int, int]]], bool]:
        """Capture a circle annotation at the given point."""
        min_x, max_x = self.x_range[0], self.x_range[1]
        min_y, max_y = self.y_range[0], self.y_range[1]
        mouse_x, mouse_y = imgui.get_mouse_pos()
        if self.pixel_ratio != 1:
            mouse_x *= self.pixel_ratio
            mouse_y *= self.pixel_ratio
        in_range = (max_x >= mouse_x) and (mouse_x >= min_x) and (max_y >= mouse_y) and (mouse_y >= min_y)
        mouse_clicked = imgui.is_mouse_clicked(self.mouse_idx)

        # If the user has right clicked, add a new point.
        if mouse_clicked and in_range:
            adj_x, adj_y = self.clip_to_view(mouse_x, mouse_y)
            wsi_x, wsi_y = self.to_wsi_coords(adj_x, adj_y)

            # Create a circle annotation with the given point as the center,
            # with a radius of 50 pixels, every 30 degrees.
            circle_coords = []
            for i in range(0, 360, 30):
                x = wsi_x + 100 * np.cos(np.radians(i))
                y = wsi_y + 100 * np.sin(np.radians(i))
                circle_coords.append((x, y))

            self.annotation_points = circle_coords
            return self.annotation_points, True

        return None, False

    def update_box_vertices(self) -> None:
        # Convert the ROI vertices (n_vertex, 2) to (n_vertex, 4, 2) for the box.
        v = self.scaled_coords
        if v is None or not len(v):
            return None
        box_vertices = np.zeros((len(v), 4, 2)).astype(np.float32)
        w = self._box_vertex_width
        box_vertices[:, :, 0] = v[:, np.newaxis, 0] + np.array([-w, w, w, -w])
        box_vertices[:, :, 1] = v[:, np.newaxis, 1] + np.array([-w, -w, w, w])
        self.scaled_box_vertices = box_vertices

    def update(self) -> None:
        """Scale the annotation capture to match the current view."""
        # Change the scaled annotation to match the current view,
        # if the view has changed.
        if self.clicking:
            return
        if not isinstance(self.viz.viewer, SlideViewer):
            return
        view_changed = (self.viz.viewer.view_params != self._last_view_params)
        n_points_changed = (len(self.annotation_points) != self._last_n_points)
        if (view_changed or n_points_changed) and len(self.annotation_points):
            self._last_view_params = self.viz.viewer.view_params
            self._last_n_points = len(self.annotation_points)
            annotation = np.array(self.annotation_points)
            self.scaled_coords, _ = self.viz.viewer._scale_roi_to_view(annotation)
            self.update_box_vertices()
            if self.scaled_coords is not None:
                self.scaled_coords = self.scaled_coords.astype(np.float32)
                self.vbo = gl_utils.create_buffer(self.scaled_coords)
                self.box_vbo = gl_utils.create_buffer(self.scaled_box_vertices)


    def render(self) -> None:
        """Render the annotation capture."""
        self.update()
        if self.scaled_coords is not None and len(self.scaled_coords):
            # Draw the ROI.
            gl_utils.draw_vbo_roi(
                self.scaled_coords,
                color=(1, 0, 0),
                alpha=1,
                linewidth=2,
                vbo=self.vbo,
                mode=gl.GL_LINE_STRIP
            )
            # Draw the boxes at each vertex.
            # Start with the white fill.
            gl_utils.draw_boxes(
                self.scaled_box_vertices,
                vbo=self.box_vbo,
                color=(1, 1, 1),
                linewidth=2,
                mode=gl.GL_POLYGON
            )
            # Draw the red outline.
            gl_utils.draw_boxes(
                self.scaled_box_vertices,
                vbo=self.box_vbo,
                color=(1, 0, 0),
                linewidth=2
            )


