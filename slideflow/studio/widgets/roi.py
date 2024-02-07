import imgui
import numpy as np
import glfw
import os
import OpenGL.GL as gl
from os.path import join, exists, dirname
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union, polygonize
from tkinter.filedialog import askopenfilename
from typing import Optional, Tuple, List, Union

from ..gui import imgui_utils, text_utils, gl_utils
from ..gui.annotator import AnnotationCapture
from ..gui.viewer import SlideViewer
from ..utils import LEFT_MOUSE_BUTTON, RIGHT_MOUSE_BUTTON

import slideflow as sf

#----------------------------------------------------------------------------

class ROIWidget:
    def __init__(self, viz: "sf.studio.Studio") -> None:
        """Widget for ROI processing control and information display.

        Args:
            viz (:class:`slideflow.studio.Studio`): The parent Slideflow Studio
                object.

        """
        self.viz                        = viz
        self.editing                    = False
        self.capturing                  = False
        self.subtracting                = False
        self.roi_toast                  = None
        self.annotator                  = AnnotationCapture(named=False)
        self.roi_grid                   = []  # Rasterized grid of ROIs in view.
        self.unique_roi_labels          = []
        self.use_rois                   = True
        self._fill_rois                 = False

        # Internals
        self._late_render               = []
        self._should_show_roi_ctx_menu  = False
        self._roi_ctx_menu_items        = []
        self._show_roi_label_menu       = None
        self._ctx_mouse_pos             = None
        self._is_clicking_ctx_menu      = False
        self._roi_hovering              = False
        self._selected_rois             = []
        self._show_roi_new_label_popup  = None
        self._new_label_popup_is_new    = True
        self._input_new_label           = ''
        self._roi_colors                = {}
        self._last_view_params          = None
        self._editing_label             = None
        self._editing_label_is_new      = True
        self._should_show_advanced_editing_window = True
        self._should_deselect_roi_on_mouse_up = True
        self._mouse_is_down             = False
        self._advanced_editor_is_new    = True
        self._roi_filter_perc           = 0.5
        self._roi_filter_center         = True
        self._capturing_roi_filter_perc = False
        self._vertex_editor             = None
        self._showing                   = False

    @property
    def roi_filter_method(self) -> Union[str, float]:
        """Get the current ROI filter method."""
        return ('center' if self._roi_filter_center else self._roi_filter_perc)

    # --- Internal ------------------------------------------------------------

    def reset_edit_state(self) -> None:
        """Reset the state of the ROI editor."""
        self.disable_roi_capture()
        self.toggle_subtracting(False)
        self._should_show_advanced_editing_window = True
        if self.roi_toast is not None:
            self.roi_toast.done()

    def _get_rois_at_mouse(self) -> List[int]:
        """Get indices of ROI(s) at the current mouse position."""

        # There are two ways to get the ROIs at the mouse position.
        # First, we can iterate through each ROI and check if the mouse point
        # is contained within the ROI polygon.
        # We do this if the rasterized ROI grid is not available.
        if self.roi_grid is None:
            mouse_point = Point(self.viz.get_mouse_pos())
            possible_rois = []
            for roi_idx, roi_array in self.viz.viewer.rois:
                try:
                    roi_poly = Polygon(roi_array)
                except ValueError:
                    continue
                if roi_poly.contains(mouse_point):
                    possible_rois.append(roi_idx)
            return possible_rois

        # However, it is also possible to do this more efficiently by
        # rasterizing the ROIs and checking if the mouse point is contained
        # within the ROI mask. We do this if the rasterized ROI grid is
        # available. The ROI grid rasterization is done in the background
        # thread, so we need to check if it is available.
        else:
            mx, my = map(int, self.viz.get_mouse_pos())
            mx -= self.viz.viewer.x_offset
            my -= self.viz.viewer.y_offset
            if (mx >= self.roi_grid.shape[0]
                or my >= self.roi_grid.shape[1]
                or mx < 0
                or my < 0):
                return []
            all_rois = self.roi_grid[mx, my, :]
            return (all_rois[all_rois.nonzero()] - 1).tolist()

    def _process_capture(self) -> None:
        """Process a newly captured ROI.

        If the ROI is valid, it is added to the slide and rendered.

        """
        viz = self.viz
        new_annotation, annotation_name = self.annotator.capture(
            x_range=(viz.viewer.x_offset, viz.viewer.x_offset + viz.viewer.width),
            y_range=(viz.viewer.y_offset, viz.viewer.y_offset + viz.viewer.height),
            pixel_ratio=viz.pixel_ratio
        )

        # Render in-progress annotations
        if new_annotation is not None:
            self.viz.suspend_mouse_input_handling()
            self.render_annotation(new_annotation, origin=(viz.viewer.x_offset, viz.viewer.y_offset))
        if annotation_name:
            wsi_coords = []
            self.viz.resume_mouse_input_handling()
            for c in new_annotation:
                _x, _y = viz.viewer.display_coords_to_wsi_coords(c[0], c[1], offset=False)
                int_coords = (int(_x), int(_y))
                if int_coords not in wsi_coords:
                    wsi_coords.append(int_coords)
            if len(wsi_coords) > 2:
                wsi_coords = np.array(wsi_coords)
                viz.wsi.load_roi_array(wsi_coords)
                # Simplify the ROI.
                self.simplify_roi([len(viz.wsi.rois)-1], tolerance=5)
                # Refresh the ROI view.
                viz.viewer.refresh_view()
                # Show a label popup if the user has just created a new ROI.
                self._show_roi_label_menu = len(viz.wsi.rois) - 1

    def _process_subtract(self) -> None:
        """Process a subtracting ROI."""
        if imgui.is_mouse_down(LEFT_MOUSE_BUTTON) and self.viz.mouse_is_over_viewer:
            self.toggle_subtracting(False)
            self.annotator.reset()
        viz = self.viz
        new_annotation, annotation_name = self.annotator.capture(
            x_range=(viz.viewer.x_offset, viz.viewer.x_offset + viz.viewer.width),
            y_range=(viz.viewer.y_offset, viz.viewer.y_offset + viz.viewer.height),
            pixel_ratio=viz.pixel_ratio
        )
        # Render in-progress subtraction annotation
        if new_annotation is not None:
            self.render_annotation(new_annotation, origin=(viz.viewer.x_offset, viz.viewer.y_offset))
        if annotation_name:
            wsi_coords = []
            for c in new_annotation:
                _x, _y = viz.viewer.display_coords_to_wsi_coords(c[0], c[1], offset=False)
                int_coords = (int(_x), int(_y))
                if int_coords not in wsi_coords:
                    wsi_coords.append(int_coords)
            if len(wsi_coords) > 2:
                self.subtract_roi_from_selected(np.array(wsi_coords))

    def _set_button_style(self) -> None:
        """Set the style for the ROI buttons."""
        imgui.push_style_color(imgui.COLOR_BUTTON, 0, 0, 0, 0)
        imgui.push_style_var(imgui.STYLE_ITEM_SPACING, [0, 0])

    def _end_button_style(self) -> None:
        """End the style for the ROI buttons."""
        imgui.pop_style_color(1)
        imgui.pop_style_var(1)

    def _update_grid(self) -> None:
        """Update the rasterized ROI grid."""
        view = self.viz.viewer
        if view is None or self.viz.wsi is None:
            return

        if not view.is_moving() and (view.view_params != self._last_view_params):
            self.roi_grid = view.rasterize_rois_in_view()
            self._last_view_params = view.view_params
        elif view.is_moving():
            self.roi_grid = None

    def get_roi_dest(self, slide: str, create: bool = False) -> Optional[str]:
        """Get the destination for a ROI file."""
        viz = self.viz
        if viz.P is None:
            return None
        dataset = viz.P.dataset()
        source = dataset.get_slide_source(slide)
        if dataset._roi_set(dataset.get_slide_source(slide)):
            filename =  (dataset.find_rois(slide)
                        or join(dataset.sources[source]['roi'], f'{slide}.csv'))
            if dirname(filename) and not exists(dirname(filename)) and create:
                os.makedirs(dirname(filename))
            return filename
        else:
            return None

    def ask_load_rois(self) -> None:
        """Ask the user to load ROIs from a CSV file."""
        viz = self.viz
        path = askopenfilename(title="Load ROIs...", filetypes=[("CSV", "*.csv",)])
        if path:
            viz.wsi.load_csv_roi(path)
            viz.viewer.refresh_view()
        self.reset_edit_state()

    # --- Callbacks -----------------------------------------------------------

    def keyboard_callback(self, key: int, action: int) -> None:
        """Handle keyboard events.

        Args:
            key (int): The key that was pressed. See ``glfw.KEY_*``.
            action (int): The action that was performed (e.g. ``glfw.PRESS``,
                ``glfw.RELEASE``, ``glfw.REPEAT``).

        """
        if (key == glfw.KEY_DELETE and action == glfw.PRESS):
            if (self.editing
               and self.viz.viewer is not None
               and self._selected_rois
               and not (self.is_vertex_editing() and self._vertex_editor.vertex_is_selected)):
                self.remove_rois(self._selected_rois)

        if self.is_vertex_editing() and self.editing:
            self._vertex_editor.keyboard_callback(key, action)

        # Only process the following shortcuts if the ROI editor pane is showing.
        if self._showing:
            if key == glfw.KEY_A and action == glfw.PRESS and not self.viz._control_down:
                self.toggle_add_roi()

            if key == glfw.KEY_E and action == glfw.PRESS:
                self.toggle_edit_roi()

            if key == glfw.KEY_S and action == glfw.PRESS and self.viz._control_down:
                self.save_rois()

            if key == glfw.KEY_L and action == glfw.PRESS and self.viz._control_down:
                self.ask_load_rois()

            if key == glfw.KEY_M and action == glfw.PRESS:
                self.merge_roi(self._selected_rois)

            if key == glfw.KEY_S and action == glfw.PRESS and not self.viz._shift_down:
                self.simplify_roi(self._selected_rois)

            if key == glfw.KEY_S and action == glfw.PRESS and self.viz._shift_down and bool(self._selected_rois):
                self.toggle_subtracting()

            if key == glfw.KEY_A and action == glfw.PRESS and self.viz._control_down:
                self.select_all()

            if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
                self.deselect_all()
                self.reset_edit_state()

    def early_render(self) -> None:
        """Render elements with OpenGL (before other UI elements are drawn)."""
        if self.is_vertex_editing() and self.editing:
            self._vertex_editor.draw()

    def late_render(self) -> None:
        """Render elements with OpenGL (after other UI elements are drawn).

        Triggers after the slide has been rendered and all other UI elements
        are drawn.

        """
        for _ in range(len(self._late_render)):
            annotation, name, kwargs = self._late_render.pop()
            gl_utils.draw_roi(annotation, **kwargs)
            if isinstance(name, str):
                tex = text_utils.get_texture(
                    name,
                    size=self.viz.gl_font_size,
                    max_width=self.viz.viewer.width,
                    max_height=self.viz.viewer.height,
                    outline=2
                )
                text_pos = (annotation.mean(axis=0))
                tex.draw(pos=text_pos, align=0.5, rint=True, color=1)

    # --- ROI selection and coloring ------------------------------------------

    def select_rois(self, rois: Union[int, List[int]]) -> None:
        """Select ROI(s)."""
        if isinstance(rois, int):
            rois = [rois]
        self._selected_rois = rois
        self.viz.viewer.highlight_roi(self._selected_rois)

    def deselect_rois(self, rois: Union[int, List[int]]) -> None:
        """Deselect ROI(s)."""
        if isinstance(rois, int):
            rois = [rois]
        self._selected_rois = [r for r in self._selected_rois if r not in rois]
        self.viz.viewer.higlight_roi(self._selected_rois)
        self.disable_vertex_editing()

    def select_all(self) -> None:
        """Select all ROIs."""
        all_rois = list(range(len(self.viz.wsi.rois)))
        self.select_rois(all_rois)

    def deselect_all(self) -> None:
        """Deselect all ROIs."""
        self._selected_rois = []
        self.viz.viewer.reset_roi_highlight()
        self.disable_vertex_editing()

    # --- Drawing -------------------------------------------------------------

    def colored_label_list(
        self,
        label_list: List[Tuple[str, Tuple[float, float, float], int]],
    ) -> Optional[int]:
        """Draw a list of colored labels."""
        viz = self.viz
        draw_list = imgui.get_window_draw_list()
        hovered = None
        for i, (label, color, counts) in enumerate(label_list):
            r, g, b = color
            with imgui.begin_group():
                _changed, _color = imgui.color_edit3(
                    f"##roi_color{i}",
                    r, g, b,
                    flags=(imgui.COLOR_EDIT_NO_INPUTS
                        | imgui.COLOR_EDIT_NO_LABEL
                        | imgui.COLOR_EDIT_NO_SIDE_PREVIEW
                        | imgui.COLOR_EDIT_NO_TOOLTIP
                        | imgui.COLOR_EDIT_NO_DRAG_DROP)
                )
                if _changed:
                    self._roi_colors[label] = _color
                    self.refresh_roi_colors()
                imgui.same_line()
                if self._editing_label and self._editing_label[0] == i:
                    if self._editing_label_is_new:
                        imgui.set_keyboard_focus_here()
                        self._editing_label_is_new = False
                    _changed, self._editing_label[1] = imgui.input_text(
                        f"##edit_roi_label{i}",
                        self._editing_label[1],
                        flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
                    )
                    if ((viz.is_mouse_down(LEFT_MOUSE_BUTTON)
                         or viz.is_mouse_down(RIGHT_MOUSE_BUTTON))
                         and not imgui.is_item_hovered()):
                        self._editing_label = None
                        self._editing_label_is_new = True
                    if _changed:
                        self.update_label_name(label, self._editing_label[1])
                        self._editing_label = None
                        self._editing_label_is_new = True
                else:
                    imgui.text(str(label))
                    if imgui.is_item_clicked():
                        self._editing_label = [i, str(label)]
                imgui_utils.right_aligned_text(str(counts), spacing=viz.spacing)
            if imgui.is_item_hovered():
                x, y = imgui.get_cursor_screen_position()
                y -= (viz.font_size * 1.4)
                draw_list.add_rect_filled(
                    x-viz.spacing,
                    y-viz.spacing,
                    x+imgui.get_content_region_max()[0],
                    y+viz.font_size+(viz.spacing*0.7),
                    imgui.get_color_u32_rgba(1, 1, 1, 0.05),
                    int(viz.font_size*0.3))
                hovered = i
        return hovered

    def draw_roi_filter_capture(self) -> Optional[Union[str, float]]:
        """Draw a widget that captures the ROI filter method."""

        viz = self.viz
        capture_success = False
        with imgui_utils.grayed_out(not (self.use_rois and viz.wsi.has_rois())):
            imgui.text('ROI filter')
            imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.font_size*7)
            _center_clicked, self._roi_filter_center = imgui.checkbox('center', self._roi_filter_center)
            if _center_clicked:
                capture_success = True
            imgui.same_line()
            with imgui_utils.item_width(viz.font_size * 3), imgui_utils.grayed_out(self._roi_filter_center):
                _percent_changed, self._roi_filter_perc = imgui.slider_float(
                    f'##percent_roi',
                    self._roi_filter_perc,
                    min_value=0.01,
                    max_value=0.99,
                    format='%.2f'
                )
                if _percent_changed and not self._roi_filter_center:
                    self._capturing_roi_filter_perc = True

                if viz.is_mouse_released() and self._capturing_roi_filter_perc:
                    capture_success = True
                    self._capturing_roi_filter_perc = False
        if capture_success:
            return self.roi_filter_method
        else:
            return None


    def draw_new_label_popup(self) -> None:
        """Prompt the user for a new ROI label."""
        viz = self.viz
        window_size = (viz.font_size * 12, viz.font_size * 5.25)
        viz.center_next_window(*window_size)
        imgui.set_next_window_size(*window_size)
        _, opened = imgui.begin('Add New ROI Label', closable=True, flags=imgui.WINDOW_NO_RESIZE)
        self.viz.suspend_keyboard_input()

        if not opened:
            self._show_roi_new_label_popup = None
            self._new_label_popup_is_new = True
            self.viz.resume_keyboard_input()

        with imgui_utils.item_width(imgui.get_content_region_max()[0] - viz.spacing*2):
            if self._new_label_popup_is_new:
                imgui.set_keyboard_focus_here()
                self._new_label_popup_is_new = False
            _changed, self._input_new_label = imgui.input_text(
                '##new_roi_label',
                self._input_new_label,
                flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
            )
        if viz.sidebar.full_button("Add", width=-1) or _changed:
            roi_idx = self._show_roi_new_label_popup
            viz.wsi.rois[roi_idx].label = self._input_new_label
            self._show_roi_new_label_popup = None
            self._input_new_label = ''
            self._new_label_popup_is_new = True
            self.viz.resume_keyboard_input()
        imgui.end()

    def should_hide_context_menu(self) -> bool:
        viz = self.viz
        return (viz.viewer is None                      # Slide not loaded.
                or not self._should_show_roi_ctx_menu   # No ROIs to show context menu for.
                or not viz.viewer.show_rois             # ROIs are not being shown.
                or not self.editing                     # Must be editing ROIs.
                or (viz.overlay is not None and viz.show_overlay))  # Overlay is being shown.

    def hide_and_reset_context_menu(self) -> None:
        """Hide and reset the ROI context menu."""
        self._should_show_roi_ctx_menu = False
        self._roi_ctx_menu_items = []
        if self._show_roi_label_menu is None:
            self._ctx_mouse_pos = None

    def remove_context_menu_if_clicked(self, clicked: bool) -> None:
        """Remove the ROI context menu if an item has been clicked."""
        viz = self.viz
        # Check if the user is currently clicking on the context menu.
        if clicked or (viz.is_mouse_down(LEFT_MOUSE_BUTTON) and not imgui.is_window_hovered()):
            self._is_clicking_ctx_menu = True

        # Cleanup the window if the user has finished clicking on a context menu item.
        if (self._is_clicking_ctx_menu and viz.is_mouse_released(LEFT_MOUSE_BUTTON)):
            self.hide_and_reset_context_menu()
            self._is_clicking_ctx_menu = False
            self._ctx_mouse_pos = None
            self.viz.viewer.reset_roi_highlight()

    def draw_context_menu(self) -> None:
        """Show the context menu for a ROI."""
        viz = self.viz
        if self.should_hide_context_menu():
            self.hide_and_reset_context_menu()
            return

        # Update the context menu mouse position and window destination.
        if self._ctx_mouse_pos is None:
            self._ctx_mouse_pos = self.viz.get_mouse_pos(scale=False)
        imgui.set_next_window_position(*self._ctx_mouse_pos)
        imgui.begin(
            "##roi_context_menu-{}".format('-'.join(map(str, self._roi_ctx_menu_items))),
            flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        )

        # Draw the context menu.
        clicked = False
        if len(self._roi_ctx_menu_items) == 1:
            with viz.bold_font():
                imgui.text(viz.wsi.rois[self._roi_ctx_menu_items[0]].name)
            imgui.separator()
            clicked = self._draw_ctx_submenu(self._roi_ctx_menu_items[0]) or clicked
            self.viz.viewer.highlight_roi(self._roi_ctx_menu_items[0])
        else:
            for roi_idx in self._roi_ctx_menu_items:
                if roi_idx < len(viz.wsi.rois):
                    if imgui.begin_menu(viz.wsi.rois[roi_idx].name):
                        clicked = self._draw_ctx_submenu(roi_idx) or clicked
                        imgui.end_menu()
                        self.viz.viewer.highlight_roi(roi_idx)

        # Cleanup the context menu if the user has clicked on an item.
        self.remove_context_menu_if_clicked(clicked)

        imgui.end()

    def draw_label_menu(self) -> None:
        """Show the label menu for a ROI."""
        viz = self.viz
        if self._show_roi_label_menu is None:
            return
        if self._ctx_mouse_pos is None:
            self._ctx_mouse_pos = self.viz.get_mouse_pos(scale=False)
        imgui.set_next_window_position(*self._ctx_mouse_pos)
        imgui.begin(
            "##roi_label_menu-{}".format((str(self._show_roi_label_menu))),
            flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        )
        with viz.bold_font():
            imgui.text("Label")
        imgui.separator()
        clicked = self._draw_label_submenu(self._show_roi_label_menu, False)
        self.viz.viewer.highlight_roi(self._show_roi_label_menu)

        # Cleanup window
        if (viz.is_mouse_down(LEFT_MOUSE_BUTTON) and not imgui.is_window_hovered()) or clicked:
            self._is_clicking_ctx_menu = True
        if (self._is_clicking_ctx_menu and viz.is_mouse_released(LEFT_MOUSE_BUTTON)):
            self._is_clicking_ctx_menu = False
            self._show_roi_label_menu = None
            self._ctx_mouse_pos = None
            self.viz.viewer.reset_roi_highlight()

        imgui.end()

    def render_annotation(
        self,
        annotation: np.ndarray,
        origin: np.ndarray,
        name: Optional[str] = None,
        color: float = 1,
        alpha: float = 1,
        linewidth: int = 3
    ):
        """Render an annotation with OpenGL.

        Annotation is prepared and appended to a list of annotations to be
        rendered at the end of frame generation.

        Args:
            annotation (np.ndarray): An array of shape (N, 2) containing the
                coordinates of the vertices of the annotation.
            origin (np.ndarray): An array of shape (2,) containing the
                coordinates of the origin of the annotation.
            name (str): A name to display with the annotation.
            color (float, tuple[float, float, float]): The color of the
                annotation. Defaults to 1 (white).
            alpha (float): The opacity of the annotation. Defaults to 1.
            linewidth (int): The width of the annotation. Defaults to 3.

        """
        kwargs = dict(color=color, linewidth=linewidth, alpha=alpha)
        self._late_render.append((np.array(annotation) + origin, name, kwargs))

    def _draw_ctx_submenu(self, index: int) -> bool:
        """Draw the context menu for a single ROI."""
        with imgui.begin_menu(f"Label##roi_{index}") as label_menu:
            if label_menu.opened:
                if self._draw_label_submenu(index):
                    return True
        if imgui.menu_item(f"Delete##roi_{index}")[0]:
            self.remove_rois(index)
            self.refresh_rois()
            return True
        return False

    def _draw_label_submenu(self, index: int, show_remove: bool = True) -> bool:
        """Draw the label submenu for an ROI."""
        for label in self.unique_roi_labels:
            if imgui.menu_item(f"{label}##roi_{index}")[0]:
                self.viz.wsi.rois[index].label = label
                return True
        if len(self.unique_roi_labels):
            imgui.separator()
        if imgui.menu_item(f"New...##roi_{index}")[0]:
            self._show_roi_new_label_popup = index
            return True
        if show_remove:
            if imgui.menu_item(f"Remove##roi_{index}")[0]:
                self.viz.wsi.rois[index].label = None
                return True
        return False

    def _show_roi_tooltip(self, hovered_rois: List[int]) -> None:
        """Show a tooltip if hovering over a ROI and no overlays are being shown."""
        viz = self.viz
        if viz.viewer.is_moving():
            return
        if (hovered_rois
            and (viz.overlay is None or not viz.show_overlay)
            and viz.viewer.show_rois
            and not self._should_show_roi_ctx_menu
            and not self._show_roi_label_menu):

            imgui.set_tooltip(
                '\n'.join(
                    [f'{viz.wsi.rois[r].name} (label: {viz.wsi.rois[r].label})'
                    for r in hovered_rois]
                )
            )

    def _process_roi_left_click(self, hovered_rois: List[int]) -> None:
        """If editing, hovering over an ROI, and left clicking, select the ROI(s)."""
        viz = self.viz

        if viz.is_mouse_down(LEFT_MOUSE_BUTTON) and viz.mouse_is_over_viewer:
            if viz.viewer.is_moving():
                self._should_deselect_roi_on_mouse_up = False
        elif viz.is_mouse_down(LEFT_MOUSE_BUTTON):
            self._should_deselect_roi_on_mouse_up = False

        # Mouse is newly released; check if ROI needs selected or deelected.
        if (not viz.is_mouse_down(LEFT_MOUSE_BUTTON)
            and self._mouse_is_down
            and not viz._shift_down
            and self._should_deselect_roi_on_mouse_up
            and not hovered_rois
            and not viz.mouse_input_is_suspended()
        ):
            # Deselect ROIs if no ROIs are hovered and the mouse is released.
            self.deselect_all()
        elif (not viz.is_mouse_down(LEFT_MOUSE_BUTTON)
              and self._mouse_is_down
              and hovered_rois
              and self._should_deselect_roi_on_mouse_up
              and not viz.mouse_input_is_suspended()
        ):
            # Select ROIs if ROIs are hovered and the mouse is released.
            # If shift is down, then select multiple ROIs.
            if viz._shift_down:
                self._selected_rois = list(set(self._selected_rois + hovered_rois))
            else:
                self._selected_rois = hovered_rois
            # If one ROI is selected, enable vertex editing.
            if len(self._selected_rois) == 1:
                self.set_roi_vertex_editing(self._selected_rois[0])
            else:
                self.disable_vertex_editing()
            self.viz.viewer.highlight_roi(self._selected_rois)

        self._mouse_is_down = viz.is_mouse_down(LEFT_MOUSE_BUTTON)
        if not self._mouse_is_down:
            self._should_deselect_roi_on_mouse_up = True

    def _process_roi_right_click(self, hovered_rois: List[int]) -> None:
        """If editing, hovering over an ROI, and right clicking, show a context menu."""

        if self.viz.is_mouse_down(RIGHT_MOUSE_BUTTON) and hovered_rois:
            if not all([r in self._selected_rois for r in hovered_rois]):
                self._selected_rois = hovered_rois
            self._should_show_roi_ctx_menu = True
            self._roi_ctx_menu_items = self._selected_rois

    # --- ROI tools -----------------------------------------------------------

    def remove_rois(
        self,
        roi_indices: Union[int, List[int]],
        *,
        refresh_view: bool = True
    ) -> None:

        """Remove ROIs by the given index or indices."""
        if not self.viz.wsi:
            return

        if not isinstance(roi_indices, (list, np.ndarray, tuple)):
            roi_indices = [roi_indices]

        # Remove the old ROIs.
        self.viz.wsi.remove_roi(roi_indices)
        for idx in roi_indices:
            if self.is_vertex_editing(idx):
                self.disable_vertex_editing()
        self._selected_rois = []

        if refresh_view and isinstance(self.viz.viewer, SlideViewer):
            # Update the ROI grid.
            self.viz.viewer._refresh_rois()
            self.roi_grid = self.viz.viewer.rasterize_rois_in_view()

            # Reset ROI colors.
            self.viz.viewer.reset_roi_highlight()

    def simplify_roi(self, roi_indices: List[int], tolerance: float = 5) -> None:
        """Simplify the given ROIs."""

        # Disable vertex editing.
        if len(roi_indices) == 1 and self.is_vertex_editing(roi_indices[0]):
            is_vertex_editing = True
        else:
            is_vertex_editing = False
        self.disable_vertex_editing()

        for idx in sorted(roi_indices, reverse=True):
            poly = Polygon(self.viz.wsi.rois[idx].coordinates)
            poly_s = poly.simplify(tolerance=tolerance)
            coords_s = np.stack(poly_s.exterior.coords.xy, axis=-1)
            old_roi = self.viz.wsi.rois[idx]
            self.viz.wsi.remove_roi(idx)
            self.viz.wsi.load_roi_array(
                coords_s,
                label=old_roi.label,
                name=old_roi.name
            )

        self.viz.viewer.refresh_view()
        if len(roi_indices) == 1:
            self._selected_rois = [len(self.viz.wsi.rois)-1]
        else:
            self._selected_rois = list(range(len(self.viz.wsi.rois)-len(roi_indices), len(self.viz.wsi.rois)))

        # Update the ROI grid.
        self.viz.viewer.highlight_roi(self._selected_rois)
        self.viz.viewer._refresh_rois()
        self.roi_grid = self.viz.viewer.rasterize_rois_in_view()
        if is_vertex_editing:
            self.set_roi_vertex_editing(self._selected_rois[0])

    def subtract_roi_from_selected(self, roi_coords: np.ndarray) -> None:
        """Subtract the given ROI from the currently selected ROIs."""

        if not len(self._selected_rois) > 0:
            return
        if not len(roi_coords) > 2:
            return

        for idx in self._selected_rois:
            poly = Polygon(self.viz.wsi.rois[idx].coordinates)
            poly_to_subtract = Polygon(roi_coords)
            # Verify the line is non-intersecting.
            polygons = list(polygonize(unary_union(poly_to_subtract)))
            if len(polygons) == 0:
                continue
            try:
                poly_s = poly.difference(poly_to_subtract)
            except Exception:
                continue
            if isinstance(poly_s, Polygon):
                coords_s = np.stack(poly_s.exterior.coords.xy, axis=-1)
                self.viz.wsi.rois[idx].coordinates = coords_s
                self.viz.viewer._refresh_rois()

    def merge_roi(self, roi_indices: List[int]) -> None:
        """Merge the given ROIs together."""

        if not len(roi_indices) > 1:
            return

        # Disable vertex editing.
        self.disable_vertex_editing()

        # Merge the polygons.
        merged_poly = unary_union([
            Polygon(self.viz.wsi.rois[idx].coordinates)
            for idx in roi_indices
        ])

        if not isinstance(merged_poly, Polygon):
            self.viz.create_toast('ROIs could not be merged.', icon='error')
            return

        # Get the coordinates of the merged ROI.
        new_roi_coords = np.stack(
            merged_poly.exterior.coords.xy,
            axis=-1
        )

        # Infer the ROI label.
        first_label = self.viz.wsi.rois[roi_indices[0]].label
        if all([self.viz.wsi.rois[idx].label == first_label for idx in roi_indices]):
            new_label = self.viz.wsi.rois[roi_indices[0]].label
        else:
            new_label = None

        # Remove the old ROIs.
        self.remove_rois(roi_indices, refresh_view=False)

        # Load the merged ROI into the slide.
        self.viz.wsi.load_roi_array(
            new_roi_coords,
            label=new_label,
        )
        self._selected_rois = [len(self.viz.wsi.rois)-1]

        # Update the view.
        if isinstance(self.viz.viewer, SlideViewer):
            self.viz.viewer._refresh_rois()
            self.roi_grid = self.viz.viewer.rasterize_rois_in_view()

    def save_rois(self) -> None:
        """Save ROIs to a CSV file."""
        viz = self.viz
        roi_file = self.get_roi_dest(viz.wsi.name, create=True)
        if roi_file is None:
            source = viz.P.dataset().get_slide_source(viz.wsi.name)
            viz.create_toast(
                'Project does not have a configured ROI folder for dataset '
                f'source {source}. Configure this folder to auto-load ROIs.',
                icon='warn'
            )
            if not exists('roi'):
                os.makedirs('roi')
            roi_file = join('roi', f'{viz.wsi.name}.csv')
        dest = viz.wsi.export_rois(roi_file)
        viz.create_toast(f'ROIs saved to {dest}', icon='success')

    # --- ROI vertex editing --------------------------------------------------

    def is_vertex_editing(self, roi_index: Optional[int] = None) -> bool:
        """Check if vertex editing is enabled for a ROI."""
        if roi_index is not None:
            return self._vertex_editor is not None and self._vertex_editor.roi_index == roi_index
        else:
            return self._vertex_editor is not None

    def set_roi_vertex_editing(self, roi_index: int) -> None:
        """Enable vertex editing for the given ROIs"""
        self._vertex_editor = VertexEditor(self.viz, roi_index)

    def disable_vertex_editing(self) -> None:
        """Disable vertex editing for all ROIs."""
        if self.is_vertex_editing():
            self._vertex_editor.close()
            self._vertex_editor = None

    # --- Control & interface -------------------------------------------------

    def set_fill_rois(self, fill: bool) -> None:
        """Set whether to fill ROIs."""
        viz = self.viz
        self._fill_rois = fill
        viz.viewer.highlight_fill = ((1, 0, 0) if fill else None)
        self.refresh_roi_colors()

    def refresh_roi_colors(self) -> None:
        """Refresh the colors of the ROIs."""
        viz = self.viz
        for label in self.unique_roi_labels:
            label_rois = [
                r for r in range(len(viz.wsi.rois))
                if viz.wsi.rois[r].label == label
            ]
            viz.viewer.reset_roi_color(label_rois)
            viz.viewer.set_roi_color(
                label_rois,
                outline=self.get_roi_color(label),
                fill=(None if not self._fill_rois else self.get_roi_color(label))
            )

    def reset(self) -> None:
        self.reset_edit_state()
        self.disable_vertex_editing()
        self._selected_rois = []

    def toggle_add_roi(self) -> None:
        """Toggle ROI capture mode."""
        if self.capturing:
            self.disable_roi_capture()
        else:
            self.enable_roi_capture()

    def enable_roi_capture(self) -> None:
        """Enable capture of ROIs with right-click and drag."""
        self.capturing = True
        self.editing = False
        if self.roi_toast is not None:
            self.roi_toast.done()
        self.roi_toast = self.viz.create_toast(
            f'Capturing new ROIs. Right click and drag to create a new ROI.',
            icon='info',
            sticky=True
        )

    def disable_roi_capture(self) -> None:
        """Disable capture of ROIs with right-click and drag."""
        self.capturing = False
        self.editing = False
        if self.roi_toast is not None:
            self.roi_toast.done()

    def toggle_edit_roi(self) -> None:
        """Toggle ROI editing mode."""
        viz = self.viz
        self.editing = not self.editing
        if self.roi_toast is not None:
            self.roi_toast.done()
        if self.editing:
            self.roi_toast = viz.create_toast(f'Editing ROIs. Right click to label or remove.', icon='info', sticky=False)
        else:
            self._should_show_advanced_editing_window = True
            self.deselect_all()
        self.capturing = False

    def toggle_subtracting(self, enable: Optional[bool] = None) -> None:
        """Toggle ROI subtraction mode."""
        if enable is not None:
            self.subtracting = enable
        else:
            self.subtracting = not self.subtracting
        if self.capturing:
            self.disable_roi_capture()
        if self.roi_toast is not None:
            self.roi_toast.done()
        if self.subtracting:
            self.roi_toast = self.viz.create_toast(f'Subtracting ROIs. Right click and drag to subtract.', icon='info', sticky=True)

    def update(self, show: bool) -> None:
        """Update the widget."""

        self._showing = show

        # Reset the widget if the slide has changed.
        if self.viz.wsi is None or not show:
            self.reset_edit_state()

        # No further updates are needed if a slide is not loaded.
        if not (isinstance(self.viz.viewer, SlideViewer) and self.viz.wsi):
            return

        # Update the rasterized ROI grid.
        self._update_grid()

        # Process ROI capture.
        if self.capturing:
            self._process_capture()

        if self.subtracting:
            self._process_subtract()

        # Draw the advanced ROI editing window.
        self.draw_advanced_editing_window()

        # Process ROI hovering and clicking.
        hovered_rois = self._get_rois_at_mouse()
        if self.editing and not self.subtracting:
            self._process_roi_left_click(hovered_rois)
            self._process_roi_right_click(hovered_rois)
        if not self.subtracting:
            self._show_roi_tooltip(hovered_rois)

        # Update ROI vertex editing.
        if self.is_vertex_editing():
            if not self.editing:
                self.disable_vertex_editing()
            else:
                self._vertex_editor.update()

    def refresh(self):
        """Refresh ROI labels & colors after a slide has been loaded."""
        self.unique_roi_labels, _ = np.unique(
            [r.label for r in self.viz.wsi.rois if r.label], return_counts=True
        )
        self.refresh_roi_colors()

    def draw(self):
        """Draw the widget."""
        self.draw_options()
        self.draw_context_menu()
        self.draw_label_menu()
        if self._show_roi_new_label_popup is not None:
            self.draw_new_label_popup()

    def draw_options(self):
        viz = self.viz

        # --- Large buttons ---------------------------------------------------
        self._set_button_style()

        # Add button.
        with viz.highlighted(self.capturing):
            if viz.sidebar.large_image_button('circle_plus', size=viz.font_size*3):
                self.toggle_add_roi()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Add ROI (A)")
        imgui.same_line()

        # Edit button.
        with viz.highlighted(self.editing):
            if viz.sidebar.large_image_button('pencil', size=viz.font_size*3):
                self.toggle_edit_roi()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Edit ROIs (E)")
        imgui.same_line()

        # Save button.
        if viz.sidebar.large_image_button('floppy', size=viz.font_size*3):
            self.save_rois()
            self.reset_edit_state()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Save ROIs (Ctrl+S)")
        imgui.same_line()

        # Load button.
        if viz.sidebar.large_image_button('folder', size=viz.font_size*3):
            self.ask_load_rois()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Load ROIs (Ctrl+L)")
        self._end_button_style()

        imgui_utils.vertical_break()

        # --- ROI labels ------------------------------------------------------
        self.unique_roi_labels, counts = np.unique(
            [r.label for r in self.viz.wsi.rois if r.label], return_counts=True
        )
        if len(self.unique_roi_labels):
            hovered = self.colored_label_list(
                [(label, self.get_roi_color(label), count)
                 for label, count in zip(self.unique_roi_labels, counts)]
            )
            if hovered is not None:
                self.viz.viewer.reset_roi_color()
                self.viz.viewer.set_roi_color(
                    [r for r in range(len(viz.wsi.rois))
                     if viz.wsi.rois[r].label == self.unique_roi_labels[hovered]],
                    outline=self.get_roi_color(self.unique_roi_labels[hovered]),
                    fill=(None if not self._fill_rois else self.get_roi_color(self.unique_roi_labels[hovered]))
                )
                self._roi_hovering = hovered
            elif self._roi_hovering is not None:
                self._roi_hovering = None
                self.refresh_roi_colors()

            imgui.separator()

        imgui.text_colored('Total ROIs', *viz.theme.dim)
        imgui_utils.right_aligned_text(str(len(self.viz.wsi.rois)))
        imgui_utils.vertical_break()

    def draw_advanced_editing_window(self):
        """Draw a window with advanced editing options."""

        if not (self.editing and self._should_show_advanced_editing_window):
            return

        # Prepare window parameters (size, position)
        imgui.set_next_window_size(self.viz.font_size*13.5, 0)
        if self._advanced_editor_is_new:
            imgui.set_next_window_position(
                self.viz.offset_x + 20,
                self.viz.offset_y + 20
            )
            self._advanced_editor_is_new = False

        _, self._should_show_advanced_editing_window = imgui.begin(
            'Edit ROIs',
            closable=True,
            flags=imgui.WINDOW_NO_RESIZE
        )
        imgui.text('{} ROI selected ({} vertices).'.format(
            len(self._selected_rois),
            sum([len(self.viz.wsi.rois[r].coordinates) for r in self._selected_rois])
        ))
        if imgui_utils.button('Merge'):
            self.merge_roi(self._selected_rois)
        if imgui.is_item_hovered():
            imgui.set_tooltip('Merge selected ROIs into a single ROI. <M>')
        imgui.same_line()
        if imgui_utils.button('Simplify'):
            self.simplify_roi(self._selected_rois, tolerance=5)
        if imgui.is_item_hovered():
            imgui.set_tooltip('Simplify the selected ROIs. <S>')
        imgui.same_line()
        if imgui_utils.button('Delete'):
            self.remove_rois(self._selected_rois)
        if imgui.is_item_hovered():
            imgui.set_tooltip('Delete the selected ROIs. <Delete>')
        imgui.same_line()
        if imgui_utils.button(('Subtract' if not self.subtracting else 'Subtracting...'),
                              enabled=bool(self._selected_rois)):
            self.toggle_subtracting()
        if imgui.is_item_hovered():
            imgui.set_tooltip('Subtract the selected ROIs from other ROIs. <Shift+S>')

        if imgui.is_window_hovered():
            self._should_deselect_roi_on_mouse_up = False

        imgui.end()

    def get_roi_color(self, label: str) -> Tuple[float, float, float, float]:
        """Get the color of an ROI label."""
        if label not in self._roi_colors:
            self._roi_colors[label] = imgui_utils.get_random_color('bright')
        return self._roi_colors[label]

    def refresh_rois(self) -> None:
        """Refresh ROIs in view and total ROI counts."""
        self.viz.viewer.refresh_view()

    def update_label_name(self, old_name: str, new_name: str) -> None:
        """Update the name of a ROI label."""
        if old_name == new_name:
            return
        self._roi_colors[new_name] = self._roi_colors.pop(old_name)
        for roi in self.viz.wsi.rois:
            if roi.label == old_name:
                roi.label = new_name

#----------------------------------------------------------------------------

class VertexEditor:

    def __init__(self, viz: "sf.studio.Studio", roi_index: int) -> None:
        self.viz = viz
        self.wsi = viz.wsi
        self.roi_index = roi_index
        self.selected_vertices = []  # Vertex indices as stored in the viewer. These are not the same as the full ROI coordinates, as they are both scaled and culled.
        self._last_roi_viewer_index = None
        self._vbo = None
        self._box_vertex_width = 5
        self._last_vertices = self.roi_vertices
        self._last_box_vertices = self._calculate_box_vertices()
        self._create_vbo()
        self._left_mouse_down = False
        self._mouse_coords_at_down = None
        self._last_mouse_coords = None
        self._mouse_down_at_vertex = False
        self._roi_is_edited = False
        self._selection_box = None
        self._selected_vertices_at_mouse_down = []
        self._select_on_release = []
        self._process_rois_on_release = False

    # --- Properties ----------------------------------------------------------

    @property
    def roi_vertices(self) -> Optional[np.ndarray]:
        """Get the vertices of the ROI."""
        if (self._last_roi_viewer_index is not None
            and len(self.viz.viewer.rois) > self._last_roi_viewer_index
            and self.viz.viewer.rois[self._last_roi_viewer_index][0] == self.roi_index):
            return self.viz.viewer.rois[self._last_roi_viewer_index][1]
        else:
            for i, (idx, vertices) in enumerate(self.viz.viewer.rois):
                if idx == self.roi_index:
                    self._last_roi_viewer_index = i
                    return vertices
            return None

    @property
    def box_vertices(self) -> Optional[np.ndarray]:
        """Get the vertices of the box around the ROI."""
        vertices = self.roi_vertices
        if vertices is None:
            return None
        if not np.all(vertices == self._last_vertices):
            last_shape = self._last_box_vertices.shape[0] if self._last_box_vertices is not None else None
            self._last_box_vertices = self._calculate_box_vertices()
            if last_shape != self._last_box_vertices.shape[0]:
                self.selected_vertices = []
            self._last_vertices = self.roi_vertices
            self._create_vbo()
        return self._last_box_vertices

    @property
    def selected_vertex_indices(self) -> List[int]:
        """Get the indices of the selected vertices w.r.t. the full ROI coordinates.

        These indices are not the same as the indices stored in the viewer
        (``.selected_vertices``). The viewer-stored indices are scaled and culled,
        while these indices are not. These indices represent all vertices of the ROI.

        """
        return self.viz.viewer._scaled_roi_ind[self.roi_index][self.selected_vertices]

    @property
    def vbo(self):
        if self._vbo is None:
            self._create_vbo()
        return self._vbo

    @vbo.setter
    def vbo(self, value):
        self._vbo = value

    @property
    def vertex_is_selected(self) -> bool:
        return bool(self.selected_vertices)

    @property
    def is_editing_vertices(self) -> bool:
        return self.vertex_is_selected or self.viz._control_down

    # --- User input ----------------------------------------------------------

    def keyboard_callback(self, key: int, action: int) -> None:
        """Handle keyboard events.

        Args:
            key (int): The key that was pressed. See ``glfw.KEY_*``.
            action (int): The action that was performed (e.g. ``glfw.PRESS``,
                ``glfw.RELEASE``, ``glfw.REPEAT``).

        """
        if key == glfw.KEY_DELETE and action == glfw.PRESS:
            if self.vertex_is_selected:
                self.remove_selected_vertices()

    def check_if_mouse_newly_down_over_vertex(self):
        """Check if the mouse is newly down over a vertex."""
        if (imgui.is_mouse_down(LEFT_MOUSE_BUTTON)
            and not self._left_mouse_down
            and self.is_editing_vertices
            and self.roi_vertices is not None
        ):
            # Mouse is newly down. Check if the mouse is over one of the box vertices.
            mouse_x, mouse_y = self.viz.get_mouse_pos() #self._mouse_coords_at_down

            # Get min/max X and Y values for the box vertices.
            w = self._box_vertex_width + 2
            min_x = self.roi_vertices[:, 0] - w
            max_x = self.roi_vertices[:, 0] + w
            min_y = self.roi_vertices[:, 1] - w
            max_y = self.roi_vertices[:, 1] + w

            # Check if the mouse is over any of the box vertices.
            inside_x = (mouse_x >= min_x) & (mouse_x <= max_x)
            inside_y = (mouse_y >= min_y) & (mouse_y <= max_y)
            inside_boxes = inside_x & inside_y

            if np.any(inside_boxes):
                return np.where(inside_boxes)[0][0]
            else:
                return None


    def handle_mouse_input(self):
        """Handle mouse input for editing vertices of an ROI."""

        # First, check if the mouse has newly clicked over a vertex.
        newly_clicked_vertex = self.check_if_mouse_newly_down_over_vertex()

        # === Suspend or resume mouse input handling ===
        # If the user is pressing control (forcing vertex view), then we should
        # suspend the mouse input handling for the viewer.
        if self.viz._control_down:
            self.viz.suspend_mouse_input_handling()
        # If the user is editing vertices and a vertex has been newly clicked,
        # then we should handle the mouse input instead of the viewer.
        elif self.is_editing_vertices and newly_clicked_vertex is not None:
            self.viz.suspend_mouse_input_handling()
        # Finally, if the user is editing vertices and they are currently being dragged,
        # then we should handle the mouse input instead of the viewer.
        elif self.is_editing_vertices and self._mouse_down_at_vertex:
            self.viz.suspend_mouse_input_handling()
        # Otherwise, we should resume the mouse input handling for the viewer.
        else:
            self.viz.resume_mouse_input_handling()
            return

        # === Handle mouse input ===
        # Check if the mouse is newly down over a vertex.
        if newly_clicked_vertex is not None:
            # Mouse is newly down over a vertex.
            self._last_mouse_coords = None
            self._selected_vertices_at_mouse_down = self.selected_vertices
            self._mouse_coords_at_down = self.viz.get_mouse_pos()
            self._mouse_down_at_vertex = True
            if self.viz._shift_down and newly_clicked_vertex not in self.selected_vertices:
                self.selected_vertices.append(newly_clicked_vertex)
            elif self.viz._shift_down:
                self.selected_vertices.remove(newly_clicked_vertex)
            elif len(self.selected_vertices) > 1:
                self._select_on_release = [newly_clicked_vertex]
            else:
                self.selected_vertices = [newly_clicked_vertex]

        elif imgui.is_mouse_down(LEFT_MOUSE_BUTTON):
            if not self._left_mouse_down:
                # Mouse is newly down, but not over a vertex.
                self._mouse_down_at_vertex = False
                self._mouse_coords_at_down = self.viz.get_mouse_pos()
                self._selected_vertices_at_mouse_down = self.selected_vertices
                if not self.viz._shift_down:
                    self.selected_vertices = []

            # Mouse is still down.
            mouse_x, mouse_y = self.viz.get_mouse_pos()
            # Check if the mouse is moving.
            if (self._mouse_coords_at_down is not None
                and (np.abs(mouse_x - self._mouse_coords_at_down[0]) > 5
                     or np.abs(mouse_y - self._mouse_coords_at_down[1]) > 5)):
                # Check if the mouse started over a vertex and is moving.
                # If so, we should drag the vertex.
                if self._mouse_down_at_vertex:
                    if self._last_mouse_coords is None:
                        self._last_mouse_coords = self._mouse_coords_at_down
                    dx = int(np.round((mouse_x - self._last_mouse_coords[0]) * self.viz.viewer.view_zoom))
                    dy = int(np.round((mouse_y - self._last_mouse_coords[1]) * self.viz.viewer.view_zoom))
                    svi = np.array(self.selected_vertex_indices)
                    coords = np.array(self.viz.wsi.rois[self.roi_index].coordinates)
                    coords[svi] += np.array([dx, dy])
                    self.viz.wsi.rois[self.roi_index].coordinates = coords
                    self.viz.viewer._refresh_rois()
                    self._select_on_release = []
                    self._last_mouse_coords = (mouse_x, mouse_y)
                # Otherwise, we should draw a selection box.
                else:
                    # Update selection box coordinates.
                    self._selection_box = [self._mouse_coords_at_down, (mouse_x, mouse_y)]
                    # Check if any vertices are inside the selection box.
                    if self.roi_vertices is not None:
                        min_x, max_x = np.sort([self._selection_box[0][0], self._selection_box[1][0]])
                        min_y, max_y = np.sort([self._selection_box[0][1], self._selection_box[1][1]])
                        inside_x = (self.roi_vertices[:, 0] >= min_x) & (self.roi_vertices[:, 0] <= max_x)
                        inside_y = (self.roi_vertices[:, 1] >= min_y) & (self.roi_vertices[:, 1] <= max_y)
                        inside_boxes = inside_x & inside_y
                        selected_by_box = np.where(inside_boxes)[0]
                        if self.viz._shift_down and all([i in self._selected_vertices_at_mouse_down for i in selected_by_box]):
                            self.selected_vertices = list(set(self._selected_vertices_at_mouse_down) - set(selected_by_box))
                        elif self.viz._shift_down:
                            self.selected_vertices = list(set(self._selected_vertices_at_mouse_down + list(selected_by_box)))
                        else:
                            self.selected_vertices = list(selected_by_box)

        elif imgui.is_mouse_released(LEFT_MOUSE_BUTTON):
            if self._select_on_release:
                self.selected_vertices = self._select_on_release
                self._select_on_release = []
                self._last_mouse_coords = None

        self._left_mouse_down = imgui.is_mouse_down(LEFT_MOUSE_BUTTON)

        if not self._left_mouse_down:
            "Mouse is not down; resetting vertex editing state."
            self._mouse_down_at_vertex = False
            self._selection_box = None

    # --- Logic ---------------------------------------------------------------

    def _create_vbo(self) -> None:
        box_vertices = self.box_vertices
        if box_vertices is not None:
            self.vbo = gl_utils.create_buffer(box_vertices)
        else:
            self.vbo = None

    def _calculate_box_vertices(self) -> Optional[np.ndarray]:
        """Calculate box outlines for each vertex in the ROI."""
        # Convert the ROI vertices (n_vertex, 2) to (n_vertex, 4, 2) for the box.
        vertices = self.roi_vertices
        if vertices is None:
            return None
        box_vertices = np.zeros((len(vertices), 4, 2)).astype(np.float32)
        w = self._box_vertex_width
        box_vertices[:, :, 0] = vertices[:, np.newaxis, 0] + np.array([-w, w, w, -w])
        box_vertices[:, :, 1] = vertices[:, np.newaxis, 1] + np.array([-w, -w, w, w])
        return box_vertices

    def remove_selected_vertices(self) -> None:
        """Remove the selected vertices from the ROI."""
        if self.selected_vertices:
            raw_roi_coords = self.viz.wsi.rois[self.roi_index].coordinates
            raw_roi_coords = np.delete(raw_roi_coords, self.selected_vertex_indices, axis=0)
            if raw_roi_coords.shape[0] < 3:
                # ROI cannot be less than 3 vertices.
                self.viz.slide_widget.roi_widget.remove_rois(self.roi_index)
                self.viz.slide_widget.roi_widget.disable_vertex_editing()
            else:
                self.viz.wsi.rois[self.roi_index].coordinates = raw_roi_coords
                self.viz.viewer.refresh_view()
                self.selected_vertices = []
                self._last_box_vertices = self._calculate_box_vertices()
                self._create_vbo()

    def close(self) -> None:
        """Close the ROI vertex editor."""
        self.viz.viewer.reset_roi_highlight()
        if self.viz._control_down:
            self.viz.resume_mouse_input_handling()

    def update(self) -> None:
        """Update the ROI vertex editor."""
        self.handle_mouse_input()
        self._last_box_vertices = self._calculate_box_vertices()
        self._create_vbo()

    # --- Drawing -------------------------------------------------------------

    def draw(self) -> None:
        """Draw the ROI vertex editor."""
        box_vertices = self.box_vertices
        # Draw the box vertices, if the ROI is in view.
        if box_vertices is not None and self.is_editing_vertices:
            gl_utils.draw_boxes(
                box_vertices,
                vbo=self.vbo,
                color=(1, 1, 1),
                linewidth=2,
                mode=gl.GL_POLYGON
            )
            gl_utils.draw_boxes(
                box_vertices,
                vbo=self.vbo,
                color=(1, 0, 0),
                linewidth=2
            )
        # Fill in the boxes for the selected vertices.
        if box_vertices is not None and self.selected_vertices and self.is_editing_vertices:
            for v in self.selected_vertices:
                gl_utils.draw_roi(
                    box_vertices[v],
                    color=(1, 0, 0),
                    linewidth=4,
                    mode=gl.GL_POLYGON
                )
        # Draw the selection box, if it exists.
        if self._selection_box is not None:
            selection_box_vertices = np.array([
                self._selection_box[0],
                (self._selection_box[0][0], self._selection_box[1][1]),
                self._selection_box[1],
                (self._selection_box[1][0], self._selection_box[0][1])
            ])
            gl_utils.draw_roi(
                selection_box_vertices,
                color=(0, 0, 0),
                linewidth=3,
                mode=gl.GL_LINE_LOOP
            )

