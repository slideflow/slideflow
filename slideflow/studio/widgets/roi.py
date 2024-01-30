import imgui
import numpy as np
import glfw
import os
from os.path import join, exists, dirname
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
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
        self.roi_toast                  = None
        self.annotator                  = AnnotationCapture(named=False)
        self.capturing                  = False
        self.editing                    = False
        self.roi_grid                   = []  # Rasterized grid of ROIs in view.
        self.unique_roi_labels          = []
        self.roi_toast                  = None
        self.use_rois                   = True

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

    @property
    def roi_filter_method(self) -> Union[str, float]:
        """Get the current ROI filter method."""
        return ('center' if self._roi_filter_center else self._roi_filter_perc)

    # --- Internal ------------------------------------------------------------

    def reset_edit_state(self):
        self.editing = False
        self.capturing = False
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
            self.render_annotation(new_annotation, origin=(viz.viewer.x_offset, viz.viewer.y_offset))
        if annotation_name:
            wsi_coords = []
            for c in new_annotation:
                _x, _y = viz.viewer.display_coords_to_wsi_coords(c[0], c[1], offset=False)
                int_coords = (int(_x), int(_y))
                if int_coords not in wsi_coords:
                    wsi_coords.append(int_coords)
            if len(wsi_coords) > 2:
                wsi_coords = np.array(wsi_coords)
                viz.wsi.load_roi_array(wsi_coords)
                viz.viewer.refresh_view()
                # Show a label popup if the user has just created a new ROI.
                self._show_roi_label_menu = len(viz.wsi.rois) - 1

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

    def get_roi_dest(self, slide: str, create: bool = False) -> str:
        """Get the destination for a ROI file."""
        viz = self.viz
        if viz.P is None:
            return None
        dataset = viz.P.dataset()
        source = dataset.get_slide_source(slide)
        filename =  (dataset.find_rois(slide)
                     or join(dataset.sources[source]['roi'], f'{slide}.csv'))
        if dirname(filename) and not exists(dirname(filename)) and create:
            os.makedirs(dirname(filename))
        return filename

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
               and hasattr(self.viz.viewer, 'selected_rois')):
                self.remove_rois(self._selected_rois)

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
                    if ((imgui.is_mouse_down(LEFT_MOUSE_BUTTON)
                         or imgui.is_mouse_down(RIGHT_MOUSE_BUTTON))
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

                if imgui.is_mouse_released() and self._capturing_roi_filter_perc:
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

        if not opened:
            self._show_roi_new_label_popup = None
            self._new_label_popup_is_new = True

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

        # Check if the user is currently clicking on the context menu.
        if clicked or (imgui.is_mouse_down(LEFT_MOUSE_BUTTON) and not imgui.is_window_hovered()):
            self._is_clicking_ctx_menu = True

        # Cleanup the window if the user has finished clicking on a context menu item.
        if (self._is_clicking_ctx_menu and imgui.is_mouse_released(LEFT_MOUSE_BUTTON)):
            self.hide_and_reset_context_menu()
            self._is_clicking_ctx_menu = False
            self._ctx_mouse_pos = None
            self.viz.viewer.deselect_roi()

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
            viz.viewer.deselect_roi()
            viz.viewer.select_roi(self._roi_ctx_menu_items[0])
        else:
            for roi_idx in self._roi_ctx_menu_items:
                if roi_idx < len(viz.wsi.rois):
                    if imgui.begin_menu(viz.wsi.rois[roi_idx].name):
                        clicked = self._draw_ctx_submenu(roi_idx) or clicked
                        imgui.end_menu()
                        viz.viewer.deselect_roi()
                        viz.viewer.select_roi(roi_idx)

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
        viz.viewer.select_roi(self._show_roi_label_menu)

        # Cleanup window
        if (imgui.is_mouse_down(LEFT_MOUSE_BUTTON) and not imgui.is_window_hovered()) or clicked:
            self._is_clicking_ctx_menu = True
        if (self._is_clicking_ctx_menu and imgui.is_mouse_released(LEFT_MOUSE_BUTTON)):
            self._is_clicking_ctx_menu = False
            self._show_roi_label_menu = None
            self._ctx_mouse_pos = None
            viz.viewer.deselect_roi()

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

        if imgui.is_mouse_down(LEFT_MOUSE_BUTTON) and self.viz.mouse_is_over_viewer:
            if self.viz.viewer.is_moving():
                self._should_deselect_roi_on_mouse_up = False
        elif imgui.is_mouse_down(LEFT_MOUSE_BUTTON):
            self._should_deselect_roi_on_mouse_up = False

        # Mouse is newly released; check if ROI needs selected or deelected.
        if (not imgui.is_mouse_down(LEFT_MOUSE_BUTTON)
            and self._mouse_is_down
            and not self.viz._shift_down
            and self._should_deselect_roi_on_mouse_up
            and not hovered_rois):
            self.viz.viewer.deselect_roi()
            self._selected_rois = []
        elif (not imgui.is_mouse_down(LEFT_MOUSE_BUTTON)
              and self._mouse_is_down
              and hovered_rois
              and self._should_deselect_roi_on_mouse_up):
            if self.viz._shift_down:
                self._selected_rois = list(set(self._selected_rois + hovered_rois))
            else:
                self._selected_rois = hovered_rois
            self.viz.viewer.deselect_roi()
            self.viz.viewer.select_roi(self._selected_rois)

        self._mouse_is_down = imgui.is_mouse_down(LEFT_MOUSE_BUTTON)
        if not self._mouse_is_down:
            self._should_deselect_roi_on_mouse_up = True

    def _process_roi_right_click(self, hovered_rois: List[int]) -> None:
        """If editing, hovering over an ROI, and right clicking, show a context menu."""

        if imgui.is_mouse_down(RIGHT_MOUSE_BUTTON) and hovered_rois:
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
        for idx in sorted(roi_indices, reverse=True):
            self.viz.wsi.remove_roi(idx)
        self._selected_rois = []

        if refresh_view and isinstance(self.viz.viewer, SlideViewer):
            # Update the ROI grid.
            self.viz.viewer._refresh_rois()
            self.roi_grid = self.viz.viewer.rasterize_rois_in_view()

            # Deselect ROIs.
            self.viz.viewer.deselect_roi()

    def simplify_roi(self, roi_indices: List[int]) -> None:
        """Simplify the given ROIs."""
        for idx in sorted(roi_indices, reverse=True):
            poly = Polygon(self.viz.wsi.rois[idx].coordinates)
            poly_s = poly.simplify(tolerance=0.1)
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
        self.viz.viewer.deselect_roi()
        self.viz.viewer.select_roi(self._selected_rois)
        self.viz.viewer._refresh_rois()
        self.roi_grid = self.viz.viewer.rasterize_rois_in_view()

    def merge_roi(self, roi_indices: List[int]) -> None:
        """Merge the given ROIs together."""

        if not len(roi_indices) > 1:
            return

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


    # --- Control & interface -------------------------------------------------

    def update(self, show: bool) -> None:
        """Update the widget."""
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

        # Draw the advanced ROI editing window.
        self.draw_advanced_editing_window()

        # Process ROI hovering and clicking.
        hovered_rois = self._get_rois_at_mouse()
        if self.editing:
            self._process_roi_left_click(hovered_rois)
            self._process_roi_right_click(hovered_rois)
        self._show_roi_tooltip(hovered_rois)

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
                self.capturing = not self.capturing
                self.editing = False
                if self.roi_toast is not None:
                    self.roi_toast.done()
                if self.capturing:
                    self.roi_toast = viz.create_toast(f'Capturing new ROIs. Right click and drag to create a new ROI.', icon='info', sticky=True)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Add ROI")
        imgui.same_line()

        # Edit button.
        with viz.highlighted(self.editing):
            if viz.sidebar.large_image_button('pencil', size=viz.font_size*3):
                self.editing = not self.editing
                if self.roi_toast is not None:
                    self.roi_toast.done()
                if self.editing:
                    self.roi_toast = viz.create_toast(f'Editing ROIs. Right click to label or remove.', icon='info', sticky=True)
                else:
                    self._should_show_advanced_editing_window = True
                    viz.viewer.deselect_roi()
                self.capturing = False
            if imgui.is_item_hovered():
                imgui.set_tooltip("Edit ROIs")
        imgui.same_line()

        # Save button.
        if viz.sidebar.large_image_button('floppy', size=viz.font_size*3):
            roi_file = self.get_roi_dest(viz.wsi.name, create=True)
            dest = viz.wsi.export_rois(roi_file)
            viz.create_toast(f'ROIs saved to {dest}', icon='success')
            self.reset_edit_state()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Save ROIs")
        imgui.same_line()

        # Load button.
        if viz.sidebar.large_image_button('folder', size=viz.font_size*3):
            path = askopenfilename(title="Load ROIs...", filetypes=[("CSV", "*.csv",)])
            if path:
                viz.wsi.load_csv_roi(path)
                viz.viewer.refresh_view()
            self.reset_edit_state()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Load ROIs")
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
                viz.viewer.select_roi(
                    [r for r in range(len(viz.wsi.rois))
                     if viz.wsi.rois[r].label == self.unique_roi_labels[hovered]],
                    outline=self.get_roi_color(self.unique_roi_labels[hovered])
                )
                self._roi_hovering = hovered
            elif self._roi_hovering is not None:
                self._roi_hovering = None
                viz.viewer.deselect_roi()
                self.viz.viewer.select_roi(self._selected_rois)

            imgui.separator()

        imgui.text_colored('Total ROIs', *viz.theme.dim)
        imgui_utils.right_aligned_text(str(len(self.viz.wsi.rois)))
        imgui_utils.vertical_break()

    def draw_advanced_editing_window(self):
        """Draw a window with advanced editing options."""

        if not (self.editing and self._should_show_advanced_editing_window):
            return

        # Prepare window parameters (size, position)
        imgui.set_next_window_size(300, 0)
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
        imgui.same_line()
        if imgui_utils.button('Simplify'):
            self.simplify_roi(self._selected_rois)
        imgui.same_line()
        if imgui_utils.button('Delete'):
            self.remove_rois(self._selected_rois)

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