import imgui
import numpy as np
import glfw
from os.path import join
from shapely.geometry import Point
from shapely.geometry import Polygon
from tkinter.filedialog import askopenfilename
from typing import Optional, Tuple, List

from ..gui import imgui_utils, text_utils, gl_utils
from ..gui.annotator import AnnotationCapture
from ..gui.viewer import SlideViewer

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
        self._show_roi_ctx_menu         = None
        self._show_roi_label_menu       = None
        self._ctx_mouse_pos             = None
        self._ctx_clicking              = False
        self._roi_hovering              = False
        self._show_roi_new_label_popup  = None
        self._new_label_popup_is_new    = True
        self._input_new_label           = ''
        self._roi_colors                = {}
        self._last_view_params          = None
        self._editing_label             = None
        self._editing_label_is_new      = True

    # --- Internal ------------------------------------------------------------

    def reset_edit_state(self):
        self.editing = False
        self.capturing = False
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
        if self.capturing:
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

        hovered_rois = None if viz.viewer.is_moving() else self._get_rois_at_mouse()

        # Check for right click, and show context menu if hovering over a ROI.
        if imgui.is_mouse_down(1):
            if hovered_rois is None:
                hovered_rois = self._get_rois_at_mouse()
            if hovered_rois:
                self._show_roi_ctx_menu = hovered_rois

        # Show a tooltip if hovering over a ROI and no overlays are being shown.
        if (hovered_rois
            and (viz.overlay is None or not viz.show_overlay)
            and viz.viewer.show_rois
            and not self._show_roi_ctx_menu
            and not self._show_roi_label_menu):
            imgui.set_tooltip(
                '\n'.join(
                    [f'{viz.wsi.rois[r].name} (label: {viz.wsi.rois[r].label})'
                     for r in hovered_rois]
                )
            )

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

    def get_roi_dest(self, slide: str) -> str:
        """Get the destination for a ROI file."""
        viz = self.viz
        if viz.P is None:
            return None
        dataset = viz.P.dataset()
        source = dataset.get_slide_source(slide)
        return (dataset.find_rois(slide)
                or join(dataset.sources[source]['roi'], f'{slide}.csv'))

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
                for idx in self.viz.viewer.selected_rois:
                    self.viz.wsi.remove_roi(idx)
                self.viz.viewer.deselect_roi()
                self.refresh_rois()

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
                    if ((imgui.is_mouse_down(0) or imgui.is_mouse_down(1))
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

    def draw_context_menu(self) -> None:
        """Show the context menu for a ROI.

        Args:
            hovered_rois (list): A list of indices of the hovered ROIs.

        """
        viz = self.viz
        if (viz.viewer is None              # Slide not loaded.
            or not self._show_roi_ctx_menu  # No ROIs to show context menu for.
            or not viz.viewer.show_rois     # ROIs are not being shown.
            or not self.editing             # Must be editing ROIs.
            or (viz.overlay is not None and viz.show_overlay)):  # Overlay is being shown.
            # Hide the context menu and reset.
            self._show_roi_ctx_menu = None
            if self._show_roi_label_menu is None:
                self._ctx_mouse_pos = None
            return

        if self._ctx_mouse_pos is None:
            self._ctx_mouse_pos = self.viz.get_mouse_pos(scale=False)
        imgui.set_next_window_position(*self._ctx_mouse_pos)
        imgui.begin(
            "##roi_context_menu-{}".format('-'.join(map(str, self._show_roi_ctx_menu))),
            flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        )
        clicked = False
        if len(self._show_roi_ctx_menu) == 1:
            with viz.bold_font():
                imgui.text(viz.wsi.rois[self._show_roi_ctx_menu[0]].name)
            imgui.separator()
            clicked = self._draw_ctx_submenu(self._show_roi_ctx_menu[0]) or clicked
            viz.viewer.select_roi(self._show_roi_ctx_menu[0])
        else:
            for roi_idx in self._show_roi_ctx_menu:
                if imgui.begin_menu(viz.wsi.rois[roi_idx].name):
                    clicked = self._draw_ctx_submenu(roi_idx) or clicked
                    imgui.end_menu()
                    viz.viewer.deselect_roi()
                    viz.viewer.select_roi(roi_idx)

        # Cleanup window
        if (imgui.is_mouse_down(0) and not imgui.is_window_hovered()) or clicked:
            self._ctx_clicking = True
        if (self._ctx_clicking and imgui.is_mouse_released(0)):
            self._ctx_clicking = False
            self._show_roi_ctx_menu = None
            self._ctx_mouse_pos = None
            viz.viewer.deselect_roi()

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
        if (imgui.is_mouse_down(0) and not imgui.is_window_hovered()) or clicked:
            self._ctx_clicking = True
        if (self._ctx_clicking and imgui.is_mouse_released(0)):
            self._ctx_clicking = False
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
            self.viz.wsi.remove_roi(index)
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

    # --- Control & interface -------------------------------------------------

    def update(self, show: bool) -> None:
        """Update the widget."""
        if self.viz.wsi is None or not show:
            self.reset_edit_state()
        if isinstance(self.viz.viewer, SlideViewer) and self.viz.wsi:
            self._update_grid()
            self._process_capture()

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
                    viz.viewer.deselect_roi()
                self.capturing = False
            if imgui.is_item_hovered():
                imgui.set_tooltip("Edit ROIs")
        imgui.same_line()

        # Save button.
        if viz.sidebar.large_image_button('floppy', size=viz.font_size*3):
            dest = viz.wsi.export_rois(self.get_roi_dest(viz.wsi.name))
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

            imgui.separator()

        imgui.text_colored('Total ROIs', *viz.theme.dim)
        imgui_utils.right_aligned_text(str(len(self.viz.wsi.rois)))
        imgui_utils.vertical_break()

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