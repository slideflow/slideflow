import pandas as pd
import imgui
import numpy as np

from shapely.geometry import Point
from shapely.geometry import Polygon
from tkinter.filedialog import askopenfilename
from ..gui import imgui_utils, text_utils, gl_utils
from ..gui.annotator import AnnotationCapture

#----------------------------------------------------------------------------

class AnnotationWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.content_height = 0
        self.annotator      = AnnotationCapture(named=False)
        self.capturing      = False
        self.editing        = False
        self.annotations    = []
        self._mouse_down    = False
        self._late_render   = []
        self._visible       = False

    @property
    def visible(self) -> bool:
        """Whether this widget is visible."""
        return self.viz.wsi is not None and self._visible

    @visible.setter
    def visible(self, val: bool):
        """Whether this widget is visible."""
        self._visible = val

    def show_menu_options(self):
        """Menu options to be shown in View -> Show"""
        _changed, _ = imgui.menu_item(
            'ROI Capture',
            selected=self._visible,
            enabled=bool(self.viz.wsi is not None)
        )
        if _changed:
            self.visible = not self.visible

    def late_render(self):
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

    def render_annotation(self, annotation, origin, name=None, color=1, alpha=1, linewidth=3):
        kwargs = dict(color=color, linewidth=linewidth, alpha=alpha)
        self._late_render.append((np.array(annotation) + origin, name, kwargs))

    def keyboard_callback(self, key, action):
        import glfw
        if (key == glfw.KEY_DELETE and action == glfw.PRESS):
            if self.editing and hasattr(self.viz.viewer, 'selected_rois'):
                for idx in self.viz.viewer.selected_rois:
                    self.viz.wsi.remove_roi(idx)
                self.viz.viewer.deselect_roi()
                self.viz.viewer.refresh_view()


    def check_for_selected_roi(self):
        mouse_down = imgui.is_mouse_down(0)

        # Mouse is newly up
        if not mouse_down:
            self._mouse_down = False
            return
        # Mouse is already down
        elif self._mouse_down:
            return
        # Mouse is newly down
        else:
            self._mouse_down = True
            mouse_point = Point(imgui.get_mouse_pos())
            for roi_idx, roi_array in self.viz.viewer.rois:
                try:
                    roi_poly = Polygon(roi_array)
                except ValueError:
                    continue
                if roi_poly.contains(mouse_point):
                    return roi_idx

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show and self.visible:
            imgui.begin('ROIs', flags=imgui.WINDOW_NO_RESIZE)

            # Add button.
            add_txt = 'Add' if not self.capturing else 'Capturing...'
            if imgui_utils.button(add_txt, width=viz.button_w, enabled=True):
                self.capturing = not self.capturing
                self.editing = False
                if self.capturing:
                    viz.create_toast(f'Capturing new ROIs. Right click and drag to create a new ROI.', icon='info')

            # Edit button.
            edit_txt = 'Edit' if not self.editing else 'Editing...'
            if imgui_utils.button(edit_txt, width=viz.button_w, enabled=True):
                self.editing = not self.editing
                if self.editing:
                    viz.create_toast(f'Editing ROIs. Click to select an ROI, and press <Del> to remove.', icon='info')
                else:
                    viz.viewer.deselect_roi()

                self.capturing = False

            # Save button.
            if imgui_utils.button('Save', width=viz.button_w, enabled=True):
                dest = viz.wsi.export_rois()
                viz.create_toast(f'ROIs saved to {dest}', icon='success')
                self.editing = False
                self.capturing = False

            # Load button.
            if imgui_utils.button('Load', width=viz.button_w, enabled=True):
                path = askopenfilename(title="Load ROIs...", filetypes=[("CSV", "*.csv",)])
                viz.wsi.load_csv_roi(path)
                viz.viewer.refresh_view()
                self.editing = False
                self.capturing = False

            imgui.end()

            if self.capturing:
                new_annotation, annotation_name = self.annotator.capture(
                    x_range=(viz.viewer.x_offset, viz.viewer.x_offset + viz.viewer.width),
                    y_range=(viz.viewer.y_offset, viz.viewer.y_offset + viz.viewer.height),
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
                    wsi_coords = np.array(wsi_coords)
                    viz.wsi.load_roi_array(wsi_coords)
                    viz.viewer.refresh_view()

            # Edit ROIs
            if self.editing:
                selected_roi = self.check_for_selected_roi()
                if imgui.is_mouse_down(1):
                    viz.viewer.deselect_roi()
                elif selected_roi is not None:
                    viz.viewer.deselect_roi()
                    viz.viewer.select_roi(selected_roi)

        else:
            self.capturing = False
            self.editing = False
#----------------------------------------------------------------------------
