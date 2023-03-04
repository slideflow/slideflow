import pandas as pd
import imgui
import numpy as np

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
        self.annotations    = []
        self._late_render   = []

    @property
    def visible(self):
        return self.viz.wsi is not None

    def late_render(self):
        for _ in range(len(self._late_render)):
            annotation, name, kwargs = self._late_render.pop()
            gl_utils.draw_roi(annotation, **kwargs)
            if isinstance(name, str):
                tex = text_utils.get_texture(name, size=self.viz.gl_font_size, max_width=self.viz.viewer.width, max_height=self.viz.viewer.height, outline=2)
                text_pos = (annotation.mean(axis=0))
                tex.draw(pos=text_pos, align=0.5, rint=True, color=1)

    def render_annotation(self, annotation, origin, name=None, color=1, alpha=1, linewidth=3):
        kwargs = dict(color=color, linewidth=linewidth, alpha=alpha)
        self._late_render.append((np.array(annotation) + origin, name, kwargs))

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show and self.visible:
            imgui.begin('ROIs', flags=imgui.WINDOW_NO_RESIZE)
            button_txt = 'Capture' if not self.capturing else 'Capturing...'
            if imgui_utils.button(button_txt, width=viz.button_w, enabled=True):
                self.capturing = not self.capturing
            if imgui_utils.button('Save', width=viz.button_w, enabled=True):
                dest = viz.wsi.export_rois()
                viz.create_toast(f'ROIs saved to {dest}', icon='success')
            if imgui_utils.button('Load', width=viz.button_w, enabled=True):
                path = askopenfilename(title="Load ROIs...", filetypes=[("CSV", "*.csv",)])
                viz.wsi.load_csv_roi(path)
                viz.viewer.refresh_view()
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

#----------------------------------------------------------------------------
