import os
import numpy as np
import pandas as pd
import imgui
from os.path import join, exists

from ..gui import imgui_utils

#----------------------------------------------------------------------------

class LayerUMAPWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.show           = True
        self.content_height = 0
        self._model_path    = None
        self._umap_layers   = []
        self.coords         = dict()
        self.visible        = False

    def refresh_model_path(self, path):
        if path != self._model_path:
            self._model_path = path
            self.coords = {}
            if path is not None and exists(join(path, 'umap_encoders')):
                self._umap_layers = [
                    d for d in os.listdir(join(path, 'umap_encoders'))
                    if os.path.isdir(join(path, 'umap_encoders', d))
                ]
                for layer in self._umap_layers:
                    layer_path = join(path, 'umap_encoders', layer)
                    df = pd.read_parquet(join(layer_path, 'slidemap.parquet'))
                    self.coords[layer] = np.stack((df.x.values, df.y.values), axis=1)
                    if self.coords[layer].shape[0] > 500:
                        idx = np.random.choice(self.coords[layer].shape[0], 500)
                        self.coords[layer] = self.coords[layer][idx]

            else:
                self._umap_layers = []

    def view_menu_options(self):
        if imgui.menu_item('Toggle Layer UMAPs', enabled=bool(self._umap_layers))[1]:
            self.show = not self.show

    def render(self):
        viz = self.viz

        self.refresh_model_path(viz._model_path)
        viz.args.use_umap_encoders = len(self._umap_layers) > 0

        # --- Draw plot with OpenGL ---------------------------------------

        if self.show and self._umap_layers:
            imgui.set_next_window_size(300 * len(self._umap_layers), 350)
            _, self.show = imgui.begin("##layer_plot", closable=True, flags=(imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE))
            _tx, _ty = imgui.get_window_position()
            _ty += 45

            for i, layer_name in enumerate(self._umap_layers):
                # Draw labeled bounding box
                tx = (300 * i) + _tx + viz.spacing
                ty = _ty
                draw_list = imgui.get_window_draw_list()
                draw_list.add_text(
                    tx + 10,
                    ty - 20,
                    imgui.get_color_u32_rgba(1, 1, 1, 1),
                    layer_name)
                draw_list.add_rect(
                    tx,
                    ty,
                    tx + 300,
                    ty + 300,
                    imgui.get_color_u32_rgba(1, 1, 1, 1),
                    thickness=1)

                # Plot reference points
                # Origin is bottom left
                for c in self.coords[layer_name]:
                    draw_list.add_circle_filled(
                        tx + (c[0] * 300),
                        ty + ((1-c[1]) * 300),
                        3,
                        imgui.get_color_u32_rgba(0.35, 0.25, 0.45, 1)
                    )

                # Plot location of tile
                if 'umap_coords' in viz.result and viz.result.umap_coords:
                    fc = viz.result.umap_coords[layer_name]
                    draw_list.add_circle_filled(
                        tx + (fc[0] * 300),
                        ty + ((1-fc[1]) * 300),
                        5,
                        imgui.get_color_u32_rgba(1, 0, 0, 1)
                    )
            imgui.end()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        return

#----------------------------------------------------------------------------
