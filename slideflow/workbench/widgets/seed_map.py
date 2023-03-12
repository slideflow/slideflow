import os
import numpy as np
import pandas as pd
import imgui
from os.path import join, exists, dirname

from ..gui import imgui_utils

#----------------------------------------------------------------------------

class SeedMapWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.show           = True
        self.content_height = 0
        self._pkl           = None
        self._umap_dfs      = dict()
        self.coords         = dict()
        self.coord_seeds    = dict()
        self.coord_classes  = dict()
        self.visible        = False
        self.nearest        = None
        self.nearest_class  = None
        self._random_seeds  = None

    def refresh_model_path(self):
        if hasattr(self.viz, 'pkl') and self._pkl != self.viz.pkl and self._pkl:
            self._pkl = self.viz.pkl
            self.coords = {}
            seed_maps_dir = join(dirname(self._pkl), 'seed_maps')
            if exists(seed_maps_dir):
                seed_maps = [u[:-8] for u in os.listdir(seed_maps_dir)
                              if u.endswith('.parquet')]
            else:
                seed_maps = []
            if self._pkl is not None and len(seed_maps):
                for layer in sorted(seed_maps):
                    df = pd.read_parquet(join(seed_maps_dir, f'{layer}.parquet'))

                    # Normalize layout
                    norm = np.stack((df.x.values, df.y.values), axis=1)
                    norm -= norm.min(axis=0)
                    norm /= (norm.max(axis=0) - norm.min(axis=0))
                    self.coords[layer] = norm
                    self.coord_seeds[layer] = df.seed.values
                    if 'class' in df.columns:
                        self.coord_classes[layer] = df['class'].values

                    self._umap_dfs[layer] = df
            else:
                self._umap_dfs = dict()

            for layer in self.coords:
                print("Layer {} | n_seeds: {} n_coords: {}".format(layer, len(self.coords[layer]), len(self.coord_seeds[layer])))

    def view_menu_options(self):
        if imgui.menu_item('Toggle Seed UMAPs', enabled=bool(self._umap_dfs))[1]:
            self.show = not self.show

    def render(self):
        viz = self.viz

        self.refresh_model_path()

        # --- Draw plot with OpenGL ---------------------------------------
        if self.show and self._umap_dfs:
            window_width = 310 * len(self._umap_dfs)
            window_height = 350
            imgui.set_next_window_size(window_width, window_height)
            _, self.show = imgui.begin("##seed_map", closable=True, flags=(imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE))
            _tx, _ty = imgui.get_window_position()
            _ty += 45

            for i, layer_name in enumerate(self._umap_dfs):
                # Adjust origin offset
                tx = (300 * i) + _tx + viz.spacing
                ty = _ty

                # Handle user input
                clicking, cx, cy = imgui_utils.click_previous_control(mouse_idx=1, enabled=True)
                cx -= tx
                cy -= ty
                cx_norm = cx / 300
                cy_norm = 1 - (cy / 300)
                if cx < 0 or cy < 0 or cx > 300 or cy > 300:
                    clicking = False

                # Draw labeled bounding box
                draw_list = imgui.get_window_draw_list()
                draw_list.add_text(
                    tx + 10,
                    ty - 20,
                    imgui.get_color_u32_rgba(1, 1, 1, 1),
                    f'Seed Map: {layer_name}')
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
                        2,
                        imgui.get_color_u32_rgba(0.19, 0.27, 0.38, 1)
                    )

                # Plot location of tile
                _c = self.coords[layer_name]
                if clicking:
                    # Find nearest point
                    dist = np.sqrt(  (_c[:, 0] - cx_norm) ** 2
                                   + (_c[:, 1] - cy_norm) ** 2)
                    _idx = np.argmin(dist)
                    self.nearest = self.coord_seeds[layer_name][_idx]
                    if self.coord_classes:
                        self.nearest_class = self.coord_classes[layer_name][_idx]
                    if hasattr(self.viz, 'latent_widget'):
                        self.viz.latent_widget.set_seed(self.nearest)
                        if self.coord_classes:
                            self.viz.latent_widget.set_class(self.nearest_class)
                if self.nearest:
                    if self.coord_classes:
                        idx = np.argwhere(((self.coord_seeds[layer_name] == self.nearest)
                                           & (self.coord_classes[layer_name] == self.nearest_class)))
                    else:
                        idx = np.argwhere(self.coord_seeds[layer_name] == self.nearest)

                    if len(idx):
                        assert len(idx) == 1
                        idx = idx[0][0]
                        draw_list.add_circle_filled(
                            tx + (_c[idx][0] * 300),
                            ty + ((1-_c[idx][1]) * 300),
                            5,
                            imgui.get_color_u32_rgba(1, 0, 0, 1)
                        )

            imgui.end()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        return

#----------------------------------------------------------------------------
