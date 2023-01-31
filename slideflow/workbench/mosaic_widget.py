import time
import os
import glfw
import numpy as np
import pandas as pd
import imgui
import slideflow as sf
import multiprocessing as mp
from os.path import join, exists
from slideflow import log
from typing import Optional

from .gui_utils import imgui_utils, gl_utils, viewer
from .utils import EasyDict


class OpenGLMosaic(sf.mosaic.Mosaic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _initialize_figure(self, figsize, background):
        pass

    def _add_patch(self, loc, size, **kwargs):
        pass

    def _plot_tile_image(self, image, extent, alpha=1):
        return image

    def _finalize_figure(self):
        pass


class MosaicViewer(viewer.Viewer):

    movable = True
    live    = False

    def __init__(self, mosaic, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mosaic = mosaic
        self.mosaic_x_offset = 0
        self.mosaic_y_offset = 0

    @property
    def view_offset(self):
        return (self.mosaic_x_offset, self.mosaic_y_offset)

    def move(self, dx: float, dy: float) -> None:
        pass

    def refresh_view(self, view_params: Optional[EasyDict] = None) -> None:
        pass

    def render(self, max_w: int, max_h: int) -> None:
        """Render the mosaic map."""

        size = min(max_w, max_h)
        if max_w > size:
            self.mosaic_x_offset = (max_w - size) / 2
            self.mosaic_y_offset = 0
        else:
            self.mosaic_x_offset = 0
            self.mosaic_y_offset = (max_h - size) / 2

        image_size = int(size / self.mosaic.num_tiles_x)
        imgui.set_next_window_bg_alpha(0)

        # Set the window position by the offset, in points (not pixels)
        imgui.set_next_window_position(self.viz.offset_x, self.viz.offset_y)
        imgui.set_next_window_size(max_w, max_h)
        imgui.begin("Mosaic", flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_FOCUS_ON_APPEARING | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
        for (x, y), image in self.mosaic.grid_images.items():
            if isinstance(image, np.ndarray):
                self.mosaic.grid_images[(x, y)] = gl_utils.Texture(
                    image=image, bilinear=True, mipmap=False
                )
            gl_id = self.mosaic.grid_images[(x, y)].gl_id
            pos = (
                self.mosaic_x_offset + x * image_size,
                self.mosaic_y_offset + size - (y * image_size)
            )
            imgui.set_cursor_pos(pos)
            imgui.image(gl_id, image_size, image_size)
        imgui.end()
        if 'message' in self.viz.result:
            del self.viz.result.message

    def zoom(self, cx: int, cy: int, dz: float) -> None:
        pass

    def reload(self, **kwargs):
        pass


#----------------------------------------------------------------------------

class MosaicWidget:
    def __init__(self, viz, width=800, s=3, debug=False):
        self.viz            = viz
        self.show           = True
        self.content_height = 0
        self._umap_path     = None
        self._umap_layers   = []
        self.coords         = dict()
        self._plot_images   = dict()
        self._plot_transforms = dict()
        self.visible        = False
        self.width          = 800
        self.show           = s
        self.dpi            = 300
        self.debug          = debug
        self.slidemap       = None
        self.num_tiles_x    = 50
        ctx = mp.get_context('fork')
        self.pool = ctx.Pool(os.cpu_count())


    def _plot_coords(self):
        import matplotlib.pyplot as plt
        from PIL import Image
        from io import BytesIO
        import matplotlib.image as mpl_image

        for layer_name in self.coords:
            # Create figure
            x, y = self.coords[layer_name][:, 0], self.coords[layer_name][:, 1]
            fig = plt.figure(figsize=(self.width/self.dpi, self.width/self.dpi), dpi=self.dpi)
            plt.scatter(x, y, s=0.2)
            plt.axis('off')
            plt.subplots_adjust(0, 0, 1, 1, 0, 0)
            plt.tight_layout()
            ax = plt.gca()

            # Convert figure as texture
            stream = BytesIO()
            plt.savefig(stream, format='raw', transparent=True, dpi=self.dpi)
            pilImage = Image.frombytes('RGBA', size=(self.width, self.width), data=stream.getvalue())
            image_array = mpl_image.pil_to_array(pilImage)
            tex_obj = self._tex_obj = gl_utils.Texture(image=image_array, bilinear=False, mipmap=False)
            self._plot_images[layer_name] = tex_obj.gl_id
            self._plot_transforms[layer_name] = ax.transData.transform

    @property
    def mosaic_kwargs(self):
        return dict(num_tiles_x=self.num_tiles_x, tile_select='first')

    def keyboard_callback(self, key, action):
        if not self.viz._control_down and (key == glfw.KEY_EQUAL and action == glfw.PRESS):
            self.increase_mosaic_resolution()
        if not self.viz._control_down and (key == glfw.KEY_MINUS and action == glfw.PRESS):
            self.decrease_mosaic_resolution()

    def increase_mosaic_resolution(self):
        self.num_tiles_x = int(self.num_tiles_x * 1.5)
        self.mosaic.generate_grid(**self.mosaic_kwargs)
        self.mosaic.plot(pool=self.pool)

    def decrease_mosaic_resolution(self):
        self.num_tiles_x = int(self.num_tiles_x / 1.5)
        self.mosaic.generate_grid(**self.mosaic_kwargs)
        self.mosaic.plot(pool=self.pool)

    # Todo: need preprocessing function for SimCLR
    def load_umap(self, path, model, layers='postconv', subsample=500):
        if path != self._umap_path:
            self._umap_path = path
            self.coords = {}
            if path is not None and exists(join(path, 'encoder')):
                self._umap_layers = ['umap']
                layer = 'umap'
                df = pd.read_parquet(join(path, 'slidemap.parquet'))
                self.coords[layer] = np.stack((df.x.values, df.y.values), axis=1)
                if subsample and self.coords[layer].shape[0] > subsample:
                    idx = np.random.choice(self.coords[layer].shape[0], subsample)
                    self.coords[layer] = self.coords[layer][idx]
                features_model, input_tensor = self.load_model(model, layers=layers)
                self.viz._umap_encoders = self.load_umap_encoder(path, features_model, input_tensor)
                self.viz._async_renderer._umap_encoders = self.viz._umap_encoders
                self._plot_coords()
                self.slidemap = sf.SlideMap(cache=join(self._umap_path, 'slidemap.parquet'))
                log.info(f"Loaded UMAP; displaying {self.coords[layer].shape[0]} points.")
            else:
                self._umap_layers = []

    def generate_mosaic(self, tfrecords):
        self.mosaic = OpenGLMosaic(self.slidemap, tfrecords=tfrecords, **self.mosaic_kwargs)
        self.mosaic.plot()
        self.viz.set_viewer(MosaicViewer(self.mosaic, **self.viz._viewer_kwargs()))

    def load_model(self, path, layers='postconv', **kwargs):
        is_simclr = sf.util.is_simclr_model_path(path)
        if is_simclr:
            import tensorflow as tf
            from slideflow.simclr import SimCLR
            model = SimCLR(2)
            model.num_features = 128
            model.num_logits = 2
            checkpoint = tf.train.Checkpoint(
                model=model,
                global_step=tf.Variable(0, dtype=tf.int64)
            )
            checkpoint.restore(path).expect_partial()
            inp = tf.keras.layers.InputLayer(input_shape=(96, 96, 3), name='input')
            input_tensor = inp.input
            model.outputs = model(inp.output, training=False)
        else:
            model = sf.model.Features(model, layers=layers, include_logits=True, **kwargs).model
            input_tensor = None
        return model, input_tensor

    def load_umap_encoder(self, path, feature_model, input_tensor=None):
        """Assumes `feature_model` has two outputs: (features, logits)"""
        import tensorflow as tf

        encoder = tf.keras.models.load_model(join(path, 'encoder'))
        encoder._name = f'umap_encoder'
        outputs = [encoder(feature_model.outputs[0])]

        # Add the logits output
        outputs += [feature_model.outputs[-1]]

        # Build the encoder model for all layers
        encoder_model = tf.keras.models.Model(
            inputs=input_tensor if input_tensor is not None else feature_model.input,
            outputs=outputs
        )
        return EasyDict(
            encoder=encoder_model,
            layers=['umap'],
            range={'umap': np.load(join(path, 'range_clip.npz'))['range']},
            clip={'umap': np.load(join(path, 'range_clip.npz'))['clip']}
        )

    def view_menu_options(self):
        if imgui.menu_item('Toggle Layer UMAPs', enabled=bool(self._umap_layers))[1]:
            self.show = not self.show

    def render(self):
        viz = self.viz

        #self.refresh_umap_path(viz._umap_path)
        viz.args.use_umap_encoders = len(self._umap_layers) > 0

        # --- Draw plot with OpenGL ---------------------------------------

        if self.show and self._umap_layers:
            imgui.set_next_window_size((self.width+viz.spacing) * len(self._umap_layers) + viz.spacing, self.width+50)
            _, self.show = imgui.begin("##layer_plot", closable=True, flags=(imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE))
            _tx, _ty = imgui.get_window_position()
            _ty += 45

            for i, layer_name in enumerate(self._umap_layers):
                # Draw labeled bounding box
                tx = (self.width * i) + _tx + viz.spacing
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
                    tx + self.width,
                    ty + self.width,
                    imgui.get_color_u32_rgba(1, 1, 1, 1),
                    thickness=1)
                transform_fn = self._plot_transforms[layer_name]
                draw_list.add_image(self._plot_images[layer_name], (tx, ty), (tx+self.width, ty+self.width))

                # Draw the bottom-left (red) and top_right (green)
                # coordinate points, to verify alignment
                if self.debug:
                    bottom_left = transform_fn([self.coords[layer_name][:, 0].min(), self.coords[layer_name][:, 1].min()])
                    top_right = transform_fn([self.coords[layer_name][:, 0].max(), self.coords[layer_name][:, 1].max()])
                    draw_list.add_circle_filled(
                        tx + bottom_left[0],
                        ty + self.width - bottom_left[1],
                        self.s,
                        imgui.get_color_u32_rgba(1, 0, 0, 1)
                    )
                    draw_list.add_circle_filled(
                        tx + top_right[0],
                        ty + self.width - top_right[1],
                        self.s,
                        imgui.get_color_u32_rgba(0, 1, 0, 1)
                    )

                ## Plot location of tile
                if 'umap_coords' in viz.result and viz.result.umap_coords:
                    fc = viz.result.umap_coords[layer_name]
                    transformed = transform_fn(fc)
                    draw_list.add_circle_filled(
                        tx + transformed[0],
                        ty + self.width - transformed[1],
                        self.s * 2,
                        imgui.get_color_u32_rgba(1, 0, 0, 1)
                    )
            imgui.end()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        return

#----------------------------------------------------------------------------
