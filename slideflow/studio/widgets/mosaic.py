import os
import numpy as np
import pandas as pd
import imgui
import slideflow as sf
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.image as mpl_image
from os.path import join, exists, dirname, abspath
from slideflow import log
from PIL import Image
from io import BytesIO
from tkinter.filedialog import askdirectory

from ._utils import Widget
from ..gui import imgui_utils, gl_utils, text_utils
from ..gui.viewer import OpenGLMosaic, MosaicViewer
from ..gui.annotator import AnnotationCapture


#----------------------------------------------------------------------------

class MosaicWidget(Widget):

    tag = 'mosaic'
    description = 'Mosaic Map'
    icon = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_mosaic.png')
    icon_highlighted = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_mosaic_highlighted.png')

    def __init__(self, viz, width=500, s=2, debug=False):
        self.viz            = viz
        self.show_umap      = True
        self.content_height = 0
        self.coords         = None
        self.visible        = False
        self.width          = width
        self.s              = s
        self.dpi            = 300
        self.debug          = debug
        self.slidemap       = None
        self.num_tiles_x    = 50
        self.pool           = None
        self.annotator      = AnnotationCapture(named=True)
        self.annotations    = []
        self.mosaic         = None
        self._plot_images   = None
        self._plot_transforms = None
        self._bottom_left   = None
        self._top_right     = None
        self._umap_width    = None
        self._clicking      = None
        self._show_popup    = False
        self._late_render_annotations = []

    def _plot_coords(self):
        """Convert coordinates to a rendered figure / texture with matplotlib."""
        # Create figure
        x, y = self.coords[:, 0], self.coords[:, 1]
        fig = plt.figure(figsize=(self.width/self.dpi, self.width/self.dpi), dpi=self.dpi)
        plt.scatter(x, y, s=0.1)
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
        self._plot_images = tex_obj.gl_id
        self._plot_transforms = ax.transData.transform

        # Save origin / key information for the images
        self._bottom_left = np.round(ax.transData.transform([self.coords[:, 0].min(), self.coords[:, 1].min()]))
        self._top_right = np.round(ax.transData.transform([self.coords[:, 0].max(), self.coords[:, 1].max()]))
        self._umap_width = self._top_right[0] - self._bottom_left[0]

    @property
    def mosaic_kwargs(self):
        return dict(num_tiles_x=self.num_tiles_x, tile_select='first')

    def view_menu_options(self):
        if imgui.menu_item('Toggle Mosaic UMAP', enabled=(self.coords is not None))[1]:
            self.show_umap = not self.show_umap

    def open_menu_options(self):
        if imgui.menu_item('Load Mosaic...')[1]:
            mosaic_path = askdirectory(title="Load mosaic (directory)...")
            if mosaic_path:
                self.load(mosaic_path)

    def close(self):
        """Close the multiprocessing pool."""
        del self.mosaic
        del self.coords
        self.viz.reset_background()
        self.annotations = []
        self.slidemap = None
        self.mosaic = None
        self.coords = None
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
        self.pool = None
        self.viz.viewer = None

    def refresh_mosaic_resolution(self):
        self.mosaic.generate_grid(**self.mosaic_kwargs)
        self.mosaic.plot(pool=self.pool)

    def load(self, obj, tfrecords=None, slides=None, **kwargs):
        """Load a UMAP from a file or SlideMap object."""
        if isinstance(obj, str):
            try:
                self.load_umap_from_path(obj, **kwargs)
            except Exception:
                self.viz.create_toast("Failed to load mosaic map.", icon='error')
                return
        elif isinstance(obj, sf.SlideMap):
            self.load_umap_from_slidemap(obj, **kwargs)
        else:
            raise ValueError(f"Unrecognized argument: {obj}")
        self.generate(tfrecords=tfrecords, slides=slides)

    def load_umap_from_slidemap(self, slidemap, subsample=5000):
        """Load a UMAP from a SlideMap object."""
        df = slidemap.data
        self.slidemap = slidemap
        self.coords = np.stack((df.x.values, df.y.values), axis=1)
        if subsample and self.coords.shape[0] > subsample:
            idx = np.random.choice(self.coords.shape[0], subsample)
            self.coords = self.coords[idx]
        self._plot_coords()
        log.info(f"Loaded UMAP; displaying {self.coords.shape[0]} points.")

    def load_umap_from_path(self, path, subsample=5000):
        """Load a saved UMAP."""
        if path is not None and exists(join(path, 'slidemap.parquet')):
            df = pd.read_parquet(join(path, 'slidemap.parquet'))
            self.coords = np.stack((df.x.values, df.y.values), axis=1)
            if subsample and self.coords.shape[0] > subsample:
                idx = np.random.choice(self.coords.shape[0], subsample)
                self.coords = self.coords[idx]
            self._plot_coords()
            self.slidemap = sf.SlideMap.load(path)
            log.info(f"Loaded UMAP; displaying {self.coords.shape[0]} points.")
        else:
            raise ValueError(f"Could not find UMAP as path {path}")

    def generate(self, tfrecords=None, slides=None):
        """Build the mosaic."""
        if self.slidemap is None:
            raise ValueError("Cannot generate mosaic; no SlideMap loaded.")
        if tfrecords is None and self.slidemap.tfrecords is None:
            raise ValueError(
                "TFRecords not found and not provided. Please provide paths "
                "to TFRecords with generate_mosaic(tfrecords=...)"
            )
        if tfrecords is None:
            tfrecords = self.slidemap.tfrecords
        if self.pool is None:
            ctx = mp.get_context('fork')
            self.pool = ctx.Pool(
                os.cpu_count(),
                initializer=sf.util.set_ignore_sigint
            )
        self.mosaic = OpenGLMosaic(self.slidemap, tfrecords=tfrecords, **self.mosaic_kwargs)
        self.mosaic.plot()
        self.viz.set_viewer(MosaicViewer(self.mosaic, slides=slides, **self.viz._viewer_kwargs()))

    def late_render(self):
        """Render UMAP plot annotations late, to ensure they are on top."""
        for _ in range(len(self._late_render_annotations)):
            annotation, name, kwargs = self._late_render_annotations.pop()
            gl_utils.draw_roi(annotation, **kwargs)
            if name is not None:
                tex = text_utils.get_texture(name, size=self.viz.gl_font_size, max_width=self.viz.viewer.width, max_height=self.viz.viewer.height, outline=2)
                text_pos = (annotation.mean(axis=0))
                tex.draw(pos=text_pos, align=0.5, rint=True, color=1)

    def render_annotation(self, annotation, origin, name=None, color=1, alpha=1, linewidth=3):
        """Convert annotations to textures."""
        kwargs = dict(color=color, linewidth=linewidth, alpha=alpha)
        self._late_render_annotations.append((np.array(annotation) + origin, name, kwargs))

    def render(self):
        """Render the figure."""
        viz = self.viz

        # --- Draw plot with OpenGL ---------------------------------------

        if self.show_umap and (self.coords is not None):
            imgui.set_next_window_size(self.width, self.width)
            _, self.show_umap = imgui.begin("##layer_plot", closable=True, flags=(imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE))
            tx, ty = imgui.get_window_position()
            draw_list = imgui.get_window_draw_list()
            transform_fn = self._plot_transforms
            draw_list.add_image(self._plot_images, (tx, ty), (tx+self.width, ty+self.width))

            # Draw the bottom-left (red) and top_right (green)
            # coordinate points, to verify alignment
            left_x = self._bottom_left[0]
            right_x = self._top_right[0]
            bottom_y = self.width - self._bottom_left[1]
            top_y = self.width - self._top_right[1]
            if self.debug:
                draw_list.add_circle_filled(
                    tx + left_x,
                    ty + bottom_y,
                    self.s,
                    imgui.get_color_u32_rgba(1, 0, 0, 1)
                )
                draw_list.add_circle_filled(
                    tx + right_x,
                    ty + top_y,
                    self.s,
                    imgui.get_color_u32_rgba(0, 1, 0, 1)
                )

            # Draw mosaic map view
            if isinstance(viz.viewer, MosaicViewer) and viz.viewer.zoomed:
                mosaic_ratio = viz.viewer.size / self._umap_width
                _x_off = viz.viewer.mosaic_x_offset / mosaic_ratio
                _y_off = viz.viewer.mosaic_y_offset / mosaic_ratio
                _w = viz.viewer.width / mosaic_ratio
                _h = viz.viewer.height / mosaic_ratio
                draw_list.add_rect(
                    tx + left_x - _x_off,
                    ty + top_y  - _y_off,
                    tx + left_x - _x_off + _w,
                    ty + top_y  - _y_off + _h ,
                    imgui.get_color_u32_rgba(1, 0, 0, 1),
                    thickness=2
                )

            # Plot location of tile
            if 'umap_coords' in viz.result and viz.result.umap_coords:
                fc = viz.result.umap_coords['mosaic_umap']
                transformed = transform_fn(fc)
                draw_list.add_circle_filled(
                    tx + transformed[0],
                    ty + self.width - transformed[1],
                    self.s * 2,
                    imgui.get_color_u32_rgba(1, 0, 0, 1)
                )

            # Capture mouse input
            new_annotation, annotation_name = self.annotator.capture(
                x_range=(tx+left_x, tx+right_x),
                y_range=(ty+top_y, ty+bottom_y),
            )
            imgui.end()

            # Render in-progress annotations
            if new_annotation is not None:
                self.render_annotation(new_annotation, origin=(tx+left_x, ty+top_y))
            if annotation_name:
                self.annotations.append((annotation_name, new_annotation))

            # Render completed annotations
            for name, annotation in self.annotations:
                self.render_annotation(annotation, origin=(tx+left_x, ty+top_y), name=name, color=(0.8, 0.4, 0.4))

    def ask_load_mosaic(self):
        mosaic_path = askdirectory(title="Load mosaic (directory)...")
        if mosaic_path:
            self.load(mosaic_path)

    def draw_info(self):
        viz = self.viz

        n_tfr = str(len(self.mosaic.tfrecords))
        n_tiles = str(len(self.coords))
        imgui.text_colored('TFRecords', *viz.theme.dim)
        imgui_utils.right_align(n_tfr)
        imgui.text(n_tfr)
        imgui.text_colored('Tiles plotted', *viz.theme.dim)
        imgui_utils.right_align(n_tiles)
        imgui.text(n_tiles)


    def draw_config_popup(self):
        viz = self.viz
        has_model = viz._model_config is not None

        if self._show_popup:
            cx, cy = imgui.get_cursor_pos()
            imgui.set_next_window_position(viz.sidebar.full_width, cy)
            imgui.begin(
                '##mosaic_popup',
                flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
            )
            if imgui.menu_item('Load mosaic')[0]:
                self.ask_load_mosaic()
            if imgui.menu_item('Close mosaic')[0]:
                self.close()
            imgui.separator()
            if imgui.menu_item('Show UMAP', enabled=self.mosaic is not None, selected=self.show_umap)[0]:
                self.show_umap = not self.show_umap

            # Hide menu if we click elsewhere
            if imgui.is_mouse_down(0) and not imgui.is_window_hovered():
                self._clicking = True
            if self._clicking and imgui.is_mouse_released(0):
                self._clicking = False
                self._show_popup = False

            imgui.end()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            with viz.header_with_buttons("Mosaic"):
                imgui.same_line(imgui.get_content_region_max()[0] - viz.font_size*1.5)
                cx, cy = imgui.get_cursor_pos()
                imgui.set_cursor_position((cx, cy-int(viz.font_size*0.25)))
                if viz.sidebar.small_button('gear'):
                    self._clicking = False
                    self._show_popup = not self._show_popup
                self.draw_config_popup()

        if show and not isinstance(viz.viewer, MosaicViewer):
            imgui_utils.padded_text('No mosaic has been loaded.', vpad=[int(viz.font_size/2), int(viz.font_size)])
            if viz.sidebar.full_button("Load a Mosaic"):
                self.ask_load_mosaic()

        elif show:
            if viz.collapsing_header('Info', default=True):
                self.draw_info()
            if viz.collapsing_header('Config', default=True):
                grid_changed, grid_size = imgui.input_int("Grid size", self.num_tiles_x, step=25, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                if grid_changed and grid_size >= 25:
                    self.num_tiles_x = grid_size
                    self.refresh_mosaic_resolution()
                _, color = imgui.color_edit3("Background", *viz._background_color[:3])
                viz._background_color = (color[0], color[1], color[2], 1)
