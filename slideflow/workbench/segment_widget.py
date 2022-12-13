import slideflow as sf
import imgui
import numpy as np
import cellpose
import cellpose.models
from functools import partial
from threading import Thread
from typing import Tuple
from PIL import Image, ImageDraw
from slideflow.slide.utils import draw_roi
from slideflow.seg.cell import segment_slide, Segmentation
from .gui_utils import imgui_utils


class SegmentWidget:
    def __init__(self, viz):
        self.viz                    = viz
        self.header                 = "Cellpose"
        self.alpha                  = 1
        self.show                   = True
        self.diam_radio_auto        = True
        self.use_gpu                = True
        self.diam_radio_manual      = False
        self.segmentation           = None
        self.diam_manual            = 10
        self.overlay                = None
        self.content_height         = 0
        self.show_mask              = True
        self.show_gradXY            = False
        self.show_centroid          = False
        self.overlay_background     = True

        self._showing_preview_overlay = False
        self._selected_model_idx    = 1
        self._auto_tile_px          = 256
        self._auto_mpp              = 0.5
        self._wsi_config            = 'auto'
        self._segment_wsi_dim       = None
        self._segment_wsi_offset    = None
        self._thread                = None
        self._segment_toast         = None
        self._centroid_toast        = None
        self._load_toast            = None
        self._outline_toast         = None

        self.mpp                    = self._auto_mpp
        self.tile_px                = self._auto_tile_px

        # Load default model
        self.supported_models = [
            'cyto', 'cyto2', 'nuclei', 'tissuenet', 'TN1', 'TN2', 'TN3',
            'livecell', 'LC1', 'LC2', 'LC3', 'LC4'
        ]
        self.models = {m: None for m in self.supported_models}


    @property
    def model(self):
        model_str = self.supported_models[self._selected_model_idx]
        if self.models[model_str] is None:
            self.models[model_str] = cellpose.models.Cellpose(
                gpu=True,
                model_type=model_str)
        return self.models[model_str]

    @property
    def diameter(self):
        return None if self.diam_radio_auto else self.diam_manual

    def is_thread_running(self):
        return self._thread is not None and self._thread.is_alive()

    def _load_segmentation(self, path, ignore_errors=False):
        loaded = np.load(path)
        if 'masks' not in loaded and not ignore_errors:
            raise TypeError(f"Unable to load '{path}'; 'masks' index not found.")
        flows = None if 'flows' not in loaded else loaded[flows]
        self.segmentation = Segmentation(loaded['masks'], flows)
        self._segment_wsi_dim = loaded['wsi_dim']
        self._segment_wsi_offset = loaded['wsi_offset']
        if 'centroids' in loaded:
            self.segmentation._centroids = loaded['centroids']
        self.refresh_segmentation_view()
        self._load_toast.done()
        self.viz.create_toast(f"Loaded {self.segmentation.masks.max()} segmentations.", icon="success")

    def calculate_centroids(self, refresh=True):
        """Calculate segmentation centroids."""
        _should_announce_complete = self.show_centroid and (self.segmentation._centroids is None)
        self.segmentation.calculate_centroids()
        if self._centroid_toast:
            self._centroid_toast.done()
        if _should_announce_complete:
            self.viz.create_toast("Centroid calculation complete.", icon="success")
        if refresh:
            refresh_toast = self.viz.create_toast(
                title=f"Rendering",
                icon='info',
                sticky=True,
                spinner=True)
            self.refresh_segmentation_view()
            refresh_toast.done()

    def drag_and_drop_hook(self, path, ignore_errors=False):
        if 'masks' in np.load(path, mmap_mode='r').files:
            self.load(path, ignore_errors=ignore_errors)

    def load(self, path, ignore_errors=False):
        """Load a .npz with saved masks."""
        if self.is_thread_running():
            self._thread.join()
        self._load_toast = self.viz.create_toast(
            title=f"Loading masks",
            icon='info',
            sticky=True,
            spinner=True)
        self._thread = Thread(target=self._load_segmentation, args=(path, ignore_errors))
        self._thread.start()

    def refresh_segmentation_view(self):
        """Refresh the Workbench view of the active segmentation."""
        if self.segmentation is None:
            return
        if self.show_mask:
            self.overlay = self.segmentation.mask_to_image(centroid=self.show_centroid)
        elif self.show_gradXY:
            if self.show_centroid:
                self.overlay = self.segmentation._draw_centroid(self.segmentation.flows)
            else:
                self.overlay = self.segmentation.flows
        elif self.show_centroid:
            self.overlay = self.segmentation.centroid_to_image()

        self._showing_preview_overlay = False
        self.viz.overlay = self.overlay
        self.viz._overlay_wsi_dim = self._segment_wsi_dim
        self.viz._overlay_offset_wsi_dim = self._segment_wsi_offset

    def render_outlines(self, color='red'):
        """Render outlines of the segmentations currently in view."""
        in_view, _, view_offset = self.viz.viewer.in_view(
            self.segmentation.masks,
            dim=self._segment_wsi_dim,
            offset=self._segment_wsi_offset,
            resize=False)

        # Calculate and draw the outlines
        outlines = [o for o in sf.seg.cell.outlines_list(in_view) if o.shape[0] >= 3]
        empty = np.zeros((in_view.shape[0], in_view.shape[1], 3), dtype=np.uint8)
        outline_img = draw_roi(empty, outlines, color=color, linewidth=2)

        self.viz.overlay = outline_img
        self.viz._overlay_wsi_dim = (
            (in_view.shape[1] / self.segmentation.masks.shape[1]) * self._segment_wsi_dim[0],
            (in_view.shape[0] / self.segmentation.masks.shape[0]) * self._segment_wsi_dim[1],
        )
        self.viz._overlay_offset_wsi_dim = (
            self._segment_wsi_offset[0] + view_offset[0],
            self._segment_wsi_offset[1] + view_offset[1]
        )
        self._outline_toast.done()
        self.viz.create_toast("Outlining complete.", icon="success")
        self.update_transparency()

    def segment_view(self):
        """Segment the current view."""
        v = self.viz.viewer
        print(f"Segmenting image with diameter {self.diameter} (shape={v.view.shape})")
        masks, flows, styles, diams = self.model.eval(
            v.view,
            channels=[[0, 0]],
            diameter=self.diameter,
        )
        self.segmentation = Segmentation(masks, flows[0])
        self._segment_wsi_dim = v.wsi_window_size
        self._segment_wsi_offset = v.origin
        self.diam_manual = int(diams)
        print(f"Segmentation of current view complete (diameter={diams:.2f}), {masks.max()} total masks).")
        self.refresh_segmentation_view()

    def segment_slide(self):
        """Generate whole-slide segmentation.

        Args:
            auto (bool): Use recommended, automatic segmentation configuration.
                If false, will use user-specified settings.
        """
        wsi = sf.WSI(self.viz.wsi.path, tile_px=self.tile_px, tile_um=int(self.tile_px * self.mpp))
        print(f"Segmenting WSI (shape={wsi.dimensions}, tile_px={self.tile_px}, mpp={self.mpp}, diameter={self.diameter})")
        if self.viz.wsi.qc_mask is not None:
            wsi.apply_qc_mask(self.viz.wsi.qc_mask)

        # Perform segmentation.
        self.segmentation, _ = segment_slide(wsi, 'cyto2', diameter=self.diameter)
        print('Mask shape:', self.segmentation.masks.shape)
        full_extract = int(wsi.tile_um / wsi.mpp)
        wsi_stride = int(full_extract / wsi.stride_div)
        self._segment_wsi_dim = (wsi_stride * (wsi.grid.shape[0]),
                                 wsi_stride * (wsi.grid.shape[1]))
        self._segment_wsi_offset = (full_extract/2 - wsi_stride/2, full_extract/2 - wsi_stride/2)

        # Done; refresh view.
        self._segment_toast.done()
        refresh_toast = self.viz.create_toast(
                title=f"Rendering segmentations",
                icon='info',
                sticky=True,
                spinner=True)
        self.refresh_segmentation_view()
        refresh_toast.done()
        self.viz.create_toast("Segmentation complete.", icon="success")

    @staticmethod
    def set_image_alpha(img, alpha, kind='mask'):
        if kind == 'mask':
            alpha_channel = np.full(img.shape[0:2], int(alpha * 255), dtype=np.uint8)
        elif kind == 'outline':
            alpha_channel = ((img[:, :, 0:3].max(axis=-1)).astype(bool).astype(np.uint8) * (255 * alpha)).astype(np.uint8)
        else:
            raise ValueError(f"Unrecognized kind {kind}")
        return np.dstack((img[:, :, 0:3], alpha_channel))

    def update_transparency(self):
        """Updates transparency of the overlay."""
        kind = 'mask' if self.overlay_background else 'outline'
        self.viz.viewer.set_overlay_alpha(partial(self.set_image_alpha, alpha=self.alpha, kind=kind))

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if not show:
            self.content_height = 0
            return

        # Set up settings interface.
        child_width = imgui.get_content_region_max()[0] / 3 - viz.spacing
        child_height = imgui.get_text_line_height_with_spacing() * 6 + viz.spacing * 2
        self.content_height = child_height + viz.spacing

        # --- Segmentation ----------------------------------------------------
        imgui.begin_child('##segment_child', width=child_width, height=child_height, border=True)
        imgui.text("Whole-slide segmentation")
        imgui.separator()

        if imgui.radio_button('Auto##config_radio_auto', self._wsi_config == 'auto'):
            self._wsi_config = 'auto'
        imgui.same_line(viz.font_size*6)
        _, self.use_gpu = imgui.checkbox('Use GPU', self.use_gpu)


        if imgui.radio_button('In view##config_radio_manual', self._wsi_config == 'in_view'):
            self._wsi_config = 'in_view'
        imgui.same_line(viz.font_size*6)
        with imgui_utils.grayed_out(self._wsi_config == 'auto'):
            if self._wsi_config == 'auto':
                self.tile_px = self._auto_tile_px
            with imgui_utils.item_width(viz.font_size*2):
                _, self.tile_px = imgui.input_int('##tile_px', self.tile_px, step=0)
            imgui.same_line()
            imgui.text('Tile (px)')

        if imgui.radio_button('Manual##config_radio_manual', self._wsi_config == 'manual'):
            self._wsi_config = 'manual'
        imgui.same_line(viz.font_size*6)
        with imgui_utils.grayed_out(self._wsi_config != 'manual'):
            if self._wsi_config == 'auto':
                self.mpp = self._auto_mpp
            elif self._wsi_config == 'in_view' and hasattr(viz, 'viewer') and hasattr(viz.viewer, 'mpp'):
                self.mpp = viz.viewer.mpp
            with imgui_utils.item_width(viz.font_size*3):
                _, self.mpp = imgui.input_float('##mpp', self.mpp)
                imgui.same_line()
            imgui.text('MPP')

        # WSI segmentation.
        if imgui_utils.button("Segment", width=viz.button_w) and not self.is_thread_running():
            self._segment_toast = viz.create_toast(
                title=f"Segmenting whole-slide image",
                icon='info',
                sticky=True,
                spinner=True)
            self._thread = Thread(target=self.segment_slide)
            self._thread.start()

        # Export
        imgui.same_line(viz.font_size*6)
        with imgui_utils.grayed_out(self.segmentation is None):
            if imgui_utils.button("Export", width=viz.button_w) and self.segmentation is not None:
                filename = f'{viz.wsi.name}-masks.npz'
                np.savez(
                    filename,
                    masks=self.segmentation.masks,
                    centroids=self.segmentation.centroids,
                    flows=self.segmentation.flows,
                    wsi_dim=self._segment_wsi_dim,
                    wsi_offset=self._segment_wsi_offset)
                viz.create_toast(f"Exported masks and centroids to {filename}", icon='success')
        imgui.end_child()

        # --- Configuration ---------------------------------------------------
        imgui.same_line()
        imgui.begin_child('##config_child', width=child_width, height=child_height, border=True)
        imgui.text("Model & cell diameter")
        imgui.separator()

        with imgui_utils.grayed_out(self._wsi_config == 'auto'):

            ## Cell segmentation model.
            with imgui_utils.item_width(child_width - viz.spacing * 2):
                _clicked, self._selected_model_idx = imgui.combo(
                    "##cellpose_model",
                    self._selected_model_idx,
                    self.supported_models)

            ## Cell Segmentation diameter.
            if imgui.radio_button('Auto diameter##diam_radio_auto', self.diam_radio_auto):
                self.diam_radio_auto = True
            if imgui.radio_button('Manual ##diam_radio_manual', not self.diam_radio_auto):
                self.diam_radio_auto = False
            imgui.same_line()
            with imgui_utils.item_width(viz.font_size*5):
                _, self.diam_manual = imgui.input_int('##diam_manual', self.diam_manual)
            if self._wsi_config == 'auto':
                self.diam_manual = 10

            # Preview segmentation.
            if imgui_utils.button("Preview", width=viz.button_w) and not (self._wsi_config == 'auto'):
                self.segment_view()

        imgui.end_child()

        # --- View ------------------------------------------------------------
        imgui.same_line()
        imgui.begin_child('##view_child', width=child_width, height=child_height, border=True)
        imgui.text("View controls")
        imgui.separator()

        with imgui_utils.grayed_out(self.segmentation is None):

            # Show segmentation mask
            _mask_clicked, self.show_mask = imgui.checkbox('Mask', self.show_mask)
            if _mask_clicked and self.show_mask:
                self.show_gradXY = False

            # Show outlines
            imgui.same_line(viz.font_size*6)
            can_outline = (viz.viewer is not None and hasattr(viz.viewer, 'mpp') and viz.viewer.mpp < 1)
            with imgui_utils.grayed_out(not can_outline):
                if (imgui_utils.button("Outline", width=viz.button_w)
                   and can_outline
                   and not self.is_thread_running()):
                    self.show_mask = False
                    self.show_centroid = False
                    self._showing_preview_overlay = True
                    self._outline_toast = viz.create_toast(
                        title=f"Calculating outlines",
                        icon='info',
                        sticky=True,
                        spinner=True)
                    self._thread = Thread(target=self.render_outlines)
                    self._thread.start()

            # Show gradXY
            with imgui_utils.grayed_out(self.segmentation is None or self.segmentation.flows is None):
                _grad_clicked, self.show_gradXY = imgui.checkbox('gradXY', self.show_gradXY)
                if self.segmentation is None or self.segmentation.flows is None:
                    _grad_clicked = False
                    self.show_gradXY = False
                elif _grad_clicked and self.show_gradXY:
                    self.show_mask = False

            # Show cellprob
            imgui.same_line(viz.font_size*6)
            with imgui_utils.grayed_out(True):
                imgui_utils.button("Cellprob", width=viz.button_w)

            # Show centroid
            _centroid_clicked, self.show_centroid = imgui.checkbox('Centroid', self.show_centroid)
            if _centroid_clicked and not self.is_thread_running():
                if self.show_centroid and self.segmentation._centroids is None:
                    self._centroid_toast = viz.create_toast(
                        title=f"Calculating centroids",
                        icon='info',
                        sticky=True,
                        spinner=True)
                self._thread = Thread(target=self.calculate_centroids)
                self._thread.start()

            # Alpha control
            imgui.same_line(viz.font_size*6)
            if imgui_utils.button("Reset", width=viz.button_w):
                self.show_mask = True
                self.show_gradXY = False
                self.show_centroid = False
                self.overlay_background = True
                self.alpha = 1
                self.refresh_segmentation_view()

            _bg_change, self.overlay_background = imgui.checkbox("Black BG", self.overlay_background)
            if _bg_change:
                self.update_transparency()

            imgui.same_line(viz.font_size*6)
            with imgui_utils.item_width(viz.button_w - viz.spacing):
                _alpha_changed, self.alpha = imgui.slider_float('##alpha',
                                                                self.alpha,
                                                                min_value=0,
                                                                max_value=1,
                                                                format='Alpha %.2f')
                if _alpha_changed:
                    self.update_transparency()

            # Refresh segmentation --------------------------------------------
            self.viz._show_overlays = any((self.show_mask, self.show_gradXY, self.show_centroid, self._showing_preview_overlay))
            if _mask_clicked or _grad_clicked: # or others
                if self.segmentation is not None:
                    self.refresh_segmentation_view()
                    self.update_transparency()

        imgui.end_child()
