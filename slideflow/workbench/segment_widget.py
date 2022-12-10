import slideflow as sf
import imgui
import numpy as np
import cellpose
import cellpose.models
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
        self.show_centroid          = False
        self.view_types             = ('Mask', 'Outline', 'Cellprob', 'gradXY')
        self.overlay                = None
        self.content_height         = 0

        self._selected_model_idx    = 1
        self._view_type_idx         = 0
        self._auto_tile_px          = 256
        self._auto_mpp              = 0.5
        self._wsi_config            = 'auto'
        self._segment_wsi_dim       = None
        self._segment_wsi_offset    = None
        self._segment_thread        = None
        self._segment_toast         = None
        self._centroid_thread       = None
        self._centroid_toast        = None

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

    def drag_and_drop_hook(self, path, ignore_errors=False):
        if 'masks' in np.load(path, mmap_mode='r').files:
            self.load(path, ignore_errors=ignore_errors)

    def load(self, path, ignore_errors=False):
        """Load a .npz with saved masks."""
        loaded = np.load(path)
        if 'masks' not in loaded and not ignore_errors:
            raise TypeError(f"Unable to load '{path}'; 'masks' index not found.")
        self.segmentation = Segmentation(loaded['masks'])
        self._segment_wsi_dim = loaded['wsi_dim']
        self._segment_wsi_offset = loaded['wsi_offset']
        if 'centroids' in loaded:
            self.segmentation._centroids = loaded['centroids']
        self.viz.create_toast(f"Loaded {self.segmentation.masks.max()} segmentations.", icon="success")
        self.refresh_segmentation_view()

    def segment_view(self):
        """Segment the current view."""
        print(f"Segmenting image with diameter {self.diameter} (shape={self.viz.viewer.view.shape})")
        masks, *_, diams = self.model.eval(
            self.viz.viewer.view,
            channels=[[0, 0]],
            diameter=self.diameter,
        )
        self.segmentation = Segmentation(masks)
        self._segment_wsi_dim = self.viz.viewer.wsi_window_size
        self._segment_wsi_offset = self.viz.viewer.origin
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
        full_extract = int(wsi.tile_um / wsi.mpp)
        wsi_stride = int(full_extract / wsi.stride_div)
        self._segment_wsi_dim = (wsi_stride * (wsi.grid.shape[0]),
                                 wsi_stride * (wsi.grid.shape[1]))
        self._segment_wsi_offset = (full_extract/2 - wsi_stride/2, full_extract/2 - wsi_stride/2)

        # Done; refresh view.
        self._segment_toast.done()
        refresh_toast = self.viz.create_toast(
                title=f"Rendering segmentations.",
                icon='info',
                sticky=True,
                spinner=True)
        self.refresh_segmentation_view()
        refresh_toast.done()
        self.viz.create_toast("Segmentation complete.", icon="success")

    def update_transparency(self):
        """Updates transparency of the overlay."""
        if self.overlay is None:
            return
        if self.view_types[self._view_type_idx] == 'Mask':
            alpha = np.full(self.overlay.shape[0:2], int(self.alpha * 255), dtype=np.uint8)
            self.overlay = np.dstack((self.overlay[:, :, 0:3], alpha))
        elif self.view_types[self._view_type_idx] == 'Outline':
            alpha = ((self.overlay[:, :, 0:3].max(axis=-1)).astype(bool).astype(np.uint8) * (255 * self.alpha)).astype(np.uint8)
            self.overlay = np.dstack((self.overlay[:, :, 0:3], alpha))

        # Ensure segmentation matches WSI view.
        self.viz.overlay = self.overlay
        self.viz._overlay_wsi_dim = self._segment_wsi_dim
        self.viz._overlay_offset_wsi_dim = self._segment_wsi_offset

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

    def refresh_segmentation_view(self):
        """Refresh the Workbench view of the active segmentation."""
        if self.segmentation is None:
            return
        if self.view_types[self._view_type_idx] == 'Mask':
            self.overlay = self.segmentation.mask_to_image(centroid=self.show_centroid)
        elif self.view_types[self._view_type_idx] == 'Outline':
            self.overlay = self.segmentation.outline_to_image(centroid=self.show_centroid)

        # Update transparency of the calculated overview.
        self.update_transparency()

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

        if (imgui_utils.button("Segment", width=viz.button_w * 1.2)
            and (self._segment_thread is None or not self._segment_thread.is_alive())):
            self._segment_toast = viz.create_toast(
                title=f"Segmenting whole-slide image.",
                icon='info',
                sticky=True,
                spinner=True)
            self._segment_thread = Thread(target=self.segment_slide)
            self._segment_thread.start()
        imgui.end_child()

        # --- Configuration ---------------------------------------------------
        imgui.same_line()
        imgui.begin_child('##config_child', width=child_width, height=child_height, border=True)

        with imgui_utils.grayed_out(self._wsi_config == 'auto'):

            ## Cell segmentation model.
            imgui.text("Model")
            imgui.same_line()
            with imgui_utils.item_width(viz.font_size * 8):
                _clicked, self._selected_model_idx = imgui.combo(
                    "##cellpose_model",
                    self._selected_model_idx,
                    self.supported_models)

            ## Cell Segmentation diameter.
            imgui.text("Diameter")
            if imgui.radio_button('##diam_radio_auto', self.diam_radio_auto):
                self.diam_radio_auto = True
            imgui.same_line()
            imgui.text("Auto")
            if imgui.radio_button('##diam_radio_manual', not self.diam_radio_auto):
                self.diam_radio_auto = False
            imgui.same_line()
            imgui.text("Manual")
            imgui.same_line()
            with imgui_utils.item_width(viz.font_size*6):
                _, self.diam_manual = imgui.input_int('##diam_manual', self.diam_manual)
            if self._wsi_config == 'auto':
                self.diam_manual = 10

            # Preview segmentation.
            if imgui_utils.button("Preview", width=viz.button_w * 1.2) and not (self._wsi_config == 'auto'):
                self.segment_view()

        imgui.end_child()

        # --- View & Export ---------------------------------------------------
        imgui.same_line()
        imgui.begin_child('##view_child', width=child_width, height=child_height, border=True)

        with imgui_utils.grayed_out(self.segmentation is None):
            for i, view_type in enumerate(self.view_types):
                with imgui_utils.grayed_out(i > 1):
                    if imgui.radio_button(f'{view_type}##view_radio{i}', self._view_type_idx == i):
                        self._view_type_idx = i
                        self.refresh_segmentation_view()
                if i == 0:
                    imgui.same_line(viz.font_size*6)
                    _, self.show = imgui.checkbox('Show', self.show)
                    self.viz._show_overlays = self.show
                if i == 1:
                    imgui.same_line(viz.font_size*6)
                    with imgui_utils.item_width(viz.button_w - viz.spacing):
                        _alpha_changed, self.alpha = imgui.slider_float('##alpha',
                                                                        self.alpha,
                                                                        min_value=0,
                                                                        max_value=1,
                                                                        format='Alpha %.2f')
                        if _alpha_changed:
                            self.update_transparency()
                if i == 2:
                    imgui.same_line(viz.font_size*6)
                    _centroid_clicked, self.show_centroid = imgui.checkbox('Centroid', self.show_centroid)
                    if _centroid_clicked and (self._centroid_thread is None or not self._centroid_thread.is_alive()):
                        if self.show_centroid and self.segmentation._centroids is None:
                            self._centroid_toast = viz.create_toast(
                                title=f"Calculating centroids.",
                                icon='info',
                                sticky=True,
                                spinner=True)
                        self._centroid_thread = Thread(target=self.calculate_centroids)
                        self._centroid_thread.start()
            if imgui_utils.button("Export", width=viz.button_w * 1.2) and self.segmentation is not None:
                filename = f'{viz.wsi.name}-masks.npz'
                np.savez(
                    filename,
                    masks=self.segmentation.masks,
                    centroids=self.segmentation.centroids,
                    wsi_dim=self._segment_wsi_dim,
                    wsi_offset=self._segment_wsi_offset)
                viz.create_toast(f"Exported masks and centroids to {filename}", icon='success')

        imgui.end_child()
