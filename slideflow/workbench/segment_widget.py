import slideflow as sf
import imgui
import numpy as np
import cellpose
import cellpose.models
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
        self.config_auto            = True
        self.diam_radio_auto        = True
        self.use_gpu                = True
        self.diam_radio_manual      = False
        self.segmentation           = None
        self.diam_manual            = 10
        self.calc_centroid          = False
        self.view_types             = ('Mask', 'Outline', 'Cellprob', 'gradXY')
        self.overlay                = None

        self._selected_model_idx    = 1
        self._view_type_idx         = 0
        self._segment_wsi_dim       = None
        self._segment_wsi_offset    = None

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
        self.diam_manual = diams
        print(f"Segmentation of current view complete (diameter={diams:.2f}), {masks.max()} total masks).")

    def segment_slide(self):
        """Generate whole-slide segmentation.

        Args:
            auto (bool): Use recommended, automatic segmentation configuration.
                If false, will use user-specified settings.
        """
        if self.config_auto:
            # Segment using recommend configuration.
            wsi = sf.WSI(self.viz.wsi.path, tile_px=256, tile_um=140)
            wsi.qc('blur')
            diameter = 10
            print(f"Segmenting WSI (shape={wsi.dimensions}, tile_px=256, tile_um=140, diameter=10)")
        else:
            # Segment using user-defined configuration.
            tile_px = 256
            tile_um = int(self.viz.viewer.mpp * 256)
            wsi = sf.WSI(self.viz.wsi.path, tile_px=tile_px, tile_um=tile_um)
            wsi.qc('blur')
            print(f"Segmenting WSI (shape={wsi.dimensions}, tile_px={tile_px}, tile_um={tile_um}, diameter={self.diameter})")

        # Perform segmentation.
        self.segmentation, _ = segment_slide(wsi, 'cyto2', diameter=self.diameter)
        full_extract = int(wsi.tile_um / wsi.mpp)
        wsi_stride = int(full_extract / wsi.stride_div)
        self._segment_wsi_dim = (wsi_stride * (wsi.grid.shape[0]),
                                    wsi_stride * (wsi.grid.shape[1]))
        self._segment_wsi_offset = (full_extract/2 - wsi_stride/2, full_extract/2 - wsi_stride/2)

    def update_transparency(self):
        """Updates transparency of the overlay."""
        if self.overlay is None:
            return
        if self.view_types[self._view_type_idx] == 'Mask':
            alpha = np.full(self.overlay.shape[0:2], int(self.alpha * 255), dtype=np.uint8)
            self.overlay = np.dstack((self.overlay[:, :, 0:3], alpha))
        elif self.view_types[self._view_type_idx] == 'Outline':
            alpha = (self.overlay[:, :, 0:3].max(axis=-1)).astype(bool).astype(np.uint8) * (255 * self.alpha)
            self.overlay = np.dstack((self.overlay[:, :, 0:3], alpha))

        # Ensure segmentation matches WSI view.
        self.viz.overlay = self.overlay
        self.viz._overlay_wsi_dim = self._segment_wsi_dim
        self.viz._overlay_offset_wsi_dim = self._segment_wsi_offset


    def refresh_segmentation_view(self):
        """Refresh the Workbench view of the active segmentation."""

        if self.view_types[self._view_type_idx] == 'Mask':
            self.overlay = self.segmentation.mask_to_image()
        elif self.view_types[self._view_type_idx] == 'Outline':
            self.overlay = self.segmentation.outline_to_image(centroid=True)

        # Update transparency of the calculated overview.
        self.update_transparency()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        # Set up settings interface.
        child_width = imgui.get_content_region_max()[0] / 3 - viz.spacing
        child_height = imgui.get_text_line_height_with_spacing() * 6 + viz.spacing * 2

        # --- Segmentation ----------------------------------------------------
        imgui.begin_child('##segment_child', width=child_width, height=child_height, border=True)
        imgui.text("Configuration")
        if imgui.radio_button('Auto##config_radio_auto', self.config_auto):
            self.config_auto = True
        if imgui.radio_button('Manual##config_radio_manual', not self.config_auto):
            self.config_auto = False
        _, self.use_gpu = imgui.checkbox('Use GPU', self.use_gpu)
        if imgui_utils.button("Segment", width=viz.button_w * 1.2):
            self.segment_slide()
            self.refresh_segmentation_view()
        imgui.end_child()

        # --- Configuration ---------------------------------------------------
        imgui.same_line()
        imgui.begin_child('##config_child', width=child_width, height=child_height, border=True)

        with imgui_utils.grayed_out(self.config_auto):

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
                _, self.diam_manual = imgui.slider_float(
                    '##diam_manual',
                    self.diam_manual,
                    min_value=1,
                    max_value=50,
                    format='Diameter %.1f')

            # Preview segmentation.
            if imgui_utils.button("Preview", width=viz.button_w * 1.2) and not self.config_auto:
                self.segment_view()
                self.refresh_segmentation_view()

        imgui.end_child()

        # --- View & Export ---------------------------------------------------
        imgui.same_line()
        imgui.begin_child('##view_child', width=child_width, height=child_height, border=True)

        with imgui_utils.grayed_out(self.segmentation is None):
            for i, view_type in enumerate(self.view_types):
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
                    _, self.calc_centroid = imgui.checkbox('Centroid', self.calc_centroid)
            if imgui_utils.button("Export", width=viz.button_w * 1.2) and self.segmentation is not None:
                filename = f'{viz.wsi.name}-masks.npz'
                np.savez(filename, masks=self.segmentation.masks, centroids=self.segmentation.centroids)
                viz.create_toast(f"Exported masks and centroids to {filename}", icon='success')

        imgui.end_child()
