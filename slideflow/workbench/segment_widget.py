import zarr
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
from slideflow.slide.seg import segment_slide, Segmentation
from .gui_utils import imgui_utils


class SegmentWidget:
    def __init__(self, viz):
        self.viz                    = viz
        self.header                 = "Cellpose"
        self.alpha                  = 1
        self.downscale              = 1
        self.show_advanced          = False
        self.spawn_workers          = True
        self.tile                   = True
        self.show                   = True
        self.diam_radio_auto        = False
        self.otsu                   = False
        self.segmentation           = None
        self.diameter_microns       = 10
        self.overlay                = None
        self.content_height         = 0
        self.show_mask              = True
        self.show_gradXY            = False
        self.show_centroid          = False
        self.overlay_background     = True
        self.tile_px                = 512
        self.save_flow              = False

        self._showing_preview_overlay = False
        self._selected_model_idx    = 1
        self._thread                = None
        self._segment_toast         = None
        self._centroid_toast        = None
        self._load_toast            = None
        self._outline_toast         = None

        # Load default model
        self.supported_models = [
            'cyto', 'cyto2', 'nuclei', 'tissuenet', 'TN1', 'TN2', 'TN3',
            'livecell', 'LC1', 'LC2', 'LC3', 'LC4'
        ]
        self.models = {m: None for m in self.supported_models}

    @property
    def diam_mean(self):
        return (17 if self.supported_models[self._selected_model_idx] == 'nuclei'
                   else 30)

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
        return None if self.diam_radio_auto else self.diam_mean

    @property
    def tile_um(self):
        return int(self.tile_px * self.mpp)

    @property
    def mpp(self):
        return self.diameter_microns / self.diam_mean

    def formatted_slide_levels(self):
        if self.viz.wsi:
            return [
                '{:.0f}x (mpp={:.2f})'.format(10/mpp, mpp)
                for mpp in self.viz.wsi.level_mpp
            ]
        else:
            return ['NA']

    def is_thread_running(self):
        return self._thread is not None and self._thread.is_alive()

    def _load_segmentation(self, path, ignore_errors=False):
        self.segmentation = Segmentation.load(path)
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
        if (sf.util.path_to_ext(path).lower() == 'zip'):
            try:
                z = zarr.open(path)
            except Exception as e:
                return
            else:
                if 'masks' in z:
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
        self.viz._overlay_wsi_dim = self.segmentation.wsi_dim
        self.viz._overlay_offset_wsi_dim = self.segmentation.wsi_offset

    def render_outlines(self, color='red'):
        """Render outlines of the segmentations currently in view."""
        in_view, _, view_offset = self.viz.viewer.in_view(
            self.segmentation.masks,
            dim=self.segmentation.wsi_dim,
            offset=self.segmentation.wsi_offset,
            resize=False)

        # Calculate and draw the outlines
        outlines = [o for o in sf.slide.seg.outlines_list(in_view) if o.shape[0] >= 3]
        empty = np.zeros((in_view.shape[0], in_view.shape[1], 3), dtype=np.uint8)
        outline_img = draw_roi(empty, outlines, color=color, linewidth=2)

        self.viz.overlay = outline_img
        self.viz._overlay_wsi_dim = (
            (in_view.shape[1] / self.segmentation.masks.shape[1]) * self.segmentation.wsi_dim[0],
            (in_view.shape[0] / self.segmentation.masks.shape[0]) * self.segmentation.wsi_dim[1],
        )
        self.viz._overlay_offset_wsi_dim = (
            self.segmentation.wsi_offset[0] + view_offset[0],
            self.segmentation.wsi_offset[1] + view_offset[1]
        )
        self._outline_toast.done()
        self.viz.create_toast("Outlining complete.", icon="success")
        self.update_transparency()

    def _segment_view(self):
        """Segment the current view."""
        v = self.viz.viewer
        view_params = v.view_params
        view_microns = (
            view_params.window_size[0] * v.wsi.mpp,
            view_params.window_size[1] * v.wsi.mpp,
        )
        view_params.target_size = (
            int(view_microns[0] / self.mpp),
            int(view_microns[1] / self.mpp)
        )
        view_img = v._read_from_pyramid(**view_params)
        print(f"Segmenting image view with diameter {self.diameter} (um={view_microns} mpp={self.mpp:.3f} shape={view_img.shape})")
        masks, flows, styles, diams = self.model.eval(
            view_img,
            channels=[[0, 0]],
            diameter=self.diameter,
        )
        self.segmentation = Segmentation(
            slide=None,
            masks=masks,
            flows=flows[0],
            wsi_dim=v.wsi_window_size,
            wsi_offset=v.origin)
        self.diameter_microns = int(diams * self.mpp)
        print(f"Segmentation of view complete (diameter={diams:.2f}, {masks.max()} total masks), shape={masks.shape}.")
        self.refresh_segmentation_view()
        if self._segment_toast is not None:
            self._segment_toast.done()

    def segment_slide(self, in_view=False):
        """Generate whole-slide segmentation."""

        # Segment single preview image if auto-diameter (diameter=None)
        if in_view:
            return self._segment_view()

        # Otherwise, segment using whole-slide image interface
        wsi = sf.WSI(self.viz.wsi.path, tile_px=self.tile_px, tile_um=self.tile_um, verbose=False)
        if self.otsu:
            wsi.qc('otsu', filter_threshold=1)
        print(f"Segmenting WSI (shape={wsi.dimensions}, tile_px={self.tile_px}, tile_um={self.tile_um}, diameter={self.diameter})")

        # Alternative method of rendering in-view preview
        if in_view:
            print("Segmenting partial WSI using current view window")
            (xi, xi_e), (yi, yi_e) = self.viz.viewer.grid_in_view(wsi)
            z = np.zeros_like(wsi.grid, dtype=bool)
            z[xi:xi_e, yi:yi_e] = True
            wsi.grid = wsi.grid * z

        # Perform segmentation.
        self.segmentation = segment_slide(
            wsi,
            'cyto2',
            save_flow=self.save_flow,
            downscale=(None if self.downscale == 1 else self.downscale),
            tile=self.tile,
            spawn_workers=self.spawn_workers)
        print('Mask shape:', self.segmentation.masks.shape)

        # Done; refresh view.
        if self._segment_toast is not None:
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
        has_viewer = (viz.viewer is not None)

        if not show:
            self.content_height = 0
            return

        # Set up settings interface.
        child_width = imgui.get_content_region_max()[0] / 3 - viz.spacing
        child_height = imgui.get_text_line_height_with_spacing() * 6 + viz.spacing * 2
        self.content_height = child_height + viz.spacing

        # --- Configuration ---------------------------------------------------
        with imgui_utils.grayed_out(not has_viewer):
            imgui.begin_child('##config_child', width=child_width, height=child_height, border=True)
            imgui.text("Model & cell diameter")
            imgui.separator()

            ## Cell segmentation model.
            with imgui_utils.item_width(child_width - viz.spacing * 2):
                _clicked, self._selected_model_idx = imgui.combo(
                    "##cellpose_model",
                    self._selected_model_idx,
                    self.supported_models)

            ## Cell Segmentation diameter.
            if imgui.radio_button('Auto-detect diameter##diam_radio_auto', self.diam_radio_auto):
                self.diam_radio_auto = True
            if imgui.radio_button('Manual: ##diam_radio_manual', not self.diam_radio_auto):
                self.diam_radio_auto = False
            imgui.same_line()
            with imgui_utils.item_width(viz.font_size*2):
                _, self.diameter_microns = imgui.input_int('##diam_manual', self.diameter_microns, step=0)
            imgui.same_line()
            imgui.text('um')

            # Preview segmentation.
            if imgui_utils.button("Preview", width=viz.button_w):
                self._segment_toast = viz.create_toast(
                    title=f"Segmenting current view",
                    icon='info',
                    sticky=True,
                    spinner=True)
                self._thread = Thread(target=self.segment_slide, args=(True,))
                self._thread.start()

            imgui.end_child()

        # --- Whole-slide segmentation ----------------------------------------
        imgui.same_line()
        with imgui_utils.grayed_out(not has_viewer or self.diam_radio_auto):
            imgui.begin_child('##segment_child', width=child_width, height=child_height, border=True)
            imgui.text("Whole-slide segmentation")
            imgui.separator()

            _, self.otsu = imgui.checkbox('Otsu threshold', self.otsu)
            _, self.save_flow = imgui.checkbox('Save flows', self.save_flow)
            _, self.show_advanced = imgui.checkbox('Show advanced', self.show_advanced)

            if self.show_advanced:
                with imgui_utils.item_width(viz.font_size*3):
                    _, self.tile_px = imgui.input_int('Window', self.tile_px, step=0)
                imgui.same_line()
                _, self.tile = imgui.checkbox('Tile', self.tile)
                with imgui_utils.item_width(viz.font_size*2):
                    _, self.downscale = imgui.input_int('Downscale factor', self.downscale, step=0)
                _, self.spawn_workers = imgui.checkbox('Enable spawn workers', self.spawn_workers)

            # WSI segmentation.
            if (imgui_utils.button("Segment", width=viz.button_w)
               and not self.is_thread_running()
               and has_viewer
               and not self.diam_radio_auto):
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
                    filename = f'{viz.wsi.name}-masks.zip'
                    self.segmentation.save(filename, centroids=True)
                    viz.create_toast(f"Exported masks and centroids to {filename}", icon='success')
            imgui.end_child()

        # --- View ------------------------------------------------------------
        imgui.same_line()
        with imgui_utils.grayed_out(self.segmentation is None):
            imgui.begin_child('##view_child', width=child_width, height=child_height, border=True)
            imgui.text("View controls")
            imgui.separator()

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
