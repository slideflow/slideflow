import imgui
import numpy as np
import cellpose
import cellpose.models
from typing import Tuple
from PIL import Image, ImageDraw
from slideflow.slide.utils import draw_roi
from .gui_utils import imgui_utils


def polygon_area(xs, ys):
    """https://en.wikipedia.org/wiki/Centroid#Of_a_polygon"""
    # https://stackoverflow.com/a/30408825/7128154
    return 0.5 * (np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))


def polygon_centroid(xs, ys):
    """https://en.wikipedia.org/wiki/Centroid#Of_a_polygon"""
    xy = np.array([xs, ys])
    c = np.dot(xy + np.roll(xy, 1, axis=1),
               xs * np.roll(ys, 1) - np.roll(xs, 1) * ys
               ) / (6 * polygon_area(xs, ys))
    return c


class SegmentWidget:
    def __init__(self, viz):
        self.viz                    = viz
        self.header                 = "Cellpose"
        self.diam_radio_auto        = True
        self.diam_radio_suggested   = False
        self.diam_radio_manual      = False
        self.diam_manual            = 10
        self._selected_model_idx    = 1

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
            self.models[model_str] = cellpose.models.Cellpose(gpu=True, model_type=model_str)
        return self.models[model_str]

    def masks_from_image(
        self,
        img: np.ndarray,
        remove_edge: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate masks from an image using Cellpose.

        Args:
            img (np.ndarray): image to segment.
            remove_edge (bool): Remove masks which touch the edge of the image.
                Used for segmentation. Defaults to False.

        Returns:
            masks, outlines
        """
        if verbose:
            print("Segmenting image with shape=", img.shape)
        masks, flows, styles, diams = self.model.eval(
            img,
            channels=[[0, 0]],
            **kwargs
        )
        outlines = cellpose.utils.outlines_list(masks)
        if verbose:
            print(f"Segmentation of current view complete ({masks.max()} total masks).")
        return masks, outlines

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        # Set up settings interface.

        ## Cell segmentation model.
        imgui.text("Model")
        imgui.same_line()
        with imgui_utils.item_width(viz.font_size * 8):
            _clicked, self._selected_model_idx = imgui.combo("##cellpose_model", self._selected_model_idx, self.supported_models)

        ## Cell Segmentation diameter.
        imgui.text("Diameter")
        if imgui.radio_button('##diam_radio_auto', self.diam_radio_auto):
            self.diam_radio_auto = True
            self.diam_radio_suggested = False
            self.diam_radio_manual = False
        imgui.same_line()
        imgui.text("Auto")

        if imgui.radio_button('##diam_radio_suggested', self.diam_radio_suggested):
            self.diam_radio_suggested = True
            self.diam_radio_auto = False
            self.diam_radio_manual = False
        imgui.same_line()
        imgui.text("Suggested")

        if imgui.radio_button('##diam_radio_manual', self.diam_radio_manual):
            self.diam_radio_manual = True
            self.diam_radio_auto = False
            self.diam_radio_suggested = False
        imgui.same_line()
        imgui.text("Manual")
        imgui.same_line()
        diam_changed, self.diam_manual = imgui_utils.input_text('##diam_manual', str(self.diam_manual), 8, width=viz.font_size*4, flags=0)

        if diam_changed:
            try:
                self.diam_manual = int(self.diam_manual)
            except:
                print("Invalid diameter: ", self.diam_manual)
                self.diam_manual = 10


        if imgui_utils.button("Segment", width=viz.button_w):

            # Calculate masks.
            diam = None if self.diam_radio_auto else int(self.diam_manual)
            masks, outlines = self.masks_from_image(viz.viewer.view, diameter=diam)

            # Convert masks to ROI drawing.
            outlined_img = draw_roi(np.zeros_like(viz.viewer.view), outlines)

            # Calculate and display centroid points.
            centroid = True
            if centroid:
                centroids = [polygon_centroid(o[:, 0], o[:, 1]) for o in outlines]
                centroid_img = Image.fromarray(outlined_img)
                draw = ImageDraw.Draw(centroid_img)
                for c in centroids:
                    draw.point((int(c[0]), int(c[1])), fill='green')
                outlined_img = np.asarray(centroid_img)

            # Set the overlay in the current viewer.
            alpha = (outlined_img.max(axis=-1)).astype(bool).astype(np.uint8) * 255
            overlay = np.dstack((outlined_img, alpha))
            viz.set_overlay(overlay)

