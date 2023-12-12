"""Tissue segmentation QC algorithm (via U-Net-like models)."""

import slideflow as sf
import numpy as np

from typing import Union, Optional, List
from scipy.ndimage import label

# -----------------------------------------------------------------------------

class Segment:

    def __init__(self, model: str):
        """Prepare tissue segmentation model for filtering a slide.

        This method uses a U-Net like model for segmentation, as trained via
        :func:``slideflow.segment.train``.

        This method works by obtaining a thumbnail of a slide (at the microns-per-pixel
        magnification used to train the segmentation model), tiling the thumbnail
        into overlapping patches, and passing each patch through the segmentation
        model. The resulting segmentation masks are then stitched and blended together
        to form a single mask for the slide. Predictions are thresholded at 0 (keep for preds > 0)

        Examples
            Apply Otsu's thresholding to a slide.

                .. code-block:: python

                    import slideflow as sf
                    from slideflow.slide import qc

                    wsi = sf.WSI(...)
                    segment = qc.Segment('/path/to/model.pth)
                    wsi.qc(segment)

        Args:
            model (str): Path to '.pth' model file, as generated via :func:``slideflow.segment.train``.
        """
        import slideflow.segment

        self.model_path = model
        self.model, self.cfg = sf.segment.load_model_and_config(model)

    def __repr__(self):
        return "Segment(model={!r})".format(
            self.model_path
        )
    
    def generate_rois(self, wsi: "sf.WSI", apply: bool = True) -> List[np.ndarray]:
        """Generate and apply ROIs to a slide using the loaded segmentation model.

        Args:
            wsi (sf.WSI): Slideflow WSI object.
        """

        try:
            from cellpose.utils import outlines_list
        except ImportError:
            raise ImportError("Cellpose must be installed for generating ROIs from a "
                              "segmentation model. Cellpose can be installed via "
                              "'pip install cellpose'.")

        # Run tiled inference.
        preds = self(wsi, threshold=None)
        
        # Threshold the predictions.
        labeled, n_rois = label(preds > 0)

        # Convert to ROIs.
        outlines = outlines_list(labeled)
        outlines = [o for o in outlines if o.shape[0]]

        # Scale the outlines.
        outlines = [o * (self.cfg.mpp / wsi.mpp) for o in outlines]

        # Load ROIs.
        if apply:
            for outline in outlines:
                wsi.load_roi_array(outline, process=False)
            wsi.process_rois()

        return outlines

    def __call__(
        self, 
        wsi: Union["sf.WSI", np.ndarray],
        threshold: Optional[Union[bool, float]] = 0
    ) -> np.ndarray:
        """Perform Otsu's thresholding on the given slide or image.

        Args:
            slide (sf.WSI, np.ndarray): Either a Slideflow WSI or a numpy array,
                with shape (h, w, c) and type np.uint8.
            mask (np.ndarray): Restrict Otsu's threshold to the area of the
                image indicated by this boolean mask. Defaults to None.

        Returns:
            np.ndarray: QC boolean mask, where True = filtered out.
        """
        if isinstance(wsi, sf.WSI):
            preds = self.model.run_slide_inference(wsi)
        else:
            preds = self.model.run_tiled_inference(wsi)
        if threshold is not None:
            return preds < threshold
        else:
            return preds