"""QC algorithm for applying tissue segmentation (via U-Net-like models)."""

import slideflow as sf
import numpy as np

from typing import Union, Optional, List
from scipy.ndimage import label

# -----------------------------------------------------------------------------

class Segment:

    def __init__(self, model: str, class_idx: Optional[int] = None):
        """Prepare tissue segmentation model for filtering a slide.

        This method uses a U-Net like model for segmentation, as trained via
        :func:``slideflow.segment.train``.

        This method works by obtaining a thumbnail of a slide (at the microns-per-pixel
        magnification used to train the segmentation model), tiling the thumbnail
        into overlapping patches, and passing each patch through the segmentation
        model. The resulting segmentation masks are then stitched and blended together
        to form a single mask for the slide.

        For binary classification, predictions are thresholded at 0 (keep for preds > 0).
        For multiclass segmentation models, the prediction with the highest probability
        is used.

        Examples
            Apply tissue segmentation to a slide.

                .. code-block:: python

                    import slideflow as sf
                    from slideflow.slide import qc

                    wsi = sf.WSI(...)
                    segment = qc.Segment('/path/to/model.pth)
                    wsi.qc(segment)

        Args:
            model (str): Path to '.pth' model file, as generated via :func:``slideflow.segment.train``.
            class_idx (int, optional): If the model is a multiclass segmentation model,
                the class index to use for filtering, starting at 1 (0 is background and is ignored).
                Defaults to None.

        """
        import slideflow.segment

        self.model_path = model
        self.class_idx = class_idx
        self.model, self.cfg = sf.segment.load_model_and_config(model)

    def __repr__(self):
        return "Segment(model={!r})".format(
            self.model_path
        )

    def generate_rois(self, wsi: "sf.WSI", apply: bool = True) -> List[np.ndarray]:
        """Generate and apply ROIs to a slide using the loaded segmentation model.

        Args:
            wsi (sf.WSI): Slideflow WSI object.
            apply (bool): Whether to apply the generated ROIs to the slide.
                Defaults to True.

        Returns:
            List[np.ndarray]: List of ROIs, where each ROI is a numpy array of
                shape (n, 2), where n is the number of vertices in the ROI.
        """

        try:
            from cellpose.utils import outlines_list
        except ImportError:
            raise ImportError("Cellpose must be installed for generating ROIs from a "
                              "segmentation model. Cellpose can be installed via "
                              "'pip install cellpose'.")

        # Run tiled inference.
        preds = self(wsi, threshold=None)

        # Threshold the predictions and convert to ROIs.
        labels = None
        if preds.ndim == 3:
            pred_max = preds.argmax(axis=0)
            outlines = []
            labels = []
            for i in range(preds.shape[0]):
                if i == 0:
                    # Skip background class.
                    continue
                labeled, n_rois = label(pred_max == i)
                _outlined = outlines_list(labeled)
                outlines +=_outlined
                if self.cfg.labels and len(self.cfg.labels) >= i:
                    lbl = self.cfg.labels[i-1]
                else:
                    lbl = i
                labels += ([lbl] * len(_outlined))
            # Remove empty ROIs
            labels = [labels[l] for l in range(len(labels)) if outlines[l].shape[0]]
            outlines = [o for o in outlines if o.shape[0]]
        else:
            labeled, n_rois = label(preds > 0)
            outlines = outlines_list(labeled)
            outlines = [o for o in outlines if o.shape[0]]

        # Scale the outlines.
        outlines = [o * (self.cfg.mpp / wsi.mpp) for o in outlines]

        if labels is not None:
            assert len(outlines) == len(labels), "Number of outlines and labels must match."

        # Load ROIs.
        if apply:
            for o, outline in enumerate(outlines):
                wsi.load_roi_array(
                    outline,
                    process=False,
                    label=(None if labels is None else labels[o])
                )
            wsi.process_rois()

        return outlines

    def __call__(
        self,
        wsi: Union["sf.WSI", np.ndarray],
        threshold: Optional[float] = 0
    ) -> np.ndarray:
        """Perform tissue segmentation on the given slide or image.

        Args:
            wsi (sf.WSI, np.ndarray): Either a Slideflow WSI or a numpy array,
                with shape (h, w, c) and type np.uint8.
            threshold (float, optional): If None, return the raw
                predictions (binary segmentation models) or post-softmax predictions (multiclass models).
                If a float, threshold the predictions at the given
                value (less than) for binary models. Defaults to 0.

        Returns:
            np.ndarray: Boolean mask, where True = filtered out (less than threshold).
        """
        import torch

        if isinstance(wsi, sf.WSI):
            preds = self.model.run_slide_inference(wsi)
        else:
            preds = self.model.run_tiled_inference(wsi)

        if preds.ndim == 3:
            if threshold is None:
                preds = torch.from_numpy(preds).softmax(dim=0).numpy()
            elif self.class_idx is not None:
                preds = preds.argmax(axis=0) != self.class_idx
            else:
                preds = preds.argmax(axis=0) == 0
        elif threshold is not None:
            preds = preds < threshold
        return preds
