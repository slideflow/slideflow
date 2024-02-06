"""QC algorithm for applying tissue segmentation (via U-Net-like models)."""

import slideflow as sf
import numpy as np

from typing import Union, Optional, List
from scipy.ndimage import label

from .strided_dl import StridedDL_V2

# -----------------------------------------------------------------------------

class Segment:

    def __init__(
        self,
        model: str,
        class_idx: Optional[int] = None,
        threshold_direction: str = 'less'
    ):
        """Prepare tissue segmentation model for filtering a slide.

        This method uses a U-Net like model for segmentation, as trained via
        :func:``slideflow.segment.train``.

        This method works by obtaining a thumbnail of a slide (at the microns-per-pixel
        magnification used to train the segmentation model), tiling the thumbnail
        into overlapping patches, and passing each patch through the segmentation
        model. The resulting segmentation masks are then stitched and blended together
        to form a single mask for the slide. As with other QC methods, the returned
        mask is a boolean mask, where True indicates areas that should be filtered
        out (e.g. background).

        This method supports binary, multiclass, and multilabel segmentation models.

        - For binary classification, predictions are thresholded at 0 (remove areas with preds < 0).

        - For multiclass segmentation models, areas predicted to be class 0 (background)
          will be removed. This class index can be changed via the ``class_idx`` argument.

        - For multilabel segmentation models, each class channel is thresholded at 0
          (remove areas with preds < 0). The resulting masks are then combined via
          logical OR. Alternatively, the ``class_idx`` argument can be used to select
          a single class channel to use for filtering, where preds < 0 for this channel
          will be removed.

        For binary and multilabel models, the thresholding direction can be changed
        via the ``threshold_direction`` argument, which can be one of 'less' (default)
        or 'greater'.


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
            threshold_direction (str, optional): Thresholding direction for binary and multilabel
                segmentation models. Can be one of 'less' (default) or 'greater'.
                Defaults to 'less'.

        """
        import slideflow.segment
        from slideflow.model.torch_utils import get_device

        if threshold_direction not in ['less', 'greater']:
            raise ValueError("Invalid threshold_direction: {}. Expected one of: less, greater".format(threshold_direction))

        self.model_path = model
        self.class_idx = class_idx
        self.model, self.cfg = sf.segment.load_model_and_config(model)
        self.model.to(get_device())
        self.model.eval()
        self.threshold_direction = threshold_direction

    def __repr__(self):
        return "Segment(model={!r})".format(
            self.model_path
        )

    @classmethod
    def from_model(
        cls,
        model: "sf.segment.SegmentModel",
        config: "sf.segment.SegmentConfig",
        class_idx: Optional[int] = None,
        threshold_direction: str = 'less'
    ):
        """Create a Segment object from a SegmentModel and SegmentConfig."""
        obj = cls.__new__(cls)
        obj.model = model
        obj.model_path = None
        obj.cfg = config
        obj.class_idx = class_idx
        obj.threshold_direction = threshold_direction
        return obj

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

        if self.cfg.mode == 'binary':
            labeled, n_rois = label(preds > 0)
            outlines = outlines_list(labeled)
            outlines = [o for o in outlines if o.shape[0]]

        elif self.cfg.mode == 'multiclass':
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

        elif self.cfg.mode == 'multilabel':
            outlines = []
            labels = []
            for i in range(preds.shape[0]):
                labeled, n_rois = label(preds[i] > 0)
                _outlined = outlines_list(labeled)
                outlines += _outlined
                if self.cfg.labels and len(self.cfg.labels) > i:
                    lbl = self.cfg.labels[i]
                else:
                    lbl = i
                labels += ([lbl] * len(_outlined))
            # Remove empty ROIs
            labels = [labels[l] for l in range(len(labels)) if outlines[l].shape[0]]
            outlines = [o for o in outlines if o.shape[0]]

        else:
            raise ValueError("Invalid loss mode: {}. Expected one of: binary, "
                             "multiclass, multilabel".format(self.cfg.mode))

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

    def get_slide_preds(self, wsi: "sf.WSI") -> np.ndarray:
        """Get the predictions for a slide using the loaded segmentation model."""
        return self.model.run_slide_inference(wsi)

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
                predictions (binary or multilabel models) or post-softmax predictions
                (multiclass models). If a float, threshold the predictions at the given
                value (less than) for binary models. Defaults to 0.

        Returns:
            np.ndarray: Boolean mask, where True = filtered out (less than threshold).

        """
        if isinstance(wsi, sf.WSI):
            preds = self.get_slide_preds(wsi)
        else:
            preds = self.model.run_tiled_inference(wsi)

        if threshold is None or threshold is False:
            return preds

        if self.cfg.mode == 'binary':
            if self.threshold_direction == 'less':
                return preds < threshold
            else:
                return preds > threshold
        elif self.cfg.mode == 'multiclass':
            if self.class_idx is not None:
                return preds.argmax(axis=0) != self.class_idx
            else:
                return preds.argmax(axis=0) == 0
        elif self.cfg.mode == 'multilabel':
            if self.class_idx is not None and self.threshold_direction == 'less':
                return preds[self.class_idx] < threshold
            elif self.class_idx is not None and self.threshold_direction == 'greater':
                return preds[self.class_idx] > threshold
            elif self.threshold_direction == 'less':
                return np.all(preds < threshold, axis=0)
            else:
                return np.all(preds > threshold, axis=0)


# -----------------------------------------------------------------------------

class StridedSegment(Segment):

    def __init__(self, *args, overlap: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self._strided_qc = StridedDL_V2(
            tile_px=self.cfg.size,
            tile_um=int(np.round(self.cfg.size*self.cfg.mpp)),
            overlap=overlap,
            filter_threads=0,
            out_classes=(self.cfg.out_classes if self.cfg.mode != 'binary' else 0)
        )
        self._strided_qc.apply = self.apply

    @classmethod
    def from_model(
        cls,
        model: "sf.segment.SegmentModel",
        config: "sf.segment.SegmentConfig",
        class_idx: Optional[int] = None,
        threshold_direction: str = 'less',
        overlap: int = 128
    ):
        """Create a StridedSegment object from a SegmentModel and SegmentConfig."""
        obj = cls.__new__(cls)
        obj.model = model
        obj.model_path = None
        obj.cfg = config
        obj.class_idx = class_idx
        obj.threshold_direction = threshold_direction
        obj._strided_qc = StridedDL_V2(
            tile_px=config.size,
            tile_um=int(np.round(config.size*config.mpp)),
            overlap=overlap,
            filter_threads=0,
            out_classes=(config.out_classes if config.mode != 'binary' else 0)
        )
        obj._strided_qc.apply = obj.apply
        return obj

    def apply(self, image: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            tensor = torch.from_numpy(image).unsqueeze(0).to(self.model.device)
            tensor = sf.io.torch.as_cwh(tensor)
            return self.model.forward(tensor).squeeze(dim=0).cpu().numpy()

    def get_slide_preds(self, wsi: "sf.WSI") -> np.ndarray:
        """Get the predictions for a slide using the loaded segmentation model."""
        return self._strided_qc(wsi)