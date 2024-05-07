"""Otsu's thresholding QC algorithm."""

import cv2
import numpy as np
import slideflow as sf
import rasterio
import shapely.affinity as sa
from slideflow import errors
from typing import Union, Optional


def _apply_mask(image, mask):
    resized_mask = cv2.resize(
        (~mask).astype(np.uint8),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    return cv2.bitwise_or(image, image, mask=resized_mask)


def _get_level_for_otsu(wsi: "sf.WSI", min_size: int = 500) -> int:
    """Find the smallest downsample level of a minimum size."""
    smallest_dim = np.array([min(L['dimensions']) for L in wsi.levels])
    level_ids = np.array([L['level'] for L in wsi.levels])
    sorted_idx = np.argsort(smallest_dim)
    try:
        best_idx = np.where(smallest_dim[sorted_idx] > min_size)[0][0]
    except IndexError:
        # If the slide is smaller than the target minimum dimension,
        # use the full slide image
        best_idx = sorted_idx[-1]
    return level_ids[sorted_idx][best_idx]

# -----------------------------------------------------------------------------

class Otsu:

    def __init__(self, slide_level: Optional[int] = None):
        """Prepare Otsu's thresholding algorithm for filtering a slide.

        This method is used to detect areas of tissue and remove background.

        This QC method works by obtaining a thumbnail of a slide, and converting
        the image into the HSV color space. The HSV image undergoes a median blur
        using OpenCV with a kernel size of 7, and the image is thresholded
        using ``cv2.THRESH_OTSU``. This results in a binary mask, which
        is then applied to the slide for filtering.

        Original paper: https://ieeexplore.ieee.org/document/4310076

        .. warning::

            Otsu's thresholding may give unexpected results with slides
            that have many large pen marks, erroneously identifying pen marks
            as tissue and removing the actual tissue as background.
            This behavior can be circumvented by applying a Gaussian filter
            before Otsu's thresholding.

            .. code-block:: python

                    import slideflow as sf
                    from slideflow.slide import qc

                    wsi = sf.WSI(...)
                    gaussian = qc.GaussianV2()
                    otsu = qc.Otsu()
                    wsi.qc([gaussian, otsu])

        Examples
            Apply Otsu's thresholding to a slide.

                .. code-block:: python

                    import slideflow as sf
                    from slideflow.slide import qc

                    wsi = sf.WSI(...)
                    otsu = qc.Otsu()
                    wsi.qc(otsu)

        Args:
            level (int): Slide pyramid level at which to perform filtering.
                Defaults to second-lowest available level.
        """
        self.level = slide_level

    def __repr__(self):
        return "Otsu(slide_level={!r})".format(
            self.level
        )

    def _thumb_from_slide(
        self,
        wsi: "sf.WSI",
    ) -> np.ndarray:
        """Get a thumbnail from the given slide.

        Args:
            wsi (sf.WSI): Whole-slide image.

        Returns:
            np.ndarray: RGB thumbnail of the whole-slide image.
        """
        if self.level is None:
            # Otsu's thresholding can be done on the smallest downsample level,
            # with the smallest dimension being at least 500 pixels
            level = _get_level_for_otsu(wsi, min_size=500)
        else:
            level = self.level

        try:
            if wsi.slide.has_levels:
                sf.log.debug("Applying Otsu's thresholding at level={}".format(level))
                thumb = wsi.slide.read_level(level=level, to_numpy=True)
            else:
                sf.log.debug("Applying Otsu's thresholding at level=None")
                thumb = wsi.slide.read_level(to_numpy=True)
        except Exception as e:
            raise errors.QCError(
                f"Thumbnail error for slide {wsi.shortname}, QC failed: {e}"
            )
        if thumb.shape[-1] == 4:
            thumb = thumb[:, :, :3]

        # Only apply Otsu thresholding within ROI, if present
        # If ROI is the ROI_issues, invert it
        if wsi.has_rois():
            ofact = 1 / wsi.slide.level_downsamples[level]
            roi_mask = np.zeros((thumb.shape[0], thumb.shape[1]))

            scaled_polys = [
                sa.scale(roi.poly, xfact=ofact, yfact=ofact, origin=(0, 0))
                for roi in wsi.rois if roi.label not in wsi.artifact_rois
            ]

            scaled_issues_polys = [
                sa.scale(roi.invert_roi(wsi.dimensions).poly, xfact=ofact, yfact=ofact, origin=(0, 0))
                for roi in wsi.rois if roi.label in wsi.artifact_rois
            ]

            if len(scaled_polys) > 0:
                roi_mask = rasterio.features.rasterize(
                    scaled_polys,
                    out_shape=thumb.shape[:2]
                )

            if len(scaled_issues_polys) > 0:
                roi_mask_issues = rasterio.features.rasterize(
                    scaled_issues_polys,
                    out_shape=thumb.shape[:2]
                )
                if len(scaled_polys) > 0:
                    roi_mask = np.minimum(roi_mask_issues, roi_mask)
                else:
                    roi_mask = roi_mask_issues
                
            if wsi.roi_method == 'outside':
                roi_mask = ~roi_mask
            thumb = cv2.bitwise_or(
                thumb,
                thumb,
                mask=roi_mask.astype(np.uint8)
            )
        # Only apply Otsu thresholding within areas not already removed
        # with other QC methods.
        if wsi.has_non_roi_qc():
            thumb = _apply_mask(thumb, wsi.get_qc_mask(roi=False))
        return thumb

    def __call__(
        self,
        wsi: Union["sf.WSI", np.ndarray],
        mask: Optional[np.ndarray] = None,
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
            thumb = self._thumb_from_slide(wsi)
        else:
            thumb = wsi
        if mask is not None:
            thumb = _apply_mask(thumb, mask)
        hsv_img = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
        img_med = cv2.medianBlur(hsv_img[:, :, 1], 7)
        flags = cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV
        _, otsu_mask = cv2.threshold(img_med, 0, 255, flags)
        return otsu_mask.astype(bool)