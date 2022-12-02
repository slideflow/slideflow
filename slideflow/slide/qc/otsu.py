"""Otsu's thresholding QC algorithm."""

import cv2
import numpy as np
import slideflow as sf
import rasterio
import shapely.affinity as sa
from slideflow import errors
from typing import Union, Optional


class Otsu:

    def __init__(self, slide_level: Optional[int] = None):
        """QC via Gaussian filtering.

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
            # Otsu's thresholding can be done on the lowest downsample level
            level = wsi.slide.level_count - 1
        else:
            level = self.level

        try:
            if wsi.slide.has_levels:
                thumb = wsi.slide.read_level(level=level, to_numpy=True)
            else:
                thumb = wsi.slide.read_level(to_numpy=True)
        except Exception as e:
            raise errors.QCError(
                f"Thumbnail error for slide {wsi.shortname}, QC failed: {e}"
            )
        if thumb.shape[-1] == 4:
            thumb = thumb[:, :, :3]

        # Only apply Otsu thresholding within ROI, if present
        if len(wsi.annPolys):
            ofact = wsi.roi_scale / wsi.slide.level_downsamples[level]
            roi_mask = np.zeros((thumb.shape[0], thumb.shape[1]))
            scaled_polys = [
                sa.scale(poly, xfact=ofact, yfact=ofact, origin=(0, 0))
                for poly in wsi.annPolys
            ]
            roi_mask = rasterio.features.rasterize(
                scaled_polys,
                out_shape=thumb.shape[:2]
            )
            thumb = cv2.bitwise_or(
                thumb,
                thumb,
                mask=roi_mask.astype(np.uint8)
            )
        return thumb

    def __call__(
        self,
        wsi: Union["sf.WSI", np.ndarray],
    ) -> np.ndarray:
        """Perform Otsu's thresholding on the given slide or image.

        Args:
            slide (sf.WSI, np.ndarray): Either a Slideflow WSI or a numpy array,
                with shape (h, w, c) and type np.uint8.

        Returns:
            np.ndarray: QC boolean mask, where True = filtered out.
        """
        if isinstance(wsi, sf.WSI):
            thumb = self._thumb_from_slide(wsi)
        else:
            thumb = wsi
        hsv_img = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
        img_med = cv2.medianBlur(hsv_img[:, :, 1], 7)
        flags = cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV
        _, mask = cv2.threshold(img_med, 0, 255, flags)
        return mask.astype(bool)