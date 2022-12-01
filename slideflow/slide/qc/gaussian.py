"""Gaussian filter QC algorithm."""

import numpy as np
import slideflow as sf
import skimage
from slideflow import errors
from typing import Union

class Gaussian:

    def __init__(
        self,
        mpp: float = 4,
        sigma: int = 3,
        threshold: float = 0.02
    ) -> None:
        """QC via Gaussian filtering.

        Args:
            mpp (float): Microns-per-pixel at which to perform filtering.
                Defaults to 4.
            sigma (int): Sigma (radius) for Gaussian filter. Defaults to 3.
            threshold (float): Gaussian threshold. Defaults to 0.02.
        """
        self.mpp = mpp
        self.sigma = sigma
        self.threshold = threshold

    def __repr__(self):
        return "Gaussian(mpp={!r}, sigma={!r}, threshold={!r})".format(
            self.mpp, self.sigma, self.threshold
        )

    def _thumb_from_slide(
        self,
        wsi: "sf.WSI"
    ) -> np.ndarray:
        """Get a thumbnail from the given slide.

        Args:
            wsi (sf.WSI): Whole-slide image.

        Returns:
            np.ndarray: RGB thumbnail of the whole-slide image.
        """
        thumb = wsi.thumb(mpp=self.mpp)
        if thumb is None:
            raise errors.QCError(
                f"Thumbnail error for slide {wsi.shortname}, QC failed"
            )
        thumb = np.array(thumb)
        if thumb.shape[-1] == 4:
            thumb = thumb[:, :, :3]
        return thumb

    def __call__(
        self,
        wsi: Union["sf.WSI", np.ndarray],
    ) -> np.ndarray:
        """Perform Gaussian filtering on the given slide or image.

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

        gray = skimage.color.rgb2gray(thumb)
        img_laplace = np.abs(skimage.filters.laplace(gray))
        gaussian = skimage.filters.gaussian(img_laplace, sigma=self.sigma)
        mask = gaussian <= self.threshold

        # Assign blur burden value
        existing_qc_mask = wsi.qc_mask
        if isinstance(wsi, sf.WSI) and existing_qc_mask is not None:
            mask = skimage.transform.resize(mask, existing_qc_mask.shape)
            mask = mask.astype(bool)
            blur = np.count_nonzero(
                np.logical_and(
                    mask,
                    np.logical_xor(mask, existing_qc_mask)
                )
            )
            wsi.blur_burden = blur / (mask.shape[0] * mask.shape[1])
            sf.log.debug(f"Blur burden: {wsi.blur_burden}")

        return mask