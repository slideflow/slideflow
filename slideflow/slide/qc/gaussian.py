"""Gaussian filter QC algorithm."""

import numpy as np
import slideflow as sf
import skimage
from slideflow import errors
from typing import Union, Optional

class Gaussian:

    def __init__(
        self,
        mpp: Optional[float] = None,
        sigma: int = 3,
        threshold: float = 0.02
    ) -> None:
        """Prepare Gaussian filtering algorithm for filtering a slide.

        This method is used to remove out-of-focus areas and pen marks.

        This QC method works by obtaining a thumbnail of a slide, and converting
        the image into grayspace. A gaussian filter with a given sigma
        (default=3) is calculated using scikit-image. Areas with blur below
        the given threshold (default=0.02) are filtered out.

        Examples
            Apply Gaussian filtering to a slide.

                .. code-block:: python

                    import slideflow as sf
                    from slideflow.slide import qc

                    wsi = sf.WSI(...)
                    otsu = qc.Otsu()
                    wsi.qc(otsu)

        Args:
            mpp (float): Microns-per-pixel at which to perform filtering.
                Defaults to 4 times the tile extraction MPP (e.g. for a
                tile_px/tile_um combination at 10X effective magnification,
                where tile_px=tile_um, the default blur_mpp would be 4, or
                effective magnification 2.5x).
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
        if self.mpp is None:
            _mpp = (wsi.tile_um/wsi.tile_px)*4
            sf.log.info(f"Performing Gaussian blur filter at mpp={_mpp:.3f}")
        else:
            _mpp = self.mpp
        thumb = wsi.thumb(mpp=_mpp)
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
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Perform Gaussian filtering on the given slide or image.

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

        gray = skimage.color.rgb2gray(thumb)
        img_laplace = np.abs(skimage.filters.laplace(gray))
        gaussian = skimage.filters.gaussian(img_laplace, sigma=self.sigma)
        blur_mask = gaussian <= self.threshold

        # Assign blur burden value
        existing_qc_mask = mask
        if mask is None and isinstance(wsi, sf.WSI):
            existing_qc_mask = wsi.qc_mask
        if existing_qc_mask is not None:
            blur_mask = skimage.transform.resize(blur_mask, existing_qc_mask.shape)
            blur_mask = blur_mask.astype(bool)
            blur = np.count_nonzero(
                np.logical_and(
                    blur_mask,
                    np.logical_xor(blur_mask, existing_qc_mask)
                )
            )
            if isinstance(wsi, sf.WSI):
                wsi.blur_burden = blur / (blur_mask.shape[0] * blur_mask.shape[1])
                sf.log.debug(f"Blur burden: {wsi.blur_burden}")

        return blur_mask