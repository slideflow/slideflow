import numpy as np
import slideflow as sf

from typing import Optional, Tuple
from .gaussian import Gaussian, blur_burden
from .strided_qc import _StridedQC_V2


class GaussianV2(_StridedQC_V2):

    def __init__(
        self,
        mpp: Optional[float] = None,
        sigma: int = 3,
        threshold: float = 0.02,
        *,
        overlap: Optional[int] = None,
        tile_px: int = 512,
        **kwargs
    ) -> None:
        """Optimized Gaussian blur filter for slide-level filtering (V2).

        This method is used to remove out-of-focus areas and pen marks,
        and is an optimization of the original
        :class:`slideflow.slide.qc.Gaussian` method.

        This method works by calculating Gaussian blur filtering for each tile
        in a slide, then stitching tile masks together to assemble the final
        slide-level mask. This approach is more computationally efficient
        and reduces memory consumption. Tiles are extracted with ``sigma * 2``
        overlap to eliminate stitching artifacts.

        Examples
            Apply Gaussian filtering to a slide.

                .. code-block:: python

                    import slideflow as sf
                    from slideflow.slide import qc

                    wsi = sf.WSI(...)
                    gaussian = qc.GaussianV2()
                    wsi.qc(gaussian)

        Args:
            mpp (float): Microns-per-pixel at which to perform filtering.
                Defaults to 4 times the tile extraction MPP (e.g. for a
                tile_px/tile_um combination at 10X effective magnification,
                where tile_px=tile_um, the default blur_mpp would be 4, or
                effective magnification 2.5x).
            sigma (int): Sigma (radius) for Gaussian filter. Defaults to 3.
            threshold (float): Gaussian threshold. Defaults to 0.02.

        Keyword args:
            tile_px (int): Size of tiles (height/width) on which to calculate
                Gaussian blur.
            verbose (bool): Show a progress bar during calculation.
            grayspace_fraction (float): Grayspace fraction to use during
                tile filtering. Defaults to 1 (disabled).
            overlap (int, optional): Specify the amount of overlap between
                tiles, in pixels, for reducing stitching artifacts. Defaults
                to ``sigma * 2``.
            filter_threads (int, optional): Number of threads to use for
                workers calculating Gaussian filter. If None, will disable
                multithreading for Gaussian filter workers. Defaults to 8.
            persistent_threads (bool): Keep thread pools alive. If False,
                will close the thread pools after this function has been called
                on a slide. Thread pools can be manually closed with
                ``.close()``. Defaults to True.
            pool (multiprocessing.Pool): Multiprocessing pool to use for tile
                extraction workers. If not provided, will create a thread pool
                with number of threads equal to the cpu count divided by 2.
                This is a separate multithreading pool than the Gaussian filter
                workers. Defaults to None.
            kwargs (Any): All remaining keyword arguments are passed to
                :meth:`slideflow.WSI.build_generator()`.

        """

        if overlap is None:
            overlap = sigma * 2

        super().__init__(tile_px=tile_px, tile_um=None, overlap=overlap, **kwargs)

        self.mpp = mpp
        self.tile_px = tile_px
        self.sigma = sigma
        self.threshold = threshold
        self._gaussian_qc = Gaussian(sigma=sigma, threshold=threshold)

    def __repr__(self):
        return "GaussianV2(mpp={!r}, sigma={!r}, threshold={!r})".format(
            self.mpp, self.sigma, self.threshold
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the QC algorithm to an image tile."""
        return self._gaussian_qc(image)

    def get_slide_and_mpp(self, wsi: "sf.WSI") -> Tuple["sf.WSI", float]:
        """Get a WSI object with the proper tile extraction size and stride."""
        if self.mpp is None:
            mpp = (wsi.tile_um/wsi.tile_px) * 4
        else:
            mpp = self.mpp
        tile_um = int(mpp * self.tile_px)
        stride_div = self.tile_px / (self.tile_px - self.overlap*2)
        qc_wsi = sf.WSI(
            wsi.path,
            tile_px=self.tile_px,
            tile_um=tile_um,
            stride_div=stride_div,
            verbose=False,
            use_edge_tiles=True,
            roi_method='ignore'
        )
        return qc_wsi, mpp

    def __call__(
        self,
        wsi: "sf.WSI",
        mask: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Apply Gaussian blur filtering to a slide."""
        blur_mask = super().__call__(wsi)

        # Assign blur burden value
        existing_qc_mask = mask
        if mask is None and isinstance(wsi, sf.WSI):
            existing_qc_mask = wsi.qc_mask
        if existing_qc_mask is not None and isinstance(wsi, sf.WSI):
            wsi.blur_burden = blur_burden(blur_mask, existing_qc_mask)
            sf.log.debug(f"Blur burden: {wsi.blur_burden}")

        return blur_mask

# -----------------------------------------------------------------------------
