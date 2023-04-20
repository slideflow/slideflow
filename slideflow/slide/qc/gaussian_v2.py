import os
import numpy as np
import slideflow as sf
import multiprocessing as mp

from functools import partial
from typing import Optional, Tuple
from tqdm import tqdm
from .gaussian import Gaussian


class GaussianV2:

    def __init__(
        self,
        mpp: Optional[float] = None,
        sigma: int = 3,
        threshold: float = 0.02,
        *,
        tile_px: int = 512,
        verbose: bool = False,
        grayspace_fraction: float = 1,
        overlap: Optional[int] = None,
        filter_threads: Optional[int] = 8,
        persistent_threads: bool = True,
        pool: Optional["mp.Pool"] = None,
        **wsi_kwargs
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
        self.mpp = mpp
        self.tile_px = tile_px
        self.sigma = sigma
        self.threshold = threshold
        self.verbose = verbose
        self.wsi_kwargs = wsi_kwargs
        self.gs_fraction = grayspace_fraction
        self.filter_threads = filter_threads
        self.persistent_threads = persistent_threads
        self._gaussian_qc = Gaussian(sigma=sigma, threshold=threshold)
        self._tile_pool = pool
        self._tile_pool_is_manual = (pool is not None)
        self._filter_pool = None
        if overlap is None:
            overlap = sigma * 2
        self.overlap = overlap
        if 'lazy_iter' not in wsi_kwargs:
            wsi_kwargs['lazy_iter'] = True

    def apply(self, image: np.ndarray) -> np.ndarray):
        """Apply the QC algorithm to an image tile."""
        return self._gaussian_qc(image)

    def close(self):
        """Close multiprocessing pools and clean up."""
        self.close_pools()

    @property
    def pool(self):
        """Return the tile worker multiprocessing pool."""
        return self._tile_pool

    @pool.setter
    def pool(self, pool):
        """Set the tile worker multiprocessing pool."""
        self._tile_pool = pool
        self._tile_pool_is_manual = (pool is not None)

    @property
    def tile_pool(self):
        """Return the tile worker thread pool used for slide reading."""
        if self._tile_pool is not None:
            return self._tile_pool
        else:
            self._tile_pool = mp.dummy.Pool(int(os.cpu_count() // 2))
            return self._tile_pool

    @property
    def filter_pool(self):
        """Return the thread pool used for Gaussian filtering."""
        if self._filter_pool is not None:
            return self._filter_pool
        elif self.filter_threads:
            self._filter_pool = mp.dummy.Pool(self.filter_threads)
        return self._filter_pool

    def close_pools(self):
        """Close multithreading pools."""
        if self._filter_pool is not None:
            self._filter_pool.close()
            self._filter_pool = None
        if self._tile_pool is not None and not self._tile_pool_is_manual:
            self._tile_pool.close()
            self._tile_pool = None

    def concatenate_and_crop(self, mask: np.ndarray, dim: Tuple[int, int]):
        """Concatenate masks into a single 2D array."""
        mask = np.concatenate(
            [
                np.concatenate([a for a in b], axis=-1)
                for b in mask
            ],
            axis=0
        )
        return mask[:dim[1], :dim[0]]

    def build_mask(self, wsi: "sf.WSI"):
        """Return an empty array for storing masks."""
        return np.ones((wsi.grid.shape[1], wsi.grid.shape[0]), dtype=object)  # type: ignore

    def build_tile_generator(self, wsi: "sf.WSI"):
        """Build a tile generator from a slide."""
        generator = wsi.build_generator(
            shuffle=False,
            show_progress=False,
            img_format='numpy',
            grayspace_fraction=self.gs_fraction,
            pool=self.tile_pool,
            **self.wsi_kwargs)
        if generator is None:
            sf.log.warning(
                "Unable to apply QC {} to slide {}; no tiles extracted.".format(
                    self.__class__.__name__,
                    wsi.name
                )
            )
            return None
        return generator()

    def _calc_mask(self, item, grid_shape):
        """Calculate a Gaussian mask from a given tile."""
        grid_i = item['grid'][1]
        grid_j = item['grid'][0]
        image = item['image']
        

        # Handle edge tiles.
        start_i = start_j = self.overlap
        end_i = end_j = self.tile_px - self.overlap
        if grid_i == 0:
            start_i = 0
        if grid_j == 0:
            start_j = 0
        if grid_i >= grid_shape[1]:
            end_i = None
        if grid_j >= grid_shape[0]:
            end_j = None

        image = image[start_i: end_i, start_j: end_j]
        g_mask = self.apply(image)
        return grid_i, grid_j, g_mask

    def __call__(
        self,
        wsi: "sf.WSI",
    ) -> Optional[np.ndarray]:
        """Apply Gaussian blur filtering to a slide."""
        if self.mpp is None:
            mpp = (wsi.tile_um/wsi.tile_px)*4
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
        )
        qc_mask = self.build_mask(qc_wsi)
        dts = self.build_tile_generator(qc_wsi)

        # Progress bar tracking
        if self.verbose:
            pb = tqdm(dts, desc="Generating...", total=qc_wsi.estimated_num_tiles)
        else:
            pb = dts

        # Apply Gaussian blur to each tile
        if self.filter_pool is not None:
            map_fn = self.filter_pool.imap_unordered
        else:
            map_fn = map
        mask_function = partial(self._calc_mask, grid_shape=qc_wsi.grid.shape)
        for (i, j, cropped) in map_fn(mask_function, pb):
            qc_mask[i, j] = cropped

        # Close pools
        if not self.persistent_threads:
            self.close_pools()

        # Calculate target dimensions for cropping
        target_dim = tuple((np.array(qc_wsi.dimensions) / (mpp / qc_wsi.mpp)).astype(int))

        return self.concatenate_and_crop(qc_mask, dim=target_dim)

# -----------------------------------------------------------------------------
