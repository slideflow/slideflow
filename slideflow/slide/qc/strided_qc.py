import os
import threading
import numpy as np
import slideflow as sf
import multiprocessing as mp

from queue import Queue
from functools import partial
from typing import Union, Optional, Tuple
from tqdm import tqdm


class _StridedQC:

    def __init__(
        self,
        tile_px: int,
        tile_um: Union[str, int],
        *,
        verbose: bool = False,
        grayspace_fraction: float = 1,
        buffer: int = 8,
        **wsi_kwargs
    ):
        """Base QC class for applying a function to a slide via subtiling.

        Should be extended by a class that implements the `.apply()` function.

        Args:
            tile_px (int): Tile size.
            tile_um (str or float): Tile size, in microns (int) or
                magnification (str).

        Keyword arguments:
            verbose (bool): Show a progress bar during calculation.
            buffer (int): Number of tiles (width and height) to extract and
                process simultaneously. Extracted tile size (width/height)
                will be  ``tile_px * buffer``. Defaults to 8.
            grayspace_fraction (float): Grayspace fraction when extracting
                tiles from slides. Defaults to 1 (disables).
            kwargs (Any): All remaining keyword arguments are passed to
                :meth:`slideflow.WSI.build_generator()`.

        """
        if isinstance(tile_um, str):
            self.tile_um = tile_um
        else:
            self.tile_um = tile_um * buffer

        self.buffer = buffer
        self.kernel = tile_px
        self.verbose = verbose
        self.wsi_kwargs = wsi_kwargs
        self.gs_fraction = grayspace_fraction
        self.pb_msg = "Generating..."
        if 'lazy_iter' not in wsi_kwargs:
            wsi_kwargs['lazy_iter'] = True

    def build_mask(self, x, y) -> np.ndarray:
        """Build the base, empty QC mask."""
        return np.ones((x, y), dtype=object)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply a QC function to an image tile. Must be extended."""
        raise NotImplementedError

    def collate_mask(self, mask: np.ndarray):
        """Collates the mask into a final form (e.g., concatenating)."""
        return mask

    def preprocess(self, image: np.ndarray):
        """Apply preprocessing to an image."""
        return image

    def __call__(
        self,
        wsi: "sf.WSI",
    ) -> Optional[np.ndarray]:
        """Apply QC function to a whole-slide image."""

        # Initialize whole-slide reader.
        b = self.buffer
        k = self.kernel
        qc_wsi = sf.WSI(wsi.path, tile_px=(k * b), tile_um=self.tile_um, verbose=False)
        existing_mask = wsi.qc_mask
        if existing_mask is not None:
            qc_wsi.apply_qc_mask(existing_mask)

        # Build tile generator.
        generator = qc_wsi.build_generator(
            shuffle=False,
            show_progress=False,
            img_format='numpy',
            grayspace_fraction=self.gs_fraction,
            **self.wsi_kwargs)
        if generator is None:
            sf.log.warning(
                "Unable to apply QC {} to slide {}; no tiles extracted.".format(
                    self.__class__.__name__,
                    qc_wsi.name
                )
            )
            return None
        dts = generator()

        # Queue for images.
        q = Queue(4)

        # Thread for generating predictions from images in queue.
        def queue_runner():
            nonlocal qc_mask

            while True:
                next_batch = q.get()
                if next_batch is None:
                    return
                grid_i, grid_j, batch = next_batch
                qc_mask[grid_i * b: grid_i * b + b,
                        grid_j * b: grid_j * b + b] = self.apply(batch)

        # Start the queue runner.
        runner = threading.Thread(target=queue_runner)
        runner.start()

        # Apply QC to tiles.
        qc_mask = self.build_mask(
            qc_wsi.grid.shape[1] * b,
            qc_wsi.grid.shape[0] * b
        )
        if self.verbose:
            pb = tqdm(dts, desc=self.pb_msg, total=qc_wsi.estimated_num_tiles)
        else:
            pb = dts
        for item in pb:
            img = self.preprocess(item['image'])
            sz = img.itemsize
            grid_i = item['grid'][1]
            grid_j = item['grid'][0]
            batch = np.lib.stride_tricks.as_strided(img,
                                                    shape=(b, b, k, k, 3),
                                                    strides=(k * sz * 3 * k * b,
                                                            k * sz * 3,
                                                            sz * 3 * k * b,
                                                            sz * 3,
                                                            sz))
            batch = batch.reshape(
                batch.shape[0] * batch.shape[1],
                *batch.shape[2:]
            )
            q.put((grid_i, grid_j, batch))
        q.put(None)
        runner.join()

        return self.collate_mask(qc_mask)


class _StridedQC_V2:
    def __init__(
        self,
        tile_px: int,
        tile_um: Union[str, int],
        *,
        verbose: bool = False,
        grayspace_fraction: float = 1,
        overlap: int = 0,
        filter_threads: Optional[int] = 8,
        persistent_threads: bool = True,
        pool: Optional["mp.Pool"] = None,
        **wsi_kwargs
    ) -> None:
        """Alternative strided slide-level QC filtering (V2).

        Args:
            tile_px (int): Size of tiles (height/width) on which to apply QC.
            tile_um (str, int): Size of tiles in microns or magnification.

        Keyword args:
            verbose (bool): Show a progress bar during calculation.
            grayspace_fraction (float): Grayspace fraction to use during
                tile filtering. Defaults to 1 (disabled).
            overlap (int, optional): Specify the amount of overlap between
                tiles, in pixels, for reducing stitching artifacts. Defaults
                to ``sigma * 2``.
            filter_threads (int, optional): Number of threads to use for
                workers calculating QC filters on tiles. If None, will disable
                multithreading for QC filter workers. Defaults to 8.
            persistent_threads (bool): Keep thread pools alive. If False,
                will close the thread pools after this function has been called
                on a slide. Thread pools can be manually closed with
                ``.close()``. Defaults to True.
            pool (multiprocessing.Pool): Multiprocessing pool to use for tile
                extraction workers. If not provided, will create a thread pool
                with number of threads equal to the cpu count divided by 2.
                This is a separate multithreading pool than the QC filter
                workers. Defaults to None.
            kwargs (Any): All remaining keyword arguments are passed to
                :meth:`slideflow.WSI.build_generator()`.

        """
        self.tile_px = tile_px
        self.tile_um = tile_um
        self.verbose = verbose
        self.overlap = overlap
        self.wsi_kwargs = wsi_kwargs
        self.gs_fraction = grayspace_fraction
        self.filter_threads = filter_threads
        self.persistent_threads = persistent_threads
        self._tile_pool = pool
        self._tile_pool_is_manual = (pool is not None)
        self._filter_pool = None
        if 'lazy_iter' not in wsi_kwargs:
            wsi_kwargs['lazy_iter'] = True

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the QC algorithm to an image tile."""
        raise NotImplementedError

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
            n_threads = max(int(sf.util.num_cpu(default=8) // 2), 1)
            sf.log.debug("Creating tile pool (n_threads={})".format(
                n_threads
            ))
            self._tile_pool = mp.dummy.Pool()
            return self._tile_pool

    @property
    def filter_pool(self):
        """Return the thread pool used for QC filtering."""
        if self._filter_pool is not None:
            return self._filter_pool
        elif self.filter_threads:
            sf.log.debug("Creating filter pool (n_threads={})".format(
                self.filter_threads
            ))
            self._filter_pool = mp.dummy.Pool(self.filter_threads)
        return self._filter_pool

    def close_pools(self):
        """Close multithreading pools."""
        sf.log.debug("Closing tile and filter pools")
        if self._filter_pool is not None:
            self._filter_pool.close()
            self._filter_pool = None
        if self._tile_pool is not None and not self._tile_pool_is_manual:
            self._tile_pool.close()
            self._tile_pool = None

    def concatenate_and_crop(self, mask: np.ndarray, dim: Tuple[int, int]):
        """Concatenate masks into a single 2D array."""
        for j in range(mask.shape[1]):
            for i in range(mask.shape[0]):
                if isinstance(mask[i, j], int):
                    mask[i, j] = self._calc_empty_mask(i, j, mask.shape)
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
        """Calculate a QC mask from a given tile."""
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

    def _calc_empty_mask(self, grid_i, grid_j, grid_shape):
        """Build an empty (1s) mask for a given tile."""
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

        g_mask = np.ones((self.tile_px, self.tile_px))
        g_mask = g_mask[start_i: end_i, start_j: end_j]
        g_mask = g_mask.astype(bool)
        return g_mask

    def get_slide_and_mpp(self, wsi: "sf.WSI") -> Tuple["sf.WSI", float]:
        """Get a WSI object with the proper tile extraction size and stride."""
        stride_div = self.tile_px / (self.tile_px - self.overlap*2)
        qc_wsi = sf.WSI(
            wsi.path,
            tile_px=self.tile_px,
            tile_um=self.tile_um,
            stride_div=stride_div,
            verbose=False,
            use_edge_tiles=True,
            roi_method='ignore'
        )
        mpp = qc_wsi.tile_um / self.tile_px
        return qc_wsi, mpp

    def __call__(
        self,
        wsi: "sf.WSI",
    ) -> Optional[np.ndarray]:
        """Apply QC filtering to a slide."""

        qc_wsi, mpp = self.get_slide_and_mpp(wsi)
        qc_mask = self.build_mask(qc_wsi)
        dts = self.build_tile_generator(qc_wsi)

        # Progress bar tracking
        if self.verbose:
            pb = tqdm(dts, desc="Generating...", total=qc_wsi.estimated_num_tiles)
        else:
            pb = dts

        # Apply QC filter to each tile
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
