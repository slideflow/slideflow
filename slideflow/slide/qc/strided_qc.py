import numpy as np
import slideflow as sf

from typing import Union, Optional
from tqdm import tqdm


class StridedQC:

    def __init__(
        self,
        tile_px: int,
        tile_um: Union[str, int],
        buffer: int = 8,
        verbose: bool = False,
        grayspace_fraction: float = 1,
        **wsi_kwargs
    ):
        """Efficiently apply a QC function to a slide via strided subtiling."""

        self.buffer = buffer
        self.kernel = tile_px
        self.tile_um = tile_um
        self.verbose = verbose
        self.wsi_kwargs = wsi_kwargs
        self.gs_fraction = grayspace_fraction
        if 'lazy_iter' not in wsi_kwargs:
            wsi_kwargs['lazy_iter'] = True

    def build_mask(self, x, y) -> np.ndarray:
        return np.ones((x, y), dtype=object)

    def apply(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def collate_mask(self, mask: np.ndarray):
        return mask

    def preprocess(self, image: np.ndarray):
        return image

    def __call__(
        self,
        wsi: "sf.WSI",
    ) -> Optional[np.ndarray]:

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

        # Apply QC to tiles.
        qc_mask = self.build_mask(
            qc_wsi.grid.shape[1] * b,
            qc_wsi.grid.shape[0] * b
        )
        if self.verbose:
            pb = tqdm(dts, desc="Generating...", total=qc_wsi.estimated_num_tiles)
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
            qc_mask[grid_i * b: grid_i * b + b,
                    grid_j * b: grid_j * b + b] = self.apply(batch)

        return self.collate_mask(qc_mask)