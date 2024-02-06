import numpy as np

from tqdm import tqdm
from contextlib import contextmanager
from typing import Callable, Union, Optional, TYPE_CHECKING
from .strided_qc import _StridedQC, _StridedQC_V2

if TYPE_CHECKING:
    import slideflow as sf


class StridedDL(_StridedQC):

    def __init__(
        self,
        model: Callable,
        pred_idx: int,
        tile_px: int,
        tile_um: Union[str, int],
        *,
        buffer: int = 8,
        verbose: bool = False,
        pred_threshold: float = 0.5,
        **wsi_kwargs
    ):
        """QC function which uses a deep learning model to generate a QC mask.

        When this QC method is applied to a slide, the given deep learning model
        generates predictions across the whole-slide image (using the class index
        specified by ``pred_idx``). Areas with a prediction above
        ``pred_threshold`` are masked, to be discarded.

        Examples
            Create a DeepFocus module that filters out-of-focus tiles.

                .. code-block:: python

                    import slideflow as sf
                    from slideflow.slide.qc import strided_dl
                    from deepfocus import deepfocus_v3

                    deepfocus = strided_dl.StridedDL(
                        model=deepfocus_v3(),
                        pred_idx=1,
                        tile_px=64,
                        tile_um='40x'
                    )
                    wsi = sf.WSI(...)
                    wsi.qc(deepfocus)


            Do the same, but using class inheritance.

                .. code-block:: python

                    import slideflow as sf
                    from slideflow.slide.qc import strided_dl
                    from deepfocus import deepfocus_v3

                    class DeepFocus(strided_dl.StridedDL):

                        def __init__(self):
                            model = deepfocus_v3()
                            checkpoint = '/path/to/checkpoint-ver5'
                            load_checkpoint(model, checkpoint)
                            super().__init__(
                                model=model,
                                pred_idx=1,
                                tile_px=64,
                                tile_um='40x'
                            )

                    wsi = sf.WSI(...)
                    deepfocus = DeepFocus()
                    wsi.qc(deepfocus)


        Args:
            model (callable): Deep learning model.
            pred_idx (int): Index of the model output to interpret as the
                final prediction.
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
            pred_threshold (float): Predictions below this value are masked.
            kwargs (Any): All remaining keyword arguments are passed to
                :meth:`slideflow.WSI.build_generator()`.

        """
        super().__init__(
            tile_px=tile_px,
            tile_um=tile_um,
            buffer=buffer,
            verbose=verbose,
            lazy_iter=True,
            deterministic=False,
            **wsi_kwargs
        )
        self.model = model
        self.pred_idx = pred_idx
        self.pred_threshold = pred_threshold

    def build_mask(self, x, y) -> np.ndarray:
        """Build the base, empty QC mask."""
        return np.ones((x, y), dtype=np.float32)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Predict focus value of an image tile using DeepFocus model."""
        y_pred = self.model(image, training=False)[:, self.pred_idx].numpy()
        return y_pred.reshape(self.buffer, self.buffer)

    def collate_mask(self, mask: np.ndarray):
        """Convert the mask from predictions to bool using a threshold."""
        if self.pred_threshold is not None:
            return mask > self.pred_threshold
        else:
            return mask

    def preprocess(self, image: np.ndarray):
        """Apply preprocessing to an image."""
        return np.clip(image.astype(np.float32) / 255, 0, 1)

    @contextmanager
    def _set_threshold(self, threshold: Optional[Union[bool, float]]):
        """Temporariliy set or disable the prediction threshold."""
        _orig_threshold = self.pred_threshold
        if isinstance(threshold, float):
            # Set the threshold to a given threshold
            self.pred_threshold = threshold
        elif threshold is False:
            # Disable thresholding (return raw values)
            self.pred_threshold = None

        yield

        # Return the threshold to irs original value
        self.pred_threshold = _orig_threshold

    def __call__(
        self,
        wsi: "sf.WSI",
        threshold: Optional[Union[bool, float]] = None
    ) -> Optional[np.ndarray]:

        with self._set_threshold(threshold):
            return super().__call__(wsi)


# -----------------------------------------------------------------------------

def _taper_mask(ly=224, lx=224, sig=7.5):
    bsize = max(224, max(ly, lx))
    xm = np.arange(bsize)
    xm = np.abs(xm - xm.mean())
    mask = 1/(1 + np.exp((xm - (bsize/2-20)) / sig))
    mask = mask * mask[:, np.newaxis]
    mask = mask[bsize//2-ly//2 : bsize//2+ly//2+ly%2,
                bsize//2-lx//2 : bsize//2+lx//2+lx%2]
    return mask

# -----------------------------------------------------------------------------

class StridedDL_V2(_StridedQC_V2):

    """Implementation of a strided deep learning QC algorithm.

    The _StrdedQC_V2 base class collates tiled QC masks into a single mask by
    cropping out the overlap regions. This approach is suitable for algorithms
    that generate artifacts at the edges of tiles, but is not adequate for
    stitching together deep learning predictions.

    This class is a subclass of _StridedQC_V2, and is designed to stitch
    together output from a deep learning QC model for tiles using a tapered mask.
    """

    def __init__(
        self,
        *args,
        out_classes: int = 0,
        **kwargs
    ):
        """Create a new StridedDL_V2 object.

        Args:
            *args (Any): Arguments to pass to the parent class.
            out_classes (int): Number of output classes from the deep learning model.
                If provided, the shape of the QC mask will be (out_classes, h, w).
                If 0 or not provided, the shape will be (h, w).
            **kwargs (Any): Keyword arguments to pass to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.out_classes = out_classes

    def _calc_mask(self, item):
        """Calculate a QC mask from a given tile."""
        grid_i = item['grid'][0]
        grid_j = item['grid'][1]
        image = item['image']

        mask = self.apply(image)
        return mask, (grid_i, grid_j)

    def build_masks(self, wsi: "sf.WSI"):
        """Return empty arrays for storing QC mask and the average (taper) mask."""
        dim = (wsi.dimensions[1], wsi.dimensions[0])
        px_ratio = wsi.tile_px / wsi.full_extract_px
        target_dim = tuple((np.array(dim) * px_ratio).astype(int))
        if self.out_classes:
            qc_dim = (self.out_classes, target_dim[0], target_dim[1])
        else:
            qc_dim = target_dim
        qc_mask = np.zeros(qc_dim, np.float32)
        avg_mask = np.zeros(target_dim, np.float32)
        return qc_mask, avg_mask

    def get_tile_bounds(self, wsi: "sf.WSI", i: int, j: int):
        """Return the bounds of a tile."""
        fy, fx = wsi.grid_to_coord(i, j, anchor="topleft")
        px_ratio = wsi.tile_px / wsi.full_extract_px
        x0 = int(fx * px_ratio)
        y0 = int(fy * px_ratio)
        x1 = x0 + wsi.tile_px
        y1 = y0 + wsi.tile_px
        return x0, x1, y0, y1

    def __call__(
        self,
        wsi: "sf.WSI",
    ) -> Optional[np.ndarray]:
        """Apply QC filtering to a slide."""

        qc_wsi, mpp = self.get_slide_and_mpp(wsi)
        qc_mask, avg_mask = self.build_masks(qc_wsi)
        dts = self.build_tile_generator(qc_wsi)

        # Get the base taper mask
        taper_mask = _taper_mask(ly=self.tile_px, lx=self.tile_px, sig=7.5)

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
        for (tile_mask, (i, j)) in map_fn(self._calc_mask, pb):
            x0, x1, y0, y1 = self.get_tile_bounds(qc_wsi, i, j)
            if self.out_classes:
                x1 = min(x1, qc_mask.shape[1])
                y1 = min(y1, qc_mask.shape[2])
                qc_mask[:, x0:x1, y0:y1] += tile_mask[:, 0: x1-x0, 0: y1-y0] * taper_mask[0: x1-x0, 0: y1-y0]
            else:
                x1 = min(x1, qc_mask.shape[0])
                y1 = min(y1, qc_mask.shape[1])
                qc_mask[x0:x1, y0:y1] += tile_mask[0: x1-x0, 0: y1-y0] * taper_mask[0: x1-x0, 0: y1-y0]
            avg_mask[x0:x1, y0:y1] += taper_mask[0: x1-x0, 0: y1-y0]

        # Normalize the mask
        qc_mask = qc_mask / avg_mask

        # Close pools
        if not self.persistent_threads:
            self.close_pools()

        return qc_mask
