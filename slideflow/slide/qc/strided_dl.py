import numpy as np

from contextlib import contextmanager
from typing import Callable, Union, Optional, TYPE_CHECKING
from .strided_qc import _StridedQC

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
