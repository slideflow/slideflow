import numpy as np

from typing import Callable, Union
from .strided_qc import _StridedQC


class StridedDL(_StridedQC):

    def __init__(
        self,
        model: Callable,
        pred_idx: int,
        tile_px: int,
        tile_um: Union[str, int],
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
            dest (str, optional): Path in which to save the qc mask.
                If None, will save in the same directory as the slide.
                Defaults to None.
                
        """
        super().__init__(
            tile_px=tile_px,
            tile_um=tile_um,
            buffer=buffer,
            verbose=verbose,
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
        y_pred = self.model(image)[:, self.pred_idx].numpy()
        return y_pred.reshape(self.buffer, self.buffer) > self.pred_threshold

    def collate_mask(self, mask: np.ndarray):
        """Convert the mask from predictions to bool using a threshold."""
        return mask > self.pred_threshold

    def preprocess(self, image: np.ndarray):
        """Apply preprocessing to an image."""
        return np.clip(image.astype(np.float32) / 255, 0, 1)