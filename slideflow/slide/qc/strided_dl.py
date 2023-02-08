import numpy as np
import slideflow as sf

from typing import Callable, Union
from tqdm import tqdm


class StridedDL:

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
        self.buffer = buffer
        self.kernel = tile_px
        self.tile_um = tile_um
        self.verbose = verbose
        self.wsi_kwargs = wsi_kwargs
        self.model = model
        self.pred_idx = pred_idx
        self.pred_threshold = pred_threshold

    def __call__(
        self,
        wsi: "sf.WSI",
    ) -> np.ndarray:

        # Initialize whole-slide reader.
        b = self.buffer
        k = self.kernel
        qc_wsi = sf.WSI(wsi.path, tile_px=(k * b), tile_um=self.tile_um, verbose=False)
        existing_mask = wsi.qc_mask
        if existing_mask is not None:
            qc_wsi.apply_qc_mask(existing_mask)

        # Build tile generator.
        dts = qc_wsi.build_generator(
            shuffle=False,
            show_progress=False,
            img_format='numpy',
            **self.wsi_kwargs)()

        # Generate prediction for slide.
        focus_mask = np.ones((qc_wsi.grid.shape[1] * b,
                              qc_wsi.grid.shape[0] * b),
                             dtype=np.float32)
        if self.verbose:
            pb = tqdm(dts, desc="Generating...", total=qc_wsi.estimated_num_tiles)
        else:
            pb = dts
        for item in pb:
            img = np.clip(item['image'].astype(np.float32) / 255, 0, 1)
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
            batch = batch.reshape(batch.shape[0] * batch.shape[1], *batch.shape[2:])
            y_pred = self.model(batch)[:, self.pred_idx].numpy()
            predictions = y_pred.reshape(b, b)
            focus_mask[grid_i * b: grid_i * b + b, grid_j * b: grid_j * b + b] = predictions

        return focus_mask > self.pred_threshold