import numpy as np
import slideflow as sf
from os.path import dirname, abspath, join
from tqdm import tqdm
from .deepfocus.keras_model import load_checkpoint, deepfocus_v3


class DeepFocus:

    def __init__(
        self,
        ckpt: str = 'ver5',
        buffer: int = 8,
        kernel: int = 64,
        mag: str = '40x',
        verbose: bool = False,
        **wsi_kwargs
    ):
        self.buffer = buffer
        self.kernel = kernel
        self.mag = mag
        self.verbose = verbose
        self.wsi_kwargs = wsi_kwargs
        self.model = deepfocus_v3()
        ckpt_path = join(dirname(abspath(__file__)), 'deepfocus/checkpoints', ckpt)
        load_checkpoint(self.model, ckpt_path)

    def __call__(
        self,
        wsi: "sf.WSI",
    ) -> np.ndarray:

        # Initialize whole-slide reader.
        b = self.buffer
        k = self.kernel
        qc_wsi = sf.WSI(wsi.path, tile_px=(k * b), tile_um=self.mag)
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
            y_pred = self.model(batch)[:, 1].numpy()
            predictions = y_pred.reshape(b, b)
            focus_mask[grid_i * b: grid_i * b + b, grid_j * b: grid_j * b + b] = predictions

        return focus_mask > 0.5