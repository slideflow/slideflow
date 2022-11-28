from os.path import dirname, abspath, join

from .strided_dl import StridedDL
from .deepfocus.keras_model import load_checkpoint, deepfocus_v3


class DeepFocus(StridedDL):

    def __init__(
        self,
        ckpt: str = 'ver5',
        **kwargs
    ):
        """Utilizes the DeepFocus QC algorithm, as published by Senaras et al:

            Senaras C, et al. DeepFocus: Detection of out-of-focus regions
            in whole slide digital images using deep
            learning. PLOS ONE 13(10): e0205387.

        The published model / saved checkpoint was converted into TF2/Keras
        format and is available at https://github.com/jamesdolezal/deepfocus.

        Args:
            ckpt (str): Checkpoint version. Only 'ver5' is available.
        """
        model = deepfocus_v3()
        ckpt_path = join(dirname(abspath(__file__)), 'deepfocus/checkpoints', ckpt)
        load_checkpoint(model, ckpt_path)
        super().__init__(model=model, pred_idx=1, tile_px=64, tile_um='40x', **kwargs)