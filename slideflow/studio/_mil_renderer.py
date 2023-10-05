import numpy as np
import slideflow as sf
from typing import Optional
from rich import print

from ._renderer import Renderer
from slideflow.mil.eval import _predict_mil, _predict_clam
from slideflow.model.extractors import rebuild_extractor

# -----------------------------------------------------------------------------

if sf.util.tf_available:
    import tensorflow as tf
    import slideflow.io.tensorflow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
if sf.util.torch_available:
    import torch
    import torchvision
    import slideflow.io.torch

# -----------------------------------------------------------------------------

class MILRenderer(Renderer):

    def __init__(self, *args, mil_model_path: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mil_model = None
        self.mil_config = None
        self.extractor = None
        self.normalizer = None
        if mil_model_path:
            self.load_model(mil_model_path)

    def load_model(self, mil_model_path: str, device: Optional[str] = None) -> None:
        sf.log.info("Loading MIL model at {}".format(mil_model_path))
        if device is None:
            from slideflow.model import torch_utils
            device = torch_utils.get_device()
        self.device = device
        self.extractor, self.normalizer = rebuild_extractor(
            mil_model_path, native_normalizer=(sf.slide_backend()=='cucim')
        )
        self.mil_model, self.mil_config = sf.mil.utils.load_model_weights(mil_model_path)
        self.mil_model.to(self.device)
        self._model = self.mil_model
        sf.log.info("Model loading successful")

    def _convert_img_to_bags(self, img):
        """Convert an image into bag format."""
        if self.extractor.backend == 'torch':
            dtype = torch.uint8
        else:
            dtype = tf.uint8
        img = np.expand_dims(img, 0)
        img = sf.io.convert_dtype(img, dtype=dtype)
        if self.extractor.backend == 'torch':
            img = img.to(self.device)
        bags = self.extractor(img)
        return bags

    def _predict_bags(self, bags, attention=False):
        """Generate MIL predictions from bags."""
        from slideflow.mil._params import ModelConfigCLAM, TrainerConfigCLAM

        # Add a batch dimension
        bags = torch.unsqueeze(bags, dim=0)

        if (isinstance(self.mil_config, TrainerConfigCLAM)
        or isinstance(self.mil_config.model_config, ModelConfigCLAM)):
            preds, att = _predict_clam(
                self.mil_model,
                bags,
                attention=attention
            )
        else:
            preds, att = _predict_mil(
                self.mil_model,
                bags,
                attention=attention,
                use_lens=self.mil_config.model_config.use_lens,
                apply_softmax=self.mil_config.model_config.apply_softmax
            )
        return preds, att

    def _run_models(self, img, res, **kwargs):
        """Generate an MIL single-bag prediction."""
        bags = self._convert_img_to_bags(img)
        preds, _ = self._predict_bags(bags)
        res.predictions = preds[0]
        res.uncertainty = None

    def _render_impl(self, res, *args, **kwargs):
        if self.mil_model is not None:
            kwargs['use_model'] = True
        super()._render_impl(res, *args, **kwargs)
