import numpy as np
import slideflow as sf
from rich import print

from ._renderer import Renderer, ABORT_RENDER
from slideflow.mil.eval import _predict_mil, _predict_clam

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

    def load_mil_model(self, mil_model, mil_config, extractor, normalizer=None):
        from slideflow.model import torch_utils
        self.device = torch_utils.get_device()
        sf.log.info("Setting up MIL renderer on device: {}".format(self.device))
        self.mil_model = self._model = mil_model.to(self.device)
        self.mil_config = mil_config
        self.extractor = extractor
        self.normalizer = normalizer

    def _convert_img_to_bags(self, img, res):
        """Convert an image into bag format."""
        if self.extractor.backend == 'torch':
            dtype = torch.uint8
        else:
            dtype = tf.uint8
        img = np.expand_dims(img, 0)
        img = sf.io.convert_dtype(img, dtype=dtype)
        if self.normalizer:
            img  = self.normalizer.transform(img)
            if self.extractor.backend == 'torch':
                img = img.to(torch.uint8)
            else:
                img = tf.cast(img, tf.uint8)
            res.normalized = img.numpy()[0].astype(np.uint8)
        if self.extractor.backend == 'torch':
            img = img.to(self.device)
        bags = self.extractor(img)
        return bags

    def _predict_bags(self, bags, attention=False):
        """Generate MIL predictions from bags."""
        from slideflow.mil._params import ModelConfigCLAM, TrainerConfigCLAM

        # Convert to torch tensor
        if sf.util.tf_available and isinstance(bags, tf.Tensor):
            bags = bags.numpy()
        if isinstance(bags, np.ndarray):
            bags = torch.from_numpy(bags)

        # Add a batch dimension & send to GPU
        bags = torch.unsqueeze(bags, dim=0)
        bags = bags.to(self.device)

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

        bags = self._convert_img_to_bags(img, res)
        preds, _ = self._predict_bags(bags)
        res.predictions = preds[0]
        res.uncertainty = None

    def _render_impl(self, res, *args, **kwargs):
        if self.mil_model is not None:
            kwargs['use_model'] = True
        super()._render_impl(res, *args, **kwargs)
        return ABORT_RENDER






