import torch
import slideflow as sf
import numpy as np

from torchvision import transforms
from huggingface_hub import hf_hub_download

from omegaconf import OmegaConf
from dinov2.eval.setup import build_model_for_eval

from ..base import BaseFeatureExtractor
from ._slide import features_from_slide

# -----------------------------------------------------------------------------


class DinoV2Features(BaseFeatureExtractor):
    """DinoV2 feature extractor.

    Feature dimensions: 1024

    GitHub: https://github.com/jamesdolezal/dinov2
    """

    tag = 'dinov2'

    def __init__(self, cfg, weights, device=None, center_crop=False):
        super().__init__(backend='torch')

        from slideflow.model import torch_utils
        self.cfg = cfg
        self.weights = weights
        self.device = torch_utils.get_device(device)
        self.model = build_model_for_eval(OmegaConf.load(cfg), weights)
        self.model.fc = torch.nn.Identity().to(self.device)
        self.model.to(self.device)

        # ---------------------------------------------------------------------
        self.num_features = 1024
        self.num_classes = 0
        all_transforms = [transforms.CenterCrop(224)] if center_crop else []
        all_transforms += [
            transforms.Lambda(lambda x: x / 255.),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
        ]
        self.transform = transforms.Compose(all_transforms)
        self.preprocess_kwargs = dict(standardize=False)
        self._center_crop = center_crop
        # ---------------------------------------------------------------------

    def __call__(self, obj, **kwargs):
        """Generate features for a batch of images or a WSI."""
        if isinstance(obj, sf.WSI):
            grid = features_from_slide(self, obj, **kwargs)
            return np.ma.masked_where(grid == -99, grid)
        elif kwargs:
            raise ValueError(
                f"{self.__class__.__name__} does not accept keyword arguments "
                "when extracting features from a batch of images."
            )
        assert obj.dtype == torch.uint8
        obj = self.transform(obj)
        return self.model(obj)

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.model.build_feature_extractor()``.

        """
        cls_name = self.__class__.__name__
        return {
            'class': f'slideflow.model.extractors.dinov2.{cls_name}',
            'kwargs': {'center_crop': self._center_crop},
            'cfg': self.cfg,
            'weights': self.weights,
        }
