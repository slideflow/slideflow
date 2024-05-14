import torch

from torchvision import transforms
from omegaconf import OmegaConf

from dinov2.eval.setup import build_model_for_eval
from slideflow.model import torch_utils

from ._factory_torch import TorchFeatureExtractor

# -----------------------------------------------------------------------------


class DinoV2Features(TorchFeatureExtractor):
    """DinoV2 feature extractor.

    Feature dimensions: 1024

    GitHub: https://github.com/jamesdolezal/dinov2
    """

    tag = 'dinov2'
    license = "Apache-2.0"
    citation = """
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
"""

    def __init__(self, cfg, weights, device=None, center_crop=False, resize=False):
        super().__init__()

        self.cfg = cfg
        self.weights = weights
        self.device = torch_utils.get_device(device)
        self.model = build_model_for_eval(OmegaConf.load(cfg), weights)
        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------------------
        self.num_features = 1024
        if center_crop:
            all_transforms = [transforms.CenterCrop(224)]
        elif resize:
            all_transforms = [transforms.Resize(224)]
        else:
            all_transforms = []
        all_transforms += [
            transforms.Lambda(lambda x: x / 255.),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
        ]
        self.transform = transforms.Compose(all_transforms)
        self.preprocess_kwargs = dict(standardize=False)
        self._center_crop = center_crop
        self._resize = resize
        # ---------------------------------------------------------------------

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.model.build_feature_extractor()``.

        """
        cls_name = self.__class__.__name__
        return {
            'class': f'slideflow.model.extractors.dinov2.{cls_name}',
            'kwargs': {
                'cfg': self.cfg,
                'weights': self.weights,
                'center_crop': self._center_crop,
                'resize': self._resize
            },
        }
