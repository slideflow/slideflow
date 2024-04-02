import torch
from torchvision import transforms
import timm

from ._factory_torch import TorchFeatureExtractor

# -----------------------------------------------------------------------------

class UNIFeatures(TorchFeatureExtractor):
    """UNI pretrained feature extractor.

    The feature extractor is a Vision Transformer (ViT) model pretrained on a
    100k slide dataset of neoplastic, infectious, inflammatory, and normal
    tissue at Harvard. It is available on Hugging Face at hf-hub:MahmoodLab/uni.

    Feature dimensions: 1024

    Manuscript: Chen, R.J., Ding, T., Lu, M.Y. et al. Towards a general-purpose
    foundation model for computational pathology. Nat Med 30, 850–862 (2024).
    https://doi.org/10.1038/s41591-024-02857-3

    Hugging Face: https://huggingface.co/MahmoodLab/uni

    """

    tag = 'uni'
    license = """Non-commercial use only. Please refer to the original authors."""
    citation = """
﻿@Article{Chen2024,
  author={Chen, Richard J. and Ding, Tong and Lu, Ming Y. and Williamson, Drew F. K. and Jaume, Guillaume and Song, Andrew H. and Chen, Bowen and Zhang, Andrew and Shao, Daniel and Shaban, Muhammad and Williams, Mane and Oldenburg, Lukas and Weishaupt, Luca L. and Wang, Judy J. and Vaidya, Anurag and Le, Long Phi and Gerber, Georg and Sahai, Sharifa and Williams, Walt and Mahmood, Faisal},
  title={Towards a general-purpose foundation model for computational pathology},
  journal={Nature Medicine},
  year={2024},
  month={Mar},
  day={01},
  volume={30},
  number={3},
  pages={850-862},
  issn={1546-170X},
  doi={10.1038/s41591-024-02857-3},
  url={https://doi.org/10.1038/s41591-024-02857-3}
}
"""

    def __init__(self, weights, device='cuda', center_crop=False, resize=False):
        super().__init__()

        from slideflow.model import torch_utils

        if center_crop and resize:
            raise ValueError("center_crop and resize cannot both be True.")

        self.device = torch_utils.get_device(device)
        self.model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0
        )
        td = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(td, strict=True)
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
                std=(0.229, 0.224, 0.225))
        ]
        self.transform = transforms.Compose(all_transforms)
        self.preprocess_kwargs = dict(standardize=False)
        self._center_crop = center_crop
        self._weights = weights
        self._resize = resize

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.model.build_feature_extractor()``.

        """
        return {
            'class': 'slideflow.model.extractors.uni.UNIFeatures',
            'kwargs': {
                'center_crop': self._center_crop,
                'resize': self._resize,
                'weights': self._weights
            }
        }
