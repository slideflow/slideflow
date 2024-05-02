"""HistoSSL Pretrained model.

Model (iBOTViT) and pretrained weights are provided by Owkin, under the
license found in the LICENSE file in the same directory as this source file.

"""
import os
import gdown
import slideflow as sf
from slideflow.util import make_cache_dir_path
from torchvision import transforms

from .._factory_torch import TorchFeatureExtractor

from .ibotvit import iBOTViT


# -----------------------------------------------------------------------------

class HistoSSLFeatures(TorchFeatureExtractor):
    """
    HistoSSL pretrained feature extractor.
    Feature dimensions: 768
    GitHub: https://github.com/owkin/HistoSSLscaling
    """

    tag = 'histossl'
    url = 'https://drive.google.com/uc?id=1uxsoNVhQFoIDxb4RYIiOtk044s6TTQXY'
    license = """
This model is developed and licensed by Owkin, Inc. The license for use is
provided in the LICENSE file in the same directory as this source file
(slideflow/model/extractors/histossl/LICENSE), and is also available
at https://github.com/owkin/HistoSSLscaling. By using this feature extractor,
you agree to the terms of the license.
"""
    citation = """
@article{Filiot2023ScalingSSLforHistoWithMIM,
	author       = {Alexandre Filiot and Ridouane Ghermi and Antoine Olivier and Paul Jacob and Lucas Fidon and Alice Mac Kain and Charlie Saillard and Jean-Baptiste Schiratti},
	title        = {Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling},
	elocation-id = {2023.07.21.23292757},
	year         = {2023},
	doi          = {10.1101/2023.07.21.23292757},
	publisher    = {Cold Spring Harbor Laboratory Press},
	url          = {https://www.medrxiv.org/content/early/2023/07/26/2023.07.21.23292757},
	eprint       = {https://www.medrxiv.org/content/early/2023/07/26/2023.07.21.23292757.full.pdf},
	journal      = {medRxiv}
}
"""
    MD5 = 'e7124eefc87fe6069bf4b864f9ed298c'

    def __init__(self, device=None, center_crop=False, resize=False, weights=None):
        super().__init__()

        from slideflow.model import torch_utils

        if center_crop and resize:
            raise ValueError("center_crop and resize cannot both be True.")

        self.print_license()
        if weights is None:
            weights = self.download()
        self.device = torch_utils.get_device(device)
        self.model = iBOTViT(
            architecture='vit_base_pancan',
            encoder='student',
            weights_path=weights
        )
        self.model.to(self.device)

        # ---------------------------------------------------------------------
        self.num_features = 768
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

    @staticmethod
    def download():
        """Download the pretrained model."""
        dest = make_cache_dir_path('histossl')
        dest = os.path.join(dest, 'ibot_vit_base_pancan.pth')
        if not os.path.exists(dest):
            gdown.download(HistoSSLFeatures.url, dest, quiet=False)
        if sf.util.md5(dest) != HistoSSLFeatures.MD5:
            raise sf.errors.ChecksumError(
                f"Downloaded weights at {dest} failed MD5 checksum."
            )
        return dest

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.model.build_feature_extractor()``.

        """
        cls_name = self.__class__.__name__
        return {
            'class': f'slideflow.model.extractors.histossl.{cls_name}',
            'kwargs': {
                'center_crop': self._center_crop,
                'resize': self._resize
            },
        }
