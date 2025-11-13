# Slideflow-NonCommercial - Add-ons for the deep learning library Slideflow
# Copyright (C) 2024 James Dolezal
#
# This file is part of Slideflow-NonCommercial.
#
# Slideflow-NonCommercial is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
#
# You are free to share, copy, and redistribute the material in any medium or format, and to adapt, remix, transform, and build upon the material, as long as you follow the terms of the license.
#
# Under the following terms:
# - Attribution: You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
# - NonCommercial: You may not use the material for commercial purposes.
#
# Slideflow-NonCommercial is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Creative Commons Attribution-NonCommercial 4.0 International License for more details.
#
# You should have received a copy of the Creative Commons Attribution-NonCommercial 4.0 International License
# along with Slideflow-NonCommercial. If not, see <https://creativecommons.org/licenses/by-nc/4.0/>.

import torch
import timm

from slideflow.model.extractors._factory_torch import TorchFeatureExtractor

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

    def __init__(self, weights: str, device: str = 'cuda', **kwargs) -> None:
        super().__init__(**kwargs)

        from slideflow.model import torch_utils

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
        self.transform = self.build_transform(img_size=224)
        self.preprocess_kwargs = dict(standardize=False)
        self._weights = weights

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.

        """
        return self._dump_config(
            class_name='slideflow.model.extractors.uni.UNIFeatures',
            weights=self._weights
        )
        
