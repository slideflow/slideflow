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
from typing import Optional

from slideflow.model.extractors._factory_torch import TorchFeatureExtractor

# -----------------------------------------------------------------------------

class ConchFeatures(TorchFeatureExtractor):
    """CONCH (Contrastive Learning from Captions for Histopathology) v1.5 feature extractor.

    CONCH is a visual-language foundation model for histopathology, pretrained using
    contrastive learning on paired image-text data. The vision encoder is based on
    Vision Transformer (ViT) architecture.

    Feature dimensions: 768 (for v1.5), 512 (for v1)

    Manuscript: Lu, M.Y., Chen, B., Williamson, D.F.K. et al. A visual-language
    foundation model for computational pathology. Nat Med 30, 863–874 (2024).
    https://doi.org/10.1038/s41591-024-02856-4

    Hugging Face: https://huggingface.co/MahmoodLab/CONCH (v1)
                  https://huggingface.co/MahmoodLab/conchv1_5 (v1.5)

    """

    tag = 'conch'
    license = """Non-commercial use only. Please refer to the original authors at MahmoodLab."""
    citation = """
@Article{Lu2024,
  author={Lu, Ming Y. and Chen, Bowen and Williamson, Drew F. K. and Chen, Richard J. and Liang, Ivy and Ding, Tong and Jaume, Guillaume and Odintsov, Igor and Zhang, Andrew and Le, Long Phi and Gerber, Georg and Glass, Anil V. and Williams, Sharifa and Mahmood, Faisal},
  title={A visual-language foundation model for computational pathology},
  journal={Nature Medicine},
  year={2024},
  month={Mar},
  day={01},
  volume={30},
  number={3},
  pages={863-874},
  issn={1546-170X},
  doi={10.1038/s41591-024-02856-4},
  url={https://doi.org/10.1038/s41591-024-02856-4}
}
"""

    def __init__(self, weights: Optional[str] = None, device: str = 'cuda', **kwargs) -> None:
        """Initialize CONCH feature extractor.

        Args:
            weights (str, optional): Path to the model weights file. If None, will attempt
                                    to download from Hugging Face hub.
            device (str): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
            **kwargs: Additional arguments passed to TorchFeatureExtractor.
        """
        super().__init__(**kwargs)

        from slideflow.model import torch_utils
        from huggingface_hub import hf_hub_download
        import os

        self.device = torch_utils.get_device(device)
        self._weights = weights

        # Download weights if not provided
        if weights is None:
            hf_token = os.getenv('HF_TOKEN')
            print("Downloading CONCH v1.5 weights from Hugging Face...")
            weights = hf_hub_download(
                repo_id='MahmoodLab/conchv1_5',
                filename='pytorch_model_vision.bin',
                token=hf_token
            )
            print(f"Weights downloaded to: {weights}")
            self._weights = weights

        # Determine version based on weights file
        # v1.5 uses 'pytorch_model_vision.bin', v1 uses 'pytorch_model.bin'
        if 'conchv1_5' in str(weights) or 'pytorch_model_vision' in str(weights):
            self.version = 'v1.5'
            self.num_features = 768
        else:
            self.version = 'v1'
            self.num_features = 512

        # Create model using timm
        # CONCH uses ViT-B/16 architecture
        self.model = timm.create_model(
            "vit_base_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,  # Remove classification head
            dynamic_img_size=True
        )

        # Load weights
        state_dict = torch.load(weights, map_location='cpu')

        # Handle different state dict formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Remove any 'model.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('model.', '') if k.startswith('model.') else k
            new_state_dict[new_key] = v

        # Load the state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)

        if missing_keys:
            print(f"Warning: Missing keys when loading CONCH weights: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading CONCH weights: {unexpected_keys[:5]}...")

        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------------------
        # Build preprocessing transform
        # CONCH uses standard ImageNet normalization
        self.transform = self.build_transform(
            img_size=224,
            norm_mean=(0.485, 0.456, 0.406),
            norm_std=(0.229, 0.224, 0.225)
        )
        self.preprocess_kwargs = dict(standardize=False)

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.

        """
        return self._dump_config(
            class_name='slideflow.model.extractors.conch.ConchFeatures',
            weights=self._weights
        )
