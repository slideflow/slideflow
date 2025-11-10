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
import torch.nn as nn
from typing import Optional, Union
import numpy as np
from slideflow.model.base import BaseFeatureExtractor


class TitanFeatures(BaseFeatureExtractor):
    """TITAN (Tile-based Integrated Transformer Aggregation Network) is a vision-language
    foundation model for whole-slide image analysis. TITAN operates on CONCH v1.5 patch
    features to generate slide-level embeddings.

    Unlike typical patch-level feature extractors, TITAN:
    - Takes pre-computed CONCH features as input (not raw images)
    - Operates at the slide level using spatial feature grids
    - Requires patch coordinates and magnification information

    Feature dimensions: Depends on model configuration

    Hugging Face: https://huggingface.co/MahmoodLab/TITAN

    Note: This extractor requires:
    1. Pre-extracted CONCH v1.5 features
    2. Patch coordinates at level 0 magnification
    3. Patch size information (typically 1024 for 40x, 512 for 20x)

    """
    tag = 'titan'
    license = """CC-BY-NC-ND-4.0 (Please check original license under https://huggingface.co/MahmoodLab/TITAN)"""
    citation = """
@misc{titan2024,
  title={TITAN: Tile-based Integrated Transformer Aggregation Network},
  author={Mahmood Lab},
  year={2024},
  publisher={Hugging Face},
  howpublished={\\url{https://huggingface.co/MahmoodLab/TITAN}}
}
"""

    def __init__(
        self,
        weights: Optional[str] = None,
        device: str = 'cuda',
        patch_size_lv0: int = 512,
        **kwargs
    ) -> None:
        """Initialize TITAN feature extractor.

        Args:
            weights (str, optional): Path to directory containing TITAN model files,
                                    or None to download from HuggingFace automatically.
                                    Should point to a directory containing model.safetensors,
                                    config.json, and custom modeling files.
            device (str): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
            patch_size_lv0 (int): Patch size at level 0 magnification.
                                  Use 1024 for 40x slides, 512 for 20x slides.
                                  Defaults to 512.
            **kwargs: Additional arguments.
        """
        super().__init__(backend='torch')

        from slideflow.model import torch_utils
        from transformers import AutoModel
        import os

        self.device = torch_utils.get_device(device)
        self.patch_size_lv0 = patch_size_lv0
        self._weights = weights

        # Load TITAN model
        if weights is None:
            # Load directly from HuggingFace Hub (auto-download)
            self.model = AutoModel.from_pretrained(
                'MahmoodLab/TITAN',
                trust_remote_code=True
            )
        else:
            # Load from local directory or cache
            # Don't use local_files_only - let transformers handle the module copying
            self.model = AutoModel.from_pretrained(
                weights,
                trust_remote_code=True
            )

        self.model.to(self.device)
        self.model.eval()

        # TITAN operates on CONCH features, not raw images
        # So it doesn't need preprocessing transforms
        self.num_features = self._get_feature_dim()
        self.preprocess_kwargs = dict(standardize=False)

    def _get_feature_dim(self) -> int:
        """Get the output feature dimension from the model."""
        # Try to infer from model configuration
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
            return self.model.config.hidden_size
        # Default fallback (typical for TITAN)
        return 768

    def encode_slide_from_patch_features(
        self,
        features: Union[torch.Tensor, np.ndarray],
        coords: Union[torch.Tensor, np.ndarray],
        patch_size_lv0: Optional[int] = None
    ) -> torch.Tensor:
        """Encode a slide from pre-computed patch features.

        Args:
            features: Patch features, shape (n_patches, feature_dim) or (1, n_patches, feature_dim).
                      Should be CONCH v1.5 features (768-dim).
                      If 2D, will be unsqueezed to add batch dimension.
            coords: Patch coordinates at level 0, shape (n_patches, 2) or (1, n_patches, 2).
                   If 2D, will be unsqueezed to add batch dimension.
            patch_size_lv0: Patch size at level 0 magnification.
                           If None, uses the value from __init__.

        Returns:
            Slide-level embedding tensor, shape (1, embedding_dim).
        """
        # Convert numpy arrays to tensors if needed
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords)

        # Convert coords to long (int64) if needed - TITAN requires integer coordinates
        if coords.dtype in [torch.float32, torch.float64]:
            coords = coords.long()

        # Move to device
        features = features.to(self.device)
        coords = coords.to(self.device)

        # Add batch dimension if needed (TITAN expects shape (1, N, C) and (1, N, 2))
        if features.dim() == 2:
            features = features.unsqueeze(0)  # (N, C) -> (1, N, C)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)  # (N, 2) -> (1, N, 2)

        # Use provided patch_size_lv0 or fall back to instance value
        patch_size = patch_size_lv0 if patch_size_lv0 is not None else self.patch_size_lv0

        # Encode slide - use correct parameter names
        with torch.inference_mode():
            slide_embedding = self.model.encode_slide_from_patch_features(
                patch_features=features,
                patch_coords=coords,
                patch_size_lv0=patch_size
            )

        return slide_embedding

    def __call__(self, obj, **kwargs):
        """Process input object.

        For TITAN, the typical usage is different from patch-level extractors.
        This method provides compatibility with the standard extractor interface,
        but users should prefer `encode_slide_from_patch_features` for direct usage.

        Args:
            obj: Can be:
                - Dict with keys 'features', 'coords', and optionally 'patch_size_lv0'
                - Tuple of (features, coords) or (features, coords, patch_size_lv0)
            **kwargs: Additional arguments passed to encode_slide_from_patch_features.

        Returns:
            Slide-level embedding.
        """
        if isinstance(obj, dict):
            features = obj['features']
            coords = obj['coords']
            patch_size_lv0 = obj.get('patch_size_lv0', None)
            return self.encode_slide_from_patch_features(
                features, coords, patch_size_lv0, **kwargs
            )
        elif isinstance(obj, (tuple, list)):
            if len(obj) == 2:
                features, coords = obj
                patch_size_lv0 = None
            elif len(obj) == 3:
                features, coords, patch_size_lv0 = obj
            else:
                raise ValueError(
                    f"Expected tuple of length 2 or 3, got {len(obj)}. "
                    "Format: (features, coords) or (features, coords, patch_size_lv0)"
                )
            return self.encode_slide_from_patch_features(
                features, coords, patch_size_lv0, **kwargs
            )
        else:
            raise TypeError(
                f"TITAN requires CONCH features and coordinates as input, "
                f"not {type(obj)}. Use encode_slide_from_patch_features() directly "
                f"or pass a dict/tuple with 'features' and 'coords'."
            )

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.
        """
        return {
            'class': 'slideflow.model.extractors.titan.TitanFeatures',
            'kwargs': {
                'weights': self._weights,
                'patch_size_lv0': self.patch_size_lv0
            }
        }
