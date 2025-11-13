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


from slideflow.model.extractors._factory_torch import TorchFeatureExtractor
from .coca_model import CoCa
import torch
from slideflow import errors


# -----------------------------------------------------------------------------

class CoCaImageFeatures(torch.nn.Module):

    def __init__(self, weights=None, device='cuda', embedding_space='image'):
        super().__init__()

        _model_cfg = {'embed_dim': 512,
            'embed_dim_caption': 768,
            'vision_cfg': {'image_size': 448,
            'patch_size': 16,
            'attentional_pool_caption': True,
            'attentional_pool_contrast': True,
            'attn_pooler_heads': 8,
            'n_queries_contrast': 1,
            'n_queries_caption': 256,
            'output_tokens': True},
            'text_cfg': {'context_length': 128,
            'vocab_size': 32007,
            'width': 768,
            'heads': 12,
            'layers': 12,
            'embed_cls': True,
            'output_tokens': True},
            'multimodal_cfg': {'context_length': 128,
            'vocab_size': 32007,
            'width': 768,
            'heads': 12,
            'layers': 12}}

        self._model = CoCa(**_model_cfg)

        if weights is not None:
            td = torch.load(weights, map_location=device)
            self._model.load_state_dict(td, strict=False)

        self.embedding_space = embedding_space

    def forward(self, image_batch):
        with torch.inference_mode():
            if self.embedding_space=='image':
                image_embs  = self._model.encode_image(image_batch, proj_contrast=False, normalize=False)
            elif self.embedding_space=='text':
                image_embs  = self._model.encode_image(image_batch, proj_contrast=True, normalize=True)
            else:
                raise errors.InvalidFeatureExtractor(f"Unrecognized CONCH embedding space: {self.embedding_space}")
            return image_embs



class ConchFeatures(TorchFeatureExtractor):
    """CONCH (CONtrastive learning from Captions for Histopathology) is a vision 
    language foundation model for histopathology, pretrained on currently the largest 
    histopathology-specific vision-language dataset of 1.17M image caption pairs. 
    Compare to other vision language foundation models, it demonstrates state-of-the-art 
    performance across 14 tasks in computational pathology ranging from image classification, 
    text-to-image, and image-to-text retrieval, captioning, and tissue segmentation.

    Feature dimensions: 768

    Manuscript: https://www.nature.com/articles/s41591-024-02856-4

    Hugging Face: https://huggingface.co/MahmoodLab/CONCH

    Github: https://github.com/mahmoodlab/CONCH

    """
    tag = 'conch'
    license = """CC-BY-NC-ND-4.0 (Please check original license under https://huggingface.co/MahmoodLab/CONCH)"""
    citation = """
@article{conch,
  title={A visual-language foundation model for computational pathology},
  author={Lu, Ming Y and Chen, Bowen and Williamson, Drew FK and Chen, Richard J and Liang, Ivy and Ding, Tong and Jaume, Guillaume and Odintsov, Igor and Le, Long Phi and Gerber, Georg and others},
  journal={Nature Medicine},
  pages={863–874},
  volume={30},
  year={2024},
  publisher={Nature Publishing Group}
}
"""

    def __init__(self, weights=None, device='cuda', embedding_space='image', **kwargs):
        super().__init__(**kwargs)

        from slideflow.model import torch_utils

        self.device = torch_utils.get_device(device)
        self.model = CoCaImageFeatures(weights=weights, device=self.device, embedding_space=embedding_space)
        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------------------
        self.num_features = 768

        self.transform = self.build_transform(img_size=224,center_crop=False,interpolation='bicubic',norm_mean=[0.48145466, 0.4578275, 0.40821073],norm_std=[0.26862954, 0.26130258, 0.27577711])
        self.preprocess_kwargs = dict(standardize=False)
        
    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.

        """
        return self._dump_config(
            class_name='slideflow.model.extractors.conch.ConchFeatures'
        )
    