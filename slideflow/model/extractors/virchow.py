import timm
import torch
from torchvision import transforms
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

from ._factory_torch import TorchFeatureExtractor

# -----------------------------------------------------------------------------

class VirchowFeatures(TorchFeatureExtractor):
    """Virchow pretrained feature extractor.

    The feature extractor is a Vision Transformer (ViT) model pretrained on a
    1.5M whole-slide dataset of histopathology images. Virchow is built and
    distributed by Paige, and is available on Hugging Face at hf-hub:paige-ai/Virchow.

    The transformer outputs both a class token (size: 1280) and patch token (256 x 1280).
    As recommended by the authors, the final downstream feature vector is a concatenation
    of the class token and an average pool of the patch token, resulting in a final
    vector size of 2560.

    Feature dimensions: 2560

    Manuscript: Vorontsov, E., et al. (2023). Virchow: A Million-Slide Digital Pathology
    Foundation Model. arXiv preprint arXiv:2309.07778.

    Hugging Face: https://huggingface.co/paige-ai/Virchow

    """

    tag = 'virchow'
    license = """Apache License 2.0 (non-commercial use only). Please see the original license at https://huggingface.co/paige-ai/Virchow."""
    citation = """
@misc{vorontsov2024virchowmillionslidedigitalpathology,
      title={Virchow: A Million-Slide Digital Pathology Foundation Model},
      author={Eugene Vorontsov and Alican Bozkurt and Adam Casson and George Shaikovski and Michal Zelechowski and Siqi Liu and Kristen Severson and Eric Zimmermann and James Hall and Neil Tenenholtz and Nicolo Fusi and Philippe Mathieu and Alexander van Eck and Donghun Lee and Julian Viret and Eric Robert and Yi Kan Wang and Jeremy D. Kunz and Matthew C. H. Lee and Jan Bernhard and Ran A. Godrich and Gerard Oakley and Ewan Millar and Matthew Hanna and Juan Retamero and William A. Moye and Razik Yousfi and Christopher Kanan and David Klimstra and Brandon Rothrock and Thomas J. Fuchs},
      year={2024},
      eprint={2309.07778},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2309.07778},
}
"""

    def __init__(self, weights, device='cuda', **kwargs):
        super().__init__(**kwargs)

        from slideflow.model import torch_utils

        self.device = torch_utils.get_device(device)
        self.model = timm.create_model(
            "vit_huge_patch14_224",
            img_size=224,
            patch_size=14,
            init_values=1e-5,
            num_classes=0,
            mlp_ratio=5.3375,
            global_pool="",
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )
        td = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(td, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------------------
        self.num_features = 2560

        # Note that Virchow uses bicubic interpolation
        # https://huggingface.co/paige-ai/Virchow/blob/main/config.json
        self.transform = self.build_transform(img_size=224, interpolation='bicubic')
        self.preprocess_kwargs = dict(standardize=False)
        self._weights = weights

    def _process_output(self, output):
        """Concatenate class and patch tokens into a single embedding."""
        class_token = output[:, 0]   # 1 x 1280
        patch_tokens = output[:, 1:] # 1 x 256 x 1280
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # 1 x 2560
        return embedding.to(torch.float32)

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.

        """
        return self._dump_config(
            class_name='slideflow.model.extractors.virchow.VirchowFeatures',
            weights=self._weights
        )