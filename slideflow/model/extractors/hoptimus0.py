import timm
import torch

from slideflow.model.extractors._factory_torch import TorchFeatureExtractor




# -----------------------------------------------------------------------------

class Hoptimus0Features(TorchFeatureExtractor):
    """H-optimus-0 pretrained feature extractor.

    This class is used to extract tile-level features from H-optimus-0.

    Feature dimensions: 1536

    Manuscript: https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0

    Hugging Face: https://huggingface.co/bioptimus/H-optimus-0

    """
    tag = 'hoptimus0'
    license = """Apache License 2.0 (License available at https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0)"""
    citation = """
@software{hoptimus0,
  author = {Saillard, Charlie and Jenatton, Rodolphe and Llinares-López, Felipe and Mariet, Zelda and Cahané, David and Durand, Eric and Vert, Jean-Philippe},
  title = {H-optimus-0},
  url = {https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0},
  year = {2024},
}
"""

    def __init__(self, weights=None, device='cuda', **kwargs):
        super().__init__(**kwargs)

        from slideflow.model import torch_utils

        self.device = torch_utils.get_device(device)
        self.model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=(weights is None),
            init_values=1e-5, 
            dynamic_img_size=False
        )
        if weights is not None:
            td = torch.load(weights, map_location=self.device)
            self.model.load_state_dict(td, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------------------
        self.num_features = 1536
        self.transform = self.build_transform(norm_mean=(0.707223, 0.578729, 0.703617),norm_std=(0.211883, 0.230117, 0.177517))
        self.preprocess_kwargs = dict(standardize=False)
        self._weights = weights


    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.

        """
        return self._dump_config(
            class_name='slideflow.model.extractors.hoptimus0.Hoptimus0Features',
            weights=self._weights
        )
    