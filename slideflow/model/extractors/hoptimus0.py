import timm
import torch

from slideflow.model.extractors._factory_torch import TorchFeatureExtractor
from slideflow.model.extractors._lora import apply_lora


# -----------------------------------------------------------------------------

class Hoptimus0Features(TorchFeatureExtractor):
    """H-optimus-0 pretrained feature extractor.

    This class is used to extract tile-level features from H-optimus-0.

    Feature dimensions: 1536

    Query/value LoRA adapters can be loaded with ``lora`` (a path to an
    adapter state dict, keys relative to ``lora_first_block``).

    Manuscript: https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0

    Hugging Face: https://huggingface.co/bioptimus/H-optimus-0

    """
    tag = 'hoptimus0'
    # sha1 of the canonical pretrained checkpoint at
    # huggingface.co/bioptimus/H-optimus-0/blob/main/pytorch_model.bin
    # (4.5 GB, hash from `sha1sum pytorch_model.bin`). Used so a
    # ``weights=None`` invocation produces the same identity hash as a
    # ``weights=<path-to-that-file>`` invocation across machines.
    weights_hash = '105e006ee8fb2f51709a51432c1a273e35ed10d4'
    license = """Apache License 2.0 (License available at https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0)"""
    citation = """
@software{hoptimus0,
  author = {Saillard, Charlie and Jenatton, Rodolphe and Llinares-López, Felipe and Mariet, Zelda and Cahané, David and Durand, Eric and Vert, Jean-Philippe},
  title = {H-optimus-0},
  url = {https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0},
  year = {2024},
}
"""

    def __init__(
        self,
        weights=None,
        device='cuda',
        *,
        lora=None,
        lora_first_block=32,
        lora_rank=8,
        lora_alpha=16,
        **kwargs
    ):
        super().__init__(**kwargs)

        from slideflow.model import torch_utils

        self.device = torch_utils.get_device(device)
        if weights is None:
            self.model = timm.create_model(
                "hf-hub:bioptimus/H-optimus-0",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=False
            )
        else:
            # same architecture as the hub config, built locally so a weights
            # file works without network access (e.g. on cluster nodes)
            self.model = timm.create_model(
                "vit_giant_patch14_reg4_dinov2",
                pretrained=False,
                num_classes=0,
                img_size=224,
                init_values=1e-5,
                dynamic_img_size=False
            )
            td = torch.load(weights, map_location='cpu', weights_only=True)
            self.model.load_state_dict(td, strict=True)
        if lora is not None:
            apply_lora(self.model, lora, first_block=lora_first_block, rank=lora_rank, alpha=lora_alpha)
        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------------------
        self.num_features = 1536
        self.transform = self.build_transform(norm_mean=(0.707223, 0.578729, 0.703617),norm_std=(0.211883, 0.230117, 0.177517), img_size=224)
        self.preprocess_kwargs = dict(standardize=False)
        self._weights = str(weights) if weights is not None else None
        self._lora = dict(lora=str(lora) if lora is not None else None, lora_first_block=lora_first_block, lora_rank=lora_rank, lora_alpha=lora_alpha)


    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.

        """
        return self._dump_config(
            class_name='slideflow.model.extractors.hoptimus0.Hoptimus0Features',
            weights=self._weights,
            **self._lora
        )
