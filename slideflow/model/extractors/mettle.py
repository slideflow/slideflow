import timm
import torch

from slideflow.model.extractors._factory_torch import TorchFeatureExtractor
from slideflow.model.extractors._lora import apply_lora


# -----------------------------------------------------------------------------

class MettleFeatures(TorchFeatureExtractor):
    """Mettle feature extractor (ViT-g/14 with four register tokens).

    This class is used to extract tile-level features from Mettle. Mettle has
    no public download, so ``weights`` must point to a local checkpoint.
    Tiles are normalized with the H-optimus-0 statistics, as when Mettle was
    trained and fine-tuned. Optionally loads query/value LoRA adapters.

    Feature dimensions: 1536

    """
    tag = 'mettle'

    def __init__(
        self,
        weights,
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
        self.model = timm.create_model(
            'vit_giant_patch14_reg4_dinov2',
            pretrained=False,
            num_classes=0,
            img_size=224,
            init_values=1e-5,
            dynamic_img_size=False
        )
        td = torch.load(weights, map_location='cpu', weights_only=True)
        # checkpoints may be wrapped, and carry a classification head that is not used here
        for key in ('state_dict', 'model', 'module'):
            if key in td and isinstance(td[key], dict):
                td = td[key]
                break
        missing, unexpected = self.model.load_state_dict(td, strict=False)
        unexpected = [u for u in unexpected if not u.startswith('head.')]
        if missing or unexpected:
            raise ValueError(
                "Mettle weights do not match the ViT-g/14 encoder (missing: {}, "
                "unexpected: {}).".format(missing[:3], unexpected[:3])
            )
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
            class_name='slideflow.model.extractors.mettle.MettleFeatures',
            weights=self._weights,
            **self._lora
        )
