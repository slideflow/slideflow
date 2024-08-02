import timm
import torch
import numpy as np
from torchvision import transforms
from typing import Optional

from ._factory_torch import TorchFeatureExtractor

# -----------------------------------------------------------------------------

class GigapathFeatures(TorchFeatureExtractor):
    """Gigapath pretrained feature extractor.

    ...

    Feature dimensions: 1536

    Manuscript: ...

    Hugging Face: ...

    """

    tag = 'gigapath'
    license = """..."""
    citation = """
    ...
"""

    def __init__(self, weights=None, device='cuda'):
        super().__init__()

        from slideflow.model import torch_utils

        self.device = torch_utils.get_device(device)
        self.model = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", 
            pretrained=(weights is not None)
        )
        if weights is not None:
            td = torch.load(weights, map_location=self.device)
            self.model.load_state_dict(td, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------------------
        self.num_features = 1536
        # This preprocessing, with resizing to 256 followed by
        # center crop to 224, is the same as the original Gigapath
        all_transforms = [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
        all_transforms += [
            transforms.Lambda(lambda x: x / 255.),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
        ]
        self.transform = transforms.Compose(all_transforms)
        self.preprocess_kwargs = dict(standardize=False)
        self._weights = weights

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.

        """
        return {
            'class': 'slideflow.model.extractors.virchow.VirchowFeatures',
            'kwargs': {
                'weights': self._weights
            }
        }

# -----------------------------------------------------------------------------

class GigapathSlideFeatures:
    """Gigapath whole-slide feature embedding generator."""

    def __init__(
        self,
        tile_encoder_weights: Optional[str] = None,
        slide_encoder_weights: Optional[str] = None,
        *,
        global_pool: bool = False,
        device: str = 'cuda',
    ):
        """Initialize the Gigapath slide feature generator."""
        try:
            from gigapath import slide_encoder
        except ImportError:
            raise ImportError("Please install the gigapath package to use this feature extractor.")

        from slideflow.model import torch_utils

        self.device = torch_utils.get_device(device)

        # Build the encoders.
        self.tile_encoder = GigapathFeatures(
            weights=tile_encoder_weights,
            device=self.device,
        )
        self.slide_encoder = slide_encoder.create_model(
            "hf_hub:prov-gigapath/prov-gigapath",
            "gigapath_slide_enc12l768d",
            1536,
            global_pool=global_pool
        ).to(self.device)
        self.slide_encoder.eval()

    def __call__(self, wsi):
        """Generate whole-slide feature embedding."""

        from slideflow.model.torch import autocast

        # Generate tile embeddings
        embed_grid = self.tile_encoder(wsi)
        unmasked_indices = np.where(~embed_grid.mask.any(axis=-1))

        # Get coordinates and reshape into a grid
        coords = wsi.get_tile_coord(anchor='center')
        tile_x = coords[:, 0]
        tile_y = coords[:, 1]
        grid_x = coords[:, 2]
        grid_y = coords[:, 3]
        width, height = wsi.shape
        coord_grid = np.zeros((height, width, 2), dtype=tile_x.dtype)
        coord_grid[grid_y, grid_x, 0] = tile_x
        coord_grid[grid_y, grid_x, 1] = tile_y

        # Reshape grids to (1, n_tiles, 1536)
        tile_embeds = torch.from_numpy(embed_grid[unmasked_indices].data)
        tile_embeds = tile_embeds.unsqueeze(0).to(self.device)
        coords = torch.from_numpy(coord_grid[unmasked_indices])
        coords = coords.unsqueeze(0).to(self.device)

        # Run slide inference
        with autocast(self.device.type, torch.float16):
            with torch.inference_mode():
                return self.slide_encoder(tile_embeds, coords)



