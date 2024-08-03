import slideflow as sf
import timm
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from typing import Optional, List, Tuple, TYPE_CHECKING
from slideflow.util.tfrecord2idx import find_index

from ._factory_torch import TorchFeatureExtractor

if TYPE_CHECKING:
    from slideflow import WSI, Dataset

# -----------------------------------------------------------------------------

class GigapathTileFeatures(TorchFeatureExtractor):
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

    def __init__(self, weights=None, device='cuda', resize=256, center_crop=224, **kwargs):
        super().__init__(**kwargs)

        from slideflow.model import torch_utils

        self.device = torch_utils.get_device(device)
        self.model = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", 
            pretrained=(weights is None)
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
        all_transforms = []

        if resize:
            all_transforms += [
                transforms.Resize(
                    256 if resize is True else resize,
                    interpolation=transforms.InterpolationMode.BICUBIC),
            ]
        if center_crop:
            all_transforms += [
                transforms.CenterCrop(
                    224 if center_crop is True else center_crop),
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
        self._resize = resize
        self._center_crop = center_crop

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.

        """
        return {
            'class': 'slideflow.model.extractors.gigapath.GigapathTileFeatures',
            'kwargs': {
                'weights': self._weights,
                'resize': self._resize,
                'center_crop': self._center_crop,
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
        self.tile_encoder = GigapathTileFeatures(
            weights=tile_encoder_weights,
            device=self.device,
        )
        if slide_encoder_weights is None:
            slide_encoder_weights = "hf_hub:prov-gigapath/prov-gigapath"
        self.slide_encoder = slide_encoder.create_model(
            slide_encoder_weights,
            "gigapath_slide_enc12l768d",
            1536,
            global_pool=global_pool
        ).to(self.device)
        self.slide_encoder.eval()

    def _reshape_coords(self, coords: np.ndarray, height: int, width: int) -> np.ndarray:
        """Reshape tile coordinates into a grid.

        Args:
            coords (np.ndarray): Tile coordinates.
            height (int): Slide height.
            width (int): Slide width.

        Returns:
            np.ndarray: Grid of tile coordinates.

        """
        tile_x = coords[:, 0]
        tile_y = coords[:, 1]
        grid_x = coords[:, 2]
        grid_y = coords[:, 3]
        coord_grid = np.zeros((height, width, 2), dtype=tile_x.dtype)
        coord_grid[grid_y, grid_x, 0] = tile_x
        coord_grid[grid_y, grid_x, 1] = tile_y
        return coord_grid

    def __call__(self, wsi: "WSI") -> torch.Tensor:
        """Generate whole-slide feature embedding.

        Args:
            wsi (WSI): Whole-slide image.

        Returns:
            torch.Tensor: Whole-slide embedding.

        """
        # Generate tile embeddings
        embed_grid = self.tile_encoder(wsi)
        unmasked_indices = np.where(~embed_grid.mask.any(axis=-1))

        # Get coordinates and reshape into a grid
        coords = wsi.get_tile_coord(anchor='center')
        width, height = wsi.shape
        coord_grid = self._reshape_coords(coords, height, width)

        # Reshape grids to (1, n_tiles, 1536)
        tile_embeds = torch.from_numpy(embed_grid[unmasked_indices].data)
        tile_embeds = tile_embeds.unsqueeze(0).to(self.device)
        coords = torch.from_numpy(coord_grid[unmasked_indices])
        coords = coords.unsqueeze(0).to(self.device)

        # Run slide inference
        return self.run_inference(tile_embeds, coords)

    def run_inference(self, tile_embed: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Run inference on a single slide.

        Args:
            tile_embed (torch.Tensor): Tile embeddings. Shape: (1, n_tiles, 1536).
            coords (torch.Tensor): Tile coordinates. Shape: (1, n_tiles, 2).

        Returns:
            torch.Tensor: Whole-slide embedding.

        """
        from slideflow.model.torch import autocast

        with autocast(self.device.type, torch.float16):
            with torch.inference_mode():
                return self.slide_encoder(tile_embed, coords)

    def inference_from_bags(self, bags: List[str], indices: List[str]) -> torch.Tensor:
        """Run inference on a set of bags.

        Args:
            bags (List[str]): List of paths to bags (tile embeddings).
            indices (List[str]): List of paths to bag indices (tile coordinates).

        Returns:
            torch.Tensor: Whole-slide embeddings.

        """
        slide_embeds = []
        for bag, index in tqdm(zip(bags, indices), total=len(bags)):
            tile_embed = torch.load(bag).unsqueeze(0).to(self.device)
            coords = torch.from_numpy(np.load(index)['arr_0']).unsqueeze(0).to(self.device)
            slide_embed = self.run_inference(tile_embed, coords)[0]
            slide_embeds.append(slide_embed)
        return torch.cat(slide_embeds, dim=0)

# -----------------------------------------------------------------------------

def generate_slide_embeddings(
    dataset: "Dataset",
    bags_path: str,
    **kwargs
) -> Tuple[torch.Tensor, List[str]]:
    """Generate slide embeddings from a dataset.

    Args:
        dataset (Dataset): Dataset.
        bags_path (str): Path to bags.

    Returns:
        torch.Tensor: Slide embeddings.

    """
    # Initialize Gigapath
    prov_gigapath = GigapathSlideFeatures(**kwargs)

    # Load bags & coords
    bags = dataset.pt_files(bags_path)
    coords = [find_index(b) for b in bags]
    slides = [sf.util.path_to_name(b) for b in bags]

    # Inference
    return prov_gigapath.inference_from_bags(bags, coords), slides
