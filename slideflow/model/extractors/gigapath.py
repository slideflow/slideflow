import os
import slideflow as sf
import timm
import torch
import numpy as np
from os.path import join, exists, isdir
from tqdm import tqdm
from torchvision import transforms
from typing import Optional, List, Tuple, Dict, Union, TYPE_CHECKING
from slideflow.util.tfrecord2idx import find_index

from ._factory_torch import TorchFeatureExtractor

if TYPE_CHECKING:
    from slideflow import WSI

# -----------------------------------------------------------------------------

class GigapathTileFeatures(TorchFeatureExtractor):
    """Gigapath pretrained feature extractor.

    This class is used to extract tile-level features from Gigapath.

    Feature dimensions: 1536

    Manuscript: https://aka.ms/gigapath

    Hugging Face: https://huggingface.co/prov-gigapath/prov-gigapath

    """
    tag = 'gigapath.tile'
    license = """License available at https://github.com/prov-gigapath/prov-gigapath"""
    citation = """
@article{xu2024gigapath,
  title={A whole-slide foundation model for digital pathology from real-world data},
  author={Xu, Hanwen and Usuyama, Naoto and Bagga, Jaspreet and Zhang, Sheng and Rao, Rajesh and Naumann, Tristan and Wong, Cliff and Gero, Zelalem and González, Javier and Gu, Yu and Xu, Yanbo and Wei, Mu and Wang, Wenhui and Ma, Shuming and Wei, Furu and Yang, Jianwei and Li, Chunyuan and Gao, Jianfeng and Rosemon, Jaylen and Bower, Tucker and Lee, Soohee and Weerasinghe, Roshanthi and Wright, Bill J. and Robicsek, Ari and Piening, Brian and Bifulco, Carlo and Wang, Sheng and Poon, Hoifung},
  journal={Nature},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
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
    """Gigapath slide-level feature embedding generator.

    This class is used to generate slide-level embeddings from tile-level embeddings.

    """

    tag = 'gigapath.slide'

    def __init__(
        self,
        weights: Optional[str] = None,
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

        # Build the encoder.
        if weights is None:
            weights = "hf_hub:prov-gigapath/prov-gigapath"
        self.slide_encoder = slide_encoder.create_model(
            weights,
            "gigapath_slide_enc12l768d",
            1536,
            global_pool=global_pool
        ).to(self.device)
        self.slide_encoder.eval()

    def __call__(self, tile_embed: torch.Tensor, coords: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate whole-slide feature embedding.

        Args:
            tile_embed (torch.Tensor): Tile embeddings. Shape: (1, n_tiles, 1536).
            coords (torch.Tensor): Tile coordinates. Shape: (1, n_tiles, 2).

        Keyword Args:
            all_layer_embed (bool): Return embeddings from all layers.
                Defaults to True.

        Returns:
            torch.Tensor: Whole-slide embedding.

        """
        return self.run_inference(tile_embed, coords, **kwargs)

    def run_inference(
        self,
        tile_embed: torch.Tensor,
        coords: torch.Tensor,
        *,
        all_layer_embed: bool = True
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Run inference on a single slide.

        Args:
            tile_embed (torch.Tensor): Tile embeddings. Shape: (1, n_tiles, 1536).
            coords (torch.Tensor): Tile coordinates. Shape: (1, n_tiles, 2).

        Keyword Args:
            all_layer_embed (bool): Return embeddings from all layers.
                Defaults to True.

        Returns:
            dict or torch.Tensor: Whole-slide embeddings. If `all_layer_embed` is True,
                returns a dict mapping layer names to embeddings from each layer.
                Otherwise, returns only the last layer embedding.

        """
        from slideflow.model.torch import autocast

        with autocast(self.device.type, torch.float16), torch.inference_mode():
            embeds = self.slide_encoder(tile_embed, coords, all_layer_embed=all_layer_embed)
            if all_layer_embed:
                return {
                    (f"layer_{i}_embed" if i < (len(embeds) - 1) else "last_layer_embed"): embed
                    for i, embed in enumerate(embeds)
                }
            else:
                return embeds[0]

    def inference_from_bags(
        self,
        bags: List[str],
        *,
        indices: Optional[List[str]] = None,
        all_layer_embed: bool = True
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Run inference on a set of bags.

        Args:
            bags (List[str]): List of paths to bags (tile embeddings).
            indices (List[str]): List of paths to bag indices (tile coordinates).
                If not provided, the indices will be inferred from the bags.

        Keyword Args:
            all_layer_embed (bool): Return embeddings from all layers.
                Defaults to True.

        Returns:
            torch.Tensor: Whole-slide embeddings.

        """
        if indices is None:
            indices = [find_index(b) for b in bags]
        slide_embeds = {} if all_layer_embed else []
        for bag, index in tqdm(zip(bags, indices), total=len(bags)):
            tile_embed = torch.load(bag).unsqueeze(0).to(self.device)
            coords = torch.from_numpy(np.load(index)['arr_0']).unsqueeze(0).to(self.device)
            slide_embed = self.run_inference(tile_embed, coords, all_layer_embed=all_layer_embed)
            if all_layer_embed:
                for layer_name, embedding in slide_embed.keys():
                    if layer_name not in slide_embeds:
                        slide_embeds[layer_name] = []
                    slide_embeds[layer_name].append(embedding)
            else:
                slide_embeds.append(slide_embed)
        if all_layer_embed:
            return {k: torch.cat(v, dim=0) for k, v in slide_embeds.items()}
        else:
            return torch.cat(slide_embeds, dim=0)

    def generate_and_save(
        self,
        bags: Union[str, List[str]],
        outdir: str,
        *,
        indices: Optional[List[str]] = None,
        all_layer_embed: bool = True,
        overwrite: bool = False
    ) -> None:
        """Generate and save whole-slide embeddings from bags.

        Args:
            bags (List[str]): List of paths to bags (tile embeddings).
            outdir (str): Output directory.

        Keyword Args:
            indices (List[str]): List of paths to bag indices (tile coordinates).
                If not provided, the indices will be inferred from the bags.
            all_layer_embed (bool): Return embeddings from all layers.
                Defaults to True.
            overwrite (bool): Overwrite existing embeddings. Defaults to False.

        """
        # Interpret the bags argument.
        if isinstance(bags, str):
            if exists(bags) and isdir(bags):
                bags = [join(bags, f) for f in os.listdir(bags) if f.endswith('.pt')]
            elif exists(bags):
                bags = [bags]
            else:
                raise ValueError("Invalid bags path: {}".format(bags))

        # Find indices for the bags (containing tile coordinates).
        if indices is None:
            indices = [find_index(b) for b in bags]

        if not exists(outdir):
            os.makedirs(outdir)

        # Run inference.
        complete = 0
        for bag, index in tqdm(zip(bags, indices), total=len(bags)):
            filename = join(outdir, sf.util.path_to_name(bag) + '.pt')
            if exists(filename):
                if overwrite:
                    sf.log.debug("Overwriting slide embedding at {}".format(filename))
                else:
                    sf.log.debug("Slide embedding already exists at {}".format(filename))
                    continue
            tile_embed = torch.load(bag).unsqueeze(0).to(self.device)
            coords = torch.from_numpy(np.load(index)['arr_0']).unsqueeze(0).to(self.device)
            slide_embed = self.run_inference(tile_embed, coords, all_layer_embed=all_layer_embed)
            torch.save(slide_embed, filename)
            complete += 1

        sf.log.info("Generated and saved {} slide embeddings ({} bags skipped)".format(complete, len(bags) - complete))


# -----------------------------------------------------------------------------

class GigapathFeatures:
    """Gigapath whole-slide feature embedding generator.

    This class is used to generate whole-slide embeddings from a whole-slide image.

    """

    tag = 'gigapath'

    def __init__(
        self,
        tile_encoder_weights: Optional[str] = None,
        slide_encoder_weights: Optional[str] = None,
        *,
        global_pool: bool = False,
        device: str = 'cuda',
        **kwargs
    ):
        """Initialize the Gigapath slide feature generator.

        Args:
            tile_encoder_weights (Optional[str]): Path to tile encoder weights.
                If not provided, the model will be loaded from the Hugging Face model hub.
                Defaults to None.
            slide_encoder_weights (Optional[str]): Path to slide encoder weights.
                If not provided, the model will be loaded from the Hugging Face model hub.
                Defaults to None.
            global_pool (bool): Use global pooling. Defaults to False.
            device (str): Device to use for inference. Defaults to 'cuda'.

        """
        from slideflow.model import torch_utils

        self.device = torch_utils.get_device(device)
        self.tile = GigapathTileFeatures(
            weights=tile_encoder_weights,
            device=self.device,
            **kwargs
        )
        self.slide = GigapathSlideFeatures(
            slide_encoder_weights,
            global_pool=global_pool,
            device=self.device
        )

    @property
    def tile_encoder(self):
        """Tile encoder model."""
        return self.tile

    @property
    def slide_encoder(self):
        """Slide encoder model."""
        return self.slide

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

    def __call__(self, wsi: "WSI", **kwargs) -> torch.Tensor:
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
        return self.slide_encoder.run_inference(tile_embeds, coords, **kwargs)

