import torch
import slideflow as sf
import rasterio
import numpy as np
import shapely.affinity as sa

from typing import Tuple, Union
from torchvision import transforms
from os.path import join, exists
from rich.progress import track
from shapely.ops import unary_union
from shapely.geometry import Polygon
from shapely.ops import unary_union
from slideflow.util import path_to_name

from .utils import topleft_pad

# -----------------------------------------------------------------------------

def get_thumb_and_mask(wsi: "sf.WSI", mpp: float):
    """Get a thumbnail and segmentation mask for a slide."""

    # Sanity check.
    width = int((wsi.mpp * wsi.dimensions[0]) / mpp)
    ds = wsi.dimensions[0] / width
    level = wsi.slide.best_level_for_downsample(ds)
    level_dim = wsi.slide.level_dimensions[level]
    if any([d > 10000 for d in level_dim]):
        sf.log.warning("Large thumbnail found ({}) at level={} for {}".format(
            level_dim, level, wsi.path
        ))

    # Get the thumbnail.
    thumb = wsi.thumb(mpp=mpp)
    img = np.array(thumb).transpose(2, 0, 1)

    assert len(wsi.roi_polys) > 0, "No ROIs found in slide"
    all_polys = unary_union([p.poly for p in wsi.roi_polys])

    xfact = thumb.size[1] / wsi.dimensions[1]
    yfact = thumb.size[0] / wsi.dimensions[0]

    # Scale ROIs to the thumbnail size.
    C = sa.scale(all_polys, xfact=xfact, yfact=yfact, origin=(0, 0))

    # Rasterize to an int mask.
    mask = rasterio.features.rasterize([C], out_shape=(thumb.size[1], thumb.size[0])).astype(int)

    # Add a dummy channel dimension.
    mask = mask[None, :, :]

    return {
        'image': img,
        'mask': mask
    }

# -----------------------------------------------------------------------------


class BufferedMaskDataset(torch.utils.data.Dataset):
    """Dataset that loads buffered image and mask pairs."""

    def __init__(self, dataset: "sf.Dataset", source: str):
        super().__init__()
        self.dataset = dataset
        self.paths = [
            join(source, s + '.pt') for s in dataset.slides()
            if exists(join(source, s + '.pt'))
        ]
        

    def __len__(self):
        return len(self.paths)
    
    def process(self, img, mask):
        """Process the image/mask and convert to a tensor."""
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        return img, mask

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load the image and mask.
        output = torch.load(self.paths[index])
        img = output['image']  # CHW (np.ndarray)
        mask = output['mask']  # 1HW (np.ndarray)
        
        # Process.
        img, mask = self.process(img, mask)

        return {
            'image': img,
            'mask': mask
        }
    

class BufferedRandomCropDataset(BufferedMaskDataset):

    def __init__(self, dataset: "sf.Dataset", source: str, size: int = 1024):
        super().__init__(dataset, source)
        self.size = size

    def process(self, img, mask):

        # Pad to target size.
        img = topleft_pad(img, self.size)
        mask = topleft_pad(mask, self.size)

        # Convert to tensor.
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        # Random crop.
        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=(self.size, self.size))
        img = transforms.functional.crop(img, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)
        return img, mask

# -----------------------------------------------------------------------------

class TileMaskDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset: "sf.Dataset",
        tile_px: int,
        tile_um: Union[int, str],
    ):
        super().__init__()

        rois = dataset.rois()
        slides_with_rois = [path_to_name(r) for r in rois]
        slides = [s for s in dataset.slide_paths()
                  if path_to_name(s) in slides_with_rois]
        kw = dict(
            tile_px=tile_px,
            tile_um=tile_um,
            rois=rois,
            verbose=False
        )
        self.coords = []
        self.all_wsi = dict()
        self.all_extract_px = dict()
        for slide in track(slides, description="Loading slides"):
            name = path_to_name(slide)
            wsi = sf.WSI(slide, roi_filter_method=0.1, **kw)
            if not len(wsi.roi_polys):
                continue
            wsi_inner = sf.WSI(slide, roi_filter_method=0.9, **kw)
            coords = np.argwhere(wsi.grid & (~wsi_inner.grid)).tolist()
            for c in coords:
                self.coords.append([name] + c)
            self.all_wsi[name] = wsi
            self.all_extract_px[name] = int(wsi.tile_um / wsi.mpp)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        slide, gx, gy = self.coords[index]
        wsi = self.all_wsi[slide]
        fe = self.all_extract_px[slide]

        # Get the image.
        img = wsi[gx, gy].transpose(2, 0, 1)

        # Determine the intersection at the given tile location.
        tile = Polygon([
            [fe*gx, fe*gy],
            [fe*gx, (fe*gy)+fe],
            [fe*(gx+1), fe*(gy+1)],
            [fe*(gx+1), fe*gy]
        ])
        assert len(wsi.roi_polys) > 0, "No ROIs found in slide"
        all_polys = unary_union([p.poly for p in wsi.roi_polys])
        A = all_polys.intersection(tile)

        # Translate polygons so the intersection origin is at (0, 0)
        B = sa.translate(A, -(fe*gx), -(fe*gy))

        # Scale to the target tile size
        xscale = yscale = wsi.tile_px / fe
        C = sa.scale(B, xfact=xscale, yfact=yscale, origin=(0, 0))
        if isinstance(C, Polygon) and not len(C.exterior.coords.xy[0]):
            mask = np.zeros((wsi.tile_px, wsi.tile_px), dtype=int)
        else:
            # Rasterize to an int mask.
            try:
                mask = rasterio.features.rasterize([C], out_shape=[wsi.tile_px, wsi.tile_px]).astype(int)
            except ValueError:
                mask = np.zeros((wsi.tile_px, wsi.tile_px), dtype=int)

        # Add a dummy channel dimension.
        mask = mask[None, :, :]

        return {
            'image': img,
            'mask': mask
        }