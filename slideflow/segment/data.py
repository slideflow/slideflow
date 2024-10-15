import torch
import slideflow as sf
import rasterio
import numpy as np
import shapely.affinity as sa

from typing import Tuple, Union, Optional, List, Dict
from torchvision import transforms
from os.path import join, exists
from rich.progress import track
from shapely.ops import unary_union
from shapely.geometry import Polygon
from shapely.ops import unary_union
from slideflow.util import path_to_name

from .utils import topleft_pad_torch

# -----------------------------------------------------------------------------

class ThumbMaskDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset: "sf.Dataset",
        mpp: float,
        roi_labels: List[str],
        *,
        mode: str = 'binary',
    ) -> None:
        """Dataset that generates thumbnails and ROI masks.

        Args:
            dataset (sf.Dataset): The dataset to use.
            mpp (float): The target microns per pixel. The thumbnail will be
                scaled to this resolution.
            roi_labels (List[str]): The ROI labels to include in the mask.

        Keyword args:
            mode (str, optional): The mode to use for the mask. One of:
                'binary', 'multiclass', 'multilabel'. Defaults to 'binary'.

        """
        super().__init__()
        self.mpp = mpp
        self.mode = mode
        self.roi_labels = roi_labels

        # Subsample dataset to only include slides with ROIs.
        self.rois = dataset.rois()
        slides = set(map(path_to_name, dataset.slide_paths()))
        slides = slides.intersection(set(map(path_to_name, self.rois)))
        dataset = dataset.filter({'slide': list(slides)})

        # Prepare WSI objects (for slides with ROIs).
        self.paths = dataset.slide_paths()

    def __len__(self) -> int:
        return len(self.paths)

    def process(
        self,
        img: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process the image/mask and convert to a tensor."""
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        return img, mask

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        # Load the image and mask.
        path = self.paths[index]
        wsi = sf.WSI(path, 299, 512, rois=self.rois, roi_filter_method=0.1, verbose=False)
        output = get_thumb_and_mask(wsi, self.mpp, self.roi_labels, skip_missing=False)
        if output is None:
            return None
        img = output['image']               # CHW (np.ndarray)
        mask = output['mask'].astype(int)   # 1HW (np.ndarray)

        if self.mode == 'multiclass':
            mask = mask * np.arange(1, mask.shape[0]+1)[:, None, None]
            mask = mask.max(axis=0)
        elif self.mode == 'binary' and mask.ndim == 3:
            mask = np.any(mask, axis=0)[None, :, :].astype(int)

        # Process.
        img, mask = self.process(img, mask)

        return {
            'image': img,
            'mask': mask
        }


class RandomCropDataset(ThumbMaskDataset):

    def __init__(self, *args, size: int = 1024, **kwargs):
        """Dataset that generates thumbnails & ROI masks, with random crops.

        Thumbnails and masks and randomly cropped and rotated together to
        a square size of `size` pixels.

        Args:
            dataset (sf.Dataset): The dataset to use.
            mpp (float): The target microns per pixel. The thumbnail will be
                scaled to this resolution.
            roi_labels (List[str]): The ROI labels to include in the mask.
            size (int, optional): The size of the random crop. Defaults to 1024.

        Keyword Args:
            mode (str, optional): The mode to use for the mask. One of:
                'binary', 'multiclass', 'multilabel'. Defaults to 'binary'.

        """
        super().__init__(*args, **kwargs)
        self.size = size

    def process(self, img, mask):
        """Randomly crop/rotate the image and mask and convert to a tensor."""
        return random_crop_and_rotate(img, mask, size=self.size)

# -----------------------------------------------------------------------------
# Buffered datasets

class BufferedMaskDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: "sf.Dataset", source: str, *, mode: str = 'binary'):
        """Dataset that loads buffered image and mask pairs.

        Args:
            dataset (sf.Dataset): The dataset to use.
            source (str): The directory containing the buffered image/mask pairs.

        Keyword Args:
            mode (str, optional): The mode to use for the mask. One of:
                'binary', 'multiclass', 'multilabel'. Defaults to 'binary'.

        """
        super().__init__()
        if mode not in ['binary', 'multiclass', 'multilabel']:
            raise ValueError("Invalid mode: {}. Expected one of: binary, "
                             "multiclass, multilabel".format(mode))
        self.dataset = dataset
        self.mode = mode
        self.paths = [
            join(source, s + '.pt') for s in dataset.slides()
            if exists(join(source, s + '.pt'))
        ]


    def __len__(self) -> int:
        return len(self.paths)

    def process(
        self,
        img: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process the image/mask and convert to a tensor."""
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        return img, mask

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load the image and mask.
        output = torch.load(self.paths[index])
        img = output['image']               # CHW (np.ndarray)
        mask = output['mask'].astype(int)   # 1HW (np.ndarray)

        if self.mode == 'multiclass':
            mask = mask * np.arange(1, mask.shape[0]+1)[:, None, None]
            mask = mask.max(axis=0)
        elif self.mode == 'binary' and mask.ndim == 3:
            mask = np.any(mask, axis=0)[None, :, :].astype(int)

        # Process.
        img, mask = self.process(img, mask)

        return {
            'image': img,
            'mask': mask
        }


class BufferedRandomCropDataset(BufferedMaskDataset):

    def __init__(self, *args, size: int = 1024, **kwargs):
        """Dataset that loads buffered image/mask pairs and randomly crops.

        Loaded thumbnails and masks and randomly cropped and rotated together to
        a square size of `size` pixels.

        Args:
            dataset (sf.Dataset): The dataset to use.
            source (str): The directory containing the buffered image/mask pairs.
            size (int, optional): The size of the random crop. Defaults to 1024.

        Keyword Args:
            mode (str, optional): The mode to use for the mask. One of:
                'binary', 'multiclass', 'multilabel'. Defaults to 'binary'.

        """
        super().__init__(*args, **kwargs)
        self.size = size

    def process(
        self,
        img: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly crop/rotate the image and mask and convert to a tensor."""
        return random_crop_and_rotate(img, mask, size=self.size)

# -----------------------------------------------------------------------------

class TileMaskDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset: "sf.Dataset",
        tile_px: int,
        tile_um: Union[int, str],
        *,
        roi_labels: Optional[List[str]] = None,
        stride_div: int = 2,
        crop_margin: int = 0,
        filter_method: str = 'otsu',
        mode: str = 'binary'
    ):
        """Dataset that generates tiles and ROI masks from slides.

        Args:
            dataset (sf.Dataset): The dataset to use.
            tile_px (int): The size of the tiles (in pixels).
            tile_um (Union[int, str]): The size of the tiles (in microns).

        Keyword args:
            stride_div (int, optional): The divisor for the stride.
                Defaults to 2.
            crop_margin (int, optional): The number of pixels to add to the
                tile size before random cropping to the target tile_px size.
                Defaults to 0 (no random cropping).
            filter_method (str, optional): The method to use for identifying
                tiles for training. If 'roi', selects only tiles that intersect
                with an ROI. If 'otsu', selects tiles based on an Otsu threshold
                of the slide. Defaults to 'roi'.

        """
        super().__init__()

        rois = dataset.rois()
        slides_with_rois = [path_to_name(r) for r in rois]
        slides = [s for s in dataset.slide_paths()
                  if path_to_name(s) in slides_with_rois]
        kw = dict(
            tile_px=tile_px + crop_margin,
            tile_um=tile_um,
            verbose=False,
            stride_div=stride_div
        )
        if roi_labels is None:
            roi_labels = []
        self.mode = mode
        self.roi_labels = roi_labels
        self.tile_px = tile_px
        self.coords = []
        self.all_wsi = dict()
        self.all_wsi_with_roi = dict()
        self.all_extract_px = dict()
        for slide in track(slides, description="Loading slides"):
            name = path_to_name(slide)
            wsi = sf.WSI(slide, **kw)
            try:
                wsi_with_rois = sf.WSI(slide, roi_filter_method=0.1, rois=rois, **kw)
            except Exception as e:
                sf.log.error("Failed to load slide {}: {}".format(slide, e))
                raise e

            # Filter ROIs to only include the specified labels.
            if self.roi_labels:
                wsi_with_rois.rois = [roi for roi in wsi_with_rois.rois if roi.label in self.roi_labels]
                wsi_with_rois.process_rois()

            if not len(wsi_with_rois.rois):
                continue
            if filter_method == 'roi':
                wsi_inner = sf.WSI(slide, roi_filter_method=0.9, rois=rois, **kw)
                if self.roi_labels:
                    wsi_inner.rois = [roi for roi in wsi_with_rois.rois if roi.label in self.roi_labels]
                    wsi_inner.process_rois()
                coords = np.argwhere(wsi_with_rois.grid & (~wsi_inner.grid)).tolist()
            elif filter_method == 'otsu':
                wsi.qc('otsu')
                coords = np.argwhere(wsi.grid).tolist()
                wsi.remove_qc()
            elif filter_method in ['all', 'none', None]:
                coords = np.argwhere(wsi_with_rois.grid).tolist()
            else:
                raise ValueError("Invalid filter method: {}. Expected one of: "
                                 "roi, otsu".format(filter_method))
            for c in coords:
                self.coords.append([name] + c)
            self.all_wsi[name] = wsi
            self.all_wsi_with_roi[name] = wsi_with_rois
            self.all_extract_px[name] = int(wsi.tile_um / wsi.mpp)

    def __len__(self):
        return len(self.coords)

    def get_scaled_and_intersecting_polys(
        self,
        polys: "Polygon",
        tile: "Polygon",
        scale: float,
        full_stride: int,
        grid_idx: Tuple[int, int]
    ):
        """Get scaled and intersecting polygons for a given tile."""
        gx, gy = grid_idx
        A = polys.intersection(tile)

        # Translate polygons so the intersection origin is at (0, 0)
        B = sa.translate(A, -(full_stride*gx), -(full_stride*gy))

        # Scale to the target tile size
        C = sa.scale(B, xfact=scale, yfact=scale, origin=(0, 0))
        return C

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get an image and mask for a given index."""
        slide, gx, gy = self.coords[index]
        wsi = self.all_wsi[slide]
        wsi_with_roi = self.all_wsi_with_roi[slide]
        fe = self.all_extract_px[slide]
        fs = wsi.full_stride
        scale = wsi.tile_px / fe

        # Get the image.
        img = wsi[gx, gy].transpose(2, 0, 1)

        # Get a polygon for the tile, used for determining overlapping ROIs.
        tile = Polygon([
            [fs*gx, fs*gy],
            [fs*gx, (fs*gy)+fe],
            [(fs*gx)+fe, (fs*gy)+fe],
            [(fs*gx)+fe, fs*gy]
        ])

        # Compute the mask from ROIs.
        if len(wsi_with_roi.rois) == 0:
            if self.roi_labels:
                mask = np.zeros((len(self.roi_labels), wsi.tile_px, wsi.tile_px), dtype=int)
            else:
                mask = np.zeros((1, wsi.tile_px, wsi.tile_px), dtype=int)

        # Handle ROIs with labels (multilabel or multiclass)
        elif self.roi_labels:
            labeled_masks = []
            for i, label in enumerate(self.roi_labels):
                wsi_polys = [p.poly for p in wsi_with_roi.rois if p.label == label]
                if len(wsi_polys) == 0:
                    mask = np.zeros((wsi.tile_px, wsi.tile_px), dtype=int)
                    labeled_masks.append(mask)
                else:
                    all_polys = unary_union(wsi_polys)
                    polys = self.get_scaled_and_intersecting_polys(
                        all_polys, tile, scale, fs, (gx, gy)
                    )
                    if isinstance(polys, Polygon) and polys.is_empty:
                        mask = np.zeros((wsi.tile_px, wsi.tile_px), dtype=int)
                    else:
                        # Rasterize to an int mask.
                        mask = rasterio.features.rasterize([polys], out_shape=[wsi.tile_px, wsi.tile_px]).astype(int)
                    labeled_masks.append(mask)
            mask = np.stack(labeled_masks, axis=0)

        # Handle ROIs without labels (binary)
        else:
            # Determine the intersection at the given tile location.
            all_polys = unary_union([p.poly for p in wsi_with_roi.rois])
            polys = self.get_scaled_and_intersecting_polys(
                all_polys, tile, scale, fs, (gx, gy)
            )

            if isinstance(polys, Polygon) and polys.is_empty:
                mask = np.zeros((wsi.tile_px, wsi.tile_px), dtype=int)
            else:
                # Rasterize to an int mask.
                try:
                    mask = rasterio.features.rasterize([polys], out_shape=[wsi.tile_px, wsi.tile_px]).astype(bool).astype(np.int32)
                except ValueError:
                    mask = np.zeros((wsi.tile_px, wsi.tile_px), dtype=int)

            # Add a dummy channel dimension.
            mask = mask[None, :, :]

        # Process according to the mode.
        if self.mode == 'multiclass':
            mask = mask * np.arange(1, mask.shape[0]+1)[:, None, None]
            mask = mask.max(axis=0)
        elif self.mode == 'binary' and mask.ndim == 3:
            mask = np.any(mask, axis=0)[None, :, :].astype(int)

        # Process.
        img, mask = self.process(img, mask)

        return {
            'image': img,
            'mask': mask
        }

    def process(
        self,
        img: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly crop/rotate the image and mask and convert to a tensor."""
        return random_crop_and_rotate(img, mask, size=self.tile_px)

# -----------------------------------------------------------------------------

def random_crop_and_rotate(img, mask, size):
    if mask.ndim == 2:
        to_squeeze = True
        mask = mask[None, :, :]
    else:
        to_squeeze = False

    # Convert to tensor.
    img = torch.from_numpy(img).permute(1, 2, 0)
    mask = torch.from_numpy(mask).permute(1, 2, 0)

    # Pad to target size.
    img = topleft_pad_torch(img, size).permute(2, 0, 1)
    mask = topleft_pad_torch(mask, size).permute(2, 0, 1)

    # Random crop.
    i, j, h, w = transforms.RandomCrop.get_params(
        img, output_size=(size, size))
    img = transforms.functional.crop(img, i, j, h, w)
    mask = transforms.functional.crop(mask, i, j, h, w)

    # Random flip.
    if np.random.rand() > 0.5:
        img = transforms.functional.hflip(img)
        mask = transforms.functional.hflip(mask)
    if np.random.rand() > 0.5:
        img = transforms.functional.vflip(img)
        mask = transforms.functional.vflip(mask)

    # Random cardinal rotation.
    r = np.random.randint(4)
    img = transforms.functional.rotate(img, r * 90)
    mask = transforms.functional.rotate(mask, r * 90)

    if to_squeeze:
        mask = mask.squeeze(0)

    return img, mask

# -----------------------------------------------------------------------------

def get_thumb_and_mask(
    wsi: "sf.WSI",
    mpp: float,
    roi_labels: Optional[List[str]] = None,
    skip_missing: bool = False
) -> Dict[str, np.ndarray]:
    """Get a thumbnail and segmentation mask for a slide."""

    if len(wsi.rois) == 0 and skip_missing:
        return None

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
    thumb = wsi.thumb(mpp=mpp).convert('RGB')
    img = np.array(thumb).transpose(2, 0, 1)
    xfact = thumb.size[1] / wsi.dimensions[1]
    yfact = thumb.size[0] / wsi.dimensions[0]

    if len(wsi.rois) == 0:
        if roi_labels:
            mask = np.zeros((len(roi_labels), thumb.size[1], thumb.size[0])).astype(bool)
        else:
            mask = np.zeros((1, thumb.size[1], thumb.size[0])).astype(bool)
    elif roi_labels:
        labeled_masks = []
        for i, label in enumerate(roi_labels):
            wsi_polys = [p.poly for p in wsi.rois if p.label == label]
            if len(wsi_polys) == 0:
                mask = np.zeros((thumb.size[1], thumb.size[0])).astype(bool)
                labeled_masks.append(mask)
            else:
                all_polys = unary_union(wsi_polys)
                # Scale ROIs to the thumbnail size.
                C = sa.scale(all_polys, xfact=xfact, yfact=yfact, origin=(0, 0))
                # Rasterize to an int mask.
                mask = rasterio.features.rasterize([C], out_shape=(thumb.size[1], thumb.size[0])).astype(bool).astype(np.int32)
                labeled_masks.append(mask)
        mask = np.stack(labeled_masks, axis=0)

    else:
        all_polys = unary_union([p.poly for p in wsi.rois])
        # Scale ROIs to the thumbnail size.
        C = sa.scale(all_polys, xfact=xfact, yfact=yfact, origin=(0, 0))
        # Rasterize to an int mask.
        mask = rasterio.features.rasterize([C], out_shape=(thumb.size[1], thumb.size[0])).astype(bool)
        # Add a dummy channel dimension.
        mask = mask[None, :, :]

    assert img.shape[1:] == mask.shape[1:], "Image and mask must have the same dimensions."
    assert mask.ndim == 3, "Mask must have 3 dimensions (C, H, W)."
    assert img.ndim == 3, "Image must have 3 dimensions (C, H, W)."

    return {
        'image': img,
        'mask': mask
    }
