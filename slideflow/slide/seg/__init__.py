import multiprocessing as mp
import numpy as np
import cellpose
import cellpose.models
import logging
import slideflow as sf

from matplotlib.colors import to_rgb
from tqdm import tqdm
from typing import Tuple, Union, Callable, Optional, Iterable, TYPE_CHECKING, List
from functools import partial
from PIL import Image, ImageDraw
from cellpose.utils import outlines_list
from slideflow.slide.utils import draw_roi
from slideflow.util import batch

from . import seg_utils

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

# -----------------------------------------------------------------------------

class Segmentation:
    def __init__(
        self,
        masks: np.ndarray,
        *,
        slide: Optional[sf.WSI] = None,
        flows: Optional[np.ndarray] = None,
        styles: Optional[np.ndarray] = None,
        diams: Optional[np.ndarray] = None,
        wsi_dim: Optional[Tuple[int, int]] = None,
        wsi_offset: Optional[Tuple[int, int]] = None
    ):
        """Segmentation mask.

        Args:
            masks (np.ndarray): Array of masks, dtype int32, where 0 represents
                non-segmented background, and each segmented mask is represented
                by unique increasing integers.
        """
        self.slide = slide
        self.masks = masks
        self.flows = flows
        self._outlines = None
        self._centroids = None
        self.wsi_dim = wsi_dim
        self.wsi_offset = wsi_offset

    @classmethod
    def from_npz(cls, path) -> "Segmentation":
        loaded = np.load(path)
        if 'masks' not in loaded:
            raise TypeError(f"Unable to load '{path}'; 'masks' index not found.")
        flows = None if 'flows' not in loaded else loaded['flows']
        obj = cls(slide=None, masks=loaded['masks'], flows=flows)
        obj.wsi_dim = loaded['wsi_dim']
        obj.wsi_offset = loaded['wsi_offset']
        if 'centroids' in loaded:
            obj._centroids = loaded['centroids']
        return obj

    @property
    def outlines(self):
        if self._outlines is None:
            self.calculate_outlines()
        return self._outlines

    @property
    def wsi_ratio(self):
        """Ratio of WSI base dimension to the mask shape."""
        if self.wsi_dim is not None:
            return self.wsi_dim[1] / self.masks.shape[0]
        else:
            return None

    def centroids(self, wsi_dim=False):
        if self._centroids is None:
            self.calculate_centroids()
        if wsi_dim:
            if self.slide is None:
                raise ValueError("Unable to calculate wsi_dim for centroids - "
                                 "slide is not set.")
            if self.wsi_dim is None:
                raise ValueError("Unable to calculate wsi_dim for centroids - "
                                 "wsi_dim is not set.")
            ratio = self.wsi_dim[1] / self.masks.shape[0]
            return ((self._centroids * ratio)[:, ::-1] + self.wsi_offset).astype(np.int32)
        else:
            return self._centroids

    def _draw_centroid(self, img, color='green'):
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        for c in self.centroids():
            x, y = np.int32(c[1]), np.int32(c[0])
            draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
        return np.asarray(pil_img)

    def calculate_centroids(self, force=False):
        if self._centroids is not None and not force:
            return
        mask_s = seg_utils.sparse_mask(self.masks)
        self._centroids = seg_utils.get_sparse_centroid(self.masks, mask_s)

    def calculate_outlines(self, force=False):
        if self._outlines is not None and not force:
            return
        self._outlines = outlines_list(self.masks)

    def centroid_to_image(self, color='green'):
        img = np.zeros((self.masks.shape[0], self.masks.shape[1], 3), dtype=np.uint8)
        return self._draw_centroid(img, color=color)

    def extract_centroids(
        self,
        slide: str,
        tile_px: int = 128,
        wsi_offset: Tuple[int, int] = (0, 0)
    ) -> Callable:
        """Return a generator which extracts tiles from a slide at the given centroids."""
        reader = sf.slide.wsi_reader(slide)
        factor = reader.dimensions[1] / self.masks.shape[0]

        def generator():
            for c in self._centroids:
                cf = c * factor + wsi_offset
                yield reader.read_from_pyramid((cf[1]-(tile_px/2), cf[0]-(tile_px/2)), (tile_px, tile_px), (tile_px, tile_px), convert='numpy', flatten=True)

        return generator

    def mask_to_image(self, centroid=False, color='cyan', centroid_color='green'):
        if isinstance(color, str):
            color = [int(c * 255) for c in to_rgb(color)]
        else:
            assert len(color) == 3
        img = np.zeros((self.masks.shape[0], self.masks.shape[1], 3), dtype=np.uint8)
        img[self.masks > 0] = color
        if centroid:
            return self._draw_centroid(img, color=centroid_color)
        else:
            return img

    def outline_to_image(self, centroid=False, color='red', centroid_color='green'):
        empty = np.zeros((self.masks.shape[0], self.masks.shape[1], 3), dtype=np.uint8)
        img = draw_roi(empty, self.outlines, color=color)
        if centroid:
            return self._draw_centroid(img, color=centroid_color)
        else:
            return img

    def save_npz(self, filename: str):
        save_dict = dict(masks=self.masks)
        if self._centroids is not None:
            save_dict['centroids'] = self._centroids
        if self.flows is not None:
            save_dict['flows'] = self.flows
        if self.wsi_dim is not None:
            save_dict['wsi_dim'] = self.wsi_dim
        if self.wsi_offset is not None:
            save_dict['wsi_offset'] = self.wsi_offset
        np.savez(filename, **save_dict)

# -----------------------------------------------------------------------------

def _mask_worker(c, slide, diameter=None, **kwargs):
    tiles = np.array([slide[x, y] for x, y in c])
    masks, flows, styles, diams = _loaded_model_.eval(tiles, channels=[[0, 0]], tile=False, diameter=diameter, **kwargs)
    return masks, flows, diams, c


def _init_worker(
    model: Union["cellpose.models.Cellpose", str],
    gpus: Optional[Union[int, Iterable[int]]] = None
):
    """Initialize pool worker, including loading model."""
    global _loaded_model_
    if isinstance(model, str):
        if gpus is not None:
            if isinstance(gpus, int):
                gpus = [gpus]
            import torch
            _id = mp.current_process()._identity
            proc = 0 if not len(_id) else _id[0]

            device = torch.device(f'cuda:{gpus[proc % len(gpus)]}')
            _loaded_model_ = cellpose.models.Cellpose(model_type=model, gpu=True, device=device)
        else:
            _loaded_model_ = cellpose.models.Cellpose(model_type=model, gpu=True)
    else:
        _loaded_model_ = model


def segment_slide(
    slide: Union[sf.WSI, str],
    model: Union["cellpose.models.Cellpose", str] = 'cyto2',
    tile_px: int = 256,
    tile_um: Union[int, str] = '40x',
    diameter: int = None,
    batch_size: int = 8,
    gpus: Optional[Union[int, Iterable[int]]] = (0,),
    num_workers: Optional[int] = None,
    pb: Optional["Progress"] = None,
    pb_tasks: Optional[List["TaskID"]] = None,
    show_progress: bool = True,
    **kwargs
) -> Segmentation:
    """Segment cells in a whole-slide image, returning masks and centroids."""

    # Quiet the logger to suppress warnings of empty masks
    logging.getLogger('cellpose').setLevel(40)

    if isinstance(slide, str):
        slide = sf.WSI(slide, tile_px=tile_px, tile_um=tile_um)
        slide.qc('otsu')
    else:
        tile_px = slide.tile_px

    # Workers and pool
    if num_workers is None and isinstance(gpus, (list, tuple)):
        num_workers = 2 * len(gpus)
    elif num_workers is None:
        num_workers = 2
    if num_workers > 0:
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(num_workers, initializer=partial(_init_worker, model=model, gpus=gpus))
    else:
        pool = mp.dummy.Pool(4, initializer=partial(_init_worker, model=model, gpus=gpus))

    diam_str = f'{diameter:.2f}' if diameter is not None else 'None'
    print(f"Segmenting slide with diameter {diam_str} (shape={slide.dimensions})")
    running_max = 0
    all_masks = np.zeros((slide.shape[0], slide.shape[1], tile_px, tile_px), dtype=np.int32)
    all_flows = np.zeros((slide.shape[0], slide.shape[1], tile_px, tile_px, 3), dtype=np.uint8)
    all_diams = []

    tile_idx = np.argwhere(slide.grid)
    mapped = pool.imap(
        partial(_mask_worker, slide=slide, diameter=diameter, batch_size=batch_size, **kwargs),
        batch(tile_idx, batch_size * 8)) # batch_size * 8

    if show_progress:
        mapped = tqdm(mapped, total=int(len(tile_idx) / (batch_size * 8)))

    for masks, flows, diams, c in mapped:
        all_diams.append(diams)
        for i in range(len(masks)):
            x = c[i][0]
            y = c[i][1]
            all_masks[x, y] = masks[i]
            all_masks[x, y][np.nonzero(masks[i])] += running_max
            running_max += masks[i].max()
            all_flows[x, y] = flows[0][i]
        # Update progress bars
        if pb is not None and pb_tasks:
            for task in pb_tasks:
                pb.advance(task, batch_size * 8)

    pool.close()
    masks = np.concatenate(np.concatenate(all_masks, axis=-1), axis=0)
    flows = np.concatenate(np.concatenate(all_flows, axis=-2), axis=0)
    return Segmentation(slide=slide, masks=masks, flows=flows, diams=all_diams)
