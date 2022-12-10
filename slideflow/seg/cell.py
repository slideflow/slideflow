import multiprocessing as mp
import numpy as np
import cellpose
import cellpose.models
import slideflow as sf
from PIL import Image, ImageDraw
from slideflow.slide.utils import draw_roi
from tqdm import tqdm
from typing import Tuple, Union
from functools import partial


class Segmentation:
    def __init__(self, masks):
        """Segmentation mask.

        Args:
            masks (np.ndarray): Array of masks, dtype int32, where 0 represents
                non-segmented background, and each segmented mask is represented
                by unique increasing integers.
        """
        self.masks = masks
        self._outlines = None
        self._centroids = None

    @property
    def outlines(self):
        if self._outlines is None:
            self._outlines = cellpose.utils.outlines_list(self.masks)
        return self._outlines

    @property
    def centroids(self):
        if self._centroids is None:
            mask_s = sparse_mask(self.masks)
            self._centroids = get_sparse_centroid(self.masks, mask_s)
        return self._centroids

    def _draw_centroid(self, img, color='green'):
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        for c in self.centroids:
            x, y = np.int32(c[1]), np.int32(c[0])
            draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
        return np.asarray(pil_img)


    def mask_to_image(self, centroid=False, centroid_color='green'):
        img = np.repeat((self.masks.astype(bool).astype(np.uint8) * 255)[:, :, np.newaxis], 3, axis=-1)
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


def get_sparse_centroid(mask, sparse_mask):
    return [np.mean(np.unravel_index(row.data, mask.shape), 1).astype(np.int32)
            for (R, row) in enumerate(sparse_mask) if R>0]


def sparse_mask(mask):
    from scipy.sparse import csr_matrix
    cols = np.arange(mask.size)
    return csr_matrix((cols, (np.ravel(mask), cols)),
                      shape=(mask.max() + 1, mask.size),
                      dtype=np.int64)


def get_masks(c, slide, model, diameter=None, **kwargs):
    x, y = c
    tile = slide[x, y]
    masks, flows, styles, diams = model.eval(tile, channels=[[0, 0]], **kwargs)
    return masks, diams, (x, y)


def segment_slide(
    slide: Union[sf.WSI, str],
    model: Union["cellpose.models.Cellpose", str] = 'cyto2',
    tile_px: int = 64,
    tile_um: Union[int, str] = '40x',
    diameter: int = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment cells in a whole-slide image, returning masks and centroids."""

    if isinstance(model, str):
        model = cellpose.models.Cellpose(model_type=model, gpu=True)

    if isinstance(slide, str):
        slide = sf.WSI(slide, tile_px=tile_px, tile_um=tile_um)
        slide.qc('otsu')
    else:
        tile_px = slide.tile_px

    diam_str = f'{diameter:.2f}' if diameter is not None else 'None'
    print(f"Segmenting slide with diameter {diam_str} (shape={slide.dimensions})")
    running_max = 0
    all_masks = np.zeros((slide.shape[0] * tile_px, slide.shape[1] * tile_px), dtype=np.int32)
    all_diams = []
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(4)

    for masks, diams, (x, y) in tqdm(pool.imap(partial(get_masks, slide=slide, model=model, diameter=diameter, **kwargs),
                                               np.argwhere(slide.grid)),
                                     total=len(np.argwhere(slide.grid))):
        all_diams.append(diams)
        all_masks[x * tile_px: (x+1) * tile_px,
                  y * tile_px: (y+1) * tile_px] = masks
        all_masks[x * tile_px: (x+1) * tile_px,
                  y * tile_px: (y+1) * tile_px][np.nonzero(masks)] += running_max
        running_max += masks.max()

    pool.close()
    seg = Segmentation(np.concatenate(np.concatenate(all_masks, axis=-1), axis=0))
    return seg, all_diams
