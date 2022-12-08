import multiprocessing as mp
import numpy as np
import cellpose
import cellpose.models
import slideflow as sf
from tqdm import tqdm
from typing import Tuple, Union
from functools import partial


def get_masks(c, slide, model, **kwargs):
    x, y = c
    tile = slide[x, y]
    masks, flows, styles, diams = model.eval(tile, channels=[[0, 0]], **kwargs)
    return masks, diams, (x, y)


def segment_slide(
    slide: Union[sf.WSI, str],
    model: Union["cellpose.models.Cellpose", str] = 'cyto2',
    tile_px=64,
    tile_um='40x',
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

    running_max = 0
    all_masks = np.zeros((slide.shape[0], slide.shape[1], tile_px, tile_px), dtype=np.int32)
    all_diams = []
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(8)

    for masks, diams, (x, y) in tqdm(pool.imap(partial(get_masks, slide=slide, model=model, **kwargs), np.argwhere(slide.grid)), total=len(np.argwhere(slide.grid))):
        all_diams.append(diams)
        masks[np.nonzero(masks)] += running_max
        all_masks[x, y] = masks
        running_max += masks.max()
    return np.concatenate(np.concatenate(all_masks, axis=-1), axis=0), all_diams
