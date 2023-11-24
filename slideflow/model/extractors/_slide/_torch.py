"""PyTorch-based feature extraction from whole-slide images."""

import slideflow as sf
import numpy as np
import torch
import torchvision

from typing import Optional, Callable, Union, TYPE_CHECKING
from slideflow import log

from ._utils import _build_grid, _log_normalizer, _use_numpy_if_png

if TYPE_CHECKING:
    from slideflow.model.base import BaseFeatureExtractor
    from slideflow.norm import StainNormalizer

# -----------------------------------------------------------------------------

class _SlideIterator(torch.utils.data.IterableDataset):
    def __init__(self, img_format, generator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_format = img_format
        self.generator = generator

    def __iter__(self):
        for image_dict in self.generator():
            img = image_dict['image']
            if self.img_format not in ('numpy', 'png'):
                np_data = torch.from_numpy(
                    np.fromstring(img, dtype=np.uint8))
                img = torchvision.io.decode_image(np_data)
            else:
                img = torch.from_numpy(img).permute(2, 0, 1)
            loc = np.array(image_dict['grid'])
            yield img, loc

# -----------------------------------------------------------------------------

def features_from_slide_torch(
    extractor: "BaseFeatureExtractor",
    slide: "sf.WSI",
    *,
    img_format: str = 'numpy',
    batch_size: int = 32,
    dtype: type = np.float16,
    grid: Optional[np.ndarray] = None,
    shuffle: bool = False,
    show_progress: bool = True,
    callback: Optional[Callable] = None,
    normalizer: Optional[Union[str, "StainNormalizer"]] = None,
    preprocess_fn: Optional[Callable] = None,
    **kwargs
) -> Optional[np.ndarray]:

    log.debug(f"Slide prediction (batch_size={batch_size}, "
              f"img_format={img_format})")

    img_format = _use_numpy_if_png(img_format)

    # Create the output array
    features_grid = _build_grid(extractor, slide, grid=grid, dtype=dtype)

    _log_normalizer(normalizer)
    opencv_norm = (isinstance(normalizer, str)
                   or (isinstance(normalizer, sf.norm.StainNormalizer)
                       and normalizer.__class__ == 'StainNormalizer'))

    # Build the tile generator
    generator = slide.build_generator(
        shuffle=shuffle,
        show_progress=show_progress,
        img_format=img_format,
        normalizer=(normalizer if opencv_norm else None),
        **kwargs)
    if not generator:
        log.error(f"No tiles extracted from slide [green]{slide.name}")
        return None

    # Build the PyTorch dataloader
    tile_dataset = torch.utils.data.DataLoader(
        _SlideIterator(img_format=img_format, generator=generator),
        batch_size=batch_size,
    )

    # Extract features from the tiles
    for i, (batch_images, batch_loc) in enumerate(tile_dataset):
        if normalizer and not opencv_norm:
            batch_images = normalizer.transform(batch_images)
        if preprocess_fn:
            batch_images = preprocess_fn(batch_images)
        batch_images = batch_images.to(extractor.device)
        model_out = sf.util.as_list(extractor(batch_images))

        # Flatten the output, relevant when
        # there are multiple outcomes / classifier heads
        _act_batch = []
        for m in model_out:
            if isinstance(m, (list, tuple)):
                _act_batch += [_m.contiguous().cpu().float().detach().numpy() for _m in m]
            else:
                _act_batch.append(m.contiguous().cpu().float().detach().numpy())
        _act_batch = np.concatenate(_act_batch, axis=-1)

        grid_idx_updated = []
        for i, act in enumerate(_act_batch):
            xi = batch_loc[i][0]
            yi = batch_loc[i][1]
            if callback:
                grid_idx_updated.append([yi, xi])
            features_grid[yi][xi] = act

        # Trigger a callback signifying that the grid has been updated.
        # Useful for progress tracking.
        if callback:
            callback(grid_idx_updated)

    return features_grid
