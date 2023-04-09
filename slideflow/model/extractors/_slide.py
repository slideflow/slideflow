import slideflow as sf
import numpy as np
from typing import Optional, Callable
from slideflow import log

# -----------------------------------------------------------------------------

def features_from_slide_torch(
    extractor: Callable,
    slide: "sf.WSI",
    *,
    img_format: str = 'auto',
    batch_size: int = 32,
    dtype: type = np.float16,
    grid: Optional[np.ndarray] = None,
    shuffle: bool = False,
    show_progress: bool = True,
    device: str = 'cuda',
    **kwargs
) -> Optional[np.ndarray]:
        """Generate features from tiles in a slide into an array."""

        import torch

        log.debug(f"Slide prediction (batch_size={batch_size})")
        total_out = extractor.num_features + extractor.num_classes
        if grid is None:
            features_grid = np.ones((
                    slide.grid.shape[1],
                    slide.grid.shape[0],
                    total_out),
                dtype=dtype)
            features_grid *= -99
        else:
            assert grid.shape == (slide.grid.shape[1], slide.grid.shape[0], total_out)
            features_grid = grid
        generator = slide.build_generator(
            shuffle=shuffle,
            show_progress=show_progress,
            img_format='numpy',
            **kwargs)
        if not generator:
            log.error(f"No tiles extracted from slide [green]{slide.name}")
            return None

        class SlideIterator(torch.utils.data.IterableDataset):
            def __init__(self, extractor, *args, **kwargs):
                super(SlideIterator).__init__(*args, **kwargs)
                self.extractor = extractor

            def __iter__(self):
                for image_dict in generator():
                    img = image_dict['image']
                    img = torch.from_numpy(img).permute(2, 0, 1)  # CWH
                    loc = np.array(image_dict['grid'])
                    yield img, loc

        tile_dataset = torch.utils.data.DataLoader(
            SlideIterator(extractor),
            batch_size=batch_size)

        for i, (batch_images, batch_loc) in enumerate(tile_dataset):
            batch_images = batch_images.to(device)
            model_out = sf.util.as_list(extractor(batch_images))
            batch_act = np.concatenate([
                m.cpu().detach().numpy()
                for m in model_out
            ])
            for i, act in enumerate(batch_act):
                xi = batch_loc[i][0]
                yi = batch_loc[i][1]
                features_grid[yi][xi] = act

        masked_grid = np.ma.masked_where(features_grid == -99, features_grid)
        return masked_grid
