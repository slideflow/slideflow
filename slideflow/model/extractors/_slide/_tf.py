"""Tensorflow-based feature extraction from whole-slide images."""

import slideflow as sf
import numpy as np
import tensorflow as tf
from typing import Optional, Callable, Union, TYPE_CHECKING
from slideflow import log

from ._utils import _build_grid, _log_normalizer, _use_numpy_if_png

if TYPE_CHECKING:
    from slideflow.model.base import BaseFeatureExtractor
    from slideflow.norm import StainNormalizer

# -----------------------------------------------------------------------------

@tf.function
def _standardize(image, loc, preprocess=None):
    """Standardize an image."""
    parsed_image = tf.image.per_image_standardization(image)
    return parsed_image, loc

def _wrap_preprocess(preprocess):

    @tf.function
    def _wrapped_preprocess(image, loc):
        return preprocess(image), loc

    return _wrapped_preprocess


def _build_slide_iterator(
    generator: Callable,
    slide: "sf.WSI",
    img_format: str,
    normalizer: Optional["StainNormalizer"],
    batch_size: int,
    preprocess_fn: Optional[Callable] = None,
):
    """Build an iterator that extracts and processes tiles from a slide."""

    # --- Utility functions ---

    def tile_generator():
        for image_dict in generator():
            yield {
                'grid': image_dict['grid'],
                'image': image_dict['image']
            }

    @tf.function
    def _parse(record):
        image = record['image']
        if img_format.lower() in ('jpg', 'jpeg'):
            image = tf.image.decode_jpeg(image, channels=3)
        image.set_shape([slide.tile_px, slide.tile_px, 3])
        loc = record['grid']
        return image, loc

    # --- Build the dataset ---

    # Generate dataset from the generator
    with tf.name_scope('dataset_input'):

        # Define the output signature and parse the dataset
        log.debug(f"Processing tiles with img_format={img_format}")
        output_signature = {
            'grid': tf.TensorSpec(shape=(2), dtype=tf.uint32)
        }
        if img_format.lower() in ('jpg', 'jpeg'):
            output_signature.update({
                'image': tf.TensorSpec(shape=(), dtype=tf.string)
            })
        else:
            output_signature.update({
                'image': tf.TensorSpec(shape=(slide.tile_px,
                                                slide.tile_px,
                                                3),
                                        dtype=tf.uint8)
            })
        tile_dataset = tf.data.Dataset.from_generator(
            tile_generator,
            output_signature=output_signature
        )
        tile_dataset = tile_dataset.map(
            _parse,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
        )

        # Apply stain normalization
        if normalizer:
            log.debug(f"Using stain normalizer: {normalizer.method}")
            if normalizer.vectorized:
                log.debug("Using vectorized normalization")
                norm_batch_size = 32 if not batch_size else batch_size
                tile_dataset = tile_dataset.batch(norm_batch_size, drop_remainder=False)
            else:
                log.debug("Using per-image normalization")
            tile_dataset = tile_dataset.map(
                normalizer.tf_to_tf,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True
            )
            if normalizer.vectorized:
                tile_dataset = tile_dataset.unbatch()
            if normalizer.method == 'macenko':
                # Drop the images that causes an error, e.g. if eigen
                # decomposition is unsuccessful.
                tile_dataset = tile_dataset.apply(tf.data.experimental.ignore_errors())

        # Standardize the images:
        tile_dataset = tile_dataset.map(
            (_standardize if preprocess_fn is None
                          else _wrap_preprocess(preprocess_fn)),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
        )

        # Batch and prefetch
        tile_dataset = tile_dataset.batch(batch_size, drop_remainder=False)
        tile_dataset = tile_dataset.prefetch(8)

    return tile_dataset

# -----------------------------------------------------------------------------

def features_from_slide_tf(
    extractor: "BaseFeatureExtractor",
    slide: "sf.WSI",
    *,
    img_format: str = 'numpy',
    batch_size: int = 32,
    dtype: type = np.float16,
    grid: Optional[np.ndarray] = None,
    shuffle: bool= False,
    show_progress: bool = True,
    callback: Optional[Callable] = None,
    normalizer: Optional[Union[str, "StainNormalizer"]] = None,
    normalizer_source: Optional[str] = None,
    preprocess_fn: Optional[Callable] = None,
    **kwargs
) -> Optional[np.ndarray]:

    log.debug(f"Slide prediction (batch_size={batch_size}, "
              f"img_format={img_format})")

    img_format = _use_numpy_if_png(img_format)

    # Create the output array
    features_grid = _build_grid(extractor, slide, grid=grid, dtype=dtype)

    # Establish stain normalization
    if isinstance(normalizer, str):
        normalizer = sf.norm.autoselect(
            normalizer,
            source=normalizer_source,
            backend='tensorflow'
        )
    _log_normalizer(normalizer)

    generator = slide.build_generator(
        img_format=img_format,
        shuffle=shuffle,
        show_progress=show_progress,
        **kwargs
    )
    if not generator:
        log.error(f"No tiles extracted from slide [green]{slide.name}")
        return None

    # Build the Tensorflow dataset
    tile_dataset = _build_slide_iterator(
        generator, slide, img_format, normalizer, batch_size, preprocess_fn
    )

    # Extract features from the tiles
    for i, (batch_images, batch_loc) in enumerate(tile_dataset):
        model_out = extractor._predict(batch_images)
        if not isinstance(model_out, (list, tuple)):
            model_out = [model_out]

        # Flatten the output, relevant when
        # there are multiple outcomes / classifier heads
        _act_batch = []
        for m in model_out:
            if isinstance(m, list):
                _act_batch += [_m.numpy() for _m in m]
            else:
                _act_batch.append(m.numpy())
        _act_batch = np.concatenate(_act_batch, axis=-1)

        _loc_batch = batch_loc.numpy()
        grid_idx_updated = []
        for i, act in enumerate(_act_batch):
            xi = _loc_batch[i][0]
            yi = _loc_batch[i][1]
            if callback:
                grid_idx_updated.append((yi, xi))
            features_grid[yi][xi] = act

        # Trigger a callback signifying that the grid has been updated.
        # Useful for progress tracking.
        if callback:
            callback(grid_idx_updated)

    return features_grid
