"""Utilities for processing PyTorch GAN output into Tensorflow tensors."""

from functools import partial
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import slideflow as sf
import tensorflow as tf
import torch
from torchvision import transforms

if TYPE_CHECKING:
    from slideflow.norm import StainNormalizer


def decode_img(
    img: np.ndarray,
    normalizer: Optional["StainNormalizer"] = None
) -> np.ndarray:
    if normalizer is not None:
        img = normalizer.rgb_to_rgb(img)
    img = np.expand_dims(img, axis=0)
    tf_img = tf.image.per_image_standardization(img)
    return {'tile_image': tf_img}


@tf.function
def decode_batch(
    img,
    normalizer: Optional["StainNormalizer"] = None,
    standardize: bool = True,
    resize_px: Optional[int] = None,
    resize_method: str = 'lanczos3',
    resize_aa: bool = True,
) -> np.ndarray:
    if resize_px is not None:
        img = tf.image.resize(img, (resize_px, resize_px), method=resize_method, antialias=resize_aa)
        img = tf.cast(img, tf.uint8)
    if normalizer is not None:
        img = normalizer.batch_to_batch(img)[0]
    if standardize:
        img = tf.image.per_image_standardization(img)
    return {'tile_image': img}


def process_gan_raw(img, normalizer=None, **kwargs):
    """Process raw GAN output, returning a
    non-normalized image Tensor (for viewing)
    and a normalized image Tensor (for inference)
    """
    img = process_gan_batch(img)
    img = decode_batch(img, **kwargs)['tile_image']
    if normalizer is not None:
        img = normalizer.batch_to_batch(img)[0]
    processed_img = tf.image.per_image_standardization(img)
    return img, processed_img


def build_gan_dataset(
    generator: Callable,
    target_px: int,
    normalizer: Optional[sf.norm.StainNormalizer] = None
) -> tf.data.Dataset:
    """Builds a processed GAN image dataset from a generator.

    Args:
        generator (Callable): GAN generator which yields a GAN image batch.
        normalizer (Optional[sf.norm.StainNormalizer], optional): Stain
            normalizer. Defaults to None.

    Returns:
        tf.data.Dataset: Processed dataset.
    """
    sig = tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.uint8)
    dts = tf.data.Dataset.from_generator(generator, output_signature=sig)
    dts = dts.map(
        partial(decode_batch, normalizer=normalizer, resize_px=target_px),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )
    return dts


def vips_resize(img, crop_width, target_px):
    import pyvips
    img_data = np.ascontiguousarray(img.numpy()).data
    vips_image = pyvips.Image.new_from_memory(img_data, crop_width, crop_width, bands=3, format="uchar")
    vips_image = vips_image.resize(target_px/crop_width)
    img = sf.slide.vips2numpy(vips_image)
    return img


def process_gan_batch(
    img: torch.Tensor,
    resize_method=None,
    gan_um: int = 400,
    gan_px: int = 512,
    target_um: int = 302,
    target_px: int = 299
) -> tf.Tensor:

    if (resize_method is not None
       and resize_method not in ('torch', 'torch_aa', 'vips')):
        raise ValueError(f'Invalid resize method {resize_method}')

    # Calculate parameters for resize/crop.
    crop_factor = target_um / gan_um
    crop_width = int(crop_factor * gan_px)
    left = int(gan_px/2 - crop_width/2)
    upper = int(gan_px/2 - crop_width/2)

    # Perform crop/resize and convert to tensor
    img = transforms.functional.crop(img, upper, left, crop_width, crop_width)

    # Resize with PyTorch
    if resize_method in ('torch', 'torch_aa'):
        img = transforms.functional.resize(img, (target_px, target_px), antialias=(resize_method=='torch_aa'))

    # Re-order the dimension from BCWH -> BWHC
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

    # Resize with VIPS
    if resize_method == 'vips':
        img = [vips_resize(i, crop_width=crop_width, target_px=target_px) for i in img]

    # Convert to Tensorflow tensor
    img = tf.convert_to_tensor(img)

    return img


def process_gan_uint8(img):
    """Process a GAN uint8 image, returning a
    non-normalized image Tensor (for viewing)
    and a normalized image Tensor (for inference)
    """
    img = torch.from_numpy(np.expand_dims(img, axis=0)).permute(0, 3, 1, 2)
    img = (img / 127.5) - 1
    return process_gan_raw(img)
