"""Utilities for processing PyTorch GAN output into Tensorflow tensors."""

from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Union

import numpy as np
import slideflow as sf
import tensorflow as tf
import torch
from torchvision import transforms

if TYPE_CHECKING:
    from slideflow.norm import StainNormalizer


@tf.function
def decode_batch(
    img: tf.Tensor,
    normalizer: Optional["StainNormalizer"] = None,
    standardize: bool = True,
    resize_px: Optional[int] = None,
    resize_method: str = 'lanczos3',
    resize_aa: bool = True,
) -> Dict[str, tf.Tensor]:
    """Process batch of tensorflow images, resizing, normalizing,
    and standardizing.

    Args:
        img (tf.Tensor): Batch of tensorflow images (uint8).
        normalizer (sf.norm.StainNormalizer, optional): Normalizer.
            Defaults to None.
        standardize (bool, optional): Standardize images. Defaults to True.
        resize_px (Optional[int], optional): Resize images. Defaults to None.
        resize_method (str, optional): Resize method. Defaults to 'lanczos3'.
        resize_aa (bool, optional): Apply antialiasing during resizing.
            Defaults to True.

    Returns:
        Dict[str, tf.Tensor]: Processed image.
    """
    if resize_px is not None:
        img = tf.image.resize(
            img,
            (resize_px, resize_px),
            method=resize_method,
            antialias=resize_aa
        )
        img = tf.cast(img, tf.uint8)
    if normalizer is not None:
        if normalizer.vectorized:
            img = normalizer.batch_to_batch(img)  # type: ignore
        else:
            img = tf.stack([normalizer.tf_to_tf(_i) for _i in img])
    if standardize:
        img = tf.image.per_image_standardization(img)
    return {'tile_image': img}


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


def vips_resize(
    img: Union[torch.Tensor, tf.Tensor],
    crop_width: int,
    target_px: int
) -> np.ndarray:
    """Resizes and crops an image tensor using libvips.resize()

    Args:
        img (Union[torch.Tensor, tf.Tensor]): Image.
        crop_width (int): Height/width of image crop (before resize).
        target_px (int): Target size of final image after resizing.

    Returns:
        np.ndarray: Resized image.
    """
    import pyvips
    img_data = np.ascontiguousarray(img.numpy()).data
    vips_image = pyvips.Image.new_from_memory(img_data, crop_width, crop_width, bands=3, format="uchar")
    vips_image = vips_image.resize(target_px/crop_width)
    return sf.slide.vips2numpy(vips_image)


def process_gan_batch(
    img: torch.Tensor,
    gan_um: int,
    gan_px: int,
    target_um: int,
    target_px: Optional[int] = None,
    resize_method: Optional[str] = None
) -> tf.Tensor:
    """Process a batch of raw GAN output, converting to a Tensorflow tensor.

    Args:
        img (torch.Tensor): Raw batch of GAN images.
        resize_method (Optional[str], optional): Resize images after crop.
            Methods include 'torch', 'torch_aa', and 'vips'. 'torch' methods
            use torchvision.transforms.resize, with (torch_aa) or without
            (torch) antialiasing. 'vips' uses the libvips resize method.
            Defaults to None (no resizing).
        gan_um (int, optional): Size of gan output images, in microns.
        gan_px (int, optional): Size of gan output images, in pixels.
        target_um (int, optional): Size of target images, in microns.
            Will crop image to meet this target.
        target_px (int, optional): Size of target images, in pixels.
            Will crop image to meet this target.

    Raises:
        ValueError: If the method is invalid.

    Returns:
        tf.Tensor: Cropped and resized image (uint8).
    """

    if (resize_method is not None
       and resize_method not in ('torch', 'torch_aa', 'vips')):
        raise ValueError(f'Invalid resize method {resize_method}')
    if resize_method is not None and target_px is None:
        raise ValueError("If resizing, target_px must not be None.")

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
        img = [
            vips_resize(i, crop_width=crop_width, target_px=target_px)  # type: ignore
            for i in img
        ]  # type: ignore

    # Convert to Tensorflow tensor
    img = tf.convert_to_tensor(img)

    return img
