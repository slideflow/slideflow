"""
Reinhard normalization based on method of:

E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.

This implementation ("fast" implementation) skips the brightness standardization step.
"""

from __future__ import division

import os
import numpy as np
from typing import Tuple, Dict, Union, Optional

import tensorflow as tf

from slideflow.norm.tensorflow import color


@tf.function
def lab_split(I: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Convert from RGB uint8 to LAB and split into channels

    Args:
        I (tf.Tensor): RGB uint8 image.

    Returns:
        tf.Tensor: I1, first channel (uint8).

        tf.Tensor: I2, first channel (uint8).

        tf.Tensor: I3, first channel (uint8).
    """
    I = tf.cast(I, tf.float32)
    I /= 255
    I = color.rgb_to_lab(I)
    I1, I2, I3 = tf.unstack(I, axis=-1)
    return I1, I2, I3


@tf.function
def merge_back(
    I1: tf.Tensor,
    I2: tf.Tensor,
    I3: tf.Tensor
) -> tf.Tensor:
    """Take seperate LAB channels and merge back to give RGB uint8

    Args:
        I1 (tf.Tensor): First channel (uint8).
        I2 (tf.Tensor): Second channel (uint8).
        I3 (tf.Tensor): Third channel (uint8).

    Returns:
        tf.Tensor: RGB uint8 image.
    """

    I = tf.stack((I1, I2, I3), axis=-1)
    I = color.lab_to_rgb(I) * 255
    return I


@tf.function
def get_mean_std(
    I1: tf.Tensor,
    I2: tf.Tensor,
    I3: tf.Tensor,
    reduce: bool = False
) -> Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
           Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    """Get mean and standard deviation of each channel.

    Args:
        I1 (tf.Tensor): First channel (uint8).
        I2 (tf.Tensor): Second channel (uint8).
        I3 (tf.Tensor): Third channel (uint8).
        reduce (bool): Reduce batch to mean across images in the batch.

    Returns:
        tf.Tensor:     Channel means, shape = (3,)
        tf.Tensor:     Channel standard deviations, shape = (3,)
    """
    m1, sd1 = tf.math.reduce_mean(I1, axis=(1,2)), tf.math.reduce_std(I1, axis=(1,2))
    m2, sd2 = tf.math.reduce_mean(I2, axis=(1,2)), tf.math.reduce_std(I2, axis=(1,2))
    m3, sd3 = tf.math.reduce_mean(I3, axis=(1,2)), tf.math.reduce_std(I3, axis=(1,2))

    if reduce:
        m1, sd1 = tf.math.reduce_mean(m1), tf.math.reduce_mean(sd1)
        m2, sd2 = tf.math.reduce_mean(m2), tf.math.reduce_mean(sd2)
        m3, sd3 = tf.math.reduce_mean(m3), tf.math.reduce_mean(sd3)

    means = tf.stack([m1, m2, m3])
    stds = tf.stack([sd1, sd2, sd3])
    return means, stds


@tf.function
def transform(
    I: tf.Tensor,
    tgt_mean: tf.Tensor,
    tgt_std: tf.Tensor
) -> tf.Tensor:
    """Transform an image using a given target means & stds.

    Args:
        I (tf.Tensor): Image to transform
        tgt_mean (tf.Tensor): Target means.
        tgt_std (tf.Tensor): Target means.

    Raises:
        ValueError: If tgt_mean or tgt_std is None.

    Returns:
        tf.Tensor: Transformed image.
    """
    if tgt_mean is None or tgt_std is None:
        raise ValueError("Normalizer has not been fit: call normalizer.fit()")

    I1, I2, I3 = lab_split(I)
    means, stds = get_mean_std(I1, I2, I3)

    I1a = tf.subtract(I1, tf.expand_dims(tf.expand_dims(means[0], axis=-1), axis=-1))
    I1b = tf.divide(tgt_std[0], stds[0])
    norm1 = (I1a * tf.expand_dims(tf.expand_dims(I1b, axis=-1), axis=-1)) + tgt_mean[0]

    I2a = tf.subtract(I2, tf.expand_dims(tf.expand_dims(means[1], axis=-1), axis=-1))
    I2b = tf.divide(tgt_std[1], stds[1])
    norm2 = (I2a * tf.expand_dims(tf.expand_dims(I2b, axis=-1), axis=-1)) + tgt_mean[1]

    I3a = tf.subtract(I3, tf.expand_dims(tf.expand_dims(means[2], axis=-1), axis=-1))
    I3b = tf.divide(tgt_std[2], stds[2])
    norm3 = (I3a * tf.expand_dims(tf.expand_dims(I3b, axis=-1), axis=-1)) + tgt_mean[2]

    merged = tf.cast(merge_back(norm1, norm2, norm3), dtype=tf.int32)
    clipped = tf.cast(tf.clip_by_value(merged, clip_value_min=0, clip_value_max=255), dtype=tf.uint8)
    return clipped


@tf.function
def fit(target: tf.Tensor, reduce: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
    """Fit a target image.

    Args:
        target (torch.Tensor): Batch of images to fit.
        reduce (bool, optional): Reduce the fit means/stds across the batch
            of images to a single mean/std array, reduced by average.
            Defaults to False (provides fit for each image in the batch).

    Returns:
        tf.Tensor: Fit means
        tf.Tensor: Fit stds
    """
    means, stds = get_mean_std(*lab_split(target), reduce=reduce)
    return means, stds


class ReinhardFastNormalizer:

    vectorized = True
    preferred_device = 'gpu'

    def __init__(self) -> None:
        """Modified Reinhard H&E stain normalizer without brightness
        standardization (Tensorflow implementation).

        Normalizes an image as defined by:

        Reinhard, Erik, et al. "Color transfer between images." IEEE
        Computer graphics and applications 21.5 (2001): 34-41.

        This implementation does not include the brightness normalization step.
        """
        package_directory = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(package_directory, '../norm_tile.jpg')
        src_img = tf.image.decode_jpeg(tf.io.read_file(img_path))
        self.fit(tf.expand_dims(src_img, axis=0))

    def fit(
        self,
        target: tf.Tensor,
        reduce: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Fit normalizer to a target image.

        Args:
            img (tf.Tensor): Target image (RGB uint8) with dimensions
                W, H, c.
            reduce (bool, optional): Reduce fit parameters across a batch of
                images by average. Defaults to False.

        Returns:
            target_means (np.ndarray):  Channel means.

            target_stds (np.ndarray):   Channel standard deviations.
        """
        if len(target.shape) == 3:
            target = tf.expand_dims(target, axis=0)
        means, stds = fit(target, reduce=reduce)
        self.target_means = means
        self.target_stds = stds
        return means, stds

    def get_fit(self) -> Dict[str, Optional[np.ndarray]]:
        """Get the current normalizer fit.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping 'target_means'
                and 'target_stds' to their respective fit values.
        """
        return {
            'target_means': None if self.target_means is None else self.target_means.numpy(),  # type: ignore
            'target_stds': None if self.target_stds is None else self.target_stds.numpy()  # type: ignore
        }

    def set_fit(
        self,
        target_means: Union[np.ndarray, tf.Tensor],
        target_stds: Union[np.ndarray, tf.Tensor]
    ) -> None:
        """Set the normalizer fit to the given values.

        Args:
            target_means (np.ndarray, tf.Tensor): Channel means. Must
                have the shape (3,).
            target_stds (np.ndarray, tf.Tensor): Channel standard deviations.
                Must have the shape (3,).
        """
        if not isinstance(target_means, tf.Tensor):
            target_means = tf.convert_to_tensor(target_means)
        if not isinstance(target_stds, tf.Tensor):
            target_stds = tf.convert_to_tensor(target_stds)
        self.target_means = target_means
        self.target_stds = target_stds

    def transform(self, I: tf.Tensor) -> tf.Tensor:
        """Normalize an H&E image.

        Args:
            img (tf.Tensor): Image, RGB uint8 with dimensions W, H, C.

        Returns:
            tf.Tensor: Normalized image (uint8)
        """
        if len(I.shape) == 3:
            return transform(tf.expand_dims(I, axis=0), self.target_means, self.target_stds)[0]
        else:
            return transform(I, self.target_means, self.target_stds)