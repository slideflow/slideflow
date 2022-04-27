"""
Reinhard normalization based on method of:

E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.

This implementation ("fast" implementation) skips the brightness standardization step.
"""

from __future__ import division

import tensorflow as tf
from typing import Tuple
from slideflow.norm.tensorflow import color


@tf.function
def lab_split(I: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Convert from RGB uint8 to LAB and split into channels
    :param I: uint8
    :return:
    """
    I = tf.cast(I, tf.float32)  # I = I.astype(np.float32)
    I /= 255
    I = color.rgb_to_lab(I)  # I = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    I1, I2, I3 = tf.unstack(I, axis=-1)  # I1, I2, I3 = cv.split(I)
    return I1, I2, I3


@tf.function
def merge_back(
    I1: tf.Tensor,
    I2: tf.Tensor,
    I3: tf.Tensor
) -> tf.Tensor:
    """
    Take seperate LAB channels and merge back to give RGB uint8
    :param I1:
    :param I2:
    :param I3:
    :return:
    """

    I = tf.stack((I1, I2, I3), axis=-1)  # I = np.clip(cv.merge((I1, I2, I3)), 0, 255).astype(np.uint8)
    I = color.lab_to_rgb(I) * 255  # cv.cvtColor(I, cv.COLOR_LAB2RGB)
    # I = tf.experimental.numpy.clip(I, 0, 255)
    return I  # tf.cast(I, tf.uint8)


@tf.function
def get_mean_std(
    I1: tf.Tensor,
    I2: tf.Tensor,
    I3: tf.Tensor,
    reduce: bool = False
) -> Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
           Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    """
    Get mean and standard deviation of each channel
    :param I: uint8
    :return:
    """
    m1, sd1 = tf.math.reduce_mean(I1, axis=(1,2)), tf.math.reduce_std(I1, axis=(1,2)) #m1, sd1 = cv.meanStdDev(I1)
    m2, sd2 = tf.math.reduce_mean(I2, axis=(1,2)), tf.math.reduce_std(I2, axis=(1,2)) #m2, sd2 = cv.meanStdDev(I2)
    m3, sd3 = tf.math.reduce_mean(I3, axis=(1,2)), tf.math.reduce_std(I3, axis=(1,2)) #m3, sd3 = cv.meanStdDev(I3)

    if reduce:
        m1, sd1 = tf.math.reduce_mean(m1), tf.math.reduce_mean(sd1)
        m2, sd2 = tf.math.reduce_mean(m2), tf.math.reduce_mean(sd2)
        m3, sd3 = tf.math.reduce_mean(m3), tf.math.reduce_mean(sd3)

    means = m1, m2, m3
    stds = sd1, sd2, sd3
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

    # norm1 = ((I1 - means[0]) * (tgt_std[0] / stds[0])) + tgt_mean[0]

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
