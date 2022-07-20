"""
Reinhard normalization based on method of:

E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.
"""

from __future__ import division

from typing import Tuple

import tensorflow as tf

import tensorflow_probability as tfp
from slideflow.norm.tensorflow.reinhard_fast import fit as fit_fast
from slideflow.norm.tensorflow.reinhard_fast import transform as transform_fast
from slideflow.norm.tensorflow.reinhard_fast import ReinhardFastNormalizer


@tf.function
def standardize_brightness(I: tf.Tensor) -> tf.Tensor:
    """

    :param I:
    :return:
    """
    p = tfp.stats.percentile(I, 90)  # p = np.percentile(I, 90)
    p = tf.cast(p, tf.float32)
    scaled = tf.cast(I, tf.float32) * tf.constant(255.0, dtype=tf.float32) / p
    scaled = tf.experimental.numpy.clip(scaled, 0, 255)
    return tf.cast(scaled, tf.uint8)  # np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)

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
    I = standardize_brightness(I)
    return transform_fast(I, tgt_mean, tgt_std)

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
    target = standardize_brightness(target)
    return fit_fast(target, reduce=reduce)

class ReinhardNormalizer(ReinhardFastNormalizer):
    """
    A stain normalization object
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(
        self,
        target: tf.Tensor,
        reduce: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if len(target.shape) == 3:
            target = tf.expand_dims(target, axis=0)
        means, stds = fit(target, reduce=reduce)
        self.target_means = means
        self.target_stds = stds
        return means, stds

    def transform(self, I: tf.Tensor) -> tf.Tensor:
        if len(I.shape) == 3:
            return transform(tf.expand_dims(I, axis=0), self.target_means, self.target_stds)[0]
        else:
            return transform(I, self.target_means, self.target_stds)