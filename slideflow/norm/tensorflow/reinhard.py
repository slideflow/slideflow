"""
Reinhard normalization based on method of:

E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.
"""

from __future__ import division

import numpy as np
from typing import Tuple, Dict, Union, Optional, Any
from packaging import version

import tensorflow as tf
import tensorflow_probability as tfp
from slideflow.norm.tensorflow import color
from slideflow.norm import utils as ut
from slideflow.util import log


@tf.function
def standardize_brightness(I: tf.Tensor) -> tf.Tensor:
    """Standardize image brightness to the 90th percentile.

    Args:
        I (tf.Tensor): Image, uint8.

    Returns:
        tf.Tensor: Brightness-standardized image (uint8)
    """
    p = tfp.stats.percentile(I, 90)  # p = np.percentile(I, 90)
    p = tf.cast(p, tf.float32)
    scaled = tf.cast(I, tf.float32) * tf.constant(255.0, dtype=tf.float32) / p
    scaled = tf.experimental.numpy.clip(scaled, 0, 255)
    return tf.cast(scaled, tf.uint8)  # np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)


@tf.function
def lab_split(I: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Convert from RGB uint8 to LAB and split into channels

    Args:
        I (tf.Tensor): RGB uint8 image.

    Returns:
        A tuple containing

            tf.Tensor: I1, first channel (float32).

            tf.Tensor: I2, second channel (float32).

            tf.Tensor: I3, third channel (float32).
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
        I1 (tf.Tensor): First channel (float32).
        I2 (tf.Tensor): Second channel (float32).
        I3 (tf.Tensor): Third channel (float32).

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
        I1 (tf.Tensor): First channel (float32).
        I2 (tf.Tensor): Second channel (float32).
        I3 (tf.Tensor): Third channel (float32).
        reduce (bool): Reduce batch to mean across images in the batch.

    Returns:
        A tuple containing

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
    tgt_std: tf.Tensor,
    mask_threshold: Optional[float] = None
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
    I1, I2, I3 = lab_split(I)
    means, stds = get_mean_std(I1, I2, I3)

    if mask_threshold:
        mask = ((I1 / 100) < mask_threshold)[:, :, :, tf.newaxis]

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

    if mask_threshold:
        return tf.where(mask, clipped, I)
    else:
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
        A tuple containing

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
        if (version.parse(tf.__version__) >= version.parse("2.8.0")
            and version.parse(tf.__version__) < version.parse("2.8.2")):
            log.warn("A bug in Tensorflow 2.8.0 prevents Reinhard GPU "
                     "acceleration; please upgrade to >= 2.8.2. Falling back "
                     "to CPU.")
            self.preferred_device = 'cpu'
        self.transform_kw = {}  # type: Dict[str, Any]
        self.set_fit(**ut.fit_presets['reinhard_fast']['v1'])  # type: ignore

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
            A tuple containing

                target_means (np.ndarray):  Channel means.

                target_stds (np.ndarray):   Channel standard deviations.
        """
        if len(target.shape) == 3:
            target = tf.expand_dims(target, axis=0)
        means, stds = fit(target, reduce=reduce)
        self.target_means = means
        self.target_stds = stds
        return means, stds

    def fit_preset(self, preset: str) -> Dict[str, np.ndarray]:
        """Fit normalizer to a preset in sf.norm.utils.fit_presets.

        Args:
            preset (str): Preset.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping fit keys to their
            fitted values.
        """
        _fit = ut.fit_presets['reinhard_fast'][preset]
        self.set_fit(**_fit)
        return _fit

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
            target_means = tf.convert_to_tensor(ut._as_numpy(target_means))
        if not isinstance(target_stds, tf.Tensor):
            target_stds = tf.convert_to_tensor(ut._as_numpy(target_stds))
        self.target_means = target_means
        self.target_stds = target_stds

    @tf.function
    def _transform_batch(self, batch: tf.Tensor) -> tf.Tensor:
        """Normalize a batch of images.

        Args:
            img (tf.Tensor): Image, RGB uint8 with dimensions BWHC.

        Returns:
            tf.Tensor: Normalized image batch (uint8)
        """

        return transform(batch, self.target_means, self.target_stds)

    @tf.function
    def transform(self, I: tf.Tensor) -> tf.Tensor:
        """Normalize an H&E image.

        Args:
            img (tf.Tensor): Image, RGB uint8 with dimensions WHC or BWHC.

        Returns:
            tf.Tensor: Normalized image (uint8)
        """
        if len(I.shape) == 3:
            return self._transform_batch(tf.expand_dims(I, axis=0))[0]
        else:
            return self._transform_batch(I)


class ReinhardNormalizer(ReinhardFastNormalizer):

    def __init__(self) -> None:
        """Reinhard H&E stain normalizer (Tensorflow implementation).

        Normalizes an image as defined by:

        Reinhard, Erik, et al. "Color transfer between images." IEEE
        Computer graphics and applications 21.5 (2001): 34-41.

        """
        super().__init__()
        self.set_fit(**ut.fit_presets['reinhard']['v1'])  # type: ignore

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
            A tuple containing

                target_means (np.ndarray):  Channel means.

                target_stds (np.ndarray):   Channel standard deviations.
        """
        if len(target.shape) == 3:
            target = tf.expand_dims(target, axis=0)
        target = standardize_brightness(target)
        means, stds = fit(target, reduce=reduce)
        self.target_means = means
        self.target_stds = stds
        return means, stds

    def fit_preset(self, preset: str) -> Dict[str, np.ndarray]:
        """Fit normalizer to a preset in sf.norm.utils.fit_presets.

        Args:
            preset (str): Preset.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping fit keys to their
                fitted values.
        """
        _fit = ut.fit_presets['reinhard'][preset]
        self.set_fit(**_fit)
        return _fit

    @tf.function
    def transform(self, I: tf.Tensor) -> tf.Tensor:
        """Normalize an H&E image.

        Args:
            img (tf.Tensor): Image, RGB uint8 with dimensions WHC or BWHC.

        Returns:
            tf.Tensor: Normalized image (uint8)
        """

        if len(I.shape) == 3:
            return self._transform_batch(standardize_brightness(tf.expand_dims(I, axis=0)))[0]
        else:
            return self._transform_batch(standardize_brightness(I))


class ReinhardFastMaskNormalizer(ReinhardFastNormalizer):

    def __init__(self, threshold: float = 0.93) -> None:
        """Modified Reinhard H&E stain normalizer only applied to
        non-whitepsace areas (Tensorflow implementation).

        Normalizes an image as defined by:

        Reinhard, Erik, et al. "Color transfer between images." IEEE
        Computer graphics and applications 21.5 (2001): 34-41.

        This "masked" implementation only normalizes non-whitespace areas.

        This normalizer contains inspiration from StainTools by Peter Byfield
        (https://github.com/Peter554/StainTools).

        Args:
            threshold (float): Whitespace fraction threshold, above which
                pixels are masked and not normalized. Defaults to 0.93.
        """
        super().__init__()
        self.threshold = threshold

    @tf.function
    def _transform_batch(self, batch: tf.Tensor) -> tf.Tensor:
        """Normalize a batch of images.

        Args:
            img (tf.Tensor): Image, RGB uint8 with dimensions BWHC.

        Returns:
            tf.Tensor: Normalized image batch (uint8)
        """

        return transform(batch, self.target_means, self.target_stds, self.threshold)


class ReinhardMaskNormalizer(ReinhardNormalizer):

    def __init__(self, threshold: float = 0.93) -> None:
        """Modified Reinhard H&E stain normalizer only applied to
        non-whitepsace areas (Tensorflow implementation).

        Normalizes an image as defined by:

        Reinhard, Erik, et al. "Color transfer between images." IEEE
        Computer graphics and applications 21.5 (2001): 34-41.

        This "masked" implementation only normalizes non-whitespace areas.

        This normalizer contains inspiration from StainTools by Peter Byfield
        (https://github.com/Peter554/StainTools).

        Args:
            threshold (float): Whitespace fraction threshold, above which
                pixels are masked and not normalized. Defaults to 0.93.
        """
        super().__init__()
        self.threshold = threshold

    @tf.function
    def _transform_batch(self, batch: tf.Tensor) -> tf.Tensor:
        """Normalize a batch of images.

        Args:
            img (tf.Tensor): Image, RGB uint8 with dimensions BWHC.

        Returns:
            tf.Tensor: Normalized image batch (uint8)
        """

        return transform(batch, self.target_means, self.target_stds, self.threshold)
