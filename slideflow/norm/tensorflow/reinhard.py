"""
Reinhard normalization based on method of:

E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.
"""

from __future__ import division

import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Union, Optional, Any
from packaging import version
from contextlib import contextmanager

from slideflow.norm.tensorflow import color
from slideflow.norm import utils as ut
from slideflow.util import log
from .utils import clip_size, standardize_brightness

# -----------------------------------------------------------------------------

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
def get_masked_mean_std(I: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Get mean and standard deviation of each channel, with white pixels masked.

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
    ones = tf.math.reduce_all(I == 255, axis=len(I.shape)-1)
    I1, I2, I3 = lab_split(I)
    I1, I2, I3 = I1[~ ones], I2[~ ones], I3[~ ones]

    m1, sd1 = tf.math.reduce_mean(I1), tf.math.reduce_std(I1)
    m2, sd2 = tf.math.reduce_mean(I2), tf.math.reduce_std(I2)
    m3, sd3 = tf.math.reduce_mean(I3), tf.math.reduce_std(I3)

    means = tf.stack([m1, m2, m3])
    stds = tf.stack([sd1, sd2, sd3])
    return means, stds


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

# -----------------------------------------------------------------------------

@tf.function
def augmented_transform(
    I: tf.Tensor,
    tgt_mean: tf.Tensor,
    tgt_std: tf.Tensor,
    means_stdev: Optional[tf.Tensor] = None,
    stds_stdev: Optional[tf.Tensor] = None,
    **kwargs
) -> tf.Tensor:
    """Transform an image using a given target means & stds, with augmentation.

    Args:
        I (tf.Tensor): Image to transform
        tgt_mean (tf.Tensor): Target means.
        tgt_std (tf.Tensor): Target means.
        means_stdev (tf.Tensor): Standard deviation of tgt_mean for
            augmentation.
        stds_stdev (tf.Tensor): Standard deviation of tgt_std for augmentation.

    Keyword args:
        ctx_mean (torch.Tensor, optional): Context channel means (e.g. from
            whole-slide image). If None, calculates means from the image.
            Defaults to None.
        ctx_std (torch.Tensor, optional): Context channel standard deviations
            (e.g. from whole-slide image). If None, calculates standard
            deviations from the image. Defaults to None.

    Returns:
        tf.Tensor: Transformed image.
    """
    if means_stdev is None and stds_stdev is None:
        raise ValueError("Must supply either means_stdev and/or stds_stdev")
    if means_stdev is not None:
        tgt_mean = tf.random.normal([3], mean=tgt_mean, stddev=means_stdev)
    if stds_stdev is not None:
        tgt_std = tf.random.normal([3], mean=tgt_std, stddev=stds_stdev)
    return transform(I, tgt_mean, tgt_std, **kwargs)


@tf.function
def transform(
    I: tf.Tensor,
    tgt_mean: tf.Tensor,
    tgt_std: tf.Tensor,
    *,
    ctx_mean: Optional[tf.Tensor] = None,
    ctx_std: Optional[tf.Tensor] = None,
    mask_threshold: Optional[float] = None
) -> tf.Tensor:
    """Transform an image using a given target means & stds.

    Args:
        I (tf.Tensor): Image to transform
        tgt_mean (tf.Tensor): Target means.
        tgt_std (tf.Tensor): Target means.

    Keyword args:
        ctx_mean (torch.Tensor, optional): Context channel means (e.g. from
            whole-slide image). If None, calculates means from the image.
            Defaults to None.
        ctx_std (torch.Tensor, optional): Context channel standard deviations
            (e.g. from whole-slide image). If None, calculates standard
            deviations from the image. Defaults to None.

    Raises:
        ValueError: If tgt_mean or tgt_std is None.

    Returns:
        tf.Tensor: Transformed image.
    """
    I1, I2, I3 = lab_split(I)

    if ctx_mean is None and ctx_std is not None:
        raise ValueError(
        "If 'ctx_stds' is provided, 'ctx_means' must not be None"
    )
    if ctx_std is None and ctx_mean is not None:
        raise ValueError(
        "If 'ctx_means' is provided, 'ctx_stds' must not be None"
    )

    if ctx_mean is not None and ctx_std is not None:
        means, stds = ctx_mean, ctx_std
    else:
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
def fit(target: tf.Tensor, reduce: bool = False, mask: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
    """Fit a target image.

    Args:
        target (torch.Tensor): Batch of images to fit.
        reduce (bool): Reduce the fit means/stds across the batch
            of images to a single mean/std array, reduced by average.
            Defaults to False (provides fit for each image in the batch).
            If ``mask`` is True, reduce will be set to ``True``.
        mask (bool): Mask out white pixels during fit. This will reduce
            the means/stdevs across batches, and will only lead to desirable
            results if a single image is provided.

    Returns:
        A tuple containing

            tf.Tensor: Fit means

            tf.Tensor: Fit stds
    """
    if mask:
        return get_masked_mean_std(target)
    else:
        return get_mean_std(*lab_split(target), reduce=reduce)


class ReinhardFastNormalizer:

    vectorized = True
    preferred_device = 'gpu'
    preset_tag = 'reinhard_fast'

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
        self._ctx_means = None  # type: Optional[tf.Tensor]
        self._ctx_stds = None  # type: Optional[tf.Tensor]
        self._augment_params = dict()  # type: Dict[str, tf.Tensor]
        self.threshold = None  # type: Optional[float]
        self.set_fit(**ut.fit_presets[self.preset_tag]['v3'])  # type: ignore
        self.set_augment(**ut.augment_presets[self.preset_tag]['v1'])  # type: ignore

    def fit(
        self,
        target: tf.Tensor,
        reduce: bool = False,
        mask: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Fit normalizer to a target image.

        Args:
            img (tf.Tensor): Target image (RGB uint8) with dimensions
                W, H, c.
            reduce (bool): Reduce fit parameters across a batch of
                images by average. Defaults to False.
            mask (bool): Ignore white pixels (255, 255, 255) when fitting.
                Defulats to False.

        Returns:
            A tuple containing

                target_means (np.ndarray):  Channel means.

                target_stds (np.ndarray):   Channel standard deviations.
        """
        if len(target.shape) == 3:
            target = tf.expand_dims(target, axis=0)
        target = clip_size(target, 2048)
        means, stds = fit(target, reduce=reduce, mask=mask)
        self.target_means = means
        self.target_stds = stds
        return means, stds

    def augment_preset(self, preset: str) -> Dict[str, np.ndarray]:
        """Configure normalizer augmentation using a preset.

        Args:
            preset (str): Preset.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping fit keys to the
                augmentation values (standard deviations).
        """
        _aug = ut.augment_presets[self.preset_tag][preset]
        self.set_augment(**_aug)
        return _aug

    def fit_preset(self, preset: str) -> Dict[str, np.ndarray]:
        """Fit normalizer to a preset in sf.norm.utils.fit_presets.

        Args:
            preset (str): Preset.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping fit keys to their
            fitted values.
        """
        _fit = ut.fit_presets[self.preset_tag][preset]
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

    def set_augment(
        self,
        means_stdev: Optional[Union[np.ndarray, tf.Tensor]] = None,
        stds_stdev: Optional[Union[np.ndarray, tf.Tensor]] = None,
    ) -> None:
        """Set the normalizer augmentation to the given values.

        Args:
            means_stdev (np.ndarray, tf.Tensor): Standard devaiation
                of target_means. Must have the shape (3,).
                Defaults to None (will not augment target means).
            stds_stdev (np.ndarray, tf.Tensor): Standard deviation
                of target_stds. Must have the shape (3,).
                Defaults to None (will not augment target stds).
        """
        if means_stdev is None and stds_stdev is None:
            raise ValueError(
                "One or both arguments 'means_stdev' and 'stds_stdev' are required."
            )
        if means_stdev is not None:
            self._augment_params['means_stdev'] = tf.convert_to_tensor(ut._as_numpy(means_stdev))
        if stds_stdev is not None:
            self._augment_params['stds_stdev'] = tf.convert_to_tensor(ut._as_numpy(stds_stdev))

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

    def _get_context_means(
        self,
        ctx_means: Optional[tf.Tensor] = None,
        ctx_stds: Optional[tf.Tensor] = None
    ) -> Tuple[Optional[tf.Tensor], Optional[tf.Tensor]]:
        if self._ctx_means is not None and self._ctx_stds is not None:
            return self._ctx_means, self._ctx_stds
        else:
            return ctx_means, ctx_stds

    def _transform_batch(
        self,
        batch: tf.Tensor,
        ctx_means: Optional[tf.Tensor] = None,
        ctx_stds: Optional[tf.Tensor] = None,
        *,
        augment: bool = False
    ) -> tf.Tensor:
        """Normalize a batch of images.

        Args:
            img (tf.Tensor): Image, RGB uint8 with dimensions BWHC.
            ctx_means (tf.Tensor, optional): Context channel means (e.g. from
                whole-slide image). If None, calculates means from the image.
                Defaults to None.
            ctx_stds (tf.Tensor, optional): Context channel standard deviations
                (e.g. from whole-slide image). If None, calculates standard
                deviations from the image. Defaults to None.

        Keyword args:
            augment (bool): Transform using stain augmentation.
                Defaults to False.

        Returns:
            tf.Tensor: Normalized image batch (uint8)
        """
        _ctx_means, _ctx_stds = self._get_context_means(ctx_means, ctx_stds)
        fn = augmented_transform if augment else transform
        aug_kw = self._augment_params if augment else {}
        return fn(
            batch,
            self.target_means,
            self.target_stds,
            ctx_mean=_ctx_means,
            ctx_std=_ctx_stds,
            mask_threshold=self.threshold,
            **aug_kw
        )

    def transform(
        self,
        I: tf.Tensor,
        ctx_means: Optional[tf.Tensor] = None,
        ctx_stds: Optional[tf.Tensor] = None,
        *,
        augment: bool = False
    ) -> tf.Tensor:
        """Normalize an H&E image.

        Args:
            img (tf.Tensor): Image, RGB uint8 with dimensions WHC or BWHC.
            ctx_means (tf.Tensor, optional): Context channel means (e.g. from
                whole-slide image). If None, calculates means from the image.
                Defaults to None.
            ctx_stds (tf.Tensor, optional): Context channel standard deviations
                (e.g. from whole-slide image). If None, calculates standard
                deviations from the image. Defaults to None.

        Keyword args:
            augment (bool): Transform using stain augmentation.

        Returns:
            tf.Tensor: Normalized image (uint8)
        """
        if augment and not any(m in self._augment_params
                               for m in ('means_stdev', 'stds_stdev')):
            raise ValueError("Augmentation space not configured.")

        _ctx_means, _ctx_stds = self._get_context_means(ctx_means, ctx_stds)
        if len(I.shape) == 3:
            return self._transform_batch(
                tf.expand_dims(I, axis=0),
                _ctx_means,
                _ctx_stds,
                augment=augment
            )[0]
        else:
            return self._transform_batch(I, _ctx_means, _ctx_stds, augment=augment)

    @contextmanager
    def image_context(self, I: Union[np.ndarray, tf.Tensor]):
        """Set the whole-slide context for the stain normalizer.

        With contextual normalization, max concentrations are determined
        from the context (whole-slide image) rather than the image being
        normalized. This may improve stain normalization for sections of
        a slide that are predominantly eosin (e.g. necrosis or low cellularity).

        When calculating max concentrations from the image context,
        white pixels (255) will be masked.

        This function is a context manager used for temporarily setting the
        image context. For example:

        .. code-block:: python

            with normalizer.image_context(slide):
                normalizer.transform(target)

        Args:
            I (np.ndarray, tf.Tensor): Context to use for normalization, e.g.
                a whole-slide image thumbnail, optionally masked with masked
                areas set to (255, 255, 255).

        """
        self.set_context(I)
        yield
        self.clear_context()

    def set_context(self, I: Union[np.ndarray, tf.Tensor]):
        """Set the whole-slide context for the stain normalizer.

        With contextual normalization, max concentrations are determined
        from the context (whole-slide image) rather than the image being
        normalized. This may improve stain normalization for sections of
        a slide that are predominantly eosin (e.g. necrosis or low cellularity).

        When calculating max concentrations from the image context,
        white pixels (255) will be masked.

        Args:
            I (np.ndarray, tf.Tensor): Context to use for normalization, e.g.
                a whole-slide image thumbnail, optionally masked with masked
                areas set to (255, 255, 255).

        """
        if not isinstance(I, tf.Tensor):
            I = tf.convert_to_tensor(ut._as_numpy(I))
        if len(I.shape) == 3:
            I = tf.expand_dims(I, axis=0)
        I = clip_size(I, 2048)
        self._ctx_means, self._ctx_stds = get_masked_mean_std(I)

    def clear_context(self):
        """Remove any previously set stain normalizer context."""
        self._ctx_means, self._ctx_stds = None, None


class ReinhardNormalizer(ReinhardFastNormalizer):

    """Reinhard H&E stain normalizer (Tensorflow implementation).

    Normalizes an image as defined by:

    Reinhard, Erik, et al. "Color transfer between images." IEEE
    Computer graphics and applications 21.5 (2001): 34-41.

    """

    preset_tag = 'reinhard'

    def fit(
        self,
        target: tf.Tensor,
        reduce: bool = False,
        mask: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Fit normalizer to a target image.

        Args:
            img (tf.Tensor): Target image (RGB uint8) with dimensions
                W, H, c.
            reduce (bool, optional): Reduce fit parameters across a batch of
                images by average. Defaults to False.
            mask (bool): Mask out white pixels during fit. This will reduce
                the means/stdevs across batches, and will only lead to desirable
                results if a single image is provided.

        Returns:
            A tuple containing

                target_means (np.ndarray):  Channel means.

                target_stds (np.ndarray):   Channel standard deviations.
        """
        if len(target.shape) == 3:
            target = tf.expand_dims(target, axis=0)
        target = clip_size(target, 2048)
        target = standardize_brightness(target, mask=mask)
        means, stds = fit(target, reduce=reduce, mask=mask)
        self.target_means = means
        self.target_stds = stds
        return means, stds

    def transform(
        self,
        I: tf.Tensor,
        ctx_means: Optional[tf.Tensor] = None,
        ctx_stds: Optional[tf.Tensor] = None,
        *,
        augment: bool = False
    ) -> tf.Tensor:
        """Normalize an H&E image.

        Args:
            img (tf.Tensor): Image, RGB uint8 with dimensions WHC or BWHC.
            ctx_means (tf.Tensor, optional): Context channel means (e.g. from
                whole-slide image). If None, calculates means from the image.
                Defaults to None.
            ctx_stds (tf.Tensor, optional): Context channel standard deviations
                (e.g. from whole-slide image). If None, calculates standard
                deviations from the image. Defaults to None.

        Keyword args:
            augment (bool): Transform using stain augmentation.
                Defaults to False.

        Returns:
            tf.Tensor: Normalized image (uint8)
        """
        if augment and not any(m in self._augment_params
                               for m in ('means_stdev', 'stds_stdev')):
            raise ValueError("Augmentation space not configured.")

        _ctx_means, _ctx_stds = self._get_context_means(ctx_means, ctx_stds)
        if len(I.shape) == 3:
            return self._transform_batch(
                standardize_brightness(tf.expand_dims(I, axis=0)),
                _ctx_means,
                _ctx_stds,
                augment=augment
            )[0]
        else:
            return self._transform_batch(
                standardize_brightness(I),
                _ctx_means,
                _ctx_stds,
                augment=augment
            )

    def set_context(self, I: tf.Tensor):
        """Set the whole-slide context for the stain normalizer.

        With contextual normalization, max concentrations are determined
        from the context (whole-slide image) rather than the image being
        normalized. This may improve stain normalization for sections of
        a slide that are predominantly eosin (e.g. necrosis or low cellularity).

        When calculating max concentrations from the image context,
        white pixels (255) will be masked.

        Args:
            I (np.ndarray, tf.Tensor): Context to use for normalization, e.g.
                a whole-slide image thumbnail, optionally masked with masked
                areas set to (255, 255, 255).

        """
        if not isinstance(I, tf.Tensor):
            I = tf.convert_to_tensor(ut._as_numpy(I))
        if len(I.shape) == 3:
            I = tf.expand_dims(I, axis=0)
        I = clip_size(I, 2048)
        I = standardize_brightness(I, mask=True)
        super().set_context(I)

    def clear_context(self):
        """Remove any previously set stain normalizer context."""
        super().clear_context()


class ReinhardFastMaskNormalizer(ReinhardFastNormalizer):

    preset_tag = 'reinhard_fast'

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


class ReinhardMaskNormalizer(ReinhardNormalizer):

    preset_tag = 'reinhard'

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
