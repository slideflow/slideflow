"""Reinhard H&E stain normalization."""

from __future__ import division

import cv2
import numpy as np
from typing import Tuple, Dict, Optional
from contextlib import contextmanager

from . import utils as ut
from .utils import lab_split_numpy as lab_split
from .utils import merge_back_numpy as merge_back

# -----------------------------------------------------------------------------

def get_mean_std(I: np.ndarray, mask: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Get mean and standard deviation of each channel.

    Args:
        I (np.ndarray): RGB uint8 image.

    Returns:
        A tuple containing

            np.ndarray:     Channel means, shape = (3,)

            np.ndarray:     Channel standard deviations, shape = (3,)
    """
    I1, I2, I3 = lab_split(I)
    if mask:
        ones = np.all(I == 255, axis=2)
        I1, I2, I3 = I1[~ ones], I2[~ ones], I3[~ ones]
    # Calculate mean and std for each channel.
    # This is about 20% faster than using np.std and np.mean
    m1, sd1 = cv2.meanStdDev(I1)
    m2, sd2 = cv2.meanStdDev(I2)
    m3, sd3 = cv2.meanStdDev(I3)
    means = m1, m2, m3
    stds = sd1, sd2, sd3
    return np.array(means), np.array(stds)


class ReinhardFastNormalizer:

    preset_tag = 'reinhard_fast'

    def __init__(self):
        """Modified Reinhard H&E stain normalizer without brightness
        standardization (numpy implementation).

        Normalizes an image as defined by:

        Reinhard, Erik, et al. "Color transfer between images." IEEE
        Computer graphics and applications 21.5 (2001): 34-41.

        This implementation does not include the brightness normalization step.

        This normalizer contains inspiration from StainTools by Peter Byfield
        (https://github.com/Peter554/StainTools).
        """
        self.threshold = None
        self._ctx_means = None
        self._ctx_stds = None
        self._augment_params = dict()  # type: Dict[str, np.ndarray]
        self.set_fit(**ut.fit_presets[self.preset_tag]['v3'])  # type: ignore
        self.set_augment(**ut.augment_presets[self.preset_tag]['v1'])  # type: ignore

    def fit(self, img: np.ndarray, mask: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Fit normalizer to a target image.

        Args:
            img (np.ndarray): Target image (RGB uint8) with dimensions W, H, C.
            mask (bool): Ignore white pixels (255, 255, 255) when fitting.
                Defulats to False.

        Returns:
            A tuple containing

                np.ndarray:  Target means (channel means).

                np.ndarray:   Target stds (channel standard deviations).
        """
        img = ut.clip_size(img, 2048)
        means, stds = get_mean_std(img, mask=mask)
        self.set_fit(means, stds)
        return means, stds

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

    def get_fit(self) -> Dict[str, np.ndarray]:
        """Get the current normalizer fit.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping 'target_means'
            and 'target_stds' to their respective fit values.
        """
        return {
            'target_means': self.target_means,
            'target_stds': self.target_stds
        }

    def _get_mean_std(
        self,
        image: np.ndarray,
        ctx_means: Optional[np.ndarray],
        ctx_stds: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get means and standard deviations from an image."""
        if ctx_means is None and ctx_stds is not None:
            raise ValueError(
            "If 'ctx_stds' is provided, 'ctx_means' must not be None"
        )
        if ctx_stds is None and ctx_means is not None:
            raise ValueError(
            "If 'ctx_means' is provided, 'ctx_stds' must not be None"
        )
        if ctx_means is not None and ctx_stds is not None:
            return ctx_means, ctx_stds
        elif self._ctx_means is not None and self._ctx_stds is not None:
            return self._ctx_means, self._ctx_stds
        else:
            return get_mean_std(image)

    def set_fit(
        self,
        target_means: np.ndarray,
        target_stds: np.ndarray
    ) -> None:
        """Set the normalizer fit to the given values.

        Args:
            target_means (np.ndarray): Channel means. Must
                have the shape (3,).
            target_stds (np.ndarray): Channel standard deviations. Must
                have the shape (3,).
        """
        target_means = ut._as_numpy(target_means).flatten()
        target_stds = ut._as_numpy(target_stds).flatten()

        if target_means.shape != (3,):
            raise ValueError("target_means must have flattened shape of (3,) - "
                             f"got {target_means.shape}")
        if target_stds.shape != (3,):
            raise ValueError("target_stds must have flattened shape of (3,) - "
                             f"got {target_stds.shape}")

        self.target_means = target_means
        self.target_stds = target_stds

    def set_augment(
        self,
        means_stdev: Optional[np.ndarray] = None,
        stds_stdev: Optional[np.ndarray] = None,
    ) -> None:
        """Set the normalizer augmentation to the given values.

        Args:
            means_stdev (np.ndarray, tf.Tensor): Standard deviation
                of the target means. Must have the shape (3,).
                Defaults to None (will not augment target means).
            stds_stdev (np.ndarray, tf.Tensor): Standard deviation
                of the target stds. Must have the shape (3,).
                Defaults to None (will not augment target stds).
        """
        if means_stdev is None and stds_stdev is None:
            raise ValueError(
                "One or both arguments 'means_stdev' and 'stds_stdev' are required."
            )
        if means_stdev is not None:
            self._augment_params['means_stdev'] = ut._as_numpy(means_stdev).flatten()
        if stds_stdev is not None:
            self._augment_params['stds_stdev'] = ut._as_numpy(stds_stdev).flatten()

    def transform(
        self,
        I: np.ndarray,
        ctx_means: Optional[np.ndarray] = None,
        ctx_stds: Optional[np.ndarray] = None,
        *,
        augment: bool = False
    ) -> np.ndarray:
        """Normalize an H&E image.

        Args:
            img (np.ndarray): Image, RGB uint8 with dimensions W, H, C.
            ctx_means (np.ndarray, optional): Context channel means (e.g. from
                whole-slide image). If None, calculates means from the image.
                Defaults to None.
            ctx_stds (np.ndarray, optional): Context channel standard deviations
                (e.g. from whole-slide image). If None, calculates standard
                deviations from the image. Defaults to None.

        Keyword args:
            augment (bool): Transform using stain augmentation.
                Defaults to False.

        Returns:
            np.ndarray: Normalized image.
        """
        if self.target_means is None or self.target_stds is None:
            raise ValueError("Normalizer has not been fit: call normalizer.fit()")

        # Augmentation; optional
        if augment and not any(m in self._augment_params
                               for m in ('means_stdev', 'stds_stdev')):
            raise ValueError("Augmentation space not configured.")
        if augment and 'means_stdev' in self._augment_params:
            target_means = np.random.normal(
                self.target_means,
                self._augment_params['means_stdev']
            )
        else:
            target_means = self.target_means
        if augment and 'stds_stdev' in self._augment_params:
            target_stds = np.random.normal(
                self.target_stds,
                self._augment_params['stds_stdev']
            )
        else:
            target_stds = self.target_stds

        I1, I2, I3 = lab_split(I)
        if self.threshold is not None:
            mask = ((I3 + 128.) / 255. < self.threshold)[:, :, np.newaxis]
        means, stds = self._get_mean_std(I, ctx_means, ctx_stds)

        norm1 = ((I1 - means[0]) * (target_stds[0] / stds[0])) + target_means[0]
        norm2 = ((I2 - means[1]) * (target_stds[1] / stds[1])) + target_means[1]
        norm3 = ((I3 - means[2]) * (target_stds[2] / stds[2])) + target_means[2]

        merged = merge_back(norm1, norm2, norm3)
        if self.threshold is not None:
            return np.where(mask, merged, I)
        else:
            return merged

    @contextmanager
    def image_context(self, I: np.ndarray):
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
            I (np.ndarray): Context to use for normalization, e.g.
                a whole-slide image thumbnail, optionally masked with masked
                areas set to (255, 255, 255).

        """
        self.set_context(I)
        yield
        self.clear_context()

    def set_context(self, I: np.ndarray):
        """Set the whole-slide context for the stain normalizer.

        With contextual normalization, max concentrations are determined
        from the context (whole-slide image) rather than the image being
        normalized. This may improve stain normalization for sections of
        a slide that are predominantly eosin (e.g. necrosis or low cellularity).

        When calculating max concentrations from the image context,
        white pixels (255) will be masked.

        Args:
            I (np.ndarray): Context to use for normalization, e.g.
                a whole-slide image thumbnail, optionally masked with masked
                areas set to (255, 255, 255).

        """
        I = ut.clip_size(I, 2048)
        self._ctx_means, self._ctx_stds = get_mean_std(I, mask=True)

    def clear_context(self):
        """Remove any previously set stain normalizer context."""
        self._ctx_means, self._ctx_stds = None, None


class ReinhardNormalizer(ReinhardFastNormalizer):

    preset_tag = 'reinhard'

    def __init__(self) -> None:
        """Reinhard H&E stain normalizer (numpy implementation).

        Normalizes an image as defined by:

        Reinhard, Erik, et al. "Color transfer between images." IEEE
        Computer graphics and applications 21.5 (2001): 34-41.

        This normalizer contains inspiration from StainTools by Peter Byfield
        (https://github.com/Peter554/StainTools).
        """
        super().__init__()
        self.set_fit(**ut.fit_presets[self.preset_tag]['v3'])  # type: ignore

    def fit(self, target: np.ndarray, mask: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Fit normalizer to a target image.

        Args:
            img (np.ndarray): Target image (RGB uint8) with dimensions W, H, C.
            mask (bool): Ignore white pixels (255, 255, 255) when fitting.
                Defulats to False.

        Returns:
            A tuple containing

                np.ndarray:  Target means (channel means).

                np.ndarray:   Target stds (channel standard deviations).
        """
        target = ut.clip_size(target, 2048)
        target = ut.standardize_brightness(target, mask=mask)
        return super().fit(target, mask=mask)

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

    def transform(
        self,
        I: np.ndarray,
        ctx_means: Optional[np.ndarray] = None,
        ctx_stds: Optional[np.ndarray] = None,
        *,
        augment: bool = False
    ) -> np.ndarray:
        """Normalize an H&E image.

        Args:
            img (np.ndarray): Image, RGB uint8 with dimensions W, H, C.
            ctx_means (np.ndarray, optional): Context channel means (e.g. from
                whole-slide image). If None, calculates means from the image.
                Defaults to None.
            ctx_stds (np.ndarray, optional): Context channel standard deviations
                (e.g. from whole-slide image). If None, calculates standard
                deviations from the image. Defaults to None.

        Keyword args:
            augment (bool): Transform using stain augmentation.
                Defaults to False.

        Returns:
            np.ndarray: Normalized image.
        """
        I = ut.standardize_brightness(I)
        return super().transform(I, ctx_means, ctx_stds, augment=augment)

    def set_context(self, I: np.ndarray):
        """Set the whole-slide context for the stain normalizer.

        With contextual normalization, max concentrations are determined
        from the context (whole-slide image) rather than the image being
        normalized. This may improve stain normalization for sections of
        a slide that are predominantly eosin (e.g. necrosis or low cellularity).

        When calculating max concentrations from the image context,
        white pixels (255) will be masked.

        Args:
            I (np.ndarray): Context to use for normalization, e.g.
                a whole-slide image thumbnail, optionally masked with masked
                areas set to (255, 255, 255).

        """
        I = ut.clip_size(I, 2048)
        I = ut.standardize_brightness(I, mask=True)
        super().set_context(I)

    def clear_context(self):
        """Remove any previously set stain normalizer context."""
        super().clear_context()


class ReinhardFastMaskNormalizer(ReinhardFastNormalizer):

    preset_tag = 'reinhard_fast'

    def __init__(self, threshold: float = 0.93) -> None:
        """Modified Reinhard H&E stain normalizer only applied to
        non-whitepsace areas (numpy implementation).

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
        non-whitepsace areas (numpy implementation).

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
