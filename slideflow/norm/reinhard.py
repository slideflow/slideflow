"""Reinhard H&E stain normalization."""

from __future__ import division

import os
import cv2
import numpy as np
from typing import Tuple, Dict

import slideflow.norm.utils as ut


def lab_split(I: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert from RGB uint8 to LAB and split into channels

    Args:
        I (np.ndarray): RGB uint8 image.

    Returns:
        np.ndarray: I1, first channel.

        np.ndarray: I2, first channel.

        np.ndarray: I3, first channel.
    """
    I = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
    I = I.astype(np.float32)
    I1, I2, I3 = cv2.split(I)
    I1 /= 2.55
    I2 -= 128.0
    I3 -= 128.0
    return I1, I2, I3


def merge_back(I1: np.ndarray, I2: np.ndarray, I3: np.ndarray) -> np.ndarray:
    """Take seperate LAB channels and merge back to give RGB uint8

    Args:
        I1 (np.ndarray): First channel.
        I2 (np.ndarray): Second channel.
        I3 (np.ndarray): Third channel.

    Returns:
        np.ndarray: RGB uint8 image.
    """
    I1 *= 2.55
    I2 += 128.0
    I3 += 128.0
    I = np.clip(cv2.merge((I1, I2, I3)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(I, cv2.COLOR_LAB2RGB)


def get_mean_std(I: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get mean and standard deviation of each channel.

    Args:
        I (np.ndarray): RGB uint8 image.

    Returns:
        A tuple containing

            np.ndarray:     Channel means, shape = (3,)

            np.ndarray:     Channel standard deviations, shape = (3,)
    """
    I1, I2, I3 = lab_split(I)
    m1, sd1 = cv2.meanStdDev(I1)
    m2, sd2 = cv2.meanStdDev(I2)
    m3, sd3 = cv2.meanStdDev(I3)
    means = m1, m2, m3
    stds = sd1, sd2, sd3
    return np.array(means), np.array(stds)



class ReinhardFastNormalizer:

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
        self.set_fit(**ut.fit_presets['reinhard_fast']['v1'])  # type: ignore

    def fit(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit normalizer to a target image.

        Args:
            img (np.ndarray): Target image (RGB uint8) with dimensions W, H, C.

        Returns:
            A tuple containing

                np.ndarray:  Target means (channel means).

                np.ndarray:   Target stds (channel standard deviations).
        """
        means, stds = get_mean_std(img)
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
        _fit = ut.fit_presets['reinhard_fast'][preset]
        self.set_fit(**_fit)
        return _fit

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


    def transform(self, I: np.ndarray) -> np.ndarray:
        """Normalize an H&E image.

        Args:
            img (np.ndarray): Image, RGB uint8 with dimensions W, H, C.

        Returns:
            np.ndarray: Normalized image.
        """
        if self.target_means is None or self.target_stds is None:
            raise ValueError("Normalizer has not been fit: call normalizer.fit()")

        I1, I2, I3 = lab_split(I)
        means, stds = get_mean_std(I)

        norm1 = ((I1 - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((I2 - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((I3 - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]

        merged = merge_back(norm1, norm2, norm3)
        return merged


class ReinhardNormalizer(ReinhardFastNormalizer):

    def __init__(self) -> None:
        """Reinhard H&E stain normalizer (numpy implementation).

        Normalizes an image as defined by:

        Reinhard, Erik, et al. "Color transfer between images." IEEE
        Computer graphics and applications 21.5 (2001): 34-41.

        This normalizer contains inspiration from StainTools by Peter Byfield
        (https://github.com/Peter554/StainTools).
        """
        super().__init__()
        self.set_fit(**ut.fit_presets['reinhard']['v1'])  # type: ignore

    def fit(self, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit normalizer to a target image.

        Args:
            img (np.ndarray): Target image (RGB uint8) with dimensions W, H, C.

        Returns:
            A tuple containing

                np.ndarray:  Target means (channel means).

                np.ndarray:   Target stds (channel standard deviations).
        """
        target = ut.standardize_brightness(target)
        return super().fit(target)

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

    def transform(self, I: np.ndarray) -> np.ndarray:
        """Normalize an H&E image.

        Args:
            img (np.ndarray): Image, RGB uint8 with dimensions W, H, C.

        Returns:
            np.ndarray: Normalized image.
        """
        I = ut.standardize_brightness(I)
        return super().transform(I)


class ReinhardFastMaskNormalizer(ReinhardFastNormalizer):

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

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Normalize an H&E image.

        Args:
            img (np.ndarray): Image, RGB uint8 with dimensions W, H, C.

        Returns:
            np.ndarray: Normalized image.
        """
        I_LAB = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        mask = (I_LAB[:, :, 0] / 255.0 < self.threshold)[:, :, np.newaxis]
        I1, I2, I3 = lab_split(image)
        means, stds = get_mean_std(image)
        norm1 = ((I1 - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((I2 - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((I3 - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]
        return np.where(mask, merge_back(norm1, norm2, norm3), image)


class ReinhardMaskNormalizer(ReinhardNormalizer):

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

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Normalize an H&E image.

        Args:
            img (np.ndarray): Image, RGB uint8 with dimensions W, H, C.

        Returns:
            np.ndarray: Normalized image.
        """
        image = ut.standardize_brightness(image)
        I_LAB = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        mask = (I_LAB[:, :, 0] / 255.0 < self.threshold)[:, :, np.newaxis]
        I1, I2, I3 = lab_split(image)
        means, stds = get_mean_std(image)
        norm1 = ((I1 - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((I2 - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((I3 - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]
        return np.where(mask, merge_back(norm1, norm2, norm3), image)