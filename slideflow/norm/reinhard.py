"""Reinhard H&E stain normalization."""

from typing import Tuple

import numpy as np

import slideflow.norm.utils as ut
from slideflow.norm.reinhard_fast import ReinhardFastNormalizer


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

    def fit(self, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit normalizer to a target image.

        Args:
            img (np.ndarray): Target image (RGB uint8) with dimensions W, H, C.

        Returns:
            target_means (np.ndarray):  Channel means.

            target_stds (np.ndarray):   Channel standard deviations.
        """
        target = ut.standardize_brightness(target)
        return super().fit(target)

    def transform(self, I: np.ndarray) -> np.ndarray:
        """Normalize an H&E image.

        Args:
            img (np.ndarray): Image, RGB uint8 with dimensions W, H, C.

        Returns:
            np.ndarray: Normalized image.
        """
        I = ut.standardize_brightness(I)
        return super().transform(I)