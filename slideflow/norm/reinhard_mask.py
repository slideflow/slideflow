"""
From https://github.com/wanghao14/Stain_Normalization
Normalize a patch stain to the target image using the method of:

E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.
"""

from __future__ import division

import os
import cv2
from typing import Tuple, Dict

import cv2 as cv
import numpy as np

import slideflow.norm.utils as ut

from slideflow.norm.reinhard_fast import lab_split, merge_back, get_mean_std
from slideflow.norm.reinhard import ReinhardNormalizer


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
        I_LAB = cv.cvtColor(image, cv.COLOR_RGB2LAB)
        I_LAB[:, :, 1] = I_LAB[:, :, 0]
        I_LAB[:, :, 2] = I_LAB[:, :, 0]
        mask = I_LAB[:, :, :] / 255.0 < self.threshold
        I1, I2, I3 = lab_split(image)
        means, stds = get_mean_std(image)
        norm1 = ((I1 - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((I2 - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((I3 - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]
        return np.where(mask, merge_back(norm1, norm2, norm3), image)
