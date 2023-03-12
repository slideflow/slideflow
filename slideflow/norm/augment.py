"""HSV stain augmentation."""

from __future__ import division

import cv2 as cv
import numpy as np
from typing import Dict

import slideflow.norm.utils as ut


class AugmentNormalizer:

    def __init__(self):
        """HSV colorspace augmentation.

        Augments an image in the HSV colorspace.
        """
        return

    def get_fit(self) -> Dict[str, np.ndarray]:
        return {}

    def set_fit(self) -> None:
        return

    def fit(self, target: np.ndarray) -> None:
        return

    def fit_preset(self, preset: str) -> None:
        pass

    def transform(self, I: np.ndarray, *, augment = None) -> np.ndarray:
        """Performs HSV colorspace augmentation.

        Args:
            I (np.ndarray): RGB uint8 image (W, H, C)

        Returns:
            np.ndarray: Augmented image.
        """
        hsv = cv.cvtColor(I, cv.COLOR_RGB2HSV)
        hsv[:, :, 2] = cv.equalizeHist(hsv[:, :, 2])
        hsv[:, :, 1] = cv.equalizeHist(hsv[:, :, 1])
        hsv = np.array(hsv, dtype=np.float64)
        hm = np.random.uniform(0.8, 1.2)
        ha = np.random.uniform(-0.2, 0.2)
        sm = np.random.uniform(0.8, 1.2)
        sa = np.random.uniform(-0.2, 0.2)
        vm = np.random.uniform(0.8, 1.2)
        va = np.random.uniform(-0.2, 0.2)
        hsv[:, :, 0] *= hm
        hsv[:, :, 1] *= sm
        hsv[:, :, 2] *= vm
        hsv[:, :, 0] += ha
        hsv[:, :, 1] += sa
        hsv[:, :, 2] += va
        hsv[hsv > 255] = 255
        hsv[hsv < 0] = 0
        hsv = np.array(hsv, dtype=np.uint8)
        img = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
        return img
