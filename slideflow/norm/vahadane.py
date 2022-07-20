"""Vahadane H&E stain normalizer."""

from __future__ import division

import os
import cv2
import numpy as np
from typing import Dict

import slideflow.norm.utils as ut
from sklearn.decomposition import DictionaryLearning


def get_stain_matrix(
    I: np.ndarray,
    threshold: float = 0.8,
    alpha: float = 0.1
) -> np.ndarray:
    """Get 2x3 stain matrix. First row H and second row E.

    Args:
        I (np.ndarray): RGB uint8 image.
        threshold (float): Threshold for determining non-white areas.
        alpha (float): Alpha value for DictionaryLearning.

    Returns:
        np.ndarray:     2x3 stain matrix (first row H, second E)
    """
    mask = ut.notwhite_mask(I, thresh=threshold).reshape((-1,))
    OD = ut.RGB_to_OD(I).reshape((-1, 3))
    OD = OD[mask]
    dl = DictionaryLearning(
        n_components=2,
        alpha=alpha,
        transform_alpha=alpha,
        fit_algorithm="lars",
        transform_algorithm="lasso_lars",
        positive_dict=True,
        verbose=False,
        max_iter=3,
        transform_max_iter=1000,
    )
    dictionary = dl.fit_transform(X=OD.T).T

    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]
    dictionary = ut.normalize_rows(dictionary)
    return dictionary


class VahadaneNormalizer:

    def __init__(self) -> None:
        """Vahadane H&E stain normalizer (numpy implementation).

        Normalizes an image as defined by:

        Vahadane, Abhishek, et al. "Structure-preserving color normalization
        and sparse stain separation for histological images."
        IEEE transactions on medical imaging 35.8 (2016): 1962-1971.

        This normalizer contains inspiration from StainTools by Peter Byfield
        (https://github.com/Peter554/StainTools).
        """
        package_directory = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(package_directory, 'norm_tile.jpg')
        self.fit(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))

    def fit(self, target: np.ndarray) -> None:
        """Fit normalizer to a target image.

        Args:
            img (np.ndarray): Target image (RGB uint8) with dimensions W, H, C.

        Returns:
            stain_matrix_target (np.ndarray):  Stain matrix (H&E)
        """
        target = ut.standardize_brightness(target)
        self.stain_matrix_target = get_stain_matrix(target)

    def get_fit(self) -> Dict[str, np.ndarray]:
        """Get the current normalizer fit.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping 'stain_matrix_target'
                to its respective fit value.
        """
        return {
            'stain_matrix_target': self.stain_matrix_target,
        }

    def set_fit(
        self,
        stain_matrix_target: np.ndarray,
    ) -> None:
        """Set the normalizer fit to the given values.

        Args:
            stain_matrix_target (np.ndarray): H&E stain matrix target. Must
                have the shape (2,3).
        """
        stain_matrix_target = ut._as_numpy(stain_matrix_target)
        if stain_matrix_target.shape != (2, 3):
            raise ValueError("stain_matrix_target must have shape of (2, 3) - "
                             f"got {stain_matrix_target.shape}")
        self.stain_matrix_target = stain_matrix_target

    def transform(self, I: np.ndarray) -> np.ndarray:
        """Normalize an H&E image.

        Args:
            img (np.ndarray): Image, RGB uint8 with dimensions W, H, C.

        Returns:
            np.ndarray: Normalized image.
        """

        if self.stain_matrix_target is None:
            raise ValueError("Normalizer has not been fit: call normalizer.fit()")

        I = ut.standardize_brightness(I)
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = ut.get_concentrations(I, stain_matrix_source)
        return (255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target).reshape(I.shape))).astype(
            np.uint8)
