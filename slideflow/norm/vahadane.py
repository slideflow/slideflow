"""Vahadane H&E stain normalizer."""

from __future__ import division

import cv2
import numpy as np
import joblib
from typing import Dict

import slideflow.norm.utils as ut
from sklearn.decomposition import DictionaryLearning


def get_stain_matrix_spams(
    I: np.ndarray,
    threshold: float = 0.8,
    alpha: float = 0.1,
    num_threads: int = 8,
    fast: bool = False,
) -> np.ndarray:
    """Get 2x3 stain matrix. First row H and second row E.

    Args:
        I (np.ndarray): RGB uint8 image.
        threshold (float): Threshold for determining non-white areas.
        alpha (float): Alpha value for DictionaryLearning.

    Returns:
        np.ndarray:     2x3 stain matrix (first row H, second E)
    """
    import spams

    mask = ut.notwhite_mask(I, thresh=threshold).reshape((-1,))
    OD = ut.RGB_to_OD(I).reshape((-1, 3))
    OD = OD[mask]
    spams_kw = dict(
        numThreads=num_threads,
        K=2,
        lambda1=alpha,
        mode=2,
        modeD=0,
        posD=True)
    if fast:
        dictionary = spams.trainDL_Memory(
            OD.T,
            **spams_kw
        ).T
    else:
        dictionary = spams.trainDL(
            OD.T,
            posAlpha=True,
            verbose=False,
            **spams_kw
        ).T
    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]
    dictionary = ut.normalize_rows(dictionary)
    return dictionary


def get_stain_matrix_sklearn(
    I: np.ndarray,
    threshold: float = 0.8,
    alpha: float = 0.1,
    num_threads: int = 8,
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
    sklearn_kw = dict(
        n_components=2,
        alpha=alpha,
        transform_alpha=alpha,
        fit_algorithm="lars",
        transform_algorithm="lasso_lars",
        positive_dict=True,
        verbose=False,
        max_iter=1000,
        transform_max_iter=1000,
    )
    with joblib.parallel_backend('threading', n_jobs=num_threads):
        dl = DictionaryLearning(**sklearn_kw)
    dictionary = dl.fit_transform(X=OD.T).T

    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]
    dictionary = ut.normalize_rows(dictionary)
    return dictionary


class VahadaneSklearnNormalizer:

    preset_tag = 'vahadane_sklearn'

    def __init__(self, threshold: float = 0.93, num_threads: int = 8) -> None:
        """Vahadane H&E stain normalizer (numpy implementation).

        Normalizes an image as defined by:

        Vahadane, Abhishek, et al. "Structure-preserving color normalization
        and sparse stain separation for histological images."
        IEEE transactions on medical imaging 35.8 (2016): 1962-1971.

        This normalizer contains inspiration from StainTools by Peter Byfield
        (https://github.com/Peter554/StainTools).

        This normalizer uses sklearn's DictionaryLearning for stain matrix
        extraction.
        """
        self.threshold = threshold
        self.num_threads = num_threads
        self.set_fit(**ut.fit_presets[self.preset_tag]['v2'])  # type: ignore

    def get_stain_matrix(self, image: np.ndarray) -> np.ndarray:
        return get_stain_matrix_sklearn(image, num_threads=self.num_threads)

    def fit(self, target: np.ndarray) -> np.ndarray:
        """Fit normalizer to a target image.

        Args:
            img (np.ndarray): Target image (RGB uint8) with dimensions W, H, C.

        Returns:
            stain_matrix_target (np.ndarray):  Stain matrix (H&E)
        """
        target = ut.standardize_brightness(target)
        stain_matrix = self.get_stain_matrix(target)
        self.set_fit(stain_matrix)
        return stain_matrix

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

    def transform(self, I: np.ndarray, *, augment: bool = False) -> np.ndarray:
        """Normalize an H&E image.

        Args:
            img (np.ndarray): Image, RGB uint8 with dimensions W, H, C.

        Returns:
            np.ndarray: Normalized image.
        """
        if augment:
            raise NotImplementedError(
                "Stain augmentation is not implemented for Vahadane normalization"
            )
        if self.stain_matrix_target is None:
            raise ValueError("Normalizer has not been fit: call normalizer.fit()")

        I = ut.standardize_brightness(I)
        I_LAB = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
        mask = (I_LAB[:, :, 0] / 255.0 < self.threshold)[:, :, np.newaxis]
        stain_matrix_source = self.get_stain_matrix(I)
        source_concentrations = ut.get_concentrations(I, stain_matrix_source)
        dot_prod = np.dot(source_concentrations, self.stain_matrix_target)
        normalized = (255 * np.exp(-1 * dot_prod.reshape(I.shape))).astype(np.uint8)
        return np.where(mask, normalized, I)


class VahadaneSpamsNormalizer(VahadaneSklearnNormalizer):

    preset_tag = 'vahadane_spams'

    def __init__(self, *args, **kwargs) -> None:
        """Vahadane H&E stain normalizer (numpy implementation).

        Normalizes an image as defined by:

        Vahadane, Abhishek, et al. "Structure-preserving color normalization
        and sparse stain separation for histological images."
        IEEE transactions on medical imaging 35.8 (2016): 1962-1971.

        This normalizer contains inspiration from StainTools by Peter Byfield
        (https://github.com/Peter554/StainTools).

        This normalizer uses the SPAMS library for stain matrix extraction.
        """
        super().__init__(*args, **kwargs)
        self.set_fit(**ut.fit_presets[self.preset_tag]['v2'])  # type: ignore

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

    def get_stain_matrix(self, image: np.ndarray) -> np.ndarray:
        return get_stain_matrix_spams(image, num_threads=self.num_threads)