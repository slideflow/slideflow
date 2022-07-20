"""
From https://github.com/wanghao14/Stain_Normalization
Stain normalization inspired by method of:

A. Vahadane et al., ‘Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images’, IEEE Transactions on Medical Imaging, vol. 35, no. 8, pp. 1962–1971, Aug. 2016.

Uses the spams package:

http://spams-devel.gforge.inria.fr/index.html

Use with python via e.g https://anaconda.org/conda-forge/python-spams
"""

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
    lamda: float = 0.1
) -> np.ndarray:
    """
    Get 2x3 stain matrix. First row H and second row E
    :param I:
    :param threshold:
    :param lamda:
    :return:
    """
    mask = ut.notwhite_mask(I, thresh=threshold).reshape((-1,))
    OD = ut.RGB_to_OD(I).reshape((-1, 3))
    OD = OD[mask]
    dl = DictionaryLearning(
        n_components=2,
        alpha=0.1,
        transform_alpha=0.1,
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


###

class VahadaneNormalizer:
    """
    A stain normalization object
    """

    def __init__(self) -> None:
        package_directory = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(package_directory, 'norm_tile.jpg')
        self.fit(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))

    def fit(self, target: np.ndarray) -> None:
        target = ut.standardize_brightness(target)
        self.stain_matrix_target = get_stain_matrix(target)

    def get_fit(self) -> Dict[str, np.ndarray]:
        return {
            'stain_matrix_target': self.stain_matrix_target,
        }

    def set_fit(
        self,
        stain_matrix_target: np.ndarray,
    ) -> None:
        self.stain_matrix_target = ut._as_numpy(stain_matrix_target)

    def target_stains(self) -> np.ndarray:
        return ut.OD_to_RGB(self.stain_matrix_target)

    def transform(self, I: np.ndarray) -> np.ndarray:

        if self.stain_matrix_target is None:
            raise ValueError("Normalizer has not been fit: call normalizer.fit()")

        I = ut.standardize_brightness(I)
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = ut.get_concentrations(I, stain_matrix_source)
        return (255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target).reshape(I.shape))).astype(
            np.uint8)

    def hematoxylin(self, I: np.ndarray) -> np.ndarray:
        I = ut.standardize_brightness(I)
        h, w, c = I.shape
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = ut.get_concentrations(I, stain_matrix_source)
        H = source_concentrations[:, 0].reshape(h, w)
        H = np.exp(-1 * H)
        return H
