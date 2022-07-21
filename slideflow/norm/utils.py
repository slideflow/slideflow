"""
From https://github.com/wanghao14/Stain_Normalization
Uses the spams package:

http://spams-devel.gforge.inria.fr/index.html

Use with python via e.g https://anaconda.org/conda-forge/python-spams
"""

from __future__ import division

from typing import Union, List
from os.path import dirname, abspath, join
import cv2 as cv
import numpy as np


# Stain normalizer default fits.
# v1 is the fit with target sf.norm.norm_tile.jpg (default for version < 1.3)
# v2 is a hand-tuned fit (default for version >= 1.3)

fit_presets = {
    'reinhard': {
        'v1': {'target_means': np.array([ 72.272896,  22.99831 , -13.860236]),
               'target_stds': np.array([15.594496,  9.642087,  9.290526])},
        'v2': {'target_means': np.array([72.909996, 20.8268, -4.9465137]),
               'target_stds': np.array([18.560713, 14.889295,  5.6756697])}
    },
    'reinhard_fast': {
        'v1': {'target_means': np.array([63.71194 ,  20.716246, -12.290746]),
               'target_stds': np.array([14.52781 ,  8.344005,  8.300264])},
        'v2': {'target_means': np.array([69.20197, 19.82498, -4.690998]),
               'target_stds': np.array([17.71583, 14.156416,  5.4176064])},
    },
    'macenko': {
        'v1': {'stain_matrix_target': np.array([[0.63111544, 0.24816133],
                                                [0.6962834 , 0.8226449 ],
                                                [0.34188122, 0.5115382 ]]),
               'target_concentrations': np.array([1.4423684, 0.9685806])},
        'v2': {'stain_matrix_target': np.array([[0.5626, 0.2159],
                                                [0.7201, 0.8012],
                                                [0.4062, 0.5581]]),
               'target_concentrations': np.array([1.9705, 1.0308])},
    },
    'vahadane_sklearn': {
        'v1': {'stain_matrix_target': np.array([[0.9840825 , 0.17771211, 0.        ],
                                                [0.        , 0.87096226, 0.49134994]])},
        'v2': {'stain_matrix_target': np.array([[0.95465684, 0.29770842, 0.        ],
                                                [0.        , 0.8053334 , 0.59282213]])},
    },
    'vahadane_spams': {
        'v1': {'stain_matrix_target': np.array([[0.54176575, 0.75441414, 0.37060648],
                                                [0.17089975, 0.8640189 , 0.4735658 ]])},
        'v2': {'stain_matrix_target': np.array([[0.4435433 , 0.7502863 , 0.4902447 ],
                                                [0.27688965, 0.8088818 , 0.5186929 ]])},
    }
}

######################################

def standardize_brightness(I):
    """

    :param I:
    :return:
    """
    p = np.percentile(I, 90)
    return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)


def remove_zeros(I):
    """
    Remove zeros, replace with 1's.
    :param I: uint8 array
    :return:
    """
    mask = (I == 0)
    I[mask] = 1
    return I


def RGB_to_OD(I):
    """
    Convert from RGB to optical density
    :param I:
    :return:
    """
    I = remove_zeros(I)
    return -1 * np.log(I / 255)


def OD_to_RGB(OD):
    """
    Convert from optical density to RGB
    :param OD:
    :return:
    """
    return (255 * np.exp(-1 * OD)).astype(np.uint8)


def normalize_rows(A):
    """
    Normalize rows of an array
    :param A:
    :return:
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


def notwhite_mask(I, thresh=0.8):
    """
    Get a binary mask where true denotes 'not white'
    :param I:
    :param thresh:
    :return:
    """
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0
    return (L < thresh)


def sign(x):
    """
    Returns the sign of x
    :param x:
    :return:
    """
    if x > 0:
        return +1
    elif x < 0:
        return -1
    elif x == 0:
        return 0


def get_concentrations(I, stain_matrix, lamda=0.01):
    """
    Get concentrations, a npix x 2 matrix
    :param I:
    :param stain_matrix: a 2x3 stain matrix
    :return:
    """

    OD = RGB_to_OD(I).reshape((-1, 3))

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(stain_matrix.T, Y, rcond=None)[0]
    return C.T

def _as_numpy(arg1: Union[List, np.ndarray]) -> np.ndarray:
    """Ensures array is a numpy array."""

    if isinstance(arg1, list):
        return np.squeeze(np.array(arg1)).astype(np.float32)
    elif isinstance(arg1, np.ndarray):
        return np.squeeze(arg1).astype(np.float32)
    else:
        raise ValueError(f'Expected numpy array; got {type(arg1)}')