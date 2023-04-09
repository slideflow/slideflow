"""
From https://github.com/wanghao14/Stain_Normalization
Uses the spams package:

http://spams-devel.gforge.inria.fr/index.html

Use with python via e.g https://anaconda.org/conda-forge/python-spams
"""

from __future__ import division

import cv2
import numpy as np
from typing import Union, List, Tuple

# -----------------------------------------------------------------------------

# Stain normalizer default fits.
# v1 is the fit with target sf.norm.norm_tile.jpg (default in version <1.6)
# v2 is a hand-tuned fit
# v3 is fit using an average of ~50k tiles across ~450 slides from TCGA (default for versions >=1.6)

fit_presets = {
    'reinhard': {
        'v1': {'target_means': np.array([ 72.272896,  22.99831 , -13.860236]),
               'target_stds': np.array([15.594496,  9.642087,  9.290526])},
        'v2': {'target_means': np.array([72.909996, 20.8268, -4.9465137]),
               'target_stds': np.array([18.560713, 14.889295,  5.6756697])},
        'v3': {'target_means': np.array([65.22132,  28.934267, -14.142519]),
               'target_stds': np.array([15.800227,  9.263783,  6.0213304])}
    },
    'reinhard_fast': {
        'v1': {'target_means': np.array([63.71194 ,  20.716246, -12.290746]),
               'target_stds': np.array([14.52781 ,  8.344005,  8.300264])},
        'v2': {'target_means': np.array([69.20197, 19.82498, -4.690998]),
               'target_stds': np.array([17.71583, 14.156416,  5.4176064])},
        'v3': {'target_means': np.array([58.12343,  26.483482, -12.701005]),
               'target_stds': np.array([14.675022,  7.5744166,  5.226378])},
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
        'v3': {'stain_matrix_target': np.array([[0.5062568, 0.2218694],
                                                [0.75322306, 0.8652155],
                                                [0.40691733, 0.42241502]]),
               'target_concentrations': np.array([1.7656903, 1.2797493])},
    },
    'macenko_fast': {
        'v1': {'stain_matrix_target': np.array([[0.6148019 , 0.21480364],
                                                [0.7010872 , 0.82317936],
                                                [0.36124164, 0.5255809 ]]),
               'target_concentrations': np.array([1.8029537, 0.9606744])},
        'v2': {'stain_matrix_target': np.array([[0.5626, 0.2159],
                                                [0.7201, 0.8012],
                                                [0.4062, 0.5581]]),
               'target_concentrations': np.array([1.9705, 1.0308])},
        'v3': {'stain_matrix_target': np.array([[0.52000326, 0.2623537 ],
                                                [0.73508584, 0.83495414],
                                                [0.4249617 , 0.4630997 ]]),
               'target_concentrations': np.array([2.0259454, 1.4088874])},
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

# Stain normalizer default augmentation spaces.
# v1 is derived from the standard deviation of fit values for ~50k tiles from ~450 slides in TCGA.

augment_presets = {
    'reinhard': {
        'v1': {'means_stdev': np.array([1.1882676, 1.3114343, 1.1200949]) * 5,
                'stds_stdev': np.array([0.5123385 , 0.37919158, 0.26019168]) * 5}
    },
    'reinhard_fast': {
        'v1': {'means_stdev': np.array([1.2963034 , 1.0061347 , 0.90867484]) * 5,
                'stds_stdev': np.array([0.47548684, 0.3956356 , 0.23499836]) * 5}
    },
    'macenko': {
        'v1': {'matrix_stdev': np.array([[0.00893346, 0.01153686],
                                         [0.00659814, 0.00722771],
                                         [0.00726339, 0.01352414]]) * 5,
               'concentrations_stdev': np.array([0.06665898, 0.06770515]) * 5}
    },
    'macenko_fast': {
        'v1': {'matrix_stdev': np.array([[0.00794701, 0.01137106],
                                         [0.00559027, 0.00642623],
                                         [0.00609103, 0.01144302]]) * 5,
               'concentrations_stdev': np.array([0.06623945, 0.08137263]) * 5}
    }
}

# -----------------------------------------------------------------------------

illuminants = {
    "A": {
        "2": (1.098466069456375, 1, 0.3558228003436005),
        "10": (1.111420406956693, 1, 0.3519978321919493),
    },
    "D50": {
        "2": (0.9642119944211994, 1, 0.8251882845188288),
        "10": (0.9672062750333777, 1, 0.8142801513128616),
    },
    "D55": {
        "2": (0.956797052643698, 1, 0.9214805860173273),
        "10": (0.9579665682254781, 1, 0.9092525159847462),
    },
    "D65": {
        "2": (0.95047, 1.0, 1.08883),
        "10": (0.94809667673716, 1, 1.0730513595166162),
    },
    "D75": {
        "2": (0.9497220898840717, 1, 1.226393520724154),
        "10": (0.9441713925645873, 1, 1.2064272211720228),
    },
    "E": {"2": (1.0, 1.0, 1.0), "10": (1.0, 1.0, 1.0)},
}

rgb_to_xyz_kernels = {
    dtype: np.array(
        [
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227],
        ],
        dtype=dtype,
    ) for dtype in ('float16', 'float32', 'float64')
}

# inv of:
# [[0.412453, 0.35758 , 0.180423],
#  [0.212671, 0.71516 , 0.072169],
#  [0.019334, 0.119193, 0.950227]]
xyz_to_rgb_kernels = {
    dtype: np.array(
        [
            [3.24048134, -1.53715152, -0.49853633],
            [-0.96925495, 1.87599, 0.04155593],
            [0.05564664, -0.20404134, 1.05731107],
        ],
        dtype=dtype,
    ) for dtype in ('float16', 'float32', 'float64')
}

######################################


def brightness_percentile(I):
    return np.percentile(I, 90)


def standardize_brightness(I, mask=False):
    """

    :param I:
    :return:
    """
    if mask:
        ones = np.all(I == 255, axis=len(I.shape)-1)
    bI = I if not mask else I[~ ones]
    p = brightness_percentile(bI)
    clipped = np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)
    if mask:
        clipped[ones] = 255
    return clipped


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
    return -1 * np.log(I / 255).astype(np.float32)


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
    I_LAB = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
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


def clip_size(I, max_size=2048):
    # Cap the context size to a maximum of (2048, 2048).
    if I.shape[0] > max_size or I.shape[1] > max_size:
        w, h = I.shape[0], I.shape[1]
        if w > h:
            h = int((h / w) * max_size)
            w = max_size
        else:
            w = int((w / h) * max_size)
            h = max_size
        I = cv2.resize(I, (h, w))
    return I


def _as_numpy(arg1: Union[List, np.ndarray]) -> np.ndarray:
    """Ensures array is a numpy array."""

    if isinstance(arg1, list):
        return np.squeeze(np.array(arg1)).astype(np.float32)
    elif isinstance(arg1, np.ndarray):
        return np.squeeze(arg1).astype(np.float32)
    else:
        raise ValueError(f'Expected numpy array; got {type(arg1)}')

# =============================================================================

import numpy as np


def unstack(a, axis = 0):
    return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis = axis)]


def rgb_to_xyz(input):
    """
    Convert a RGB image to CIE XYZ.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    assert input.dtype in (np.float16, np.float32, np.float64)

    kernel = rgb_to_xyz_kernels[str(input.dtype)]
    value = np.where(
        input > 0.04045,
        np.power((input + 0.055) / 1.055, 2.4),
        input / 12.92,
    )
    return np.tensordot(value, np.transpose(kernel), axes=((-1,), (0,)))


def xyz_to_rgb(input):
    """
    Convert a CIE XYZ image to RGB.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    assert input.dtype in (np.float16, np.float32, np.float64)

    kernel = xyz_to_rgb_kernels[str(input.dtype)]
    value = np.tensordot(input, np.transpose(kernel), axes=((-1,), (0,)))
    value = np.where(
        value > 0.0031308,
        np.power(np.clip(value, 0, None), 1.0 / 2.4) * 1.055 - 0.055,
        value * 12.92,
    )
    return np.clip(value, 0, 1)


def lab_to_rgb(input, illuminant="D65", observer="2"):
    """
    Convert a CIE LAB image to RGB.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
      observer : {"2", "10"}, optional
        The aperture angle of the observer.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    assert input.dtype in (np.float16, np.float32, np.float64)

    lab = input
    lab = unstack(lab, axis=-1)
    l, a, b = lab[0], lab[1], lab[2]

    y = (l + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)

    z = np.clip(z, 0, None)

    xyz = np.stack([x, y, z], axis=-1)

    xyz = np.where(
        xyz > 0.2068966,
        np.power(xyz, 3.0),
        (xyz - 16.0 / 116.0) / 7.787,
    )

    coords = np.array(illuminants[illuminant.upper()][observer], input.dtype)

    xyz = xyz * coords

    return xyz_to_rgb(xyz)


def rgb_to_lab(input, illuminant="D65", observer="2"):
    """
    Convert a RGB image to CIE LAB.
    Args:
      input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
      observer : {"2", "10"}, optional
        The aperture angle of the observer.
      name: A name for the operation (optional).
    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    assert input.dtype in (np.float16, np.float32, np.float64)

    coords = np.array(illuminants[illuminant.upper()][observer], input.dtype)

    xyz = rgb_to_xyz(input)

    xyz = xyz / coords

    xyz = np.where(
        xyz > 0.008856,
        np.power(xyz, 1.0 / 3.0),
        xyz * 7.787 + 16.0 / 116.0,
    )

    xyz = unstack(xyz, axis=-1)
    x, y, z = xyz[0], xyz[1], xyz[2]

    # Vector scaling
    L = (y * 116.0) - 16.0
    A = (x - y) * 500.0
    B = (y - z) * 200.0

    return np.stack([L, A, B], axis=-1)

# -----------------------------------------------------------------------------


# --- Numpy and CV2-based LAB-RGB utility functions. -----------------------------

def merge_back_cv2(I1: np.ndarray, I2: np.ndarray, I3: np.ndarray) -> np.ndarray:
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

def lab_split_cv2(I: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

# -----------------------------------------------------------------------------

def lab_split_numpy(I: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert from RGB uint8 to LAB and split into channels

    Args:
        I (np.ndarray): RGB uint8 image.

    Returns:
        np.ndarray: I1, first channel.

        np.ndarray: I2, first channel.

        np.ndarray: I3, first channel.
    """
    I = I.astype(np.float32)
    I /= 255
    I = rgb_to_lab(I)
    return unstack(I, axis=-1)


def merge_back_numpy(I1: np.ndarray, I2: np.ndarray, I3: np.ndarray) -> np.ndarray:
    """Take seperate LAB channels and merge back to give RGB uint8

    Args:
        I1 (np.ndarray): First channel.
        I2 (np.ndarray): Second channel.
        I3 (np.ndarray): Third channel.

    Returns:
        np.ndarray: RGB uint8 image.
    """
    I = np.stack((I1, I2, I3), axis=-1)
    I = lab_to_rgb(I) * 255
    I = I.astype(np.int32)
    I = np.clip(I, 0, 255).astype(np.uint8)
    return I
