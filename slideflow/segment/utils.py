import torch
import torch.nn.functional as F
import numpy as np

from ._cp_utils import make_tiles, average_tiles, outlines_list

# -----------------------------------------------------------------------------

def topleft_pad(img, size, padval=0):
    """Pad an image to the top-left.

    Args:
        img (np.ndarray or torch.Tensor): The image to pad, in the form (W, H, C).
        size (int): The target size (height/width).
        padval (int): The value to pad with.

    """
    if isinstance(img, np.ndarray):
        return topleft_pad_numpy(img, size, padval)
    elif isinstance(img, torch.Tensor):
        return topleft_pad_torch(img, size, padval)
    else:
        raise ValueError(f"Unknown image type: {type(img)}")

def center_square_pad(img, size, padval=0):
    """Pad an image to the center.

    Args:
        img (np.ndarray or torch.Tensor): The image to pad, in the form (W, H, C).
        size (int): The target size (height/width).
        padval (int): The value to pad with.

    """
    if isinstance(img, np.ndarray):
        return center_square_pad_numpy(img, size, padval)
    elif isinstance(img, torch.Tensor):
        return center_square_pad_torch(img, size, padval)
    else:
        raise ValueError(f"Unknown image type: {type(img)}")

def _get_center_padding(img, size):
    """Get the padding required to center an image.

    Args:
        img: The image to pad, in the form (W, H, C).
        size: The target size.

    Returns:
        The padding required to center the image.

    """
    if img.shape[0] < size:
        pad_xi = pad_xj = int((size - img.shape[0]) / 2)
        if (pad_xi * 2) + img.shape[0] < size:
            pad_xj += 1
    else:
        pad_xi = pad_xj = 0
    if img.shape[1] < size:
        pad_yi = pad_yj = int((size - img.shape[1]) / 2)
        if (pad_yi * 2) + img.shape[1] < size:
            pad_yj += 1
    else:
        pad_yi = pad_yj = 0

    return pad_xi, pad_xj, pad_yi, pad_yj

# -----------------------------------------------------------------------------

# Numpy functions for padding images.

def center_square_pad_numpy(img, size, padval=0):
    """Pad an image to the center.

    Args:
        img (np.ndarray): The image to pad, in the form (W, H, C).
        size (int): The target size (height/width).
        padval (int): The value to pad with.

    """
    pad_xi, pad_xj, pad_yi, pad_yj = _get_center_padding(img, size)
    padded = np.pad(
        img,
        ((pad_xi, pad_xj), (pad_yi, pad_yj), (0, 0)),
        mode='constant',
        constant_values=padval
    )
    return padded

def topleft_pad_numpy(img, size, padval=0):
    """Pad an image to the top-left.

    Args:
        img (np.ndarray): The image to pad, in the form (W, H, C).
        size (int): The target size (height/width).
        padval (int): The value to pad with.

    """
    # Pad to target size.
    if img.shape[0] < size:
        pad_x = (size - img.shape[0])
    else:
        pad_x = 0
    if img.shape[1] < size:
        pad_y = (size - img.shape[1])
    else:
        pad_y = 0
    padded = np.pad(img, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant', constant_values=padval)
    return padded

# -----------------------------------------------------------------------------

# PyTorch functions for padding images.

def center_square_pad_torch(img, size, padval=0):
    """Pad an image to the center.

    Args:
        img (torch.Tensor): The image to pad, in the form (W, H, C).
        size (int): The target size (height/width).
        padval (int): The value to pad with.

    """
    pad_xi, pad_xj, pad_yi, pad_yj = _get_center_padding(img, size)

    # PyTorch requires padding in the form (pad_channel_start, pad_channel_end, pad_top, pad_bottom, pad_left, pad_right).
    padded = F.pad(img, (0, 0, pad_yi, pad_yj, pad_xi, pad_xj), mode='constant', value=padval)

    return padded

def topleft_pad_torch(img, size, padval=0):
    """Pad an image to the top-left.

    Args:
        img (torch.Tensor): The image to pad, in the form (W, H, C).
        size (int): The target size (height/width).
        padval (int): The value to pad with.

    """
    # Pad to target size.
    if img.shape[0] < size:
        pad_x = (size - img.shape[0])
    else:
        pad_x = 0
    if img.shape[1] < size:
        pad_y = (size - img.shape[1])
    else:
        pad_y = 0

    # PyTorch requires padding in the form (pad_channel_start, pad_channel_end, pad_top, pad_bottom, pad_left, pad_right).
    padded = F.pad(img, (0, 0, 0, pad_y, 0, pad_x), mode='constant', value=padval)

    return padded