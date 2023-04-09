from __future__ import absolute_import

import imghdr
import io
import os
import struct
import sys
import numpy as np
from typing import List, Optional, Tuple, Any, Union, TYPE_CHECKING

from slideflow import errors, log
from slideflow.util import tfrecord2idx

if TYPE_CHECKING:
    import tensorflow as tf
    import torch


def _np_float_to_uint8(img):
    return ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)


def _np_uint8_to_float(img):
    return ((img.astype(np.float32) / 127.5) - 1)


def _is_np_uint8(img):
    return isinstance(img, np.ndarray) and img.dtype == np.uint8


def _is_np_float32(img):
    return isinstance(img, np.ndarray) and img.dtype == np.float32


def _is_tf_uint8(img):
    import tensorflow as tf
    return isinstance(img, tf.Tensor) and img.dtype == tf.uint8


def _is_tf_float(img):
    import tensorflow as tf
    return (isinstance(img, tf.Tensor) and
            img.dtype == tf.float16 or img.dtype == tf.float32)


def _is_torch_uint8(img):
    import torch
    return isinstance(img, torch.Tensor) and img.dtype == torch.uint8


def _is_torch_float(img):
    import torch
    return (isinstance(img, torch.Tensor) and
            img.dtype == torch.float16 or img.dtype == torch.float32)


def detect_tfrecord_format(tfr: str) -> Tuple[Optional[List[str]],
                                              Optional[str]]:
    '''Detects tfrecord format.

    Args:
        tfr (str): Path to tfrecord.

    Returns:
        A tuple containing

            list(str): List of detected features.

            str: Image file type (png/jpeg)
    '''
    try:
        record = tfrecord2idx.get_tfrecord_by_index(tfr, index=0)
    except errors.EmptyTFRecordsError:
        log.debug(f"Unable to detect format for {tfr}; file empty.")
        return None, None
    img_type = imghdr.what('', record['image_raw'])
    return list(record.keys()), img_type


def convert_dtype(
    img: Any,
    dtype: Union[np.dtype, "tf.dtypes.DType", "torch.dtype"]
) -> Any:
    """Converts an image from one type to another.

    Images can be converted to and from numpy arrays, Torch Tensors and
    Tensorflow Tensors. Images can also be converted from standardized
    float images to RGB uint8 images, and vice versa.

    Supported formats for starting and ending dtype include:

    .. list-table::
        :widths: 20 80
        :header-rows: 0

        * - ``np.uint8``
          - Image in RGB (WHC) uint8 format.
        * - ``np.float32``
          - RGB (WHC) image. If the source image is a numpy uint8 or torch uint8,
            it will be standardized with ``(img / 127.5) - 1``.
            If the source image is a tensorflow image,
            standardization uses ``tf.image.per_image_standardization()``.
        * - ``torch.uint8``
          - Image in RGB (CWH) uint8 format.
        * - ``torch.float32``
          - Image converted with ``(img / 127.5) - 1`` and WHC -> CWH.
        * - ``tf.uint8``
          - Image in RGB (WHC) uint8 format.
        * - ``tf.float32``
          - Image converted with ``tf.image.per_image_standardization()``

    Args:
        img (Any): Input image or batch of images.
        start_dtype (type): Starting dtype.
        end_dtype (type): Target dtype for conversion.

    Returns:
        Converted image or batch of images.
    """

    # Import necessary packages
    if 'tensorflow' in sys.modules:
        import tensorflow as tf
    if 'torch' in sys.modules:
        import torch
        from slideflow.io.torch import cwh_to_whc, whc_to_cwh

    # Verify dtypes are valid
    def _valid_dtype(_dtype):
        if 'tensorflow' in sys.modules:
            if _dtype in (tf.uint8, tf.float32, tf.float16):
                return True
        if 'torch' in sys.modules:
            if _dtype in (torch.uint8, torch.float32, torch.float16):
                return True
        return _dtype in (np.uint8, np.float32)

    _valid_str = ("np.uint8, np.float32, "
                  "tf.uint8, tf.float16, tf.float32, "
                  "torch.uint8, torch.float16, torch.float32")
    if not _valid_dtype(dtype):
        raise ValueError(f"Unrecognized dtype {dtype}. Expected: {_valid_str}")
    if not _valid_dtype(img.dtype):
        raise ValueError(f"Image has unrecognized dtype {img.dtype}. "
                         f"Expected: {_valid_str}")

    # --- np.uint8 conversions ------------------------------------------------
    elif _is_np_uint8(img):

        if dtype is np.uint8:
            return img

        if dtype is np.float32:
            return _np_uint8_to_float(img)

        if 'torch' in sys.modules and dtype is torch.uint8:
            return whc_to_cwh(torch.from_numpy(img))

        if 'torch' in sys.modules and dtype in (torch.float16, torch.float32):
            assert isinstance(dtype, torch._C.dtype)
            return (whc_to_cwh(torch.from_numpy(img).to(dtype)) / 127.5) - 1

        if 'tensorflow' in sys.modules and dtype is tf.uint8:
            return tf.convert_to_tensor(img, dtype=tf.uint8)

        if 'tensorflow' in sys.modules and dtype in (tf.float16, tf.float32):
            return tf.cast(
                tf.image.per_image_standardization(
                    tf.convert_to_tensor(img, dtype=tf.uint8)), dtype)

    # --- np.float32 conversions ----------------------------------------------
    elif _is_np_float32(img):

        if dtype is np.float32:
            return img

        if dtype is np.uint8:
            return _np_float_to_uint8(img)

        if 'torch' in sys.modules and dtype is torch.uint8:
            return whc_to_cwh(torch.from_numpy(_np_float_to_uint8(img)))

        if 'torch' in sys.modules and dtype in (torch.float16, torch.float32):
            assert isinstance(dtype, torch._C.dtype)
            return whc_to_cwh(torch.from_numpy(img).to(dtype))

        if 'tensorflow' in sys.modules and dtype is tf.uint8:
            return tf.convert_to_tensor(_np_float_to_uint8(img))

        if 'tensorflow' in sys.modules and dtype in (tf.float16, tf.float32):
            return tf.cast(
                tf.image.per_image_standardization(
                    tf.convert_to_tensor(_np_float_to_uint8(img))), dtype)

    # --- torch.uint8 conversions ---------------------------------------------
    elif 'torch' in sys.modules and _is_torch_uint8(img):

        if dtype is torch.uint8:
            return img

        if dtype is np.uint8:
            return img.cpu().numpy()

        if dtype is np.float32:
            return _np_uint8_to_float(img.cpu().numpy())

        if dtype in (torch.float16, torch.float32):
            return (img.to(dtype) / 127.5) - 1

        if 'tensorflow' in sys.modules and dtype is tf.uint8:
            return tf.convert_to_tensor(cwh_to_whc(img).cpu().numpy())

        if 'tensorflow' in sys.modules and dtype in (tf.float16, tf.float32):
            return tf.cast(
                tf.image.per_image_standardization(
                    tf.convert_to_tensor(cwh_to_whc(img).cpu().numpy())), dtype)

    # --- torch.float32 conversions -------------------------------------------
    elif 'torch' in sys.modules and _is_torch_float(img):

        if dtype in (torch.float16, torch.float32) and dtype == img.dtype:
            return img

        if dtype is np.uint8:
            return _np_float_to_uint8(cwh_to_whc(img).cpu().numpy())

        if dtype is np.float32:
            return cwh_to_whc(img).cpu().numpy()

        if dtype is torch.uint8:
            return ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)

        if 'tensorflow' in sys.modules and dtype is tf.uint8:
            return tf.convert_to_tensor(
                cwh_to_whc(
                    ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)).cpu().numpy())

        if 'tensorflow' in sys.modules and dtype in (tf.float16, tf.float32):
            return tf.cast(
                tf.image.per_image_standardization(
                    tf.convert_to_tensor(
                        cwh_to_whc(
                            ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)).cpu().numpy())), dtype)

    # --- tf.uint8 conversions ------------------------------------------------
    elif 'tensorflow' in sys.modules and _is_tf_uint8(img):

        if dtype is tf.uint8:
            return img

        if dtype is np.uint8:
            return img.numpy()

        if dtype is np.float32:
            return tf.cast(
                tf.image.per_image_standardization(img), tf.float32).numpy()

        if 'torch' in sys.modules and dtype in (torch.float16, torch.float32):
            assert isinstance(dtype, torch._C.dtype)
            return (torch.from_numpy(img.numpy()).to(dtype) / 127.5) - 1

        if 'torch' in sys.modules and dtype is torch.uint8:
            return torch.from_numpy(img.numpy())

        if dtype in (tf.float16, tf.float32):
            return tf.cast(
                tf.image.per_image_standardization(img), dtype)

    # --- tf.float32 conversions ----------------------------------------------
    elif 'tensorflow' in sys.modules and _is_tf_float(img):

        if dtype in (tf.float16, tf.float32) and dtype == img.dtype:
            return img

        if dtype is np.float32:
            return img.numpy()

        if (dtype in (tf.uint8, np.uint8)
           or ('torch' in sys.modules and dtype is torch.uint8)):
            raise ValueError(
                "Unable to convert standardized Tensorflow tensors to "
                "uint8 (Tensorflow standardization is uni-directional)")

        if 'torch' in sys.modules and dtype in (torch.float16, torch.float32):
            raise ValueError(
                "Unable to convert standardized Tensorflow tensors to "
                "PyTorch-standardized tensors (Tensorflow standardization is "
                "uni-directional)")

    else:
        raise ValueError(f"Unable to convert from {img.dtype} to {dtype}")

