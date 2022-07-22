"""This module provides H&E stain normalization tools, with numpy, PyTorch,
and Tensorflow implementations for several stain normalization methods.

Overview
--------

The main normalizer interface, :class:`slideflow.norm.StainNormalizer`, offers
efficient numpy implementations for the Macenko, Reinhard, Reinhard-Fast,
Reinhard (masked), and Vahadane H&E stain normalization algorithms, as well
as an HSV colorspace stain augmentation method. This normalizer can convert
images to and from Tensors, numpy arrays, and raw JPEG/PNG images.

In addition to these numpy implementations, PyTorch-native and Tensorflow-native
implementations are also provided, which offer performance improvements
and/or vectorized application. The native normalizers are found in
``slideflow.norm.tensorflow`` and ``slideflow.norm.torch``, respectively.
Tensorflow-native normalizer methods include Macenko, Reinhard, and
Reinhard-fast. Torch-native normalizer methods include Reinhard and
Reinhard-fast.

The Vahadane normalizer has two numpy implementations available: SPAMS
(``vahadane_spams``) and sklearn (``vahadane_sklearn``). As of version 1.2.3,
the sklearn implementation will be used if unspecified (``method='vahadane'``).

Performance
-----------

The Numpy implementations contain all functions necessary for normalizing
Tensors from both Tensorflow and PyTorch, but may be slower than backend-native
implementations when available. Performance benchmarks for the normalizer
implementations are given below:

.. list-table:: **Performance Benchmarks** (256 x 256 images, Slideflow 1.2.3, benchmarked on 3960X and A100 40GB)
    :header-rows: 1

    * -
      - Tensorflow backend
      - PyTorch backend
    * - macenko
      - 1,295 img/s (**native**)
      - 142 img/s
    * - reinhard
      - 1,536 img/s (**native**)
      - 1,840 img/s (**native**)
    * - reinhard_fast
      - 8,599 img/s (**native**)
      - 2,590 img/s (**native**)
    * - reinhard_mask
      - 1,537 img/s (**native**)
      - 1,581 img/s
    * - reinhard_fast_mask
      - 7,885 img/s (**native**)
      - 2,116 img/s
    * - vahadane_spams
      - 0.7 img/s
      - 2.2 img/s
    * - vahadane_sklearn
      - 0.9 img/s
      - 1.0 img/s


Use :func:`slideflow.norm.autoselect` to get the fastest available normalizer
for a given method and active backend (Tensorflow/PyTorch).
"""

from __future__ import absolute_import

import os
import sys
import multiprocessing as mp
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import slideflow as sf
from PIL import Image
from slideflow import errors
from slideflow.dataset import Dataset
from slideflow.util import detuple, log
from tqdm import tqdm

if TYPE_CHECKING:
    import tensorflow as tf
    import torch

from slideflow.norm import (augment, macenko, reinhard, vahadane)


class StainNormalizer:

    vectorized = False
    normalizers = {
        'macenko':  macenko.MacenkoNormalizer,
        'reinhard': reinhard.ReinhardNormalizer,
        'reinhard_fast': reinhard.ReinhardFastNormalizer,
        'reinhard_mask': reinhard.ReinhardMaskNormalizer,
        'reinhard_fast_mask': reinhard.ReinhardFastMaskNormalizer,
        'vahadane': vahadane.VahadaneSklearnNormalizer,
        'vahadane_sklearn': vahadane.VahadaneSklearnNormalizer,
        'vahadane_spams': vahadane.VahadaneSpamsNormalizer,
        'augment': augment.AugmentNormalizer
    }  # type: Dict[str, Any]

    def __init__(self, method: str, **kwargs) -> None:
        """H&E Stain normalizer supporting various normalization methods.

        The stain normalizer supports numpy images, PNG or JPG strings,
        Tensorflow tensors, and PyTorch tensors. The default ``.transform()``
        method will attempt to preserve the original image type while minimizing
        conversions to and from Tensors.

        Alternatively, you can manually specify the image conversion type
        by using the appropriate function. For example, to convert a Tensor
        to a normalized numpy RGB image, use ``.tf_to_rgb()``.

        Args:
            method (str): Normalization method. Options include 'macenko',
                'reinhard', 'reinhard_fast', 'reinhard_mask',
                'reinhard_fast_mask', 'vahadane', 'vahadane_spams',
                'vahadane_sklearn', and 'augment'.

        Keyword args:
            stain_matrix_target (np.ndarray, optional): Set the stain matrix
                target for the normalizer. May raise an error if the normalizer
                does not have a stain_matrix_target fit attribute.
            target_concentrations (np.ndarray, optional): Set the target
                concentrations for the normalizer. May raise an error if the
                normalizer does not have a target_concentrations fit attribute.
            target_means (np.ndarray, optional): Set the target means for the
                normalizer. May raise an error if the normalizer does not have
                a target_means fit attribute.
            target_stds (np.ndarray, optional): Set the target standard
                deviations for the normalizer. May raise an error if the
                normalizer does not have a target_stds fit attribute.

        Raises:
            ValueError: If the specified normalizer method is not available.

        Examples
            Normalize a numpy image using the default fit.

                >>> import slideflow as sf
                >>> macenko = sf.norm.StainNormalizer('macenko')
                >>> macenko.transform(image)

            Fit the normalizer to a target image (numpy or path).

                >>> macenko.fit(target_image)

            Fit the normalizer to all images in a Dataset.

                >>> dataset = sf.Dataset(...)
                >>> macenko.fit(dataset)

            Normalize an image and convert from Tensor to RGB.

                >>> macenko.tf_to_rgb(image)

            Normalize images during DataLoader pre-processing.

                >>> dataset = sf.Dataset(...)
                >>> dataloader = dataset.torch(..., normalizer=macenko)
                >>> dts = dataset.tensorflow(..., normalizer=macenko)

        """
        if method not in self.normalizers:
            raise ValueError(f"Unrecognized normalizer method {method}")

        self.method = method
        self.n = self.normalizers[method]()

        if kwargs:
            self.n.fit(**kwargs)

    def __repr__(self):
        base = "{}(\n".format(self.__class__.__name__)
        base += "  method = {!r},\n".format(self.method)
        for fit_param, fit_val in self.get_fit().items():
            base += "  {} = {!r},\n".format(fit_param, fit_val)
        base += ")"
        return base

    def _torch_transform(self, inp: "torch.Tensor") -> "torch.Tensor":
        """Normalize a torch uint8 image (CWH), via intermediate
        conversion to WHC.

        Args:
            inp (torch.Tensor): Image, uint8. Images are normalized in
                W x H x C space. Images provided as C x W x H will be
                auto-converted and permuted back after normalization.

        Returns:
            torch.Tensor:   Image, uint8.

        """
        import torch
        from slideflow.io.torch import cwh_to_whc, whc_to_cwh

        if len(inp.shape) == 4:
            return torch.stack([self._torch_transform(img) for img in inp])
        elif inp.shape[0] == 3:
            # Convert from CWH -> WHC (normalize) -> CWH
            return whc_to_cwh(torch.from_numpy(self.rgb_to_rgb(cwh_to_whc(inp).cpu().numpy())))
        else:
            return torch.from_numpy(self.rgb_to_rgb(inp.cpu().numpy()))

    def fit(
        self,
        arg1: Optional[Union[Dataset, np.ndarray, str]],
        batch_size: int = 64,
        num_threads: Union[str, int] = 'auto',
        **kwargs,
    ) -> None:
        """Fit the normalizer to a target image or dataset of images.

        Args:
            arg1: (Dataset, np.ndarray, str): Target to fit. May be a numpy
                image array (uint8), path to an image, or a Slideflow Dataset.
                If a Dataset is provided, will average fit values across
                all images in the dataset.
            batch_size (int, optional): Batch size during fitting, if fitting
                to dataset. Defaults to 64.
            num_threads (Union[str, int], optional): Number of threads to use
                during fitting, if fitting to a dataset. Defaults to 'auto'.
        """

        # Fit to a dataset
        if isinstance(arg1, Dataset):
            # Set up thread pool
            if num_threads == 'auto':
                num_threads = os.cpu_count()  # type: ignore
            log.debug(f"Setting up pool (size={num_threads}) for norm fitting")
            log.debug(f"Using normalizer batch size of {batch_size}")
            pool = mp.dummy.Pool(num_threads)  # type: ignore

            dataset = arg1
            if sf.backend() == 'tensorflow':
                dts = dataset.tensorflow(
                    None,
                    batch_size,
                    standardize=False,
                    infinite=False
                )
            elif sf.backend() == 'torch':
                dts = dataset.torch(
                    None,
                    batch_size,
                    standardize=False,
                    infinite=False,
                    num_workers=8
                )
            all_fit_vals = []  # type: ignore
            pb = tqdm(
                desc='Fitting normalizer...',
                ncols=80,
                total=dataset.num_tiles
            )
            for img_batch, slide in dts:
                if sf.backend() == 'torch':
                    img_batch = img_batch.permute(0, 2, 3, 1)  # BCWH -> BWHC

                mapped = pool.imap(lambda x: self.n.fit(x.numpy()), img_batch)
                for fit_vals in mapped:
                    if all_fit_vals == []:
                        all_fit_vals = [[] for _ in range(len(fit_vals))]
                    for v, val in enumerate(fit_vals):
                        all_fit_vals[v] += [np.squeeze(val)]
                pb.update(batch_size)
            self.n.set_fit(*[np.array(v).mean(axis=0) for v in all_fit_vals])
            pool.close()

        # Fit to numpy image
        elif isinstance(arg1, np.ndarray):
            self.n.fit(arg1)

        # Fit to a preset
        elif isinstance(arg1, str) and arg1 in ('v1', 'v2'):
            self.n.fit_preset(arg1)

        # Fit to a path to an image
        elif isinstance(arg1, str):
            self.src_img = cv2.cvtColor(cv2.imread(arg1), cv2.COLOR_BGR2RGB)
            self.n.fit(self.src_img)

        elif arg1 is None and kwargs:
            self.set_fit(**kwargs)

        else:
            raise ValueError(f'Unrecognized args for fit: {arg1}')

        log.debug('Fit normalizer: {}'.format(
            ', '.join([f"{fit_key} = {fit_val}"
            for fit_key, fit_val in self.get_fit().items()])
        ))

    def get_fit(self, as_list: bool = False):
        """Get the current normalizer fit.

        Args:
            as_list (bool). Convert the fit values (numpy arrays) to list
                format. Defaults to False.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping fit parameters (e.g.
            'target_concentrations') to their respective fit values.
        """
        _fit = self.n.get_fit()
        if as_list:
            return {k: v.tolist() for k, v in _fit.items()}
        else:
            return _fit

    def set_fit(self, **kwargs) -> None:
        """Set the normalizer fit to teh given values.

        Keyword args:
            stain_matrix_target (np.ndarray, optional): Set the stain matrix
                target for the normalizer. May raise an error if the normalizer
                does not have a stain_matrix_target fit attribute.
            target_concentrations (np.ndarray, optional): Set the target
                concentrations for the normalizer. May raise an error if the
                normalizer does not have a target_concentrations fit attribute.
            target_means (np.ndarray, optional): Set the target means for the
                normalizer. May raise an error if the normalizer does not have
                a target_means fit attribute.
            target_stds (np.ndarray, optional): Set the target standard
                deviations for the normalizer. May raise an error if the
                normalizer does not have a target_stds fit attribute.
        """
        self.n.set_fit(**{k:v for k, v in kwargs.items() if v is not None})

    def transform(
        self,
        image: Union[str, bytes, np.ndarray, "tf.Tensor", "torch.Tensor"]
    ) -> Union[str, bytes, np.ndarray, "tf.Tensor", "torch.Tensor"]:
        """Normalize a target image, attempting to preserve the original type.

        Args:
            image (np.ndarray, tf.Tensor, or torch.Tensor): Image.

        Returns:
            Normalized image of the original type.
        """
        if isinstance(image, (str, bytes)):
            raise ValueError("Unable to auto-transform bytes or str; please "
                             "use .png_to_png() or .jpeg_to_jpeg().")
        if 'tensorflow' in sys.modules:
            import tensorflow as tf
            if isinstance(image, tf.Tensor):
                return self.tf_to_tf(image)
        if 'torch' in sys.modules:
            import torch
            if isinstance(image, torch.Tensor):
                return self.torch_to_torch(image)
        if isinstance(image, np.ndarray):
            return self.rgb_to_rgb(image)
        raise ValueError(f"Unrecognized image type {type(image)}; expected "
                         "np.ndarray, tf.Tensor, or torch.Tensor")

    def jpeg_to_jpeg(
        self,
        jpeg_string: Union[str, bytes],
        quality: int = 100
    ) -> bytes:
        """Normalize a JPEG image, returning a JPEG image.

        Args:
            jpeg_string (str, bytes): JPEG image data.
            quality (int, optional): Quality level for creating the resulting
                normalized JPEG image. Defaults to 100.

        Returns:
            bytes:  Normalized JPEG image.
        """
        cv_image = self.jpeg_to_rgb(jpeg_string)
        with BytesIO() as output:
            Image.fromarray(cv_image).save(
                output,
                format="JPEG",
                quality=quality
            )
            return output.getvalue()

    def jpeg_to_rgb(self, jpeg_string: Union[str, bytes]) -> np.ndarray:
        """Normalize a JPEG image, returning a numpy uint8 array.

        Args:
            jpeg_string (str, bytes): JPEG image data.

        Returns:
            np.ndarray: Normalized image, uint8, W x H x C.
        """
        cv_image = cv2.imdecode(
            np.fromstring(jpeg_string, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return self.n.transform(cv_image)

    def png_to_png(self, png_string: Union[str, bytes]) -> bytes:
        """Normalize a PNG image, returning a PNG image.

        Args:
            png_string (str, bytes): PNG image data.

        Returns:
            bytes: Normalized PNG image.
        """
        cv_image = self.png_to_rgb(png_string)
        with BytesIO() as output:
            Image.fromarray(cv_image).save(output, format="PNG")
            return output.getvalue()

    def png_to_rgb(self, png_string: Union[str, bytes]) -> np.ndarray:
        """Normalize a PNG image, returning a numpy uint8 array.

        Args:
            png_string (str, bytes): PNG image data.

        Returns:
            np.ndarray: Normalized image, uint8, W x H x C.
        """
        return self.jpeg_to_rgb(png_string)  # It should auto-detect format

    def rgb_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """Normalize a numpy array (uint8), returning a numpy array (uint8).

        Args:
            image (np.ndarray): Image (uint8).

        Returns:
            np.ndarray: Normalized image, uint8, W x H x C.
        """
        return self.n.transform(image)

    def tf_to_rgb(self, image: "tf.Tensor") -> np.ndarray:
        """Normalize a tf.Tensor (uint8), returning a numpy array (uint8).

        Args:
            image (tf.Tensor): Image (uint8).

        Returns:
            np.ndarray: Normalized image, uint8, W x H x C.
        """
        return self.rgb_to_rgb(np.array(image))

    def tf_to_tf(
        self,
        image: Union[Dict, "tf.Tensor"],
        *args: Any
    ) -> Tuple[Union[Dict, "tf.Tensor"], ...]:
        """Normalize a tf.Tensor (uint8), returning a numpy array (uint8).

        Args:
            image (tf.Tensor, Dict): Image (uint8) either as a raw Tensor,
                or a Dictionary with the image under the key 'tile_image'.
            args (Any, optional): Any additional arguments, which will be passed
                and returned unmodified.

        Returns:
            A tuple containing the normalized tf.Tensor image (uint8,
            W x H x C) and any additional arguments provided.
        """
        import tensorflow as tf

        if isinstance(image, dict):
            image['tile_image'] = tf.py_function(
                self.tf_to_rgb,
                [image['tile_image']],
                tf.int32
            )
        elif len(image.shape) == 4:
            image = tf.stack([self.tf_to_tf(_i) for _i in image])
        else:
            image = tf.py_function(self.tf_to_rgb, [image], tf.int32)
        return detuple(image, args)

    def torch_to_torch(
        self,
        image: Union[Dict, "torch.Tensor"],
        *args
    ) -> Tuple[Union[Dict, "torch.Tensor"], ...]:
        """Normalize a torch.Tensor (uint8), returning a numpy array (uint8).

        Args:
            image (torch.Tensor, Dict): Image (uint8) either as a raw Tensor,
                or a Dictionary with the image under the key 'tile_image'.
            args (Any, optional): Any additional arguments, which will be passed
                and returned unmodified.

        Returns:
            A tuple containing

                np.ndarray: Normalized tf.Tensor image, uint8, C x W x H.

                args (Any, optional): Any additional arguments provided, unmodified.
        """
        if isinstance(image, dict):
            to_return = {
                k: v for k, v in image.items()
                if k != 'tile_image'
            }
            to_return['tile_image'] = self._torch_transform(image['tile_image'])
            return detuple(to_return, args)
        else:
            return detuple(self._torch_transform(image), args)


def autoselect(
    method: str,
    source: Optional[str] = None,
    **kwargs
) -> StainNormalizer:
    """Select the best normalizer for a given method, and fit to a given source.

    If a normalizer method has a native implementation in the current backend
    (Tensorflow or PyTorch), the native normalizer will be used.
    If not, the default numpy implementation will be used.

    Currently, the PyTorch-native normalizers are NOT used by default, as they
    are slower than the numpy implementations. Thus, with the PyTorch backend,
    all normalizers will be the default numpy implementations.

    Args:
        method (str): Normalization method. Options include 'macenko',
            'reinhard', 'reinhard_fast', 'reinhard_mask', 'reinhard_fast_mask',
            'vahadane', 'vahadane_spams', 'vahadane_sklearn', and 'augment'.
        source (str, optional): Path to a source image. If provided, will
            fit the normalizer to this image. Defaults to None.

    Returns:
        StainNormalizer:    Initialized StainNormalizer.
    """

    if sf.backend() == 'tensorflow':
        import slideflow.norm.tensorflow
        BackendNormalizer = sf.norm.tensorflow.TensorflowStainNormalizer
    elif sf.backend() == 'torch':
        log.debug("Not attempting to use PyTorch-native normalizer.")
        BackendNormalizer = StainNormalizer  # type: ignore
    else:
        raise errors.UnrecognizedBackendError

    if method in BackendNormalizer.normalizers:
        normalizer = BackendNormalizer(method, **kwargs)
    else:
        normalizer = StainNormalizer(method, **kwargs)  # type: ignore

    if source is not None and source != 'dataset':
        normalizer.fit(source)

    return normalizer