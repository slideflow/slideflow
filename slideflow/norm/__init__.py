"""Stain normalization methods, including both OpenCV (individual image)
and Tensorflow/PyTorch (vectorized) implementations."""

from __future__ import absolute_import

import multiprocessing as mp
import os
from io import BytesIO
from os.path import join
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

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

from slideflow.norm import (augment, macenko, reinhard, reinhard_fast,
                            reinhard_mask, vahadane)


class StainNormalizer:
    """Supervises stain normalization of images."""

    vectorized = False
    normalizers = {
        'macenko':  macenko.MacenkoNormalizer,
        'reinhard': reinhard.ReinhardNormalizer,
        'reinhard_fast': reinhard_fast.ReinhardFastNormalizer,
        'reinhard_mask': reinhard_mask.ReinhardMaskNormalizer,
        'vahadane': vahadane.VahadaneNormalizer,
        'augment': augment.AugmentNormalizer
    }  # type: Dict[str, Any]

    def __init__(self, method: str, **kwargs) -> None:
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

    def _torch_transform(self, inp):
        import torch
        from slideflow.io.torch import cwh_to_whc, whc_to_cwh

        if len(inp.shape) == 4:
            return torch.stack([self._torch_transform(img) for img in inp])
        elif inp.shape[0] == 3:
            return whc_to_cwh(self._torch_transform(cwh_to_whc(inp)))
        else:
            return torch.from_numpy(self.rgb_to_rgb(inp.cpu().numpy()))

    def get_fit(self, as_list: bool = False):
        _fit = self.n.get_fit()
        if as_list:
            return {k: v.tolist() for k, v in _fit.items()}
        else:
            return _fit

    def set_fit(self, **kwargs) -> None:
        self.n.set_fit(**kwargs)

    def fit(
        self,
        arg1: Optional[Union[Dataset, np.ndarray, str]],
        batch_size: int = 64,
        num_threads: Union[str, int] = 'auto',
        **kwargs,
    ) -> None:
        """Fit the normalizer.

        Args:
            arg1: (Dataset, np.ndarray, str): Target to fit.
            batch_size (int, optional): Batch size during fitting, if fitting
                to dataset. Defaults to 64.
            num_threads (Union[str, int], optional): Number of threads to use
                during fitting, if fitting to a dataset. Defaults to 'auto'.

        Raises:
            NotImplementedError: If attempting to fit Dataset using
            non-vectorized normalizer.

            errors.NormalizerError: If unrecognized arguments are provided.
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

        # Fit to a path to an image
        elif isinstance(arg1, str):
            self.src_img = cv2.cvtColor(cv2.imread(arg1), cv2.COLOR_BGR2RGB)
            self.n.fit(self.src_img)

        elif arg1 is None and kwargs:
            self.set_fit(**kwargs)

        else:
            raise errors.NormalizerError(f'Unrecognized args for fit: {arg1}')

        log.debug('Fit normalizer: {}'.format(
            ', '.join([f"{fit_key} = {fit_val}"
            for fit_key, fit_val in self.get_fit().items()])
        ))

    def tf_to_tf(
        self,
        image: Union[Dict, "tf.Tensor"],
        *args: Any
    ) -> Tuple[Union[Dict, "tf.Tensor"], ...]:
        '''Tensorflow RGB image (or batch)
            -> normalized Tensorflow RGB image (or batch)'''
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
        '''Non-normalized PyTorch image -> normalized RGB PyTorch image'''
        if isinstance(image, dict):
            to_return = {
                k: v for k, v in image.items()
                if k != 'tile_image'
            }
            to_return['tile_image'] = self._torch_transform(image['tile_image'])
            return detuple(to_return, args)
        else:
            return detuple(self._torch_transform(image), args)

    def tf_to_rgb(self, image: "tf.Tensor") -> np.ndarray:
        '''Non-normalized tensorflow RGB array -> normalized RGB numpy array'''
        return self.rgb_to_rgb(np.array(image))

    def rgb_to_rgb(self, image: np.ndarray) -> np.ndarray:
        '''Non-normalized RGB numpy array -> normalized RGB numpy array'''
        return self.n.transform(image)

    def jpeg_to_rgb(self, jpeg_string: Union[str, bytes]) -> np.ndarray:
        '''Non-normalized compressed JPG data -> normalized RGB numpy array'''
        cv_image = cv2.imdecode(
            np.fromstring(jpeg_string, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return self.n.transform(cv_image)

    def png_to_rgb(self, png_string: Union[str, bytes]) -> np.ndarray:
        '''Non-normalized compressed PNG data -> normalized RGB numpy array'''
        return self.jpeg_to_rgb(png_string)  # It should auto-detect format

    def jpeg_to_jpeg(
        self,
        jpeg_string: Union[str, bytes],
        quality: int = 75
    ) -> bytes:
        '''Non-normalized compressed JPG string data -> normalized
        compressed JPG data
        '''
        cv_image = self.jpeg_to_rgb(jpeg_string)
        with BytesIO() as output:
            Image.fromarray(cv_image).save(
                output,
                format="JPEG",
                quality=quality
            )
            return output.getvalue()

    def png_to_png(self, png_string: Union[str, bytes]) -> bytes:
        '''Non-normalized PNG string data -> normalized PNG string data'''
        cv_image = self.png_to_rgb(png_string)
        with BytesIO() as output:
            Image.fromarray(cv_image).save(output, format="PNG")
            return output.getvalue()


def autoselect(
    method: str,
    source: Optional[str] = None
) -> StainNormalizer:

    if sf.backend() == 'tensorflow':
        import slideflow.norm.tensorflow
        BackendNormalizer = sf.norm.tensorflow.TensorflowStainNormalizer
    elif sf.backend() == 'torch':
        import slideflow.norm.torch
        BackendNormalizer = sf.norm.torch.TorchStainNormalizer  # type: ignore
    else:
        raise Exception(f"Unrecognized backend: {sf.backend()}")

    if method in BackendNormalizer.normalizers:
        normalizer = BackendNormalizer(method)
    else:
        normalizer = StainNormalizer(method)  # type: ignore

    if source is not None and source != 'dataset':
        normalizer.fit(source)

    return normalizer