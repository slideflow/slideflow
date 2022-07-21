from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from slideflow import errors
from slideflow.dataset import Dataset
from slideflow.norm import StainNormalizer
from slideflow.norm.tensorflow import reinhard, macenko
from slideflow.util import detuple, log
from tqdm import tqdm

import tensorflow as tf


class TensorflowStainNormalizer(StainNormalizer):

    normalizers = {
        'reinhard': reinhard.ReinhardNormalizer,
        'reinhard_fast': reinhard.ReinhardFastNormalizer,
        'reinhard_mask': reinhard.ReinhardMaskNormalizer,
        'reinhard_fast_mask': reinhard.ReinhardFastMaskNormalizer,
        'macenko': macenko.MacenkoNormalizer
    }

    def __init__(
        self,
        method: str,
        device: Optional[str] = None,
        **kwargs
    ) -> None:
        """Tensorflow-native H&E Stain normalizer.

        The stain normalizer supports numpy images, PNG or JPG strings,
        Tensorflow tensors, and PyTorch tensors. The default `.transform()`
        method will attempt to preserve the original image type while minimizing
        conversions to and from Tensors.

        Alternatively, you can manually specify the image conversion type
        by using the appropriate function. For example, to convert a JPEG
        image to a normalized numpy RGB image, use `.jpeg_to_rgb()`.

        Args:
            method (str): Normalization method to use.
            device (str, optional): Device on which to perform normalization
                (e.g. 'cpu', 'gpu:0', 'gpu'). If None, will default to using
                the preferred device for the corresponding normalizer
                ('cpu' for Macenko, 'gpu' for Reinhard).

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
            Please see :class:`slideflow.norm.StainNormalizer` for examples.
        """
        super().__init__(method, **kwargs)
        self._device = device

    @property
    def vectorized(self) -> bool:  # type: ignore
        """Returns whether the associated normalizer is vectorized.

        Returns:
            bool: Normalizer is vectorized.
        """
        return self.n.vectorized

    @property
    def device(self) -> str:
        """Device (e.g. cpu, gpu) on which normalization should be performed.

        Returns:
            str: Device that will be used.
        """
        if self._device is not None:
            return self._device
        elif hasattr(self.n, 'preferred_device'):
            return self.n.preferred_device
        else:
            return 'cpu'

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

    def fit(
        self,
        arg1: Optional[Union[Dataset, np.ndarray, str]],
        batch_size: int = 64,
        num_threads: Union[str, int] = 'auto',
        **kwargs
    ) -> None:
        """Fit the normalizer to a target image or dataset of images.

        Args:
            arg1: (Dataset, np.ndarray, str): Target to fit. May be a numpy
                image array (uint8), path to an image, or a Slideflow Dataset.
                If a Dataset is provided, will average fit values across
                all images in the dataset.
            batch_size (int, optional): Batch size during fitting, if fitting
                to dataset. Defaults to 64.
        """
        if isinstance(arg1, Dataset):
            # Prime the normalizer
            dataset = arg1
            dts = dataset.tensorflow(
                None,
                batch_size,
                standardize=False,
                infinite=False
            )
            all_fit_vals = []  # type: ignore
            pb = tqdm(
                desc='Fitting normalizer...',
                ncols=80,
                total=dataset.num_tiles
            )
            for i, slide in dts:
                if self.vectorized:
                    fit_vals = self.n.fit(i, reduce=True)
                else:
                    _img_fits = zip(*[self.n.fit(_i) for _i in i])
                    fit_vals = [tf.reduce_mean(tf.stack(v), axis=0) for v in _img_fits]
                if all_fit_vals == []:
                    all_fit_vals = [[] for _ in range(len(fit_vals))]
                for v, val in enumerate(fit_vals):
                    all_fit_vals[v] += [val]
                pb.update(batch_size)
            self.n.set_fit(*[tf.math.reduce_mean(tf.stack(v), axis=0) for v in all_fit_vals])

        elif isinstance(arg1, np.ndarray):
            self.n.fit(tf.convert_to_tensor(arg1))

        # Fit to a preset
        elif isinstance(arg1, str) and arg1 in ('v1', 'v2'):
            self.n.fit_preset(arg1)

        elif isinstance(arg1, str):
            self.src_img = tf.image.decode_jpeg(tf.io.read_file(arg1))
            self.n.fit(self.src_img)

        elif arg1 is None and kwargs:
            self.set_fit(**kwargs)

        else:
            raise errors.NormalizerError(f'Unrecognized args for fit: {arg1}')

        log.debug('Fit normalizer: {}'.format(
            ', '.join([f"{fit_key} = {fit_val}"
            for fit_key, fit_val in self.get_fit().items()])
        ))

    @tf.function
    def tf_to_tf(
        self,
        batch: Union[Dict, tf.Tensor],
        *args: Any
    ) -> Union[Dict, tf.Tensor, Tuple[Union[Dict, tf.Tensor], ...]]:
        """Normalize a Tensor image or batch of image Tensors.

        Args:
            batch (Union[Dict, tf.Tensor]): Dict of tensors with image batches
                (via key "tile_image") or a batch of images.

        Returns:
            Tuple[Union[Dict, tf.Tensor], ...]: Normalized images in the same
            format as the input.
        """
        with tf.device(self.device):
            if isinstance(batch, dict):
                to_return = {
                    k: v for k, v in batch.items()
                    if k != 'tile_image'
                }
                to_return['tile_image'] = self.n.transform(batch['tile_image'])
                return detuple(to_return, args)
            else:
                return detuple(self.n.transform(batch), args)

    def tf_to_rgb(self, image: tf.Tensor) -> np.ndarray:
        """Normalize a tf.Tensor (uint8), returning a numpy array (uint8).

        Args:
            image (tf.Tensor): Image (uint8).

        Returns:
            np.ndarray: Normalized image, uint8, W x H x C.
        """
        return self.tf_to_tf(image).numpy()

    def rgb_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """Normalize a numpy array (uint8), returning a numpy array (uint8).

        Args:
            image (np.ndarray): Image (uint8).

        Returns:
            np.ndarray: Normalized image, uint8, W x H x C.
        """
        image = tf.convert_to_tensor(image)
        return self.n.transform(image).numpy()

    def jpeg_to_rgb(self, jpeg_string: Union[str, bytes]) -> np.ndarray:
        """Normalize a JPEG image, returning a numpy uint8 array.

        Args:
            jpeg_string (str, bytes): JPEG image data.

        Returns:
            np.ndarray: Normalized image, uint8, W x H x C.
        """
        return self.tf_to_rgb(tf.image.decode_jpeg(jpeg_string))

    def png_to_rgb(self, png_string: Union[str, bytes]) -> np.ndarray:
        """Normalize a PNG image, returning a numpy uint8 array.

        Args:
            png_string (str, bytes): PNG image data.

        Returns:
            np.ndarray: Normalized image, uint8, W x H x C.
        """
        return self.tf_to_rgb(tf.image.decode_png(png_string, channels=3))
