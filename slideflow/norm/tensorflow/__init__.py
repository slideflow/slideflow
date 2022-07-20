from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from slideflow import errors
from slideflow.dataset import Dataset
from slideflow.norm import StainNormalizer
from slideflow.norm.tensorflow import reinhard, reinhard_fast, macenko
from slideflow.util import detuple, log
from tqdm import tqdm

import tensorflow as tf


class TensorflowStainNormalizer(StainNormalizer):

    normalizers = {
        'reinhard': reinhard.ReinhardNormalizer,
        'reinhard_fast': reinhard_fast.ReinhardFastNormalizer,
        'macenko': macenko.MacenkoNormalizer
    }

    def __init__(
        self,
        method: str,
        device: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initializes the Tensorflow stain normalizer

        Args:
            method (str, optional): Normalizer method. Defaults to 'reinhard'.
            source (Optional[str], optional): Normalizer source image,
                'dataset', or None. Defaults to None.
        """
        super().__init__(method, **kwargs)
        self._device = device

    @property
    def vectorized(self) -> bool:  # type: ignore
        return self.n.vectorized

    @property
    def device(self) -> str:
        if self._device is not None:
            return self._device
        elif hasattr(self.n, 'preferred_device'):
            return self.n.preferred_device
        else:
            return 'cpu'

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
        **kwargs
    ) -> None:
        """Fit the normalizer.

        Args:
            target_means (_type_, optional): Target means. Defaults to None.
            target_stds (_type_, optional): Target stds. Defaults to None.
            stain_matrix_target (_type_, optional): Stain matrix target.
                Defaults to None.
            target_concentrations (_type_, optional): Target concentrations.
                Defaults to None.
            batch_size (int, optional): Batch size during dataset fitting.
                Defaults to 64.

        Raises:
            errors.NormalizerError: If unrecognized arguments are provided.
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
                fit_vals = self.n.fit(i, reduce=True)
                if all_fit_vals == []:
                    all_fit_vals = [[] for _ in range(len(fit_vals))]
                for v, val in enumerate(fit_vals):
                    all_fit_vals[v] += [val]
                pb.update(batch_size)
            self.n.set_fit(*[tf.math.reduce_mean(tf.stack(v), axis=0) for v in all_fit_vals])

        elif isinstance(arg1, np.ndarray):
            self.n.fit(tf.convert_to_tensor(arg1))

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
        """Normalize a batch of tensors.

        Args:
            batch (Union[Dict, tf.Tensor]): Dict of tensors with image batches
                (via key "tile_image") or a batch of images.

        Returns:
            Tuple[Union[Dict, tf.Tensor], ...]: Normalized images in the same
            format as the input.
        """
        #with tf.device(self.device):
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
        return self.tf_to_tf(image).numpy()

    def rgb_to_rgb(self, image: np.ndarray) -> np.ndarray:
        '''Non-normalized RGB numpy array -> normalized RGB numpy array'''
        image = tf.convert_to_tensor(image)
        return self.n.transform(image).numpy()

    def jpeg_to_rgb(self, jpeg_string: Union[str, bytes]) -> np.ndarray:
        '''Non-normalized compressed JPG data -> normalized RGB numpy array'''
        return self.tf_to_rgb(tf.image.decode_jpeg(jpeg_string))

    def png_to_rgb(self, png_string: Union[str, bytes]) -> np.ndarray:
        '''Non-normalized compressed PNG data -> normalized RGB numpy array'''
        return self.tf_to_rgb(tf.image.decode_png(png_string, channels=3))
