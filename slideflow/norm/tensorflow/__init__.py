import os
import numpy as np
import tensorflow as tf
from typing import Optional, Union, List, Dict, Any, Tuple

from os.path import join
from tqdm import tqdm
from slideflow.dataset import Dataset
from slideflow.util import log
from slideflow.norm import StainNormalizer
from slideflow.norm.tensorflow import reinhard, reinhard_fast
from slideflow import errors


class TensorflowStainNormalizer(StainNormalizer):
    vectorized = True
    backend = 'tensorflow'
    normalizers = {
        'reinhard': reinhard,
        'reinhard_fast': reinhard_fast
    }

    def __init__(
        self,
        method: str='reinhard',
        source: Optional[str] = None
    ) -> None:
        """Initializes the Tensorflow stain normalizer

        Args:
            method (str, optional): Normalizer method. Defaults to 'reinhard'.
            source (Optional[str], optional): Normalizer source image,
                'dataset', or None. Defaults to None.
        """
        self.method = method
        self._source = source
        self.n = self.normalizers[method]
        if not source:
            package_directory = os.path.dirname(os.path.abspath(__file__))
            source = join(package_directory, '../norm_tile.jpg')
        if source != 'dataset':
            self.src_img = tf.image.decode_jpeg(tf.io.read_file(source))
            means, stds = self.n.fit(tf.expand_dims(self.src_img, axis=0))
            self.target_means = tf.concat(means, 0)
            self.target_stds = tf.concat(stds, 0)
        else:
            self.target_means = None
            self.target_stds = None
        self.stain_matrix_target = None
        self.target_concentrations = None

    def __repr__(self) -> str:
        src = "" if not self._source else ", source={!r}".format(self._source)
        return "TensorflowStainNormalizer(method={!r}{})".format(self.method, src)

    # === Normalizer parameters ===============================================

    # --- Target means --------------------------------------------------------

    @property
    def target_means(self) -> Optional[np.ndarray]:
        return self._target_means

    @target_means.setter
    def target_means(self, val: Optional[Union[np.ndarray, tf.Tensor]]) -> None:
        if val is not None:
            if isinstance(val, tf.Tensor):
                self._target_means = val.numpy()
                self._target_means_tensor = val
            else:
                self._target_means = val
                self._target_means_tensor = tf.constant(val)
        else:
            self._target_means = None
            self._target_means_tensor = None

    @property
    def target_means_tensor(self) -> Optional[tf.Tensor]:
        return self._target_means_tensor

    # --- Target stds ---------------------------------------------------------

    @property
    def target_stds(self) -> Optional[np.ndarray]:
        return self._target_stds

    @target_stds.setter
    def target_stds(self, val: Optional[Union[np.ndarray, tf.Tensor]]) -> None:
        if val is not None:
            if isinstance(val, tf.Tensor):
                self._target_stds = val.numpy()
                self._target_stds_tensor = val
            else:
                self._target_stds = val
                self._target_stds_tensor = tf.constant(val)
        else:
            self._target_stds = None
            self._target_stds_tensor = None

    @property
    def target_stds_tensor(self) -> Optional[tf.Tensor]:
        return self._target_stds_tensor

    # --- Target stain matrix -------------------------------------------------

    @property
    def stain_matrix_target(self) -> Optional[np.ndarray]:
        return self._stain_matrix_target

    @stain_matrix_target.setter
    def stain_matrix_target(
        self,
        val: Optional[Union[np.ndarray, tf.Tensor]]
    ) -> None:
        if val is not None:
            if isinstance(val, tf.Tensor):
                self._stain_matrix_target = val.numpy()
                self._stain_matrix_target_tensor = val
            else:
                self._stain_matrix_target = val
                self._stain_matrix_target_tensor = tf.constant(val)
        else:
            self._stain_matrix_target = None
            self._stain_matrix_target_tensor = None

    @property
    def stain_matrix_target_tensor(self) -> Optional[tf.Tensor]:
        return self._stain_matrix_target_tensor

    # --- Target concentrations -----------------------------------------------

    @property
    def target_concentrations(self) -> Optional[np.ndarray]:
        return self._target_concentrations

    @target_concentrations.setter
    def target_concentrations(
        self,
        val: Optional[Union[np.ndarray, tf.Tensor]]
    ) -> None:
        if val is not None:
            if isinstance(val, tf.Tensor):
                self._target_concentrations = val.numpy()
                self._target_concentrations_tensor = val
            else:
                self._target_concentrations = val
                self._target_concentrations_tensor = tf.constant(val)
        else:
            self._target_concentrations = None
            self._target_concentrations_tensor = None

    @property
    def target_concentrations_tensor(self) -> Optional[tf.Tensor]:
        return self._target_concentrations_tensor

    # =========================================================================

    def fit(
        self,
        *args: Optional[Union[Dataset, np.ndarray, str]],
        target_means: Optional[Union[List[float], np.ndarray]] = None,
        target_stds: Optional[Union[List[float], np.ndarray]] = None,
        stain_matrix_target: Optional[Union[List[float], np.ndarray]] = None,
        target_concentrations: Optional[Union[List[float], np.ndarray]] = None,
        batch_size: int = 64,
        num_threads: Union[str, int] = 'auto'
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
        if len(args) and isinstance(args[0], Dataset):
            # Prime the normalizer
            dataset = args[0]
            dts = dataset.tensorflow(
                None,
                batch_size,
                standardize=False,
                infinite=False
            )
            means, stds = [], []
            pb = tqdm(
                desc='Fitting normalizer...',
                ncols=80,
                total=dataset.num_tiles
            )
            for i, slide in dts:
                _m, _s = self.n.fit(i, reduce=True)
                means += [_m]
                stds += [_s]
                pb.update(batch_size)
            self.target_means = tf.math.reduce_mean(tf.stack(means), axis=0)
            self.target_stds = tf.math.reduce_mean(tf.stack(stds), axis=0)

        elif len(args) and isinstance(args[0], np.ndarray) and len(args) == 1:
            if len(args[0].shape) == 3:
                img = tf.expand_dims(tf.constant(args[0]), axis=0)
            else:
                img = tf.constant(args[0])
            means, stds = self.n.fit(img)
            self.target_means = tf.concat(means, 0)
            self.target_stds = tf.concat(stds, 0)

        elif len(args) and isinstance(args[0], str):
            src_img = tf.image.decode_jpeg(tf.io.read_file(args[0]))
            self.src_img = tf.expand_dims(src_img, axis=0)
            means, stds = self.n.fit(self.src_img)
            self.target_means = tf.concat(means, 0)
            self.target_stds = tf.concat(stds, 0)

        elif target_means is not None:
            self.target_means = tf.constant(
                np.array(target_means),
                dtype=tf.float32
            )
            self.target_stds = tf.constant(
                np.array(target_stds),
                dtype=tf.float32
            )
        elif (stain_matrix_target is not None
              and target_concentrations is not None):
            self.stain_matrix_target = tf.constant(
                np.array(stain_matrix_target),
                dtype=tf.float32
            )
            self.target_concentrations = tf.constant(
                np.array(target_concentrations),
                dtype=tf.float32
            )
        elif stain_matrix_target is not None:
            self.stain_matrix_target = tf.constant(
                np.array(stain_matrix_target),
                dtype=tf.float32
            )
        else:
            raise errors.NormalizerError(f'Unrecognized args for fit: {args}')
        log.info(
            f"Fit normalizer to mean {self.target_means}, "
            f"stddev {self.target_stds}"
        )

    def get_fit(self) -> Dict[str, Optional[List[float]]]:
        return {
            'target_means': None if self.target_means is None else self.target_means.tolist(),
            'target_stds': None if self.target_stds is None else self.target_stds.tolist(),
            'stain_matrix_target': None if self.stain_matrix_target is None else self.stain_matrix_target.tolist(),
            'target_concentrations': None if self.target_concentrations is None else self.target_concentrations.tolist()
        }

    @tf.function
    def batch_to_batch(
        self,
        batch: Union[Dict, tf.Tensor],
        *args: Any
    ) -> Tuple[Union[Dict, tf.Tensor], ...]:
        """Normalize a batch of tensors.

        Args:
            batch (Union[Dict, tf.Tensor]): Dict of tensors with image batches
                (via key "tile_image") or a batch of images.

        Returns:
            Tuple[Union[Dict, tf.Tensor], ...]: Normalized images in the same
            format as the input.
        """
        with tf.device('gpu:0'):
            if isinstance(batch, dict):
                to_return = {
                    k: v for k, v in batch.items()
                    if k != 'tile_image'
                }
                to_return['tile_image'] = self.tf_to_tf(batch['tile_image'])
                return tuple([to_return] + list(args))
            else:
                return tuple([self.tf_to_tf(batch)] + list(args))

    @tf.function
    def tf_to_tf(self, image: tf.Tensor) -> tf.Tensor:
        if len(image.shape) == 3:
            image = tf.expand_dims(image, axis=0)
            return self.n.transform(image, self.target_means_tensor, self.target_stds_tensor)[0]
        else:
            return self.n.transform(image, self.target_means_tensor, self.target_stds_tensor)

    def tf_to_rgb(self, image: tf.Tensor) -> np.ndarray:
        return self.tf_to_tf(image).numpy()

    def rgb_to_rgb(self, image: np.ndarray) -> np.ndarray:
        '''Non-normalized RGB numpy array -> normalized RGB numpy array'''
        image = tf.expand_dims(tf.constant(image, dtype=tf.uint8), axis=0)
        return self.n.transform(image, self.target_means_tensor, self.target_stds_tensor).numpy()[0]

    def jpeg_to_rgb(self, jpeg_string: Union[str, bytes]) -> np.ndarray:
        '''Non-normalized compressed JPG data -> normalized RGB numpy array'''
        return self.tf_to_rgb(tf.image.decode_jpeg(jpeg_string))

    def png_to_rgb(self, png_string: Union[str, bytes]) -> np.ndarray:
        '''Non-normalized compressed PNG data -> normalized RGB numpy array'''
        return self.tf_to_rgb(tf.image.decode_png(png_string, channels=3))
