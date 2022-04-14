import os
import numpy as np
import tensorflow as tf

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

    def __init__(self, method='reinhard', source=None):
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

    def __repr__(self):
        src = "" if not self._source else ", source={!r}".format(self._source)
        return "TensorflowStainNormalizer(method={!r}{})".format(self.method, src)

    @property
    def target_means(self):
        return self._target_means

    @target_means.setter
    def target_means(self, val):
        self._target_means = val

    @property
    def target_stds(self):
        return self._target_stds

    @target_stds.setter
    def target_stds(self, val):
        self._target_stds = val

    @property
    def stain_matrix_target(self):
        return self._stain_matrix_target

    @stain_matrix_target.setter
    def stain_matrix_target(self, val):
        self._stain_matrix_target = val

    @property
    def target_concentrations(self):
        return self._target_concentrations

    @target_concentrations.setter
    def target_concentrations(self, val):
        self._target_concentrations = val

    def fit(self, *args, target_means=None, target_stds=None,
            stain_matrix_target=None, target_concentrations=None,
            batch_size=64):

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

        msg = f"Fit normalizer to mean {self.target_means.numpy()}, "
        msg += f"stddev {self.target_stds.numpy()}"
        log.info(msg)

    def get_fit(self):

        def to_np(a):
            return a.numpy().tolist()

        return {
            'target_means': None if self.target_means is None else to_np(self.target_means),
            'target_stds': None if self.target_stds is None else to_np(self.target_stds),
            'stain_matrix_target': None if self.stain_matrix_target is None else to_np(self.stain_matrix_target),
            'target_concentrations': None if self.target_concentrations is None else to_np(self.target_concentrations)
        }

    @tf.function
    def batch_to_batch(self, batch, *args):
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
    def tf_to_tf(self, image):
        if len(image.shape) == 3:
            image = tf.expand_dims(image, axis=0)
            return self.n.transform(image, self.target_means, self.target_stds)[0]
        else:
            return self.n.transform(image, self.target_means, self.target_stds)

    def tf_to_rgb(self, image):
        return self.tf_to_tf(image).numpy()

    def rgb_to_rgb(self, image):
        '''Non-normalized RGB numpy array -> normalized RGB numpy array'''
        image = tf.expand_dims(tf.constant(image, dtype=tf.uint8), axis=0)
        return self.n.transform(image, self.target_means, self.target_stds).numpy()[0]

    def jpeg_to_rgb(self, jpeg_string):
        '''Non-normalized compressed JPG data -> normalized RGB numpy array'''
        return self.tf_to_rgb(tf.image.decode_jpeg(jpeg_string))

    def png_to_rgb(self, png_string):
        '''Non-normalized compressed PNG data -> normalized RGB numpy array'''
        return self.tf_to_rgb(tf.image.decode_png(png_string, channels=3))
