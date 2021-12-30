import os
import numpy as np
from io import BytesIO
from os.path import join
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from slideflow.dataset import Dataset
from slideflow.util import log

from slideflow.slide import tf_reinhard

class TFStainNormalizer:
    vectorized = True
    normalizers = {
        'reinhard': tf_reinhard
    }

    def __init__(self, method='reinhard', source=None):

        self.method = method
        self.n = self.normalizers[method]
        self._source = source
        if not source:
            package_directory = os.path.dirname(os.path.abspath(__file__))
            source = join(package_directory, 'norm_tile.jpg')
        self.src_img = tf.image.decode_jpeg(tf.io.read_file(source))
        means, stds = self.n.fit(tf.expand_dims(self.src_img, axis=0))
        self.target_means = tf.concat(means, 0)
        self.target_stds = tf.concat(stds, 0)

    def __repr__(self):
        src = "" if not self._source else ", source={!r}".format(self._source)
        return "TFStainNormalizer(method={!r}{})".format(self.method, src)

    def fit(self, *args):
        if isinstance(args[0], Dataset):
            # Prime the normalizer
            dataset = args[0]
            batch_size = 32
            dts = dataset.tensorflow(None, batch_size, standardize=False, infinite=False)
            m, s = [], []
            pb = tqdm(desc='Fitting normalizer...', total=dataset.num_tiles)
            for i, slide in dts:
                _m, _s = self.n.fit(i, reduce=True)
                m += [_m]
                s += [_s]
                pb.update(batch_size)
            dts_mean = tf.math.reduce_mean(tf.stack(m), axis=0)
            dts_std = tf.math.reduce_mean(tf.stack(s), axis=0)
            self.target_means = dts_mean
            self.target_stds = dts_std
        elif isinstance(args[0], np.ndarray) and len(args) == 1:
            if len(args[0].shape) == 3:
                img = tf.expand_dims(tf.constant(args[0]), axis=0)
            else:
                img = tf.constant(args[0])
            means, stds = self.n.fit(img)
            self.target_means = tf.concat(means, 0)
            self.target_stds = tf.concat(stds, 0)
        elif isinstance(args[0], str):
            self.src_img = tf.expand_dims(tf.image.decode_jpeg(tf.io.read_file(args[0])), axis=0)
            means, stds = self.n.fit(self.src_img)
            self.target_means = tf.concat(means, 0)
            self.target_stds = tf.concat(stds, 0)
        elif isinstance(args[0], np.ndarray) and len(args) == 2:
            self.target_means = tf.constant(args[0], dtype=tf.float32)
            self.target_stds = tf.constant(args[1], dtype=tf.float32)
        log.info(f"Fit normalizer to mean {self.target_means.numpy()}, stddev {self.target_stds.numpy()}")

    @tf.function
    def batch_to_batch(self, batch, *args):
        with tf.device('gpu:0'):
            if isinstance(batch, dict):
                to_return = {k:v for k,v in batch.items() if k != 'tile_image'}
                to_return['tile_image'] = self.tf_to_tf(batch['tile_image'])
                return to_return, *args
            else:
                return self.tf_to_tf(batch), *args

    @tf.function
    def tf_to_tf(self, image):
        if len(image.shape) == 3:
            return self.n.transform( tf.expand_dims(image, axis=0), self.target_means, self.target_stds)[0]
        else:
            return self.n.transform(image, self.target_means, self.target_stds)

    def tf_to_rgb(self, image):
        return self.tf_to_tf(image).numpy()

    def pil_to_pil(self, image):
        '''Non-normalized PIL.Image -> normalized PIL.Image'''
        tf_image = self.rgb_to_rgb(np.array(image.convert('RGB')))
        return Image.fromarray(tf_image)

    def rgb_to_rgb(self, image):
        '''Non-normalized RGB numpy array -> normalized RGB numpy array'''
        cv_image = self.n.transform(tf.tensor(image), self.target_means, self.target_stds)
        return cv_image.numpy()

    def jpeg_to_rgb(self, jpeg_string):
        '''Non-normalized compressed JPG string data -> normalized RGB numpy array'''
        tf_image = tf.image.decode_jpeg(jpeg_string)
        return self.tf_to_rgb(tf_image)

    def png_to_rgb(self, png_string):
        '''Non-normalized compressed PNG string data -> normalized RGB numpy array'''
        tf_image = tf.image.decode_png(png_string, channels=3)
        return self.tf_to_rgb(tf_image)

    def jpeg_to_jpeg(self, jpeg_string, quality=75):
        '''Non-normalized compressed JPG string data -> normalized compressed JPG string data'''
        np_image = self.jpeg_to_rgb(jpeg_string)
        with BytesIO() as output:
            Image.fromarray(np_image).save(output, format="JPEG", quality=quality)
            return output.getvalue()
