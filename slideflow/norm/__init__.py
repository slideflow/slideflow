import os
import cv2
import numpy as np
import slideflow as sf

from io import BytesIO
from os.path import join
from PIL import Image
from slideflow.dataset import Dataset
from slideflow.util import log
from tqdm import tqdm

if sf.backend() == 'tensorflow':
    import tensorflow as tf

from slideflow.norm import macenko,  \
                           reinhard, \
                           reinhard_fast, \
                           reinhard_mask, \
                           vahadane, \
                           augment

class GenericStainNormalizer:
    """Object to supervise stain normalization for images and efficiently convert between common image types."""

    vectorized = False
    backend = 'cv'
    normalizers = {
        'macenko':  macenko.Normalizer,
        'reinhard': reinhard.Normalizer,
        'reinhard_fast': reinhard_fast.Normalizer,
        'reinhard_mask': reinhard_mask.Normalizer,
        'vahadane': vahadane.Normalizer,
        'augment': augment.Normalizer
    }

    def __init__(self, method='reinhard', source=None):
        """Initializer. Establishes normalization method.

        Args:
            method (str): Either 'macenko', 'reinhard', or 'vahadane'. Defaults to 'reinhard'.
            source (str): Path to source image for normalizer. If not provided, defaults to an internal example image.
        """

        self.method = method
        self._source = source
        self.n = self.normalizers[method]()
        if not source:
            package_directory = os.path.dirname(os.path.abspath(__file__))
            source = join(package_directory, 'norm_tile.jpg')
        self.src_img = cv2.cvtColor(cv2.imread(source), cv2.COLOR_BGR2RGB)
        self.n.fit(self.src_img)

    def __repr__(self):
        src = "" if not self._source else ", source={!r}".format(self._source)
        return "GenericStainNormalizer(method={!r}{})".format(self.method, src)

    @property
    def target_means(self):
        return self.n.target_means

    @property
    def target_stds(self):
        return self.n.target_stds

    @property
    def stain_matrix_target(self):
        return self.n.stain_matrix_target

    @property
    def target_concentrations(self):
        return self.n.target_concentrations

    def fit(self, *args):
        '''if isinstance(args[0], Dataset):
            # Prime the normalizer
            dataset = args[0]
            dts = dataset.tensorflow(None, None, standardize=False, infinite=False)
            m, s = [], []
            pb = tqdm(desc='Fitting normalizer...', ncols=80, total=dataset.num_tiles)
            for i, slide in dts:
                _m, _s = self.n.fit(i.numpy())
                m += [tf.squeeze(_m)]
                s += [tf.squeeze(_s)]
                pb.update()
            dts_mean = np.array(m).mean(axis=0)
            dts_std = np.array(s).mean(axis=0)
            self.target_means = dts_mean
            self.target_stds = dts_std'''
        if isinstance(args[0], Dataset):
            raise NotImplementedError("Dataset fitting not supported for non-vectorized normalizers.")
        elif isinstance(args[0], np.ndarray) and len(args) == 1:
            self.n.fit(args[0])
        elif isinstance(args[0], str):
            self.src_img = cv2.cvtColor(cv2.imread(args[0]), cv2.COLOR_BGR2RGB)
            self.n.fit(self.src_img)
        else:
            raise ValueError('Unrecognized arguments for fit()')
            #TODO: support manually setting target_means and target_stds
        log.info(f"Fit normalizer to mean {self.target_means}, stddev {self.target_stds}")

    def batch_to_batch(self, *args):
        raise NotImplementedError(f"Vectorized functions not available for GenericStainNormalizer (method={self.method})")

    def tf_to_tf(self, image, *args):
        if isinstance(image, dict):
            image['tile_image'] = tf.py_function(self.tf_to_rgb, [image['tile_image']], tf.int32)
            return image, *args
        else:
            return tf.py_function(self.tf_to_rgb, [image], tf.int32), *args

    def tf_to_rgb(self, image):
        '''Non-normalized tensorflow image array -> normalized RGB numpy array'''
        return self.rgb_to_rgb(np.array(image))

    def rgb_to_rgb(self, image):
        '''Non-normalized RGB numpy array -> normalized RGB numpy array'''
        cv_image = self.n.transform(image)
        return cv_image

    def jpeg_to_rgb(self, jpeg_string):
        '''Non-normalized compressed JPG string data -> normalized RGB numpy array'''
        cv_image = cv2.imdecode(np.fromstring(jpeg_string, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return self.n.transform(cv_image)

    def png_to_rgb(self, png_string):
        '''Non-normalized compressed PNG string data -> normalized RGB numpy array'''
        return self.jpeg_to_rgb(png_string) # It should auto-detect format

    def jpeg_to_jpeg(self, jpeg_string, quality=75):
        '''Non-normalized compressed JPG string data -> normalized compressed JPG string data'''
        cv_image = self.jpeg_to_rgb(jpeg_string)
        with BytesIO() as output:
            Image.fromarray(cv_image).save(output, format="JPEG", quality=quality)
            return output.getvalue()

    def png_to_png(self, png_string):
        '''Non-normalized PNG string data -> normalized PNG string data'''
        cv_image = self.png_to_rgb(png_string)
        with BytesIO() as output:
            Image.fromarray(cv_image).save(output, format="PNG")
            return output.getvalue()

def autoselect(method, source=None, prefer_vectorized=True):
    '''Auto-selects best normalizer based on method, choosing backend-appropriate vectorized normalizer if available.'''

    if sf.backend() == 'tensorflow' and prefer_vectorized:
        from slideflow.norm.tensorflow import TensorflowStainNormalizer as VectorizedNormalizer
    elif sf.backend() == 'torch' and prefer_vectorized:
        from slideflow.norm.torch import TorchStainNormalizer as VectorizedNormalizer

    if prefer_vectorized and method in VectorizedNormalizer.normalizers:
        return VectorizedNormalizer(method, source)
    else:
        return GenericStainNormalizer(method, source)