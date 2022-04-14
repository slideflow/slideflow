import os
import cv2
import numpy as np
import slideflow as sf
import multiprocessing as mp

from io import BytesIO
from os.path import join
from PIL import Image
from slideflow.dataset import Dataset
from slideflow.util import log
from tqdm import tqdm
from slideflow import errors

if sf.backend() == 'tensorflow':
    import tensorflow as tf

from slideflow.norm import macenko,  \
                           reinhard, \
                           reinhard_fast, \
                           reinhard_mask, \
                           vahadane, \
                           augment


class StainNormalizer:
    """Supervises stain normalization for images and efficiently
    convert between common image types."""

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
            method (str): Either 'macenko', 'reinhard', or 'vahadane'.
                Defaults to 'reinhard'.
            source (str): Path to source image for normalizer. If not provided,
                defaults to slideflow.norm.norm_tile.jpg
        """

        self.method = method
        self._source = source
        self.n = self.normalizers[method]()
        if not source:
            package_directory = os.path.dirname(os.path.abspath(__file__))
            source = join(package_directory, 'norm_tile.jpg')
        if source != 'dataset':
            self.src_img = cv2.cvtColor(cv2.imread(source), cv2.COLOR_BGR2RGB)
            self.n.fit(self.src_img)

    def __repr__(self):
        src = "" if not self._source else ", source={!r}".format(self._source)
        return "StainNormalizer(method={!r}{})".format(self.method, src)

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

    def fit(self, *args, target_means=None, target_stds=None,
            stain_matrix_target=None, target_concentrations=None,
            batch_size=64, num_threads='auto'):

        if (len(args)
           and isinstance(args[0], Dataset)
           and self.method in ('reinhard', 'reinhard_fast')):
            # Set up thread pool
            if num_threads == 'auto':
                num_threads = os.cpu_count()
            log.debug(f"Setting up pool (size={num_threads}) for norm fitting")
            log.debug(f"Using normalizer batch size of {batch_size}")
            pool = mp.dummy.Pool(num_threads)

            dataset = args[0]
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
            means, stds = [], []
            pb = tqdm(
                desc='Fitting normalizer...',
                ncols=80,
                total=dataset.num_tiles
            )
            for img_batch, slide in dts:
                if sf.backend() == 'torch':
                    img_batch = img_batch.permute(0, 2, 3, 1)  # BCWH -> BWHC

                mapped = pool.imap(lambda x: self.n.fit(x.numpy()), img_batch)
                for _m, _s in mapped:
                    means += [np.squeeze(_m)]
                    stds += [np.squeeze(_s)]
                pb.update(batch_size)
            self.n.target_means = np.array(means).mean(axis=0)
            self.n.target_stds = np.array(stds).mean(axis=0)

        elif len(args) and isinstance(args[0], Dataset):
            msg = f"Dataset fitting not supported for method '{self.method}'."
            raise NotImplementedError(msg)

        elif len(args) and isinstance(args[0], np.ndarray) and len(args) == 1:
            self.n.fit(args[0])

        elif len(args) and isinstance(args[0], str):
            self.src_img = cv2.cvtColor(cv2.imread(args[0]), cv2.COLOR_BGR2RGB)
            self.n.fit(self.src_img)

        elif target_means is not None:
            self.n.target_means = np.array(target_means)
            self.n.target_stds = np.array(target_stds)

        elif (stain_matrix_target is not None
              and target_concentrations is not None):
            self.n.stain_matrix_target = np.array(stain_matrix_target)
            self.n.target_concentrations = np.array(target_concentrations)

        elif stain_matrix_target is not None:
            self.n.stain_matrix_target = np.array(stain_matrix_target)

        else:
            raise errors.NormalizerError(f'Unrecognized args for fit: {args}')

        fit_msg = f"Fit normalizer to mean {self.target_means}"
        fit_msg += f", stddev {self.target_stds}"
        log.info(fit_msg)

    def get_fit(self):
        return {
            'target_means': self.n.target_means.tolist(),
            'target_stds': self.n.target_stds.tolist(),
            'stain_matrix_target': self.n.stain_matrix_target.tolist(),
            'target_concentrations': self.n.target_concentrations.tolist()
        }

    def batch_to_batch(self, *args):
        msg = "Vectorized functions not available for StainNormalizer "
        msg += f"(method={self.method})"
        raise NotImplementedError(msg)

    def tf_to_tf(self, image, *args):
        if isinstance(image, dict):
            image['tile_image'] = tf.py_function(
                self.tf_to_rgb,
                [image['tile_image']],
                tf.int32
            )
        else:
            image = tf.py_function(self.tf_to_rgb, [image], tf.int32)
        return tuple([image] + list(args))

    def tf_to_rgb(self, image):
        '''Non-normalized tensorflow RGB array -> normalized RGB numpy array'''
        return self.rgb_to_rgb(np.array(image))

    def rgb_to_rgb(self, image):
        '''Non-normalized RGB numpy array -> normalized RGB numpy array'''
        cv_image = self.n.transform(image)
        return cv_image

    def jpeg_to_rgb(self, jpeg_string):
        '''Non-normalized compressed JPG data -> normalized RGB numpy array'''
        cv_image = cv2.imdecode(
            np.fromstring(jpeg_string, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return self.n.transform(cv_image)

    def png_to_rgb(self, png_string):
        '''Non-normalized compressed PNG data -> normalized RGB numpy array'''
        return self.jpeg_to_rgb(png_string)  # It should auto-detect format

    def jpeg_to_jpeg(self, jpeg_string, quality=75):
        '''Non-normalized compressed JPG string data -> normalized compressed JPG data'''  # noqa F401
        cv_image = self.jpeg_to_rgb(jpeg_string)
        with BytesIO() as output:
            Image.fromarray(cv_image).save(
                output,
                format="JPEG",
                quality=quality
            )
            return output.getvalue()

    def png_to_png(self, png_string):
        '''Non-normalized PNG string data -> normalized PNG string data'''
        cv_image = self.png_to_rgb(png_string)
        with BytesIO() as output:
            Image.fromarray(cv_image).save(output, format="PNG")
            return output.getvalue()


def autoselect(method, source=None, prefer_vectorized=True):
    '''Auto-selects best normalizer based on method,
    choosing backend-appropriate vectorized normalizer if available.
    '''

    if sf.backend() == 'tensorflow' and prefer_vectorized:
        from slideflow.norm.tensorflow import TensorflowStainNormalizer as VectorizedNormalizer  # noqa F401
    elif sf.backend() == 'torch' and prefer_vectorized:
        from slideflow.norm.torch import TorchStainNormalizer as VectorizedNormalizer  # noqa F401
    elif prefer_vectorized:
        raise errors.BackendError(f"Unknown backend {sf.backend()}")

    if prefer_vectorized and method in VectorizedNormalizer.normalizers:
        return VectorizedNormalizer(method, source)
    else:
        return StainNormalizer(method, source)
