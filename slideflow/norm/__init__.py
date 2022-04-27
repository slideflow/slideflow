import os
import cv2
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from io import BytesIO
from os.path import join
from PIL import Image
from typing import Optional, Union, Dict, Any, Tuple, List, TYPE_CHECKING

import slideflow as sf
from slideflow.dataset import Dataset
from slideflow.util import log
from slideflow import errors
from slideflow.norm.utils import BaseNormalizer

if sf.backend() == 'tensorflow':
    import tensorflow as tf
elif TYPE_CHECKING:
    import tensorflow as tf

from slideflow.norm import (macenko,
                            reinhard,
                            reinhard_fast,
                            reinhard_mask,
                            vahadane,
                            augment)


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
    }  # type: Dict[str, Any]

    def __init__(
        self,
        method: str = 'reinhard',
        source: Optional[str] = None
    ) -> None:
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
    def target_means(self) -> Optional[np.ndarray]:
        return self.n.target_means

    @property
    def target_stds(self) -> Optional[np.ndarray]:
        return self.n.target_stds

    @property
    def stain_matrix_target(self) -> Optional[np.ndarray]:
        return self.n.stain_matrix_target

    @property
    def target_concentrations(self) -> Optional[np.ndarray]:
        return self.n.target_concentrations

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
            target_means (Optional[np.ndarray], optional): Target means.
                Defaults to None.
            target_stds (Optional[np.ndarray], optional): Target stds.
                Defaults to None.
            stain_matrix_target (Optional[np.ndarray], optional): Target stain
                matrix. Defaults to None.
            target_concentrations (Optional[np.ndarray], optional): Target
                concentrations. Defaults to None.
            batch_size (int, optional): Batch size during fitting, if fitting
                to dataset. Defaults to 64.
            num_threads (Union[str, int], optional): Number of threads to use
                during fitting, if fitting to a dataset. Defaults to 'auto'.

        Raises:
            NotImplementedError: If attempting to fit Dataset using
            non-vectorized normalizer.

            errors.NormalizerError: If unrecognized arguments are provided.
        """
        if (len(args)
           and isinstance(args[0], Dataset)
           and self.method in ('reinhard', 'reinhard_fast')):
            # Set up thread pool
            if num_threads == 'auto':
                num_threads = os.cpu_count()  # type: ignore
            log.debug(f"Setting up pool (size={num_threads}) for norm fitting")
            log.debug(f"Using normalizer batch size of {batch_size}")
            pool = mp.dummy.Pool(num_threads)  # type: ignore

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
            pool.close()

        elif len(args) and isinstance(args[0], Dataset):
            raise NotImplementedError(
                f"Dataset fitting not supported for method '{self.method}'."
            )

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
        log.info(
            f"Fit normalizer to mean {self.target_means}"
            f", stddev {self.target_stds}"
        )

    def get_fit(self) -> Dict[str, Optional[List[float]]]:
        return {
            'target_means': self.n.target_means.tolist(),
            'target_stds': self.n.target_stds.tolist(),
            'stain_matrix_target': self.n.stain_matrix_target.tolist(),
            'target_concentrations': self.n.target_concentrations.tolist()
        }

    def tf_to_tf(
        self,
        image: Union[Dict, "tf.Tensor"],
        *args: Any
    ) -> Tuple[Union[Dict, "tf.Tensor"], ...]:
        if isinstance(image, dict):
            image['tile_image'] = tf.py_function(
                self.tf_to_rgb,
                [image['tile_image']],
                tf.int32
            )
        else:
            image = tf.py_function(self.tf_to_rgb, [image], tf.int32)
        return tuple([image] + list(args))

    def tf_to_rgb(self, image: "tf.Tensor") -> np.ndarray:
        '''Non-normalized tensorflow RGB array -> normalized RGB numpy array'''
        return self.rgb_to_rgb(np.array(image))

    def rgb_to_rgb(self, image: np.ndarray) -> np.ndarray:
        '''Non-normalized RGB numpy array -> normalized RGB numpy array'''
        cv_image = self.n.transform(image)
        return cv_image

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
    source: Optional[str] = None,
    prefer_vectorized: bool = True
) -> StainNormalizer:
    '''Auto-selects best normalizer based on method,
    choosing backend-appropriate vectorized normalizer if available.
    '''
    if sf.backend() == 'tensorflow' and prefer_vectorized:
        from slideflow.norm.tensorflow import TensorflowStainNormalizer
        if method in TensorflowStainNormalizer.normalizers:
            return TensorflowStainNormalizer(method, source)
    elif sf.backend() == 'torch' and prefer_vectorized:
        from slideflow.norm.torch import TorchStainNormalizer
        if method in TorchStainNormalizer.normalizers:
            return TorchStainNormalizer(method, source)
    return StainNormalizer(method, source)
