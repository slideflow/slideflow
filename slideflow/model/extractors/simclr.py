import slideflow as sf
import tensorflow as tf
import numpy as np

from typing import Optional
from slideflow import simclr

from ..base import BaseFeatureExtractor
from ._slide import features_from_slide

# -----------------------------------------------------------------------------

def crop(image, size):
    """Crops to center of an image.

    Args:
        image (tf.Tensor): Image tensor.
        size (int): Size of crop (width/height).

    Returns:
        tf.Tensor: Cropped image.
    """
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    top = (h - size) // 2
    left = (w - size) // 2
    return image[top:top+size, left:left+size]


class SimCLR_Features(BaseFeatureExtractor):
    """SimCLR feature extractor.

    Loads trained SimCLRv2 model and provides interface for generating
    features from a batch of images or a WSI.

    Feature dimensions: <variable>

    GitHub: https://github.com/jamesdolezal/simclr
    """

    tag = 'simclr'

    def __init__(
        self,
        ckpt: str,
        center_crop: Optional[bool] = None,
        tile_px: Optional[int] = None,
        resize_crop: bool = True,
    ) -> None:
        super().__init__(backend='tensorflow')

        self.model = simclr.load(ckpt)
        self.simclr_args = simclr.load_model_args(ckpt)
        self.num_features = self.simclr_args.proj_out_dim
        self.num_classes = self.simclr_args.num_classes
        self._model_path = ckpt
        self._resize_crop = resize_crop

        # ---------------------------------------------------------------------
        preprocess = simclr.data.get_preprocess_fn(
            is_training=False,
            is_pretrain=False,
            image_size=self.simclr_args.image_size,
            center_crop=resize_crop  # The SimCLR center-crop crops an image by
                                     # the proportion 0.875 and then resizes to
                                     # the original dimensions. This differs
                                     # from the center cropping below.
        )

        # Determine whether input images should be cropped.
        # Images should be cropped if center_crop is True, or if tile_px is
        # specified and is not equal to the SimCLR training image size.
        self._center_crop = (center_crop
                             or (tile_px is not None
                                 and tile_px != self.simclr_args.image_size))
        if self._center_crop:
            self.transform = lambda x:  preprocess(
                                            crop(x, self.simclr_args.image_size)
                                        )
        else:
            self.transform = preprocess
        if center_crop is False and self._center_crop:
            raise ValueError(
                'Cannot use both center_crop and tile_px options.'
                'If tile_px is specified, center_crop is automatically'
                'enabled.'
            )

        # Image preprocessing should preferentially be done within
        # a tf.data.Dataset pipeline, as this greatly improves performance.
        # The below preprocessing kwargs are passed to `Dataset.tensorflow()`
        # when creating an input pipeline for feature generation with
        # `DatasetFeatures`.

        # However, for flexibility, if this `SimCLR_Features`` interface is used
        # on unprocessed (uint8) images outside a Dataset pipeline, the
        # transform function will be applied to each image automatically when
        # using the __call_function, via `self.transform()`.
        self.preprocess_kwargs = dict(
            standardize=False,
            transform=self.transform
        )
        # ---------------------------------------------------------------------

    @tf.function
    def _predict(self, batch_images):
        if batch_images.dtype == tf.uint8:
            batch_images = tf.map_fn(
                self.transform,
                batch_images,
                dtype=tf.float32
            )
        return self.model(batch_images, training=False)[0]

    def __call__(self, obj, **kwargs):
        """Generate features for a batch of images or a WSI."""
        if isinstance(obj, sf.WSI):
            grid = features_from_slide(self, obj, preprocess_fn=self.transform, **kwargs)
            return np.ma.masked_where(grid == -99, grid)
        elif kwargs:
            raise ValueError(
                f"{self.__class__.__name__} does not accept keyword arguments "
                "when extracting features from a batch of images."
            )
        assert obj.dtype in (tf.float32, tf.uint8)
        return self._predict(obj)

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.model.build_feature_extractor()``.

        """
        return {
            'class': 'slideflow.model.extractors.simclr.SimCLR_Features',
            'kwargs': {
                'center_crop': self._center_crop,
                'resize_crop': self._resize_crop,
                'ckpt': self._model_path
            }
        }