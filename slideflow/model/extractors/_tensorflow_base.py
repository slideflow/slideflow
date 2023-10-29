import numpy as np
import slideflow as sf
import tensorflow as tf

from ..base import BaseFeatureExtractor
from ._slide import features_from_slide


class TensorflowFeatureExtractor(BaseFeatureExtractor):
    """Base feature extractor for Tensorflow models."""

    def __init__(self):
        super().__init__(backend='tensorflow')

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
        import tensorflow as tf
        if isinstance(obj, sf.WSI):
            grid = features_from_slide(self, obj, preprocess_fn=self.transform, **kwargs)
            return np.ma.masked_where(grid == sf.heatmap.MASK, grid)
        elif kwargs:
            raise ValueError(
                f"{self.__class__.__name__} does not accept keyword arguments "
                "when extracting features from a batch of images."
            )
        assert obj.dtype in (tf.float32, tf.uint8)
        return self._predict(obj)