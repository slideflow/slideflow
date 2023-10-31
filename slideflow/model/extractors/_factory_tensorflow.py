"""Factory for building Tensorflow feature extractors."""

import importlib
import numpy as np
import slideflow as sf
from slideflow import errors

from ._slide import features_from_slide
from ._registry import _tf_extractors, is_tensorflow_extractor, register_tf
from ..base import BaseFeatureExtractor


def build_tensorflow_feature_extractor(name, **kwargs):
    if is_tensorflow_extractor(name):
        if name in _tf_extractors:
            return _tf_extractors[name](**kwargs)
        else:
            return _tf_extractors[name+'_imagenet'](**kwargs)
    else:
        raise errors.InvalidFeatureExtractor(f"Unrecognized feature extractor: {name}")

# -----------------------------------------------------------------------------

@register_tf
def simclr(ckpt, **kwargs):
    from .simclr import SimCLR_Features
    return SimCLR_Features(ckpt, **kwargs)

@register_tf
def xception_imagenet(tile_px, **kwargs):
    return TensorflowImagenetLayerExtractor('xception', tile_px, **kwargs)

@register_tf
def vgg16_imagenet(tile_px, **kwargs):
    return TensorflowImagenetLayerExtractor('vgg16', tile_px, **kwargs)

@register_tf
def vgg19_imagenet(tile_px, **kwargs):
    return TensorflowImagenetLayerExtractor('vgg19', tile_px, **kwargs)

@register_tf
def resnet50_imagenet(tile_px, **kwargs):
    return TensorflowImagenetLayerExtractor('resnet50', tile_px, **kwargs)

@register_tf
def resnet101_imagenet(tile_px, **kwargs):
    return TensorflowImagenetLayerExtractor('resnet101', tile_px, **kwargs)

@register_tf
def resnet101_imagenet(tile_px, **kwargs):
    return TensorflowImagenetLayerExtractor('resnet101', tile_px, **kwargs)

@register_tf
def resnet152_imagenet(tile_px, **kwargs):
    return TensorflowImagenetLayerExtractor('resnet152', tile_px, **kwargs)

@register_tf
def resnet152_imagenet(tile_px, **kwargs):
    return TensorflowImagenetLayerExtractor('resnet152', tile_px, **kwargs)

@register_tf
def resnet50_v2_imagenet(tile_px, **kwargs):
    return TensorflowImagenetLayerExtractor('resnet50_v2', tile_px, **kwargs)

@register_tf
def resnet101_v2_imagenet(tile_px, **kwargs):
    return TensorflowImagenetLayerExtractor('resnet101_v2', tile_px, **kwargs)

@register_tf
def resnet152_v2_imagenet(tile_px, **kwargs):
    return TensorflowImagenetLayerExtractor('resnet152_v2', tile_px, **kwargs)

@register_tf
def inception_imagenet(tile_px, **kwargs):
    return TensorflowImagenetLayerExtractor('inception', tile_px, **kwargs)

@register_tf
def nasnet_large_imagenet(tile_px, **kwargs):
    return TensorflowImagenetLayerExtractor('nasnet_large', tile_px, **kwargs)

@register_tf
def inception_resnet_v2_imagenet(tile_px, **kwargs):
    return TensorflowImagenetLayerExtractor('inception_resnet_v2', tile_px, **kwargs)

@register_tf
def mobilenet_imagenet(tile_px, **kwargs):
    return TensorflowImagenetLayerExtractor('mobilenet', tile_px, **kwargs)

@register_tf
def mobilenet_v2_imagenet(tile_px, **kwargs):
    return TensorflowImagenetLayerExtractor('mobilenet_v2', tile_px, **kwargs)

# -----------------------------------------------------------------------------

class TensorflowImagenetLayerExtractor(BaseFeatureExtractor):
    """Feature extractor that calculates layer activations for
    imagenet-pretrained Tensorflow models."""

    def __init__(self, model_name, tile_px, **kwargs):
        super().__init__(backend='tensorflow')

        from ..tensorflow import ModelParams, Features
        import tensorflow as tf

        _hp = ModelParams(tile_px=tile_px, model=model_name, include_top=False, hidden_layers=0)
        model = _hp.build_model(num_classes=1, pretrain='imagenet')
        submodule = importlib.import_module(f'tensorflow.keras.applications.{model_name}')
        self.model_name = model_name
        self.ftrs = Features.from_model(model, **kwargs)
        self.tag = model_name + "_" + '-'.join(self.ftrs.layers)
        self.num_features = self.ftrs.num_features
        self._tile_px = tile_px

        @tf.function
        def _transform(x):
            x = tf.cast(x, tf.float32)
            return submodule.preprocess_input(x)

        self.transform = _transform
        self.preprocess_kwargs = dict(standardize=False)

    @property
    def tile_px(self):
        return self._tile_px

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "<TensorflowImagenetLayerExtractor model={} layers={} n_features={}>".format(
            self.model_name,
            self.ftrs.layers,
            self.num_features,
        )

    def _predict(self, batch_images):
        """Generate features for a batch of images."""
        import tensorflow as tf
        if batch_images.dtype == tf.uint8:
            batch_images = tf.cast(batch_images, tf.float32)
            batch_images = self.transform(batch_images)
        return self.ftrs._predict(batch_images)

    def __call__(self, obj, **kwargs):
        """Generate features for a batch of images or a WSI."""
        if isinstance(obj, sf.WSI):
            grid = features_from_slide(self, obj, preprocess_fn=self.transform, **kwargs)
            return np.ma.masked_where(grid == sf.heatmap.MASK, grid)
        elif kwargs:
            raise ValueError(
                f"{self.__class__.__name__} does not accept keyword arguments "
                "when extracting features from a batch of images."
            )
        else:
            return self._predict(obj)

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.model.build_feature_extractor()``.

        """
        return {
            'class': 'slideflow.model.extractors.TensorflowImagenetLayerExtractor',
            'kwargs': {
                'model_name': self.model_name,
                'tile_px': self._tile_px,
                'layers': self.ftrs.layers,
                'pooling': self.ftrs._pooling,
            }
        }

