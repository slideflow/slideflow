"""Factory for building Tensorflow feature extractors."""

import importlib
from slideflow import errors

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
def xception_imagenet(tile_px, **kwargs):
    return _TensorflowImagenetLayerExtractor('xception', tile_px, **kwargs)

@register_tf
def vgg16_imagenet(tile_px, **kwargs):
    return _TensorflowImagenetLayerExtractor('vgg16', tile_px, **kwargs)

@register_tf
def vgg19_imagenet(tile_px, **kwargs):
    return _TensorflowImagenetLayerExtractor('vgg19', tile_px, **kwargs)

@register_tf
def resnet50_imagenet(tile_px, **kwargs):
    return _TensorflowImagenetLayerExtractor('resnet50', tile_px, **kwargs)

@register_tf
def resnet101_imagenet(tile_px, **kwargs):
    return _TensorflowImagenetLayerExtractor('resnet101', tile_px, **kwargs)

@register_tf
def resnet101_imagenet(tile_px, **kwargs):
    return _TensorflowImagenetLayerExtractor('resnet101', tile_px, **kwargs)

@register_tf
def resnet152_imagenet(tile_px, **kwargs):
    return _TensorflowImagenetLayerExtractor('resnet152', tile_px, **kwargs)

@register_tf
def resnet152_imagenet(tile_px, **kwargs):
    return _TensorflowImagenetLayerExtractor('resnet152', tile_px, **kwargs)

@register_tf
def resnet50_v2_imagenet(tile_px, **kwargs):
    return _TensorflowImagenetLayerExtractor('resnet50_v2', tile_px, **kwargs)

@register_tf
def resnet101_v2_imagenet(tile_px, **kwargs):
    return _TensorflowImagenetLayerExtractor('resnet101_v2', tile_px, **kwargs)

@register_tf
def resnet152_v2_imagenet(tile_px, **kwargs):
    return _TensorflowImagenetLayerExtractor('resnet152_v2', tile_px, **kwargs)

@register_tf
def inception_imagenet(tile_px, **kwargs):
    return _TensorflowImagenetLayerExtractor('inception', tile_px, **kwargs)

@register_tf
def nasnet_large_imagenet(tile_px, **kwargs):
    return _TensorflowImagenetLayerExtractor('nasnet_large', tile_px, **kwargs)

@register_tf
def inception_resnet_v2_imagenet(tile_px, **kwargs):
    return _TensorflowImagenetLayerExtractor('inception_resnet_v2', tile_px, **kwargs)

@register_tf
def mobilenet_imagenet(tile_px, **kwargs):
    return _TensorflowImagenetLayerExtractor('mobilenet', tile_px, **kwargs)

@register_tf
def mobilenet_v2_imagenet(tile_px, **kwargs):
    return _TensorflowImagenetLayerExtractor('mobilenet_v2', tile_px, **kwargs)

# -----------------------------------------------------------------------------

class _TensorflowImagenetLayerExtractor(BaseFeatureExtractor):
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
        self.num_classes = 0

        @tf.function
        def _transform(x):
            x = tf.cast(x, tf.float32)
            return submodule.preprocess_input(x)

        self.transform = _transform
        self.preprocess_kwargs = dict(standardize=False)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "<TensorflowImagenetLayerExtractor model={} layers={} n_features={}>".format(
            self.model_name,
            self.ftrs.layers,
            self.num_features,
        )

    def __call__(self, batch_images):
        import tensorflow as tf
        assert batch_images.dtype == tf.uint8
        batch_images = tf.cast(batch_images, tf.float32)
        batch_images = self.transform(batch_images)
        return self.ftrs._predict(batch_images)

