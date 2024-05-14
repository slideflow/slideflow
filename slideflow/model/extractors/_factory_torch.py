"""Factory for building PyTorch feature extractors."""

import numpy as np
import slideflow as sf
from slideflow import errors

from ._slide import features_from_slide
from ._registry import _torch_extractors, is_torch_extractor, register_torch
from ..base import BaseFeatureExtractor


def build_torch_feature_extractor(name, **kwargs):
    if is_torch_extractor(name):
        if name in _torch_extractors:
            return _torch_extractors[name](**kwargs)
        else:
            return _torch_extractors[name+'_imagenet'](**kwargs)
    else:
        raise errors.InvalidFeatureExtractor(f"Unrecognized feature extractor: {name}")

# -----------------------------------------------------------------------------

@register_torch
def uni(weights, **kwargs):
    from .uni import UNIFeatures
    return UNIFeatures(weights, **kwargs)

@register_torch
def vit(**kwargs):
    from .vit import ViTFeatures
    return ViTFeatures(**kwargs)

@register_torch
def histossl(**kwargs):
    from .histossl import HistoSSLFeatures
    return HistoSSLFeatures(**kwargs)

@register_torch
def plip(**kwargs):
    from .plip import PLIPFeatures
    return PLIPFeatures(**kwargs)

@register_torch
def dinov2(**kwargs):
    from .dinov2 import DinoV2Features
    return DinoV2Features(**kwargs)

@register_torch
def ctranspath(**kwargs):
    from .ctranspath import CTransPathFeatures
    return CTransPathFeatures(**kwargs)

@register_torch
def retccl(**kwargs):
    from .retccl import RetCCLFeatures
    return RetCCLFeatures(**kwargs)

@register_torch
def resnet18_imagenet(tile_px, **kwargs):
    return TorchImagenetLayerExtractor('resnet18', tile_px, **kwargs)

@register_torch
def resnet50_imagenet(tile_px, **kwargs):
    return TorchImagenetLayerExtractor('resnet50', tile_px, **kwargs)

@register_torch
def alexnet_imagenet(tile_px, **kwargs):
    return TorchImagenetLayerExtractor('alexnet', tile_px, **kwargs)

@register_torch
def squeezenet_imagenet(tile_px, **kwargs):
    return TorchImagenetLayerExtractor('squeezenet', tile_px, **kwargs)

@register_torch
def densenet_imagenet(tile_px, **kwargs):
    return TorchImagenetLayerExtractor('densenet', tile_px, **kwargs)

@register_torch
def inception_imagenet(tile_px, **kwargs):
    return TorchImagenetLayerExtractor('inception', tile_px, **kwargs)

@register_torch
def googlenet_imagenet(tile_px, **kwargs):
    return TorchImagenetLayerExtractor('googlenet', tile_px, **kwargs)

@register_torch
def shufflenet_imagenet(tile_px, **kwargs):
    return TorchImagenetLayerExtractor('shufflenet', tile_px, **kwargs)

@register_torch
def resnext50_32x4d_imagenet(tile_px, **kwargs):
    return TorchImagenetLayerExtractor('resnext50_32x4d', tile_px, **kwargs)

@register_torch
def vgg16_imagenet(tile_px, **kwargs):
    return TorchImagenetLayerExtractor('vgg16', tile_px, **kwargs)

@register_torch
def mobilenet_v2_imagenet(tile_px, **kwargs):
    return TorchImagenetLayerExtractor('mobilenet_v2', tile_px, **kwargs)

@register_torch
def mobilenet_v3_small_imagenet(tile_px, **kwargs):
    return TorchImagenetLayerExtractor('mobilenet_v3_small', tile_px, **kwargs)

@register_torch
def mobilenet_v3_large_imagenet(tile_px, **kwargs):
    return TorchImagenetLayerExtractor('mobilenet_v3_large', tile_px, **kwargs)

@register_torch
def wide_resnet50_2_imagenet(tile_px, **kwargs):
    return TorchImagenetLayerExtractor('wide_resnet50_2', tile_px, **kwargs)

@register_torch
def mnasnet_imagenet(tile_px, **kwargs):
    return TorchImagenetLayerExtractor('mnasnet', tile_px, **kwargs)

@register_torch
def xception_imagenet(tile_px, **kwargs):
    from torchvision import transforms
    extractor = TorchImagenetLayerExtractor('xception', tile_px, **kwargs)
    extractor.transform = transforms.Compose([
        transforms.Lambda(lambda x: x / 255.),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
    ])
    return extractor

@register_torch
def nasnet_large_imagenet(tile_px, **kwargs):
    from torchvision import transforms
    extractor = TorchImagenetLayerExtractor('nasnet_large', tile_px, **kwargs)
    extractor.transform = transforms.Compose([
        transforms.Lambda(lambda x: x / 255.),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
    ])
    return extractor

# -----------------------------------------------------------------------------

class TorchFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for PyTorch models."""

    def __init__(self, channels_last=False, mixed_precision=False):
        from .. import torch_utils

        super().__init__(backend='torch')
        self.device = torch_utils.get_device()
        self.channels_last = channels_last
        self.mixed_precision = mixed_precision

    def __call__(self, obj, **kwargs):
        """Generate features for a batch of images or a WSI."""
        import torch
        from slideflow.model.torch import autocast

        if isinstance(obj, sf.WSI):
            grid = features_from_slide(self, obj, **kwargs)
            return np.ma.masked_where(grid == sf.heatmap.MASK, grid)
        elif kwargs:
            raise ValueError(
                f"{self.__class__.__name__} does not accept keyword arguments "
                "when extracting features from a batch of images."
            )
        assert obj.dtype == torch.uint8
        obj = obj.to(self.device)
        obj = self.transform(obj)
        with autocast(self.device.type, mixed_precision=self.mixed_precision):
            with torch.no_grad():
                if self.channels_last:
                    obj = obj.to(memory_format=torch.channels_last)
                return self.model(obj)


class TorchImagenetLayerExtractor(BaseFeatureExtractor):
    """Feature extractor that calculates layer activations for
    imagenet-pretrained PyTorch models."""

    def __init__(self, model_name, tile_px, device=None, **kwargs):
        super().__init__(backend='torch')

        from ..torch import ModelParams, Features
        from .. import torch_utils
        from torchvision import transforms


        self.device = torch_utils.get_device(device)
        _hp = ModelParams(tile_px=tile_px, model=model_name, include_top=False, hidden_layers=0)
        model = _hp.build_model(num_classes=1, pretrain='imagenet').to(self.device)
        self.model_name = model_name
        self.ftrs = Features.from_model(model, tile_px=tile_px, **kwargs)
        self.tag = model_name + "_" + '-'.join(self.ftrs.layers)
        self.num_features = self.ftrs.num_features
        self._tile_px = tile_px

        # Normalization for Imagenet pretrained models
        # as described here: https://pytorch.org/vision/0.11/models.html
        all_transforms = [
            transforms.Lambda(lambda x: x / 255.),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
        ]
        self.transform = transforms.Compose(all_transforms)
        self.preprocess_kwargs = dict(standardize=False)

    @property
    def mixed_precision(self):
        return self.ftrs.mixed_precision

    @mixed_precision.setter
    def mixed_precision(self, value):
        self.ftrs.mixed_precision = value

    @property
    def channels_last(self):
        return self.ftrs.channels_last

    @channels_last.setter
    def channels_last(self, value):
        self.ftrs.channels_last = value

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "<TorchImagenetLayerExtractor model={} layers={} n_features={}>".format(
            self.model_name,
            self.ftrs.layers,
            self.num_features,
        )

    def __call__(self, obj, **kwargs):
        """Generate features for a batch of images or a WSI."""
        if isinstance(obj, sf.WSI):
            grid = features_from_slide(self, obj, **kwargs)
            return np.ma.masked_where(grid == sf.heatmap.MASK, grid)
        elif kwargs:
            raise ValueError(
                f"{self.__class__.__name__} does not accept keyword arguments "
                "when extracting features from a batch of images."
            )
        else:
            import torch
            assert obj.dtype == torch.uint8
            obj = self.transform(obj).to(self.device)
            return self.ftrs._predict(obj)

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.model.build_feature_extractor()``.

        """
        return {
            'class': 'slideflow.model.extractors.TorchImagenetLayerExtractor',
            'kwargs': {
                'model_name': self.model_name,
                'tile_px': self._tile_px,
                'layers': self.ftrs.layers,
                'pooling': self.ftrs._pooling,
            }
        }