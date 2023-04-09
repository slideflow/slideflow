"""Factory for building PyTorch feature extractors."""

from slideflow import errors

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
def ctranspath(tile_px, **kwargs):
    from .ctranspath import CTransPathFeatures
    return CTransPathFeatures(center_crop=(tile_px != 224), **kwargs)

@register_torch
def retccl(tile_px, **kwargs):
    from .retccl import RetCCLFeatures
    return RetCCLFeatures(center_crop=(tile_px != 256), **kwargs)

@register_torch
def resnet18_imagenet(tile_px, **kwargs):
    return _TorchImagenetLayerExtractor('resnet18', tile_px, **kwargs)

@register_torch
def resnet50_imagenet(tile_px, **kwargs):
    return _TorchImagenetLayerExtractor('resnet50', tile_px, **kwargs)

@register_torch
def alexnet_imagenet(tile_px, **kwargs):
    return _TorchImagenetLayerExtractor('alexnet', tile_px, **kwargs)

@register_torch
def squeezenet_imagenet(tile_px, **kwargs):
    return _TorchImagenetLayerExtractor('squeezenet', tile_px, **kwargs)

@register_torch
def densenet_imagenet(tile_px, **kwargs):
    return _TorchImagenetLayerExtractor('densenet', tile_px, **kwargs)

@register_torch
def inception_imagenet(tile_px, **kwargs):
    return _TorchImagenetLayerExtractor('inception', tile_px, **kwargs)

@register_torch
def googlenet_imagenet(tile_px, **kwargs):
    return _TorchImagenetLayerExtractor('googlenet', tile_px, **kwargs)

@register_torch
def shufflenet_imagenet(tile_px, **kwargs):
    return _TorchImagenetLayerExtractor('shufflenet', tile_px, **kwargs)

@register_torch
def resnext50_32x4d_imagenet(tile_px, **kwargs):
    return _TorchImagenetLayerExtractor('resnext50_32x4d', tile_px, **kwargs)

@register_torch
def vgg16_imagenet(tile_px, **kwargs):
    return _TorchImagenetLayerExtractor('vgg16', tile_px, **kwargs)

@register_torch
def mobilenet_v2_imagenet(tile_px, **kwargs):
    return _TorchImagenetLayerExtractor('mobilenet_v2', tile_px, **kwargs)

@register_torch
def mobilenet_v3_small_imagenet(tile_px, **kwargs):
    return _TorchImagenetLayerExtractor('mobilenet_v3_small', tile_px, **kwargs)

@register_torch
def mobilenet_v3_large_imagenet(tile_px, **kwargs):
    return _TorchImagenetLayerExtractor('mobilenet_v3_large', tile_px, **kwargs)

@register_torch
def wide_resnet50_2_imagenet(tile_px, **kwargs):
    return _TorchImagenetLayerExtractor('wide_resnet50_2', tile_px, **kwargs)

@register_torch
def mnasnet_imagenet(tile_px, **kwargs):
    return _TorchImagenetLayerExtractor('mnasnet', tile_px, **kwargs)

@register_torch
def xception_imagenet(tile_px, **kwargs):
    from torchvision import transforms
    extractor = _TorchImagenetLayerExtractor('xception', tile_px, **kwargs)
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
    extractor = _TorchImagenetLayerExtractor('nasnet_large', tile_px, **kwargs)
    extractor.transform = transforms.Compose([
        transforms.Lambda(lambda x: x / 255.),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
    ])
    return extractor

# -----------------------------------------------------------------------------

class _TorchImagenetLayerExtractor(BaseFeatureExtractor):
    """Feature extractor that calculates layer activations for
    imagenet-pretrained PyTorch models."""

    def __init__(self, model_name, tile_px, device=None, **kwargs):
        super().__init__(backend='torch')

        import torch
        from ..torch import ModelParams, Features
        from torchvision import transforms

        self.device = device if device is not None else torch.device('cuda')
        _hp = ModelParams(tile_px=tile_px, model=model_name, include_top=False, hidden_layers=0)
        model = _hp.build_model(num_classes=1, pretrain='imagenet').to(self.device)
        self.model_name = model_name
        self.ftrs = Features.from_model(model, tile_px=tile_px, **kwargs)
        self.tag = model_name + "_" + '-'.join(self.ftrs.layers)
        self.num_features = self.ftrs.num_features
        self.num_classes = 0

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

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "<TorchImagenetLayerExtractor model={} layers={} n_features={}>".format(
            self.model_name,
            self.ftrs.layers,
            self.num_features,
        )

    def __call__(self, batch_images):
        import torch
        assert batch_images.dtype == torch.uint8
        batch_images = self.transform(batch_images).to(self.device)
        return self.ftrs._predict(batch_images)
