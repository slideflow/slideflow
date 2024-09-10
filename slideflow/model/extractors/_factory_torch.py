"""Factory for building PyTorch feature extractors."""

import inspect
import slideflow as sf
from typing import Tuple, Generator, Optional, TYPE_CHECKING
from slideflow import errors

from ._slide import features_from_slide
from ._registry import _torch_extractors, is_torch_extractor, register_torch
from ..base import BaseFeatureExtractor

if TYPE_CHECKING:
    import torch


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
def virchow(weights, **kwargs):
    from .virchow import VirchowFeatures
    return VirchowFeatures(weights, **kwargs)

@register_torch
def vit(**kwargs):
    from .vit import ViTFeatures
    return ViTFeatures(**kwargs)

@register_torch
def dinov2(**kwargs):
    from .dinov2 import DinoV2Features
    return DinoV2Features(**kwargs)

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

    def __init__(
        self, 
        channels_last: bool = False, 
        mixed_precision: bool = True,
        **transform_kwargs
    ) -> None:
        from .. import torch_utils

        super().__init__(backend='torch')
        self.device = torch_utils.get_device()
        self.channels_last = channels_last
        self.mixed_precision = mixed_precision
        self.inference_mode = True
        self.force_uint8 = True
        self.transform_kwargs = self._verify_transform_args(transform_kwargs)

    def _process_output(self, output):
        """Process model output."""
        import torch
        return output.to(torch.float32)

    def __call__(self, obj, **kwargs):
        """Generate features for a batch of images or a WSI."""
        import torch
        from slideflow.model.torch import autocast

        if isinstance(obj, sf.WSI):
            # Returns masked array of features
            return features_from_slide(self, obj, **kwargs)
        elif isinstance(obj, str) and obj.endswith('.tfrecords'):
            tfr_generator = self.tfrecord_inference(obj, **kwargs)
            # Concatenate features from all batches
            features = []
            locations = []
            for batch_features, batch_locations in tfr_generator:
                features.append(batch_features)
                locations.append(batch_locations)
            return torch.cat(features), torch.cat(locations)
        elif kwargs:
            raise ValueError(
                f"{self.__class__.__name__} does not accept keyword arguments "
                "when extracting features from a batch of images."
            )
        elif not isinstance(obj, torch.Tensor):
            raise ValueError(
                "Expected first argument to be either a WSI object, a path "
                " to a tfrecord, or a Tensor. Got: {}".format(type(obj))
            )
        if not (obj.dtype == torch.uint8) and self.force_uint8:
            raise RuntimeError("Expected input to be a uint8 tensor, got: {}".format(
                obj.dtype
            ))
        obj = obj.to(self.device)
        obj = self.transform(obj)
        with autocast(self.device.type, mixed_precision=self.mixed_precision):
            with torch.inference_mode(self.inference_mode):
                if self.channels_last:
                    obj = obj.to(memory_format=torch.channels_last)
                return self._process_output(self.model(obj))

    def _verify_transform_args(self, kwargs):
        sig = inspect.signature(self.get_transforms)
        valid_kwargs = [
            p.name for p in sig.parameters.values() 
            if (p.kind == p.KEYWORD_ONLY
                and p.name != 'img_size')
        ]
        for k in kwargs:
            if k not in valid_kwargs:
                raise ValueError(f"Unrecognized argument: {k}")
        return kwargs

    @staticmethod
    def _get_interpolation(mode: str):
        from torchvision.transforms import InterpolationMode
        if mode == 'bilinear':
            return InterpolationMode.BILINEAR
        if mode == 'bicubic':
            return InterpolationMode.BICUBIC
        if mode == 'nearest':
            return InterpolationMode.NEAREST
        if mode == 'nearest_exact':
            return InterpolationMode.NEAREST_EXACT
        raise ValueError("Unrecognized interpolation mode: {}".format(mode))

    def get_transforms(
        self,
        *,
        img_size: Optional[int] = None,
        center_crop: Optional[int] = None,
        resize: Optional[int] = None,
        interpolation: str = 'bilinear',
        antialias: bool = False,
        norm_mean: Optional[Tuple[float, float, float]] = (0.485, 0.456, 0.406),
        norm_std: Optional[Tuple[float, float, float]] = (0.229, 0.224, 0.225),
    ):
        """Get a list of preprocessing image transforms."""
        from torchvision import transforms

        all_transforms = []
        if center_crop:
            all_transforms += [
                transforms.CenterCrop(
                    img_size if center_crop is True else center_crop
                )
            ]
        if resize:
            all_transforms += [
                transforms.Resize(
                    img_size if resize is True else resize,
                    interpolation=self._get_interpolation(interpolation),
                    antialias=antialias
                )
            ]
        all_transforms += [
            transforms.Lambda(lambda x: x / 255.),
            transforms.Normalize(
                mean=norm_mean,
                std=norm_std
            )
        ]
        return all_transforms

    def build_transform(self, **kwargs):
        """Compose the preprocessing transforms, updated with user keyword arguments"""
        # Use the user-specified transforms, if provided.
        from torchvision import transforms
        kwargs.update(self.transform_kwargs)
        return transforms.Compose(self.get_transforms(**kwargs))
    

    def tfrecord_inference(
        self,
        tfrecord_path: str,
        batch_size: int = 32,
        num_workers: int = 2
    ) -> Generator[Tuple["torch.Tensor", "torch.Tensor"], None, None]:
        """Generate features from a TFRecord file."""
        import torch
        from torch.utils.data import DataLoader

        tfr = sf.TFRecord(tfrecord_path, decode_images=True)
        tfr_dl = DataLoader(
            tfr,
            batch_size=batch_size,
            num_workers=num_workers
        )
        for batch in tfr_dl:
            features = self(sf.io.torch.whc_to_cwh(batch['image_raw']))
            locations = torch.stack([batch['loc_x'], batch['loc_y']], dim=1)
            yield features, locations

    def _dump_config(self, class_name, **kwargs):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.

        """
        return {
            'class': class_name,
            'kwargs': {
                **self.transform_kwargs,
                **kwargs
            }
        }


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
            # Returns masked array of features
            return features_from_slide(self, obj, **kwargs)
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
        feature extractor, using ``slideflow.build_feature_extractor()``.

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