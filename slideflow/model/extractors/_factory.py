"""Factory for building feature extractors."""

import slideflow as sf
from slideflow import errors

from ._registry import (is_tensorflow_extractor, is_torch_extractor,
                        _tf_extractors, _torch_extractors)
from ._factory_tensorflow import build_tensorflow_feature_extractor
from ._factory_torch import build_torch_feature_extractor


def build_feature_extractor(name, **kwargs):
    """Build a feature extractor.

    The returned feature extractor is a callable object, which returns
    features (often layer activations) for a batch of images.
    Images are expected to be in (B, W, H, C) format and non-standardized
    (scaled 0-255) with dtype uint8. The feature extractors perform
    all needed preprocessing on the fly.

    Args:
        name (str): Name of the feature extractor to build. Available
            feature extractors are listed with
            :func:`slideflow.model.list_extractors()`.

    Keyword arguments:
        tile_px (int): Tile size (input image size), in pixels.
        **kwargs (Any): All remaining keyword arguments are passed
            to the feature extractor factory function, and may be different
            for each extractor.

    Returns:
        A callable object which accepts a batch of images (B, W, H, C) of dtype
        uint8 and returns a batch of features (dtype float32).

    Examples
        Create an extractor that calculates post-convolutional layer activations
        from an imagenet-pretrained Resnet50 model.

            .. code-block:: python

                from slideflow.model import build_feature_extractor

                extractor = build_feature_extractor(
                    'resnet50_imagenet'
                )

        Create an extractor that calculates 'conv4_block4_2_relu' activations
        from an imagenet-pretrained Resnet50 model.

            .. code-block:: python

                from slideflow.model import build_feature_extractor

                extractor = build_feature_extractor(
                    'resnet50_imagenet',
                    layers='conv4_block4_2_relu
                )

        Create a pretrained "CTransPath" extractor.

            .. code-block:: python

                from slideflow.model import build_feature_extractor

                extractor = build_feature_extractor('ctranspath')

        Use an extractor to calculate layer activations for an entire dataset.

            .. code-block:: python

                import slideflow as sf

                # Load a project and dataset
                P = sf.load_project(...)
                dataset = P.dataset(...)

                # Create a feature extractor
                resnet = sf.model.build_feature_extractor(
                    'resnet50_imagenet'
                )

                # Calculate features for the entire dataset
                features = sf.DatasetFeatures(
                    resnet,
                    dataset=dataset
                )

    """
    if is_tensorflow_extractor(name) and is_torch_extractor(name):
        sf.log.info(
            f"Feature extractor {name} available in both Tensorflow and "
            f"PyTorch backends; using active backend {sf.backend()}")
        if sf.backend() == 'tensorflow':
            return build_tensorflow_feature_extractor(name, **kwargs)
        else:
            return build_torch_feature_extractor(name, **kwargs)
    if is_tensorflow_extractor(name):
        return build_tensorflow_feature_extractor(name, **kwargs)
    elif is_torch_extractor(name):
        return build_torch_feature_extractor(name, **kwargs)
    else:
        raise errors.InvalidFeatureExtractor(f"Unrecognized feature extractor: {name}")
