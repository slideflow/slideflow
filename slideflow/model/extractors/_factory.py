"""Factory for building feature extractors."""

import importlib
import slideflow as sf
from os.path import join, exists
from typing import Optional, Tuple, TYPE_CHECKING
from slideflow import errors
from slideflow.model import BaseFeatureExtractor

from ._registry import (is_tensorflow_extractor, is_torch_extractor,
                        _tf_extractors, _torch_extractors)
from ._factory_tensorflow import build_tensorflow_feature_extractor
from ._factory_torch import build_torch_feature_extractor

if TYPE_CHECKING:
    from slideflow.norm import StainNormalizer

# -----------------------------------------------------------------------------

def build_feature_extractor(
    name: str,
    backend: Optional[str] = None,
    **kwargs
) -> BaseFeatureExtractor:
    """Build a feature extractor.

    The returned feature extractor is a callable object, which returns
    features (often layer activations) for either a batch of images or a
    :class:`slideflow.WSI` object.

    If generating features for a batch of images, images are expected to be in
    (B, W, H, C) format and non-standardized (scaled 0-255) with dtype uint8.
    The feature extractors perform all needed preprocessing on the fly.

    If generating features for a slide, the slide is expected to be a
    :class:`slideflow.WSI` object. The feature extractor will generate features
    for each tile in the slide, returning a numpy array of shape (W, H, F),
    where F is the number of features.

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

        Generate a map of features across a slide.

            .. code-block:: python

                import slideflow as sf
                from slideflow.model import build_feature_extractor

                # Load a slide
                wsi = sf.WSI(...)

                # Create a feature extractor
                retccl = build_feature_extractor(
                    'retccl',
                    tile_px=299
                )

                # Create a feature map, a 2D array of shape
                # (W, H, F), where F is the number of features.
                features = retccl(wsi)

    """
    # Build feature extractor according to manually specified backend
    if backend is not None and backend not in ('tensorflow', 'torch'):
        raise ValueError(f"Invalid backend: {backend}")

    # Build a feature extractor from a finetuned model
    if sf.util.is_tensorflow_model_path(name):
        model_config = sf.util.get_model_config(name)
        if model_config['hp']['uq']:
            from slideflow.model.tensorflow import UncertaintyInterface
            return UncertaintyInterface(name, **kwargs)
        else:
            from slideflow.model.tensorflow import Features
            return Features(name, **kwargs)
    elif sf.util.is_torch_model_path(name):
        model_config = sf.util.get_model_config(name)
        if model_config['hp']['uq']:
            from slideflow.model.torch import UncertaintyInterface
            return UncertaintyInterface(name, **kwargs)
        else:
            from slideflow.model.torch import Features  # noqa: F401
            return Features(name, **kwargs)

    # Build feature extractor with a specific backend
    if backend == 'tensorflow':
        if not is_tensorflow_extractor(name):
            raise errors.InvalidFeatureExtractor(
                f"Feature extractor {name} not available in Tensorflow backend")
        return build_tensorflow_feature_extractor(name, **kwargs)
    elif backend == 'torch':
        if not is_torch_extractor(name):
            raise errors.InvalidFeatureExtractor(
                f"Feature extractor {name} not available in PyTorch backend")
        return build_torch_feature_extractor(name, **kwargs)

    # Auto-build feature extractor according to available backends
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


def rebuild_extractor(
    bags_or_model: str,
    allow_errors: bool = False,
    native_normalizer: bool = True
) -> Tuple[Optional["BaseFeatureExtractor"], Optional["StainNormalizer"]]:
    """Recreate the extractor used to generate features stored in bags.

    Args:
        bags_or_model (str): Either a path to directory containing feature bags,
            or a path to a trained MIL model. If a path to a trained MIL model,
            the extractor used to generate features will be recreated.
        allow_errors (bool): If True, return None if the extractor
            cannot be rebuilt. If False, raise an error. Defaults to False.
        native_normalizer (bool, optional): Whether to use PyTorch/Tensorflow-native
            stain normalization, if applicable. If False, will use the OpenCV/Numpy
            implementations. Defaults to True.

    Returns:
        Optional[BaseFeatureExtractor]: Extractor function, or None if ``allow_errors`` is
            True and the extractor cannot be rebuilt.

        Optional[StainNormalizer]: Stain normalizer used when generating
            feature bags, or None if no stain normalization was used.

    """
    # Load bags configuration
    is_bag_config = bags_or_model.endswith('bags_config.json')
    is_bag_dir = exists(join(bags_or_model, 'bags_config.json'))
    is_model_dir = exists(join(bags_or_model, 'mil_params.json'))
    if not (is_bag_dir or is_model_dir or is_bag_config):
        if allow_errors:
            return None, None
        else:
            raise ValueError(
                'Could not find bags or MIL model configuration at '
                f'{bags_or_model}.'
            )
    if is_bag_config:
        bags_config = sf.util.load_json(bags_or_model)
    elif is_model_dir:
        mil_config = sf.util.load_json(join(bags_or_model, 'mil_params.json'))
        if 'bags_extractor' not in mil_config:
            if allow_errors:
                return None, None
            else:
                raise ValueError(
                    'Could not rebuild extractor from configuration at '
                    f'{bags_or_model}; missing "bags_extractor" key in '
                    'mil_params.json.'
                )
        bags_config = mil_config['bags_extractor']
    else:
        bags_config = sf.util.load_json(join(bags_or_model, 'bags_config.json'))
    if ('extractor' not in bags_config
       or any(n not in bags_config['extractor'] for n in ['class', 'kwargs'])):
        if allow_errors:
            return None, None
        else:
            raise ValueError(
                'Could not rebuild extractor from configuration at '
                f'{bags_or_model}; missing "extractor" class or kwargs.'
            )

    # Rebuild extractor
    extractor_name = bags_config['extractor']['class'].split('.')
    extractor_class = extractor_name[-1]
    extractor_kwargs = bags_config['extractor']['kwargs']
    module = importlib.import_module('.'.join(extractor_name[:-1]))
    try:
        extractor = getattr(module, extractor_class)(**extractor_kwargs)
    except Exception:
        if allow_errors:
            return None
        else:
            raise ValueError(
                f'Could not rebuild extractor from configuration at {bags_or_model}.'
            )

    # Rebuild stain normalizer
    if bags_config['normalizer'] is not None:
        normalizer = sf.norm.autoselect(
            bags_config['normalizer']['method'],
            backend=(extractor.backend if native_normalizer else 'opencv')
        )
        normalizer.set_fit(**bags_config['normalizer']['fit'])
    else:
        normalizer = None
    if (hasattr(extractor, 'normalizer')
       and extractor.normalizer is not None
       and normalizer is not None):
        sf.log.warning(
            'Extractor already has a stain normalizer. Overwriting with '
            'normalizer from bags configuration.'
        )
        extractor.normalizer = normalizer
    elif hasattr(extractor, 'normalizer') and extractor.normalizer is not None:
        normalizer = extractor.normalizer

    return extractor, normalizer
