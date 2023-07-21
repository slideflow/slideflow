"""Utility functions for MIL."""

import slideflow as sf
import importlib

from os.path import exists, join
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from slideflow.model.base import BaseFeatureExtractor
    from slideflow.norm import StainNormalizer

# -----------------------------------------------------------------------------

def build_bag_encoder(
    bags_or_model: str,
    allow_errors: bool = False
) -> Tuple[Optional["BaseFeatureExtractor"], Optional["StainNormalizer"]]:
    """Recreate the encoder used to generate features stored in bags.

    Args:
        bags_or_model (str): Either a path to directory containing feature bags,
            or a path to a trained MIL model. If a path to a trained MIL model,
            the encoder used to generate features will be recreated.
        allow_errors (bool): If True, return None if the encoder
            cannot be rebuilt. If False, raise an error. Defaults to False.

    Returns:
        Optional[BaseFeatureExtractor]: Encoder function, or None if ``allow_errors`` is
            True and the encoder cannot be rebuilt.

        Optional[StainNormalizer]: Stain normalizer used when generating
            feature bags, or None if no stain normalization was used.

    """
    # Load bags configuration
    is_bag_dir = exists(join(bags_or_model, 'bags_config.json'))
    is_model_dir = exists(join(bags_or_model, 'mil_params.json'))
    if not is_bag_dir and not is_model_dir:
        if allow_errors:
            return None, None
        else:
            raise ValueError(
                'Could not find bags or MIL model configuration at '
                f'{bags_or_model}.'
            )
    if is_model_dir:
        mil_config = sf.util.load_json(join(bags_or_model, 'mil_params.json'))
        if 'bags_encoder' not in mil_config:
            if allow_errors:
                return None, None
            else:
                raise ValueError(
                    'Could not rebuild extractor from configuration at '
                    f'{bags_or_model}.'
                )
        bags_config = mil_config['bags_encoder']
    else:
        bags_config = sf.util.load_json(join(bags_or_model, 'bags_config.json'))
    if ('extractor' not in bags_config
       or any(n not in bags_config['extractor'] for n in ['class', 'kwargs'])):
        if allow_errors:
            return None, None
        else:
            raise ValueError(
                'Could not rebuild extractor from configuration at '
                f'{bags_or_model}.'
            )

    # Rebuild encoder
    encoder_name = bags_config['extractor']['class'].split('.')
    encoder_class = encoder_name[-1]
    encoder_kwargs = bags_config['extractor']['kwargs']
    module = importlib.import_module('.'.join(encoder_name[:-1]))
    try:
        encoder = getattr(module, encoder_class)(**encoder_kwargs)
    except Exception:
        if allow_errors:
            return None
        else:
            raise ValueError(
                f'Could not rebuild extractor from configuration at {bags}.'
            )

    # Rebuild stain normalizer
    if bags_config['normalizer'] is not None:
        normalizer = sf.norm.autoselect(
            bags_config['normalizer']['method'],
            backend=encoder.backend
        )
        normalizer.set_fit(**bags_config['normalizer']['fit'])
    else:
        normalizer = None
    if (hasattr(encoder, 'normalizer')
       and encoder.normalizer is not None
       and normalizer is not None):
        sf.log.warning(
            'Encoder already has a stain normalizer. Overwriting with '
            'normalizer from bags configuration.'
        )
        encoder.normalizer = normalizer
    elif hasattr(encoder, 'normalizer') and encoder.normalizer is not None:
        normalizer = encoder.normalizer

    return encoder, normalizer
