"""Utility functions for MIL."""

import slideflow as sf
import importlib
import numpy as np

from os.path import exists, join, isdir
from typing import Optional, Tuple, Union, TYPE_CHECKING
from slideflow import errors, log
from ._params import (
    _TrainerConfig, ModelConfigCLAM, TrainerConfigCLAM
)

if TYPE_CHECKING:
    import torch
    from slideflow.model.base import BaseFeatureExtractor
    from slideflow.norm import StainNormalizer

# -----------------------------------------------------------------------------

def load_model_weights(
    weights: str,
    config: Optional[_TrainerConfig] = None,
    *,
    input_shape: Optional[int] = None,
    output_shape: Optional[int] = None,
) -> Tuple["torch.nn.Module", _TrainerConfig]:
    """Load weights and build model.

    Args:
        weights (str): Path to model weights.
        config (:class:`slideflow.mil.TrainerConfigFastAI` or :class:`slideflow.mil.TrainerConfigCLAM`):
            Configuration for building model. If ``weights`` is a path to a
            model directory, will attempt to read ``mil_params.json`` from this
            location and load saved configuration. Defaults to None.

    Keyword Args:
        input_shape (int): Number of features in the input data.
        output_shape (int): Number of output classes.

    Returns:
        :class:`torch.nn.Module`: Loaded model.
    """
    import torch

    if isinstance(config, TrainerConfigCLAM):
        raise NotImplementedError

    if exists(join(weights, 'mil_params.json')):
        mil_params = sf.util.load_json(join(weights, 'mil_params.json'))
    else:
        mil_params = None

    # Read configuration from saved model, if available
    if config is None:
        if mil_params is None:
            raise errors.ModelError(
                f"Could not find `mil_params.json` at {weights}. Check the "
                "provided model/weights path, or provide a configuration "
                "with 'config'."
            )
        else:
            config = sf.mil.mil_config(trainer=mil_params['trainer'],
                                       **mil_params['params'])

    # Determine the input and output shapes, reading from the model
    # configuration, if necessary.
    if input_shape is None or output_shape is None:
        if mil_params is None:
            raise errors.ModelError(
                f"Could not find `mil_params.json` at {weights}. Check the "
                "provided model/weights path, or provide the input and output "
                "shape via input_shape and output_shape."
            )
        else:
            if input_shape is None and 'input_shape' in mil_params:
                input_shape = mil_params['input_shape']
            elif input_shape is None:
                raise errors.ModelError(
                    'Could not find input_shape in `mil_params.json`.'
                )
            if output_shape is None and 'output_shape' in mil_params:
                output_shape = mil_params['output_shape']
            elif output_shape is None:
                raise errors.ModelError(
                    'Could not find output_shape in `mil_params.json`.'
                )

    # Build the model
    if isinstance(config, TrainerConfigCLAM):
        config_size = config.model_fn.sizes[config.model_config.model_size]
        _size = [input_shape] + config_size[1:]
        model = config.model_fn(size=_size)
        log.info(f"Building model {config.model_fn.__name__} (size={_size})")
    elif isinstance(config.model_config, ModelConfigCLAM):
        config_size = config.model_fn.sizes[config.model_config.model_size]
        _size = [input_shape] + config_size[1:]
        model = config.model_fn(size=_size)
        log.info(f"Building model {config.model_fn.__name__} (size={_size})")
    else:
        model = config.model_fn(input_shape, output_shape)
        log.info(f"Building model {config.model_fn.__name__} "
                 f"(in={input_shape}, out={output_shape})")
    if isdir(weights):
        if exists(join(weights, 'models', 'best_valid.pth')):
            weights = join(weights, 'models', 'best_valid.pth')
        elif exists(join(weights, 'results', 's_0_checkpoint.pt')):
            weights = join(weights, 'results', 's_0_checkpoint.pt')
        else:
            raise errors.ModelError(
                f"Could not find model weights at path {weights}"
            )
    log.info(f"Loading model weights from [green]{weights}[/]")
    model.load_state_dict(torch.load(weights))

    # Prepare device.
    if hasattr(model, 'relocate'):
        model.relocate()  # type: ignore
    model.eval()
    return model, config


def _load_bag(bag: Union[str, np.ndarray, "torch.Tensor"]) -> "torch.Tensor":
    """Load bag from file or convert to torch.Tensor."""
    import torch

    if isinstance(bag, str):
        return torch.load(bag)
    elif isinstance(bag, np.ndarray):
        return torch.from_numpy(bag)
    elif isinstance(bag, torch.Tensor):
        return bag
    else:
        raise ValueError(
            "Unrecognized bag type '{}'".format(type(bag))
        )