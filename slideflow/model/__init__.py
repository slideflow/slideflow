'''Submodule that includes tools for intermediate layer activations.

Supports both PyTorch and Tensorflow backends, importing either model.tensorflow
or model.pytorch based on the environmental variable SF_BACKEND.
'''

import warnings
from typing import Any, Dict, List

import slideflow as sf
from slideflow import errors
from .base import BaseFeatureExtractor
from .features import DatasetFeatures
from .extractors import (
    list_extractors, list_torch_extractors, list_tensorflow_extractors,
    is_extractor, is_torch_extractor, is_tensorflow_extractor,
    build_feature_extractor, build_torch_feature_extractor,
    build_tensorflow_feature_extractor
)

# --- Backend-specific imports ------------------------------------------------

if sf.backend() == 'tensorflow':
    from slideflow.model.tensorflow import (CPHTrainer, Features, load, # noqa F401
                                            LinearTrainer, ModelParams,
                                            Trainer, UncertaintyInterface)
elif sf.backend() == 'torch':
    from slideflow.model.torch import (CPHTrainer, Features, load, # noqa F401
                                       LinearTrainer, ModelParams,
                                       Trainer, UncertaintyInterface)
else:
    raise errors.UnrecognizedBackendError

# -----------------------------------------------------------------------------


def is_tensorflow_tensor(arg: Any) -> bool:
    """Checks if the given object is a Tensorflow Tensor."""
    if sf.util.tf_available:
        import tensorflow as tf
        return isinstance(arg, tf.Tensor)
    else:
        return False


def is_torch_tensor(arg: Any) -> bool:
    """Checks if the given object is a Tensorflow Tensor."""
    if sf.util.torch_available:
        import torch
        return isinstance(arg, torch.Tensor)
    else:
        return False


def is_tensorflow_model(arg: Any) -> bool:
    """Checks if the object is a Tensorflow Model or path to Tensorflow model."""
    if isinstance(arg, str):
        return sf.util.is_tensorflow_model_path(arg)
    elif sf.util.tf_available:
        import tensorflow as tf
        return isinstance(arg, tf.keras.models.Model)
    else:
        return False


def is_torch_model(arg: Any) -> bool:
    """Checks if the object is a PyTorch Module or path to PyTorch model."""
    if isinstance(arg, str):
        return sf.util.is_torch_model_path(arg)
    elif sf.util.torch_available:
        import torch
        return isinstance(arg, torch.nn.Module)
    else:
        return False


def trainer_from_hp(*args, **kwargs):
    warnings.warn(
        "sf.model.trainer_from_hp() is deprecated. Please use "
        "sf.model.build_trainer().",
        DeprecationWarning
    )
    return build_trainer(*args, **kwargs)


def build_trainer(
    hp: "ModelParams",
    outdir: str,
    labels: Dict[str, Any],
    **kwargs
) -> Trainer:
    """From the given :class:`slideflow.ModelParams` object, returns
    the appropriate instance of :class:`slideflow.model.Trainer`.

    Args:
        hp (:class:`slideflow.ModelParams`): ModelParams object.
        outdir (str): Path for event logs and checkpoints.
        labels (dict): Dict mapping slide names to outcome labels (int or
            float format).

    Keyword Args:
        slide_input (dict): Dict mapping slide names to additional
            slide-level input, concatenated after post-conv.
        name (str, optional): Optional name describing the model, used for
            model saving. Defaults to 'Trainer'.
        feature_sizes (list, optional): List of sizes of input features.
            Required if providing additional input features as input to
            the model.
        feature_names (list, optional): List of names for input features.
            Used when permuting feature importance.
        outcome_names (list, optional): Name of each outcome. Defaults to
            "Outcome {X}" for each outcome.
        mixed_precision (bool, optional): Use FP16 mixed precision (rather
            than FP32). Defaults to True.
        allow_tf32 (bool): Allow internal use of Tensorfloat-32 format.
                Defaults to False.
        config (dict, optional): Training configuration dictionary, used
            for logging. Defaults to None.
        use_neptune (bool, optional): Use Neptune API logging.
            Defaults to False
        neptune_api (str, optional): Neptune API token, used for logging.
            Defaults to None.
        neptune_workspace (str, optional): Neptune workspace.
            Defaults to None.
        load_method (str): Either 'full' or 'weights'. Method to use
                when loading a Tensorflow model. If 'full', loads the model with
                ``tf.keras.models.load_model()``. If 'weights', will read the
                ``params.json`` configuration file, build the model architecture,
                and then load weights from the given model with
                ``Model.load_weights()``. Loading with 'full' may improve
                compatibility across Slideflow versions. Loading with 'weights'
                may improve compatibility across hardware & environments.
        custom_objects (dict, Optional): Dictionary mapping names
                (strings) to custom classes or functions. Defaults to None.
    """
    if hp.model_type() == 'categorical':
        return Trainer(hp, outdir, labels, **kwargs)
    if hp.model_type() == 'linear':
        return LinearTrainer(hp, outdir, labels, **kwargs)
    if hp.model_type() == 'cph':
        return CPHTrainer(hp, outdir, labels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {hp.model_type()}")


def read_hp_sweep(
    filename: str,
    models: List[str] = None
) -> Dict[str, "ModelParams"]:
    """Organizes a list of hyperparameters ojects and associated models names.

    Args:
        filename (str): Path to hyperparameter sweep JSON file.
        models (list(str)): List of model names. Defaults to None.
            If not supplied, returns all valid models from batch file.

    Returns:
        List of (Hyperparameter, model_name) for each HP combination
    """
    if models is not None and not isinstance(models, list):
        raise ValueError("If supplying models, must be list(str) "
                         "with model names.")
    if isinstance(models, list) and not list(set(models)) == models:
        raise ValueError("Duplicate model names provided.")

    hp_list = sf.util.load_json(filename)

    # First, ensure all indicated models are in the batch train file
    if models:
        valid_models = []
        for hp_dict in hp_list:
            model_name = list(hp_dict.keys())[0]
            if ((not models)
               or (isinstance(models, str) and model_name == models)
               or model_name in models):
                valid_models += [model_name]
        missing = [m for m in models if m not in valid_models]
        if missing:
            raise ValueError(f"Unable to find models {', '.join(missing)}")
    else:
        valid_models = [list(hp_dict.keys())[0] for hp_dict in hp_list]

    # Read the batch train file and generate HyperParameter objects
    # from the given configurations
    loaded = {}
    for hp_dict in hp_list:
        name = list(hp_dict.keys())[0]
        if name in valid_models:
            loaded.update({
                name: ModelParams.from_dict(hp_dict[name])
            })
    return loaded  # type: ignore
