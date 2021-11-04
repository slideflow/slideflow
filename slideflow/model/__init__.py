import os
import slideflow as sf
from slideflow.util import log
from slideflow.model.base import HyperParameterError

# --- Backend-specific imports ------------------------------------------------

if os.environ['SF_BACKEND'] == 'tensorflow':
    from slideflow.model.tensorflow import ModelParams, Trainer, LinearTrainer, CPHTrainer
elif os.environ['SF_BACKEND'] == 'torch':
    from slideflow.model.torch import ModelParams, Trainer, LinearTrainer, CPHTrainer
else:
    raise ValueError(f"Unknown backend {os.environ['SF_BACKEND']}")

# -----------------------------------------------------------------------------

def trainer_from_hp(hp, **kwargs):
    """From the given :class:`slideflow.model.ModelParams` object, returns the appropriate instance of
    :class:`slideflow.model.Model`.

    Args:
        hp (:class:`slideflow.model.ModelParams`): ModelParams object.

    Keyword Args:
        outdir (str): Location where event logs and checkpoints will be written.
        annotations (dict): Nested dict, mapping slide names to a dict with patient name (key 'submitter_id'),
            outcome labels (key 'outcome_label'), and any additional slide-level inputs (key 'input').
        name (str, optional): Optional name describing the model, used for model saving. Defaults to None.
        manifest (dict, optional): Manifest dictionary mapping TFRecords to number of tiles. Defaults to None.
        model_type (str, optional): Type of model outcome, 'categorical' or 'linear'. Defaults to 'categorical'.
        feature_sizes (list, optional): List of sizes of input features. Required if providing additional
            input features as input to the model.
        feature_names (list, optional): List of names for input features. Used when permuting feature importance.
        outcome_names (list, optional): Name of each outcome. Defaults to "Outcome {X}" for each outcome.
        mixed_precision (bool, optional): Use FP16 mixed precision (rather than FP32). Defaults to True.
    """

    if hp.model_type() == 'categorical':
        return Trainer(hp=hp, **kwargs)
    if hp.model_type() == 'linear':
        return LinearTrainer(hp=hp, **kwargs)
    if hp.model_type() == 'cph':
        return CPHTrainer(hp=hp, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {hp.model_type()}")

def get_hp_from_batch_file(filename, models=None):
    """Organizes a list of hyperparameters ojects and associated models names.

    Args:
        filename (str): Path to hyperparameter sweep JSON file.
        models (list(str)): List of model names. Defaults to None.
            If not supplied, returns all valid models from batch file.

    Returns:
        List of (Hyperparameter, model_name) for each HP combination
    """

    if models is not None and not isinstance(models, list):
        raise sf.util.UserError("If supplying models, must be a list of strings containing model names.")
    if isinstance(models, list) and not list(set(models)) == models:
        raise sf.util.UserError("Duplicate model names provided.")

    hp_list = sf.util.load_json(filename)

    # First, ensure all indicated models are in the batch train file
    if models:
        valid_models = []
        for hp_dict in hp_list:
            model_name = list(hp_dict.keys())[0]
            if (not models) or (isinstance(models, str) and model_name==models) or model_name in models:
                valid_models += [model_name]
        missing_models = [m for m in models if m not in valid_models]
        if missing_models:
            raise ValueError(f"Unable to find the following models in the batch train file: {', '.join(missing_models)}")
    else:
        valid_models = [list(hp_dict.keys())[0] for hp_dict in hp_list]

    # Read the batch train file and generate HyperParameter objects from the given configurations
    hyperparameters = {}

    loaded = {}
    for hp_dict in hp_list:
        name = list(hp_dict.keys())[0]
        if name in valid_models:
            loaded.update({name: sf.model.ModelParams.from_dict(hp_dict[name])})

    return loaded