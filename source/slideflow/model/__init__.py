import os
import csv
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
        normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
        normalizer_source (str, optional): Path to normalizer source image. Defaults to None.
            If None but using a normalizer, will use an internal tile for normalization.
            Internal default tile can be found at slideflow.util.norm_tile.jpg
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

def get_hp_from_batch_file(batch_train_file, models=None):
    """Organizes a list of hyperparameters ojects and associated models names.

    Args:
        batch_train_file (str): Path to train train TSV file.
        models (list(str)): List of model names. Defaults to None.
            If not supplied, returns all valid models from batch file.

    Returns:
        List of (Hyperparameter, model_name) for each HP combination
    """

    if models is not None and not isinstance(models, list):
        raise sf.util.UserError("If supplying models, must be a list of strings containing model names.")
    if isinstance(models, list) and not list(set(models)) == models:
        raise sf.util.UserError("Duplicate model names provided.")

    # First, ensure all indicated models are in the batch train file
    if models:
        valid_models = []
        with open(batch_train_file) as csv_file:
            reader = csv.reader(csv_file, delimiter='\t')
            header = next(reader)
            try:
                model_name_i = header.index('model_name')
            except:
                err_msg = "Unable to find column 'model_name' in the batch training config file."
                log.error(err_msg)
                raise ValueError(err_msg)
            for row in reader:
                model_name = row[model_name_i]
                # First check if this row is a valid model
                if (not models) or (isinstance(models, str) and model_name==models) or model_name in models:
                    # Now verify there are no duplicate model names
                    if model_name in valid_models:
                        err_msg = f'Duplicate model names found in {sf.util.green(batch_train_file)}.'
                        log.error(err_msg)
                        raise ValueError(err_msg)
                    valid_models += [model_name]
        missing_models = [m for m in models if m not in valid_models]
        if missing_models:
            raise ValueError(f"Unable to find the following models in the batch train file: {', '.join(missing_models)}")

    # Read the batch train file and generate HyperParameter objects from the given configurations
    hyperparameters = {}
    batch_train_rows = []
    with open(batch_train_file) as csv_file:
        reader = csv.reader(csv_file, delimiter='\t')
        header = next(reader)
        for row in reader:
            batch_train_rows += [row]

    for row in batch_train_rows:
        try:
            hp, hp_model_name = get_hp_from_row(row, header)
        except HyperParameterError as e:
            log.error('Invalid Hyperparameter combination: ' + str(e))
            return
        if models and hp_model_name not in models: continue
        hyperparameters[hp_model_name] = hp
    return hyperparameters

def get_hp_from_row(row, header):
    """Converts a row in the batch_train CSV file into a ModelParams object."""

    model_name_i = header.index('model_name')
    args = header[0:model_name_i] + header[model_name_i+1:]
    model_name = row[model_name_i]
    hp = ModelParams()
    for arg in args:
        value = row[header.index(arg)]
        if arg in hp._get_args():
            if arg != 'epochs':
                arg_type = type(getattr(hp, arg))
                if arg_type == bool:
                    if value.lower() in ['true', 'yes', 'y', 't']:
                        bool_val = True
                    elif value.lower() in ['false', 'no', 'n', 'f']:
                        bool_val = False
                    else:
                        raise ValueError(f'Unable to parse "{value}" for batch file argument "{arg}" into a bool.')
                    setattr(hp, arg, bool_val)
                elif arg in ('L2_weight', 'dropout'):
                    setattr(hp, arg, float(value))
                else:
                    setattr(hp, arg, arg_type(value))
            else:
                epochs = [int(i) for i in value.translate(str.maketrans({'[':'', ']':''})).split(',')]
                setattr(hp, arg, epochs)
        else:
            log.error(f"Unknown argument '{arg}' found in training config file.", 0)
    return hp, model_name