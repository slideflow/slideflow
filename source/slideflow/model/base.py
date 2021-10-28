import json
import os
import csv
import numpy as np
from slideflow.util import log

"""Base classes to be extended by framework-specific implementations."""

class ModelParams:
    """Build a set of hyperparameters."""

    OptDict = {}
    ModelDict = {}
    _LinearLoss = []
    _AllLoss = []

    def __init__(self, tile_px=299, tile_um=302, epochs=10, toplayer_epochs=0,
                 model='Xception', pooling='max', loss='sparse_categorical_crossentropy',
                 learning_rate=0.0001, learning_rate_decay=0, learning_rate_decay_steps=100000,
                 batch_size=16, hidden_layers=1, hidden_layer_width=500, optimizer='Adam',
                 early_stop=False, early_stop_patience=0, early_stop_method='loss',
                 training_balance='category', validation_balance='none',
                 trainable_layers=0, L2_weight=0, dropout=0, augment='xyrj', drop_images=False):

        """Collection of hyperparameters used for model building and training

        Args:
            tile_px (int, optional): Tile width in pixels. Defaults to 299.
            tile_um (int, optional): Tile width in microns. Defaults to 302.
            epochs (int, optional): Number of epochs to train the full model. Defaults to 10.
            toplayer_epochs (int, optional): Number of epochs to only train the fully-connected layers. Defaults to 0.
            model (str, optional): Base model architecture name. Defaults to 'Xception'.
            pooling (str, optional): Post-convolution pooling. 'max', 'avg', or 'none'. Defaults to 'max'.
            loss (str, optional): Loss function. Defaults to 'sparse_categorical_crossentropy'.
            learning_rate (float, optional): Learning rate. Defaults to 0.0001.
            learning_rate_decay (int, optional): Learning rate decay rate. Defaults to 0.
            learning_rate_decay_steps (int, optional): Learning rate decay steps. Defaults to 100000.
            batch_size (int, optional): Batch size. Defaults to 16.
            hidden_layers (int, optional): Number of post-convolutional fully-connected hidden layers. Defaults to 1.
            hidden_layer_width (int, optional): Width of fully-connected hidden layers. Defaults to 500.
            optimizer (str, optional): Name of optimizer. Defaults to 'Adam'.
            early_stop (bool, optional): Use early stopping. Defaults to False.
            early_stop_patience (int, optional): Patience for early stopping, in epochs. Defaults to 0.
            early_stop_method (str, optional): Metric to monitor for early stopping. Defaults to 'loss'.
            training_balance ([type], optional): Type of batch-level balancing to use during training.
                Defaults to 'category'.
            validation_balance ([type], optional): Type of batch-level balancing to use during validation.
                Defaults to 'none'.
            trainable_layers (int, optional): Number of layers which are traininable. If 0, trains all layers. Defaults to 0.
            L2_weight (int, optional): L2 regularization weight. Defaults to 0.
            dropout (int, optional): Post-convolution dropout rate. Defaults to 0.
            augment (str): Image augmentations to perform. String containing characters designating augmentations.
                'x' indicates random x-flipping, 'y' y-flipping, 'r' rotating, and 'j' JPEG compression/decompression
                at random quality levels. Passing either 'xyrj' or True will use all augmentations.
            drop_images (bool, optional): Drop images, using only other slide-level features as input. Defaults to False.
        """

        # Additional hyperparameters to consider:
        # beta1 0.9
        # beta2 0.999
        # epsilon 1.0
        # batch_norm_decay 0.99

        # Assert provided hyperparameters are valid
        assert isinstance(tile_px, int)
        assert isinstance(tile_um, int)
        assert isinstance(toplayer_epochs, int)
        assert isinstance(epochs, (int, list))
        if isinstance(epochs, list):
            assert all([isinstance(t, int) for t in epochs])
        assert pooling in ['max', 'avg', 'none']
        assert isinstance(learning_rate, float)
        assert isinstance(learning_rate_decay, (int, float))
        assert isinstance(learning_rate_decay_steps, (int))
        assert isinstance(batch_size, int)
        assert isinstance(hidden_layers, int)

        assert isinstance(early_stop, bool)
        assert isinstance(early_stop_patience, int)
        assert early_stop_method in ['loss', 'accuracy']
        assert training_balance in ['tile', 'category', 'patient', 'slide', 'none', None]
        assert validation_balance in ['tile', 'category', 'patient', 'slide', 'none', None]
        assert isinstance(hidden_layer_width, int)
        assert isinstance(trainable_layers, int)
        assert isinstance(L2_weight, (int, float))
        assert isinstance(dropout, (int, float))
        assert isinstance(augment, (bool, str))
        assert isinstance(drop_images, bool)

        assert 0 <= learning_rate_decay <= 1
        assert 0 <= L2_weight <= 1
        assert 0 <= dropout <= 1

        self.tile_px = tile_px
        self.tile_um = tile_um
        self.toplayer_epochs = toplayer_epochs
        self.epochs = epochs if isinstance(epochs, list) else [epochs]
        self.model = model
        self.pooling = pooling if pooling != 'none' else None
        self.loss = loss
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.early_stop = early_stop
        self.early_stop_method = early_stop_method
        self.early_stop_patience = early_stop_patience
        self.hidden_layers = hidden_layers
        self.training_balance = training_balance
        self.validation_balance = validation_balance
        self.augment = augment
        self.hidden_layer_width = hidden_layer_width
        self.trainable_layers = trainable_layers
        self.L2_weight = float(L2_weight)
        self.dropout = dropout
        self.drop_images = drop_images

        # Perform check to ensure combination of HPs are valid
        self.validate()

    def _get_args(self):
        return [arg for arg in dir(self) if not arg[0]=='_' and arg not in ['get_opt',
                                                                            'build_model',
                                                                            'model_type',
                                                                            'validate',
                                                                            'get_dict',
                                                                            'load_dict',
                                                                            'OptDict',
                                                                            'ModelDict',
                                                                            'LinearLossDict',
                                                                            'AllLossDict']]

    def get_dict(self):
        d = {}
        for arg in self._get_args():
            d.update({arg: getattr(self, arg)})
        return d

    def load_dict(self, hp_dict):
        for key, value in hp_dict.items():
            try:
                setattr(self, key, value)
            except:
                log.error(f'Unrecognized hyperparameter {key}; unable to load')

    def __str__(self):
        args = sorted(self._get_args(), key=lambda arg: arg.lower())
        arg_dict = {arg: getattr(self, arg) for arg in args}
        return json.dumps(arg_dict, indent=2)

    def _detect_classes_from_labels(self, labels):
        outcome_labels = np.array(list(labels.values()))
        if len(outcome_labels.shape) == 1:
            outcome_labels = np.expand_dims(outcome_labels, axis=1)

        if self.model_type() == 'categorical':
            return {i: np.unique(outcome_labels[:,i]).shape[0] for i in range(outcome_labels.shape[1])}
        else:
            try:
                return outcome_labels.shape[1]
            except TypeError:
                raise HyperParameterError('Incorrect formatting of outcome labels for linear model; must be an ndarray.')

    def validate(self):
        """Check that hyperparameter combinations are valid."""
        if (self.model_type() != 'categorical' and ((self.training_balance == 'category') or
                                                    (self.validation_balance == 'category'))):
            raise HyperParameterError(f'Cannot combine category-level balancing with model type "{self.model_type()}".')
        return True

    def model_type(self):
        """Returns either 'linear', 'categorical', or 'cph' depending on the loss type."""
        if self.loss == 'negative_log_likelihood':
            return 'cph'
        elif self.loss in self._LinearLoss:
            return 'linear'
        else:
            return 'categorical'

class HyperParameterError(Exception):
    pass

class ModelError(Exception):
    def __init__(self, message, errors=None):
        log.error(message)
        super().__init__(message)

class no_scope():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False

def log_summary(model, neptune_run=None):
    # Print to terminal
    if log.getEffectiveLevel() <= 20:
        print()
        model.summary()

    # Log to neptune
    if neptune_run:
        summary_string = []
        model.summary(print_fn=lambda x: summary_string.append(x))
        neptune_run['model_info/summary'] = "\n".join(summary_string)

def log_manifest(train_tfrecords=None, val_tfrecords=None, labels=None, save_loc=None):
    out = ''
    if save_loc:
        save_file = open(os.path.join(save_loc), 'w')
        writer = csv.writer(save_file)
        writer.writerow(['slide', 'dataset', 'outcome_label'])
    if train_tfrecords or val_tfrecords:
        if train_tfrecords:
            for tfrecord in train_tfrecords:
                slide = tfrecord.split('/')[-1][:-10]
                outcome_label = labels[slide] if labels else 'NA'
                out += ' '.join([slide, 'training', str(outcome_label)])
                if save_loc:
                    writer.writerow([slide, 'training', outcome_label])
        if val_tfrecords:
            for tfrecord in val_tfrecords:
                slide = tfrecord.split('/')[-1][:-10]
                outcome_label = labels[slide] if labels else 'NA'
                out += ' '.join([slide, 'validation', str(outcome_label)])
                if save_loc:
                    writer.writerow([slide, 'training', outcome_label])
    if save_loc:
        save_file.close()
    return out

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