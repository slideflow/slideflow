'''Submodule that includes base classes to be extended by framework-specific implementations.'''

import json
import os
import csv
import numpy as np
from slideflow.util import log
from slideflow.slide import StainNormalizer

class FeatureError(Exception):
    pass

class _ModelParams:
    """Build a set of hyperparameters."""

    def __init__(self, tile_px=299, tile_um=302, epochs=3, toplayer_epochs=0, model='xception', pooling='max',
                 loss='sparse_categorical_crossentropy', learning_rate=0.0001, learning_rate_decay=0,
                 learning_rate_decay_steps=100000, batch_size=16, hidden_layers=0, hidden_layer_width=500,
                 optimizer='Adam', early_stop=False, early_stop_patience=0, early_stop_method='loss',
                 training_balance='category', validation_balance='none', trainable_layers=0, L2_weight=0, dropout=0,
                 augment='xyrj', normalizer=None, normalizer_source=None, include_top=True, drop_images=False):

        """Collection of hyperparameters used for model building and training

        Args:
            tile_px (int, optional): Tile width in pixels. Defaults to 299.
            tile_um (int, optional): Tile width in microns. Defaults to 302.
            epochs (int, optional): Number of epochs to train the full model. Defaults to 3.
            toplayer_epochs (int, optional): Number of epochs to only train the fully-connected layers. Defaults to 0.
            model (str, optional): Base model architecture name. Defaults to 'xception'.
            pooling (str, optional): Post-convolution pooling. 'max', 'avg', or 'none'. Defaults to 'max'.
            loss (str, optional): Loss function. Defaults to 'sparse_categorical_crossentropy'.
            learning_rate (float, optional): Learning rate. Defaults to 0.0001.
            learning_rate_decay (int, optional): Learning rate decay rate. Defaults to 0.
            learning_rate_decay_steps (int, optional): Learning rate decay steps. Defaults to 100000.
            batch_size (int, optional): Batch size. Defaults to 16.
            hidden_layers (int, optional): Number of fully-connected hidden layers after core model. Defaults to 0.
            hidden_layer_width (int, optional): Width of fully-connected hidden layers. Defaults to 500.
            optimizer (str, optional): Name of optimizer. Defaults to 'Adam'.
            early_stop (bool, optional): Use early stopping. Defaults to False.
            early_stop_patience (int, optional): Patience for early stopping, in epochs. Defaults to 0.
            early_stop_method (str, optional): Metric to monitor for early stopping. Defaults to 'loss'.
            training_balance ([type], optional): Type of batch-level balancing to use during training.
                Defaults to 'category'.
            validation_balance ([type], optional): Type of batch-level balancing to use during validation.
                Defaults to 'none'.
            trainable_layers (int, optional): Number of layers which are traininable. If 0, trains all layers.
                Defaults to 0.
            L2_weight (int, optional): L2 regularization weight. Defaults to 0.
            dropout (int, optional): Post-convolution dropout rate. Defaults to 0.
            augment (str): Image augmentations to perform. String containing characters designating augmentations.
                'x' indicates random x-flipping, 'y' y-flipping, 'r' rotating, and 'j' JPEG compression/decompression
                at random quality levels. Passing either 'xyrj' or True will use all augmentations.
            normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image. Defaults to None.
                If None but using a normalizer, will use an internal tile for normalization.
                Internal default tile can be found at slideflow.slide.norm_tile.jpg
            include_top (bool, optional): Include post-convolution fully-connected layers from the core model. Defaults
                to True. include_top=False is not currently compatible with the PyTorch backend.
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
        assert isinstance(include_top, bool)

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
        self.hidden_layer_width = hidden_layer_width
        self.trainable_layers = trainable_layers
        self.L2_weight = float(L2_weight)
        self.dropout = dropout
        self.normalizer = normalizer
        self.normalizer_source = normalizer_source
        self.augment = augment
        self.drop_images = drop_images
        self.include_top = include_top

        # Perform check to ensure combination of HPs are valid
        self.validate()

    def __repr__(self):
        base = "ModelParams("
        for arg in self._get_args():
            base += "\n  {} = {!r},".format(arg, getattr(self, arg))
        base += "\n)"
        return base

    @classmethod
    def from_dict(cls, hp_dict):
        obj = cls()
        obj.load_dict(hp_dict)
        return obj

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        if isinstance(m, dict):
            assert len(m) == 1
            model_name, model = list(m.items())[0]
            assert isinstance(model_name, str)
            self.ModelDict.update(m)
            self._model = model_name
        elif isinstance(m, str):
            assert m in self.ModelDict
            self._model = m
        else:
            self.ModelDict.update({'custom': m})
            self._model = 'custom'

    def _get_args(self):
        return [arg for arg in dir(self) if not arg[0]=='_' and arg not in ['get_opt',
                                                                            'build_model',
                                                                            'model_type',
                                                                            'validate',
                                                                            'from_dict',
                                                                            'get_dict',
                                                                            'get_loss',
                                                                            'get_normalizer',
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

    def get_normalizer(self):
        return None if not self.normalizer else StainNormalizer(method=self.normalizer, source=self.normalizer_source)

    def load_dict(self, hp_dict):
        for key, value in hp_dict.items():
            if not hasattr(self, key):
                log.error(f'Unrecognized hyperparameter {key}; unable to load')
            try:
                setattr(self, key, value)
            except:
                log.error(f'Error setting hyperparameter {key} to {value}; unable to hyperparameters')

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
        if (self.model_type() != 'categorical' and self.early_stop_method == 'accuracy'):
            raise HyperParameterError(f'Model type "{self.model_type()}" is not compatible with early stopping method "accuracy"')
        return True

    def model_type(self):
        """Returns either 'linear', 'categorical', or 'cph' depending on the loss type."""
        if self.loss == 'negative_log_likelihood':
            return 'cph'
        elif self.loss in self.LinearLossDict:
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
                    writer.writerow([slide, 'validation', outcome_label])
    if save_loc:
        save_file.close()
    return out