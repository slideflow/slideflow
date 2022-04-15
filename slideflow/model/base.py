'''Submodule that includes base classes to be extended by framework-specific implementations.'''

import json
import os
import csv
import numpy as np
import slideflow as sf
from slideflow.util import log
from slideflow import errors


class _ModelParams:
    """Build a set of hyperparameters."""

    def __init__(self, tile_px=299, tile_um=302, epochs=3, toplayer_epochs=0, model='xception', pooling='max',
                 loss='sparse_categorical_crossentropy', learning_rate=0.0001, learning_rate_decay=0,
                 learning_rate_decay_steps=100000, batch_size=16, hidden_layers=0, hidden_layer_width=500,
                 optimizer='Adam', early_stop=False, early_stop_patience=0, early_stop_method='loss',
                 manual_early_stop_epoch=None, manual_early_stop_batch=None, uq=False, training_balance='auto',
                 validation_balance='none', trainable_layers=0, l1=0, l2=0, l1_dense=None, l2_dense=None, dropout=0,
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
            manual_early_stop_epoch (int, optional): Manually override early stopping to occur at this epoch/batch.
                Defaults to None.
            manual_early_stop_batch (int, optional): Manually override early stopping to occur at this epoch/batch.
                Defaults to None.
            training_balance ([type], optional): Type of batch-level balancing to use during training.
                Options include 'tile', 'category', 'patient', 'slide', and None. Defaults to 'category' if a
                categorical loss is provided, and 'patient' if a linear loss is provided.
            validation_balance ([type], optional): Type of batch-level balancing to use during validation.
                Options include 'tile', 'category', 'patient', 'slide', and None. Defaults to 'none'.
            trainable_layers (int, optional): Number of layers which are traininable. If 0, trains all layers.
                Defaults to 0.
            l1 (int, optional): L1 regularization weight. Defaults to 0.
            l2 (int, optional): L2 regularization weight. Defaults to 0.
            l1_dense (int, optional): L1 regularization weight for Dense layers. Defaults to the value of l1.
            l2_dense (int, optional): L2 regularization weight for Dense layers. Defaults to the value of l2.
            dropout (int, optional): Post-convolution dropout rate. Defaults to 0.
            uq (bool, optional): Use uncertainty quantification with dropout. Requires dropout > 0. Defaults to False.
            augment (str): Image augmentations to perform. String containing characters designating augmentations.
                'x' indicates random x-flipping, 'y' y-flipping, 'r' rotating, and 'j' JPEG compression/decompression
                at random quality levels. Passing either 'xyrj' or True will use all augmentations.
            normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image. Defaults to None.
                If None but using a normalizer, will use an internal tile for normalization.
                Internal default tile can be found at slideflow.slide.norm_tile.jpg
            include_top (bool, optional): Include post-convolution fully-connected layers from the core model. Defaults
                to True. include_top=False is not currently compatible with the PyTorch backend.
            drop_images (bool, optional): Drop images, using only other slide-level features as input.
                Defaults to False.
        """
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
        assert isinstance(manual_early_stop_batch, int) or manual_early_stop_batch is None
        assert isinstance(manual_early_stop_epoch, int) or manual_early_stop_epoch is None
        assert isinstance(early_stop, bool)
        assert isinstance(early_stop_patience, int)
        assert early_stop_method in ['loss', 'accuracy', 'manual']
        assert training_balance in ['auto', 'tile', 'category', 'patient', 'slide', 'none', None]
        assert validation_balance in ['tile', 'category', 'patient', 'slide', 'none', None]
        assert isinstance(hidden_layer_width, int)
        assert isinstance(trainable_layers, int)
        assert isinstance(l1, (int, float))
        assert isinstance(l2, (int, float))
        assert isinstance(dropout, (int, float))
        assert isinstance(uq, bool)
        assert isinstance(augment, (bool, str))
        assert isinstance(drop_images, bool)
        assert isinstance(include_top, bool)

        assert 0 <= learning_rate_decay <= 1
        assert 0 <= l1 <= 1
        assert 0 <= l2 <= 1
        assert 0 <= dropout <= 1

        if l1_dense is not None:
            assert isinstance(l1_dense, (int, float))
            assert 0 <= l1_dense <= 1

        if l2_dense is not None:
            assert isinstance(l2_dense, (int, float))
            assert 0 <= l2_dense <= 1

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
        self.manual_early_stop_batch = manual_early_stop_batch
        self.manual_early_stop_epoch = manual_early_stop_epoch
        self.hidden_layers = hidden_layers
        if training_balance == 'auto':
            self.training_balance = 'category' if self.model_type() == 'categorical' else 'patient'
        else:
            self.training_balance = training_balance
        self.validation_balance = validation_balance
        self.hidden_layer_width = hidden_layer_width
        self.trainable_layers = trainable_layers
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.l1_dense = self.l1 if l1_dense is None else float(l1_dense)
        self.l2_dense = self.l2 if l2_dense is None else float(l2_dense)
        self.dropout = dropout
        self.uq = uq
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
        to_ignore = [
            'get_opt',
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
            'AllLossDict'
        ]
        args = [
            arg for arg in dir(self)
            if arg[0] != '_' and arg not in to_ignore
        ]
        return args

    def get_dict(self):
        d = {}
        for arg in self._get_args():
            d.update({arg: getattr(self, arg)})
        return d

    def get_normalizer(self, **kwargs):
        if not self.normalizer:
            return None
        else:
            return sf.norm.autoselect(self.normalizer, self.normalizer_source, **kwargs)

    def load_dict(self, hp_dict):
        for key, value in hp_dict.items():
            if not hasattr(self, key):
                log.error(f'Unrecognized hyperparameter {key}; unable to load')
            try:
                setattr(self, key, value)
            except Exception:
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
            return {i: np.unique(outcome_labels[:, i]).shape[0] for i in range(outcome_labels.shape[1])}
        else:
            try:
                return outcome_labels.shape[1]
            except TypeError:
                raise errors.ModelParamsError('Incorrect formatting of outcomes for linear model; expected ndarray.')

    def validate(self):
        """Check that hyperparameter combinations are valid."""
        if (self.model_type() != 'categorical' and ((self.training_balance == 'category') or
                                                    (self.validation_balance == 'category'))):
            msg = f'Cannot combine category-level balancing with model type "{self.model_type()}".'
            raise errors.ModelParamsError(msg)
        if (self.model_type() != 'categorical' and self.early_stop_method == 'accuracy'):
            msg = f'Model type "{self.model_type()}" is not compatible with early stopping method "accuracy"'
            raise errors.ModelParamsError(msg)
        if self.uq and not self.dropout:
            msg = f"Uncertainty quantification (uq=True) requires dropout > 0 (got: dropout={self.dropout})"
            raise errors.ModelParamsError(msg)
        if (self.early_stop
           and self.early_stop_method == 'manual'
           and (self.manual_early_stop_epoch is None
                or self.manual_early_stop_batch is None)):
            msg = f'HP warning: both manual_early_stop_batch (got: {self.manual_early_stop_batch}) and '
            msg += f'manual_early_stop_epoch (got: {self.manual_early_stop_epoch}) must be != None '
            msg += 'to trigger manual early stopping.'
            raise errors.ModelParamsError(msg)
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
        neptune_run['summary'] = "\n".join(summary_string)


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
