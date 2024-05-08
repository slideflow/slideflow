'''Submodule that includes base classes to be extended by framework-specific implementations.'''

import json
import pandas as pd
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from pandas.api.types import is_float_dtype, is_integer_dtype

import numpy as np
import slideflow as sf
from slideflow import errors
from slideflow.util import log, log_manifest, no_scope  # noqa: F401

if TYPE_CHECKING:
    from slideflow.norm import StainNormalizer


class _ModelParams:
    """Build a set of hyperparameters."""

    ModelDict = {}  # type: Dict
    LinearLossDict = {}  # type: Dict
    AllLossDict = {}  # type: Dict

    def __init__(
        self,
        *,
        tile_px: int = 299,
        tile_um: Union[int, str] = 302,
        epochs: Union[int, List[int]] = 3,
        toplayer_epochs: int = 0,
        model: str = 'xception',
        pooling: str = 'max',
        loss: Union[str, Dict] = 'sparse_categorical_crossentropy',
        learning_rate: float = 0.0001,
        learning_rate_decay: float = 0,
        learning_rate_decay_steps: float = 100000,
        batch_size: int = 16,
        hidden_layers: int = 0,
        hidden_layer_width: int = 500,
        optimizer: str = 'Adam',
        early_stop: bool = False,
        early_stop_patience: int = 0,
        early_stop_method: str = 'loss',
        manual_early_stop_epoch: Optional[int] = None,
        manual_early_stop_batch: Optional[int] = None,
        uq: bool = False,
        training_balance: Optional[str] = 'auto',
        validation_balance: Optional[str] = 'none',
        trainable_layers: int = 0,
        l1: Optional[float] = 0,
        l2: Optional[float] = 0,
        l1_dense: Optional[float] = 0,
        l2_dense: Optional[float] = 0,
        dropout: Optional[float] = 0,
        augment: Optional[str] = 'xyrj',
        normalizer: Optional[str] = None,
        normalizer_source: Optional[str] = None,
        include_top: bool = True,
        drop_images: bool = False
    ) -> None:
        """Configure a set of training parameters via keyword arguments.

        Parameters are configured in the context of the current deep learning
        backend (Tensorflow or PyTorch), which can be viewed with
        :func:`slideflow.backend`. While most model parameters are
        cross-compatible between Tensorflow and PyTorch, some parameters are
        unique to a backend, so this object should be configured in the same
        backend that the model will be trained in.

        Args:
            tile_px (int): Tile width in pixels. Defaults to 299.
            tile_um (int or str): Tile width in microns (int) or
                magnification (str, e.g. "20x"). Defaults to 302.
            epochs (int): Number of epochs to train the full model. Defaults to 3.
            toplayer_epochs (int): Number of epochs to only train the fully-connected layers. Defaults to 0.
            model (str): Base model architecture name. Defaults to 'xception'.
            pooling (str): Post-convolution pooling. 'max', 'avg', or 'none'. Defaults to 'max'.
            loss (str): Loss function. Defaults to 'sparse_categorical_crossentropy'.
            learning_rate (float): Learning rate. Defaults to 0.0001.
            learning_rate_decay (int): Learning rate decay rate. Defaults to 0.
            learning_rate_decay_steps (int): Learning rate decay steps. Defaults to 100000.
            batch_size (int): Batch size. Defaults to 16.
            hidden_layers (int): Number of fully-connected hidden layers after core model. Defaults to 0.
            hidden_layer_width (int): Width of fully-connected hidden layers. Defaults to 500.
            optimizer (str): Name of optimizer. Defaults to 'Adam'.
            early_stop (bool): Use early stopping. Defaults to False.
            early_stop_patience (int): Patience for early stopping, in epochs. Defaults to 0.
            early_stop_method (str): Metric to monitor for early stopping. Defaults to 'loss'.
            manual_early_stop_epoch (int, optional): Manually override early stopping to occur at this epoch/batch.
                Defaults to None.
            manual_early_stop_batch (int, optional): Manually override early stopping to occur at this epoch/batch.
                Defaults to None.
            training_balance (str, optional): Type of batch-level balancing to use during training.
                Options include 'tile', 'category', 'patient', 'slide', and None. Defaults to 'category' if a
                categorical loss is provided, and 'patient' if a linear loss is provided.
            validation_balance (str, optional): Type of batch-level balancing to use during validation.
                Options include 'tile', 'category', 'patient', 'slide', and None. Defaults to 'none'.
            trainable_layers (int): Number of layers which are traininable. If 0, trains all layers.
                Defaults to 0.
            l1 (int, optional): L1 regularization weight. Defaults to 0.
            l2 (int, optional): L2 regularization weight. Defaults to 0.
            l1_dense (int, optional): L1 regularization weight for Dense layers. Defaults to the value of l1.
            l2_dense (int, optional): L2 regularization weight for Dense layers. Defaults to the value of l2.
            dropout (int, optional): Post-convolution dropout rate. Defaults to 0.
            uq (bool, optional): Use uncertainty quantification with dropout. Requires dropout > 0. Defaults to False.
            augment (str, optional): Image augmentations to perform. Characters in the string designate augmentations.
                Combine these characters to define the augmentation pipeline. For example, 'xyrj' will perform x-flip,
                y-flip, rotation, and JPEG compression. True will use all augmentations. Defaults to 'xyrj'.

                .. list-table::
                    :header-rows: 1
                    :widths: 10 90

                    * - Character
                      - Augmentation
                    * - x
                      - Random x-flipping
                    * - y
                      - Random y-flipping
                    * - r
                      - Random cardinal rotation
                    * - j
                      - Random JPEG compression (10% chance to JPEG compress with quality between 50-100%)
                    * - b
                      - Random Guassian blur (50% chance to blur with sigma between 0.5 - 2.0)
                    * - n
                      - :ref:`stain_augmentation` (requires stain normalizer)


            normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Stain normalization preset or
                path to a source image. Valid presets include 'v1', 'v2', and
                'v3'. If None, will use the default present ('v3').
                Defaults to None.
            include_top (bool): Include post-convolution fully-connected layers from the core model. Defaults
                to True. include_top=False is not currently compatible with the PyTorch backend.
            drop_images (bool): Drop images, using only other slide-level features as input.
                Defaults to False.
        """
        if isinstance(tile_um, str):
            sf.util.assert_is_mag(tile_um)
            tile_um = tile_um.lower()

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
            self.training_balance = training_balance  # type: ignore
        self.validation_balance = validation_balance
        self.hidden_layer_width = hidden_layer_width
        self.trainable_layers = trainable_layers
        self.l1 = l1 if l1 is None else float(l1)
        self.l2 = l2 if l2 is None else float(l2)
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

    def __repr__(self) -> str:
        base = "ModelParams("
        for arg in self._get_args():
            base += "\n  {} = {!r},".format(arg, getattr(self, arg))
        base += "\n)"
        return base

    def __str__(self) -> str:
        args = sorted(self._get_args(), key=lambda arg: arg.lower())
        arg_dict = {arg: getattr(self, arg) for arg in args}
        return json.dumps(arg_dict, indent=2)

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    @classmethod
    def from_dict(cls, hp_dict: Dict) -> "_ModelParams":
        obj = cls()
        obj.load_dict(hp_dict)
        return obj

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, m: Union[str, Any, Dict[str, Any]]) -> None:
        if isinstance(m, dict):
            assert len(m) == 1
            model_name, model = list(m.items())[0]
            assert isinstance(model_name, str)
            self.ModelDict.update(m)
            self._model = model_name
        elif isinstance(m, str):
            assert m in self.ModelDict or m.startswith('timm_')
            self._model = m
        else:
            self.ModelDict.update({'custom': m})
            self._model = 'custom'

    @property
    def loss(self) -> str:
        return self._loss

    @loss.setter
    def loss(self, l: Union[str, Dict])  -> None:
        if isinstance(l, dict):
            # Verify that the custom loss dictionary provided is valid.
            valid_loss_types = ('cph', 'linear', 'categorical')
            if 'type' not in l or 'fn' not in l:
                raise ValueError("If supplying a custom loss, dictionary must "
                                 "have the keys 'type' and 'fn'.")
            if l['type'] not in valid_loss_types:
                raise ValueError("Custom loss type must be one of: ",
                                 ', '.join(valid_loss_types))
            loss_name = 'custom_' + l['type']
            self.AllLossDict.update({loss_name: l['fn']})
            self._loss = loss_name
        elif isinstance(l, str):
            assert l in self.AllLossDict
            self._loss = l

    def _get_args(self) -> List[str]:
        to_ignore = [
            'get_opt',
            'build_model',
            'model_type',
            'validate',
            'to_dict',
            'from_dict',
            'get_dict',
            'get_loss',
            'get_normalizer',
            'load_dict',
            'OptDict',
            'ModelDict',
            'LinearLossDict',
            'AllLossDict',
            'get_model_loader'
        ]
        args = [
            arg for arg in dir(self)
            if arg[0] != '_' and arg not in to_ignore
        ]
        return args

    def get_dict(self) -> Dict[str, Any]:
        """Deprecated. Alias of ModelParams.to_dict()."""
        warnings.warn(
            "ModelParams.get_dict() is deprecated. Please use .to_dict()",
            DeprecationWarning
        )
        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary of configured parameters."""
        d = {}
        for arg in self._get_args():
            d.update({arg: getattr(self, arg)})
        return d

    def get_normalizer(self, **kwargs) -> Optional["StainNormalizer"]:
        """Return a configured :class:`slideflow.StainNormalizer`."""
        if not self.normalizer:
            return None
        else:
            return sf.norm.autoselect(self.normalizer, self.normalizer_source, **kwargs)

    def load_dict(self, hp_dict: Dict[str, Any]) -> None:
        for key, value in hp_dict.items():
            if not hasattr(self, key):
                log.error(f'Unrecognized hyperparameter {key}; unable to load')
            try:
                setattr(self, key, value)
            except Exception:
                log.error(f'Error setting hyperparameter {key} to {value}; unable to hyperparameters')
        self.validate()

    def _detect_classes_from_labels(
        self,
        labels: Dict
    ) -> Union[int, Dict[int, int]]:

        if isinstance(labels, dict):
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
        elif isinstance(labels, pd.DataFrame):
            if 'label' not in labels.columns:
                raise errors.ModelError("Expected DataFrame with 'label' "
                                        "column.")
            if is_float_dtype(labels.label):
                return 1

            elif self.model_type() == 'categorical':
                unique = labels.label.unique()
                if is_integer_dtype(labels.label):
                    return {0: unique.max()+1}
                else:
                    return {0: len(unique)}
            else:
                print(self.model_type)
                raise errors.ModelError(
                    "Expected integer or float labels. Got: {}".format(
                        labels.label.dtype
                    )
                )
        else:
            raise errors.ModelError(
                "Expected dict or DataFrame. Got: {}".format(type(labels))
            )


    def validate(self) -> bool:
        """Check that hyperparameter combinations are valid."""

        # Assertions.
        assert isinstance(self.tile_px, int)
        assert isinstance(self.tile_um, (str, int))
        assert isinstance(self.toplayer_epochs, int)
        assert isinstance(self.epochs, (int, list))
        if isinstance(self.epochs, list):
            assert all([isinstance(t, int) for t in self.epochs])
        assert self.pooling in ['max', 'avg', 'none']
        assert isinstance(self.learning_rate, float)
        assert isinstance(self.learning_rate_decay, (int, float))
        assert isinstance(self.learning_rate_decay_steps, (int))
        assert isinstance(self.batch_size, int)
        assert isinstance(self.hidden_layers, int)
        assert isinstance(self.manual_early_stop_batch, int) or self.manual_early_stop_batch is None
        assert isinstance(self.manual_early_stop_epoch, int) or self.manual_early_stop_epoch is None
        assert isinstance(self.early_stop, bool)
        assert isinstance(self.early_stop_patience, int)
        assert self.early_stop_method in ['loss', 'accuracy', 'manual']
        assert self.training_balance in ['auto', 'tile', 'category', 'patient', 'slide', 'none', None]
        assert self.validation_balance in ['tile', 'category', 'patient', 'slide', 'none', None]
        assert isinstance(self.hidden_layer_width, int)
        assert isinstance(self.trainable_layers, int)
        assert isinstance(self.l1, (int, float))
        assert isinstance(self.l2, (int, float))
        assert isinstance(self.dropout, (int, float))
        assert isinstance(self.uq, bool)
        assert isinstance(self.augment, (bool, str))
        assert isinstance(self.drop_images, bool)
        assert isinstance(self.include_top, bool)

        assert 0 <= self.learning_rate_decay <= 1
        assert 0 <= self.l1 <= 1
        assert 0 <= self.l2 <= 1
        assert 0 <= self.dropout <= 1

        if self.l1_dense is not None:
            assert isinstance(self.l1_dense, (int, float))
            assert 0 <= self.l1_dense <= 1

        if self.l2_dense is not None:
            assert isinstance(self.l2_dense, (int, float))
            assert 0 <= self.l2_dense <= 1

        # Augmentation checks.
        valid_aug = 'xyrjbn' if sf.backend() == 'tensorflow' else 'xyrdspbjnc'
        if isinstance(self.augment, str) and not all(s in valid_aug for s in self.augment):
            raise errors.ModelParamsError(
                "Unrecognized augmentation(s): {}".format(
                    ','.join([s for s in self.augment if s not in valid_aug])
                )
            )

        # Specific considerations.
        if isinstance(self.tile_um, str):
            sf.util.assert_is_mag(self.tile_um)
            self.tile_um = self.tile_um.lower()
        if self.training_balance == 'auto':
            self.training_balance = 'category' if self.model_type() == 'categorical' else 'patient'
        if not isinstance(self.epochs, list):
            self.epochs = [self.epochs]

        # PyTorch checks.
        if sf.backend() == 'torch':
            if self.l2_dense:
                log.warn(
                    "'l2_dense' is not implemented in PyTorch backend. "
                    "L2 regularization must be applied to the whole model "
                    "by setting 'l2' instead. 'l1_dense' will be ignored.")
            if self.l1_dense or self.l1:
                log.warn(
                    "L1 regularization is not implemented in PyTorch backend "
                    "and will be ignored.")

        # Model type validations.
        if (self.model_type() != 'categorical' and ((self.training_balance == 'category') or
                                                    (self.validation_balance == 'category'))):
            raise errors.ModelParamsError(
                f'Cannot combine category-level balancing with model type "{self.model_type()}".'
            )
        if (self.model_type() != 'categorical' and self.early_stop_method == 'accuracy'):
            raise errors.ModelParamsError(
                f'Model type "{self.model_type()}" is not compatible with early stopping method "accuracy"'
            )
        if self.uq and not self.dropout:
            raise errors.ModelParamsError(
                f"Uncertainty quantification (uq=True) requires dropout > 0 (got: dropout={self.dropout})"
            )
        if (self.early_stop
           and self.early_stop_method == 'manual'
           and (self.manual_early_stop_epoch is None
                or self.manual_early_stop_batch is None)):
            raise errors.ModelParamsError(
                f'HP warning: both manual_early_stop_batch (got: {self.manual_early_stop_batch}) and '
                f'manual_early_stop_epoch (got: {self.manual_early_stop_epoch}) must be != None '
                'to trigger manual early stopping.'
            )
        return True

    def model_type(self) -> str:
        """Returns either 'linear', 'categorical', or 'cph' depending on the loss type."""
        #check if loss is custom_[type] and returns type
        if self.loss.startswith('custom'):
            return self.loss[7:]
        elif self.loss == 'negative_log_likelihood' or self.loss == 'CoxProportionalHazardsLoss':
            return 'cph'
        elif self.loss in self.LinearLossDict:
            return 'linear'
        else:
            return 'categorical'


class BaseFeatureExtractor:
    """Base feature extractor, to be extended by subclasses.

    Considerations when extending this class:
    - Be sure to set .num_classes and .num_features accordingly
    - Be sure to included any needed preprocessing steps in .preprocess_kwargs.
      These keyword arguments will be passed to :meth:`Dataset.tensorflow()` and
      :meth:`Dataset.torch()` when datasets are being prepared to interface
      with this feature extractor.
    - __call__ should return feature vectors for a batch of pre-processed images.
    - If you want to perform custom preprocessing that cannot be supported
      using preprocess_kwargs, set .preprocess_kwargs = {'standardize': False}
      and include all preprocessing steps in __call__.

    """

    tag = 'generic_extractor'
    license = ''
    citation = ''

    def __init__(self, backend: str, include_preds: bool = False) -> None:
        """Initialize the base feature extractor.

        Args:
            backend (str): Either 'tensorflow' or 'torch'. Used to determine
                which Tensor format this feature extractor can work with.
            include_preds (bool): Whether the output of this extractor
                also returns predictions. If so, they should be returned
                after the features. Defaults to False.
        """

        assert backend in ('tensorflow', 'torch')
        self.include_preds = include_preds

        # ---------------------------------------------------------------------
        self.num_classes = 0
        self.num_features = 0
        self.num_uncertainty = 0
        self.preprocess_kwargs = {}
        # ---------------------------------------------------------------------

        self.backend = backend
        self.img_format = None
        self.wsi_normalizer = None
        self.include_preds = include_preds


    def __str__(self):
        return "<{} n_features={}, n_classes={}>".format(
            self.__class__.__name__,
            self.num_features,
            self.num_classes,
        )

    @property
    def normalizer(self) -> Optional["StainNormalizer"]:
        """Returns the configured whole-slide image normalizer."""
        return self.wsi_normalizer

    @normalizer.setter
    def normalizer(self, normalizer: "StainNormalizer") -> None:
        """Sets the normalizer property."""
        self.wsi_normalizer = normalizer

    def is_torch(self):
        return self.backend == 'torch'

    def is_tensorflow(self):
        return self.backend == 'tensorflow'

    def __call__(self, obj, **kwargs):
        raise NotImplementedError

    def dump_config(self):
        """Dump the configuration of this feature extractor.

        The configuration should be a dictionary of all parameters needed to
        re-instantiate this feature extractor. The dictionary should have the
        keys 'class' and 'kwargs', where 'class' is the name of the class, and
        'kwargs' is a dictionary of keyword arguments.
        """
        raise NotImplementedError

    def print_license(self) -> None:
        """Print the license statement for the pretrained model."""
        if self.license:
            print(self.license)
        else:
            print("No license available.")

    def cite(self):
        """Print the citation for the pretrained model in Nature format."""
        if self.citation:
            print(self.citation)
        else:
            print("No citation available.")


class HyperParameterError(Exception):
    pass
