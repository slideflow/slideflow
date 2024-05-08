'''Tensorflow backend for the slideflow.model submodule.'''

from __future__ import absolute_import, division, print_function

import atexit
import inspect
import json
import logging
import os
import shutil
import numpy as np
import multiprocessing as mp
import tensorflow as tf
from packaging import version
from os.path import dirname, exists, join
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Callable, Iterable
)
from tensorflow.keras import applications as kapps

import slideflow as sf
import slideflow.model.base as _base
import slideflow.util.neptune_utils
from slideflow import errors
from slideflow.util import log, NormFit, no_scope

from . import tensorflow_utils as tf_utils
from .base import log_manifest, BaseFeatureExtractor
from .tensorflow_utils import unwrap, flatten, eval_from_model, build_uq_model  # type: ignore

# Set the tensorflow logger
if sf.getLoggingLevel() == logging.DEBUG:
    logging.getLogger('tensorflow').setLevel(logging.DEBUG)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
else:
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sf.util.allow_gpu_memory_growth()

if TYPE_CHECKING:
    import pandas as pd
    from slideflow.norm import StainNormalizer


class StaticDropout(tf.keras.layers.Dropout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        return super().call(inputs, training=True)


class ModelParams(_base._ModelParams):
    """Build a set of hyperparameters."""

    ModelDict = {
        'xception': kapps.Xception,
        'vgg16': kapps.VGG16,
        'vgg19': kapps.VGG19,
        'resnet50': kapps.ResNet50,
        'resnet101': kapps.ResNet101,
        'resnet152': kapps.ResNet152,
        'resnet50_v2': kapps.ResNet50V2,
        'resnet101_v2': kapps.ResNet101V2,
        'resnet152_v2': kapps.ResNet152V2,
        'inception': kapps.InceptionV3,
        'nasnet_large': kapps.NASNetLarge,
        'inception_resnet_v2': kapps.InceptionResNetV2,
        'mobilenet': kapps.MobileNet,
        'mobilenet_v2': kapps.MobileNetV2,
        'densenet_121': kapps.DenseNet121,
        'densenet_169': kapps.DenseNet169,
        'densenet_201': kapps.DenseNet201,
        # 'ResNeXt50': kapps.ResNeXt50,
        # 'ResNeXt101': kapps.ResNeXt101,
        # 'NASNet': kapps.NASNet
    }
    OptDict = {
        'Adam': tf.keras.optimizers.Adam,
        'SGD': tf.keras.optimizers.SGD,
        'RMSprop': tf.keras.optimizers.RMSprop,
        'Adagrad': tf.keras.optimizers.Adagrad,
        'Adadelta': tf.keras.optimizers.Adadelta,
        'Adamax': tf.keras.optimizers.Adamax,
        'Nadam': tf.keras.optimizers.Nadam
    }
    if hasattr(kapps, 'EfficientNetV2B0'):
        ModelDict.update({'efficientnet_v2b0': kapps.EfficientNetV2B0})
    if hasattr(kapps, 'EfficientNetV2B1'):
        ModelDict.update({'efficientnet_v2b1': kapps.EfficientNetV2B1})
    if hasattr(kapps, 'EfficientNetV2B2'):
        ModelDict.update({'efficientnet_v2b2': kapps.EfficientNetV2B2})
    if hasattr(kapps, 'EfficientNetV2B3'):
        ModelDict.update({'efficientnet_v2b3': kapps.EfficientNetV2B3})
    if hasattr(kapps, 'EfficientNetV2S'):
        ModelDict.update({'efficientnet_v2s': kapps.EfficientNetV2S})
    if hasattr(kapps, 'EfficientNetV2M'):
        ModelDict.update({'efficientnet_v2m': kapps.EfficientNetV2M})
    if hasattr(kapps, 'EfficientNetV2L'):
        ModelDict.update({'efficientnet_v2l': kapps.EfficientNetV2L})
    LinearLossDict = {
        loss: getattr(tf.keras.losses, loss)
        for loss in [
            'mean_squared_error',
            'mean_absolute_error',
            'mean_absolute_percentage_error',
            'mean_squared_logarithmic_error',
            'squared_hinge',
            'hinge',
            'logcosh'
        ]
    }
    LinearLossDict.update({
        'negative_log_likelihood': tf_utils.negative_log_likelihood
    })
    AllLossDict = {
        loss: getattr(tf.keras.losses, loss)
        for loss in [
            'mean_squared_error',
            'mean_absolute_error',
            'mean_absolute_percentage_error',
            'mean_squared_logarithmic_error',
            'squared_hinge',
            'hinge',
            'categorical_hinge',
            'logcosh',
            'huber',
            'categorical_crossentropy',
            'sparse_categorical_crossentropy',
            'binary_crossentropy',
            'kullback_leibler_divergence',
            'poisson'
        ]
    }
    AllLossDict.update({
        'batch_loss_crossentropy': tf_utils.batch_loss_crossentropy,
        'negative_log_likelihood': tf_utils.negative_log_likelihood
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.model in self.ModelDict.keys()
        assert self.optimizer in self.OptDict.keys()
        assert self.loss in self.AllLossDict.keys()

    def _add_hidden_layers(
        self,
        model: tf.keras.Model,
        regularizer: tf.keras.layers.Layer
    ) -> Tuple[tf.keras.Model, tf.keras.layers.Layer]:
        """Adds hidden layer(s) to a model.

        Args:
            model (tf.keras.Model): Tensorflow model.
            regularizer (tf.keras.layers.Layer): Regularization for hidden layers.

        Returns:
            A tuple containing

                tf.keras.Model: Model with hidden layers added.

                tf.keras.layers.Layer: Last linear layer.
        """
        log.debug("Using Batch normalization")
        last_linear = None
        for i in range(self.hidden_layers):
            model = tf.keras.layers.Dense(self.hidden_layer_width,
                                          name=f'hidden_{i}',
                                          activation='relu',
                                          kernel_regularizer=regularizer)(model)
            model = tf.keras.layers.BatchNormalization()(model)
            last_linear = model
            if self.uq:
                model = StaticDropout(self.dropout)(model)
            elif self.dropout:
                model = tf.keras.layers.Dropout(self.dropout)(model)
        return model, last_linear

    def _get_dense_regularizer(self) -> Optional[tf.keras.layers.Layer]:
        """Return regularizer for dense (hidden) layers."""

        if self.l2_dense and not self.l1_dense:
            log.debug(f"Using L2 regularization for dense layers (weight={self.l2_dense})")
            return tf.keras.regularizers.l2(self.l2_dense)
        elif self.l1_dense and not self.l2_dense:
            log.debug(f"Using L1 regularization for dense layers (weight={self.l1_dense})")
            return tf.keras.regularizers.l1(self.l1_dense)
        elif self.l1_dense and self.l2_dense:
            log.debug(f"Using L1 (weight={self.l1_dense}) and L2 (weight={self.l2_dense}) reg for dense layers")
            return tf.keras.regularizers.l1_l2(l1=self.l1_dense, l2=self.l2_dense)
        else:
            log.debug("Not using regularization for dense layers")
            return None

    def _add_regularization(self, model: tf.keras.Model) -> tf.keras.Model:
        """Add non-hidden layer regularization.

        Args:
            model (tf.keras.Model): Tensorflow model.

        Returns:
            tf.keras.Model: Tensorflow model with regularization added.
        """
        if self.l2 and not self.l1:
            log.debug(f"Using L2 regularization for base model (weight={self.l2})")
            regularizer = tf.keras.regularizers.l2(self.l2)
        elif self.l1 and not self.l2:
            log.debug(f"Using L1 regularization for base model (weight={self.l1})")
            regularizer = tf.keras.regularizers.l1(self.l1)
        elif self.l1 and self.l2:
            log.debug(f"Using L1 (weight={self.l1}) and L2 (weight={self.l2}) regularization for base model")
            regularizer = tf.keras.regularizers.l1_l2(l1=self.l1, l2=self.l2)
        else:
            log.debug("Not using regularization for base model")
            regularizer = None
        if regularizer is not None:
            model = tf_utils.add_regularization(model, regularizer)
        return model

    def _freeze_layers(self, model: tf.keras.Model) -> tf.keras.Model:
        """Freeze last X layers, where X = self.trainable_layers.

        Args:
            model (tf.keras.Model): Tensorflow model.

        Returns:
            tf.keras.Model: Tensorflow model with frozen layers.
        """
        freezeIndex = int(len(model.layers) - (self.trainable_layers - 1))  # - self.hp.hidden_layers - 1))
        log.info(f'Only training on last {self.trainable_layers} layers (of {len(model.layers)} total)')
        for layer in model.layers[:freezeIndex]:
            layer.trainable = False
        return model

    def _get_core(self, weights: Optional[str] = None) -> tf.keras.Model:
        """Returns a Keras model of the appropriate architecture, input shape,
        pooling, and initial weights.

        Args:
            weights (Optional[str], optional): Pretrained weights to use.
                Defaults to None.

        Returns:
            tf.keras.Model: Core model.
        """
        input_shape = (self.tile_px, self.tile_px, 3)
        model_fn = self.ModelDict[self.model]
        model_kwargs = {
            'input_shape': input_shape,
            'include_top': self.include_top,
            'pooling': self.pooling,
            'weights': weights
        }
        # Only pass kwargs accepted by model function
        model_fn_sig = inspect.signature(model_fn)
        model_kw = [
            param.name
            for param in model_fn_sig.parameters.values()
            if param.kind == param.POSITIONAL_OR_KEYWORD
        ]
        model_kwargs = {key: model_kwargs[key] for key in model_kw if key in model_kwargs}
        return model_fn(**model_kwargs)

    def _build_base(
        self,
        pretrain: Optional[str] = 'imagenet',
        load_method: str = 'weights'
    ) -> tf.keras.Model:
        """"Builds the base image model, from a Keras model core, with the
        appropriate input tensors and identity layers.

        Args:
            pretrain (str, optional): Pretrained weights to load.
                Defaults to 'imagenet'.
            load_method (str): Either 'full' or 'weights'. Method to use
                when loading a Tensorflow model. If 'full', loads the model with
                ``tf.keras.models.load_model()``. If 'weights', will read the
                ``params.json`` configuration file, build the model architecture,
                and then load weights from the given model with
                ``Model.load_weights()``. Loading with 'full' may improve
                compatibility across Slideflow versions. Loading with 'weights'
                may improve compatibility across hardware & environments.

        Returns:
            tf.keras.Model: Base model.
        """
        image_shape = (self.tile_px, self.tile_px, 3)
        tile_input_tensor = tf.keras.Input(shape=image_shape, name='tile_image')
        if pretrain:
            log.debug(f'Using pretraining from [magenta]{pretrain}')
        if pretrain and pretrain != 'imagenet':
            pretrained_model = load(pretrain, method=load_method, training=True)
            try:
                # This is the tile_image input
                pretrained_input = pretrained_model.get_layer(name='tile_image').input
                # Name of the pretrained model core, which should be at layer 1
                pretrained_name = pretrained_model.get_layer(index=1).name
                # This is the post-convolution layer
                pretrained_output = pretrained_model.get_layer(name='post_convolution').output
                base_model = tf.keras.Model(inputs=pretrained_input,
                                            outputs=pretrained_output,
                                            name=f'pretrained_{pretrained_name}').layers[1]
            except ValueError:
                log.warning('Unable to automatically read pretrained model, will try legacy format')
                base_model = pretrained_model.get_layer(index=0)
        else:
            base_model = self._get_core(weights=pretrain)
            if self.include_top:
                base_model = tf.keras.Model(
                    inputs=base_model.input,
                    outputs=base_model.layers[-2].output,
                    name=base_model.name
                )
        # Add regularization
        base_model = self._add_regularization(base_model)

        # Allow only a subset of layers in the base model to be trainable
        if self.trainable_layers != 0:
            base_model = self._freeze_layers(base_model)

        # This is an identity layer that simply returns the last layer, allowing us to name and access this layer later
        post_convolution_identity_layer = tf.keras.layers.Activation('linear', name='post_convolution')
        layers = [tile_input_tensor, base_model]
        if not self.pooling:
            layers += [tf.keras.layers.Flatten()]
        layers += [post_convolution_identity_layer]
        if self.uq:
            layers += [StaticDropout(self.dropout)]
        elif self.dropout:
            layers += [tf.keras.layers.Dropout(self.dropout)]
        tile_image_model = tf.keras.Sequential(layers)
        model_inputs = [tile_image_model.input]
        return tile_image_model, model_inputs

    def _build_categorical_or_linear_model(
        self,
        num_classes: Union[int, Dict[Any, int]],
        num_slide_features: int = 0,
        activation: str = 'softmax',
        pretrain: str = 'imagenet',
        checkpoint: Optional[str] = None,
        load_method: str = 'weights'
    ) -> tf.keras.Model:
        """Assembles categorical or linear model, using pretraining (imagenet)
        or the base layers of a supplied model.

        Args:
            num_classes (int or dict): Either int (single categorical outcome,
                indicating number of classes) or dict (dict mapping categorical
                outcome names to number of unique categories in each outcome).
            num_slide_features (int): Number of slide-level features separate
                from image input. Defaults to 0.
            activation (str): Type of final layer activation to use.
                Defaults to softmax.
            pretrain (str): Either 'imagenet' or path to model to use as
                pretraining. Defaults to 'imagenet'.
            checkpoint (str): Path to checkpoint from which to resume model
                training. Defaults to None.
            load_method (str): Either 'full' or 'weights'. Method to use
                when loading a Tensorflow model. If 'full', loads the model with
                ``tf.keras.models.load_model()``. If 'weights', will read the
                ``params.json`` configuration file, build the model architecture,
                and then load weights from the given model with
                ``Model.load_weights()``. Loading with 'full' may improve
                compatibility across Slideflow versions. Loading with 'weights'
                may improve compatibility across hardware & environments.
        """
        tile_image_model, model_inputs = self._build_base(pretrain, load_method)
        if num_slide_features:
            log.debug(f'Model has {num_slide_features} slide input features')
            slide_feature_input_tensor = tf.keras.Input(
                shape=(num_slide_features),
                name='slide_feature_input'
            )
        else:
            log.debug('Not using any slide-level input features.')

        # Merge layers
        if num_slide_features and ((self.tile_px == 0) or self.drop_images):
            log.info('Generating model with only slide-level input - no images')
            merged_model = slide_feature_input_tensor
            model_inputs += [slide_feature_input_tensor]
        elif num_slide_features:
            # Add slide feature input tensors
            merged_model = tf.keras.layers.Concatenate(name='input_merge')(
                [slide_feature_input_tensor, tile_image_model.output]
            )
            model_inputs += [slide_feature_input_tensor]
        else:
            merged_model = tile_image_model.output

        # Add hidden layers
        regularizer = self._get_dense_regularizer()
        merged_model, last_linear = self._add_hidden_layers(
            merged_model, regularizer
        )

        # Multi-categorical outcomes
        if isinstance(num_classes, dict):
            outputs = []
            for c in num_classes:
                final_dense_layer = tf.keras.layers.Dense(
                    num_classes[c],
                    kernel_regularizer=regularizer,
                    name=f'logits-{c}'
                )(merged_model)
                outputs += [
                    tf.keras.layers.Activation(
                        activation,
                        dtype='float32',
                        name=f'out-{c}'
                    )(final_dense_layer)
                ]
        else:
            final_dense_layer = tf.keras.layers.Dense(
                num_classes,
                kernel_regularizer=regularizer,
                name='logits'
            )(merged_model)
            outputs = [
                tf.keras.layers.Activation(
                    activation,
                    dtype='float32',
                    name='output'
                )(final_dense_layer)
            ]
        # Assemble final model
        log.debug(f'Using {activation} activation')
        model = tf.keras.Model(inputs=model_inputs, outputs=outputs)
        # Disable experimental batch loss
        if False:
            model.add_loss(tf_utils.batch_loss_crossentropy(last_linear))

        if checkpoint:
            log.info(f'Loading checkpoint weights from [green]{checkpoint}')
            model.load_weights(checkpoint)

        return model

    def _build_cph_model(
        self,
        num_classes: Union[int, Dict[Any, int]],
        num_slide_features: int = 1,
        pretrain: Optional[str] = None,
        checkpoint: Optional[str] = None,
        load_method: str = 'weights',
        training: bool = True
    ) -> tf.keras.Model:
        """Assembles a Cox Proportional Hazards (CPH) model, using pretraining
        (imagenet) or the base layers of a supplied model.

        Args:
            num_classes (int or dict): Either int (single categorical outcome,
                indicating number of classes) or dict (dict mapping categorical
                outcome names to number of unique categories in each outcome).
            num_slide_features (int): Number of slide-level features separate
                from image input. Defaults to 0.
            activation (str): Type of final layer activation to use.
                Defaults to softmax.
            pretrain (str): Either 'imagenet' or path to model to use as
                pretraining. Defaults to 'imagenet'.
            checkpoint (str): Path to checkpoint from which to resume model
                training. Defaults to None.
            load_method (str): Either 'full' or 'weights'. Method to use
                when loading a Tensorflow model. If 'full', loads the model with
                ``tf.keras.models.load_model()``. If 'weights', will read the
                ``params.json`` configuration file, build the model architecture,
                and then load weights from the given model with
                ``Model.load_weights()``. Loading with 'full' may improve
                compatibility across Slideflow versions. Loading with 'weights'
                may improve compatibility across hardware & environments.
        """
        activation = 'linear'
        tile_image_model, model_inputs = self._build_base(pretrain, load_method)

        # Add slide feature input tensors, if there are more slide features
        # than just the event input tensor for CPH models
        if training:
            event_input_tensor = tf.keras.Input(shape=(1), name='event_input')
        if not (num_slide_features == 1):
            slide_feature_input_tensor = tf.keras.Input(
                shape=(num_slide_features - 1),
                name='slide_feature_input'
            )
        # Merge layers
        if num_slide_features and ((self.tile_px == 0) or self.drop_images):
            # Add images
            log.info('Generating model with only slide-level input - no images')
            merged_model = slide_feature_input_tensor
            model_inputs += [slide_feature_input_tensor]
            if training:
                model_inputs += [event_input_tensor]
        elif num_slide_features and num_slide_features > 1:
            # Add slide feature input tensors, if there are more slide features
            # than just the event input tensor for CPH models
            merged_model = tf.keras.layers.Concatenate(name='input_merge')(
                [slide_feature_input_tensor, tile_image_model.output]
            )
            model_inputs += [slide_feature_input_tensor]
            if training:
                model_inputs += [event_input_tensor]
        else:
            merged_model = tile_image_model.output
            if training:
                model_inputs += [event_input_tensor]

        # Add hidden layers
        regularizer = self._get_dense_regularizer()
        merged_model, last_linear = self._add_hidden_layers(
            merged_model, regularizer
        )
        log.debug(f'Using {activation} activation')

        # Multi-categorical outcomes
        if type(num_classes) == dict:
            outputs = []
            for c in num_classes:
                final_dense_layer = tf.keras.layers.Dense(
                    num_classes[c],
                    kernel_regularizer=regularizer,
                    name=f'logits-{c}'
                )(merged_model)
                outputs += [tf.keras.layers.Activation(
                    activation,
                    dtype='float32',
                    name=f'out-{c}'
                )(final_dense_layer)]
        else:
            final_dense_layer = tf.keras.layers.Dense(
                num_classes,
                kernel_regularizer=regularizer,
                name='logits'
            )(merged_model)
            outputs = [tf.keras.layers.Activation(
                activation,
                dtype='float32',
                name='output'
            )(final_dense_layer)]
        if training:
            outputs[0] = tf.keras.layers.Concatenate(
                name='output_merge_CPH',
                dtype='float32'
            )([outputs[0], event_input_tensor])

        # Assemble final model
        model = tf.keras.Model(inputs=model_inputs, outputs=outputs)

        if checkpoint:
            log.info(f'Loading checkpoint weights from [green]{checkpoint}')
            model.load_weights(checkpoint)

        return model

    def build_model(
        self,
        labels: Optional[Dict] = None,
        num_classes: Optional[Union[int, Dict[Any, int]]] = None,
        **kwargs
    ) -> tf.keras.Model:
        """Auto-detects model type (categorical, linear, CPH) from parameters
        and builds, using pretraining or the base layers of a supplied model.

        Args:
            labels (dict, optional): Dict mapping slide names to outcomes.
                Used to detect number of outcome categories.
            num_classes (int or dict, optional): Either int (single categorical
                outcome, indicating number of classes) or dict (dict mapping
                categorical outcome names to number of unique categories in
                each outcome). Must supply either `num_classes` or `label`
                (can detect number of classes from labels)
            num_slide_features (int, optional): Number of slide-level features
                separate from image input. Defaults to 0.
            activation (str, optional): Type of final layer activation to use.
                Defaults to 'softmax' (categorical models) or 'linear'
                (linear or CPH models).
            pretrain (str, optional): Either 'imagenet' or path to model to use
                as pretraining. Defaults to 'imagenet'.
            checkpoint (str, optional): Path to checkpoint from which to resume
                model training. Defaults to None.
            load_method (str): Either 'full' or 'weights'. Method to use
                when loading a Tensorflow model. If 'full', loads the model with
                ``tf.keras.models.load_model()``. If 'weights', will read the
                ``params.json`` configuration file, build the model architecture,
                and then load weights from the given model with
                ``Model.load_weights()``. Loading with 'full' may improve
                compatibility across Slideflow versions. Loading with 'weights'
                may improve compatibility across hardware & environments.
        """

        assert num_classes is not None or labels is not None
        if num_classes is None:
            num_classes = self._detect_classes_from_labels(labels)  # type: ignore

        if self.model_type() == 'categorical':
            return self._build_categorical_or_linear_model(
                num_classes, **kwargs, activation='softmax'
            )
        elif self.model_type() == 'linear':
            return self._build_categorical_or_linear_model(
                num_classes, **kwargs, activation='linear'
            )
        elif self.model_type() == 'cph':
            return self._build_cph_model(num_classes, **kwargs)
        else:
            raise errors.ModelError(f'Unknown model type: {self.model_type()}')

    def get_loss(self) -> tf.keras.losses.Loss:
        return self.AllLossDict[self.loss]

    def get_opt(self) -> tf.keras.optimizers.Optimizer:
        """Returns optimizer with appropriate learning rate."""
        if self.learning_rate_decay not in (0, 1):
            initial_learning_rate = self.learning_rate
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=self.learning_rate_decay_steps,
                decay_rate=self.learning_rate_decay,
                staircase=True
            )
            return self.OptDict[self.optimizer](learning_rate=lr_schedule)
        else:
            return self.OptDict[self.optimizer](learning_rate=self.learning_rate)

    def model_type(self) -> str:
        """Returns 'linear', 'categorical', or 'cph', reflecting the loss."""
        #check if loss is custom_[type] and returns type
        if self.loss.startswith('custom'):
            return self.loss[7:]
        elif self.loss == 'negative_log_likelihood':
            return 'cph'
        elif self.loss in self.LinearLossDict:
            return 'linear'
        else:
            return 'categorical'


class _PredictionAndEvaluationCallback(tf.keras.callbacks.Callback):

    """Prediction and Evaluation Callback used during model training."""

    def __init__(self, parent: "Trainer", cb_args: SimpleNamespace) -> None:
        super(_PredictionAndEvaluationCallback, self).__init__()
        self.parent = parent
        self.hp = parent.hp
        self.cb_args = cb_args
        self.early_stop = False
        self.early_stop_batch = 0
        self.early_stop_epoch = 0
        self.last_ema = -1  # type: float
        self.moving_average = []  # type: List
        self.ema_two_checks_prior = -1  # type: float
        self.ema_one_check_prior = -1  # type: float
        self.epoch_count = cb_args.starting_epoch
        self.model_type = self.hp.model_type()
        self.results = {'epochs': {}}  # type: Dict[str, Dict]
        self.neptune_run = self.parent.neptune_run
        self.global_step = 0
        self.train_summary_writer = tf.summary.create_file_writer(
            join(self.parent.outdir, 'train'))
        self.val_summary_writer = tf.summary.create_file_writer(
            join(self.parent.outdir, 'validation'))

        # Circumvents buffer overflow error with Python 3.10.
        # Without this, a buffer overflow error will be encountered when
        # attempting to make a matplotlib figure (with the tkagg backend)
        # during model evaluation. I have not yet been able to track down
        # the root cause.
        if self.cb_args.using_validation:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.close()

    def _log_training_metrics(self, logs):
        """Log training metrics to Tensorboard/Neptune."""
        # Log to Tensorboard.
        with self.train_summary_writer.as_default():
            for _log in logs:
                tf.summary.scalar(
                    f'batch_{_log}',
                    data=logs[_log],
                    step=self.global_step)
        # Log to neptune.
        if self.neptune_run:
            self.neptune_run['metrics/train/batch/loss'].log(
                logs['loss'],
                step=self.global_step)
            sf.util.neptune_utils.list_log(
                self.neptune_run,
                'metrics/train/batch/accuracy',
                logs['accuracy'],
                step=self.global_step)

    def _log_validation_metrics(self, metrics):
        """Log validation metrics to Tensorboard/Neptune."""
        # Tensorboard logging for validation metrics
        with self.val_summary_writer.as_default():
            for _log in metrics:
                tf.summary.scalar(
                    f'batch_{_log}',
                    data=metrics[_log],
                    step=self.global_step)
        # Log to neptune
        if self.neptune_run:
            for v in metrics:
                self.neptune_run[f"metrics/val/batch/{v}"].log(
                    round(metrics[v], 3),
                    step=self.global_step
                )
            if self.last_ema != -1:
                self.neptune_run["metrics/val/batch/exp_moving_avg"].log(
                    round(self.last_ema, 3),
                    step=self.global_step
                )
            self.neptune_run["early_stop/stopped_early"] = False

    def _log_epoch_evaluation(self, epoch_results, metrics, accuracy, loss, logs={}):
        """Log the end-of-epoch evaluation to CSV, Tensorboard, and Neptune."""
        epoch = self.epoch_count
        run = self.neptune_run
        sf.util.update_results_log(
            self.cb_args.results_log,
            'trained_model',
            {f'epoch{epoch}': epoch_results}
        )
        with self.val_summary_writer.as_default():
            # Note: Tensorboard epoch logging starts with index=0,
            # whereas all other logging starts with index=1
            if isinstance(accuracy, (list, tuple, np.ndarray)):
                for i in range(len(accuracy)):
                    tf.summary.scalar(f'epoch_accuracy-{i}', data=accuracy[i], step=epoch-1)
            elif accuracy is not None:
                tf.summary.scalar(f'epoch_accuracy', data=accuracy, step=epoch-1)
            if isinstance(loss, (list, tuple, np.ndarray)):
                for i in range(len(loss)):
                    tf.summary.scalar(f'epoch_loss-{i}', data=loss[i], step=epoch-1)
            else:
                tf.summary.scalar(f'epoch_loss', data=loss, step=epoch-1)

        # Log epoch results to Neptune
        if run:
            # Training epoch metrics
            run['metrics/train/epoch/loss'].log(logs['loss'], step=epoch)
            sf.util.neptune_utils.list_log(
                run,
                'metrics/train/epoch/accuracy',
                logs['accuracy'],
                step=epoch
            )
            # Validation epoch metrics
            run['metrics/val/epoch/loss'].log(loss, step=epoch)
            sf.util.neptune_utils.list_log(
                run,
                'metrics/val/epoch/accuracy',
                accuracy,
                step=epoch
            )
            for metric in metrics:
                if metrics[metric]['tile'] is None:
                    continue
                for outcome in metrics[metric]['tile']:
                    # If only one outcome, log to metrics/val/epoch/[metric].
                    # If more than one outcome, log to
                    # metrics/val/epoch/[metric]/[outcome_name]
                    def metric_label(s):
                        if len(metrics[metric]['tile']) == 1:
                            return f'metrics/val/epoch/{s}_{metric}'
                        else:
                            return f'metrics/val/epoch/{s}_{metric}/{outcome}'

                    tile_metric = metrics[metric]['tile'][outcome]
                    slide_metric = metrics[metric]['slide'][outcome]
                    patient_metric = metrics[metric]['patient'][outcome]

                    # If only one value for a metric, log to .../[metric]
                    # If more than one value for a metric (e.g. AUC for each
                    # category), log to .../[metric]/[i]
                    sf.util.neptune_utils.list_log(
                        run,
                        metric_label('tile'),
                        tile_metric,
                        step=epoch
                    )
                    sf.util.neptune_utils.list_log(
                        run,
                        metric_label('slide'),
                        slide_metric,
                        step=epoch
                    )
                    sf.util.neptune_utils.list_log(
                        run,
                        metric_label('patient'),
                        patient_metric,
                        step=epoch
                    )

    def _metrics_from_dataset(
        self,
        epoch_label: str,
    ) -> Tuple[Dict, float, float]:
        return sf.stats.metrics_from_dataset(
            self.model,
            model_type=self.hp.model_type(),
            patients=self.parent.patients,
            dataset=self.cb_args.validation_data,
            outcome_names=self.parent.outcome_names,
            label=epoch_label,
            data_dir=self.parent.outdir,
            num_tiles=self.cb_args.num_val_tiles,
            save_predictions=self.cb_args.save_predictions,
            reduce_method=self.cb_args.reduce_method,
            loss=self.hp.get_loss(),
            uq=bool(self.hp.uq),
        )

    def on_epoch_end(self, epoch: int, logs={}) -> None:
        if sf.getLoggingLevel() <= 20:
            print('\r\033[K', end='')
        self.epoch_count += 1
        if (self.epoch_count in [e for e in self.hp.epochs]
           or self.early_stop):
            if self.parent.name:
                model_name = self.parent.name
            else:
                model_name = 'trained_model'
            model_path = os.path.join(
                self.parent.outdir,
                f'{model_name}_epoch{self.epoch_count}'
            )
            if self.cb_args.save_model:
                self.model.save(model_path)
                log.info(f'Trained model saved to [green]{model_path}')

                # Try to copy model settings/hyperparameters file
                # into the model folder
                params_dest = join(model_path, 'params.json')
                if not exists(params_dest):
                    try:
                        config_path = join(dirname(model_path), 'params.json')
                        if self.neptune_run:
                            config = sf.util.load_json(config_path)
                            config['neptune_id'] = self.neptune_run['sys/id'].fetch()
                            sf.util.write_json(config, config_path)

                        shutil.copy(config_path, params_dest)
                        shutil.copy(
                            join(dirname(model_path), 'slide_manifest.csv'),
                            join(model_path, 'slide_manifest.csv')
                        )
                    except Exception as e:
                        log.warning(e)
                        log.warning('Unable to copy params.json/slide_manifest'
                                    '.csv files into model folder.')

            if self.cb_args.using_validation:
                self.evaluate_model(logs)
        elif self.early_stop:
            self.evaluate_model(logs)
        self.model.stop_training = self.early_stop

    def on_train_batch_end(self, batch: int, logs={}) -> None:
        # Tensorboard logging for training metrics
        if batch > 0 and batch % self.cb_args.log_frequency == 0:
            #with self.train_summary_writer.as_default():
            self._log_training_metrics(logs)

        # Check if manual early stopping has been triggered
        if (self.hp.early_stop
           and self.hp.early_stop_method == 'manual'):

            assert self.hp.manual_early_stop_batch is not None
            assert self.hp.manual_early_stop_epoch is not None

            if (self.hp.manual_early_stop_epoch <= (self.epoch_count+1)
               and self.hp.manual_early_stop_batch <= batch):

                log.info('Manual early stop triggered: epoch '
                         f'{self.epoch_count+1}, batch {batch}')
                self.model.stop_training = True
                self.early_stop = True
                self.early_stop_batch = batch
                self.early_stop_epoch = self.epoch_count + 1

        # Validation metrics
        if (self.cb_args.using_validation and self.cb_args.validate_on_batch
           and (batch > 0)
           and (batch % self.cb_args.validate_on_batch == 0)):
            _, acc, loss = eval_from_model(
                self.model,
                self.cb_args.mid_train_validation_data,
                model_type=self.hp.model_type(),
                uq=False,
                loss=self.hp.get_loss(),
                steps=self.cb_args.validation_steps,
                verbosity='quiet',
            )
            val_metrics = {'loss': loss}
            val_log_metrics = {'loss': loss}
            if isinstance(acc, float):
                val_metrics['accuracy'] = acc
                val_log_metrics['accuracy'] = acc
            elif acc is not None:
                val_metrics.update({f'accuracy-{i+1}': acc[i] for i in range(len(acc))})
                val_log_metrics.update({f'out-{i}_accuracy': acc[i] for i in range(len(acc))})

            val_loss = val_metrics['loss']
            self.model.stop_training = False
            if (self.hp.early_stop_method == 'accuracy'
               and 'accuracy' in val_metrics):
                early_stop_value = val_metrics['accuracy']
                val_acc = f"{val_metrics['accuracy']:.3f}"
            else:
                early_stop_value = val_loss
                val_acc = ', '.join([
                    f'{val_metrics[v]:.3f}'
                    for v in val_metrics
                    if 'accuracy' in v
                ])
            if 'accuracy' in logs:
                train_acc = f"{logs['accuracy']:.3f}"
            else:
                train_acc = ', '.join([
                    f'{logs[v]:.3f}'
                    for v in logs
                    if 'accuracy' in v
                ])
            if sf.getLoggingLevel() <= 20:
                print('\r\033[K', end='')
            self.moving_average += [early_stop_value]

            self._log_validation_metrics(val_log_metrics)
            # Log training metrics if not already logged this batch
            if batch % self.cb_args.log_frequency > 0:
                self._log_training_metrics(logs)

            # Base logging message
            batch_msg = f'[blue]Batch {batch:<5}[/]'
            loss_msg = f"[green]loss[/]: {logs['loss']:.3f}"
            val_loss_msg = f"[magenta]val_loss[/]: {val_loss:.3f}"
            if self.model_type == 'categorical':
                acc_msg = f"[green]acc[/]: {train_acc}"
                val_acc_msg = f"[magenta]val_acc[/]: {val_acc}"
                log_message = f"{batch_msg} {loss_msg}, {acc_msg} | "
                log_message += f"{val_loss_msg}, {val_acc_msg}"
            else:
                log_message = f"{batch_msg} {loss_msg} | {val_loss_msg}"

            # Calculate exponential moving average of validation accuracy
            if len(self.moving_average) <= self.cb_args.ema_observations:
                log.info(log_message)
            else:
                # Only keep track of the last [ema_observations] val accuracies
                self.moving_average.pop(0)
                if self.last_ema == -1:
                    # Calculate simple moving average
                    self.last_ema = (sum(self.moving_average)
                                     / len(self.moving_average))
                    log.info(log_message + f' (SMA: {self.last_ema:.3f})')
                else:
                    # Update exponential moving average
                    sm = self.cb_args.ema_smoothing
                    obs = self.cb_args.ema_observations
                    self.last_ema = ((early_stop_value * (sm / (1 + obs)))
                                     + (self.last_ema * (1 - (sm / (1 + obs)))))
                    log.info(log_message + f' (EMA: {self.last_ema:.3f})')

            # If early stopping and our patience criteria has been met,
            #   check if validation accuracy is still improving
            steps_per_epoch = self.cb_args.steps_per_epoch
            if (self.hp.early_stop
               and self.hp.early_stop_method in ('loss', 'accuracy')
               and self.last_ema != -1
               and ((float(batch) / steps_per_epoch) + self.epoch_count)
                    > self.hp.early_stop_patience):

                if (self.ema_two_checks_prior != -1
                    and ((self.hp.early_stop_method == 'accuracy'
                          and self.last_ema <= self.ema_two_checks_prior)
                         or (self.hp.early_stop_method == 'loss'
                             and self.last_ema >= self.ema_two_checks_prior))):

                    log.info(f'Early stop: epoch {self.epoch_count+1}, batch '
                             f'{batch}')
                    self.model.stop_training = True
                    self.early_stop = True
                    self.early_stop_batch = batch
                    self.early_stop_epoch = self.epoch_count + 1

                    # Log early stop to neptune
                    if self.neptune_run:
                        self.neptune_run["early_stop/early_stop_epoch"] = self.epoch_count
                        self.neptune_run["early_stop/early_stop_batch"] = batch
                        self.neptune_run["early_stop/method"] = self.hp.early_stop_method
                        self.neptune_run["early_stop/stopped_early"] = self.early_stop
                        self.neptune_run["sys/tags"].add("early_stopped")
                else:
                    self.ema_two_checks_prior = self.ema_one_check_prior
                    self.ema_one_check_prior = self.last_ema

        # Update global step (for tracking metrics across epochs)
        self.global_step += 1

    def on_train_end(self, logs={}) -> None:
        if sf.getLoggingLevel() <= 20:
            print('\r\033[K')
        if self.neptune_run:
            self.neptune_run['sys/tags'].add('training_complete')

    def evaluate_model(self, logs={}) -> None:
        log.debug("Evaluating model from evaluation callback")
        epoch = self.epoch_count
        metrics, acc, loss = self._metrics_from_dataset(f'val_epoch{epoch}')

        # Note that Keras loss during training includes regularization losses,
        # so this loss will not match validation loss calculated during training
        val_metrics = {'accuracy': acc, 'loss': loss}
        log.info('Validation metrics: ' + json.dumps(val_metrics, indent=4))
        self.results['epochs'][f'epoch{epoch}'] = {
            'train_metrics': {k: v for k, v in logs.items() if k[:3] != 'val'},
            'val_metrics': val_metrics
        }
        if self.early_stop:
            self.results['epochs'][f'epoch{epoch}'].update({
                'early_stop_epoch': self.early_stop_epoch,
                'early_stop_batch': self.early_stop_batch,
            })
        for m in metrics:
            if metrics[m]['tile'] is None:
                continue
            self.results['epochs'][f'epoch{epoch}'][f'tile_{m}'] = metrics[m]['tile']
            self.results['epochs'][f'epoch{epoch}'][f'slide_{m}'] = metrics[m]['slide']
            self.results['epochs'][f'epoch{epoch}'][f'patient_{m}'] = metrics[m]['patient']

        epoch_results = self.results['epochs'][f'epoch{epoch}']
        self._log_epoch_evaluation(
            epoch_results, metrics=metrics, accuracy=acc, loss=loss, logs=logs
        )


class Trainer:
    """Base trainer class containing functionality for model building, input
    processing, training, and evaluation.

    This base class requires categorical outcome(s). Additional outcome types
    are supported by :class:`slideflow.model.LinearTrainer` and
    :class:`slideflow.model.CPHTrainer`.

    Slide-level (e.g. clinical) features can be used as additional model input
    by providing slide labels in the slide annotations dictionary, under the
    key 'input'.
    """

    _model_type = 'categorical'

    def __init__(
        self,
        hp: ModelParams,
        outdir: str,
        labels: Dict[str, Any],
        *,
        slide_input: Optional[Dict[str, Any]] = None,
        name: str = 'Trainer',
        feature_sizes: Optional[List[int]] = None,
        feature_names: Optional[List[str]] = None,
        outcome_names: Optional[List[str]] = None,
        mixed_precision: bool = True,
        allow_tf32: bool = False,
        config: Dict[str, Any] = None,
        use_neptune: bool = False,
        neptune_api: Optional[str] = None,
        neptune_workspace: Optional[str] = None,
        load_method: str = 'weights',
        custom_objects: Optional[Dict[str, Any]] = None,
        transform: Optional[Union[Callable, Dict[str, Callable]]] = None,
    ) -> None:

        """Sets base configuration, preparing model inputs and outputs.

        Args:
            hp (:class:`slideflow.ModelParams`): ModelParams object.
            outdir (str): Path for event logs and checkpoints.
            labels (dict): Dict mapping slide names to outcome labels (int or
                float format).
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
            load_method (str): Either 'full' or 'weights'. Method to use
                when loading a Tensorflow model. If 'full', loads the model with
                ``tf.keras.models.load_model()``. If 'weights', will read the
                ``params.json`` configuration file, build the model architecture,
                and then load weights from the given model with
                ``Model.load_weights()``. Loading with 'full' may improve
                compatibility across Slideflow versions. Loading with 'weights'
                may improve compatibility across hardware & environments.
            config (dict, optional): Training configuration dictionary, used
                for logging and image format verification. Defaults to None.
            use_neptune (bool, optional): Use Neptune API logging.
                Defaults to False
            neptune_api (str, optional): Neptune API token, used for logging.
                Defaults to None.
            neptune_workspace (str, optional): Neptune workspace.
                Defaults to None.
            custom_objects (dict, Optional): Dictionary mapping names
                (strings) to custom classes or functions. Defaults to None.
            transform (callable or dict, optional): Optional transform to
                apply to input images. If dict, must have the keys 'train'
                and/or 'val', mapping to callables that takes a single
                image Tensor as input and returns a single image Tensor.
                If None, no transform is applied. If a single callable is
                provided, it will be applied to both training and validation
                data. If a dict is provided, the 'train' transform will be
                applied to training data and the 'val' transform will be
                applied to validation data. If a dict is provided and either
                'train' or 'val' is None, no transform will be applied to
                that data. Defaults to None.
        """

        if load_method not in ('full', 'weights'):
            raise ValueError("Unrecognized value for load_method, must be "
                             "either 'full' or 'weights'.")

        self.outdir = outdir
        self.tile_px = hp.tile_px
        self.labels = labels
        self.hp = hp
        self.slides = list(labels.keys())
        self.slide_input = slide_input
        self.feature_names = feature_names
        self.feature_sizes = feature_sizes
        self.num_slide_features = 0 if not feature_sizes else sum(feature_sizes)
        self.mixed_precision = mixed_precision
        self._allow_tf32 = allow_tf32
        self.name = name
        self.neptune_run = None
        self.annotations_tables = []
        self.eval_callback = _PredictionAndEvaluationCallback  # type: tf.keras.callbacks.Callback
        self.load_method = load_method
        self.custom_objects = custom_objects
        self.patients = dict()

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Format outcome labels (ensures compatibility with single
        # and multi-outcome models)
        self._process_outcome_labels(outcome_names)
        self._setup_inputs()

        # Normalization setup
        self.normalizer = self.hp.get_normalizer()
        if self.normalizer:
            log.info(f'Using realtime {self.hp.normalizer} normalization')

        # Mixed precision and Tensorfloat-32
        if self.mixed_precision:
            _policy = 'mixed_float16'
            log.debug(f'Enabling mixed precision ({_policy})')
            if version.parse(tf.__version__) > version.parse("2.8"):
                tf.keras.mixed_precision.set_global_policy(_policy)
            else:
                policy = tf.keras.mixed_precision.experimental.Policy(_policy)
                tf.keras.mixed_precision.experimental.set_policy(policy)
        tf.config.experimental.enable_tensor_float_32_execution(allow_tf32)

        # Custom transforms
        self._process_transforms(transform)

        # Log parameters
        if config is None:
            config = {
                'slideflow_version': sf.__version__,
                'backend': sf.backend(),
                'git_commit': sf.__gitcommit__,
                'model_name': self.name,
                'full_model_name': self.name,
                'outcomes': self.outcome_names,
                'model_type': self.hp.model_type(),
                'img_format': None,
                'tile_px': self.hp.tile_px,
                'tile_um': self.hp.tile_um,
                'input_features': None,
                'input_feature_sizes': None,
                'input_feature_labels': None,
                'hp': self.hp.to_dict()
            }
        sf.util.write_json(config, join(self.outdir, 'params.json'))
        self.config = config
        self.img_format = config['img_format'] if 'img_format' in config else None

        # Initialize Neptune
        self.use_neptune = use_neptune
        if self.use_neptune:
            if neptune_api is None or neptune_workspace is None:
                raise ValueError("If using Neptune, must supply values "
                                 "neptune_api and neptune_workspace.")
            self.neptune_logger = sf.util.neptune_utils.NeptuneLog(
                neptune_api,
                neptune_workspace
            )

    def _process_outcome_labels(self, outcome_names: Optional[List[str]]) -> None:
        outcome_labels = np.array(list(self.labels.values()))
        if len(outcome_labels.shape) == 1:
            outcome_labels = np.expand_dims(outcome_labels, axis=1)
        if not outcome_names:
            outcome_names = [
                f'Outcome {i}'
                for i in range(outcome_labels.shape[1])
            ]
        outcome_names = sf.util.as_list(outcome_names)
        if self.labels and (len(outcome_names) != outcome_labels.shape[1]):
            num_names = len(outcome_names)
            num_outcomes = outcome_labels.shape[1]
            raise errors.ModelError(f'Size of outcome_names ({num_names}) != '
                                    f'number of outcomes {num_outcomes}')
        self.outcome_names = outcome_names

        if self.labels:
            self.num_classes = self.hp._detect_classes_from_labels(self.labels)
            with tf.device('/cpu'):
                for oi in range(outcome_labels.shape[1]):
                    self.annotations_tables += [tf.lookup.StaticHashTable(
                        tf.lookup.KeyValueTensorInitializer(
                            self.slides,
                            outcome_labels[:, oi]
                        ), -1
                    )]
        else:
            self.num_classes = None  # type: ignore

    def _process_transforms(
        self,
        transform: Optional[Union[Callable, Dict[str, Callable]]] = None
    ) -> None:
        """Process custom transformations for training and/or validation."""
        if not isinstance(transform, dict):
            transform = {'train': transform, 'val': transform}
        if any([t not in ('train', 'val') for t in transform]):
            raise ValueError("transform must be a callable or dict with keys "
                             "'train' and/or 'val'")
        if 'train' not in transform:
            transform['train'] = None
        if 'val' not in transform:
            transform['val'] = None
        self.transform = transform

    def _setup_inputs(self) -> None:
        """Setup slide-level input."""
        if self.num_slide_features:
            assert self.slide_input is not None
            try:
                if self.num_slide_features:
                    log.info(f'Training with both images and '
                             f'{self.num_slide_features} slide-level input'
                             'features')
            except KeyError:
                raise errors.ModelError("Unable to find slide-level input at "
                                        "'input' key in annotations")
            for slide in self.slides:
                if len(self.slide_input[slide]) != self.num_slide_features:
                    num_in_feature_table = len(self.slide_input[slide])
                    raise errors.ModelError(
                        f'Length of input for slide {slide} does not match '
                        f'feature_sizes; expected {self.num_slide_features}, '
                        f'got {num_in_feature_table}'
                    )

    def _compile_model(self) -> None:
        """Compile keras model."""
        self.model.compile(
            optimizer=self.hp.get_opt(),
            loss=self.hp.get_loss(),
            metrics=['accuracy']
        )

    def _fit_normalizer(self, norm_fit: Optional[NormFit]) -> None:
        """Fit the Trainer normalizer using the specified fit, if applicable.

        Args:
            norm_fit (Optional[Dict[str, np.ndarray]]): Normalizer fit.
        """
        if norm_fit is not None and not self.normalizer:
            raise ValueError("norm_fit supplied, but model params do not"
                             "specify a normalizer.")
        if self.normalizer and norm_fit is not None:
            self.normalizer.set_fit(**norm_fit)  # type: ignore
        elif (self.normalizer
              and 'norm_fit' in self.config
              and self.config['norm_fit'] is not None):
            log.debug("Detecting normalizer fit from model config")
            self.normalizer.set_fit(**self.config['norm_fit'])

    def _parse_tfrecord_labels(
        self,
        image: tf.Tensor,
        slide: tf.Tensor
    ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        """Parses raw entry read from TFRecord."""

        image_dict = {'tile_image': image}

        if self.num_classes is None:
            label = None
        elif len(self.num_classes) > 1:  # type: ignore
            label = {
                f'out-{oi}': self.annotations_tables[oi].lookup(slide)
                for oi in range(len(self.num_classes))  # type: ignore
            }
        else:
            label = self.annotations_tables[0].lookup(slide)

        # Add additional non-image feature inputs if indicated,
        #     excluding the event feature used for CPH models
        if self.num_slide_features:

            def slide_lookup(s):
                return self.slide_input[s.numpy().decode('utf-8')]

            num_features = self.num_slide_features
            slide_feature_input_val = tf.py_function(
                func=slide_lookup,
                inp=[slide],
                Tout=[tf.float32] * num_features
            )
            image_dict.update({'slide_feature_input': slide_feature_input_val})

        return image_dict, label

    def _retrain_top_layers(
        self,
        train_data: tf.data.Dataset,
        steps_per_epoch: int,
        callbacks: tf.keras.callbacks.Callback = None,
        epochs: int = 1
    ) -> Dict:
        """Retrain only the top layer, leaving all other layers frozen."""
        log.info('Retraining top layer')
        # Freeze the base layer
        self.model.layers[0].trainable = False
        #val_steps = 200 if validation_data else None
        self._compile_model()

        toplayer_model = self.model.fit(
            train_data,
            epochs=epochs,
            verbose=(sf.getLoggingLevel() <= 20),
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks
        )
        # Unfreeze the base layer
        self.model.layers[0].trainable = True
        return toplayer_model.history

    def _detect_patients(self, *args):
        self.patients = dict()
        for dataset in args:
            if dataset is None:
                continue
            dataset_patients = dataset.patients()
            if not dataset_patients:
                self.patients.update({s: s for s in self.slides})
            else:
                self.patients.update(dataset_patients)

    def _interleave_kwargs(self, **kwargs) -> Dict[str, Any]:
        args = SimpleNamespace(
            labels=self._parse_tfrecord_labels,
            normalizer=self.normalizer,
            **kwargs
        )
        return vars(args)

    def _interleave_kwargs_val(self, **kwargs) -> Dict[str, Any]:
        return self._interleave_kwargs(**kwargs)

    def _metric_kwargs(self, **kwargs) -> Dict[str, Any]:
        args = SimpleNamespace(
            model=self.model,
            model_type=self._model_type,
            patients=self.patients,
            outcome_names=self.outcome_names,
            data_dir=self.outdir,
            neptune_run=self.neptune_run,
            **kwargs
        )
        return vars(args)

    def _verify_img_format(self, dataset, *datasets: Optional["sf.Dataset"]) -> str:
        """Verify that the image format of the dataset matches the model config.

        Args:
            dataset (sf.Dataset): Dataset to check.
            *datasets (sf.Dataset): Additional datasets to check. May be None.

        Returns:
            str: Image format, either 'png' or 'jpg', if a consistent image
                format was found, otherwise None.

        """
        # First, verify all datasets have the same image format
        img_formats = set([d.img_format for d in datasets if d])
        if len(img_formats) > 1:
            log.error("Multiple image formats detected: {}.".format(
                ', '.join(img_formats)
            ))
            return None
        elif self.img_format and not dataset.img_format:
            log.warning("Unable to verify image format (PNG/JPG) of dataset.")
            return None
        elif self.img_format and dataset.img_format != self.img_format:
            log.error(
                "Mismatched image formats. Expected '{}' per model config, "
                "but dataset has format '{}'.".format(
                    self.img_format,
                    dataset.img_format))
            return None
        else:
            return dataset.img_format

    def load(self, model: str, **kwargs) -> tf.keras.Model:
        self.model = load(
            model,
            method=self.load_method,
            custom_objects=self.custom_objects,
            **kwargs
        )

    def predict(
        self,
        dataset: "sf.Dataset",
        batch_size: Optional[int] = None,
        norm_fit: Optional[NormFit] = None,
        format: str = 'parquet',
        from_wsi: bool = False,
        roi_method: str = 'auto',
        reduce_method: Union[str, Callable] = 'average',
    ) -> Dict[str, "pd.DataFrame"]:
        """Perform inference on a model, saving tile-level predictions.

        Args:
            dataset (:class:`slideflow.dataset.Dataset`): Dataset containing
                TFRecords to evaluate.
            batch_size (int, optional): Evaluation batch size. Defaults to the
                same as training (per self.hp)
            norm_fit (Dict[str, np.ndarray]): Normalizer fit, mapping fit
                parameters (e.g. target_means, target_stds) to values
                (np.ndarray). If not provided, will fit normalizer using
                model params (if applicable). Defaults to None.
            format (str, optional): Format in which to save predictions. Either
                'csv', 'feather', or 'parquet'. Defaults to 'parquet'.
            from_wsi (bool): Generate predictions from tiles dynamically
                extracted from whole-slide images, rather than TFRecords.
                Defaults to False (use TFRecords).
            roi_method (str): ROI method to use if from_wsi=True (ignored if
                from_wsi=False).  Either 'inside', 'outside', 'auto', 'ignore'.
                If 'inside' or 'outside', will extract tiles in/out of an ROI,
                and raise errors.MissingROIError if an ROI is not available.
                If 'auto', will extract tiles inside an ROI if available,
                and across the whole-slide if no ROI is found.
                If 'ignore', will extract tiles across the whole-slide
                regardless of whether an ROI is available.
                Defaults to 'auto'.
            reduce_method (str, optional): Reduction method for calculating
                slide-level and patient-level predictions for categorical
                outcomes. Options include 'average', 'mean', 'proportion',
                'median', 'sum', 'min', 'max', or a callable function.
                'average' and 'mean' are  synonymous, with both options kept
                for backwards compatibility. If  'average' or 'mean', will
                reduce with average of each logit across  tiles. If
                'proportion', will convert tile predictions into onehot encoding
                then reduce by averaging these onehot values. For all other
                values, will reduce with the specified function, applied via
                the pandas ``DataFrame.agg()`` function. Defaults to 'average'.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary with keys 'tile', 'slide', and
            'patient', and values containing DataFrames with tile-, slide-,
            and patient-level predictions.
        """

        if format not in ('csv', 'feather', 'parquet'):
            raise ValueError(f"Unrecognized format {format}")

        self._detect_patients(dataset)

        # Verify image format
        self._verify_img_format(dataset)

        # Fit normalizer
        self._fit_normalizer(norm_fit)

        # Load and initialize model
        if not self.model:
            raise errors.ModelNotLoadedError
        log_manifest(
            None,
            dataset.tfrecords(),
            labels=self.labels,
            filename=join(self.outdir, 'slide_manifest.csv')
        )
        if not batch_size:
            batch_size = self.hp.batch_size
        with tf.name_scope('input'):
            interleave_kwargs = self._interleave_kwargs_val(
                batch_size=batch_size,
                infinite=False,
                transform=self.transform['val'],
                augment=False
            )
            tf_dts_w_slidenames = dataset.tensorflow(
                incl_loc=True,
                incl_slidenames=True,
                from_wsi=from_wsi,
                roi_method=roi_method,
                **interleave_kwargs
            )
        # Generate predictions
        log.info('Generating predictions...')
        dfs = sf.stats.predict_dataset(
            model=self.model,
            dataset=tf_dts_w_slidenames,
            model_type=self._model_type,
            uq=bool(self.hp.uq),
            num_tiles=dataset.num_tiles,
            outcome_names=self.outcome_names,
            patients=self.patients,
            reduce_method=reduce_method,
        )
        # Save predictions
        sf.stats.metrics.save_dfs(dfs, format=format, outdir=self.outdir)
        return dfs

    def evaluate(
        self,
        dataset: "sf.Dataset",
        batch_size: Optional[int] = None,
        save_predictions: Union[bool, str] = 'parquet',
        reduce_method: Union[str, Callable] = 'average',
        norm_fit: Optional[NormFit] = None,
        uq: Union[bool, str] = 'auto',
        from_wsi: bool = False,
        roi_method: str = 'auto',
    ) -> Dict[str, Any]:
        """Evaluate model, saving metrics and predictions.

        Args:
            dataset (:class:`slideflow.dataset.Dataset`): Dataset containing
                TFRecords to evaluate.
            batch_size (int, optional): Evaluation batch size. Defaults to the
                same as training (per self.hp)
            save_predictions (bool or str, optional): Save tile, slide, and
                patient-level predictions at each evaluation. May be 'csv',
                'feather', or 'parquet'. If False, will not save predictions.
                Defaults to 'parquet'.
            reduce_method (str, optional): Reduction method for calculating
                slide-level and patient-level predictions for categorical
                outcomes. Options include 'average', 'mean', 'proportion',
                'median', 'sum', 'min', 'max', or a callable function.
                'average' and 'mean' are  synonymous, with both options kept
                for backwards compatibility. If  'average' or 'mean', will
                reduce with average of each logit across  tiles. If
                'proportion', will convert tile predictions into onehot encoding
                then reduce by averaging these onehot values. For all other
                values, will reduce with the specified function, applied via
                the pandas ``DataFrame.agg()`` function. Defaults to 'average'.
            norm_fit (Dict[str, np.ndarray]): Normalizer fit, mapping fit
                parameters (e.g. target_means, target_stds) to values
                (np.ndarray). If not provided, will fit normalizer using
                model params (if applicable). Defaults to None.
            uq (bool or str, optional): Enable UQ estimation (for
                applicable models). Defaults to 'auto'.

        Returns:
            Dictionary of evaluation metrics.
        """
        if uq != 'auto':
            if not isinstance(uq, bool):
                raise ValueError(f"Unrecognized value {uq} for uq")
            self.hp.uq = uq

        self._detect_patients(dataset)

        # Verify image format
        self._verify_img_format(dataset)

        # Perform evaluation
        _unit_type = 'slides' if from_wsi else 'tfrecords'
        log.info(f'Evaluating {len(dataset.tfrecords())} {_unit_type}')

        # Fit normalizer
        self._fit_normalizer(norm_fit)

        # Load and initialize model
        if not self.model:
            raise errors.ModelNotLoadedError
        log_manifest(
            None,
            dataset.tfrecords(),
            labels=self.labels,
            filename=join(self.outdir, 'slide_manifest.csv')
        )
        # Neptune logging
        if self.use_neptune:
            assert self.neptune_run is not None
            self.neptune_run = self.neptune_logger.start_run(
                self.name,
                self.config['project'],
                dataset,
                tags=['eval']
            )
            self.neptune_logger.log_config(self.config, 'eval')
            self.neptune_run['data/slide_manifest'].upload(
                join(self.outdir, 'slide_manifest.csv')
            )

        if not batch_size:
            batch_size = self.hp.batch_size
        with tf.name_scope('input'):
            interleave_kwargs = self._interleave_kwargs_val(
                batch_size=batch_size,
                infinite=False,
                transform=self.transform['val'],
                augment=False
            )
            tf_dts_w_slidenames = dataset.tensorflow(
                incl_slidenames=True,
                incl_loc=True,
                from_wsi=from_wsi,
                roi_method=roi_method,
                **interleave_kwargs
            )
        # Generate performance metrics
        log.info('Calculating performance metrics...')
        metric_kwargs = self._metric_kwargs(
            dataset=tf_dts_w_slidenames,
            num_tiles=dataset.num_tiles,
            label='eval'
        )
        metrics, acc, loss = sf.stats.metrics_from_dataset(
            save_predictions=save_predictions,
            reduce_method=reduce_method,
            loss=self.hp.get_loss(),
            uq=bool(self.hp.uq),
            **metric_kwargs
        )
        results = {'eval': {}}  # type: Dict[str, Dict[str, float]]
        for metric in metrics:
            if metrics[metric]:
                log.info(f"Tile {metric}: {metrics[metric]['tile']}")
                log.info(f"Slide {metric}: {metrics[metric]['slide']}")
                log.info(f"Patient {metric}: {metrics[metric]['patient']}")
                results['eval'].update({
                    f'tile_{metric}': metrics[metric]['tile'],
                    f'slide_{metric}': metrics[metric]['slide'],
                    f'patient_{metric}': metrics[metric]['patient']
                })

        # Note that Keras loss during training includes regularization losses,
        # so this loss will not match validation loss calculated during training
        val_metrics = {'accuracy': acc, 'loss': loss}
        results_log = os.path.join(self.outdir, 'results_log.csv')
        log.info('Evaluation metrics:')
        for m in val_metrics:
            log.info(f'{m}: {val_metrics[m]}')
        results['eval'].update(val_metrics)
        sf.util.update_results_log(results_log, 'eval_model', results)

        # Update neptune log
        if self.neptune_run:
            self.neptune_run['eval/results'] = val_metrics
            self.neptune_run.stop()

        return results

    def train(
        self,
        train_dts: "sf.Dataset",
        val_dts: Optional["sf.Dataset"],
        log_frequency: int = 100,
        validate_on_batch: int = 0,
        validation_batch_size: int = None,
        validation_steps: int = 200,
        starting_epoch: int = 0,
        ema_observations: int = 20,
        ema_smoothing: int = 2,
        use_tensorboard: bool = True,
        steps_per_epoch_override: int = 0,
        save_predictions: Union[bool, str] = 'parquet',
        save_model: bool = True,
        resume_training: Optional[str] = None,
        pretrain: Optional[str] = 'imagenet',
        checkpoint: Optional[str] = None,
        save_checkpoints: bool = True,
        multi_gpu: bool = False,
        reduce_method: Union[str, Callable] = 'average',
        norm_fit: Optional[NormFit] = None,
        from_wsi: bool = False,
        roi_method: str = 'auto',
    ) -> Dict[str, Any]:
        """Builds and trains a model from hyperparameters.

        Args:
            train_dts (:class:`slideflow.Dataset`): Training dataset. Will call
                the `.tensorflow()` method to retrieve the tf.data.Dataset
                used for model fitting.
            val_dts (:class:`slideflow.Dataset`): Validation dataset. Will call
                the `.tensorflow()` method to retrieve the tf.data.Dataset
                used for model fitting.
            log_frequency (int, optional): How frequent to update Tensorboard
                logs, in batches. Defaults to 100.
            validate_on_batch (int, optional): Validation will also be performed
                every N batches. Defaults to 0.
            validation_batch_size (int, optional): Validation batch size.
                Defaults to same as training (per self.hp).
            validation_steps (int, optional): Number of batches to use for each
                instance of validation. Defaults to 200.
            starting_epoch (int, optional): Starts training at the specified
                epoch. Defaults to 0.
            ema_observations (int, optional): Number of observations over which
                to perform exponential moving average smoothing. Defaults to 20.
            ema_smoothing (int, optional): Exponential average smoothing value.
                Defaults to 2.
            use_tensoboard (bool, optional): Enable tensorboard callbacks.
                Defaults to False.
            steps_per_epoch_override (int, optional): Manually set the number
                of steps per epoch. Defaults to 0 (automatic).
            save_predictions (bool or str, optional): Save tile, slide, and
                patient-level predictions at each evaluation. May be 'csv',
                'feather', or 'parquet'. If False, will not save predictions.
                Defaults to 'parquet'.
            save_model (bool, optional): Save models when evaluating at
                specified epochs. Defaults to True.
            resume_training (str, optional): Path to model to continue training.
                Only valid in Tensorflow backend. Defaults to None.
            pretrain (str, optional): Either 'imagenet' or path to Tensorflow
                model from which to load weights. Defaults to 'imagenet'.
            checkpoint (str, optional): Path to cp.ckpt from which to load
                weights. Defaults to None.
            save_checkpoint (bool, optional): Save checkpoints at each epoch.
                Defaults to True.
            multi_gpu (bool, optional): Enable multi-GPU training using
                Tensorflow/Keras MirroredStrategy.
            reduce_method (str, optional): Reduction method for calculating
                slide-level and patient-level predictions for categorical
                outcomes. Options include 'average', 'mean', 'proportion',
                'median', 'sum', 'min', 'max', or a callable function.
                'average' and 'mean' are  synonymous, with both options kept
                for backwards compatibility. If  'average' or 'mean', will
                reduce with average of each logit across  tiles. If
                'proportion', will convert tile predictions into onehot encoding
                then reduce by averaging these onehot values. For all other
                values, will reduce with the specified function, applied via
                the pandas ``DataFrame.agg()`` function. Defaults to 'average'.
            norm_fit (Dict[str, np.ndarray]): Normalizer fit, mapping fit
                parameters (e.g. target_means, target_stds) to values
                (np.ndarray). If not provided, will fit normalizer using
                model params (if applicable). Defaults to None.

        Returns:
            dict: Nested results dict with metrics for each evaluated epoch.
        """

        if self.hp.model_type() != self._model_type:
            hp_model = self.hp.model_type()
            raise errors.ModelError(f"Incompatible models: {hp_model} (hp) and "
                                    f"{self._model_type} (model)")

        self._detect_patients(train_dts, val_dts)

        # Verify image format across datasets.
        img_format = self._verify_img_format(train_dts, val_dts)
        if img_format and self.config['img_format'] is None:
            self.config['img_format'] = img_format
            sf.util.write_json(self.config, join(self.outdir, 'params.json'))

        # Clear prior Tensorflow graph to free memory
        tf.keras.backend.clear_session()
        results_log = os.path.join(self.outdir, 'results_log.csv')

        # Fit the normalizer to the training data and log the source mean/stddev
        if self.normalizer and self.hp.normalizer_source == 'dataset':
            self.normalizer.fit(train_dts)
        else:
            self._fit_normalizer(norm_fit)

        if self.normalizer:
            config_path = join(self.outdir, 'params.json')
            if not exists(config_path):
                config = {
                    'slideflow_version': sf.__version__,
                    'hp': self.hp.to_dict(),
                    'backend': sf.backend()
                }
            else:
                config = sf.util.load_json(config_path)
            config['norm_fit'] = self.normalizer.get_fit(as_list=True)
            sf.util.write_json(config, config_path)

        # Prepare multiprocessing pool if from_wsi=True
        if from_wsi:
            pool = mp.Pool(
                sf.util.num_cpu(default=8),
                initializer=sf.util.set_ignore_sigint
            )
        else:
            pool = None

        # Save training / validation manifest
        if val_dts is None:
            val_paths = None
        elif from_wsi:
            val_paths = val_dts.slide_paths()
        else:
            val_paths = val_dts.tfrecords()
        log_manifest(
            train_dts.tfrecords(),
            val_paths,
            labels=self.labels,
            filename=join(self.outdir, 'slide_manifest.csv')
        )

        # Neptune logging
        if self.use_neptune:
            tags = ['train']
            if 'k-fold' in self.config['validation_strategy']:
                tags += [f'k-fold{self.config["k_fold_i"]}']
            self.neptune_run = self.neptune_logger.start_run(
                self.name,
                self.config['project'],
                train_dts,
                tags=tags
            )
            self.neptune_logger.log_config(self.config, 'train')
            self.neptune_run['data/slide_manifest'].upload(  # type: ignore
                os.path.join(self.outdir, 'slide_manifest.csv')
            )

        # Set up multi-GPU strategy
        if multi_gpu:
            strategy = tf.distribute.MirroredStrategy()
            log.info('Multi-GPU training with '
                     f'{strategy.num_replicas_in_sync} devices')
            # Fixes "OSError: [Errno 9] Bad file descriptor" after training
            atexit.register(strategy._extended._collective_ops._pool.close)
        else:
            strategy = None

        with strategy.scope() if strategy else no_scope():
            # Build model from ModelParams
            if resume_training:
                self.model = load(resume_training, method='weights', training=True)
            else:
                model = self.hp.build_model(
                    labels=self.labels,
                    num_slide_features=self.num_slide_features,
                    pretrain=pretrain,
                    checkpoint=checkpoint,
                    load_method=self.load_method
                )
                self.model = model
                tf_utils.log_summary(model, self.neptune_run)

            with tf.name_scope('input'):
                t_kwargs = self._interleave_kwargs(
                    batch_size=self.hp.batch_size,
                    infinite=True,
                    augment=self.hp.augment,
                    transform=self.transform['train'],
                    from_wsi=from_wsi,
                    pool=pool,
                    roi_method=roi_method
                )
                train_data = train_dts.tensorflow(drop_last=True, **t_kwargs)
                log.debug(f"Training: {train_dts.num_tiles} total tiles.")

            # Set up validation data
            using_validation = (val_dts
                                and (len(val_dts.tfrecords()) if not from_wsi
                                     else len(val_dts.slide_paths())))
            if using_validation:
                assert val_dts is not None
                with tf.name_scope('input'):
                    if not validation_batch_size:
                        validation_batch_size = self.hp.batch_size
                    v_kwargs = self._interleave_kwargs_val(
                        batch_size=validation_batch_size,
                        infinite=False,
                        augment=False,
                        transform=self.transform['val'],
                        from_wsi=from_wsi,
                        pool=pool,
                        roi_method=roi_method
                    )
                    validation_data = val_dts.tensorflow(
                        incl_slidenames=True,
                        incl_loc=True,
                        drop_last=True,
                        **v_kwargs
                    )
                    log.debug(f"Validation: {val_dts.num_tiles} total tiles.")
                if validate_on_batch:
                    log.debug('Validation during training: every '
                              f'{validate_on_batch} steps and at epoch end')
                    mid_v_kwargs = v_kwargs.copy()
                    mid_v_kwargs['infinite'] = True
                    mid_train_validation_data = iter(val_dts.tensorflow(
                        incl_slidenames=True,
                        incl_loc=True,
                        drop_last=True,
                        **mid_v_kwargs
                    ))
                else:
                    log.debug('Validation during training: at epoch end')
                    mid_train_validation_data = None
                if validation_steps:
                    num_samples = validation_steps * self.hp.batch_size
                    log.debug(f'Using {validation_steps} batches ({num_samples}'
                              ' samples) each validation check')
                else:
                    log.debug('Using entire validation set each val check')
            else:
                log.debug('Validation during training: None')
                validation_data = None
                mid_train_validation_data = None
                validation_steps = 0

            # Calculate parameters
            if from_wsi:
                train_tiles = train_data.est_num_tiles
                val_tiles = validation_data.est_num_tiles
            else:
                train_tiles = train_dts.num_tiles
                val_tiles = 0 if val_dts is None else val_dts.num_tiles
            if max(self.hp.epochs) <= starting_epoch:
                max_epoch = max(self.hp.epochs)
                log.error(f'Starting epoch ({starting_epoch}) cannot be greater'
                          f' than max target epoch ({max_epoch})')
            if (self.hp.early_stop and self.hp.early_stop_method == 'accuracy'
               and self._model_type != 'categorical'):
                log.error("Unable to use 'accuracy' early stopping with model "
                          f"type '{self.hp.model_type()}'")
            if starting_epoch != 0:
                log.info(f'Starting training at epoch {starting_epoch}')
            if steps_per_epoch_override:
                steps_per_epoch = steps_per_epoch_override
            else:
                steps_per_epoch = round(train_tiles / self.hp.batch_size)

            cb_args = SimpleNamespace(
                starting_epoch=starting_epoch,
                using_validation=using_validation,
                validate_on_batch=validate_on_batch,
                validation_steps=validation_steps,
                ema_observations=ema_observations,
                ema_smoothing=ema_smoothing,
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_data,
                mid_train_validation_data=mid_train_validation_data,
                num_val_tiles=val_tiles,
                save_predictions=save_predictions,
                save_model=save_model,
                results_log=results_log,
                reduce_method=reduce_method,
                log_frequency=log_frequency
            )

            # Create callbacks for early stopping, checkpoint saving,
            # summaries, and history
            val_callback = self.eval_callback(self, cb_args)
            callbacks = [tf.keras.callbacks.History(), val_callback]
            if save_checkpoints:
                cp_callback = tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(self.outdir, 'cp.ckpt'),
                    save_weights_only=True,
                    verbose=(sf.getLoggingLevel() <= 20)
                )
                callbacks += [cp_callback]
            if use_tensorboard:
                log.debug(
                    "Logging with Tensorboard to {} every {} batches.".format(
                        self.outdir, log_frequency
                    ))
                tensorboard_callback = tf.keras.callbacks.TensorBoard(
                    log_dir=self.outdir,
                    histogram_freq=0,
                    write_graph=False,
                    update_freq='batch'
                )
                callbacks += [tensorboard_callback]

            # Retrain top layer only, if using transfer learning and
            # not resuming training
            total_epochs = (self.hp.toplayer_epochs
                            + (max(self.hp.epochs) - starting_epoch))
            if self.hp.toplayer_epochs:
                self._retrain_top_layers(
                    train_data,
                    steps_per_epoch,
                    callbacks=None,
                    epochs=self.hp.toplayer_epochs
                )
            self._compile_model()

            # Train the model
            log.info('Beginning training')
            try:
                self.model.fit(
                    train_data,
                    steps_per_epoch=steps_per_epoch,
                    epochs=total_epochs,
                    verbose=(sf.getLoggingLevel() <= 20),
                    initial_epoch=self.hp.toplayer_epochs,
                    callbacks=callbacks
                )
            except tf.errors.ResourceExhaustedError as e:
                log.error(f"Training failed for [bold]{self.name}[/]. "
                          f"Error: \n {e}")
            results = val_callback.results
            if self.use_neptune and self.neptune_run is not None:
                self.neptune_run['results'] = results['epochs']
                self.neptune_run.stop()

            # Cleanup
            if pool is not None:
                pool.close()
            del mid_train_validation_data

            return results


class LinearTrainer(Trainer):

    """Extends the base :class:`slideflow.model.Trainer` class to add support
    for linear/continuous outcomes. Requires that all outcomes be continuous,
    with appropriate linear loss function. Uses R-squared as the evaluation
    metric, rather than AUROC."""

    _model_type = 'linear'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _compile_model(self) -> None:
        self.model.compile(optimizer=self.hp.get_opt(),
                           loss=self.hp.get_loss(),
                           metrics=[self.hp.get_loss()])

    def _parse_tfrecord_labels(
        self,
        image: Union[Dict[str, tf.Tensor], tf.Tensor],
        slide: tf.Tensor
    ) -> Tuple[Union[Dict[str, tf.Tensor], tf.Tensor], tf.Tensor]:
        image_dict = {'tile_image': image}
        if self.num_classes is None:
            label = None
        else:
            label = [
                self.annotations_tables[oi].lookup(slide)
                for oi in range(self.num_classes)  # type: ignore
            ]

        # Add additional non-image feature inputs if indicated,
        #     excluding the event feature used for CPH models
        if self.num_slide_features:

            def slide_lookup(s):
                return self.slide_input[s.numpy().decode('utf-8')]

            num_features = self.num_slide_features
            slide_feature_input_val = tf.py_function(
                func=slide_lookup,
                inp=[slide],
                Tout=[tf.float32] * num_features
            )
            image_dict.update({'slide_feature_input': slide_feature_input_val})

        return image_dict, label


class CPHTrainer(LinearTrainer):

    """Cox Proportional Hazards model. Requires that the user provide event
    data as the first input feature, and time to outcome as the linear outcome.
    Uses concordance index as the evaluation metric."""

    _model_type = 'cph'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self.num_slide_features:
            raise errors.ModelError('Model error - CPH models must '
                                    'include event input')

    def _setup_inputs(self) -> None:
        # Setup slide-level input
        try:
            num_features = self.num_slide_features - 1
            if num_features:
                log.info(f'Training with both images and {num_features} '
                         'categories of slide-level input')
                log.info('Interpreting first feature as event for CPH model')
            else:
                log.info('Training with images alone. Interpreting first '
                         'feature as event for CPH model')
        except KeyError:
            raise errors.ModelError("Unable to find slide-level input at "
                                    "'input' key in annotations")
        assert self.slide_input is not None
        for slide in self.slides:
            if len(self.slide_input[slide]) != self.num_slide_features:
                num_in_feature_table = len(self.slide_input[slide])
                raise errors.ModelError(
                    f'Length of input for slide {slide} does not match '
                    f'feature_sizes; expected {self.num_slide_features}, got '
                    f'{num_in_feature_table}'
                )

    def load(self, model: str, **kwargs) -> tf.keras.Model:
        if self.load_method == 'full':
            custom_objects = {
                'negative_log_likelihood': tf_utils.negative_log_likelihood,
                'concordance_index': tf_utils.concordance_index
            }
            self.model = tf.keras.models.load_model(
                model,
                custom_objects=custom_objects
            )
            self.model.compile(
                loss=tf_utils.negative_log_likelihood,
                metrics=tf_utils.concordance_index
            )
        else:
            self.model = load(model, method=self.load_method, **kwargs)

    def _compile_model(self) -> None:
        self.model.compile(optimizer=self.hp.get_opt(),
                           loss=tf_utils.negative_log_likelihood,
                           metrics=tf_utils.concordance_index)

    def _parse_tfrecord_labels(
        self,
        image: Union[Dict[str, tf.Tensor], tf.Tensor],
        slide: tf.Tensor
    ) -> Tuple[Union[Dict[str, tf.Tensor], tf.Tensor], tf.Tensor]:
        image_dict = {'tile_image': image}
        if self.num_classes is None:
            label = None
        else:
            label = [
                self.annotations_tables[oi].lookup(slide)
                for oi in range(self.num_classes)  # type: ignore
            ]

        # Add additional non-image feature inputs if indicated,
        #     excluding the event feature used for CPH models
        if self.num_slide_features:
            # Time-to-event data must be added as a separate feature

            def slide_lookup(s):
                return self.slide_input[s.numpy().decode('utf-8')][1:]

            def event_lookup(s):
                return self.slide_input[s.numpy().decode('utf-8')][0]

            num_features = self.num_slide_features - 1
            event_input_val = tf.py_function(
                func=event_lookup,
                inp=[slide],
                Tout=[tf.float32]
            )
            image_dict.update({'event_input': event_input_val})
            slide_feature_input_val = tf.py_function(
                func=slide_lookup,
                inp=[slide],
                Tout=[tf.float32] * num_features
            )
            # Add slide input features, excluding the event feature
            # used for CPH models
            if not (self.num_slide_features == 1):
                image_dict.update(
                    {'slide_feature_input': slide_feature_input_val}
                )
        return image_dict, label


class Features(BaseFeatureExtractor):
    """Interface for obtaining predictions and features from intermediate layer
    activations from Slideflow models.

    Use by calling on either a batch of images (returning outputs for a single
    batch), or by calling on a :class:`slideflow.WSI` object, which will
    generate an array of spatially-mapped activations matching the slide.

    Examples
        *Calling on batch of images:*

        .. code-block:: python

            interface = Features('/model/path', layers='postconv')
            for image_batch in train_data:
                # Return shape: (batch_size, num_features)
                batch_features = interface(image_batch)

        *Calling on a slide:*

        .. code-block:: python

            slide = sf.WSI(...)
            interface = Features('/model/path', layers='postconv')
            # Returns shape:
            # (slide.grid.shape[0], slide.grid.shape[1], num_features)
            activations_grid = interface(slide)

    Note:
        When this interface is called on a batch of images, no image processing
        or stain normalization will be performed, as it is assumed that
        normalization will occur during data loader image processing. When the
        interface is called on a `slideflow.WSI`, the normalization strategy
        will be read from the model configuration file, and normalization will
        be performed on image tiles extracted from the WSI. If this interface
        was created from an existing model and there is no model configuration
        file to read, a slideflow.norm.StainNormalizer object may be passed
        during initialization via the argument `wsi_normalizer`.
    """

    def __init__(
        self,
        path: Optional[str],
        layers: Optional[Union[str, List[str]]] = 'postconv',
        include_preds: bool = False,
        load_method: str = 'weights',
        pooling: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> None:
        """Creates a features interface from a saved slideflow model which
        outputs feature activations at the designated layers.

        Intermediate layers are returned in the order of layers.
        predictions are returned last.

        Args:
            path (str): Path to saved Slideflow model.
            layers (list(str), optional): Layers from which to generate
                activations.  The post-convolution activation layer is accessed
                via 'postconv'. Defaults to 'postconv'.
            include_preds (bool, optional): Include predictions in output. Will be
                returned last. Defaults to False.
            load_method (str): Either 'full' or 'weights'. Method to use
                when loading a Tensorflow model. If 'full', loads the model with
                ``tf.keras.models.load_model()``. If 'weights', will read the
                ``params.json`` configuration file, build the model architecture,
                and then load weights from the given model with
                ``Model.load_weights()``. Loading with 'full' may improve
                compatibility across Slideflow versions. Loading with 'weights'
                may improve compatibility across hardware & environments.
        """
        super().__init__('tensorflow', include_preds=include_preds)
        if layers and isinstance(layers, str):
            layers = [layers]
        self.layers = layers
        self.path = path
        self.device = device
        if isinstance(device, str):
            self.device = device.replace('cuda', 'gpu')
        self._pooling = None
        self._include_preds = None
        if path is not None:
            self._model = load(self.path, method=load_method)  # type: ignore
            config = sf.util.get_model_config(path)
            if 'img_format' in config:
                self.img_format = config['img_format']
            self.hp = sf.ModelParams()
            self.hp.load_dict(config['hp'])
            self.wsi_normalizer = self.hp.get_normalizer()
            if 'norm_fit' in config and config['norm_fit'] is not None:
                if self.wsi_normalizer is None:
                    log.warn('norm_fit found in model config file, but model '
                             'params does not use a normalizer. Ignoring.')
                else:
                    self.wsi_normalizer.set_fit(**config['norm_fit'])
            self._build(
                layers=layers, include_preds=include_preds, pooling=pooling  # type: ignore
            )

    @classmethod
    def from_model(
        cls,
        model: tf.keras.Model,
        layers: Optional[Union[str, List[str]]] = 'postconv',
        include_preds: bool = False,
        wsi_normalizer: Optional["StainNormalizer"] = None,
        pooling: Optional[Any] = None,
        device: Optional[str] = None
    ):
        """Creates a features interface from a loaded slideflow model which
        outputs feature activations at the designated layers.

        Intermediate layers are returned in the order of layers.
        predictions are returned last.

        Args:
            model (:class:`tensorflow.keras.models.Model`): Loaded model.
            layers (list(str), optional): Layers from which to generate
                activations.  The post-convolution activation layer is accessed
                via 'postconv'. Defaults to 'postconv'.
            include_preds (bool, optional): Include predictions in output. Will be
                returned last. Defaults to False.
            wsi_normalizer (:class:`slideflow.norm.StainNormalizer`): Stain
                normalizer to use on whole-slide images. Not used on
                individual tile datasets via __call__. Defaults to None.
        """
        obj = cls(None, layers, include_preds, device=device)
        if isinstance(model, tf.keras.models.Model):
            obj._model = model
        else:
            raise errors.ModelError(f"Model {model} is not a valid Tensorflow "
                                    "model.")
        obj._build(
            layers=layers, include_preds=include_preds, pooling=pooling  # type: ignore
        )
        obj.wsi_normalizer = wsi_normalizer
        return obj

    def __repr__(self):
        return ("{}(\n".format(self.__class__.__name__) +
                "    path={!r},\n".format(self.path) +
                "    layers={!r},\n".format(self.layers) +
                "    include_preds={!r},\n".format(self._include_preds) +
                "    pooling={!r},\n".format(self._pooling) +
                ")")

    def __call__(
        self,
        inp: Union[tf.Tensor, "sf.WSI"],
        **kwargs
    ) -> Optional[Union[np.ndarray, tf.Tensor]]:
        """Process a given input and return features and/or predictions.
        Expects either a batch of images or a :class:`slideflow.WSI`.

        When calling on a `WSI` object, keyword arguments are passed to
        :meth:`slideflow.WSI.build_generator()`.

        """
        if isinstance(inp, sf.WSI):
            return self._predict_slide(inp, **kwargs)
        else:
            return self._predict(inp)

    def _predict_slide(
        self,
        slide: "sf.WSI",
        *,
        img_format: str = 'auto',
        batch_size: int = 32,
        dtype: type = np.float16,
        grid: Optional[np.ndarray] = None,
        shuffle: bool = False,
        show_progress: bool = True,
        callback: Optional[Callable] = None,
        normalizer: Optional[Union[str, "sf.norm.StainNormalizer"]] = None,
        normalizer_source: Optional[str] = None,
        **kwargs
    ) -> Optional[np.ndarray]:
        """Generate activations from slide => activation grid array."""

        # Check image format
        if img_format == 'auto' and self.img_format is None:
            raise ValueError(
                'Unable to auto-detect image format (png or jpg). Set the '
                'format by passing img_format=... to the call function.'
            )
        elif img_format == 'auto':
            assert self.img_format is not None
            img_format = self.img_format

        return sf.model.extractors.features_from_slide(
            self,
            slide,
            img_format=img_format,
            batch_size=batch_size,
            dtype=dtype,
            grid=grid,
            shuffle=shuffle,
            show_progress=show_progress,
            callback=callback,
            normalizer=(normalizer if normalizer else self.wsi_normalizer),
            normalizer_source=normalizer_source,
            **kwargs
        )

    @tf.function
    def _predict(self, inp: tf.Tensor) -> tf.Tensor:
        """Return activations for a single batch of images."""
        with tf.device(self.device) if self.device else no_scope():
            return self.model(inp, training=False)

    def _build(
        self,
        layers: Optional[Union[str, List[str]]],
        include_preds: bool = True,
        pooling: Optional[Any] = None
    ) -> None:
        """Builds the interface model that outputs feature activations at the
        designated layers and/or predictions. Intermediate layers are returned in
        the order of layers. predictions are returned last."""

        self._pooling = pooling
        self._include_preds = include_preds

        if isinstance(pooling, str):
            if pooling == 'avg':
                pooling = tf.keras.layers.GlobalAveragePooling2D
            elif pooling == 'max':
                pooling = tf.keras.layers.GlobalMaxPool2D
            else:
                raise ValueError(f"Unrecognized pooling value {pooling}. "
                                 "Expected 'avg', 'max', or Keras layer.")

        if layers and not isinstance(layers, list):
            layers = [layers]
        if layers:
            if 'postconv' in layers:
                layers[layers.index('postconv')] = 'post_convolution'  # type: ignore
            log.debug(f"Setting up interface to return activations from layers "
                      f"{', '.join(layers)}")
        else:
            layers = []

        def pool_if_3d(tensor):
            if pooling is not None and len(tensor.shape) == 4:
                return pooling()(tensor)
            else:
                return tensor

        # Find the desired layers
        outputs = {}
        outer_layer_outputs = {
            self._model.layers[i].name: self._model.layers[i].output
            for i in range(len(self._model.layers))
        }
        core_layer_outputs = {}
        inner_layers = [la for la in layers if la not in outer_layer_outputs]
        if inner_layers:
            intermediate_core = tf.keras.models.Model(
                inputs=self._model.layers[1].input,
                outputs=[
                    pool_if_3d(self._model.layers[1].get_layer(il).output)
                    for il in inner_layers
                ]
            )
            if len(inner_layers) > 1:
                int_out = intermediate_core(self._model.input)
                for la, layer in enumerate(inner_layers):
                    core_layer_outputs[layer] = int_out[la]
            else:
                outputs[inner_layers[0]] = intermediate_core(self._model.input)
        for layer in layers:
            if layer in outer_layer_outputs:
                outputs[layer] = outer_layer_outputs[layer]
            elif layer in core_layer_outputs:
                outputs[layer] = core_layer_outputs[layer]

        # Build a model that outputs the given layers
        outputs_list = [] if not layers else [outputs[la] for la in layers]
        if include_preds:
            outputs_list += [self._model.output]
        self.model = tf.keras.models.Model(
            inputs=self._model.input,
            outputs=outputs_list
        )
        self.num_features = sum([outputs[o].shape[1] for o in outputs])
        self.num_outputs = len(outputs_list)
        if isinstance(self._model.output, list) and include_preds:
            log.warning("Multi-categorical outcomes is experimental "
                        "for this interface.")
            self.num_classes = sum(o.shape[1] for o in self._model.output)
        elif include_preds:
            self.num_classes = self._model.output.shape[1]
        else:
            self.num_classes = 0

        if include_preds:
            log.debug(f'Number of classes: {self.num_classes}')
        log.debug(f'Number of activation features: {self.num_features}')

    def dump_config(self):
        return {
            'class': 'slideflow.model.tensorflow.Features',
            'kwargs': {
                'path': self.path,
                'layers': self.layers,
                'include_preds': self._include_preds,
                'pooling': self._pooling
            }
        }

class UncertaintyInterface(Features):
    def __init__(
        self,
        path: Optional[str],
        layers: Optional[Union[str, List[str]]] = 'postconv',
        load_method: str = 'weights',
        pooling: Optional[Any] = None
    ) -> None:
        super().__init__(
            path,
            layers=layers,
            include_preds=True,
            load_method=load_method,
            pooling=pooling
        )
        # TODO: As the below to-do suggests, this should be updated
        # for multi-class
        self.num_uncertainty = 1
        if self.num_classes > 2:
            log.warn("UncertaintyInterface not yet implemented for multi-class"
                     " models")

    @classmethod
    def from_model(  # type: ignore
        cls,
        model: tf.keras.Model,
        layers: Optional[Union[str, List[str]]] = None,
        wsi_normalizer: Optional["StainNormalizer"] = None,
        pooling: Optional[Any] = None
    ):
        obj = cls(None, layers)
        if isinstance(model, tf.keras.models.Model):
            obj._model = model
        else:
            raise errors.ModelError(f"Model {model} is not a valid Tensorflow "
                                    "model.")
        obj._build(
            layers=layers, include_preds=True, pooling=pooling  # type: ignore
        )
        obj.wsi_normalizer = wsi_normalizer
        return obj

    def __repr__(self):
        return ("{}(\n".format(self.__class__.__name__) +
                "    path={!r},\n".format(self.path) +
                "    layers={!r},\n".format(self.layers) +
                "    pooling={!r},\n".format(self._pooling) +
                ")")

    @tf.function
    def _predict(self, inp):
        """Return activations (mean), predictions (mean), and uncertainty
        (stdev) for a single batch of images."""

        out_drop = [[] for _ in range(self.num_outputs)]
        for _ in range(30):
            yp = self.model(inp, training=False)
            for n in range(self.num_outputs):
                out_drop[n] += [(yp[n] if self.num_outputs > 1 else yp)]
        for n in range(self.num_outputs):
            out_drop[n] = tf.stack(out_drop[n], axis=0)
        predictions = tf.math.reduce_mean(out_drop[-1], axis=0)

        # TODO: Only takes STDEV from first outcome category which works for
        # outcomes with 2 categories, but a better solution is needed
        # for num_categories > 2
        uncertainty = tf.math.reduce_std(out_drop[-1], axis=0)[:, 0]
        uncertainty = tf.expand_dims(uncertainty, axis=-1)

        if self.num_outputs > 1:
            out = [
                tf.math.reduce_mean(out_drop[n], axis=0)
                for n in range(self.num_outputs-1)
            ]
            return out + [predictions, uncertainty]
        else:
            return predictions, uncertainty

    def dump_config(self):
        return {
            'class': 'slideflow.model.tensorflow.UncertaintyInterface',
            'kwargs': {
                'path': self.path,
                'layers': self.layers,
                'pooling': self._pooling
            }
        }

def load(
    path: str,
    method: str = 'weights',
    custom_objects: Optional[Dict[str, Any]] = None,
    training: bool = False
) -> tf.keras.models.Model:
    """Load a model trained with Slideflow.

    Args:
        path (str): Path to saved model. Must be a model trained in Slideflow.
        method (str): Method to use when loading the model; either 'full' or
            'weights'. If 'full', will load the saved model with
            ``tf.keras.models.load_model()``. If 'weights', will read the
            ``params.json`` configuration file, build the model architecture,
            and then load weights from the given model with
            ``Model.load_weights()``. Loading with 'full' may improve
            compatibility across Slideflow versions. Loading with 'weights'
            may improve compatibility across hardware & environments.
        custom_objects (dict, Optional): Dictionary mapping names
            (strings) to custom classes or functions. Defaults to None.

    Returns:
        tf.keras.models.Model: Loaded model.
    """
    if method not in ('full', 'weights'):
        raise ValueError(f"Unrecognized method {method}, expected "
                         "either 'full' or 'weights'")
    log.debug(f"Loading model with method='{method}'")
    if method == 'full':
        return tf.keras.models.load_model(path, custom_objects=custom_objects)
    else:
        config = sf.util.get_model_config(path)
        hp = ModelParams.from_dict(config['hp'])
        if len(config['outcomes']) == 1 or config['model_type'] == 'linear':
            num_classes = len(list(config['outcome_labels'].keys()))
        else:
            num_classes = {
                outcome: len(list(config['outcome_labels'][outcome].keys()))
                for outcome in config['outcomes']
            }  # type: ignore
        if config['model_type'] == 'cph':
            cph_kw = dict(training=training)
        else:
            cph_kw = dict()
        model = hp.build_model(  # type: ignore
            num_classes=num_classes,
            num_slide_features=0 if not config['input_feature_sizes'] else sum(config['input_feature_sizes']),
            pretrain=None,
            **cph_kw
        )
        model.load_weights(join(path, 'variables/variables'))
        return model
