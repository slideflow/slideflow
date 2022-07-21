'''Tensorflow backend for the slideflow.model submodule.'''

from __future__ import absolute_import, division, print_function

import atexit
import inspect
import json
import os
import shutil
from os.path import dirname, exists, join
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import slideflow as sf
import slideflow.model.base as _base
import slideflow.util.neptune_utils
from packaging import version
from slideflow import errors
from slideflow.model import tensorflow_utils as tf_utils
from slideflow.model.base import log_manifest, no_scope
from slideflow.model.tensorflow_utils import unwrap
from slideflow.util import NormFit, Path
from slideflow.util import colors as col
from slideflow.util import log

import tensorflow as tf
from tensorflow.keras import applications as kapps

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
        # 'ResNeXt50': kapps.ResNeXt50,
        # 'ResNeXt101': kapps.ResNeXt101,
        # 'DenseNet': kapps.DenseNet,
        # 'NASNet': kapps.NASNet
    }

    def __init__(self, *args, **kwargs):
        self.OptDict = {
            'Adam': tf.keras.optimizers.Adam,
            'SGD': tf.keras.optimizers.SGD,
            'RMSprop': tf.keras.optimizers.RMSprop,
            'Adagrad': tf.keras.optimizers.Adagrad,
            'Adadelta': tf.keras.optimizers.Adadelta,
            'Adamax': tf.keras.optimizers.Adamax,
            'Nadam': tf.keras.optimizers.Nadam
        }
        if hasattr(kapps, 'EfficientNetV2B0'):
            self.ModelDict.update({'efficientnet_v2b0': kapps.EfficientNetV2B0})
        if hasattr(kapps, 'EfficientNetV2B1'):
            self.ModelDict.update({'efficientnet_v2b1': kapps.EfficientNetV2B1})
        if hasattr(kapps, 'EfficientNetV2B2'):
            self.ModelDict.update({'efficientnet_v2b2': kapps.EfficientNetV2B2})
        if hasattr(kapps, 'EfficientNetV2B3'):
            self.ModelDict.update({'efficientnet_v2b3': kapps.EfficientNetV2B3})
        if hasattr(kapps, 'EfficientNetV2S'):
            self.ModelDict.update({'efficientnet_v2s': kapps.EfficientNetV2S})
        if hasattr(kapps, 'EfficientNetV2M'):
            self.ModelDict.update({'efficientnet_v2m': kapps.EfficientNetV2M})
        if hasattr(kapps, 'EfficientNetV2L'):
            self.ModelDict.update({'efficientnet_v2l': kapps.EfficientNetV2L})
        self.LinearLossDict = {
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
        self.LinearLossDict.update({
            'negative_log_likelihood': tf_utils.negative_log_likelihood
        })
        self.AllLossDict = {
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
        self.AllLossDict.update({
            'batch_loss_crossentropy': tf_utils.batch_loss_crossentropy,
            'negative_log_likelihood': tf_utils.negative_log_likelihood
        })
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
        pretrain: Optional[str] = 'imagenet'
    ) -> tf.keras.Model:
        """"Builds the base image model, from a Keras model core, with the
        appropriate input tensors and identity layers.

        Args:
            pretrain (str, optional): Pretrained weights to load.
                Defaults to 'imagenet'.

        Returns:
            tf.keras.Model: Base model.
        """
        image_shape = (self.tile_px, self.tile_px, 3)
        tile_input_tensor = tf.keras.Input(shape=image_shape, name='tile_image')
        if pretrain:
            log.info(f'Using pretraining from {col.green(pretrain)}')
        if pretrain and pretrain != 'imagenet':
            pretrained_model = tf.keras.models.load_model(pretrain)
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
        checkpoint: Optional[str] = None
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
        """
        tile_image_model, model_inputs = self._build_base(pretrain)
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
        elif num_slide_features and num_slide_features > 1:
            # Add slide feature input tensors, if there are more slide features
            #    than just the event input tensor for CPH models
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
                    name=f'prelogits-{c}'
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
                name='prelogits'
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
            log.info(f'Loading checkpoint weights from {col.green(checkpoint)}')
            model.load_weights(checkpoint)

        return model

    def _build_cph_model(
        self,
        num_classes: Union[int, Dict[Any, int]],
        num_slide_features: int = 1,
        pretrain: Optional[str] = None,
        checkpoint: Optional[str] = None
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
        """
        activation = 'linear'
        tile_image_model, model_inputs = self._build_base(pretrain)

        # Add slide feature input tensors, if there are more slide features
        #    than just the event input tensor for CPH models
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
            model_inputs += [slide_feature_input_tensor, event_input_tensor]
        elif num_slide_features and num_slide_features > 1:
            # Add slide feature input tensors, if there are more slide features
            #    than just the event input tensor for CPH models
            merged_model = tf.keras.layers.Concatenate(name='input_merge')(
                [slide_feature_input_tensor, tile_image_model.output]
            )
            model_inputs += [slide_feature_input_tensor, event_input_tensor]
        else:
            merged_model = tile_image_model.output
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
                    name=f'prelogits-{c}'
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
                name='prelogits'
            )(merged_model)
            outputs = [tf.keras.layers.Activation(
                activation,
                dtype='float32',
                name='output'
            )(final_dense_layer)]
        outputs[0] = tf.keras.layers.Concatenate(
            name='output_merge_CPH',
            dtype='float32'
        )([outputs[0], event_input_tensor])

        # Assemble final model
        model = tf.keras.Model(inputs=model_inputs, outputs=outputs)

        if checkpoint:
            log.info(f'Loading checkpoint weights from {col.green(checkpoint)}')
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
        if self.loss == 'negative_log_likelihood':
            return 'cph'
        elif self.loss in self.LinearLossDict:
            return 'linear'
        else:
            return 'categorical'


class _PredictionAndEvaluationCallback(tf.keras.callbacks.Callback):
    # TODO: log early stopping batch number, and record

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

    def _metrics_from_dataset(
        self,
        epoch_label: str,
        pred_args: SimpleNamespace
    ) -> Tuple[Dict, float, float]:
        return sf.stats.metrics_from_dataset(
            self.model,
            model_type=self.hp.model_type(),
            patients=self.parent.patients,
            dataset=self.cb_args.validation_data_with_slidenames,
            outcome_names=self.parent.outcome_names,
            label=epoch_label,
            data_dir=self.parent.outdir,
            num_tiles=self.cb_args.num_val_tiles,
            save_predictions=self.cb_args.save_predictions,
            reduce_method=self.cb_args.reduce_method,
            pred_args=pred_args,
            incl_loc=True,
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
                log.info(f'Trained model saved to {col.green(model_path)}')

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
        # Neptune logging for training metrics
        if self.neptune_run:
            self.neptune_run['metrics/train/batch/loss'].log(
                logs['loss'],
                step=self.global_step
            )
            sf.util.neptune_utils.list_log(
                self.neptune_run,
                'metrics/train/batch/accuracy',
                logs['accuracy'],
                step=self.global_step
            )

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
            val_metrics = self.model.evaluate(
                self.cb_args.validation_data,
                verbose=0,
                steps=self.cb_args.validation_steps,
                return_dict=True
            )
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

            # Log to neptune
            if self.neptune_run:
                for v in val_metrics:
                    self.neptune_run[f"metrics/val/batch/{v}"].log(
                        round(val_metrics[v], 3),
                        step=self.global_step
                    )
                if self.last_ema != -1:
                    self.neptune_run["metrics/val/batch/exp_moving_avg"].log(
                        round(self.last_ema, 3),
                        step=self.global_step
                    )
                self.neptune_run["early_stop/stopped_early"] = False

            # Base logging message
            batch_msg = col.blue(f'Batch {batch:<5}')
            loss_msg = f"{col.green('loss')}: {logs['loss']:.3f}"
            val_loss_msg = f"{col.purple('val_loss')}: {val_loss:.3f}"
            if self.model_type == 'categorical':
                acc_msg = f"{col.green('acc')}: {train_acc}"
                val_acc_msg = f"{col.purple('val_acc')}: {val_acc}"
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
        metrics, acc, loss = self._metrics_from_dataset(
            f'val_epoch{epoch}',
            SimpleNamespace(loss=self.hp.get_loss(), uq=bool(self.hp.uq))
        )

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
        sf.util.update_results_log(
            self.cb_args.results_log,
            'trained_model',
            {f'epoch{epoch}': epoch_results}
        )
        # Log epoch results to Neptune
        if self.neptune_run:
            # Training epoch metrics
            self.neptune_run['metrics/train/epoch/loss'].log(
                logs['loss'],
                step=epoch
            )
            sf.util.neptune_utils.list_log(
                self.neptune_run,
                'metrics/train/epoch/accuracy',
                logs['accuracy'],
                step=epoch
            )
            # Validation epoch metrics
            self.neptune_run['metrics/val/epoch/loss'].log(
                val_metrics['loss'],
                step=epoch
            )
            sf.util.neptune_utils.list_log(
                self.neptune_run,
                'metrics/val/epoch/accuracy',
                val_metrics['accuracy'],
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
                        self.neptune_run,
                        metric_label('tile'),
                        tile_metric,
                        step=epoch
                    )
                    sf.util.neptune_utils.list_log(
                        self.neptune_run,
                        metric_label('slide'),
                        slide_metric,
                        step=epoch
                    )
                    sf.util.neptune_utils.list_log(
                        self.neptune_run,
                        metric_label('patient'),
                        patient_metric,
                        step=epoch
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
        patients: Dict[str, str],
        slide_input: Optional[Dict[str, Any]] = None,
        name: str = 'Trainer',
        manifest: Optional[Dict[str, int]] = None,
        feature_sizes: Optional[List[int]] = None,
        feature_names: Optional[List[str]] = None,
        outcome_names: Optional[List[str]] = None,
        mixed_precision: bool = True,
        config: Dict[str, Any] = None,
        use_neptune: bool = False,
        neptune_api: Optional[str] = None,
        neptune_workspace: Optional[str] = None
    ) -> None:

        """Sets base configuration, preparing model inputs and outputs.

        Args:
            hp (:class:`slideflow.model.ModelParams`): ModelParams object.
            outdir (str): Path for event logs and checkpoints.
            labels (dict): Dict mapping slide names to outcome labels (int or
                float format).
            patients (dict): Dict mapping slide names to patient ID, as some
                patients may have multiple slides. If not provided, assumes 1:1
                mapping between slide names and patients.
            slide_input (dict): Dict mapping slide names to additional
                slide-level input, concatenated after post-conv.
            name (str, optional): Optional name describing the model, used for
                model saving. Defaults to 'Trainer'.
            manifest (dict, optional): Manifest dictionary mapping TFRecords to
                number of tiles. Defaults to None.
            feature_sizes (list, optional): List of sizes of input features.
                Required if providing additional input features as input to
                the model.
            feature_names (list, optional): List of names for input features.
                Used when permuting feature importance.
            outcome_names (list, optional): Name of each outcome. Defaults to
                "Outcome {X}" for each outcome.
            mixed_precision (bool, optional): Use FP16 mixed precision (rather
                than FP32). Defaults to True.
            config (dict, optional): Training configuration dictionary, used
                for logging. Defaults to None.
            use_neptune (bool, optional): Use Neptune API logging.
                Defaults to False
            neptune_api (str, optional): Neptune API token, used for logging.
                Defaults to None.
            neptune_workspace (str, optional): Neptune workspace.
                Defaults to None.
        """
        self.outdir = outdir
        self.manifest = manifest
        self.tile_px = hp.tile_px
        self.labels = labels
        self.hp = hp
        self.slides = list(labels.keys())
        self.slide_input = slide_input
        self.feature_names = feature_names
        self.feature_sizes = feature_sizes
        self.num_slide_features = 0 if not feature_sizes else sum(feature_sizes)
        self.mixed_precision = mixed_precision
        self.name = name
        self.neptune_run = None
        self.annotations_tables = []
        self.eval_callback = _PredictionAndEvaluationCallback  # type: tf.keras.callbacks.Callback

        if patients:
            self.patients = patients
        else:
            self.patients = {s: s for s in self.slides}

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Format outcome labels (ensures compatibility with single
        # and multi-outcome models)
        outcome_labels = np.array(list(labels.values()))
        if len(outcome_labels.shape) == 1:
            outcome_labels = np.expand_dims(outcome_labels, axis=1)
        if not outcome_names:
            outcome_names = [
                f'Outcome {i}'
                for i in range(outcome_labels.shape[1])
            ]
        outcome_names = sf.util.as_list(outcome_names)
        if len(outcome_names) != outcome_labels.shape[1]:
            num_names = len(outcome_names)
            num_outcomes = outcome_labels.shape[1]
            raise errors.ModelError(f'Size of outcome_names ({num_names}) != '
                                    f'number of outcomes {num_outcomes}')
        self.outcome_names = outcome_names
        self._setup_inputs()
        if labels:
            self.num_classes = self.hp._detect_classes_from_labels(labels)
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

        # Normalization setup
        self.normalizer = self.hp.get_normalizer()
        if self.normalizer:
            log.info(f'Using realtime {self.hp.normalizer} normalization')

        if self.mixed_precision:
            _policy = 'mixed_float16'
            log.debug(f'Enabling mixed precision ({_policy})')
            if version.parse(tf.__version__) > version.parse("2.8"):
                tf.keras.mixed_precision.set_global_policy(_policy)
            else:
                policy = tf.keras.mixed_precision.experimental.Policy(_policy)
                tf.keras.mixed_precision.experimental.set_policy(policy)

        # Log parameters
        if config is None:
            config = {
                'slideflow_version': sf.__version__,
                'hp': self.hp.get_dict(),
                'backend': sf.backend()
            }
        self.config = config
        sf.util.write_json(config, join(self.outdir, 'params.json'))

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

    def _setup_inputs(self) -> None:
        """Setup slide-level input."""
        if self.num_slide_features:
            assert self.slide_input is not None
            try:
                if self.num_slide_features:
                    log.info(f'Training with both images and '
                             f'{self.num_slide_features} categories of slide-'
                             'level input')
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
        validation_data: tf.data.Dataset,
        steps_per_epoch: int,
        callbacks: tf.keras.callbacks.Callback = None,
        epochs: int = 1
    ) -> Dict:
        """Retrain only the top layer, leaving all other layers frozen."""
        log.info('Retraining top layer')
        # Freeze the base layer
        self.model.layers[0].trainable = False
        val_steps = 200 if validation_data else None
        self._compile_model()

        toplayer_model = self.model.fit(
            train_data,
            epochs=epochs,
            verbose=(sf.getLoggingLevel() <= 20),
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_data,
            validation_steps=val_steps,
            callbacks=callbacks
        )
        # Unfreeze the base layer
        self.model.layers[0].trainable = True
        return toplayer_model.history

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

    def load(self, model: str) -> tf.keras.Model:
        self.model = tf.keras.models.load_model(model)

    def predict(
        self,
        dataset: "sf.Dataset",
        batch_size: Optional[int] = None,
        norm_fit: Optional[NormFit] = None,
        format: str = 'parquet'
    ) -> "pd.DataFrame":
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

        Returns:
            pandas.DataFrame of tile-level predictions.
        """

        if format not in ('csv', 'feather', 'parquet'):
            raise ValueError(f"Unrecognized format {format}")

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
                augment=False
            )
            tf_dts_w_slidenames = dataset.tensorflow(
                incl_slidenames=True,
                incl_loc=True,
                **interleave_kwargs
            )
        # Generate predictions
        log.info('Generating predictions...')
        pred_args = SimpleNamespace(uq=bool(self.hp.uq))
        dfs = sf.stats.predict_from_dataset(
            model=self.model,
            dataset=tf_dts_w_slidenames,
            model_type=self._model_type,
            pred_args=pred_args,
            num_tiles=dataset.num_tiles,
            outcome_names=self.outcome_names,
            incl_loc=True
        )

        # Save predictions
        sf.stats.metrics.save_dfs(dfs, format=format, outdir=self.outdir)

        return dfs

    def evaluate(
        self,
        dataset: "sf.Dataset",
        batch_size: Optional[int] = None,
        save_predictions: Union[bool, str] = 'parquet',
        reduce_method: str = 'average',
        norm_fit: Optional[NormFit] = None,
        uq: Union[bool, str] = 'auto'
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
                slide-level and patient-level predictions for categorical outcomes.
                Either 'average' or 'proportion'. If 'average', will reduce with
                average of each logit across tiles. If 'proportion', will convert
                tile predictions into onehot encoding then reduce by averaging
                these onehot values. Defaults to 'average'.
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
                augment=False
            )
            tf_dts_w_slidenames = dataset.tensorflow(
                incl_slidenames=True,
                incl_loc=True,
                **interleave_kwargs
            )
        # Generate performance metrics
        log.info('Calculating performance metrics...')
        metric_kwargs = self._metric_kwargs(
            dataset=tf_dts_w_slidenames,
            num_tiles=dataset.num_tiles,
            label='eval'
        )
        pred_args = SimpleNamespace(
            loss=self.hp.get_loss(),
            uq=bool(self.hp.uq)
        )
        metrics, acc, loss = sf.stats.metrics_from_dataset(
            save_predictions=save_predictions,
            pred_args=pred_args,
            reduce_method=reduce_method,
            incl_loc=True,
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
        reduce_method: str = 'average',
        norm_fit: Optional[NormFit] = None,
    ):
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
            resume_training (str, optional): Path to Tensorflow model to
                continue training. Defaults to None.
            pretrain (str, optional): Either 'imagenet' or path to Tensorflow
                model from which to load weights. Defaults to 'imagenet'.
            checkpoint (str, optional): Path to cp.ckpt from which to load
                weights. Defaults to None.
            save_checkpoint(bool, optional): Save checkpoints at each epoch.
                Defaults to True.
            multi_gpu(bool, optional): Enable multi-GPU training using
                Tensorflow/Keras MirroredStrategy.
            reduce_method (str, optional): Reduction method for calculating
                slide-level and patient-level predictions for categorical outcomes.
                Either 'average' or 'proportion'. If 'average', will reduce with
                average of each logit across tiles. If 'proportion', will convert
                tile predictions into onehot encoding then reduce by averaging
                these onehot values. Defaults to 'average'.
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
                    'hp': self.hp.get_dict(),
                    'backend': sf.backend()
                }
            else:
                config = sf.util.load_json(config_path)
            config['norm_fit'] = self.normalizer.get_fit(as_list=True)
            sf.util.write_json(config, config_path)

        # Save training / validation manifest
        val_tfrecords = None if val_dts is None else val_dts.tfrecords()
        log_manifest(
            train_dts.tfrecords(),
            val_tfrecords,
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
            # Build model from ModeLParams
            if resume_training:
                self.model = tf.keras.load_model(resume_training)
            else:
                model = self.hp.build_model(
                    labels=self.labels,
                    num_slide_features=self.num_slide_features,
                    pretrain=pretrain,
                    checkpoint=checkpoint
                )
                self.model = model
                tf_utils.log_summary(model, self.neptune_run)

            with tf.name_scope('input'):
                t_kwargs = self._interleave_kwargs(
                    batch_size=self.hp.batch_size,
                    infinite=True,
                    augment=self.hp.augment
                )
                train_data = train_dts.tensorflow(drop_last=True, **t_kwargs)

            # Set up validation data
            using_validation = (val_dts and len(val_dts.tfrecords()))
            if using_validation:
                assert val_dts is not None
                with tf.name_scope('input'):
                    if not validation_batch_size:
                        validation_batch_size = self.hp.batch_size
                    v_kwargs = self._interleave_kwargs_val(
                        batch_size=validation_batch_size,
                        infinite=False,
                        augment=False
                    )
                    val_data = val_dts.tensorflow(**v_kwargs)
                    val_data_w_slidenames = val_dts.tensorflow(
                        incl_slidenames=True,
                        incl_loc=True,
                        drop_last=True,
                        **v_kwargs
                    )
                if validate_on_batch:
                    log.debug('Validation during training: every '
                              f'{validate_on_batch} steps and at epoch end')
                else:
                    log.debug('Validation during training: at epoch end')
                if validation_steps:
                    validation_data_for_training = val_data.repeat()
                    num_samples = validation_steps * self.hp.batch_size
                    log.debug(f'Using {validation_steps} batches ({num_samples}'
                              ' samples) each validation check')
                else:
                    validation_data_for_training = val_data
                    log.debug('Using entire validation set each val check')
            else:
                log.debug('Validation during training: None')
                validation_data_for_training = None
                val_data = None
                val_data_w_slidenames = None
                validation_steps = 0

            # Calculate parameters
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
                steps_per_epoch = round(train_dts.num_tiles/self.hp.batch_size)

            cb_args = SimpleNamespace(
                starting_epoch=starting_epoch,
                using_validation=using_validation,
                validate_on_batch=validate_on_batch,
                validation_data=val_data,
                validation_steps=validation_steps,
                ema_observations=ema_observations,
                ema_smoothing=ema_smoothing,
                steps_per_epoch=steps_per_epoch,
                validation_data_with_slidenames=val_data_w_slidenames,
                num_val_tiles=(0 if val_dts is None else val_dts.num_tiles),
                save_predictions=save_predictions,
                save_model=save_model,
                results_log=results_log,
                reduce_method=reduce_method
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
                tensorboard_callback = tf.keras.callbacks.TensorBoard(
                    log_dir=self.outdir,
                    histogram_freq=0,
                    write_graph=False,
                    update_freq=log_frequency
                )
                callbacks += [tensorboard_callback]

            # Retrain top layer only, if using transfer learning and
            # not resuming training
            total_epochs = (self.hp.toplayer_epochs
                            + (max(self.hp.epochs) - starting_epoch))
            if self.hp.toplayer_epochs:
                self._retrain_top_layers(
                    train_data,
                    validation_data_for_training,
                    steps_per_epoch,
                    callbacks=None,
                    epochs=self.hp.toplayer_epochs
                )
            # Train the model
            self._compile_model()
            log.info('Beginning training')
            try:
                self.model.fit(
                    train_data,
                    steps_per_epoch=steps_per_epoch,
                    epochs=total_epochs,
                    verbose=(sf.getLoggingLevel() <= 20),
                    initial_epoch=self.hp.toplayer_epochs,
                    validation_data=validation_data_for_training,
                    validation_steps=validation_steps,
                    callbacks=callbacks
                )
            except tf.errors.ResourceExhaustedError as e:
                log.debug(e)
                log.error(f"Training failed for {col.bold(self.name)}, "
                          "GPU memory exceeded.")
            results = val_callback.results
            if self.use_neptune and self.neptune_run is not None:
                self.neptune_run['results'] = results['epochs']
                self.neptune_run.stop()

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

    def load(self, model: str) -> tf.keras.Model:
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


class Features:
    """Interface for obtaining logits and features from intermediate layer
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
        path: Optional[Path],
        layers: Optional[Union[str, List[str]]] = 'postconv',
        include_logits: bool = False
    ) -> None:
        """Creates a features interface from a saved slideflow model which
        outputs feature activations at the designated layers.

        Intermediate layers are returned in the order of layers.
        Logits are returned last.

        Args:
            path (str): Path to saved Slideflow model.
            layers (list(str), optional): Layers from which to generate
                activations.  The post-convolution activation layer is accessed
                via 'postconv'. Defaults to 'postconv'.
            include_logits (bool, optional): Include logits in output. Will be
                returned last. Defaults to False.
        """
        self.path = path
        self.num_logits = 0
        self.num_features = 0
        self.num_uncertainty = 0
        self.img_format = None
        log.debug('Setting up Features interface')
        if path is not None:
            self._model = tf.keras.models.load_model(self.path)
            config = sf.util.get_model_config(path)
            if 'img_format' in config:
                self.img_format = config['img_format']
            self.hp = sf.model.ModelParams()
            self.hp.load_dict(config['hp'])
            self.wsi_normalizer = self.hp.get_normalizer()
            if 'norm_fit' in config and config['norm_fit'] is not None:
                if self.wsi_normalizer is None:
                    log.warn('norm_fit found in model config file, but model '
                             'params does not use a normalizer. Ignoring.')
                else:
                    self.wsi_normalizer.set_fit(**config['norm_fit'])
            self._build(
                layers=layers, include_logits=include_logits  # type: ignore
            )

    @classmethod
    def from_model(
        cls,
        model: tf.keras.Model,
        layers: Optional[Union[str, List[str]]] = 'postconv',
        include_logits: bool = False,
        wsi_normalizer: Optional["StainNormalizer"] = None
    ):
        """Creates a features interface from a loaded slideflow model which
        outputs feature activations at the designated layers.

        Intermediate layers are returned in the order of layers.
        Logits are returned last.

        Args:
            model (:class:`tensorflow.keras.models.Model`): Loaded model.
            layers (list(str), optional): Layers from which to generate
                activations.  The post-convolution activation layer is accessed
                via 'postconv'. Defaults to 'postconv'.
            include_logits (bool, optional): Include logits in output. Will be
                returned last. Defaults to False.
            wsi_normalizer (:class:`slideflow.norm.StainNormalizer`): Stain
                normalizer to use on whole-slide images. Is not used on
                individual tile datasets via __call__. Defaults to None.
        """
        obj = cls(None, layers, include_logits)
        if isinstance(model, tf.keras.models.Model):
            obj._model = model
        else:
            raise errors.ModelError("Model is not a valid Tensorflow model.")
        obj._build(
            layers=layers, include_logits=include_logits  # type: ignore
        )
        obj.wsi_normalizer = wsi_normalizer
        return obj

    def __call__(
        self,
        inp: Union[tf.Tensor, "sf.WSI"],
        **kwargs
    ) -> Optional[Union[np.ndarray, tf.Tensor]]:
        """Process a given input and return features and/or logits.
        Expects either a batch of images or a :class:`slideflow.WSI`."""

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
        **kwargs
    ) -> Optional[np.ndarray]:
        """Generate activations from slide => activation grid array."""

        log.debug(f"Slide prediction (batch_size={batch_size}, "
                  f"img_format={img_format})")
        if img_format == 'auto' and self.img_format is None:
            raise ValueError(
                'Unable to auto-detect image format (png or jpg). Set the '
                'format by passing img_format=... to the call function.'
            )
        elif img_format == 'auto':
            assert self.img_format is not None
            img_format = self.img_format
        total_out = self.num_features + self.num_logits + self.num_uncertainty
        features_grid = np.ones((
                slide.grid.shape[1],
                slide.grid.shape[0],
                total_out),
            dtype=dtype)
        features_grid *= -1
        generator = slide.build_generator(
            shuffle=False,
            show_progress=True,
            img_format=img_format,
            **kwargs
        )
        if not generator:
            log.error(f"No tiles extracted from slide {col.green(slide.name)}")
            return None

        def tile_generator():
            for image_dict in generator():
                yield {
                    'grid': image_dict['grid'],
                    'image': image_dict['image']
                }

        @tf.function
        def _parse_function(record):
            image = record['image']
            if img_format.lower() in ('jpg', 'jpeg'):
                image = tf.image.decode_jpeg(image, channels=3)
            elif img_format.lower() in ('png',):
                image = tf.image.decode_png(image, channels=3)
            loc = record['grid']
            if self.wsi_normalizer:
                image = self.wsi_normalizer.tf_to_tf(image)
            parsed_image = tf.image.per_image_standardization(image)
            parsed_image.set_shape([slide.tile_px, slide.tile_px, 3])
            return parsed_image, loc

        # Generate dataset from the generator
        with tf.name_scope('dataset_input'):
            output_signature = {
                'image': tf.TensorSpec(shape=(), dtype=tf.string),
                'grid': tf.TensorSpec(shape=(2), dtype=tf.uint32)
            }
            tile_dataset = tf.data.Dataset.from_generator(
                tile_generator,
                output_signature=output_signature
            )
            tile_dataset = tile_dataset.map(
                _parse_function,
                num_parallel_calls=8
            )
            tile_dataset = tile_dataset.batch(batch_size, drop_remainder=False)
            tile_dataset = tile_dataset.prefetch(8)

        act_arr = []
        loc_arr = []
        for i, (batch_images, batch_loc) in enumerate(tile_dataset):
            model_out = self._predict(batch_images)
            if not isinstance(model_out, (list, tuple)):
                model_out = [model_out]
            act_arr += [np.concatenate([m.numpy() for m in model_out], axis=-1)]
            loc_arr += [batch_loc.numpy()]

        act_arr = np.concatenate(act_arr)
        loc_arr = np.concatenate(loc_arr)

        for i, act in enumerate(act_arr):
            xi = loc_arr[i][0]
            yi = loc_arr[i][1]
            features_grid[yi][xi] = act

        return features_grid

    @tf.function
    def _predict(self, inp: tf.Tensor) -> tf.Tensor:
        """Return activations for a single batch of images."""
        return self.model(inp, training=False)

    def _build(
        self,
        layers: Optional[Union[str, List[str]]],
        include_logits: bool = True
    ) -> None:
        """Builds the interface model that outputs feature activations at the
        designated layers and/or logits. Intermediate layers are returned in
        the order of layers. Logits are returned last."""
        if layers and not isinstance(layers, list):
            layers = [layers]
        if layers:
            log.debug(f"Setting up interface to return activations from layers "
                      f"{', '.join(layers)}")
            other_layers = [la for la in layers if la != 'postconv']
        else:
            other_layers = []
        outputs = {}
        if layers:
            intermediate_core = tf.keras.models.Model(
                inputs=self._model.layers[1].input,
                outputs=[
                    self._model.layers[1].get_layer(ol).output
                    for ol in other_layers
                ]
            )
            if len(other_layers) > 1:
                int_out = intermediate_core(self._model.input)
                for la, layer in enumerate(other_layers):
                    outputs[layer] = int_out[la]
            elif len(other_layers):
                outputs[other_layers[0]] = intermediate_core(self._model.input)
            if 'postconv' in layers:
                outputs['postconv'] = self._model.layers[1].get_output_at(0)
        outputs_list = [] if not layers else [outputs[la] for la in layers]
        if include_logits:
            outputs_list += [self._model.output]
        self.model = tf.keras.models.Model(
            inputs=self._model.input,
            outputs=outputs_list
        )
        self.num_features = sum([outputs[o].shape[1] for o in outputs])
        self.num_outputs = len(outputs_list)
        if isinstance(self._model.output, list):
            log.warning("Multi-categorical outcomes not yet supported "
                        "for this interface.")
            self.num_logits = 0
        elif include_logits:
            self.num_logits = self._model.output.shape[1]
        else:
            self.num_logits = 0

        if include_logits:
            log.debug(f'Number of logits: {self.num_logits}')
        log.debug(f'Number of activation features: {self.num_features}')


class UncertaintyInterface(Features):
    def __init__(
        self,
        path: Path,
        layers: Optional[Union[str, List[str]]] = None
    ) -> None:
        log.debug('Setting up UncertaintyInterface')
        super().__init__(path, layers=layers, include_logits=True)
        # TODO: As the below to-do suggests, this should be updated
        # for multi-class
        self.num_uncertainty = 1
        if self.num_logits > 2:
            log.warn("UncertaintyInterface not yet implemented for multi-class"
                     " models")

    @classmethod
    def from_model(  # type: ignore
        cls,
        model: tf.keras.Model,
        layers: Optional[Union[str, List[str]]] = None,
        wsi_normalizer: Optional["StainNormalizer"] = None,
    ) -> None:
        super().from_model(
            model,
            layers=layers,
            include_logits=True,
            wsi_normalizer=wsi_normalizer
        )

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
        logits = tf.math.reduce_mean(out_drop[-1], axis=0)

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
            return out + [logits, uncertainty]
        else:
            return logits, uncertainty
