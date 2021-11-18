'''Tensorflow backend for the slideflow.model submodule.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import warnings
import shutil
import types
import inspect

warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
import slideflow as sf
import slideflow.model.base as _base
import slideflow.util.neptune_utils

from os.path import join, exists, dirname
from slideflow.util import log
from slideflow.model.base import ModelError, FeatureError, no_scope, log_summary, log_manifest
from slideflow.model.tensorflow_utils import *

class ModelParams(_base._ModelParams):
    """Build a set of hyperparameters."""

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
        self.ModelDict = {
            'xception': tf.keras.applications.Xception,
            'vgg16': tf.keras.applications.VGG16,
            'vgg19': tf.keras.applications.VGG19,
            'resnet50': tf.keras.applications.ResNet50,
            'resnet101': tf.keras.applications.ResNet101,
            'resnet152': tf.keras.applications.ResNet152,
            'resnet50_v2': tf.keras.applications.ResNet50V2,
            'resnet101_v2': tf.keras.applications.ResNet101V2,
            'resnet152_v2': tf.keras.applications.ResNet152V2,
            #'ResNeXt50': tf.keras.applications.ResNeXt50,
            #'ResNeXt101': tf.keras.applications.ResNeXt101,
            'inception': tf.keras.applications.InceptionV3,
            'nasnet_large': tf.keras.applications.NASNetLarge,
            'inception_resnet_v2': tf.keras.applications.InceptionResNetV2,
            'mobilenet': tf.keras.applications.MobileNet,
            'mobilenet_v2': tf.keras.applications.MobileNetV2,
            #'DenseNet': tf.keras.applications.DenseNet,
            #'NASNet': tf.keras.applications.NASNet
        }
        self.LinearLossDict = {
            'mean_squared_error': tf.keras.losses.mean_squared_error,
            'mean_absolute_error': tf.keras.losses.mean_absolute_error,
            'mean_absolute_percentage_error': tf.keras.losses.mean_absolute_percentage_error,
            'mean_squared_logarithmic_error': tf.keras.losses.mean_squared_logarithmic_error,
            'squared_hinge': tf.keras.losses.squared_hinge,
            'hinge': tf.keras.losses.hinge,
            'logcosh': tf.keras.losses.logcosh,
            'negative_log_likelihood': negative_log_likelihood
        }
        self.AllLossDict = {
            'mean_squared_error': tf.keras.losses.mean_squared_error,
            'mean_absolute_error': tf.keras.losses.mean_absolute_error,
            'mean_absolute_percentage_error': tf.keras.losses.mean_absolute_percentage_error,
            'mean_squared_logarithmic_error': tf.keras.losses.mean_squared_logarithmic_error,
            'squared_hinge': tf.keras.losses.squared_hinge,
            'hinge': tf.keras.losses.hinge,
            'categorical_hinge': tf.keras.losses.categorical_hinge,
            'logcosh': tf.keras.losses.logcosh,
            'huber': tf.keras.losses.huber,
            'categorical_crossentropy': tf.keras.losses.categorical_crossentropy,
            'sparse_categorical_crossentropy': tf.keras.losses.sparse_categorical_crossentropy,
            'binary_crossentropy': tf.keras.losses.binary_crossentropy,
            'kullback_leibler_divergence': tf.keras.losses.kullback_leibler_divergence,
            'poisson': tf.keras.losses.poisson,
            'negative_log_likelihood': negative_log_likelihood
        }
        super().__init__(*args, **kwargs)
        assert self.model in self.ModelDict.keys()
        assert self.optimizer in self.OptDict.keys()
        assert self.loss in self.AllLossDict.keys()

    def _add_hidden_layers(self, model, regularizer):
        for i in range(self.hidden_layers):
            model = tf.keras.layers.Dense(self.hidden_layer_width,
                                          name=f'hidden_{i}',
                                          activation='relu',
                                          kernel_regularizer=regularizer)(model)
        return model

    def _add_regularization(self, model):
        # Add L2 regularization to all compatible layers in the base model
        if self.L2_weight != 0:
            regularizer = tf.keras.regularizers.l2(self.L2_weight)
            model = add_regularization(model, regularizer)
        else:
            regularizer = None
        return model, regularizer

    def _freeze_layers(self, model):
        freezeIndex = int(len(model.layers) - (self.trainable_layers - 1 ))# - self.hp.hidden_layers - 1))
        log.info(f'Only training on last {self.trainable_layers} layers (of {len(model.layers)} total)')
        for layer in model.layers[:freezeIndex]:
            layer.trainable = False
        return model

    def _get_core(self, weights=None):
        """Returns a Keras model of the appropriate architecture, input shape, pooling, and initial weights."""
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
        model_kw = [param.name for param in model_fn_sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
        model_kwargs = {key:model_kwargs[key] for key in model_kw if key in model_kwargs}
        return model_fn(**model_kwargs)

    def _build_base(self, pretrain='imagenet'):
        """Builds the base image model, from a Keras model core, with the appropriate input tensors and identity layers."""
        image_shape = (self.tile_px, self.tile_px, 3)
        tile_input_tensor = tf.keras.Input(shape=image_shape, name='tile_image')
        if pretrain: log.info(f'Using pretraining from {sf.util.green(pretrain)}')
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
                base_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output, name=base_model.name)

        # Add regularization
        base_model, regularizer = self._add_regularization(base_model)

        # Allow only a subset of layers in the base model to be trainable
        if self.trainable_layers != 0:
            base_model = self._freeze_layers(base_model)

        # This is an identity layer that simply returns the last layer, allowing us to name and access this layer later
        post_convolution_identity_layer = tf.keras.layers.Activation('linear', name='post_convolution')
        layers = [tile_input_tensor, base_model]
        if not self.pooling:
            layers += [tf.keras.layers.Flatten()]
        layers += [post_convolution_identity_layer]
        if self.dropout:
            layers += [tf.keras.layers.Dropout(self.dropout)]
        tile_image_model = tf.keras.Sequential(layers)
        model_inputs = [tile_image_model.input]
        return tile_image_model, model_inputs, regularizer

    def _build_categorical_model(self, num_classes, num_slide_features=0, activation='softmax',
                                pretrain='imagenet', checkpoint=None):

        """Assembles categorical model, using pretraining (imagenet) or the base layers of a supplied model.

        Args:
            num_classes (int or dict): Either int (single categorical outcome, indicating number of classes) or dict
                (dict mapping categorical outcome names to number of unique categories in each outcome).
            num_slide_features (int): Number of slide-level features separate from image input. Defaults to 0.
            activation (str): Type of final layer activation to use. Defaults to softmax.
            pretrain (str): Either 'imagenet' or path to model to use as pretraining. Defaults to 'imagenet'.
            checkpoint (str): Path to checkpoint from which to resume model training. Defaults to None.
        """

        tile_image_model, model_inputs, regularizer = self._build_base(pretrain)

        if num_slide_features:
            slide_feature_input_tensor = tf.keras.Input(shape=(num_slide_features), name='slide_feature_input')

        # Merge layers
        if num_slide_features and ((self.tile_px == 0) or self.drop_images):
            log.info('Generating model with just clinical variables and no images')
            merged_model = slide_feature_input_tensor
            model_inputs += [slide_feature_input_tensor]
        else:
            merged_model = tile_image_model.output

        # Add hidden layers
        merged_model = self._add_hidden_layers(merged_model, regularizer)

        # Multi-categorical outcomes
        if isinstance(num_classes, dict):
            outputs = []
            for c in num_classes:
                final_dense_layer = tf.keras.layers.Dense(num_classes[c],
                                                          kernel_regularizer=regularizer,
                                                          name=f'prelogits-{c}')(merged_model)
                outputs += [tf.keras.layers.Activation(activation, dtype='float32', name=f'out-{c}')(final_dense_layer)]
        else:
            final_dense_layer = tf.keras.layers.Dense(num_classes,
                                                      kernel_regularizer=regularizer,
                                                      name='prelogits')(merged_model)
            outputs = [tf.keras.layers.Activation(activation, dtype='float32', name='output')(final_dense_layer)]

        # Assemble final model
        log.debug(f'Using {activation} activation')
        model = tf.keras.Model(inputs=model_inputs, outputs=outputs)

        if checkpoint:
            log.info(f'Loading checkpoint weights from {sf.util.green(checkpoint)}')
            model.load_weights(checkpoint)

        return model

    def _build_cph_model(self, num_classes, num_slide_features=1, pretrain=None, checkpoint=None):
        """Assembles a Cox Proportional Hazards (CPH) model, using pretraining (imagenet) or the base layers
        of a supplied model.

        Args:
            num_classes (int or dict): Either int (single categorical outcome, indicating number of classes) or dict
                (dict mapping categorical outcome names to number of unique categories in each outcome).
            num_slide_features (int): Number of slide-level features separate from image input. Defaults to 0.
            activation (str): Type of final layer activation to use. Defaults to softmax.
            pretrain (str): Either 'imagenet' or path to model to use as pretraining. Defaults to 'imagenet'.
            checkpoint (str): Path to checkpoint from which to resume model training. Defaults to None.
        """

        activation = 'linear'
        tile_image_model, model_inputs, regularizer = self._build_base(pretrain)

        # Add slide feature input tensors, if there are more slide features
        #    than just the event input tensor for CPH models
        event_input_tensor = tf.keras.Input(shape=(1), name='event_input')
        if not (num_slide_features == 1):
            slide_feature_input_tensor = tf.keras.Input(shape=(num_slide_features - 1),
                                                        name='slide_feature_input')

        # Merge layers
        if num_slide_features and ((self.tile_px == 0) or self.drop_images):
            # Add images
            log.info('Generating model with just clinical variables and no images')
            merged_model = slide_feature_input_tensor
            model_inputs += [slide_feature_input_tensor, event_input_tensor]
        elif num_slide_features and num_slide_features > 1:
            # Add slide feature input tensors, if there are more slide features
            #    than just the event input tensor for CPH models
            merged_model = tf.keras.layers.Concatenate(name='input_merge')([slide_feature_input_tensor,
                                                                            tile_image_model.output])
            model_inputs += [slide_feature_input_tensor, event_input_tensor]
        else:
            merged_model = tile_image_model.output
            model_inputs += [event_input_tensor]

        # Add hidden layers
        merged_model = self._add_hidden_layers(merged_model, regularizer)

        log.debug(f'Using {activation} activation')

        # Multi-categorical outcomes
        if type(num_classes) == dict:
            outputs = []
            for c in self.num_classes:
                final_dense_layer = tf.keras.layers.Dense(num_classes[c],
                                                          kernel_regularizer=regularizer,
                                                          name=f'prelogits-{c}')(merged_model)
                outputs += [tf.keras.layers.Activation(activation, dtype='float32', name=f'out-{c}')(final_dense_layer)]
        else:
            final_dense_layer = tf.keras.layers.Dense(num_classes,
                                                      kernel_regularizer=regularizer,
                                                      name='prelogits')(merged_model)
            outputs = [tf.keras.layers.Activation(activation, dtype='float32', name='output')(final_dense_layer)]
        outputs[0] = tf.keras.layers.Concatenate(name='output_merge_CPH',
                                                 dtype='float32')([outputs[0], event_input_tensor])

        # Assemble final model
        model = tf.keras.Model(inputs=model_inputs, outputs=outputs)

        if checkpoint:
            log.info(f'Loading checkpoint weights from {sf.util.green(checkpoint)}')
            model.load_weights(checkpoint)

        return model

    def build_model(self, labels=None, num_classes=None, **kwargs):
        """Auto-detects model type (categorical, linear, CPH) from parameters and builds, using pretraining (imagenet)
        or the base layers of a supplied model.

        Args:
            labels (dict, optional): Dict mapping slide names to outcomes. Used to detect number of outcome categories.
            num_classes (int or dict, optional): Either int (single categorical outcome, indicating number of classes)
                or dict (dict mapping categorical outcome names to number of unique categories in each outcome).
                Must supply either `num_classes` or `label` (can detect number of classes from labels)
            num_slide_features (int, optional): Number of slide-level features separate from image input. Defaults to 0.
            activation (str, optional): Type of final layer activation to use. Defaults to 'softmax' (categorical models)
                or 'linear' (linear or CPH models).
            pretrain (str, optional): Either 'imagenet' or path to model to use as pretraining. Defaults to 'imagenet'.
            checkpoint (str, optional): Path to checkpoint from which to resume model training. Defaults to None.
        """

        assert num_classes is not None or labels is not None
        if num_classes is None:
            num_classes = self._detect_classes_from_labels(labels)

        if self.model_type() == 'categorical':
            return self._build_categorical_model(num_classes, **kwargs, activation='softmax')
        elif self.model_type() == 'linear':
            return self._build_categorical_model(num_classes, **kwargs, activation='linear')
        elif self.model_type() == 'cph':
            return self._build_cph_model(num_classes, **kwargs)
        else:
            raise ModelError(f'Unknown model type: {self.model_type()}')

    def get_loss(self):
        return self.AllLossDict[self.loss]

    def get_opt(self):
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
            return self.OptDict[self.optimizer](lr=self.learning_rate)

    def model_type(self):
        """Returns either 'linear', 'categorical', or 'cph' depending on the loss type."""
        if self.loss == 'negative_log_likelihood':
            return 'cph'
        elif self.loss in self.LinearLossDict:
            return 'linear'
        else:
            return 'categorical'

class _PredictionAndEvaluationCallback(tf.keras.callbacks.Callback):
    # TODO: log early stopping batch number, and record

    """Prediction and Evaluation Callback used during model training."""

    def __init__(self, parent, cb_args):
        super(_PredictionAndEvaluationCallback, self).__init__()
        self.parent = parent
        self.hp = parent.hp
        self.cb_args = cb_args
        self.early_stop = False
        self.last_ema = -1
        self.moving_average = []
        self.ema_two_checks_prior = -1
        self.ema_one_check_prior = -1
        self.epoch_count = cb_args.starting_epoch
        self.model_type = self.hp.model_type()
        self.results = {'epochs': {}}
        self.neptune_run = self.parent.neptune_run

    def on_epoch_end(self, epoch, logs={}):
        if log.getEffectiveLevel() <= 20: print('\r\033[K', end='')
        self.epoch_count += 1
        if self.epoch_count in [e for e in self.hp.epochs]:
            model_name = self.parent.name if self.parent.name else 'trained_model'
            model_path = os.path.join(self.parent.outdir, f'{model_name}_epoch{self.epoch_count}')
            self.model.save(model_path)

            # Try to copy model settings/hyperparameters file into the model folder
            if not exists(join(model_path, 'params.json')):
                try:
                    config_path = join(dirname(model_path), 'params.json')
                    if self.neptune_run:
                        config = sf.util.load_json(config_path)
                        config['neptune_id'] = self.neptune_run['sys/id'].fetch()
                        sf.util.write_json(config, config_path)
                    shutil.copy(config_path,
                                join(model_path, 'params.json'), )
                    shutil.copy(join(dirname(model_path), 'slide_manifest.csv'),
                                join(model_path, 'slide_manifest.csv'), )
                except:
                    log.warning('Unable to copy params.json/slide_manifest.csv files into model folder.')

            log.info(f'Trained model saved to {sf.util.green(model_path)}')
            if self.cb_args.using_validation:
                self.evaluate_model(logs)
        elif self.early_stop:
            self.evaluate_model(logs)
        self.model.stop_training = self.early_stop

    def on_train_batch_end(self, batch, logs={}):
        if (self.cb_args.using_validation and self.cb_args.validate_on_batch
            and (batch > 0) and (batch % self.cb_args.validate_on_batch == 0)):

            val_metrics = self.model.evaluate(self.cb_args.validation_data,
                                                verbose=0,
                                                steps=self.cb_args.validation_steps,
                                                return_dict=True)
            val_loss = val_metrics['loss']
            self.model.stop_training = False
            if self.hp.early_stop_method == 'accuracy' and 'val_accuracy' in val_metrics:
                early_stop_value = val_metrics['val_accuracy']
                val_acc = f"{val_metrics['val_accuracy']:3f}"
            else:
                early_stop_value = val_loss
                val_acc = ', '.join([f'{val_metrics[v]:.3f}' for v in val_metrics if 'accuracy' in v])
            if 'accuracy' in logs:
                train_acc = f"{logs['accuracy']:.3f}"
            else:
                train_acc = ', '.join([f'{logs[v]:.3f}' for v in logs if 'accuracy' in v])
            if log.getEffectiveLevel() <= 20: print('\r\033[K', end='')
            self.moving_average += [early_stop_value]

            # Log to neptune
            if self.neptune_run:
                self.neptune_run["metrics/batch/batch"].log(batch)
                self.neptune_run["metrics/batch/epoch"].log(self.epoch_count)
                for v in val_metrics:
                    self.neptune_run[f"metrics/batch/val_{v}"].log(round(val_metrics[v], 3))
                self.neptune_run["metrics/batch/exp_moving_average"].log(round(self.last_ema, 3))
                self.neptune_run["early_stop/stopped_early"] = False

            # Base logging message
            batch_msg = sf.util.blue(f'Batch {batch:<5}')
            loss_msg = f"{sf.util.green('loss')}: {logs['loss']:.3f}"
            val_loss_msg = f"{sf.util.purple('val_loss')}: {val_loss:.3f}"
            if self.model_type == 'categorical':
                acc_msg = f"{sf.util.green('acc')}: {train_acc}"
                val_acc_msg = f"{sf.util.purple('val_acc')}: {val_acc}"
                log_message = f"{batch_msg} {loss_msg}, {acc_msg} | {val_loss_msg}, {val_acc_msg}"
            else:
                log_message = f"{batch_msg} {loss_msg} | {val_loss_msg}"

            # Calculate exponential moving average of validation accuracy
            if len(self.moving_average) <= self.cb_args.ema_observations:
                log.info(log_message)
            else:
                # Only keep track of the last [ema_observations] validation accuracies
                self.moving_average.pop(0)
                if self.last_ema == -1:
                    # Calculate simple moving average
                    self.last_ema = sum(self.moving_average) / len(self.moving_average)
                    log.info(log_message +  f' (SMA: {self.last_ema:.3f})')
                else:
                    # Update exponential moving average
                    self.last_ema = (early_stop_value * (self.cb_args.ema_smoothing/(1+self.cb_args.ema_observations))) + \
                                    (self.last_ema * (1-(self.cb_args.ema_smoothing/(1+self.cb_args.ema_observations))))
                    log.info(log_message + f' (EMA: {self.last_ema:.3f})')

            # If early stopping and our patience criteria has been met,
            #   check if validation accuracy is still improving
            if (self.hp.early_stop and
                (self.last_ema != -1) and
                (float(batch)/self.cb_args.steps_per_epoch)+self.epoch_count > self.hp.early_stop_patience):

                if (self.ema_two_checks_prior != -1 and
                    ((self.hp.early_stop_method == 'accuracy' and self.last_ema <= self.ema_two_checks_prior) or
                     (self.hp.early_stop_method == 'loss'     and self.last_ema >= self.ema_two_checks_prior))):

                    log.info(f'Early stop triggered: epoch {self.epoch_count+1}, batch {batch}')
                    self.model.stop_training = True
                    self.early_stop = True

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

    def on_train_end(self, logs={}):
        if log.getEffectiveLevel() <= 20: print('\r\033[K')
        if self.neptune_run:
            self.neptune_run['sys/tags'].add('training_complete')

    def evaluate_model(self, logs={}):
        epoch = self.epoch_count
        epoch_label = f'val_epoch{epoch}'
        pred_args = types.SimpleNamespace(loss=self.hp.get_loss())
        metrics, acc, loss = sf.statistics.metrics_from_dataset(
            self.model,
            model_type=self.hp.model_type(),
            labels=self.parent.labels,
            patients=self.parent.patients,
            dataset=self.cb_args.validation_data_with_slidenames,
            outcome_names=self.parent.outcome_names,
            label=epoch_label,
            data_dir=self.parent.outdir,
            num_tiles=self.cb_args.num_val_tiles,
            histogram=False,
            verbose=True,
            save_predictions=self.cb_args.save_predictions,
            pred_args=pred_args
        )

        # Note that Keras loss during training includes regularization losses,
        #  so this loss will not match validation loss calculated during training
        val_metrics = {'accuracy': acc, 'loss': loss}
        #val_metrics = self.model.evaluate(self.cb_args.validation_data, verbose=0, return_dict=True)
        log.info(f'Validation metrics: ' + json.dumps(val_metrics, indent=4))
        self.results['epochs'][f'epoch{epoch}'] = {'train_metrics': {k:v for k,v in logs.items() if k[:3] != 'val'},
                                                   'val_metrics': val_metrics }
        for metric in metrics:
            if metrics[metric]['tile'] is None: continue
            self.results['epochs'][f'epoch{epoch}']['tile'] = metrics[metric]['tile']
            self.results['epochs'][f'epoch{epoch}']['slide'] = metrics[metric]['slide']
            self.results['epochs'][f'epoch{epoch}']['patient'] = metrics[metric]['patient']

        epoch_results = self.results['epochs'][f'epoch{epoch}']
        sf.util.update_results_log(self.cb_args.results_log, 'trained_model', {f'epoch{epoch}': epoch_results})

class Trainer:
    """Base trainer class containing functionality for model building, input processing, training, and evaluation.

    This base class requires categorical outcome(s). Additional outcome types are supported by
    :class:`slideflow.model.LinearTrainer` and :class:`slideflow.model.CPHTrainer`.

    Slide-level (e.g. clinical) features can be used as additional model input by providing slide labels
    in the slide annotations dictionary, under the key 'input'.
    """

    _model_type = 'categorical'

    def __init__(self, hp, outdir, labels, patients, slide_input=None, name=None, manifest=None, feature_sizes=None,
                 feature_names=None, outcome_names=None, mixed_precision=True, config=None, use_neptune=False,
                 neptune_api=None, neptune_workspace=None):

        """Sets base configuration, preparing model inputs and outputs.

        Args:
            hp (:class:`slideflow.model.ModelParams`): ModelParams object.
            outdir (str): Location where event logs and checkpoints will be written.
            labels (dict): Dict mapping slide names to outcome labels (int or float format).
            patients (dict): Dict mapping slide names to patient ID, as some patients may have multiple slides.
                If not provided, assumes 1:1 mapping between slide names and patients.
            slide_input (dict): Dict mapping slide names to additional slide-level input, concatenated after post-conv.
            name (str, optional): Optional name describing the model, used for model saving. Defaults to None.
            manifest (dict, optional): Manifest dictionary mapping TFRecords to number of tiles. Defaults to None.
            model_type (str, optional): Type of model outcome, 'categorical' or 'linear'. Defaults to 'categorical'.
            feature_sizes (list, optional): List of sizes of input features. Required if providing additional
                input features as input to the model.
            feature_names (list, optional): List of names for input features. Used when permuting feature importance.
            outcome_names (list, optional): Name of each outcome. Defaults to "Outcome {X}" for each outcome.
            mixed_precision (bool, optional): Use FP16 mixed precision (rather than FP32). Defaults to True.
            config (dict, optional): Training configuration dictionary, used for logging. Defaults to None.
            use_neptune (bool, optional): Use Neptune API logging. Defaults to False
            neptune_api (str, optional): Neptune API token, used for logging. Defaults to None.
            neptune_workspace (str, optional): Neptune workspace, used for logging. Defaults to None.
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
        self.outcome_names = outcome_names
        self.mixed_precision = mixed_precision
        self.name = name
        self.config = config
        self.neptune_run = None

        if patients:
            self.patients = patients
        else:
            self.patients = {s:s for s in self.slides}

        if not os.path.exists(outdir): os.makedirs(outdir)

        # Format outcome labels (ensures compatibility with single and multi-outcome models)
        outcome_labels = np.array(list(labels.values()))
        if len(outcome_labels.shape) == 1:
            outcome_labels = np.expand_dims(outcome_labels, axis=1)
        if not self.outcome_names:
            self.outcome_names = [f'Outcome {i}' for i in range(outcome_labels.shape[1])]
        if not isinstance(self.outcome_names, list):
            self.outcome_names = [self.outcome_names]
        if len(self.outcome_names) != outcome_labels.shape[1]:
            num_names = len(self.outcome_names)
            num_outcomes = outcome_labels.shape[1]
            raise ModelError(f'Size of outcome_names ({num_names}) does not match number of outcomes {num_outcomes}')

        self._setup_inputs()
        self.num_classes = self.hp._detect_classes_from_labels(labels)

        # Normalization setup
        self.normalizer = self.hp.get_normalizer()
        if self.normalizer: log.info(f'Using realtime {self.hp.normalizer} normalization')

        if self.mixed_precision:
            log.debug('Enabling mixed precision')
            policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
            tf.keras.mixed_precision.experimental.set_policy(policy)

        with tf.device('/cpu'):
            self.annotations_tables = []
            for oi in range(outcome_labels.shape[1]):
                self.annotations_tables += [tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(self.slides, outcome_labels[:,oi]), -1
                )]

        # Initialize Neptune
        self.use_neptune = use_neptune
        if self.use_neptune:
            self.neptune_logger = sf.util.neptune_utils.NeptuneLog(neptune_api, neptune_workspace)

    def _setup_inputs(self):
        # Setup slide-level input
        if self.num_slide_features:
            try:
                if self.num_slide_features:
                    log.info(f'Training with both images and {self.num_slide_features} categories of slide-level input')
            except KeyError:
                raise ModelError("Unable to find slide-level input at 'input' key in annotations")
            for slide in self.slides:
                if len(self.slide_input[slide]) != self.num_slide_features:
                    err_msg = f'Length of input for slide {slide} does not match feature_sizes'
                    num_in_feature_table = len(self.slide_input[slide])
                    raise ModelError(f'{err_msg}; expected {self.num_slide_features}, got {num_in_feature_table}')

    def _compile_model(self):
        '''Compiles keras model.'''

        self.model.compile(optimizer=self.hp.get_opt(),
                           loss=self.hp.get_loss(),
                           metrics=['accuracy'])

    def _parse_tfrecord_labels(self, image, slide):
        '''Parses raw entry read from TFRecord.'''

        image_dict = { 'tile_image': image }

        if len(self.num_classes) > 1:
            label = {f'out-{oi}': self.annotations_tables[oi].lookup(slide) for oi in range(len(self.num_classes))}
        else:
            label = self.annotations_tables[0].lookup(slide)

        # Add additional non-image feature inputs if indicated,
        #     excluding the event feature used for CPH models
        if self.num_slide_features:
            def slide_lookup(s): return self.slide_input[s.numpy().decode('utf-8')]
            num_features = self.num_slide_features
            slide_feature_input_val = tf.py_function(func=slide_lookup, inp=[slide], Tout=[tf.float32] * num_features)
            image_dict.update({'slide_feature_input': slide_feature_input_val})

        return image_dict, label

    def _retrain_top_layers(self, train_data, validation_data, steps_per_epoch, callbacks=None, epochs=1):
        '''Retrains only the top layer of this object's model, while leaving all other layers frozen.'''
        log.info('Retraining top layer')
        # Freeze the base layer
        self.model.layers[0].trainable = False
        val_steps = 200 if validation_data else None

        self._compile_model()

        toplayer_model = self.model.fit(train_data,
                                        epochs=epochs,
                                        verbose=(log.getEffectiveLevel() <= 20),
                                        steps_per_epoch=steps_per_epoch,
                                        validation_data=validation_data,
                                        validation_steps=val_steps,
                                        callbacks=callbacks)

        # Unfreeze the base layer
        self.model.layers[0].trainable = True
        return toplayer_model.history

    def _interleave_kwargs(self, **kwargs):
        args = types.SimpleNamespace(
            labels=self._parse_tfrecord_labels,
            normalizer=self.normalizer,
            **kwargs
        )
        return vars(args)

    def _metric_kwargs(self, **kwargs):
        args = types.SimpleNamespace(
            model=self.model,
            model_type=self._model_type,
            labels=self.labels,
            patients=self.patients,
            outcome_names=self.outcome_names,
            data_dir=self.outdir,
            neptune_run=self.neptune_run,
            **kwargs
        )
        return vars(args)

    def load(self, model):
        self.model = tf.keras.models.load_model(model)

    def evaluate(self, dataset, batch_size=None, permutation_importance=False, histogram=False, save_predictions=False):

        """Evaluate model, saving metrics and predictions.

        Args:
            dataset (:class:`slideflow.dataset.Dataset`): Dataset containing TFRecords to evaluate.
            checkpoint (list, optional): Path to cp.cpkt checkpoint to load. Defaults to None.
            batch_size (int, optional): Evaluation batch size. Defaults to the same as training (per self.hp)
            permutation_importance (bool, optional): Run permutation feature importance to define relative benefit
                of histology and each clinical slide-level feature input, if provided.
            histogram (bool, optional): Save histogram of tile predictions. Poorly optimized, uses seaborn, may
                drastically increase evaluation time. Defaults to False.
            save_predictions (bool, optional): Save tile, slide, and patient-level predictions to CSV. Defaults to False.

        Returns:
            Dictionary of evaluation metrics.
        """

        # Load and initialize model
        if not self.model:
            raise sf.util.UserError("Model has not been loaded, unable to evaluate.")
        log_manifest(None, dataset.tfrecords(), self.labels, join(self.outdir, 'slide_manifest.csv'))

        # Neptune logging
        if self.use_neptune:
            self.neptune_run = self.neptune_logger.start_run(self.name, self.config['project'], dataset, tags='eval')
            self.neptune_logger.log_config(self.config, 'eval')
            self.neptune_run['data/slide_manifest'].upload(os.path.join(self.outdir, 'slide_manifest.csv'))

        if not batch_size: batch_size = self.hp.batch_size
        with tf.name_scope('input'):
            interleave_kwargs = self._interleave_kwargs(batch_size=batch_size, infinite=False, augment=False)
            tf_dts = dataset.tensorflow(**interleave_kwargs)
            tf_dts_w_slidenames = dataset.tensorflow(incl_slidenames=True, **interleave_kwargs)

        # Generate performance metrics
        log.info('Calculating performance metrics...')
        metric_kwargs = self._metric_kwargs(dataset=tf_dts_w_slidenames, num_tiles=dataset.num_tiles, label='eval')
        if permutation_importance:
            drop_images = ((self.hp.tile_px == 0) or self.hp.drop_images)
            metrics = sf.statistics.permutation_feature_importance(feature_names=self.feature_names,
                                                                   feature_sizes=self.feature_sizes,
                                                                   drop_images=drop_images,
                                                                   **metric_kwargs)
        else:
            metrics, acc, loss = sf.statistics.metrics_from_dataset(histogram=histogram,
                                                                    verbose=True,
                                                                    save_predictions=save_predictions,
                                                                    pred_args=types.SimpleNamespace(loss=self.hp.get_loss()),
                                                                    **metric_kwargs)
            results_dict = { 'eval': {} }
        for metric in metrics:
            if metrics[metric]:
                log.info(f"Tile {metric}: {metrics[metric]['tile']}")
                log.info(f"Slide {metric}: {metrics[metric]['slide']}")
                log.info(f"Patient {metric}: {metrics[metric]['patient']}")
                results_dict['eval'].update({
                    f'tile_{metric}': metrics[metric]['tile'],
                    f'slide_{metric}': metrics[metric]['slide'],
                    f'patient_{metric}': metrics[metric]['patient']
                })


        # Note that Keras loss during training includes regularization losses,
        #  so this loss will not match validation loss calculated during training
        # val_metrics = self.model.evaluate(tf_dts, verbose=(log.getEffectiveLevel() <= 20), return_dict=True)
        val_metrics = {'accuracy': acc, 'loss': loss}
        results_log = os.path.join(self.outdir, 'results_log.csv')
        log.info(f'Evaluation metrics:')
        for m in val_metrics:
            log.info(f'{m}: {val_metrics[m]}')
        results_dict['eval'].update(val_metrics)
        sf.util.update_results_log(results_log, 'eval_model', results_dict)

        # Update neptune log
        if self.neptune_run:
            self.neptune_run['eval/results'] = val_metrics
            self.neptune_run.stop()

        return val_metrics

    def train(self, train_dts, val_dts, log_frequency=100, validate_on_batch=512, validation_batch_size=32,
              validation_steps=200, starting_epoch=0, ema_observations=20, ema_smoothing=2, use_tensorboard=True,
              steps_per_epoch_override=None, save_predictions=False, resume_training=None,
              pretrain='imagenet', checkpoint=None, multi_gpu=False):

        """Builds and trains a model from hyperparameters.

        Args:
            train_dts (:class:`slideflow.dataset.Dataset`): Dataset containing TFRecords for training.
            val_dts (:class:`slideflow.dataset.Dataset`): Dataset containing TFRecords for validation.
            log_frequency (int, optional): How frequent to update Tensorboard logs, in batches. Defaults to 100.
            validate_on_batch (int, optional): How frequent o perform validation, in batches. Defaults to 512.
            validation_batch_size (int, optional): Batch size to use during validation. Defaults to 32.
            validation_steps (int, optional): Number of batches to use for each instance of validation. Defaults to 200.
            starting_epoch (int, optional): Starts training at the specified epoch. Defaults to 0.
            ema_observations (int, optional): Number of observations over which to perform exponential moving average
                smoothing. Defaults to 20.
            ema_smoothing (int, optional): Exponential average smoothing value. Defaults to 2.
            use_tensoboard (bool, optional): Enable tensorboard callbacks. Defaults to False.
            steps_per_epoch_override (int, optional): Manually set the number of steps per epoch. Defaults to None.
            save_predictions (bool, optional): Save tile, slide, and patient-level predictions at each evaluation.
                Defaults to False.
            resume_training (str, optional): Path to Tensorflow model to continue training. Defaults to None.
            pretrain (str, optional): Either 'imagenet' or path to Tensorflow model from which to load weights.
                Defaults to 'imagenet'.
            checkpoint (str, optional): Path to cp.ckpt from which to load weights. Defaults to None.

        Returns:
            Nested results dictionary containing metrics for each evaluated epoch.
        """

        if self.hp.model_type() != self._model_type:
            raise ModelError(f"Incomptable model types: {self.hp.model_type()} (hp) and {self._model_type} (model)")
        tf.keras.backend.clear_session() # Clear prior Tensorflow graph to free memory

        # Save training / validation manifest
        val_tfrecords = None if val_dts is None else val_dts.tfrecords()
        log_manifest(train_dts.tfrecords(), val_tfrecords, self.labels, join(self.outdir, 'slide_manifest.csv'))

        # Neptune logging
        if self.use_neptune:
            tags = ['train']
            if 'k-fold' in self.config['validation_strategy']:
                tags += [f'k-fold{self.config["k_fold_i"]}']
            self.neptune_run = self.neptune_logger.start_run(self.name, self.config['project'], train_dts, tags=tags)
            self.neptune_logger.log_config(self.config, 'train')
            self.neptune_run['data/slide_manifest'].upload(os.path.join(self.outdir, 'slide_manifest.csv'))

        if multi_gpu:
            strategy = tf.distribute.MirroredStrategy()
            log.info(f'Multi-GPU training with {strategy.num_replicas_in_sync} devices')
        else:
            strategy = None

        with strategy.scope() if strategy else no_scope():
            # Build model from ModeLParams
            if resume_training:
                self.model = tf.keras.load_model(resume_training)
            else:
                model = self.hp.build_model(labels=self.labels,
                                            num_slide_features=self.num_slide_features,
                                            pretrain=pretrain,
                                            checkpoint=checkpoint)
                self.model = model
                log_summary(model, self.neptune_run)

            with tf.name_scope('input'):
                t_kwargs = self._interleave_kwargs(batch_size=self.hp.batch_size, infinite=True, augment=self.hp.augment)
                train_data = train_dts.tensorflow(**t_kwargs)

            # Set up validation data
            using_validation = (val_dts and len(val_dts.tfrecords()))
            if using_validation:
                with tf.name_scope('input'):
                    v_kwargs = self._interleave_kwargs(batch_size=validation_batch_size, infinite=False, augment=False)
                    val_data = val_dts.tensorflow(**v_kwargs)
                    val_data_w_slidenames = val_dts.tensorflow(incl_slidenames=True, **v_kwargs)

                val_log_msg = '' if not validate_on_batch else f'every {str(validate_on_batch)} steps and '
                log.debug(f'Validation during training: {val_log_msg}at epoch end')
                if validation_steps:
                    validation_data_for_training = val_data.repeat()
                    num_samples = validation_steps * self.hp.batch_size
                    log.debug(f'Using {validation_steps} batches ({num_samples} samples) each validation check')
                else:
                    validation_data_for_training = val_data
                    log.debug(f'Using entire validation set each validation check')
            else:
                log.debug('Validation during training: None')
                validation_data_for_training = None
                val_data = None
                val_data_w_slidenames = None
                validation_steps = 0

            # Calculate parameters
            if max(self.hp.epochs) <= starting_epoch:
                max_epoch = max(self.hp.epochs)
                log.error(f'Starting epoch ({starting_epoch}) cannot be greater than the max target epoch ({max_epoch})')
            if self.hp.early_stop and self.hp.early_stop_method == 'accuracy' and self._model_type != 'categorical':
                log.error(f"Unable to use 'accuracy' early stopping with model type '{self.hp.model_type()}'")
            if starting_epoch != 0:
                log.info(f'Starting training at epoch {starting_epoch}')
            total_epochs = self.hp.toplayer_epochs + (max(self.hp.epochs) - starting_epoch)
            if steps_per_epoch_override:
                steps_per_epoch = steps_per_epoch_override
            else:
                steps_per_epoch = round(train_dts.num_tiles/self.hp.batch_size)
            results_log = os.path.join(self.outdir, 'results_log.csv')

            cb_args = types.SimpleNamespace(
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
                results_log=results_log,
            )

            # Create callbacks for early stopping, checkpoint saving, summaries, and history
            history_callback = tf.keras.callbacks.History()
            checkpoint_path = os.path.join(self.outdir, 'cp.ckpt')
            evaluation_callback = _PredictionAndEvaluationCallback(self, cb_args)
            cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                            save_weights_only=True,
                                                            verbose=(log.getEffectiveLevel() <= 20))
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.outdir,
                                                                histogram_freq=0,
                                                                write_graph=False,
                                                                update_freq=log_frequency)

            callbacks = [history_callback, evaluation_callback, cp_callback]
            if use_tensorboard:
                callbacks += [tensorboard_callback]
            if self.neptune_run:
                from neptune.new.integrations.tensorflow_keras import NeptuneCallback
                neptune_cb = NeptuneCallback(run=self.neptune_run, base_namespace='metrics')
                callbacks += [neptune_cb]

            # Retrain top layer only, if using transfer learning and not resuming training
            if self.hp.toplayer_epochs:
                self._retrain_top_layers(train_data, validation_data_for_training, steps_per_epoch,
                                        callbacks=None, epochs=self.hp.toplayer_epochs)

            # Train the model
            self._compile_model()
            log.info('Beginning training')
            #tf.debugging.enable_check_numerics()

            try:
                self.model.fit(train_data,
                                steps_per_epoch=steps_per_epoch,
                                epochs=total_epochs,
                                verbose=(log.getEffectiveLevel() <= 20),
                                initial_epoch=self.hp.toplayer_epochs,
                                validation_data=validation_data_for_training,
                                validation_steps=validation_steps,
                                callbacks=callbacks)
            except tf.errors.ResourceExhaustedError as e:
                print()
                print(e)
                log.error(f"Training failed for {sf.util.bold(self.name)}, GPU memory exceeded.")

            results = evaluation_callback.results
            if self.use_neptune:
                self.neptune_run['results/logged_epochs'] = [int(e[5:]) for e in results['epochs'] if e[:5] == 'epoch']
                self.neptune_run['results/epochs'] = results['epochs']
                self.neptune_run.stop()

            return results

class LinearTrainer(Trainer):

    """Extends the base :class:`slideflow.model.Trainer` class to add support for linear outcomes. Requires that all
    outcomes be linear, with appropriate linear loss function. Uses R-squared as the evaluation metric, rather
    than AUROC."""

    _model_type = 'linear'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compile_model(self):
        self.model.compile(optimizer=self.hp.get_opt(),
                           loss=self.hp.get_loss(),
                           metrics=[self.hp.get_loss()])

    def _parse_tfrecord_labels(self, image, slide):
        image_dict = { 'tile_image': image }
        label = [self.annotations_tables[oi].lookup(slide) for oi in range(self.num_classes)]

        # Add additional non-image feature inputs if indicated,
        #     excluding the event feature used for CPH models
        if self.num_slide_features:
            def slide_lookup(s): return self.slide_input[s.numpy().decode('utf-8')]
            num_features = self.num_slide_features
            slide_feature_input_val = tf.py_function(func=slide_lookup, inp=[slide], Tout=[tf.float32] * num_features)
            image_dict.update({'slide_feature_input': slide_feature_input_val})

        return image_dict, label

class CPHTrainer(LinearTrainer):

    """Cox Proportional Hazards model. Requires that the user provide event data as the first input feature,
    and time to outcome as the linear outcome. Uses concordance index as the evaluation metric."""

    _model_type = 'cph'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.num_slide_features:
            raise ModelError('Model error - CPH models must include event input')

    def _setup_inputs(self):
        # Setup slide-level input
        try:
            num_features = self.num_slide_features - 1
            if num_features:
                log.info(f'Training with both images and {num_features} categories of slide-level input')
                log.info('Interpreting first feature as event for CPH model')
            else:
                log.info(f'Training with images alone. Interpreting first feature as event for CPH model')
        except KeyError:
            raise ModelError("Unable to find slide-level input at 'input' key in annotations")
        for slide in self.slides:
            if len(self.slide_input[slide]) != self.num_slide_features:
                err_msg = f'Length of input for slide {slide} does not match feature_sizes'
                num_in_feature_table = len(self.slide_input[slide])
                raise ModelError(f'{err_msg}; expected {self.num_slide_features}, got {num_in_feature_table}')

    def load(self, model):
        custom_objects = {'negative_log_likelihood':negative_log_likelihood,
                          'concordance_index':concordance_index}
        self.model = tf.keras.models.load_model(model, custom_objects=custom_objects)
        self.model.compile(loss=negative_log_likelihood, metrics=concordance_index)

    def _compile_model(self):
        self.model.compile(optimizer=self.hp.get_opt(),
                           loss=negative_log_likelihood,
                           metrics=concordance_index)

    def _parse_tfrecord_labels(self, image, slide):
        image_dict = { 'tile_image': image }
        label = [self.annotations_tables[oi].lookup(slide) for oi in range(self.num_classes)]

        # Add additional non-image feature inputs if indicated,
        #     excluding the event feature used for CPH models
        if self.num_slide_features:
            # Time-to-event data must be added as a separate feature
            def slide_lookup(s): return self.slide_input[s.numpy().decode('utf-8')][1:]
            def event_lookup(s): return self.slide_input[s.numpy().decode('utf-8')][0]
            num_features = self.num_slide_features - 1

            event_input_val = tf.py_function(func=event_lookup, inp=[slide], Tout=[tf.float32])
            image_dict.update({'event_input': event_input_val})

            slide_feature_input_val = tf.py_function(func=slide_lookup, inp=[slide], Tout=[tf.float32] * num_features)

            # Add slide input features, excluding the event feature used for CPH models
            if not (self.num_slide_features == 1):
                image_dict.update({'slide_feature_input': slide_feature_input_val})
        return image_dict, label

class Features:
    """Interface for obtaining logits and features from intermediate layer activations from Slideflow models.

    Use by calling on either a batch of images (returning outputs for a single batch), or by calling on a
    :class:`slideflow.WSI` object, which will generate an array of spatially-mapped activations matching
    the slide.

    Examples
        *Calling on batch of images:*

        .. code-block:: python

            interface = Features('/model/path', layers='postconv')
            for image_batch in train_data:
                # Return shape: (batch_size, num_features)
                batch_features = interface(image_batch)

        *Calling on a slide:*

        .. code-block:: python

            slide = sf.slide.WSI(...)
            interface = Features('/model/path', layers='postconv')
            # Return shape: (slide.grid.shape[0], slide.grid.shape[1], num_features):
            activations_grid = interface(slide)

    Note:
        When this interface is called on a batch of images, no image processing or stain normalization will be
        performed, as it is assumed that normalization will occur during data loader image processing.
        When the interface is called on a `slideflow.WSI`, the normalization strategy will be read from the model
        configuration file, and normalization will be performed on image tiles extracted from the WSI. If this interface
        was created from an existing model and there is no model configuration file to read, a
        slideflow.slide.StainNormalizer object may be passed during initialization via the argument `wsi_normalizer`.

    """

    def __init__(self, path, layers='postconv', include_logits=False):
        """Creates a features interface from a saved slideflow model which outputs feature activations
        at the designated layers.

        Intermediate layers are returned in the order of layers. Logits are returned last.

        Args:
            path (str): Path to saved Slideflow model.
            layers (list(str), optional): Layers from which to generate activations.  The post-convolution activation layer
                is accessed via 'postconv'. Defaults to 'postconv'.
            include_logits (bool, optional): Include logits in output. Will be returned last. Defaults to False.
        """

        if layers and not isinstance(layers, list): layers = [layers]
        self.path = path
        self.num_logits = 0
        self.num_features = 0
        if path is not None:
            self._model = tf.keras.models.load_model(self.path)
            try:
                config = sf.util.get_model_config(path)
            except:
                log.warning(f"Unable to find configuration for model {path}; unable to determine normalization " + \
                            "strategy for WSI processing.")

            self.hp = sf.model.ModelParams()
            self.hp.load_dict(config['hp'])
            self.wsi_normalizer = self.hp.get_normalizer()
            self._build(layers=layers, include_logits=include_logits)

    @classmethod
    def from_model(cls, model, layers='postconv', include_logits=False, wsi_normalizer=None):
        """Creates a features interface from a loaded slideflow model which outputs feature activations
        at the designated layers.

        Intermediate layers are returned in the order of layers. Logits are returned last.

        Args:
            model (:class:`tensorflow.keras.models.Model`): Loaded model.
            layers (list(str), optional): Layers from which to generate activations.  The post-convolution activation layer
                is accessed via 'postconv'. Defaults to 'postconv'.
            include_logits (bool, optional): Include logits in output. Will be returned last. Defaults to False.
            wsi_normalizer (:class:`slideflow.slide.StainNormalizer`): Stain normalizer to use on whole-slide images.
                Is not used on individual tile datasets via __call__. Defaults to None.
        """

        obj = cls(None, layers, include_logits)
        if isinstance(model, tf.keras.models.Model):
            obj._model = model
        else:
            raise TypeError("Provided model is not a valid Tensorflow model.")
        obj._build(layers=layers, include_logits=include_logits)
        obj.wsi_normalizer = wsi_normalizer
        return obj

    def __call__(self, inp, **kwargs):
        """Process a given input and return features and/or logits. Expects either a batch of images or
        a :class:`slideflow.slide.WSI` object."""

        if isinstance(inp, sf.slide.WSI):
            return self._predict_slide(inp, **kwargs)
        else:
            return self._predict(inp)

    def _predict_slide(self, slide, batch_size=128, dtype=np.float16, **kwargs):
        """Generate activations from slide => activation grid array."""
        total_out = self.num_features + self.num_logits
        features_grid = np.zeros((slide.grid.shape[1], slide.grid.shape[0], total_out), dtype=dtype)
        generator = slide.build_generator(shuffle=False, include_loc='grid', show_progress=True, **kwargs)

        if not generator:
            log.error(f"No tiles extracted from slide {sf.util.green(slide.name)}")
            return

        def _parse_function(record):
            image = record['image']
            loc = record['loc']
            parsed_image = tf.image.per_image_standardization(image)
            parsed_image.set_shape([slide.tile_px, slide.tile_px, 3])
            if self.wsi_normalizer:
                image = tf.py_function(self.wsi_normalizer.tf_to_rgb, [image], tf.int32)
            return parsed_image, loc

        # Generate dataset from the generator
        with tf.name_scope('dataset_input'):
            output_signature={'image':tf.TensorSpec(shape=(slide.tile_px,slide.tile_px,3), dtype=tf.uint8),
                              'loc':tf.TensorSpec(shape=(2), dtype=tf.uint32)}
            tile_dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
            tile_dataset = tile_dataset.map(_parse_function, num_parallel_calls=8)
            tile_dataset = tile_dataset.batch(batch_size, drop_remainder=False)
            tile_dataset = tile_dataset.prefetch(8)

        act_arr = []
        loc_arr = []
        for i, (batch_images, batch_loc) in enumerate(tile_dataset):
            model_out = self._predict(batch_images)
            if not isinstance(model_out, list): model_out = [model_out]
            act_arr += [np.concatenate([m.numpy() for m in model_out])]
            loc_arr += [batch_loc.numpy()]

        act_arr = np.concatenate(act_arr)
        loc_arr = np.concatenate(loc_arr)

        for i, act in enumerate(act_arr):
            xi = loc_arr[i][0]
            yi = loc_arr[i][1]
            features_grid[yi][xi] = act

        return features_grid

    @tf.function
    def _predict(self, inp):
        """Return activations for a single batch of images."""
        return self.model(inp, training=False)

    def _build(self, layers, include_logits=True):
        """Builds the interface model that outputs feature activations at the designated layers and/or logits.
            Intermediate layers are returned in the order of layers. Logits are returned last."""

        if layers:
            log.debug(f"Setting up interface to return activations from layers {', '.join(layers)}")
            other_layers = [l for l in layers if l != 'postconv']
        else:
            other_layers = []
        outputs = {}
        if layers:
            intermediate_core = tf.keras.models.Model(inputs=self._model.layers[1].input,
                                                      outputs=[self._model.layers[1].get_layer(l).output for l in other_layers])
            if len(other_layers) > 1:
                int_out = intermediate_core(self._model.input)
                for l, layer in enumerate(other_layers):
                    outputs[layer] = int_out[l]
            elif len(other_layers):
                outputs[other_layers[0]] = intermediate_core(self._model.input)
            if 'postconv' in layers:
                outputs['postconv'] = self._model.layers[1].get_output_at(0)
        outputs_list = [] if not layers else [outputs[l] for l in layers]
        if include_logits:
            outputs_list += [self._model.output]
        self.model = tf.keras.models.Model(inputs=self._model.input, outputs=outputs_list)
        self.num_features = sum([outputs[o].shape[1] for o in outputs])
        if isinstance(self._model.output, list):
            log.warning("Multi-categorical outcomes not yet supported for this interface.")
            self.num_logits = 0
        else:
            self.num_logits = 0 if not include_logits else self._model.output.shape[1]
        if include_logits:
            log.debug(f'Number of logits: {self.num_logits}')
        log.debug(f'Number of activation features: {self.num_features}')