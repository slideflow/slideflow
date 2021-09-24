# Copyright (C) James Dolezal - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, September 2019
# ==========================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import warnings
import shutil
import json
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
import slideflow as sf

from tqdm import tqdm
from slideflow.util import log
from slideflow.io.tfrecords import interleave_tfrecords
from slideflow.model_utils import *
from slideflow.util import StainNormalizer

BALANCE_BY_CATEGORY = 'BALANCE_BY_CATEGORY'
BALANCE_BY_PATIENT = 'BALANCE_BY_PATIENT'
NO_BALANCE = 'NO_BALANCE'
MODEL_FORMAT_1_9 = '1.9'
MODEL_FORMAT_CURRENT = MODEL_FORMAT_1_9
MODEL_FORMAT_LEGACY = 'legacy'

#TODO: Fix ModelActivationsInterface for multiple categorical outcomes
#TODO: make different Model class for each type of input & overload base class
#TODO: convert annotations_tables to standard python lookup dicts

class ModelActivationsInterface:
    '''Provides an interface to obtain logits and post-convolutional activations
        from saved Slideflow Keras models. Provides support for newer models (v1.9.1+)
        and legacy slideflow models (1.9.0b and earlier)'''

    def __init__(self, path, model_format=None, layers='postconv', include_logits=True):
        '''Initializer.

        Args:
            path:           Path to saved Slideflow Keras model
            model_format:   Either slideflow.model.MODEL_FORMAT_CURRENT or _LEGACY.
                                Indicates how the saved model should be processed,
                                as older versions of Slideflow had models constructed differently,
                                with differing naming of Keras layers.
            layers:         Layers from which to generate activations.
                                The post-convolution activation layer is accessed via 'postconv'
        '''
        if not model_format: model_format = MODEL_FORMAT_CURRENT
        if not isinstance(layers, list): layers = [layers]
        self.model_format = model_format
        self.path = path
        self.hp = sf.util.get_model_hyperparameters(path)
        self.num_classes = 0
        self.num_features = 0
        self.duplicate_input = (model_format == MODEL_FORMAT_LEGACY)
        self._model = tf.keras.models.load_model(self.path)
        self._build(layers=[l for l in layers if l != 'postconv'],
                    include_postconv=('postconv' in layers),
                    include_logits=include_logits)

    def _build(self,
               layers,
               pooling=None,
               include_postconv=False,
               include_logits=True):

        '''Builds a model that outputs feature activations at the designated layers
            and concatenates into a single final output vector.'''

        if not layers:
            log.debug('Setting up interface to return activations from post-conv activations')
        else:
            msg_tail = 'and post-conv activations' if include_postconv else ''
            log.debug(f"Setting up interface to return activations from layers {', '.join(layers)} {msg_tail}")

        assert layers or include_postconv
        if pooling:
            assert type(pooling) == list and len(pooling) == len(layers)

        if self.model_format == MODEL_FORMAT_1_9:
            model_core = self._model.layers[1]
        if self.model_format == MODEL_FORMAT_LEGACY:
            model_core = self._model.layers[0]

        self.duplicate_input = True #TODO: Why is this here?

        layer_sources = {}
        if layers:
            for layer_name in layers:
                if layer_name in [l.name for l in self._model.layers]:
                    layer_sources[layer_name] = self._model
                elif layer_name in [l.name for l in model_core.layers]:
                    layer_sources[layer_name] = model_core
                elif self.model_format != MODEL_FORMAT_LEGACY:
                    log.warning('Unable to read model using modern format, will try using legacy format')
                    self.model_format = MODEL_FORMAT_LEGACY
                    self._build(layers, include_postconv)
                    return
                else:
                    raise ModelError(f'Unable to read model: could not find layer {layer_name}')
            raw_output_layers = [layer_sources[l].get_layer(l).output for l in layers]

            if pooling:
                activation_layers = []
                for idx, al in enumerate(raw_output_layers):
                    if pooling[idx]:
                        activation_layers += [tf.keras.layers.GlobalAveragePooling2D(name=f'act_pooling_{idx}')(al)]
                    else:
                        activation_layers += [al]
            activation_layers = raw_output_layers
        else:
            activation_layers = []

        if include_postconv:
            activation_layers += [model_core.output]

        if len(activation_layers) == 1:
            merged_output = activation_layers[0]
        else:
            merged_output = tf.keras.layers.Concatenate(axis=1)(activation_layers)

        if include_logits:
            outputs = [merged_output, self._model.output]
        else:
            outputs = [merged_output]

        try:
            self.model = tf.keras.models.Model(inputs=[self._model.input, model_core.input],
                                               outputs=outputs)
        except AttributeError:
            if self.model_format != MODEL_FORMAT_LEGACY:
                log.warning('Unable to read model using modern format, will try using legacy format')
                self.model_format = MODEL_FORMAT_LEGACY
                self._build(layers, include_postconv=include_postconv)
                return
            else:
                raise ModelError('Unable to read model.')

        if include_logits:
            self.num_features = self.model.output_shape[0][1]
            self.num_classes = self.model.output_shape[1][1]
            log.debug(f'Number of logits: {self.num_classes}')
        else:
            self.num_features = self.model.output_shape[1]
        log.debug(f'Number of activation features: {self.num_features}')

    @tf.function
    def _predict(self, inp):
        return self.model(inp, training=False)

    def predict(self, image_batch):
        '''Given a batch of images, will return a batch of post-convolutional activations and a batch of logits.'''

        if self.duplicate_input:
            return self._predict([image_batch, image_batch])
        else:
            return self._predict(image_batch)

class HyperParameters:
    '''Object to supervise construction of a set of hyperparameters for Slideflow models.'''
    _OptDict = {
        'Adam':    tf.keras.optimizers.Adam,
        'SGD': tf.keras.optimizers.SGD,
        'RMSprop': tf.keras.optimizers.RMSprop,
        'Adagrad': tf.keras.optimizers.Adagrad,
        'Adadelta': tf.keras.optimizers.Adadelta,
        'Adamax': tf.keras.optimizers.Adamax,
        'Nadam': tf.keras.optimizers.Nadam
    }
    _ModelDict = {
        'Xception': tf.keras.applications.Xception,
        'VGG16': tf.keras.applications.VGG16,
        'VGG19': tf.keras.applications.VGG19,
        'ResNet50': tf.keras.applications.ResNet50,
        'ResNet101': tf.keras.applications.ResNet101,
        'ResNet152': tf.keras.applications.ResNet152,
        'ResNet50V2': tf.keras.applications.ResNet50V2,
        'ResNet101V2': tf.keras.applications.ResNet101V2,
        'ResNet152V2': tf.keras.applications.ResNet152V2,
        #'ResNeXt50': tf.keras.applications.ResNeXt50,
        #'ResNeXt101': tf.keras.applications.ResNeXt101,
        'InceptionV3': tf.keras.applications.InceptionV3,
        'NASNetLarge': tf.keras.applications.NASNetLarge,
        'InceptionResNetV2': tf.keras.applications.InceptionResNetV2,
        'MobileNet': tf.keras.applications.MobileNet,
        'MobileNetV2': tf.keras.applications.MobileNetV2,
        #'DenseNet': tf.keras.applications.DenseNet,
        #'NASNet': tf.keras.applications.NASNet
    }
    _LinearLoss = ['mean_squared_error',
                   'mean_absolute_error',
                   'mean_absolute_percentage_error',
                   'mean_squared_logarithmic_error',
                   'squared_hinge',
                   'hinge',
                   'logcosh',
                   'negative_log_likelihood']

    _AllLoss = ['mean_squared_error',
                'mean_absolute_error',
                'mean_absolute_percentage_error',
                'mean_squared_logarithmic_error',
                'squared_hinge',
                'hinge'
                'categorical_hinge',
                'logcosh',
                'huber_loss',
                'categorical_crossentropy',
                'sparse_categorical_crossentropy',
                'binary_crossentropy',
                'kullback_leibler_divergence',
                'poisson',
                'cosine_proximity',
                'is_categorical_crossentropy',
                'negative_log_likelihood']

    def __init__(self, tile_px=299, tile_um=302, finetune_epochs=10, toplayer_epochs=0,
                 model='Xception', pooling='max', loss='sparse_categorical_crossentropy',
                 learning_rate=0.0001, learning_rate_decay=0, learning_rate_decay_steps=100000,
                 batch_size=16, hidden_layers=1, hidden_layer_width=500, optimizer='Adam',
                 early_stop=False, early_stop_patience=0, early_stop_method='loss',
                 balanced_training=BALANCE_BY_CATEGORY, balanced_validation=NO_BALANCE,
                 trainable_layers=0, L2_weight=0, dropout=0, augment=True, drop_images=False):

        # Additional hyperparameters to consider:
        # beta1 0.9
        # beta2 0.999
        # epsilon 1.0
        # batch_norm_decay 0.99

        # Assert provided hyperparameters are valid
        assert isinstance(tile_px, int)
        assert isinstance(tile_um, int)
        assert isinstance(toplayer_epochs, int)
        assert isinstance(finetune_epochs, (int, list))
        if isinstance(finetune_epochs, list):
            assert all([isinstance(t, int) for t in finetune_epochs])
        assert model in self._ModelDict.keys()
        assert pooling in ['max', 'avg', 'none']
        assert loss in self._AllLoss
        assert isinstance(learning_rate, float)
        assert isinstance(learning_rate_decay, (int, float))
        assert isinstance(learning_rate_decay_steps, (int))
        assert isinstance(batch_size, int)
        assert isinstance(hidden_layers, int)
        assert optimizer in self._OptDict.keys()
        assert isinstance(early_stop, bool)
        assert isinstance(early_stop_patience, int)
        assert early_stop_method in ['loss', 'accuracy']
        assert balanced_training in [BALANCE_BY_CATEGORY, BALANCE_BY_PATIENT, NO_BALANCE]
        assert isinstance(hidden_layer_width, int)
        assert isinstance(trainable_layers, int)
        assert isinstance(L2_weight, (int, float))
        assert isinstance(dropout, (int, float))
        assert isinstance(augment, bool)
        assert isinstance(drop_images, bool)

        assert 0 <= learning_rate_decay <= 1
        assert 0 <= L2_weight <= 1
        assert 0 <= dropout <= 1

        self.tile_px = tile_px
        self.tile_um = tile_um
        self.toplayer_epochs = toplayer_epochs
        self.finetune_epochs = finetune_epochs if isinstance(finetune_epochs, list) else [finetune_epochs]
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
        self.balanced_training = balanced_training
        self.balanced_validation = balanced_validation
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
                                                                            'get_model',
                                                                            'model_type',
                                                                            'validate',
                                                                            'get_dict',
                                                                            'load_dict']]
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

    def validate(self):
        '''Ensures that hyperparameter combinations are valid.'''
        if (self.model_type() != 'categorical' and ((self.balanced_training == BALANCE_BY_CATEGORY) or
                                                    (self.balanced_validation == BALANCE_BY_CATEGORY))):
            raise HyperParameterError(f'Cannot combine category-level balancing with model type "{self.model_type()}".')
        return True

    def get_opt(self):
        '''Returns optimizer with appropriate learning rate.'''
        if self.learning_rate_decay not in (0, 1):
            initial_learning_rate = self.learning_rate
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=self.learning_rate_decay_steps,
                decay_rate=self.learning_rate_decay,
                staircase=True
            )
            return self._OptDict[self.optimizer](learning_rate=lr_schedule)
        else:
            return self._OptDict[self.optimizer](lr=self.learning_rate)

    def get_model(self, input_tensor=None, weights=None):
        '''Returns a Keras model of the appropriate architecture, input shape, pooling, and initial weights.'''
        if self.model == 'NASNetLarge':
            input_shape = (self.tile_px, self.tile_px, 3)
        else:
            input_shape = None
        return self._ModelDict[self.model](
            input_shape=input_shape,
            input_tensor=input_tensor,
            include_top=False,
            pooling=self.pooling,
            weights=weights
        )

    def model_type(self):
        '''Returns either 'linear' or 'categorical' depending on the loss type.'''
        if self.loss == 'negative_log_likelihood':
            return 'cph'
        elif self.loss in self._LinearLoss:
            return 'linear'
        else:
            return 'categorical'

class Model:
    ''' Model containing all functions necessary to build input dataset pipelines,
    build a training and validation set model, and monitor and execute training.'''

    def __init__(self, outdir, tile_px, annotations, train_tfrecords, val_tfrecords, name=None, manifest=None,
                 model_type='categorical', feature_sizes=None, feature_names=None, normalizer=None,
                 normalizer_source=None, outcome_names=None, mixed_precision=True):

        """Model initializer.

        Args:
            outdir (str): Location where event logs and checkpoints will be written.
            tile_px (int): Int, width/height of input image in pixels.
            annotations (dict): Nested dict, mapping slide names to a dict with patient name (key 'submitter_id'),
                outcome labels (key 'outcome_label'), and any additional slide-level inputs (key 'input').
            train_tfrecords (list): List of tfrecord paths for training.
            val_tfrecords (list): List of tfrecord paths for validation.
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

        self.outdir = outdir
        self.manifest = manifest
        self.tile_px = tile_px
        self.annotations = annotations
        self.train_tfrecords = train_tfrecords
        self.val_tfrecords = val_tfrecords
        self.model_type = model_type
        self.slides = list(annotations.keys())
        self.feature_names = feature_names
        self.feature_sizes = feature_sizes
        self.num_slide_features = 0 if not feature_sizes else sum(feature_sizes)
        self.outcome_names = outcome_names
        self.mixed_precision = mixed_precision
        self.name = name

        if not os.path.exists(outdir): os.makedirs(outdir)

        # Format outcome labels (ensures compatibility with single and multi-outcome models)
        outcome_labels = np.array([annotations[slide]['outcome_label'] for slide in self.slides])
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

        if model_type not in ('categorical', 'linear', 'cph'):
            raise ModelError(f'Unknown model type {model_type}')

        # Setup slide-level input
        if self.num_slide_features:
            try:
                self.slide_feature_table = {slide: annotations[slide]['input'] for slide in self.slides}
                num_features = self.num_slide_features if model_type != 'cph' else self.num_slide_features - 1
                if num_features:
                    log.info(f'Training with both images and {num_features} categories of slide-level input')
                    if model_type == 'cph':
                        log.info('Interpreting first feature as event for CPH model')
                elif model_type == 'cph':
                    log.info(f'Training with images alone. Interpreting first feature as event for CPH model')
            except KeyError:
                raise ModelError("Unable to find slide-level input at 'input' key in annotations")
            for slide in self.slides:
                if len(self.slide_feature_table[slide]) != self.num_slide_features:
                    err_msg = f'Length of input for slide {slide} does not match feature_sizes'
                    num_in_feature_table = len(self.slide_feature_table[slide])
                    raise ModelError(f'{err_msg}; expected {self.num_slide_features}, got {num_in_feature_table}')

        # Normalization setup
        if normalizer: log.info(f'Using realtime {normalizer} normalization')
        self.normalizer = None if not normalizer else StainNormalizer(method=normalizer, source=normalizer_source)

        # Set up number of outcome classes
        if model_type in ['linear', 'cph']:
            try:
                self.num_classes = outcome_labels.shape[1]
            except TypeError:
                raise ModelError('Incorrect formatting of outcome labels for linear model; must be an ndarray.')
        elif model_type == 'categorical':
            self.num_classes = {i: np.unique(outcome_labels[:,i]).shape[0] for i in range(outcome_labels.shape[1])}

        with tf.device('/cpu'):
            self.annotations_tables = []
            for oi in range(outcome_labels.shape[1]):
                self.annotations_tables += [tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(self.slides, outcome_labels[:,oi]), -1
                )]

    def _build_model(self, hp, pretrain=None, checkpoint=None):
        ''' Assembles base model, using pretraining (imagenet) or the base layers of a supplied model.

        Args:
            hp:            HyperParameters object
            pretrain:    Either 'imagenet' or path to model to use as pretraining
            checkpoint:    Path to checkpoint from which to resume model training
        '''
        if self.mixed_precision:
            log.debug('Training with mixed precision')
            policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
            tf.keras.mixed_precision.experimental.set_policy(policy)

        # Setup inputs
        image_shape = (self.tile_px, self.tile_px, 3)
        tile_input_tensor = tf.keras.Input(shape=image_shape, name='tile_image')
        if self.num_slide_features:
            if self.model_type == 'cph':
                event_input_tensor = tf.keras.Input(shape=(1), name='event_input')
                # Add slide feature input tensors, if there are more slide features
                #    than just the event input tensor for CPH models
                if not ((self.num_slide_features == 1) and (self.model_type == 'cph')):
                    slide_feature_input_tensor = tf.keras.Input(shape=(self.num_slide_features - 1),
                                                                name='slide_feature_input')
            else:
                slide_feature_input_tensor = tf.keras.Input(shape=(self.num_slide_features),
                                                            name='slide_feature_input')
        if self.model_type == 'cph' and not self.num_slide_features:
            raise ModelError('Model error - CPH models must include event input')

        # Load pretrained model if applicable
        if pretrain: log.info(f'Using pretraining from {sf.util.green(pretrain)}')
        if pretrain and pretrain!='imagenet':
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
            # Create core model
            base_model = hp.get_model(weights=pretrain)

        # Add L2 regularization to all compatible layers in the base model
        if hp.L2_weight != 0:
            regularizer = tf.keras.regularizers.l2(hp.L2_weight)
            base_model = add_regularization(base_model, regularizer)
        else:
            regularizer = None

        # Allow only a subset of layers in the base model to be trainable
        if hp.trainable_layers != 0:
            freezeIndex = int(len(base_model.layers) - (hp.trainable_layers - 1 ))# - hp.hidden_layers - 1))
            log.info(f'Only training on last {hp.trainable_layers} layers (of {len(base_model.layers)} total)')
            for layer in base_model.layers[:freezeIndex]:
                layer.trainable = False

        # Create sequential tile model:
        #     tile image --> convolutions --> pooling/flattening --> hidden layers ---> prelogits --> softmax/logits
        #                             additional slide input --/

        # This is an identity layer that simply returns the last layer, allowing us to name and access this layer later
        post_convolution_identity_layer = tf.keras.layers.Lambda(lambda x: x, name='post_convolution')
        layers = [tile_input_tensor, base_model]
        if not hp.pooling:
            layers += [tf.keras.layers.Flatten()]
        layers += [post_convolution_identity_layer]
        if hp.dropout:
            layers += [tf.keras.layers.Dropout(hp.dropout)]
        tile_image_model = tf.keras.Sequential(layers)
        model_inputs = [tile_image_model.input]

        # Merge layers
        if self.num_slide_features:
            # Add images
            if (hp.tile_px == 0) or hp.drop_images:
                log.info('Generating model with just clinical variables and no images')
                merged_model = slide_feature_input_tensor
                model_inputs += [slide_feature_input_tensor]
            elif not ((self.num_slide_features == 1) and (self.model_type == 'cph')):
                # Add slide feature input tensors, if there are more slide features
                #    than just the event input tensor for CPH models
                merged_model = tf.keras.layers.Concatenate(name='input_merge')([slide_feature_input_tensor,
                                                                                tile_image_model.output])
                model_inputs += [slide_feature_input_tensor]
            else:
                merged_model = tile_image_model.output

            # Add event tensor if this is a CPH model
            if self.model_type == 'cph':
                model_inputs += [event_input_tensor]
        else:
            merged_model = tile_image_model.output

        # Add hidden layers
        for i in range(hp.hidden_layers):
            merged_model = tf.keras.layers.Dense(hp.hidden_layer_width,
                                                 name=f'hidden_{i}',
                                                 activation='relu',
                                                 kernel_regularizer=regularizer)(merged_model)

        # Add the softmax prediction layer
        if hp.model_type() in ['linear', 'cph']:
            activation = 'linear'
        else:
            activation = 'softmax'
        log.debug(f'Using {activation} activation')

        # Multi-categorical outcomes
        if type(self.num_classes) == dict:
            outputs = []
            for c in self.num_classes:
                final_dense_layer = tf.keras.layers.Dense(self.num_classes[c],
                                                          kernel_regularizer=regularizer,
                                                          name=f'prelogits-{c}')(merged_model)

                outputs += [tf.keras.layers.Activation(activation, dtype='float32', name=f'out-{c}')(final_dense_layer)]

        else:
            final_dense_layer = tf.keras.layers.Dense(self.num_classes,
                                                      kernel_regularizer=regularizer,
                                                      name='prelogits')(merged_model)


            outputs = [tf.keras.layers.Activation(activation, dtype='float32', name='output')(final_dense_layer)]

        if self.model_type == 'cph':
            outputs[0] = tf.keras.layers.Concatenate(name='output_merge_CPH',
                                                     dtype='float32')([outputs[0], event_input_tensor])

        # Assemble final model
        model = tf.keras.Model(inputs=model_inputs, outputs=outputs)

        if checkpoint:
            log.info(f'Loading checkpoint weights from {sf.util.green(checkpoint)}')
            model.load_weights(checkpoint)

        # Print model summary
        if log.getEffectiveLevel() <= 20:
            print()
            model.summary()

        return model

    def _compile_model(self, hp):
        '''Compiles keras model.

        Args:
            hp        Hyperparameter object.
        '''

        if self.model_type == 'cph':
            metrics = concordance_index
        elif self.model_type == 'linear':
            metrics = [hp.loss]
        else:
            metrics = ['accuracy']

        loss_fn = negative_log_likelihood if self.model_type=='cph' else hp.loss
        self.model.compile(optimizer=hp.get_opt(),
                           loss=loss_fn,
                           metrics=metrics)

    def _parse_tfrecord_labels(self, record, base_parser, include_slidenames=True):
        '''Parses raw entry read from TFRecord.'''

        # Note: multi-image functionality removed in version 1.11 due to lack of use and changes in tfrecord processing
        # If desired, this can be re-added at this stage by simply returning multiple images in the resulting image_dict

        slide, image = base_parser(record)
        image_dict = { 'tile_image': image }

        if self.model_type in ['linear', 'cph']:
            label = [self.annotations_tables[oi].lookup(slide) for oi in range(self.num_classes)]
        elif len(self.num_classes) > 1:
            label = {f'out-{oi}': self.annotations_tables[oi].lookup(slide) for oi in range(len(self.num_classes))}
        else:
            label = self.annotations_tables[0].lookup(slide)

        # Add additional non-image feature inputs if indicated,
        #     excluding the event feature used for CPH models
        if self.num_slide_features:
            # If CPH model is used, time-to-event data must be added as a separate feature
            if self.model_type == 'cph':
                def slide_lookup(s): return self.slide_feature_table[s.numpy().decode('utf-8')][1:]
                def event_lookup(s): return self.slide_feature_table[s.numpy().decode('utf-8')][0]
                num_features = self.num_slide_features - 1

                event_input_val = tf.py_function(func=event_lookup, inp=[slide], Tout=[tf.float32])
                image_dict.update({'event_input': event_input_val})
            else:
                def slide_lookup(s): return self.slide_feature_table[s.numpy().decode('utf-8')]
                num_features = self.num_slide_features

            slide_feature_input_val = tf.py_function(func=slide_lookup, inp=[slide], Tout=[tf.float32] * num_features)

            # Add slide input features, excluding the event feature used for CPH models
            if not ((self.num_slide_features == 1) and (self.model_type == 'cph')):
                image_dict.update({'slide_feature_input': slide_feature_input_val})

        if include_slidenames:
            return image_dict, label, slide
        else:
            return image_dict, label

    def _retrain_top_layers(self, hp, train_data, validation_data, steps_per_epoch, callbacks=None, epochs=1):
        '''Retrains only the top layer of this object's model, while leaving all other layers frozen.'''
        log.info('Retraining top layer')
        # Freeze the base layer
        self.model.layers[0].trainable = False
        val_steps = 200 if validation_data else None

        self._compile_model(hp)

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

    def save_manifest(self):
        # Record which slides are used for training and validation, and to which categories they belong
        if self.train_tfrecords or self.val_tfrecords:
            with open(os.path.join(self.outdir, 'slide_manifest.log'), 'w') as slide_manifest:
                writer = csv.writer(slide_manifest)
                writer.writerow(['slide', 'dataset', 'outcome_label'])
                if self.train_tfrecords:
                    for tfrecord in self.train_tfrecords:
                        slide = tfrecord.split('/')[-1][:-10]
                        if slide in self.slides:
                            outcome_label = self.annotations[slide]['outcome_label']
                            writer.writerow([slide, 'training', outcome_label])
                if self.val_tfrecords:
                    for tfrecord in self.val_tfrecords:
                        slide = tfrecord.split('/')[-1][:-10]
                        if slide in self.slides:
                            outcome_label = self.annotations[slide]['outcome_label']
                            writer.writerow([slide, 'validation', outcome_label])

    def evaluate(self, tfrecords, hp=None, model=None, model_type='categorical', checkpoint=None, batch_size=None,
                    max_tiles_per_slide=0, min_tiles_per_slide=0, permutation_importance=False,
                    histogram=False, save_predictions=False):
        '''Evaluate model.

        Args:
            tfrecords:              List of TFrecords paths to load for evaluation.
            hp:                     HyperParameters object
            model:                  Optional; Tensorflow model to load for evaluation.
                                    If None, will build using hyperparameters.
            model_type:             Either linear or categorical.
            checkpoint:             Path to cp.cpkt checkpoint. If provided, will update model with checkpoint weights.
            batch_size:             Evaluation batch size.
            max_tiles_per_slide:    If provided, will select only up to this maximum number of tiles from each slide.
            min_tiles_per_slide:    If provided, will only evaluate slides with a given minimum number of tiles.
            permutation_importance: If true, will run permutation feature importance to define relative benefit
                                        of histology and each clinical slide-level feature input, if provided.

        Returns:
            Keras history object.'''

        # Load and initialize model
        if not hp and checkpoint:
            raise ModelError('If using a checkpoint for evaluation, hyperparameters must be specified.')
        self.save_manifest()
        if not batch_size: batch_size = hp.batch_size
        with tf.name_scope('input'):
            dataset, dataset_with_slidenames, num_tiles = interleave_tfrecords(tfrecords,
                                                                               img_size=self.tile_px,
                                                                               batch_size=batch_size,
                                                                               balance=NO_BALANCE,
                                                                               finite=True,
                                                                               model_type=self.model_type,
                                                                               label_parser=self._parse_tfrecord_labels,
                                                                               annotations=self.annotations,
                                                                               max_tiles=max_tiles_per_slide,
                                                                               min_tiles=min_tiles_per_slide,
                                                                               include_slidenames=True,
                                                                               augment=False,
                                                                               normalizer=self.normalizer,
                                                                               manifest=self.manifest,
                                                                               slides=self.slides)
        if model:
            if model_type == 'cph':
                custom_objects = {'negative_log_likelihood':negative_log_likelihood,
                                  'concordance_index':concordance_index}
                self.model = tf.keras.models.load_model(model, custom_objects=custom_objects)
                self.model.compile(loss=negative_log_likelihood, metrics=concordance_index)
            else:
                self.model = tf.keras.models.load_model(model)

        elif checkpoint:
            self.model = self._build_model(hp)
            self.model.load_weights(checkpoint)

        # Generate performance metrics
        log.info('Calculating performance metrics...')
        if permutation_importance:
            drop_images = ((hp.tile_px==0) or hp.drop_images)
            auc, r_squared, c_index = sf.statistics.permutation_feature_importance(self.model,
                                                                             dataset_with_slidenames,
                                                                             self.annotations,
                                                                             model_type,
                                                                             self.outdir,
                                                                             outcome_names=self.outcome_names,
                                                                             label='eval',
                                                                             manifest=self.manifest,
                                                                             num_tiles=num_tiles,
                                                                             feature_names=self.feature_names,
                                                                             feature_sizes=self.feature_sizes,
                                                                             drop_images=drop_images)
        else:
            auc, r_squared, c_index = sf.statistics.metrics_from_dataset(self.model,
                                                                   model_type=model_type,
                                                                   annotations=self.annotations,
                                                                   manifest=self.manifest,
                                                                   dataset=dataset_with_slidenames,
                                                                   outcome_names=self.outcome_names,
                                                                   label='eval',
                                                                   data_dir=self.outdir,
                                                                   num_tiles=num_tiles,
                                                                   histogram=histogram,
                                                                   verbose=True,
                                                                   save_predictions=save_predictions)

        if model_type == 'categorical':
            log.info(f"Tile AUC: {auc['tile']}")
            log.info(f"Slide AUC: {auc['slide']}")
            log.info(f"Patient AUC: {auc['patient']}")
        if model_type == 'linear':
            log.info(f"Tile R-squared: {r_squared['tile']}")
            log.info(f"Slide R-squared: {r_squared['slide']}")
            log.info(f"Patient R-squared: {r_squared['patient']}")
        if model_type == 'cph':
            log.info(f"Tile c-index: {c_index['tile']}")
            log.info(f"Slide c-index: {c_index['slide']}")
            log.info(f"Patient c-index: {c_index['patient']}")

        val_metrics = self.model.evaluate(dataset, verbose=(log.getEffectiveLevel() <= 20), return_dict=True)

        results_log = os.path.join(self.outdir, 'results_log.csv')
        log.info(f'Evaluation metrics:')
        for m in val_metrics:
            log.info(f'{m}: {val_metrics[m]:.4f}')

        results_dict =     { 'eval': val_metrics }

        if model_type == 'categorical':
            results_dict['eval'].update({
                'tile_auc': auc['tile'],
                'slide_auc': auc['slide'],
                'patient_auc': auc['patient']
            })
        if model_type == 'linear':
            results_dict['eval'].update({
                'tile_r_squared': r_squared['tile'],
                'slide_r_squared': r_squared['slide'],
                'patient_r_squared': r_squared['patient']
            })
        if model_type == 'cph':
            results_dict['eval'].update({
                'tile_r_squared': c_index['tile'],
                'slide_r_squared': c_index['slide'],
                'patient_r_squared': c_index['patient']
            })

        sf.util.update_results_log(results_log, 'eval_model', results_dict)

        return val_metrics

    def train(self,
              hp,
              pretrain='imagenet',
              resume_training=None,
              checkpoint=None,
              log_frequency=100,
              validate_on_batch=512,
              validation_batch_size=32,
              validation_steps=200,
              max_tiles_per_slide=0,
              min_tiles_per_slide=0,
              starting_epoch=0,
              ema_observations=20,
              ema_smoothing=2,
              steps_per_epoch_override=None,
              use_tensorboard=False,
              multi_gpu=True,
              save_predictions=False,
              skip_metrics=False):

        '''Train the model.

        Args:
            hp:                         HyperParameters object
            pretrain:                   Either None, 'imagenet' or path to Tensorflow model for pretrained weights
            resume_training:            If True, will attempt to resume previously aborted training
            checkpoint:                 Path to cp.cpkt checkpoint file. If provided, will load checkpoint weights
            log_frequency:              How frequent to update Tensorboard logs
            validate_on_batch:          Validation will be performed every X batches
            validation_batch_size:      Batch size to use during validation
            validation_steps:           Number of batches to use for each instance of validation
            max_tiles_per_slide:        If provided, will select only up to this maximum number of tiles from each slide
            min_tiles_per_slide:        If provided, will only evaluate slides with a given minimum number of tiles
            starting_epoch:             Starts training at the specified epoch
            ema_observations:           Number of observations over which to perform exponential moving average smoothing
            ema_smoothing:              Exponential average smoothing value
            steps_per_epoch_override:   If provided, will manually set the number of steps per epoch.

        Returns:
            Results dictionary, Keras history object'''

        if multi_gpu:
            strategy = tf.distribute.MirroredStrategy()
            log.info(f'Multi-GPU training with {strategy.num_replicas_in_sync} devices')
        else:
            strategy = None
        self.save_manifest()
        with strategy.scope() if strategy is not None else no_scope():
            with tf.name_scope('input'):
                train_data, _, num_tiles = interleave_tfrecords(self.train_tfrecords,
                                                                img_size=self.tile_px,
                                                                batch_size=hp.batch_size,
                                                                balance=hp.balanced_training,
                                                                finite=False,
                                                                model_type=self.model_type,
                                                                label_parser=self._parse_tfrecord_labels,
                                                                annotations=self.annotations,
                                                                max_tiles=max_tiles_per_slide,
                                                                min_tiles=min_tiles_per_slide,
                                                                include_slidenames=False,
                                                                augment=hp.augment,
                                                                normalizer=self.normalizer,
                                                                manifest=self.manifest,
                                                                slides=self.slides)
            # Set up validation data
            using_validation = (self.val_tfrecords and len(self.val_tfrecords))
            if using_validation:
                with tf.name_scope('input'):
                    interleave_results = interleave_tfrecords(self.val_tfrecords,
                                                              img_size=self.tile_px,
                                                              batch_size=validation_batch_size,
                                                              balance=hp.balanced_validation,
                                                              finite=True,
                                                              model_type=self.model_type,
                                                              label_parser=self._parse_tfrecord_labels,
                                                              annotations=self.annotations,
                                                              max_tiles=max_tiles_per_slide,
                                                              min_tiles=min_tiles_per_slide,
                                                              include_slidenames=True,
                                                              augment=False,
                                                              normalizer=self.normalizer,
                                                              manifest=self.manifest,
                                                              slides=self.slides)
                    validation_data, validation_data_with_slidenames, num_val_tiles = interleave_results

                val_log_msg = '' if not validate_on_batch else f'every {str(validate_on_batch)} steps and '
                log.debug(f'Validation during training: {val_log_msg}at epoch end')
                if validation_steps:
                    validation_data_for_training = validation_data.repeat()
                    num_samples = validation_steps * hp.batch_size
                    log.debug(f'Using {validation_steps} batches ({num_samples} samples) each validation check')
                else:
                    validation_data_for_training = validation_data
                    log.debug(f'Using entire validation set each validation check')
            else:
                log.debug('Validation during training: None')
                validation_data_for_training = None
                validation_steps = 0

        # Prepare results
        results = {'epochs': {}}

        # Calculate parameters
        if max(hp.finetune_epochs) <= starting_epoch:
            max_epoch = max(hp.finetune_epochs)
            log.error(f'Starting epoch ({starting_epoch}) cannot be greater than the max target epoch ({max_epoch})')
            return None, None
        if hp.early_stop and hp.early_stop_method == 'accuracy' and hp.model_type() != 'categorical':
            log.error(f"Unable to use 'accuracy' early stopping with model type '{hp.model_type()}'")
        if starting_epoch != 0:
            log.info(f'Starting training at epoch {starting_epoch}')
        total_epochs = hp.toplayer_epochs + (max(hp.finetune_epochs) - starting_epoch)
        if steps_per_epoch_override:
            steps_per_epoch = steps_per_epoch_override
        else:
            steps_per_epoch = round(num_tiles/hp.batch_size)
        results_log = os.path.join(self.outdir, 'results_log.csv')

        # Create callbacks for early stopping, checkpoint saving, summaries, and history
        history_callback = tf.keras.callbacks.History()
        checkpoint_path = os.path.join(self.outdir, 'cp.ckpt')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=(log.getEffectiveLevel() <= 20))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.outdir,
                                                              histogram_freq=0,
                                                              write_graph=False,
                                                              update_freq=log_frequency)
        parent = self

        class PredictionAndEvaluationCallback(tf.keras.callbacks.Callback):
            # TODO: log early stopping batch number, and record

            def __init__(self):
                super(PredictionAndEvaluationCallback, self).__init__()
                self.early_stop = False
                self.last_ema = -1
                self.moving_average = []
                self.ema_two_checks_prior = -1
                self.ema_one_check_prior = -1
                self.epoch_count = starting_epoch
                self.model_type = hp.model_type()

            def on_epoch_end(self, epoch, logs={}):
                if log.getEffectiveLevel() <= 20: print('\r\033[K', end='')
                self.epoch_count += 1
                if self.epoch_count in [e for e in hp.finetune_epochs]:
                    model_name = parent.name if parent.name else 'trained_model'
                    model_path = os.path.join(parent.DATA_DIR, f'{model_name}_epoch{self.epoch_count}')
                    self.model.save(model_path)

                    # Try to copy model settings/hyperparameters file into the model folder
                    try:
                        shutil.copy(os.path.join(os.path.dirname(model_path), 'hyperparameters.json'),
                                    os.path.join(model_path, 'hyperparameters.json'), )
                        shutil.copy(os.path.join(os.path.dirname(model_path), 'slide_manifest.log'),
                                    os.path.join(model_path, 'slide_manifest.log'), )
                    except:
                        log.warning('Unable to copy hyperparameters.json/slide_manifest.log files into model folder.')

                    log.info(f'Trained model saved to {sf.util.green(model_path)}')
                    if parent.val_tfrecords and len(parent.val_tfrecords):
                        self.evaluate_model(logs)
                elif self.early_stop:
                    self.evaluate_model(logs)
                self.model.stop_training = self.early_stop

            def on_train_batch_end(self, batch, logs={}):
                if using_validation and validate_on_batch and (batch > 0) and (batch % validate_on_batch == 0):
                    val_metrics = self.model.evaluate(validation_data,
                                                      verbose=0,
                                                      steps=validation_steps,
                                                      return_dict=True)
                    val_loss = val_metrics['loss']
                    self.model.stop_training = False
                    if hp.early_stop_method == 'accuracy' and 'val_accuracy' in val_metrics:
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

                    # First, skip moving average calculations if using an invalid metric
                    if self.model_type != 'categorical' and hp.early_stop_method == 'accuracy':
                        log.info(log_message)
                    else:
                        # Calculate exponential moving average of validation accuracy
                        if len(self.moving_average) <= ema_observations:
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
                                self.last_ema = (early_stop_value * (ema_smoothing/(1+ema_observations))) + \
                                                (self.last_ema * (1-(ema_smoothing/(1+ema_observations))))
                                log.info(log_message + f' (EMA: {self.last_ema:.3f})')

                        # If early stopping and our patience criteria has been met,
                        #   check if validation accuracy is still improving
                        if (hp.early_stop and
                            (self.last_ema != -1) and
                            (float(batch)/steps_per_epoch)+self.epoch_count > hp.early_stop_patience):

                            if (self.ema_two_checks_prior != -1 and
                                ((hp.early_stop_method == 'accuracy' and self.last_ema <= self.ema_two_checks_prior) or
                                 (hp.early_stop_method == 'loss' and self.last_ema >= self.ema_two_checks_prior))):

                                log.info(f'Early stop triggered: epoch {self.epoch_count+1}, batch {batch}')
                                self.model.stop_training = True
                                self.early_stop = True
                            else:
                                self.ema_two_checks_prior = self.ema_one_check_prior
                                self.ema_one_check_prior = self.last_ema

            def on_train_end(self, logs={}):
                if log.getEffectiveLevel() <= 20: print('\r\033[K')

            def evaluate_model(self, logs={}):
                epoch = self.epoch_count
                epoch_label = f'val_epoch{epoch}'
                if not skip_metrics:
                    auc, r_squared, c_index = sf.statistics.metrics_from_dataset(self.model,
                                                                           model_type=hp.model_type(),
                                                                           annotations=parent.annotations,
                                                                           manifest=parent.manifest,
                                                                           dataset=validation_data_with_slidenames,
                                                                           outcome_names=parent.outcome_names,
                                                                           label=epoch_label,
                                                                           data_dir=parent.DATA_DIR,
                                                                           num_tiles=num_val_tiles,
                                                                           histogram=False,
                                                                           verbose=True,
                                                                           save_predictions=save_predictions)

                val_metrics = self.model.evaluate(validation_data, verbose=0, return_dict=True)
                log.info(f'Validation metrics:')
                for m in val_metrics:
                    log.info(f'{m}: {val_metrics[m]:.4f}')
                results['epochs'][f'epoch{epoch}'] = {'train_metrics': logs,
                                                        'val_metrics': val_metrics }
                if not skip_metrics:
                    for i, c in enumerate(auc['tile']):
                        results['epochs'][f'epoch{epoch}'][f'tile_auc{i}'] = c
                    for i, c in enumerate(auc['slide']):
                        results['epochs'][f'epoch{epoch}'][f'slide_auc{i}'] = c
                    for i, c in enumerate(auc['patient']):
                        results['epochs'][f'epoch{epoch}'][f'patient_auc{i}'] = c
                    results['epochs'][f'epoch{epoch}']['r_squared'] = r_squared
                    results['epochs'][f'epoch{epoch}']['c_index'] = c_index

                epoch_results = results['epochs'][f'epoch{epoch}']
                sf.util.update_results_log(results_log, 'trained_model', {f'epoch{epoch}': epoch_results})

        callbacks = [history_callback, PredictionAndEvaluationCallback(), cp_callback]
        if use_tensorboard:
            callbacks += [tensorboard_callback]

        with strategy.scope() if strategy is not None else no_scope():
            # Build or load model
            if resume_training:
                log.info(f'Resuming training from {sf.util.green(resume_training)}')
                self.model = tf.keras.models.load_model(resume_training)
            else:
                self.model = self._build_model(hp,
                                            pretrain=pretrain,
                                            checkpoint=checkpoint)

            # Retrain top layer only, if using transfer learning and not resuming training
            if hp.toplayer_epochs:
                self._retrain_top_layers(hp, train_data, validation_data_for_training, steps_per_epoch,
                                        callbacks=None, epochs=hp.toplayer_epochs)

            # Fine-tune the model
            log.info('Beginning fine-tuning')
            self._compile_model(hp)

            #tf.debugging.enable_check_numerics()

            history = self.model.fit(train_data,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=total_epochs,
                                 verbose=(log.getEffectiveLevel() <= 20),
                                 initial_epoch=hp.toplayer_epochs,
                                 validation_data=validation_data_for_training,
                                 validation_steps=validation_steps,
                                 callbacks=callbacks)

        return results, history.history
