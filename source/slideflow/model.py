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
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf

from tensorflow.keras.mixed_precision import experimental as mixed_precision
from functools import partial
from slideflow.util import log
from slideflow.io.tfrecords import detect_tfrecord_format, get_tfrecord_parser
from slideflow.model_utils import *

import slideflow.util as sfutil
from slideflow.util import StainNormalizer
import slideflow.statistics as sfstats

BALANCE_BY_CATEGORY = 'BALANCE_BY_CATEGORY'
BALANCE_BY_PATIENT = 'BALANCE_BY_PATIENT'
NO_BALANCE = 'NO_BALANCE'
MODEL_FORMAT_1_9 = "1.9"
MODEL_FORMAT_CURRENT = MODEL_FORMAT_1_9
MODEL_FORMAT_LEGACY = "legacy"

class ModelActivationsInterface:
	'''Provides an interface to obtain logits and post-convolutional activations 
		from saved Slideflow Keras models. Provides support for newer models (v1.9.1+) 
		and legacy slideflow models (1.9.0b and earlier)'''

	def __init__(self, path, model_format=None, layers=['postconv'], include_logits=True):
		'''Initializer.
		
		Args:
			path:			Path to saved Slideflow Keras model
			model_format:	Either slideflow.model.MODEL_FORMAT_CURRENT or _LEGACY. 
								Indicates how the saved model should be processed,
								as older versions of Slideflow had models constructed differently, 
								with differing naming of Keras layers.
			layers:			Layers from which to generate activations. The post-convolution activation layer is accessed via 'postconv'
		'''
		if not model_format: model_format = MODEL_FORMAT_CURRENT
		self.model_format = model_format
		self.path = path
		self.num_classes = 0
		self.num_features = 0
		self.duplicate_input = (model_format == MODEL_FORMAT_LEGACY)
		self._model = tf.keras.models.load_model(self.path)
		self._build(layer_names=[l for l in layers if l != 'postconv'], 
					include_postconv=('postconv' in layers),
					include_logits=include_logits)

	def _build(self, layer_names=['block4_pool', 'block13_pool'], pooling=None, include_postconv=False, include_logits=True):
		'''Builds a model that outputs feature activations at the designated layers and concatenates into a single final output vector.'''

		if layer_names in [None, []]:
			log.info(f"Setting up activations interface to return activations from post-conv activations", 1)
		else:
			log.info(f"Setting up activations interface to return activations from layers {', '.join(layer_names)} {'and post-conv activations' if include_postconv else ''}", 1)
		
		assert (layer_names is not None) or include_postconv
		if pooling:
			assert type(pooling) == list and len(pooling) == len(layer_names)

		if self.model_format == MODEL_FORMAT_1_9:
			model_core = self._model.layers[1]
		if self.model_format == MODEL_FORMAT_LEGACY:
			model_core = self._model.layers[0]

		inputs = model_core.input
		self.duplicate_input = True # Why is this here?

		layer_sources = {}
		if layer_names:
			for layer_name in layer_names:
				if layer_name in [l.name for l in self._model.layers]:
					layer_sources[layer_name] = self._model
				elif layer_name in [l.name for l in model_core.layers]:
					layer_sources[layer_name] = model_core
				elif self.model_format != MODEL_FORMAT_LEGACY:
					log.warn("Unable to read model using modern format, will try using legacy format", 1)
					self.model_format = MODEL_FORMAT_LEGACY
					self._build(layer_names, include_postconv)
					return
				else:
					raise ModelError(f"Unable to read model: could not find layer {layer_name}")
			raw_output_layers = [layer_sources[l].get_layer(l).output for l in layer_names]

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

		merged_output = activation_layers[0] if len(activation_layers) == 1 else tf.keras.layers.Concatenate(axis=1)(activation_layers)
		
		if include_logits:
			outputs = [merged_output, self._model.output]
		else:
			outputs = [merged_output]

		try:
			self.model = tf.keras.models.Model(inputs=[self._model.input, model_core.input],
											   outputs=outputs)
		except AttributeError:
			if self.model_format != MODEL_FORMAT_LEGACY:
				log.warn("Unable to read model using modern format, will try using legacy format", 1)
				self.model_format = MODEL_FORMAT_LEGACY
				self._build(layer_names, include_postconv)
				return
			else:
				raise ModelError("Unable to read model.")

		if include_logits:
			self.num_features = self.model.output_shape[0][1]
			self.num_classes = self.model.output_shape[1][1]
			log.info(f"Number of logits: {self.model.output_shape[1][1]}", 2)
		else:
			self.num_features = self.model.output_shape[1]
		log.info(f"Number of activation features: {self.num_features}", 2)

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
		'Adam':	tf.keras.optimizers.Adam,
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
		assert (isinstance(finetune_epochs, list) and all([isinstance(t, int) for t in finetune_epochs])) or isinstance(finetune_epochs, int)
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
		return [arg for arg in dir(self) if not arg[0]=='_' and arg not in ['get_opt', 'get_model', 'model_type', 'validate']]

	def _get_dict(self):
		d = {}
		for arg in self._get_args():
			d.update({arg: getattr(self, arg)})
		return d

	def _load_dict(self, hp_dict):
		for key, value in hp_dict.items():
			try:
				setattr(self, key, value)
			except:
				log.error(f"Unrecognized hyperparameter {key}; unable to load")

	def __str__(self):
		output = "Hyperparameters:\n"
			
		args = sorted(self._get_args(), key=lambda arg: arg.lower())
		for arg in args:
			value = getattr(self, arg)
			output += log.empty(f"{sfutil.header(arg)} = {value}\n", 2, None)
		return output

	def validate(self):
		'''Ensures that hyperparameter combinations are valid.'''
		if (self.model_type() != 'categorical' and ((self.balanced_training == BALANCE_BY_CATEGORY) or 
													(self.balanced_validation == BALANCE_BY_CATEGORY))):
			raise HyperParameterError(f'Invalid hyperparameter combination: balancing type "{BALANCE_BY_CATEGORY}" and model type "{self.model_type()}".')
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

	def get_model(self, image_shape=None, input_tensor=None, weights=None):
		'''Returns a Keras model of the appropriate architecture, input shape, pooling, and initial weights.'''
		return self._ModelDict[self.model](
			input_shape=image_shape,
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

class SlideflowModel:
	''' Model containing all functions necessary to build input dataset pipelines,
	build a training and validation set model, and monitor and execute training.'''

	def __init__(self, data_directory, image_size, slide_annotations, train_tfrecords, validation_tfrecords, 
					manifest=None, mixed_precision=True, model_type='categorical', normalizer=None, normalizer_source=None, 
					feature_sizes=None, feature_names=None, outcome_names=None):
		'''Model initializer.

		Args:
			data_directory:			Location where event logs and checkpoints will be written
			image_size:				Int, width/height of input image in pixels.
			slide_annotations:		Dictionary mapping slide names to both patient names and outcome labels
			train_tfrecords:		List of tfrecord paths for training
			validation_tfrecords:	List of tfrecord paths for validation
			manifest:				Manifest dictionary mapping TFRecords to number of tiles
			mixed_precision:		Bool, if True, will use FP16 mixed precision (rather than FP32)
			model_type:				Type of model outcome label, either 'categorical' or 'linear'
			normalizer:				Tile image normalization to perform in real-time during training
			normalizer_source:		Source image for normalization if being performed in real-time
		'''
		self.DATA_DIR = data_directory
		self.MANIFEST = manifest
		self.IMAGE_SIZE = image_size
		self.SLIDE_ANNOTATIONS = slide_annotations
		self.TRAIN_TFRECORDS = train_tfrecords
		self.VALIDATION_TFRECORDS = validation_tfrecords
		self.MODEL_TYPE = model_type
		self.SLIDES = list(slide_annotations.keys())
		self.FEATURE_SIZES = feature_sizes
		self.NUM_SLIDE_FEATURES = 0 if not feature_sizes else sum(feature_sizes)
		self.FEATURE_NAMES = feature_names
		self.mixed_precision = mixed_precision

		# Format outcome labels (ensures compatibility with single and multi-outcome models)
		outcome_labels = np.array([slide_annotations[slide]['outcome_label'] for slide in self.SLIDES])
		if len(outcome_labels.shape) == 1:
			outcome_labels = np.expand_dims(outcome_labels, axis=1)
		
		self.OUTCOME_NAMES = [f'Outcome {i}' for i in range(outcome_labels.shape[1])] if not outcome_names else outcome_names
		self.OUTCOME_NAMES = [self.OUTCOME_NAMES] if not isinstance(self.OUTCOME_NAMES, list) else self.OUTCOME_NAMES

		if len(self.OUTCOME_NAMES) != outcome_labels.shape[1]:
			raise ModelError(f"Provided outcome_names (len: {len(self.OUTCOME_NAMES)}) does not match number of outcomes {outcome_labels.shape[1]}")

		if model_type not in ('categorical', 'linear', 'cph'):
			raise ModelError(f"Unknown model type {model_type}")

		# Setup slide-level input
		if self.NUM_SLIDE_FEATURES:
			try:
				self.SLIDE_FEATURE_TABLE = {slide: slide_annotations[slide]['input'] for slide in self.SLIDES}
				num_features = self.NUM_SLIDE_FEATURES if model_type != 'cph' else self.NUM_SLIDE_FEATURES - 1
				if num_features:
					log.info(f"Training with both images and {num_features} categories of slide-level input", 1)
					if model_type == 'cph':
						log.info("Interpreting first feature as event for CPH model", 1)
				elif model_type == 'cph':
					log.info(f"Training with images alone. Interpreting first feature as event for CPH model", 1)
				
			except KeyError:
				raise ModelError("If num_slide_features > 0, slide-level input must be provided via 'input' key in slide_annotations")
			for slide in self.SLIDES:
				if len(self.SLIDE_FEATURE_TABLE[slide]) != self.NUM_SLIDE_FEATURES:
					raise ModelError(f"Length of input for slide {slide} does not match num_slide_features; expected {self.NUM_SLIDE_FEATURES}, got {len(self.SLIDE_FEATURE_TABLE[slide])}")

		# Normalization setup
		if normalizer: log.info(f"Using realtime {normalizer} normalization", 1)
		self.normalizer = None if not normalizer else StainNormalizer(method=normalizer, source=normalizer_source)

		if model_type in ['linear', 'cph']:
			try:
				self.NUM_CLASSES = outcome_labels.shape[1]
			except TypeError:
				raise ModelError("Incorrect formatting of outcome labels for a linear model; must be formatted as an array.")

		if model_type == 'categorical':
			self.NUM_CLASSES = {i: np.unique(outcome_labels[:,i]).shape[0] for i in range(outcome_labels.shape[1])}

		with tf.device('/cpu'):
			self.ANNOTATIONS_TABLES = []
			for oi in range(outcome_labels.shape[1]):
				self.ANNOTATIONS_TABLES += [tf.lookup.StaticHashTable(
					tf.lookup.KeyValueTensorInitializer(self.SLIDES, outcome_labels[:,oi]), -1
				)]
			#self.RNA_SEQ_TABLE = {self.SLIDES[i]:outcome_labels[i] for i in range(len(self.SLIDES))}

		if not os.path.exists(self.DATA_DIR):
			os.makedirs(self.DATA_DIR)
		
		# Record which slides are used for training and validation, and to which categories they belong
		if train_tfrecords or validation_tfrecords:
			with open(os.path.join(self.DATA_DIR, 'slide_manifest.log'), 'w') as slide_manifest:
				writer = csv.writer(slide_manifest)
				writer.writerow(['slide', 'dataset', 'outcome_label'])
				if train_tfrecords:
					for tfrecord in train_tfrecords:
						slide = tfrecord.split('/')[-1][:-10]
						if slide in self.SLIDES:
							outcome_label = slide_annotations[slide]['outcome_label']
							writer.writerow([slide, 'training', outcome_label])
				if validation_tfrecords:
					for tfrecord in validation_tfrecords:
						slide = tfrecord.split('/')[-1][:-10]
						if slide in self.SLIDES:
							outcome_label = slide_annotations[slide]['outcome_label']
							writer.writerow([slide, 'validation', outcome_label])

	def _build_model(self, hp, pretrain=None, pretrain_model_format=None, checkpoint=None):
		''' Assembles base model, using pretraining (imagenet) or the base layers of a supplied model.

		Args:
			hp:			HyperParameters object
			pretrain:	Either 'imagenet' or path to model to use as pretraining
			checkpoint:	Path to checkpoint from which to resume model training
		'''
		if self.mixed_precision:
			log.info("Training with mixed precision", 1)
			policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
			mixed_precision.set_policy(policy)

		# Setup inputs
		image_shape = (self.IMAGE_SIZE, self.IMAGE_SIZE, 3)
		tile_input_tensor = tf.keras.Input(shape=image_shape, name="tile_image")
		if self.NUM_SLIDE_FEATURES:
			if self.MODEL_TYPE == 'cph':
				event_input_tensor = tf.keras.Input(shape=(1), name="event_input")
				# Add slide feature input tensors, if there are more slide features
				#    than just the event input tensor for CPH models
				if not ((self.NUM_SLIDE_FEATURES == 1) and (self.MODEL_TYPE == 'cph')):
					slide_feature_input_tensor = tf.keras.Input(shape=(self.NUM_SLIDE_FEATURES - 1), name="slide_feature_input")
			else:
				slide_feature_input_tensor = tf.keras.Input(shape=(self.NUM_SLIDE_FEATURES), name="slide_feature_input")
		if self.MODEL_TYPE == 'cph' and not self.NUM_SLIDE_FEATURES:
			raise ModelError("Model error - CPH models must include event input")

		# Load pretrained model if applicable
		if pretrain: log.info(f"Using pretraining from {sfutil.green(pretrain)}", 1)
		if pretrain and pretrain!='imagenet':
			pretrained_model = tf.keras.models.load_model(pretrain)
			if not pretrain_model_format or pretrain_model_format == MODEL_FORMAT_CURRENT:
				try:
					# This is the tile_image input
					pretrained_input = pretrained_model.get_layer(name="tile_image").input
					# Name of the pretrained model core, which should be at layer 1
					pretrained_name = pretrained_model.get_layer(index=1).name 
					# This is the post-convolution layer
					pretrained_output = pretrained_model.get_layer(name="post_convolution").output 
					base_model = tf.keras.Model(inputs=pretrained_input, 
												outputs=pretrained_output, 
												name=f"pretrained_{pretrained_name}").layers[1]
				except ValueError:
					log.warn("Unable to read pretrained model using modern format, will try legacy model format", 1)
					pretrain_model_format = MODEL_FORMAT_LEGACY
			if pretrain_model_format == MODEL_FORMAT_LEGACY:
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
			# freezeIndex is the layer from 0 up to Index that should be frozen (not trainable). 
			# Per Jakob's models, all but last 10, 20, or 30 layers were frozen. His last three layers were a 1000-node fully connected layer (eqv. to our hidden layers), 
			# then softmax, then classification. It looks like we don't add a classification layer on though, I don't see it added anywhere.
			# I see below that we add on the hidden layer and softmax layer, so I am freezing (10-2=8) only, since when we add the last layers on it will add up to 10, 20, 30
			freezeIndex = int(len(base_model.layers) - (hp.trainable_layers - 1 ))# - hp.hidden_layers - 1))
			log.info(f"Only training on last {hp.trainable_layers} layers (of {len(base_model.layers)} total layers)", 2)
			for layer in base_model.layers[:freezeIndex]:
				layer.trainable = False

		# Create sequential tile model: 
		# 	tile image --> convolutions --> pooling/flattening --> hidden layers ---> prelogits --> softmax/logits
		#                             additional slide input --/

		# This is an identity layer that simply returns the last layer, allowing us to name and access this layer later
		post_convolution_identity_layer = tf.keras.layers.Lambda(lambda x: x, name="post_convolution") 
		layers = [tile_input_tensor, base_model]
		if not hp.pooling:
			layers += [tf.keras.layers.Flatten()]
		layers += [post_convolution_identity_layer]
		if hp.dropout:
			layers += [tf.keras.layers.Dropout(hp.dropout)]
		tile_image_model = tf.keras.Sequential(layers)
		model_inputs = [tile_image_model.input]

		# Merge layers
		if self.NUM_SLIDE_FEATURES:
			# Add images
			if (hp.tile_px == 0) or hp.drop_images:
				log.info("Generating model with just clinical variables and no images", 1)
				merged_model = slide_feature_input_tensor
				model_inputs += [slide_feature_input_tensor]		
			elif not ((self.NUM_SLIDE_FEATURES == 1) and (self.MODEL_TYPE == 'cph')):
				# Add slide feature input tensors, if there are more slide features
				#    than just the event input tensor for CPH models
				merged_model = tf.keras.layers.Concatenate(name="input_merge")([slide_feature_input_tensor, tile_image_model.output])
				model_inputs += [slide_feature_input_tensor]
			else:
				merged_model = tile_image_model.output

			# Add event tensor if this is a CPH model
			if self.MODEL_TYPE == 'cph':
				model_inputs += [event_input_tensor]
		else:
			merged_model = tile_image_model.output

		# Add hidden layers
		for i in range(hp.hidden_layers):
			merged_model = tf.keras.layers.Dense(hp.hidden_layer_width, 
												 name=f"hidden_{i}", 
												 activation='relu', 
												 kernel_regularizer=regularizer)(merged_model)
		
		'''merged_model = tf.keras.layers.Dense(512, name=f"hidden_0", activation=None, kernel_regularizer=regularizer)(merged_model)
		merged_model = tf.keras.layers.LeakyReLU()(merged_model)
		merged_model = tf.keras.layers.Dense(128, name=f"hidden_1", activation=None, kernel_regularizer=regularizer)(merged_model)
		merged_model = tf.keras.layers.LeakyReLU()(merged_model)
		merged_model = tf.keras.layers.Dense(32, name=f"autoencoder", activation='tanh', kernel_regularizer=regularizer)(merged_model)
		merged_model = tf.keras.layers.Dense(64, name=f"reencode_0", activation=None, kernel_regularizer=regularizer)(merged_model)
		merged_model = tf.keras.layers.LeakyReLU()(merged_model)
		merged_model = tf.keras.layers.Dense(128, name=f"reencode_1", activation=None, kernel_regularizer=regularizer)(merged_model)
		merged_model = tf.keras.layers.LeakyReLU()(merged_model)
		merged_model = tf.keras.layers.Dense(256, name=f"reencode_2", activation=None, kernel_regularizer=regularizer)(merged_model)
		merged_model = tf.keras.layers.LeakyReLU()(merged_model)
		merged_model = tf.keras.layers.Dense(512, name=f"reencode_3", activation=None, kernel_regularizer=regularizer)(merged_model)
		merged_model = tf.keras.layers.LeakyReLU()(merged_model)
		merged_model = tf.keras.layers.Dense(1024, name=f"reencode_4", activation=None, kernel_regularizer=regularizer)(merged_model)
		merged_model = tf.keras.layers.LeakyReLU()(merged_model)
		merged_model = tf.keras.layers.Dense(2048, name=f"reencode_5", activation=None, kernel_regularizer=regularizer)(merged_model)
		merged_model = tf.keras.layers.LeakyReLU()(merged_model)'''

		# Add the softmax prediction layer
		if hp.model_type() in ['linear', 'cph']:
			activation = 'linear'
		else:
			activation = 'softmax'
		log.info(f"Using {activation} activation", 1)

		# Multi-categorical outcomes
		if type(self.NUM_CLASSES) == dict:
			outputs = []
			for c in self.NUM_CLASSES:
				final_dense_layer = tf.keras.layers.Dense(self.NUM_CLASSES[c], kernel_regularizer=regularizer, name=f"prelogits-{c}")(merged_model)
				outputs += [tf.keras.layers.Activation(activation, dtype='float32', name=f'out-{c}')(final_dense_layer)]

		else:
			final_dense_layer = tf.keras.layers.Dense(self.NUM_CLASSES, kernel_regularizer=regularizer, name="prelogits")(merged_model)
			#final_dense_layer = tf.keras.layers.Dropout(0.2)(final_dense_layer) # include for rna seq
			outputs = [tf.keras.layers.Activation(activation, dtype='float32', name='output')(final_dense_layer)]

		if self.MODEL_TYPE == 'cph':
			outputs[0] = tf.keras.layers.Concatenate(name="output_merge_CPH", dtype='float32')([outputs[0], event_input_tensor])

		# Assemble final model
		model = tf.keras.Model(inputs=model_inputs, outputs=outputs)

		if checkpoint:
			log.info(f"Loading checkpoint weights from {sfutil.green(checkpoint)}", 1)
			model.load_weights(checkpoint)

		# Print model summary
		if log.INFO_LEVEL > 0:
			print()
			model.summary()

		return model
	
	def _build_multi_image_model(self, hp, pretrain=None, checkpoint=None):
		'''Builds a sample test model that reads multiple images from each TFRecord entry.

		Args:
			hp:			HyperParameters object
			pretrain:	Either 'imagenet' or path to model to use as pretraining
			checkpoint:	Path to checkpoint from which to resume model training
		'''
		base_model = tf.keras.applications.Xception(input_shape=(299,299,3),
													include_top=False,
													pooling='max',
													weights='imagenet')

		base_model_i = tf.keras.applications.InceptionV3(input_shape=(299,299,3),
														 include_top=False,
														 pooling='max',
														 weights='imagenet')

		hidden = tf.keras.layers.Dense(200, activation='relu')(base_model.output)
		hidden_i = tf.keras.layers.Dense(200, activation='relu')(base_model_i.output)
		combined = tf.keras.layers.Concatenate()([hidden, hidden_i])
		hidden_c = tf.keras.layers.Dense(100, activation='relu')(combined)
		predictions = tf.keras.layers.Dense(self.NUM_CLASSES, activation='softmax')(hidden)

		model = tf.keras.Model(inputs=[base_model.input, base_model_i.input], outputs=predictions)

		return model

	def _compile_model(self, hp):
		'''Compiles keras model.
		
		Args:
			hp		Hyperparameter object.	
		'''
		
		if self.MODEL_TYPE == 'cph':
			metrics = concordance_index
		elif self.MODEL_TYPE == 'linear':
			metrics = [hp.loss]
		else:
			metrics = ['accuracy']

		loss_fn = negative_log_likelihood if self.MODEL_TYPE=='cph' else hp.loss
		self.model.compile(optimizer=hp.get_opt(),
						   loss=loss_fn,
						   metrics=metrics)

	def _interleave_tfrecords(self, tfrecords, batch_size, balance, finite, max_tiles=None, min_tiles=None, 
								multi_image=False, drop_remainder=False, include_slidenames=False, augment=True):

		'''Generates an interleaved dataset from a collection of tfrecord files,
		sampling from tfrecord files randomly according to balancing if provided.
		Requires self.MANIFEST. Assumes TFRecord files are named by slide.

		Args:
			tfrecords:				Array of paths to TFRecord files
			batch_size:				Batch size
			balance:				Whether to use balancing for batches. Options are BALANCE_BY_CATEGORY,
										BALANCE_BY_PATIENT, and NO_BALANCE. If finite option is used, will drop
										tiles in order to maintain proportions across the interleaved dataset.
			augment:					Whether to use data augmentation (random flip/rotate)
			finite:					Whether create finite or infinite datasets. WARNING: If finite option is 
										used with balancing, some tiles will be skipped.
			max_tiles:				Maximum number of tiles to use per slide.
			min_tiles:				Minimum number of tiles that each slide must have to be included.
			multi_image:			Bool, if True, will read multiple images from each TFRecord record.
		'''				 
		log.info(f"Interleaving {len(tfrecords)} tfrecords: finite={finite}, max_tiles={max_tiles}, min_tiles={min_tiles}", 1)
		datasets = []
		datasets_categories = []
		num_tiles = []
		global_num_tiles = 0
		categories = {}
		categories_prob = {}
		categories_tile_fraction = {}
		prob_weights = None
		base_parser = None
		detected_format = None
		num_tfrecords_empty = 0
		num_tfrecords_less_than_min = 0
		
		if tfrecords == []:
			raise ModelError("No TFRecords found.")

		if self.MANIFEST:
			pb = sfutil.ProgressBar(len(tfrecords), counter_text='files', leadtext="Interleaving tfrecords... ")
			for filename in tfrecords:
				slide_name = sfutil.path_to_name(filename)
				
				if slide_name not in self.SLIDES:
					continue

				# Determine total number of tiles available in TFRecord
				try:
					tiles = self.MANIFEST[filename]['total']
				except KeyError:
					log.error(f"Manifest not finished, unable to find {sfutil.green(filename)}", 1)
					raise ManifestError(f"Manifest not finished, unable to find {filename}")
				
				# Ensure TFRecord has minimum number of tiles; otherwise, skip
				if not min_tiles and tiles == 0:
					num_tfrecords_empty += 1
					continue
				elif tiles < min_tiles:
					num_tfrecords_less_than_min
					continue
				
				# Get the base TFRecord parser, based on the first tfrecord
				if detected_format is None:
					detected_format = detect_tfrecord_format(filename)
				elif detected_format != detect_tfrecord_format(filename):
					raise ModelError("Inconsistent TFRecord formatting among tfrecords. All records must be internally formatted the same.")
				if base_parser is None:
					base_parser = get_tfrecord_parser(filename, 
							('slide', 'image_raw'),
							standardize=True,
							img_size=self.IMAGE_SIZE,
							normalizer=self.normalizer,
							augment=augment)

				# Assign category by outcome if this is a categorical model, 
				#	Merging category names if there are multiple outcomes (balancing across all combinations of outcome categories equally)
				# Otherwise, consider all slides from the same category (effectively skipping balancing); appropriate for linear models.
				if self.MODEL_TYPE == 'categorical':
					category = self.SLIDE_ANNOTATIONS[slide_name]['outcome_label']
					category = [category] if not isinstance(category, list) else category
					category = '-'.join(map(str, category))
				else:
					category = 1
				
				datasets += [tf.data.TFRecordDataset(filename, num_parallel_reads=16)]
				datasets_categories += [category]

				# Cap number of tiles to take from TFRecord at maximum specified
				if max_tiles and tiles > max_tiles:
					log.info(f"Only taking maximum of {max_tiles} (of {tiles}) tiles from {sfutil.green(filename)}", 2)
					tiles = max_tiles
				
				if category not in categories.keys():
					categories.update({category: {'num_slides': 1,
												'num_tiles': tiles}})
				else:
					categories[category]['num_slides'] += 1
					categories[category]['num_tiles'] += tiles
				num_tiles += [tiles]
				pb.increase_bar_value()
			pb.end()

			if num_tfrecords_empty:
				log.info(f"Skipped {num_tfrecords_empty} empty tfrecords", 2)
			if num_tfrecords_less_than_min:
				log.info(f"Skipped {num_tfrecords_less_than_min} tfrecords with less than {min_tiles} tiles", 2)

			for category in categories:
				lowest_category_slide_count = min([categories[i]['num_slides'] for i in categories])
				lowest_category_tile_count = min([categories[i]['num_tiles'] for i in categories])
				categories_prob[category] = lowest_category_slide_count / categories[category]['num_slides']
				categories_tile_fraction[category] = lowest_category_tile_count / categories[category]['num_tiles']

			# Balancing
			if balance == NO_BALANCE:
				log.info(f"Not balancing input", 2)
				prob_weights = [i/sum(num_tiles) for i in num_tiles]
			if balance == BALANCE_BY_PATIENT:
				log.info(f"Balancing input across slides", 2)
				prob_weights = [1.0] * len(datasets)
				if finite:
					# Only take as many tiles as the number of tiles in the smallest dataset
					minimum_tiles = min(num_tiles)
					for i in range(len(datasets)):
						num_tiles[i] = minimum_tiles
			if balance == BALANCE_BY_CATEGORY:
				log.info(f"Balancing input across categories", 2)
				prob_weights = [categories_prob[datasets_categories[i]] for i in range(len(datasets))]
				if finite:
					# Only take as many tiles as the number of tiles in the smallest category
					for i in range(len(datasets)):
						num_tiles[i] = int(num_tiles[i] * categories_tile_fraction[datasets_categories[i]])
						log.empty(f"Tile fraction (dataset {i+1}/{len(datasets)}): {categories_tile_fraction[datasets_categories[i]]}, taking {num_tiles[i]}", 3)
					log.empty(f"Global num tiles: {global_num_tiles}", 3)
			
			# Take the calculcated number of tiles from each dataset and calculate global number of tiles
			for i in range(len(datasets)):
				datasets[i] = datasets[i].take(num_tiles[i])
				if not finite:
					datasets[i] = datasets[i].repeat()
			global_num_tiles = sum(num_tiles)

		else:
			log.error("No manifest detected! Unable to perform tfrecord balancing or any tile-level selection operations (min or max tiles per slide)", 1)
			for filename in tfrecords:
				slide_name = sfutil.path_to_name(filename)
				
				if slide_name not in self.SLIDES:
					continue

				datasets += [tf.data.TFRecordDataset(filename, num_parallel_reads=16)]

		# Interleave and batch datasets
		try:
			sampled_dataset = tf.data.experimental.sample_from_datasets(datasets, weights=prob_weights)
			dataset = self._get_parsed_datasets(sampled_dataset, base_parser, include_slidenames=False)			
			dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
			#dataset = dataset.prefetch(tf.data.AUTOTUNE)
		except IndexError:
			raise ModelError("No TFRecords found after filter criteria; please ensure all tiles have been extracted and all TFRecords are in the appropriate folder")

		if include_slidenames:
			dataset_with_slidenames = self._get_parsed_datasets(sampled_dataset, base_parser, include_slidenames=True)
			dataset_with_slidenames = dataset_with_slidenames.batch(batch_size, drop_remainder=drop_remainder)
			#dataset_with_slidenames = dataset_with_slidenames.prefetch(tf.data.AUTOTUNE)

		else:
			dataset_with_slidenames = None
		
		return dataset, dataset_with_slidenames, global_num_tiles

	def _get_parsed_datasets(self, tfrecord_dataset, base_parser, include_slidenames=False):
		if include_slidenames:
			training_parser_with_slidenames = partial(self._parse_tfrecord_function, base_parser=base_parser, include_slidenames=True)
			dataset_with_slidenames = tfrecord_dataset.map(training_parser_with_slidenames, num_parallel_calls=32)#tf.data.AUTOTUNE)
			return dataset_with_slidenames
		else:
			training_parser = partial(self._parse_tfrecord_function, base_parser=base_parser, include_slidenames=False)
			dataset = tfrecord_dataset.map(training_parser, num_parallel_calls=32)#tf.data.AUTOTUNE)
			return dataset

	def _parse_tfrecord_function(self, record, base_parser, include_slidenames=True, multi_image=False):
		'''Parses raw entry read from TFRecord.'''

		# Note: multi-image functionality removed in version 1.11 due to lack of use and changes in tfrecord processing
		# If desired, this can be re-added at this stage by simply returning multiple images in the resulting image_dict

		slide, image = base_parser(record)
		image_dict = { 'tile_image': image }

		if self.MODEL_TYPE in ['linear', 'cph']:
			label = [self.ANNOTATIONS_TABLES[oi].lookup(slide) for oi in range(self.NUM_CLASSES)]
		elif len(self.NUM_CLASSES) > 1:
			label = {f'out-{oi}': self.ANNOTATIONS_TABLES[oi].lookup(slide) for oi in range(len(self.NUM_CLASSES))}
		else:
			label = self.ANNOTATIONS_TABLES[0].lookup(slide)

		# === RNA SEQ ==========
		#def rna_seq_lookup(s): return self.RNA_SEQ_TABLE[s.numpy().decode('utf-8')]

		#label = tf.py_function(func=rna_seq_lookup,
		#						inp=[slide],
		#						Tout=tf.float32)
		# ====================

		# Add additional non-image feature inputs if indicated,
		# 	excluding the event feature used for CPH models
		if self.NUM_SLIDE_FEATURES:
			# If CPH model is used, time-to-event data must be added as a separate feature
			if self.MODEL_TYPE == 'cph':
				def slide_lookup(s): return self.SLIDE_FEATURE_TABLE[s.numpy().decode('utf-8')][1:]
				def event_lookup(s): return self.SLIDE_FEATURE_TABLE[s.numpy().decode('utf-8')][0]
				num_features = self.NUM_SLIDE_FEATURES - 1

				event_input_val = tf.py_function(func=event_lookup,
													inp=[slide],
													Tout=[tf.float32])
				image_dict.update({'event_input': event_input_val})
			else:
				def slide_lookup(s): return self.SLIDE_FEATURE_TABLE[s.numpy().decode('utf-8')]
				num_features = self.NUM_SLIDE_FEATURES

			slide_feature_input_val = tf.py_function(func=slide_lookup, 
														inp=[slide],
														Tout=[tf.float32] * num_features)

			# Add slide input features, excluding the event feature used for CPH models
			if not ((self.NUM_SLIDE_FEATURES == 1) and (self.MODEL_TYPE == 'cph')):
				image_dict.update({'slide_feature_input': slide_feature_input_val}) 

		if include_slidenames:
			return image_dict, label, slide
		else:
			return image_dict, label

	def _retrain_top_layers(self, hp, train_data, validation_data, steps_per_epoch, callbacks=None, epochs=1):
		'''Retrains only the top layer of this object's model, while leaving all other layers frozen.'''
		log.info("Retraining top layer", 1)
		# Freeze the base layer
		self.model.layers[0].trainable = False
		val_steps = 200 if validation_data else None

		self._compile_model(hp)

		toplayer_model = self.model.fit(train_data,
										epochs=epochs,
										verbose=(log.INFO_LEVEL > 0),
										steps_per_epoch=steps_per_epoch,
										validation_data=validation_data,
										validation_steps=val_steps,
										callbacks=callbacks)

		# Unfreeze the base layer
		self.model.layers[0].trainable = True
		return toplayer_model.history

	def evaluate(self, tfrecords, hp=None, model=None, model_type='categorical', checkpoint=None, batch_size=None, 
					max_tiles_per_slide=0, min_tiles_per_slide=0, multi_image=False, permutation_importance=False, 
					histogram=False, save_predictions=False):
		'''Evaluate model.

		Args:
			tfrecords:				List of TFrecords paths to load for evaluation.
			hp:						HyperParameters object
			model:					Optional; Tensorflow model to load for evaluation. If None, will build model using hyperparameters.
			model_type:				Either linear or categorical.
			checkpoint:				Path to cp.cpkt checkpoint. If provided, will update model with given checkpoint weights.
			batch_size:				Evaluation batch size.
			max_tiles_per_slide:	If provided, will select only up to this maximum number of tiles from each slide.
			min_tiles_per_slide:	If provided, will only evaluate slides with a given minimum number of tiles.
			multi_image:			If true, will evaluate model with multi-image inputs.
			permutation_importance:	If true, will run permutation feature importance to define relative benefit of histology
										and each clinical slide-level feature input, if provided.

		Returns:
			Keras history object.'''

		# Load and initialize model
		if not hp and checkpoint:
			raise ModelError("If using a checkpoint for evaluation, hyperparameters must be specified.")
		if not batch_size: batch_size = hp.batch_size
		with tf.name_scope('input'):
			dataset, dataset_with_slidenames, num_tiles = self._interleave_tfrecords(tfrecords,
																					batch_size=batch_size,
																					balance=NO_BALANCE,
																					finite=True,
																					max_tiles=max_tiles_per_slide,
																					min_tiles=min_tiles_per_slide,
																					multi_image=multi_image,
																					include_slidenames=True,
																					augment=False)
		if model:
			if model_type == 'cph':
				self.model = tf.keras.models.load_model(model, custom_objects = {
																				  'negative_log_likelihood':negative_log_likelihood,
																				  'concordance_index':concordance_index 
																				})
				self.model.compile(loss=negative_log_likelihood, metrics=concordance_index)
			else:
				self.model = tf.keras.models.load_model(model)

		elif checkpoint:
			self.model = self._build_model(hp)
			self.model.load_weights(checkpoint)

		# Generate performance metrics
		log.info("Calculating performance metrics...", 1)
		if permutation_importance:
			auc, r_squared, c_index = sfstats.permutation_feature_importance(self.model, 
																			 dataset_with_slidenames,
																			 self.SLIDE_ANNOTATIONS, 
																			 model_type,
																			 self.DATA_DIR,
																			 outcome_names=self.OUTCOME_NAMES,
																			 label="eval",
																			 manifest=self.MANIFEST,
																			 num_tiles=num_tiles,
																			 feature_names=self.FEATURE_NAMES,
																			 feature_sizes=self.FEATURE_SIZES,
																			 drop_images=((hp.tile_px==0) or hp.drop_images))
		else:
			auc, r_squared, c_index = sfstats.metrics_from_dataset(self.model,
																   model_type=model_type,
																   annotations=self.SLIDE_ANNOTATIONS, 
																   manifest=self.MANIFEST,
																   dataset=dataset_with_slidenames,
																   outcome_names=self.OUTCOME_NAMES,
																   label="eval",
																   data_dir=self.DATA_DIR,
																   num_tiles=num_tiles,
																   histogram=histogram,
																   verbose=True,
																   save_predictions=save_predictions)

		if model_type == 'categorical':
			log.info(f"Tile AUC: {auc['tile']}", 1)
			log.info(f"Slide AUC: {auc['slide']}", 1)
			log.info(f"Patient AUC: {auc['patient']}", 1)
		if model_type == 'linear':
			log.info(f"Tile R-squared: {r_squared['tile']}", 1)
			log.info(f"Slide R-squared: {r_squared['slide']}", 1)
			log.info(f"Patient R-squared: {r_squared['patient']}", 1)
		if model_type == 'cph':
			log.info(f"Tile c-index: {c_index['tile']}", 1)
			log.info(f"Slide c-index: {c_index['slide']}", 1)
			log.info(f"Patient c-index: {c_index['patient']}", 1)
		
		val_metrics = self.model.evaluate(dataset, verbose=(log.INFO_LEVEL > 0), return_dict=True)

		results_log = os.path.join(self.DATA_DIR, 'results_log.csv')
		log.info(f"Evaluation metrics:", 1)
		for m in val_metrics:
			log.info(f"{m}: {val_metrics[m]:.4f}", 2)

		results_dict = 	{ 'eval': val_metrics }

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

		sfutil.update_results_log(results_log, 'eval_model', results_dict)
		
		return val_metrics

	def train(self, hp, pretrain='imagenet', pretrain_model_format=None, resume_training=None, checkpoint=None, log_frequency=100, multi_image=False, 
				validate_on_batch=512, val_batch_size=32, validation_steps=200, max_tiles_per_slide=0, min_tiles_per_slide=0, starting_epoch=0,
				ema_observations=20, ema_smoothing=2, steps_per_epoch_override=None, use_tensorboard=False, multi_gpu=False, save_predictions=False,
				skip_metrics=False):
		'''Train the model.
		
		Args:
			hp:						HyperParameters object
			pretrain:				Either None, 'imagenet' or path to Tensorflow model for pretrained weights
			resume_training:		If True, will attempt to resume previously aborted training
			checkpoint:				Path to cp.cpkt checkpoint file. If provided, will load checkpoint weights
			log_frequency:			How frequent to update Tensorboard logs
			multi_image:			If True, will train model with multi-image inputs
			validate_on_batch:		Validation will be performed every X batches
			val_batch_size:			Batch size to use during validation
			validation_steps:		Number of batches to use for each instance of validation
			max_tiles_per_slide:	If provided, will select only up to this maximum number of tiles from each slide
			min_tiles_per_slide:	If provided, will only evaluate slides with a given minimum number of tiles
			starting_epoch:			Starts training at the specified epoch
			ema_observations:		Number of observations over which to perform exponential moving average smoothing
			ema_smoothing:			Exponential average smoothing value
			steps_per_epoch_override:	If provided, will manually set the number of steps per epoch.
			
		Returns:
			Results dictionary, Keras history object'''

		if multi_gpu:
			strategy = tf.distribute.MirroredStrategy()
			log.info(f"Multi-GPU training with {strategy.num_replicas_in_sync} devices", 1)
		else:
			strategy = None

		with strategy.scope() if strategy is not None else no_scope():
			# Build inputs
			with tf.name_scope('input'):
				train_data, _, num_tiles = self._interleave_tfrecords(self.TRAIN_TFRECORDS, 
																	batch_size=hp.batch_size, 
																	balance=hp.balanced_training, 
																	finite=False,
																	max_tiles=max_tiles_per_slide,
																	min_tiles=min_tiles_per_slide,
																	multi_image=multi_image,
																	include_slidenames=False,
																	augment=hp.augment)
			# Set up validation data
			using_validation = (self.VALIDATION_TFRECORDS and len(self.VALIDATION_TFRECORDS))
			if using_validation:
				with tf.name_scope('input'):
					validation_data, validation_data_with_slidenames, num_val_tiles = self._interleave_tfrecords(self.VALIDATION_TFRECORDS, 
																									batch_size=val_batch_size, 
																									balance=hp.balanced_validation, 
																									finite=True,
																									max_tiles=max_tiles_per_slide,
																									min_tiles=min_tiles_per_slide,
																									multi_image=multi_image,
																									include_slidenames=True,
																									augment=False)

				val_log_msg = "" if not validate_on_batch else f"every {sfutil.bold(str(validate_on_batch))} steps and "
				log.info(f"Validation during training: {val_log_msg}at epoch end", 1)
				if validation_steps:
					validation_data_for_training = validation_data.repeat()
					log.empty(f"Using {validation_steps} batches ({validation_steps * hp.batch_size} samples) each validation check", 2)
				else:
					validation_data_for_training = validation_data
					log.empty(f"Using entire validation set each validation check", 2)
			else:
				log.info("Validation during training: None", 1)
				validation_data_for_training = None
				validation_steps = 0

		# Prepare results
		results = {'epochs': {}}

		# Calculate parameters
		if max(hp.finetune_epochs) <= starting_epoch:
			log.error(f"Starting epoch ({starting_epoch}) cannot be greater than the maximum target epoch ({max(hp.finetune_epochs)})", 1)
			return None, None
		if hp.early_stop and hp.early_stop_method == 'accuracy' and hp.model_type() != 'categorical':
			log.error(f"Unable to use early stopping method 'accuracy' with a non-categorical model type (type: '{hp.model_type()}')")
		if starting_epoch != 0:
			log.info(f"Starting training at epoch {starting_epoch}", 1)
		total_epochs = hp.toplayer_epochs + (max(hp.finetune_epochs) - starting_epoch)
		steps_per_epoch = round(num_tiles/hp.batch_size) if steps_per_epoch_override is None else steps_per_epoch_override
		results_log = os.path.join(self.DATA_DIR, 'results_log.csv')

		# Create callbacks for early stopping, checkpoint saving, summaries, and history
		history_callback = tf.keras.callbacks.History()
		checkpoint_path = os.path.join(self.DATA_DIR, "cp.ckpt")
		cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
														 save_weights_only=True,
														 verbose=(log.INFO_LEVEL > 0))
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.DATA_DIR, 
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
				if log.INFO_LEVEL > 0: print("\r\033[K", end="")
				self.epoch_count += 1
				if self.epoch_count in [e for e in hp.finetune_epochs]:
					model_path = os.path.join(parent.DATA_DIR, f"trained_model_epoch{self.epoch_count}")
					self.model.save(model_path)

					# Try to copy model settings/hyperparameters file into the model folder
					try:
						shutil.copy(os.path.join(os.path.dirname(model_path), 'hyperparameters.json'), os.path.join(model_path, 'hyperparameters.json'), )
						shutil.copy(os.path.join(os.path.dirname(model_path), 'slide_manifest.log'), os.path.join(model_path, 'slide_manifest.log'), )
					except:
						log.warning("Unable to copy model hyperparameters.json & slide_manifest.log files into model directory.", 1)

					log.complete(f"Trained model saved to {sfutil.green(model_path)}", 1)
					if parent.VALIDATION_TFRECORDS and len(parent.VALIDATION_TFRECORDS):
						self.evaluate_model(logs)
				elif self.early_stop:
					self.evaluate_model(logs)
				self.model.stop_training = self.early_stop

			def on_train_batch_end(self, batch, logs={}):
				if using_validation and validate_on_batch and (batch > 0) and (batch % validate_on_batch == 0):
					val_metrics = self.model.evaluate(validation_data, verbose=0, steps=validation_steps, return_dict=True)
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
					if log.INFO_LEVEL > 0: print("\r\033[K", end="")
					self.moving_average += [early_stop_value]
					# Base logging message
					if self.model_type == 'categorical':
						log_message = f"{sfutil.info(f'Batch {batch:<5}')} {sfutil.green('loss')}: {logs['loss']:.3f}, {sfutil.green('acc')}: {train_acc} | {sfutil.purple('val_loss')}: {val_loss:.3f}, {sfutil.purple('val_acc')}: {val_acc}"
					else:
						log_message = f"{sfutil.info(f'Batch {batch:<5}')} {sfutil.green('loss')}: {logs['loss']:.3f} | {sfutil.purple('val_loss')}: {val_loss:.3f}"
					# First, skip moving average calculations if using an invalid metric
					if self.model_type != 'categorical' and hp.early_stop_method == 'accuracy':
						log.empty(log_message)
					else:
						# Calculate exponential moving average of validation accuracy
						if len(self.moving_average) <= ema_observations:
							log.empty(log_message)
						else:
							# Only keep track of the last [ema_observations] validation accuracies
							self.moving_average.pop(0)
							if self.last_ema == -1:
								# Calculate simple moving average
								self.last_ema = sum(self.moving_average) / len(self.moving_average)
								log.empty(log_message +  f" (SMA: {self.last_ema:.3f})")
							else:
								# Update exponential moving average
								self.last_ema = (early_stop_value * (ema_smoothing/(1+ema_observations))) + (self.last_ema * (1-(ema_smoothing/(1+ema_observations))))
								log.empty(log_message + f" (EMA: {self.last_ema:.3f})")

						# If early stopping and our patience criteria has been met, check if validation accuracy is still improving 
						if hp.early_stop and (self.last_ema != -1) and (float(batch)/steps_per_epoch)+self.epoch_count > hp.early_stop_patience:
							if (self.ema_two_checks_prior != -1 and
								((hp.early_stop_method == 'accuracy' and self.last_ema <= self.ema_two_checks_prior) or 
								 (hp.early_stop_method == 'loss' and self.last_ema >= self.ema_two_checks_prior))):

								log.info(f"Early stop triggered: epoch {self.epoch_count+1}, batch {batch}", 1)
								self.model.stop_training = True
								self.early_stop = True
							else:
								self.ema_two_checks_prior = self.ema_one_check_prior
								self.ema_one_check_prior = self.last_ema

			def on_train_end(self, logs={}):
				if log.INFO_LEVEL > 0: print("\r\033[K")

			def evaluate_model(self, logs={}):
				epoch = self.epoch_count
				epoch_label = f"val_epoch{epoch}"
				if not skip_metrics:
					auc, r_squared, c_index = sfstats.metrics_from_dataset(self.model,
																		   model_type=hp.model_type(),
																		   annotations=parent.SLIDE_ANNOTATIONS,
																		   manifest=parent.MANIFEST,
																		   dataset=validation_data_with_slidenames,
																		   outcome_names=parent.OUTCOME_NAMES,
																		   label=epoch_label,
																		   data_dir=parent.DATA_DIR,
																		   num_tiles=num_val_tiles,
																		   histogram=False,
																		   verbose=True,
																		   save_predictions=save_predictions)

				val_metrics = self.model.evaluate(validation_data, verbose=0, return_dict=True)
				log.info(f"Validation metrics:", 1)
				for m in val_metrics:
					log.info(f"{m}: {val_metrics[m]:.4f}", 2)
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
				sfutil.update_results_log(results_log, 'trained_model', {f'epoch{epoch}': epoch_results})
							
		callbacks = [history_callback, PredictionAndEvaluationCallback(), cp_callback]
		if use_tensorboard:
			callbacks += [tensorboard_callback]
		
		with strategy.scope() if strategy is not None else no_scope():
			# Build or load model
			if resume_training:
				log.info(f"Resuming training from {sfutil.green(resume_training)}", 1)
				self.model = tf.keras.models.load_model(resume_training)
			elif multi_image:
				self.model = self._build_multi_image_model(hp, pretrain=pretrain, checkpoint=checkpoint)
			else:
				self.model = self._build_model(hp, 
											pretrain=pretrain, 
											pretrain_model_format=pretrain_model_format, 
											checkpoint=checkpoint)

			# Retrain top layer only, if using transfer learning and not resuming training
			if hp.toplayer_epochs:
				self._retrain_top_layers(hp, train_data, validation_data_for_training, steps_per_epoch, 
										callbacks=None, epochs=hp.toplayer_epochs)

			# Fine-tune the model
			log.info("Beginning fine-tuning", 1)
			self._compile_model(hp)

			#tf.debugging.enable_check_numerics()

			history = self.model.fit(train_data,
								 steps_per_epoch=steps_per_epoch,
								 epochs=total_epochs,
								 verbose=(log.INFO_LEVEL > 0),
								 initial_epoch=hp.toplayer_epochs,
								 validation_data=validation_data_for_training,
								 validation_steps=validation_steps,
								 callbacks=callbacks)

		return results, history.history
