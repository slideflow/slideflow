# Copyright (C) James Dolezal - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, September 2019
# ==========================================================================

# Update 3/2/2019: Beginning tf.data implementation
# Update 5/29/2019: Supports both loose image tiles and TFRecords, 
#   annotations supplied by separate annotation file upon initial model call

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import sys
import shutil
import pickle
import argparse
import gc
import csv
import random
import warnings
import tempfile
warnings.filterwarnings('ignore')

import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorflow.python.framework import ops
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from datetime import datetime
from glob import glob
from scipy import stats
from statistics import median
from sklearn import metrics
from matplotlib import pyplot as plt
from functools import partial
from slideflow.util import TCGA, log
from slideflow.io import tfrecords

import slideflow.util as sfutil
from slideflow.util import StainNormalizer
import slideflow.statistics as sfstats

BALANCE_BY_CATEGORY = 'BALANCE_BY_CATEGORY'
BALANCE_BY_PATIENT = 'BALANCE_BY_PATIENT'
NO_BALANCE = 'NO_BALANCE'
MODEL_FORMAT_1_9 = "1.9"
MODEL_FORMAT_CURRENT = MODEL_FORMAT_1_9
MODEL_FORMAT_LEGACY = "legacy"


#def negative_log_likelihood(input_data):
#	def loss(y_true,y_pred):
#		hazard_ratio = tf.math.exp(y_pred)
#		log_risk = tf.math.log(tf.math.cumsum(hazard_ratio))
#		uncensored_likelihood = tf.transpose(y_pred) - log_risk
#		censored_likelihood = uncensored_likelihood * input_data
#		neg_likelihood = -tf.reduce_sum(censored_likelihood)
#		return neg_likelihood
#	return loss
	
	
def negative_log_likelihood(y_true, y_pred):
	import sys
	E = y_pred[:, -1]
	y_pred = y_pred[:, :-1]
	E = tf.reshape(E, [-1])
	y_pred = tf.reshape(y_pred, [-1])
	y_true = tf.reshape(y_true, [-1])
	#tf.print("y_pred: ", y_pred, output_stream=sys.stdout)
	#tf.print("y_true: ", y_true, output_stream=sys.stdout)
	#tf.print("E: ", E, output_stream=sys.stdout)
	order = tf.argsort(y_true)
	E = tf.gather(E, order)
	y_pred = tf.gather(y_pred, order)
	#tf.print("y_pred, sort: ", y_pred, output_stream=sys.stdout)
	#tf.print("E, sort: ", E, output_stream=sys.stdout)
	#hazard_ratio = tf.math.exp(y_pred)
	#log_risk = tf.math.log(tf.math.cumsum(hazard_ratio))
	#uncensored_likelihood = tf.transpose(y_pred) - log_risk
	#censored_likelihood = uncensored_likelihood * E
	#neg_likelihood = -tf.reduce_sum(censored_likelihood)
	
	gamma = tf.math.reduce_max(y_pred)
	eps = tf.constant(1e-7, dtype=tf.float16)
	log_cumsum_h = tf.math.add(tf.math.log(tf.math.add(tf.math.cumsum(tf.math.exp(tf.math.subtract(y_pred, gamma))), eps)), gamma)
	return -tf.math.divide(tf.reduce_sum(tf.math.multiply(tf.subtract(y_pred, log_cumsum_h), E)),tf.reduce_sum(E))
	
	#return neg_likelihood


# Fix for broken batch normalization in TF 1.14
#tf.keras.layers.BatchNormalization = sfutil.UpdatedBatchNormalization


def concordance_index(y_true, y_pred):
	E = y_pred[:, -1]
	y_pred = y_pred[:, :-1]
	E = tf.reshape(E, [-1])
	y_pred = tf.reshape(y_pred, [-1])
	y_pred = -y_pred #negative of log hazard ratio to have correct relationship with survival
	g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
	g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)
	f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
	event = tf.multiply(tf.transpose(E), E)
	f = tf.multiply(tf.cast(f, tf.float32), event)
	f = tf.compat.v1.matrix_band_part(tf.cast(f, tf.float32), -1, 0)
	g = tf.reduce_sum(tf.multiply(g, f))
	f = tf.reduce_sum(f)
	return tf.where(tf.equal(f, 0), 0.0, g/f)


def add_regularization(model, regularizer):
	'''Adds regularization (e.g. L2) to all eligible layers of a model.
	This function is from "https://sthalles.github.io/keras-regularizer/" '''
	if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
		print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
		return model

	for layer in model.layers:
		for attr in ['kernel_regularizer']:
			if hasattr(layer, attr):
				setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
	model_json = model.to_json()

	# Save the weights before reloading the model.
	tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
	model.save_weights(tmp_weights_path)

	# load the model from the config
	model = tf.keras.models.model_from_json(model_json)

	# Reload the model weights
	model.load_weights(tmp_weights_path, by_name=True)
	return model

class HyperParameterError(Exception):
	pass

class ManifestError(Exception):
	pass

class ModelError(Exception):
	pass

class ModelActivationsInterface:
	'''Provides an interface to obtain logits and post-convolutional activations from saved Slideflow Keras models.
	Provides support for newer models (v1.9.1+) and legacy slideflow models (1.9.0b and earlier)'''

	def __init__(self, path, model_format=None):
		'''Initializer.
		
		Args:
			path:			Path to saved Slideflow Keras model
			model_format:	Either slideflow.model.MODEL_FORMAT_CURRENT or _LEGACY. Indicates how the saved model should be processed,
								as older versions of Slideflow had models constructed differently, with differing naming of Keras layers.
		'''
		if not model_format: model_format = MODEL_FORMAT_CURRENT
		self.model_format = model_format

		self.path = path
		_model = tf.keras.models.load_model(path)

		# CONSIDER IMPLEMENTING THIS MORE EFFICIENT VERSION: ===========
		#inputs = xception.input
		#outputs = [xception.get_layer(name=layer_name).output for layer_name in activation_layer_names]
		#self.functor = tf.keras.backend.function(inputs, outputs)
		# ===============================================================
		
		if model_format == MODEL_FORMAT_1_9:
			try:
				loaded_model = tf.keras.models.Model(inputs=[_model.input],
													outputs=[_model.get_layer(name="post_convolution").output, _model.output])
				model_input = tf.keras.layers.Input(shape=loaded_model.input_shape[1:])
				model_output = loaded_model(model_input)
				self.model = tf.keras.Model(model_input, model_output)
			except ValueError:
				log.warn("Unable to read model using modern format, will try legacy model format", 1)
				model_format = MODEL_FORMAT_LEGACY

		if model_format == MODEL_FORMAT_LEGACY:
			loaded_model = tf.keras.models.Model(inputs=[_model.input, _model.layers[0].layers[0].input],
											outputs=[_model.layers[0].layers[-1].output, _model.layers[-1].output])
			model_input = tf.keras.layers.Input(shape=loaded_model.input_shape[0][1:])
			model_output = loaded_model([model_input, model_input])
			self.model = tf.keras.Model(model_input, model_output)

		self.NUM_CLASSES = _model.layers[-1].output_shape[-1]

	def predict(self, image_batch):
		'''Given a batch of images, will return a batch of post-convolutional activations and a batch of logits.'''
		# ======================
		#return self.functor(image_batch)
		# ======================
		if self.model_format == MODEL_FORMAT_1_9:
			return self.model.predict(image_batch)
		elif self.model_format == MODEL_FORMAT_LEGACY:
			return self.model.predict([image_batch, image_batch])

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
		#'ResNet101': tf.keras.applications.ResNet101,
		#'ResNet152': tf.keras.applications.ResNet152,
		#'ResNet50V2': tf.keras.applications.ResNet50V2,
		#'ResNet101V2': tf.keras.applications.ResNet101V2,
		#'ResNet152V2': tf.keras.applications.ResNet152V2,
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
	_LinearLoss = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'logcosh', 'negative_log_likelihood']

	_AllLoss = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge'
				'categorical_hinge', 'logcosh', 'huber_loss', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy',
				'kullback_leibler_divergence', 'poisson', 'cosine_proximity', 'is_categorical_crossentropy', 'negative_log_likelihood']


	def __init__(self, tile_px=299, tile_um=302, finetune_epochs=10, toplayer_epochs=0, model='Xception', pooling='max', loss='sparse_categorical_crossentropy',
				 learning_rate=0.0001, batch_size=16, hidden_layers=1, hidden_layer_width=500, optimizer='Adam', early_stop=False, 
				 early_stop_patience=0, early_stop_method='loss', balanced_training=BALANCE_BY_CATEGORY, balanced_validation=NO_BALANCE, 
				 trainable_layers=0, L2_weight=0, augment=True, drop_images=False):
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
		assert isinstance(augment, bool)
		assert isinstance(drop_images, bool)
		
		self.tile_px = tile_px
		self.tile_um = tile_um
		self.toplayer_epochs = toplayer_epochs
		self.finetune_epochs = finetune_epochs if isinstance(finetune_epochs, list) else [finetune_epochs]
		self.model = model
		self.pooling = pooling if pooling != 'none' else None
		self.loss = loss
		self.learning_rate = learning_rate
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
			return False
		return True

	def get_opt(self):
		'''Returns optimizer with appropriate learning rate.'''
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
		model_type = 'linear' if self.loss in self._LinearLoss else 'categorical'
		model_type = 'cph' if self.loss == 'negative_log_likelihood' else model_type
		return model_type

class SlideflowModel:
	''' Model containing all functions necessary to build input dataset pipelines,
	build a training and validation set model, and monitor and execute training.'''
	def __init__(self, data_directory, image_size, slide_annotations, train_tfrecords, validation_tfrecords, 
					manifest=None, use_fp16=True, model_type='categorical', normalizer=None, normalizer_source=None, num_slide_input=0, feature_sizes = None, feature_names = None):
		'''Model initializer.

		Args:
			data_directory:			Location where event logs and checkpoints will be written
			image_size:				Int, width/height of input image in pixels.
			slide_annotations:		Dictionary mapping slide names to both patient names and outcome
			train_tfrecords:		List of tfrecord paths for training
			validation_tfrecords:	List of tfrecord paths for validation
			manifest:				Manifest dictionary mapping TFRecords to number of tiles
			use_fp16:				Bool, if True, will use FP16 (rather than FP32)
			model_type:				Type of model outcome, either 'categorical' or 'linear'
			normalizer:				Tile image normalization to perform in real-time during training
			normalizer_source:		Source image for normalization if being performed in real-time
		'''
		self.DATA_DIR = data_directory
		self.MANIFEST = manifest
		self.IMAGE_SIZE = image_size
		self.DTYPE = 'float16' if use_fp16 else 'float32'
		self.SLIDE_ANNOTATIONS = slide_annotations
		self.TRAIN_TFRECORDS = train_tfrecords
		self.VALIDATION_TFRECORDS = validation_tfrecords
		self.MODEL_TYPE = model_type
		self.SLIDES = list(slide_annotations.keys())
		self.DATASETS = {}
		self.NUM_SLIDE_INPUT = num_slide_input
		self.EVENT_TENSOR = None
		self.FEATURE_SIZES = feature_sizes
		self.FEATURE_NAMES = feature_names
		
		
		outcomes = [slide_annotations[slide]['outcome'] for slide in self.SLIDES]

		# Setup slide-level input
		if num_slide_input:
			try:
				self.SLIDE_INPUT_TABLE = {slide: slide_annotations[slide]['input'] for slide in self.SLIDES}
				log.info(f"Training with both images and {num_slide_input} categories of slide-level input", 1)
			except KeyError:
				raise ModelError("If num_slide_input > 0, slide-level input must be provided via 'input' key in slide_annotations")
			for slide in self.SLIDES:
				if len(self.SLIDE_INPUT_TABLE[slide]) != num_slide_input:
					raise ModelError(f"Length of input for slide {slide} does not match num_slide_input; expected {num_slide_input}, got {len(self.SLIDE_INPUT_TABLE[slide])}")

		# Normalization setup
		if normalizer: log.info(f"Using realtime {normalizer} normalization", 1)
		self.normalizer = None if not normalizer else StainNormalizer(method=normalizer, source=normalizer_source)

		# Setup outcome hash tables
		if model_type == 'categorical':
			try:
				self.NUM_CLASSES = len(list(set(outcomes)))
			except TypeError:
				log.error("Unable to use multiple outcome variables with categorical model type.")
				sys.exit()
			with tf.device('/cpu'):
				self.ANNOTATIONS_TABLES = [tf.lookup.StaticHashTable(
					tf.lookup.KeyValueTensorInitializer(self.SLIDES, outcomes), -1
				)]
		elif model_type == 'linear' or model_type == 'cph':
			try:
				self.NUM_CLASSES = len(outcomes[0])
			except TypeError:
				log.error("Incorrect formatting of outcomes for a linear model; must be formatted as an array.")
				sys.exit()
			with tf.device('/cpu'):
				self.ANNOTATIONS_TABLES = []
				for oi in range(self.NUM_CLASSES):
					self.ANNOTATIONS_TABLES += [tf.lookup.StaticHashTable(
						tf.lookup.KeyValueTensorInitializer(self.SLIDES, [o[oi] for o in outcomes]), -1
					)]
		else:
			log.error(f"Unknown model type {model_type}")
			sys.exit()

		if not os.path.exists(self.DATA_DIR):
			os.makedirs(self.DATA_DIR)
		
		# Record which slides are used for training and validation, and to which categories they belong
		if train_tfrecords or validation_tfrecords:
			with open(os.path.join(self.DATA_DIR, 'slide_manifest.log'), 'w') as slide_manifest:
				writer = csv.writer(slide_manifest)
				header = ['slide', 'dataset', 'outcome']
				writer.writerow(header)
				if train_tfrecords:
					for tfrecord in train_tfrecords:
						slide = tfrecord.split('/')[-1][:-10]
						if slide in self.SLIDES:
							outcome = slide_annotations[slide]['outcome']
							writer.writerow([slide, 'training', outcome])
				if validation_tfrecords:
					for tfrecord in validation_tfrecords:
						slide = tfrecord.split('/')[-1][:-10]
						if slide in self.SLIDES:
							outcome = slide_annotations[slide]['outcome']
							writer.writerow([slide, 'validation', outcome])

	def _build_dataset_inputs(self, tfrecords, batch_size, balance, augment, finite=False, max_tiles=None, 
								min_tiles=0, include_slidenames=False, multi_image=False, parse_fn=None, drop_remainder=False):
		'''Assembles dataset inputs from tfrecords.
		
		Args:
			tfrecords:				List of tfrecords paths
			batch_size:				Batch size
			balance:				Whether to use input balancing; options are BALANCE_BY_PATIENT, BALANCE_BY_CATEGORY, NO_BALANCE
								 		(only available if TFRECORDS_BY_PATIENT=True)
			augment:				Bool, whether to perform image augmentation (random flipping/rotating)
			finite:					Bool, whether dataset should be finite or infinite (with dataset.repeat())
			max_tiles:				Int, limits number of tiles to use for each TFRecord if supplied
			min_tiles:				Int, only includes TFRecords with this minimum number of tiles
			include_slidenames:		Bool, if True, dataset will include slidename (each entry will return image, label, and slidename)
			multi_image:			Bool, if True, will read multiple images from each TFRecord record.
		'''
		self.AUGMENT = augment
		with tf.name_scope('input'):
			dataset, dataset_with_slidenames, num_tiles = self._interleave_tfrecords(tfrecords, batch_size, balance, finite, max_tiles=max_tiles,
																															 min_tiles=min_tiles,
																															 include_slidenames=include_slidenames,
																															 multi_image=multi_image,
																															 parse_fn=parse_fn,
																															 drop_remainder=drop_remainder)
		return dataset, dataset_with_slidenames, num_tiles

	def _build_model(self, hp, pretrain=None, pretrain_model_format=None, checkpoint=None):
		''' Assembles base model, using pretraining (imagenet) or the base layers of a supplied model.

		Args:
			hp:			HyperParameters object
			pretrain:	Either 'imagenet' or path to model to use as pretraining
			checkpoint:	Path to checkpoint from which to resume model training
		'''
		if self.DTYPE == 'float16':
			log.info("Training with mixed precision", 1)
			policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
			mixed_precision.set_policy(policy)

		# Setup inputs
		image_shape = (self.IMAGE_SIZE, self.IMAGE_SIZE, 3)
		tile_input_tensor = tf.keras.Input(shape=image_shape, name="tile_image")
		if self.NUM_SLIDE_INPUT:
			if self.MODEL_TYPE == 'cph':
				slide_input_tensor = tf.keras.Input(shape=(self.NUM_SLIDE_INPUT - 1), name="slide_input")
			else:
				slide_input_tensor = tf.keras.Input(shape=(self.NUM_SLIDE_INPUT), name="slide_input")

		if hp.loss == 'negative_log_likelihood':
			event_input_tensor = tf.keras.Input(shape=(1), name="event_input")
		# Load pretrained model if applicable
		if pretrain: log.info(f"Using pretraining from {sfutil.green(pretrain)}", 1)
		if pretrain and pretrain!='imagenet':
			pretrained_model = tf.keras.models.load_model(pretrain)
			if not pretrain_model_format or pretrain_model_format == MODEL_FORMAT_CURRENT:
				try:
					pretrained_input = pretrained_model.get_layer(name="tile_image").input # This is the tile_image input
					pretrained_name = pretrained_model.get_layer(index=1).name # Name of the pretrained model core, which should be at layer 1
					pretrained_output = pretrained_model.get_layer(name="post_convolution").output # This is the post-convolution layer
					base_model = tf.keras.Model(inputs=pretrained_input, outputs=pretrained_output, name=f"pretrained_{pretrained_name}")
				except ValueError:
					log.warn("Unable to read pretrained model using modern format, will try legacy model format", 1)
					pretrain_model_format = MODEL_FORMAT_LEGACY
			if pretrain_model_format == MODEL_FORMAT_LEGACY:
				base_model = pretrained_model.get_layer(index=0)
		else:
			# Create core model
			base_model = hp.get_model(#input_tensor=tile_input_tensor,
									  weights=pretrain)

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
			freezeIndex = int(len(base_model.layers) - (hp.trainable_layers - hp.hidden_layers - 1))
			for layer in base_model.layers[:freezeIndex]:
				layer.trainable = False

		# Create sequential tile model: 
		# 	tile image --> convolutions --> pooling/flattening --> hidden layers ---> prelogits --> softmax/logits
		#                             additional slide input --/
		post_convolution_identity_layer = tf.keras.layers.Lambda(lambda x: x, name="post_convolution") # This is an identity layer that simply returns the last layer, allowing us to name and access this layer later
		layers = [tile_input_tensor, base_model]
		if not hp.pooling:
			layers += [tf.keras.layers.Flatten()]
		layers += [post_convolution_identity_layer]
		tile_image_model = tf.keras.Sequential(layers)
		model_inputs = [tile_image_model.input]

		# Merge layers
		if self.NUM_SLIDE_INPUT:
			if hp.drop_images:
				log.info("Generating model with just clinical variables and no images", 1)
				merged_model = slide_input_tensor				
			else:
				merged_model = tf.keras.layers.Concatenate(name="input_merge")([slide_input_tensor, tile_image_model.output])
			model_inputs += [slide_input_tensor]
			if hp.loss == 'negative_log_likelihood':
				self.EVENT_TENSOR = event_input_tensor
				model_inputs += [event_input_tensor]
		else:
			merged_model = tile_image_model.output

		# Add hidden layers
		for i in range(hp.hidden_layers):
			merged_model = tf.keras.layers.Dense(hp.hidden_layer_width, name=f"hidden_{i}", activation='relu', kernel_regularizer=regularizer)(merged_model)

		# Add the softmax prediction layer
		activation = 'linear' if (hp.model_type() == 'linear' or hp.model_type() == 'cph') else 'softmax'
		final_dense_layer = tf.keras.layers.Dense(self.NUM_CLASSES, kernel_regularizer=regularizer, name="prelogits")(merged_model)
		softmax_output = tf.keras.layers.Activation(activation, dtype='float32', name='logits')(final_dense_layer)
		if hp.loss == 'negative_log_likelihood':
			softmax_output = tf.keras.layers.Concatenate(name="output_merge_CPH")([softmax_output, event_input_tensor])
		# Assemble final model
		model = tf.keras.Model(inputs=model_inputs, outputs=softmax_output)

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

	def _interleave_tfrecords(self, tfrecords, batch_size, balance, finite, max_tiles=None, min_tiles=None,
								include_slidenames=False, multi_image=False, parse_fn=None, drop_remainder=False):
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
			include_slidenames:		Bool, if True, dataset will include slidename (each entry will return image, label, and slidename)
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
		if not parse_fn: parse_fn = self._parse_tfrecord_function
		
		if tfrecords == []:
			log.error(f"No TFRecords found.", 1)
			sys.exit()

		for filename in tfrecords:
			slide_name = filename.split('/')[-1][:-10]
			
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
				log.info(f"Skipping empty tfrecord {sfutil.green(slide_name)}", 2)
				continue
			elif tiles < min_tiles:
				log.info(f"Skipping tfrecord {sfutil.green(slide_name)}; has {tiles} tiles (minimum: {min_tiles})", 2)
				continue
			
			# Assign category by outcome if this is a categorical model.
			# Otherwise, consider all slides from the same category (effectively skipping balancing); appropriate for linear models.
			category = self.SLIDE_ANNOTATIONS[slide_name]['outcome'] if self.MODEL_TYPE == 'categorical' else 1
			if filename not in self.DATASETS:
				self.DATASETS.update({filename: tf.data.TFRecordDataset(filename, num_parallel_reads=32)}) #buffer_size=1024*1024*100 num_parallel_reads=tf.data.experimental.AUTOTUNE
			datasets += [self.DATASETS[filename]]
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
		for category in categories:
			lowest_category_slide_count = min([categories[i]['num_slides'] for i in categories])
			lowest_category_tile_count = min([categories[i]['num_tiles'] for i in categories])
			categories_prob[category] = lowest_category_slide_count / categories[category]['num_slides']
			categories_tile_fraction[category] = lowest_category_tile_count / categories[category]['num_tiles']

		# Balancing
		if balance == NO_BALANCE:
			log.empty(f"Not balancing input", 2)
			prob_weights = [i/sum(num_tiles) for i in num_tiles]
		if balance == BALANCE_BY_PATIENT:
			log.empty(f"Balancing input across slides", 2)
			prob_weights = [1.0] * len(datasets)
			if finite:
				# Only take as many tiles as the number of tiles in the smallest dataset
				minimum_tiles = min(num_tiles)
				for i in range(len(datasets)):
					num_tiles[i] = minimum_tiles
		if balance == BALANCE_BY_CATEGORY:
			log.empty(f"Balancing input across categories", 2)
			prob_weights = [categories_prob[datasets_categories[i]] for i in range(len(datasets))]
			if finite:
				# Only take as many tiles as the number of tiles in the smallest category
				for i in range(len(datasets)):
					num_tiles[i] = int(num_tiles[i] * categories_tile_fraction[datasets_categories[i]])
					log.empty(f"Tile fraction (dataset {i+1}/{len(datasets)}): {categories_tile_fraction[datasets_categories[i]]}, taking {num_tiles[i]}", 2)
				log.empty(f"Global num tiles: {global_num_tiles}", 2)
		
		# Take the calculcated number of tiles from each dataset and calculate global number of tiles
		for i in range(len(datasets)):
			datasets[i] = datasets[i].take(num_tiles[i])
			if not finite:
				datasets[i] = datasets[i].repeat()
		global_num_tiles = sum(num_tiles)

		# Interleave datasets
		try:
			dataset = tf.data.experimental.sample_from_datasets(datasets, weights=prob_weights)
		except IndexError:
			log.error(f"No TFRecords found after filter criteria; please ensure all tiles have been extracted and all TFRecords are in the appropriate folder", 1)
			sys.exit()
		if include_slidenames:
			dataset_with_slidenames = dataset.map(partial(parse_fn, include_slidenames=True, multi_image=multi_image), num_parallel_calls=32) #tf.data.experimental.AUTOTUNE
			dataset_with_slidenames = dataset_with_slidenames.batch(batch_size, drop_remainder=drop_remainder)
		else:
			dataset_with_slidenames = None
		dataset = dataset.map(partial(parse_fn, include_slidenames=False, multi_image=multi_image), num_parallel_calls = 8)
		dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
		
		return dataset, dataset_with_slidenames, global_num_tiles

	def _parse_tfrecord_function(self, record, include_slidenames=True, multi_image=False):
		'''Parses raw entry read from TFRecord.'''
		feature_description = tfrecords.FEATURE_DESCRIPTION if not multi_image else tfrecords.FEATURE_DESCRIPTION_MULTI
		features = tf.io.parse_single_example(record, feature_description)
		slide = features['slide']
		if self.MODEL_TYPE == 'linear' or self.MODEL_TYPE == 'cph':
			label = [self.ANNOTATIONS_TABLES[oi].lookup(slide) for oi in range(self.NUM_CLASSES)]
		else:
			label = self.ANNOTATIONS_TABLES[0].lookup(slide)

		if multi_image:
			image_dict = {}
			inputs = [inp for inp in list(feature_description.keys()) if inp != 'slide']
			for i in inputs:
				image_string = features[i]
				image = self._process_image(image_string, self.AUGMENT)
				image_dict.update({
					i: image
				})
			if include_slidenames:
				return image_dict, label, slide
			else:
				return image_dict, label	
		else:
			image_string = features['image_raw']
			image = self._process_image(image_string, self.AUGMENT)
			image_dict = { 'tile_image': image }

			if self.NUM_SLIDE_INPUT:
				if self.MODEL_TYPE == 'cph':
					def slide_lookup(s): return self.SLIDE_INPUT_TABLE[s.numpy().decode('utf-8')][1:]
					slide_input_val = tf.py_function(func=slide_lookup, inp=[slide], Tout=[tf.float32] * (self.NUM_SLIDE_INPUT - 1))
					def event_lookup(s): return self.SLIDE_INPUT_TABLE[s.numpy().decode('utf-8')][0]
					event_input_val = tf.py_function(func=event_lookup, inp=[slide], Tout=[tf.float32])
					image_dict.update({'slide_input': slide_input_val})
					image_dict.update({'event_input': event_input_val})
				else:
					def slide_lookup(s): return self.SLIDE_INPUT_TABLE[s.numpy().decode('utf-8')]
					slide_input_val = tf.py_function(func=slide_lookup, inp=[slide], Tout=[tf.float32] * self.NUM_SLIDE_INPUT)
					image_dict.update({'slide_input': slide_input_val})
			if include_slidenames:
				return image_dict, label, slide
			else:
				return image_dict, label

	def _process_image(self, image_string, augment):
		'''Converts a JPEG-encoded image string into RGB array, using normalization if specified.'''
		image = tf.image.decode_jpeg(image_string, channels = 3)

		if self.normalizer:
			image = tf.py_function(self.normalizer.tf_to_rgb, [image], tf.int32)

		image = tf.image.per_image_standardization(image)

		if augment:
			# Apply augmentations
			# Rotate 0, 90, 180, 270 degrees
			image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

			# Random flip and rotation
			image = tf.image.random_flip_left_right(image)
			image = tf.image.random_flip_up_down(image)

		image = tf.image.convert_image_dtype(image, tf.float32)
		image.set_shape([self.IMAGE_SIZE, self.IMAGE_SIZE, 3])
		return image

	def _retrain_top_layers(self, hp, train_data, validation_data, steps_per_epoch, callbacks=None, epochs=1):
		'''Retrains only the top layer of this object's model, while leaving all other layers frozen.'''
		log.info("Retraining top layer", 1)
		# Freeze the base layer
		self.model.layers[0].trainable = False
		val_steps = 200 if validation_data else None
		metrics = ['accuracy'] if hp.model_type() != 'linear' else [hp.loss]
		event_input_tensor = tf.keras.Input(shape=(1), name="event_input")
		if hp.loss == 'negative_log_likelihood':
			self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=hp.learning_rate),
						  loss=negative_log_likelihood(event_input_tensor),
						  metrics=metrics)
		else:
			self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=hp.learning_rate),
						  loss=hp.loss,
						  metrics=metrics)

		toplayer_model = self.model.fit(train_data,
				  epochs=epochs,
				  verbose=(log.INFO_LEVEL > 0),
				  steps_per_epoch=steps_per_epoch,
				  validation_data=validation_data,
				  validation_steps=val_steps,
				  callbacks=callbacks)

		# Unfreeze the base layer
		model.layers[0].trainable = True
		return toplayer_model.history

	def evaluate(self, tfrecords, hp=None, model=None, model_type='categorical', checkpoint=None, batch_size=None, 
					max_tiles_per_slide=0, min_tiles_per_slide=0, multi_image=False, feature_importance=True):
		'''Evaluate model.

		Args:
			tfrecords:				List of TFrecords paths to load for evaluation.
			hp:						HyperParameters object
			model:					Optional; .h5 model to load for evaluation. If None, will build model using hyperparameters.
			model_type:				Either linear or categorical.
			checkpoint:				Path to cp.cpkt checkpoint. If provided, will update model with given checkpoint weights.
			batch_size:				Evaluation batch size.
			max_tiles_per_slide:	If provided, will select only up to this maximum number of tiles from each slide.
			min_tiles_per_slide:	If provided, will only evaluate slides with a given minimum number of tiles.
			multi_image:			If true, will evaluate model with multi-image inputs.
			
		Returns:
			Keras history object.'''

		# Load and initialize model
		if not hp and checkpoint:
			log.error("If using a checkpoint for evaluation, hyperparameters must be specified.")
			sys.exit()
		if not batch_size: batch_size = hp.batch_size
		dataset, dataset_with_slidenames, num_tiles = self._build_dataset_inputs(tfrecords, batch_size, NO_BALANCE, augment=False, 
																													finite=True,
																													max_tiles=max_tiles_per_slide,
																													min_tiles=min_tiles_per_slide,
																													include_slidenames=True,
																													multi_image=multi_image)
		if model:
			if model_type == 'cph':
				self.model = tf.keras.models.load_model(model, custom_objects={'negative_log_likelihood':negative_log_likelihood, 'concordance_index':concordance_index})
			else:
				self.model = tf.keras.models.load_model(model)
		elif checkpoint:
			self.model = self._build_model(hp)
			self.model.load_weights(checkpoint)

		# Generate performance metrics
		log.info("Calculating performance metrics...", 1)
		if feature_importance:
			sfstats.permutation_feature_importance(self.model, dataset_with_slidenames, self.SLIDE_ANNOTATIONS, 
																						   model_type, self.DATA_DIR, label="eval", manifest=self.MANIFEST, num_tiles=num_tiles, num_input = self.NUM_SLIDE_INPUT, feature_names = self.FEATURE_NAMES, feature_sizes = self.FEATURE_SIZES, drop_images = hp.drop_images)
												   
		tile_auc, slide_auc, patient_auc, r_squared, c_index = sfstats.generate_performance_metrics(self.model, dataset_with_slidenames, self.SLIDE_ANNOTATIONS, 
																						   model_type, self.DATA_DIR, label="eval", manifest=self.MANIFEST, num_tiles=num_tiles)

		log.info(f"Tile AUC: {tile_auc}", 1)
		log.info(f"Slide AUC: {slide_auc}", 1)
		log.info(f"Patient AUC: {patient_auc}", 1)
		log.info(f"R-squared: {r_squared}", 1)
		log.info(f"c-index: {c_index}", 1)
		
		val_loss, val_acc = self.model.evaluate(dataset, verbose=log.INFO_LEVEL > 0)

		# Log results
		results_log = os.path.join(self.DATA_DIR, 'results_log.csv')
		results_dict = {
			'eval': {
				'val_loss': val_loss,
				'val_acc': val_acc,
				'tile_auc': tile_auc,
				'slide_auc': slide_auc,
				'patient_auc': patient_auc,
				'r_squared': r_squared,
				'c_index': c_index
			}
		}
		sfutil.update_results_log(results_log, 'eval_model', results_dict)
		
		return val_acc

	def train(self, hp, pretrain='imagenet', pretrain_model_format=None, resume_training=None, checkpoint=None, log_frequency=100, multi_image=False, 
				validate_on_batch=512, val_batch_size=32, validation_steps=200, max_tiles_per_slide=0, min_tiles_per_slide=0, starting_epoch=0,
				ema_observations=20, ema_smoothing=2, steps_per_epoch_override=None):
		'''Train the model for a number of steps, according to flags set by the argument parser.
		
		Args:
			hp:						HyperParameters object
			pretrain:				Either None, 'imagenet' or path to .h5 file for pretrained weights
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

		# Build inputs
		train_data, _, num_tiles = self._build_dataset_inputs(self.TRAIN_TFRECORDS, hp.batch_size, hp.balanced_training, hp.augment, finite=False,
																																	 max_tiles=max_tiles_per_slide,
																																	 min_tiles=min_tiles_per_slide,
																																	 include_slidenames=False,
																																	 multi_image=multi_image)
		# Set up validation data
		using_validation = (self.VALIDATION_TFRECORDS and len(self.VALIDATION_TFRECORDS))
		if using_validation:
			validation_data, validation_data_with_slidenames, _ = self._build_dataset_inputs(self.VALIDATION_TFRECORDS, val_batch_size, hp.balanced_validation, augment=False, 
																																							   finite=True,
																																							   max_tiles=max_tiles_per_slide,
																																							   min_tiles=min_tiles_per_slide,
																																							   include_slidenames=True, 
																																							   multi_image=multi_image)
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
		metrics = ['accuracy'] if hp.model_type() != 'linear' else [hp.loss]

		# Create callbacks for early stopping, checkpoint saving, summaries, and history
		history_callback = tf.keras.callbacks.History()
		checkpoint_path = os.path.join(self.DATA_DIR, "cp.ckpt")
		cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=(log.INFO_LEVEL > 0))
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.DATA_DIR, 
															  histogram_freq=0,
															  write_graph=False,
															  update_freq=log_frequency)
		parent = self

		class PredictionAndEvaluationCallback(tf.keras.callbacks.Callback):
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
					model_path = os.path.join(parent.DATA_DIR, f"trained_model_epoch{self.epoch_count}.h5")
					self.model.save(model_path)
					log.complete(f"Trained model saved to {sfutil.green(model_path)}", 1)
					if parent.VALIDATION_TFRECORDS and len(parent.VALIDATION_TFRECORDS):
						self.evaluate_model(logs)
				self.model.stop_training = self.early_stop

			def on_train_batch_end(self, batch, logs={}):
				if using_validation and validate_on_batch and (batch > 0) and (batch % validate_on_batch == 0):
					val_loss, val_acc = self.model.evaluate(validation_data, verbose=0, steps=validation_steps)
					self.model.stop_training = False
					early_stop_value = val_acc if hp.early_stop_method == 'accuracy' else val_loss
					if log.INFO_LEVEL > 0: print("\r\033[K", end="")
					self.moving_average += [early_stop_value]
					# Base logging message
					if self.model_type == 'categorical':
						log_message = f"Batch {batch:<5} loss: {logs['loss']:.3f}, acc: {logs['accuracy']:.3f} | val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f}"
					else:
						log_message = f"Batch {batch:<5} loss: {logs['loss']:.3f} | val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f}"
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
				if hp.model_type() != 'linear':
					train_acc = logs['accuracy']
				else:
					train_acc = logs[hp.loss]
				tile_auc, slide_auc, patient_auc, r_squared = sfstats.generate_performance_metrics(self.model, validation_data_with_slidenames, 
																									parent.SLIDE_ANNOTATIONS, hp.model_type(), 
																									parent.DATA_DIR, label=epoch_label, manifest=parent.MANIFEST, num_tiles=num_tiles)
				val_loss, val_acc = self.model.evaluate(validation_data, verbose=0)
				log.info(f"Validation loss: {val_loss:.4f} | accuracy: {val_acc:.4f}", 1)
				results['epochs'][f'epoch{epoch}'] = {}
				results['epochs'][f'epoch{epoch}']['train_acc'] = np.amax(train_acc)
				results['epochs'][f'epoch{epoch}']['val_loss'] = val_loss
				results['epochs'][f'epoch{epoch}']['val_acc'] = val_acc
				for i, auc in enumerate(tile_auc):
					results['epochs'][f'epoch{epoch}'][f'tile_auc{i}'] = auc
				for i, auc in enumerate(slide_auc):
					results['epochs'][f'epoch{epoch}'][f'slide_auc{i}'] = auc
				for i, auc in enumerate(patient_auc):
					results['epochs'][f'epoch{epoch}'][f'patient_auc{i}'] = auc
				results['epochs'][f'epoch{epoch}']['r_squared'] = r_squared
				epoch_results = results['epochs'][f'epoch{epoch}']
				sfutil.update_results_log(results_log, 'trained_model', {f'epoch{epoch}': epoch_results})
							
		callbacks = [history_callback, PredictionAndEvaluationCallback(), cp_callback]#, tensorboard_callback]

		# Build or load model
		if resume_training:
			log.info(f"Resuming training from {sfutil.green(resume_training)}", 1)
			self.model = tf.keras.models.load_model(resume_training)
		elif multi_image:
			self.model = self._build_multi_image_model(hp, pretrain=pretrain, checkpoint=checkpoint)
		else:
			self.model = self._build_model(hp, pretrain=pretrain, pretrain_model_format=pretrain_model_format, checkpoint=checkpoint)

		# Retrain top layer only if using transfer learning and not resuming training
		if hp.toplayer_epochs:
			self._retrain_top_layers(hp, train_data, validation_data_for_training, steps_per_epoch, 
									callbacks=None, epochs=hp.toplayer_epochs)

		# Fine-tune the model
		log.info("Beginning fine-tuning", 1)
		if hp.loss == 'negative_log_likelihood':
			self.model.compile(loss=negative_log_likelihood, optimizer=hp.get_opt(), metrics=concordance_index)
		else:
			self.model.compile(loss=hp.loss,
					   optimizer=hp.get_opt(),
					   metrics=metrics)

		history = self.model.fit(train_data,
								 steps_per_epoch=steps_per_epoch,
								 epochs=total_epochs,
								 verbose=(log.INFO_LEVEL > 0),
								 initial_epoch=hp.toplayer_epochs,
								 validation_data=validation_data_for_training,
								 validation_steps=validation_steps,
								 callbacks=callbacks)

		return results, history.history