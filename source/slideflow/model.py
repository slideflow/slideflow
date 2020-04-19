# Copyright (C) James Dolezal - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, September 2019
# ==========================================================================

# Update 3/2/2019: Beginning tf.data implementation
# Update 5/29/2019: Supports both loose image tiles and TFRecords, 
#   annotations supplied by separate annotation file upon initial model call

'''Builds a CNN model.'''

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
import slideflow.statistics as sfstats

BALANCE_BY_CATEGORY = 'BALANCE_BY_CATEGORY'
BALANCE_BY_PATIENT = 'BALANCE_BY_PATIENT'
NO_BALANCE = 'NO_BALANCE'

def add_regularization(model, regularizer):
	# this function is from "https://sthalles.github.io/keras-regularizer/"
	# adds regularizer to layers where you can set such an attribute, in this case, L2 regularization.
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
		'InceptionResNetV2': tf.keras.applications.InceptionResNetV2,
		'MobileNet': tf.keras.applications.MobileNet,
		'MobileNetV2': tf.keras.applications.MobileNetV2,
		#'DenseNet': tf.keras.applications.DenseNet,
		#'NASNet': tf.keras.applications.NASNet
	}
	_LinearLoss = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'logcosh']

	def __init__(self, finetune_epochs=10, toplayer_epochs=0, model='Xception', pooling='max', loss='sparse_categorical_crossentropy',
				 learning_rate=0.0001, batch_size=16, hidden_layers=1, optimizer='Adam', early_stop=False, 
				 early_stop_patience=0, balanced_training=BALANCE_BY_CATEGORY, balanced_validation=NO_BALANCE, 
				 hidden_layer_width=500, trainable_layers=0, L2_weight=0, augment=True):
		''' Additional hyperparameters to consider:
		beta1 0.9
		beta2 0.999
		epsilon 1.0
		batch_norm_decay 0.99
		'''
		self.toplayer_epochs = toplayer_epochs
		self.finetune_epochs = finetune_epochs if type(finetune_epochs) == list else [finetune_epochs]
		self.model = model
		self.pooling = pooling
		self.loss = loss
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.optimizer = optimizer
		self.early_stop = early_stop
		self.hidden_layers = hidden_layers
		self.early_stop_patience = early_stop_patience
		self.balanced_training = balanced_training
		self.balanced_validation = balanced_validation
		self.augment = augment
		self.hidden_layer_width = hidden_layer_width
		self.trainable_layers = trainable_layers
		self.L2_weight = float(L2_weight)

	def _get_args(self):
		return [arg for arg in dir(self) if not arg[0]=='_' and arg not in ['get_opt', 'get_model', 'model_type']]

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

	def get_opt(self):
		'''Returns optimizer with appropriate learning rate.'''
		return self._OptDict[self.optimizer](lr=self.learning_rate)

	def get_model(self, input_shape, weights):
		'''Returns a Keras model of the appropriate architecture, input shape, pooling, and initial weights.'''
		return self._ModelDict[self.model](
			input_shape=input_shape,
			include_top=False,
			pooling=self.pooling,
			weights=weights
		)

	def model_type(self):
		'''Returns either 'linear' or 'categorical' depending on the loss type.'''
		model_type = 'linear' if self.loss in self._LinearLoss else 'categorical'
		return model_type

class SlideflowModel:
	''' Model containing all functions necessary to build input dataset pipelines,
	build a training and validation set model, and monitor and execute training.'''
	def __init__(self, data_directory, image_size, slide_annotations, train_tfrecords, validation_tfrecords, manifest=None, use_fp16=True, model_type='categorical'):
		self.DATA_DIR = data_directory # Directory where to write event logs and checkpoints.
		self.MANIFEST = manifest
		self.IMAGE_SIZE = image_size
		self.USE_FP16 = use_fp16
		self.DTYPE = tf.float16 if self.USE_FP16 else tf.float32
		self.SLIDE_ANNOTATIONS = slide_annotations # Dictionary mapping slide names to both patient names and outcome
		self.TRAIN_TFRECORDS = train_tfrecords
		self.VALIDATION_TFRECORDS = validation_tfrecords
		self.MODEL_TYPE = model_type
		self.SLIDES = list(slide_annotations.keys())
		outcomes = [slide_annotations[slide]['outcome'] for slide in self.SLIDES]

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
		elif model_type == 'linear':
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
								min_tiles=None, include_slidenames=False, multi_input=False):
		'''Assembles dataset inputs from tfrecords.
		
		Args:
			folders:		Array of directories in which to search for slides (subfolders) containing tfrecords
			balance:		Whether to use input balancing; options are BALANCE_BY_PATIENT, BALANCE_BY_CATEGORY, NO_BALANCE
								 (only available if TFRECORDS_BY_PATIENT=True)'''
		self.AUGMENT = augment
		with tf.name_scope('input'):
			dataset, dataset_with_slidenames, num_tiles = self._interleave_tfrecords(tfrecords, batch_size, balance, finite, max_tiles=max_tiles,
																															 min_tiles=min_tiles,
																															 include_slidenames=include_slidenames,
																															 multi_input=multi_input)
		return dataset, dataset_with_slidenames, num_tiles

	def _build_model(self, hp, pretrain=None, checkpoint=None):
		# Assemble base model, using pretraining (imagenet) or the base layers of a supplied model
		if pretrain:
			log.info(f"Using pretraining from {sfutil.green(pretrain)}", 1)
		if pretrain and pretrain!='imagenet':
			# Load pretrained model
			pretrained_model = tf.keras.models.load_model(pretrain)
			base_model = pretrained_model.get_layer(index=0)
		else:
			# Create model using ImageNet if specified
			base_model = hp.get_model(input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3),
									  weights=pretrain)

		# Add L2 regularization to all compatible layers in the base model
		if hp.L2_weight != 0:
			regularizer = tf.keras.regularizers.l2(hp.L2_weight)
			base_model = add_regularization(base_model, regularizer)
		else:
			regularizer = None

		# Allow only a subset of layers to be trainable
		if hp.trainable_layers != 0:
			# freezeIndex is the layer from 0 up to Index that should be frozen (not trainable). 
			# Per Jakob's models, all but last 10, 20, or 30 layers were frozen. His last three layers were a 1000-node fully connected layer (eqv. to our hidden layers), 
			# then softmax, then classification. It looks like we don't add a classification layer on though, I don't see it added anywhere.
			# I see below that we add on the hidden layer and softmax layer, so I am freezing (10-2=8) only, since when we add the last layers on it will add up to 10, 20, 30
			freezeIndex = int(len(base_model.layers) - (hp.trainable_layers - hp.hidden_layers - 1))
			for layer in base_model.layers[:freezeIndex]:
				layer.trainable = False

		# Base model and pooling
		# James 4/18/2020: The model is collapsed into one layer to allow for easier loading of pretrained models
		layers = [base_model]
		if not hp.pooling:
			layers += [tf.keras.layers.Flatten()]

		# Add hidden layers if specified
		for i in range(hp.hidden_layers):
				layers += [tf.keras.layers.Dense(hp.hidden_layer_width, activation='relu', kernel_regularizer=regularizer)]

		# Add the softmax prediction layer
		activation = 'linear' if hp.model_type() == 'linear' else 'softmax'
		layers += [tf.keras.layers.Dense(self.NUM_CLASSES, activation=activation, kernel_regularizer=regularizer)]
		model = tf.keras.Sequential(layers)

		if checkpoint:
			log.info(f"Loading checkpoint weights from {sfutil.green(checkpoint)}", 1)
			model.load_weights(checkpoint)

		return model
	
	def _build_multi_input_model(self, hp, pretrain=None, checkpoint=None):
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

	def _interleave_tfrecords(self, tfrecords, batch_size, balance, finite, max_tiles=None, min_tiles=None, include_slidenames=False, multi_input=False):
		'''Generates an interleaved dataset from a collection of tfrecord files,
		sampling from tfrecord files randomly according to balancing if provided.
		Requires self.MANIFEST. Assumes TFRecord files are named by slide.

		Args:
			tfrecords			Array of paths to TFRecord files
			batch_size			Batch size
			balance				Whether to use balancing for batches. Options are BALANCE_BY_CATEGORY,
									BALANCE_BY_PATIENT, and NO_BALANCE. If finite option is used, will drop
									tiles in order to maintain proportions across the interleaved dataset.
			augment				Whether to use data augmentation (random flip/rotate)
			finite				Whether create finite or infinite datasets. WARNING: If finite option is 
									used with balancing, some tiles will be skipped.
			max_tiles			Maximum number of tiles to use per slide.
			min_tiles			Minimum number of tiles that each slide must have to be included. '''
							 
		log.info(f"Interleaving {len(tfrecords)} tfrecords: finite={finite}, max_tiles={max_tiles}, min_tiles={min_tiles}", 1)
		datasets = []
		datasets_categories = []
		num_tiles = []
		global_num_tiles = 0
		categories = {}
		categories_prob = {}
		categories_tile_fraction = {}
		
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
				sys.exit()
			
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
			dataset_to_add = tf.data.TFRecordDataset(filename) if finite else tf.data.TFRecordDataset(filename).repeat()
			datasets += [dataset_to_add]
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
		global_num_tiles = sum(num_tiles)

		# Interleave datasets
		try:
			dataset = tf.data.experimental.sample_from_datasets(datasets, weights=prob_weights)
		except IndexError:
			log.error(f"No TFRecords found after filter criteria; please ensure all tiles have been extracted and all TFRecords are in the appropriate folder", 1)
			sys.exit()
		if include_slidenames:
			dataset_with_slidenames = dataset.map(partial(self._parse_tfrecord_function, include_slidenames=True, multi_input=multi_input), num_parallel_calls = 8)
			dataset_with_slidenames = dataset_with_slidenames.batch(batch_size)
		else:
			dataset_with_slidenames = None
		dataset = dataset.map(partial(self._parse_tfrecord_function, include_slidenames=False, multi_input=multi_input), num_parallel_calls = 8)
		dataset = dataset.batch(batch_size)
		
		return dataset, dataset_with_slidenames, global_num_tiles

	def _parse_tfrecord_function(self, record, include_slidenames=True, multi_input=False):
		feature_description = tfrecords.FEATURE_DESCRIPTION if not multi_input else tfrecords.FEATURE_DESCRIPTION_MULTI
		features = tf.io.parse_single_example(record, feature_description)
		slide = features['slide']
		if self.MODEL_TYPE == 'linear':
			label = [self.ANNOTATIONS_TABLES[oi].lookup(slide) for oi in range(self.NUM_CLASSES)]
		else:
			label = self.ANNOTATIONS_TABLES[0].lookup(slide)
		if multi_input:
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
			if include_slidenames:
				return image, label, slide
			else:
				return image, label

	def _process_image(self, image_string, augment):
		image = tf.image.decode_jpeg(image_string, channels = 3)
		image = tf.image.per_image_standardization(image)

		if augment:
			# Apply augmentations
			# Rotate 0, 90, 180, 270 degrees
			image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

			# Random flip and rotation
			image = tf.image.random_flip_left_right(image)
			image = tf.image.random_flip_up_down(image)

		dtype = tf.float16 if self.USE_FP16 else tf.float32
		image = tf.image.convert_image_dtype(image, dtype)
		image.set_shape([self.IMAGE_SIZE, self.IMAGE_SIZE, 3])
		return image

	def _retrain_top_layers(self, hp, train_data, validation_data, steps_per_epoch, callbacks=None, epochs=1):
		log.info("Retraining top layer", 1)
		# Freeze the base layer
		self.model.layers[0].trainable = False
		val_steps = 100 if validation_data else None
		metrics = ['acc'] if hp.model_type() != 'linear' else [hp.loss]

		self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=hp.learning_rate),
					  loss=hp.loss,
					  metrics=metrics)

		toplayer_model = self.model.fit(train_data,
				  epochs=epochs,
				  verbose=1,
				  steps_per_epoch=steps_per_epoch,
				  validation_data=validation_data,
				  validation_steps=val_steps,
				  callbacks=callbacks)

		# Unfreeze the base layer
		model.layers[0].trainable = True
		return toplayer_model.history

	def evaluate(self, tfrecords, hp=None, model=None, model_type='categorical', checkpoint=None, batch_size=None, 
					max_tiles_per_slide=0, min_tiles_per_slide=0, multi_input=False):
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
			multi_input:			If true, will evaluate model with multi-image inputs.
			
		Returns:
			Keras history object.'''

		# Load and initialize model
		if not hp and checkpoint:
			log.error("If using a checkpoint for evaluation, hyperparameters must be specified.")
			sys.exit()
		batch_size = batch_size if not hp else hp.batch_size
		dataset, dataset_with_slidenames, num_tiles = self._build_dataset_inputs(tfrecords, batch_size, NO_BALANCE, augment=False, 
																													finite=True,
																													max_tiles=max_tiles_per_slide,
																													min_tiles=min_tiles_per_slide,
																													include_slidenames=True,
																													multi_input=multi_input)
		if model:
			self.model = tf.keras.models.load_model(model)
		elif checkpoint:
			self.model = self._build_model(hp)
			self.model.load_weights(checkpoint)

		# Generate performance metrics
		log.info("Calculating performance metrics...", 1)
		tile_auc, slide_auc, patient_auc, r_squared = sfstats.generate_performance_metrics(self.model, dataset_with_slidenames, self.SLIDE_ANNOTATIONS, 
																						   model_type, self.DATA_DIR, label="eval", manifest=self.MANIFEST)

		log.info(f"Tile AUC: {tile_auc}", 1)
		log.info(f"Slide AUC: {slide_auc}", 1)
		log.info(f"Patient AUC: {patient_auc}", 1)
		log.info(f"R-squared: {r_squared}", 1)

		val_loss, val_acc = self.model.evaluate(dataset)

		# Log results
		results_log = os.path.join(self.DATA_DIR, 'results_log.csv')
		results_dict = {
			'eval': {
				'val_loss': val_loss,
				'val_acc': val_acc,
				'tile_auc': tile_auc,
				'slide_auc': slide_auc,
				'patient_auc': patient_auc,
				'r_squared': r_squared
			}
		}
		sfutil.update_results_log(results_log, 'eval_model', results_dict)

		#with open(results_log, "w") as results_file:
		#	writer = csv.writer(results_file)
		#	writer.writerow(['val_loss', 'val_acc', 'tile_auc', 'slide_auc', 'patient_auc', 'r_squared'])
		#	writer.writerow([val_loss, val_acc, tile_auc, slide_auc, patient_auc, r_squared])
		
		return val_acc

	def train(self, hp, pretrain='imagenet', resume_training=None, checkpoint=None, log_frequency=100, multi_input=False, 
				validate_on_batch=256, max_tiles_per_slide=0, min_tiles_per_slide=0, ema_observations=8, ema_smoothing=2):
		'''Train the model for a number of steps, according to flags set by the argument parser.
		
		Args:
			hp:						HyperParameters object
			pretrain				Either None, 'imagenet' or path to .h5 file for pretrained weights
			resume_training			If True, will attempt to resume previously aborted training
			checkpoint				Path to cp.cpkt checkpoint file. If provided, will load checkpoint weights
			log_frequency			How frequent to update Tensorboard logs
			multi_input				If True, will train model with multi-image inputs
			validate_on_batch		Validation will be performed every X batches.
			max_tiles_per_slide		If provided, will select only up to this maximum number of tiles from each slide.
			min_tiles_per_slide		If provided, will only evaluate slides with a given minimum number of tiles.
			
			
		Returns:
			Results dictionary, Keras history object'''

		#tf.keras.layers.BatchNormalization = sfutil.UpdatedBatchNormalization

		# Build inputs
		train_data, _, num_tiles = self._build_dataset_inputs(self.TRAIN_TFRECORDS, hp.batch_size, hp.balanced_training, hp.augment, finite=False,
																																	 max_tiles=max_tiles_per_slide,
																																	 min_tiles=min_tiles_per_slide,
																																	 include_slidenames=False,
																																	 multi_input=multi_input)
		using_validation = (self.VALIDATION_TFRECORDS and len(self.VALIDATION_TFRECORDS))
		if using_validation:
			validation_data, validation_data_with_slidenames, _ = self._build_dataset_inputs(self.VALIDATION_TFRECORDS, hp.batch_size, hp.balanced_validation, augment=False, 
																																							   finite=True,
																																							   max_tiles=max_tiles_per_slide,
																																							   min_tiles=min_tiles_per_slide,
																																							   include_slidenames=True, 
																																							   multi_input=multi_input)
		validation_data_for_training = None if not using_validation else validation_data.repeat()
		val_steps = 0 if not using_validation else 200

		# Prepare results
		results = {'epochs': {}}

		# Calculate parameters
		total_epochs = hp.toplayer_epochs + max(hp.finetune_epochs)
		steps_per_epoch = round(num_tiles/hp.batch_size)
		initialized_optimizer = hp.get_opt()
		results_log = os.path.join(self.DATA_DIR, 'results_log.csv')
		metrics = ['acc'] if hp.model_type() != 'linear' else [hp.loss]

		# Create callbacks for early stopping, checkpoint saving, summaries, and history
		history_callback = tf.keras.callbacks.History()
		checkpoint_path = os.path.join(self.DATA_DIR, "cp.ckpt")
		cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
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
				self.val_acc_moving_average = []
				self.ema_two_checks_prior = 0
				self.ema_one_check_prior = 0
				self.epoch_count = 0

			def on_epoch_end(self, epoch, logs={}):
				print("\r\033[K", end="")
				self.epoch_count += 1
				if epoch+1 in [e for e in hp.finetune_epochs]:
					model_path = os.path.join(parent.DATA_DIR, f"trained_model_epoch{epoch+1}.h5")
					self.model.save(model_path)
					log.complete(f"Trained model saved to {sfutil.green(model_path)}", 1)
					if parent.VALIDATION_TFRECORDS and len(parent.VALIDATION_TFRECORDS):
						self.evaluate_model(logs)
				self.model.stop_training = self.early_stop

			def on_train_batch_end(self, batch, logs={}):
				if (batch > 0) and (batch % validate_on_batch == 0) and (parent.VALIDATION_TFRECORDS and len(parent.VALIDATION_TFRECORDS)):
					val_loss, val_acc = self.model.evaluate(validation_data, verbose=0)
					print("\r\033[K", end="")
					self.val_acc_moving_average += [val_acc]
					# Calculate exponential moving average of validation accuracy
					if len(self.val_acc_moving_average) <= ema_observations:
						log.empty(f"Batch {batch:<5} loss: {logs['loss']:.3f}, acc: {logs['acc']:.3f} | val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f}")
					else:
						# Only keep track of the last [ema_observations] validation accuracies
						self.val_acc_moving_average.pop(0)
						if self.last_ema == -1:
							# Calculate simple moving average
							self.last_ema = sum(self.val_acc_moving_average) / len(self.val_acc_moving_average)
							log.empty(f"Batch {batch:<5} loss: {logs['loss']:.3f}, acc: {logs['acc']:.3f} | val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f} (SMA: {self.last_ema:.3f})")
						else:
							# Update exponential moving average
							self.last_ema = (val_acc * (ema_smoothing/(1+ema_observations))) + (self.last_ema * (1-(ema_smoothing/(1+ema_observations))))
							log.empty(f"Batch {batch:<5} loss: {logs['loss']:.3f}, acc: {logs['acc']:.3f} | val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f} (EMA: {self.last_ema:.3f})")

					# If early stopping and our patience criteria has been met, check if validation accuracy is still improving 
					if hp.early_stop and (self.last_ema != -1) and (float(batch)/steps_per_epoch)+self.epoch_count > hp.early_stop_patience:
						if self.last_ema <= self.ema_two_checks_prior:
							log.info(f"Early stop triggered: epoch {self.epoch_count+1}, batch {batch}", 1)
							self.model.stop_training = True
							self.early_stop = True
						else:
							self.ema_two_checks_prior = self.ema_one_check_prior
							self.ema_one_check_prior = self.last_ema

			def on_train_end(self, logs={}):
				print("\r\033[K")

			def evaluate_model(self, logs={}):
				epoch = self.epoch_count
				epoch_label = f"val_epoch{epoch}"
				if hp.model_type() != 'linear':
					train_acc = logs['acc']
				else:
					train_acc = logs[hp.loss]
				tile_auc, slide_auc, patient_auc, r_squared = sfstats.generate_performance_metrics(parent.model, validation_data_with_slidenames, 
																									parent.SLIDE_ANNOTATIONS, hp.model_type(), 
																									parent.DATA_DIR, label=epoch_label, manifest=parent.MANIFEST)
				val_loss, val_acc = parent.model.evaluate(validation_data, verbose=0)
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
							
		callbacks = [history_callback, PredictionAndEvaluationCallback(), cp_callback, tensorboard_callback]

		# Build or load model
		if resume_training:
			log.info(f"Resuming training from {sfutil.green(resume_training)}", 1)
			self.model = tf.keras.models.load_model(resume_training)
		elif multi_input:
			self.model = self._build_multi_input_model(hp, pretrain=pretrain, checkpoint=checkpoint)
		else:
			self.model = self._build_model(hp, pretrain=pretrain, checkpoint=checkpoint)

		# Retrain top layer only if using transfer learning and not resuming training
		if hp.toplayer_epochs:
			self._retrain_top_layers(hp, train_data.repeat(), validation_data_for_training, steps_per_epoch, 
									callbacks=None, epochs=hp.toplayer_epochs)

		# Fine-tune the model
		log.info("Beginning fine-tuning", 1)

		self.model.compile(loss=hp.loss,
						   optimizer=initialized_optimizer,
						   metrics=metrics)

		history = self.model.fit(train_data.repeat(),
								 steps_per_epoch=steps_per_epoch,
								 epochs=total_epochs,
								 verbose=1,
								 initial_epoch=hp.toplayer_epochs,
								 validation_data=validation_data_for_training,
								 validation_steps=val_steps,
								 callbacks=callbacks)

		return results, history.history
