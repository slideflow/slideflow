# Copyright (C) James Dolezal - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, October 2017
# ==========================================================================

# Update 3/2/2019: Beginning tf.data implementation
# Update 5/29/2019: Supports both loose image tiles and TFRecords, 
#   annotations supplied by separate annotation file upon initial model call

''''Builds a CNN model.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import sys
import shutil
from datetime import datetime

import numpy as np
import pickle
import argparse

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorboard.plugins.custom_scalar import layout_pb2

from glob import glob
from scipy.stats import linregress
from statistics import median
from numpy.random import choice

from util import tfrecords, sfutil
from util.sfutil import TCGAAnnotations

BALANCE_BY_CATEGORY = 'BALANCE_BY_CATEGORY'
BALANCE_BY_CASE = 'BALANCE_BY_CASE'
NO_BALANCE = 'NO_BALANCE'

class HyperParameters:
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
		'ResNet': tf.keras.applications.ResNet,
		'ResNetV2': tf.keras.applications.ResNetV2,
		'ResNeXt': tf.keras.applications.ResNeXt,
		'InceptionV3': tf.keras.applications.InceptionV3,
		'InceptionResNetV2': tf.keras.applications.InceptionResNetV2,
		'MobileNet': tf.keras.applications.MobileNet,
		'MobileNetV2': tf.keras.applications.MobileNetV2,
		'DenseNet': tf.keras.applications.DenseNet,
		'NASNet': tf.keras.applications.NASNet
	}
	def __init__(self, toplayer_epochs=5, finetune_epochs=50, model='InceptionV3', pooling='avg', loss='sparse_categorical_crossentropy',
				 learning_rate=0.1, batch_size=16, hidden_layers=0, optimizer='Adam', early_stop=False, 
				 early_stop_patience=0, balanced_training=BALANCE_BY_CATEGORY, balanced_validation=NO_BALANCE, 
				 augment=True):
		''' Additional hyperparameters to consider:
		beta1 0.9
		beta2 0.999
		epsilon 1.0
		batch_norm_decay 0.99
		'''
		self.toplayer_epochs = toplayer_epochs
		self.finetune_epochs = finetune_epochs
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

	def get_opt(self):
		return self._OptDict[self.optimizer](lr=self.learning_rate)

	def get_model(self, input_shape, weights):
		return self._ModelDict[self.model](
			input_shape=input_shape,
			include_top=False,
			pooling=self.pooling,
			weights=weights
		)

	def _get_args(self):
		return [arg for arg in dir(self) if not arg[0]=='_' and arg not in ['get_opt', 'get_model']]

	def __str__(self):
		output = f" + [{sfutil.info('INFO')}] Hyperparameters:\n"
		args = self._get_args()
		for arg in args:
			value = getattr(self, arg)
			output += f"   - {sfutil.header(arg)} = {value}\n"
		return output

class SlideflowModel:
	''' Model containing all functions necessary to build input dataset pipelines,
	build a training and validation set model, and monitor and execute training.'''
	def __init__(self, data_directory, input_directory, annotations_file, image_size, num_classes,
				 manifest=None, use_tfrecord=True, tfrecords_by_case=False, use_fp16=True, num_tiles=10000):
		self.DATA_DIR = data_directory
		self.INPUT_DIR = input_directory
		self.MODEL_DIR = self.DATA_DIR # Directory where to write event logs and checkpoints.
		self.TRAIN_DIR = os.path.join(self.MODEL_DIR, 'train') # Directory where to write eval logs and summaries.
		self.TEST_DIR = os.path.join(self.MODEL_DIR, 'test') # Directory where to write eval logs and summaries.
		self.TRAIN_FILES = os.path.join(self.INPUT_DIR, "train_data/*/*.jpg")
		self.TEST_FILES = os.path.join(self.INPUT_DIR, "eval_data/*/*.jpg")
		self.USE_TFRECORD = use_tfrecord
		self.ANNOTATIONS_FILE = annotations_file
		self.MANIFEST = manifest
		self.TFRECORDS_BY_CASE = tfrecords_by_case # Whether to expect a tfrecord file for each case/slide
		self.IMAGE_SIZE = image_size
		self.USE_FP16 = use_fp16
		self.DTYPE = tf.float16 if self.USE_FP16 else tf.float32

		self.NUM_TILES = 1000 #num_tiles # TODO calculate this automatically
		self.NUM_CLASSES = num_classes # TODO calculate this automatically

		annotations = sfutil.get_annotations_dict(annotations_file, key_name="slide", value_name="category")

		with tf.device('/cpu'):
			self.ANNOTATIONS_TABLE = tf.lookup.StaticHashTable(
				tf.lookup.KeyValueTensorInitializer(list(annotations.keys()), list(annotations.values())), -1
			)

		if not os.path.exists(self.MODEL_DIR):
			os.makedirs(self.MODEL_DIR)

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

	def _parse_tfrecord_function(self, record):
		features = tf.io.parse_single_example(record, tfrecords.FEATURE_DESCRIPTION)
		case = features['case']
		label = self.ANNOTATIONS_TABLE.lookup(case)
		image_string = features['image_raw']
		image = self._process_image(image_string, self.AUGMENT)
		return image, label

	def _gen_batched_dataset(self, filename, batch_size, augment):
		self.AUGMENT = augment
		dataset = tf.data.TFRecordDataset(filename)
		dataset = dataset.shuffle(1000)
		dataset = dataset.map(self._parse_tfrecord_function, num_parallel_calls = 8)
		dataset = dataset.batch(batch_size)
		return dataset

	def _interleave_tfrecords(self, folder, batch_size, balance, augment, finite):
		'''Generates an interleaved dataset from a collection of tfrecord files,
		sampling from tfrecord files randomly according to balancing if provided.
		Requires self.MANIFEST. Assumes TFRecord files are named by case.

		Args:
			folder		Location to search for TFRecord files
			batch_size	Batch size
			balance		Whether to use balancing for batches. Options are BALANCE_BY_CATEGORY,
							BALANCE_BY_CASE, and NO_BALANCE. If finite option is used, will drop
							tiles in order to maintain proportions across the interleaved dataset.
			augment		Whether to use data augmentation (random flip/rotate)
			finite		Whether create finite or infinite datasets. WARNING: If finite option is 
							used with balancing, some tiles will be skipped.'''
		self.AUGMENT = augment
		annotations = sfutil.get_annotations_dict(self.ANNOTATIONS_FILE, key_name=sfutil.TCGAAnnotations.case, 
																		 value_name="category")
		datasets = []
		datasets_categories = []
		num_tiles = []
		categories = {}
		categories_prob = {}
		categories_tile_fraction = {}
		search_folder = os.path.join(self.INPUT_DIR, folder)
		tfrecord_files = glob(os.path.join(search_folder, "*.tfrecords"))
		if tfrecord_files == []:
			print(f" + [{sfutil.fail('ERROR')}] No TFRecords found in {sfutil.green(search_folder)}")
			sys.exit()
		for filename in tfrecord_files:
			dataset_to_add = tf.data.TFRecordDataset(filename) if finite else tf.data.TFRecordDataset(filename).repeat()
			datasets += [dataset_to_add]
			case_shortname = filename.split('/')[-1][:-10]
			category = annotations[case_shortname]
			datasets_categories += [category]
			tiles = self.MANIFEST[filename]['total']
			if category not in categories.keys():
				categories.update({category: {'num_cases': 1,
											  'num_tiles': tiles}})
			else:
				categories[category]['num_cases'] += 1
				categories[category]['num_tiles'] += tiles
			num_tiles += [tiles]
		for category in categories:
			lowest_category_case_count = min([categories[i]['num_cases'] for i in categories])
			lowest_category_tile_count = min([categories[i]['num_tiles'] for i in categories])
			categories_prob[category] = lowest_category_case_count / categories[category]['num_cases']
			categories_tile_fraction[category] = lowest_category_tile_count / categories[category]['num_tiles']
		if balance == NO_BALANCE:
			print(f" + [{sfutil.info('INFO')}] Not balancing input from {sfutil.green(folder)}")
			prob_weights = [i/sum(num_tiles) for i in num_tiles]
		if balance == BALANCE_BY_CASE:
			print(f" + [{sfutil.info('INFO')}] Balancing input from {sfutil.green(folder)} across cases")
			prob_weights = None
			if finite:
				# Only take as many tiles as the number of tiles in the smallest dataset
				for i in range(len(datasets)):
					num_to_take = min(num_tiles)
					datasets[i] = datasets[i].take(num_to_take)
		if balance == BALANCE_BY_CATEGORY:
			print(f" + [{sfutil.info('INFO')}] Balancing input from {sfutil.green(folder)} across categories")
			prob_weights = [categories_prob[datasets_categories[i]] for i in range(len(datasets))]
			if finite:
				# Only take as many tiles as the number of tiles in the smallest category
				for i in range(len(datasets)):
					num_to_take = num_tiles[i] * categories_tile_fraction[datasets_categories[i]]
					datasets[i] = datasets[i].take(num_to_take)
		# Remove empty cases
		for i in sorted(range(len(prob_weights)), reverse=True):
			if num_tiles[i] == 0:
				del(num_tiles[i])
				del(datasets[i])
				del(datasets_categories[i])
				del(prob_weights[i])
		dataset = tf.data.experimental.sample_from_datasets(datasets, weights=prob_weights)
		#dataset = dataset.shuffle(1000)
		dataset = dataset.map(self._parse_tfrecord_function, num_parallel_calls = 8)
		dataset = dataset.batch(batch_size)
		return dataset

	def build_dataset_inputs(self, filename_dir, subfolder, tfrecord_file, batch_size, balance, augment, finite=False):
		'''Args:
			filename_dir:	Directory in which to search for image tiles, if applicable
			subfolder:		Sub-directory in which to search for tfrecords, if applicable
			tfrecord_file:	Name of tfrecord file containing all records, if applicable
			balance:		Whether to use input balancing; options are BALANCE_BY_CASE, BALANCE_BY_CATEGORY, NO_BALANCE
								 (only available if TFRECORDS_BY_CASE=True)'''
		with tf.name_scope('input'):
			if self.TFRECORDS_BY_CASE:
				dataset = self._interleave_tfrecords(subfolder, batch_size, balance, augment, finite)
			else:
				if balance:
					print(f" + [{sfutil.warn('WARN')}] Unable to use balanced inputs unless each case/slide has its own tfrecord file")
				dataset = self._gen_batched_dataset(os.path.join(self.INPUT_DIR, tfrecord_file), batch_size, augment)
		return dataset

	def build_model(self, hp, pretrain=None, checkpoint=None):
		# Assemble base model, using pretraining (imagenet) or the base layers of a supplied model
		if pretrain:
			print(f" + [{sfutil.info('INFO')}] Using pretraining from {sfutil.green(pretrain)}")
		if pretrain and pretrain!='imagenet':
			# Load pretrained model
			pretrained_model = tf.keras.models.load_model(pretrain)
			base_model = pretrained_model.get_layer(index=0)
		else:
			# Create model using ImageNet if specified
			base_model = hp.get_model(input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3),
									  weights=pretrain)

		# Combine base model with top layer (classification/prediction layer)
		layers = [base_model]
		for i in range(hp.hidden_layers):
			layers += [tf.keras.layers.Dense(1000, activation='relu')]
		layers += [tf.keras.layers.Dense(self.NUM_CLASSES, activation='softmax')]
		model = tf.keras.Sequential(layers)
		
		if checkpoint:
			print(f" + [{sfutil.info('INFO')}] Loading checkpoint weights from {sfutil.green(checkpoint)}")
			model.load_weights(checkpoint)

		return model

	def evaluate(self, hp, model=None, checkpoint=None):
		data_to_eval = self.build_dataset_inputs(self.TEST_FILES, 'eval', 'test.tfrecords', hp.batch_size, False, hp.augment, finite=True)
		if model:
			loaded_model = tf.keras.models.load_model(model)
		elif checkpoint:
			loaded_model = self.build_model(hp)
			loaded_model.load_weights(checkpoint)
		loaded_model.compile(loss='sparse_categorical_crossentropy',
					optimizer=tf.keras.optimizers.Adam(lr=hp.learning_rate),
					metrics=['accuracy'])
		results = loaded_model.evaluate(data_to_eval)
		return results

	def retrain_top_layers(self, model, hp, train_data, test_data, callbacks=None, epochs=1):
		print(f" + [{sfutil.info('INFO')}] Retraining top layer")
		# Freeze the base layer
		model.layers[0].trainable = False

		steps_per_epoch = round(self.NUM_TILES/hp.batch_size)
		val_steps = 100 if test_data else None
		lr_fast = hp.learning_rate * 10

		model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr_fast),
					  loss='sparse_categorical_crossentropy',
					  metrics=['accuracy'])

		toplayer_model = model.fit(train_data,
				  epochs=epochs,
				  steps_per_epoch=steps_per_epoch,
				  validation_data=test_data,
				  validation_steps=val_steps,
				  callbacks=callbacks)

		# Unfreeze the base layer
		model.layers[0].trainable = True
		return toplayer_model.history

	def train_unsupervised(self, hp, pretrain='imagenet', resume_training=None, checkpoint=None):
		'''Trains a model with the given hyperparameters (hp)'''
		# Todo: for pretraining, support imagenet, full model, and checkpoints

		# Calculated parameters
		total_epochs = hp.toplayer_epochs + hp.finetune_epochs
		initialized_optimizer = hp.get_opt()
		steps_per_epoch = round(self.NUM_TILES/hp.batch_size)
		tf.keras.layers.BatchNormalization = sfutil.UpdatedBatchNormalization

		train_data = self.build_dataset_inputs(self.TRAIN_FILES, 'train', 'train.tfrecords', hp.batch_size, hp.balanced_training, hp.augment)
		test_data = self.build_dataset_inputs(self.TEST_FILES, 'eval', 'test.tfrecords', hp.batch_size, hp.balanced_validation, hp.augment, finite=True)
		
		history = tf.keras.callbacks.History()
		if hp.early_stop:
			early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=hp.early_stop_patience)
			callbacks = [history, early_stop]
		else:
			callbacks = [history]
		model = self.build_model(hp, pretrain=pretrain)

		if hp.toplayer_epochs:
			self.retrain_top_layers(model, hp, train_data.repeat(), None, callbacks=[history], epochs=hp.toplayer_epochs)
		
		model.compile(loss=hp.loss,
					optimizer=initialized_optimizer,
					metrics=['accuracy'])
		
		finetune_model = model.fit(train_data.repeat(),
			steps_per_epoch=steps_per_epoch,
			epochs=total_epochs,
			verbose=1,
			initial_epoch=hp.toplayer_epochs,
			validation_data=None,
			callbacks=callbacks)

		print("Training complete")
		train_acc = finetune_model.history['accuracy']
		print("Starting validation")
		val_loss, val_acc = model.evaluate(test_data, verbose=1)
		return train_acc, val_loss, val_acc

	def train_supervised(self, hp, pretrain='imagenet', resume_training=None, checkpoint=None, log_frequency=20):
		'''Train the model for a number of steps, according to flags set by the argument parser.'''
		# Calculated parameters
		total_epochs = hp.toplayer_epochs + hp.finetune_epochs
		initialized_optimizer = hp.get_opt()
		steps_per_epoch = round(self.NUM_TILES/hp.batch_size)
		tf.keras.layers.BatchNormalization = sfutil.UpdatedBatchNormalization

		val_steps = 200

		train_data = self.build_dataset_inputs(self.TRAIN_FILES, 'train', 'train.tfrecords', hp.batch_size, hp.balance, hp.augment)
		test_data = self.build_dataset_inputs(self.TEST_FILES, 'eval', 'test.tfrecords', hp.batch_size, hp.balance, hp.augment)

		# Create callbacks for checkpoint saving, summaries, and history
		checkpoint_path = os.path.join(self.MODEL_DIR, "cp.ckpt")
		cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
														 save_weights_only=True,
														 verbose=1)
		
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.DATA_DIR, 
															  histogram_freq=0,
															  write_graph=False,
															  update_freq=hp.batch_size*log_frequency)
		history = tf.keras.callbacks.History()
		callbacks = [cp_callback, tensorboard_callback, history]

		# Load the model
		if resume_training:
			print(f" + [{sfutil.info('INFO')}] Resuming training from {sfutil.green(resume_training)}")
			model = tf.keras.models.load_model(resume_training)
		else:
			model = self.build_model(hp, pretrain=pretrain, checkpoint=checkpoint)

		model.save(os.path.join(self.DATA_DIR, "untrained_model.h5"))

		# TODO: simplify to (if toplayer_epochs) only once hyperparameter support implemented
		# Retrain top layer only if using transfer learning and not resuming training
		if pretrain and not (resume_training or checkpoint) and hp.toplayer_epochs:
			self.retrain_top_layers(model, hp, train_data.repeat(), test_data.repeat(), callbacks=callbacks, epochs=hp.toplayer_epochs)

		model.save(os.path.join(self.DATA_DIR, "toplayer_trained_model.h5"))

		# Fine-tune the model
		print(f" + [{sfutil.info('INFO')}] Beginning fine-tuning")

		# Code for fixing a model that did not have batch_norm update
		'''for layer in model.layers[0].layers:
			if "batch_normalization" not in layer.name:
				layer.trainable=False'''

		model.compile(loss='sparse_categorical_crossentropy',
					optimizer=initialized_optimizer,
					metrics=['accuracy'])

		# Fine-tune model
		finetune_model = model.fit(train_data.repeat(),
			steps_per_epoch=steps_per_epoch,
			epochs=total_epochs,
			initial_epoch=hp.toplayer_epochs,
			validation_data=test_data.repeat(),
			validation_steps=val_steps,
			callbacks=callbacks)

		model.save(os.path.join(self.DATA_DIR, "trained_model.h5"))
		return finetune_model.history['val_accuracy']