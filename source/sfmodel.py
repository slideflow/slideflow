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

from util import tfrecords, sfutil
from util.sfutil import TCGAAnnotations

# Calculate accuracy with https://stackoverflow.com/questions/50111438/tensorflow-validate-accuracy-with-batch-data
# TODO: try next, comment out line 254 (results in calculating total_loss before update_ops is called)
# TODO: visualize graph, memory usage, and compute time with https://www.tensorflow.org/guide/graph_viz
# TODO: export logs to file for monitoring remotely

class SFModelConfig:
	def __init__(self, image_size, num_classes, batch_size, num_tiles=10000, augment=False, 
				learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1.0, batch_norm_decay=0.99, 
				early_stop=0.015, max_epoch=300, log_frequency=20, test_frequency=600, use_fp16=True):
		''' Declare constants describing the model and training process.
		Args:
			image_size						Size of input images in pixels
			num_classes						Number of classes
			batch_size						Batch size for training
			augment							Whether or not to perform data augmentation
			learning_rate					Learning rate for the Adams Optimizer
			beta1							Beta1 for AdamOptimizer
			beta2							Beta2 for AdamOptimizer
			epsilon							Epsilon for AdamOptimizer
			batch_norm_decay				Decay rate for batch_norm (0.999 default, use lower numbers if poor validation performance)
			early_stop						Rate of validation loss decay that should trigger early stopping
			max_epoch						Maximum number of times to repeat through training set
			log_frequency					How often to log results to console, in steps
			test_frequency					How often to run validation testing, in steps
			use_fp16						Whether to use FP16 or not (vs. FP32)
		'''		
		self.image_size = image_size
		self.num_classes = num_classes
		self.batch_size = batch_size
		self.num_tiles = num_tiles
		self.augment = augment
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.batch_norm_decay = batch_norm_decay
		self.early_stop = early_stop
		self.max_epoch = max_epoch
		self.log_frequency = log_frequency
		self.test_frequency = test_frequency
		self.use_fp16 = use_fp16

	def get_args(self):
		ignored_args = ['image_size', 'num_classes', 'batch_size', 'use_fp16', 'get_args', 'print_config']
		return [i for i in dir(self) if (not i[0]=='_') and (i not in ignored_args)]

	def print_config(self):
		print(f" + [{sfutil.info('INFO')}] Model configuration:")
		for arg in self.get_args():
			value = getattr(self, arg)
			print(f"   - {sfutil.header(arg)} = {value}")

class SlideflowModel:
	''' Model containing all functions necessary to build input dataset pipelines,
	build a training and validation set model, and monitor and execute training.'''

	def __init__(self, data_directory, input_directory, annotations_file, manifest=None, use_tfrecord=True):
		self.DATA_DIR = data_directory
		self.INPUT_DIR = input_directory
		self.MODEL_DIR = self.DATA_DIR # Directory where to write event logs and checkpoints.
		self.TRAIN_DIR = os.path.join(self.MODEL_DIR, 'train') # Directory where to write eval logs and summaries.
		self.TEST_DIR = os.path.join(self.MODEL_DIR, 'test') # Directory where to write eval logs and summaries.
		self.TRAIN_FILES = os.path.join(self.INPUT_DIR, "train_data/*/*.jpg")
		self.TEST_FILES = os.path.join(self.INPUT_DIR, "eval_data/*/*.jpg")
		self.USE_TFRECORD = use_tfrecord
		self.ANNOTATIONS_FILE = annotations_file
		self.MANIFEST = manifest # Used for balanced augmentation

		annotations = sfutil.get_annotations_dict(annotations_file, key_name="slide", value_name="category")

		# TODO: use verification done by parent slideflow module; if not done, offer to use again
		#tfrecord_files = [self.TRAIN_TFRECORD, self.EVAL_TFRECORD] if self.USE_TFRECORD else []
		#sfutil.verify_tiles(annotations, self.INPUT_DIR, tfrecord_files)

		with tf.device('/cpu'):
			self.ANNOTATIONS_TABLE = tf.lookup.StaticHashTable(
				tf.lookup.KeyValueTensorInitializer(list(annotations.keys()), list(annotations.values())), -1
			)

		if not os.path.exists(self.MODEL_DIR):
			#shutil.rmtree(self.MODEL_DIR)
			os.makedirs(self.MODEL_DIR)

	def config(self, config):
		self.IMAGE_SIZE = config.image_size
		self.NUM_CLASSES = config.num_classes
		self.BATCH_SIZE = config.batch_size
		self.NUM_TILES = config.num_tiles
		self.AUGMENT = config.augment
		self.LEARNING_RATE = config.learning_rate
		self.BETA1 = config.beta1
		self.BETA2 = config.beta2
		self.EPSILON = config.epsilon
		self.BATCH_NORM_DECAY = config.batch_norm_decay
		self.VALIDATION_EARLY_STOP_SLOPE = config.early_stop
		self.MAX_EPOCH = config.max_epoch
		self.LOG_FREQUENCY = config.log_frequency
		self.TEST_FREQUENCY = config.test_frequency
		self.USE_FP16 = config.use_fp16
		self.DTYPE = tf.float16 if self.USE_FP16 else tf.float32
		config.print_config()

	def _process_image(self, image_string):
		image = tf.image.decode_jpeg(image_string, channels = 3)
		image = tf.image.per_image_standardization(image)

		if self.AUGMENT:
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

	def _parse_function(self, filename):
		case = filename.split('/')[-2]
		label = self.ANNOTATIONS_TABLE.lookup(case)
		image_string = tf.read_file(filename)
		image = self._process_image(image_string)
		return image, label

	def _gen_batched_dataset(self, filenames):
		# Replace the below dataset with one that uses a Python generator for flexibility of labeling
		dataset = tf.data.Dataset.from_tensor_slices(filenames)
		dataset = dataset.shuffle(tf.size(filenames, out_type=tf.int64))
		dataset = dataset.map(self._parse_function, num_parallel_calls = 8)
		dataset = dataset.batch(self.BATCH_SIZE)
		return dataset

	def _parse_tfrecord_function(self, record):
		features = tf.io.parse_single_example(record, tfrecords.FEATURE_DESCRIPTION)
		case = features['case']
		label = self.ANNOTATIONS_TABLE.lookup(case)
		image_string = features['image_raw']
		image = self._process_image(image_string)
		return image, label

	def _interleave_tfrecords(self, folder, balanced):
		annotations = sfutil.get_annotations_dict(self.ANNOTATIONS_FILE, key_name=sfutil.TCGAAnnotations.case, 
																		 value_name="category")
		datasets = []
		categories = {}
		category_keep_prob = {}
		keep_prob_weights = [] if self.MANIFEST else None
		tfrecord_files = glob(os.path.join(self.INPUT_DIR, f"{folder}/*.tfrecords"))
		if self.MANIFEST:
			for filename in tfrecord_files:
				datasets += [tf.data.TFRecordDataset(filename).repeat()]
				case_shortname = filename.split('/')[-1][:-10]
				category = annotations[case_shortname]
				if category not in categories.keys():
					categories.update({category: 1})
				else:
					categories[category] += 1
			for category in categories:
				category_keep_prob[category] = min(categories.values()) / categories[category]
			for filename in tfrecord_files:
				case = filename.split('/')[-1][:-10]
				category = annotations[case]
				keep_prob_weights += [category_keep_prob[category]]
		else:
			for filename in tfrecord_files:
				datasets += [tf.data.TFRecordDataset(filename).repeat()]			
		interleaved_dataset = tf.data.experimental.sample_from_datasets(datasets, weights=keep_prob_weights)
		#interleaved_dataset = interleaved_dataset.shuffle(1000)
		interleaved_dataset = interleaved_dataset.map(self._parse_tfrecord_function, num_parallel_calls = 8)
		interleaved_dataset = interleaved_dataset.batch(self.BATCH_SIZE)
		return interleaved_dataset

	def build_inputs(self, balanced=True):
		'''Construct input for the model.'''
		with tf.name_scope('input'):
			if not self.USE_TFRECORD:
				train_dataset = self._gen_batched_dataset(tf.io.match_filenames_once(self.TRAIN_FILES))
				test_dataset = self._gen_batched_dataset(tf.io.match_filenames_once(self.TEST_FILES))
			else:
				train_dataset = self._interleave_tfrecords('train', balanced=balanced)
				test_dataset = self._interleave_tfrecords('eval', balanced=balanced)
		return train_dataset, test_dataset

	def build_model(self, pretrain=None):
		# Assemble base model, using pretraining (imagenet) or the base layers of a supplied model
		if pretrain:
			print(f" + [{sfutil.info('INFO')}] Using pretraining from {sfutil.green(pretrain)}")
		if pretrain and pretrain!='imagenet':
			# Load pretrained model
			pretrained_model = tf.keras.models.load_model(pretrain)
			base_model = pretrained_model.get_layer(index=0)
		else:
			# Create model using ImageNet if specified
			base_model = tf.keras.applications.InceptionV3(
				input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3),
				include_top=False,
				pooling='avg',
				weights=pretrain
			)

		# Combine base model with top layer (classification/prediction layer)
		#fully_connected_layer = tf.keras.layers.Dense(1000, activation='relu')
		prediction_layer = tf.keras.layers.Dense(self.NUM_CLASSES, activation='softmax')
		model = tf.keras.Sequential([
			base_model,
			#fully_connected_layer,
			prediction_layer
		])
		return model

	def retrain_top_layers(self, model, train_data, test_data, callbacks=None, epochs=1):
		print(f" + [{sfutil.info('INFO')}] Retraining top layer")
		# Freeze the base layer
		model.layers[0].trainable = False

		steps_per_epoch = round(self.NUM_TILES/self.BATCH_SIZE)
		lr_fast = self.LEARNING_RATE * 10

		model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr_fast),
					  loss='sparse_categorical_crossentropy',
					  metrics=['accuracy'])

		toplayer_model = model.fit(train_data.repeat(),
				  epochs=epochs,
				  steps_per_epoch=steps_per_epoch,
				  validation_data=test_data.repeat(),
				  validation_steps=100,
				  callbacks=callbacks)

		# Unfreeze the base layer
		model.layers[0].trainable = True
		model.save(os.path.join(self.DATA_DIR, "toplayer_trained_model.h5"))
		return toplayer_model.history

	def evaluate(self, model=None, checkpoint=None, dataset='train'):
		train_data, test_data = self.build_inputs()
		data_to_eval = train_data if dataset=='train' else test_data
		if model:
			loaded_model = tf.keras.models.load_model(model)
		elif checkpoint:
			loaded_model = self.build_model()
			loaded_model.load_weights(checkpoint)
		loaded_model.compile(loss='sparse_categorical_crossentropy',
					optimizer=tf.keras.optimizers.Adam(lr=self.LEARNING_RATE),
					metrics=['accuracy'])
		results = loaded_model.evaluate(train_data)
		print(results)

	def train(self, pretrain='imagenet', resume_training=None, checkpoint=None):
		'''Train the model for a number of steps, according to flags set by the argument parser.'''
		
		tf.keras.layers.BatchNormalization = sfutil.UpdatedBatchNormalization
		train_data, test_data = self.build_inputs()
		steps_per_epoch = round(self.NUM_TILES/self.BATCH_SIZE)
		val_steps = 200
		toplayer_epochs = 5
		finetune_epochs = self.MAX_EPOCH
		total_epochs = toplayer_epochs + finetune_epochs

		# Create callbacks for checkpoint saving, summaries, and history
		checkpoint_path = os.path.join(self.MODEL_DIR, "cp.ckpt")
		cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
														 save_weights_only=True,
														 verbose=1)
		
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.DATA_DIR, 
															  histogram_freq=0,
															  write_graph=False,
															  update_freq=self.BATCH_SIZE*self.LOG_FREQUENCY)
		history = tf.keras.callbacks.History()
		callbacks = [cp_callback, tensorboard_callback, history]

		# Load the model
		if resume_training:
			print(f" + [{sfutil.info('INFO')}] Resuming training from {sfutil.green(resume_training)}")
			model = tf.keras.models.load_model(resume_training)
		else:
			model = self.build_model(pretrain)
		if checkpoint:
			print(f" + [{sfutil.info('INFO')}] Loading checkpoint weights from {sfutil.green(checkpoint)}")
			model.load_weights(checkpoint)

		# Retrain top layer only if using transfer learning and not resuming training
		if pretrain and not (resume_training or checkpoint):
			self.retrain_top_layers(model, train_data, test_data, callbacks=callbacks, epochs=toplayer_epochs)
		
		# Fine-tune the model
		print(f" + [{sfutil.info('INFO')}] Beginning fine-tuning")

		for layer in model.layers[0].layers:
			if "batch_normalization" not in layer.name:
				layer.trainable=False

		model.compile(loss='sparse_categorical_crossentropy',
					optimizer=tf.keras.optimizers.Adam(lr=self.LEARNING_RATE),
					metrics=['accuracy'])

		# Fine-tune model
		finetune_model = model.fit(train_data.repeat(),
			steps_per_epoch=steps_per_epoch,
			epochs=total_epochs,
			initial_epoch=toplayer_epochs,
			validation_data=test_data.repeat(),
			validation_steps=val_steps,
			callbacks=callbacks)

		model.save(os.path.join(self.DATA_DIR, "trained_model.h5"))
		return finetune_model.history['val_accuracy']

if __name__ == "__main__":
	#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	#tf.logging.set_verbosity(tf.logging.ERROR)

	parser = argparse.ArgumentParser(description = "Train a CNN using an Inception-v4 network")
	parser.add_argument('-d', '--dir', help='Path to root directory for saving model.')
	parser.add_argument('-i', '--input', help='Path to root directory with training and eval data.')
	parser.add_argument('-r', '--retrain', help='Path to directory containing model to use as pretraining')
	parser.add_argument('-a', '--annotation', help='Path to root directory with training and eval data.')
	args = parser.parse_args()

	#SFM = SlideflowModel(args.dir, args.input, args.annotation)
	#model_config = SFModelConfig(args.size, args.classes, args.batch, augment=True, use_fp16=args.use_fp16)
	#SFM.config(model_config)
	#SFM.train(restore_checkpoint = args.retrain)
