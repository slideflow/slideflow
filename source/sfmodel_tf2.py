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
'''from tensorflow.contrib.framework import arg_scope
from tensorflow.summary import FileWriterCache
import tensorflow.contrib.lookup'''
from tensorboard.plugins.custom_scalar import layout_pb2

#import inception_v4
#from inception_utils import inception_arg_scope
from glob import glob
from scipy.stats import linregress

from util import tfrecords, sfutil
from util.sfutil import TCGAAnnotations

#slim = tf.contrib.slim

#RUN_OPTS = tf.RunOptions(report_tensor_allocations_upon_oom = True)

# Calculate accuracy with https://stackoverflow.com/questions/50111438/tensorflow-validate-accuracy-with-batch-data
# TODO: try next, comment out line 254 (results in calculating total_loss before update_ops is called)
# TODO: visualize graph, memory usage, and compute time with https://www.tensorflow.org/guide/graph_viz
# TODO: export logs to file for monitoring remotely

class SFModelConfig:
	def __init__(self, image_size, num_classes, batch_size, augment=False, learning_rate=0.01, 
				beta1=0.9, beta2=0.999, epsilon=1.0, batch_norm_decay=0.99, early_stop=0.015, 
				max_epoch=300, log_frequency=20, test_frequency=600, use_fp16=True):
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

	def __init__(self, data_directory, input_directory, annotations_file):
		self.DATA_DIR = data_directory
		self.INPUT_DIR = input_directory
		self.MODEL_DIR = self.DATA_DIR # Directory where to write event logs and checkpoints.
		self.TRAIN_DIR = os.path.join(self.MODEL_DIR, 'train') # Directory where to write eval logs and summaries.
		self.TEST_DIR = os.path.join(self.MODEL_DIR, 'test') # Directory where to write eval logs and summaries.
		self.TRAIN_FILES = os.path.join(self.INPUT_DIR, "train_data/*/*.jpg")
		self.TEST_FILES = os.path.join(self.INPUT_DIR, "eval_data/*/*.jpg")
		self.TRAIN_TFRECORD = os.path.join(self.INPUT_DIR, "train.tfrecords")
		self.EVAL_TFRECORD = os.path.join(self.INPUT_DIR, "eval.tfrecords")
		self.USE_TFRECORD = (os.path.exists(self.TRAIN_TFRECORD) and os.path.exists(self.EVAL_TFRECORD))

		annotations = sfutil.get_annotations_dict(annotations_file, key_name="slide", value_name="category")
		# TODO: use verification done by parent slideflow module; if not done, offer to use again
		#tfrecord_files = [self.TRAIN_TFRECORD, self.EVAL_TFRECORD] if self.USE_TFRECORD else []
		#sfutil.verify_tiles(annotations, self.INPUT_DIR, tfrecord_files)

		# Reset default graph
		#tf.reset_default_graph()

		with tf.device('/cpu'):
			#with tf.variable_scope("annotations"):
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

	def _gen_filenames_op(self, dir_string):
		filenames_op = tf.train.match_filenames_once(dir_string)
		labels_op = tf.map_fn(lambda f: self.ANNOTATIONS_TABLE.lookup(tf.string_split([f], '/').values[tf.constant(-2, dtype=tf.int32)]),
								filenames_op, dtype=tf.int32)
		return filenames_op, labels_op

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

	def _parse_function(self, filename, label):
		image_string = tf.read_file(filename)
		image = self._process_image(image_string)
		return image, label

	def _parse_tfrecord_function(self, tfrecord_features):
		case = tfrecord_features['case']
		label = self.ANNOTATIONS_TABLE.lookup(case)
		image_string = tfrecord_features['image_raw']
		image = self._process_image(image_string)
		return image, label

	def _gen_batched_dataset(self, filenames, labels):
		# Replace the below dataset with one that uses a Python generator for flexibility of labeling
		dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
		dataset = dataset.shuffle(tf.size(filenames, out_type=tf.int64))
		dataset = dataset.map(self._parse_function, num_parallel_calls = 8)
		dataset = dataset.batch(self.BATCH_SIZE)
		return dataset

	def _gen_batched_dataset_from_tfrecord(self, tfrecord):
		raw_image_dataset = tf.data.TFRecordDataset(tfrecord)
		feature_description = tfrecords.FEATURE_DESCRIPTION

		def _parse_image_function(example_proto):
			"""Parses the input tf.Example proto using the above feature dictionary."""
			return tf.io.parse_single_example(example_proto, feature_description)

		dataset = raw_image_dataset.map(_parse_image_function)
		dataset = dataset.shuffle(100000)
		dataset = dataset.map(self._parse_tfrecord_function, num_parallel_calls = 8)
		dataset = dataset.batch(self.BATCH_SIZE)
		return dataset

	def build_inputs(self):
		'''Construct input for the model.'''

		if not self.USE_TFRECORD:
			with tf.name_scope('filename_input'):
				train_filenames_op, train_labels_op = self._gen_filenames_op(self.TRAIN_FILES)
				test_filenames_op, test_labels_op = self._gen_filenames_op(self.TEST_FILES)
			train_dataset = self._gen_batched_dataset(train_filenames_op, train_labels_op)
			test_dataset = self._gen_batched_dataset(test_filenames_op, test_labels_op)
		else:
			with tf.name_scope('input'):
				train_dataset = self._gen_batched_dataset_from_tfrecord(self.TRAIN_TFRECORD)
				test_dataset = self._gen_batched_dataset_from_tfrecord(self.EVAL_TFRECORD)
		
		return train_dataset, test_dataset

	def train(self):
		'''Train the model for a number of steps, according to flags set by the argument parser.'''
		
		train_data, test_data = self.build_inputs()

		# Create callback for checkpoint saving
		checkpoint_path = os.path.join(self.MODEL_DIR, "cp.ckpt")
		cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
														 save_weights_only=True,
														 verbose=1)

		# Callbacks for summary writing
		logdir = self.DATA_DIR
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, 
															  histogram_freq=0,
															  write_graph=False,
															  update_freq=self.BATCH_SIZE*self.LOG_FREQUENCY)

		# Get pretrained model
		base_model = tf.keras.applications.InceptionV3(
			input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3),
			include_top=False,
			pooling='max',
			weights='imagenet'
		)
		# Freeze pretrained weights
		base_model.trainable = False
		
		# Create a trainable classification head / final layer, then link with base
		fully_connected_layer = tf.keras.layers.Dense(1536, activation='relu')
		prediction_layer = tf.keras.layers.Dense(self.NUM_CLASSES, activation='softmax')
		model = tf.keras.Sequential([
			base_model,
			fully_connected_layer,
			prediction_layer
		])

		# Compile the model
		lr_fast = self.LEARNING_RATE * 10
		model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr_fast),
					  loss='sparse_categorical_crossentropy',
					  metrics=['accuracy']
		)

		# Train final layer of the model
		num_epochs=0
		steps_per_epoch=round(83000/self.BATCH_SIZE)
		val_steps=20
		model.fit(train_data.repeat(),
				  epochs=num_epochs,
				  steps_per_epoch=steps_per_epoch,
				  validation_data=test_data.repeat(),
				  validation_steps=val_steps,
				  callbacks=[cp_callback, tensorboard_callback])

		# Now, fine-tune the model
		# Unfreeze all layers
		print(f" + [{sfutil.info('INFO')}] Beginning fine-tuning")
		base_model.trainable = True

		'''# Refreeze layers until the layers we want to fine-tune ???
		for layer in base_model.layers[:100]:
			layer.trainable=False'''

		# Recompile the model
		lr_finetune = self.LEARNING_RATE
		model.compile(loss='sparse_categorical_crossentropy',
					  optimizer=tf.keras.optimizers.Adam(lr=lr_finetune),
					  metrics=['accuracy'])

		# Increase training epochs for fine-tuning
		fine_tune_epochs = 30
		total_epochs = num_epochs + fine_tune_epochs

		# Fine-tune model
		# Note: will need to set initial_epoch to begin training after epoch 30
		# Since we just trained for 30 epochs
		model.fit(train_data.repeat(),
			steps_per_epoch=steps_per_epoch,
			epochs=total_epochs,
			initial_epoch=num_epochs,
			validation_data=test_data.repeat(),
			validation_steps=val_steps,
			callbacks=[cp_callback, tensorboard_callback])

		model.save(os.path.join(self.DATA_DIR, "trained_model.h5"))

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