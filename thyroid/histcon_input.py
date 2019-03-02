# Copyright (C) James Dolezal - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, October 2017
# ==========================================================================

# Update 3/2/2019: Beginning tf.data implementation

"""Builds the HISTCON network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import argparse

from six.moves import urllib, xrange
import tensorflow as tf

# Process images of the below size. If this number is altered, the
# model architecture will change and will need to be retrained.

parser = argparse.ArgumentParser()

# Model parameters.

parser.add_argument('--batch_size', type=int, default=2,
	help='Number of images to process in a batch.')

parser.add_argument('--data_dir', type=str, default='/Users/james/thyroid',
	help='Path to the HISTCON data directory.')

parser.add_argument('--use_fp16', type=bool, default=True,
	help='Train the model using fp16.')

parser.add_argument('--model_dir', type=str, default='/Users/james/thyroid/models/active',
	help='Directory where to write event logs and checkpoints.')

parser.add_argument('--eval_dir', type=str, default='/Users/james/thyroid/models/eval',
	help='Directory where to write eval logs and summaries.')

parser.add_argument('--conv_dir', type=str, default='/Users/james/thyroid/models/conv',
	help='Directory where to write logs and summaries for the convoluter.')

parser.add_argument('--max_epoch', type=int, default=30,
	help='Number of batches to run.')

parser.add_argument('--log_frequency', type=int, default=1,
	help='How often to log results to the console.')

parser.add_argument('--summary_steps', type=int, default=25,
	help='How often to save summaries for Tensorboard display, in steps.')

parser.add_argument('--eval_data', type=str, default='test',
	help='Either "test" or "train", indicating the type of data to use for evaluation.')

parser.add_argument('--eval_interval_secs', type=int, default=300,
	help='How often to run the eval.')

parser.add_argument('--num_examples', type=int, default=10000,
	help='Number of examples to run.')

parser.add_argument('--run_once', type=bool, default=False,
	help='True/False; whether to run eval only once.')

parser.add_argument('--whole_image', type=str,
	help='Filename of whole image (JPG) to evaluate with saved model.')

FLAGS = parser.parse_args()

IMAGE_SIZE = 512

# Global constants describing the histopathologic annotations.
NUM_CLASSES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1024
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1024

def _parse_function(filename, label):
	image_string = tf.read_file(filename)
	image = tf.image.decode_jpeg(image_string, channels = 3)
	image = tf.image.per_image_standardization(image)

	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	image = tf.image.convert_image_dtype(image, dtype)
	image = tf.image.resize_images(image, [128,128])
	image.set_shape([128, 128, 3])#[IMAGE_SIZE, IMAGE_SIZE, 3])

	# Optional image resizing
	
	return image, label

def _train_preprocess(image, label):
	# Optional pre-processing
	#image = tf.image.random_flip_left_right(image)
	#image = tf.image.random_brightness(image, max_delta = 32.0 / 255.0)
	#image = tf.image.random_saturation(image, lower=0.5, upper = 1.5)
	#image = tf.clip_by_value(image, 0.0, 1.0)

	return image, label

def inputs(data_dir, batch_size, eval_data):
	'''Construct input for HISTCON evaluation.

	Args:
		eval_data: bool indicating if one should use the training or eval data set.
		data_dir: Path to the data directory.
		batch_size: Number of images per batch.

	Returns:
		next_batch_images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
		next_batch_labels: Labels. 1D tensor of [batch_size] size.
	'''
	if not eval_data:
		files = os.path.join(data_dir, "train_data/*/*/*.jpg")
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
	else:
		files = os.path.join(data_dir, "eval_data/*/*/*.jpg")
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

	with tf.name_scope('filename_input'):
		with tf.Session() as sess:
			tf_filenames = tf.train.match_filenames_once(files)
			tf_labels = tf.map_fn(lambda f: tf.string_to_number(tf.string_split([f], '/').values[tf.constant(-3, dtype=tf.int32)],
															out_type=tf.int32), tf_filenames, dtype=tf.int32)

			init = (tf.global_variables_initializer(), tf.local_variables_initializer())
			sess.run(init)
			filenames, labels = sess.run([tf_filenames, tf_labels])

	with tf.name_scope('input'):

		dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
		dataset = dataset.shuffle(tf.size(filenames, out_type=tf.int64))
		dataset = dataset.map(_parse_function, num_parallel_calls = 8)
		dataset = dataset.map(_train_preprocess, num_parallel_calls = 8)
		dataset = dataset.batch(FLAGS.batch_size)
		dataset = dataset.repeat(FLAGS.max_epoch)
		dataset = dataset.prefetch(1)

		with tf.name_scope('iterator'):
			iterator = dataset.make_one_shot_iterator()
			next_batch_images, next_batch_labels = iterator.get_next()

	return next_batch_images, next_batch_labels
