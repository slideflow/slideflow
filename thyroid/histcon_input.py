# Copyright (C) James Dolezal - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, October 2017
# ==========================================================================

"""Builds the HISTCON network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import urllib, xrange
import tensorflow as tf

# Process images of the below size. If this number is altered, the
# model architecture will change and will need to be retrained.

IMAGE_SIZE = 512

# Global constants describing the histopathologic annotations.
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 100000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1024

def _generate_image_and_label_batch(image, label, min_queue_images,
									batch_size, shuffle):
	'''Construct a queued batch of images and labels.

	Args:
		image: 3D Tensor of [height, width, 3] of type.float32.
		label: 1D Tensor of type.int32
		min_queue_images: int32, minimum number of samples to retain
			in the queue that provides batches of images.
		batch_size: Number of images per batch.
		shuffle: bool, whether to use a shuffling queue.

	Return:
		images: Images. 4D Tensor of [batch_size, height, width, 3] size.
		labels: Labels. 1D Tensor of [batch_size] size.
	'''
	# Create a queue that shuffles images, and then
	# read "batch_size" number of images + labels from the queue.
	num_preprocess_threads = 8
	if shuffle:
		images, label_batch = tf.train.shuffle_batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_images + 3 * batch_size,
			min_after_dequeue=min_queue_images)
	else:
		images, label_batch = tf.train.batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_images + 3 * batch_size)

	# Display the training images in Tensorboard.
	tf.summary.image('images', images, max_outputs = 4)

	return images, tf.reshape(label_batch, [batch_size])

def processed_inputs(data_dir, batch_size, eval_data):
	'''Applies some sort of pre-processing, corruption, or distortion to images for training.

	Currently, this feature has not been implemented.
	'''
	return inputs(data_dir, batch_size, eval_data)

def inputs(data_dir, batch_size, eval_data):
	'''Construct input for HISTCON evaluation.

	Args:
		eval_data: bool indicating if one should use the training or eval data set.
		data_dir: Path to the data directory.
		batch_size: Number of images per batch.

	Returns:
		images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
		labels: Labels. 1D tensor of [batch_size] size.
	'''
	if not eval_data:
		filenames = os.path.join(data_dir, "train_data/*/*.jpg")
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
	else:
		filenames = os.path.join(data_dir, "eval_data/*/*.jpg")
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

	with tf.name_scope('input'):

		with tf.name_scope('queue'):
			# Create a queue that produces filenames to read
			filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(filenames))

		with tf.name_scope('image_reader'):
			# Create Tensor that reads an image and label from the filename queue
			image_reader = tf.WholeFileReader()
			key, image_file = image_reader.read(filename_queue)
			S = tf.string_split([key],'/')
			label = tf.string_to_number(S.values[tf.constant(-2, dtype=tf.int32)],
										out_type=tf.int32)
			image = tf.image.decode_jpeg(image_file)

		with tf.name_scope('image_processing'):
			# Image processing
			# To input: resize image as appropriate (e.g. down-scaling if necessary)
			# Subtract off the mean and divide by the variance of the pixels.
			float_image = tf.image.per_image_standardization(image)

			float_image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

		with tf.name_scope('batching'):
			# Ensure that random shuffling has good mixing properties
			min_fraction_of_examples_in_queue = 0.01
			min_queue_examples = int(num_examples_per_epoch *
									min_fraction_of_examples_in_queue)

			# Generate a batch of images and labels by building a queue.
			return _generate_image_and_label_batch(float_image, label,
												min_queue_examples, batch_size,
												shuffle=False)
