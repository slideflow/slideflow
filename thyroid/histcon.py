# Copyright (C) James Dolezal - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, October 2017
# ==========================================================================

# Update 3/2/2019: Beginning tf.data implementation

# In the process of merging histcon & histcon_input

''''Builds the HISTCON network.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import os, sys

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

import numpy as np

import histcon
import inception_v4
from inception_utils import inception_arg_scope
from six.moves import urllib, xrange

class HistconModel:
	''' Model containing all functions necessary to build input dataset pipelines,
	build a training and validation set model, and monitor and execute training.'''

	# Global constants describing the HISTCON data set.

	# Process images of the below size. If this number is altered, the
	# model architecture will change and will need to be retrained.

	IMAGE_SIZE = 128
	NUM_CLASSES = 5

	NUM_EXAMPLES_PER_EPOCH = 1024

	# Constants for the training process.
	MOVING_AVERAGE_DECAY = 0.9999 		# Decay to use for the moving average.
	NUM_EPOCHS_PER_DECAY = 240.0		# Epochs after which learning rate decays.
	LEARNING_RATE_DECAY_FACTOR = 0.05	# Learning rate decay factor.
	INITIAL_LEARNING_RATE = 0.001			# Initial learning rate.

	# Variables previous created with parser & FLAGS
	BATCH_SIZE = 2
	WHOLE_IMAGE = '' # Filename of whole image (JPG) to evaluate with saved model
	MAX_EPOCH = 30000
	LOG_FREQUENCY = 1 # How often to log results to console, in steps
	TEST_FREQUENCY = 10 # How often to run validation testing, in steps
	SUMMARY_STEPS = 5 # How often to save summaries for Tensorboard display, in steps
	EVAL_INTERVAL_SECS = 300 # How often to run eval/validation
	NUM_EXAMPLES = 10000 # Number of examples to run?
	USE_FP16 = False

	''' ANSWER: 
	https://stackoverflow.com/questions/46111072/how-to-use-feedable-iterator-from-tensorflow-dataset-api-along-with-monitoredtra
	https://github.com/jke-zq/tensorflow120/blob/master/dataset_switch.py
	https://github.com/tensorflow/tensorflow/issues/14613
	I don't think there was any change that would cause this error. From the error message though it looks like the handle might have been captured (using Iterator.to_string_handle()) in a different session from which it is being used. Since you're using tf.train.MonitoredTrainingSession, is it possible that the session has been recreated? If that's the case, you'll need to make sure to recompute the handle—and potentially reinitialize the iterator—in the new session.

	A possible workaround, which might or might not work depending on the actual cause of the problem, would be to set a shared_name when you create the iterator. I believe that should make the handle string stable across restarts of the session (as long as it's not in a task that could have failed).
	'''

	def __init__(self, data_directory):
		self.DATA_DIR = data_directory
		self.MODEL_DIR = os.path.join(self.DATA_DIR, 'models/active') # Directory where to write event logs and checkpoints.
		self.EVAL_DIR = os.path.join(self.DATA_DIR, 'models/eval') # Directory where to write eval logs and summaries.
		self.CONV_DIR = os.path.join(self.DATA_DIR, 'models/conv') # Directory where to write logs and summaries for the convoluter.
		self.TRAIN_FILES = os.path.join(self.DATA_DIR, "train_data/*/*/*.jpg")
		self.TEST_FILES = os.path.join(self.DATA_DIR, "train_data/*/*/*.jpg")

		if tf.gfile.Exists(self.MODEL_DIR):
			tf.gfile.DeleteRecursively(self.MODEL_DIR)
		tf.gfile.MakeDirs(self.MODEL_DIR)

	def _parse_function(self, filename, label):
		'''Loads image file data into Tensor.

		Args:
			filename: 	a string containing directory/filename of .jpg file
			label: 		accompanying image label

		Returns:
			image: a Tensor of shape [size, size, 3] containing image data
			label: accompanying label
		'''
		image_string = tf.read_file(filename)
		image = tf.image.decode_jpeg(image_string, channels = 3)
		image = tf.image.per_image_standardization(image)

		dtype = tf.float16 if self.USE_FP16 else tf.float32
		image = tf.image.convert_image_dtype(image, dtype)
		image = tf.image.resize_images(image, [128,128])
		image.set_shape([self.IMAGE_SIZE, self.IMAGE_SIZE, 3])

		# Optional image resizing
		
		return image, label

	def _train_preprocess(self, image, label):
		'''Performs image pre-processing, including flipping, and changes to brightness and saturation.

		Args:
			image: a Tensor of shape [size, size, 3]
			label: accompanying label

		Returns:
			image: a Tensor of shape [size, size, 3]
			label: accompanying label
		'''

		# Optional pre-processing
		#image = tf.image.random_flip_left_right(image)
		#image = tf.image.random_brightness(image, max_delta = 32.0 / 255.0)
		#image = tf.image.random_saturation(image, lower=0.5, upper = 1.5)
		#image = tf.clip_by_value(image, 0.0, 1.0)

		return image, label

	def gen_filenames_op(self, dir_string):
		filenames_op = tf.train.match_filenames_once(dir_string)
		labels_op = tf.map_fn(lambda f: tf.string_to_number(tf.string_split([f], '/').values[tf.constant(-3, dtype=tf.int32)],
													out_type=tf.int32), filenames_op, dtype=tf.int32)
		return filenames_op, labels_op

	def generate_batched_dataset(self, filenames, labels):
		dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
		dataset = dataset.shuffle(tf.size(filenames, out_type=tf.int64))
		dataset = dataset.map(self._parse_function, num_parallel_calls = 8)
		dataset = dataset.map(self._train_preprocess, num_parallel_calls = 8)
		dataset = dataset.batch(self.BATCH_SIZE)
		return dataset

	def build_inputs(self):
		'''Construct input for HISTCON evaluation.

		Args:
			sess: active tensorflow session

		Returns:
			next_batch_images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
			next_batch_labels: Labels. 1D tensor of [batch_size] size.
		'''
	
		with tf.name_scope('filename_input'):
			train_filenames_op, train_labels_op = self.gen_filenames_op(self.TRAIN_FILES)
			test_filenames_op, test_labels_op = self.gen_filenames_op(self.TEST_FILES)

		with tf.name_scope('input'):
			train_dataset = self.generate_batched_dataset(train_filenames_op, train_labels_op)
			train_dataset = train_dataset.repeat(self.MAX_EPOCH)
			train_dataset = train_dataset.prefetch(1)

			test_dataset = self.generate_batched_dataset(test_filenames_op, test_labels_op)
			test_dataset = test_dataset.prefetch(1)

			with tf.name_scope('iterator'):
				train_iterator = train_dataset.make_initializable_iterator()

				# Will likely need to be re-initializable iterator to repeat testing
				test_iterator = test_dataset.make_initializable_iterator()

				train_iterator_handle = train_iterator.string_handle()
				test_iterator_handle = test_iterator.string_handle()

				handle = tf.placeholder(tf.string, shape=[])
				iterator = tf.data.Iterator.from_string_handle(handle, 
															   train_iterator.output_types,
															   train_iterator.output_shapes)

			next_batch_images, next_batch_labels = iterator.get_next()

			if self.USE_FP16: next_batch_images = tf.cast(next_batch_images, dtype=tf.float16)														   
	
			handles = {'iterator':handle, 'train': train_iterator_handle, 'test':test_iterator_handle,
						'train_init':train_iterator.initializer, 'test_init':test_iterator.initializer}

		return next_batch_images, next_batch_labels, handles

	def loss(self, logits, labels):
		'''Add L2Loss to all trainable variables, and a summary for "Loss" and "Loss/avg".

		Args:
			logits: Logits from inference().
			labels: Labels from distorted_inputs() or inputs(). 1D tensor of shape [batch_size]

		Returns:
			Loss Tensor of type float.
		'''
		# Calculate average cross entropy loss across the batch.
		labels = tf.cast(labels, tf.int64)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=labels, logits=logits, name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy_mean)

		# Total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss)
		return tf.add_n(tf.get_collection('losses'), name='total_loss')

	def _add_loss_summaries(self, total_loss):
		'''Add summaries for losses in HISTCON model.

		Generates moving average for all losses and associated sumaries for
		visualizing network performance with Tensorboard.

		Args:
			total_loss: Total loss from loss().

		Returns:
			loss_averages_op: op fro generating moving averages of losses.
		'''
		# Compute moving average of all individual losses and the total loss.
		loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
		losses = tf.get_collection('losses')
		loss_averages_op = loss_averages.apply(losses + [total_loss])

		# Attach a scalar summary to all individual losses and the total loss; do the
		# same for the averaged version of the losses.
		for l in losses + [total_loss]:
			# Name each loss as '(raw)' and name the moving average version of the loss as
			# the original loss name.
			tf.summary.scalar(l.op.name + ' (raw)', l)
			tf.summary.scalar(l.op.name, loss_averages.average(l))

		return loss_averages_op

	def build_train_op(self, total_loss, global_step):
		'''Train the HISTCON model.

		Create an optimizer and apply to all trainable variables. Add moving
		average for all trainable variables.

		Args:
			total_loss: Total loss from loss().
			global_step: Integer variable counting the number of training steps processed.

		Returns:
			train_op: op for training.
		'''

		# Variables that affect learning rate.
		num_batches_per_epoch = self.NUM_EXAMPLES_PER_EPOCH / self.BATCH_SIZE
		decay_steps = int(num_batches_per_epoch * self.NUM_EPOCHS_PER_DECAY)

		# Decay the learning rate exponentially based on the number of steps.
		lr = tf.train.exponential_decay(self.INITIAL_LEARNING_RATE,
										global_step,
										decay_steps,
										self.LEARNING_RATE_DECAY_FACTOR,
										staircase=True)
		tf.summary.scalar('learning_rate', lr)

		# Generate moving averages of all losses and associated summaries.
		loss_averages_op = self._add_loss_summaries(total_loss)

		# Compute gradients.
		with tf.control_dependencies([loss_averages_op]):
			opt = tf.train.GradientDescentOptimizer(lr)
			grads = opt.compute_gradients(total_loss)

		# Apply gradients.
		apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

		# Add histograms for trainable variables.
		for var in tf.trainable_variables():
			tf.summary.histogram(var.op.name, var)

		# Add histograms for gradients.
		for grad, var in grads:
			if grad is not None:
				tf.summary.histogram(var.op.name + '/gradients', grad)

		# Track moving averages of all trainable variables.
		variable_averages = tf.train.ExponentialMovingAverage(
			self.MOVING_AVERAGE_DECAY, global_step)
		variables_averages_op = variable_averages.apply(tf.trainable_variables())

		with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
			train_op = tf.no_op(name='train')

		return train_op

	def train(self):
		'''Train HISTCON for a number of steps, according to flags set by the argument parser.'''
		
		global_step = tf.train.get_or_create_global_step()

		# Force input pipeline to CPU:0 to avoid operations ending up on GPU.
		with tf.device('/cpu'):
			next_batch_images, next_batch_labels, handles = self.build_inputs()

		train_iterator_str = handles['train']
		test_iterator_str = handles['test']
		iterator = handles['iterator']
		train_initializer = handles['train_init'] 
		test_initializer = handles['test_init']

		with arg_scope(inception_arg_scope()):
			logits, end_points = inception_v4.inception_v4(next_batch_images, num_classes=self.NUM_CLASSES)
		
		# Calculate training loss.
		loss = self.loss(logits, next_batch_labels)
		
		# Create summary op for validation
		'''losses = tf.get_collection('losses')
		for l in losses + [loss]:
			summary_op = tf.summary.scalar(l.op.name + ' (raw)', l)'''

		# Define summary writer for saving validation logs
		#  (training logs handled by MonitoredTrainingSession)
		test_writer = tf.summary.FileWriter("./logs/validation", graph=tf.get_default_graph())

		train_op = self.build_train_op(loss, global_step)

		# Create an averaging op to follow validation accuracy
		with tf.name_scope('mean_validation_loss'):
			validation_accuracy, validation_accuracy_update = tf.metrics.mean(loss)

		stream_vars = [v for v in tf.local_variables() if v.name.startswith('mean_validation_loss')]
		stream_vars_reset = [v.initializer for v in stream_vars]
		
		validation_accuracy_scalar = tf.summary.scalar(loss.name+ ' (raw)', validation_accuracy)

		# Op to reset validation mean (for use between epochs)
		#reset_validation_average = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

		init = (tf.global_variables_initializer(), tf.local_variables_initializer())

		# Visualize CNN activations
		#tf_cnnvis.deconv_visualization(tf.get_default_graph(), None, input_tensor = images)

		class _LoggerHook(tf.train.SessionRunHook):
			'''Logs loss and runtime.'''
			def __init__(self, train_str, test_str, parent):
				self.parent = parent
				self.train_str = train_str
				self.test_str = test_str
				self.train_handle = None
				self.test_handle = None

			def after_create_session(self, session, coord):
				del coord
				print ('doing string-handle work...')
				if self.train_str is not None:
					self.train_iterator_handle, self.test_iterator_handle = session.run([self.train_str, self.test_str])
					session.run([train_initializer, test_initializer])
					session.run(init)

				print ('String handle work done')
					
			def begin(self):
				self._step = -1
				self._start_time = time.time()

			def before_run(self, run_context):
				feed_dict = run_context.original_args.feed_dict
				if feed_dict and feed_dict[iterator] == self.train_iterator_handle:
					self._step += 1
					return tf.train.SessionRunArgs(loss) # Asks for loss value.

			def after_run(self, run_context, run_values):
				if (self._step % self.parent.LOG_FREQUENCY == 0) and (run_context.original_args.feed_dict[iterator] == self.train_iterator_handle):
					current_time = time.time()
					duration = current_time - self._start_time
					self._start_time = current_time

					loss_value = run_values.results
					images_per_sec = self.parent.LOG_FREQUENCY * self.parent.BATCH_SIZE / duration
					sec_per_batch = float(duration / self.parent.LOG_FREQUENCY)

					format_str = ('%s: step %d, loss = %.2f (%.1f images/sec; %.3f sec/batch)')
					print(format_str % (datetime.now(), self._step, loss_value,
										images_per_sec, sec_per_batch))

		loggerhook = _LoggerHook(train_iterator_str, test_iterator_str, self)

		with tf.train.MonitoredTrainingSession(
			checkpoint_dir = self.MODEL_DIR,
			hooks = [loggerhook],#tf.train.NanTensorHook(loss),
			config = tf.ConfigProto(
					log_device_placement=False),
			save_summaries_steps = self.SUMMARY_STEPS) as mon_sess:
			
			while not mon_sess.should_stop():
				_, step = mon_sess.run([train_op, global_step], feed_dict={iterator:loggerhook.train_iterator_handle})

				if (step % self.TEST_FREQUENCY == 0):
					# Validation testing
					print("Validation testing...")

					# Reset validation testing average
					mon_sess.run(stream_vars_reset, feed_dict={iterator:loggerhook.test_iterator_handle})#, feed_dict={iterator:loggerhook.test_iterator_handle})
					print("Finished average reset.")
					while True:
						try:
							_, val_acc_sum, val_acc = mon_sess.run([validation_accuracy_update, validation_accuracy_scalar, validation_accuracy], feed_dict={iterator:loggerhook.test_iterator_handle})
						except tf.errors.OutOfRangeError:
							break
					#summary = tf.summary.scalar(loss.name+ ' (raw)', average_loss)
					print("Validation testing almost done.")
					test_writer.add_summary(val_acc_sum, step)
					print("Validation testing finished: {}".format(val_acc))
					mon_sess.run(test_initializer, feed_dict={iterator:loggerhook.test_iterator_handle})

def main(argv=None):
	'''Initialize directories and start the main Tensorflow app.'''
	histcon = HistconModel('/Users/james/histcon')
	histcon.train()

if __name__ == "__main__":
	tf.app.run()
