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

from datetime import datetime
import time

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.python import debug as tf_debug

import histcon
import inception_v4
from inception_utils import inception_arg_scope

def train():
	'''Train HISTCON for a number of steps, according to flags set by the argument parser.'''
	with tf.Graph().as_default():
		global_step = tf.train.get_or_create_global_step()

		# Get images and labels.
		# Force input pipeline to CPU:0 to avoid operations ending up on GPU.
		with tf.device('/cpu'):
			next_batch_images, next_batch_labels = histcon.inputs()

		# Build a Graph that computes the logits predictions from
		# the inference model.
		
		with arg_scope(inception_arg_scope()):
			logits, end_points = inception_v4.inception_v4(next_batch_images, num_classes=histcon.NUM_CLASSES)
		
		# Calculate training loss.
		loss = histcon.loss(logits, next_batch_labels)
		
		# Create summary op for validation
		losses = tf.get_collection('losses')
		for l in losses + [loss]
			summary_op = tf.summary.scalar(l.op.name + ' (raw)', l)

		# Build a Graph that trains the model with one batch of
		# examples and updates the model parameters.
		train_op = histcon.train(loss, global_step)

		# Visualize CNN activations
		#tf_cnnvis.deconv_visualization(tf.get_default_graph(), None, input_tensor = images)

		class _LoggerHook(tf.train.SessionRunHook):
			'''Logs loss and runtime.'''
			
			def validation(self):
				# Run through validation dataset
				vlosses = []
				while True:
					try:
						#vlosses.append(sess.run([loss], feed_dict={...})
						pass
					except tf.OutOfRangeError:
						break
				average_loss = np.mean(vlosses)
				summary_op = tf.summary.scalar(loss.name + ' (raw)', loss)
				
					
			def begin(self):
				self._step = -1
				self._start_time = time.time()

			def before_run(self, run_context):
				self._step += 1
				return tf.train.SessionRunArgs(loss) # Asks for loss value.

			def after_run(self, run_context, run_values):
				if self._step % histcon.LOG_FREQUENCY == 0:
					current_time = time.time()
					duration = current_time - self._start_time
					self._start_time = current_time

					loss_value = run_values.results
					images_per_sec = histcon.LOG_FREQUENCY * histcon.BATCH_SIZE / duration
					sec_per_batch = float(duration / histcon.LOG_FREQUENCY)

					format_str = ('%s: step %d, loss = %.2f (%.1f images/sec; %.3f sec/batch)')
					print(format_str % (datetime.now(), self._step, loss_value,
										images_per_sec, sec_per_batch))

		with tf.train.MonitoredTrainingSession(
			checkpoint_dir = histcon.MODEL_DIR,
			hooks = [tf.train.NanTensorHook(loss),
					_LoggerHook()],
			config = tf.ConfigProto(
					log_device_placement=False),
			save_summaries_steps = histcon.SUMMARY_STEPS) as mon_sess:

			while not mon_sess.should_stop():
				mon_sess.run(train_op)

def main(argv=None):
	'''Initialize directories and start the main Tensorflow app.'''
	if tf.gfile.Exists(histcon.MODEL_DIR):
		tf.gfile.DeleteRecursively(histcon.MODEL_DIR)
	tf.gfile.MakeDirs(histcon.MODEL_DIR)
	train()

if __name__ == "__main__":
	tf.app.run()