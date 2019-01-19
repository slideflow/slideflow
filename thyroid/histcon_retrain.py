# Copyright (C) James Dolezal - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, October 2017
# ==========================================================================

"""Builds the HISTCON network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

from tensorflow.python.platform import gfile
from tensorflow.contrib.framework import arg_scope
from tensorflow.python import debug as tf_debug
import tensorflow as tf

import histcon
import pickle
#import tf_cnnvis
import inception_v4
from inception_utils import inception_arg_scope

import sys

parser = histcon.parser

RETRAIN_MODEL = '/home/shawarma/thyroid/models/inception_v4_2018_04_27/inception_v4.pb'
MODEL_VALUES_FILE = '/home/shawarma/thyroid/thyroid/obj/inception_v4_imagenet_pretrained.pkl'
DTYPE = tf.float16 if histcon.FLAGS.use_fp16 else tf.float32

def retrain():
	# Do not import the final layer of the saved network, as we will be working with 
	#  different output classes

	with open(MODEL_VALUES_FILE, 'rb') as f:
		var_dict = pickle.load(f)

	# Start a new graph and build a new inception_v4 network
	with tf.Graph().as_default():
		global_step = tf.train.get_or_create_global_step()

		with tf.device('/cpu'):
			images, labels = histcon.processed_inputs()

		with arg_scope(inception_arg_scope()):
			logits, end_points = inception_v4.inception_v4(images, num_classes=histcon.NUM_CLASSES)

			assign_ops = []

			# For each trainable variable, create an op that will assign to it the value we saved
			for trainable_var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
				var_name = trainable_var.name
				if var_name in var_dict:
					assign_op = trainable_var.assign(var_dict[var_name])
					assign_ops.append(assign_op)

		# Run the ops which will execute the variable data transfer
		with tf.Session() as sess:
			sess.run(assign_ops)

		loss = histcon.loss(logits, labels)
		train_op = histcon.train(loss, global_step)		

		class _LoggerHook(tf.train.SessionRunHook):
			'''Logs loss and runtime.'''

			def begin(self):
				self._step = -1
				self._start_time = time.time()

			def before_run(self, run_context):
				self._step += 1
				return tf.train.SessionRunArgs(loss) # Asks for loss value.

			def after_run(self, run_context, run_values):
				if self._step % FLAGS.log_frequency == 0:
					current_time = time.time()
					duration = current_time - self._start_time
					self._start_time = current_time

					loss_value = run_values.results
					images_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
					sec_per_batch = float(duration / FLAGS.log_frequency)

					format_str = ('%s: step %d, loss = %.2f (%.1f images/sec; %.3f sec/batch)')
					print(format_str % (datetime.now(), self._step, loss_value,
										images_per_sec, sec_per_batch))

		with tf.train.MonitoredTrainingSession(
			checkpoint_dir = FLAGS.model_dir,
			hooks = [tf.train.StopAtStepHook(last_step = FLAGS.max_steps),
					tf.train.NanTensorHook(loss),
					_LoggerHook()],
			config = tf.ConfigProto(
					log_device_placement=False),
			save_summaries_steps = FLAGS.summary_steps) as mon_sess:

			while not mon_sess.should_stop():
				mon_sess.run(train_op)

def main(argv=None):
	if tf.gfile.Exists(FLAGS.model_dir):
		tf.gfile.DeleteRecursively(FLAGS.model_dir)
	tf.gfile.MakeDirs(FLAGS.model_dir)
	retrain()

if __name__ == "__main__":
	FLAGS = parser.parse_args()
	tf.app.run()
