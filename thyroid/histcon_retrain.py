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
from tensorflow.python.framework import tensor_util
from tensorflow.contrib.framework import arg_scope
import tensorflow as tf

import histcon
#import tf_cnnvis
import inception_v4
from inception_utils import inception_arg_scope

import sys

parser = histcon.parser

RETRAIN_MODEL = '/home/shawarma/thyroid/models/inception_v4_2018_04_27/inception_v4.pb'
DTYPE = tf.float16 if histcon.FLAGS.use_fp16 else tf.float32

def retrain():
	# Do not import the final layer of the saved network, as we will be working with 
	#  different output classes
	variables_to_ignore = ("InceptionV4/Logits/Logits/weights:0", "InceptionV4/Logits/Logits/biases:0")

	with tf.Graph().as_default():
		# Import the graph from the saved *.rb file
		with gfile.GFile(RETRAIN_MODEL, 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def, name='')

		input_placeholder = tf.placeholder(DTYPE, shape=[None, histcon.IMAGE_SIZE, histcon.IMAGE_SIZE, 3])

		imported_variable_dict = {}

		# Create a second, trainable inception_v4 network
		with arg_scope(inception_arg_scope()):
			with tf.variable_scope('NewModel'):
				inception_v4.inception_v4(input_placeholder, num_classes=histcon.NUM_CLASSES, create_aux_logits=False)		

			# Start a session to extract the imported tensor values
			with tf.Session() as sess:
				# First, get a list of trainable variables from our newly created inception_v4 network
				for trainable_var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='NewModel'):
					variable_name = trainable_var.name[9:]
					if variable_name in variables_to_ignore:
						continue
					# For each trainable variable, put the saved value from our imported graph
					#  into a python dictionary for later use
					imported_value = tf.get_default_graph().get_tensor_by_name(variable_name).eval(session=sess)
					imported_variable_dict[variable_name] = imported_value

	# Reset the graph to clear all tensors, including those from our imported graph
	tf.reset_default_graph()

	# Start a new graph and build a new inception_v4 network
	with tf.Graph().as_default():
		global_step = tf.train.get_or_create_global_step()

		with tf.device('/cpu'):
			images, labels = histcon.processed_inputs()

		with arg_scope(inception_arg_scope()):
			logits, end_points = inception_v4.inception_v4(images, num_classes=histcon.NUM_CLASSES, create_aux_logits=True)

			assign_ops = []

			# For each trainable variable, create an op that will assign to it the value we saved
			for trainable_var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
				var_name = trainable_var.name
				if variable_name in variables_to_ignore:
					continue
				assign_op = trainable_var.assign(imported_variable_dict[var_name])
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
