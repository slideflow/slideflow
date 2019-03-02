# Copyright (C) James Dolezal - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, October 2017
# ==========================================================================

''''Builds the HISTCON network.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import histcon

def eval_once(saver, summary_writer, top_k_op, summary_op):
	'''Run eval once.

	Args:
		saver: Saver.
		summary_writer: Summary writer.
		top_k_op: Top K op.
		summary_op: Summary op.
	'''
	with tf.Session() as sess:
		# Initialize variables (not in example code)
		init = (tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init)

		# Restore checkpoint
		ckpt = tf.train.get_checkpoint_state(histcon.MODEL_DIR)
		if ckpt and ckpt.model_checkpoint_path:
			# Restores from checkpoint
			saver.restore(sess, ckpt.model_checkpoint_path)
			# Assuming model_checkpoint_path looks like:
			# 	/directory_path/model.ckpt-0,
			# extract the step from the filename.
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		else:
			print('No checkpoint file found.')
			return

		# Start the queue runners
		coord = tf.train.Coordinator()
		try:
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

			num_iter = int(math.ceil(histcon.NUM_EXAMPLES / histcon.BATCH_SIZE)) # Number of examples to test.
			true_count = 0 # Counts the number of correct predictions.
			total_sample_count = num_iter * histcon.BATCH_SIZE
			step = 0
			while step < num_iter and not coord.should_stop():
				predictions = sess.run([top_k_op])
				true_count += np.sum(predictions)
				step += 1

			# Compute precision @ 1.
			precision = true_count / total_sample_count
			print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

			summary = tf.Summary()
			summary.ParseFromString(sess.run(summary_op))
			summary.value.add(tag='Precision @ 1', simple_value=precision)
			summary_writer.add_summary(summary, global_step)
		except Exception as e:
			coord.request_stop(e)

		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=10)

def evaluate():
	'''Eval HISTCON for a number of steps.'''
	with tf.Graph().as_default() as g:
		# Get images and labels for HISTCON.
		eval_data = histcon.EVAL_DATA == 'test'
		images, labels = histcon.inputs(eval_data=eval_data)

		# Build a Graph that computes the logits predictions from the inference model.
		logits = histcon.inference(images)

		# Calculate predictions.
		top_k_op = tf.nn.in_top_k(logits, labels, 1)

		# Restore the moving average version of the learned variables for eval.
		variable_averages = tf.train.ExponentialMovingAverage(
			histcon.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		# Build the summary op based on the TF collection of Summaries.
		summary_op = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter(histcon.EVAL_DIR, g)

		while True:
			eval_once(saver, summary_writer, top_k_op, summary_op)
			time.sleep(histcon.EVAL_INTERVAL_SECS)

def main(argv=None):
	if tf.gfile.Exists(histcon.EVAL_DIR):
		tf.gfile.DeleteRecursively(histcon.EVAL_DIR)
	tf.gfile.MakeDirs(histcon.EVAL_DIR)
	evaluate()

if __name__ == '__main__':
	tf.app.run()


