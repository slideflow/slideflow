# Copyright (C) James Dolezal - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, March 2019
# ==========================================================================

'''Convolutionally applies a saved Tensorflow model to a larger image, displaying
the result as a heatmap overlay.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import warnings

import progress_bar

import tensorflow as tf
import numpy as np
import inception_v4
from tensorflow.contrib.framework import arg_scope
from inception_utils import inception_arg_scope
from PIL import Image

from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcol

# TODO: extract_image_patches expands data too unnecessarily, creating large memory pressure.
#  Need to find iterator method to only extract as many patches as are needed to generate a batch

class Convoluter:
	# Model variables
	SIZE = 512
	NUM_CLASSES = 5
	BATCH_SIZE = 512
	USE_FP16 = True

	# Display variables
	stride_divisor = 4
	STRIDES = [1, int(SIZE/stride_divisor), int(SIZE/stride_divisor), 1]
	WINDOW_SIZE = 6000

	WHOLE_IMAGE = '/home/shawarma/thyroid/images/WSI_25/234807-1_25.jpg'
	MODEL_DIR = '/home/shawarma/thyroid/models/active' # Directory where to write event logs and checkpoints.

	def __init__(self):
		self.DTYPE = tf.float16 if self.USE_FP16 else tf.float32
		self.DTYPE_INT = tf.int16 if self.USE_FP16 else tf.int32

	def scan_image(self):
		warnings.simplefilter('ignore', Image.DecompressionBombWarning)
		with tf.Graph().as_default() as g:

			image_string = tf.read_file(self.WHOLE_IMAGE)
			image = tf.cast(tf.image.decode_jpeg(image_string, channels = 3), tf.int32)

			window_size = [self.SIZE, self.SIZE]
			window_stride = [int(self.SIZE/4), int(self.SIZE/4)]

			with tf.Session() as sess:
				init = (tf.global_variables_initializer(), tf.local_variables_initializer())
				sess.run(init)

				shape = sess.run(tf.shape(image))

				print("Loading image of size {} x {}".format(shape[0], shape[1]))

				if (window_size[0] > shape[0]) or (window_size[1] > shape[1]):
					raise IndexError("Window size is too large")

				coord = []

				self.Y_SIZE = shape[0] - window_size[0]
				self.X_SIZE = shape[1] - window_size[1]

				for y in range(0, (shape[0]+1) - window_size[0], window_stride[0]):
					for x in range(0, (shape[1]+1) - window_size[1], window_stride[1]):
						coord.append([y, x])

				coord_dataset = tf.data.Dataset.from_tensor_slices(coord)
				coord_dataset = coord_dataset.map(lambda c: tf.cast(c, dtype=tf.int32))
				coord_dataset = coord_dataset.map(lambda c: image[c[0]:c[0]+window_size[0], c[1]:c[1]+window_size[1],], num_parallel_calls = 8)
				coord_dataset = coord_dataset.map(lambda patch: tf.cast(tf.image.per_image_standardization(patch), self.DTYPE), num_parallel_calls = 8)

				coord_dataset = coord_dataset.batch(self.BATCH_SIZE, drop_remainder = False)
				coord_dataset.prefetch(1)

				iterator = coord_dataset.make_one_shot_iterator()
				batch = iterator.get_next()

				# Pad the batch if necessary to create a batch of minimum size BATCH_SIZE
				padded_batch = tf.concat([batch, tf.zeros([self.BATCH_SIZE - tf.shape(batch)[0], 512, 512, 3], 
															dtype=self.DTYPE)], 0)

			padded_batch.set_shape([self.BATCH_SIZE, self.SIZE, self.SIZE, 3])

			with arg_scope(inception_arg_scope()):
				_, end_points = inception_v4.inception_v4(padded_batch, num_classes = self.NUM_CLASSES)

			slogits = end_points['Predictions']
			saver = tf.train.Saver()

			with tf.Session() as sess:
				init = (tf.global_variables_initializer(), tf.local_variables_initializer())
				sess.run(init)

				ckpt = tf.train.get_checkpoint_state(self.MODEL_DIR)
				if ckpt and ckpt.model_checkpoint_path:
					print("Restoring saved checkpoint model.")
					saver.restore(sess, ckpt.model_checkpoint_path)
				else:
					raise Exception('Unable to find checkpoint file.')

				logits_arr = []

				x_logits_len = int(self.X_SIZE / window_stride[0])+1
				y_logits_len = int(self.Y_SIZE / window_stride[1])+1
				total_logits_count = x_logits_len * y_logits_len

				if total_logits_count != len(coord):
					raise Exception("The expected total number of window tiles does not match the number of generated starting points for window tiles.")

				count = 0

				while True:
					try:
						count = min(count, total_logits_count)
						progress_bar.bar(count, total_logits_count, text = "Calculated {} images out of {}. "
																			.format(min(count, total_logits_count),
																			 total_logits_count))
						new_logits = sess.run(tf.cast(slogits, tf.float32))
						logits_arr = new_logits if logits_arr == [] else np.concatenate([logits_arr, new_logits])
					except tf.errors.OutOfRangeError:
						print("End of image detected.")
						break
					count += self.BATCH_SIZE
				progress_bar.end()
			
			# Crop the output to exclude padding
			print("total size of padded dataset: {}".format(logits_arr.shape))
			logits_arr = logits_arr[0:total_logits_count]
			logits_out = np.resize(logits_arr, [y_logits_len, x_logits_len, self.NUM_CLASSES])
			
			self.display(self.WHOLE_IMAGE, logits_out, self.SIZE)

	def display(self, image_file, logits, size):
		'''Displays logits calculated using scan_image as a heatmap overlay.'''
		print("Received logits, size=%s, (%s x %s)" % (size, len(logits), len(logits[0])))
		print("Calculating overlay matrix...")

		axis_color = 'lightgoldenrodyellow'

		fig = plt.figure()
		ax = fig.add_subplot(111)

		fig.subplots_adjust(bottom = 0.25)

		im = plt.imread(image_file)
		implot = ax.imshow(im, zorder=0)
		gca = plt.gca()
		gca.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

		# Calculations to determine appropriate offset for heatmap
		im_extent = implot.get_extent()
		extent = [im_extent[0] + size/2, im_extent[1] - size/2, im_extent[2] - size/2, im_extent[3] + size/2]

		# Define color map
		jetMap = np.linspace(0.45, 0.95, 255)
		cmMap = cm.nipy_spectral(jetMap)
		newMap = mcol.ListedColormap(cmMap)

		heatmap_dict = {}

		def slider_func(val):
			for h, s in heatmap_dict.values():
				h.set_alpha(s.val)

		# Make heatmaps and sliders
		for i in range(self.NUM_CLASSES):
			ax_slider = fig.add_axes([0.25, 0.2-(0.2/self.NUM_CLASSES)*i, 0.5, 0.03], facecolor=axis_color)
			heatmap = ax.imshow(logits[:, :, i], extent=extent, cmap=newMap, alpha = 0.0, interpolation='bicubic', zorder=10)
			slider = Slider(ax_slider, 'Class {}'.format(i), 0, 1, valinit = 0)
			heatmap_dict.update({"Class{}".format(i): [heatmap, slider]})
			slider.on_changed(slider_func)

		plt.show()

if __name__==('__main__'):
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	tf.logging.set_verbosity(tf.logging.ERROR)
	c = Convoluter()
	c.scan_image()

