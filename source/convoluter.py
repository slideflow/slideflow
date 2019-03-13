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
import imageio
import inception_v4
from tensorflow.contrib.framework import arg_scope
from inception_utils import inception_arg_scope
from PIL import Image
import argparse
import pickle
from scipy.misc import imsave

from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcol

Image.MAX_IMAGE_PIXELS = 100000000000

class Convoluter:
	def __init__(self, whole_image, model_dir, size, num_classes, batch_size, use_fp16):
		self.WHOLE_IMAGE = whole_image
		self.MODEL_DIR = model_dir
		self.SIZE = size
		self.NUM_CLASSES = num_classes
		self.BATCH_SIZE = batch_size
		self.USE_FP16 = use_fp16
		self.DTYPE = tf.float16 if self.USE_FP16 else tf.float32
		self.DTYPE_INT = tf.int16 if self.USE_FP16 else tf.int32

		# Display variables
		stride_divisor = 4
		self.STRIDES = [1, int(size/stride_divisor), int(size/stride_divisor), 1]

	def scan_image(self, display=True):
		warnings.simplefilter('ignore', Image.DecompressionBombWarning)

		# Load whole-slide-image into Numpy array
		whole_slide_image = imageio.imread(self.WHOLE_IMAGE)
		shape = whole_slide_image.shape

		print("Loading image of size {} x {}".format(shape[0], shape[1]))

		# Calculate window sizes, strides, and coordinates for windows
		window_size = [self.SIZE, self.SIZE]
		window_stride = [int(self.SIZE/4), int(self.SIZE/4)]

		#tf.cast(***, tf.int32)

		if (window_size[0] > shape[0]) or (window_size[1] > shape[1]):
				raise IndexError("Window size is too large")

		coord = []
		coord_d = []

		self.Y_SIZE = shape[0] - window_size[0]
		self.X_SIZE = shape[1] - window_size[1]

		for y in range(0, (shape[0]+1) - window_size[0], window_stride[0]):
			for x in range(0, (shape[1]+1) - window_size[1], window_stride[1]):
				# f is a flag (number 0 -7) which corresponds to a flip/rotation combination
				for f in range(8):
					coord.append([y, x, f])
					# coord_d is purely for debugging purposes
					coord_d.append([y/self.Y_SIZE, x/self.X_SIZE, f])

		def rotate_image(np_array, flag):
			if flag == 0:
				return np_array
			elif flag == 1:
				return np.rot90(np_array)
			elif flag == 2:
				return np.rot90(np_array, 2)
			elif flag == 3:
				return np.rot90(np_array, 3)
			elif flag == 4:
				return np.fliplr(np_array)
			elif flag == 5:
				return np.fliplr(np.rot90(np_array))
			elif flag == 6:
				return np.fliplr(np.rot90(np_array, 2))
			elif flag == 7:
				return np.fliplr(np.rot90(np_array, 3))
			else:
				raise IndexError("Invalid flag selection for image flip/rotation, must be integer 0 - 7.")

		def gen_slice():
			for c in coord:
				rotated = rotate_image(whole_slide_image[c[0]:c[0] + window_size[0], c[1]:c[1] + window_size[1],], c[2])
				coord_label =str (c[0])+'-'+str(c[1])
				#matrix_mean = np.matrix(rotated[:,:,0]).mean()
				#imsave('img/c{}.jpg'.format(str(c[0])+'-'+str(c[1])+'-'+str(c[2])), rotated)
				yield rotated, coord_label
				#yield [matrix_mean, c[0], c[1], c[2], c[0]]

		# Function for testing/troubleshooting
		def gen_coord_d():
			for c in coord_d:
				yield [c[0], c[1], c[0], c[1], c[0]]

		with tf.Graph().as_default() as g:
			# Generate dataset from coordinates
			tile_dataset = tf.data.Dataset.from_generator(gen_slice, (self.DTYPE)) # replace gen_slice with gen_coord_d for testing purposes
			tile_dataset = tile_dataset.batch(self.BATCH_SIZE, drop_remainder = False)
			tile_dataset = tile_dataset.prefetch(2)
			tile_iterator = tile_dataset.make_one_shot_iterator()
			next_batch_images, next_batch_labels = tile_iterator.get_next()
			#next_coord = tf.cast(next_coord, tf.int32)

			# Generate ops that will convert batch of coordinates to extracted & processed image patches from whole-slide-image
			#image_patches = tf.map_fn(get_image_slice, next_batch)
			image_patches = tf.map_fn(lambda patch: tf.cast(tf.image.per_image_standardization(patch), self.DTYPE), next_batch_images)

			# Pad the batch if necessary to create a batch of minimum size BATCH_SIZE
			padded_batch = tf.concat([image_patches, tf.zeros([self.BATCH_SIZE - tf.shape(image_patches)[0], self.SIZE, self.SIZE, 3], # image_patches instead of next_batch
														dtype=self.DTYPE)], 0)
			padded_batch.set_shape([self.BATCH_SIZE, self.SIZE, self.SIZE, 3])

			'''padded_batch = tf.concat([next_batch, tf.zeros([self.BATCH_SIZE - tf.shape(next_batch)[0], 5], dtype=self.DTYPE)], 0)
			padded_batch.set_shape([self.BATCH_SIZE, 5])'''

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
				labels_arr = []

				x_logits_len = int(self.X_SIZE / window_stride[1])+1
				y_logits_len = int(self.Y_SIZE / window_stride[0])+1
				total_logits_count = x_logits_len * y_logits_len * 8	# Multiplied by 8, as there are 8 different flip/rotation combinations

				if total_logits_count != len(coord):
					raise Exception("The expected total number of window tiles does not match the number of generated starting points for window tiles.")

				count = 0

				while True:
					try:
						count = min(count, total_logits_count)
						progress_bar.bar(count, total_logits_count, text = "Calculated {} images out of {}. "
																			.format(min(count, total_logits_count),
																			 total_logits_count))
						new_logits, new_labels = sess.run([tf.cast(slogits, tf.float32), next_batch_labels])

						# Execute this code for debugging purposes if using gen_coord_d
						#new_logits = sess.run(tf.cast(padded_batch, tf.float32))

						logits_arr = new_logits if logits_arr == [] else np.concatenate([logits_arr, new_logits])
						labels_arr = new_labels if labels_arr == [] else np.concatenate([labels_arr, new_labels])
					except tf.errors.OutOfRangeError:
						print("End of image detected.")
						break
					count += self.BATCH_SIZE
				progress_bar.end()
			
			# Crop the output to exclude padding
			print("total size of padded dataset: {}".format(logits_arr.shape))
			logits_arr = logits_arr[0:total_logits_count]
			labels_arr = labels_arr[0:total_logits_count]

			print('First value of unmerged array:')
			print(logits_arr[0])

			print('Sorting arrays')
			sorted_indices = labels_arr.argsort()
			logits_arr = logits_arr[sorted_indices]
			labels_arr = labels_arr[sorted_indices]

			# Average logits across different flips/rotations
			if logits_arr.shape[0] % 8 != 0:
				raise IndexError("Output logits not divisible by 8, likely error with padding.")

			# Save output in Python Pickle format for later display
			case_name = ''.join(self.WHOLE_IMAGE.split('/')[-1].split('.')[:-1])
			pkl_name =  case_name + '.pkl'

			with open('raw_'+pkl_name, 'wb') as handle:
				pickle.dump(logits_arr.reshape(-1, 8, self.NUM_CLASSES), handle)
			
			logits_arr = np.mean(logits_arr.reshape(-1, 8, self.NUM_CLASSES), axis=1)

			print('First value of merged array:')
			print(logits_arr[0])

			# Organize array into 2D format corresponding to where each logit was calculated
			logits_out = np.resize(logits_arr, [y_logits_len, x_logits_len, self.NUM_CLASSES])

			with open(pkl_name, 'wb') as handle:
				pickle.dump(logits_out, handle)
			
			if display: 
				self.display(self.WHOLE_IMAGE, logits_out, self.SIZE, case_name)

	def display(self, image_file, logits, size, name):
		'''Displays logits calculated using scan_image as a heatmap overlay.'''
		print("Received logits, size=%s, (%s x %s)" % (size, len(logits), len(logits[0])))
		print("Calculating overlay matrix...")

		axis_color = 'lightgoldenrodyellow'

		fig = plt.figure()
		ax = fig.add_subplot(111)

		fig.subplots_adjust(bottom = 0.25, top=0.95)

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
			heatmap = ax.imshow(logits[:, :, i], extent=extent, cmap=newMap, alpha = 0.0, interpolation='none', zorder=10) #bicubic
			slider = Slider(ax_slider, 'Class {}'.format(i), 0, 1, valinit = 0)
			heatmap_dict.update({"Class{}".format(i): [heatmap, slider]})
			slider.on_changed(slider_func)

		fig.canvas.set_window_title(name)
		plt.show()

	def load_pkl_and_scan_image(self, pkl_file):
		print("Loading pre-calculated logits...")
		with open(pkl_file, 'rb') as handle:
			logits = pickle.load(handle)

		# Remove this 
		'''self.Y_SIZE = shape[0] - self.SIZE
		self.X_SIZE = shape[1] - self.SIZE
		window_stride = [int(self.SIZE/4), int(self.SIZE/4)]

		x_logits_len = int(self.X_SIZE / window_stride[1])+1
		y_logits_len = int(self.Y_SIZE / window_stride[0])+1'''

		logits = logits[:, 0, :]
		logits_arr = np.resize(logits, [217, 167, self.NUM_CLASSES])

		self.display(self.WHOLE_IMAGE, logits_arr, self.SIZE, pkl_file.split('/')[-1])

if __name__==('__main__'):
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	tf.logging.set_verbosity(tf.logging.ERROR)

	parser = argparse.ArgumentParser(description = 'Convolutionally applies a saved Tensorflow model to a larger image, displaying the result as a heatmap overlay.')
	parser.add_argument('-d', '--dir', help='Path to model directory containing stored checkpoint.')
	parser.add_argument('-l', '--load', help='Python Pickle file containing pre-calculated weights to load')
	parser.add_argument('-i', '--image', help='Image on which to apply heatmap.')
	parser.add_argument('-s', '--size', type=int, help='Size of image patches to analyze.')
	parser.add_argument('-c', '--classes', type=int, help='Number of unique output classes contained in the model.')
	parser.add_argument('-b', '--batch', type=int, default = 64, help='Batch size for which to run the analysis.')
	parser.add_argument('--fp16', action="store_true", help='Use Float16 operators (half-precision) instead of Float32.')
	args = parser.parse_args()

	if args.load:
		c = Convoluter(args.image, None, args.size, args.classes, args.batch, args.fp16)
		c.load_pkl_and_scan_image(args.load)
	else:
		c = Convoluter(args.image, args.dir, args.size, args.classes, args.batch, args.fp16)
		c.scan_image()

