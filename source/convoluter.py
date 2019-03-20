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
import csv
from scipy.misc import imsave

from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcol
from matplotlib import pyplot as mp

from fastim import FastImshow

Image.MAX_IMAGE_PIXELS = 100000000000

class Convoluter:
	def __init__(self, whole_image, model_dir, size, num_classes, batch_size, use_fp16, save_folder = ''):
		self.WHOLE_IMAGE = whole_image
		self.MODEL_DIR = model_dir
		self.SIZE = size
		self.NUM_CLASSES = num_classes
		self.BATCH_SIZE = batch_size
		self.USE_FP16 = use_fp16
		self.DTYPE = tf.float16 if self.USE_FP16 else tf.float32
		self.DTYPE_INT = tf.int16 if self.USE_FP16 else tf.int32
		self.SAVE_FOLDER = save_folder

		# Display variables
		self.STRIDE = 4

	def scan_image(self, display=True, prefix='', save = False, export_tiles = False, final_layer = True):
		warnings.simplefilter('ignore', Image.DecompressionBombWarning)

		# Load whole-slide-image into Numpy array
		whole_slide_image = imageio.imread(self.WHOLE_IMAGE)
		shape = whole_slide_image.shape
		case_name = ''.join(self.WHOLE_IMAGE.split('/')[-1].split('.')[:-1])
		pkl_name =  case_name + '.pkl'
		if final_layer: display = False

		print(f"Loading image of size {shape[0]} x {shape[1]}")

		# Calculate window sizes, strides, and coordinates for windows
		window_size = [self.SIZE, self.SIZE]
		window_stride = [int(self.SIZE/self.STRIDE), int(self.SIZE/self.STRIDE)]

		if (window_size[0] > shape[0]) or (window_size[1] > shape[1]):
				raise IndexError("Window size is too large")

		coord = []

		self.Y_SIZE = shape[0] - window_size[0]
		self.X_SIZE = shape[1] - window_size[1]

		for y in range(0, (shape[0]+1) - window_size[0], window_stride[0]):
			for x in range(0, (shape[1]+1) - window_size[1], window_stride[1]):
				coord.append([y, x])

		def gen_slice():
			for ci in range(len(coord)):
				c = coord[ci]
				region = whole_slide_image[c[0]:c[0] + window_size[0], c[1]:c[1] + window_size[1],]
				coord_label = ci
				if export_tiles:
					imsave(f'tiles/{case_name}_{ci}.jpg', region)
				yield region, coord_label

		with tf.Graph().as_default() as g:
			# Generate dataset from coordinates
			tile_dataset = tf.data.Dataset.from_generator(gen_slice, (self.DTYPE, tf.int64))
			tile_dataset = tile_dataset.batch(self.BATCH_SIZE, drop_remainder = False)
			tile_dataset = tile_dataset.prefetch(2)
			tile_iterator = tile_dataset.make_one_shot_iterator()
			next_batch_images, next_batch_labels  = tile_iterator.get_next() #next_batch_labels

			# Generate ops that will convert batch of coordinates to extracted & processed image patches from whole-slide-image
			image_patches = tf.map_fn(lambda patch: tf.cast(tf.image.per_image_standardization(patch), self.DTYPE), next_batch_images)

			# Pad the batch if necessary to create a batch of minimum size BATCH_SIZE
			padded_batch = tf.concat([image_patches, tf.zeros([self.BATCH_SIZE - tf.shape(image_patches)[0], self.SIZE, self.SIZE, 3], # image_patches instead of next_batch
														dtype=self.DTYPE)], 0)
			padded_batch.set_shape([self.BATCH_SIZE, self.SIZE, self.SIZE, 3])

			with arg_scope(inception_arg_scope()):
				_, end_points = inception_v4.inception_v4(padded_batch, num_classes = self.NUM_CLASSES)

			prelogits = end_points['PreLogitsFlatten']
			slogits = end_points['Predictions']
			num_tensors_final_layer = prelogits.get_shape().as_list()[1]
			vars_to_restore = []

			for var_to_restore in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
				if ((var_to_restore.name[12:21] != "AuxLogits") and 
					((var_to_restore.name[:25] != "InceptionV4/Logits/Logits") or not final_layer)):
					vars_to_restore.append(var_to_restore)

			saver = tf.train.Saver(vars_to_restore)

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
						
						endpoint = prelogits if final_layer else slogits
						new_logits, new_labels = sess.run([tf.cast(endpoint, tf.float32), next_batch_labels])

						logits_arr = new_logits if logits_arr == [] else np.concatenate([logits_arr, new_logits])
						labels_arr = new_labels if labels_arr == [] else np.concatenate([labels_arr, new_labels])
					except tf.errors.OutOfRangeError:
						print("End of image detected.")
						break
					count += self.BATCH_SIZE
				progress_bar.end()
			
			# Crop the output to exclude padding
			print(f"total size of padded dataset: {logits_arr.shape}")
			logits_arr = logits_arr[0:total_logits_count]
			labels_arr = labels_arr[0:total_logits_count]

			print('Sorting arrays')
			sorted_indices = labels_arr.argsort()
			logits_arr = logits_arr[sorted_indices]
			labels_arr = labels_arr[sorted_indices]

			# Organize array into 2D format corresponding to where each logit was calculated
			if final_layer:
				print(f"Resizing to {total_logits_count} x {num_tensors_final_layer}")
				output = np.resize(logits_arr, [total_logits_count, num_tensors_final_layer])
				self.save_csv(output, labels_arr, case_name, "NIFTP")
			else:
				output = np.resize(logits_arr, [y_logits_len, x_logits_len, self.NUM_CLASSES])
				with open(os.path.join(self.SAVE_FOLDER, prefix+pkl_name), 'wb') as handle:
					pickle.dump(output, handle)
			
			if display: 
				self.fast_display(self.WHOLE_IMAGE, output, self.SIZE, case_name)

			if save and not final_layer:
				self.save_heatmaps(self.WHOLE_IMAGE, output, self.SIZE, case_name)

	def save_csv(self, output, labels, name, category):
		print("Writing csv...")
		with open(os.path.join(self.SAVE_FOLDER, name+'_final_layer_weights.csv'), 'w') as csv_file:
			csv_writer = csv.writer(csv_file, delimiter = ',')
			csv_writer.writerow(["Tile_num", "Category"] + [f"Node{n}" for n in range(len(output[0]))])
			for l in range(len(labels)):
				label = labels[l]
				out = output[l].tolist()
				csv_writer.writerow([label, category] + out)

	def save_heatmaps(self, image_file, logits, size, name):
		'''Displays logits calculated using scan_image as a heatmap overlay.'''
		print(f"Loading image and assembling heatmaps for image {image_file}...")

		fig = plt.figure(figsize=(18, 16))
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

		# Make heatmaps and sliders
		for i in range(self.NUM_CLASSES):
			heatmap = ax.imshow(logits[:, :, i], extent=extent, cmap=newMap, alpha = 0.0, interpolation='none', zorder=10) #bicubic
			#slider = Slider(ax_slider, 'Class {}'.format(i), 0, 1, valinit = 0)
			heatmap_dict.update({i: heatmap})

		mp.savefig(os.path.join(self.SAVE_FOLDER, f'{name}-raw.png'), bbox_inches='tight')

		for i in range(self.NUM_CLASSES):
			heatmap_dict[i].set_alpha(0.6)
			mp.savefig(os.path.join(self.SAVE_FOLDER, f'{name}-{i}.png'), bbox_inches='tight')
			heatmap_dict[i].set_alpha(0.0)

		mp.close()

	def fast_display(self, image_file, logits, size, name):
		'''*** Experimental ***'''
		print("Received logits, size=%s, (%s x %s)" % (size, len(logits), len(logits[0])))
		print("Calculating overlay matrix and displaying with dynamic resampling...")

		fig = plt.figure()
		ax = fig.add_subplot(111)
		fig.subplots_adjust(bottom = 0.25, top=0.95)

		buf = plt.imread(image_file)
		implot = FastImshow(buf, ax, extent=None, tgt_res=1024)
		gca = plt.gca()
		gca.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

		# Calculations to determine appropriate offset for heatmap
		im_extent = implot.extent
		extent = [im_extent[0] + size/2, im_extent[1] - size/2, im_extent[2] + size/2, im_extent[3] - size/2]

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
			ax_slider = fig.add_axes([0.25, 0.2-(0.2/self.NUM_CLASSES)*i, 0.5, 0.03], facecolor='lightgoldenrodyellow')
			heatmap = ax.imshow(logits[:, :, i], extent=extent, cmap=newMap, alpha = 0.0, interpolation='none', zorder=10) #bicubic
			slider = Slider(ax_slider, f'Class {i}', 0, 1, valinit = 0)
			heatmap_dict.update({f"Class{i}": [heatmap, slider]})
			slider.on_changed(slider_func)

		fig.canvas.set_window_title(name)
		implot.show()
		plt.show()

	def load_pkl_and_scan_image(self, pkl_file):
		print("Loading pre-calculated logits...")
		with open(pkl_file, 'rb') as handle:
			logits = pickle.load(handle)

		if not self.NUM_CLASSES:
			self.NUM_CLASSES = logits.shape[2] 

		self.fast_display(self.WHOLE_IMAGE, logits, self.SIZE, pkl_file.split('/')[-1])

	def load_pkl_and_save_heatmaps(self, pkl_file):
		print("Loading pre-calculated logits...")
		with open(pkl_file, 'rb') as handle:
			logits = pickle.load(handle)

		if not self.NUM_CLASSES:
			self.NUM_CLASSES = logits.shape[2] 

		self.save_heatmaps(self.WHOLE_IMAGE, logits, self.SIZE, pkl_file.split('/')[-1])

if __name__==('__main__'):
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	tf.logging.set_verbosity(tf.logging.ERROR)

	parser = argparse.ArgumentParser(description = 'Convolutionally applies a saved Tensorflow model to a larger image, displaying the result as a heatmap overlay.')
	parser.add_argument('-m', '--model', help='Path to model directory containing stored checkpoint.')
	parser.add_argument('-f', '--folder', help='Folder to search for whole-slide-images to analyze.')
	parser.add_argument('-l', '--load', help='Python Pickle file containing pre-calculated weights to load')
	parser.add_argument('-i', '--image', help='Image on which to apply heatmap.')
	parser.add_argument('-s', '--size', type=int, help='Size of image patches to analyze.')
	parser.add_argument('-c', '--classes', type=int, default = 1, help='Number of unique output classes contained in the model.')
	parser.add_argument('-b', '--batch', type=int, default = 64, help='Batch size for which to run the analysis.')
	parser.add_argument('--fp16', action="store_true", help='Use Float16 operators (half-precision) instead of Float32.')
	parser.add_argument('--save', action="store_true", help='Save heatmaps to PNG file instead of displaying.')
	parser.add_argument('--final', action="store_true", help='Calculate and export image tiles and final layer weights.')
	args = parser.parse_args()

	if args.load:
		c = Convoluter(args.image, None, args.size, args.classes, 1, args.fp16)
		c.load_pkl_and_scan_image(args.load)
	elif args.folder:
		if args.model:
			# Load images from a directory and calculate logits
			c = Convoluter('', args.model, args.size, args.classes, args.batch, args.fp16, save_folder = args.folder)
			if args.final: c.STRIDE = 1
			for f in [f for f in os.listdir(args.folder) if (os.path.isfile(os.path.join(args.folder, f)) and (f[-3:] == "jpg"))]:
				c.WHOLE_IMAGE = os.path.join(args.folder, f)
				c.scan_image(False, '', save = args.save)
		elif args.save:
			# Load images from a directory and save heatmaps as image files
			for f in [f for f in os.listdir(args.folder) if (os.path.isfile(os.path.join(args.folder, f)) and (f[-3:] == "pkl"))]:
				pkl = os.path.join(args.folder, f)
				#the [7: component below is a temporary workaround since I exported my pkl files with the prefix 'active_'
				image = os.path.join(args.folder, f[7:-4]+'.jpg')
				c = Convoluter(image, None, args.size, args.classes, 1, args.fp16, save_folder = args.folder)
				c.load_pkl_and_save_heatmaps(pkl)
	else:
		c = Convoluter(args.image, args.model, args.size, args.classes, args.batch, args.fp16)
		c.scan_image()

