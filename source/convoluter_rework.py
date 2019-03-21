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
from os.path import join, isfile

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

# TODO: memory management
# TODO: add logits to CSV file as metadata

class Convoluter:
	def __init__(self, size, num_classes, batch_size, use_fp16, save_folder = ''):
		self.IMAGES = {}
		self.CATEGORIES = {}
		self.MODEL_DIR = None
		self.PKL_DICT = {}
		self.SIZE = size
		self.NUM_CLASSES = num_classes
		self.BATCH_SIZE = batch_size
		self.DTYPE = tf.float16 if use_fp16 else tf.float32
		self.DTYPE_INT = tf.int16 if use_fp16 else tf.int32
		self.SAVE_FOLDER = save_folder
		self.STRIDE = 4

	def load_images(self, whole_image_array, directory, category=None):
		for image in whole_image_array:
			name = image[:-4]
			self.IMAGES.update({name: join(directory, image)})
			if category:
				self.CATEGORIES.update({name: category})

	def load_pkl(self, pkl_array, directory):
		for pkl in pkl_array:
			p_name = pkl[:-4]
			self.PKL_DICT.update({p_name: join(directory, pkl)})

	def build_model(self, model_dir):
		self.MODEL_DIR = model_dir


	def convolute_all_images(self, save_heatmaps, display_heatmaps, save_final_layer, export_tiles):
		'''Parent function to guide convolution across a whole-slide image and execute desired functions.

		Args:
			save_heatmaps: 				Bool, if true will save heatmap overlays as PNG files
			display_heatmaps:			Bool, if true will display interactive heatmap for each whole-slide image
			save_final_layer: 			Bool, if true will calculate and save final layer weights in CSV file
			export_tiles:				Bool, if true will save convoluted image tiles to subdirectory "tiles"

		Returns:
			None
		'''
		if export_tiles and not os.path.exists(join(self.SAVE_FOLDER, "tiles")):
			os.makedirs(join(self.SAVE_FOLDER, "tiles"))
		if not save_heatmaps and not display_heatmaps:
			# No need to calculate overlapping tiles
			print("Calculating only non-overlapping tiles for final layer weight extraction.")
			self.STRIDE = 1
		for case_name in self.IMAGES:
			print(f"Working on case {case_name}")
			image_path = self.IMAGES[case_name]
			# Use PKL logits if available
			if case_name in self.PKL_DICT:
				with open(self.PKL_DICT[case], 'rb') as handle:
					logits = pickle.load(handle)
			# Otherwise recalculate
			else:
				logits, final_layer, final_layer_labels = self.scan_image(image_path, case_name, 
																		  export_tiles, save_final_layer,
																		  save_pkl=(save_heatmaps or display_heatmaps))
			if save_heatmaps:
				self.export_heatmaps(image_path, logits, self.SIZE, case_name)
			if save_final_layer:
				category = "None" if case_name not in self.CATEGORIES else self.CATEGORIES[case_name]
				self.save_csv(final_layer, final_layer_labels, case_name, category)
			if display_heatmaps:
				self.fast_display(image_path, logits, self.SIZE, case_name)

	def scan_image(self, image_path, case_name, export_tiles=False, final_layer=False, save_pkl=True):
		'''Returns logits and final layer weights'''
		warnings.simplefilter('ignore', Image.DecompressionBombWarning)
		# Reset graph to prevent OOM errors when convoluting across multiple images
		# TODO: split this code into two functions: one that loads a model, and another that handles
		#  alternating inputs
		tf.reset_default_graph()

		# Load whole-slide-image into Numpy array and prepare pkl output
		whole_slide_image = imageio.imread(image_path)
		shape = whole_slide_image.shape
		pkl_name =  case_name + '.pkl'

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
				# Check if this is a unique tile without overlap (e.g. if stride was 1)
				if (y % self.SIZE == 0) and (x % self.SIZE == 0):
					# If true, this tile is eligible for final layer weight calculation
					coord.append([y, x, True])
				else:
					coord.append([y, x, False])

		def gen_slice():
			for ci in range(len(coord)):
				c = coord[ci]
				region = whole_slide_image[c[0]:c[0] + window_size[0], c[1]:c[1] + window_size[1],]
				coord_label = ci
				unique_tile = c[2]
				if export_tiles and unique_tile:
					imsave(join(self.SAVE_FOLDER, f'tiles/{case_name}_{ci}.jpg'), region)
				yield region, coord_label, unique_tile

		with tf.Graph().as_default() as g:
			# Generate dataset from coordinates
			with tf.name_scope('dataset_input'):
				tile_dataset = tf.data.Dataset.from_generator(gen_slice, (self.DTYPE, tf.int64, tf.bool))
				tile_dataset = tile_dataset.batch(self.BATCH_SIZE, drop_remainder = False)
				tile_dataset = tile_dataset.prefetch(1)
				tile_iterator = tile_dataset.make_one_shot_iterator()
				next_batch_images, next_batch_labels, next_batch_unique  = tile_iterator.get_next()

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
				prelogits_arr = []	# Final layer weights
				logits_arr = []		# Logits (predictions)
				unique_arr = []		# Boolean array indicating whether tile is unique (non-overlapping)

				while True:
					try:
						count = min(count, total_logits_count)
						progress_bar.bar(count, total_logits_count, text = "Calculated {} images out of {}. "
																			.format(min(count, total_logits_count),
																			 total_logits_count))
						if final_layer:
							new_logits, new_prelogits, new_labels, new_unique = sess.run([tf.cast(slogits, tf.float32),
																			  tf.cast(prelogits, tf.float32),
																			  next_batch_labels,
																			  next_batch_unique])
							prelogits_arr = new_prelogits if prelogits_arr == [] else np.concatenate([prelogits_arr, new_prelogits])
							unique_arr = new_unique if unique_arr == [] else np.concatenate([unique_arr, new_unique])
						else:
							new_logits, new_labels = sess.run([tf.cast(slogits, tf.float32), next_batch_labels])

						logits_arr = new_logits if logits_arr == [] else np.concatenate([logits_arr, new_logits])
						labels_arr = new_labels if labels_arr == [] else np.concatenate([labels_arr, new_labels])
					except tf.errors.OutOfRangeError:
						progress_bar.end()
						break
					count += self.BATCH_SIZE

			# Crop the output to exclude padding
			logits_arr = logits_arr[0:total_logits_count]
			labels_arr = labels_arr[0:total_logits_count]

			sorted_indices = labels_arr.argsort()
			logits_arr = logits_arr[sorted_indices]
			labels_arr = labels_arr[sorted_indices]

			if final_layer:
				prelogits_arr = prelogits_arr[0:total_logits_count]
				prelogits_arr = prelogits_arr[sorted_indices]
				unique_arr = unique_arr[0:total_logits_count]
				unique_arr = unique_arr[sorted_indices]

			# Organize array into 2D format corresponding to where each logit was calculated
			if final_layer:
				#print(f"Resizing final layer with {num_tensors_final_layer} features")
				#prelogits_out = np.resize(prelogits_arr, [total_logits_count, num_tensors_final_layer])
				prelogits_out = [prelogits_arr[p] for p in range(len(prelogits_arr)) if unique_arr[p]]
				prelogits_labels = [l for l in range(len(unique_arr)) if unique_arr[l]]
			else:
				prelogits_out = None
			logits_out = np.resize(logits_arr, [y_logits_len, x_logits_len, self.NUM_CLASSES])
			if save_pkl:
				with open(os.path.join(self.SAVE_FOLDER, pkl_name), 'wb') as handle:
					pickle.dump(logits_out, handle)

			return logits_out, prelogits_out, prelogits_labels

	def save_csv(self, output, labels, name, category):
		print("Writing csv...")
		csv_started = os.path.exists(join(self.SAVE_FOLDER, 'final_layer_weights.csv'))
		write_mode = 'a' if csv_started else 'w'
		with open(join(self.SAVE_FOLDER, 'final_layer_weights.csv'), write_mode) as csv_file:
			csv_writer = csv.writer(csv_file, delimiter = ',')
			if not csv_started:
				csv_writer.writerow(["Tile_num", "Case", "Category"] + [f"Node{n}" for n in range(len(output[0]))])
			for l in range(len(output)):
				out = output[l].tolist()
				csv_writer.writerow([labels[l], name, category] + out)

	def export_heatmaps(self, image_file, logits, size, name):
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

def get_args():
	parser = argparse.ArgumentParser(description = 'Convolutionally applies a saved Tensorflow model to a larger image, displaying the result as a heatmap overlay.')
	parser.add_argument('-m', '--model', help='Path to model directory containing stored checkpoint.')
	parser.add_argument('-i', '--image', help='Path to whole-slide image or folder of images to analyze.')
	parser.add_argument('-p', '--pkl', help='Python Pickle file, or folder of pkl files, containing pre-calculated weights to load')
	parser.add_argument('-o', '--out', help='Path to directory in which exported images and data will be saved.')
	parser.add_argument('-s', '--size', type=int, help='Size of image patches to analyze.')
	parser.add_argument('-c', '--classes', type=int, default = 1, help='Number of unique output classes contained in the model.')
	parser.add_argument('-b', '--batch', type=int, default = 64, help='Batch size for which to run the analysis.')
	parser.add_argument('--fp16', action="store_true", help='Use Float16 operators (half-precision) instead of Float32.')
	parser.add_argument('--save', action="store_true", help='Save heatmaps to PNG file instead of displaying.')
	parser.add_argument('--final', action="store_true", help='Calculate and export image tiles and final layer weights.')
	parser.add_argument('--display', action="store_true", help='Display results with interactive heatmap for each whole-slide image.')
	parser.add_argument('--export', action="store_true", help='Export calculated images tiles.')
	return parser.parse_args()

if __name__==('__main__'):
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	tf.logging.set_verbosity(tf.logging.ERROR)
	args = get_args()

	'''
	--- New args -----
	Flag 			Description 						Use
	args.model 		Path to model directory				Will calculate new logits if supplied
	args.image 		Path to single image or directory	Whole-slide image(s); either loads single image or all within directory; 
														if subfolders present, calculates labels
	args.pkl 		Path to single pkl or directory		Loads pkl file(s) if supplied
														If both model and pkl are supplied, will use pkl files
	args.out 		Path to export directory			Where to save pkl files, heatmap files, and CSVs, defaults to empty string
	args.size 		Same
	args.classes 	Same
	args.batch 		Same
	fp16 			Same
	save 			Same
	final 			Same 								Will raise error if model not supplied
	display 		Display interactive heatmap 		Will pause after each image convolution to display calculated heatmap
	export 			Export calculated tiles 			Will save convoluted tiles if flag provided
	'''

	# New method
	c = Convoluter(args.size, args.classes, args.batch, args.fp16, args.out)

	if isfile(args.image):
		image_list = [args.image.split('/'[-1])]
		image_dir = "/".join(args.image.split('/')[:-1])
		c.load_images(image_list, image_dir)
	else:
		# First, load images in the directory, not assigning any category
		image_list = [i for i in os.listdir(args.image) if (isfile(join(args.image, i)) and (i[-3:] == "jpg"))]	
		c.load_images(image_list, args.image)
		# Next, load images in subdirectories, assigning category by subdirectory name
		dir_list = [d for d in os.listdir(args.image) if not isfile(join(args.image, d))]
		for directory in dir_list:
			image_list = [i for i in os.listdir(join(args.image, directory)) if (isfile(join(args.image, directory, i)) and (i[-3:] == "jpg"))]	
			c.load_images(image_list, join(args.image, directory), category=directory)
	if args.pkl and isfile(args.pkl):
		pkl_list = [args.pkl.split('/'[-1])]
		pkl_dir = "/".join(args.pkl.split('/')[:-1])
	elif args.pkl:
		pkl_list = [p for p in os.listdir(args.pkl) if (isfile(join(args.pkl, p)) and (p[-3:] == "pkl"))]
		pkl_dir = args.pkl
	else:
		pkl_list = []
		pkl_dir = ''

	c.load_pkl(pkl_list, pkl_dir)
	c.build_model(args.model)
	c.convolute_all_images(args.save, args.display, args.final, args.export)