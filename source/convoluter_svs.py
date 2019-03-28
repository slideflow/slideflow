# Copyright (C) James Dolezal - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, March 2019
# ==========================================================================

'''Convolutionally applies a saved Tensorflow model to a larger image, displaying
the result as a heatmap overlay.

This version will convolute across an SVS image instead of JPG chunks.

Requires: Openslide (https://openslide.org/download/)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import warnings
from os.path import join, isfile, exists

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
import openslide as ops
import shapely.geometry as sg
import cv2

from multiprocessing.dummy import Pool as ThreadPool

from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcol
from matplotlib import pyplot as mp

from fastim import FastImshow

Image.MAX_IMAGE_PIXELS = 100000000000
NUM_THREADS = 6

class AnnotationObject:
	def __init__(self, name):
		self.name = name
		self.coordinates = []
	def add_coord(self, coord):
		self.coordinates.append(coord)
	def scaled_area(self, scale):
		return np.multiply(self.coordinates, 1/scale)
	def print_coord(self):
		for c in self.coordinates: print(c)

class SVSReader:
	def __init__(self, path, export_folder=None, pb=None):
		self.annotations = []
		self.export_folder = export_folder
		self.pb = pb # Progress bar
		self.p_id = None
		self.print = print if not pb else pb.print
		self.name = path[:-4].split('/')[-1]
		annotation_path = path[:-4] + ".csv"
		if not exists(annotation_path):
			#raise FileNotFoundError("Unable to locate associated '.csv' annotation file. Generate this file using QuPath and the included Groovy script.")
			self.print(f" ! [{self.name}] WARNING: No annotation file found, using whole slide.")
		else:
			self.load_annotations(annotation_path)
		try:
			self.slide = ops.OpenSlide(path)
		except ops.lowlevel.OpenSlideUnsupportedFormatError:
			self.print(f"Unable to read SVS file from {path} , skipping")
			self.shape = None
			return None
		self.shape = self.slide.dimensions
		self.filter_dimensions = self.slide.level_dimensions[-1]
		self.filter_magnification = self.filter_dimensions[0] / self.shape[0]
		self.thumb = self.slide.get_thumbnail((4096, 4096))
		self.thumb_file = join('/'.join(path.split('/')[:-1]), f'{self.name}_thumb.jpg')
		imageio.imwrite(self.thumb_file, self.thumb)
		self.MPP = float(self.slide.properties[ops.PROPERTY_NAME_MPP_X])
		self.print(f" * [{self.name}] Microns per pixel: {self.MPP}")
		self.print(f" * [{self.name}] Loaded SVS of size {self.shape[0]} x {self.shape[1]}")

	def loaded_correctly(self):
		return bool(self.shape)

	def build_generator(self, size_px, size_um, stride_div, case_name, export=False, augment=False):
		# Calculate window sizes, strides, and coordinates for windows
		tiles_path = join(self.export_folder, "tiles", case_name)
		if not os.path.exists(tiles_path): os.makedirs(tiles_path)
		# Calculate pixel size of extraction window
		extract_px = int(size_um / self.MPP)
		stride = int(extract_px / stride_div)
		self.print(f" * [{self.name}] Extracting tiles of pixel size {extract_px}px for a size of {size_um}um")
		if size_px > extract_px:
			self.print(f"WARNING: Tiles will be up-scaled with cubic interpolation ({extract_px}px -> {size_px}px)")
		coord = []
		slide_x_size = self.shape[0] - extract_px
		slide_y_size = self.shape[1] - extract_px

		for y in range(0, (self.shape[1]+1) - extract_px, stride):
			for x in range(0, (self.shape[0]+1) - extract_px, stride):
				# Check if this is a unique tile without overlap (e.g. if stride was 1)
				is_unique = ((y % extract_px == 0) and (x % extract_px == 0))
				coord.append([x, y, is_unique])

		# Load annotations as shapely.geometry objects
		annPolys = [sg.Polygon(annotation.coordinates) for annotation in self.annotations]
		# Create mask for indicating whether tile was extracted
		tile_mask = np.asarray([0 for i in range(len(coord))])
		self.tile_mask = None
		self.p_id = None if not self.pb else self.pb.add_bar(0, len(coord), endtext=case_name)

		def generator():
			for ci in range(len(coord)):
				if self.pb:
					self.pb.update(self.p_id, ci)
				c = coord[ci]
				c_f = np.multiply(c, self.filter_magnification) # Filter coordinates
				filter_px = int(extract_px * self.filter_magnification)

				# Check if the center of the current window lies within any annotation; if not, skip
				if bool(annPolys) and not any([annPoly.contains(sg.Point(int(c[0]+extract_px/2), int(c[1]+extract_px/2))) for annPoly in annPolys]):
					continue

				# Read the low-mag level for filter checking
				filter_region = np.asarray(self.slide.read_region(c, self.slide.level_count-1, [filter_px, filter_px]))[:,:,:-1]
				median_brightness = int(sum(np.median(filter_region, axis=(0, 1))))
				if median_brightness > 660:
					# Discard tile; median brightness (average RGB pixel) > 220
					continue

				# Read the full-res region and discard the alpha pixels
				region = np.asarray(self.slide.read_region(c, 0, [extract_px, extract_px]))[:,:,:-1]
				region = cv2.resize(region, dsize=(size_px, size_px), interpolation=cv2.INTER_CUBIC)
				tile_mask[ci] = 1
				coord_label = ci
				unique_tile = c[2]
				if export and unique_tile:
					imageio.imwrite(join(tiles_path, f'{case_name}_{ci}.jpg'), region)
					if augment:
						imageio.imwrite(join(tiles_path, f'{case_name}_{ci}_aug1.jpg'), np.rot90(region))
						imageio.imwrite(join(tiles_path, f'{case_name}_{ci}_aug2.jpg'), np.flipud(region))
						imageio.imwrite(join(tiles_path, f'{case_name}_{ci}_aug3.jpg'), np.flipud(np.rot90(region)))
						imageio.imwrite(join(tiles_path, f'{case_name}_{ci}_aug4.jpg'), np.fliplr(region))
						imageio.imwrite(join(tiles_path, f'{case_name}_{ci}_aug5.jpg'), np.fliplr(np.rot90(region)))
						imageio.imwrite(join(tiles_path, f'{case_name}_{ci}_aug6.jpg'), np.flipud(np.fliplr(region)))
						imageio.imwrite(join(tiles_path, f'{case_name}_{ci}_aug7.jpg'), np.flipud(np.fliplr(np.rot90(region))))
				yield region, coord_label, unique_tile
			if self.pb: 
				self.pb.end(self.p_id)
				self.print(f" * [{self.name}] Size of coord: {len(coord)} and total exported: {sum(tile_mask)}")
			self.tile_mask = tile_mask

		return generator, slide_x_size, slide_y_size, stride

	def load_annotations(self, path):
		with open(path, "r") as csvfile:
			reader = csv.reader(csvfile, delimiter=';')
			for row in reader:
				if row[0] == 'X_thumb':
					# New object detected
					self.annotations.append(AnnotationObject(f"Object{len(self.annotations)}"))
					continue
				x_coord = int(float(row[2]))
				y_coord = int(float(row[3]))
				self.annotations[-1].add_coord((x_coord, y_coord))
			self.print(f" * [{self.name}] Total annotation objects detected: {len(self.annotations)}")

class Convoluter:
	def __init__(self, size_px, size_um, num_classes, batch_size, use_fp16, save_folder='', augment=False):
		self.SVS = {}
		self.CATEGORIES = {}
		self.MODEL_DIR = None
		self.PKL_DICT = {}
		self.SIZE_PX = size_px
		self.SIZE_UM = size_um
		self.NUM_CLASSES = num_classes
		self.BATCH_SIZE = batch_size
		self.DTYPE = tf.float16 if use_fp16 else tf.float32
		self.DTYPE_INT = tf.int16 if use_fp16 else tf.int32
		self.SAVE_FOLDER = save_folder
		self.STRIDE_DIV = 4
		self.MODEL_DIR = None
		self.AUGMENT = augment

	def load_svs(self, whole_svs_array, directory, category=None):
		for svs in whole_svs_array:
			name = svs[:-4]
			svs_path = svs if not directory else join(directory, svs)
			self.SVS.update({name: svs_path})
			if category:
				self.CATEGORIES.update({name: category})

	def load_pkl(self, pkl_array, directory):
		for pkl in pkl_array:
			self.PKL_DICT.update({pkl[:-4]: join(directory, pkl)})

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
			self.STRIDE_DIV = 1

		if export_tiles and not (display_heatmaps or save_final_layer or save_heatmaps):
			print("Exporting tiles only, no new calculations or heatmaps will be generated.")
			pb = progress_bar.ProgressBar()

			def export_tiles(case_name):
				svs_path = self.SVS[case_name]
				category = "None" if case_name not in self.CATEGORIES else self.CATEGORIES[case_name]
				pb.print(f"Working on case {case_name} ({category})")
				self.export_tiles(svs_path, case_name, pb)

			pool = ThreadPool(NUM_THREADS)
			pool.map(export_tiles, self.SVS)
		else:
			for case_name in self.SVS:
				svs_path = self.SVS[case_name]
				category = "None" if case_name not in self.CATEGORIES else self.CATEGORIES[case_name]
				print(f"Working on case {case_name} ({category})")

				# Use PKL logits if available
				if case_name in self.PKL_DICT and not save_final_layer:
					with open(self.PKL_DICT[case_name], 'rb') as handle:
						logits = pickle.load(handle)
				# Otherwise recalculate
				else:
					logits, final_layer, final_layer_labels, logits_flat = self.scan_image(svs_path, case_name, 
																			export_tiles, save_final_layer,
																			save_pkl=((save_heatmaps or display_heatmaps) and case_name not in self.PKL_DICT))
				if save_heatmaps:
					self.export_heatmaps(svs_path, logits, self.SIZE_PX, case_name)
				if save_final_layer:
					self.save_csv(final_layer, final_layer_labels, logits_flat, case_name, category)
				if display_heatmaps:
					self.fast_display(svs_path, logits, self.SIZE_PX, case_name)

	def export_tiles(self, svs_path, case_name, pb = None):
		whole_svs = SVSReader(svs_path, self.SAVE_FOLDER, pb=pb)
		if not whole_svs.loaded_correctly(): return
		gen_slice, _, _, _ = whole_svs.build_generator(self.SIZE_PX, self.SIZE_UM, self.STRIDE_DIV, case_name, 
															  export=True, 
															  augment=self.AUGMENT)
		for tile, coord, unique in gen_slice():
			pass

	def scan_image(self, svs_path, case_name, export_tiles=False, final_layer=False, save_pkl=True):
		'''Returns logits and final layer weights'''
		warnings.simplefilter('ignore', Image.DecompressionBombWarning)

		# Reset graph to prevent OOM errors when convoluting across multiple images
		tf.reset_default_graph()

		# Load whole-slide-image into Numpy array and prepare pkl output
		whole_svs = SVSReader(svs_path, self.SAVE_FOLDER)
		pkl_name =  case_name + '.pkl'

		# load SVS generator
		gen_slice, x_size, y_size, stride_px = whole_svs.build_generator(self.SIZE_PX, self.SIZE_UM, self.STRIDE_DIV, case_name, 
																		 export=export_tiles)

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
				padded_batch = tf.concat([image_patches, tf.zeros([self.BATCH_SIZE - tf.shape(image_patches)[0], self.SIZE_PX, self.SIZE_PX, 3], # image_patches instead of next_batch
															dtype=self.DTYPE)], 0)
				padded_batch.set_shape([self.BATCH_SIZE, self.SIZE_PX, self.SIZE_PX, 3])

			with arg_scope(inception_arg_scope()):
				_, end_points = inception_v4.inception_v4(padded_batch, num_classes=self.NUM_CLASSES, is_training=False, create_aux_logits=False)

			prelogits = end_points['PreLogitsFlatten']
			slogits = end_points['Predictions']
			vars_to_restore = []
			for var_to_restore in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
				if ((var_to_restore.name[12:21] != "AuxLogits")):# and 
					#((var_to_restore.name[:25] != "InceptionV4/Logits/Logits") or not final_layer)):
					vars_to_restore.append(var_to_restore)

			saver = tf.train.Saver(vars_to_restore)

			with tf.Session() as sess:
				init = (tf.global_variables_initializer(), tf.local_variables_initializer())
				sess.run(init)

				ckpt = tf.train.get_checkpoint_state(self.MODEL_DIR)
				if ckpt and ckpt.model_checkpoint_path:
					print(" + Restoring saved checkpoint model.")
					saver.restore(sess, ckpt.model_checkpoint_path)
				else:
					raise Exception('Unable to find checkpoint file.')

				logits_arr = []
				labels_arr = []
				x_logits_len = int(x_size / stride_px) + 1
				y_logits_len = int(y_size / stride_px) + 1
				total_logits_count = x_logits_len * y_logits_len	

				count = 0
				prelogits_arr = [] # Final layer weights 
				logits_arr = []	# Logits (predictions) 
				unique_arr = []	# Boolean array indicating whether tile is unique (non-overlapping) 

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
							#print(new_prelogits[:,0].tolist())
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
			
			flat_unique_logits = None
			if final_layer:
				prelogits_arr = prelogits_arr[0:total_logits_count]
				prelogits_arr = prelogits_arr[sorted_indices]
				unique_arr = unique_arr[0:total_logits_count]
				unique_arr = unique_arr[sorted_indices]
				# Find logits from non-overlapping tiles (will be used for metadata for saved final layer weights CSV)
				flat_unique_logits = [logits_arr[l] for l in range(len(logits_arr)) if unique_arr[l]]

			# Organize array into 2D format corresponding to where each logit was calculated
			if final_layer:
				prelogits_out = [prelogits_arr[p] for p in range(len(prelogits_arr)) if unique_arr[p]]
				prelogits_labels = [l for l in range(len(unique_arr)) if unique_arr[l]]
			else:
				prelogits_out = None
				prelogits_labels = None

			# Next, expand the logits back to a full 2D map spanning the whole SVS file,
			#  supplying values of "-1" where tiles were skipped
			expanded_logits = [[-1] * self.NUM_CLASSES] * len(whole_svs.tile_mask)
			li = 0
			for i in range(len(expanded_logits)):
				if whole_svs.tile_mask[i] == 1:
					expanded_logits[i] = logits_arr[li]
					li += 1

			# Resize logits array into a two-dimensional array for heatmap display
			# Previously, this was using logits_arr instead of expanded_logits
			expanded_logits = np.asarray(expanded_logits)
			print(f" * Expanded_logits size: {expanded_logits.shape}; resizing to y:{y_logits_len} and x:{x_logits_len}")
			logits_out = np.resize(expanded_logits, [y_logits_len, x_logits_len, self.NUM_CLASSES])
			if save_pkl:
				with open(os.path.join(self.SAVE_FOLDER, pkl_name), 'wb') as handle:
					pickle.dump(logits_out, handle)

			return logits_out, prelogits_out, prelogits_labels, flat_unique_logits

	def save_csv(self, output, labels, logits, name, category):
		print("Writing csv...")
		csv_started = os.path.exists(join(self.SAVE_FOLDER, 'final_layer_weights.csv'))
		write_mode = 'a' if csv_started else 'w'
		with open(join(self.SAVE_FOLDER, 'final_layer_weights.csv'), write_mode) as csv_file:
			csv_writer = csv.writer(csv_file, delimiter = ',')
			if not csv_started:
				#csv_writer.writerow(["Tile_num", "Case", "Category"] + [f"Node{n}" for n in range(len(output[0]))])
				csv_writer.writerow(["Tile_num", "Case", "Category"] + [f"Logits{l}" for l in range(len(logits[0]))] + [f"Node{n}" for n in range(len(output[0]))])
			for l in range(len(output)):
				out = output[l].tolist()
				logit = logits[l].tolist()
				csv_writer.writerow([labels[l], name, category] + logit + out)

	def export_heatmaps(self, image_file, logits, size, name):
		'''Displays logits calculated using scan_image as a heatmap overlay.'''
		print(f"Loading image and assembling heatmaps for image {image_file}...")

		fig = plt.figure(figsize=(18, 16))
		ax = fig.add_subplot(111)
		fig.subplots_adjust(bottom = 0.25, top=0.95)

		if image_file[-4:] == ".svs":
			whole_svs = SVSReader(image_file, self.SAVE_FOLDER)
			im = whole_svs.thumb #plt.imread(whole_svs.thumb)
		else:
			im = plt.imread(image_file)
		implot = ax.imshow(im, zorder=0)
		gca = plt.gca()
		gca.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

		# Calculations to determine appropriate offset for heatmap
		im_extent = implot.get_extent()
		#extent = [im_extent[0] + size/2, im_extent[1] - size/2, im_extent[2] - size/2, im_extent[3] + size/2]
		extent = im_extent

		# Define color map
		jetMap = np.linspace(0.45, 0.95, 255)
		cmMap = cm.nipy_spectral(jetMap)
		newMap = mcol.ListedColormap(cmMap)

		heatmap_dict = {}

		# Make heatmaps and sliders
		for i in range(self.NUM_CLASSES):
			heatmap = ax.imshow(logits[:, :, i], extent=extent, cmap=newMap, alpha = 0.0, vmin = 0.0, vmax = 1.0, interpolation='none', zorder=10) #bicubic
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

		if image_file[-4:] == ".svs":
			whole_svs = SVSReader(image_file, self.SAVE_FOLDER)
			buf = plt.imread(whole_svs.thumb_file) #plt.imread(whole_svs.thumb)
		else:
			buf = plt.imread(image_file)

		implot = FastImshow(buf, ax, extent=None, tgt_res=1024)
		gca = plt.gca()
		gca.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

		# Calculations to determine appropriate offset for heatmap
		im_extent = implot.extent
		extent = im_extent #[im_extent[0] + size/2, im_extent[1] - size/2, im_extent[2] + size/2, im_extent[3] - size/2]

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
			heatmap = ax.imshow(logits[:, :, i], extent=extent, cmap=newMap, alpha = 0.0,  vmin = 0.0, vmax = 1.0, interpolation='none', zorder=10) #bicubic
			slider = Slider(ax_slider, f'Class {i}', 0, 1, valinit = 0)
			heatmap_dict.update({f"Class{i}": [heatmap, slider]})
			slider.on_changed(slider_func)

		fig.canvas.set_window_title(name)
		implot.show()
		plt.show()

def get_args():
	parser = argparse.ArgumentParser(description = 'Convolutionally applies a saved Tensorflow model to a larger image, displaying the result as a heatmap overlay.')
	parser.add_argument('-m', '--model', help='Path to Tensorflow model directory containing stored checkpoint.')
	parser.add_argument('-i', '--svs', help='Path to whole-slide image (SVS format) or folder of images (SVS) to analyze.')
	parser.add_argument('-p', '--pkl', help='Python Pickle file, or folder of pkl files, containing pre-calculated weights to load. If both a PKL file and model are supplied, will default to using PKL file.')
	parser.add_argument('-o', '--out', help='Path to directory in which exported images and data will be saved.')
	parser.add_argument('-c', '--classes', type=int, default = 1, help='Number of unique output classes contained in the model.')
	parser.add_argument('-b', '--batch', type=int, default = 64, help='Batch size for which to run the analysis.')
	parser.add_argument('--px', type=int, default=512, help='Size of image patches to analyze, in pixels.')
	parser.add_argument('--um', type=float, default=127.6928, help='Size of image patches to analyze, in microns.')
	parser.add_argument('--fp16', action="store_true", help='Use Float16 operators (half-precision) instead of Float32.')
	parser.add_argument('--save', action="store_true", help='Save heatmaps to PNG file instead of displaying.')
	parser.add_argument('--final', action="store_true", help='Calculate and export image tiles and final layer weights.')
	parser.add_argument('--display', action="store_true", help='Display results with interactive heatmap for each whole-slide image.')
	parser.add_argument('--export', action="store_true", help='Save extracted image tiles.')
	parser.add_argument('--augment', action="store_true", help='Augment extracted tiles with flipping/rotating.')
	return parser.parse_args()

if __name__==('__main__'):
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	tf.logging.set_verbosity(tf.logging.ERROR)
	args = get_args()

	c = Convoluter(args.px, args.um, args.classes, args.batch, args.fp16, args.out, args.augment)

	if not args.pkl: args.pkl = args.out

	if isfile(args.svs):
		image_list = [args.svs.split('/')[-1]]
		image_dir = "/".join(args.svs.split('/')[:-1])
		c.load_svs(image_list, image_dir)
	else:
		# First, load images in the directory, not assigning any category
		image_list = [i for i in os.listdir(args.svs) if (isfile(join(args.svs, i)) and (i[-3:] == "svs"))]	
		c.load_svs(image_list, args.svs)
		# Next, load images in subdirectories, assigning category by subdirectory name
		dir_list = [d for d in os.listdir(args.svs) if not isfile(join(args.svs, d))]
		for directory in dir_list:
			image_list = [i for i in os.listdir(join(args.svs, directory)) if (isfile(join(args.svs, directory, i)) and (i[-3:] == "svs"))]	
			c.load_svs(image_list, join(args.svs, directory), category=directory)
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