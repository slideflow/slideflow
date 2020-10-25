import sys
import os
import csv
import pickle
import time
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.colors as mcol
import seaborn as sns
import tensorflow as tf
import scipy.stats as stats
import slideflow.util as sfutil
import slideflow.io as sfio
import shapely.geometry as sg

from io import StringIO
from slideflow.util import log, ProgressBar, TCGA, StainNormalizer
from slideflow.util.fastim import FastImshow
from slideflow.mosaic import Mosaic
from slideflow.model import ModelActivationsInterface
from os.path import join, isfile, exists
from random import sample
from statistics import mean
from math import isnan
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from functools import partial
from multiprocessing.dummy import Process as DProcess
from multiprocessing.dummy import Pool as DPool
from sklearn.neighbors import NearestNeighbors
from PIL import Image

# TODO: add check that cached PKL corresponds to current and correct model & slides

def create_bool_mask(x, y, w, sx, sy):
	l = max(0,  int(x-(w/2.)))
	r = min(sx, int(x+(w/2.)))
	t = max(0,  int(y-(w/2.)))
	b = min(sy, int(y+(w/2.)))
	m = np.array([[[True]*3]*sx]*sy)
	for yi in range(m.shape[1]):
		for xi in range(m.shape[0]):
			if (t < yi < b) and (l < xi < r):
				m[yi][xi] = [False, False, False]
	return m

class ActivationsError(Exception):
	pass

class ActivationsVisualizer:
	'''Loads annotations, saved layer activations, and prepares output saving directories.
		Will also read/write processed activations to a PKL cache file to save time in future iterations.'''
	
	def __init__(self, model, tfrecords, root_dir, image_size, annotations=None, outcome_header=None, 
					focus_nodes=[], use_fp16=True, normalizer=None, normalizer_source=None, 
					use_activations_cache=False, activations_cache='default', batch_size=32,
					activations_export=None, max_tiles_per_slide=100, manifest=None, model_format=None):
		'''Object initializer.

		Args:
			model:					Path to model from which to calculate activations
			tfrecords:				List of tfrecords paths
			root_dir:				Root directory in which to save cache files and output files
			image_size:				Int, width/height of input images in pixels
			annotations:			Path to CSV file containing slide annotations
			outcome_header:			String, name of outcome header in annotations file, used to compare activations between categories
			focus_nodes:			List of int, nodes on which to focus when generating cross-category statistics
			use_fp16:				Bool, whether to use FP16 (rather than FP32)
			normalizer:				String, which real-time normalization to use on images taken from TFRecords
			noramlizer_source:		String, path to image to use as source for real-time normalization
			use_activations_cache:	Bool, if true, will store activations in a PKL cache file for rapid re-use
			activations_cache:		File in which to store activations PKL cache
			batch_size:				Batch size to use during activations calculations
			activations_export:		Filename for CSV export of activations
			max_tiles_per_slide:	Maximum number of tiles from which to generate activations for each slide
			manifest:				Optional, dict mapping tfrecords to number of tiles contained. Used for progress bars.
		'''
		self.categories = []
		self.used_categories = []
		self.slide_category_dict = {}
		self.slide_node_dict = {}
		self.slide_logits_dict = {}
		self.umap = None
		
		self.focus_nodes = focus_nodes
		self.manifest = manifest
		self.MAX_TILES_PER_SLIDE = max_tiles_per_slide
		self.IMAGE_SIZE = image_size
		self.tfrecords = np.array(tfrecords)
		self.slides = sorted([sfutil.path_to_name(tfr) for tfr in self.tfrecords])

		self.STATS_CSV_FILE = join(root_dir, "stats", "slide_level_summary.csv")
		self.STATS_ROOT = join(root_dir, "stats")
		self.ACTIVATIONS_CACHE = join(root_dir, "stats", "activations_cache.pkl") if activations_cache=='default' else join(root_dir, 'stats', activations_cache)
		if not exists(join(root_dir, "stats")):
			os.makedirs(join(root_dir, "stats"))

		# Load annotations if provided
		if annotations and outcome_header:
			self.load_annotations(annotations, outcome_header)

		# Load activations
		# Load from PKL (cache) if present
		if self.ACTIVATIONS_CACHE and exists(self.ACTIVATIONS_CACHE): 
			# Load saved PKL cache
			log.empty(f"Loading pre-calculated predictions and activations from {self.ACTIVATIONS_CACHE}...", 1)
			with open(self.ACTIVATIONS_CACHE, 'rb') as pt_pkl_file:
				
				self.slide_node_dict, self.slide_logits_dict = pickle.load(pt_pkl_file)
				first_slide = list(self.slide_node_dict.keys())[0]
				logits = 	  list(self.slide_logits_dict[first_slide].keys())
				self.nodes =  list(self.slide_node_dict[first_slide].keys())
				if max_tiles_per_slide:
					log.info(f"Filtering activations to maximum {max_tiles_per_slide} tiles per slide", 1)
					for slide in self.slide_node_dict.keys():
						for n in self.nodes:
							if len(self.slide_node_dict[slide][n]) > max_tiles_per_slide:
								self.slide_node_dict[slide][n] = self.slide_node_dict[slide][n][:max_tiles_per_slide]
						for l in logits:
							if len(self.slide_logits_dict[slide][l]) > max_tiles_per_slide:
								self.slide_logits_dict[slide][l] = self.slide_logits_dict[slide][l][:max_tiles_per_slide]

		# Otherwise will need to generate new activations from a given model
		else:
			self.generate_activations_from_model(model, tfrecords, model_format=model_format, 
																   use_fp16=use_fp16,
																   batch_size=batch_size,
																   export=activations_export,
																   normalizer=normalizer,
																   normalizer_source=normalizer_source)
			self.nodes = list(self.slide_node_dict[list(self.slide_node_dict.keys())[0]].keys())

		# Now delete slides not included in our filtered TFRecord list
		loaded_slides = list(self.slide_node_dict.keys())
		for loaded_slide in loaded_slides:
			if loaded_slide not in self.slides:
				del self.slide_node_dict[loaded_slide]
				del self.slide_logits_dict[loaded_slide]

		# Now screen for missing slides in activations
		missing_slides = []
		for slide in self.slides:
			if slide not in self.slide_node_dict:
				missing_slides += [slide]
			elif self.slide_node_dict[slide][0] == []:
				missing_slides += [slide]
		log.info(f"Loaded activations from {(len(self.slides)-len(missing_slides))}/{len(self.slides)} slides ({len(missing_slides)} missing)", 2)
		if len(missing_slides):
			missing_tfrecords = [tfr for tfr in self.tfrecords if sfutil.path_to_name(tfr) in missing_slides]
			log.empty(f"Generating activations for {len(missing_slides)} missing slides...", 1)
			self.generate_activations_from_model(model, missing_tfrecords, use_fp16=use_fp16, batch_size=batch_size, export=activations_export, normalizer=normalizer, normalizer_source=normalizer_source)
		
		# Record which categories have been included in the specified tfrecords
		if self.categories:
			self.used_categories = list(set([self.slide_category_dict[slide] for slide in self.slides]))
			self.used_categories.sort()
		log.info(f"Observed categories (total: {len(self.used_categories)}):", 2)
		for c in self.used_categories:
			log.empty(f"\t{c}", 2)

	def _save_node_statistics_to_csv(self, sorted_nodes, slide_node_dict, filename, tile_stats=None, slide_stats=None):
		'''Internal function to exports statistics (ANOVA p-values and slide-level averages) to CSV.
		
		Args:
			sorted_nodes:		List of node IDs (int) sorted in order of significance.
			slide_node_dict:	Dict mapping slides to dict of nodes mapping to slide-level values as calculated elsewhere.
									Slide-level node values could be mean, median, thresholded, or other.
			filename:			Filename
			tile_stats:			Dictionary mapping nodes to a dict of tile-level stats containing 'p' (ANOVA P-value) and 'f' (ANOVA F-value) for each node
									As calculated elsewhere by comparing node activations between categories
			slide_stats:		Dictionary mapping nodes to a dict of slide-level stats containing 'p' (ANOVA P-value), 'f' (ANOVA F-value),
									and 'num_above_threshold' for each node
									As calculated elsewhere by comparing node activations between categories
		'''
		# Save results to CSV
		log.empty(f"Writing results to {sfutil.green(filename)}...", 1)
		with open(filename, 'w') as outfile:
			csv_writer = csv.writer(outfile)
			header = ['slide', 'category'] + [f"FLNode{n}" for n in sorted_nodes]
			csv_writer.writerow(header)
			for slide in self.slides:
				category = self.slide_category_dict[slide]
				row = [slide, category] + [slide_node_dict[slide][n] for n in sorted_nodes]
				csv_writer.writerow(row)
			if tile_stats:
				csv_writer.writerow(['Tile-level statistic', 'ANOVA P-value'] + [tile_stats[n]['p'] for n in sorted_nodes])
				csv_writer.writerow(['Tile-level statistic', 'ANOVA F-value'] + [tile_stats[n]['f'] for n in sorted_nodes])
			if slide_stats:
				csv_writer.writerow(['Slide-level statistic', 'ANOVA P-value'] + [slide_stats[n]['p'] for n in sorted_nodes])
				csv_writer.writerow(['Slide-level statistic', 'ANOVA F-value'] + [slide_stats[n]['f'] for n in sorted_nodes])
				#csv_writer.writerow(['Slide-level statistic', 'Num above threshold'] + [slide_stats[n]['num_above_thresh'] for n in sorted_nodes])

	def slide_tile_dict(self):
		'''Generates dictionary mapping slides to list of node activations for each tile.

		Example (3 nodes):
			{ 'Slide1': [[0.1, 0.2, 0.4], # Slide1, node activations for tile1
						 [0.5, 0.1, 0.7], # Slide1, node activations for tile2
						 [0.6, 0.9, 0.1]] # Slide1, node activations for tile3
			}
		'''
		result = {}
		for slide in self.slides:
			num_tiles = len(self.slide_node_dict[slide][0])
			result.update({slide: [[self.slide_node_dict[slide][node][tile_index] for node in self.nodes] for tile_index in range(num_tiles)]})
		return result

	def map_to_whitespace(self, whitespace_threshold=230, num_threads=16):

		def _parse_function(record):
			features = tf.io.parse_single_example(record, sfio.tfrecords.FEATURE_DESCRIPTION)
			slide = features['slide']
			image_string = features['image_raw']
			raw_image = tf.image.decode_jpeg(image_string, channels=3)
			return raw_image, slide

		def map_to_tfrecord(tfrecord, pb=None):
			umap_x, umap_y, umap_meta = [], [], []
			dataset = tf.data.TFRecordDataset(tfrecord)
			dataset = dataset.map(_parse_function, num_parallel_calls=8)
			for i, data in enumerate(dataset):
				if self.MAX_TILES_PER_SLIDE and i >= self.MAX_TILES_PER_SLIDE: break
				image, slide = data
				if pb: pb.increase_bar_value()
				#fraction = (np.mean(image.numpy(), axis=2) > whitespace_threshold).sum() / (self.IMAGE_SIZE**2)
				#avg = np.mean(image)
				#umap_x += [fraction]
				#umap_y += [avg]
				grayspace_threshold = 0.05
				hsv_image = mcol.rgb_to_hsv(image.numpy())
				fraction = (hsv_image[:,:,1] < grayspace_threshold).sum() / (self.IMAGE_SIZE**2)
				hue = np.mean(hsv_image[:,:,0])
				umap_x += [fraction]
				umap_y += [hue]

				umap_meta += [{
					'slide': slide.numpy().decode('utf-8'),
					'index': i
				}]
			return np.array(umap_x), np.array(umap_y), np.array(umap_meta)

		total_tiles = 0
		if self.manifest:
			try:
				total_tiles = sum([min(self.manifest[tfrecord]['total'], self.MAX_TILES_PER_SLIDE) if self.MAX_TILES_PER_SLIDE else self.manifest[tfrecord]['total'] for tfrecord in self.tfrecords])
			except:
				pass
		
		pb = ProgressBar(total_tiles, counter_text='tiles', leadtext="Calculating whitespace... ", show_counter=True, show_eta=True) if total_tiles else None
		
		pool = DPool(num_threads)
		result = pool.map(partial(map_to_tfrecord, pb=pb), self.tfrecords)
		pool.close()
		print("\r\033[K", end="")
		log.empty("Finished whitespace calculations", 1)
		umap_x = np.concatenate([r[0] for r in result])
		umap_y = np.concatenate([r[1] for r in result])
		umap_meta = np.concatenate([r[2] for r in result]).tolist()
		return umap_x, umap_y, umap_meta

	def map_to_predictions(self, x=0, y=0):
		'''Returns coordinates and metadata for tile-level predictions for all tiles,
		which can be used to create a TFRecordMap.
		
		Args:
			x:			Int, identifies the outcome category id for which predictions will be mapped to the X-axis
			y:			Int, identifies the outcome category id for which predictions will be mapped to the Y-axis
		
		Returns:
			mapped_x:	List of x-axis coordinates (predictions for the category 'x')
			mapped_y:	List of y-axis coordinates (predictions for the category 'y')
			umap_meta:	List of dictionaries containing tile-level metadata (used for TFRecordMap)
		'''
		umap_x, umap_y, umap_meta = [], [], []
		for slide in self.slides:
			for tile_index in range(len(self.slide_logits_dict[slide][0])):
				umap_x += [self.slide_logits_dict[slide][x][tile_index]]
				umap_y += [self.slide_logits_dict[slide][y][tile_index]]
				umap_meta += [{
					'slide': slide,
					'index': tile_index
				}]
		return np.array(umap_x), np.array(umap_y), umap_meta

	def get_activations(self):
		'''Returns dictionary mapping slides to tile-level node activations.

		Example (3 nodes):
			{ 'Slide1': [[0.1, 0.5, 0.6], # Slide1, node1 activations for all tiles
						 [0.2, 0.1, 0.7], # Slide1, node2 activations for all tiles
						 [0.4, 0.7, 0.1]] # Slide1, node3 activations for all tiles
			}
		'''
		return self.slide_node_dict

	def get_predictions(self):
		'''Returns dictionary mapping slides to tile-level logit predictions.

		Example (2 outcome categories):
			{ 'Slide1': [[0.1, 0.9, 0.6], # Slide1, logit predictions for category 1 for all tiles
						 [0.9, 0.1, 0.4], # Slide1, logit predictions for category 2 for all tiles
			}
		'''
		return self.slide_logits_dict

	def get_slide_level_linear_predictions(self):
		'''Returns slide-level predictions assuming the model is predicting a linear outcome.

		Returns:
			dict:		Dictionary mapping slide names to final slide-level predictions
							for each outcome cateogry, calculated as the average predicted value
							in the outcome category for all tiles in the slide.
							Example:
								{ 'slide1': {
									0: 0.24,	# Outcome category 0
									1: 0.15,	# Outcome category 1
									2: 0.61 }}	# Outcome category 2
		'''
		first_slide = list(self.slide_logits_dict.keys())[0]
		outcomes = sorted(list(self.slide_logits_dict[first_slide].keys()))
		slide_predictions = {slide: {o: mean(self.slide_logits_dict[slide][o]) for o in outcomes}
																			   for slide in self.slide_logits_dict}
		return slide_predictions

	def get_slide_level_categorical_predictions(self, prediction_filter=None):
		'''Returns slide-level predictions assuming the model is predicting a categorical outcome.

		Args:
			prediction_filter:	(optional) List of int. If provided, will restrict predictions to only these
									categories, with final prediction being based based on highest logit
									among these categories.

		Returns:
			slide_predictions:	Dictionary mapping slide names to final slide-level predictions.
			slide_percentages:	This is a dictionary mapping slide names to a dictionary for each category,
									which maps the category id to the percent of tiles in the slide
									predicted to be this category.
									Example:
										{ 'slide1': {
											0: 0.24,
											1: 0.15,
											2: 0.61 }}
								If linear model, this is the same as slide_predictions.
		'''
		slide_predictions = {}
		slide_percentages = {}
		first_slide = list(self.slide_logits_dict.keys())[0]
		outcomes = sorted(list(self.slide_logits_dict[first_slide].keys()))
		for slide in self.slide_logits_dict:
			num_tiles = len(self.slide_logits_dict[slide][0])
			tile_predictions = []
			for i in range(num_tiles):
				calculated_logits = [self.slide_logits_dict[slide][o][i] for o in outcomes]
				if prediction_filter:
					filtered_calculated_logits = [calculated_logits[o] for o in prediction_filter]
				else:
					filtered_calculated_logits = calculated_logits
				tile_predictions += [calculated_logits.index(max(filtered_calculated_logits))]
			slide_prediction_values = {o: (tile_predictions.count(o)/len(tile_predictions)) for o in outcomes}
			slide_percentages.update({slide: slide_prediction_values})
			slide_predictions.update({slide: max(slide_prediction_values, key=lambda l: slide_prediction_values[l])})
		return slide_predictions, slide_percentages

	def load_annotations(self, annotations, outcome_header):
		'''Loads annotations from a given file with the specified outcome header.
		
		Args:
			annotations:		Path to CSV annotations file.
			outcome_header:		String, name of column header from which to read outcome variables.
		'''
		with open(annotations, 'r') as ann_file:
			log.info("Reading annotations...", 1)
			ann_reader = csv.reader(ann_file)
			header = next(ann_reader)
			slide_i = header.index(TCGA.slide)
			category_i = header.index(outcome_header)
			for row in ann_reader:
				slide = row[slide_i]
				category = row[category_i]
				if slide not in self.slides: continue
				self.slide_category_dict.update({slide:category})

		self.categories = list(set(self.slide_category_dict.values()))

		if self.slide_node_dict:
			# If initial loading has been completed already, make note of observed categories in given header
			for slide in self.slides:
				try:
					if self.slide_node_dict[slide][0] != []:
						self.used_categories = list(set(self.used_categories + [self.slide_category_dict[slide]]))
						self.used_categories.sort()
				except KeyError:
					# Skip unknown slide
					pass
			log.info(f"Observed categories (total: {len(self.used_categories)}):", 1)
			for c in self.used_categories:
				log.empty(f"\t{c}", 2)

	def get_tile_node_activations_by_category(self, node):
		'''For each outcome category, calculates activations of a given node across all tiles in the category.
		Requires annotations to have been loaded with load_annotations()

		Args:
			node:		Int, id of node.
		
		Returns:
			List of node activations separated by category.
				Example:
				[[0.1, 0.2, 0.7, 0.1, 0.0], # Activations for node "N" across all tiles from slides in category 1
				 [0.8, 0.2, 0.1]] 			# Activations for node "N" across all tiles from slides in category 2
		'''
		if not self.categories: 
			log.warn("Unable to calculate node activations by category; annotations not loaded. Please load with load_annotations()")
			return
		tile_node_activations_by_category = []
		for c in self.used_categories:
			nodelist = [self.slide_node_dict[pt][node] for pt in self.slides if self.slide_category_dict[pt] == c]
			tile_node_activations_by_category += [[nodeval for nl in nodelist for nodeval in nl]]
		return tile_node_activations_by_category

	def get_top_nodes_by_slide(self):
		'''First, slide-level average node activations are calculated for all slides. 
			Then, the significance of the difference in average node activations between for slides
			belonging to the different outcome categories is calculated using ANOVA.
			This function then returns a list of all nodes, sorted by ANOVA p-value (most significant first).
		'''
		# First ensure basic stats have been calculated
		if not hasattr(self, 'sorted_nodes'):
			self.calculate_activation_averages_and_stats()
		
		return self.sorted_nodes

	def get_top_nodes_by_tile(self):
		'''First, tile-level average node activations are calculated for all tiles.
			Then, the significance of the difference in node activations for tiles
			belonging to the different outcome categories is calculated using ANOVA.
			This function then returns a list of all nodes, sorted by ANOVA p-value (most significant first).
		'''
		# First ensure basic stats have been calculated
		if not hasattr(self, 'sorted_nodes'):
			self.calculate_activation_averages_and_stats()
		
		return self.nodes

	def find_neighbors(self, neighbor_AV, neighbor_slides, n_neighbors, algorithm='ball_tree'):
		'''Finds neighboring tiles for a given ActivationsVisualizer and list of slides.

		Args:
			neighbor_AV:		ActivationsVisualizer, will be used to look for neighbors
			neighbor_slides:	Either a single slide name or a list of slide names. Corresponds to slides in the provided neighboring AV.
									Will look for neighbors to all tiles in these slides.
			n_neighbors:		Number of neighbors to find for each tile
			algorithm:			NearestNeighbors algorithm, either 'kd_tree', 'ball_tree', or 'brute'

		Returns:
			Dict mapping slide names (from self.slides) to tile indices for tiles that were found to be neighbors to the provided neighbor_AV and neighbor_slides.
		'''
		if type(neighbor_slides) != list: neighbor_slides = [neighbor_slides]
		
		# Setup source slide-tile indices
		slide_tile_indices = []
		for slide in self.slide_node_dict:
			for tile_index in range(len(self.slide_node_dict[slide][0])):
				slide_tile_indices += [(slide, tile_index)]

		# Setup neighbor slide-tile indices
		neighbor_slide_tile_indices = []
		for slide in neighbor_slides:
			if slide not in neighbor_AV.slide_node_dict:
				raise ActivationsError(f"Slide {slide} not found in neighboring ActivationsVisualizer")
			for tile_index in range(len(neighbor_AV.slide_node_dict[slide][0])):
				neighbor_slide_tile_indices += [(slide, tile_index)]
		
		log.empty("Initializing nearest neighbor search...", 1)
		X = np.array([[self.slide_node_dict[slide][n][tile_index] for n in self.nodes] for (slide, tile_index) in slide_tile_indices])
		nbrs = NearestNeighbors(n_neighbors=5, algorithm=algorithm, n_jobs=-1).fit(X)
		neighbors = {}

		log.empty("Searching for nearest neighbors...", 1)
		neighbor_activations = [[neighbor_AV.slide_node_dict[slide][n][tile_index] for n in neighbor_AV.nodes] for (slide, tile_index) in neighbor_slide_tile_indices]
		distances, all_indices = nbrs.kneighbors(neighbor_activations)

		for indices_by_tile in all_indices:
			for index in indices_by_tile:
				neighboring_slide, tile_index = slide_tile_indices[index]
				if neighboring_slide in neighbors and tile_index not in neighbors[neighboring_slide]:
					neighbors[neighboring_slide] += [tile_index]
				else:
					neighbors.update({neighboring_slide: [tile_index]})

		#filtered_predictions = {}
		#filtered_predictions[neighbor_slide] = []
		#sentinel = {}
		
		#for tile_index, neighbor_indices in enumerate(all_indices):
			#neighbor_indices = neighbor_indices[:50]
			#prediction = 'Ras-like' if self.slide_logits_dict[slide][0][tile_index] > 0.5 else ('Braf-like' if self.slide_logits_dict[slide][0][tile_index] < -0.5 else 'None')
			#neighbor_categories = [tumap.slide_labels[tumap.point_meta[_i]['slide']] for _i in neighbor_indices]
			#most_common_category_in_neighbors = max(set(neighbor_categories), key=neighbor_categories.count)
			#percent_matching_categories = len([_i for _i in neighbor_indices if tumap.slide_labels[tumap.point_meta[_i]['slide']] == most_common_category_in_neighbors])/len(neighbor_indices)
		#	for neighbor_index in neighbor_indices:
		#		neighbor_slide = tumap.point_meta[neighbor_index]['slide']
		#		neighbor_index = tumap.point_meta[neighbor_index]['index']
		#		if neighbor_slide in neighbors:
		#			neighbors[neighbor_slide] += [neighbor_index]
		#		else:
		#			neighbors.update({neighbor_slide: [neighbor_index]})
			
			# Hotspot / sentinel zone defined by:
			#  - 99% of neighbors belong to the category that the tile is predicted to be
			#  - At least 10 different slides in this category can be found among the neighbors
			
			#if percent_matching_categories > 0.95 and most_common_category_in_neighbors == prediction:
			#	filtered_predictions[slide] += [self.slide_logits_dict[slide][0][tile_index]]
			#	num_unique_matching_slides = len(list(set([tumap.point_meta[_i]['slide'] for _i in neighbor_indices if tumap.slide_labels[tumap.point_meta[_i]['slide']] == prediction])))
			#	if num_unique_matching_slides >= 10:
			#		if slide in sentinel:
			#			if prediction in sentinel[slide]:
			#				sentinel[slide][prediction] += 1
			#			else:
			#				sentinel[slide].update({prediction: 1})
			#		else:
			#			sentinel.update({slide: {prediction: 1}})

		#if not exists('neighbor_dump.pkl'):
		#	with open('neighbor_dump.pkl', 'wb') as dumpfile:
		#		pickle.dump(neighbors, dumpfile)

		#for slide in sentinel:
		#	for cat in sentinel[slide]:
		#		print(slide, cat, sentinel[slide][cat])
		#for slide in filtered_predictions:
		#	if not filtered_predictions[slide]: continue
		#	print(slide, mean(filtered_predictions[slide]))
		return neighbors

	def generate_activations_from_model(self, model, tfrecords, use_fp16=True, batch_size=16, export=None, normalizer=None, normalizer_source=None, model_format=None):
		'''Calculates activations from a given model.

		Args:
			model:		Path to .h5 file from which to calculate final layer activations.
			use_fp16:	If true, uses Float16 (default) instead of Float32.
			batch_size:	Batch size for model predictions.
			export:		String (default: None). If provided, will export CSV of activations with this filename.'''

		# Rename tfrecord_array to tfrecords
		log.info(f"Calculating layer activations from {sfutil.green(model)}, max {self.MAX_TILES_PER_SLIDE} tiles per slide.", 1)

		# Load model
		combined_model = ModelActivationsInterface(model, model_format=model_format)

		unique_slides = list(set([sfutil.path_to_name(tfr) for tfr in tfrecords]))

		# Prepare normalizer
		if normalizer: log.info(f"Using realtime {normalizer} normalization", 1)
		normalizer = None if not normalizer else StainNormalizer(method=normalizer, source=normalizer_source)

		# Prepare PKL export dictionary
		for slide in unique_slides:
			if slide not in self.slide_node_dict:
				self.slide_node_dict.update({slide: {}})
			if slide not in self.slide_logits_dict:
				self.slide_logits_dict.update({slide: {}})

		def _parse_function(record):
			features = tf.io.parse_single_example(record, sfio.tfrecords.FEATURE_DESCRIPTION)
			slide = features['slide']
			image_string = features['image_raw']
			raw_image = tf.image.decode_jpeg(image_string, channels=3)

			if normalizer:
				raw_image = tf.py_function(normalizer.tf_to_rgb, [raw_image], tf.int32)

			processed_image = tf.image.convert_image_dtype(raw_image, tf.float32)
			processed_image = tf.image.per_image_standardization(processed_image)
			processed_image.set_shape([self.IMAGE_SIZE, self.IMAGE_SIZE, 3])
			return processed_image, slide

		# Calculate final layer activations for each tfrecord
		fla_start_time = time.time()
		nodes_names, logits_names = [], []
		detected_logit_structure=False
		if export:
			outfile = open(export, 'w')
			csvwriter = csv.writer(outfile)

		for t, tfrecord in enumerate(tfrecords):
			dataset = tf.data.TFRecordDataset(tfrecord)

			dataset = dataset.map(_parse_function, num_parallel_calls=8)
			dataset = dataset.batch(batch_size, drop_remainder=False)
			
			fl_activations_combined, logits_combined, slides_combined = [], [], []

			for i, data in enumerate(dataset):
				batch_processed_images, batch_slides = data
				batch_slides = batch_slides.numpy()
				batch_slides = np.array([unique_slides.index(bs.decode('utf-8')) for bs in batch_slides], dtype=np.uint32)

				fl_activations, logits = combined_model.predict(batch_processed_images)

				
				fl_activations_combined = fl_activations if fl_activations_combined == [] else np.concatenate([fl_activations_combined, fl_activations])
				logits_combined = logits if logits_combined == [] else np.concatenate([logits_combined, logits])
				slides_combined = batch_slides if slides_combined == [] else np.concatenate([slides_combined, batch_slides])

				sys.stdout.write(f"\r(TFRecord {t+1:>3}/{len(tfrecords):>3}) (Batch {i+1:>3}) ({len(fl_activations_combined):>5} images): {sfutil.green(sfutil.path_to_name(tfrecord))}")
				sys.stdout.flush()

				if self.MAX_TILES_PER_SLIDE and (len(fl_activations_combined) >= self.MAX_TILES_PER_SLIDE):
					break
			
			# Check if TFRecord was empty
			if fl_activations_combined == []:
				log.warn(f"Unable to calculate activations from {sfutil.green(sfutil.path_to_name(tfrecord))}; is the TFRecord empty?", 1)
				continue

			if not detected_logit_structure:
				nodes_names = [f"FLNode{f}" for f in range(fl_activations_combined.shape[1])]
				logits_names = [f"Logits{l}" for l in range(logits_combined.shape[1])]
				if export:
					header = ["Slide"] + logits_names + nodes_names
					csvwriter.writerow(header)
				for n in range(len(nodes_names)):
					for slide in unique_slides:
						self.slide_node_dict[slide].update({n: []})
				for l in range(len(logits_names)):
					for slide in unique_slides:
						self.slide_logits_dict[slide].update({l: []})
				detected_logit_structure=True

			if self.MAX_TILES_PER_SLIDE and len(fl_activations_combined) > self.MAX_TILES_PER_SLIDE:
				slides_combined = slides_combined[:self.MAX_TILES_PER_SLIDE]
				logits_combined = logits_combined[:self.MAX_TILES_PER_SLIDE]
				fl_activations_combined = fl_activations_combined[:self.MAX_TILES_PER_SLIDE]

			# Export to memory and CSV
			for i in range(len(fl_activations_combined)):
				slide = unique_slides[slides_combined[i]]
				activations_vals = fl_activations_combined[i].tolist()
				logits_vals = logits_combined[i].tolist()
				# Write to CSV
				if export:
					row = [slide] + logits_vals + activations_vals
					csvwriter.writerow(row)
				# Write to memory
				for n in range(len(nodes_names)):
					val = activations_vals[n]
					self.slide_node_dict[slide][n] += [val]
				for l in range(len(logits_names)):
					val = logits_vals[l]
					self.slide_logits_dict[slide][l] += [val]

		if export:
			outfile.close()

		fla_calc_time = time.time()
		print()
		log.info(f"Activation calculation time: {fla_calc_time-fla_start_time:.0f} sec", 1)
		if export:
			log.complete(f"Final layer activations saved to {sfutil.green(export)}", 1)
		
		# Dump PKL dictionary to file
		if self.ACTIVATIONS_CACHE:
			with open(self.ACTIVATIONS_CACHE, 'wb') as pt_pkl_file:
				pickle.dump([self.slide_node_dict, self.slide_logits_dict], pt_pkl_file)
			log.complete(f"Predictions and activations cached to {sfutil.green(self.ACTIVATIONS_CACHE)}", 1)

		return self.slide_node_dict, self.slide_logits_dict

	def export_activations_to_csv(self, filename, nodes=None):
		'''Exports calculated activations to csv.

		Args:
			filename:	Path to CSV file for export.
			nodes:		(optional) List of int. Activations of these nodes will be exported. 
							If None, activations for all nodes will be exported.
		'''
		with open(filename, 'w') as outfile:
			csvwriter = csv.writer(outfile)
			nodes = self.nodes if not nodes else nodes
			header = ["Slide"] + [f"FLNode{f}" for f in nodes]
			csvwriter.writerow(header)
			for slide in self.slide_node_dict:
				row = [slide]
				for n in nodes:
					row += [self.slide_node_dict[slide][n]]
				csvwriter.writewrow(row)

	def calculate_activation_averages_and_stats(self, filename=None, node_method='avg', threshold=0.5):
		'''Calculates activation averages across categories, 
			as well as tile-level and patient-level statistics using ANOVA, 
			exporting to CSV if desired.
			
		Args:
			filename:		Path to CSV file for export.
			node_method:	Either 'avg' (default) or 'threshold'. If avg, slide-level node data is calculated by averaging
								node activations across all tiles. If threshold, slide-level node data is calculated by counting
								the number of tiles with node activations > threshold and dividing by the total number of tiles.

		Returns:
			Dict mapping slides to dict of nodes mapping to slide-level node values;
			Dict mapping nodes to tile-level dict of statistics ('p', 'f');
			Dict mapping nodes to slide-level dict of statistics ('p', 'f');
		'''
		if not self.categories:
			log.warn("Unable to calculate activations statistics; annotations not loaded. Please load with load_annotations().'")
			return
		if node_method not in ('avg', 'threshold'):
			raise ActivationsError(f"'node_method' must be either 'avg' or 'threshold', not {node_method}")
		#if node_method == 'threshold' and not threshold_category:
		#	raise ActivationsError(f"If 'node_method' is threshold, then threshold_category must be supplied.")

		empty_category_dict = {}
		self.node_cat_dict = {}
		slide_node_dict = {}
		tile_node_stats = {}
		slide_node_stats = {}
		for category in self.categories:
			empty_category_dict.update({category: []})
		if not hasattr(self, 'nodes'):
			log.error("Activations have not been generated, unable to calculate averages")
			return
		for node in self.nodes:
			self.node_cat_dict.update({node: deepcopy(empty_category_dict)})

		self.categories = self.categories[::-1]

		log.empty("Calculating activation averages & stats across nodes...", 1)
		for node in self.nodes:
			# For each node, calculate average across tiles found in a slide
			for slide in self.slides:
				if slide not in slide_node_dict: slide_node_dict.update({slide: {}})
				pt_cat = self.slide_category_dict[slide]
				if node_method == 'avg':
					node_val = mean(self.slide_node_dict[slide][node])
				elif node_method == 'threshold':
					node_val = sum([1 for t in self.slide_node_dict[slide][node] if t > threshold]) / len(self.slide_node_dict[slide][node])
				self.node_cat_dict[node][pt_cat] += [node_val]
				slide_node_dict[slide][node] = node_val
			
			# Tile-level ANOVA
			fvalue, pvalue = stats.f_oneway(*self.get_tile_node_activations_by_category(node))
			if not isnan(fvalue) and not isnan(pvalue): 
				tile_node_stats.update({node: {'f': fvalue,
										  'p': pvalue} })
			else:
				tile_node_stats.update({node: {'f': -1,
										  'p': 1} })

			# Patient-level ANOVA
			fvalue, pvalue = stats.f_oneway(*[self.node_cat_dict[node][c] for c in self.used_categories])
			if not isnan(fvalue) and not isnan(pvalue): 
				slide_node_stats.update({node: {'f': fvalue,
												 'p': pvalue} })
			else:
				slide_node_stats.update({node: {'f': -1,
										  		 'p': 1} })

			# Thresholding
			#if threshold_category:
			#	pt_vals = []
			#	for c in [c for c in self.used_categories if c != threshold_category]:
			#		pt_vals += self.node_cat_dict[node][c]
			#	if not pt_vals:
			#		slide_node_stats[node]['num_above_thresh'] = 0
			#	else:
			#		cat_thresh = max(pt_vals)
			#		num_above_thresh = len([v for v in self.node_cat_dict[node][threshold_category] if v > cat_thresh])

		try:
			self.nodes = sorted(self.nodes, key=lambda n: tile_node_stats[n]['p'])
			#if node_method == 'avg':
			self.sorted_nodes = sorted(self.nodes, key=lambda n: slide_node_stats[n]['p'])
			#elif node_method == 'threshold':
			#	self.sorted_nodes = sorted(self.nodes, key=lambda n: slide_node_stats[n]['num_above_thresh'], reverse=True)
		except:
			log.warn("No stats calculated; unable to sort nodes.", 1)
			self.sorted_nodes = self.nodes
			
		for i, node in enumerate(self.nodes):
			if self.focus_nodes and (node not in self.focus_nodes): continue
			try:
				log.info(f"Tile-level P-value ({node}): {tile_node_stats[node]['p']}", 1)
				log.info(f"Patient-level P-value: ({node}): {slide_node_stats[node]['p']}", 1)
			except:
				log.warn(f"No stats calculated for node {node}", 1)
			if (not self.focus_nodes) and i>9: break

		# Export results
		export_file = self.STATS_CSV_FILE if not filename else filename
		self._save_node_statistics_to_csv(self.sorted_nodes, slide_node_dict, filename=export_file, tile_stats=tile_node_stats, slide_stats=slide_node_stats)
		return slide_node_dict, tile_node_stats, slide_node_stats

	def logistic_regression(self, slide_method='avg', node_threshold=0.5):
		'''Experimental function, creates a logistic regression model to generate slide-level predictions from slide-level statistics.'''
		from sklearn.linear_model import LogisticRegression
		from sklearn.metrics import classification_report, confusion_matrix

		slide_node_dict, _, _ = self.calculate_activation_averages_and_stats(None, slide_method, node_threshold)

		log.empty("Working on logistic regression...", 1)
		x = np.array([[slide_node_dict[slide][n] for n in self.nodes] for slide in self.slides])
		y = np.array([self.categories.index(self.slide_category_dict[slide]) for slide in self.slides])
		
		model = LogisticRegression(solver='lbfgs', max_iter=100, multi_class='ovr').fit(x, y)
		yhat = model.predict(x)
		log.complete(f"Regression complete, accuracy: {model.score(x, y):.3f}", 1)
		return model

	def generate_box_plots(self, export_folder=None):
		'''Generates box plots comparing nodal activations at the slide-level and tile-level.
		
		Args:
			export_folder:	(optional) Path to directory in which to save box plots.
								If None, will save boxplots to STATS_ROOT directory.
		'''

		if not self.categories:
			log.warn("Unable to generate box plots; annotations not loaded. Please load with load_annotations().")
			return

		# First ensure basic stats have been calculated
		if not hasattr(self, 'sorted_nodes'):
			self.calculate_activation_averages_and_stats()
		if not export_folder: export_folder = self.STATS_ROOT

		# Display tile-level box plots & stats
		log.empty("Generating box plots...")
		for i, node in enumerate(self.nodes):
			if self.focus_nodes and (node not in self.focus_nodes): continue
			plt.clf()
			snsbox = sns.boxplot(data=self.get_tile_node_activations_by_category(node))
			title = f"{node} (tile-level)"
			snsbox.set_title(title)
			snsbox.set(xlabel="Category", ylabel="Activation")
			plt.xticks(plt.xticks()[0], self.used_categories)
			boxplot_filename = join(export_folder, f"boxplot_{title}.png")
			plt.savefig(boxplot_filename, bbox_inches='tight')
			if (not self.focus_nodes) and i>4: break

		# Print slide_level box plots & stats
		for i, node in enumerate(self.sorted_nodes):
			if self.focus_nodes and (node not in self.focus_nodes): continue
			plt.clf()
			snsbox = sns.boxplot(data=[self.node_cat_dict[node][c] for c in self.used_categories])
			title = f"{node} (slide-level)"
			snsbox.set_title(title)
			snsbox.set(xlabel="Category",ylabel="Average tile activation")
			plt.xticks(plt.xticks()[0], self.used_categories)
			boxplot_filename = join(export_folder, f"boxplot_{title}.png")
			plt.savefig(boxplot_filename, bbox_inches='tight')
			if (not self.focus_nodes) and i>4: break

	def save_example_tiles_gradient(self, nodes=None, export_folder=None, tile_filter=None):
		'''For a given set of activation nodes, saves image tiles named according 
			to their corresponding node activations, for easy sorting and visualization.
			Duplicate image tiles will be saved for each node, organized into subfolders named according to node id.

		Args:
			nodes:			List of int, nodes to evaluate
			export_folder:	Path to folder in which to save examples tiles
			tile_filter:	(optional) Dict mapping slide names to tile indices.
								If provided, will only save image tiles from this list.
								Example:
								{'slide1': [0, 16, 200]}
		'''
		if not export_folder: export_folder = self.SORTED_DIR = join(self.STATS_ROOT, "sorted_tiles")
		if not nodes: nodes = self.focus_nodes

		for node in nodes:
			if not exists(join(export_folder, node)):
				os.makedirs(join(export_folder, node))
			
			gradient = []
			for slide in self.slides:
				for i, tile in enumerate(self.slide_node_dict[slide][node]):
					if tile_filter and (slide not in tile_filter) or (i not in tile_filter[slide]):
						continue
					gradient += [{
									'val': tile,
									'slide': slide,
									'index': i
					}]
			gradient = sorted(gradient, key=lambda k: k['val'])
			for i, g in enumerate(gradient):
				print(f"Extracting tile {i} of {len(gradient)} for node {node}...", end="\r")
				for tfr in self.tfrecords:
					if sfutil.path_to_name(tfr) == g['slide']:
						tfr_dir = tfr
				if not tfr_dir:
					log.warn(f"TFRecord location not found for slide {g['slide']}", 1)
				slide, image = sfio.tfrecords.get_tfrecord_by_index(tfr_dir, g['index'], decode=False)
				tile_filename = f"{i}-tfrecord{g['slide']}-{g['index']}-{g['val']:.2f}.jpg"
				image_string = open(join(export_folder, node, tile_filename), 'wb')
				image_string.write(image.numpy())
				image_string.close()
			print()

	def save_example_tiles_high_low(self, nodes, slides, export_folder=None):
		'''For a given set of activation nodes, saves images of tiles with the highest and lowest
		activations in these nodes, restricted to the set of slides designated.

		Args:
			nodes:			List of int. Nodes with which to perform this function.
			slides:			List of slide names. Will load tile images from these slides. 
			export_folder:	Path to directory in which to save image tiles.
		'''
		if not export_folder: export_folder = join(self.STATS_ROOT, "example_tiles")
		for node in nodes:
			for slide in slides:
				sorted_index = np.argsort(self.slide_node_dict[slide][node])
				lowest = sorted_index[:10]
				highest = sorted_index[-10:]
				lowest_dir = join(export_folder, node, slide, 'lowest')
				highest_dir = join(export_folder, node, slide, 'highest')
				if not exists(lowest_dir): os.makedirs(lowest_dir)
				if not exists(highest_dir): os.makedirs(highest_dir)

				for tfr in self.tfrecords:
					if sfutil.path_to_name(tfr) == slide:
						tfr_dir = tfr
				if not tfr_dir:
					log.warn(f"TFRecord location not found for slide {slide}", 1)

				def extract_by_index(indices, directory):
					for index in indices:
						slide, image = sfio.tfrecords.get_tfrecord_by_index(tfr_dir, index, decode=False)
						tile_filename = f"tfrecord{slide.numpy()}-tile{index}.jpg"
						image_string = open(join(directory, tile_filename), 'wb')
						image_string.write(image.numpy())
						image_string.close()

				extract_by_index(lowest, lowest_dir)
				extract_by_index(highest, highest_dir)

class TileVisualizer:
	'''Class to supervize visualization of node activations across an image tile.
	Visualization is accomplished by performing sequential convolutional masking 
		and determining impact of masking on node activation. In this way,
		the masking reveals spatial importance with respect to activation of the given node.
	'''

	def __init__(self, model, node, tile_px, mask_width=None, normalizer=None, normalizer_source=None, model_format=None):
		'''Object initializer.

		Args:
			model:				Path to .h5 model file
			node:				Int, activation node to analyze
			tile_px:			Int, width/height of image tiles
			mask_width:			Width of mask to convolutionally apply. Defaults to 1/6 of tile_px
			normalizer:			String, normalizer to apply to tiles in real-time.
			normalizer_source:	Path to normalizer source image.
		'''
		self.NODE = node
		self.IMAGE_SHAPE = (tile_px, tile_px, 3)
		self.MASK_WIDTH = mask_width if mask_width else int(self.IMAGE_SHAPE[0]/6)
		self.normalizer = None if not normalizer else StainNormalizer(method=normalizer, source=normalizer_source)

		log.info("Initializing tile visualizer", 1)
		log.info(f"Node: {sfutil.bold(str(node))} | Shape: ({self.IMAGE_SHAPE}) | Window size: {self.MASK_WIDTH}", 1)
		log.info(f"Loading Tensorflow model at {sfutil.green(model)}...", 1)

		self.loaded_model = ModelActivationsInterface(model, model_format=model_format)

	def _calculate_activation_map(self, stride_div=4):
		'''Creates map of importance through convolutional masking and
		examining changes in node activations.'''
		sx = self.IMAGE_SHAPE[0]
		sy = self.IMAGE_SHAPE[1]
		w  = self.MASK_WIDTH
		stride = int(self.MASK_WIDTH / stride_div)
		min_x  = int(w/2)
		max_x  = int(sx - w/2)
		min_y  = int(w/2)
		max_y  = int(sy - w/2)

		act_array = []
		for yi in range(min_y, max_y, stride):
			for xi in range(min_x, max_x, stride):
				mask = create_bool_mask(xi, yi, w, sx, sy)
				masked = self.tf_processed_image.numpy() * mask
				act, _ = self.loaded_model.predict(np.array([masked]))
				act_array += [act[0][self.NODE]]
				print(f"Calculating activations at x:{xi}, y:{yi}; act={act[0][self.NODE]}", end='\033[K\r')
		max_center_x = max(range(min_x, max_x, stride))
		max_center_y = max(range(min_y, max_y, stride))
		reshaped_array = np.reshape(np.array(act_array), [len(range(min_x, max_x, stride)), 
														  len(range(min_y, max_y, stride))])
		print()
		return reshaped_array, max_center_x, max_center_y

	def _predict_masked(self, x, y, index):
		mask = create_bool_mask(x, y, self.MASK_WIDTH, self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1])
		masked = self.tf_processed_image.numpy() * mask
		act, _ = self.loaded_model.predict(np.array([masked]))
		return act[0][index]	

	def visualize_tile(self, tfrecord=None, index=None, image_jpg=None, export_folder=None, zoomed=True, interactive=False):
		'''Visualizes tiles, either interactively or saving to directory.
		
		Args:
			tfrecord:			If provided, will visualize tile from the designated tfrecord.
									Must supply either a tfrecord and index or image_jpg
			index:				Index of tile to visualize within tfrecord, if provided
			image_jpeg:			JPG image to perform analysis on
			export_folder:		Folder in which to save heatmap visualization
			zoomed:				Bool. If true, will crop image to space containing heatmap (otherwise a small border will be seen)
			interactive:		If true, will display as interactive map using matplotlib
		'''
		if not (image_jpg or tfrecord):
			raise ActivationsError("Must supply either tfrecord or image_jpg")

		if image_jpg:
			log.info(f"Processing tile at {sfutil.green(image_jpg)}...", 1)
			tilename = sfutil.path_to_name(image_jpg)
			self.tile_image = Image.open(image_jpg)
			image_file = open(image_jpg, 'rb')
			tf_decoded_image = tf.image.decode_jpeg(image_file.read(), channels=3)
		else:
			slide, tf_decoded_image = sfio.tfrecords.get_tfrecord_by_index(tfrecord, index, decode=True)
			tilename = f'{slide.numpy().decode("utf-8")}-{index}'
			self.tile_image = Image.fromarray(tf_decoded_image.numpy())

		# Normalize PIL image & TF image
		if self.normalizer: 
			self.tile_image = self.normalizer.pil_to_pil(self.tile_image)
			tf_decoded_image = tf.py_function(self.normalizer.tf_to_rgb, [self.tile_image], tf.int32)

		# Next, process image with Tensorflow
		self.tf_processed_image = tf.image.convert_image_dtype(self.tf_processed_image, tf.float16)
		self.tf_processed_image = tf.image.per_image_standardization(tf_decoded_image)
		self.tf_processed_image.set_shape(self.IMAGE_SHAPE)

		# Now create the figure
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111)
		self.implot = plt.imshow(self.tile_image)

		if interactive:
			self.rect = patches.Rectangle((0, 0), self.MASK_WIDTH, self.MASK_WIDTH, facecolor='white', zorder=20)
			self.ax.add_patch(self.rect)

		activation_map, max_center_x, max_center_y = self._calculate_activation_map()

		# Prepare figure
		filename = join(export_folder, f'{tilename}-heatmap.png')

		def hover(event):
			if event.xdata:
				self.rect.set_xy((event.xdata-self.MASK_WIDTH/2, event.ydata-self.MASK_WIDTH/2))
				print(self._predict_masked(event.xdata, event.ydata, index=self.NODE), end='\r')
				self.fig.canvas.draw_idle()

		def click(event):
			if event.button == 1:
				self.MASK_WIDTH = min(min(self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1]), self.MASK_WIDTH + 25)
				self.rect.set_width(self.MASK_WIDTH)
				self.rect.set_height(self.MASK_WIDTH)
			else:
				self.MASK_WIDTH = max(0, self.MASK_WIDTH - 25)
				self.rect.set_width(self.MASK_WIDTH)
				self.rect.set_height(self.MASK_WIDTH)
			self.fig.canvas.draw_idle()	

		if interactive:
			self.fig.canvas.mpl_connect('motion_notify_event', hover)
			self.fig.canvas.mpl_connect('button_press_event', click)
		
		if activation_map is not None:
			# Define color map
			jetMap = np.flip(np.linspace(0.45, 0.95, 255))
			cmMap = cm.nipy_spectral(jetMap)
			newMap = mcol.ListedColormap(cmMap)

			# Calculate boundaries of heatmap
			hw = int(self.MASK_WIDTH/2)
			if zoomed:
				extent = (hw, max_center_x, max_center_y, hw)
			else:
				extent = (0, max_center_x+hw, max_center_y+hw, 0)
			
			# Heatmap
			self.ax.imshow(activation_map, extent=extent,
										   cmap=newMap,
										   alpha=0.6 if not interactive else 0.0,
										   interpolation='bicubic',
										   zorder=10)
		if filename:
			plt.savefig(filename, bbox_inches='tight')
			log.complete(f"Heatmap saved to {filename}", 1)
		if interactive:
			print()
			plt.show()

class Heatmap:
	'''Generates heatmap by calculating predictions from a sliding scale window across a slide.'''

	def __init__(self, slide_path, model_path, size_px, size_um, use_fp16=True, stride_div=2, roi_dir=None, 
					roi_list=None, roi_method='inside', buffer=True, normalizer=None, normalizer_source=None,
					batch_size=16, skip_thumb=False, model_format=None):
		'''Convolutes across a whole slide, calculating logits and saving predictions internally for later use.

		Args:
			slide_path:			Path to slide
			model_path:			Path to .h5 model file
			size_px:			Size of image tiles, in pixels
			size_um:			Size of image tiles, in microns
			use_fp16:			Bool, whether to use FP16 (vs FP32)
			stride_div:			Divisor for stride when convoluting across slide
			roi_dir:			Directory in which slide ROI is contained
			roi_list:			If a roi_dir is not supplied, a list of paths to ROI CSVs can be provided
			roi_method:			Either 'inside', 'outside', or 'ignore'. If inside, tiles will be extracted inside ROI region
									If outside, tiles will be extracted outside ROI region
			buffer:				Either 'vmtouch' or path to directory to use for buffering slides
									Significantly improves performance for slides on HDDs
			normalizer:			Normalization strategy to use on image tiles
			normalizer_source:	Path to normalizer source image
			batch_size:			Batch size when calculating predictions
			skip_thumb:			If true, will skip thumbnail generation (can save time if saving heatmap without thumbnail image)
		'''
		from slideflow.slide import SlideReader

		#self.DTYPE = tf.float16 if use_fp16 else tf.float32
		self.logits = None

		# Setup normalization
		self.normalizer = normalizer
		self.normalizer_source = normalizer_source

		# Create progress bar
		pb = ProgressBar(1, counter_text='tiles', leadtext="Generating heatmap... ", show_counter=True, show_eta=True)
		self.print = pb.print

		# Load the slide
		self.slide = SlideReader(slide_path, size_px, size_um, stride_div, enable_downsample=False, 
																		   roi_dir=roi_dir, 
																		   roi_list=roi_list,
																		   roi_method=roi_method,
																		   silent=True,
																		   buffer=buffer,
																		   pb=pb)
		pb.BARS[0].end_value = self.slide.estimated_num_tiles

		# First, load the designated model
		self.model = ModelActivationsInterface(model_path, model_format=model_format)

		# Record the number of classes in the model
		self.NUM_CLASSES = self.model.NUM_CLASSES#_model.layers[-1].output_shape[-1]

		if not self.slide.loaded_correctly():
			raise ActivationsError(f"Unable to load slide {self.slide.name} for heatmap generation")

		# Pre-load thumbnail in separate thread
		if not skip_thumb:
			thumb_process = DProcess(target=self.slide.thumb)
			thumb_process.start()

		# Create tile coordinate generator
		gen_slice = self.slide.build_generator(normalizer=self.normalizer,
											   normalizer_source=self.normalizer_source)

		if not gen_slice:
			log.error(f"No tiles extracted from slide {sfutil.green(self.slide.name)}", 1)

		# Generate dataset from the generator
		with tf.name_scope('dataset_input'):
			tile_dataset = tf.data.Dataset.from_generator(gen_slice, (tf.uint32))
			tile_dataset = tile_dataset.map(self._parse_function, num_parallel_calls=8)
			tile_dataset = tile_dataset.batch(batch_size, drop_remainder=False)
			tile_dataset = tile_dataset.prefetch(8)

		# Iterate through generator to calculate logits +/- final layer activations for all tiles
		logits_arr = []		# Logits (predictions)
		postconv_arr = []	# Post-convolutional layer (post-convolutional activations)
		for batch_images in tile_dataset:
			postconv, logits = self.model.predict(batch_images)
			logits_arr = logits if logits_arr == [] else np.concatenate([logits_arr, logits])
			postconv_arr = postconv if postconv_arr == [] else np.concatenate([postconv_arr, postconv])

		num_postconv_nodes = postconv_arr.shape[1]

		if not skip_thumb:
			print('\r\033[KFinished predictions. Waiting on thumbnail...', end="")
			thumb_process.join()

		if (self.slide.tile_mask is not None) and (self.slide.extracted_x_size) and (self.slide.extracted_y_size) and (self.slide.full_stride):
			# Expand logits back to a full 2D map spanning the whole slide,
			#  supplying values of "0" where tiles were skipped by the tile generator
			x_logits_len = int(self.slide.extracted_x_size / self.slide.full_stride) + 1
			y_logits_len = int(self.slide.extracted_y_size / self.slide.full_stride) + 1
			expanded_logits = [[-1] * self.NUM_CLASSES] * len(self.slide.tile_mask)
			expanded_postconv = [[-1] * num_postconv_nodes] * len(self.slide.tile_mask)
			li = 0
			for i in range(len(expanded_logits)):
				if self.slide.tile_mask[i] == 1:
					expanded_logits[i] = logits_arr[li]
					expanded_postconv[i] = postconv_arr[li]
					li += 1
			try:
				expanded_logits = np.asarray(expanded_logits, dtype=float)
				expanded_postconv = np.asarray(expanded_postconv, dtype=float)
			except ValueError:
				raise ActivationsError("Mismatch with number of categories in model output and expected number of categories")

			# Resize logits array into a two-dimensional array for heatmap display
			self.logits = np.resize(expanded_logits, [y_logits_len, x_logits_len, self.NUM_CLASSES])
			self.postconv = np.resize(expanded_postconv, [y_logits_len, x_logits_len, num_postconv_nodes])
		else:
			self.logits = logits_arr
			self.postconv = postconv_arr

		if (type(self.logits) == bool) and (not self.logits):
			log.error(f"Unable to create heatmap for slide {sfutil.green(self.slide.name)}", 1)

	def _parse_function(self, image):
		parsed_image = tf.image.per_image_standardization(image)
		#parsed_image = tf.image.convert_image_dtype(parsed_image, self.DTYPE)
		parsed_image.set_shape([299, 299, 3])
		return parsed_image

	def _prepare_figure(self, show_roi=True):
		self.fig = plt.figure(figsize=(18, 16))
		self.ax = self.fig.add_subplot(111)
		self.fig.subplots_adjust(bottom = 0.25, top=0.95)
		gca = plt.gca()
		gca.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
		jetMap = np.linspace(0.45, 0.95, 255)
		cmMap = cm.nipy_spectral(jetMap)
		self.newMap = mcol.ListedColormap(cmMap)
		
		# Plot ROIs
		if show_roi:
			print("\r\033[KPlotting ROIs...", end="")
			ROI_SCALE = self.slide.full_shape[0]/2048
			annPolys = [sg.Polygon(annotation.scaled_area(ROI_SCALE)) for annotation in self.slide.rois]
			for poly in annPolys:
				x,y = poly.exterior.xy
				plt.plot(x, y, zorder=20, color='k', linewidth=5)

	def display(self, show_roi=True, interpolation='none', logit_cmap=None, skip_thumb=False):
		'''Interactively displays calculated logits as a heatmap.
		
		Args:
			show_roi:			Bool, whether to overlay ROIs onto heatmap image
			interpolation:		Interpolation strategy to use for smoothing heatmap
			logit_cmap:			Either function or a dictionary use to create heatmap colormap.
									Each image tile will generate a list of predictions of length O, 
									where O is the number of outcome categories.
									If logit_cmap is a function, then this logit prediction list will be passed to the function,
										and the function is expected to return [R, G, B] values which will be displayed.
									If the logit_cmap is a dictionary, it should map 'r', 'g', and 'b' to outcome indices;
										The prediction for these outcome categories will be mapped to the corresponding colors.
										Thus, the corresponding color will only reflect predictions of up to three outcome categories.
										Example (this would map prediction for outcome 0 to the red colorspace, outcome 3 to green colorspace, and so on):
										{
											'r': 0,
											'g': 3,
											'b': 1
										}
			skip_thumb:			Bool, whether to skip thumbnail (vs displaying with heatmap)
		'''
		self._prepare_figure(show_roi=False)
		heatmap_dict = {}

		if not skip_thumb:
			if show_roi: thumb = self.slide.annotated_thumb()
			else: thumb = self.slide.thumb()
			implot = FastImshow(thumb, self.ax, extent=None, tgt_res=1024)

		def slider_func(val):
			for h, s in heatmap_dict.values():
				h.set_alpha(s.val)

		if logit_cmap:
			if callable(logit_cmap):
				map_logit = logit_cmap
			else:
				def map_logit(l): 
					# Make heatmap with specific logit predictions mapped to r, g, and b
					return (l[logit_cmap['r']], l[logit_cmap['g']], l[logit_cmap['b']])
			extent = None if skip_thumb else implot.extent
			heatmap = self.ax.imshow([[map_logit(l) for l in row] for row in self.logits], extent=extent, interpolation=interpolation, zorder=10)
		else:
			for i in range(self.NUM_CLASSES):
				heatmap = self.ax.imshow(self.logits[:, :, i], extent=implot.extent, cmap=self.newMap, alpha = 0.0, interpolation=interpolation, zorder=10) #bicubic
				ax_slider = self.fig.add_axes([0.25, 0.2-(0.2/self.NUM_CLASSES)*i, 0.5, 0.03], facecolor='lightgoldenrodyellow')
				slider = Slider(ax_slider, f'Class {i}', 0, 1, valinit = 0)
				heatmap_dict.update({f"Class{i}": [heatmap, slider]})
				slider.on_changed(slider_func)

		self.fig.canvas.set_window_title(self.slide.name)
		implot.show()
		plt.show()

	def save(self, save_folder, show_roi=True, interpolation='none', logit_cmap=None, skip_thumb=False):
		'''Saves calculated logits as heatmap overlays.
		
		Args:
			show_roi:			Bool, whether to overlay ROIs onto heatmap image
			interpolation:		Interpolation strategy to use for smoothing heatmap
			logit_cmap:			Either function or a dictionary use to create heatmap colormap.
									Each image tile will generate a list of predictions of length O, 
									where O is the number of outcome categories.
									If logit_cmap is a function, then this logit prediction list will be passed to the function,
										and the function is expected to return [R, G, B] values which will be displayed.
									If the logit_cmap is a dictionary, it should map 'r', 'g', and 'b' to outcome indices;
										The prediction for these outcome categories will be mapped to the corresponding colors.
										Thus, the corresponding color will only reflect predictions of up to three outcome categories.
										Example (this would map prediction for outcome 0 to the red colorspace, outcome 3 to green colorspace, and so on):
										{
											'r': 0,
											'g': 3,
											'b': 1
										}
			skip_thumb:			Bool, whether to skip thumbnail (vs displaying with heatmap)
		'''
		print("\r\033[KPreparing figure...", end="")
		self._prepare_figure(show_roi=False)

		if not skip_thumb:
			# Save plot without heatmaps
			print("\r\033[KSaving base figure...", end="")
			if show_roi: thumb = self.slide.annotated_thumb()
			else: thumb = self.slide.thumb()
			implot = self.ax.imshow(thumb, zorder=0)
			plt.savefig(os.path.join(save_folder, f'{self.slide.name}-raw.png'), bbox_inches='tight')

		if logit_cmap:
			if callable(logit_cmap):
				map_logit = logit_cmap
			else:
				def map_logit(l): 
					# Make heatmap with specific logit predictions mapped to r, g, and b
					return (l[logit_cmap['r']], l[logit_cmap['g']], l[logit_cmap['b']])
			extent = None if skip_thumb else implot.get_extent()
			heatmap = self.ax.imshow([[map_logit(l) for l in row] for row in self.logits], extent=extent, interpolation=interpolation, zorder=10)
			plt.savefig(os.path.join(save_folder, f'{self.slide.name}-custom.png'), bbox_inches='tight')
		else:
			# Make heatmap plots and sliders for each outcome category
			for i in range(self.NUM_CLASSES):
				print(f"\r\033[KMaking heatmap {i+1} of {self.NUM_CLASSES}...", end="")
				heatmap = self.ax.imshow(self.logits[:, :, i], extent=implot.get_extent(),
															cmap=self.newMap,
															vmin=0,
															vmax=1,
															alpha=0.6,
															interpolation=interpolation, #bicubic
															zorder=10)
				plt.savefig(os.path.join(save_folder, f'{self.slide.name}-{i}.png'), bbox_inches='tight')
				heatmap.remove()

		plt.close()
		print("\r\033[K", end="")
		log.complete(f"Saved heatmaps for {sfutil.green(self.slide.name)}", 1)