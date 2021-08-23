import os
import sys
import csv

from numpy.lib.arraysetops import unique
import umap
import types
import time
import pickle

import seaborn as sns
import numpy as np
import pandas as pd
import multiprocessing as mp
import slideflow.util as sfutil

from functools import partial
from slideflow.util import ProgressBar
from os.path import join
from slideflow.util import log
from scipy import stats
from random import sample
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib.widgets import LassoSelector
from lifelines.utils import concordance_index as c_index

# TODO: remove 'hidden_0' reference as this may not be present if the model does not have hidden layers

class StatisticsError(Exception):
	pass

class TFRecordMap:
	'''Map of tiles from TFRecords, mapped either explicitly with pre-specified coordinates, 
			or mapped with dimensionality reduction from post-convolutional layer weights, 
			as provided from sf.activations.ActivationsVisualizer.'''

	def __init__(self, slides, tfrecords, cache=None):
		''' Backend for mapping TFRecords into two dimensional space. Can use an ActivationsVisualizer object
		to map TFRecords according to UMAP of activations, or map according to pre-specified coordinates. 
		
		Args:
			slides:		List of slide names
			tfrecords:	List of paths to tfrecords
			cache:		(optional) String, path name. If provided, will cache activations to this PKL file.'''

		self.slides = slides
		self.tfrecords = tfrecords
		self.cache = cache

		self.x = []
		self.y = []
		self.point_meta = []
		self.values = []
		self.map_meta = {}

		# Try to load from cache
		if self.cache:
			if self.load_cache():
				return

	@classmethod
	def from_precalculated(cls, slides, x, y, meta, values=None, tfrecords=None, cache=None):
		''' Initializes map from precalculated coordinates.

		Args:
			slides:		List of slide names
			x:			X coordinates for tfrecords
			y:			Y coordinates for tfrecords
			meta:		List of dicts. Metadata for each point on the map (representing a single tfrecord). 
			values:		Values used to overlay colors during plotting.
			tfrecords:	(optional) List of paths to tfrecords. Not required, used to store for use by other objects. *** TODO: REMOVE ***
			cache:		(optional) String, path name. If provided, will cache umap coordinates to this PKL file. '''

		obj = cls(slides, tfrecords)
		obj.x = np.array(x) if type(x) == list else x
		obj.y = np.array(y) if type(y) == list else y
		obj.point_meta = np.array(meta) if type(meta) == list else meta
		obj.cache = cache
		obj.values = np.array(values) if type(values) == list else values
		if obj.values == []:
			obj.values = np.array(['None' for i in range(len(obj.point_meta))])
		obj.save_cache()
		return obj

	@classmethod
	def from_activations(cls, activations, exclude_slides=None, prediction_filter=None, force_recalculate=False, 
						 map_slide=None, cache=None, low_memory=False, max_tiles_per_slide=0, umap_dim=2):
		'''Initializes map from an activations visualizer.

		Args:
			activations:		ActivationsVisualizer class
			exclude_slides:		(optional) List of names of slides to exclude from map.
			prediction_filter:	(optional) List. Will restrict predictions to only these provided categories.
			force_recalculate:	(optional) Will force recalculation of umap despite presence of cache.
			use_centroid:		(optional) Will calculate and map centroid activations.
			map_slide:			Either None (default), 'centroid', or 'average'. If none, will map all tiles from each slide.
			cache:				(optional) String, path name. If provided, will cache umap coordinates to this file. '''

		if map_slide is not None and map_slide not in ('centroid', 'average'):
			raise StatisticsError(f"map_slide must be None (default), 'centroid', or 'average', not '{map_slide}'")

		slides = activations.slides if not exclude_slides else [slide for slide in activations.slides 
																	  if slide not in exclude_slides]

		tfrecords = activations.tfrecords if not exclude_slides else [tfr for tfr in activations.tfrecords 
																		  if sfutil.path_to_name(tfr) not in exclude_slides]

		obj = cls(slides, tfrecords, cache=cache)
		obj.AV = activations
		if map_slide:
			obj._calculate_from_slides(method=map_slide,
									   prediction_filter=prediction_filter,
									   force_recalculate=force_recalculate,
									   low_memory=low_memory)
		else:
			obj._calculate_from_tiles(prediction_filter=prediction_filter,
									  force_recalculate=force_recalculate, 
									  low_memory=low_memory, 
									  max_tiles_per_slide=max_tiles_per_slide, 
									  dim=umap_dim)
		return obj

	def _calculate_from_tiles(self, prediction_filter=None, force_recalculate=False, 
							  low_memory=False, max_tiles_per_slide=0, dim=2):

		''' Internal function to guide calculation of UMAP from final layer activations, as provided via ActivationsVisualizer nodes.'''

		if len(self.x) and len(self.y) and not force_recalculate:
			log.info("UMAP loaded from cache, will not recalculate", 1)

			# First, filter out slides not included in provided activations
			self.x = np.array([self.x[i] for i in range(len(self.x)) if self.point_meta[i]['slide'] in self.AV.slides])
			self.y = np.array([self.y[i] for i in range(len(self.y)) if self.point_meta[i]['slide'] in self.AV.slides])
			self.point_meta = np.array([self.point_meta[i] for i in range(len(self.point_meta)) 
														   if self.point_meta[i]['slide'] in self.AV.slides])
			self.values = np.array(['None' for i in range(len(self.point_meta)) 
										   if self.point_meta[i]['slide'] in self.AV.slides])
			
			# If UMAP already calculated, only update predictions if prediction filter is provided
			if prediction_filter:
				log.info("Updating UMAP predictions according to filter restrictions", 1)
				
				num_logits = len(self.AV.slide_logits_dict[self.slides[0]])

				for i in range(len(self.point_meta)):	
					slide = self.point_meta[i]['slide']
					tile_index = self.point_meta[i]['index']		
					logits = [self.AV.slide_logits_dict[slide][l][tile_index] for l in range(num_logits)]
					filtered_logits = [logits[l] for l in prediction_filter]
					prediction = logits.index(max(filtered_logits))
					self.point_meta[i]['logits'] = logits
					self.point_meta[i]['prediction'] = prediction

			if max_tiles_per_slide:
				log.info(f"Restricting map to maximum of {max_tiles_per_slide} tiles per slide", 1)
				new_x, new_y, new_meta = [], [], []
				slide_tile_count = {}
				for i, pm in enumerate(self.point_meta):
					slide = pm['slide']
					if slide not in slide_tile_count:
						slide_tile_count.update({slide: 1})
					elif slide_tile_count[slide] < max_tiles_per_slide:
						new_x += [self.x[i]]
						new_y += [self.y[i]]
						new_meta += [pm]
						slide_tile_count[slide] += 1
				self.x, self.y, self.point_meta = np.array(new_x), np.array(new_y), np.array(new_meta)
				self.values = np.array(['None' for i in range(len(self.point_meta))])
			return

		# Calculate UMAP
		node_activations = []
		self.map_meta['nodes'] = self.AV.nodes
		log.empty("Calculating UMAP...", 1)
		for slide in self.slides:
			first_node = list(self.AV.slide_node_dict[slide].keys())[0]
			num_vals = len(self.AV.slide_node_dict[slide][first_node])
			num_logits = len(self.AV.slide_logits_dict[slide])
			for i in range(num_vals):
				node_activations += [[self.AV.slide_node_dict[slide][n][i] for n in self.AV.nodes]]
				logits = [self.AV.slide_logits_dict[slide][l][i] for l in range(num_logits)]
				location = self.AV.slide_loc_dict[slide][i]
				# if prediction_filter is supplied, calculate prediction based on maximum value of allowed outcomes
				if prediction_filter:
					filtered_logits = [logits[l] for l in prediction_filter]
					prediction = logits.index(max(filtered_logits))
				elif logits:
					prediction = logits.index(max(logits))
				else:
					prediction = None

				self.point_meta += [{
					'slide': slide,
					'index': i,
					'prediction': prediction,
					'logits': logits,
					'loc': location
				}]

		coordinates = gen_umap(np.array(node_activations), 
							   n_components=dim,
							   n_neighbors=100,
							   min_dist=0.1,
							   low_memory=low_memory)

		self.x = np.array([c[0] for c in coordinates])
		if dim > 1:
			self.y = np.array([c[1] for c in coordinates])
		else:
			self.y = np.array([0 for i in range(len(self.x))])
		self.values = np.array(['None' for i in range(len(self.point_meta))])
		self.save_cache()

	def _calculate_from_slides(self, method='centroid', prediction_filter=None,
							   force_recalculate=False, low_memory=False):

		''' Internal function to guide calculation of UMAP from final layer activations for eadch tile, 
			as provided via ActivationsVisualizer nodes, and then map only the centroid tile for each slide.
			
		Args:
			method:					Either 'centroid' or 'average'. If centroid, will calculate UMAP only 
										from centroid tiles for each slide. If average, will calculate UMAP 
										based on average node activations across all tiles within the slide, 
										then display the centroid tile for each slide.
			prediction_filter:		(optional) List of int. If provided, will restrict predictions to only these categories.
			force_recalculate:		Bool, default=False. If true, will force recalculation of UMAP despite loading from cache.
			low_memory:				Bool, if True, will calculate UMAP in low-memory mode (less memory, more computations)
		'''

		if method not in ('centroid', 'average'):
			raise StatisticsError(f'Method must be either "centroid" or "average", not {method}')

		log.info("Calculating centroid indices...", 1)
		optimal_slide_indices, centroid_activations = calculate_centroid(self.AV.slide_node_dict)
		
		# Restrict mosaic to only slides that had enough tiles to calculate an optimal index from centroid
		successful_slides = list(optimal_slide_indices.keys())
		num_warned = 0
		warn_threshold = 3
		for slide in self.AV.slides:
			print_func = print if num_warned < warn_threshold else None
			if slide not in successful_slides:
				log.warn(f"Unable to calculate centroid for slide {sfutil.green(slide)}; will not include in Mosaic", 1, print_func)
				num_warned += 1
		if num_warned >= warn_threshold:
			log.warn(f"...{num_warned} total warnings, see {sfutil.green(log.logfile)} for details", 1)

		if len(self.x) and len(self.y) and not force_recalculate:
			log.info("UMAP loaded from cache, will filter to include only provided tiles", 1)
			new_x, new_y, new_meta = [], [], []
			for i in range(len(self.point_meta)):
				slide = self.point_meta[i]['slide']
				if slide in optimal_slide_indices and self.point_meta[i]['index'] == optimal_slide_indices[slide]:
					new_x += [self.x[i]]
					new_y += [self.y[i]]
					if prediction_filter:
						num_logits = len(self.AV.slide_logits_dict[slide])
						tile_index = self.point_meta[i]['index']
						logits = [self.AV.slide_logits_dict[slide][l][tile_index] for l in range(num_logits)]
						filtered_logits = [logits[l] for l in prediction_filter]
						prediction = logits.index(max(filtered_logits))
						meta = {
							'slide': slide,
							'index': tile_index,
							'logits': logits,
							'prediction': prediction,
						}
					else:
						meta = self.point_meta[i]
					new_meta += [meta]
			self.x = np.array(new_x)
			self.y = np.array(new_y)
			self.point_meta = np.array(new_meta)
			self.values = np.array(['None' for i in range(len(self.point_meta))])
		else:
			log.empty(f"Calculating UMAP from slide-level {method}...", 1)
			umap_input = []
			for slide in self.slides:
				if method == 'centroid':
					umap_input += [centroid_activations[slide]]
				elif method == 'average':
					activation_averages = [np.mean(self.AV.slide_node_dict[slide][n]) for n in self.AV.nodes]
					umap_input += [activation_averages]
				self.point_meta += [{
					'slide': slide,
					'index': optimal_slide_indices[slide],
					'logits': [],
					'prediction': 0
				}]

			coordinates = gen_umap(np.array(umap_input), n_neighbors=50, min_dist=0.1, low_memory=low_memory)
			self.x = np.array([c[0] for c in coordinates])
			self.y = np.array([c[1] for c in coordinates])
			self.values = np.array(['None' for i in range(len(self.point_meta))])
			self.save_cache()

	def cluster(self, n_clusters):
		'''Performs clustering on data and adds to metadata labels. Requires an ActivationsVisualizer backend. '''
		activations = [[self.AV.slide_node_dict[pm['slide']][n][pm['index']] for n in self.AV.nodes] for pm in self.point_meta]
		log.info(f"Calculating K-means clustering (n={n_clusters})", 1)
		kmeans = KMeans(n_clusters=n_clusters).fit(activations)
		labels = kmeans.labels_
		for i, label in enumerate(labels):
			self.point_meta[i]['cluster'] = label

	def export_to_csv(self, filename):
		'''Exports calculated UMAP coordinates to csv.'''
		with open(filename, 'w') as outfile:
			csvwriter = csv.writer(outfile)
			header = ['slide', 'index', 'x', 'y']
			csvwriter.writerow(header)
			for index in range(len(self.point_meta)):
				x = self.x[index]
				y = self.y[index]
				meta = self.point_meta[index]
				slide = meta['slide']
				index = meta['index']
				row = [slide, index, x, y]
				csvwriter.writerow(row)

	def calculate_neighbors(self, slide_categories=None, algorithm='kd_tree'):
		'''Calculates neighbors among tiles in this map, assigning neighboring statistics 
			to tile metadata 'num_unique_neighbors' and 'percent_matching_categories'.
		
		Args:
			slide_categories:	Optional, dict mapping slides to categories. If provided, will be used to 
									calculate 'percent_matching_categories' statistic.
			algorithm:			NearestNeighbor algorithm, either 'kd_tree', 'ball_tree', or 'brute'
		'''
		from sklearn.neighbors import NearestNeighbors
		log.empty("Initializing neighbor search...", 1)
		X = np.array([[self.AV.slide_node_dict[self.point_meta[i]['slide']][n][self.point_meta[i]['index']] for n in self.AV.nodes] for i in range(len(self.x))])
		nbrs = NearestNeighbors(n_neighbors=100, algorithm=algorithm, n_jobs=-1).fit(X)
		log.empty("Calculating nearest neighbors...", 1)
		distances, indices = nbrs.kneighbors(X)
		for i, ind in enumerate(indices):
			num_unique_slides = len(list(set([self.point_meta[_i]['slide'] for _i in ind])))
			self.point_meta[i]['num_unique_neighbors'] = num_unique_slides
			if slide_categories:
				percent_matching_categories = len([_i for _i in ind if slide_categories[self.point_meta[_i]['slide']] == slide_categories[self.point_meta[i]['slide']]])/len(ind)
				self.point_meta[i]['percent_matching_categories'] = percent_matching_categories

	def filter(self, slides):
		'''Filters map to only show tiles from the given slides.'''
		if not hasattr(self, 'full_x'):
			# Backup full coordinates
			self.full_x, self.full_y, self.full_meta = self.x, self.y, self.point_meta
		else:
			# Restore backed up full coordinates
			self.x, self.y, self.point_meta = self.full_x, self.full_y, self.full_meta
		
		self.point_meta = np.array([pm for pm in self.point_meta if pm['slide'] in slides])
		self.x = np.array([self.x[xi] for xi in range(len(self.x)) if self.point_meta[xi]['slide'] in slides])
		self.y = np.array([self.y[yi] for yi in range(len(self.y)) if self.point_meta[yi]['slide'] in slides])

	def show_neighbors(self, neighbor_AV, slide):
		'''Filters map to only show neighbors with a corresponding neighbor ActivationsVisualizer and neighbor slide.

		Args:
			neighbor_AV:		ActivationsVisualizer containing activations for neighboring slide
			slide:				Name of neighboring slide
		'''
		if slide not in neighbor_AV.slide_node_dict:
			raise StatisticsError(f"Slide {slide} not found in the provided ActivationsVisualizer, unable to find neighbors")
		if not hasattr(self, 'AV'):
			raise StatisticsError(f"TFRecordMap does not have a linked ActivationsVisualizer, unable to calculate neighbors")

		tile_neighbors = self.AV.find_neighbors(neighbor_AV, slide, n_neighbors=5)

		if not hasattr(self, 'full_x'):
			# Backup full coordinates
			self.full_x, self.full_y, self.full_meta = self.x, self.y, self.point_meta
		else:
			# Restore backed up full coordinates
			self.x, self.y, self.point_meta = self.full_x, self.full_y, self.full_meta

		self.x = np.array([self.x[i] for i in range(len(self.x)) if self.point_meta[i]['slide'] in tile_neighbors and self.point_meta[i]['index'] in tile_neighbors[self.point_meta[i]['slide']]])
		self.y = np.array([self.y[i] for i in range(len(self.y)) if self.point_meta[i]['slide'] in tile_neighbors and self.point_meta[i]['index'] in tile_neighbors[self.point_meta[i]['slide']]])
		self.meta = np.array([self.point_meta[i] for i in range(len(self.point_meta)) if self.point_meta[i]['slide'] in tile_neighbors and self.point_meta[i]['index'] in tile_neighbors[self.point_meta[i]['slide']]])

	def label_by_logits(self, index):
		'''Displays each point with label equal to the logits (linear from 0-1)

		Args:
			index:				Logit index
		'''
		self.values = np.array([m['logits'][index] for m in self.point_meta])

	def label_by_slide(self, slide_labels=None):
		'''Displays each point as the name of the corresponding slide. 
			If slide_labels is provided, will use this dictionary to label slides.

		Args:
			slide_labels:		(Optional) Dict mapping slide names to labels.
		'''
		if slide_labels:
			self.values = np.array([slide_labels[m['slide']] for m in self.point_meta])
		else:
			self.values = np.array([m['slide'] for m in self.point_meta])

	def label_by_tile_meta(self, tile_meta, translation_dict=None):
		'''Displays each point with label equal a value in tile metadata (e.g. 'prediction')

		Args:
			tile_meta:			String, key to metadata from which to read
			translation_dict:	Optional, if provided, will translate the read metadata through this dictionary
		'''
		if translation_dict:
			try:
				self.values = np.array([translation_dict[m[tile_meta]] for m in self.point_meta])
			except KeyError:
				# Try by converting metadata to string
				self.values = np.array([translation_dict[str(m[tile_meta])] for m in self.point_meta])
		else:
			self.values = np.array([m[tile_meta] for m in self.point_meta])

	def save_2d_plot(self, filename, subsample=None, title=None, cmap=None, 
					 use_float=False, xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), 
					 dpi=300, xlabel=None, ylabel=None, legend=None):
		'''Saves plot of data to a provided filename.

		Args:
			filename:		Saves image of plot to this file.
			slide_labels:	(optional) Dictionary mapping slide names to labels.
			slide_filter:	(optional) List, restricts map to the provided slides.
			show_tile_meta:	(optional) String (key), if provided, will label tiles 
								according to this key as provided in the tile-level meta (self.point_meta)
			outcome_labels:	(optional) Dictionary to translate outcomes (provided 
								via show_tile_meta and point_meta) to human readable label.
			subsample:		(optional) Int, if provided, will only 
								include this number of tiles on plot (randomly selected)
			title:			(optional) String, title for plot
			cmap:			(optional) Dicionary mapping labels to colors
			use_float:		(optional) Interpret labels as float for linear coloring'''	

		# Subsampling
		if subsample:
			ri = sample(range(len(self.x)), min(len(self.x), subsample))
		else:
			ri = list(range(len(self.x)))
		x = self.x[ri]
		y = self.y[ri]
		values = self.values[ri]

		# Prepare pandas dataframe
		df = pd.DataFrame()
		df['umap_x'] = x
		df['umap_y'] = y
		df['category'] = values if use_float else pd.Series(values, dtype='category')

		# Prepare color palette
		if use_float:
			cmap = None
			palette = None
		else:
			unique_categories = list(set(values))
			unique_categories.sort()
			if len(unique_categories) <= 12:
				seaborn_palette = sns.color_palette("Paired", len(unique_categories))
			else:
				seaborn_palette = sns.color_palette('hls', len(unique_categories))
			palette = {unique_categories[i]:seaborn_palette[i] for i in range(len(unique_categories))}

		# Make plot
		plt.clf()
		umap_2d = sns.scatterplot(x=x, y=y, data=df, hue='category', s=30, palette=cmap if cmap else palette)
		plt.gca().set_ylim(*((None, None) if not ylim else ylim))
		plt.gca().set_xlim(*((None, None) if not xlim else xlim))
		umap_2d.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1, title=legend)
		umap_2d.set(xlabel=xlabel, ylabel=ylabel)
		umap_figure = umap_2d.get_figure()
		umap_figure.set_size_inches(6, 4.5)
		if title: umap_figure.axes[0].set_title(title)
		umap_figure.canvas.start_event_loop(sys.float_info.min)
		umap_figure.savefig(filename, bbox_inches='tight', dpi=dpi)
		log.complete(f"Saved 2D UMAP to {sfutil.green(filename)}", 1)

		def onselect(verts):
			print(verts)

		lasso = LassoSelector(plt.gca(), onselect)

	def save_3d_plot(self, z=None, node=None, filename=None, subsample=None):
		'''Saves a plot of a 3D umap, with the 3rd dimension representing values provided by argument "z" 
		
		Args: 
			z: 			Values for z axis (optional).
			node:		Int, node to plot on 3rd axis (optional). Ignored if z is supplied.
			filename:	Filename to save image of plot
			subsample:	(optionanl) int, if provided will subsample data to include only this number of tiles as max'''

		title = f"UMAP with node {node} focus"

		if not filename:
			filename = "3d_plot.png"

		if (z is None) and (node is None):
			raise StatisticsError("Must supply either 'z' or 'node'.")

		# Get node activations for 3rd dimension
		if z is None:
			z = np.array([self.AV.slide_node_dict[m['slide']][node][m['index']] for m in self.point_meta])

		# Subsampling
		if subsample:
			ri = sample(range(len(self.x)), min(len(self.x), subsample))
		else:
			ri = list(range(len(self.x)))

		x = self.x[ri]
		y = self.y[ri]
		z = z[ri]

		# Plot tiles on a 3D coordinate space with 2 coordinates from UMAP & 3rd from the value of the excluded node
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter(x, y, z, c=z,
							cmap='viridis',
							linewidth=0.5,
							edgecolor="black")
		ax.set_title(title)
		log.info(f"Saving 3D UMAP to {sfutil.green(filename)}...", 1)
		plt.savefig(filename, bbox_inches='tight')

	def get_tiles_in_area(self, x_lower=-999, x_upper=999, y_lower=-999, y_upper=999):
		'''Returns dictionary of slide names mapping to tile indices, 
			or tiles that fall within the specified location on the umap.'''

		# Find tiles that meet UMAP location criteria
		filtered_tiles = {}
		num_selected = 0
		for i in range(len(self.point_meta)):
			if (x_lower < self.x[i] < x_upper) and (y_lower < self.y[i] < y_upper):
				slide = self.point_meta[i]['slide']
				tile_index = self.point_meta[i]['index']
				if slide not in filtered_tiles:
					filtered_tiles.update({slide: [tile_index]})
				else:
					filtered_tiles[slide] += [tile_index]
				num_selected += 1
		log.info(f"Selected {num_selected} tiles by filter criteria.", 1)
		return filtered_tiles

	def save_cache(self):
		if self.cache:
			try:
				with open(self.cache, 'wb') as cache_file:
					pickle.dump([self.x, self.y, self.point_meta, self.map_meta], cache_file)
					log.info(f"Wrote UMAP cache to {sfutil.green(self.cache)}", 1)
			except:
				log.info(f"Error attempting to write UMAP cache to {sfutil.green(self.cache)}", 1)

	def load_cache(self):
		try:
			with open(self.cache, 'rb') as cache_file:
				self.x, self.y, self.point_meta, self.map_meta = pickle.load(cache_file)
				log.info(f"Loaded UMAP cache from {sfutil.green(self.cache)}", 1)
				return True
		except FileNotFoundError:
			log.info(f"No UMAP cache found at {sfutil.green(self.cache)}", 1)
		return False

def generate_tile_roc(i, y_true, y_pred, data_dir, label_start, histogram=False):
	'''Generates tile-level ROC. Defined as a separate function for use with multiprocessing.'''
	try:
		roc_auc, average_precision, optimal_threshold = generate_roc(y_true[:, i], y_pred[:, i], data_dir, f'{label_start}tile_ROC{i}')
		if histogram:
			save_histogram(y_true[:, i], y_pred[:, i], data_dir, f'{label_start}tile_histogram{i}')			
	except IndexError:
		log.warn(f"Unable to generate tile-level stats for outcome {i}", 1)
		return None, None, None
	return roc_auc, average_precision, optimal_threshold

def get_average_by_group(prediction_array, prediction_label, unique_groups, tile_to_group, y_true_group,
							num_cat, label_end, save_predictions=False, data_dir=None, label='group'):
	'''Function to generate group-level averages (e.g. slide-level or patient-level).
	
	For a given tile-level prediction array, calculate spercent predictions 
	in each outcome by group (e.g. patient, slide), and saves to CSV if specified.'''

	groups = {g:[] for g in unique_groups}

	def update_group(ttg):
		nonlocal groups
		i, g = ttg
		groups[g] += [prediction_array[i]]

	with mp.dummy.Pool(processes=16) as p:
		p.map(update_group, enumerate(tile_to_group))

	group_percents = {g:np.array(groups[g]).mean(axis=0) for g in unique_groups}
	avg_by_group = np.array([group_percents[g] for g in unique_groups])

	if save_predictions:
		save_path = join(data_dir, f"{label}_predictions{label_end}.csv")
		with open(save_path, 'w') as outfile:
			writer = csv.writer(outfile)
			header = [label] + [f"y_true{i}" for i in range(num_cat)] + [f"{prediction_label}{j}" for j in range(num_cat)]
			writer.writerow(header)
			for i, group in enumerate(unique_groups):
				row = np.concatenate([ [group], y_true_group[group], avg_by_group[i] ])
				writer.writerow(row)
	return avg_by_group

def get_centroid_index(input_array):
	'''Calculate index of centroid from a given two-dimensional input array.'''
	km = KMeans(n_clusters=1).fit(input_array)
	closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, input_array)
	return closest[0]

def calculate_centroid(slide_node_dict):
	'''Calcultes slide-level centroid indices for a provided slide-node dict.'''
	optimal_indices = {}
	centroid_activations = {}
	nodes = list(slide_node_dict[list(slide_node_dict.keys())[0]].keys())
	for slide in slide_node_dict:
		slide_nodes = slide_node_dict[slide]
		# Reorganize "slide_nodes" into an array of node activations for each tile
		# Final size of array should be (num_nodes, num_tiles_in_slide) 
		activations = [[slide_nodes[n][i] for n in nodes] for i in range(len(slide_nodes[0]))]
		if not len(activations): continue
		km = KMeans(n_clusters=1).fit(activations)
		closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, activations)
		closest_index = closest[0]
		closest_activations = activations[closest_index]

		optimal_indices.update({slide: closest_index})
		centroid_activations.update({slide: closest_activations})
	return optimal_indices, centroid_activations

def normalize_layout(layout, min_percentile=1, max_percentile=99, relative_margin=0.1):
	"""Removes outliers and scales layout to between [0,1]."""

	# compute percentiles
	mins = np.percentile(layout, min_percentile, axis=(0))
	maxs = np.percentile(layout, max_percentile, axis=(0))

	# add margins
	mins -= relative_margin * (maxs - mins)
	maxs += relative_margin * (maxs - mins)

	# `clip` broadcasts, `[None]`s added only for readability
	clipped = np.clip(layout, mins, maxs)

	# embed within [0,1] along both axes
	clipped -= clipped.min(axis=0)
	clipped /= clipped.max(axis=0)

	return clipped

def gen_umap(array, n_components=2, n_neighbors=20, min_dist=0.01, metric='cosine', low_memory=False):
	'''Generates and returns a umap from a given array.'''
	try:
		layout = umap.UMAP(n_components=n_components,
						   verbose=(log.INFO_LEVEL > 0),
						   n_neighbors=n_neighbors,
						   min_dist=min_dist,
						   metric=metric,
						   low_memory=low_memory).fit_transform(array)
	except ValueError:
		raise StatisticsError("Error performing UMAP. Please make sure you are supplying a non-empty TFRecord array and that the TFRecords are not empty.")
		
	return normalize_layout(layout)

def save_histogram(y_true, y_pred, data_dir, name='histogram'):
	'''Generates histogram of y_pred, labeled by y_true'''

	cat_false = [yp for i, yp in enumerate(y_pred) if y_true[i] == 0]
	cat_true = [yp for i, yp in enumerate(y_pred) if y_true[i] == 1]

	plt.clf()
	plt.title('Tile-level Predictions')
	try:
		sns.displot( cat_false , color="skyblue", label="Negative")
		sns.displot( cat_true , color="red", label="Positive")
	except np.linalg.LinAlgError:
		log.warn("Unable to generate histogram, insufficient data", 1)
	plt.legend()
	plt.savefig(os.path.join(data_dir, f'{name}.png'))

def to_onehot(val, num_cat):
	'''Converts value to one-hot encoding
	
	Args:
		val:		Value to encode
		num_cat:	Maximum value (length of onehot encoding)'''

	onehot = [0] * num_cat
	onehot[val] = 1
	return onehot

def generate_roc(y_true, y_pred, save_dir=None, name='ROC'):
	'''Generates and saves an ROC with a given set of y_true, y_pred values.'''
	# ROC
	try:
		fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
	except:
		log.error("Error with ROC curve:")
		print(y_true)
		print(y_pred)
		sys.exit()
	roc_auc = metrics.auc(fpr, tpr)

	# Precision recall
	precision, recall, pr_thresholds = metrics.precision_recall_curve(y_true, y_pred)
	average_precision = metrics.average_precision_score(y_true, y_pred)

	# Calculate optimal cutoff via maximizing Youden's J statistic (sens+spec-1, or TPR - FPR)
	try:
		optimal_threshold = threshold[list(zip(tpr,fpr)).index(max(zip(tpr,fpr), key=lambda x: x[0]-x[1]))]
	except:
		optimal_threshold = -1

	# Plot
	if save_dir:
		# ROC
		plt.clf()
		plt.title('ROC Curve')
		plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
		plt.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.ylabel('TPR')
		plt.xlabel('FPR')
		plt.savefig(os.path.join(save_dir, f'{name}.png'))
		
		# Precision recall
		plt.clf()
		plt.title('Precision-Recall Curve')
		plt.plot(precision, recall, 'b', label = 'AP = %0.2f' % average_precision)
		plt.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.ylabel('Recall')
		plt.xlabel('Precision')
		plt.savefig(os.path.join(save_dir, f'{name}-PRC.png'))
	return roc_auc, average_precision, optimal_threshold

def generate_combined_roc(y_true, y_pred, save_dir, labels, name='ROC'):
	'''Generates and saves overlapping ROCs with a given combination of y_true and y_pred.'''
	# Plot
	plt.clf()
	plt.title(name)
	colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

	rocs = []
	for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
		fpr, tpr, threshold = metrics.roc_curve(yt, yp)
		roc_auc = metrics.auc(fpr, tpr)
		rocs += [roc_auc]
		plt.plot(fpr, tpr, colors[i % len(colors)], label = labels[i] + f' (AUC: {roc_auc:.2f})')	

	# Finish plot
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('TPR')
	plt.xlabel('FPR')
	
	plt.savefig(os.path.join(save_dir, f'{name}.png'))
	return rocs

def read_predictions(predictions_file, level):
	'''Reads predictions from a previously saved CSV file.'''
	predictions = {}
	y_pred_label = "percent_tiles_positive" if level in ("patient", "slide") else "y_pred"
	with open(predictions_file, 'r') as csvfile:
		reader = csv.reader(csvfile)
		header = next(reader)
		prediction_labels = [h.split('y_true')[-1] for h in header if "y_true" in h]
		for label in prediction_labels:
			predictions.update({label: {
				'y_true': [],
				'y_pred': []
			}})
		for row in reader:
			for label in prediction_labels:
				yti = header.index(f'y_true{label}')
				ypi = header.index(f'{y_pred_label}{label}')
				predictions[label]['y_true'] += [int(row[yti])]
				predictions[label]['y_pred'] += [float(row[ypi])]
	return predictions

def generate_scatter(y_true, y_pred, data_dir, name='_plot', plot=True):
	'''Generate and save scatter plots and calculate R2 statistic for each outcome variable.
		y_true and y_pred are both 2D arrays; the first dimension is each observation, 
		the second dimension is each outcome variable.'''

	# Error checking
	if y_true.shape != y_pred.shape:
		log.error(f"Y_true (shape: {y_true.shape}) and y_pred (shape: {y_pred.shape}) must have the same shape to generate a scatter plot", 1)
		return
	if y_true.shape[0] < 2:
		log.error(f"Must have more than one observation to generate a scatter plot with R2 statistics.", 1)
		return

	# Perform scatter for each outcome variable
	r_squared = []
	for i in range(y_true.shape[1]):

		# Statistics
		slope, intercept, r_value, p_value, std_err = stats.linregress(y_true[:,i], y_pred[:,i])
		r_squared += [r_value ** 2]

		if plot:
			# Plot
			p = sns.jointplot(y_true[:,i], y_pred[:,i], kind="reg")
			p.set_axis_labels('y_true', 'y_pred')
			plt.savefig(os.path.join(data_dir, f'Scatter{name}-{i}.png'))

	return r_squared

def basic_metrics(y_true, y_pred):
	'''Generates basic performance metrics, including sensitivity, specificity, and accuracy.'''
	assert(len(y_true) == len(y_pred))
	assert([y in [0,1] for y in y_true])
	assert([y in [0,1] for y in y_pred])

	TP = 0 # True positive
	TN = 0 # True negative
	FP = 0 # False positive
	FN = 0 # False negative

	for i, yt in enumerate(y_true):
		yp = y_pred[i]
		if yt == 1 and yp == 1:
			TP += 1
		elif yt == 1 and yp == 0:
			FN += 1
		elif yt == 0 and yp == 1:
			FP += 1
		elif yt == 0 and yp == 0:
			TN += 1

	accuracy = (TP + TN) / (TP + TN + FP + FN)
	sensitivity = TP / (TP + FN)
	specificity = TN / (TN + FP)

	# Additional metrics with sklearn
	precision = metrics.precision_score(y_true, y_pred)
	recall = metrics.recall_score(y_true, y_pred)
	f1_score = metrics.f1_score(y_true, y_pred)
	kappa = metrics.cohen_kappa_score(y_true, y_pred)

	return accuracy, sensitivity, specificity, precision, recall, f1_score, kappa

def concordance_index(y_true, y_pred):
	E = y_pred[:, -1]  # HERE
	y_pred = y_pred[:, :-1]  # HERE
	y_pred = y_pred.flatten()
	E = E.flatten()
	y_true = y_true.flatten()
	y_pred = - y_pred # Need to take negative to get concordance index since these are log hazard ratios
	return c_index(y_true, y_pred, E)

def _cph_metrics(args):
	'''Internal function to calculate tile, slide, and patient level metrics for a CPH outcome.'''
	# Detect number of outcome categories
	num_cat = args.y_pred.shape[1] 

	# Generate c_index
	args.c_index['tile'] = concordance_index(args.y_true, args.y_pred)
	
	# Generate and save slide-level averages of each outcome
	averages_by_slide = get_average_by_group(args.y_pred, 
											prediction_label="average",
											unique_groups=args.unique_slides,
											tile_to_group=args.tile_to_slides,
											y_true_group=args.y_true_slide,
											num_cat=num_cat,
											label_end=args.label_end,
											save_predictions=args.save_slide_predictions,
											data_dir=args.data_dir,
											label="slide")
	y_true_by_slide = np.array([args.y_true_slide[slide] for slide in args.unique_slides])
	args.c_index['slide'] = concordance_index(y_true_by_slide, averages_by_slide)
	if not args.patient_error:
		# Generate and save patient-level averages of each outcome
		averages_by_patient = get_average_by_group(args.y_pred,
													prediction_label="average",
													unique_groups=args.patients,
													tile_to_group=args.tile_to_patients,
													y_true_group=args.y_true_patient,
													num_cat=num_cat,
													label_end=args.label_end,
													save_predictions=args.save_patient_predictions,
													data_dir=args.data_dir,
													label="slide")
		y_true_by_patient = np.array([args.y_true_patient[patient] for patient in args.patients])
		args.c_index['patient'] = concordance_index(y_true_by_patient, averages_by_patient)	

def _linear_metrics(args):
	'''Internal function to calculate tile, slide, and patient level metrics for a linear outcome.'''
	# Detect number of outcome categories
	num_cat = args.y_pred.shape[1] 

	# Main loop
	# Generate R-squared
	args.r_squared['tile'] = generate_scatter(args.y_true, args.y_pred, args.data_dir, args.label_end, plot=args.plot)

	# Generate and save slide-level averages of each outcome
	averages_by_slide = get_average_by_group(args.y_pred,
											prediction_label="average",
											unique_groups=args.unique_slides,
											tile_to_group=args.tile_to_slides,
											y_true_group=args.y_true_slide,
											num_cat=num_cat,
											label_end=args.label_end,
											save_predictions=args.save_slide_predictions,
											data_dir=args.data_dir,
											label="slide")
	y_true_by_slide = np.array([args.y_true_slide[slide] for slide in args.unique_slides])
	args.r_squared['slide'] = generate_scatter(y_true_by_slide, averages_by_slide, args.data_dir, args.label_end+"_by_slide")			
	if not args.patient_error:
		# Generate and save patient-level averages of each outcome
		averages_by_patient = get_average_by_group(args.y_pred,
													prediction_label="average",
													unique_groups=args.patients,
													tile_to_group=args.tile_to_patients,
													y_true_group=args.y_true_patient, 
													num_cat=num_cat,
													label_end=args.label_end,
													save_predictions=args.save_patient_predictions,
													data_dir=args.data_dir,
													label="slide")

		y_true_by_patient = np.array([args.y_true_patient[patient] for patient in args.patients])
		args.r_squared['patient'] = generate_scatter(y_true_by_patient, averages_by_patient, args.data_dir, args.label_end+"_by_patient")

def _categorical_metrics(args, outcome_name, starttime=None):
	'''Internal function to calculate tile, slide, and patient level metrics for a categorical outcome.'''
	start = starttime
	num_observed_outcome_categories = max(args.y_true)+1
	if num_observed_outcome_categories != args.y_pred.shape[1]:
		log.warn(f"Model predictions have different number of outcome categories ({args.y_pred.shape[1]}) than provided annotations ({num_observed_outcome_categories})!", 1)

	num_cat = max(num_observed_outcome_categories, args.y_pred.shape[1])

	# For categorical models, convert to one-hot encoding
	args.y_true = np.array([to_onehot(i, num_cat) for i in args.y_true])
	args.y_true_slide = {k:to_onehot(v, num_cat) for k,v in args.y_true_slide.items()}
	args.y_true_patient = {k:to_onehot(v, num_cat) for k,v in args.y_true_patient.items()}

	args.auc['tile'][outcome_name] = []
	args.auc['slide'][outcome_name] = []
	args.auc['patient'][outcome_name] = []

	with mp.Pool(processes=8) as p:
		# TODO: this is memory inefficient as it copies y_true / y_pred to each subprocess
		# Furthermore, it copies all categories when only one category is needed for each process
		# Consider implementing shared memory, ideally compatible with python 3.7
		for i, (auc, ap, thresh) in enumerate(p.imap(partial(generate_tile_roc, y_true=args.y_true,
																				y_pred=args.y_pred,
																				data_dir=args.data_dir,
																				label_start=args.label_start + outcome_name + "_",
																				histogram=args.histogram), range(num_cat))):
			args.auc['tile'][outcome_name] += [auc]
			if args.verbose:		
				log.info(f"Tile-level AUC (cat #{i:>2}): {auc:.3f}, AP: {ap:.3f} (opt. threshold: {thresh:.3f})", 1)

	# Convert predictions to one-hot encoding
	onehot_predictions = np.array([to_onehot(x, num_cat) for x in np.argmax(args.y_pred, axis=1)])

	# Compare one-hot predictions to one-hot y_true for category-level accuracy
	split_predictions = np.split(onehot_predictions, num_cat, 1)
	for ci, cat_pred_array in enumerate(split_predictions):
		try:
			y_true_in_category = args.y_true[:, ci]
			num_tiles_in_category = np.sum(y_true_in_category)
			num_correctly_predicted_in_category = np.sum(cat_pred_array[np.argwhere(y_true_in_category>0)])
			category_accuracy = num_correctly_predicted_in_category / num_tiles_in_category
			cat_percent_acc = category_accuracy * 100
			if args.verbose:
				log.info(f"Category {ci} accuracy: {cat_percent_acc:.1f}% ({num_correctly_predicted_in_category}/{num_tiles_in_category})", 1)
		except IndexError:
			log.warn(f"Unable to generate category-level accuracy stats for category index {ci}", 1)

	# Generate slide-level percent calls
	percent_calls_by_slide = get_average_by_group(onehot_predictions,
												  prediction_label="percent_tiles_positive",
												  unique_groups=args.unique_slides,
												  tile_to_group=args.tile_to_slides,
												  y_true_group=args.y_true_slide,
												  num_cat=num_cat,
												  label_end="_" + outcome_name + args.label_end,
												  save_predictions=args.save_slide_predictions,
												  data_dir=args.data_dir,
												  label="slide")

	# Generate slide-level ROC
	for i in range(num_cat):
		try:
			slide_y_pred = percent_calls_by_slide[:, i]
			slide_y_true = [args.y_true_slide[slide][i] for slide in args.unique_slides]
			roc_auc, average_precision, optimal_threshold = generate_roc(slide_y_true, slide_y_pred, args.data_dir, f'{args.label_start}{outcome_name}_slide_ROC{i}')
			args.auc['slide'][outcome_name] += [roc_auc]
			if args.verbose:
				log.info(f"Slide-level AUC (cat #{i:>2}): {roc_auc:.3f}, AP: {average_precision:.3f} (opt. threshold: {optimal_threshold:.3f})", 1)
		except IndexError:
			log.warn(f"Unable to generate slide-level stats for outcome {i}", 1)

	if not args.patient_error:
		# Generate patient-level percent calls
		percent_calls_by_patient = get_average_by_group(onehot_predictions,
														prediction_label="percent_tiles_positive",
														unique_groups=args.patients,
														tile_to_group=args.tile_to_patients,
														y_true_group=args.y_true_patient,
														num_cat=num_cat,
														label_end="_" + outcome_name + args.label_end,
														save_predictions=args.save_patient_predictions,
														data_dir=args.data_dir,
														label="slide")

		# Generate patient-level ROC
		for i in range(num_cat):
			try:
				patient_y_pred = percent_calls_by_patient[:, i]
				patient_y_true = np.array([args.y_true_patient[patient][i] for patient in args.patients])
				roc_auc, average_precision, optimal_threshold = generate_roc(patient_y_true, patient_y_pred, args.data_dir, f'{args.label_start}{outcome_name}_patient_ROC{i}')
				args.auc['patient'][outcome_name] += [roc_auc]
				if args.verbose:
					log.info(f"Patient-level AUC (cat #{i:>2}): {roc_auc:.3f}, AP: {average_precision:.3f} (opt. threshold: {optimal_threshold:.3f})", 1)
			except IndexError:
				log.warn(f"Unable to generate patient-level stats for outcome {i}", 1)

def save_predictions_to_csv(y_true, y_pred, tile_to_slides, data_dir, label_end, outcome_names=None):
	# Save tile-level predictions
	if type(y_true) == list:
		assert len(y_true) == len(y_pred), "Number of outcomes in y_true and y_pred must match"
		assert len(y_true) == len(outcome_names), "Number of provided outcome names must equal the number of y_true outcomes"

	tile_csv_dir = os.path.join(data_dir, f"tile_predictions{label_end}.csv")
	with open(tile_csv_dir, 'w') as outfile:
		writer = csv.writer(outfile)
		# If multiple outcomes are present
		if type(y_true) == list:
			y_true_is_reduced = (len(y_true[0].shape) == 1)
			y_pred_is_reduced = (len(y_pred[0].shape) == 1)
			y_true_header = []
			y_pred_header = []
			for cat_i in range(len(y_true)):
				y_true_cat_header = [f"{outcome_names[cat_i]}_y_true0"] if y_true_is_reduced else [f"{outcome_names[cat_i]}_y_true{i}" for i in range(y_true[cat_i].shape[1])]
				y_pred_cat_header = [f"{outcome_names[cat_i]}_y_pred0"] if y_pred_is_reduced else [f"{outcome_names[cat_i]}_y_pred{i}" for i in range(y_pred[cat_i].shape[1])]
				y_true_header += y_true_cat_header
				y_pred_header += y_pred_cat_header
			header = ['slide'] + y_true_header + y_pred_header
			writer.writerow(header)
			for i in range(len(y_true)):
				y_true_str_list = []
				y_pred_str_list = []
				for cat_i in range(len(y_true)):
					y_true_str_cat_list = [str(y_true[cat_i][i])] if y_true_is_reduced else [str(yti) for yti in y_true[cat_i][i]]
					y_pred_str_cat_list = [str(y_pred[cat_i][i])] if y_pred_is_reduced else [str(ypi) for ypi in y_pred[cat_i][i]]
					y_true_str_list += y_true_str_cat_list
					y_pred_str_list += y_pred_str_cat_list
				row = np.concatenate([[tile_to_slides[i]], y_true_str_list, y_pred_str_list])
				writer.writerow(row)
		# If there is only a single outcome
		else:
			y_true_is_reduced = (len(y_true.shape) == 1)
			y_pred_is_reduced = (len(y_pred.shape) == 1)
			y_true_header = ["y_true0"] if y_true_is_reduced else [f"y_true{i}" for i in range(y_true.shape[1])]
			header = ['slide'] + y_true_header + [f"y_pred{j}" for j in range(y_pred.shape[1])]
			writer.writerow(header)
			for i in range(len(y_true)):
				y_true_str_list = [str(y_true[i])] if y_true_is_reduced else [str(yti) for yti in y_true[i]]
				y_pred_str_list = [str(y_pred[i])] if y_pred_is_reduced else [str(ypi) for ypi in y_pred[i]]
				row = np.concatenate([[tile_to_slides[i]], y_true_str_list, y_pred_str_list])
				writer.writerow(row)
	log.complete(f"Predictions saved to {sfutil.green(data_dir)}", 1)

def metrics_from_predictions(y_true,
							 y_pred,
							 tile_to_slides,
							 annotations,
							 model_type,
							 manifest,
							 outcome_names=None,
							 label=None,
							 min_tiles_per_slide=0,
							 data_dir=None,
							 verbose=True,
							 save_predictions=True,
							 histogram=False,
							 plot=True):
	'''
		For multiple outcomes, y_true and y_pred are expected to be a list of numpy arrays (each numpy array corresponding to whole-dataset predictions for a single outcome)
	'''

	start = time.time()
	label_end = "" if not label else f"_{label}"
	label_start = "" if not label else f"{label}_"

	tile_to_patients = np.array([annotations[slide][sfutil.TCGA.patient] for slide in tile_to_slides])
	patients = np.unique(tile_to_patients)
	unique_slides = np.unique(tile_to_slides)

	# Filter out slides not meeting minimum tile number criteria, if specified
	slides_to_filter = []
	num_total_slides = len(unique_slides)
	if manifest:
		for tfrecord in manifest:
			tfrecord_name = sfutil.path_to_name(tfrecord)
			num_tiles_tfrecord = manifest[tfrecord]['total']
			if num_tiles_tfrecord < min_tiles_per_slide:
				if verbose:	log.info(f"Filtering out {tfrecord_name}: {num_tiles_tfrecord} tiles", 2)
				slides_to_filter += [tfrecord_name]
	else:
		log.warn("Manifest not provided, unable to filter tfrecords by min_tiles_per_slide", 1)
	unique_slides = [us for us in unique_slides if us not in slides_to_filter]
	if verbose:
		log.info(f"Filtered out {num_total_slides - len(unique_slides)} of {num_total_slides} slides in evaluation set (minimum tiles per slide: {min_tiles_per_slide})", 1)

	# Set up annotations
	y_true_slide = {s: annotations[s]['outcome_label'] for s in annotations}
	y_true_patient = {annotations[s][sfutil.TCGA.patient]: annotations[s]['outcome_label'] for s in annotations}

	# Verify patient outcomes are consistent if multiples slides are present for each patient
	patient_error = False
	for slide in annotations:
		patient = annotations[slide][sfutil.TCGA.patient]
		if  y_true_slide[slide] != y_true_patient[patient]:
			log.error("Data integrity failure when generating ROCs; patient assigned to multiple slides with different outcomes", 1)
			patient_error = True

	# Function to determine which predictions, if any, should be exported to CSV
	def should_save_predictions(group):
		return (save_predictions == True or 
			   (type(save_predictions) == str and save_predictions == group) or
			   (type(save_predictions) == list and group in save_predictions))

	metric_args = types.SimpleNamespace(
		y_true = y_true,
		y_pred = y_pred,
		unique_slides = unique_slides,
		tile_to_slides = tile_to_slides,
		tile_to_patients = tile_to_patients,
		label_start = label_start,
		label_end = label_end,
		save_slide_predictions = should_save_predictions('slide'),
		save_patient_predictions = should_save_predictions('patient'),
		save_tile_predictions = should_save_predictions('tile'),
		data_dir = data_dir,
		patient_error = patient_error,
		patients = patients,
		r_squared = {'tile': None, 'slide': None, 'patient': None},
		c_index = {'tile': None, 'slide': None, 'patient': None},
		auc = {'tile': {}, 'slide': {}, 'patient': {}},
		plot = plot,
		histogram = histogram,
		verbose = verbose,
	)

	if model_type == 'categorical':

		# Detect the number of outcomes by y_true
		if type(y_true) == list:
			num_outcomes_by_y_true = len(y_true)
		elif len(y_true.shape) == 1:
			num_outcomes_by_y_true = 1
		else:
			raise StatisticsError(f"y_true expected to be formated as list of numpy arrays for each outcome category.")

		# Confirm that the number of outcomes provided by y_true match the provided outcome names
		if not outcome_names:
			outcome_names = {f"Outcome {i}" for i in range(num_outcomes_by_y_true)}
		elif len(outcome_names) != num_outcomes_by_y_true:
			raise StatisticsError(f"Number of outcome names {len(outcome_names)} does not match y_true {num_outcomes_by_y_true}")

		for oi, outcome in enumerate(outcome_names):
			if len(outcome_names) > 1:
				metric_args.y_true_slide = {s:v[oi] for s,v in y_true_slide.items()}
				metric_args.y_true_patient = {s:v[oi] for s,v in y_true_patient.items()}
				metric_args.y_pred = y_pred[oi]
				metric_args.y_true = y_true[oi]
			else: 
				metric_args.y_true_slide = y_true_slide
				metric_args.y_true_patient = y_true_patient
				metric_args.y_pred = y_pred
				metric_args.y_true = y_true

			log.info(f"Validation metrics for outcome {sfutil.green(outcome)}")
			_categorical_metrics(metric_args, outcome, starttime=start)

	elif model_type == 'linear':
		metric_args.y_true_slide = y_true_slide
		metric_args.y_true_patient = y_true_patient
		_linear_metrics(metric_args)

	elif model_type == 'cph': 
		metric_args.y_true_slide = y_true_slide
		metric_args.y_true_patient = y_true_patient
		_cph_metrics(metric_args)

	if metric_args.save_tile_predictions:
		try:
			save_predictions_to_csv(y_true, y_pred, tile_to_slides, data_dir, label_end, outcome_names)
		except:
			log.error("Unable to save predictions to CSV - not yet implemented for multiple outcomes")
		
	return metric_args.auc, metric_args.r_squared, metric_args.c_index

def predict_from_model(model, dataset, num_tiles=0):
	'''Generates predictions (y_true, y_pred, tile_to_slide) from a given model and dataset.'''
	import tensorflow as tf

	@tf.function
	def get_predictions(img):
		return model(img, training=False)

	start = time.time()
	y_true, y_pred, tile_to_slides = [], [], []
	detected_batch_size = 0
	if log.INFO_LEVEL > 0:
		pb = ProgressBar(num_tiles, counter_text='images', leadtext="Generating predictions... ", show_counter=True, show_eta=True) if num_tiles else None
	else:
		pb = None

	# Get predictions and performance metrics
	for i, (img, yt, slide) in enumerate(dataset):
		if pb:
			pb.increase_bar_value(detected_batch_size)
		elif log.INFO_LEVEL > 0:
			sys.stdout.write(f"\rGenerating predictions (batch {i})...")
			sys.stdout.flush()
		
		y_pred += [get_predictions(img)]
		
		if type(yt) == dict:
			y_true += [[yt[f'out-{o}'].numpy() for o in range(len(yt))]]
		else:
			y_true += [yt.numpy()]

		tile_to_slides += [slide_bytes.decode('utf-8') for slide_bytes in slide.numpy()]
		if not detected_batch_size: detected_batch_size = len(tile_to_slides)

	if pb: pb.end()
	if log.INFO_LEVEL > 0: sfutil.clear_console()

	tile_to_slides = np.array(tile_to_slides)
	if type(y_pred[0]) == list:
		# Concatenate predictions for each outcome
		y_pred = [np.concatenate(yp) for yp in zip(*y_pred)]
	else:
		y_pred = np.concatenate(y_pred)
	if type(y_true[0]) == list:
		# Concatenate y_true for each outcome
		y_true = [np.concatenate(yt) for yt in zip(*y_true)]
	else:
		y_true = np.concatenate(y_true)

	end = time.time()
	log.info(f"Prediction complete. Time to completion: {int(end-start)} s", 1)

	return y_true, y_pred, tile_to_slides

def predict_from_layer(model, layer_input, input_layer_name='hidden_0', ouput_layer_index=None):
	import tensorflow as tf
	from slideflow.model_utils import get_layer_index_by_name

	first_hidden_layer_index = get_layer_index_by_name(model, input_layer_name)
	input_shape = model.layers[first_hidden_layer_index].get_input_shape_at(0) # get the input shape of desired layer
	x = input_tensor = tf.keras.Input(shape=input_shape) # a new input tensor to be able to feed the desired layer

	# create the new nodes for each layer in the path
	# For CPH models, include hidden layers excluding the final concatenation
	# 	(softmax + event tensor) layer
	if ouput_layer_index is not None:
		for layer in model.layers[first_hidden_layer_index:ouput_layer_index]:
			x = layer(x)	
	else:
		for layer in model.layers[first_hidden_layer_index:]:
			x = layer(x)

	# create the model
	new_model = tf.keras.Model(input_tensor, x)
	y_pred = new_model.predict(layer_input)
	return y_pred

def metrics_from_dataset(model,
						 model_type,
						 annotations,
						 manifest,
						 dataset,
						 outcome_names=None,
						 label=None,
						 min_tiles_per_slide=0,
						 data_dir=None,
						 num_tiles=0,
						 histogram=False,
						 verbose=True,
						 save_predictions=True):

	'''Evaluate performance of a given model on a given TFRecord dataset, 
	generating a variety of statistical outcomes and graphs.

	Args:
		model						Keras model to evaluate
		dataset		TFRecord dataset which include three items: raw image data, labels, and slide names.
		annotations					dictionary mapping slidenames to patients (TCGA.patient) and outcomes (outcome)
		model_type					'linear' or 'categorical'
		data_dir					directory in which to save performance metrics and graphs
		label						(optional) label with which to annotation saved files and graphs
		manifest					(optional) manifest as provided by Dataset, used to filter slides that do not have minimum number of tiles
		min_tiles_per_slide			(optional) if provided, will only perform calculations on slides that have a given minimum number of tiles
		num_tiles					(optional) total number of tiles across dataset, used for progress bar.

	Returns:
		auc, r_squared, c_index
	'''

	y_true, y_pred, tile_to_slides = predict_from_model(model, dataset, num_tiles=num_tiles)

	before_metrics = time.time()
	metrics = metrics_from_predictions(y_true=y_true,
										y_pred=y_pred,
										tile_to_slides=tile_to_slides,
										annotations=annotations,
										model_type=model_type,
										manifest=manifest,
										outcome_names=outcome_names,
										label=label,
										min_tiles_per_slide=min_tiles_per_slide,
										data_dir=data_dir,
										verbose=verbose,
										save_predictions=save_predictions,
										histogram=histogram,
										plot=True)
	after_metrics = time.time()
	log.info(f'Validation metrics generated, time: {after_metrics-before_metrics:.2f} s')
	return metrics
	
def permutation_feature_importance(model,
								   dataset_with_slidenames,
								   annotations,
								   model_type,
								   data_dir,
								   outcome_names=None,
								   label=None,
								   manifest=None,
								   min_tiles_per_slide=0,
								   num_tiles=0,
								   feature_names=None,
								   feature_sizes=None,
								   drop_images=False):
								   
	'''Calculate metrics (tile, slide, and patient AUC) from a given model that accepts clinical, slide-level feature 
		inputs, and permute to find relative feature performance.

	Args:
		model						Keras model to evaluate
		dataset_with_slidenames		TFRecord dataset which include three items: raw image data, labels, and slide names.
		annotations					dictionary mapping slidenames to patients (TCGA.patient) and outcomes (outcome)
		model_type					'linear' or 'categorical'
		data_dir					directory in which to save performance metrics and graphs
		label						(optional) label with which to annotate saved files and graphs
		manifest					(optional) manifest as provided by Dataset, used to filter slides that do not have minimum number of tiles
		min_tiles_per_slide			(optional) if provided, will only perform calculations on slides that have a given minimum number of tiles
		num_tiles					(optional) total number of tiles across dataset, used for progress bar.
		feature_names				Names for each of the clinical input features.
		feature_sizes				Sizes for each of the clinical input features.
		drop_images					Bool. If True, will exclude images from model (making predictions from clinical features alone)

	Returns:
		Dictiory of AUCs with keys 'tile', 'slide', and 'patient'

	'''
	import tensorflow as tf
	
	y_true = [] # True outcomes for each tile
	tile_to_slides = [] # Associated slide name for each tile
	pre_hl = [] # Activations pre-hidden layers for each tile
	detected_batch_size = 0
	metrics = {}
	
	# Establish the output layer for the intermediate model.
	#   This layer is just prior to the hidden layers, and includes
	#   input from clinical features (if present) merged with 
	#   post-convolution activations from image data (if present)
	hidden_layer_input = "slide_feature_input" if drop_images else "input_merge"
	intermediate_layer_model = tf.keras.Model(inputs=model.input,
									 		  outputs=model.get_layer(hidden_layer_input).output)
	# Setup progress bar
	pb = None
	if log.INFO_LEVEL > 0:
		msg = f"Generating model activations at layer '{hidden_layer_input}'..."
		sys.stdout.write(f"\r{msg}")
		if num_tiles:
			pb = ProgressBar(num_tiles,
							 counter_text='images',
							 leadtext=msg,
							 show_counter=True,
							 show_eta=True)

	# Create the time-to-event input used for CPH models
	if model_type == 'cph':
		event_input = tf.keras.Model(inputs=model.input, outputs=model.get_layer("event_input").output)
		events = []

	# For all tiles, calculate the intermediate layer (pre-hidden layer) activations,
	# 	and if a CPH model is being used, include time-to-event data
	for i, batch in enumerate(dataset_with_slidenames):
		if pb: pb.increase_bar_value(detected_batch_size)
		elif log.INFO_LEVEL > 0:
			sys.stdout.write(f"\rGenerating predictions (batch {i})...")
			sys.stdout.flush()
		if not detected_batch_size: detected_batch_size = len(batch[1].numpy())

		tile_to_slides += [slide_bytes.decode('utf-8') for slide_bytes in batch[2].numpy()]
		y_true += [batch[1].numpy()]
		pre_hl += [intermediate_layer_model.predict_on_batch(batch[0])]
		if model_type == 'cph':
			events += [event_input.predict_on_batch(batch[0])]
	
	# Concatenate arrays
	pre_hl = np.concatenate(pre_hl)
	if model_type == 'cph':
		events = np.concatenate(events)
	y_true = np.concatenate(y_true)
	tile_to_slides = np.array(tile_to_slides)
	
	if log.INFO_LEVEL > 0:
		sys.stdout.write("\r\033[K")
		sys.stdout.flush()

	# Generate baseline model predictions from hidden layers, 
	# 	Using the pre-hidden layer activations generated just above.
	#	These baseline predictions should be identical to running 
	# 	the complete model all at once.
	if model_type == 'cph':
		y_pred = predict_from_layer(model, pre_hl, input_layer_name='hidden_0', ouput_layer_index=-1)
		y_pred = np.concatenate((y_pred, events), axis = 1)
	else:
		y_pred = predict_from_layer(model, pre_hl, input_layer_name='hidden_0')

	# Generate the AUC, R-squared, and C-index metrics
	# 	From the generated baseline predictions.
	base_auc, base_r_squared, base_c_index = metrics_from_predictions(y_true=y_true,
												  					  y_pred=y_pred,
																	  tile_to_slides=tile_to_slides,
																	  annotations=annotations,
																	  model_type=model_type,
																	  manifest=manifest,
																	  outcome_names=outcome_names,
																	  label=label,
																	  min_tiles_per_slide=min_tiles_per_slide,
																	  data_dir=data_dir,
																	  verbose=True,
																	  histogram=False,
																	  plot=False)
	base_auc_list = np.array([base_auc['tile'], base_auc['slide'], base_auc['patient']])

	total_features = sum(feature_sizes)
	if model_type == 'cph':
		feature_sizes = feature_sizes[1:]
		feature_names = feature_names[1:]
		total_features -= 1

	if not drop_images:
		feature_names += ["Histology"]

	# For each feature, generate permutation metrics
	curCount = 0
	for i, feature in enumerate(feature_names):
		pre_hl_new = np.copy(pre_hl)
		
		if feature == "Histology":
			pre_hl_new[:,total_features:] = np.random.permutation(pre_hl_new[:,total_features:])
		else:
			if feature_sizes[i] == 1:
				pre_hl_new[:,curCount] = np.random.permutation(pre_hl_new[:,curCount])
			else:
				pre_hl_new[:,curCount:curCount + feature_sizes[i]] = np.random.permutation(pre_hl_new[:,curCount:curCount + feature_sizes[i]])
			
			curCount = curCount + feature_sizes[i]

		if model_type == 'cph':
			y_pred = predict_from_layer(model, pre_hl_new, input_layer_name='hidden_0', ouput_layer_index=-1)
			y_pred = np.concatenate((y_pred, events), axis = 1)
		else:
			y_pred = predict_from_layer(model, pre_hl_new, input_layer_name='hidden_0')

		new_auc, _, _ = metrics_from_predictions(y_true=y_true,
												y_pred=y_pred,
												tile_to_slides=tile_to_slides,
												annotations=annotations,
												model_type=model_type,
												manifest=manifest,
												outcome_names=outcome_names,
												label=None, #label[i] ? 
												min_tiles_per_slide=min_tiles_per_slide,
												data_dir=data_dir,
												verbose=False,
												histogram=False,
												plot=False)
		metrics[feature] = base_auc_list - np.array([new_auc['tile'], new_auc['slide'], new_auc['patient']])

	#Probably makes sense to measure only at the tile level - unless we write code to do permutation of patient level data which would be probably more work than its worth
	feature_text = ""
	for feature in feature_names:
		if model_type == 'categorical':
			feature_text += feature + ": " + str(metrics[feature][0][0]) + ", "
		else:
			feature_text += feature + ": " + str(metrics[feature][0]) + ", "
	log.info("Feature importance, tile level: " + feature_text, 1)
	
	return base_auc, base_r_squared, base_c_index