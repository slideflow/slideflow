import argparse
import json
import sys
import os
import math
import csv
import cv2
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.colors as mcol
import seaborn as sns
import pandas as pd
import tensorflow as tf
import scipy.stats as stats

from os.path import join, isfile, exists
from random import shuffle, sample
from mpl_toolkits import mplot3d
from statistics import mean
from math import isnan
from copy import deepcopy

import slideflow.util as sfutil
import slideflow.util.statistics as sfstats
from slideflow.util import log, progress_bar, tfrecords, TCGA
from PIL import Image

# TODO: merge Mosaic class into ActivationsVisualizer

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

class Mosaic:
	FOCUS_SLIDE = None
	BATCH_SIZE = 16
	SLIDES = {}
	GRID = []

	stride_div = 1
	ax_thumbnail = None
	metadata, points, tiles = [], [], []
	tfrecord_paths = []
	tile_point_distances = []
	rectangles = {}
	logits = {}

	def __init__(self, leniency=1.5, expanded=True, tile_zoom=15, num_tiles_x=50, resolution='high', 
					export=True, tile_um=None, use_fp16=True, save_dir=None):
		# Global variables
		self.max_distance_factor = leniency
		self.mapping_method = 'expanded' if expanded else 'strict'
		self.tile_zoom_factor = tile_zoom
		self.export = export
		self.num_tiles_x = num_tiles_x
		self.save_dir = save_dir
		self.DTYPE = tf.float16 if use_fp16 else tf.float32
		self.PT_NODE_DICT_PKL = join(save_dir, "stats", "activation_node_dict.pkl")

		# Variables used only when loading from slides
		self.tile_um = tile_um
		
		# Initialize figure
		log.info("Initializing figure...", 1)
		if resolution not in ('high', 'low'):
			log.warn(f"Unknown resolution option '{resolution}', defaulting to low resolution", 1)
		if resolution == 'high':
			self.fig = plt.figure(figsize=(200,200))
			self.ax = self.fig.add_subplot(111, aspect='equal')
		else:
			self.fig = plt.figure(figsize=(24,18))
			self.ax = self.fig.add_subplot(121, aspect='equal')
		self.ax.set_facecolor("#dfdfdf")
		self.fig.tight_layout()
		plt.subplots_adjust(left=0.02, bottom=0, right=0.98, top=1, wspace=0.1, hspace=0)
		self.ax.set_aspect('equal', 'box')
		self.ax.set_xticklabels([])
		self.ax.set_yticklabels([])

	def generate_final_layer_from_tfrecords(self, tfrecord_array, model, image_size):
		log.info(f"Calculating final layer activations from model {sfutil.green(model)}", 1)

		# Load model
		_model = tf.keras.models.load_model(model)
		complete_model=False
		try:
			loaded_model = tf.keras.models.Model(inputs=[_model.input, _model.layers[0].layers[0].input],
												outputs=[_model.layers[0].layers[-1].output, _model.layers[-1].output])
		except AttributeError:
			# Provides support for complete models that were not generated using Slideflow
			complete_model=True
			loaded_model = tf.keras.models.Model(inputs=[_model.input],
												 outputs=[_model.layers[-2].output])
		
		fl_activations_all, logits_all, slides_all, indices_all, tile_indices_all, tfrecord_all = [], [], [], [], [], []
		unique_slides = list(set([sfutil.path_to_name(tfr) for tfr in tfrecord_array]))

		def _parse_function(record):
			features = tf.io.parse_single_example(record, tfrecords.FEATURE_DESCRIPTION)
			slide = features['slide']
			image_string = features['image_raw']
			raw_image = tf.image.decode_jpeg(image_string, channels=3)
			processed_image = tf.image.per_image_standardization(raw_image)
			processed_image = tf.image.convert_image_dtype(processed_image, self.DTYPE)
			processed_image.set_shape([image_size, image_size, 3])
			return processed_image, slide

		# Calculate final layer activations for each tfrecord
		for tfrecord in tfrecord_array:
			log.info(f"Calculating activations from {sfutil.green(tfrecord)}", 2)
			dataset = tf.data.TFRecordDataset(tfrecord)

			dataset = dataset.map(_parse_function, num_parallel_calls=8)
			dataset = dataset.batch(self.BATCH_SIZE, drop_remainder=False)
			self.tfrecord_paths += [tfrecord]
			tfrecord_index = self.tfrecord_paths.index(tfrecord)

			fl_activations_arr, logits_arr, slides_arr, indices_arr, tile_indices_arr = [], [], [], [], []
  
			for i, data in enumerate(dataset):
				batch_processed_images, batch_slides = data
				batch_slides = batch_slides.numpy()
				sys.stdout.write(f"\r - Working on batch {i}")
				sys.stdout.flush()

				# Calculate global and tfrecord-specific indices
				indices = list(range(len(indices_arr), len(indices_arr) + len(batch_slides)))
				tile_indices = list(range(i * len(batch_slides), i * len(batch_slides) + len(batch_slides)))

				if not complete_model:
					fl_activations, logits = loaded_model.predict([batch_processed_images, batch_processed_images])
				else:
					fl_activations = loaded_model.predict([batch_processed_images])
					logits = [[-1]] * self.BATCH_SIZE

				fl_activations_arr = fl_activations if fl_activations_arr == [] else np.concatenate([fl_activations_arr, fl_activations])
				logits_arr = logits if logits_arr == [] else np.concatenate([logits_arr, logits])
				slides_arr = batch_slides if slides_arr == [] else np.concatenate([slides_arr, batch_slides])
				indices_arr = indices if indices_arr == [] else np.concatenate([indices_arr, indices])
				tile_indices_arr = tile_indices if tile_indices_arr == [] else np.concatenate([tile_indices_arr, tile_indices])

			sys.stdout.write("\r\033[K")
			sys.stdout.flush()

			tfrecord_arr = np.array([tfrecord_index] * len(slides_arr))

			fl_activations_all = fl_activations_arr if fl_activations_all == [] else np.concatenate([fl_activations_all, fl_activations_arr])
			logits_all = logits_arr if logits_all == [] else np.concatenate([logits_all, logits_arr])
			slides_all = slides_arr if slides_all == [] else np.concatenate([slides_all, slides_arr])
			indices_all = indices_arr if indices_all == [] else np.concatenate([indices_all, indices_arr])
			tile_indices_all = tile_indices_arr if tile_indices_all == [] else np.concatenate([tile_indices_all, tile_indices_arr])
			tfrecord_all = tfrecord_arr if tfrecord_all == [] else np.concatenate([tfrecord_all, tfrecord_arr])

		# Save final layer activations to CSV file
		# and export PKL
		nodes = [f"FLNode{f}" for f in range(fl_activations_all.shape[1])]
		logits = [f"Logits{l}" for l in range(logits_all.shape[1])]
		header = ["Slide"] + logits + nodes
		flactivations_file = join(self.save_dir, "final_layer_activations.csv")

		# Prepare PKL export dictionary
		slide_node_dict = {}
		for slide in unique_slides:
			slide_node_dict.update({slide: {}})
		for node in nodes:
			for slide in unique_slides:
				slide_node_dict[slide].update({node: []})

		# Export to CSV
		with open(flactivations_file, 'w') as outfile:
			csvwriter = csv.writer(outfile)
			csvwriter.writerow(header)
			for i in range(len(slides_all)):
				slide = [slides_all[i].decode('utf-8')]
				logits = logits_all[i].tolist()
				flactivations = fl_activations_all[i].tolist()
				row = slide + logits + flactivations
				csvwriter.writerow(row)

				# Export to PKL dictionary
				for node in nodes:
					node_i = header.index(node)
					val = row[node_i]
					slide_node_dict[slide][node] += [val]

		# Dump PKL dictionary to file
		with open(self.PT_NODE_DICT_PKL, 'wb') as pt_pkl_file:
			pickle.dump(slide_node_dict, pt_pkl_file)

		# Returns a 2D array, with each element containing FL activations, logits, slide name, tfrecord name, and tfrecord indices
		return fl_activations_all, logits_all, slides_all, indices_all, tile_indices_all, tfrecord_all	

	def generate_from_tfrecords(self, tfrecord_array, model, image_size, focus=None):
		fl_activations, logits, slides, indices, tile_indices, tfrecords = self.generate_final_layer_from_tfrecords(tfrecord_array, model, image_size)
		
		dl_coord = sfstats.gen_umap(fl_activations)
		self.load_coordinates(dl_coord, [slides, tfrecords, indices, tile_indices])

		self.place_tile_outlines()
		self.calculate_distances()
		self.pair_tiles_and_points()
		if focus:
			self.focus_tfrecords(focus)
		self.finish_mosaic()

	def load_coordinates(self, coord, meta):
		log.empty("Loading dimensionality reduction coordinates and plotting points...", 1)
		points_x = []
		points_y = []
		point_index = 0
		slides, tfrecords, indices, tile_indices = meta
		for i, p in enumerate(coord):
			points_x.append(p[0])
			points_y.append(p[1])
			self.points.append({'x':p[0],
								'y':p[1],
								'index':point_index,
								'neighbors':[],
								'category':'none',
								'slide':slides[i],
								'tfrecord':self.tfrecord_paths[tfrecords[i]],
								'tfrecord_index':indices[i],
								'tile_index':tile_indices[i],
								'paired_tile':None })
			point_index += 1
		x_points = [p['x'] for p in self.points]
		y_points = [p['y'] for p in self.points]
		_x_width = max(x_points) - min(x_points)
		_y_width = max(y_points) - min(y_points)
		buffer = (_x_width + _y_width)/2 * 0.05
		max_x = max(x_points) + buffer
		min_x = min(x_points) - buffer
		max_y = max(y_points) + buffer
		min_y = min(y_points) - buffer

		log.info(f"Loaded {len(self.points)} points.", 2)

		#self.tsne_plot = self.ax.scatter(points_x, points_y, s=1000, facecolors='none', edgecolors='green', alpha=0)# markersize = 5
		self.tile_size = (max_x - min_x) / self.num_tiles_x
		self.num_tiles_y = int((max_y - min_y) / self.tile_size)
		self.max_distance = math.sqrt(2*((self.tile_size/2)**2)) * self.max_distance_factor
		self.tile_coord_x = [(i*self.tile_size)+min_x for i in range(self.num_tiles_x)]
		self.tile_coord_y = [(j*self.tile_size)+min_y for j in range(self.num_tiles_y)]

		# Initialize grid
		for j in range(self.num_tiles_y):
			for i in range(self.num_tiles_x):
				self.GRID.append({'x': ((self.tile_size/2) + min_x) + (self.tile_size * i),
									'y': ((self.tile_size/2) + min_y) + (self.tile_size * j),
									'x_index': i,
									'y_index': j,
									'index': len(self.GRID),
									'size': self.tile_size,
									'points':[],
									'distances':[],
									'active': False,
									'image': None})

		# Add point indices to grid
		points_added = 0
		for point in self.points:
			x_index = int((point['x'] - min_x) / self.tile_size)
			y_index = int((point['y'] - min_y) / self.tile_size)
			for g in self.GRID:
				if g['x_index'] == x_index and g['y_index'] == y_index:
					g['points'].append(point['index'])
					points_added += 1
		for g in self.GRID:
			shuffle(g['points'])
		log.info(f"{points_added} points added to grid", 2)

	def place_tile_outlines(self):
		log.empty("Placing tile outlines...", 1)
		# Find max GRID density
		max_grid_density = 1
		for g in self.GRID:
			max_grid_density = max(max_grid_density, len(g['points']))
		for grid_tile in self.GRID:
			rect_size = min((len(grid_tile['points']) / max_grid_density) * self.tile_zoom_factor, 1) * self.tile_size

			tile = patches.Rectangle((grid_tile['x'] - rect_size/2, 
							  		  grid_tile['y'] - rect_size/2), 
									  rect_size, 
							  		  rect_size, 
									  fill=True, alpha=1, facecolor='white', edgecolor="#cccccc")
			self.ax.add_patch(tile)

			grid_tile['size'] = rect_size
			grid_tile['rectangle'] = tile
			grid_tile['neighbors'] = []
			grid_tile['paired_point'] = None

	def calculate_distances(self):
		log.empty("Calculating tile-point distances...", 1)
		pb = progress_bar.ProgressBar()
		pb_id = pb.add_bar(0, len(self.GRID))
		for i, tile in enumerate(self.GRID):
			pb.update(pb_id, i)
			if self.mapping_method == 'strict':
				# Calculate distance for each point from center
				distances = []
				for point_index in tile['points']:
					point = self.points[point_index]
					distance = math.sqrt((point['x']-tile['x'])**2 + (point['y']-tile['y'])**2)
					distances.append([point['index'], distance])
				distances.sort(key=lambda d: d[1])
				tile['distances'] = distances
			elif self.mapping_method == 'expanded':
				# Calculate distance for each point from center
				distances = []
				for point in self.points:
					distance = math.sqrt((point['x']-tile['x'])**2 + (point['y']-tile['y'])**2)
					distances.append([point['index'], distance])
				distances.sort(key=lambda d: d[1])
				for d in distances:
					if d[1] <= self.max_distance:
						tile['neighbors'].append(d)
						self.points[d[0]]['neighbors'].append([tile['index'], d[1]])
						self.tile_point_distances.append({'distance': d[1],
													'tile_index':tile['index'],
													'point_index':d[0]})
					else:
						break
			else:
				raise TypeError("Unknown mapping method")
		pb.end()
		if self.mapping_method == 'expanded':
			self.tile_point_distances.sort(key=lambda d: d['distance'])

	def pair_tiles_and_points(self):
		log.empty("Placing image tiles...", 1)
		num_placed = 0
		if self.mapping_method == 'strict':
			for tile in self.GRID:
				if not len(tile['distances']): continue
				closest_point = tile['distances'][0][0]
				point = self.points[closest_point]
				#tile_image = plt.imread(point['image_path'])
				#tile_image = point['image_data']
				_, tile_image = tfrecords.get_tfrecord_by_index(point['tfrecord'], point['tile_index'])
				tile_alpha, num_slide, num_other = 1, 0, 0
				if self.FOCUS_SLIDE and len(tile['points']):
					for point_index in tile['points']:
						point = self.points[point_index]
						if point['slide'] == self.FOCUS_SLIDE:
							num_slide += 1
						else:
							num_other += 1
					fraction_slide = num_slide / (num_other + num_slide)
					tile_alpha = fraction_slide
				if not self.export:
					tile_image = cv2.resize(tile_image, (0,0), fx=0.25, fy=0.25)
				image = self.ax.imshow(tile_image, aspect='equal', origin='lower', extent=[tile['x']-tile['size']/2, 
																						tile['x']+tile['size']/2,
																						tile['y']-tile['size']/2,
																						tile['y']+tile['size']/2], zorder=99, alpha=tile_alpha)
				tile['image'] = image
				num_placed += 1
		elif self.mapping_method == 'expanded':
			for distance_pair in self.tile_point_distances:
				# Attempt to place pair, skipping if unable (due to other prior pair)
				point = self.points[distance_pair['point_index']]
				tile = self.GRID[distance_pair['tile_index']]
				if not (point['paired_tile'] or tile['paired_point']):
					point['paired_tile'] = True
					tile['paired_point'] = True

					#tile_image = plt.imread(point['image_path'])
					#tile_image = point['image_data']
					_, tile_image = tfrecords.get_tfrecord_by_index(point['tfrecord'], point['tile_index'])
					if not self.export:
						tile_image = cv2.resize(tile_image, (0,0), fx=0.25, fy=0.25)
					image = self.ax.imshow(tile_image, aspect='equal', origin='lower', extent=[tile['x']-self.tile_size/2, 
																					tile['x']+self.tile_size/2,
																					tile['y']-self.tile_size/2,
																					tile['y']+self.tile_size/2], zorder=99)		
					tile['image'] = image
					num_placed += 1
		log.info(f"Num placed: {num_placed}", 2)

	def focus_tfrecords(self, tfrecord_list):
		for tile in self.GRID:
			if not len(tile['points']): continue
			num_cat, num_other = 0, 0
			for point_index in tile['points']:
				point = self.points[point_index]
				if point['tfrecord'] in tfrecord_list:
					num_cat += 1
				else:
					num_other += 1
			alpha = num_cat / (num_other + num_cat)
			tile['image'].set_alpha(alpha)

	def finish_mosaic(self):
		log.empty("Displaying/exporting figure...", 1)
		self.ax.autoscale(enable=True, tight=None)
		if self.export:
			save_path = join(self.save_dir, f'Mosaic-{self.num_tiles_x}.png')
			plt.savefig(save_path, bbox_inches='tight')
			log.complete(f"Saved figure to {sfutil.green(save_path)}", 1)
			plt.close()
		else:
			while True:
				try:
					plt.show()
				except UnicodeDecodeError:
					continue
				break

class ActivationsVisualizer:
	missing_slides = []
	used_categories = []
	umaps = []

	def __init__(self, annotations, category_header, tfrecords, root_dir, focus_nodes=[]):
		self.focus_nodes = focus_nodes
		self.CATEGORY_HEADER = category_header
		self.ANNOTATIONS = annotations
		self.TFRECORDS = np.array(tfrecords)
		self.slides_to_include = [sfutil.path_to_name(tfr) for tfr in self.TFRECORDS]

		self.FLA = join(root_dir, "stats", "final_layer_activations.csv")
		self.STATS_CSV_FILE = join(root_dir, "stats", "slide_level_summary.csv")
		self.PT_NODE_DICT_PKL = join(root_dir, "stats", "activation_node_dict.pkl")
		self.UMAP_CACHE = join(root_dir, "stats", "umap_cache.pkl")
		self.EXAMPLE_TILES_DIR = join(root_dir, "stats", "example_tiles")
		self.SORTED_DIR = join(root_dir, "stats", "sorted_tiles")
		if not exists(join(root_dir, "stats")):
			os.makedirs(join(root_dir, "stats"))

		# Initial loading and preparation
		self.load_annotations()
		self.load_activations()

	def load_annotations(self):
		self.slide_category_dict = {}
		with open(self.ANNOTATIONS, 'r') as ann_file:
			log.empty("Reading annotations...", 1)
			ann_reader = csv.reader(ann_file)
			header = next(ann_reader)
			slide_i = header.index(TCGA.slide)
			category_i = header.index(self.CATEGORY_HEADER)
			for row in ann_reader:
				slide = row[slide_i]
				category = row[category_i]
				if slide not in self.slides_to_include: continue
				self.slide_category_dict.update({slide:category})

		self.categories = list(set(self.slide_category_dict.values()))
		self.slides = list(self.slide_category_dict.keys())

	def load_pkl(self):
		log.empty("Loading pre-calculated activations from pickled files...", 1)
		with open(self.PT_NODE_DICT_PKL, 'rb') as pt_pkl_file:
			self.slide_node_dict = pickle.load(pt_pkl_file)
			self.nodes = list(self.slide_node_dict[list(self.slide_node_dict.keys())[0]].keys())

	def write_pkl(self):
		with open(self.PT_NODE_DICT_PKL, 'wb') as pt_pkl_file:
			pickle.dump(self.slide_node_dict, pt_pkl_file)

	def load_activations(self):
		if exists(self.PT_NODE_DICT_PKL): 
			self.load_pkl()
		else:
			self.slide_node_dict = {}
			for slide in self.slides:
				self.slide_node_dict.update({slide: {}})
			with open(self.FLA, 'r') as fl_file:
				log.empty(f"Reading final layer activations from {sfutil.green(self.FLA)}...", 1)
				fl_reader = csv.reader(fl_file)
				header = next(fl_reader)
				self.nodes = [h for h in header if h[:6] == "FLNode"]
				slide_i = header.index("Slide")

				for node in self.nodes:
					for slide in self.slides:
						self.slide_node_dict[slide].update({node: []})

				for i, row in enumerate(fl_reader):
					print(f"Reading activations for tile {i}...", end="\r")
					slide = row[slide_i]
					for node in self.nodes:
						node_i = header.index(node)
						val = float(row[node_i])
						self.slide_node_dict[slide][node] += [val]
			print()
			self.write_pkl()

		# Now delete slides not included in our filtered TFRecord list
		loaded_slides = list(self.slide_node_dict.keys())
		for loaded_slide in loaded_slides:
			if loaded_slide not in self.slides_to_include:
				del self.slide_node_dict[loaded_slide]

		# Now screen for missing slides in activations
		for slide in self.slides:
			try:
				if self.slide_node_dict[slide]['FLNode0'] == []:
					self.missing_slides += [slide]
				else:
					self.used_categories = list(set(self.used_categories + [self.slide_category_dict[slide]]))
					self.used_categories.sort()
			except KeyError:
				log.warn(f"Skipping unknown slide {slide}", 1)
				self.missing_slides += [slide]
		log.info(f"Loaded activations from {(len(self.slides)-len(self.missing_slides))}/{len(self.slides)} slides ({len(self.missing_slides)} missing)", 1)
		log.info(f"Observed categories (total: {len(self.used_categories)}):", 1)
		for c in self.used_categories:
			log.empty(f"\t{c}", 2)

	def get_tile_node_activations_by_category(self, node):
		tile_node_activations_by_category = []
		for c in self.used_categories:
			nodelist = [self.slide_node_dict[pt][node] for pt in self.slides if (pt not in self.missing_slides and self.slide_category_dict[pt] == c)]
			tile_node_activations_by_category += [[nodeval for nl in nodelist for nodeval in nl]]
		return tile_node_activations_by_category

	def get_top_nodes_by_slide(self):
		# First ensure basic stats have been calculated
		if not hasattr(self, 'nodes_avg_pt'):
			self.calculate_activation_averages_and_stats()
		
		return self.nodes_avg_pt

	def get_top_nodes_by_tile(self):
		# First ensure basic stats have been calculated
		if not hasattr(self, 'nodes_avg_pt'):
			self.calculate_activation_averages_and_stats()
		
		return self.nodes

	def calculate_activation_averages_and_stats(self):
		empty_category_dict = {}
		self.node_dict_avg_pt = {}
		node_stats = {}
		node_stats_avg_pt = {}
		for category in self.categories:
			empty_category_dict.update({category: []})
		for node in self.nodes:
			self.node_dict_avg_pt.update({node: deepcopy(empty_category_dict)})

		for node in self.nodes:
			# For each node, calculate average across tiles found in a slide
			print(f"Calculating activation averages & stats for node {node}...", end="\r")
			for slide in self.slides:
				if slide in self.missing_slides: continue
				pt_cat = self.slide_category_dict[slide]
				avg = mean(self.slide_node_dict[slide][node])
				self.node_dict_avg_pt[node][pt_cat] += [avg]
			
			# Tile-level ANOVA
			fvalue, pvalue = stats.f_oneway(*self.get_tile_node_activations_by_category(node))
			if not isnan(fvalue) and not isnan(pvalue): 
				node_stats.update({node: {'f': fvalue,
										  'p': pvalue} })

			# Patient-level ANOVA
			fvalue, pvalue = stats.f_oneway(*[self.node_dict_avg_pt[node][c] for c in self.used_categories])
			if not isnan(fvalue) and not isnan(pvalue): 
				node_stats_avg_pt.update({node: {'f': fvalue,
												 'p': pvalue} })
		print()

		try:
			self.nodes = sorted(self.nodes, key=lambda n: node_stats[n]['p'])
			self.nodes_avg_pt = sorted(self.nodes, key=lambda n: node_stats_avg_pt[n]['p'])
		except:
			log.warn("No stats calculated; unable to sort nodes.", 1)
			self.nodes_avg_pt = self.nodes
			
		for i, node in enumerate(self.nodes):
			if self.focus_nodes and (node not in self.focus_nodes): continue
			try:
				log.info(f"Tile-level P-value ({node}): {node_stats[node]['p']}", 1)
				log.info(f"Patient-level P-value: ({node}): {node_stats_avg_pt[node]['p']}", 1)
			except:
				log.warn(f"No stats calculated for node {node}", 1)
			if (not self.focus_nodes) and i>9: break
		if not exists(self.STATS_CSV_FILE):
			self.save_to_csv(self.nodes_avg_pt, tile_stats=node_stats, slide_stats=node_stats_avg_pt)
		else:
			log.info(f"Stats file already generated at {sfutil.green(self.STATS_CSV_FILE)}; not regenerating", 1)

	def generate_box_plots(self):
		# First ensure basic stats have been calculated
		if not hasattr(self, 'nodes_avg_pt'):
			self.calculate_activation_averages_and_stats()

		# Display tile-level box plots & stats
		log.empty("Generating box plots...")
		for i, node in enumerate(self.nodes):
			if self.focus_nodes and (node not in self.focus_nodes): continue
			snsbox = sns.boxplot(data=self.get_tile_node_activations_by_category(node))
			snsbox.set_title(f"{node} (tile-level)")
			snsbox.set(xlabel="Category", ylabel="Activation")
			plt.xticks(plt.xticks()[0], self.used_categories)
			plt.show()
			if (not self.focus_nodes) and i>4: break

		# Print slide_level box plots & stats
		for i, node in enumerate(self.nodes_avg_pt):
			if self.focus_nodes and (node not in self.focus_nodes): continue
			snsbox = sns.boxplot(data=[self.node_dict_avg_pt[node][c] for c in self.used_categories])
			snsbox.set_title(f"{node} (slide-level)")
			snsbox.set(xlabel="Category",ylabel="Average tile activation")
			plt.xticks(plt.xticks()[0], self.used_categories)
			plt.show()
			if (not self.focus_nodes) and i>4: break
	
	def save_to_csv(self, nodes_avg_pt, tile_stats=None, slide_stats=None):
		# Save results to CSV
		log.empty(f"Writing results to {sfutil.green(self.STATS_CSV_FILE)}...", 1)
		with open(self.STATS_CSV_FILE, 'w') as outfile:
			csv_writer = csv.writer(outfile)
			header = ['slide', 'category'] + nodes_avg_pt
			csv_writer.writerow(header)
			for slide in self.slides:
				if slide in self.missing_slides: continue
				category = self.slide_category_dict[slide]
				row = [slide, category] + [mean(self.slide_node_dict[slide][n]) for n in nodes_avg_pt]
				csv_writer.writerow(row)
			if tile_stats:
				csv_writer.writerow(['Tile-level statistic', 'ANOVA P-value'] + [tile_stats[n]['p'] for n in nodes_avg_pt])
				csv_writer.writerow(['Tile-level statistic', 'ANOVA F-value'] + [tile_stats[n]['f'] for n in nodes_avg_pt])
			if slide_stats:
				csv_writer.writerow(['Slide-level statistic', 'ANOVA P-value'] + [slide_stats[n]['p'] for n in nodes_avg_pt])
				csv_writer.writerow(['Slide-level statistic', 'ANOVA F-value'] + [slide_stats[n]['f'] for n in nodes_avg_pt])

	def load_umap_cache(self):
		try:
			with open(self.UMAP_CACHE, 'rb') as umap_file:
				self.umaps = pickle.load(umap_file)
				log.info(f"Loaded UMAP cache from {sfutil.green(self.UMAP_CACHE)}", 1)
				return True
		except:
			log.info(f"No UMAP cache found at {sfutil.green(self.UMAP_CACHE)}", 1)
			return False

	def check_if_umap_calculated(self, nodes):
		slides = [slide for slide in self.slides if slide not in self.missing_slides]
		for um in self.umaps:
			# Check to see if this has already been cached
			if (sorted(um['nodes']) == sorted(nodes)) and (sorted(um['slides']) == sorted(slides)):
				return um
		return False

	def cache_umap(self, umap_x, umap_y, umap_meta, nodes):
		slides = [slide for slide in self.slides if slide not in self.missing_slides]
		umap = {
			'nodes': nodes,
			'slides': slides,
			'umap_x': umap_x,
			'umap_y': umap_y,
			'umap_meta': umap_meta
		}
		self.umaps += [umap]
		with open(self.UMAP_CACHE, 'wb') as umap_file:
			pickle.dump(self.umaps, umap_file)
			log.info(f"Wrote UMAP cache to {sfutil.green(self.UMAP_CACHE)}", 1)
		return umap

	def calculate_umap(self, exclude_node=None):
		nodes_to_include = [n for n in self.nodes if n != exclude_node]
		# Check if UMAP has already been cached in self.umap and presumably stored
		# Otherwise, calculate and cache a new UMAP
		self.load_umap_cache()
		umap_check = self.check_if_umap_calculated(nodes_to_include)
		if umap_check:
			log.info("UMAP results already calculated and cached", 1)
			return umap_check
		else:
			log.info("No compatible UMAP results found in cache; recalculating", 1)
			node_activations = []
			umap_meta = []
			log.empty("Calculating UMAP...", 1)
			for slide in self.slides:
				if slide in self.missing_slides: continue
				first_node = list(self.slide_node_dict[slide].keys())[0]
				num_vals = len(self.slide_node_dict[slide][first_node])
				for i in range(num_vals):
					node_activations += [[self.slide_node_dict[slide][n][i] for n in nodes_to_include]]
					umap_meta += [{
						'slide': slide,
						'index': i,
					}]
			coordinates = sfstats.gen_umap(np.array(node_activations))
			umap_x = np.array([c[0] for c in coordinates])
			umap_y = np.array([c[1] for c in coordinates])
			umap = self.cache_umap(umap_x, umap_y, umap_meta, nodes_to_include)
			return umap

	def plot_2D_umap(self, node=None, exclusion=False, subsample=None):
		umap = self.calculate_umap(exclude_node=node if exclusion else None)
		categories = np.array([self.slide_category_dict[m['slide']] for m in umap['umap_meta']])
		
		# Subsampling
		if subsample:
			ri = sample(range(len(umap['umap_x'])), subsample)
		else:
			ri = list(range(len(umap['umap_x'])))

		unique_categories = list(set(categories[ri]))

		# Prepare pandas dataframe
		df = pd.DataFrame()
		df['umap_x'] = umap['umap_x'][ri]
		df['umap_y'] = umap['umap_y'][ri]
		df['category'] = pd.Series(categories[ri], dtype='category')

		# Make plot
		log.info("Displaying 2D UMAP...", 1)
		sns.scatterplot(x=umap['umap_x'][ri], y=umap['umap_y'][ri], data=df, hue='category', palette=sns.color_palette('Set1', len(unique_categories)))
		plt.show()
		return umap

	def plot_3D_umap(self, node, exclusion=False, subsample=1000):
		umap = self.calculate_umap(exclude_node=node if exclusion else None)

		# Subsampling
		if subsample:
			ri = sample(range(len(umap['umap_x'])), subsample)
		else:
			ri = list(range(len(umap['umap_x'])))

		umap_x = umap['umap_x'][ri]
		umap_y = umap['umap_y'][ri]

		node_vals = np.array([self.slide_node_dict[m['slide']][node][m['index']] for m in umap['umap_meta']])
		z = node_vals[ri]

		# Plot tiles on a 3D coordinate space with 2 coordinates from UMAP & 3rd from the value of the excluded node
		log.info("Displaying 3D UMAP...", 1)
		ax = plt.axes(projection='3d')
		ax.scatter(umap_x, umap_y, z, c=z,
									  cmap='viridis',
									  linewidth=0.5,
									  edgecolor="black")
		ax.set_title(f"UMAP with node {node} focus")
		plt.show()
		return umap

	def filter_tiles_by_umap(self, umap, x_lower=-999, x_upper=999, y_lower=-999, y_upper=999):
		# Find tiles that meet UMAP location criteria
		umap_x = umap['umap_x']
		umap_y = umap['umap_y']
		umap_meta = umap['umap_meta']
		filter_criteria = {}
		num_selected = 0
		for i in range(len(umap_meta)):
			if (x_lower < umap_x[i] < x_upper) and (y_lower < umap_y[i] < y_upper):
				slide = umap_meta[i]['slide']
				tile_index = umap_meta[i]['index']
				if slide not in filter_criteria:
					filter_criteria.update({slide: [tile_index]})
				else:
					filter_criteria[slide] += [tile_index]
				num_selected += 1
		log.info(f"Selected {num_selected} tiles by filter criteria.", 1)
		return filter_criteria

	def save_example_tiles_gradient(self, nodes=None, tile_filter=None):
		if not nodes:
			nodes = self.focus_nodes
		for node in nodes:
			if not exists(join(self.SORTED_DIR, node)):
				os.makedirs(join(self.SORTED_DIR, node))
			
			gradient = []
			for slide in self.slides:
				if slide in self.missing_slides: continue
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
				for tfr in self.TFRECORDS:
					if sfutil.path_to_name(tfr) == g['slide']:
						tfr_dir = tfr
				if not tfr_dir:
					log.warn(f"TFRecord location not found for slide {g['slide']}", 1)
				slide, image = tfrecords.get_tfrecord_by_index(tfr_dir, g['index'], decode=False)
				slide = slide.numpy()
				image = image.numpy()
				tile_filename = f"{i}-tfrecord{g['slide']}-{g['index']}-{g['val']:.2f}.jpg"
				image_string = open(join(self.SORTED_DIR, node, tile_filename), 'wb')
				image_string.write(image)
				image_string.close()
			print()

	def save_example_tiles_high_low(self, focus_slides):
		# Extract samples of tiles with highest and lowest values in a particular node
		# for a subset of slides
		for fn in self.focus_nodes:
			for slide in focus_slides:
				sorted_index = np.argsort(self.slide_node_dict[slide][fn])
				lowest = sorted_index[:10]
				highest = sorted_index[-10:]
				lowest_dir = join(self.EXAMPLE_TILES_DIR, fn, slide, 'lowest')
				highest_dir = join(self.EXAMPLE_TILES_DIR, fn, slide, 'highest')
				if not exists(lowest_dir): os.makedirs(lowest_dir)
				if not exists(highest_dir): os.makedirs(highest_dir)

				for tfr in self.TFRECORDS:
					if sfutil.path_to_name(tfr) == slide:
						tfr_dir = tfr
				if not tfr_dir:
					log.warn(f"TFRecord location not found for slide {slide}", 1)

				def extract_by_index(indices, directory):
					for index in indices:
						slide, image = tfrecords.get_tfrecord_by_index(tfr_dir, index, decode=False)
						slide = slide.numpy()
						image = image.numpy()
						tile_filename = f"tfrecord{slide}-tile{index}.jpg"
						image_string = open(join(directory, tile_filename), 'wb')
						image_string.write(image)
						image_string.close()

				extract_by_index(lowest, lowest_dir)
				extract_by_index(highest, highest_dir)

class TileVisualizer:
	def __init__(self, model, node, shape, tile_width=None, interactive=False):
		self.NODE = node
		self.IMAGE_SHAPE = shape
		self.TILE_WIDTH = tile_width if tile_width else int(self.IMAGE_SHAPE[0]/6)
		self.interactive = interactive
		log.info("Initializing tile visualizer", 1)
		log.info(f"Node: {sfutil.bold(str(node))} | Shape: ({shape[0]}, {shape[1]}, {shape[2]}) | Window size: {self.TILE_WIDTH}", 1)
		log.info(f"Loading Tensorflow model at {sfutil.green(model)}...", 1)
		_model = tf.keras.models.load_model(model)
		self.loaded_model = tf.keras.models.Model(inputs=[_model.input, _model.layers[0].layers[0].input],
												  outputs=[_model.layers[0].layers[-1].output, _model.layers[-1].output])

	def visualize_tile(self, tile, save_dir=None, zoomed=True):
		log.info(f"Processing tile at {sfutil.green(tile)}...", 1)
		tilename = sfutil.path_to_name(tile)
		# First, open tile image
		self.tile_image = Image.open(tile)
		image_file = open(tile, 'rb')
		self.tf_raw_image = image_file.read()

		# Next, process image with Tensorflow
		tf_decoded_image = tf.image.decode_jpeg(self.tf_raw_image, channels=3)
		self.tf_processed_image = tf.image.per_image_standardization(tf_decoded_image)
		self.tf_processed_image = tf.image.convert_image_dtype(self.tf_processed_image, tf.float16)
		self.tf_processed_image.set_shape(self.IMAGE_SHAPE)
		image_batch = np.array([self.tf_processed_image.numpy()])

		# Calculate baseline activations/predictions
		base_activations, base_logits = self.loaded_model.predict([image_batch, image_batch])

		# Now create the figure
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111)
		self.implot = plt.imshow(self.tile_image)

		if self.interactive:
			self.rect = patches.Rectangle((0, 0), self.TILE_WIDTH, self.TILE_WIDTH, facecolor='white', zorder=20)
			self.ax.add_patch(self.rect)

		activation_map, max_center_x, max_center_y = self.calculate_activation_map()

		self.generate_figure(tilename, activation_map, max_x=max_center_x,
													   max_y=max_center_y,
													   save_dir=save_dir,
													   zoomed_extent=zoomed)

	def predict_masked(self, x, y, index):
		mask = create_bool_mask(x, y, self.TILE_WIDTH, self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1])
		masked = self.tf_processed_image.numpy() * mask
		act, _ = self.loaded_model.predict([np.array([masked]), np.array([masked])])
		return act[0][index]	

	def calculate_activation_map(self, stride_div=4):
		sx = self.IMAGE_SHAPE[0]
		sy = self.IMAGE_SHAPE[1]
		w  = self.TILE_WIDTH
		stride = int(self.TILE_WIDTH / stride_div)
		min_x  = int(w/2)
		max_x  = int(sx - w/2)
		min_y  = int(w/2)
		max_y  = int(sy - w/2)

		act_array = []
		for yi in range(min_y, max_y, stride):
			for xi in range(min_x, max_x, stride):
				mask = create_bool_mask(xi, yi, w, sx, sy)
				masked = self.tf_processed_image.numpy() * mask
				act, _ = self.loaded_model.predict([np.array([masked]), np.array([masked])])
				act_array += [act[0][self.NODE]]
				print(f"Calculating activations at x:{xi}, y:{yi}; act={act[0][self.NODE]}", end='\033[K\r')
		max_center_x = max(range(min_x, max_x, stride))
		max_center_y = max(range(min_y, max_y, stride))
		reshaped_array = np.reshape(np.array(act_array), [len(range(min_x, max_x, stride)), 
														  len(range(min_y, max_y, stride))])
		print()

		return reshaped_array, max_center_x, max_center_y

	def generate_figure(self, name, activation_map=None, max_x=None, max_y=None, save_dir=None, zoomed_extent=False):
		# Create hover and click events
		def hover(event):
			if event.xdata:
				self.rect.set_xy((event.xdata-self.TILE_WIDTH/2, event.ydata-self.TILE_WIDTH/2))
				print(self.predict_masked(event.xdata, event.ydata, index=self.NODE), end='\r')
				self.fig.canvas.draw_idle()

		def click(event):
			if event.button == 1:
				self.TILE_WIDTH = min(min(self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1]), self.TILE_WIDTH + 25)
				self.rect.set_width(self.TILE_WIDTH)
				self.rect.set_height(self.TILE_WIDTH)
			else:
				self.TILE_WIDTH = max(0, self.TILE_WIDTH - 25)
				self.rect.set_width(self.TILE_WIDTH)
				self.rect.set_height(self.TILE_WIDTH)
			self.fig.canvas.draw_idle()	

		if self.interactive:
			self.fig.canvas.mpl_connect('motion_notify_event', hover)
			self.fig.canvas.mpl_connect('button_press_event', click)
		
		if activation_map is not None:
			# Define color map
			jetMap = np.flip(np.linspace(0.45, 0.95, 255))
			cmMap = cm.nipy_spectral(jetMap)
			newMap = mcol.ListedColormap(cmMap)

			# Calculate boundaries of heatmap
			hw = int(self.TILE_WIDTH/2)
			if zoomed_extent:
				extent = (hw, max_x, max_y, hw)
			else:
				extent = (0, max_x+hw, max_y+hw, 0)
			
			# Heatmap
			self.ax.imshow(activation_map, extent=extent,
										   cmap=newMap,
										   alpha=0.6 if not self.interactive else 0.0,
										   interpolation='bicubic',
										   zorder=10)
		if save_dir:
			heatmap_loc = join(save_dir, f'{name}-heatmap.png')
			plt.savefig(heatmap_loc, bbox_inches='tight')
			log.complete(f"Heatmap saved to {heatmap_loc}", 1)
		if self.interactive:
			print()
			plt.show()