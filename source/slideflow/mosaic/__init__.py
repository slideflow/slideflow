import argparse
import json
import sys
import os
import math
import csv
import cv2
import umap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from os.path import join, isfile, exists

import tensorflow as tf

import slideflow.util as sfutil
from slideflow.util import log, progress_bar, tfrecords

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

def gen_umap(array):
	layout = umap.UMAP(n_components=2, verbose=True, n_neighbors=20, min_dist=0.01, metric="cosine").fit_transform(array)
	return normalize_layout(layout)
	
#def gen_tsne_array(array):
#    layout = TSNE(n_components=2, verbose=True, metric="cosine", learning_rate=10, perplexity=50).fit_transform(array)
#    return normalize_layout(layout)

class Mosaic:
	SVS = None
	BATCH_SIZE = 64
	SLIDES = {}
	GRID = []
	stride_div = 1
	ax_thumbnail = None
	svs_background = None
	metadata, tsne_points, tiles = [], [], []
	tfrecord_paths = []
	tile_point_distances = []
	rectangles = {}
	final_layer_weights = {}
	logits = {}

	def __init__(self, leniency=1.5, expanded=True, focus=None, tile_zoom=15, num_tiles_x=50, resolution='high', 
					export=False, tile_um=None, use_fp16=True):
		# Global variables
		self.max_distance_factor = leniency
		self.mapping_method = 'expanded' if expanded else 'strict'
		self.focus = focus
		self.tile_zoom_factor = tile_zoom
		self.export = export
		self.DTYPE = tf.float16 if use_fp16 else tf.float32

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

	def load_slides(self, slides_array, category="None"):
		print(f"[SVS] Loading SVS slides ...")
		for slide in slides_array:
			name = slide[:-4]
			filetype = slide[-3:]
			path = slide

			try:
				slide = ops.OpenSlide(path)
			except ops.lowlevel.OpenSlideUnsupportedFormatError:
				print(f"Unable to read file from {path} , skipping")
				return None
	
			shape = slide.dimensions
			goal_thumb_area = 400*400
			y_x_ratio = shape[1] / shape[0]
			thumb_x = math.sqrt(goal_thumb_area / y_x_ratio)
			thumb_y = thumb_x * y_x_ratio
			thumb_ratio = thumb_x / shape[0]
			thumb = slide.get_thumbnail((int(thumb_x), int(thumb_y)))
			MPP = float(slide.properties[ops.PROPERTY_NAME_MPP_X]) # Microns per pixel

			# Calculate tile index -> cooordinates dictionary
			coords = []
			extract_px = int(self.tile_um / MPP)
			stride = int(extract_px / self.stride_div)
			for y in range(0, (shape[1]+1) - extract_px, stride):
				for x in range(0, (shape[0]+1) - extract_px, stride):
					if ((y % extract_px == 0) and (x % extract_px == 0)):
						# Indicates unique (non-overlapping tile)
						coords.append([x, y])

			self.SLIDES.update({name: { "name": name,
										"path": path,
										"type": filetype,
										"category": category,
										"thumb": thumb,
										"ratio": thumb_ratio,
										"MPP": MPP,
										"tile_extr_px": int(self.tile_um / MPP),
										"tile_size": int(self.tile_um / MPP) * thumb_ratio,
										'coords':coords} })
			self.SVS = name

	def generate_final_layer_from_tfrecords(self, tfrecord_array, model, image_size):
		log.info(f"Calculating final layer weights from model {sfutil.green(model)}", 1)

		# Load model
		_model = tf.keras.models.load_model(model)
		loaded_model = tf.keras.models.Model(inputs=[_model.input, _model.layers[0].layers[0].input],
											 outputs=[_model.layers[0].layers[-1].output, _model.layers[-1].output])
		num_classes = _model.layers[-1].output_shape[-1]

		results = []

		def _parse_function(record):
			features = tf.io.parse_single_example(record, tfrecords.FEATURE_DESCRIPTION)
			case = features['case']
			image_string = features['image_raw']
			image = tf.image.decode_jpeg(image_string, channels=3)
			image = tf.image.per_image_standardization(image)
			image = tf.image.convert_image_dtype(image, self.DTYPE)
			image.set_shape([image_size, image_size, 3])
			return image, case

		# Calculate final layer weights for each tfrecord
		for tfrecord in tfrecord_array:
			log.info(f"Calculating weights from {tfrecord}", 2)
			dataset = tf.data.TFRecordDataset(tfrecord)
			dataset = dataset.map(_parse_function, num_parallel_calls=8)
			dataset = dataset.batch(self.BATCH_SIZE, drop_remainder=False)
			self.tfrecord_paths += [tfrecord]
			tfrecord_index = self.tfrecord_paths.index(tfrecord)

			fl_weights_arr = []
			logits_arr = []
			cases_arr = []
			indices_arr = []
			for i, batch_images, batch_cases in enumerate(dataset):
				sys.stdout.write(f"\r - Working on batch {i}")
				sys.stdout.flush()
				indices = list(range(0+i*self.BATCH_SIZE, self.BATCH_SIZE+i*self.BATCH_SIZE))
				fl_weights, logits = loaded_model.predict([batch_images, batch_images])
				fl_weights_arr = fl_weights if fl_weights_arr == [] else np.concatenate([fl_weights_arr, fl_weights])
				logits_arr = logits if logits_arr == [] else np.concatenate([logits_arr, logits])
				cases_arr = batch_cases if cases_arr == [] else np.concatenate([cases_arr, batch_cases])
				indices_arr = indices if indices_arr == [] else np.concatenate([indices_arr, indices])
			sys.stdout.write("\r\033[K")
			sys.stdout.flush()

			tfrecord_label = np.array([tfrecord_index] * len(cases_arr))
			# Join the weights, logits, and case labels into a 2D stack
			tfrecord_results = np.stack(fl_weights_arr, logits_arr, cases_arr, tfrecord_label, indices_arr)
			results = tfrecord_results if results == [] else np.concatenate([results, tfrecord_results])

		# Returns a 2D array, with each element containing FL weights, logits, case name, tfrecord name, and tfrecord indices
		return results           

	def generate_from_tfrecords(self, tfrecord_array, model, image_size):
		self.final_layer_weights = self.generate_final_layer_from_tfrecords(tfrecord_array, model, image_size)
		print("Printing final layer weights...")
		print(self.final_layer_weights)
		sys.exit()

		dl_coord = gen_umap(self.final_layer_weights[:,0])
		self.load_coordinates(dl_coord)
		self.place_tile_outlines()
		if len(self.SLIDES):
			self.generate_hover_events()
		self.calculate_distances()
		self.pair_tiles_and_points()
		if len(self.SLIDES):
			self.draw_slides()
		if self.focus:
			self.focus_category(self.focus)

	def load_coordinates(self, coord):
		log.info("Loading dimensionality reduction coordinates and plotting points...", 1)
		points_x = []
		points_y = []
		point_index = 0
		for i, p in enumerate(coord):
			meta = self.final_layer_weights[i]
			tile_num = int(meta[0])
			case = meta[2]
			tfrecord = meta[3]
			tfrecord_index = meta[4]
			category = 'none'#meta[2]
			points_x.append(p[0])
			points_y.append(p[1])
			self.points.append({'x':p[0],
								'y':p[1],
								'index':point_index,
								'tile_num':tile_num,
								'neighbors':[],
								'category':category,
								'case':case,
								'tfrecord':tfrecord,
								'tfrecord_index':tfrecord_index,
								'paired_tile':None,
								'image_path':join(self.tile_root, case, f"{case}_{tile_num}.jpg")})
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
		for point in self.points:
			x_index = int((point['x'] - min_x) / self.tile_size)
			y_index = int((point['y'] - min_y) / self.tile_size)
			for g in self.GRID:
				if g['x_index'] == x_index and g['y_index'] == y_index:
					g['points'].append(point['index'])

	def place_tile_outlines(self):
		log.info("Placing tile outlines...", 1)
		# Find max GRID density
		max_grid_density = 1
		for g in self.GRID:
			max_grid_density = max(max_grid_density, len(g['points']))
		for grid_tile in self.GRID:
			rect_size = min((len(grid_tile['points']) / max_grid_density) * self.tile_zoom_factor, 1) * self.tile_size

			tile = Rectangle((grid_tile['x'] - rect_size/2, 
							  grid_tile['y'] - rect_size/2), 
							  rect_size, 
							  rect_size, 
							  fill=True, alpha=1, facecolor='white', edgecolor="#cccccc")
			self.ax.add_patch(tile)

			grid_tile['size'] = rect_size
			grid_tile['rectangle'] = tile
			grid_tile['neighbors'] = []
			grid_tile['paired_point'] = None

	def generate_hover_events(self):
		def hover(event):
			# Check if mouse hovering over scatter plot
			prior_tile = None
			empty = True
			if self.ax.contains(event)[0]:
				for tile in self.GRID:
					if tile['rectangle'].contains(event)[0]:
						if prior_tile == tile: return
						if prior_tile:
							prior_tile['active'] = False
						tile['active'] = True
						prior_tile = tile
						empty = False

						if self.svs_background: self.fig.canvas.restore_region(self.svs_background)
						for index in tile['points']:
							point = self.points[index]
							case = point['case']
							if case in self.SLIDES:
								slide = self.SLIDES[case]
								size = slide['tile_size']
								origin_x, origin_y = slide['coords'][point['tile_num']]
								origin_x *= slide['ratio']
								origin_y *= slide['ratio']
								tile_outline = Rectangle((origin_x,# - size/2, 
														origin_y),# - size/2), 
														size, 
														size, 
														fill=None, alpha=1, color='green',
														zorder=100)
								self.ax_thumbnail.add_artist(tile_outline)
								self.ax_thumbnail.draw_artist(tile_outline)
						self.fig.canvas.blit(self.ax_thumbnail.bbox)
						return
			if not empty:
				self.fig.canvas.restore_region(self.svs_background)
				self.fig.canvas.blit(self.ax_thumbnail.bbox)
				empty = True
				
		def resize(event):
			'''for rect in list(self.rectangles):
				self.rectangles[rect].remove()'''
			self.fig.canvas.restore_region(self.svs_background)
			self.fig.canvas.draw()
			self.svs_background = self.fig.canvas.copy_from_bbox(self.ax_thumbnail.bbox)

		self.fig.canvas.mpl_connect('motion_notify_event', hover)
		self.fig.canvas.mpl_connect('resize_event', resize)

	def calculate_distances(self):
		log.info("Calculating tile-point distances...")
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
		log.info("Placing image tiles...", 1)
		num_placed = 0
		if self.mapping_method == 'strict':
			for tile in self.GRID:
				if not len(tile['distances']): continue
				closest_point = tile['distances'][0][0]
				point = self.points[closest_point]
				tile_image = plt.imread(point['image_path'])
				tile_alpha, num_case, num_other = 1, 0, 0
				if self.SVS and len(tile['points']):
					for point_index in tile['points']:
						point = self.points[point_index]
						if point['case'] == self.SVS:
							num_case += 1
						else:
							num_other += 1
					fraction_svs = num_case / (num_other + num_case)
					tile_alpha = fraction_svs
				if not self.export:
					tile_image = cv2.resize(tile_image, (0,0), fx=0.25, fy=0.25)
				image = self.ax.imshow(tile_image, aspect='equal', origin='lower', extent=[tile['x']-tile['size']/2, 
																						tile['x']+tile['size']/2,
																						tile['y']-tile['size']/2,
																						tile['y']+tile['size']/2], zorder=99, alpha=tile_alpha)
				tile['image'] = image
				num_placed += 1
			print(f"[INFO] Num placed: {num_placed}")
		elif self.mapping_method == 'expanded':
			for distance_pair in self.tile_point_distances:
				# Attempt to place pair, skipping if unable (due to other prior pair)
				point = self.points[distance_pair['point_index']]
				tile = self.GRID[distance_pair['tile_index']]
				if not (point['paired_tile'] or tile['paired_point']):
					point['paired_tile'] = True
					tile['paired_point'] = True

					tile_image = plt.imread(point['image_path'])
					if not self.export:
						tile_image = cv2.resize(tile_image, (0,0), fx=0.25, fy=0.25)
					image = self.ax.imshow(tile_image, aspect='equal', origin='lower', extent=[tile['x']-self.tile_size/2, 
																					tile['x']+self.tile_size/2,
																					tile['y']-self.tile_size/2,
																					tile['y']+self.tile_size/2], zorder=99)		
					tile['image'] = image
					num_placed += 1
			print(f"[INFO] Num placed: {num_placed}")

	def draw_slides(self):
		log.info("Drawing slides...", 1)
		self.ax_thumbnail = self.fig.add_subplot(122)
		self.ax_thumbnail.set_xticklabels([])
		self.ax_thumbnail.set_yticklabels([])
		name = list(self.SLIDES)[0]
		self.ax_thumbnail.imshow(self.SLIDES[name]['thumb'])
		self.fig.canvas.draw()
		self.svs_background = self.fig.canvas.copy_from_bbox(self.ax_thumbnail.bbox)
		self.SLIDES[name]['plot'] = self.ax_thumbnail
		self.highlight_slide_on_mosaic(name)

	def focus_category(self, category):
		for tile in self.GRID:
			if not len(tile['points']): continue
			num_cat, num_other = 0, 0
			for point_index in tile['points']:
				point = self.points[point_index]
				if point['category'] == category:
					num_cat += 1
				else:
					num_other += 1
			alpha = num_cat / (num_other + num_cat)
			tile['image'].set_alpha(alpha)

	def finish_mosaic(self, export):
		log.info("Displaying/exporting figure...", 1)
		self.ax.autoscale(enable=True, tight=None)
		if export:
			save_path = join(self.tile_root, f'Mosaic-{self.num_tiles_x}.png')
			plt.savefig(save_path, bbox_inches='tight')
			log.complete(f"Saved figure to {save_path}", 1)
			plt.close()
		else:
			while True:
				try:
					plt.show()
				except UnicodeDecodeError:
					continue
				break

# TODO
# - load metadata
# - umap pkl
# - replace plt.imread with automatic use of TFRecord    
# - metadata with tfrecords / categories