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
from random import shuffle

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
	try:
		layout = umap.UMAP(n_components=2, verbose=True, n_neighbors=20, min_dist=0.01, metric="cosine").fit_transform(array)
	except ValueError:
		log.error("Error performing UMAP. Please make sure you are supplying a non-empty TFRecord array and that the TFRecords are not empty.")
		sys.exit()
	return normalize_layout(layout)
	
#def gen_tsne_array(array):
#    layout = TSNE(n_components=2, verbose=True, metric="cosine", learning_rate=10, perplexity=50).fit_transform(array)
#    return normalize_layout(layout)

class Mosaic:
	SVS = None
	BATCH_SIZE = 16
	SLIDES = {}
	GRID = []
	stride_div = 1
	ax_thumbnail = None
	svs_background = None
	metadata, points, tiles = [], [], []
	tfrecord_paths = []
	tile_point_distances = []
	rectangles = {}
	final_layer_weights = {}
	logits = {}

	def __init__(self, leniency=1.5, expanded=True, tile_zoom=15, num_tiles_x=50, resolution='high', 
					export=True, tile_um=None, use_fp16=True, save_dir=None):
		# Global variables
		self.max_distance_factor = leniency
		self.mapping_method = 'expanded' if expanded else 'strict'
		self.tile_zoom_factor = tile_zoom
		self.export = export
		self.DTYPE = tf.float16 if use_fp16 else tf.float32
		self.num_tiles_x = num_tiles_x
		self.save_dir = save_dir

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
		log.info(f"Loading SVS slides ...", 1)
		for slide in slides_array:
			name = sfutil.path_to_name(slide)
			filetype = sfutil.path_to_ext(slide)
			path = slide

			try:
				slide = ops.OpenSlide(path)
			except ops.lowlevel.OpenSlideUnsupportedFormatError:
				log.warn(f"Unable to read file from {path} , skipping", 1)
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
		complete_model=False
		try:
			loaded_model = tf.keras.models.Model(inputs=[_model.input, _model.layers[0].layers[0].input],
												outputs=[_model.layers[0].layers[-1].output, _model.layers[-1].output])
			num_classes = _model.layers[-1].output_shape[-1]
		except AttributeError:
			# Provides support for complete models that were not generated using Slideflow
			complete_model=True
			loaded_model = tf.keras.models.Model(inputs=[_model.input],
												 outputs=[_model.layers[-2].output])
			num_classes = 1
		

		results = []
		fl_weights_all, logits_all, slides_all, indices_all, images_all, tfrecord_all = [], [], [], [], [], []

		def _parse_function(record):
			features = tf.io.parse_single_example(record, tfrecords.FEATURE_DESCRIPTION)
			slide = features['slide']
			image_string = features['image_raw']
			raw_image = tf.image.decode_jpeg(image_string, channels=3)
			processed_image = tf.image.per_image_standardization(raw_image)
			processed_image = tf.image.convert_image_dtype(processed_image, self.DTYPE)
			processed_image.set_shape([image_size, image_size, 3])
			return processed_image, raw_image, slide

		# Calculate final layer weights for each tfrecord
		for tfrecord in tfrecord_array:
			log.info(f"Calculating weights from {sfutil.green(tfrecord)}", 2)
			dataset = tf.data.TFRecordDataset(tfrecord)

			dataset = dataset.map(_parse_function, num_parallel_calls=8)
			dataset = dataset.batch(self.BATCH_SIZE, drop_remainder=False)
			self.tfrecord_paths += [tfrecord]
			tfrecord_index = self.tfrecord_paths.index(tfrecord)

			fl_weights_arr, logits_arr, slides_arr, indices_arr, images_arr = [], [], [], [], []
			for i, data in enumerate(dataset):
				batch_processed_images, batch_raw_images, batch_slides = data
				batch_raw_images_np = batch_raw_images.numpy()
				sys.stdout.write(f"\r - Working on batch {i}")
				sys.stdout.flush()
				indices = list(range(len(indices_arr), len(indices_arr) + len(batch_processed_images)))
				if not complete_model:
					fl_weights, logits = loaded_model.predict([batch_processed_images, batch_processed_images])
				else:
					fl_weights = loaded_model.predict([batch_processed_images])
					logits = [-1] * self.BATCH_SIZE
				fl_weights_arr = fl_weights if fl_weights_arr == [] else np.concatenate([fl_weights_arr, fl_weights])
				logits_arr = logits if logits_arr == [] else np.concatenate([logits_arr, logits])
				slides_arr = batch_slides if slides_arr == [] else np.concatenate([slides_arr, batch_slides])
				images_arr = batch_raw_images_np if images_arr == [] else np.concatenate([images_arr, batch_raw_images_np])
				indices_arr = indices if indices_arr == [] else np.concatenate([indices_arr, indices])
			sys.stdout.write("\r\033[K")
			sys.stdout.flush()

			tfrecord_arr = np.array([tfrecord_index] * len(slides_arr))

			fl_weights_all = fl_weights_arr if fl_weights_all == [] else np.concatenate([fl_weights_all, fl_weights_arr])
			logits_all = logits_arr if logits_all == [] else np.concatenate([logits_all, logits_arr])
			slides_all = slides_arr if slides_all == [] else np.concatenate([slides_all, slides_arr])
			images_all = images_arr if images_all == [] else np.concatenate([images_all, images_arr])
			indices_all = indices_arr if indices_all == [] else np.concatenate([indices_all, indices_arr])
			tfrecord_all = tfrecord_arr if tfrecord_all == [] else np.concatenate([tfrecord_all, tfrecord_arr])\

		# Save final layer weights to CSV file
		header = ["Slide"] + [f"Logits{l}" for l in range(logits_all.shape[1])] + [f"FLNode{f}" for f in range(fl_weights_all.shape[1])]
		flweights_file = join(self.save_dir, "final_layer_weights.csv")
		with open(flweights_file, 'w') as outfile:
			csvwriter = csv.writer(outfile)
			csvwriter.writerow(header)
			for i in range(len(slides_all)):
				slide = [slides_all[i].decode('utf-8')]
				logits = logits_all[i].tolist()
				flweights = fl_weights_all[i].tolist()
				row = slide + logits + flweights
				csvwriter.writerow(row)

		# Returns a 2D array, with each element containing FL weights, logits, slide name, tfrecord name, and tfrecord indices
		return fl_weights_all, logits_all, slides_all, images_all, indices_all, tfrecord_all	

	def generate_from_tfrecords(self, tfrecord_array, model, image_size, focus=None):
		fl_weights, logits, slides, images, indices, tfrecords = self.generate_final_layer_from_tfrecords(tfrecord_array, model, image_size)
		
		dl_coord = gen_umap(fl_weights)
		self.load_coordinates(dl_coord, [slides, tfrecords, indices, images])
		self.place_tile_outlines()
		#if len(self.SLIDES):
		#	self.generate_hover_events()
		self.calculate_distances()
		self.pair_tiles_and_points()
		#if len(self.SLIDES):
		#	self.draw_slides()
		if focus:
			self.focus_tfrecords(focus)
		self.finish_mosaic()

	def load_coordinates(self, coord, meta):
		log.empty("Loading dimensionality reduction coordinates and plotting points...", 1)
		points_x = []
		points_y = []
		point_index = 0
		slides, tfrecords, indices, images = meta
		for i, p in enumerate(coord):
			slide = slides[i]
			tfrecord = self.tfrecord_paths[tfrecords[i]]
			tfrecord_index = indices[i]
			category = 'none'
			points_x.append(p[0])
			points_y.append(p[1])
			self.points.append({'x':p[0],
								'y':p[1],
								'index':point_index,
								'neighbors':[],
								'category':category,
								'slide':slide,
								'tfrecord':tfrecord,
								'tfrecord_index':tfrecord_index,
								'paired_tile':None,
								'image_path':None,
								'image_data':images[i]})
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

	'''def generate_hover_events(self):
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
							slide = point['slide']
							if slide in self.SLIDES:
								slide = self.SLIDES[slide]
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
			#for rect in list(self.rectangles):
			#	self.rectangles[rect].remove()
			self.fig.canvas.restore_region(self.svs_background)
			self.fig.canvas.draw()
			self.svs_background = self.fig.canvas.copy_from_bbox(self.ax_thumbnail.bbox)

		self.fig.canvas.mpl_connect('motion_notify_event', hover)
		self.fig.canvas.mpl_connect('resize_event', resize)'''

	'''def draw_slides(self):
		log.info("Drawing slides...", 1)
		self.ax_thumbnail = self.fig.add_subplot(122)
		self.ax_thumbnail.set_xticklabels([])
		self.ax_thumbnail.set_yticklabels([])
		name = list(self.SLIDES)[0]
		self.ax_thumbnail.imshow(self.SLIDES[name]['thumb'])
		self.fig.canvas.draw()
		self.svs_background = self.fig.canvas.copy_from_bbox(self.ax_thumbnail.bbox)
		self.SLIDES[name]['plot'] = self.ax_thumbnail
		#self.highlight_slide_on_mosaic(name)'''

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
				tile_image = point['image_data']
				tile_alpha, num_slide, num_other = 1, 0, 0
				if self.SVS and len(tile['points']):
					for point_index in tile['points']:
						point = self.points[point_index]
						if point['slide'] == self.SVS:
							num_slide += 1
						else:
							num_other += 1
					fraction_svs = num_slide / (num_other + num_slide)
					tile_alpha = fraction_svs
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
					tile_image = point['image_data']
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

# TODO
# - use pkl
# - use automatic TFRecord image reading to reduce RAM usage    