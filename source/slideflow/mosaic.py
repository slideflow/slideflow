import math
import time
import cv2
import sys
import csv

import numpy as np
import matplotlib.pyplot as plt
import slideflow.util as sfutil
import slideflow.io as sfio

from random import shuffle
from matplotlib import patches
from os.path import join
from slideflow.util import log, StainNormalizer
from slideflow.statistics import get_centroid_index
from multiprocessing.dummy import Pool as DPool
from functools import partial

class MosaicError(Exception):
	pass

class Mosaic:
	'''Visualization of tiles as mapped using dimensionality reduction.'''
	GRID = []
	points = []

	def __init__(self, umap, focus=None, leniency=1.5, expanded=False, tile_zoom=15, num_tiles_x=50, resolution='high', 
					relative_size=False, tile_select='nearest', tile_meta=None, normalizer=None, normalizer_source=None):
		'''Generate a mosaic map.

		Args:
			umap:				TFRecordMap object
			focus:				List of tfrecords (paths) to highlight on the mosaic
			leniency:			UMAP leniency
			expanded:			If true, will try to fill in blank spots on the UMAP with nearby tiles. Takes exponentially longer to generate.
			tile_zoom:			Zoom level
			num_tiles_x:		Mosaic map grid size
			resolution:			Resolution of exported figure; either 'high', 'medium', or 'low'.
			relative_size:		If True, will physically size grid images in proportion to the number of tiles within the grid space.
			tile_select:		Determines how to choose a tile for display on each grid space. Either 'nearest' or 'centroid'. 
									If nearest, will display tile nearest to center of grid.
									If centroid, for each grid, will calculate which tile is nearest to centroid using data in tile_meta
			tile_meta:			Dictionary. Metadata for tiles, used if tile_select. Dictionary should have slide names as keys, mapped to
									List of metadata (length of list = number of tiles in slide)
			normalizer:			String. Normalizer to apply to images taken from TFRecords.
			normalizer_source:	String, path. Path to image to use as normalizer source.'''

		FOCUS_SLIDE = None
		tile_point_distances = []	
		max_distance_factor = leniency
		mapping_method = 'expanded' if expanded else 'strict'
		tile_zoom_factor = tile_zoom # TODO: investigate if this argument is required
		self.mapped_tiles = {}
		self.umap = umap
		self.num_tiles_x = num_tiles_x
		self.tfrecords_paths = umap.tfrecords

		# Setup normalization
		if normalizer: log.info(f"Using realtime {normalizer} normalization", 1)
		self.normalizer = None if not normalizer else StainNormalizer(method=normalizer, source=normalizer_source)
		
		# Initialize figure
		log.empty("Initializing figure...", 1)
		if resolution not in ('high', 'low'):
			log.warn(f"Unknown resolution option '{resolution}', defaulting to low resolution", 1)
		if resolution == 'high':
			fig = plt.figure(figsize=(200,200))
			ax = fig.add_subplot(111, aspect='equal')
		else:
			fig = plt.figure(figsize=(24,18))
			ax = fig.add_subplot(121, aspect='equal')
		ax.set_facecolor("#dfdfdf")
		fig.tight_layout()
		plt.subplots_adjust(left=0.02, bottom=0, right=0.98, top=1, wspace=0.1, hspace=0)
		ax.set_aspect('equal', 'box')
		ax.set_xticklabels([])
		ax.set_yticklabels([])

		# First, load UMAP coordinates	
		log.empty("Loading coordinates and plotting points...", 1)
		for i in range(len(umap.x)):
			slide = umap.point_meta[i]['slide']
			self.points.append({'coord':np.array((umap.x[i], umap.y[i])),
								'global_index': i,
								'category':'none',
								'slide':slide,
								'tfrecord':self._get_tfrecords_from_slide(slide),
								'tfrecord_index':umap.point_meta[i]['index'],
								'paired_tile':None,
								'meta':None if not tile_meta else tile_meta[slide][umap.point_meta[i]['index']]})
		x_points = [p['coord'][0] for p in self.points]
		y_points = [p['coord'][1] for p in self.points]
		_x_width = max(x_points) - min(x_points)
		_y_width = max(y_points) - min(y_points)
		buffer = (_x_width + _y_width)/2 * 0.05
		max_x = max(x_points) + buffer
		min_x = min(x_points) - buffer
		max_y = max(y_points) + buffer
		min_y = min(y_points) - buffer

		log.info(f"Loaded {len(self.points)} points.", 2)

		tile_size = (max_x - min_x) / self.num_tiles_x
		self.num_tiles_y = int((max_y - min_y) / tile_size)
		max_distance = math.sqrt(2*((tile_size/2)**2)) * max_distance_factor

		# Initialize grid
		for j in range(self.num_tiles_y):
			for i in range(self.num_tiles_x):
				x = ((tile_size/2) + min_x) + (tile_size * i)
				y = ((tile_size/2) + min_y) + (tile_size * j)
				self.GRID.append({	'coord': np.array((x, y)),
									'x_index': i,
									'y_index': j,
									'grid_index': len(self.GRID),
									'size': tile_size,
									'points':[],
									'nearest_index':[],
									'active': False,
									'image': None})

		# Add point indices to grid
		points_added = 0
		for point in self.points:
			x_index = int((point['coord'][0] - min_x) / tile_size)
			y_index = int((point['coord'][1] - min_y) / tile_size)
			for g in self.GRID:
				if g['x_index'] == x_index and g['y_index'] == y_index:
					g['points'].append(point['global_index'])
					points_added += 1
		for g in self.GRID:
			shuffle(g['points'])
		log.info(f"{points_added} points added to grid", 2)

		# Next, prepare mosaic grid by placing tile outlines
		log.empty("Placing tile outlines...", 1)
		max_grid_density = 1
		for g in self.GRID:
			max_grid_density = max(max_grid_density, len(g['points']))
		for grid_tile in self.GRID:
			rect_size = min((len(grid_tile['points']) / max_grid_density) * tile_zoom_factor, 1) * tile_size

			tile = patches.Rectangle((grid_tile['coord'][0] - rect_size/2, 
							  		  grid_tile['coord'][1] - rect_size/2), 
									  rect_size, 
							  		  rect_size, 
									  fill=True, alpha=1, facecolor='white', edgecolor="#cccccc")
			ax.add_patch(tile)

			grid_tile['size'] = rect_size
			grid_tile['rectangle'] = tile
			grid_tile['paired_point'] = None

		# Then, calculate distances from each point to each spot on the grid
		if mapping_method not in ('strict', 'expanded'):
			raise TypeError("Unknown mapping method; must be strict or expanded")
		else:
			log.info(f"Mapping method: {mapping_method}", 2)

		if tile_select not in ('nearest', 'centroid'):
			raise TypeError("Unknown tile selection method; must be nearest or centroid")
		else:
			log.info(f"Tile selection method: {tile_select}", 2)

		def calc_distance(tile, global_point_coords):
			if mapping_method == 'strict':
				# Calculate distance for each point within the grid tile from center of the grid tile
				point_coords = np.asarray([self.points[global_index]['coord'] for global_index in tile['points']])
				if len(point_coords):
					if tile_select == 'nearest':
						distances = np.linalg.norm(point_coords - tile['coord'], ord=2, axis=1.)
						tile['nearest_index'] = tile['points'][np.argmin(distances)]
					elif not tile_meta:
						raise MosaicError("Unable to calculate centroid for mosaic if tile_meta not provided.")
					else:
						centroid_index = get_centroid_index([self.points[global_index]['meta'] for global_index in tile['points']])
						tile['nearest_index'] = tile['points'][centroid_index]
			elif mapping_method == 'expanded':
				# Calculate distance for each point within the entire grid from center of the grid tile
				distances = np.linalg.norm(global_point_coords - tile['coord'], ord=2, axis=1.)
				for i, distance in enumerate(distances):
					if distance <= max_distance:
						tile_point_distances.append({'distance': distance,
													'grid_index':tile['grid_index'],
													'point_index':self.points[i]['global_index']})

		log.empty("Calculating tile-point distances...", 1)
		tile_point_start = time.time()
		global_point_coords = np.asarray([p['coord'] for p in self.points])
		pool = DPool(8)
		for i, _ in enumerate(pool.imap_unordered(partial(calc_distance, global_point_coords=global_point_coords), self.GRID), 1):
			sys.stderr.write(f'\rCompleted {i/len(self.GRID):.2%}')
		pool.close()
		pool.join()
		tile_point_end = time.time()
		sys.stdout.write("\r\033[K")
		log.info(f"Calculations complete ({tile_point_end-tile_point_start:.0f} sec)", 2)

		if mapping_method == 'expanded':
			tile_point_distances.sort(key=lambda d: d['distance'])

		# Then, pair grid tiles and points according to their distances
		log.empty("Placing image tiles...", 1)
		num_placed = 0
		if mapping_method == 'strict':
			for tile in self.GRID:
				if not len(tile['points']): continue
				closest_point = tile['nearest_index']
				point = self.points[closest_point]

				if not point['tfrecord']:
					log.error(f"The tfrecord {point['slide']} was not found in the list of paths provided by the input umap; please ensure the TFRecord exists.", 1)
					continue

				_, tile_image = sfio.tfrecords.get_tfrecord_by_index(point['tfrecord'], point['tfrecord_index'], decode=False)
				self.mapped_tiles.update({point['tfrecord']: point['tfrecord_index']})
				tile_image = self._decode_image_string(tile_image.numpy())
				
				tile_alpha, num_slide, num_other = 1, 0, 0
				display_size = tile_size
				if relative_size:
					if FOCUS_SLIDE and len(tile['points']):
						for point_index in tile['points']:
							point = self.points[point_index]
							if point['slide'] == FOCUS_SLIDE:
								num_slide += 1
							else:
								num_other += 1
						fraction_slide = num_slide / (num_other + num_slide)
						tile_alpha = fraction_slide
					display_size = tile['size']
				image = ax.imshow(tile_image, aspect='equal', origin='lower', extent=[tile['coord'][0]-display_size/2, 
																						tile['coord'][0]+display_size/2,
																						tile['coord'][1]-display_size/2,
																						tile['coord'][1]+display_size/2], zorder=99, alpha=tile_alpha)
				tile['image'] = image
				num_placed += 1
		elif mapping_method == 'expanded':
			for i, distance_pair in enumerate(tile_point_distances):
				print(f"\rPlacing tile {i}/{len(tile_point_distances)}...", end="")
				# Attempt to place pair, skipping if unable (due to other prior pair)
				point = self.points[distance_pair['point_index']]
				tile = self.GRID[distance_pair['grid_index']]
				if not (point['paired_tile'] or tile['paired_point']):
					point['paired_tile'] = True
					tile['paired_point'] = True

					_, tile_image = sfio.tfrecords.get_tfrecord_by_index(point['tfrecord'], point['tfrecord_index'], decode=False)
					self.mapped_tiles.update({point['tfrecord']: point['tfrecord_index']})
					tile_image = self._decode_image_string(tile_image.numpy())					

					image = ax.imshow(tile_image, aspect='equal', origin='lower', extent=[tile['coord'][0]-tile_size/2,
																					tile['coord'][0]+tile_size/2,
																					tile['coord'][1]-tile_size/2,
																					tile['coord'][1]+tile_size/2], zorder=99)
					tile['image'] = image
					num_placed += 1
			print("\r\033[K", end="")
		log.info(f"Num placed: {num_placed}", 2)

		# Focus on a subset of TFRecords if desired
		if focus: self.focus(focus)

		# Finally, finish the mosaic figure
		ax.autoscale(enable=True, tight=None)

	def _get_tfrecords_from_slide(self, slide):
		'''Using the internal list of TFRecord paths, returns the path to a TFRecord for a given corresponding slide.'''
		for tfr in self.tfrecords_paths:
			if sfutil.path_to_name(tfr) == slide:
				return tfr
		log.error(f"Unable to find TFRecord path for slide {sfutil.green(slide)}", 1)

	def _decode_image_string(self, string):	
		'''Internal method to convert a JPEG string (as stored in TFRecords) to an RGB array.'''
		if self.normalizer:
			tile_image = self.normalizer.jpeg_to_rgb(string)
		else:
			image_arr = np.fromstring(string, np.uint8)
			tile_image_bgr = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
			tile_image = cv2.cvtColor(tile_image_bgr, cv2.COLOR_BGR2RGB)
		return tile_image

	def focus(self, tfrecords):
		'''Highlights certain tiles according to a focus list if list provided, or resets highlighting if no tfrecords provided.'''
		if tfrecords:
			for tile in self.GRID:
				if not len(tile['points']) or not tile['image']: continue
				num_cat, num_other = 0, 0
				for point_index in tile['points']:
					point = self.points[point_index]
					if point['tfrecord'] in tfrecords:
						num_cat += 1
					else:
						num_other += 1
				alpha = num_cat / (num_other + num_cat)
				tile['image'].set_alpha(alpha)
		else:
			for tile in self.GRID:
				if not len(tile['points']) or not tile['image']: continue
				tile['image'].set_alpha(1)

	def save(self, filename):
		'''Saves the mosaic map figure to the given filename.'''
		log.empty("Exporting figure...", 1)
		plt.savefig(filename, bbox_inches='tight')
		log.complete(f"Saved figure to {sfutil.green(filename)}", 1)
		plt.close()

	def save_report(self, filename):
		'''Saves a report of which tiles (and their corresponding slide) were displayed on the Mosaic map, in CSV format.'''
		with open(filename, 'w') as f:
			writer = csv.writer(f)
			writer.writerow(['slide', 'index'])
			for tfr in self.mapped_tiles:
				writer.writerow([tfr, self.mapped_tiles[tfr]])
		log.complete(f"Mosaic report saved to {sfutil.green(filename)}", 1)

	def display(self):
		'''Displays the mosaic map as an interactive matplotlib figure.'''
		log.empty("Displaying figure...")
		while True:
			try:
				plt.show()
			except UnicodeDecodeError:
				continue
			break