import math
import time
import cv2
import sys

import numpy as np
import matplotlib.pyplot as plt
import slideflow.util as sfutil
import slideflow.io as sfio

from random import shuffle
from matplotlib import patches
from os.path import join
from slideflow.util import log, ProgressBar

from multiprocessing.dummy import Pool as DPool

class Mosaic:
	GRID = []
	points = []

	def __init__(self, umap, focus=None, leniency=1.5, expanded=False, tile_zoom=15, num_tiles_x=50, resolution='high', 
					export=True):
		'''Generate a mosaic map.

		Args:
			umap:			TFRecordUMAP object
			focus:			List of tfrecords to highlight on the mosaic
			leniency:		UMAP leniency
			expanded:		If true, will try to fill in blank spots on the UMAP with nearby tiles. Takes exponentially longer to generate.
			tile_zoom:		Zoom level
			num_tiles_x:	Mosaic map grid size
			resolution:		Resolution of exported figure; either 'high', 'medium', or 'low'.'''

		FOCUS_SLIDE = None
		tile_point_distances = []	
		max_distance_factor = leniency
		mapping_method = 'expanded' if expanded else 'strict'
		tile_zoom_factor = tile_zoom
		export = export
		self.num_tiles_x = num_tiles_x
		self.tfrecords_paths = umap.tfrecords
		
		# Initialize figure
		log.info("Initializing figure...", 1)
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
								'neighbors':[],
								'category':'none',
								'slide':slide,
								'tfrecord':self._get_tfrecords_from_slide(slide),
								'tfrecord_index':umap.point_meta[i]['index'],
								'paired_tile':None })
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
									'distances':[],
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
			grid_tile['neighbors'] = []
			grid_tile['paired_point'] = None

		# Then, calculate distances from each point to each spot on the grid
		if mapping_method not in ('strict', 'expanded'):
			raise TypeError("Unknown mapping method")

		# ---- new function test
		def calc_distance(tile):
			if mapping_method == 'strict':
				# Calculate distance for each point within the grid tile from center of the grid tile
				distances = []
				for point_index in tile['points']:
					point = self.points[point_index]
					distance = np.linalg.norm(tile['coord'] - point['coord'])
					distances.append([point['global_index'], distance])
				distances.sort(key=lambda d: d[1])
				tile['distances'] = distances
			elif mapping_method == 'expanded':
				# Calculate distance for each point within the entire grid from center of the grid tile
				distances = []
				for point in self.points:
					distance = np.linalg.norm(tile['coord'] - point['coord'])
					distances.append([point['global_index'], distance])
				distances.sort(key=lambda d: d[1])
				for d in distances:
					if d[1] <= max_distance:
						tile['neighbors'].append(d)
						self.points[d[0]]['neighbors'].append([tile['grid_index'], d[1]])
						tile_point_distances.append({'distance': d[1],
													'grid_index':tile['grid_index'],
													'point_index':d[0]})
					else:
						break

		log.empty("Calculating tile-point distances...", 1)
		tile_point_start = time.time()
		pool = DPool(8)

		for i, _ in enumerate(pool.imap_unordered(calc_distance, self.GRID), 1):
			sys.stderr.write(f'\rCompleted {i/len(self.GRID):.2%}')

		pool.close()
		pool.join()
		tile_point_end = time.time()
		log.info(f"Calculations complete ({tile_point_end-tile_point_start:.0f} sec)", 1)

		# ----------------------

		'''
		# Calculate tile-point distances
		log.empty("Calculating tile-point distances...", 1)
		tile_point_start = time.time()
		pb = ProgressBar()
		pb_id = pb.add_bar(0, len(self.GRID))
		for i, tile in enumerate(self.GRID):
			pb.update(pb_id, i)
			calc_distance(tile)		
		pb.end()
		tile_point_end = time.time()
		log.info(f"Calculations complete ({tile_point_end-tile_point_start:.0f} sec)", 1)
		'''

		if mapping_method == 'expanded':
			tile_point_distances.sort(key=lambda d: d['distance'])

		# Then, pair grid tiles and points according to their distances
		log.empty("Placing image tiles...", 1)
		num_placed = 0
		if mapping_method == 'strict':
			for tile in self.GRID:
				if not len(tile['distances']): continue
				closest_point = tile['distances'][0][0]
				point = self.points[closest_point]

				_, tile_image = sfio.tfrecords.get_tfrecord_by_index(point['tfrecord'], point['tfrecord_index'], decode=False)
				image_arr = np.fromstring(tile_image.numpy(), np.uint8)
				tile_image_bgr = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
				tile_image = cv2.cvtColor(tile_image_bgr, cv2.COLOR_BGR2RGB)				

				tile_alpha, num_slide, num_other = 1, 0, 0
				if FOCUS_SLIDE and len(tile['points']):
					for point_index in tile['points']:
						point = self.points[point_index]
						if point['slide'] == FOCUS_SLIDE:
							num_slide += 1
						else:
							num_other += 1
					fraction_slide = num_slide / (num_other + num_slide)
					tile_alpha = fraction_slide
				if not export:
					tile_image = cv2.resize(tile_image, (0,0), fx=0.25, fy=0.25)
				image = ax.imshow(tile_image, aspect='equal', origin='lower', extent=[tile['x']-tile['size']/2, 
																						tile['x']+tile['size']/2,
																						tile['y']-tile['size']/2,
																						tile['y']+tile['size']/2], zorder=99, alpha=tile_alpha)
				tile['image'] = image
				num_placed += 1
		elif mapping_method == 'expanded':
			for distance_pair in tile_point_distances:
				# Attempt to place pair, skipping if unable (due to other prior pair)
				point = self.points[distance_pair['point_index']]
				tile = self.GRID[distance_pair['grid_index']]
				if not (point['paired_tile'] or tile['paired_point']):
					point['paired_tile'] = True
					tile['paired_point'] = True

					_, tile_image = sfio.tfrecords.get_tfrecord_by_index(point['tfrecord'], point['tfrecord_index'], decode=False)
					image_arr = np.fromstring(tile_image.numpy(), np.uint8)
					tile_image_bgr = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
					tile_image = cv2.cvtColor(tile_image_bgr, cv2.COLOR_BGR2RGB)

					if not export:
						tile_image = cv2.resize(tile_image, (0,0), fx=0.25, fy=0.25)
					image = ax.imshow(tile_image, aspect='equal', origin='lower', extent=[tile['x']-tile_size/2,
																					tile['x']+tile_size/2,
																					tile['y']-tile_size/2,
																					tile['y']+tile_size/2], zorder=99)
					tile['image'] = image
					num_placed += 1
		log.info(f"Num placed: {num_placed}", 2)

		# If desired, highlight certain tiles according to a focus list
		if focus:
			for tile in self.GRID:
				if not len(tile['points']): continue
				num_cat, num_other = 0, 0
				for point_index in tile['points']:
					point = self.points[point_index]
					if point['tfrecord'] in focus:
						num_cat += 1
					else:
						num_other += 1
				alpha = num_cat / (num_other + num_cat)
				tile['image'].set_alpha(alpha)

		# Finally, finish the mosaic figure
		ax.autoscale(enable=True, tight=None)

	def _get_tfrecords_from_slide(self, slide):
		for tfr in self.tfrecords_paths:
			if sfutil.path_to_name(tfr) == slide:
				return tfr
		log.error(f"Unable to find TFRecord path for slide {sfutil.green(slide)}", 1)

	def focus(self, tfrecords):
		# If desired, highlight certain tiles according to a focus list
		if tfrecords:
			for tile in self.GRID:
				if not len(tile['points']): continue
				num_cat, num_other = 0, 0
				for point_index in tile['points']:
					point = self.points[point_index]
					if point['tfrecord'] in focus:
						num_cat += 1
					else:
						num_other += 1
				alpha = num_cat / (num_other + num_cat)
				tile['image'].set_alpha(alpha)
		else:
			for tile in self.GRID:
				if not len(tile['points']): continue
				tile['image'].set_alpha(1)

	def save(self, directory):
		log.empty("Exporting figure...", 1)
		save_path = join(directory, f'Mosaic-{self.num_tiles_x}.png')
		plt.savefig(save_path, bbox_inches='tight')
		log.complete(f"Saved figure to {sfutil.green(save_path)}", 1)
		plt.close()

	def display(self):
		log.empty("Displaying figure...")
		while True:
			try:
				plt.show()
			except UnicodeDecodeError:
				continue
			break