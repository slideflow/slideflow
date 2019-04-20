import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

import argparse
import json
import sys
import math
import csv
import pickle
import cv2

import openslide as ops

import os
from os.path import join, isfile, exists

# TODO: Consider converting to pyqtgraph to increase speed

class Mosaic:
	def __init__(self, args):
		self.metadata = []
		self.tsne_points = [] # format: (x, y, index)
		self.tiles = []
		self.rectangles = {}
		self.tile_um = args.um
		self.SLIDES = {}
		self.GRID = []
		self.num_tiles_x = args.detail
		self.stride_div = 1
		self.max_distance_factor = args.leniency
		self.tile_root = args.tile
		self.export = args.export
		self.ax_thumbnail = None
		self.tile_zoom_factor = 10
		self.SVS = None

		self.initiate_figure()
		self.load_metadata(args.meta)
		self.load_bookmark_state(args.bookmark)

	def generate(self):
		self.place_tile_outlines()
		self.generate_hover_events()
		self.calculate_distances()
		self.pair_tiles_and_points()
		if len(self.SLIDES): self.draw_slides()
		self.finish_mosaic(self.export)

	def initiate_figure(self):
		print("[Core] Initializing figure...")
		if self.export:
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

	def load_metadata(self, path):
		print("[Core] Loading metadata...")
		with open(path, 'r') as metadata_file:
			reader = csv.reader(metadata_file, delimiter='\t')
			self.meta_headers = next(reader, None)
			for row in reader:
				self.metadata.append(row)

	def load_bookmark_state(self, path):
		print("[Core] Loading t-SNE bookmark and plotting points...")
		with open(path, 'r') as bookmark_file:
			state = json.load(bookmark_file)
			projection_points = state[0]['projections']
			points_x = []
			points_y = []
			for i, p in enumerate(projection_points):
				if 'tsne-1' in p:
					meta = self.metadata[i]
					tile_num = int(meta[0])	
					points_x.append(p['tsne-0'])
					points_y.append(p['tsne-1'])
					point_dict = {'x':p['tsne-0'],
								  'y':p['tsne-1'],
								  'index':len(self.tsne_points),
								  'tile_num':tile_num,
								  'neighbors':[],
								  'paired_tile':None,
								  'image_path':join(self.tile_root, meta[1], f"{meta[1]}_{tile_num}.jpg")}
					for meta_index in range(1, len(self.meta_headers)):
						point_dict.update({self.meta_headers[meta_index].lower(): meta[meta_index]})
					self.tsne_points.append(point_dict)
			x_points = [p['x'] for p in self.tsne_points]
			y_points = [p['y'] for p in self.tsne_points]
			_x_width = max(x_points) - min(x_points)
			_y_width = max(y_points) - min(y_points)
			max_x = max(x_points) + _x_width * 0.02
			min_x = min(x_points) - _x_width * 0.02
			max_y = max(y_points) + _y_width * 0.02
			min_y = min(y_points) - _y_width * 0.02

		self.tile_size = (max_x - min_x) / self.num_tiles_x
		self.num_tiles_y = int((max_y - min_y) / self.tile_size)
		self.max_distance = math.sqrt(2*((self.tile_size/2)**2)) * self.max_distance_factor

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
								  'active': False})

		# Add point indices to grid
		for point in self.tsne_points:
			x_index = int((point['x'] - min_x) / self.tile_size)
			y_index = int((point['y'] - min_y) / self.tile_size)
			for g in self.GRID:
				if g['x_index'] == x_index and g['y_index'] == y_index:
					g['points'].append(point['index'])

		# Plot points
		self.tsne_plot = self.ax.scatter(points_x, points_y, s=1000, facecolors='none', edgecolors='green', alpha=0)# markersize = 5

		print(f"[Core] Loaded {len(self.tsne_points)} points.")

	def place_tile_outlines(self):
		print("[Mosaic] Placing tile outlines...")
		# tile_index = 0
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

	def load_slides(self, slides_array, directory, category="None"):
		print(f"[SVS] Loading SVS files from {directory} ...")
		for slide in slides_array:
			name = slide[:-4]
			filetype = slide[-3:]
			path = slide if not directory else join(directory, slide)

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

	def draw_slides(self):
		print("[SVS] Drawing slides...")
		self.ax_thumbnail = self.fig.add_subplot(122)
		self.ax_thumbnail.set_xticklabels([])
		self.ax_thumbnail.set_yticklabels([])
		name = list(self.SLIDES)[0]
		self.ax_thumbnail.imshow(self.SLIDES[name]['thumb'])
		self.fig.canvas.draw()
		self.svs_background = self.fig.canvas.copy_from_bbox(self.ax_thumbnail.bbox)
		self.SLIDES[name]['plot'] = self.ax_thumbnail
		self.highlight_slide_on_mosaic(name)

	def highlight_slide_on_mosaic(self, name):
		pass

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

						self.fig.canvas.restore_region(self.svs_background)
						for index in tile['points']:
							point = self.tsne_points[index]
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
			self.fig.canvas.draw()
			self.svs_background = self.fig.canvas.copy_from_bbox(self.ax_thumbnail.bbox)

		self.fig.canvas.mpl_connect('motion_notify_event', hover)
		self.fig.canvas.mpl_connect('resize_event', resize)

	def calculate_distances(self):
		print("[Mosaic] Calculating tile-point distances...")
		for tile in self.GRID:
			# Calculate distance for each point from center
			distances = []
			for point_index in tile['points']:
				point = self.tsne_points[point_index]
				distance = math.sqrt((point['x']-tile['x'])**2 + (point['y']-tile['y'])**2)
				distances.append([point['index'], distance])
			distances.sort(key=lambda d: d[1])
			tile['distances'] = distances

	def pair_tiles_and_points(self):
		print("[Mosaic] Placing image tiles...")
		num_placed = 0
		for tile in self.GRID:
			if not len(tile['distances']): continue
			closest_point = tile['distances'][0][0]
			point = self.tsne_points[closest_point]
			tile_image = plt.imread(point['image_path'])
			tile_alpha, num_case, num_other = 1, 0, 0
			if self.SVS and len(tile['points']):
				for point_index in tile['points']:
					point = self.tsne_points[point_index]
					if point['case'] == self.SVS:
						num_case += 1
					else:
						num_other += 1
				fraction_svs = num_case / (num_other + num_case)
				tile_alpha = fraction_svs
			if not self.export:
				tile_image = cv2.resize(tile_image, (0,0), fx=0.25, fy=0.25)
			self.ax.imshow(tile_image, aspect='equal', origin='lower', extent=[tile['x']-tile['size']/2, 
																			   tile['x']+tile['size']/2,
																			   tile['y']-tile['size']/2,
																			   tile['y']+tile['size']/2], zorder=99, alpha=tile_alpha)
			num_placed += 1
		print(f"[INFO] Num placed: {num_placed}")

	def finish_mosaic(self, export):
		print("[Core] Displaying/exporting figure...")
		self.ax.autoscale(enable=True, tight=None)
		if export:
			save_path = join(self.tile_root, f'Mosaic-{self.num_tiles_x}.png')
			plt.savefig(save_path, bbox_inches='tight')
			print(f"Saved figure to {save_path}")
			plt.close()
		else:
			plt.show()

def get_args():
	parser = argparse.ArgumentParser(description = 'Creates a t-SNE histology tile mosaic using a saved t-SNE bookmark generated with Tensorboard.')
	parser.add_argument('-b', '--bookmark', help='Path to saved Tensorboard *.txt bookmark file.')
	parser.add_argument('-m', '--meta', help='Path to Tensorboard metadata.tsv file.')
	parser.add_argument('-t', '--tile', help='Path to root directory containing image tiles, separated in directories according to case name.')
	parser.add_argument('-s', '--slide', help='(Optional) Path to whole slide images (SVS or JPG format)')
	parser.add_argument('-d', '--detail', type=int, default=70, help='(Optional) Reflects how many tiles should be generated on the mosaic; default is 70.')
	parser.add_argument('-f', '--figsize', type=int, default=200, help='(Optional) Reflects how large the output figure size should be ; default is 200.')
	parser.add_argument('-l', '--leniency', type=int, default=1.5, help='(Optional) How lenient the algorithm should be when placing tiles ; default is 1.5.')
	parser.add_argument('--um', type=float, help='(Necessary if plotting SVS) Size of extracted image tiles in microns.')
	parser.add_argument('--export', action="store_true", help='Save mosaic to png file.')
	return parser.parse_args()

if __name__ == '__main__':
	args = get_args()
	mosaic = Mosaic(args)

	if args.slide and not args.um:
		raise ValueError("Size of extracted tiles (in microns) must be supplied when viewing SVS files; please use the --um flag.")
	if args.slide and isfile(args.slide) and not args.export:
		# Load a single SVS
		slide_list = [args.slide.split('/')[-1]]
		slide_dir = "/".join(args.slide.split('/')[:-1])
		mosaic.load_slides(slide_list, slide_dir)
	elif args.slide and not args.export:
		# First, load images in the directory, not assigning any category
		slide_list = [i for i in os.listdir(args.slide) if (isfile(join(args.slide, i)) and (i[-3:].lower() in ("svs", "jpg")))]	
		mosaic.load_slides(slide_list, args.slide)
		# Next, load images in subdirectories, assigning category by subdirectory name
		dir_list = [d for d in os.listdir(args.slide) if not isfile(join(args.slide, d))]
		for directory in dir_list:
			# Ignore images if in the thumbnails or QuPath project directory
			if directory in ["thumbs", "ignore", "QuPath_Project"]: continue
			slide_list = [i for i in os.listdir(join(args.slide, directory)) if (isfile(join(args.slide, directory, i)) and (i[-3:].lower() in ("svs", "jpg")))]	
			mosaic.load_slides(slide_list, join(args.slide, directory), category=directory)

	mosaic.generate()