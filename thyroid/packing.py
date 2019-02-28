# Copyright (C) James Dolezal - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, October 2017, Updated 12/16/18
# ==========================================================================
"""For a given whole-slide image (WSI), subdivides the image into smaller tiles 
  using an annotated *.json file generated with LabelMe.

James M Dolezal, 2019
"""

import json
import os, sys
import warnings
import shapely.geometry as sg
import matplotlib.pyplot as plt
import numpy as np
import progress_bar
from matplotlib import path
from PIL import Image
from os.path import isfile, join


Image.MAX_IMAGE_PIXELS = None
TILE_SIZE = 2048
ANNOTATION_COMPRESSION_FACTOR = 10
DIRECTORY = "/media/shawarma/Backup/Other files/Thyroid/Annotations/All/to_convert"

class Packer:
	"""Module which loads a LabelMe json annotation file and subdivides annotations
	into smaller tiles in an optimal fashion.

	Example use:
		p = Packer(json_data, case_id, 1, 256, "/home/full_image.jpg")
		p.subdivide_all_annotations()
		p.export_all_tiles("/home/export_directory")
		p.show_graph()
	"""

	def __init__(self, case_data, case_id, factor, size, full_image):
		"""Initializes module by opening *.json and starting MatPlotLib graph
		Args:
			case_data:	LabelMe json data
			case_id:	Unique number identifying the case
			factor:		Factor by which annotation file coordinates are compressed (e.g. 10)
			size:		Size of tiles to generate
			full_image:	Location of full image from which to extract tiles
		"""
		self.data = case_data['shapes']
		self.case_id = case_id
		self.factor = factor
		self.size = size
		self.background_shape = None # Shape which designates bounding box for the background
		self.other_shapes = [] # Collection of all non-background shapes
		self.full_image = full_image

		plt.axes(label="Packing Result")
		self.gca = plt.gca()
		self.gca.invert_yaxis()
		self.gca.tick_params(axis="x", top=True, labeltop=True, bottom = False, 
								labelbottom = False)

	def optimize_annotations(self, background_label, ignore_background):
		"""For all annotated shapes, find the starting tile placement which maximizes
		the number of tiles placed.

		Args:
			background_label:		Label in the json file which designates the bounding background shape
			ignore_background:		If true, will not extract any image tiles from the background shape
		"""
		for index, shape in enumerate(self.data):
			area = shape['points']
			area_full = np.multiply(area, self.factor)

			label = shape['label']

			if label == background_label and not ignore_background:
				mPolygon = plt.Polygon(area_full, facecolor="none", edgecolor="r")
				self.gca.add_patch(mPolygon)
				self.background_shape = shape
			elif label != background_label:
				mPolygon = plt.Polygon(area_full)
				self.other_shapes.append(shape)
				squares, coordinates = self.tile_iterator(area, index, int(self.size / self.factor), [4,4])
				print("Number of tiles: %i" % squares)
				if squares > 0:
					print ("Optimal starting point: (%s, %s)" %
							(coordinates[0] * self.factor, coordinates[1] * self.factor))
				self.data[index]['square_offset'] = coordinates
				self.data[index]['max_squares'] = squares

	def export_tiles(self, export_dir, background_label = "background", ignore_background = True, origin_optimization = True):
		"""For all annotated shapes, exports extracted tiles from a full-size image.

		Args:
			export_dir:				Directory in which to export tiles
			background_label:		Label in the json file which designates the bounding background shape
			ignore_background:		If true, will not extract any image tiles from the background shape
			origin_optimization:	Increases number of tiles extracted by ~10%, but CPU-intensive
		"""

		if origin_optimization:
			self.optimize_annotations(background_label, ignore_background)

		im = Image.open(self.full_image)

		print('Opened %s image from %s | %s %s' % (im.format, self.full_image, 
													im.size, im.mode))

		for index, shape in enumerate(self.data):
			# Increase annotation coordinates by factor
			area_reduced = shape['points']
			area = np.multiply(area_reduced, self.factor)

			if origin_optimization:
				try:
					offset = shape['square_offset']
					max_squares = shape['max_squares']
				except KeyError:
					print("No annotation found for shape %s, skipping" % index)
					continue
			else:
				offset = [0,0]

			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				self.place_tiles(area, offset, self.size, export_dir, im, index, False)

		print("Export complete.")

	def tile_iterator(self, area, _id, tile_size, stride = [1, 1], exclusions = None):
		"""Iterate through possible tile placement starting points and 
		find the placement which maximizes number of tiles placed."""

		max_squares = 0
		max_coord = []

		max_it = tile_size*tile_size

		for j in range(0, tile_size, stride[1]):
			for i in range(0, tile_size, stride[0]):
				sys.stdout.write("\rOptimizing annotation #%s: " % _id)
				progress_bar.bar((j*tile_size)+i, max_it, newline=False)
				count = self.place_tiles(area, [i, j], tile_size, exclusions = exclusions)
				if count >= max_squares: 
					max_squares = count
					max_coord = [i, j]

		sys.stdout.write("\n")

		return max_squares, max_coord

	def place_tiles(self, area, offset, tile_size, export_dir = None, image = None, _id = None, graph = False, exclusions = None, color='g'):
		"""Fills a given area with squares and returns the total number of squares/tiles placed.
		If an image is provided, will export extracted tiles.

		Args:
			area:			Bounding box shape, extracted from LabelMe json, in which to place tiles
			offset:			Coordinates for which to offset tile placement
			tile_size:		Pixel size of tiles to be placed
			export_dir:		Directory in which to save  tiles, if extraction from whole image is performed.
			image: 			Full-size image from which to extract tiles.
			_id: 			ID of the annotation being tiled
			max_squares:	Maximum number of tiles that can be placed, if already determined.
			graph:			Whether or not to display tiling results
			exclusions:		Array of shapes to exclude from tiling
			color:			Color code for displaying tile placement, e.g. 'r', 'g'

		Returns:
			Total number of tiles placed.
		"""

		x_min = min(p[0] for p in area) + offset[0]
		x_max = max(p[0] for p in area) + offset[0]
		x_range = x_max - x_min

		y_min = min(p[1] for p in area) + offset[1]
		y_max = max(p[1] for p in area) + offset[1]
		y_range = y_max - y_min

		square_count = 0

		areaPoly = sg.Polygon(area)

		for j in range(int(y_range/tile_size)):
			for i in range(int(x_range/tile_size)):
				x = x_min + i*tile_size
				y = y_min + j*tile_size
				sTestSquare = sg.Polygon([(x, y), (x, y+tile_size), 
										  (x+tile_size, y+tile_size),
										  (x+tile_size, y)])

				if graph: mTestSquare = plt.Rectangle((x, y), tile_size, tile_size, 
														fc=color)

				# Check to see if square falls within the current annotation polygon
				if areaPoly.contains(sTestSquare): 
					if exclusions:
						in_area = False
						for ex in exclusions:
							if sg.Polygon(ex['points']).intersects(sTestSquare):
								in_area = True
								break
						if in_area: continue
					if graph: self.gca.add_patch(mTestSquare)
					if image:
						region = image.crop((x, y, x+tile_size, y+tile_size))
						region.save(join(export_dir, "%s_%s_%s_0.jpg" % (self.case_id, _id, square_count)), "jpeg")
						region.transpose(Image.FLIP_LEFT_RIGHT).save(join(export_dir, "%s_%s_%s_0F.jpg" % (self.case_id, _id, square_count)), "jpeg")
						region.transpose(Image.ROTATE_90).save(join(export_dir, "%s_%s_%s_90.jpg" % (self.case_id, _id, square_count)), "jpeg")
						region.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT).save(join(export_dir, "%s_%s_%s_90F.jpg" % (self.case_id, _id, square_count)), "jpeg")
						region.transpose(Image.ROTATE_180).save(join(export_dir, "%s_%s_%s_180.jpg" % (self.case_id, _id, square_count)), "jpeg")
						region.transpose(Image.ROTATE_180).transpose(Image.FLIP_LEFT_RIGHT).save(join(export_dir, "%s_%s_%s_180F.jpg" % (self.case_id, _id, square_count)), "jpeg")
						region.transpose(Image.ROTATE_270).save(join(export_dir, "%s_%s_%s_270.jpg" % (self.case_id, _id, square_count)), "jpeg")
						region.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT).save(join(export_dir, "%s_%s_%s_270F.jpg" % (self.case_id, _id, square_count)), "jpeg")
					square_count += 1
		return square_count

	def show_graph(self):
		plt.axis('scaled')
		#plt.show()

if __name__==('__main__'):
	cases = [f[:-5] for f in os.listdir(DIRECTORY) if (isfile(join(DIRECTORY, f)) 
										and f[-4:] == 'json')]

	for case_id in cases:
		with open (join(DIRECTORY, "%s.json" % case_id)) as case:
			full_image = join(DIRECTORY, "%s.jpg" % case_id)
			export_dir = join(DIRECTORY, str(TILE_SIZE), case_id)
			if not os.path.exists(export_dir): os.makedirs(export_dir)

			case_data = json.load(case)

			packer = Packer(case_data, case_id, ANNOTATION_COMPRESSION_FACTOR, TILE_SIZE, full_image)
			packer.export_tiles(export_dir, origin_optimization = True)

			# Show results
			packer.show_graph()