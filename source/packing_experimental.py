# Copyright (C) James Dolezal - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, October 2017, Updated 12/16/18
# ==========================================================================
"""For a given whole-slide image (WSI), subdivides the image into smaller tiles 
  using an annotated *.json file generated with LabelMe.

James M Dolezal
"""

import json
import os, sys
import warnings
import shapely.geometry as sg
import matplotlib.pyplot as plt
import numpy as np
import progress
from matplotlib import path
from PIL import Image
from os.path import isfile, join

# TODO: Progress bar for subdividing/saving stage
# TODO: Need to filter out normal image boxes that are all white

SIZE = 256
FACTOR = 1 # Annotation compression factor, to multiply annotation coordinates
ROOT_FOLDER = "/Users/James/thyroid/images"
ANNOTATED_FOLDER = join(ROOT_FOLDER, "Annotated") # Folder with annotated .json files
SAVE_FOLDER = join(ROOT_FOLDER, str(SIZE))
CASES = [f[:-5] for f in os.listdir(AN_F) if (isfile(join(AN_F, f)) 
										and f[-4:] == 'json')]
class Packer:
	"""Module used to contain automatic packing functions.

	Args:
		size: Full-size area will be subdivided into tiles of this size
		factor: Annotated *.json file is compressed at this factor.  
				Subdivion testing is performed *
	"""

	def __init__(self, size, factor, case_data):
		"""Initializes module by opening *.json and starting MatPlotLib graph"""
		self.data = case_data
		self.factor = factor
		self.size = size

		# Initialize grid for drawing
		plt.axes(label="Packing Result")
		self.gca = plt.gca()
		self.gca.invert_yaxis()
		self.gca.tick_params(axis="x", top=True, labeltop=True, bottom = False, 
								labelbottom = False)

	def subdivide_all_annotations(self, exclude_annotations = None, display=True):
		"""Finds optimal tile subdivision solution for all annotations, generating
		generates tiles from whole image, and graphically displays results."""

		for index, shape in enumerate(self.data['shapes']):
			if shape['label'] not in exclude_annotations: 
				square_count, coordinates = self.subdivide(shape)

		if display:
			plt.axis('scaled')
			plt.show()

	def subdivide(self, shape):
		"""Finds optimal tile subdivision solution, generates tiles from whole
		image, and graphically displays results."""
		self.find_max_tiles(shape)
		self.tile_whole_image(shape)

		# Subdivide normal background
		# print('Subdividing normal background...')
		# place_squares(case_shape['points'], 0, 0, gca, graph=True, image=im, label='normal', num=0, exclusions = other_shapes, color='r')


	def find_max_tiles(self, shape):
		"""Iterates through all annotations and finds starting points
		which produce maximum number of subdivided tiles."""

		label = shape['label']
		area = shape['points']
		area_full = np.multiply(area, self.factor)
		area_small = np.divide(area, self.factor)

		# If the current shape is the case (outlining) shape, display it as a single outline
		if label == "case":
			mPolygon = plt.Polygon(area_full, facecolor="none", edgecolor="r")
		else: 
			mPolygon = plt.Polygon(area_full)
		
		self.gca.add_patch(mPolygon)

		sys.stdout.write("\rAnalyzing annotation #%s: " % ann_num)
		squares, coordinates = self.square_iterator(area_small)

		s['square_offset'] = coordinates
		s['number'] = ann_num
		s['max_squares'] = squares

	def tile_whole_image(self):
		"""Based on a given tile size, area, and starting position, subdivides 
		a whole image into smaller tiles"""

		# Open the image
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			im = Image.open((join(ANNOTATED_FOLDER, "%s.jpg" % CASE_ID)))
		print('Opened %s image "%s.jpg" %s %s' % (im.format, CASE_ID, 
													im.size, im.mode))

		case_shape = None
		other_shapes = []
		
		# Subdivide annotations into squares
		for s in d['shapes']:
			label = s['label']

			if label == "case":
				case_shape = s
				continue
			else:
				other_shapes.append(s)

			area_reduced = s['points']
			area = np.multiply(area_reduced, FACTOR)
			offset = s['square_offset']
			ann_num = s['number']
			max_squares = s['max_squares']

			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				place_squares(area, offset[0], offset[1], gca, graph=True, 
								image=im, label=CASE_ID, num=ann_num, 
								max_squares = max_squares)

			sys.stdout.write("\n")

	def place_squares(self, offset_x, offset_y):
		pass

	def square_iterator(self, area, exclusions = None):
		"""Iterate through possible square placement starting points and 
	   find the placement which maximizes number of squares placed."""

		tile_size = int(self.size/(self.factor ** 2))
		max_it = tile_size ** 2
		max_squares = 0
		max_coord = []

		for j in range(tile_size):
			for i in range(tile_size):
				progress.bar((j*tile_size)+i, max_it, newline=False)
				count = self.place_squares(area, i, j, self.gca, exclusions = exclusions, 
										tile_size = tile_size)
				if count >= max_squares: 
					max_squares = count
					max_coord = [i, j]
		sys.stdout.write("\n")

		print("Number of sections: %i" % max_squares)
					if max_squares > 0: print("Starting point: (%s, %s)\n" % 
											(max_coord[0]*self.factor, max_coord[1]*self.factor))
					else: print("\n")

		# Show final result
		# place_squares(area, max_coord[0], max_coord[1], gca, True)

		return max_squares, max_coord

	def subdivider():
		pass


def place_squares(area, offset_x = 0, offset_y = 0, gca = None, graph = False, 
					image = None, label = None, num = None, exclusions = None, 
					tile_size = SIZE, color='g', max_squares = None):
	"""Fill annotation area with squares.

	Args:
		area:	*
		offset_x: *
		offset_y: *
		gca: *
		graph: *
		image: *
		label: *
		num: *
		exclusions: *
		tile_size: *
		color: *
		max_squares: *

	Returns:
		Number of total squares (or "tiles") placed
	"""
	x_min = min(p[0] for p in area) + offset_x
	x_max = max(p[0] for p in area) + offset_x
	x_range = x_max - x_min
	y_min = min(p[1] for p in area) + offset_y
	y_max = max(p[1] for p in area) + offset_y
	y_range = y_max - y_min

	square_count = 0

	areaPoly = sg.Polygon(area)

	# Place test squares
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
				if graph: gca.add_patch(mTestSquare) #mTestSquare.set_color('g')
				if image:
					if max_squares:
						sys.stdout.write("\rSubdividing annotation #%s: " % ann_num)
						progress.bar(square_count, max_squares)
					region = image.crop((x, y, x+tile_size, y+tile_size))
					folder = join(SAVE_FOLDER, label)
					if not os.path.exists(folder): os.makedirs(folder)

					region.save(join(folder, "%s_%s_%s_0.jpg" % (label, num, square_count)), "jpeg")
					region.transpose(Image.FLIP_LEFT_RIGHT).save(join(folder, "%s_%s_%s_0F.jpg" % (label, num, square_count)), "jpeg")
					region.transpose(Image.ROTATE_90).save(join(folder, "%s_%s_%s_90.jpg" % (label, num, square_count)), "jpeg")
					region.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT).save(join(folder, "%s_%s_%s_90F.jpg" % (label, num, square_count)), "jpeg")
					region.transpose(Image.ROTATE_180).save(join(folder, "%s_%s_%s_180.jpg" % (label, num, square_count)), "jpeg")
					region.transpose(Image.ROTATE_180).transpose(Image.FLIP_LEFT_RIGHT).save(join(folder, "%s_%s_%s_180F.jpg" % (label, num, square_count)), "jpeg")
					region.transpose(Image.ROTATE_270).save(join(folder, "%s_%s_%s_270.jpg" % (label, num, square_count)), "jpeg")
					region.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT).save(join(folder, "%s_%s_%s_270F.jpg" % (label, num, square_count)), "jpeg")
				square_count += 1
	return square_count

if __name__==("__main__"):
	for CASE_ID in CASES:
		with open(join(ANNOTATED_FOLDER, '%s.json' % CASE_ID)) as case
			print("Working on case %s" % CASE_ID)
			case_data = json.load(case)
			packer = Packer(SIZE, FACTOR, CASE_ID, case_data)
			packer.subdivide_all_annotations(False)