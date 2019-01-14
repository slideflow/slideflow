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

SIZE = 256
FACTOR = 1
DIRECTORY = "/Users/James/thyroid/images/Annotated"

class Packer:
	def __init__(self, case_data, factor, size, full_image):
		"""Initializes module by opening *.json and starting MatPlotLib graph"""
		self.data = case_data
		self.factor = factor
		self.size = size
		self.background_shapes = None # Shape which designates bounding box for the background
		self.other_shapes = [] # Collection of all non-background shapes
		self.full_image = full_image

		plt.axes(label="Packing Result")
		self.gca = plt.gca()
		self.gca.invert_yaxis()
		self.gca.tick_params(axis="x", top=True, labeltop=True, bottom = False, 
								labelbottom = False)

	def subdivide_all_annotations(self, background_label = "case", ignore_background = True):
		for index, shape in enumerate(self.data):
			area = shape['points']
			area_full = np.multiply(area, self.factor)
			area_small = np.divide(area, self.factor)
			tile_screening_size = int(SIZE/(FACTOR**2))

			label = shape['label']

			if label == background_label and not ignore_background:
				mPolygon = plt.Polygon(area_full, facecolor="none", edgecolor="r")
				self.gca.add_patch(mPolygon)
				self.background = shape
			elif label != background_label:
				mPolygon = plt.Polygon(area_full)
				self.shapes.append(shape)
				squares, coordinates = tile_iterator(area_small, index, tile_screening_size)
				print("Number of sections: %i" % squares)
				if squares > 0:
					print ("Starting point: (%s, %s)" %
							(coordinates[0] * self.factor, coordinates[1] * self.factor))
				shape['square_offset'] = coordinates
				shape['max_squares'] = squares

	def export_all_tiles(self):
		"""For all annotated shapes, exports extracted tiles."""

		# Open full-sized image
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			im = Image.open(self.full_image)

		print('Opened %s image from %s | %s %s' % (im.format, self.full_image, 
													im.size, im.mode))

		for index, shape in enumerate(self.data):
			# Increase annotation coordinates by factor
			area_reduced = shape['points']
			area = np.multiply(area_reduced, self.factor)

			offset = shape['square_offset']
			max_squares = shape['max_squares']

			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				place_tiles(area, offset[0], offset[1], self.gca, graph=True, 
								image=im, num=index, 
								max_squares = max_squares)

			sys.stdout.write("\n")

	def tile_iterator(self, area, _id, tile_size, exclusions = None):
		"""Iterate through possible tile placement starting points and 
		find the placement which maximizes number of tiles placed."""

		max_squares = 0
		max_coord = []

		max_it = tile_size*tile_size

		for j in range(tile_size):
			for i in range(tile_size):
				sys.stdout.write("\rAnalyzing annotation #%s: " % _id)
				progress.bar((j*tile_size)+i, max_it, newline=False)
				count = place_tiles(area, i, j, self.gca, exclusions = exclusions, 
										tile_size = tile_size)
				if count >= max_squares: 
					max_squares = count
					max_coord = [i, j]

		sys.stdout.write("\n")

		return max_squares, max_coord

	def place_tiles(self, area, offset, tile_size, image = None, color='g', max_squares = None):
		"""Fills a given area with squares and returns the total number of squares/tiles placed.
		If an image is provided, will export extracted tiles.

		Args:
			image: Full-size image from which to extract tiles.
			max_squares: Maximum number of tiles that can be placed, if already determined.

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


if __name__==('__main__'):
	cases = [f[:-5] for f in os.listdir(DIRECTORY) if (isfile(join(DIRECTORY, f)) 
										and f[-4:] == 'json')]

	for case_id in cases:
		with open (join(DIRECTORY, "%s.json" % case_id)) as case:
			case_data = json.load(case)
			full_image = join(DIRECTORY, "%s.jpg" % case_id)
			packer = Packer(case_data, FACTOR, SIZE, full_image)
