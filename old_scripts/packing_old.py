# Histcon Sampler & Packing Algorithm
# Copyright (C) James Dolezal - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, October 2017
# ==========================================================================

"""Subdivides an annotated image (from labelme) into smaller subsections of 
	a given size, for use in the Histon convolutional network. """

import json
import os
import sys
import time
import argparse
import warnings

import shapely.geometry as sg
import matplotlib.pyplot as plt
import numpy as np
import progress_bar as progress
from multiprocessing import Pool
from matplotlib import path
from contextlib import closing
from PIL import Image

CASE_DIR = "cases/"
RESULTS_DIR = "packing_results/"

parser = argparse.ArgumentParser()

parser.add_argument('--case', type=str, default='LE107_1',
	help="Name of the case to analyze.")

parser.add_argument('--size', type=int, default=128,
	help='Size of image patches to create.')

FLAGS = parser.parse_args()


class Loc():
	def __init__(self, area, x, y, gca, exclusions):
		self.area=area
		self.x=x
		self.y=y
		self.gca=gca
		self.exclusions=exclusions

def square_count(coord):
	area = coord.area
	offset_x = coord.x
	offset_y = coord.y
	gca = coord.gca
	exclusions = coord.exclusions
	graph = False
	image = None
	label = None
	num = None
	color = 'g'
	
	''' Fill annotation area with squares '''
	x_min = min(p[0] for p in area) + offset_x
	x_max = max(p[0] for p in area) + offset_x
	x_range = x_max - x_min
	y_min = min(p[1] for p in area) + offset_y
	y_max = max(p[1] for p in area) + offset_y
	y_range = y_max - y_min

	square_count = 0

	areaPoly = sg.Polygon(area)

	# Place test squares
	for j in range(int(y_range/FLAGS.size)):
		for i in range(int(x_range/FLAGS.size)):
			x = x_min + i*FLAGS.size
			y = y_min + j*FLAGS.size
			sTestSquare = sg.Polygon([(x, y), (x, y+FLAGS.size), (x+FLAGS.size, y+FLAGS.size), (x+FLAGS.size, y)])
			if graph: mTestSquare = plt.Rectangle((x, y), FLAGS.size, FLAGS.size, fc=color)

			# Check to see if square falls within the current annotation polygon
			if areaPoly.contains(sTestSquare): 
				if exclusions:
					in_area = False
					for ex in exclusions:
						if sg.Polygon(ex['points']).intersects(sTestSquare):
							in_area = True
							break
					if in_area: continue
				if graph: gca.add_patch(mTestSquare)
				if image:
					region = image.crop((x, y, x+FLAGS.size, y+FLAGS.size))
					if not os.path.exists(RESULTS_DIR + label): os.makedirs(RESULTS_DIR + label)
					region.save(RESULTS_DIR + "%s/%s_%s_%s_0.jpg" % (label, FLAGS.case+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.FLIP_LEFT_RIGHT).save(RESULTS_DIR + "%s/%s_%s_%s_0F.jpg" % (label, FLAGS.case+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.ROTATE_90).save(RESULTS_DIR + "%s/%s_%s_%s_90.jpg" % (label, FLAGS.case+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT).save(RESULTS_DIR + "%s/%s_%s_%s_90F.jpg" % (label, FLAGS.case+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.ROTATE_180).save(RESULTS_DIR + "%s/%s_%s_%s_180.jpg" % (label, FLAGS.case+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.ROTATE_180).transpose(Image.FLIP_LEFT_RIGHT).save(RESULTS_DIR + "%s/%s_%s_%s_180F.jpg" % (label, FLAGS.case+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.ROTATE_270).save(RESULTS_DIR + "%s/%s_%s_%s_270.jpg" % (label, FLAGS.case+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT).save(RESULTS_DIR + "%s/%s_%s_%s_270F.jpg" % (label, FLAGS.case+"-"+label, num, square_count), "jpeg")
				square_count += 1
	return square_count



class SquareIterator:
	'''	Iterate through all possible square placement starting points and
		find the placement which maximizes number of squares placed.
	'''
	max_squares = 0
	max_coord = []
	complete = 0

	def __init__(self, gca, area, _id, exclusions=None):
		self._id = _id
		#self.gca = gca
		#self.area = area
		#self.exclusions = exclusions

		# Build matrix of coordinates
		self.matrix = []
		for i in range(FLAGS.size):
			for j in range(FLAGS.size):

				self.matrix.append(Loc(area, i, j, gca, exclusions))

		print("Analyzing annotation %s..." % _id)

		start_time = time.time()

		# Map analyses onto matrix of coordinates

		#with closing( Pool(processes=4) ) as pool:
			#counts = pool.map(self, self.matrix)
			#counts = pool.map(square_count, self.matrix)

		#counts = map(square_count, self.matrix)

		pool = Pool(8)
		#counts = pool.map(self, self.matrix)
		counts = pool.map(square_count, self.matrix)
		pool.close()
		pool.join()

		#counts = map(self, self.matrix)
		#counts = map(square_count, self.matrix)

		end_time = time.time()

		print("...complete (%.2f sec)" % (end_time-start_time))

		# Find the coordinates which result in the highest square placement
		self.max_squares = max(counts)
		for i in range(len(counts)):
			if counts[i] == self.max_squares:
				self.max_coord = [self.matrix[i].x, self.matrix[i].y]
				break

		#for j in range(FLAGS.size):
		#	for i in range(FLAGS.size):
		#		if counts[j][i] == self.max_squares:
		#			self.max_coord = [i, j]
		#			break

	def __call__(self, row):
		return self.analyze_row(row)

	def analyze_row(self, row):
		return map(self.square_count, row)

	def square_count(self, place):
		return place_squares(self.area, place[0], place[1], self.gca, self.exclusions)

def place_squares(area, offset_x = 0, offset_y = 0, gca = None, graph = False, image = None, label = None, num = None, exclusions = None, color='g'):
	''' Fill annotation area with squares '''
	x_min = min(p[0] for p in area) + offset_x
	x_max = max(p[0] for p in area) + offset_x
	x_range = x_max - x_min
	y_min = min(p[1] for p in area) + offset_y
	y_max = max(p[1] for p in area) + offset_y
	y_range = y_max - y_min

	square_count = 0

	areaPoly = sg.Polygon(area)

	# Place test squares
	for j in range(int(y_range)/FLAGS.size):
		for i in range(int(x_range)/FLAGS.size):
			x = x_min + i*FLAGS.size
			y = y_min + j*FLAGS.size
			sTestSquare = sg.Polygon([(x, y), (x, y+FLAGS.size), (x+FLAGS.size, y+FLAGS.size), (x+FLAGS.size, y)])
			if graph: mTestSquare = plt.Rectangle((x, y), FLAGS.size, FLAGS.size, fc=color)

			# Check to see if square falls within the current annotation polygon
			if areaPoly.contains(sTestSquare): 
				if exclusions:
					in_area = False
					for ex in exclusions:
						if sg.Polygon(ex['points']).intersects(sTestSquare):
							in_area = True
							break
					if in_area: continue
				if graph: gca.add_patch(mTestSquare)
				if image:
					region = image.crop((x, y, x+FLAGS.size, y+FLAGS.size))
					if not os.path.exists(RESULTS_DIR + label): os.makedirs(RESULTS_DIR + label)
					region.save(RESULTS_DIR + "%s/%s_%s_%s_0.jpg" % (label, FLAGS.case+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.FLIP_LEFT_RIGHT).save(RESULTS_DIR + "%s/%s_%s_%s_0F.jpg" % (label, FLAGS.case+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.ROTATE_90).save(RESULTS_DIR + "%s/%s_%s_%s_90.jpg" % (label, FLAGS.case+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT).save(RESULTS_DIR + "%s/%s_%s_%s_90F.jpg" % (label, FLAGS.case+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.ROTATE_180).save(RESULTS_DIR + "%s/%s_%s_%s_180.jpg" % (label, FLAGS.case+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.ROTATE_180).transpose(Image.FLIP_LEFT_RIGHT).save(RESULTS_DIR + "%s/%s_%s_%s_180F.jpg" % (label, FLAGS.case+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.ROTATE_270).save(RESULTS_DIR + "%s/%s_%s_%s_270.jpg" % (label, FLAGS.case+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT).save(RESULTS_DIR + "%s/%s_%s_%s_270F.jpg" % (label, FLAGS.case+"-"+label, num, square_count), "jpeg")
				square_count += 1
	return square_count

def subdivider(area, label, offset, im):
	''' Based on a pre-calculated optimal starting point, subdivide
	    area into squares'''

	x_min = min(p[0] for p in area) + offset[0]
	x_max = max(p[0] for p in area) + offset[1]
	x_range = x_max - x_min
	y_min = min(p[1] for p in area) + offset[0]
	y_max = max(p[1] for p in area) + offset[1]
	y_range = y_max - y_min

	for j in range(int(y_range)/FLAGS.size):
		for i in range(int(x_range)/FLAGS.size):
			x = x_

def main():
	with open (CASE_DIR + "%s.json" % FLAGS.case) as case:
		with Image.open(CASE_DIR + "%s.jpg" % FLAGS.case) as im:

			print('Opened %s image "%s.jpg" %s %s' % (im.format, FLAGS.case, im.size, im.mode))

			warnings.simplefilter('ignore', Image.DecompressionBombWarning)

			d = json.load(case)
			case_shape = []
			other_shapes = []

			# Initiate grid for drawing
			plt.axes()
			gca = plt.gca()
			gca.invert_yaxis()
			gca.tick_params(axis="x", top=True, labeltop=True, bottom = False, labelbottom = False)

			# Iterate through each annotation
			ann_num = 1
			for s in d['shapes']:
				area = s['points']
				label = s['label']

				if label == "case":
					mPolygon = plt.Polygon(area, facecolor="none", edgecolor="r")
				else: 
					mPolygon = plt.Polygon(area)

				# Add annotation polygon to the grid
				gca.add_patch(mPolygon)		

				if label == "case": 
					# Subdivide normal background
					case_shape = s
					
				else:
					other_shapes.append(s)
					iterator = SquareIterator(gca, area, ann_num)
					squares, offset = iterator.max_squares, iterator.max_coord

					print("Number of sections: %i" % squares)
					if squares > 0: print("Starting point: (%s, %s)\n" % (offset[0], offset[1]))
					else: print("\n")

					print('Subdividing annotation #%s...' % ann_num)
					place_squares(area, offset[0], offset[1], gca, graph=True, image=im, label=label, num=ann_num)

					ann_num += 1

			if case_shape == []:
				print("Error: no case shape provided.")
				sys.exit()
			else:
				print('Subdividing normal background...')
				place_squares(case_shape['points'], 0, 0, gca, graph=True, image=im, label='normal', num=0, exclusions = other_shapes, color='r')

			print('Finished.')

			# Show grid
			plt.axis('scaled')
			plt.show()

if __name__=='__main__':
	main()