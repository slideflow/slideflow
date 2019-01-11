# Histcon sampler & Packing Algorithm
# Copyright (C) James Dolezal - All Rights Reserved
# Unauthorized copying of this file, via any medium, is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, October 2017
# ==========================================================================

import os
import sys
import json
import time
import random
import warnings
import argparse
import progress_bar

from PIL import Image
from multiprocessing import Pool
import shapely.geometry as sg
import matplotlib.pyplot as plt
import numpy as np

CASE_DIR = "/home/falafel/histcon/cases/"
RESULTS_DIR = "/home/falafel/histcon/packing_results/"
CASE = "LE107_1"
SIZE = 64
VERBOSE = True

def save_squares(area, index, label, num, image, gca=None, graph=False, exclusions=None, save = True):
	x_off = index[0]
	y_off = index[1]

	x_min = min(p[0] for p in area) + x_off
	x_max = max(p[0] for p in area) + x_off
	x_range = x_max - x_min

	y_min = min(p[1] for p in area) + y_off
	y_max = max(p[1] for p in area) + y_off
	y_range = y_max - y_min

	square_count = 0
	filter_count = 0
	areaPoly = sg.Polygon(area)

	for j in range(int(y_range/SIZE)):
		for i in range(int(x_range/SIZE)):
			x = x_min + i*SIZE
			y = y_min + j*SIZE
			sTestSquare = sg.Polygon([(x, y), (x, y+SIZE), (x+SIZE, y+SIZE), (x+SIZE, y)])

			if areaPoly.contains(sTestSquare):
				if exclusions:
					in_area = False
					for ex in exclusions:
						if sg.Polygon(ex['points']).intersects(sTestSquare):
							in_area = True
							break
					if in_area: continue

				region = None

				if label == "normal":
					# Filter the image first to remove background
					region = image.crop((x, y, x+SIZE, y+SIZE))
					ra = np.array(region)
					if np.mean(ra) > 230 and np.std(ra) < 5:
						gca.add_patch( plt.Rectangle((x, y), SIZE, SIZE, fc='y') )
						filter_count += 1
						continue
				elif graph:
					gca.add_patch( plt.Rectangle((x, y), SIZE, SIZE, fc='g') )

				if save:
					if not region: region = image.crop((x, y, x+SIZE, y+SIZE))
					if not os.path.exists(RESULTS_DIR + label): os.makedirs(RESULTS_DIR + label)
					region.save(RESULTS_DIR + "%s/%s_%s_%s_0.jpg" % (label, CASE+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.FLIP_LEFT_RIGHT).save(RESULTS_DIR + "%s/%s_%s_%s_0F.jpg" % (label, CASE+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.ROTATE_90).save(RESULTS_DIR + "%s/%s_%s_%s_90.jpg" % (label, CASE+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT).save(RESULTS_DIR + "%s/%s_%s_%s_90F.jpg" % (label, CASE+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.ROTATE_180).save(RESULTS_DIR + "%s/%s_%s_%s_180.jpg" % (label, CASE+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.ROTATE_180).transpose(Image.FLIP_LEFT_RIGHT).save(RESULTS_DIR + "%s/%s_%s_%s_180F.jpg" % (label, CASE+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.ROTATE_270).save(RESULTS_DIR + "%s/%s_%s_%s_270.jpg" % (label, CASE+"-"+label, num, square_count), "jpeg")
					region.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT).save(RESULTS_DIR + "%s/%s_%s_%s_270F.jpg" % (label, CASE+"-"+label, num, square_count), "jpeg")
				square_count += 1
	if VERBOSE and label=="normal": print("Filtered %s normal squares." % filter_count)

def calc_placement(datum):
	x_off = datum[0]
	y_off = datum[1]
	area = datum[2]

	x_min = min(p[0] for p in area) + x_off
	x_max = max(p[0] for p in area) + x_off
	x_range = x_max - x_min

	y_min = min(p[1] for p in area) + y_off
	y_max = max(p[1] for p in area) + y_off
	y_range = y_max - y_min

	square_count = 0

	areaPoly = sg.Polygon(area)

	for j in range(int(y_range/SIZE)):
		for i in range(int(x_range/SIZE)):
			x = x_min + i*SIZE
			y = y_min + j*SIZE
			sTestSquare = sg.Polygon([(x, y), (x, y+SIZE), (x+SIZE, y+SIZE), (x+SIZE, y)])

			if areaPoly.contains(sTestSquare):
				square_count += 1

	return square_count

def main(export, plot):
	warnings.simplefilter('ignore', Image.DecompressionBombWarning)

	edit_case_data = False

	with open(os.path.join(CASE_DIR, "%s.json" % CASE)) as case:
		with Image.open(CASE_DIR + "%s.jpg" % CASE) as im:
			if VERBOSE: print('Opened %s image "%s.jpg" %s %s' % (im.format, CASE, im.size, im.mode))

			case_data = json.load(case)
			inner_shapes = []
			case_shape = None
			num_shapes = len(case_data['shapes'])

			# Initiate grid for drawing
			plt.axes()
			gca = plt.gca()
			gca.invert_yaxis()
			gca.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

			start_time = time.time()

			# Enumerate through shapes, looking for case shape
			for index, shape in enumerate(case_data['shapes']):
				area = shape['points']
				label = shape['label']

				if not VERBOSE: progress_bar.bar(index, num_shapes)

				if label == 'case':
					# Subdivide normal background
					# Prepare for GCA display
					case_shape = shape
					gca.add_patch(plt.Polygon(area, facecolor="none", edgecolor="r"))
				else:
					inner_shapes.append(shape)
					gca.add_patch(plt.Polygon(area))

					# First look to see if placement has already been calculated
					if ('max_coord' in shape) and (str(SIZE) in shape['max_coord']):
						if VERBOSE: print('Found previously calculated coordinates')
						prev = shape['max_coord'][str(SIZE)]
						max_squares = prev['max_squares']
						max_coord = prev['coord']
					else:
						edit_case_data = True
						# Find optimal placement start point
						index_matrix = []
						for j in range(SIZE):
							for i in range(SIZE):
								index_matrix.append([i, j, area])

						if VERBOSE: print("\nWorking on annotation %s/%s..." % (index, num_shapes))
						pool = Pool(8)
						result = pool.map(calc_placement, index_matrix)
						pool.close()
						pool.join()

						end_time = time.time()

						max_squares = max(result)
						max_index = result.index(max_squares)
						max_coord = index_matrix[max_index][:2]

						if 'max_coord' not in shape:
							case_data['shapes'][index].update({'max_coord': {SIZE: { 'max_squares' : max_squares, 'coord': max_coord  }}})
						else:
							case_data['shapes'][index]['max_coord'].update({SIZE: { 'max_squares' : max_squares, 'coord': max_coord  }})

						if VERBOSE: print("Max:", max(result), index_matrix[max_index][:2])

					if export and VERBOSE: print("Saving...")
					save_squares(area, max_coord, label, index, im, gca=gca, graph=True, save = export)
					if export and VERBOSE: print("...done.")

			# Subdivide the case shape
			if export and VERBOSE: print("Subdividing normal background...")
			save_squares(case_shape['points'], [0, 0], 'normal', 0, im, gca=gca, exclusions=inner_shapes, save = export)
			if export and VERBOSE: print("...finished.")
			if(plot):
				plt.axis('scaled')
				plt.show()
			plt.clf()
			end_time = time.time()
			if not VERBOSE:
				progress_bar.end()
				print(" (%.2f sec)" % (end_time-start_time))
	if edit_case_data:
		if VERBOSE: print("Writing updated annotation file...")
		with open(os.path.join(CASE_DIR, "%s-ann.json" % CASE), 'w') as outfile:
			json.dump(case_data, outfile)

if __name__=='__main__':
	main(False, True)
