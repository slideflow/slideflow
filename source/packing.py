import json
import os, sys
import warnings
import shapely.geometry as sg
import matplotlib.pyplot as plt
import numpy as np
import progress_bar as progress
from matplotlib import path
from PIL import Image
from os import listdir
from os.path import isfile, join

# TODO: Progress bar for subdividing/saving stage
# TODO: Need to filter out normal image boxes that are all white

SIZE = 1024
FACTOR = 10 # Annotation compression factor, with which to multiply annotation coordinates
AN_F = "/media/falafel/Backup/Other files/Thyroid Research - Files/JPG/Annotated" # Folder conntaining annotated .json files
SAVE_FOLDER = "/media/falafel/Backup/Other files/Thyroid Research - Files/%s" % SIZE
CASES = [f[:-5] for f in listdir(AN_F) if isfile(join(AN_F, f)) and f[-4:] == 'json']

def place_squares(area, offset_x = 0, offset_y = 0, gca = None, graph = False, image = None, label = None, num = None, exclusions = None, tile_size = SIZE, color='g', max_squares = None):
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
	for j in range(int(y_range)/tile_size):
		for i in range(int(x_range)/tile_size):
			x = x_min + i*tile_size
			y = y_min + j*tile_size
			sTestSquare = sg.Polygon([(x, y), (x, y+tile_size), (x+tile_size, y+tile_size), (x+tile_size, y)])
			if graph: mTestSquare = plt.Rectangle((x, y), tile_size, tile_size, fc=color)

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

def square_iterator(gca, area, ann_num, tile_size, exclusions=None):
	'''Iterate through possible square placement starting points and 
	   find the placement which maximizes number of squares placed'''

	max_squares = 0
	max_coord = []

	max_it = tile_size*tile_size

	for j in range(tile_size):
		for i in range(tile_size):
			sys.stdout.write("\rAnalyzing annotation #%s: " % ann_num)
			progress.bar((j*tile_size)+i, max_it)
			count = place_squares(area, i, j, gca, exclusions = exclusions, tile_size = tile_size)
			if count >= max_squares: 
				max_squares = count
				max_coord = [i, j]

	sys.stdout.write("\n")

	# Show final result
	#place_squares(area, max_coord[0], max_coord[1], gca, True)

	return max_squares, max_coord

def subdivider(area, label, offset, im):
	''' Based on a pre-calculated optimal starting point, subdivide
	    area into squares'''
	x_min = min(p[0] for p in area) + offset[0]
	x_max = max(p[0] for p in area) + offset[1]
	x_range = x_max - x_min
	y_min = min(p[1] for p in area) + offset[0]
	y_max = max(p[1] for p in area) + offset[1]
	y_range = y_max - y_min

	for j in range(int(y_range)/SIZE):
		for i in range(int(x_range)/SIZE):
			x = x_

for CASE_ID in CASES:
	print("Working on case %s" % CASE_ID)
	with open (join(AN_F, "%s.json" % CASE_ID)) as case:
		d = json.load(case)

		# Initiate grid for drawing
		plt.axes(label=CASE_ID)
		gca = plt.gca()
		gca.invert_yaxis()
		gca.tick_params(axis="x", top=True, labeltop=True, bottom = False, labelbottom = False)

		# Iterate through each annotation
		ann_num = 1
		for s in d['shapes']:
			area = s['points']
			area_full = np.multiply(area, FACTOR)
			area_small = np.divide(area, FACTOR)

			label = s['label']

			if label == "case":
				mPolygon = plt.Polygon(area_full, facecolor="none", edgecolor="r")
			else: 
				mPolygon = plt.Polygon(area_full)

			# Add annotation polygon to the grid
			gca.add_patch(mPolygon)		

			if label == "case": continue

			#squares, sq_co = square_iterator(gca, area, ann_num, int(SIZE/FACTOR))
			squares, sq_co = square_iterator(gca, area_small, ann_num, int(SIZE/(FACTOR*FACTOR)))

			print("Number of sections: %i" % squares)
			if squares > 0: print("Starting point: (%s, %s)\n" % (sq_co[0]*FACTOR, sq_co[1]*FACTOR))
			else: print("\n")

			s['square_offset'] = sq_co
			s['number'] = ann_num
			s['max_squares'] = squares

			ann_num += 1

		# Now open the image
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			im = Image.open((join(AN_F, "%s.jpg" % CASE_ID)))
		print('Opened %s image "%s.jpg" %s %s' % (im.format, CASE_ID, im.size, im.mode))

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
				place_squares(area, offset[0], offset[1], gca, graph=True, image=im, label=CASE_ID, num=ann_num, max_squares = max_squares)

			sys.stdout.write("\n")

		# Subdivide normal background
		#print('Subdividing normal background...')
		#place_squares(case_shape['points'], 0, 0, gca, graph=True, image=im, label='normal', num=0, exclusions = other_shapes, color='r')

		# Show grid
		#plt.axis('scaled')
		#plt.show()
