import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

import json
import sys
import math
import csv

from os.path import join

bookmark = '/home/shawarma/data/Thyroid/tsne/state.txt'
metadata_path = '/home/shawarma/data/Thyroid/tsne/metadata.tsv'
tile_root = '/home/shawarma/data/Thyroid/tiles'

num_tiles_x = 70

fig = plt.figure(figsize=(200, 200))
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom = 0.05, top=0.95)
ax.set_aspect('equal', 'box')

tsne_points = [] #format x, y, index
metadata = []

with open(metadata_path, 'r') as metadata_file:
	reader = csv.reader(metadata_file, delimiter='\t')
	headers = next(reader, None)
	for row in reader:
		metadata.append(row)

with open(bookmark, 'r') as bookmark_file:
	state = json.load(bookmark_file)
	projection_points = state[0]['projections']
	point_index = 0
	for i, p in enumerate(projection_points):
		meta = metadata[i]
		tile_num = meta[0]
		case = meta[1]
		category = meta[2]
		if 'tsne-1' in p:
			tsne_points.append({'x':p['tsne-0'],
								'y':p['tsne-1'],
								'index':point_index,
								'neighbors':[],
								'paired_tile':None,
								'image_path':join(tile_root, case, f"{case}_{tile_num}.jpg")})
			point_index += 1
	x_points = [p['x'] for p in tsne_points]
	y_points = [p['y'] for p in tsne_points]
	_x_width = max(x_points) - min(x_points)
	_y_width = max(y_points) - min(y_points)
	buffer = (_x_width + _y_width)/2 * 0.1
	max_x = max(x_points) + buffer
	min_x = min(x_points) - buffer
	max_y = max(y_points) + buffer
	min_y = min(y_points) - buffer

tile_size = (max_x - min_x) / num_tiles_x
num_tiles_y = int((max_y - min_y) / tile_size)
max_distance = math.sqrt(2*((tile_size/2)**2))

tile_coord_x = [(i*tile_size)+min_x for i in range(num_tiles_x)]
tile_coord_y = [(j*tile_size)+min_y for j in range(num_tiles_y)]

tiles = []

i=0
for y in tile_coord_y:
	for x in tile_coord_x:
		tile = Rectangle((x - tile_size/2, y - tile_size/2), tile_size, tile_size, fill=None, alpha=1, color='white')
		ax.add_patch(tile)
		tiles.append({'rectangle':tile,
					  'x':x,
					  'y':y,
					  'index':i,
					  'neighbors':[],
					  'paired_point':None})
		i+=1

num_placed = 0

tile_point_distances = []

for tile in tiles:
	# Calculate distance for each point from center
	distances = []
	for point in tsne_points:
		distance = math.sqrt((point['x']-tile['x'])**2 + (point['y']-tile['y'])**2)
		distances.append([point['index'], distance])
	distances.sort(key=lambda d: d[1])
	#eligible_distances = []
	for d in distances:
		if d[1] <= max_distance:
			#eligible_distances.append(d)
			tile['neighbors'].append(d)
			tsne_points[d[0]]['neighbors'].append([tile['index'], d[1]])
			tile_point_distances.append({'distance': d[1],
										 'tile_index':tile['index'],
										 'point_index':d[0]})
		else:
			break

tile_point_distances.sort(key=lambda d: d['distance'])
for distance_pair in tile_point_distances:
	# Attempt to place pair, skipping if unable (due to other prior pair)
	point = tsne_points[distance_pair['point_index']]
	tile = tiles[distance_pair['tile_index']]
	if not (point['paired_tile'] or tile['paired_point']):
		point['paired_tile'] = True
		tile['paired_point'] = True
		#ax.plot(point['x'], point['y'], 'go')

		#draw image
		tile_image = plt.imread(point['image_path'])
		ax.imshow(tile_image, aspect='equal', origin='lower', extent=[tile['x']-tile_size/2, tile['x']+tile_size/2, tile['y']-tile_size/2, tile['y']+tile_size/2], zorder=99)

		tile['rectangle'].set_color('red')
		tile['rectangle'].set_fill(True)
		tile['rectangle'].set_alpha(0)
		num_placed += 1

print(f"Num placed: {num_placed}")
plt.autoscale()
plt.savefig(join(tile_root, 'Mosaic_map.png'), bbox_inches='tight')
plt.close()

#plt.show()