from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

import numpy as np
import argparse
import json
import csv
import math

from os.path import join

class Mosaic:
	def __init__(self, args):
		'''N = 1000
		random_x = np.random.randn(N)
		random_y = np.random.randn(N)
		trace = go.Scatter(x = random_x,
						   y = random_y,
						   mode = 'markers')
		data = [trace]
		plot(data, filename='basic-scatter')'''
		self.tsne_points = []
		self.metadata = []
		self.GRID = []
		self.plotly_shapes = []
		self.rectangle_coords_x = []
		self.rectangle_coords_y = []
		self.plotly_images = []
		self.tile_zoom_factor = 15

		self.tile_root = args.tile
		self.num_tiles_x = args.detail
		self.max_distance_factor = args.leniency

		self.load_metadata(args.meta)
		self.load_bookmark_state(args.bookmark)
		self.place_tile_outlines()
		self.make_plot()

	def make_plot(self):
		'''for i in range(10):
			self.plotly_images.append(
				dict(
					source= 'img/234800_1220.jpg', #'/Users/james/results/Thyroid/Mosaic/tiles/234800/234800_1220.jpg', #'https://images.plot.ly/language-icons/api-home/python-logo.png', 
					xref= "x",
					yref= "y",
					x= i * 5,
					y= 0,
					sizex= 5,
					sizey= 5,
					sizing = "contain",
					opacity = 1.0,
					xanchor= "center",
					yanchor= "center",
					visible = True,
					layer = "below"
				)
			)'''

		print(self.tile_size)

		trace = go.Scatter(
			x=self.rectangle_coords_x, 
			y=self.rectangle_coords_y,
			mode='markers',
			marker = dict(
				sizeref = 0.01,
				sizemode = 'area',
				size = [g['size'] for g in self.GRID],#[self.tile_size] * len(self.rectangle_coords_x),
				symbol = 1
			)
		)

		layout = go.Layout(
			xaxis = dict(
				nticks = 10,
				#domain = [0, 0.45],
				title = "shared X axis"
			),
			yaxis = dict(
				scaleanchor = "x",
				#domain = [0, 0.45],
				title = "1:1"
			),
			#images = self.plotly_images,
			#shapes = self.plotly_shapes
		)

		fig = go.Figure(data=[trace], layout=layout)
		plot(fig, config={'scrollZoom': True}, filename='basic-scatter')

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
								  'active': False,
								  'image': None})

		# Add point indices to grid
		for point in self.tsne_points:
			x_index = int((point['x'] - min_x) / self.tile_size)
			y_index = int((point['y'] - min_y) / self.tile_size)
			for g in self.GRID:
				if g['x_index'] == x_index and g['y_index'] == y_index:
					g['points'].append(point['index'])

		print(f"[Core] Loaded {len(self.tsne_points)} points.")

	def place_tile_outlines(self):
		print("[Mosaic] Placing tile outlines...")
		max_grid_density = 1
		for g in self.GRID:
			max_grid_density = max(max_grid_density, len(g['points']))
		for grid_tile in self.GRID:
			rect_size = min((len(grid_tile['points']) / max_grid_density) * self.tile_zoom_factor, 1) * self.tile_size

			shape = {
				'type': 'rect',
				'x0': grid_tile['x'] - rect_size/2,
				'y0': grid_tile['y'] - rect_size/2,
				'x1': grid_tile['x'] + rect_size/2,
				'y1': grid_tile['y'] + rect_size/2,
				'line': {
					'color': 'rgba(128, 0, 128, 1)',
					'width': 2,
				},
				'fillcolor': 'rgba(128, 0, 128, 0.7)',
			}

			self.plotly_shapes.append(shape)
			self.rectangle_coords_x.append(grid_tile['x'])
			self.rectangle_coords_y.append(grid_tile['y'])

			'''tile = Rectangle((grid_tile['x'] - rect_size/2, 
							  grid_tile['y'] - rect_size/2), 
							  rect_size, 
							  rect_size, 
							  fill=True, alpha=1, facecolor='white', edgecolor="#cccccc")
			self.ax.add_patch(tile)'''

			grid_tile['size'] = rect_size
			#grid_tile['rectangle'] = tile
			grid_tile['neighbors'] = []
			grid_tile['paired_point'] = None

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