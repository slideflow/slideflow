import numpy as np
import argparse
import json
import csv
import math

import os
from os.path import join

from bokeh.plotting import figure, show, output_file
from bokeh.io import curdoc
from bokeh.models.widgets import Button, Dropdown
from bokeh.models.glyphs import Quad
from bokeh.models import CustomJS, ColumnDataSource
from bokeh.layouts import column

class Mosaic:
	def __init__(self, args):
		self.tsne_points = []
		self.metadata = []
		self.GRID = []
		self.SVS = False
		self.plotly_shapes = []
		self.rectangle_coords_x = []
		self.rectangle_coords_y = []
		self.plotly_images = []
		self.tile_zoom_factor = 150
		self.category = "PTC-classic"

		self.tile_root = args.tile
		self.num_tiles_x = args.detail
		self.max_distance_factor = args.leniency

		self.load_metadata(args.meta)
		self.load_bookmark_state(args.bookmark)
		self.place_tile_outlines()
		self.calculate_distances()
		self.pair_tiles_and_points()
		self.make_plot()

	def make_plot(self):
		TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,save,box_select"
		p = figure(tools=TOOLS, match_aspect=True, plot_width=1200, plot_height=1200)
		
		image_paths, x, y, w, h = [], [], [], [], []

		'''for g in self.GRID:
			if g['image_path']:
				image_paths.append(g['image_path'])
				x.append(g['x'] - g['size']/2)
				y.append(g['y'] + g['size']/2)
				w.append(g['size'])
				h.append(g['size'])
				
		p.image_url(url=image_paths,
					x=x,
					y=y,
					w=w,
					h=h,
					alpha=0.3)#[g['alpha'] for g in self.GRID])'''


# -----------------------
		#follicular = ColumnDataSource(data=dict(x=[g['alpha'] for g in self.GRID]))
		follicular = ColumnDataSource(data=dict(x=self.get_category_alpha("PTC-follicular")))

		for g in self.GRID:
			if g['image_path']:
				p.image_url(url=[g['image_path']],
						x=g['x'] - g['size']/2,
						y=g['y'] + g['size']/2,
						w=g['size'],
						h=g['size'])
						#alpha=g['alpha'])

		original_source = ColumnDataSource( dict(
						top = [g['y'] + g['size']/2 for g in self.GRID], # self.rectangle_coords_x,
						bottom = [g['y'] - g['size']/2 for g in self.GRID], #self.rectangle_coords_y,
						left = [g['x'] + g['size']/2 for g in self.GRID],
						right = [g['x'] - g['size']/2 for g in self.GRID],
						fill_alpha = [1-g['alpha'] for g in self.GRID],
						line_alpha = [1-g['alpha'] for g in self.GRID] ))

		glyph = Quad(left="left", right="right", top="top", bottom="bottom", fill_color="white", line_color='lightgray',
					 fill_alpha="fill_alpha", line_alpha="line_alpha")

		p.add_glyph(original_source, glyph)

		'''callback = CustomJS(args=dict(p=p, source=original_source, alpha_list=follicular), code="""
			var alphas = alpha_list.getv('data');
			var data = source.data;
			alpha1 = data['fill_alpha']
			alpha2 = data['line_alpha']
			for (i = 0; i < alpha1.length; i++) {
				alpha1[i] = alphas['x'][i]
				alpha2[i] = alphas['x'][i]
			}
			p.change.emit()
		""")'''

		#def doit(attr, old, new):
		#	for i in range
		
		button = Button(label="Change!", button_type="success")
		#button.callback = callback
		button.on_change('value', doit)
		curdoc().add_root(column(p, button))

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
								  'alpha': 1,
								  'image_path': None})

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

			grid_tile['size'] = rect_size
			#grid_tile['rectangle'] = tile
			grid_tile['neighbors'] = []
			grid_tile['paired_point'] = None

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
			#for i in range(len(tile['distances'])):
			closest_point = tile['distances'][0][0]
			point = self.tsne_points[closest_point]
			if not os.path.exists(join(os.getcwd(), point['image_path'])):
				print(f'Does not exist: {point["image_path"]}')
			#		continue
			#	else:
			tile['image_path'] = point['image_path']
			#		break

			tile_alpha, num_case, num_other = 1, 0, 0
			if self.category and len(tile['points']):
				for point_index in tile['points']:
					point = self.tsne_points[point_index]
					if point['category'] == self.category:
						num_case += 1
					else:
						num_other += 1
				fraction_svs = num_case / (num_other + num_case)
				tile_alpha = fraction_svs
				tile['alpha'] = tile_alpha
			num_placed += 1
		print(f"[INFO] Num placed: {num_placed}")

	def get_category_alpha(self, category):
		alpha_list = []
		for tile in self.GRID:
			if not len(tile['points']): 
				alpha_list.append(1)
				continue
			num_cat, num_other = 0, 0
			for point_index in tile['points']:
				point = self.tsne_points[point_index]
				if point['category'] == category:
					num_cat += 1
				else:
					num_other += 1
			alpha = num_cat / (num_other + num_cat)
			alpha_list.append(1-alpha)
		return alpha_list

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

#if __name__ == '__main__':
	#args = get_args()
	#mosaic = Mosaic(args)
class Args:
	bookmark = '/Users/james/results/Thyroid/Test_Set_B/higher_perplexity_set_b.txt'
	meta = '/Users/james/results/Thyroid/Test_Set_B/metadata.tsv'
	tile = join(os.path.basename(os.path.dirname(__file__)), 'static', 'img')#'./img'#'/home/shawarma/data/Thyroid/SVS_Test_Set_B/tiles'
	detail = 50
	figsize = 100
	leniency = 2
	def __init__(self): pass
args = Args()
mosaic = Mosaic(args)