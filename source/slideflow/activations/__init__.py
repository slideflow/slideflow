import sys
import os
import math
import csv
import cv2
import pickle
import time
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.colors as mcol
import seaborn as sns
import pandas as pd
import tensorflow as tf
import scipy.stats as stats

from os.path import join, isfile, exists
from random import shuffle, sample
from statistics import mean
from math import isnan
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D

import slideflow.util as sfutil
import slideflow.util.statistics as sfstats
from slideflow.util import log, ProgressBar, tfrecords, TCGA
from PIL import Image

# TODO: add check that cached PKL corresponds to current and correct model & slides
# TODO: re-calculate new activations if some slides not present in cache

def sizeof(obj):
    size = sys.getsizeof(obj)
    if isinstance(obj, dict): return size + sum(map(sizeof, obj.keys())) + sum(map(sizeof, obj.values()))
    if isinstance(obj, (list, tuple, set, frozenset)): return size + sum(map(sizeof, obj))
    return size

def sizeof_fmt(num, suffix='B'):
	''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
	for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
		if abs(num) < 1024.0:
			return "%3.1f %s%s" % (num, unit, suffix)
		num /= 1024.0
	return "%.1f %s%s" % (num, 'Yi', suffix)

def create_bool_mask(x, y, w, sx, sy):
	l = max(0,  int(x-(w/2.)))
	r = min(sx, int(x+(w/2.)))
	t = max(0,  int(y-(w/2.)))
	b = min(sy, int(y+(w/2.)))
	m = np.array([[[True]*3]*sx]*sy)
	for yi in range(m.shape[1]):
		for xi in range(m.shape[0]):
			if (t < yi < b) and (l < xi < r):
				m[yi][xi] = [False, False, False]
	return m

class ActivationsVisualizer:
	missing_slides = []
	umaps = []
	tfrecords_paths = []
	slides = []				# List of slide names (without extension or path)
	slide_category_dict = {}
	categories = []
	used_categories = []
	slide_node_dict = {}

	def __init__(self, model, tfrecords, root_dir, image_size, annotations=None, category_header=None, 
					focus_nodes=[], use_fp16=True, batch_size=16, export_csv=False, max_tiles_per_slide=500):
		'''Loads annotations, saved layer activations, and prepares output saving directories.
		Will also read/write processed activations to a PKL cache file to save time in future iterations.'''

		self.focus_nodes = focus_nodes
		self.MAX_TILES_PER_SLIDE = max_tiles_per_slide
		self.IMAGE_SIZE = image_size
		self.tfrecords_paths = np.array(tfrecords)
		self.slides_to_include = [sfutil.path_to_name(tfr) for tfr in self.tfrecords_paths]

		self.FLA = join(root_dir, "stats", "final_layer_activations.csv")
		self.STATS_CSV_FILE = join(root_dir, "stats", "slide_level_summary.csv")
		self.PT_NODE_DICT_PKL = join(root_dir, "stats", "activation_node_dict.pkl")
		self.UMAP_CACHE = join(root_dir, "stats", "umap_cache.pkl")
		self.EXAMPLE_TILES_DIR = join(root_dir, "stats", "example_tiles")
		self.SORTED_DIR = join(root_dir, "stats", "sorted_tiles")
		self.STATS_ROOT = join(root_dir, "stats")
		if not exists(join(root_dir, "stats")):
			os.makedirs(join(root_dir, "stats"))

		# Load annotations if provided
		if annotations and category_header:
			self.load_annotations(annotations, category_header)

		# Load activations
		# Load from PKL (cache) if present
		if exists(self.PT_NODE_DICT_PKL): 
			# Load saved PKL cache
			log.info("Loading pre-calculated activations from pickled files...", 1)
			with open(self.PT_NODE_DICT_PKL, 'rb') as pt_pkl_file:
				self.slide_node_dict = pickle.load(pt_pkl_file)
				self.nodes = list(self.slide_node_dict[list(self.slide_node_dict.keys())[0]].keys())
		# Otherwise try loading from CSV if present
		elif exists(self.FLA):
			self.slide_node_dict = {}
			for slide in self.slides_to_include:
				self.slide_node_dict.update({slide: {}})
			with open(self.FLA, 'r') as fl_file:
				log.info(f"Reading final layer activations from {sfutil.green(self.FLA)}...", 1)
				fl_reader = csv.reader(fl_file)
				header = next(fl_reader)
				csv_node_names = [h for h in header if h[:6] == "FLNode"]
				try:
					slide_i = header.index("Slide")
				except:
					log.error(f"Unable to load activations from CSV at {sfutil.green(self.FLA)}; format incorrect", 1)
					return

				for node in csv_node_names:
					node_num = int(node.strip("FLNode"))
					for slide in self.slides_to_include:
						self.slide_node_dict[slide].update({node_num: []})

				self.nodes = list(range(len(csv_node_names)))
				if self.nodes != [int(n.strip("FLNode")) for n in csv_node_names]:
					log.error(f'Unable to load activations CSV at {sfutil.green(self.FLA)}: incorrect header format.')
					return

				for i, row in enumerate(fl_reader):
					print(f"Reading activations for tile {i}...", end="\r")
					slide = row[slide_i]
					for node in csv_node_names:
						node_i = header.index(node)
						node_num = int(node.strip("FLNode"))
						val = float(row[node_i])
						self.slide_node_dict[slide][node_num] += [val]
			print()
			# Write PKL cache
			with open(self.PT_NODE_DICT_PKL, 'wb') as pt_pkl_file:
				pickle.dump(self.slide_node_dict, pt_pkl_file)
		# Otherwise will need to generate new activations from a given model
		else:
			self.generate_activations_from_model(model, use_fp16=use_fp16, batch_size=batch_size, export_csv=export_csv)
			self.nodes = list(self.slide_node_dict[list(self.slide_node_dict.keys())[0]].keys())

		# Now delete slides not included in our filtered TFRecord list
		loaded_slides = list(self.slide_node_dict.keys())
		for loaded_slide in loaded_slides:
			if loaded_slide not in self.slides_to_include:
				del self.slide_node_dict[loaded_slide]

		# Now screen for missing slides in activations
		for slide in self.slides_to_include:
			try:
				if self.slide_node_dict[slide][0] == []:
					self.missing_slides += [slide]
				elif self.categories:
					self.used_categories = list(set(self.used_categories + [self.slide_category_dict[slide]]))
					self.used_categories.sort()
			except KeyError:
				log.warn(f"Skipping unknown slide {slide}", 1)
				self.missing_slides += [slide]
		log.info(f"Loaded activations from {(len(self.slides_to_include)-len(self.missing_slides))}/{len(self.slides_to_include)} slides ({len(self.missing_slides)} missing)", 1)
		log.info(f"Observed categories (total: {len(self.used_categories)}):", 1)
		for c in self.used_categories:
			log.empty(f"\t{c}", 2)

	def load_annotations(self, annotations, category_header):
		with open(annotations, 'r') as ann_file:
			log.info("Reading annotations...", 1)
			ann_reader = csv.reader(ann_file)
			header = next(ann_reader)
			slide_i = header.index(TCGA.slide)
			category_i = header.index(category_header)
			for row in ann_reader:
				slide = row[slide_i]
				category = row[category_i]
				if slide not in self.slides_to_include: continue
				self.slide_category_dict.update({slide:category})

		self.categories = list(set(self.slide_category_dict.values()))

		if self.slide_node_dict:		
			# If initial loading has been completed already, make note of observed categories in given header
			for slide in self.slides_to_include:
				try:
					if self.slide_node_dict[slide][0] != []:
						self.used_categories = list(set(self.used_categories + [self.slide_category_dict[slide]]))
						self.used_categories.sort()
				except KeyError:
					# Skip unknown slide
					pass
			log.info(f"Observed categories (total: {len(self.used_categories)}):", 1)
			for c in self.used_categories:
				log.empty(f"\t{c}", 2)

	def get_tile_node_activations_by_category(self, node):
		if not self.categories: 
			log.warn("Unable to calculate node activations by category; annotations not loaded. Please load with load_annotations()")
			return
		tile_node_activations_by_category = []
		for c in self.used_categories:
			nodelist = [self.slide_node_dict[pt][node] for pt in self.slides_to_include if (pt not in self.missing_slides and self.slide_category_dict[pt] == c)]
			tile_node_activations_by_category += [[nodeval for nl in nodelist for nodeval in nl]]
		return tile_node_activations_by_category

	def get_top_nodes_by_slide(self):
		# First ensure basic stats have been calculated
		if not hasattr(self, 'nodes_avg_pt'):
			self.calculate_activation_averages_and_stats()
		
		return self.nodes_avg_pt

	def get_top_nodes_by_tile(self):
		# First ensure basic stats have been calculated
		if not hasattr(self, 'nodes_avg_pt'):
			self.calculate_activation_averages_and_stats()
		
		return self.nodes

	def get_tfrecords_path(self, slide):
		for tfr in self.tfrecords_paths:
			if sfutil.path_to_name(tfr) == slide:
				return tfr
		log.error(f"Unable to find TFRecord path for slide {sfutil.green(slide)}", 1)
		sys.exit()

	def generate_activations_from_model(self, model, use_fp16=True, batch_size=16, export_csv=True):
		# Rename tfrecord_array to tfrecords
		log.info(f"Calculating final layer activations from model {sfutil.green(model)}", 1)

		# Load model
		_model = tf.keras.models.load_model(model)
		#complete_model = False
		#try:
		loaded_model = tf.keras.models.Model(inputs=[_model.input, _model.layers[0].layers[0].input],
											 outputs=[_model.layers[0].layers[-1].output, _model.layers[-1].output])
		#except AttributeError:
		#	# DANGEROUS: UNKNOWN RESULTS MAY OCCUR; WILL NOT IMPLEMENT FOR NOW
		#	# Provides support for complete models that were not generated using Slideflow
		#	complete_model = True
		#	loaded_model = tf.keras.models.Model(inputs=[_model.input],
		#										 outputs=[_model.layers[-2].output])
		
		unique_slides = list(set([sfutil.path_to_name(tfr) for tfr in self.tfrecords_paths]))

		# Prepare PKL export dictionary
		self.slide_node_dict = {}
		for slide in unique_slides:
			self.slide_node_dict.update({slide: {}})

		def _parse_function(record):
			features = tf.io.parse_single_example(record, tfrecords.FEATURE_DESCRIPTION)
			slide = features['slide']
			image_string = features['image_raw']
			raw_image = tf.image.decode_jpeg(image_string, channels=3)
			processed_image = tf.image.per_image_standardization(raw_image)
			processed_image = tf.image.convert_image_dtype(processed_image, tf.float16 if use_fp16 else tf.float32)
			processed_image.set_shape([self.IMAGE_SIZE, self.IMAGE_SIZE, 3])
			return processed_image, slide

		# Calculate final layer activations for each tfrecord
		fla_start_time = time.time()
		nodes_names, logits_names = [], []
		if export_csv:
			outfile = open(self.FLA, 'w')
			csvwriter = csv.writer(outfile)

		for t, tfrecord in enumerate(self.tfrecords_paths):
			#log.empty(f"Calculating activations from {sfutil.green(tfrecord)}", 2)
			dataset = tf.data.TFRecordDataset(tfrecord)

			dataset = dataset.map(_parse_function, num_parallel_calls=8)
			dataset = dataset.batch(batch_size, drop_remainder=False)
			
			fl_activations_combined, logits_combined, slides_combined = [], [], []

			for i, data in enumerate(dataset):
				batch_processed_images, batch_slides = data
				batch_slides = batch_slides.numpy()
				sys.stdout.write(f"\r(TFRecord {t+1:>3}/{len(self.tfrecords_paths):>3}) (Batch {i+1:>3}) ({len(fl_activations_combined):>5} images): {sfutil.green(sfutil.path_to_name(tfrecord))}")
				sys.stdout.flush()

				#if not complete_model:
				fl_activations, logits = loaded_model.predict([batch_processed_images, batch_processed_images])
				#else:
				#	fl_activations = loaded_model.predict([batch_processed_images])
				#	logits = [[-1]] * batch_size
				fl_activations_combined = fl_activations if fl_activations_combined == [] else np.concatenate([fl_activations_combined, fl_activations])
				logits_combined = logits if logits_combined == [] else np.concatenate([logits_combined, logits])
				slides_combined = batch_slides if slides_combined == [] else np.concatenate([slides_combined, batch_slides])

			if not nodes_names and not logits_names:
				nodes_names = [f"FLNode{f}" for f in range(fl_activations_combined.shape[1])]
				logits_names = [f"Logits{l}" for l in range(logits_combined.shape[1])]
				header = ["Slide"] + logits_names + nodes_names
				if export_csv:
					csvwriter.writerow(header)
				for n in range(len(nodes_names)):
					for slide in unique_slides:
						self.slide_node_dict[slide].update({n: []})

			# Randomly select tiles, only taking as much as specified
			if self.MAX_TILES_PER_SLIDE and len(fl_activations_combined) > self.MAX_TILES_PER_SLIDE:
				selected_indices = random.sample(list(range(len(fl_activations_combined))), self.MAX_TILES_PER_SLIDE)
				slides_combined = slides_combined[selected_indices]
				logits_combined = logits_combined[selected_indices]
				fl_activations_combined = fl_activations_combined[selected_indices]

			# Export to PKL and CSV
			for i in range(len(fl_activations_combined)):
				slide = slides_combined[i].decode('utf-8')
				activations_vals = fl_activations_combined[i].tolist()
				logits_vals = logits_combined[i].tolist()
				# Write to CSV
				if export_csv:
					row = [slide] + logits_vals + activations_vals
					csvwriter.writerow(row)
				# Write to PKL
				for n in range(len(nodes_names)):
					val = activations_vals[n]
					self.slide_node_dict[slide][n] += [val]

			print(f"{'Size':>6}: {sizeof_fmt(sizeof(self.slide_node_dict)):>8}")

		if export_csv:
			outfile.close()

		fla_calc_time = time.time()
		log.info(f"Activation calculation time: {fla_calc_time-fla_start_time:.0f} sec", 1)
		if export_csv:
			log.complete(f"Final layer activations saved to {sfutil.green(self.FLA)}", 1)
		
		# Dump PKL dictionary to file
		with open(self.PT_NODE_DICT_PKL, 'wb') as pt_pkl_file:
			pickle.dump(self.slide_node_dict, pt_pkl_file)
		log.complete(f"Final layer activations cached to {sfutil.green(self.PT_NODE_DICT_PKL)}", 1)

	def export_activations_to_csv(self, nodes=None):
		with open(self.FLA, 'w') as outfile:
			csvwriter = csv.writer(outfile)
			nodes = self.nodes if not nodes else nodes
			header = ["Slide"] + [f"FLNode{f}" for f in nodes]
			csvwriter.writerow(header)
			for slide in self.slide_node_dict:
				row = [slide]
				for n in nodes:
					row += [self.slide_node_dict[slide][n]]
				csvwriter.writewrow(row)

	def calculate_activation_averages_and_stats(self, annotations=None, category_header=None):
		if not self.categories and (annotations and category_header):
			self.load_annotations(annotations, category_header)
		elif not self.categories:
			log.warn("Unable to calculate activations statistics; annotations not loaded. Please load with load_annotations() or provide 'annotations' and 'category_header'")
			return
		empty_category_dict = {}
		self.node_dict_avg_pt = {}
		node_stats = {}
		node_stats_avg_pt = {}
		for category in self.categories:
			empty_category_dict.update({category: []})
		for node in self.nodes:
			self.node_dict_avg_pt.update({node: deepcopy(empty_category_dict)})

		for node in self.nodes:
			# For each node, calculate average across tiles found in a slide
			print(f"Calculating activation averages & stats for node {node}...", end="\r")
			for slide in self.slides_to_include:
				if slide in self.missing_slides: continue
				pt_cat = self.slide_category_dict[slide]
				avg = mean(self.slide_node_dict[slide][node])
				self.node_dict_avg_pt[node][pt_cat] += [avg]
			
			# Tile-level ANOVA
			fvalue, pvalue = stats.f_oneway(*self.get_tile_node_activations_by_category(node))
			if not isnan(fvalue) and not isnan(pvalue): 
				node_stats.update({node: {'f': fvalue,
										  'p': pvalue} })
			else:
				node_stats.update({node: {'f': -1,
										  'p': 1} })

			# Patient-level ANOVA
			fvalue, pvalue = stats.f_oneway(*[self.node_dict_avg_pt[node][c] for c in self.used_categories])
			if not isnan(fvalue) and not isnan(pvalue): 
				node_stats_avg_pt.update({node: {'f': fvalue,
												 'p': pvalue} })
			else:
				node_stats_avg_pt.update({node: {'f': -1,
										  		 'p': 1} })
		print()

		try:
			self.nodes = sorted(self.nodes, key=lambda n: node_stats[n]['p'])
			self.nodes_avg_pt = sorted(self.nodes, key=lambda n: node_stats_avg_pt[n]['p'])
		except:
			log.warn("No stats calculated; unable to sort nodes.", 1)
			self.nodes_avg_pt = self.nodes
			
		for i, node in enumerate(self.nodes):
			if self.focus_nodes and (node not in self.focus_nodes): continue
			try:
				log.info(f"Tile-level P-value ({node}): {node_stats[node]['p']}", 1)
				log.info(f"Patient-level P-value: ({node}): {node_stats_avg_pt[node]['p']}", 1)
			except:
				log.warn(f"No stats calculated for node {node}", 1)
			if (not self.focus_nodes) and i>9: break
		if not exists(self.STATS_CSV_FILE):
			self.save_node_statistics_to_csv(self.nodes_avg_pt, tile_stats=node_stats, slide_stats=node_stats_avg_pt)
		else:
			log.info(f"Stats file already generated at {sfutil.green(self.STATS_CSV_FILE)}; not regenerating", 1)

	def generate_box_plots(self, annotations=None, category_header=None, interactive=False):
		if not self.categories and (annotations and category_header):
			self.load_annotations(annotations, category_header)
		elif not self.categories:
			log.warn("Unable to generate box plots; annotations not loaded. Please load with load_annotations() or provide 'annotations' and 'category_header'")
			return

		# First ensure basic stats have been calculated
		if not hasattr(self, 'nodes_avg_pt'):
			self.calculate_activation_averages_and_stats()

		# Display tile-level box plots & stats
		log.empty("Generating box plots...")
		for i, node in enumerate(self.nodes):
			if self.focus_nodes and (node not in self.focus_nodes): continue
			plt.clf()
			snsbox = sns.boxplot(data=self.get_tile_node_activations_by_category(node))
			title = f"{node} (tile-level)"
			snsbox.set_title(title)
			snsbox.set(xlabel="Category", ylabel="Activation")
			plt.xticks(plt.xticks()[0], self.used_categories)
			if interactive:
				plt.show()
			boxplot_filename = join(self.STATS_ROOT, f"boxplot_{title}.png")
			plt.savefig(boxplot_filename, bbox_inches='tight')
			if (not self.focus_nodes) and i>4: break

		# Print slide_level box plots & stats
		for i, node in enumerate(self.nodes_avg_pt):
			if self.focus_nodes and (node not in self.focus_nodes): continue
			plt.clf()
			snsbox = sns.boxplot(data=[self.node_dict_avg_pt[node][c] for c in self.used_categories])
			title = f"{node} (slide-level)"
			snsbox.set_title(title)
			snsbox.set(xlabel="Category",ylabel="Average tile activation")
			plt.xticks(plt.xticks()[0], self.used_categories)
			if interactive:
				plt.show()
			boxplot_filename = join(self.STATS_ROOT, f"boxplot_{title}.png")
			plt.savefig(boxplot_filename, bbox_inches='tight')
			if (not self.focus_nodes) and i>4: break
	
	def save_node_statistics_to_csv(self, nodes_avg_pt, tile_stats=None, slide_stats=None):
		# Save results to CSV
		log.empty(f"Writing results to {sfutil.green(self.STATS_CSV_FILE)}...", 1)
		with open(self.STATS_CSV_FILE, 'w') as outfile:
			csv_writer = csv.writer(outfile)
			header = ['slide', 'category'] + [f"FLNode{n}" for n in nodes_avg_pt]
			csv_writer.writerow(header)
			for slide in self.slides_to_include:
				if slide in self.missing_slides: continue
				category = self.slide_category_dict[slide]
				row = [slide, category] + [mean(self.slide_node_dict[slide][n]) for n in nodes_avg_pt]
				csv_writer.writerow(row)
			if tile_stats:
				csv_writer.writerow(['Tile-level statistic', 'ANOVA P-value'] + [tile_stats[n]['p'] for n in nodes_avg_pt])
				csv_writer.writerow(['Tile-level statistic', 'ANOVA F-value'] + [tile_stats[n]['f'] for n in nodes_avg_pt])
			if slide_stats:
				csv_writer.writerow(['Slide-level statistic', 'ANOVA P-value'] + [slide_stats[n]['p'] for n in nodes_avg_pt])
				csv_writer.writerow(['Slide-level statistic', 'ANOVA F-value'] + [slide_stats[n]['f'] for n in nodes_avg_pt])

	def calculate_umap(self, exclude_node=None):
		# First, try to load prior umap cache
		try:
			with open(self.UMAP_CACHE, 'rb') as umap_file:
				self.umaps = pickle.load(umap_file)
				log.info(f"Loaded UMAP cache from {sfutil.green(self.UMAP_CACHE)}", 1)
		except:
			log.info(f"No UMAP cache found at {sfutil.green(self.UMAP_CACHE)}", 1)

		# Check if UMAP cache has already been calculated; if so, return
		slides = [slide for slide in self.slides_to_include if slide not in self.missing_slides]
		nodes_to_include = [n for n in self.nodes if n != exclude_node]
		for um in self.umaps:
			# Check to see if this has already been cached
			if (sorted(um['nodes']) == sorted(nodes_to_include)) and (sorted(um['slides']) == sorted(slides)):
				log.info("UMAP results already calculated and cached", 1)
				return um

		# Otherwise, generate a new umap
		log.info("No compatible UMAP results found in cache; recalculating", 1)
		node_activations = []
		umap_meta = []
		log.empty("Calculating UMAP...", 1)
		for slide in self.slides_to_include:
			if slide in self.missing_slides: continue
			first_node = list(self.slide_node_dict[slide].keys())[0]
			num_vals = len(self.slide_node_dict[slide][first_node])
			for i in range(num_vals):
				node_activations += [[self.slide_node_dict[slide][n][i] for n in nodes_to_include]]
				umap_meta += [{
					'slide': slide,
					'index': i,
				}]
		coordinates = sfstats.gen_umap(np.array(node_activations))
		umap_x = np.array([c[0] for c in coordinates])
		umap_y = np.array([c[1] for c in coordinates])

		# Cache new umap
		umap = {
			'nodes': nodes_to_include,
			'slides': slides,
			'umap_x': umap_x,
			'umap_y': umap_y,
			'umap_meta': umap_meta
		}
		self.umaps += [umap]
		with open(self.UMAP_CACHE, 'wb') as umap_file:
			pickle.dump(self.umaps, umap_file)
			log.info(f"Wrote UMAP cache to {sfutil.green(self.UMAP_CACHE)}", 1)

		return umap

	def generate_mosaic(self, umap=None, focus=None, leniency=1.5, expanded=True, tile_zoom=15, num_tiles_x=50, resolution='high', 
					export=True, tile_um=None, use_fp16=True, save_dir=None):
		
		FOCUS_SLIDE = None
		GRID = []
		points= []
		tile_point_distances = []	

		max_distance_factor = leniency
		mapping_method = 'expanded' if expanded else 'strict'
		tile_zoom_factor = tile_zoom
		export = export
		num_tiles_x = num_tiles_x
		save_dir = self.STATS_ROOT if not save_dir else save_dir

		if not umap:
			umap = self.calculate_umap()

		# Variables used only when loading from slides
		tile_um = tile_um
		
		# Initialize figure
		log.info("Initializing figure...", 1)
		if resolution not in ('high', 'low'):
			log.warn(f"Unknown resolution option '{resolution}', defaulting to low resolution", 1)
		if resolution == 'high':
			fig = plt.figure(figsize=(200,200))
			ax = fig.add_subplot(111, aspect='equal')
		else:
			fig = plt.figure(figsize=(24,18))
			ax = fig.add_subplot(121, aspect='equal')
		ax.set_facecolor("#dfdfdf")
		fig.tight_layout()
		plt.subplots_adjust(left=0.02, bottom=0, right=0.98, top=1, wspace=0.1, hspace=0)
		ax.set_aspect('equal', 'box')
		ax.set_xticklabels([])
		ax.set_yticklabels([])

		# First, load UMAP coordinates	
		log.empty("Loading dimensionality reduction coordinates and plotting points...", 1)
		for i in range(len(umap['umap_x'])):
			slide = umap['umap_meta'][i]['slide']
			points.append({'x':umap['umap_x'][i],
								'y':umap['umap_y'][i],
								'global_index': i,
								'neighbors':[],
								'category':'none',
								'slide':slide,
								'tfrecord':self.get_tfrecords_path(slide),
								'tfrecord_index':umap['umap_meta'][i]['index'],
								'paired_tile':None })
		x_points = [p['x'] for p in points]
		y_points = [p['y'] for p in points]
		_x_width = max(x_points) - min(x_points)
		_y_width = max(y_points) - min(y_points)
		buffer = (_x_width + _y_width)/2 * 0.05
		max_x = max(x_points) + buffer
		min_x = min(x_points) - buffer
		max_y = max(y_points) + buffer
		min_y = min(y_points) - buffer

		log.info(f"Loaded {len(points)} points.", 2)

		tile_size = (max_x - min_x) / num_tiles_x
		num_tiles_y = int((max_y - min_y) / tile_size)
		max_distance = math.sqrt(2*((tile_size/2)**2)) * max_distance_factor

		# Initialize grid
		for j in range(num_tiles_y):
			for i in range(num_tiles_x):
				GRID.append({'x': ((tile_size/2) + min_x) + (tile_size * i),
									'y': ((tile_size/2) + min_y) + (tile_size * j),
									'x_index': i,
									'y_index': j,
									'grid_index': len(GRID),
									'size': tile_size,
									'points':[],
									'distances':[],
									'active': False,
									'image': None})

		# Add point indices to grid
		points_added = 0
		for point in points:
			x_index = int((point['x'] - min_x) / tile_size)
			y_index = int((point['y'] - min_y) / tile_size)
			for g in GRID:
				if g['x_index'] == x_index and g['y_index'] == y_index:
					g['points'].append(point['global_index'])
					points_added += 1
		for g in GRID:
			shuffle(g['points'])
		log.info(f"{points_added} points added to grid", 2)

		# Next, prepare mosaic grid by placing tile outlines
		log.empty("Placing tile outlines...", 1)
		max_grid_density = 1
		for g in GRID:
			max_grid_density = max(max_grid_density, len(g['points']))
		for grid_tile in GRID:
			rect_size = min((len(grid_tile['points']) / max_grid_density) * tile_zoom_factor, 1) * tile_size

			tile = patches.Rectangle((grid_tile['x'] - rect_size/2, 
							  		  grid_tile['y'] - rect_size/2), 
									  rect_size, 
							  		  rect_size, 
									  fill=True, alpha=1, facecolor='white', edgecolor="#cccccc")
			ax.add_patch(tile)

			grid_tile['size'] = rect_size
			grid_tile['rectangle'] = tile
			grid_tile['neighbors'] = []
			grid_tile['paired_point'] = None

		# Then, calculate distances from each point to each spot on the grid
		if mapping_method not in ('strict', 'expanded'):
			raise TypeError("Unknown mapping method")

		log.empty("Calculating tile-point distances...", 1)
		tile_point_start = time.time()
		pb = ProgressBar()
		pb_id = pb.add_bar(0, len(GRID))
		for i, tile in enumerate(GRID):
			pb.update(pb_id, i)
			if mapping_method == 'strict':
				# Calculate distance for each point from center
				distances = []
				for point_index in tile['points']:
					point = points[point_index]
					distance = math.sqrt((point['x']-tile['x'])**2 + (point['y']-tile['y'])**2)
					distances.append([point['global_index'], distance])
				distances.sort(key=lambda d: d[1])
				tile['distances'] = distances
			elif mapping_method == 'expanded':
				# Calculate distance for each point from center
				distances = []
				for point in points:
					distance = math.sqrt((point['x']-tile['x'])**2 + (point['y']-tile['y'])**2)
					distances.append([point['global_index'], distance])
				distances.sort(key=lambda d: d[1])
				for d in distances:
					if d[1] <= max_distance:
						tile['neighbors'].append(d)
						points[d[0]]['neighbors'].append([tile['grid_index'], d[1]])
						tile_point_distances.append({'distance': d[1],
													'grid_index':tile['grid_index'],
													'point_index':d[0]})
					else:
						break
		pb.end()
		tile_point_end = time.time()
		log.info(f"Calculations complete ({tile_point_end-tile_point_start:.0f} sec)", 1)

		if mapping_method == 'expanded':
			tile_point_distances.sort(key=lambda d: d['distance'])

		# Then, pair grid tiles and points according to their distances
		log.empty("Placing image tiles...", 1)
		num_placed = 0
		if mapping_method == 'strict':
			for tile in GRID:
				if not len(tile['distances']): continue
				closest_point = tile['distances'][0][0]
				point = points[closest_point]
				_, tile_image = tfrecords.get_tfrecord_by_index(point['tfrecord'], point['tfrecord_index'])
				tile_alpha, num_slide, num_other = 1, 0, 0
				if FOCUS_SLIDE and len(tile['points']):
					for point_index in tile['points']:
						point = points[point_index]
						if point['slide'] == FOCUS_SLIDE:
							num_slide += 1
						else:
							num_other += 1
					fraction_slide = num_slide / (num_other + num_slide)
					tile_alpha = fraction_slide
				if not export:
					tile_image = cv2.resize(tile_image, (0,0), fx=0.25, fy=0.25)
				image = ax.imshow(tile_image, aspect='equal', origin='lower', extent=[tile['x']-tile['size']/2, 
																						tile['x']+tile['size']/2,
																						tile['y']-tile['size']/2,
																						tile['y']+tile['size']/2], zorder=99, alpha=tile_alpha)
				tile['image'] = image
				num_placed += 1
		elif mapping_method == 'expanded':
			for distance_pair in tile_point_distances:
				# Attempt to place pair, skipping if unable (due to other prior pair)
				point = points[distance_pair['point_index']]
				tile = GRID[distance_pair['grid_index']]
				if not (point['paired_tile'] or tile['paired_point']):
					point['paired_tile'] = True
					tile['paired_point'] = True
					_, tile_image = tfrecords.get_tfrecord_by_index(point['tfrecord'], point['tfrecord_index'])
					if not export:
						tile_image = cv2.resize(tile_image, (0,0), fx=0.25, fy=0.25)
					image = ax.imshow(tile_image, aspect='equal', origin='lower', extent=[tile['x']-tile_size/2,
																					tile['x']+tile_size/2,
																					tile['y']-tile_size/2,
																					tile['y']+tile_size/2], zorder=99)
					tile['image'] = image
					num_placed += 1
		log.info(f"Num placed: {num_placed}", 2)

		# If desired, highlight certain tiles according to a focus list
		if focus:
			for tile in GRID:
				if not len(tile['points']): continue
				num_cat, num_other = 0, 0
				for point_index in tile['points']:
					point = points[point_index]
					if point['tfrecord'] in focus:
						num_cat += 1
					else:
						num_other += 1
				alpha = num_cat / (num_other + num_cat)
				tile['image'].set_alpha(alpha)

		# Finally, finish the mosaic figure
		log.empty("Displaying/exporting figure...", 1)
		ax.autoscale(enable=True, tight=None)
		if export:
			save_path = join(save_dir, f'Mosaic-{num_tiles_x}.png')
			plt.savefig(save_path, bbox_inches='tight')
			log.complete(f"Saved figure to {sfutil.green(save_path)}", 1)
			plt.close()
		else:
			while True:
				try:
					plt.show()
				except UnicodeDecodeError:
					continue
				break

	def plot_2D_umap(self, node=None, exclusion=False, subsample=None, interactive=False, filename=None):
		umap = self.calculate_umap(exclude_node=node if exclusion else None)
		umap_save_dir = join(self.STATS_ROOT, "2d_umap.png") if not filename else filename
		if self.slide_category_dict:
			categories = np.array([self.slide_category_dict[m['slide']] for m in umap['umap_meta']])
		else:
			categories = ["None" for m in umap['umap_meta']]

		# Subsampling
		if subsample:
			ri = sample(range(len(umap['umap_x'])), subsample)
		else:
			ri = list(range(len(umap['umap_x'])))

		unique_categories = list(set(categories[ri]))

		# Prepare pandas dataframe
		df = pd.DataFrame()
		df['umap_x'] = umap['umap_x'][ri]
		df['umap_y'] = umap['umap_y'][ri]
		df['category'] = pd.Series(categories[ri], dtype='category')

		# Make plot
		plt.clf()
		sns.scatterplot(x=umap['umap_x'][ri], y=umap['umap_y'][ri], data=df, hue='category', palette=sns.color_palette('Set1', len(unique_categories)))
		if interactive:
			log.info("Displaying 2D UMAP...", 1)
			plt.show()

		log.info(f"Saving 2D UMAP to {sfutil.green(umap_save_dir)}...", 1)
		plt.savefig(umap_save_dir, bbox_inches='tight')
		return umap

	def plot_3D_umap(self, node, exclusion=False, subsample=1000, interactive=False, filename=None):
		umap = self.calculate_umap(exclude_node=node if exclusion else None)
		umap_name = "3d_umap.png" if not node else f"3d_umap_{node}.png"
		umap_save_dir = join(self.STATS_ROOT, umap_name) if not filename else filename

		# Subsampling
		if subsample:
			ri = sample(range(len(umap['umap_x'])), subsample)
		else:
			ri = list(range(len(umap['umap_x'])))

		umap_x = umap['umap_x'][ri]
		umap_y = umap['umap_y'][ri]

		node_vals = np.array([self.slide_node_dict[m['slide']][node][m['index']] for m in umap['umap_meta']])
		z = node_vals[ri]

		# Plot tiles on a 3D coordinate space with 2 coordinates from UMAP & 3rd from the value of the excluded node
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter(umap_x, umap_y, z, c=z,
									  cmap='viridis',
									  linewidth=0.5,
									  edgecolor="black")
		ax.set_title(f"UMAP with node {node} focus")
		if interactive:
			log.info("Displaying 3D UMAP...", 1)
			plt.show()
		log.info(f"Saving 3D UMAP to {sfutil.green(umap_save_dir)}...", 1)
		plt.savefig(umap_save_dir, bbox_inches='tight')
		
		return umap

	def filter_tiles_by_umap(self, umap, x_lower=-999, x_upper=999, y_lower=-999, y_upper=999):
		# Find tiles that meet UMAP location criteria
		umap_x = umap['umap_x']
		umap_y = umap['umap_y']
		umap_meta = umap['umap_meta']
		filtered_tiles = {}
		num_selected = 0
		for i in range(len(umap_meta)):
			if (x_lower < umap_x[i] < x_upper) and (y_lower < umap_y[i] < y_upper):
				slide = umap_meta[i]['slide']
				tile_index = umap_meta[i]['index']
				if slide not in filtered_tiles:
					filtered_tiles.update({slide: [tile_index]})
				else:
					filtered_tiles[slide] += [tile_index]
				num_selected += 1
		log.info(f"Selected {num_selected} tiles by filter criteria.", 1)
		return filtered_tiles

	def save_example_tiles_gradient(self, nodes=None, tile_filter=None):
		if not nodes:
			nodes = self.focus_nodes
		for node in nodes:
			if not exists(join(self.SORTED_DIR, node)):
				os.makedirs(join(self.SORTED_DIR, node))
			
			gradient = []
			for slide in self.slides_to_include:
				if slide in self.missing_slides: continue
				for i, tile in enumerate(self.slide_node_dict[slide][node]):
					if tile_filter and (slide not in tile_filter) or (i not in tile_filter[slide]):
						continue
					gradient += [{
									'val': tile,
									'slide': slide,
									'index': i
					}]
			gradient = sorted(gradient, key=lambda k: k['val'])
			for i, g in enumerate(gradient):
				print(f"Extracting tile {i} of {len(gradient)} for node {node}...", end="\r")
				for tfr in self.tfrecords_paths:
					if sfutil.path_to_name(tfr) == g['slide']:
						tfr_dir = tfr
				if not tfr_dir:
					log.warn(f"TFRecord location not found for slide {g['slide']}", 1)
				slide, image = tfrecords.get_tfrecord_by_index(tfr_dir, g['index'], decode=False)
				slide = slide.numpy()
				image = image.numpy()
				tile_filename = f"{i}-tfrecord{g['slide']}-{g['index']}-{g['val']:.2f}.jpg"
				image_string = open(join(self.SORTED_DIR, node, tile_filename), 'wb')
				image_string.write(image)
				image_string.close()
			print()

	def save_example_tiles_high_low(self, focus_slides):
		# Extract samples of tiles with highest and lowest values in a particular node
		# for a subset of slides
		for fn in self.focus_nodes:
			for slide in focus_slides:
				sorted_index = np.argsort(self.slide_node_dict[slide][fn])
				lowest = sorted_index[:10]
				highest = sorted_index[-10:]
				lowest_dir = join(self.EXAMPLE_TILES_DIR, fn, slide, 'lowest')
				highest_dir = join(self.EXAMPLE_TILES_DIR, fn, slide, 'highest')
				if not exists(lowest_dir): os.makedirs(lowest_dir)
				if not exists(highest_dir): os.makedirs(highest_dir)

				for tfr in self.tfrecords_paths:
					if sfutil.path_to_name(tfr) == slide:
						tfr_dir = tfr
				if not tfr_dir:
					log.warn(f"TFRecord location not found for slide {slide}", 1)

				def extract_by_index(indices, directory):
					for index in indices:
						slide, image = tfrecords.get_tfrecord_by_index(tfr_dir, index, decode=False)
						slide = slide.numpy()
						image = image.numpy()
						tile_filename = f"tfrecord{slide}-tile{index}.jpg"
						image_string = open(join(directory, tile_filename), 'wb')
						image_string.write(image)
						image_string.close()

				extract_by_index(lowest, lowest_dir)
				extract_by_index(highest, highest_dir)

class TileVisualizer:
	def __init__(self, model, node, shape, tile_width=None, interactive=False):
		self.NODE = node
		self.IMAGE_SHAPE = shape
		self.TILE_WIDTH = tile_width if tile_width else int(self.IMAGE_SHAPE[0]/6)
		self.interactive = interactive
		log.info("Initializing tile visualizer", 1)
		log.info(f"Node: {sfutil.bold(str(node))} | Shape: ({shape[0]}, {shape[1]}, {shape[2]}) | Window size: {self.TILE_WIDTH}", 1)
		log.info(f"Loading Tensorflow model at {sfutil.green(model)}...", 1)
		_model = tf.keras.models.load_model(model)
		self.loaded_model = tf.keras.models.Model(inputs=[_model.input, _model.layers[0].layers[0].input],
												  outputs=[_model.layers[0].layers[-1].output, _model.layers[-1].output])

	def visualize_tile(self, tile, save_dir=None, zoomed=True):
		log.info(f"Processing tile at {sfutil.green(tile)}...", 1)
		tilename = sfutil.path_to_name(tile)
		# First, open tile image
		self.tile_image = Image.open(tile)
		image_file = open(tile, 'rb')
		self.tf_raw_image = image_file.read()

		# Next, process image with Tensorflow
		tf_decoded_image = tf.image.decode_jpeg(self.tf_raw_image, channels=3)
		self.tf_processed_image = tf.image.per_image_standardization(tf_decoded_image)
		self.tf_processed_image = tf.image.convert_image_dtype(self.tf_processed_image, tf.float16)
		self.tf_processed_image.set_shape(self.IMAGE_SHAPE)
		image_batch = np.array([self.tf_processed_image.numpy()])

		# Calculate baseline activations/predictions
		base_activations, base_logits = self.loaded_model.predict([image_batch, image_batch])

		# Now create the figure
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111)
		self.implot = plt.imshow(self.tile_image)

		if self.interactive:
			self.rect = patches.Rectangle((0, 0), self.TILE_WIDTH, self.TILE_WIDTH, facecolor='white', zorder=20)
			self.ax.add_patch(self.rect)

		activation_map, max_center_x, max_center_y = self.calculate_activation_map()

		self.generate_figure(tilename, activation_map, max_x=max_center_x,
													   max_y=max_center_y,
													   save_dir=save_dir,
													   zoomed_extent=zoomed)

	def predict_masked(self, x, y, index):
		mask = create_bool_mask(x, y, self.TILE_WIDTH, self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1])
		masked = self.tf_processed_image.numpy() * mask
		act, _ = self.loaded_model.predict([np.array([masked]), np.array([masked])])
		return act[0][index]	

	def calculate_activation_map(self, stride_div=4):
		sx = self.IMAGE_SHAPE[0]
		sy = self.IMAGE_SHAPE[1]
		w  = self.TILE_WIDTH
		stride = int(self.TILE_WIDTH / stride_div)
		min_x  = int(w/2)
		max_x  = int(sx - w/2)
		min_y  = int(w/2)
		max_y  = int(sy - w/2)

		act_array = []
		for yi in range(min_y, max_y, stride):
			for xi in range(min_x, max_x, stride):
				mask = create_bool_mask(xi, yi, w, sx, sy)
				masked = self.tf_processed_image.numpy() * mask
				act, _ = self.loaded_model.predict([np.array([masked]), np.array([masked])])
				act_array += [act[0][self.NODE]]
				print(f"Calculating activations at x:{xi}, y:{yi}; act={act[0][self.NODE]}", end='\033[K\r')
		max_center_x = max(range(min_x, max_x, stride))
		max_center_y = max(range(min_y, max_y, stride))
		reshaped_array = np.reshape(np.array(act_array), [len(range(min_x, max_x, stride)), 
														  len(range(min_y, max_y, stride))])
		print()

		return reshaped_array, max_center_x, max_center_y

	def generate_figure(self, name, activation_map=None, max_x=None, max_y=None, save_dir=None, zoomed_extent=False):
		# Create hover and click events
		def hover(event):
			if event.xdata:
				self.rect.set_xy((event.xdata-self.TILE_WIDTH/2, event.ydata-self.TILE_WIDTH/2))
				print(self.predict_masked(event.xdata, event.ydata, index=self.NODE), end='\r')
				self.fig.canvas.draw_idle()

		def click(event):
			if event.button == 1:
				self.TILE_WIDTH = min(min(self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1]), self.TILE_WIDTH + 25)
				self.rect.set_width(self.TILE_WIDTH)
				self.rect.set_height(self.TILE_WIDTH)
			else:
				self.TILE_WIDTH = max(0, self.TILE_WIDTH - 25)
				self.rect.set_width(self.TILE_WIDTH)
				self.rect.set_height(self.TILE_WIDTH)
			self.fig.canvas.draw_idle()	

		if self.interactive:
			self.fig.canvas.mpl_connect('motion_notify_event', hover)
			self.fig.canvas.mpl_connect('button_press_event', click)
		
		if activation_map is not None:
			# Define color map
			jetMap = np.flip(np.linspace(0.45, 0.95, 255))
			cmMap = cm.nipy_spectral(jetMap)
			newMap = mcol.ListedColormap(cmMap)

			# Calculate boundaries of heatmap
			hw = int(self.TILE_WIDTH/2)
			if zoomed_extent:
				extent = (hw, max_x, max_y, hw)
			else:
				extent = (0, max_x+hw, max_y+hw, 0)
			
			# Heatmap
			self.ax.imshow(activation_map, extent=extent,
										   cmap=newMap,
										   alpha=0.6 if not self.interactive else 0.0,
										   interpolation='bicubic',
										   zorder=10)
		if save_dir:
			heatmap_loc = join(save_dir, f'{name}-heatmap.png')
			plt.savefig(heatmap_loc, bbox_inches='tight')
			log.complete(f"Heatmap saved to {heatmap_loc}", 1)
		if self.interactive:
			print()
			plt.show()