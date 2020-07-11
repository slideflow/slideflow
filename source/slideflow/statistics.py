import os
import sys
import csv
import umap
import pickle

import seaborn as sns
import numpy as np
import pandas as pd

import slideflow.util as sfutil

from os.path import join
from slideflow.util import log
from scipy import stats
from random import sample
from statistics import median
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib.widgets import LassoSelector

class TFRecordUMAP:

	def __init__(self, slides, tfrecords, cache=None):
		''' slides = self.slides_to_include '''
		self.slides = slides
		self.tfrecords = tfrecords
		self.cache = cache

		self.x = []
		self.y = []
		self.point_meta = []
		self.map_meta = {}

		# Try to load from cache
		if self.cache:
			if self.load_cache():
				return

	@classmethod
	def from_precalculated(cls, slides, x, y, meta, tfrecords=None, cache=None):
		obj = cls(slides, tfrecords)
		obj.x = x
		obj.y = y
		obj.point_meta = meta
		obj.cache = cache
		obj.save_cache()
		return obj

	@classmethod
	def from_activations(cls, activations, exclude_slides=None, prediction_filter=None, force_recalculate=False, use_centroid=False, cache=None):
		slides = activations.slides if not exclude_slides else [slide for slide in activations.slides if slide not in exclude_slides]
		tfrecords = activations.tfrecords if not exclude_slides else [tfr for tfr in activations.tfrecords if sfutil.path_to_name(tfr) not in exclude_slides]

		obj = cls(slides, tfrecords, cache=cache)
		obj.AV = activations
		if use_centroid:
			obj._calculate_from_centroid(prediction_filter=prediction_filter, force_recalculate=force_recalculate)
		else:
			obj._calculate_from_nodes(prediction_filter=prediction_filter, force_recalculate=force_recalculate)
		return obj

	def _calculate_from_nodes(self, prediction_filter=None, force_recalculate=False):
		if len(self.x) and len(self.y) and not force_recalculate:
			log.info("UMAP loaded from cache, will not recalculate", 1)

			# First, filter out slides not included in provided activations
			self.x = np.array([self.x[i] for i in range(len(self.x)) if self.point_meta[i]['slide'] in self.AV.slides])
			self.y = np.array([self.y[i] for i in range(len(self.y)) if self.point_meta[i]['slide'] in self.AV.slides])
			self.point_meta = np.array([self.point_meta[i] for i in range(len(self.point_meta)) if self.point_meta[i]['slide'] in self.AV.slides])
			
			# If UMAP already calculated, only update predictions if prediction filter is provided
			if prediction_filter:
				log.info("Updating UMAP predictions according to filter restrictions", 1)
				
				num_logits = len(self.AV.slide_logits_dict[self.slides[0]])

				for i in range(len(self.point_meta)):	
					slide = self.point_meta[i]['slide']
					tile_index = self.point_meta[i]['index']		
					logits = [self.AV.slide_logits_dict[slide][l][tile_index] for l in range(num_logits)]
					filtered_logits = [logits[l] for l in prediction_filter]
					prediction = logits.index(max(filtered_logits))
					self.point_meta[i]['prediction'] = prediction
			return

		# Calculate UMAP
		node_activations = []
		self.map_meta['nodes'] = self.AV.nodes
		log.empty("Calculating UMAP...", 1)
		for slide in self.slides:
			first_node = list(self.AV.slide_node_dict[slide].keys())[0]
			num_vals = len(self.AV.slide_node_dict[slide][first_node])
			num_logits = len(self.AV.slide_logits_dict[slide])
			for i in range(num_vals):
				node_activations += [[self.AV.slide_node_dict[slide][n][i] for n in self.AV.nodes]]
				logits = [self.AV.slide_logits_dict[slide][l][i] for l in range(num_logits)]
				# if prediction_filter is supplied, calculate prediction based on maximum value of allowed outcomes
				if prediction_filter:
					filtered_logits = [logits[l] for l in prediction_filter]
					prediction = logits.index(max(filtered_logits))
				else:
					prediction = logits.index(max(logits))

				self.point_meta += [{
					'slide': slide,
					'index': i,
					'prediction': prediction,
				}]

		coordinates = gen_umap(np.array(node_activations))
		self.x = np.array([c[0] for c in coordinates])
		self.y = np.array([c[1] for c in coordinates])
		self.save_cache()

	def _calculate_from_centroid(self, prediction_filter=None, force_recalculate=False):
		log.info("Calculating centroid indices for slide-level UMAP...", 1)
		optimal_slide_indices, centroid_activations = calculate_centroid(self.AV.slide_node_dict)
		
		# Restrict mosaic to only slides that had enough tiles to calculate an optimal index from centroid
		successful_slides = list(optimal_slide_indices.keys())
		num_warned = 0
		warn_threshold = 3
		for slide in self.AV.slides:
			print_func = print if num_warned < warn_threshold else None
			if slide not in successful_slides:
				log.warn(f"Unable to calculate centroid for slide {sfutil.green(slide)}; will not include in Mosaic", 1, print_func)
				num_warned += 1
		if num_warned >= warn_threshold:
			log.warn(f"...{num_warned} total warnings, see {sfutil.green(log.logfile)} for details", 1)

		if len(self.x) and len(self.y) and not force_recalculate:
			log.info("UMAP loaded from cache, will filter to include only provided tiles", 1)
			new_x = []
			new_y = []
			new_meta = []
			for i in range(len(self.point_meta)):
				slide = self.point_meta[i]['slide']
				if slide in optimal_slide_indices and self.point_meta[i]['index'] == optimal_slide_indices[slide]:
					new_x += [self.x[i]]
					new_y += [self.y[i]]
					
					if prediction_filter:
						num_logits = len(self.AV.slide_logits_dict[slide])
						tile_index = self.point_meta[i]['index']
						logits = [self.AV.slide_logits_dict[slide][l][tile_index] for l in range(num_logits)]
						filtered_logits = [logits[l] for l in prediction_filter]
						prediction = logits.index(max(filtered_logits))
						meta = {
							'slide': slide,
							'index': tile_index,
							'prediction': prediction,
						}
					else:
						meta = self.point_meta[i]
						
					new_meta += [meta]
			self.x = np.array(new_x)
			self.y = np.array(new_y)
			self.point_meta = np.array(new_meta)
		else:
			log.empty("Calculating UMAP...", 1)
			umap_input = []
			for slide in self.slides:
				umap_input += [centroid_activations[slide]]
				self.point_meta += [{
					'slide': slide,
					'index': optimal_slide_indices[slide],
					'prediction': 0
				}]

			coordinates = gen_umap(np.array(umap_input))
			self.x = np.array([c[0] for c in coordinates])
			self.y = np.array([c[1] for c in coordinates])
			self.save_cache()

	def cluster(self, n_clusters):
		activations = [[self.AV.slide_node_dict[pm['slide']][n][pm['index']] for n in self.AV.nodes] for pm in self.point_meta]
		log.info(f"Calculating K-means clustering (n={n_clusters})", 1)
		kmeans = KMeans(n_clusters=n_clusters).fit(activations)
		labels = kmeans.labels_
		for i, label in enumerate(labels):
			self.point_meta[i]['cluster'] = label

	def export_to_csv(self, filename):
		'''Exports calculated UMAP coordinates to csv.'''
		with open(filename, 'w') as outfile:
			csvwriter = csv.writer(outfile)
			header = ['slide', 'index', 'x', 'y']
			csvwriter.writerow(header)
			for index in range(len(self.point_meta)):
				x = self.x[index]
				y = self.y[index]
				meta = self.point_meta[index]
				slide = meta['slide']
				index = meta['index']
				row = [slide, index, x, y]
				csvwriter.writerow(row)

	def save_2d_plot(self, filename, slide_labels=None, slide_filter=None, show_tile_meta=None,
						outcome_labels=None, subsample=None, title=None, cmap=None, use_float=False):

		# Warn the user if both slide_labels (manual category labeling) and show_tile_meta are supplied
		if slide_labels and show_tile_meta:
			log.warn("Both `slide_labels` and `show_tile_meta` provided; will ignore the former and show only tile metadata", 2)
		
		# Filter out slides
		if slide_labels and not slide_filter:
			slide_filter = list(slide_labels.keys())
		if slide_filter:
			meta = [pm for pm in self.point_meta if pm['slide'] in slide_filter]
			x = np.array([self.x[xi] for xi in range(len(self.x)) if self.point_meta[xi]['slide'] in slide_filter])
			y = np.array([self.y[yi] for yi in range(len(self.y)) if self.point_meta[yi]['slide'] in slide_filter])
		else:
			meta, x, y = self.point_meta, self.x, self.y

		# Establish category labeling
		if show_tile_meta:
			if outcome_labels:
				try:
					categories = np.array([outcome_labels[m[show_tile_meta]] for m in meta])
				except KeyError:
					# Try by converting metadata to string
					categories = np.array([outcome_labels[str(m[show_tile_meta])] for m in meta])
			else:
				categories = np.array([m[show_tile_meta] for m in meta])
		elif slide_labels:
			categories = np.array([slide_labels[m['slide']] for m in meta])		
		else:
			categories = np.array(["None" for m in meta])

		# Subsampling
		if subsample:
			ri = sample(range(len(x)), min(len(x), subsample))
		else:
			ri = list(range(len(x)))
		x = x[ri]
		y = y[ri]

		# Prepare pandas dataframe
		df = pd.DataFrame()
		df['umap_x'] = x
		df['umap_y'] = y
		df['category'] = categories[ri] if use_float else pd.Series(categories[ri], dtype='category')

		# Prepare color palette
		if use_float:
			cmap = None
			palette = None
		else:
			unique_categories = list(set(categories[ri]))
			unique_categories.sort()
			seaborn_palette = sns.color_palette("Paired", len(unique_categories)) if len(unique_categories) <= 12 else sns.color_palette('hls', len(unique_categories))
			palette = {unique_categories[i]:seaborn_palette[i] for i in range(len(unique_categories))}

		# Make plot
		plt.clf()
		umap_2d = sns.scatterplot(x=x, y=y, data=df, hue='category', palette=cmap if cmap else palette)
		umap_2d.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
		umap_figure = umap_2d.get_figure()
		if title: umap_figure.axes[0].set_title(title)
		umap_figure.savefig(filename, bbox_inches='tight')
		log.complete(f"Saved 2D UMAP to {sfutil.green(filename)}", 1)

		def onselect(verts):
			print(verts)

		lasso = LassoSelector(plt.gca(), onselect)

	def save_3d_plot(self, z, filename, title="UMAP", subsample=None):
		'''Saves a plot of a 3D umap, with the 3rd dimension representing values provided by argument "z" '''

		# Subsampling
		if subsample:
			ri = sample(range(len(self.x)), min(len(self.x), subsample))
		else:
			ri = list(range(len(self.x)))

		x = self.x[ri]
		y = self.y[ri]
		z = z[ri]

		# Plot tiles on a 3D coordinate space with 2 coordinates from UMAP & 3rd from the value of the excluded node
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter(x, y, z, c=z,
							cmap='viridis',
							linewidth=0.5,
							edgecolor="black")
		ax.set_title(title)
		log.info(f"Saving 3D UMAP to {sfutil.green(filename)}...", 1)
		plt.savefig(filename, bbox_inches='tight')

	def get_tiles_in_area(self, x_lower=-999, x_upper=999, y_lower=-999, y_upper=999):
		'''Returns dictionary of slide names mapping to tile indices, for tiles that fall within the specified location on the umap.'''
		# Find tiles that meet UMAP location criteria
		filtered_tiles = {}
		num_selected = 0
		for i in range(len(self.point_meta)):
			if (x_lower < self.x[i] < x_upper) and (y_lower < self.y[i] < y_upper):
				slide = self.point_meta[i]['slide']
				tile_index = self.point_meta[i]['index']
				if slide not in filtered_tiles:
					filtered_tiles.update({slide: [tile_index]})
				else:
					filtered_tiles[slide] += [tile_index]
				num_selected += 1
		log.info(f"Selected {num_selected} tiles by filter criteria.", 1)
		return filtered_tiles

	def save_cache(self):
		if self.cache:
			try:
				with open(self.cache, 'wb') as cache_file:
					pickle.dump([self.x, self.y, self.point_meta, self.map_meta], cache_file)
					log.info(f"Wrote UMAP cache to {sfutil.green(self.cache)}", 1)
			except:
				log.info(f"Error attempting to write UMAP cache to {sfutil.green(self.cache)}", 1)

	def load_cache(self):
		try:
			with open(self.cache, 'rb') as cache_file:
				self.x, self.y, self.point_meta, self.map_meta = pickle.load(cache_file)
				log.info(f"Loaded UMAP cache from {sfutil.green(self.cache)}", 1)
				return True
		except FileNotFoundError:
			log.info(f"No UMAP cache found at {sfutil.green(self.cache)}", 1)
		return False

def get_centroid_index(input_array):
	km = KMeans(n_clusters=1).fit(input_array)
	closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, input_array)
	return closest[0]

def calculate_centroid(slide_node_dict):
	optimal_indices = {}
	centroid_activations = {}
	nodes = list(slide_node_dict[list(slide_node_dict.keys())[0]].keys())
	for slide in slide_node_dict:
		slide_nodes = slide_node_dict[slide]
		# Reorganize "slide_nodes" into an array of node activations for each tile
		# Final size of array should be (num_nodes, num_tiles_in_slide) 
		activations = [[slide_nodes[n][i] for n in nodes] for i in range(len(slide_nodes[0]))]
		km = KMeans(n_clusters=1).fit(activations)
		closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, activations)
		closest_index = closest[0]
		closest_activations = activations[closest_index]

		optimal_indices.update({slide: closest_index})
		centroid_activations.update({slide: closest_activations})
	return optimal_indices, centroid_activations

def normalize_layout(layout, min_percentile=1, max_percentile=99, relative_margin=0.1):
	"""Removes outliers and scales layout to between [0,1]."""

	# compute percentiles
	mins = np.percentile(layout, min_percentile, axis=(0))
	maxs = np.percentile(layout, max_percentile, axis=(0))

	# add margins
	mins -= relative_margin * (maxs - mins)
	maxs += relative_margin * (maxs - mins)

	# `clip` broadcasts, `[None]`s added only for readability
	clipped = np.clip(layout, mins, maxs)

	# embed within [0,1] along both axes
	clipped -= clipped.min(axis=0)
	clipped /= clipped.max(axis=0)

	return clipped

def gen_umap(array):
	'''Generates and returns a umap from a given array.'''
	try:
		layout = umap.UMAP(n_components=2, verbose=True, n_neighbors=20, min_dist=0.01, metric="cosine").fit_transform(array)
	except ValueError:
		log.error("Error performing UMAP. Please make sure you are supplying a non-empty TFRecord array and that the TFRecords are not empty.")
		sys.exit()
	return normalize_layout(layout)

def generate_histogram(y_true, y_pred, data_dir, name='histogram'):
	'''Generates histogram of y_pred, labeled by y_true'''
	cat_false = [y_pred[i] for i in range(len(y_pred)) if y_true[i] == 0]
	cat_true = [y_pred[i] for i in range(len(y_pred)) if y_true[i] == 1]

	plt.clf()
	plt.title('Tile-level Predictions')
	try:
		sns.distplot( cat_false , color="skyblue", label="Negative")
		sns.distplot( cat_true , color="red", label="Positive")
	except np.linalg.LinAlgError:
		log.warn("Unable to generate histogram, insufficient data", 1)
	plt.legend()
	plt.savefig(os.path.join(data_dir, f'{name}.png'))

def generate_roc(y_true, y_pred, save_dir=None, name='ROC'):
	'''Generates and saves an ROC with a given set of y_true, y_pred values.'''
	# Statistics
	fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
	roc_auc = metrics.auc(fpr, tpr)
	# Calculate optimal cutoff via maximizing Youden's J statistic (sens+spec-1, or TPR - FPR)
	try:
		optimal_threshold = threshold[list(zip(tpr,fpr)).index(max(zip(tpr,fpr), key=lambda x: x[0]-x[1]))]
	except:
		optimal_threshold = -1

	# Plot
	if save_dir:
		plt.clf()
		plt.title('ROC Curve')
		plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
		plt.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.ylabel('TPR')
		plt.xlabel('FPR')
		plt.savefig(os.path.join(save_dir, f'{name}.png'))
	return roc_auc, optimal_threshold

# The below function is deprecated and will be removed in the next version
'''def generate_combined_roc(y_true, y_pred, save_dir, labels, name='ROC'):
	# Generates and saves overlapping ROCs with a given combination of y_true and y_pred.
	# Plot
	plt.clf()
	plt.title(name)
	colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

	rocs = []
	for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
		fpr, tpr, threshold = metrics.roc_curve(yt, yp)
		roc_auc = metrics.auc(fpr, tpr)
		rocs += [roc_auc]
		plt.plot(fpr, tpr, colors[i % len(colors)], label = labels[i] + f' (AUC: {roc_auc:.2f})')	

	# Finish plot
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('TPR')
	plt.xlabel('FPR')
	
	plt.savefig(os.path.join(save_dir, f'{name}.png'))
	return rocs'''

def read_predictions(predictions_file, level):
	predictions = {}
	y_pred_label = "percent_tiles_positive" if level in ("patient", "slide") else "y_pred"
	with open(predictions_file, 'r') as csvfile:
		reader = csv.reader(csvfile)
		header = next(reader)
		prediction_labels = [h.split('y_true')[-1] for h in header if "y_true" in h]
		for label in prediction_labels:
			predictions.update({label: {
				'y_true': [],
				'y_pred': []
			}})
		for row in reader:
			for label in prediction_labels:
				yti = header.index(f'y_true{label}')
				ypi = header.index(f'{y_pred_label}{label}')
				predictions[label]['y_true'] += [int(row[yti])]
				predictions[label]['y_pred'] += [float(row[ypi])]
	return predictions

def generate_scatter(y_true, y_pred, data_dir, name='_plot'):
	'''Generate and save scatter plots and calculate R2 statistic for each outcome variable.
	y_true and y_pred are both 2D arrays; the first dimension is each observation, the second dimension is each outcome variable.'''
	# Error checking
	if y_true.shape != y_pred.shape:
		log.error(f"Y_true (shape: {y_true.shape}) and y_pred (shape: {y_pred.shape}) must have the same shape to generate a scatter plot", 1)
		return
	if y_true.shape[0] < 2:
		log.error(f"Must have more than one observation to generate a scatter plot with R2 statistics.", 1)
		return
	
	# Perform scatter for each outcome variable
	r_squared = []
	for i in range(y_true.shape[1]):

		# Statistics
		slope, intercept, r_value, p_value, std_err = stats.linregress(y_true[:,i], y_pred[:,i])
		r_squared += [r_value ** 2]

		def r2(x, y):
			return stats.pearsonr(x, y)[0] ** 2

		# Plot
		p = sns.jointplot(y_true[:,i], y_pred[:,i], kind="reg", stat_func=r2)
		p.set_axis_labels('y_true', 'y_pred')
		plt.savefig(os.path.join(data_dir, f'Scatter{name}-{i}.png'))

	return r_squared

def generate_basic_performance_metrics(y_true, y_pred):
	'''Generates basic performance metrics, including sensitivity, specificity, and accuracy.'''
	assert(len(y_true) == len(y_pred))
	assert([y in [0,1] for y in y_true])
	assert([y in [0,1] for y in y_pred])

	TP = 0 # True positive
	TN = 0 # True negative
	FP = 0 # False positive
	FN = 0 # False negative

	for i, yt in enumerate(y_true):
		yp = y_pred[i]
		if yt == 1 and yp == 1:
			TP += 1
		elif yt == 1 and yp == 0:
			FN += 1
		elif yt == 0 and yp == 1:
			FP += 1
		elif yt == 0 and yp == 0:
			TN += 1

	accuracy = (TP + TN) / (TP + TN + FP + FN)
	sensitivity = TP / (TP + FN)
	specificity = TN / (TN + FP)
	
	return accuracy, sensitivity, specificity

def generate_performance_metrics(model, dataset_with_slidenames, annotations, model_type, data_dir, label=None, manifest=None, min_tiles_per_slide=0):
	'''Evaluate performance of a given model on a given TFRecord dataset, 
	generating a variety of statistical outcomes and graphs.

	Args:
		model						Keras model to evaluate
		dataset_with_slidenames		TFRecord dataset which include three items: raw image data, labels, and slide names.
		annotations					dictionary mapping slidenames to patients (TCGA.patient) and outcomes (outcome)
		model_type					'linear' or 'categorical'
		data_dir					directory in which to save performance metrics and graphs
		label						optional label with which to annotation saved files and graphs
	'''
	
	# Get predictions and performance metrics
	sys.stdout.write("\rGenerating predictions...")
	label_end = "" if not label else f"_{label}"
	label_start = "" if not label else f"{label}_"
	y_true, y_pred, tile_to_slides = [], [], []
	for i, batch in enumerate(dataset_with_slidenames):
		sys.stdout.write(f"\rGenerating predictions (batch {i})...")
		sys.stdout.flush()
		tile_to_slides += [slide_bytes.decode('utf-8') for slide_bytes in batch[2].numpy()]
		y_true += [batch[1].numpy()]
		y_pred += [model.predict_on_batch(batch[0])]
	patients = list(set([annotations[slide][sfutil.TCGA.patient] for slide in tile_to_slides]))
	sys.stdout.write("\r\033[K")
	sys.stdout.flush()
	y_pred = np.concatenate(y_pred)
	y_true = np.concatenate(y_true)
	num_true_outcome_categories = max(y_true)+1

	if num_true_outcome_categories != len(y_pred[0]):
		log.warn(f"Model predictions have different number of outcome categories ({len(y_pred[0])}) than provided annotations ({num_true_outcome_categories})!", 1)
	num_tiles = len(tile_to_slides)
	unique_slides = list(set(tile_to_slides))

	# Filter out slides not meeting minimum tile number criteria, if specified
	slides_to_filter = []
	num_total_slides = len(unique_slides)
	for tfrecord in manifest:
		tfrecord_name = sfutil.path_to_name(tfrecord)
		num_tiles_tfrecord = manifest[tfrecord]['total']
		if num_tiles_tfrecord < min_tiles_per_slide:
			log.info(f"Filtering out {tfrecord_name}: {num_tiles_tfrecord} tiles", 2)
			slides_to_filter += [tfrecord_name]
	unique_slides = [us for us in unique_slides if us not in slides_to_filter]
	log.info(f"Filtered out {num_total_slides - len(unique_slides)} of {num_total_slides} slides in evaluation set (minimum tiles per slide: {min_tiles_per_slide})", 1)

	tile_auc = []
	slide_auc = []
	patient_auc = []
	r_squared = None

	if model_type == 'categorical':
		# Convert y_true to one_hot encoding
		num_cat = max(num_true_outcome_categories, len(y_pred[0]))
		def to_onehot(val):
			onehot = [0] * num_cat
			onehot[val] = 1
			return onehot
		y_true = np.array([to_onehot(i) for i in y_true])

	if model_type == 'linear':
		#y_true = np.array([[i] for i in y_true])
		num_cat = len(y_pred[0])
		r_squared = generate_scatter(y_true, y_pred, data_dir, label_end)

	# Create dictionary mapping slides to one_hot category encoding
	y_true_slide = {}
	y_true_patient = {}
	patient_error = False
	for i in range(len(tile_to_slides)):
		slidename = tile_to_slides[i]
		patient = annotations[slidename][sfutil.TCGA.patient]
		if slidename not in y_true_slide:
			y_true_slide.update({slidename: y_true[i]})
		# Now check for data integrity problems (slide assigned to multiple different outcomes)
		elif not np.array_equal(y_true_slide[slidename], y_true[i]):
			log.error("Data integrity failure when generating ROCs; slide assigned to multiple different one-hot outcomes", 1)
			sys.exit()
		if patient not in y_true_patient:
			y_true_patient.update({patient: y_true[i]})
		elif not patient_error and not np.array_equal(y_true_patient[patient], y_true[i]):
			log.error("Data integrity failure when generating ROCs; patient assigned to multiple slides with different outcomes", 1)
			patient_error = True

	def get_average_by_slide(prediction_array, prediction_label):
		'''For a given tile-level prediction array, calculate percent predictions in each outcome by patient and save predictions to CSV'''
		avg_by_slide = []
		save_path = join(data_dir, f"slide_predictions{label_end}.csv")
		cat_index_warn = []
		for slide in unique_slides:
			percent_predictions = []
			for cat_index in range(num_cat):
				try:
					sum_of_outcome = sum([ prediction_array[i][cat_index] for i in range (num_tiles) if tile_to_slides[i] == slide ])
					num_total_tiles = len([i for i in range(len(tile_to_slides)) if tile_to_slides[i] == slide])
					percent_predictions += [sum_of_outcome / num_total_tiles] 
				except IndexError:
					if cat_index not in cat_index_warn:
						log.warn(f"Unable to generate slide-level stats for category index {cat_index}", 1)
						cat_index_warn += [cat_index]
			avg_by_slide += [percent_predictions]
		avg_by_slide = np.array(avg_by_slide)
		with open(save_path, 'w') as outfile:
			writer = csv.writer(outfile)
			header = ['slide'] + [f"y_true{i}" for i in range(num_cat)] + [f"{prediction_label}{j}" for j in range(num_cat)]
			writer.writerow(header)
			for i, slide in enumerate(unique_slides):
				row = np.concatenate([ [slide], y_true_slide[slide], avg_by_slide[i] ])
				writer.writerow(row)
		return avg_by_slide

	def get_average_by_patient(prediction_array, prediction_label):
		'''For a given tile-level prediction array, calculate percent predictions in each outcome by patient and save predictions to CSV'''
		avg_by_patient = []
		save_path = join(data_dir, f"patient_predictions{label_end}.csv")
		cat_index_warn = []
		for patient in patients:
			percent_predictions = []
			for cat_index in range(num_cat):
				try:
					sum_of_outcome = sum([ prediction_array[i][cat_index] for i in range (num_tiles) if annotations[tile_to_slides[i]][sfutil.TCGA.patient] == patient ])
					num_total_tiles = len([i for i in range(len(tile_to_slides)) if annotations[tile_to_slides[i]][sfutil.TCGA.patient] == patient])
					percent_predictions += [sum_of_outcome / num_total_tiles]
				except IndexError:
					if cat_index not in cat_index_warn:
						log.warn(f"Unable to generate patient-level stats for category index {cat_index}", 1)
						cat_index_warn += [cat_index]
			avg_by_patient += [percent_predictions]
		avg_by_patient = np.array(avg_by_patient)
		with open(save_path, 'w') as outfile:
			writer = csv.writer(outfile)
			header = ['patient'] + [f"y_true{i}" for i in range(num_cat)] + [f"{prediction_label}{j}" for j in range(num_cat)]
			writer.writerow(header)
			for i, patient in enumerate(patients):
				row = np.concatenate([ [patient], y_true_patient[patient], avg_by_patient[i] ])
				writer.writerow(row)
		return avg_by_patient
	
	if model_type == 'categorical':
		# Generate tile-level ROC
		for i in range(num_cat):
			try:
				auc, optimal_threshold = generate_roc(y_true[:, i], y_pred[:, i], data_dir, f'{label_start}tile_ROC{i}')
				generate_histogram(y_true[:, i], y_pred[:, i], data_dir, f'{label_start}tile_histogram{i}')
				tile_auc += [auc]
				log.info(f"Tile-level AUC (cat #{i}): {auc} (opt. threshold: {optimal_threshold})", 1)
			except IndexError:
				log.warn(f"Unable to generate tile-level stats for outcome {i}", 1)

		# Convert predictions to one-hot encoding
		onehot_predictions = []
		for x in range(len(y_pred)):
			predictions = y_pred[x]
			onehot_predictions += [[1 if pred == max(predictions) else 0 for pred in predictions]]
		
		# Compare one-hot predictions to one-hot y_true for category-level accuracy
		for cat_index in range(num_cat):
			try:
				num_tiles_in_category = sum([yt[cat_index] for yt in y_true])
				num_correctly_predicted_in_category = sum([yp[cat_index] for i, yp in enumerate(onehot_predictions) if y_true[i][cat_index]])
				category_accuracy = num_correctly_predicted_in_category / num_tiles_in_category
				cat_percent_acc = category_accuracy * 100
				log.info(f"Category {cat_index} accuracy: {cat_percent_acc:.1f}% ({num_correctly_predicted_in_category}/{num_tiles_in_category})", 1)
			except IndexError:
				log.warn(f"Unable to generate category-level accuracy stats for category index {cat_index}", 1)

		# Generate slide-level percent calls
		percent_calls_by_slide = get_average_by_slide(onehot_predictions, "percent_tiles_positive")

		# Generate slide-level ROC
		for i in range(num_cat):
			try:
				slide_y_pred = percent_calls_by_slide[:, i]
				slide_y_true = [y_true_slide[slide][i] for slide in unique_slides]
				auc, optimal_threshold = generate_roc(slide_y_true, slide_y_pred, data_dir, f'{label_start}slide_ROC{i}')
				slide_auc += [auc]
				log.info(f"Slide-level AUC (cat #{i}): {auc} (opt. threshold: {optimal_threshold})", 1)
			except IndexError:
				log.warn(f"Unable to generate slide-level stats for outcome {i}", 1)

		if not patient_error:
			# Generate patient-level percent calls
			percent_calls_by_patient = get_average_by_patient(onehot_predictions, "percent_tiles_positive")

			# Generate patient-level ROC
			for i in range(num_cat):
				try:
					patient_y_pred = percent_calls_by_patient[:, i]
					patient_y_true = [y_true_patient[patient][i] for patient in patients]
					auc, optimal_threshold = generate_roc(patient_y_true, patient_y_pred, data_dir, f'{label_start}patient_ROC{i}')
					patient_auc += [auc]
					log.info(f"Patient-level AUC (cat #{i}): {auc} (opt. threshold: {optimal_threshold})", 1)
				except IndexError:
					log.warn(f"Unable to generate patient-level stats for outcome {i}", 1)

	if model_type == 'linear':
		# Generate and save slide-level averages of each outcome
		averages_by_slide = get_average_by_slide(y_pred, "average")
		y_true_by_slide = np.array([y_true_slide[slide] for slide in unique_slides])
		r_squared_slide = generate_scatter(y_true_by_slide, averages_by_slide, data_dir, label_end+"_by_slide")			

		if not patient_error:
			# Generate and save patient-level averages of each outcome
			averages_by_patient = get_average_by_patient(y_pred, "average")
			y_true_by_patient = np.array([y_true_patient[patient] for patient in patients])
			r_squared_patient = generate_scatter(y_true_by_patient, averages_by_patient, data_dir, label_end+"_by_patient")			

	# Save tile-level predictions
	tile_csv_dir = os.path.join(data_dir, f"tile_predictions{label_end}.csv")
	with open(tile_csv_dir, 'w') as outfile:
		writer = csv.writer(outfile)
		header = ['slide'] + [f"y_true{i}" for i in range(y_true.shape[1])] + [f"y_pred{j}" for j in range(len(y_pred[0]))]
		writer.writerow(header)
		for i in range(len(y_true)):
			y_true_str_list = [str(yti) for yti in y_true[i]]
			y_pred_str_list = [str(ypi) for ypi in y_pred[i]]
			row = np.concatenate([[tile_to_slides[i]], y_true_str_list, y_pred_str_list])
			writer.writerow(row)

	log.complete(f"Predictions saved to {sfutil.green(data_dir)}", 1)
	return tile_auc, slide_auc, patient_auc, r_squared