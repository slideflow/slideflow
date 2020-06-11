import os
import sys
import csv
import umap

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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

class TFRecordUMAP:
	x = []
	y = []
	point_meta = []
	map_meta = {}

	def __init__(self, tfrecords, slides, cache=None):
		''' slides = self.slides_to_include '''
		self.tfrecords = tfrecords
		self.slides = slides
		self.cache = cache

		# Try to load from cache
		if self.cache:
			if self.load_cache():
				return
	
	def calculate_from_nodes(self, slide_node_dict, slide_logits_dict, nodes, exclude_slides=None):
		self.map_meta['nodes'] = nodes

		# Calculate UMAP
		node_activations = []
		log.empty("Calculating UMAP...", 1)
		for slide in self.slides:
			if slide in exclude_slides: continue
			first_node = list(slide_node_dict[slide].keys())[0]
			num_vals = len(slide_node_dict[slide][first_node])
			num_logits = len(slide_logits_dict[slide])
			for i in range(num_vals):
				node_activations += [[slide_node_dict[slide][n][i] for n in nodes]]
				logits = [slide_logits_dict[slide][l][i] for l in range(num_logits)]
				prediction = logits.index(max(logits))

				self.point_meta += [{
					'slide': slide,
					'index': i,
					'prediction': prediction,
				}]

		coordinates = gen_umap(np.array(node_activations))
		self.x = np.array([c[0] for c in coordinates])
		self.y = np.array([c[1] for c in coordinates])

	def load_precalculated(self, x, y, meta):
		self.x = x
		self.y = y
		self.point_meta = meta
		self.save_cache()

	def save_2d_plot(self, filename, slide_category_dict=None, show_prediction=False, outcome_labels=None,
					subsample=None, title=None):
		# Prepare plotting categories
		if slide_category_dict:
			categories = np.array([slide_category_dict[m['slide']] for m in self.point_meta])
		elif show_prediction:
			if outcome_labels:
				categories = np.array([outcome_labels[m['prediction']] for m in self.point_meta])
			else:
				categories = np.array([m['prediction'] for m in self.point_meta])
		else:
			categories = np.array(["None" for m in self.point_meta])

		# Subsampling
		if subsample:
			ri = sample(range(len(self.x)), min(len(self.x), subsample))
		else:
			ri = list(range(len(self.x)))
		x = self.x[ri]
		y = self.y[ri]

		unique_categories = list(set(categories[ri]))	

		# Prepare pandas dataframe
		df = pd.DataFrame()
		df['umap_x'] = x
		df['umap_y'] = y
		df['category'] = pd.Series(categories[ri], dtype='category')

		# Make plot
		plt.clf()
		fig = plt.figure()
		umap_2d = sns.scatterplot(x=x, y=y, data=df, hue='category', palette=sns.color_palette('Set1', len(unique_categories)))
		umap_2d.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
		log.info(f"Saving 2D UMAP to {sfutil.green(filename)}...", 1)
		umap_figure = umap_2d.get_figure()
		if title: umap_figure.axes[0].set_title(title)
		umap_figure.savefig(filename, bbox_inches='tight')

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
		except:
			log.info(f"No UMAP cache found at {sfutil.green(self.cache)}", 1)
		return False

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

def generate_roc(y_true, y_pred, save_dir, name='ROC'):
	'''Generates and saves an ROC with a given set of y_true, y_pred values.'''
	# Statistics
	fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
	roc_auc = metrics.auc(fpr, tpr)

	# Plot
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
	return roc_auc

def generate_combined_roc(y_true, y_pred, save_dir, labels, name='ROC'):
	'''Generates and saves overlapping ROCs with a given combination of y_true and y_pred.'''
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
	return rocs

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
		num_cat = len(y_pred[0])
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
		'''For a given tile-level prediction array, calculate averages in each outcome by patient and save predictions to CSV'''
		avg_by_slide = []
		save_path = join(data_dir, f"slide_predictions{label_end}.csv")
		for slide in unique_slides:
			averages = []
			for cat_index in range(num_cat):
				sum_of_outcome = sum([ prediction_array[i][cat_index] for i in range (num_tiles) if tile_to_slides[i] == slide ])
				num_total_tiles = len([i for i in range(len(tile_to_slides)) if tile_to_slides[i] == slide])
				averages += [sum_of_outcome / num_total_tiles] 
			avg_by_slide += [averages]
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
		'''For a given tile-level prediction array, calculate averages in each outcome by patient and save predictions to CSV'''
		avg_by_patient = []
		save_path = join(data_dir, f"patient_predictions{label_end}.csv")
		for patient in patients:
			averages = []
			for cat_index in range(num_cat):
				sum_of_outcome = sum([ prediction_array[i][cat_index] for i in range (num_tiles) if annotations[tile_to_slides[i]][sfutil.TCGA.patient] == patient ])
				num_total_tiles = len([i for i in range(len(tile_to_slides)) if annotations[tile_to_slides[i]][sfutil.TCGA.patient] == patient])
				averages += [sum_of_outcome / num_total_tiles]
			avg_by_patient += [averages]
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
			auc = generate_roc(y_true[:, i], y_pred[:, i], data_dir, f'{label_start}tile_ROC{i}')
			generate_histogram(y_true[:, i], y_pred[:, i], data_dir, f'{label_start}tile_histogram{i}')
			tile_auc += [auc]
			log.info(f"Tile-level AUC (cat #{i}): {auc}", 1)

		# Convert predictions to one-hot encoding
		onehot_predictions = []
		for x in range(len(y_pred)):
			predictions = y_pred[x]
			onehot_predictions += [[1 if pred == max(predictions) else 0 for pred in predictions]]
		
		# Compare one-hot predictions to one-hot y_true for category-level accuracy
		for cat_index in range(num_cat):
			num_tiles_in_category = sum([yt[cat_index] for yt in y_true])
			num_correctly_predicted_in_category = sum([yp[cat_index] for i, yp in enumerate(onehot_predictions) if y_true[i][cat_index]])
			category_accuracy = num_correctly_predicted_in_category / num_tiles_in_category
			cat_percent_acc = category_accuracy * 100
			log.info(f"Category {cat_index} accuracy: {cat_percent_acc:.1f}% ({num_correctly_predicted_in_category}/{num_tiles_in_category})", 1)

		# Generate slide-level percent calls
		percent_calls_by_slide = get_average_by_slide(onehot_predictions, "percent_tiles_positive")

		# Generate slide-level ROC
		for i in range(num_cat):
			slide_y_pred = percent_calls_by_slide[:, i]
			slide_y_true = [y_true_slide[slide][i] for slide in unique_slides]
			auc = generate_roc(slide_y_true, slide_y_pred, data_dir, f'{label_start}slide_ROC{i}')
			slide_auc += [auc]
			log.info(f"Slide-level AUC (cat #{i}): {auc}", 1)

		if not patient_error:
			# Generate patient-level percent calls
			percent_calls_by_patient = get_average_by_patient(onehot_predictions, "percent_tiles_positive")

			# Generate patient-level ROC
			for i in range(num_cat):
				patient_y_pred = percent_calls_by_patient[:, i]
				patient_y_true = [y_true_patient[patient][i] for patient in patients]
				auc = generate_roc(patient_y_true, patient_y_pred, data_dir, f'{label_start}patient_ROC{i}')
				patient_auc += [auc]
				log.info(f"Patient-level AUC (cat #{i}): {auc}", 1)

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
		header = ['slide'] + [f"y_true{i}" for i in range(num_cat)] + [f"y_pred{j}" for j in range(num_cat)]
		writer.writerow(header)
		for i in range(len(y_true)):
			y_true_str_list = [str(yti) for yti in y_true[i]]
			y_pred_str_list = [str(ypi) for ypi in y_pred[i]]
			row = np.concatenate([[tile_to_slides[i]], y_true_str_list, y_pred_str_list])
			writer.writerow(row)

	log.complete(f"Predictions saved to {sfutil.green(data_dir)}", 1)
	return tile_auc, slide_auc, patient_auc, r_squared