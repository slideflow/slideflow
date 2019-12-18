import os
import sys
import csv
import umap

import seaborn as sns
import numpy as np
from os.path import join

from slideflow.util import log
from scipy import stats
from statistics import median
from sklearn import metrics
from matplotlib import pyplot as plt

import slideflow.util as sfutil

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

def generate_roc(y_true, y_pred, data_dir, name='ROC'):
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
	plt.savefig(os.path.join(data_dir, f'{name}.png'))
	return roc_auc

def generate_scatter(y_true, y_pred, data_dir, name='_plot'):
	'''Generate and save scatter plots and calculate R2 statistic for each outcome variable.
	y_true and y_pred are both 2D arrays; the first dimension is each observation, the second dimension is each outcome variable.'''
	# Error checking
	if y_true.shape != y_pred.shape:
		log.error(f"Y_true (shape: {y_true.shape}) and y_pred (shape: {y_pred.shape}) must have the same shape to generate a scatter plot")
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

def generate_performance_metrics(model, dataset_with_slidenames, annotations, model_type, data_dir, label=None):
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
	log.empty("Generating predictions...", 1)
	label_end = "" if not label else f"_{label}"
	label_start = "" if not label else f"{label}_"
	y_true, y_pred, tile_to_slides = [], [], []
	for i, batch in enumerate(dataset_with_slidenames):
		sys.stdout.write(f"\r   - Working on batch {i}...")
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
		elif not np.array_equal(y_true_patient[patient], y_true[i]):
			log.error("Data integrity failure when generating ROCs; patient assigned to multiple slides with different outcomes", 1)
			sys.exit()

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
			log.info(f"Category {cat_index} accuracy: {cat_percent_acc:.1f}% ({num_correctly_predicted_in_category}/{num_tiles_in_category})")

		# Generate slide-level percent calls
		percent_calls_by_slide = get_average_by_slide(onehot_predictions, "percent_tiles_positive")

		# Generate slide-level ROC
		for i in range(num_cat):
			slide_y_pred = percent_calls_by_slide[:, i]
			slide_y_true = [y_true_slide[slide][i] for slide in unique_slides]
			auc = generate_roc(slide_y_true, slide_y_pred, data_dir, f'{label_start}slide_ROC{i}')
			slide_auc += [auc]
			log.info(f"Slide-level AUC (cat #{i}): {auc}", 1)

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