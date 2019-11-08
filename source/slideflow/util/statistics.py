import os
import sys
import csv

import seaborn as sns
import numpy as np

from slideflow.util import log
from scipy import stats
from statistics import median
from sklearn import metrics
from matplotlib import pyplot as plt

import slideflow.util as sfutil

DATA_DIR = ""

def generate_histogram(y_true, y_pred, data_dir, name='histogram'):
	'''Generates histogram of y_pred, labeled by y_true'''
	cat_false = [y_pred[i] for i in range(len(y_pred)) if y_true[i] == 0]
	cat_true = [y_pred[i] for i in range(len(y_pred)) if y_true[i] == 1]

	plt.clf()
	plt.title('Tile-level Predictions')
	sns.distplot( cat_false , color="skyblue", label="Negative")
	sns.distplot( cat_true , color="red", label="Positive")
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
	log.info("Generating predictions...", 1)
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

	tile_auc = []
	slide_auc = []
	patient_auc = []
	r_squared = None

	if model_type == 'linear':
		#y_true = np.array([[i] for i in y_true])
		num_cat = len(y_pred[0])
		r_squared = generate_scatter(y_true, y_pred, data_dir, label_end)
			
	if model_type == 'categorical':
		# Convert y_true to one_hot encoding
		num_cat = len(y_pred[0])
		def to_onehot(val):
			onehot = [0] * num_cat
			onehot[val] = 1
			return onehot
		y_true = np.array([to_onehot(i) for i in y_true])

		# Create dictionary mapping slides to one_hot category encoding
		slide_onehot = {}
		patient_onehot = {}
		for i in range(len(tile_to_slides)):
			slidename = tile_to_slides[i]
			patient = annotations[slidename][sfutil.TCGA.patient]
			if slidename not in slide_onehot:
				slide_onehot.update({slidename: y_true[i]})
			# Now check for data integrity problems (slide assigned to multiple different outcomes)
			elif not np.array_equal(slide_onehot[slidename], y_true[i]):
				log.error("Data integrity failure when generating ROCs; slide assigned to multiple different one-hot outcomes", 1)
				sys.exit()
			if patient not in patient_onehot:
				patient_onehot.update({patient: y_true[i]})
			elif not np.array_equal(patient_onehot[patient], y_true[i]):
				log.error("Data integrity failure when generating ROCs; patient assigned to multiple slides with different outcomes", 1)
				sys.exit()

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

		# Generate slide-level ROC
		unique_slides = list(set(tile_to_slides))
		percent_calls_by_slide = []
		for slide in unique_slides:
			percentages = []
			for cat_index in range(num_cat):
				num_predictions = sum([ onehot_predictions[i][cat_index] for i in range(num_tiles) if tile_to_slides[i] == slide ])
				num_total_tiles = len([i for i in range(len(tile_to_slides)) if tile_to_slides[i] == slide])
				percentages += [num_predictions / num_total_tiles] 
			percent_calls_by_slide += [percentages]
		percent_calls_by_slide = np.array(percent_calls_by_slide)

		for i in range(num_cat):
			slide_y_pred = percent_calls_by_slide[:, i]
			slide_y_true = [slide_onehot[slide][i] for slide in unique_slides]
			auc = generate_roc(slide_y_true, slide_y_pred, data_dir, f'{label_start}slide_ROC{i}')
			slide_auc += [auc]
			log.info(f"Slide-level AUC (cat #{i}): {auc}", 1)

		# Save slide-level predictions
		slide_csv_dir = os.path.join(data_dir, f"slide_predictions{label_end}.csv")
		with open(slide_csv_dir, 'w') as outfile:
			writer = csv.writer(outfile)
			header = ['slide'] + [f"y_true{i}" for i in range(num_cat)] + [f"percent_tiles_positive{j}" for j in range(num_cat)]
			writer.writerow(header)
			for i, slide in enumerate(unique_slides):
				slide_y_true_onehot = slide_onehot[slide]
				row = np.concatenate([ [slide], slide_y_true_onehot, percent_calls_by_slide[i] ])
				writer.writerow(row)

		# Generate patient-level ROC
		percent_calls_by_patient = []
		for patient in patients:
			percentages = []
			for cat_index in range(num_cat):
				num_predictions = sum([ onehot_predictions[i][cat_index] for i in range(num_tiles) if annotations[tile_to_slides[i]][sfutil.TCGA.patient] == patient ])
				num_total_tiles = len([i for i in range(len(tile_to_slides)) if annotations[tile_to_slides[i]][sfutil.TCGA.patient] == patient])
				percentages += [num_predictions / num_total_tiles]
			percent_calls_by_patient += [percentages]
		percent_calls_by_patient = np.array(percent_calls_by_patient)

		for i in range(num_cat):
			patient_y_pred = percent_calls_by_patient[:, i]
			patient_y_true = [patient_onehot[patient][i] for patient in patients]
			auc = generate_roc(patient_y_true, patient_y_pred, data_dir, f'{label_start}patient_ROC{i}')
			patient_auc += [auc]
			log.info(f"Patient-level AUC (cat #{i}): {auc}", 1)

		# Save patient-level predictions
		patient_csv_dir = os.path.join(data_dir, f"patient_predictions{label_end}.csv")
		with open(patient_csv_dir, 'w') as outfile:
			writer = csv.writer(outfile)
			header = ['patient'] + [f"y_true{i}" for i in range(num_cat)] + [f"percent_tiles_positive{j}" for j in range(num_cat)]
			writer.writerow(header)
			for i, patient in enumerate(patients):
				patient_y_true_onehot = patient_onehot[patient]
				row = np.concatenate([ [patient], patient_y_true_onehot, percent_calls_by_patient[i] ])
				writer.writerow(row)
	
	# Save tile-level predictions
	tile_csv_dir = os.path.join(data_dir, f"tile_predictions{label_end}.csv")
	with open(tile_csv_dir, 'w') as outfile:
		writer = csv.writer(outfile)
		header = ['slide'] + [f"y_true{i}" for i in range(num_cat)] + [f"y_pred{j}" for j in range(num_cat)]
		writer.writerow(header)
		for i in range(len(y_true)):
			row = np.concatenate([[tile_to_slides[i]], [str(y_true[i])], [str(y_pred[i])]])
			writer.writerow(row)

	log.complete(f"Predictions saved to {sfutil.green(data_dir)}", 1)
	return tile_auc, slide_auc, patient_auc, r_squared