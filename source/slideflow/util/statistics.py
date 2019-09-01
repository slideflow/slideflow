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
		log.error("Y_true and y_pred must have the same shape in order to generate a scatter plot")
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
		p = sns.jointplot(y_true, y_pred, kind="reg", stat_func=r2)
		p.set_axis_labels('y_true', 'y_pred')
		plt.savefig(os.path.join(data_dir, f'Scatter{name}.png'))

	return r_squared

def generate_predictions_and_roc(model, dataset_with_casenames, model_type, data_dir, label=None):
	'''Dataset must include three items: raw image data, labels, and case names.'''

	# TODO: return array of aucs for each outcome variable instead of just last
	
	# Get predictions and performance metrics
	log.info("Generating predictions...", 1)
	label_end = "" if not label else f"_{label}"
	label_start = "" if not label else f"{label}_"
	y_true, y_pred, cases = [], [], []
	for i, batch in enumerate(dataset_with_casenames):
		sys.stdout.write(f"\r   - Working on batch {i}...")
		sys.stdout.flush()
		cases += [case_bytes.decode('utf-8') for case_bytes in batch[2].numpy()]
		y_true += [batch[1].numpy()]
		y_pred += [model.predict_on_batch(batch[0])]
	sys.stdout.write("\r\033[K")
	sys.stdout.flush()
	y_pred = np.concatenate(y_pred)
	y_true = np.concatenate(y_true)

	tile_auc = []
	slide_auc = []
	r_squared = None

	if model_type == 'linear':
		y_true = np.array([[i] for i in y_true])
		num_cat = len(y_pred[0])
		r_squared = generate_scatter(y_true, y_pred, label_end, data_dir)
			
	if model_type == 'categorical':
		# Convert y_true to one_hot encoding
		num_cat = len(y_pred[0])
		def to_onehot(val):
			onehot = [0] * num_cat
			onehot[val] = 1
			return onehot
		y_true = np.array([to_onehot(i) for i in y_true])

		# Scan datasets for case-level one_hot encoding
		case_onehot = {}
		for i in range(len(cases)):
			casename = cases[i]
			if casename not in case_onehot:
				case_onehot.update({casename: y_true[i]})
			# Now check for data integrity problems (case assigned to multiple different outcomes)
			elif not np.array_equal(case_onehot[casename], y_true[i]):
				log.error("Data integrity failure when generating ROCs", 1)
				sys.exit()

		# Generate tile-level ROC
		for i in range(num_cat):
			auc = generate_roc(y_true[:, i], y_pred[:, i], data_dir, f'{label_start}tile_ROC{i}')
			tile_auc += [auc]
			log.info(f"Tile-level AUC (cat #{i}): {auc}", 1)

		# Generate slide-level ROC
		onehot_predictions = []
		for x in range(len(y_pred)):
			predictions = y_pred[x]
			onehot_predictions += [[1 if pred == max(predictions) else 0 for pred in predictions]]

		unique_cases = list(set(cases))
		percent_calls_by_case = []
		for case in unique_cases:
			percentages = []
			for cat_index in range(num_cat):
				percentages += [sum([ onehot_predictions[i][cat_index] for i in range(len(cases)) if cases[i] == case ]) / len([i for i in range(len(cases)) if cases[i] == case])]
			percent_calls_by_case += [percentages]
		percent_calls_by_case = np.array(percent_calls_by_case)

		for i in range(num_cat):
			case_y_pred = percent_calls_by_case[:, i]
			case_y_true = [case_onehot[case][i] for case in unique_cases]
			auc = generate_roc(case_y_true, case_y_pred, data_dir, f'{label_start}slide_ROC{i}')
			slide_auc += [auc]
			log.info(f"Slide-level AUC (cat #{i}): {auc}", 1)
			# Write raw data for debugging
			#with open(os.path.join(data_dir, f"ROC_{i}.txt"), 'w') as dumpfile:
			#	writer = csv.writer(dumpfile)	

		# Save slide-level predictions
		slide_csv_dir = os.path.join(data_dir, f"slide_predictions{label_end}.csv")
		with open(slide_csv_dir, 'w') as outfile:
			writer = csv.writer(outfile)
			header = ['case'] + [f"y_true{i}" for i in range(num_cat)] + [f"percent_tiles_positive{j}" for j in range(num_cat)]
			writer.writerow(header)
			for i, case in enumerate(unique_cases):
				case_y_true_onehot = case_onehot[case]
				row = np.concatenate([ [case], case_y_true_onehot, percent_calls_by_case[i] ])
				writer.writerow(row)
	
	# Save tile-level predictions
	tile_csv_dir = os.path.join(data_dir, f"tile_predictions{label_end}.csv")
	with open(tile_csv_dir, 'w') as outfile:
		writer = csv.writer(outfile)
		header = ['case'] + [f"y_true{i}" for i in range(num_cat)] + [f"y_pred{j}" for j in range(num_cat)]
		writer.writerow(header)
		for i in range(len(y_true)):
			row = np.concatenate([[cases[i]], y_true[i], y_pred[i]])
			writer.writerow(row)

	log.complete(f"Predictions saved to {sfutil.green(data_dir)}", 1)
	return tile_auc, slide_auc, r_squared