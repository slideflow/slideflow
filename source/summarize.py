import os
import csv
import time
import json
import pprint
import hashlib
import argparse

import slideflow.util as sfutil
import slideflow.util.statistics as sfstats

from statistics import mean
from os.path import join, exists, isdir, getmtime
from slideflow.util import log, TCGA
from slideflow.util.datasets import Dataset
from tabulate import tabulate

# Organization heirarchy:
# DatasetGroup
#  |- Project
#     |- Outcome
#        |- Patient subset
#           |- Model

# TODO: check that k-folds are truly compatible (rather than just checking to see if patient lists are the same)
# TODO: Automatically link evaluations to model training
# TODO: find r_squared from parent directory
# TODO: export results to CSV

class DatasetGroup:
	def __init__(self, names, config):
		self.names = [names] if type(names) != list else names
		self.config = config
		self.outcomes = []

	def has_outcome(self, outcome):
		return self.get_outcome(outcome) != -1

	def get_outcome(self, outcome_headers):
		for outcome in self.outcomes:
			if outcome.outcome_headers == outcome_headers:
				return outcome
		return -1

	def add_outcome(self, outcome_headers, outcome_labels):
		outcome = Outcome(outcome_headers, outcome_labels)
		self.outcomes += [outcome]
		return outcome

	def print_summary(self, grouped=False, show_model_names=False):
		datasets_string = []
		with open(self.config, 'r') as config_file:
			config = json.load(config_file)
			for dataset_name in self.names:
				if dataset_name not in config:
					log.error(f'Unable to load dataset "{dataset_name}" configuration', 1)
					return
				description = "" if 'description' not in config[dataset_name] else f" ({config[dataset_name]['description']})"
				datasets_string += [f"{sfutil.bold(dataset_name)}{description}"]
		if len(self.outcomes):
			print(' and '.join(datasets_string))
		for outcome in self.outcomes:
			outcome.print_summary(grouped=grouped, show_model_names=show_model_names)

class Project:
	def __init__(self, path):
		with open(join(path, "settings.json")) as settings_file:
			self.settings = json.load(settings_file)

		if not all(x in self.settings for x in ("dataset_config", "datasets", "name")):
			self.settings = False
		else:
			self.name = self.settings['name']

class Outcome:
	def __init__(self, outcome_headers, outcome_labels):
		self.outcome_headers = [outcome_headers] if type(outcome_headers) != list else outcome_headers
		self.outcome_labels = outcome_labels
		self.subsets = []
		self.string = ', '.join(self.outcome_headers)

	def get_subset(self, validation_strategy, k_fold_i, manifest_hash):
		for subset in self.subsets:
			if subset.validation_strategy == validation_strategy:
				if validation_strategy == 'k-fold':
					for kfold in subset.k_folds:
						if (kfold == k_fold_i) and subset.k_folds[kfold]['manifest_hash'] == manifest_hash:
							return subset
				else:
					if subset.manifest_hash == manifest_hash:
						return subset
		return False

	def get_compatible_subset(self, validation_strategy, k_fold_i, manifest_slide_hash):
		for subset in self.subsets:
			if subset.validation_strategy == validation_strategy:
				if validation_strategy == 'k-fold':
					for kfold in subset.k_folds:
						if (kfold != k_fold_i) and subset.k_folds[kfold]['slide_hash'] == manifest_slide_hash:
							return subset
		return False

	def add_subset(self, subset):
		self.subsets += [subset]

	def print_summary(self, grouped=False, show_model_names=False):

		print(f"\t{self.string}")
		for subset in self.subsets:
			subset.print_summary(grouped=grouped, show_model_names=show_model_names)

class Subset:
	def __init__(self, _id, outcome, slide_list, filters, validation_strategy, total_k_folds):
		self.model_groups = []
		self.id = _id
		self.slide_list = slide_list
		self.filters = filters
		self.num_slides = len(slide_list)
		self.validation_strategy = validation_strategy
		self.k_folds = {} if validation_strategy == "k-fold" else None
		self.total_k_folds = total_k_folds if validation_strategy == 'k-fold' else 0
		self.manifest_hash = None
		self.outcome = outcome
		self.outcome_labels = None
		self.model_type = None

	def add_model(self, model):
		group = self.get_compatible_model_group(model)
		if group:
			group.add_model(model)
		else:
			group = ModelGroup([model], len(self.model_groups))
			group.subset = self
			self.model_groups += [group]
			if not self.outcome_labels:
				self.outcome_labels = group.outcome_labels
				self.model_type = group.model_type
			elif (self.outcome_labels != group.outcome_labels) or (self.model_type != group.model_type):
				log.warn("WARNING: added a group to a subset with mismatching outcome labels or model type")

		if (model.validation_strategy != self.validation_strategy):
			print("Incompatible model type: unable to add to subset")
			return
		if model.k_fold_i and model.k_fold_i in self.k_folds:
			self.k_folds[model.k_fold_i]['models'] += [model]
		elif model.k_fold_i:
			self.k_folds.update({model.k_fold_i: {
				'models': [model],
				'manifest_hash': model.manifest.hash,
				'slide_hash': model.manifest.slide_hash
			}})
		model.subset = self

	def get_compatible_model_group(self, model):
		for group in self.model_groups:
			if group.is_compatible(model):
				return group
		return False

	def print_summary(self, metrics=None, grouped=False, show_model_names=False):
		print(f"\t\tSubset {self.id}" + (f" ({self.total_k_folds}-fold cross-validation)" if self.validation_strategy=='k-fold' else "") + f": {len(self.slide_list)} slides")
		print(f"\t\tFilters: {self.filters}")
		print(f"\t\tOutcomes: {self.outcome_labels}")

		if not metrics:
			if self.model_type == 'linear':
				metrics = ['r_squared']
			else: 
				metrics = ['slide_auc', 'tile_auc']

		def get_metrics(e, mi):
			m = []
			for metric in metrics:
				try:
					if metric in e:
						m += [float(e[metric][1:-1].split(', ')[mi])]
					elif f'{metric}{mi}' in e:
						m += [float(e[f'{metric}{mi}'])]
					else:
						m += [-1]
				except:
					m += [-1]
			return m

		def format_epoch(s):
			return s.split('epoch')[-1]

		tabbed_results = {
			'Group ID': [],
			'Epoch': [],
			'K-fold': [],
		}
		for metric in metrics:
			if self.outcome_labels:
				for label in self.outcome_labels.values():
					tabbed_results.update({f'{metric} ({label})': []})
			else:
				tabbed_results.update({metric: []})
		tabbed_results.update({"Model names": []})

		for group in self.model_groups:
			if group.k_fold and not group.stage=="evaluation" and grouped:
				# Generates new figure overlaying ROCs from each k-fold
				group.gen_combined_roc()

				models_by_kfold = group.get_models_by_kfold()
				for e in group.epochs:
					used_k_str = [sfutil.purple('-')] * group.k_fold
					
					metrics_results = {}
					for metric in metrics:
						if self.outcome_labels:
							for label in self.outcome_labels.values():
								metric_label = f'{metric} ({label})'
								metrics_results.update({metric_label: {
																	'str': [sfutil.purple(' -  ')] * group.k_fold,
																	'val': []
																}
													})
						else:
							metrics_results.update({metric: {
																	'str': [sfutil.purple(' -  ')] * group.k_fold,
																	'val': []
																}
													})
					model_names = []
					for k in models_by_kfold:
						for model in models_by_kfold[k]:
							if e in model.results:
								# Do something with it
								used_k_str[k-1] = str(k)

								for i, metric in enumerate(metrics):
									if self.outcome_labels:
										for l in self.outcome_labels:
											metrics_results[f'{metric} ({self.outcome_labels[l]})']['str'][k-1] = f'{get_metrics(model.results[e], mi=int(l))[i]:.2f}'
											metrics_results[f'{metric} ({self.outcome_labels[l]})']['val'] += [get_metrics(model.results[e], mi=int(l))[i]]
									else:
										metrics_results[metric]['str'][k-1] = f'{get_metrics(model.results[e], mi=0)[i]:.2f}'
										metrics_results[metric]['val'] += [get_metrics(model.results[e], mi=0)[i]]

								model_names += [model.name]
							else:
								continue
					tabbed_results['Group ID'] += [group.id]
					tabbed_results['Epoch'] += [format_epoch(e)]
					tabbed_results['K-fold'] += [" / ".join(used_k_str)]
					tabbed_results['Model names'] += [" / ".join(model_names)]

					for metric in metrics:
						if self.outcome_labels:
							for label in self.outcome_labels.values():
								metric_label = f'{metric} ({label})'
								if len(metrics_results[metric_label]['val']):
									mean_metric = f'{mean(metrics_results[metric_label]["val"]):.2f}'
								else:
									mean_metric = '-'
								tabbed_results[metric_label] += [" / ".join(metrics_results[metric_label]['str']) + ' (' + sfutil.green(sfutil.bold(mean_metric)) + ')']
						else:
							if len(metrics_results[metric]['val']):
								mean_metric = f'{mean(metrics_results[metric]["val"]):.2f}'
							else:
								mean_metric = '-'
							tabbed_results[metric] += [" / ".join(metrics_results[metric]['str']) + ' (' + sfutil.green(sfutil.bold(mean_metric)) + ')']
			else:
				epochs = []
				for model in group.models:
					for epoch in model.results:
						epochs += [(model, epoch)]

				#epochs.sort(key=lambda v: get_metrics(v[0].results[v[1]], mi=0)[0], reverse=True)
				#epochs.sort(key=lambda v: v[0].group.id)

				for e in epochs:
					tabbed_results['Group ID'] += [group.id]
					tabbed_results['Epoch'] += [format_epoch(e[1])]
					tabbed_results['K-fold'] += [e[0].k_fold_i]
					tabbed_results['Model names'] += [e[0].name]

					if self.outcome_labels:
						for l in self.outcome_labels:
							metrics_results = get_metrics(e[0].results[e[1]], mi=int(l))
							for i, metric in enumerate(metrics):
								tabbed_results[f'{metric} ({self.outcome_labels[l]})'] += [metrics_results[i]]
					else:
						metrics_results = get_metrics(e[0].results[e[1]], mi=0)
						for i, metric in enumerate(metrics):
							tabbed_results[metric] += [metrics_results[i]]

		if not show_model_names:
			del(tabbed_results['Model names'])

		print("\n\t\t\t" + tabulate(tabbed_results, headers="keys").replace("\n", "\n\t\t\t") + "\n")

class ModelGroup:
	def __init__(self, models, _id):
		self.models = []
		self.epochs = []
		self.tile_px = models[0].tile_px
		self.tile_um = models[0].tile_um
		self.hp_key = models[0].hp_key
		self.k_fold = models[0].k_fold
		self.stage = models[0].stage
		self.outcome_labels = models[0].outcome_labels
		self.model_type = models[0].model_type
		self.add_models(models)
		self.id = _id
		self.subset = None

	def add_model(self, model):
		if self.is_compatible(model):
			self.models += [model]
			for e in model.results:
				if e not in self.epochs:
					self.epochs += [e]
			model.group = self
		else:
			log.error(f"Incompatible model, unable to add model {model.name} to group")
	
	def add_models(self, models):
		for model in models:
			self.add_model(model)
	
	def is_compatible(self, model):
		if ((model.hp_key == self.hp_key) and 
			(model.k_fold == self.k_fold) and 
			(model.tile_px == self.tile_px) and 
			(model.tile_um == self.tile_um) and 
			(model.stage == self.stage) and 
			(model.k_fold_i not in [m.k_fold_i for m in self.models])):
			return True
		else:
			return False

	def get_models_by_kfold(self):
		if not self.k_fold:
			return False
		else:
			models_by_kfold = {}
			for k in range(1, self.k_fold+1):
				for model in self.models:
					if (model.k_fold_i == k) and k not in models_by_kfold:
						models_by_kfold.update({k: [model]})
					elif (model.k_fold_i == k):
						models_by_kfold[k] += [model]
			return models_by_kfold

	def gen_combined_roc(self):
		save_dir = join(self.models[0].project.settings['root'], 'stats')
		if not exists(save_dir):
			os.makedirs(save_dir)

		for epoch in self.epochs:
			y_true_all, y_pred_all = {}, {}
			for model in self.models:
				#epochs = [e.split('epoch')[-1] for e in list(model.results.keys())]
				#if not len(epochs): continue
				if epoch not in model.results.keys():
					continue
				pred = model.get_predictions(epoch.split('epoch')[-1])

				if not pred:
					continue

				for label in pred:
					if label not in y_true_all:
						y_true_all.update({label: []})
						y_pred_all.update({label: []})

					y_true_all[label] += [pred[label]['y_true']]
					y_pred_all[label] += [pred[label]['y_pred']]

			labels = list(y_true_all.keys())
			labels.sort()
			for label in labels:
				#print(" | ".join([m.name for m in self.models]) + f" : [{label}] : {len(y_true_all[label])}")
				outcome_str = "" if not self.subset else self.subset.outcome.string
				sfstats.generate_combined_roc(y_true_all[label], y_pred_all[label], save_dir, labels=[f'K-fold {m.k_fold_i}' for m in self.models], name=f"Combined ROC [{outcome_str}-Group{self.id}-{label}-{epoch}]")
				#print(f"Saved combined ROCs to {sfutil.green(save_dir)}")

class Model:
	def __init__(self, models_dir, name, project, interactive=True):
		self.dir = join(models_dir, name)
		self.name = name
		self.project = project
		self.results = {}
		self.group = None
		self.outcome_labels = None
		self.last_modified = None
		self.subset = None
		if not exists(join(self.dir, "hyperparameters.json")): 
			self.hyperparameters = None
		else:
			with open(join(self.dir, "hyperparameters.json"), 'r') as hp_file:
				self.hyperparameters = json.load(hp_file)
			if "model_type" not in self.hyperparameters:
				self.hyperparameters.update({'model_type': 'categorical'})
				sfutil.write_json(self.hyperparameters, join(self.dir, 'hyperparameters.json'))
				log.info(f"Updated {sfutil.green(join(self.dir, 'hyperparameters.json'))} to specify model_type='categorical'")
			if "outcome_headers" not in self.hyperparameters:
				if not interactive or not self.get_outcome_headers():
					self.hyperparameters = None
			if self.hyperparameters and ("outcome_labels" not in self.hyperparameters):
				if not interactive or not self.get_outcomes():
					self.hyperparameters = None
			if self.hyperparameters:
				try:
					if "stage" in self.hyperparameters:
						self.stage = self.hyperparameters["stage"]
					else:
						self.stage = "training"
					self.tile_px = int(self.hyperparameters["tile_px"])
					self.tile_um = int(self.hyperparameters["tile_um"])
					self.validation_strategy = self.hyperparameters["validation_strategy"]
					self.k_fold_i = self.hyperparameters["k_fold_i"]
					self.k_fold = self.hyperparameters["validation_k_fold"]
					self.filters = self.hyperparameters['filters']
					self.manifest = SlideManifest(join(self.dir, "slide_manifest.log"))
					self.load_results(join(self.dir, "results_log.csv"))
					self.hp_key = tuple(sorted(self.hyperparameters['hp'].items()))
					self.model_type = self.hyperparameters['model_type']
					self.outcome_labels = self.hyperparameters['outcome_labels']
				except KeyError:
					log.error(f"Model {join(self.dir, 'hyperparameters.json')} file incorrectly formatted")
					self.hyperparameters = None

	def load_results(self, results_log):
		try:
			self.last_modified = getmtime(results_log)
		except FileNotFoundError:
			return
		with open(results_log, 'r') as results_file:
			reader = csv.reader(results_file)
			header = next(reader)
			if 'epoch' in header:
				epoch_i = header.index('epoch')
			elif 'model_name' in header:
				epoch_i = header.index('model_name')
			else:
				epoch_i = None
			meta = [h for h in header if h != 'epoch']
			meta_i = [header.index(h) for h in meta]
			for row in reader:
				if epoch_i != None:
					epoch = row[epoch_i]
				else:
					epoch = 'Unknown'
				self.results.update({
					epoch: dict(zip(meta, [row[mi] for mi in meta_i]))
				})

	def get_outcome_headers(self):
		with open(self.project.settings['annotations'], 'r') as ann_file:
			reader = csv.reader(ann_file)
			headers = next(reader)
			outcome_headers = [h for h in headers if h not in (TCGA.patient, TCGA.project, TCGA.slide)]
			outcome_headers += ["<skip>"]
		for i, outcome_header in enumerate(outcome_headers):
			print(f"{i+1}. {outcome_header}")
		ohi = sfutil.choice_input(f"What are the outcome header(s) for model {self.name} in project {self.project.name}?\n  (respond with number(s), separated by commas if multiple) ", list(range(1, len(outcome_headers)+1)), multi_choice=True, input_type=int)
		oh = [outcome_headers[i-1] for i in ohi]
		if "<skip>" in oh:
			return False
		self.hyperparameters.update({'outcome_headers': oh})
		sfutil.write_json(self.hyperparameters, join(self.dir, "hyperparameters.json"))
		log.info(f"Updated {sfutil.green(join(self.dir, 'hyperparameters.json'))} with 'outcome_headers'={oh}", 2)
		return True

	def get_outcomes(self):
		log.info(f"Outcomes not found in model hyperparameter log ({sfutil.green(self.dir)}), attempting to automatically detect...", 2)
		dataset = Dataset(config_file=self.project.settings['dataset_config'], sources=self.project.settings['datasets'])
		dataset.load_annotations(self.project.settings['annotations'])
		try:
			outcomes, unique_outcomes = dataset.get_outcomes_from_annotations(self.hyperparameters['outcome_headers'], 
																			filters=self.hyperparameters['filters'], 
																	 	 	filter_blank=self.hyperparameters['outcome_headers'],
																	 		use_float=(self.hyperparameters['model_type'] == 'linear'))
		except TypeError:
			log.error(f"Unable to load results for model {sfutil.green(self.dir)}; model_type is {self.hyperparameters['model_type']} but outcomes in annotations file cannot be converted into float", 1)
			return False
		except IndexError:
			filters = self.hyperparameters['filters']
			annotations = self.project.settings['annotations']
			log.error(f'Unable to load results for model {sfutil.green(self.dir)}; could not find all filters "{filters}" in the annotations file {annotations})')
			return False
		self.hyperparameters.update({"outcome_labels": None if self.hyperparameters['model_type'] != 'categorical' else dict(zip(range(len(unique_outcomes)), unique_outcomes))})
		sfutil.write_json(self.hyperparameters, join(self.dir, "hyperparameters.json"))
		log.info(f"Updated {sfutil.green(join(self.dir, 'hyperparameters.json'))} with 'outcome_labels'={self.hyperparameters['outcome_labels']}", 2)
		return True

	def get_predictions(self, epoch, level='patient'):
		if level not in ('patient', 'slide', 'tile'):
			log.error(f"Unknown prediction level {level}; must be 'patient', 'slide', or 'tile'.", 1)
			return None

		predictions = join(self.dir, f'{level}_predictions_val_epoch{epoch}.csv')
		if not exists(predictions):
			log.error(f"Unable to find predictions file {sfutil.bold(predictions)} in {sfutil.green(self.dir)}", 1)
			return None

		predictions = sfstats.read_predictions(predictions, level)
		return predictions

class SlideManifest:
	def __init__(self, path):
		self.metadata = {}
		with open(path, 'r') as f:
			contents = f.read()
			self.hash = hashlib.md5(contents.encode('utf-8')).hexdigest()
			f.seek(0)
			reader = csv.reader(f)
			header = next(reader)
			slide_i = header.index('slide')
			dataset_i = header.index('dataset')
			meta_headers = [h for h in header if h not in ('slide', 'dataset')]
			meta_headers_i = [header.index(h) for h in meta_headers]
			for row in reader:
				slide = row[slide_i]
				dataset = row[dataset_i]
				meta = [row[mi] for mi in meta_headers_i]
				self.metadata.update({
					slide: {
						'dataset': dataset,
						'meta': meta
					}
				})
			self.slide_list = list(self.metadata.keys())
			self.slide_list.sort()
			slide_list_str = ", ".join(self.slide_list)
			self.slide_hash = hashlib.md5(slide_list_str.encode('utf-8')).hexdigest()

def get_projects_from_folder(directory):
	return [join(directory, d) for d in os.listdir(directory) 
									   if (isdir(join(directory, d)) and exists(join(directory, d, "settings.json")))]

def get_compatible_dataset(project, datasets):
	for dataset in datasets:
		if dataset.config == project.settings['dataset_config'] and dataset.names == project.settings['datasets']:
			return dataset
	return False

def load_from_directory(search_directory, nested=False, starttime=None, shownames=False):
	if nested:
		project_folders = []
		for _dir in [join(search_directory, d) for d in os.listdir(search_directory) if isdir(join(search_directory, d))]:
			project_folders += get_projects_from_folder(_dir)
	else:
		project_folders = get_projects_from_folder(search_directory)

	datasets = []

	for pf in project_folders:
		# Initialize project
		project = Project(pf)
		if not project.settings: continue

		# Find compatible dataset; if none exists; create
		dataset = get_compatible_dataset(project, datasets)
		if not dataset:
			dataset = DatasetGroup(project.settings['datasets'], project.settings['dataset_config'])
			datasets += [dataset]

		# Find models in the project
		if exists(join(pf, "models")):
			models = os.listdir(join(pf, "models"))
		else:
			continue

		for model_name in models:
			# For each detected model, create a Model object
			model = Model(join(pf, "models"), model_name, project)
			if starttime and model.last_modified and model.last_modified-starttime < 0:
				continue
			if not model.hyperparameters: continue
			model_outcome = model.hyperparameters['outcome_headers']
			model_outcome = [model_outcome] if type(model_outcome) != list else model_outcome

			if dataset.has_outcome(model_outcome):
				outcome = dataset.get_outcome(model_outcome)
			else:
				outcome = dataset.add_outcome(model_outcome, model.outcome_labels)

			subset = outcome.get_subset(model.validation_strategy, model.k_fold_i, model.manifest.hash)
			if subset:
				subset.add_model(model)
				continue
			subset = outcome.get_compatible_subset(model.validation_strategy, model.k_fold_i, model.manifest.slide_hash)
			if subset:
				subset.k_folds.update({
					model.k_fold_i: {
						'manifest_hash': model.manifest.hash,
						'slide_hash': model.manifest.slide_hash,
						'models': []
					}
				})
				subset.add_model(model)
				continue
			subset = Subset(len(outcome.subsets), outcome, model.manifest.slide_list, model.filters, model.validation_strategy, model.k_fold)
			subset.add_model(model)
			outcome.add_subset(subset)
	
	for dataset in datasets:
		dataset.print_summary(grouped=True, show_model_names=shownames)

def valid_date(s):
	try:
		return time.mktime(time.strptime(s, "%Y-%m-%d"))
	except ValueError:
		try:
			return time.mktime(time.strptime(s, "%Y-%m-%d-%H-%M-%S"))
		except ValueError:
			msg = f"Not a valid date: '{s}'"
			raise argparse.ArgumentTypeError(msg)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Summarizes Slideflow project results.")
	parser.add_argument('-d', '--dir', required=True, type=str, help='Path to parent directory containings slideflow projects.')
	parser.add_argument('-n', '--nested', action="store_true", help='Whether directory specified contains further nested directories to search.')
	parser.add_argument('-s', '--since', type=valid_date, help='Print results from this starting date (Format: YYYY-mm-dd or YYYY-mm-dd-HH-MM-SS)')
	parser.add_argument('--names', action="store_true", help='Print model names with results.')
	args = parser.parse_args()

	load_from_directory(args.dir, args.nested, args.since, args.names)