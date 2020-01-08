import os
import json
import csv
import pprint
import hashlib
import argparse

import slideflow.util as sfutil

from os.path import join, exists
from slideflow.util import log, TCGA
from tabulate import tabulate

# Organization heirarchy:
# Dataset
#  |- Project
#     |- Outcome
#        |- Patient subset
#           |- Model

# TODO: check that k-folds are truly compatible (rather than just checking to see if patient lists are the same)
# TODO: Automatically link evaluations to model training

class Project:
	def __init__(self, path):
		with open(join(path, "settings.json")) as settings_file:
			self.settings = json.load(settings_file)

		if not all(x in self.settings for x in ("dataset_config", "datasets", "name")):
			self.settings = False
		else:
			self.name = self.settings['name']
			self.dataset = self.settings['datasets']
			self.outcomes = []

	def has_outcome(self, outcome):
		return self.get_outcome(outcome) != -1

	def get_outcome(self, outcome_headers):
		for outcome in self.outcomes:
			if outcome.outcome_headers == outcome_headers:
				return outcome
		return -1

	def add_outcome(self, outcome_headers):
		outcome = Outcome(outcome_headers)
		self.outcomes += [outcome]
		return outcome

	def print_summary(self):
		datasets = [self.settings['datasets']] if type(self.settings['datasets']) != list else self.settings['datasets']
		datasets_string = []
		with open(self.settings['dataset_config'], 'r') as config_file:
			config = json.load(config_file)
			for dataset in datasets:
				if dataset not in config:
					log.error(f'Unable to load dataset "{dataset}" configuration for project {self.name}', 1)
					return
				description = "" if 'description' not in config[dataset] else f" ({config[dataset]['description']})"
				datasets_string += [f"{sfutil.bold(dataset)}{description}"]
		print(' and '.join(datasets_string))
		for outcome in self.outcomes:
			outcome.print_summary()  

class Outcome:
	def __init__(self, outcome_headers):
		self.outcome_headers = outcome_headers
		self.subsets = []

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

	def print_summary(self):
		headers = [self.outcome_headers] if type(self.outcome_headers) != list else self.outcome_headers
		print(f"\t{', '.join(headers)}")
		for subset in self.subsets:
			subset.print_summary()

class Subset:
	def __init__(self, _id, slide_list, dataset, filters, validation_strategy, total_k_folds):
		self.models = []
		self.model_groups = []
		self.id = _id
		self.slide_list = slide_list
		self.dataset = dataset
		self.filters = filters
		self.num_slides = len(slide_list)
		self.validation_strategy = validation_strategy
		self.k_folds = {} if validation_strategy == "k-fold" else None
		self.total_k_folds = total_k_folds if validation_strategy == 'k-fold' else 0
		self.manifest_hash = None

	def add_model(self, model):
		self.models += [model]
		
		#self.add_model_to_group(model)
		group = self.get_compatible_model_group(model)
		if group:
			group.add_model(model)
		else:
			group = ModelGroup([model], len(self.model_groups))
			self.model_groups += [group]

		if (model.validation_strategy != self.validation_strategy):
			print("Incompatible model type: unable to add to subset")
		if model.k_fold_i and model.k_fold_i in self.k_folds:
			self.k_folds[model.k_fold_i]['models'] += [model]
		elif model.k_fold_i:
			self.k_folds.update({model.k_fold_i: {
				'models': [model],
				'manifest_hash': model.manifest.hash,
				'slide_hash': model.manifest.slide_hash
			}})

	def get_compatible_model_group(self, model):
		for group in self.model_groups:
			if group.is_compatible(model):
				return group
		return False

	def print_models(self):
		if self.validation_strategy == 'k-fold':
			for k in self.k_folds:
				print(f"K-fold {k}: {self.k_folds[k]}")
		else:
			print(self.models)

	def print_summary(self):
		print(f"\t\tSubset {self.id}" + f" ({self.total_k_folds}-fold cross-validation)" if self.validation_strategy=='k-fold' else "")
		print(f"\t\t{len(self.slide_list)} slides")
		print(f"\t\tFilters: {self.filters}")

		def sorted_val(e):
			return float(e['slide_auc'][1:-1].split(', ')[0])

		epochs = []
		for model in self.models:
			for epoch in model.results:
				epochs += [(model, epoch)]

		epochs.sort(key=lambda v: sorted_val(v[0].results[v[1]]), reverse=True)
		epochs.sort(key=lambda v: v[0].group.id)

		tabbed_results = {
			"Model name": [e[0].name for e in epochs],
			"Epoch": [e[1] for e in epochs],
			"Slide AUC": [sorted_val(e[0].results[e[1]]) for e in epochs],
			"Group": [e[0].group.id for e in epochs]
		}
		if self.k_folds:
			tabbed_results.update({
				"K-fold": [e[0].k_fold_i for e in epochs]
			})
		print("\t\t\t" + tabulate(tabbed_results, headers="keys").replace("\n", "\n\t\t\t"))

class ModelGroup:
	def __init__(self, models, _id):
		self.models = []
		self.hp_key = models[0].hp_key
		self.add_models(models)
		self.id = _id

	def add_model(self, model):
		if self.is_compatible(model):
			self.models += [model]
			model.group = self
		else:
			log.error("Incompatible model, unable to add to group")
	
	def add_models(self, models):
		for model in models:
			self.add_model(model)
	
	def is_compatible(self, model):
		if model.hp_key == self.hp_key:
			return True
		else:
			return False

class Model:
	def __init__(self, models_dir, name, project, interactive=True):
		self.dir = join(models_dir, name)
		self.name = name
		self.project = project
		self.results = {}
		self.group = None
		if not exists(join(self.dir, "hyperparameters.json")): 
			self.hyperparameters = False		
		else:
			with open(join(self.dir, "hyperparameters.json"), 'r') as hp_file:
				self.hyperparameters = json.load(hp_file)
			if "outcome_headers" not in self.hyperparameters:
				if not interactive or not self.get_outcome_headers():
					self.hyperparameters = False
			if self.hyperparameters:
				self.validation_strategy = self.hyperparameters["validation_strategy"]
				self.k_fold_i = self.hyperparameters["k_fold_i"]
				self.k_fold = self.hyperparameters["validation_k_fold"]
				self.filters = self.hyperparameters['filters']
				self.manifest = SlideManifest(join(self.dir, "slide_manifest.log"))
				self.load_results(join(self.dir, "results_log.csv"))
				self.hp_key = tuple(sorted(self.hyperparameters['hp'].items()))

	def load_results(self, results_log):
		with open(results_log, 'r') as results_file:
			reader = csv.reader(results_file)
			header = next(reader)
			epoch_i = header.index('epoch')
			meta = [h for h in header if h != 'epoch']
			meta_i = [header.index(h) for h in meta]
			for row in reader:
				epoch = row[epoch_i]
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
		ohi = sfutil.choice_input(f"What are the outcome header(s) for model {self.name} in project {self.project.name}?\n  (respond with number(s), separated by commas if multiple) ", list(range(len(outcome_headers))), multi_choice=True, input_type=int)
		oh = [outcome_headers[i-1] for i in ohi]
		if "<skip>" in oh:
			return False
		self.hyperparameters.update({'outcome_headers': oh})
		sfutil.write_json(self.hyperparameters, join(self.dir, "hyperparameters.json"))
		log.info(f"Updated {sfutil.green(join(self.dir, 'hyperparameters.json'))} with 'outcome_headers'={oh}", 2)
		return True

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

def load_from_directory(project_directory):
	project_folders = [join(project_directory, d) for d in os.listdir(project_directory) 
												  if exists(join(project_directory, d, "settings.json"))]
	for pf in project_folders:
		project = Project(pf)
		if not project.settings: continue

		models = os.listdir(join(pf, "models"))

		for model_name in models:
			model = Model(join(pf, "models"), model_name, project)
			if not model.hyperparameters: continue
			model_outcome = model.hyperparameters['outcome_headers']
			model_outcome = [model_outcome] if type(model_outcome) != list else model_outcome

			if project.has_outcome(model_outcome):
				outcome = project.get_outcome(model_outcome)
			else:
				outcome = project.add_outcome(model_outcome)

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
			subset = Subset(len(outcome.subsets), model.manifest.slide_list, project.dataset, model.filters, model.validation_strategy, model.k_fold)
			subset.add_model(model)
			outcome.add_subset(subset)
		
		project.print_summary()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Summarizes Slideflow project results.")
	parser.add_argument('-d', '--dir', required=True, help='Path to parent directory containings slideflow projects.')
	args = parser.parse_args()

	load_from_directory(args.dir)