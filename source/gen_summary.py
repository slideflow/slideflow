import os
import json
import csv
import pprint
import hashlib

import slideflow.util as sfutil

from os.path import join, exists
from slideflow.util import log

# Automatically link evaluations to model training

# Organization heirarchy:
# Dataset
#  |- Project
#     |- Outcome
#        |- Patient subset
#           |- Model

# TODO: check that k-folds are truly compatible (rather than just checking to see if patient lists are the same)

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
        outcome = Outcome(outcome_id, outcome_headers)
        self.outcomes += [outcome]
        return outcome

class Outcome:
    def __init__(self, outcome_id, outcome_headers):
        self.id = outcome_id
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

class Subset:
    def __init__(self, _id, slide_list, dataset, filters, validation_strategy):
        self.models = []
        self.id = _id
        self.slide_list = slide_list
        self.dataset = dataset
        self.filters = filters
        self.num_slides = len(slide_list)
        self.validation_strategy = validation_strategy
        self.k_folds = {} if validation_strategy == "k-fold" else None
        self.manifest_hash = None

    def add_model(self, model):
        self.models += [model]
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

    def print_models(self):
        if self.validation_strategy == 'k-fold':
            for k in self.k_folds:
                print(f"K-fold {k}: {self.k_folds[k]}")
        else:
            print(self.models)

class Model:
    def __init__(self, models_dir, name):
        self.dir = join(models_dir, name)
        if not exists(join(self.dir, "hyperparameters.json")): 
            self.hyperparameters = False
        else:
            with open(join(self.dir, "hyperparameters.json")) as hp_file:
                self.hyperparameters = json.load(hp_file)
            if "outcome_headers" not in self.hyperparameters:
                self.hyperparameters = False
            else:
                self.validation_strategy = self.hyperparameters["validation_strategy"]
                self.k_fold_i = self.hyperparameters["k_fold_i"]
                self.filters = self.hyperparameters['filters']
                self.manifest = SlideManifest(join(self.dir, "slide_manifest.log"))

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

def gen_summary(project_directory):
    project_folders = [join(project_directory, d) for d in os.listdir(project_directory) 
                                                  if exists(join(project_directory, d, "settings.json"))]

    ALL_SUBSETS = []

    for pf in project_folders:
        project = Project(pf)
        if not project.settings: continue

        models = os.listdir(join(pf, "models"))

        for model_name in models:
            model = Model(join(pf, "models"), model_name)
            if not model.hyperparameters: continue

            if project.has_outcome(model.hyperparameters['outcome_headers']):
                outcome = project.get_outcome(model.hyperparameters['outcome_headers'])
            else:
                outcome = project.add_outcome(model.hyperparameters['outcome_headers'])

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
            subset = Subset(len(outcome.subsets), model.manifest.slide_list, project.dataset, model.filters, model.validation_strategy)
            subset.add_model(model)
            outcome.add_subset(subset)
            ALL_SUBSETS += [subset]

    for sub in ALL_SUBSETS:
        print(f"Subset {sub.id}:")
        sub.print_models()

if __name__=='__main__':
    gen_summary('/home/shawarma/data/slideflow_projects')