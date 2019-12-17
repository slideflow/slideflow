import argparse
import os
import sys
import shutil
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from comet_ml import Experiment
import tensorflow as tf

from os.path import join, isfile, exists, isdir
from pathlib import Path
from glob import glob
from random import shuffle, choice
from string import ascii_lowercase
import csv

import gc
import atexit
import subprocess
import itertools
import multiprocessing

import slideflow.trainer.model as sfmodel
import slideflow.util as sfutil
from slideflow.util import TCGA, log
from slideflow.util.datasets import Dataset
from slideflow.mosaic import ActivationsVisualizer, TileVisualizer

# TODO: allow datasets to have filters (would address evaluate() function)

__version__ = "1.3.4"

SKIP_VERIFICATION = False
NUM_THREADS = 4
EVAL_BATCH_SIZE = 64
GPU_LOCK = None
NO_LABEL = 'no_label'
SILENT = 'SILENT'
SOURCE_DIR = os.path.dirname(os.path.realpath(__file__))
VALIDATION_ID = ''.join(choice(ascii_lowercase) for i in range(10))
COMET_API_KEY = "A3VWRcPaHgqc4H5K0FoCtRXbp"
DEBUGGING = True

def set_logging_level(level):
	if level == SILENT:
		sfutil.LOGGING_LEVEL.SILENT = True
	else:
		sfutil.LOGGING_LEVEL.INFO = level

def autoselect_gpu(number_available, reverse=True):
	global GPU_LOCK
	'''Automatically claims a free GPU and creates a lock file to prevent 
	other instances of slideflow from using the same GPU.'''
	gpus = range(number_available) if not reverse else reversed(range(number_available))
	for n in gpus:
		if not exists(join(SOURCE_DIR, f"gpu{n}.lock")):
			log.empty(f"Requesting GPU #{n}")
			os.environ["CUDA_VISIBLE_DEVICES"]=str(n)
			open(join(SOURCE_DIR, f"gpu{n}.lock"), 'a').close()
			GPU_LOCK = n
			return
	log.error(f"No free GPUs detected; try deleting 'gpu[#].lock' files in the slideflow directory if GPUs are not in use.")

def select_gpu(number):
	global GPU_LOCK
	log.empty(f"Requesting GPU #{number}")
	GPU_LOCK = number
	os.environ["CUDA_VISIBLE_DEVICES"]=str(number)

def release_gpu():
	global GPU_LOCK
	log.empty("Cleaning up...")
	if GPU_LOCK != None and exists(join(SOURCE_DIR, f"gpu{GPU_LOCK}.lock")):
		log.empty(f"Freeing GPU {GPU_LOCK}...")
		os.remove(join(SOURCE_DIR, f"gpu{GPU_LOCK}.lock"))
	
atexit.register(release_gpu)

class SlideflowProject:
	MANIFEST = None

	def __init__(self, project_folder, interactive=True):
		'''Initializes project by creating project folder, prompting user for project settings, and project
		settings to "settings.json" within the project directory.'''
		os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
		log.header(f"Slideflow v{__version__}\n================")
		log.header("Loading project...")
		if project_folder and not os.path.exists(project_folder):
			if interactive:
				if sfutil.yes_no_input(f'Directory "{project_folder}" does not exist. Create directory and set as project root? [Y/n] ', default='yes'):
					os.makedirs(project_folder)
				else:
					project_folder = sfutil.dir_input("Where is the project root directory? ", create_on_invalid=True, absolute=True)
			else:
				log.info(f"Project directory {project_folder} not found; will create.")
				os.makedirs(project_folder)
		if not project_folder:
			project_folder = sfutil.dir_input("Where is the project root directory? ", create_on_invalid=True, absolute=True)
		sfutil.PROJECT_DIR = project_folder

		if exists(join(project_folder, "settings.json")):
			self.load_project(project_folder)
		elif interactive:
			self.create_project()
		
	def extract_tiles(self, tile_um=None, tile_px=None, filters=None, skip_validation=False, generate_tfrecords=True):
		'''Extract tiles from a group of slides; save a percentage of tiles for validation testing if the 
		validation target is 'per-patient'; and generate TFRecord files from the raw images.'''
		import slideflow.convoluter as convoluter

		log.header("Extracting image tiles...")
		tile_um = self.PROJECT['tile_um'] if not tile_um else tile_um
		tile_px = self.PROJECT['tile_px'] if not tile_px else tile_px
		convoluter.NUM_THREADS = NUM_THREADS

		# Load dataset for evaluation
		extracting_dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])

		for dataset_name in self.PROJECT['datasets']:
			log.empty(f"Working on dataset {sfutil.bold(dataset_name)}", 1)
			unfiltered_slide_list = extracting_dataset.get_slides_by_dataset(dataset_name)
			slide_list = sfutil.filter_slide_paths(unfiltered_slide_list, filters=filters)
			log.info(f"Extracting tiles from {len(slide_list)} slides ({tile_um} um, {tile_px} px)", 1)
			
			save_folder = join(extracting_dataset.datasets[dataset_name]['tiles'], extracting_dataset.datasets[dataset_name]['label'])
			roi_dir = extracting_dataset.datasets[dataset_name]['roi']

			if not os.path.exists(save_folder):
				os.makedirs(save_folder)

			c = convoluter.Convoluter(tile_px, tile_um, batch_size=None,
														use_fp16=self.PROJECT['use_fp16'], 
														stride_div=2,
														save_folder=save_folder, 
														roi_dir=roi_dir)
			c.load_slides(slide_list)
			c.convolute_slides(export_tiles=True)

			if not skip_validation and self.PROJECT['validation_target'] == 'per-tile':
				if self.PROJECT['validation_target'] == 'per-tile':
					if self.PROJECT['validation_strategy'] == 'boostrap':
						log.warn("Validation bootstrapping is not supported when the validation target is per-tile; will generate random fixed validation target", 1)
					if self.PROJECT['validation_strategy'] in ('bootstrap', 'fixed'):
						# Split the extracted tiles into two groups
						sfutil.datasets.split_tiles(save_folder, fraction=[-1, self.PROJECT['validation_fraction']], names=['training', 'validation'])
					if self.PROJECT['validation_strategy'] == 'k-fold':
						sfutil.datasets.split_tiles(save_folder, fraction=[-1] * self.PROJECT['validation_k_fold'], names=[f'kfold-{i}' for i in range(self.PROJECT['validation_k_fold'])])

		if generate_tfrecords:
			self.generate_tfrecord()

	def generate_tfrecord(self):
		'''Create tfrecord files from a collection of raw images'''
		log.header('Writing TFRecord files...')

		# Load dataset for evaluation
		working_dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])
		
		for d in working_dataset.datasets:
			log.empty(f"Working on dataset {d}", 1)
			config = working_dataset.datasets[d]
			tfrecord_dir = join(config["tfrecords"], config["label"])
			tiles_dir = join(config["tiles"], config["label"])

			# Check to see if subdirectories in the target folders are slide directories (contain images)
			#  or are further subdirectories (e.g. validation and training)
			log.info('Scanning tile directory structure...', 2)
			if sfutil.contains_nested_subdirs(tiles_dir):
				subdirs = [_dir for _dir in os.listdir(tiles_dir) if isdir(join(tiles_dir, _dir))]
				for subdir in subdirs:
					tfrecord_subdir = join(tfrecord_dir, subdir)
					sfutil.tfrecords.write_tfrecords_multi(join(tiles_dir, subdir), tfrecord_subdir)
			else:
				sfutil.tfrecords.write_tfrecords_multi(tiles_dir, tfrecord_dir)

			self.update_manifest()

			if self.PROJECT['delete_tiles']:
				shutil.rmtree(tiles_dir)

	def update_tfrecords(self):
		log.header('Updating TFRecords...')
		working_dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])

		for d in working_dataset.datasets:
			config = working_dataset.datasets[d]
			tfrecord_folder = join(config["tfrecords"], config["label"])
			num_updated = 0
			log.info(f"Updating TFRecords in {sfutil.green(tfrecord_folder)}...")
			num_updated += sfutil.tfrecords.update_tfrecord_dir(tfrecord_folder, slide='case', image_raw='image_raw')
		log.complete(f"Updated {sfutil.bold(num_updated)} TFRecords files")

	def initialize_model(self, model_name, dataset, train_tfrecords, validation_tfrecords, outcomes, model_type='categorical'):
		'''Prepares a Slideflow model using the provided outcome variable (outcome_header) 
		and a given set of training and validation tfrecords.'''

		# Using the project annotation file, assemble list of slides for training, as well as the slide annotations dictionary (output labels)
		model_dir = join(self.PROJECT['models_dir'], model_name)

		# Build a model using the slide list as input and the annotations dictionary as output labels
		SFM = sfmodel.SlideflowModel(model_dir, self.PROJECT['tile_px'], outcomes, train_tfrecords, validation_tfrecords,
																				manifest=dataset.get_manifest(),
																				use_fp16=self.PROJECT['use_fp16'],
																				model_type=model_type)
		return SFM

	def evaluate(self, model_name, outcome_header, model_type='categorical', model_file="trained_model.h5", hyperparameters=None, filters=None, checkpoint=None, eval_k_fold=None):
		'''Evaluates a saved model on a given set of tfrecords.'''
		log.header(f"Evaluating model {sfutil.bold(model_name)}...")
		model_root = join(self.PROJECT['models_dir'], model_name)
		if sfutil.path_to_name(model_file) != model_file:
			model_fullpath = join(model_root, model_file)
		else:
			model_fullpath= model_file

		# Load hyperparameters from saved model
		hp_file = hyperparameters if hyperparameters else join(model_root, 'hyperparameters.json')
		hp_data = sfutil.load_json(hp_file)
		hp = sfmodel.HyperParameters()
		hp._load_dict(hp_data['hp'])
		
		# Filter out slides that are blank in the outcome category
		filter_blank = [outcome_header] if type(outcome_header) != list else outcome_header

		# Load annotations / outcomes
		outcomes = sfutil.get_outcomes_from_annotations(outcome_header, filters=filters, filter_blank=filter_blank, use_float=(model_type=='linear'))

		# Load dataset for evaluation
		eval_dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])
		# If using a specific k-fold, load validation plan
		if eval_k_fold:
			log.info(f"Using {sfutil.bold('k-fold iteration ' + str(eval_k_fold))}", 1)
			validation_log = join(self.PROJECT['root'], "validation_plans.json")
			_, eval_tfrecords = sfutil.tfrecords.get_training_and_validation_tfrecords(eval_dataset, validation_log, outcomes, model_type,
																										validation_target=hp_data['validation_target'],
																										validation_strategy=hp_data['validation_strategy'],
																										validation_fraction=hp_data['validation_fraction'],
																										validation_k_fold=hp_data['validation_k_fold'],
																										k_fold_iter=eval_k_fold)
		# Otherwise use all TFRecords
		else:
			unfiltered_eval_tfrecords = eval_dataset.get_tfrecords(ask_to_merge_subdirs=True)
			eval_tfrecords = sfutil.filter_tfrecords_paths(unfiltered_eval_tfrecords, filters=filters)

		# Set up model for evaluation
		SFM = self.initialize_model(f"eval-{model_name}", eval_dataset, None, None, outcomes, model_type=model_type)
		log.info(f"Evaluating {sfutil.bold(len(eval_tfrecords))} tfrecords", 1)
		results = SFM.evaluate(tfrecords=eval_tfrecords, hp=hp, model=model_fullpath, model_type=model_type, checkpoint=checkpoint, batch_size=EVAL_BATCH_SIZE)
		return results

	def _get_hp(self, row, header):
		'''Internal function used to convert a row in the batch_train CSV file into a HyperParameters object.'''
		model_name_i = header.index('model_name')
		args = header[0:model_name_i] + header[model_name_i+1:]
		model_name = row[model_name_i]
		hp = sfmodel.HyperParameters()
		for arg in args:
			value = row[header.index(arg)]
			if arg in hp._get_args():
				if arg != 'finetune_epochs':
					arg_type = type(getattr(hp, arg))
					if arg_type == bool:
						if value.lower() in ['true', 'yes', 'y', 't']:
							bool_val = True
						elif value.lower() in ['false', 'no', 'n', 'f']:
							bool_val = False
						else:
							log.warn(f'Unable to parse arg "{arg}" with value "{value}" in batch training file into true/false; will default to True', 1)
							bool_val = True
						setattr(hp, arg, bool_val)
					else:
						setattr(hp, arg, arg_type(value))
				else:
					epochs = [int(i) for i in value.split(',')]
					setattr(hp, arg, epochs)
			else:
				log.error(f"Unknown argument '{arg}' found in training config file.", 0)
		return hp, model_name

	def _valid_hp(self, hp):
		if (hp.model_type() != 'categorical' and ((hp.balanced_training == sfmodel.BALANCE_BY_CATEGORY) or 
											    (hp.balanced_validation == sfmodel.BALANCE_BY_CATEGORY))):
			log.error(f'Invalid hyperparameter combination: balancing type "{sfmodel.BALANCE_BY_CATEGORY}" and model type "{hp.model_type()}".', 1)
			return False
		return True

	def _get_valid_models(self, batch_train_file, models):
		'''Internal function used to scan a batch_train file for valid, trainable models.'''
		models_to_train = []
		with open(batch_train_file) as csv_file:
			reader = csv.reader(csv_file, delimiter="\t")
			header = next(reader)
			try:
				model_name_i = header.index('model_name')
			except:
				log.error("Unable to find column 'model_name' in the batch training config file.", 0)
				sys.exit() 
			for row in reader:
				model_name = row[model_name_i]
				# First check if this row is a valid model
				if (not models) or (type(models)==str and model_name==models) or model_name in models:
					# Now verify there are no duplicate model names
					if model_name in models_to_train:
						log.error(f"Duplicate model names found in {sfutil.green(batch_train_file)}.", 0)
						sys.exit()
					models_to_train += [model_name]
		return models_to_train

	def _update_results_log(self, results_log_path, model_name, results_dict):
		'''Internal function used to dynamically update results_log when recording training metrics.'''
		# First, read current results log into a dictionary
		results_log = {}
		if exists(results_log_path):
			with open(results_log_path, "r") as results_file:
				reader = csv.reader(results_file)
				headers = next(reader)
				model_name_i = headers.index('model_name')
				result_keys = [k for k in headers if k != 'model_name']
				for row in reader:
					name = row[model_name_i]
					results_log[name] = {}
					for result_key in result_keys:
						result = row[headers.index(result_key)]
						results_log[name][result_key] = result
			# Move the current log file into a temporary file
			shutil.move(results_log_path, f"{results_log_path}.temp")

		# Next, update the results log with the new results data
		for epoch in results_dict:
			results_log.update({f'{model_name}-{epoch}': results_dict[epoch]})

		# Finally, create a new log file incorporating the new data
		with open(results_log_path, "w") as results_file:
			writer = csv.writer(results_file)
			result_keys = []
			# Search through results to find all results keys
			for model in results_log:
				result_keys += list(results_log[model].keys())
			# Remove duplicate result keys
			result_keys = list(set(result_keys))
			# Write header labels
			writer.writerow(['model_name'] + result_keys)
			# Iterate through model results and record
			for model in results_log:
				row = [model]
				# Include all saved metrics
				for result_key in result_keys:
					if result_key in results_log[model]:
						row += [results_log[model][result_key]]
					else:
						row += [""]
				writer.writerow(row)

		# Delete the old results log file
		if exists(f"{results_log_path}.temp"):
			os.remove(f"{results_log_path}.temp")

	def train(self, models=None, outcome_header='category', multi_outcome=False, filters=None, resume_training=None, checkpoint=None, 
				pretrain='imagenet', supervised=True, batch_file=None, hyperparameters=None, model_label=None, model_type='categorical',
				validation_target=None, validation_strategy=None, validation_fraction=None, validation_k_fold=None, k_fold_iter=None):
		'''Train model(s) given configurations found in batch_train.tsv.

		Args:
			models				(optional): Either string representing a model name or an array of strings containing model names. 
									Will train models with these names in the batch_train.tsv config file.
									Defaults to None, which will train all models in the batch_train.tsv config file.
			outcome_header		(optional): String or list. Specifies which header(s) in the annotation file to use for the output category. 
									Defaults to 'category'.	If a list is provided, will loop through all outcomes and perform HP sweep on each.
			multi_outcome		(optional): If True, will train to multiple outcomes simultaneously instead of looping through the
									list of outcomes in "outcome_header". Defaults to False.
			filters				(optional): Dictionary of column names mapping to column values by which to filter slides using the annotation file.
			resume_training		(optional): Path to .h5 model to continue training
			checkpoint			(optional): Path to cp.ckpt from which to load weights
			supervised			(optional): Whether to use verbose output and save training progress to Tensorboard
			batch_file			(optional): Manually specify batch file to use for a hyperparameter sweep. If not specified, will use project default.
			hyperparameters		(optional): Manually specify hyperparameter combination to use for training. If specified, will ignore batch training file.
			model_label			(optional): Name/label of model. Must be supplied if hyperparameters are provided.
			model_type			(optional): Type of output variable, either categorical (default) or linear.
			validation_target 	(optional): Whether to select validation data on a 'per-patient' or 'per-tile' basis. If not specified, will use project default.
			validation_strategy	(optional): Validation dataset selection strategy (bootstrap, k-fold, fixed, none). If not specified, will use project default.
			validation_fraction	(optional): Fraction of data to use for validation testing. If not specified, will use project default.
			validation_k_fold 	(optional): K, if using k-fold validation. If not specified, will use project default.
			k_fold_iter			(optional): Which iteration to train if using k-fold validation. Defaults to training all iterations.

		Returns:
			A dictionary containing model names mapped to train_acc, val_loss, and val_acc
		'''
		# Get list of slides for training and establish validation plan
		batch_train_file = self.PROJECT['batch_train_config'] if not batch_file else sfutil.global_path(batch_file)
		validation_target = self.PROJECT['validation_target'] if not validation_target else validation_target
		validation_strategy = self.PROJECT['validation_strategy'] if not validation_strategy else validation_strategy
		validation_fraction = self.PROJECT['validation_fraction'] if not validation_fraction else validation_fraction
		validation_k_fold = self.PROJECT['validation_k_fold'] if not validation_k_fold else validation_k_fold
		validation_log = join(self.PROJECT['root'], "validation_plans.json")
		results_log_path = os.path.join(self.PROJECT['root'], "results_log.csv")
		k_fold_iter = [k_fold_iter] if (k_fold_iter != None and type(k_fold_iter) != list) else k_fold_iter
		k_fold = validation_k_fold if validation_strategy in ('k-fold', 'bootstrap') else 0
		valid_k = [] if not k_fold else [kf for kf in range(k_fold) if ((k_fold_iter and kf in k_fold_iter) or (not k_fold_iter))]
		outcome_header = [outcome_header] if type(outcome_header) != list else outcome_header

		if multi_outcome and model_type != "linear":
			log.error("Multiple outcome variables only supported for linear outcome variables.")
			sys.exit()

		if hyperparameters and not model_label:
			log.error("If specifying hyperparameters, 'model_label' must be supplied. ", 1)
			return

		# Load dataset for training
		training_dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])

		# Quickly scan for errors (duplicate model names in batch training file) and prepare models to train
		log.header("Performing hyperparameter sweep...")
		if not hyperparameters:
			hp_models_to_train = self._get_valid_models(batch_train_file, models)
		else:
			hp_models_to_train = [model_label]

		if multi_outcome:
			log.info(f"Training ({len(hp_models_to_train)} models) using {len(outcome_header)} variables as simultaneous input:", 1)
		else:
			log.header(f"Training ({len(hp_models_to_train)} models) for each of {len(outcome_header)} outcome variables:", 1)
		for outcome in outcome_header:
			log.empty(outcome, 2)
		print()

		# Next, prepare the multiprocessing manager (needed to free VRAM after training and keep track of results)
		manager = multiprocessing.Manager()
		results_dict = manager.dict()

		# Create a worker that can execute one round of training
		def trainer(results_dict, outcomes, model_name, hp, k_fold_i=None):
			if supervised:
				k_fold_msg = "" if not k_fold_i else f" ({validation_strategy} iteration #{k_fold_i})"
				log.empty(f"Training model {sfutil.bold(model_name)}{k_fold_msg}...", 1)
				log.info(hp, 1)
			full_model_name = model_name if not k_fold_i else model_name+f"-kfold{k_fold_i}"

			# Initialize Comet experiment
			if not DEBUGGING:
				experiment = Experiment(COMET_API_KEY, project_name=self.PROJECT['name'])
				experiment.log_parameters(hp._get_dict())
				experiment.log_other('model_name', model_name)
				if k_fold_i:
					experiment.log_other('k_fold_iter', k_fold_i)

			# Get TFRecords for training and validation
			training_tfrecords, validation_tfrecords = sfutil.tfrecords.get_training_and_validation_tfrecords(training_dataset, validation_log, outcomes, model_type,
																										validation_target=validation_target,
																										validation_strategy=validation_strategy,
																										validation_fraction=validation_fraction,
																										validation_k_fold=validation_k_fold,
																										k_fold_iter=k_fold_i)
			# Initialize model
			SFM = self.initialize_model(full_model_name, training_dataset, training_tfrecords, validation_tfrecords, outcomes, model_type=model_type)

			# Log model settings and hyperparameters
			hp_file = join(self.PROJECT['models_dir'], full_model_name, 'hyperparameters.json')
			hp_data = {
				"model_name": model_name,
				"tile_px": self.PROJECT['tile_px'],
				"tile_um": self.PROJECT['tile_um'],
				"model_type": model_type,
				"dataset_config": self.PROJECT['dataset_config'],
				"datasets": self.PROJECT['datasets'],
				"annotations": self.PROJECT['annotations'],
				"validation_target": validation_target,
				"validation_strategy": validation_strategy,
				"validation_fraction": validation_fraction,
				"validation_k_fold": validation_k_fold,
				"k_fold_i": k_fold_i,
				"filters": filters,
				"hp": hp._get_dict()
			}
			sfutil.write_json(hp_data, hp_file)

			# Execute training
			try:
				results, keras = SFM.train(hp, pretrain=pretrain, 
													   resume_training=resume_training, 
													   checkpoint=checkpoint,
													   supervised=supervised)
				results_dict.update({full_model_name: results})
				logged_epochs = [int(e[5:]) for e in results['epochs'].keys() if e[:5] == 'epoch']
				
				if not DEBUGGING: experiment.log_metrics(results['epochs'][f'epoch{max(logged_epochs)}'])
				del(SFM)
				return keras
			except tf.errors.ResourceExhaustedError:
				log.error(f"Training failed for {sfutil.bold(model_name)}, GPU memory exceeded.", 0)
				del(SFM)
				return None

		def train_to_outcome(selected_outcome_headers):
			outcomes = sfutil.get_outcomes_from_annotations(selected_outcome_headers, filters=filters, 
																					  filter_blank=selected_outcome_headers,
																					  use_float=(model_type == 'linear'))
			print()
			
			# First, prepare hyperparameters and model names
			hyperparameter_list = []	
			if not hyperparameters:
				# Assembling list of models and hyperparameters from batch_train.tsv file
				batch_train_rows = []
				with open(batch_train_file) as csv_file:
					reader = csv.reader(csv_file, delimiter='\t')
					header = next(reader)
					for row in reader:
						batch_train_rows += [row]
					
				for row in batch_train_rows:
					# Read hyperparameters
					hp, hp_model_name = self._get_hp(row, header)
					if hp_model_name not in hp_models_to_train: continue

					# Verify HP combinations are valid
					if not self._valid_hp(hp):
						return

					hyperparameter_list += [[hp, hp_model_name]]
			else:
				if not self._valid_hp(hyperparameters):
					return
				hyperparameter_list = [[hyperparameters, model_label]]
			
			single_model = (len(hyperparameter_list) == 1) and (not k_fold or (k_fold and len(valid_k) == 1))
			keras_results = None

			for hp, hp_model_name in hyperparameter_list:
				# Generate model name
				outcome_string = "-".join(selected_outcome_headers) if type(selected_outcome_headers) == list else selected_outcome_headers
				model_name = f"{outcome_string}-{hp_model_name}"
				model_iterations = [model_name] if not k_fold else [f"{model_name}-kfold{k+1}" for k in valid_k]

				# Perform training
				if k_fold:
					for k in valid_k:
						if DEBUGGING or single_model:
							keras_results = trainer(results_dict, outcomes, model_name, hp, k+1)
						else:
							process = multiprocessing.Process(target=trainer, args=(results_dict, outcomes, model_name, hp, k+1))
							process.start()
							process.join()
				else:
					if DEBUGGING or single_model:
						keras_results = trainer(results_dict, outcomes, model_name, hp)
					else:
						process = multiprocessing.Process(target=trainer, args=(results_dict, outcomes, model_name, hp))
						process.start()
						process.join()

				# Record results
				for mi in model_iterations:
					if mi not in results_dict:
						log.error(f"Training failed for model {model_name} for an unknown reason")
					else:
						self._update_results_log(results_log_path, mi, results_dict[mi]['epochs'])
				log.complete(f"Training complete for model {model_name}, results saved to {sfutil.green(results_log_path)}")

			# Print summary of all models
			log.complete("Training complete; validation accuracies:", 0)
			for model in results_dict:
				last_epoch = max([int(e.split('epoch')[-1]) for e in results_dict[model]['epochs'].keys() if 'epoch' in e ])
				final_metrics = results_dict[model]['epochs'][f'epoch{last_epoch}']
				log.empty(f" - {sfutil.green(model)}: Train_Acc={str(final_metrics['train_acc'])}, " +
					f"Val_loss={final_metrics['val_loss']}, Val_Acc={final_metrics['val_acc']}" )
			return keras_results

		# If using multiple outcomes, initiate hyperparameter sweep
		if multi_outcome:
			keras_results = train_to_outcome(outcome_header)
			return results_dict, keras_results

		# If not training to multiple outcome, perform full hyperparameter sweep
		# for each outcome category specified
		else:
			for out in outcome_header:
				#results_dict.clear() # This is no longer needed since each outcome header will correspond to a different model name
				keras_results = train_to_outcome(out)
			return results_dict, keras_results
		
	def generate_heatmaps(self, model_name, filters=None, resolution='medium'):
		'''Creates predictive heatmap overlays on a set of slides. 

		Args:
			model_name:		Which model to use for generating predictions
			filter_header:	Column name for filtering input slides based on the project annotations file. 
			filter_values:	List of values to include when filtering slides according to filter_header.
			resolution:		Heatmap resolution (determines stride of tile predictions). 
								"low" uses a stride equal to tile width.
								"medium" uses a stride equal 1/2 tile width.
								"high" uses a stride equal to 1/4 tile width.
		'''
		import slideflow.convoluter as convoluter
		
		log.header("Generating heatmaps...")
		resolutions = {'low': 1, 'medium': 2, 'high': 4}
		try:
			stride_div = resolutions[resolution]
		except KeyError:
			log.error(f"Invalid resolution '{resolution}': must be either 'low', 'medium', or 'high'.")
			return

		# Load dataset for evaluation
		heatmaps_dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])
		unfiltered_slide_list = heatmaps_dataset.get_slide_paths()
		slide_list = sfutil.filter_slide_paths(unfiltered_slide_list, filters=filters)
		roi_list = heatmaps_dataset.get_rois()
		heatmaps_folder = os.path.join(self.PROJECT['root'], 'heatmaps')
		if not os.path.exists(heatmaps_folder): os.makedirs(heatmaps_folder)
		model_path = model_name if model_name[-3:] == ".h5" else join(self.PROJECT['models_dir'], model_name, 'trained_model.h5')

		c = convoluter.Convoluter(self.PROJECT['tile_px'], self.PROJECT['tile_um'], batch_size=64,
																					use_fp16=self.PROJECT['use_fp16'],
																					stride_div=stride_div,
																					save_folder=heatmaps_folder,
																					roi_list=roi_list)
		c.load_slides(slide_list)
		c.build_model(model_path)
		c.convolute_slides(save_heatmaps=True, save_final_layer=True, export_tiles=False)

	def generate_mosaic(self, model, filters=None, focus_filters=None, resolution="medium", num_tiles_x=50):
		'''Generates a mosaic map with dimensionality reduction on penultimate layer activations. Tile data is extracted from the provided
		set of TFRecords and predictions are calculated using the specified model.'''
		log.header("Generating mosaic map...")

		# Load dataset for evaluation
		mosaic_dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])
		mosaic_tfrecords = mosaic_dataset.get_tfrecords(ask_to_merge_subdirs=True)
		model_path = model if model[-3:] == ".h5" else join(self.PROJECT['models_dir'], model, 'trained_model.h5')

		tfrecords_list = sfutil.filter_tfrecords_paths(mosaic_tfrecords, filters=filters)
		if focus_filters:
			focus_list = sfutil.filter_tfrecords_paths(mosaic_tfrecords, filters=focus_filters)
		else:
			focus_list = None
		log.info(f"Generating mosaic from {len(tfrecords_list)} slides, with focus on {0 if not focus_list else len(focus_list)} slides.", 1)

		AV = ActivationsVisualizer(model=model_path,
								   tfrecords=tfrecords_list, 
								   root_dir=self.PROJECT['root'],
								   image_size=self.PROJECT['tile_px'],
								   focus_nodes=None,
								   use_fp16=self.PROJECT['use_fp16'])

		AV.generate_mosaic(focus=focus_list,
						   num_tiles_x=num_tiles_x,
						   resolution=resolution)

		return AV

	def generate_activations_analytics(self, model, outcome_header, filters=None, focus_nodes=[], node_exclusion=False):
		'''Calculates final layer activations and displays information regarding the most significant final layer nodes.'''
		log.header("Generating final layer activation analytics...")

		# Load dataset for evaluation
		activations_dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])
		activations_tfrecords = activations_dataset.get_tfrecords(ask_to_merge_subdirs=True)
		model_path = model if model[-3:] == ".h5" else join(self.PROJECT['models_dir'], model, 'trained_model.h5')

		tfrecords_list = sfutil.filter_tfrecords_paths(activations_tfrecords, filters=filters)
		log.info(f"Visualizing activations from {len(tfrecords_list)} slides", 1)

		AV = ActivationsVisualizer(model=model_path,
								   tfrecords=tfrecords_list,
								   root_dir=self.PROJECT['root'],
								   image_size=self.PROJECT['tile_px'],
								   annotations=self.PROJECT['annotations'],
								   category_header=outcome_header,
								   focus_nodes=focus_nodes,
								   use_fp16=self.PROJECT['use_fp16'])

		return AV

	def visualize_tiles(self, model, directory, node, num_to_visualize=20, window=None):
		tiles = [o for o in os.listdir(directory) if not isdir(join(directory, o))]
		tiles.sort(key=lambda x: int(x.split('-')[0]))
		tiles.reverse()

		TV = TileVisualizer(model=model, 
							node=node,
							shape=[self.PROJECT['tile_px'], self.PROJECT['tile_px'], 3],
							tile_width=window)

		for tile in tiles[:20]:
			tile_loc = join(directory, tile)
			TV.visualize_tile(tile_loc, save_dir=directory)
 
	def create_blank_train_config(self, filename=None):
		'''Creates a CSV file with the batch training structure.'''
		if not filename:
			filename = self.PROJECT['batch_train_config']
		with open(filename, 'w') as csv_outfile:
			writer = csv.writer(csv_outfile, delimiter='\t')
			# Create headers and first row
			header = ['model_name']
			firstrow = ['model1']
			default_hp = sfmodel.HyperParameters()
			for arg in default_hp._get_args():
				header += [arg]
				firstrow += [getattr(default_hp, arg)]
			writer.writerow(header)
			writer.writerow(firstrow)

	def create_hyperparameter_sweep(self, finetune_epochs, toplayer_epochs, model, pooling, loss, learning_rate, batch_size, hidden_layers,
									optimizer, early_stop, early_stop_patience, balanced_training, balanced_validation, augment, filename=None):
		'''Prepares a hyperparameter sweep using the batch train config file.'''
		log.header("Preparing hyperparameter sweep...")
		# Assemble all possible combinations of provided hyperparameters
		pdict = locals()
		del(pdict['self'])
		del(pdict['filename'])
		del(pdict['finetune_epochs'])
		args = list(pdict.keys())
		args.reverse()
		for arg in args:
			if type(pdict[arg]) != list:
				pdict[arg] = [pdict[arg]]
		argsv = list(pdict.values())
		argsv.reverse()
		sweep = list(itertools.product(*argsv))

		if not filename:
			filename = self.PROJECT['batch_train_config']
		with open(filename, 'w') as csv_outfile:
			writer = csv.writer(csv_outfile, delimiter='\t')
			# Create headers
			header = ['model_name', 'finetune_epochs']
			for arg in args:
				header += [arg]
			writer.writerow(header)
			# Iterate through sweep
			for i, params in enumerate(sweep):
				row = [f'HPSweep{i}', ','.join([str(f) for f in finetune_epochs])]
				full_params = [finetune_epochs] + list(params)
				hp = sfmodel.HyperParameters(*full_params)
				for arg in args:
					row += [getattr(hp, arg)]
				writer.writerow(row)
		log.complete(f"Wrote {len(sweep)} combinations for sweep to {sfutil.green(filename)}")

	def resize_tfrecords(self, size, filters=None):
		log.header(f"Resizing TFRecord tiles to ({size}, {size})")
		resize_dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])
		resize_tfrecords = resize_dataset.get_tfrecords()
		tfrecords_list = sfutil.filter_tfrecords_paths(resize_tfrecords, filters=filters)

		log.info(f"Resizing {len(tfrecords_list)} tfrecords", 1)

		for tfr in tfrecords_list:
			sfutil.tfrecords.transform_tfrecord(tfr, tfr+".transformed", resize=size)

	def create_blank_annotations_file(self, outfile=None):
		'''Creates an example blank annotations file.'''
		if not outfile: 
			outfile = self.PROJECT['annotations']
		with open(outfile, 'w') as csv_outfile:
			csv_writer = csv.writer(csv_outfile, delimiter=',')
			header = [TCGA.patient, 'dataset', 'category']
			csv_writer.writerow(header)

	def associate_slide_names(self):
		'''Experimental function used to automatically associated patient names with slide filenames in the annotations file.'''
		# Load dataset
		dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])

		sfutil.update_annotations_with_slidenames(self.PROJECT['annotations'], dataset)
		sfutil.load_annotations(self.PROJECT['annotations'], dataset)

	def update_manifest(self, force_update=False):
		'''Updates manifest file in the TFRecord directory, used to track number of records and verify annotations.
		
		Args:
			force_update:	If True, will re-validate contents of all TFRecords. If False, will only validate
								contents of TFRecords not yet in the manifest
		'''
		tfrecords_folders = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets']).get_tfrecords_folders()
		for tfr_folder in tfrecords_folders:
			sfutil.update_tfrecord_manifest(directory=tfr_folder, 
											force_update=force_update)

	def load_project(self, directory):
		'''Loads a saved and pre-configured project.'''
		if exists(join(directory, "settings.json")):
			self.PROJECT = sfutil.load_json(join(directory, "settings.json"))
			log.empty("Project configuration loaded.\n")
		else:
			raise OSError(f'Unable to locate settings.json at location "{directory}".')

		# Enable logging
		log.logfile = sfutil.global_path("log.log")

		# Load dataset for evaluation
		try:
			dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])
			# Load annotations
			sfutil.load_annotations(self.PROJECT['annotations'], dataset)

			if not SKIP_VERIFICATION:
				log.header("Verifying Annotations...")
				sfutil.verify_annotations_slides(dataset)
				log.header("Verifying TFRecord manifest...")
				self.update_manifest()
		except FileNotFoundError:
			log.warn("No datasets configured.")

	def load_datasets(self, path):
		try:
			datasets_data = sfutil.load_json(path)
			datasets_names = list(datasets_data.keys())
			datasets_names.sort()
		except FileNotFoundError:
			datasets_data = {}
			datasets_names = []
		return datasets_data, datasets_names

	def add_dataset(self, name, slides, roi, tiles, tfrecords, label, path=None):
		if not path:
			path = self.PROJECT['dataset_config']
		try:
			datasets_data = sfutil.load_json(path)
		except FileNotFoundError:
			datasets_data = {}
		datasets_data.update({name: {
			'slides': slides,
			'roi': roi,
			'tiles': tiles,
			'tfrecords': tfrecords,
			'label': label
		}})
		sfutil.write_json(datasets_data, path)
		log.info(f"Saved dataset {name} to {path}")

	def save_project(self):
		'''Saves current project configuration as "settings.json".'''
		sfutil.write_json(self.PROJECT, join(self.PROJECT['root'], 'settings.json'))

	def create_project(self):
		'''Prompts user to provide all relevant project configuration and saves configuration to "settings.json".'''
		# General setup and slide configuration
		project = {'root': sfutil.PROJECT_DIR}
		project['name'] = input("What is the project name? ")
		
		# Ask for annotations file location; if one has not been made, offer to create a blank template and then exit
		if not sfutil.yes_no_input("Has an annotations (CSV) file already been created? [y/N] ", default='no'):
			if sfutil.yes_no_input("Create a blank annotations file? [Y/n] ", default='yes'):
				project['annotations'] = sfutil.file_input("Where will the annotation file be located? [./annotations.csv] ", 
									default='./annotations.csv', filetype="csv", verify=False)
				self.create_blank_annotations_file(project['annotations'])
		else:
			project['annotations'] = sfutil.file_input("Where is the project annotations (CSV) file located? [./annotations.csv] ", 
									default='./annotations.csv', filetype="csv")

		# Dataset configuration
		project['dataset_config'] = sfutil.file_input("Where is the dataset configuration file located? [./datasets.json] ",
													default='./datasets.json', filetype='json', verify=False)

		project['datasets'] = []
		while not project['datasets']:
			datasets_data, datasets_names = self.load_datasets(project['dataset_config'])

			print(sfutil.bold("Detected datasets:"))
			if not len(datasets_names):
				print(" [None]")
			else:
				for i, name in enumerate(datasets_names):
					print(f" {i+1}. {name}")
				print(f" {len(datasets_names)+1}. ADD NEW")
				dataset_selection = sfutil.choice_input(f"Which datasets should be used? (choose {len(datasets_names)+1} to add a new dataset) ", valid_choices=[str(l) for l in list(range(1, len(datasets_names)+2))], multi_choice=True)

			if not len(datasets_names) or str(len(datasets_names)+1) in dataset_selection:
				# Create new dataset
				print(f"{sfutil.bold('Creating new dataset')}")
				dataset_name = input("What is the dataset name? ")
				dataset_slides = sfutil.dir_input("Where are the slides stored? [./slides] ",
										default='./slides', create_on_invalid=True)
				dataset_roi = sfutil.dir_input("Where are the ROI files (CSV) stored? [./slides] ",
										default='./slides', create_on_invalid=True)
				dataset_tiles = sfutil.dir_input("Where will the tessellated image tiles be stored? (recommend SSD) [./tiles] ",
										default='./tiles', create_on_invalid=True)
				dataset_tfrecords = sfutil.dir_input("Where should the TFRecord files be stored? (recommend HDD) [./tfrecord] ",
										default='./tfrecord', create_on_invalid=True)

				self.add_dataset(name=dataset_name,
								 slides=dataset_slides,
								 roi=dataset_roi,
								 tiles=dataset_tiles,
								 tfrecords=dataset_tfrecords,
								 label=NO_LABEL,
								 path=project['dataset_config'])

				print("Updated dataset configuration file.")
			else:
				try:
					project['datasets'] = [datasets_names[int(j)-1] for j in dataset_selection]
				except TypeError:
					print(f'Invalid selection: {dataset_selection}')
					continue

		# Other tessellation
		project['delete_tiles'] = sfutil.yes_no_input("Should raw tile images be deleted after TFRecord storage? [Y/n] ", default='yes')

		# Training
		project['models_dir'] = sfutil.dir_input("Where should the saved models be stored? [./models] ",
									default='./models', create_on_invalid=True)
		project['tile_um'] = sfutil.int_input("What is the tile width in microns? [280] ", default=280)
		project['tile_px'] = sfutil.int_input("What is the tile width in pixels? [224] ", default=224)
		project['use_fp16'] = sfutil.yes_no_input("Should FP16 be used instead of FP32? (recommended) [Y/n] ", default='yes')
		project['batch_train_config'] = sfutil.file_input("Location for the batch training TSV config file? [./batch_train.tsv] ",
													default='./batch_train.tsv', filetype='tsv', verify=False)
		
		if not exists(project['batch_train_config']):
			print("Batch training file not found, creating blank")
			self.create_blank_train_config(project['batch_train_config'])
		
		# Validation strategy
		project['validation_fraction'] = sfutil.float_input("What fraction of training data should be used for validation testing? [0.2] ", valid_range=[0,1], default=0.2)
		project['validation_target'] = sfutil.choice_input("How should validation data be selected by default, per-tile or per-patient? [per-patient] ", valid_choices=['per-tile', 'per-patient'], default='per-patient')
		if project['validation_target'] == 'per-patient':
			project['validation_strategy'] = sfutil.choice_input("Which validation strategy should be used by default, k-fold, bootstrap, or fixed? [k-fold]", valid_choices=['k-fold', 'bootstrap', 'fixed', 'none'], default='k-fold')
		else:
			project['validation_strategy'] = sfutil.choice_input("Which validation strategy should be used by default, k-fold or fixed? ", valid_choices=['k-fold', 'fixed', 'none'])
		if project['validation_strategy'] == 'k-fold':
			project['validation_k_fold'] = sfutil.int_input("What is K? [3] ", default=3)
		elif project['validation_strategy'] == 'bootstrap':
			project['validation_k_fold'] = sfutil.int_input("How many iterations should be performed when bootstrapping? [3] ", default=3)
		else:
			project['validation_k_fold'] = 0

		sfutil.write_json(project, join(sfutil.PROJECT_DIR, 'settings.json'))
		self.PROJECT = project

		# Write a sample actions.py file
		with open(join(SOURCE_DIR, 'sample_actions.py'), 'r') as sample_file:
			sample_actions = sample_file.read()
			with open(os.path.join(sfutil.PROJECT_DIR, 'actions.py'), 'w') as actions_file:
				actions_file.write(sample_actions)

		print("\nProject configuration saved.\n")
		self.load_project(sfutil.PROJECT_DIR)
