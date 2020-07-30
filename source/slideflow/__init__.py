import os
import io
import sys
import shutil
import logging
import gc
import atexit
import itertools
import warnings
import csv
import numpy as np
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing

from os.path import join, isfile, exists, isdir, dirname
from pathlib import Path
from glob import glob
from random import shuffle, choice
from string import ascii_lowercase
from multiprocessing.dummy import Pool as DPool
from functools import partial

import slideflow.model as sfmodel
import slideflow.util as sfutil
import slideflow.io as sfio

from slideflow.io.datasets import Dataset
from slideflow.util import TCGA, ProgressBar, log
from slideflow.activations import ActivationsVisualizer, TileVisualizer, Heatmap
from slideflow.statistics import TFRecordUMAP, calculate_centroid
from slideflow.mosaic import Mosaic
from comet_ml import Experiment

# TODO: change Dataset handling to exclude tile size specificity in naming

__version__ = "1.8.2"

NO_LABEL = 'no_label'
SILENT = 'SILENT'
SOURCE_DIR = os.path.dirname(os.path.realpath(__file__))
COMET_API_KEY = "A3VWRcPaHgqc4H5K0FoCtRXbp"

DEFAULT_FLAGS = {
	'use_comet': False,
	'skip_verification': False,
	'eval_batch_size': 64,
	'num_threads': 4
}

def evaluator(outcome_header, model, project_config, results_dict, filters=None, 
				hyperparameters=None, checkpoint=None, eval_k_fold=None, max_tiles_per_slide=0,
				min_tiles_per_slide=0, flags=None):

	if not flags: flags = DEFAULT_FLAGS

	model_root = dirname(model)

	# Load hyperparameters from saved model
	hp_file = hyperparameters if hyperparameters else join(model_root, 'hyperparameters.json')
	hp_data = sfutil.load_json(hp_file)
	hp = sfmodel.HyperParameters()
	hp._load_dict(hp_data['hp'])
	model_name = f"eval-{hp_data['model_name']}-{sfutil.path_to_name(model)}"
	model_type = hp.model_type()

	# Filter out slides that are blank in the outcome category
	filter_blank = [outcome_header] if not isinstance(outcome_header, list) else outcome_header

	# Load dataset and annotations for evaluation
	eval_dataset = Dataset(config_file=project_config['dataset_config'],
						   sources=project_config['datasets'],
						   annotations=project_config['annotations'],
						   filters=filters)
	outcomes, unique_outcomes = eval_dataset.get_outcomes_from_annotations(outcome_header, use_float=(hp.model_type()=='linear'))

	# If using a specific k-fold, load validation plan
	if eval_k_fold:
		log.info(f"Using {sfutil.bold('k-fold iteration ' + str(eval_k_fold))}", 1)
		validation_log = join(project_config['root'], "validation_plans.json")
		_, eval_tfrecords = sfio.tfrecords.get_training_and_validation_tfrecords(eval_dataset, validation_log, outcomes, hp.model_type(),
																									validation_target=hp_data['validation_target'],
																									validation_strategy=hp_data['validation_strategy'],
																									validation_fraction=hp_data['validation_fraction'],
																									validation_k_fold=hp_data['validation_k_fold'],
																									k_fold_iter=eval_k_fold)
	# Otherwise use all TFRecords
	else:
		eval_tfrecords = eval_dataset.get_tfrecords(merge_subdirs=True)

	# Set up model for evaluation
	# Using the project annotation file, assemble list of slides for training, as well as the slide annotations dictionary (output labels)
	model_dir = join(project_config['models_dir'], model_name)

	# Build a model using the slide list as input and the annotations dictionary as output labels
	SFM = sfmodel.SlideflowModel(model_dir, project_config['tile_px'], outcomes, 
																		train_tfrecords=None,
																		validation_tfrecords=eval_tfrecords,
																		manifest=eval_dataset.get_manifest(),
																		use_fp16=project_config['use_fp16'],
																		model_type=hp.model_type())

	# Log model settings and hyperparameters
	hp_file = join(model_dir, 'hyperparameters.json')
	hp_data = {
		"model_name": model_name,
		"model_path": model,
		"stage": "evaluation",
		"tile_px": project_config['tile_px'],
		"tile_um": project_config['tile_um'],
		"model_type": hp.model_type(),
		"outcome_headers": outcome_header,
		"outcome_labels": None if hp.model_type() != 'categorical' else dict(zip(range(len(unique_outcomes)), unique_outcomes)),
		"dataset_config": project_config['dataset_config'],
		"datasets": project_config['datasets'],
		"annotations": project_config['annotations'],
		"validation_target": hp_data['validation_target'],
		"validation_strategy": hp_data['validation_strategy'],
		"validation_fraction": hp_data['validation_fraction'],
		"validation_k_fold": hp_data['validation_k_fold'],
		"k_fold_i": eval_k_fold,
		"filters": filters,
		"pretrain": None,
		"resume_training": None,
		"checkpoint": checkpoint,
		"comet_experiment": None,
		"hp": hp._get_dict()
	}
	sfutil.write_json(hp_data, hp_file)

	# Perform evaluation
	log.info(f"Evaluating {sfutil.bold(len(eval_tfrecords))} tfrecords", 1)
	
	results = SFM.evaluate(tfrecords=eval_tfrecords, 
						   hp=hp,
						   model=model,
						   model_type=hp.model_type(),
						   checkpoint=checkpoint,
						   batch_size=flags['eval_batch_size'],
						   max_tiles_per_slide=max_tiles_per_slide,
						   min_tiles_per_slide=min_tiles_per_slide)

	# Load results into multiprocessing dictionary
	results_dict['results'] = results
	return results_dict

def heatmap_generator(slide, model_name, model_path, save_folder, roi_list, show_roi, resolution, interpolation, project_config, 
						logit_cmap=None, skip_thumb=False, flags=None):
	import slideflow.slide as sfslide
	if not flags: flags = DEFAULT_FLAGS

	resolutions = {'low': 1, 'medium': 2, 'high': 4}
	try:
		stride_div = resolutions[resolution]
	except KeyError:
		log.error(f"Invalid resolution '{resolution}': must be either 'low', 'medium', or 'high'.")
		return

	if exists(join(save_folder, f'{sfutil.path_to_name(slide)}-custom.png')):
		log.empty(f"Skipping already-completed heatmap for slide {sfutil.path_to_name(slide)}", 1)
		return

	heatmap = Heatmap(slide, model_path, project_config['tile_px'], project_config['tile_um'], 
																	use_fp16=project_config['use_fp16'],
																	stride_div=stride_div,
																	save_folder=save_folder,
																	roi_list=roi_list,
																	thumb_folder=join(project_config['root'], 'thumbs'))

	heatmap.generate(batch_size=flags['eval_batch_size'], skip_thumb=skip_thumb)
	heatmap.save(show_roi=show_roi, interpolation=interpolation, logit_cmap=logit_cmap, skip_thumb=skip_thumb)

def trainer(outcome_headers, model_name, project_config, results_dict, hp, validation_strategy, 
			validation_target, validation_fraction, validation_k_fold, validation_log, validation_dataset=None, 
			validation_annotations=None, validation_filters=None, k_fold_i=None, filters=None, pretrain=None, 
			resume_training=None, checkpoint=None, validate_on_batch=0, validation_steps=200, max_tiles_per_slide=0, 
			min_tiles_per_slide=0, starting_epoch=0, flags=None):

	import tensorflow as tf

	if not flags: flags = DEFAULT_FLAGS

	# First, clear prior Tensorflow graph to free memory
	tf.keras.backend.clear_session()

	# Log current model name and k-fold iteration, if applicable
	k_fold_msg = "" if not k_fold_i else f" ({validation_strategy} iteration #{k_fold_i})"
	log.empty(f"Training model {sfutil.bold(model_name)}{k_fold_msg}...")
	log.empty(hp, 1)
	full_model_name = model_name if not k_fold_i else model_name+f"-kfold{k_fold_i}"

	# Initialize Comet experiment
	if flags['use_comet']:
		experiment = Experiment(COMET_API_KEY, project_name=project_config['name'])
		experiment.log_parameters(hp._get_dict())
		experiment.log_other('model_name', model_name)
		if k_fold_i:
			experiment.log_other('k_fold_iter', k_fold_i)

	# Load dataset and annotations for training
	training_dataset = Dataset(config_file=project_config['dataset_config'],
							   sources=project_config['datasets'],
							   annotations=project_config['annotations'],
							   filters=filters,
							   filter_blank=outcome_headers)

	# Load outcomes
	outcomes, unique_outcomes = training_dataset.get_outcomes_from_annotations(outcome_headers, use_float=(hp.model_type() == 'linear'))
	if hp.model_type() == 'categorical': 
		outcome_labels = dict(zip(range(len(unique_outcomes)), unique_outcomes))
	else:
		outcome_labels = dict(zip(range(len(outcome_headers)), outcome_headers))

	# Get TFRecords for training and validation
	manifest = training_dataset.get_manifest()
	training_tfrecords, validation_tfrecords = sfio.tfrecords.get_training_and_validation_tfrecords(training_dataset, validation_log, outcomes, hp.model_type(),
																									validation_target=validation_target,
																									validation_strategy=validation_strategy,
																									validation_fraction=validation_fraction,
																									validation_k_fold=validation_k_fold,
																									k_fold_iter=k_fold_i)
	# Use external validation dataset if specified
	if validation_dataset:
		validation_dataset = Dataset(config_file=project_config['dataset_config'],
									 sources=validation_dataset,
									 annotations=validation_annotations,
									 filters=validation_filters,
									 filter_blank=outcome_headers)
		validation_tfrecords = validation_dataset.get_tfrecords()
		manifest.update(validation_dataset.get_manifest())
		validation_outcomes, _ = validation_dataset.get_outcomes_from_annotations(outcome_headers, use_float=(hp.model_type() == 'linear'))
		outcomes.update(validation_outcomes)

	# Initialize model
	# Using the project annotation file, assemble list of slides for training, as well as the slide annotations dictionary (output labels)
	model_dir = join(project_config['models_dir'], full_model_name)

	# Build a model using the slide list as input and the annotations dictionary as output labels
	SFM = sfmodel.SlideflowModel(model_dir, project_config['tile_px'], outcomes, training_tfrecords, validation_tfrecords,
																			manifest=manifest,
																			use_fp16=project_config['use_fp16'],
																			model_type=hp.model_type())

	# Log model settings and hyperparameters
	hp_file = join(project_config['models_dir'], full_model_name, 'hyperparameters.json')
	hp_data = {
		"model_name": model_name,
		"stage": "training",
		"tile_px": project_config['tile_px'],
		"tile_um": project_config['tile_um'],
		"model_type": hp.model_type(),
		"outcome_headers": outcome_headers,
		"outcome_labels": outcome_labels,
		"dataset_config": project_config['dataset_config'],
		"datasets": project_config['datasets'],
		"annotations": project_config['annotations'],
		"validation_target": validation_target,
		"validation_strategy": validation_strategy,
		"validation_fraction": validation_fraction,
		"validation_k_fold": validation_k_fold,
		"k_fold_i": k_fold_i,
		"filters": filters,
		"pretrain": pretrain,
		"resume_training": resume_training,
		"checkpoint": checkpoint,
		"comet_experiment": None if not flags['use_comet'] else experiment.get_key(),
		"hp": hp._get_dict()
	}
	sfutil.write_json(hp_data, hp_file)

	# Execute training
	try:
		results, history = SFM.train(hp, pretrain=pretrain, 
										 resume_training=resume_training, 
										 checkpoint=checkpoint,
										 validate_on_batch=validate_on_batch,
										 validation_steps=validation_steps,
										 max_tiles_per_slide=max_tiles_per_slide,
										 min_tiles_per_slide=min_tiles_per_slide,
										 starting_epoch=starting_epoch)
		results['history'] = history
		results_dict.update({full_model_name: results})
		logged_epochs = [int(e[5:]) for e in results['epochs'].keys() if e[:5] == 'epoch']
		
		if flags['use_comet']: experiment.log_metrics(results['epochs'][f'epoch{max(logged_epochs)}'])
		del(SFM)
		return history
	except tf.errors.ResourceExhaustedError:
		log.empty("\n")
		log.error(f"Training failed for {sfutil.bold(model_name)}, GPU memory exceeded.", 0)
		del(SFM)
		return None

class SlideflowProject:
	FLAGS = DEFAULT_FLAGS
	GPU_LOCK = None

	def __init__(self, project_folder, num_gpu=1, reverse_select_gpu=True, force_gpu=None, interactive=True):
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
					project_folder = sfutil.dir_input("Where is the project root directory? ", None, create_on_invalid=True, absolute=True)
			else:
				log.info(f"Project directory {project_folder} not found; will create.")
				os.makedirs(project_folder)
		if not project_folder:
			project_folder = sfutil.dir_input("Where is the project root directory? ", None, create_on_invalid=True, absolute=True)

		if exists(join(project_folder, "settings.json")):
			self.load_project(project_folder)
		elif interactive:
			self.create_project(project_folder)

		# Set up GPU
		if force_gpu is not None:
			self.select_gpu(force_gpu)
		else:
			self.autoselect_gpu(num_gpu, reverse=reverse_select_gpu)
		atexit.register(self.release_gpu)

	def autoselect_gpu(self, number_available, reverse=True):
		'''Automatically claims a free GPU and creates a lock file to prevent 
		other instances of slideflow from using the same GPU.'''
		log.header("Selecting GPU...")

		if not number_available:
			os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
			log.warn(f"Disabling GPU access.")
		else:
			gpus = range(number_available) if not reverse else reversed(range(number_available))
			gpu_selected = -1
			for n in gpus:
				if not exists(join(SOURCE_DIR, f"gpu{n}.lock")):
					self.select_gpu(n)
					gpu_selected = n
					break
			if gpu_selected == -1 and number_available:
				log.warn(f"No GPU selected; tried selecting one from a user-specified pool of {number_available}.", 1)
				log.empty(f"Try deleting 'gpu[#].lock' files in {sfutil.green(SOURCE_DIR)} if GPUs are not in use.", 2)
		import tensorflow as tf

	def release_gpu(self):
		log.header("Cleaning up...")
		if self.GPU_LOCK != None and exists(join(SOURCE_DIR, f"gpu{self.GPU_LOCK}.lock")):
			log.empty(f"Freeing GPU {self.GPU_LOCK}...", 1)
			os.remove(join(SOURCE_DIR, f"gpu{self.GPU_LOCK}.lock"))

	def select_gpu(self, number):
		if number == -1:
			os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
			log.warn(f"Disabling GPU access.")
		else:
			log.empty(f"Using GPU #{number}", 1)
			self.GPU_LOCK = number
			os.environ["CUDA_VISIBLE_DEVICES"]=str(number)
			open(join(SOURCE_DIR, f"gpu{number}.lock"), 'a').close()
		
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

	def _get_hyperparameter_combinations(self, hyperparameters, models, batch_train_file):
		'''Returns list of hyperparameters ojects and associated models names, either from specified hyperparameters or from a batch_train file
		if hyperparameters is None.'''
		if not hyperparameters:
			hp_models_to_train = self._get_valid_models(batch_train_file, models)
		else:
			hp_models_to_train = [models]

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
				try:
					hp, hp_model_name = self._get_hp(row, header)
				except sfmodel.HyperParameterError as e:
					log.error("Invalid Hyperparameter combination: " + str(e))
					return

				if hp_model_name not in hp_models_to_train: continue

				hyperparameter_list += [[hp, hp_model_name]]
		elif isinstance(hyperparameters, list) and isinstance(models, list):
			if len(models) != len(hyperparameters):
				log.error(f"Unable to iterate through hyperparameters provided; length of hyperparameters ({len(hyperparameters)}) much match length of models ({len(models)})", 1)
				return
			for i in range(len(models)):
				if not hyperparameters[i].validate():
					return
				hyperparameter_list += [[hyperparameters[i], models[i]]]
		else:
			if not hyperparameters.validate():
				return
			hyperparameter_list = [[hyperparameters, models]]
		return hyperparameter_list

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
				if (not models) or (isinstance(models, str) and model_name==models) or model_name in models:
					# Now verify there are no duplicate model names
					if model_name in models_to_train:
						log.error(f"Duplicate model names found in {sfutil.green(batch_train_file)}.", 0)
						sys.exit()
					models_to_train += [model_name]
		return models_to_train

	def add_dataset(self, name, slides, roi, tiles, tfrecords, label, path=None):
		'''Adds a dataset to the dataset configuration file.

		Args:
			name:		Dataset name.
			slides:		Path to directory containing slides.
			roi:		Path to directory containing CSV ROIs.
			tiles:		Path to directory in which to store extracted tiles.
			tfrecords:	Path to directory in which to store TFRecords of extracted tiles.
			label:		Label to give dataset (typically identifies tile size in pixels/microns).'''

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

	def associate_slide_names(self):
		'''Experimental function used to automatically associated patient names with slide filenames in the annotations file.'''
		log.header("Associating slide names...")
		# Load dataset
		dataset = self.get_dataset()
		dataset.update_annotations_with_slidenames(self.PROJECT['annotations'])
		#dataset.load_annotations()

	def create_blank_annotations_file(self, outfile=None):
		'''Creates an example blank annotations file.'''
		if not outfile: 
			outfile = self.PROJECT['annotations']
		with open(outfile, 'w') as csv_outfile:
			csv_writer = csv.writer(csv_outfile, delimiter=',')
			header = [TCGA.patient, 'dataset', 'category']
			csv_writer.writerow(header)

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
									optimizer, early_stop, early_stop_patience, early_stop_method, balanced_training, balanced_validation, 
									augment, hidden_layer_width, trainable_layers, L2_weight, filename=None):
		'''Prepares a hyperparameter sweep using the batch train config file.'''
		log.header("Preparing hyperparameter sweep...")
		# Assemble all possible combinations of provided hyperparameters
		pdict = locals()
		del(pdict['self'])
		del(pdict['filename'])
		del(pdict['finetune_epochs'])
		args = list(pdict.keys())
		for arg in args:
			if not isinstance(pdict[arg], list):
				pdict[arg] = [pdict[arg]]
		argsv = list(pdict.values())
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
				full_params = dict(zip(['finetune_epochs'] + args, [finetune_epochs] + list(params)))
				hp = sfmodel.HyperParameters(**full_params)
				for arg in args:
					row += [getattr(hp, arg)]
				writer.writerow(row)
		log.complete(f"Wrote {len(sweep)} combinations for sweep to {sfutil.green(filename)}")

	def create_project(self, project_folder):
		'''Prompts user to provide all relevant project configuration and saves configuration to "settings.json".'''
		# General setup and slide configuration
		project = {
			'root': project_folder,
			'slideflow_version': __version__
		}
		project['name'] = input("What is the project name? ")
		
		# Ask for annotations file location; if one has not been made, offer to create a blank template and then exit
		if not sfutil.yes_no_input("Has an annotations (CSV) file already been created? [y/N] ", default='no'):
			if sfutil.yes_no_input("Create a blank annotations file? [Y/n] ", default='yes'):
				project['annotations'] = sfutil.file_input("Where will the annotation file be located? [./annotations.csv] ", 
									root=project['root'], default='./annotations.csv', filetype="csv", verify=False)
				self.create_blank_annotations_file(project['annotations'])
		else:
			project['annotations'] = sfutil.file_input("Where is the project annotations (CSV) file located? [./annotations.csv] ", 
									root=project['root'], default='./annotations.csv', filetype="csv")

		# Dataset configuration
		project['dataset_config'] = sfutil.file_input("Where is the dataset configuration file located? [./datasets.json] ",
													root=project['root'], default='./datasets.json', filetype='json', verify=False)

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
										root=project['root'], default='./slides', create_on_invalid=True)
				dataset_roi = sfutil.dir_input("Where are the ROI files (CSV) stored? [./slides] ",
										root=project['root'], default='./slides', create_on_invalid=True)
				dataset_tiles = sfutil.dir_input("Where will the tessellated image tiles be stored? (recommend SSD) [./tiles] ",
										root=project['root'], default='./tiles', create_on_invalid=True)
				dataset_tfrecords = sfutil.dir_input("Where should the TFRecord files be stored? (recommend HDD) [./tfrecord] ",
										root=project['root'], default='./tfrecord', create_on_invalid=True)

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

		# Training
		project['models_dir'] = sfutil.dir_input("Where should the saved models be stored? [./models] ",
									root=project['root'], default='./models', create_on_invalid=True)
		project['tile_um'] = sfutil.int_input("What is the tile width in microns? [280] ", default=280)
		project['tile_px'] = sfutil.int_input("What is the tile width in pixels? [224] ", default=224)
		project['use_fp16'] = sfutil.yes_no_input("Should FP16 be used instead of FP32? (recommended) [Y/n] ", default='yes')
		project['batch_train_config'] = sfutil.file_input("Location for the batch training TSV config file? [./batch_train.tsv] ",
													root=project['root'], default='./batch_train.tsv', filetype='tsv', verify=False)
		
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

		sfutil.write_json(project, join(project_folder, 'settings.json'))
		self.PROJECT = project

		# Write a sample actions.py file
		with open(join(SOURCE_DIR, 'sample_actions.py'), 'r') as sample_file:
			sample_actions = sample_file.read()
			with open(os.path.join(project_folder, 'actions.py'), 'w') as actions_file:
				actions_file.write(sample_actions)

		print("\nProject configuration saved.\n")
		self.load_project(project_folder)

	def evaluate(self, model, outcome_header, hyperparameters=None, filters=None, checkpoint=None,
					eval_k_fold=None, max_tiles_per_slide=0, min_tiles_per_slide=0):
		'''Evaluates a saved model on a given set of tfrecords.
		
		Args:
			model:					Path to .h5 model to evaluate.
			outcome_header:			Annotation column header that specifies the outcome label.
			hyperparameters:		Path to model's hyperparameters.json file. If None, searches for this file in the same directory as the model.
			filters:				Filters to use when selecting tfrecords on which to perform evaluation.
			checkpoint:				Path to cp.ckpt file to load, if evaluating a saved checkpoint.
			eval_k_fold:			K-fold iteration number to evaluate. If None, will evaluate all tfrecords irrespective of K-fold.
			max_tiles_per_slide:	Will only use up to this many tiles from each slide for evaluation. If zero, will include all tiles.
			min_tiles_per_slide:	Minimum number of tiles a slide must have to be included in evaluation. Default is 0, but
										for best slide-level AUC, a minimum of at least 10 tiles per slide is recommended.'''
										
		log.header(f"Evaluating model {sfutil.green(model)}...")

		manager = multiprocessing.Manager()
		results_dict = manager.dict()
		ctx = multiprocessing.get_context('spawn')
		
		process = ctx.Process(target=evaluator, args=(outcome_header, model, self.PROJECT, results_dict, filters, hyperparameters, 
														checkpoint, eval_k_fold, max_tiles_per_slide, min_tiles_per_slide, self.FLAGS))
		process.start()
		log.empty(f"Spawning evaluation process (PID: {process.pid})")
		process.join()

		return results_dict

	def extract_dual_tiles(self, tile_um=None, tile_px=None, stride_div=1, filters=None):
		import slideflow.slide as sfslide
		import tensorflow as tf
		from PIL import Image

		# Filter out warnings and allow loading large images
		warnings.simplefilter('ignore', Image.DecompressionBombWarning)
		Image.MAX_IMAGE_PIXELS = 100000000000

		tile_um = self.PROJECT['tile_um'] if not tile_um else tile_um
		tile_px = self.PROJECT['tile_px'] if not tile_px else tile_px

		log.header("Extracting dual-image tiles...")
		extracting_dataset = get_dataset(filters=filters)

		def extract_tiles_from_slide(slide_path, roi_list, dataset_config, pb):
			root_path = join(dataset_config["tfrecords"], dataset_config["label"])
			if not exists(root_path): 
					os.makedirs(root_path)

			whole_slide = sfslide.SlideReader(slide_path, tile_px, tile_um, stride_div, roi_list=roi_list, pb=pb)
			small_tile_generator = whole_slide.build_generator(dual_extract=True)
			tfrecord_name = sfutil.path_to_name(slide_path)
			tfrecord_path = join(root_path, f"{tfrecord_name}.tfrecords")
			records = []

			for image_dict in small_tile_generator():
				label = bytes(tfrecord_name, 'utf-8')
				image_string_dict = {}
				for image_label in image_dict:
					np_image = image_dict[image_label]
					image = Image.fromarray(np_image).convert('RGB')
					with io.BytesIO() as output:
						image.save(output, format="JPEG")
						image_string = output.getvalue()
						image_string_dict.update({
							image_label: image_string
						})
				records += [[label, image_string_dict]]

			shuffle(records)
			
			with tf.io.TFRecordWriter(tfrecord_path) as writer:
				for label, image_string_dict in records:
					tf_example = sfio.tfrecords.multi_image_example(label, image_string_dict)
					writer.write(tf_example.SerializeToString())

		for dataset_name in self.PROJECT['datasets']:
			log.empty(f"Working on dataset {sfutil.bold(dataset_name)}", 1)
			slide_list = extracting_dataset.get_slide_paths(dataset=dataset_name)
			roi_list = extracting_dataset.get_rois()
			dataset_config = extracting_dataset.datasets[dataset_name]
			log.info(f"Extracting tiles from {len(slide_list)} slides ({tile_um} um, {tile_px} px)", 1)
			pb = None#ProgressBar(bar_length=5, counter_text='tiles')

			if self.FLAGS['num_threads'] > 1:
				pool = DPool(self.FLAGS['num_threads'])
				pool.map(partial(extract_tiles_from_slide, roi_list=roi_list, dataset_config=dataset_config, pb=pb), slide_list)
				pool.close()
			else:
				for slide_path in slide_list:
					extract_tiles_from_slide(slide_path, roi_list, dataset_config, pb)
		
		self.update_manifest()

	def extract_tiles(self, tile_um=None, tile_px=None, filters=None, stride_div=1, tma=False, save_tiles=False, save_tfrecord=True,
						delete_tiles=True, enable_downsample=False, roi_method='inside', skip_missing_roi=True, skip_extracted=True, dataset=None):
		'''Extract tiles from a group of slides; save a percentage of tiles for validation testing if the 
		validation target is 'per-patient'; and generate TFRecord files from the raw images.
		
		Args:
			tile_um:			Tile size in microns. If None, will use project default.
			tile_px:			Tile size in pixels. If None, will use project default.
			filters:			Dataset filters to use when selecting slides for tile extraction.
			stride_div:			Stride divisor to use when extracting tiles. A stride of 1 will extract non-overlapping tiles. 
									A stride_div of 2 will extract overlapping tiles, with a stride equal to 50% of the tile width.
			tma:				Bool. If True, reads slides as Tumor Micro-Arrays (TMAs), detecting and extracting tumor cores.
			delete_tiles:		Bool. If True, will delete loose tile images after storing into TFRecords.
			enable_downsample:	Bool. If True, enables the use of downsampling while reading slide images. This may result in corrupted image tiles
									if downsampled slide layers are corrupted or not fully generated. Manual confirmation of tile integrity is recommended.
			roi_method:			Either 'inside' or 'outside'. Whether to extract tiles inside or outside the ROIs.'''

		import slideflow.slide as sfslide
		from PIL import Image

		# Filter out warnings and allow loading large images
		warnings.simplefilter('ignore', Image.DecompressionBombWarning)
		Image.MAX_IMAGE_PIXELS = 100000000000

		log.header("Extracting image tiles...")
		tile_um = self.PROJECT['tile_um'] if not tile_um else tile_um
		tile_px = self.PROJECT['tile_px'] if not tile_px else tile_px
		#self.FLAGS['num_threads'] = 1

		if not save_tiles and not save_tfrecord:
			log.error("Either save_tiles or save_tfrecord must be true to extract tiles.", 1)
			return
		
		if dataset: datasets = [dataset] if not isinstance(dataset, list) else dataset
		else:		datasets = self.PROJECT['datasets']

		# Load dataset for evaluation
		extracting_dataset = self.get_dataset(filters=filters)

		# Prepare validation/training subsets if per-tile validation is being used
		if self.PROJECT['validation_target'] == 'per-tile':
			if self.PROJECT['validation_strategy'] == 'boostrap':
				log.warn("Validation bootstrapping is not supported when the validation target is per-tile; will generate random fixed validation target", 1)
			if self.PROJECT['validation_strategy'] in ('bootstrap', 'fixed'):
				# Split the extracted tiles into two groups
				split_fraction = [-1, self.PROJECT['validation_fraction']]
				split_names = ['training', 'validation']
			if self.PROJECT['validation_strategy'] == 'k-fold':
				split_fraction = [-1] * self.PROJECT['validation_k_fold']
				split_names = [f'kfold-{i}' for i in range(self.PROJECT['validation_k_fold'])]
		else:
			split, split_fraction, split_names = None, None, None

		for dataset_name in datasets:
			log.empty(f"Working on dataset {sfutil.bold(dataset_name)}", 1)

			tiles_folder = join(extracting_dataset.datasets[dataset_name]['tiles'], extracting_dataset.datasets[dataset_name]['label'])
			roi_dir = extracting_dataset.datasets[dataset_name]['roi']
			dataset_config = extracting_dataset.datasets[dataset_name]
			tfrecord_dir = join(dataset_config["tfrecords"], dataset_config["label"])
			if save_tfrecord and not exists(tfrecord_dir):
				os.makedirs(tfrecord_dir)
			if save_tiles and not os.path.exists(tiles_folder):
				os.makedirs(tiles_folder)

			# Prepare list of slides for extraction
			slide_list = extracting_dataset.get_slide_paths(dataset=dataset_name)
			already_extracted_tfrecords = [sfutil.path_to_name(tfr) for tfr in extracting_dataset.get_tfrecords(dataset=dataset_name)]
			if skip_extracted:
				# First, check for interrupted extraction
				interrupted = [sfutil.path_to_name(marker) for marker in glob(join((tfrecord_dir if tfrecord_dir else tiles_folder), '*.unfinished'))]
				if len(interrupted):
					log.info(f'Interrupted tile extraction detected in {len(interrupted)} tfrecords, will re-extract these slides')
					for interrupted_slide in interrupted:
						log.empty(interrupted_slide, 2)
						if interrupted_slide in already_extracted_tfrecords:
							del(already_extracted_tfrecords[already_extracted_tfrecords.index(interrupted_slide)])
					
				slide_list = [slide for slide in slide_list if sfutil.path_to_name(slide) not in already_extracted_tfrecords]
				if len(already_extracted_tfrecords):
					log.info(f"Skipping tile extraction for {len(already_extracted_tfrecords)} slides; TFRecords already generated.", 1)	
			log.info(f"Extracting tiles from {len(slide_list)} slides ({tile_um} um, {tile_px} px)", 1)

			# Verify slides and estimate total number of tiles
			log.info("Verifying slides...", 1)
			total_tiles = 0
			for slide_path in slide_list:
				slide = sfslide.SlideReader(slide_path, tile_px, tile_um, stride_div, roi_dir=roi_dir,
																					  roi_method=roi_method,
																					  skip_missing_roi=skip_missing_roi,
																					  silent=True)
				print(f"\r\033[KVerified {sfutil.green(slide.name)} (approx. {slide.estimated_num_tiles} tiles)", end="")
				total_tiles += slide.estimated_num_tiles
				del(slide)
			print("\r\033[K", end='')
			log.info(f"Verification complete. Total estimated tiles to extract: {total_tiles}", 1)
			
			if total_tiles:
				pb = ProgressBar(total_tiles, counter_text='tiles', leadtext="Extracting tiles... ", show_counter=True, show_eta=True)
			else:
				pb = None

			# Function to extract tiles from a slide
			def extract_tiles_from_slide(slide_path, pb):
				print_func = print if not pb else pb.print
				log.empty(f"Exporting tiles for slide {sfutil.path_to_name(slide_path)}", 1, print_func)

				if not tma:
					whole_slide = sfslide.SlideReader(slide_path, tile_px, tile_um, stride_div, enable_downsample=enable_downsample, 
																								roi_dir=roi_dir,
																								roi_method=roi_method,
																								skip_missing_roi=skip_missing_roi,
																								pb=pb)
				else:
					whole_slide = sfslide.TMAReader(slide_path, tile_px, tile_um, stride_div, enable_downsample=enable_downsample,
																							  export_folder=tiles_folder, 
																							  pb=pb)

				if not whole_slide.loaded_correctly():
					return

				whole_slide.export_tiles(tfrecord_dir=tfrecord_dir if save_tfrecord else None,
										 tiles_dir=tiles_folder if save_tiles else None,
										 split_fraction=split_fraction,
										 split_names=split_names)

			# Use multithreading if specified, extracting tiles from all slides in the filtered list
			if self.FLAGS['num_threads'] > 1 and len(slide_list):
				pool = DPool(self.FLAGS['num_threads'])
				pool.map(partial(extract_tiles_from_slide, pb=pb), slide_list)
				pool.close()
			else:
				for slide_path in slide_list:
					extract_tiles_from_slide(slide_path, pb)

		# Update manifest
		self.update_manifest()

	def generate_activations_analytics(self, model, outcome_header=None, filters=None, focus_nodes=[], node_exclusion=False, activations_export=None,
										activations_cache='default'):
		'''Calculates final layer activations and displays information regarding the most significant final layer nodes.
		
		Note: GPU memory will remain in use, as the Keras model associated with the visualizer is active.'''
		log.header("Generating final layer activation analytics...")

		# Load dataset for evaluation
		activations_dataset = self.get_dataset(filters=filters)
		tfrecords_list = activations_dataset.get_tfrecords(ask_to_merge_subdirs=True)
		model_path = model if model[-3:] == ".h5" else join(self.PROJECT['models_dir'], model, 'trained_model.h5')
		log.info(f"Visualizing activations from {len(tfrecords_list)} slides", 1)

		AV = ActivationsVisualizer(model=model_path,
								   tfrecords=tfrecords_list,
								   root_dir=self.PROJECT['root'],
								   image_size=self.PROJECT['tile_px'],
								   annotations=self.PROJECT['annotations'],
								   outcome_header=outcome_header,
								   focus_nodes=focus_nodes,
								   use_fp16=self.PROJECT['use_fp16'],
								   activations_export=activations_export,
								   activations_cache=activations_cache)

		return AV

	def generate_heatmaps(self, model, filters=None, directory=None, resolution='low', interpolation='none', show_roi=True, logit_cmap=None, skip_thumb=False, single_thread=False):
		'''Creates predictive heatmap overlays on a set of slides. 

		Args:
			model:			Path to .h5 model with which predictions will be generated.
			filters:		Dataset filters to use when selecting slides for which to generate heatmaps.
			resolution:		Heatmap resolution (determines stride of tile predictions). 
								"low" uses a stride equal to tile width.
								"medium" uses a stride equal 1/2 tile width.
								"high" uses a stride equal to 1/4 tile width.
		'''
		log.header("Generating heatmaps...")

		# Prepare dataset
		heatmaps_dataset = self.get_dataset(filters=filters)
		slide_list = heatmaps_dataset.get_slide_paths()
		roi_list = heatmaps_dataset.get_rois()
		model_path = model if model[-3:] == ".h5" else join(self.PROJECT['models_dir'], model, 'trained_model.h5')

		# Attempt to auto-detect supplied model name
		detected_model_name = sfutil.path_to_name(model_path)
		hp_file = join(*model_path.split('/')[:-1], 'hyperparameters.json')
		if exists(hp_file):
			loaded_hp = sfutil.load_json(hp_file)
			if 'model_name' in loaded_hp:
				detected_model_name = loaded_hp['model_name']
		
		# Make output directory
		heatmaps_folder = directory if directory else os.path.join(self.PROJECT['root'], 'heatmaps', detected_model_name)
		if not exists(heatmaps_folder): os.makedirs(heatmaps_folder)

		# Heatmap processes
		ctx = multiprocessing.get_context('spawn')
		for slide in slide_list:
			if single_thread:
				heatmap_generator(slide, model, model_path, heatmaps_folder, roi_list, show_roi,
									resolution, interpolation, self.PROJECT, logit_cmap, skip_thumb, self.FLAGS)
			else:
				process = ctx.Process(target=heatmap_generator, args=(slide, model, model_path, heatmaps_folder, roi_list, show_roi, 
																		resolution, interpolation, self.PROJECT, logit_cmap, skip_thumb, self.FLAGS))
				process.start()
				log.empty(f"Spawning heatmaps process (PID: {process.pid})")
				process.join()

	def generate_mosaic(self, model, header_category=None, filters=None, focus_filters=None, resolution="low", num_tiles_x=50, max_tiles_per_slide=100,
						expanded=False, map_centroid=False, show_prediction=None, restrict_prediction=None, outcome_labels=None, cmap=None, model_type=None,
						umap_cache='default', activations_cache='default', mosaic_filename=None, umap_filename=None, activations_export=None, umap_export=None,
						use_float=False, normalize=None):
		'''Generates a mosaic map with dimensionality reduction on penultimate layer activations. Tile data is extracted from the provided
		set of TFRecords and predictions are calculated using the specified model.
		
		Args:
			model:					Path to .h5 file to use when generating layer activations.
			filters:				Dataset filters to use when selecting slides to include the mosaic.
			focus_filters:			Dataset filters to use when selecting slides to highlight on the mosaic.
			resolution:				Resolution of the mosaic map. Impacts size of the final figure. Either low, medium, or high.
			num_tiles_x:			Specifies the size of the mosaic map grid.
			max_tiles_per_slide:	Limits the number of tiles taken from each slide. Too high of a number may introduce memory issues.
			export_activations:		Bool. If true, will save calculated layer activations to a CSV.'''

		log.header("Generating mosaic map...")

		# Prepare dataset & model
		mosaic_dataset = self.get_dataset(filters=filters)
		tfrecords_list = mosaic_dataset.get_tfrecords()
		if focus_filters:
			mosaic_dataset.apply_filters(focus_filters)
			focus_list = mosaic_dataset.get_tfrecords()
		else:
			focus_list = None
		log.info(f"Generating mosaic from {len(tfrecords_list)} slides, with focus on {0 if not focus_list else len(focus_list)} slides.", 1)

		# Set up paths
		model_path = model if model[-3:] == ".h5" else join(self.PROJECT['models_dir'], model, 'trained_model.h5')
		if umap_cache == 'default':
			umap_cache = join(self.PROJECT['root'], 'stats', 'umap_cache.pkl')
		else:
			umap_cache = join(self.PROJECT['root'], 'stats', umap_cache)

		# If a header category is supplied and we are not showing predictions, then assign slide labels from annotations
		if header_category and (show_prediction is None):
			outcomes_category, unique_outcomes = mosaic_dataset.get_outcomes_from_annotations(header_category, use_float=use_float)
			if use_float:
				slide_labels = {k:outcomes_category[k]['outcome'] for k, v in outcomes_category.items()}
			else:
				slide_labels = {k:unique_outcomes[v['outcome']] for k, v in outcomes_category.items()}			
		else:
			slide_labels = {}

		# If showing predictions, try to automatically load prediction labels
		if (show_prediction is not None) and (not outcome_labels):
			if exists(join(dirname(model_path), 'hyperparameters.json')):
				model_hyperparameters = sfutil.load_json(join(dirname(model_path), 'hyperparameters.json'))
				outcome_labels = model_hyperparameters['outcome_labels']
				model_type = model_type if model_type else model_hyperparameters['model_type']
				log.info(f'Automatically loaded prediction labels found at {sfutil.green(dirname(model_path))}', 1)
			else:
				log.info(f'Unable to auto-detect prediction labels from model hyperparameters file', 1)
				
		# Initialize mosaic, umap, and ActivationsVisualizer
		mosaic, umap = None, None

		AV = ActivationsVisualizer(model=model_path,
								   tfrecords=tfrecords_list, 
								   root_dir=self.PROJECT['root'],
								   image_size=self.PROJECT['tile_px'],
								   focus_nodes=None,
								   use_fp16=self.PROJECT['use_fp16'],
								   batch_size=self.FLAGS['eval_batch_size'],
								   activations_export=activations_export,
								   max_tiles_per_slide=max_tiles_per_slide,
								   activations_cache=activations_cache)

		umap = TFRecordUMAP.from_activations(AV, use_centroid=map_centroid, prediction_filter=restrict_prediction, cache=umap_cache)

		# If displaying centroid AND predictions, then show slide-level predictions rather than tile-level predictions
		if map_centroid and show_prediction is not None:
			log.info("Showing slide-level predictions at point of centroid", 1)

			# If not model has not been assigned, assume categorical model
			model_type = model_type if model_type else 'categorical'

			# Get predictions
			slide_predictions, slide_percentages = AV.get_slide_level_predictions(model_type=model_type, prediction_filter=restrict_prediction)

			# Assign outcome label to prediction
			# If show_prediction is provided (either a number or string), then display ONLY the prediction for the provided category, as a colormap
			if type(show_prediction) == int:
				log.info(f"Showing prediction for outcome {show_prediction} as colormap", 1)
				slide_labels = {k:v[show_prediction] for k, v in slide_percentages.items()}
				show_prediction = None
				use_float = True
			elif type(show_prediction) == str:
				log.info(f"Showing prediction for outcome {show_prediction} as colormap", 1)
				reversed_labels = {v:k for k, v in outcome_labels.items()}
				if show_prediction not in reversed_labels:
					raise ValueError(f"Unknown outcome category `{show_prediction}`")
				slide_labels = {k:v[int(reversed_labels[show_prediction])] for k, v in slide_percentages.items()}
				show_prediction = None
				use_float = True
			elif use_float:
				# Displaying linear predictions needs to be implemented here
				raise TypeError("If showing prediction and use_float is True, please pass desired outcome category for prediction to `show_prediction`.")
			# Otherwise, show_prediction is assumed to be just "True", in which case show categorical predictions
			else:
				try:
					slide_labels = {k:outcome_labels[v] for (k,v) in slide_predictions.items()}
				except KeyError:
					# Try interpreting prediction label keys as strings
					slide_labels = {k:outcome_labels[str(v)] for (k,v) in slide_predictions.items()}

		umap.save_2d_plot(umap_filename if umap_filename else join(self.PROJECT['root'], 'stats', '2d_mosaic_umap.png'), slide_labels=slide_labels,
																														 slide_filter=mosaic_dataset.get_slides(),
																														 show_tile_meta='prediction' if (show_prediction and not map_centroid) else None,
																														 outcome_labels=outcome_labels,
																														 cmap=cmap,
																														 use_float=use_float)
		if umap_export:
			umap.export_to_csv(umap_export)

		if mosaic_filename:
			mosaic = Mosaic(umap, leniency=1.5,
								expanded=expanded,
								tile_zoom=15,
								num_tiles_x=num_tiles_x,
								resolution=resolution,
								normalize=normalize)
			mosaic.focus(focus_list)
			mosaic.save(mosaic_filename if mosaic_filename else join(self.STATS_ROOT, 'Mosaic.png'))
			
		return AV, mosaic, umap

	def generate_mosaic_from_predictions(self, model, x, y, filters=None, focus_filters=None, header_category=None, resolution='low', num_tiles_x=50,
											expanded=False, max_tiles_per_slide=0, normalize=None):

		dataset = self.get_dataset(filters=filters, filter_blank=header_category)
		outcomes_category, unique_outcomes = dataset.get_outcomes_from_annotations(header_category)
		slide_to_category = {k:unique_outcomes[v['outcome']] for k, v in outcomes_category.items()}

		AV = ActivationsVisualizer(model=model,
								   tfrecords=dataset.get_tfrecords(), 
								   root_dir=self.PROJECT['root'],
								   image_size=self.PROJECT['tile_px'],
								   use_fp16=self.PROJECT['use_fp16'],
								   batch_size=self.FLAGS['eval_batch_size'],
								   max_tiles_per_slide=max_tiles_per_slide)

		umap_x, umap_y, umap_meta = AV.get_mapped_predictions(x, y)

		umap = TFRecordUMAP.from_precalculated(tfrecords=dataset.get_tfrecords(),
											   slides=dataset.get_slides(),
											   x=umap_x,
											   y=umap_y,
											   meta=umap_meta)
 
		umap.save_2d_plot(join(self.PROJECT['root'], 'stats', '2d_mosaic_umap.png'), slide_to_category)

		mosaic_map = Mosaic(umap, leniency=1.5,
								  expanded=expanded,
								  tile_zoom=15,
								  num_tiles_x=num_tiles_x,
								  resolution=resolution,
								  normalize=normalize)

		mosaic_map.save(join(self.PROJECT['root'], 'stats'))
		mosaic_map.save_report(join(self.PROJECT['root'], 'stats', 'mosaic_report.csv'))

	def generate_mosaic_from_annotations(self, header_x, header_y, header_category=None, filters=None, focus_filters=None, resolution='low', num_tiles_x=50,
											expanded=False, use_optimal_tile=False, model=None, max_tiles_per_slide=100, mosaic_filename=None, umap_filename=None,
											activations_cache='default', normalize=None):

		dataset = self.get_dataset(filters=filters, filter_blank=[header_x, header_y])
		# We are assembling a list of slides from the TFRecords path list, because we only want to use slides that have a corresponding TFRecord
		#  (some slides did not have a large enough ROI for tile extraction, and some slides may be in the annotations but are missing a slide image)
		slides = [sfutil.path_to_name(tfr) for tfr in dataset.get_tfrecords()]
		outcomes, _ = dataset.get_outcomes_from_annotations([header_x, header_y], use_float=True)
		outcomes_category, unique_outcomes = dataset.get_outcomes_from_annotations(header_category)
		slide_to_category = {k:unique_outcomes[v['outcome']] for k, v in outcomes_category.items()}

		umap_x = np.array([outcomes[slide]['outcome'][0] for slide in slides])
		umap_y = np.array([outcomes[slide]['outcome'][1] for slide in slides])

		if use_optimal_tile and not model:
			log.error("Unable to calculate optimal tile if no model is specified.")
			return
		elif use_optimal_tile:
			# Calculate most representative tile in each slide/TFRecord for display
			AV = ActivationsVisualizer(model=model,
									   tfrecords=dataset.get_tfrecords(), 
									   root_dir=self.PROJECT['root'],
									   image_size=self.PROJECT['tile_px'],
									   use_fp16=self.PROJECT['use_fp16'],
									   batch_size=self.FLAGS['eval_batch_size'],
									   max_tiles_per_slide=max_tiles_per_slide,
									   activations_cache='default')

			optimal_slide_indices, _ = calculate_centroid(AV.slide_node_dict)

			# Restrict mosaic to only slides that had enough tiles to calculate an optimal index from centroid
			successful_slides = list(optimal_slide_indices.keys())
			num_warned = 0
			warn_threshold = 3
			for slide in slides:
				print_func = print if num_warned < warn_threshold else None
				if slide not in successful_slides:
					log.warn(f"Unable to calculate optimal tile for slide {sfutil.green(slide)}; will not include in Mosaic", 1, print_func)
					num_warned += 1
			if num_warned >= warn_threshold:
				log.warn(f"...{num_warned} total warnings, see {sfutil.green(log.logfile)} for details", 1)

			umap_x = np.array([outcomes[slide]['outcome'][0] for slide in successful_slides])
			umap_y = np.array([outcomes[slide]['outcome'][1] for slide in successful_slides])
			umap_meta = [{'slide': slide, 'index': optimal_slide_indices[slide]} for slide in successful_slides]
		else:
			# Take the first tile from each slide/TFRecord
			umap_meta = [{'slide': slide, 'index': 0} for slide in slides]

		umap = TFRecordUMAP.from_precalculated(tfrecords=dataset.get_tfrecords(),
											   slides=slides,
											   x=umap_x,
											   y=umap_y,
											   meta=umap_meta)

		mosaic_map = Mosaic(umap, leniency=1.5,
								  expanded=expanded,
								  tile_zoom=15,
								  num_tiles_x=num_tiles_x,
								  tile_select='centroid' if use_optimal_tile else 'nearest',
								  resolution=resolution,
								  normalize=normalize)

		mosaic_map.save(mosaic_filename if mosaic_filename else join(self.PROJECT['root'], 'stats', 'Mosaic.png'))
		mosaic_map.save_report(join(self.PROJECT['root'], 'stats', 'mosaic_report.csv'))
		umap.save_2d_plot(umap_filename if umap_filename else join(self.PROJECT['root'], 'stats', '2d_mosaic_umap.png'), slide_to_category)

	def generate_thumbnails(self, filters=None, filter_blank=None, enable_downsample=False):
		'''Generates slide thumbnails and saves to project folder.'''
		log.header('Generating thumbnails...')

		thumb_folder = join(self.PROJECT['root'], 'thumbs')
		dataset = self.get_dataset(filters=filters, filter_blank=filter_blank)		
		slide_list = dataset.get_slide_paths(dataset=dataset_name)
		log.info(f"Saving thumbnails to {sfutil.green(thumb_folder)}", 1)

		for slide_path in slide_list:
			log.empty(f"Working on {sfutil.green(sfutil.path_to_name(slide_path))}...", 1)
			whole_slide = sfslide.SlideReader(slide_path, 0, 0, 1, enable_downsample=enable_downsample,
																   skip_missing_roi=False)

	def generate_tfrecords_from_tiles(self, delete_tiles=True):
		'''Create tfrecord files from a collection of raw images'''
		log.header('Writing TFRecord files...')

		# Load dataset for evaluation
		working_dataset = Dataset(config_file=self.PROJECT['dataset_config'], sources=self.PROJECT['datasets'])
		
		for d in working_dataset.datasets:
			log.empty(f"Working on dataset {d}", 1)
			config = working_dataset.datasets[d]
			tfrecord_dir = join(config["tfrecords"], config["label"])
			tiles_dir = join(config["tiles"], config["label"])
			if not exists(tiles_dir):
				log.warn(f"No tiles found for dataset {sfutil.bold(d)}", 1)
				continue

			# Check to see if subdirectories in the target folders are slide directories (contain images)
			#  or are further subdirectories (e.g. validation and training)
			log.info('Scanning tile directory structure...', 2)
			if sfutil.contains_nested_subdirs(tiles_dir):
				subdirs = [_dir for _dir in os.listdir(tiles_dir) if isdir(join(tiles_dir, _dir))]
				for subdir in subdirs:
					tfrecord_subdir = join(tfrecord_dir, subdir)
					sfio.tfrecords.write_tfrecords_multi(join(tiles_dir, subdir), tfrecord_subdir)
			else:
				sfio.tfrecords.write_tfrecords_multi(tiles_dir, tfrecord_dir)

			self.update_manifest()

			if delete_tiles:
				shutil.rmtree(tiles_dir)
	
	def get_dataset(self, filters=None, filter_blank=None):
		return Dataset(config_file=self.PROJECT['dataset_config'], 
					   sources=self.PROJECT['datasets'],
					   annotations=self.PROJECT['annotations'],
					   filters=filters,
					   filter_blank=filter_blank)

	def load_datasets(self, path):
		'''Loads datasets from a given datasets.json file.'''
		try:
			datasets_data = sfutil.load_json(path)
			datasets_names = list(datasets_data.keys())
			datasets_names.sort()
		except FileNotFoundError:
			datasets_data = {}
			datasets_names = []
		return datasets_data, datasets_names

	def load_project(self, directory):
		'''Loads a saved and pre-configured project.'''
		if exists(join(directory, "settings.json")):
			self.PROJECT = sfutil.load_json(join(directory, "settings.json"))
			log.empty("Project configuration loaded.")
		else:
			raise OSError(f'Unable to locate settings.json at location "{directory}".')

		# Enable logging
		log.logfile = join(self.PROJECT['root'], "log.log")

		# Load dataset for evaluation
		try:
			dataset = self.get_dataset()

			if not self.FLAGS['skip_verification']:
				log.header("Verifying Annotations...")
				dataset.verify_annotations_slides()
				log.header("Verifying TFRecord manifest...")
				self.update_manifest()
		except FileNotFoundError:
			log.warn("No datasets configured.")

	def resize_tfrecords(self, size, filters=None):
		'''Resizes TFRecords to a given pixel size.'''
		log.header(f"Resizing TFRecord tiles to ({size}, {size})")
		resize_dataset = self.get_dataset(filters=filters)
		tfrecords_list = resize_dataset.get_tfrecords()
		log.info(f"Resizing {len(tfrecords_list)} tfrecords", 1)

		for tfr in tfrecords_list:
			sfio.tfrecords.transform_tfrecord(tfr, tfr+".transformed", resize=size)
	
	def extract_tiles_from_tfrecords(self, destination=None, filters=None):
		'''Extracts all tiles from a set of TFRecords'''
		log.header(f"Extracting tiles from TFRecords")
		to_extract_dataset = self.get_dataset(filters=filters)
		
		for dataset_name in self.PROJECT['datasets']:
			to_extract_tfrecords = to_extract_dataset.get_tfrecords(dataset=dataset_name)
			if destination:
				tiles_dir = destination
			else:
				tiles_dir = join(to_extract_dataset.datasets[dataset_name]['tiles'], to_extract_dataset.datasets[dataset_name]['label'])
				if not exists(tiles_dir):
					os.makedirs(tiles_dir)
			for tfr in to_extract_tfrecords:
				sfio.tfrecords.extract_tiles(tfr, tiles_dir)		

	def save_project(self):
		'''Saves current project configuration as "settings.json".'''
		sfutil.write_json(self.PROJECT, join(self.PROJECT['root'], 'settings.json'))

	def train(self, models=None, outcome_header='category', multi_outcome=False, filters=None, resume_training=None, checkpoint=None, 
				pretrain='imagenet', batch_file=None, hyperparameters=None, validation_target=None, validation_strategy=None,
				validation_fraction=None, validation_k_fold=None, k_fold_iter=None,
				validation_dataset=None, validation_annotations=None, validation_filters=None, validate_on_batch=256, validation_steps=200,
				max_tiles_per_slide=0, min_tiles_per_slide=0, starting_epoch=0):
		'''Train model(s) given configurations found in batch_train.tsv.

		Args:
			models:					Either a string representing a model name, or an array of strings containing multiple model names. 
										Required if training to a single hyperparameter combination with the "hyperparameters" argument.
										If performing a hyperparameter sweep, will only train models with these names in the batch_train.tsv config file.
										May supply None if performing a hyperparameter sweep, in which case all models in the batch_train.tsv config file will be trained.
			outcome_header:			String or list. Specifies which header(s) in the annotation file to use for the output category. 
										Defaults to 'category'.	If a list is provided, will loop through all outcomes and perform HP sweep on each.
			multi_outcome:			If True, will train to multiple outcomes simultaneously instead of looping through the
										list of outcomes in "outcome_header". Defaults to False.
			filters:				Dictionary of column names mapping to column values by which to filter slides using the annotation file.
			resume_training:		Path to .h5 model to continue training
			checkpoint:				Path to cp.ckpt from which to load weights
			pretrain:				Pretrained weights to load. Default is imagenet. May supply a compatible .h5 file from which to load weights.
			batch_file:				Manually specify batch file to use for a hyperparameter sweep. If not specified, will use project default.
			hyperparameters:		Manually specify hyperparameter combination to use for training. If specified, will ignore batch training file.
			validation_target: 		Whether to select validation data on a 'per-patient' or 'per-tile' basis. If not specified, will use project default.
			validation_strategy:	Validation dataset selection strategy (bootstrap, k-fold, fixed, none). If not specified, will use project default.
			validation_fraction:	Fraction of data to use for validation testing. If not specified, will use project default.
			validation_k_fold: 		K, if using k-fold validation. If not specified, will use project default.
			k_fold_iter:			Which iteration to train if using k-fold validation. Defaults to training all iterations.
			validation_dataset:		If specified, will use a separate dataset on which to perform validation.
			validation_annotations:	If using a separate dataset for validation, the annotations CSV must be supplied.
			validation_filters:		If using a separate dataset for validation, these filters are used to select a subset of slides for validation.
			validate_on_batch:		Validation will be performed every X batches.
			max_tiles_per_slide:	Will only use up to this many tiles from each slide for training. If zero, will include all tiles.
			min_tiles_per_slide:	Minimum number of tiles a slide must have to be included in training. 

		Returns:
			A dictionary containing model names mapped to train_acc, val_loss, and val_acc
		'''
		# Reconcile provided arguments with project defaults
		batch_train_file = self.PROJECT['batch_train_config'] if not batch_file else join(self.PROJECT['root'], batch_file)
		validation_strategy = self.PROJECT['validation_strategy'] if not validation_strategy else validation_strategy
		validation_target = self.PROJECT['validation_target'] if not validation_target else validation_target
		validation_fraction = self.PROJECT['validation_fraction'] if not validation_fraction else validation_fraction
		validation_k_fold = self.PROJECT['validation_k_fold'] if not validation_k_fold else validation_k_fold
		validation_log = join(self.PROJECT['root'], "validation_plans.json")

		# Quickly scan for errors (duplicate model names in batch training file) and prepare models to train
		if hyperparameters and not models:
			log.error("If specifying hyperparameters, 'models' must be supplied. ", 1)
			return

		# Prepare hyperparameters
		log.header("Performing hyperparameter sweep...")
		
		hyperparameter_list = self._get_hyperparameter_combinations(hyperparameters, models, batch_train_file)

		outcome_header = [outcome_header] if not isinstance(outcome_header, list) else outcome_header
		if multi_outcome:
			log.info(f"Training ({len(hyperparameter_list)} models) using {len(outcome_header)} variables as simultaneous input:", 1)
		else:
			log.header(f"Training ({len(hyperparameter_list)} models) for each of {len(outcome_header)} outcome variables:", 1)
		for outcome in outcome_header:
			log.empty(outcome, 2)
		print()
		outcome_header = [outcome_header] if multi_outcome else outcome_header

		# Prepare k-fold validation configuration
		results_log_path = os.path.join(self.PROJECT['root'], "results_log.csv")
		k_fold_iter = [k_fold_iter] if (k_fold_iter != None and not isinstance(k_fold_iter, list)) else k_fold_iter
		k_fold = validation_k_fold if validation_strategy in ('k-fold', 'bootstrap') else 0
		valid_k = [] if not k_fold else [kf for kf in range(1, k_fold+1) if ((k_fold_iter and kf in k_fold_iter) or (not k_fold_iter))]

		# Next, prepare the multiprocessing manager (needed to free VRAM after training and keep track of results)
		manager = multiprocessing.Manager()
		results_dict = manager.dict()
		ctx = multiprocessing.get_context('spawn')

		# If using multiple outcomes, initiate hyperparameter sweep for each outcome category specified
		# If not training to multiple outcome, perform full hyperparameter sweep of the combined outcomes
		for selected_outcome_headers in outcome_header:
			# For each hyperparameter combination, perform training
			for hp, hp_model_name in hyperparameter_list:
				if multi_outcome and hp.model_type() != 'linear':
					log.error("Multiple outcome variables only supported for linear outcome variables.")
					return
				# Generate model name
				if isinstance(selected_outcome_headers, list):
					outcome_string = "-".join(selected_outcome_headers)
				else:
					outcome_string = selected_outcome_headers
					selected_outcome_headers = [selected_outcome_headers]

				model_name = f"{outcome_string}-{hp_model_name}"
				model_iterations = [model_name] if not k_fold else [f"{model_name}-kfold{k}" for k in valid_k]

				def start_training_process(k):
					# Using a separate process ensures memory is freed once training has completed
					process = ctx.Process(target=trainer, args=(selected_outcome_headers, model_name,self.PROJECT,
																results_dict, hp, validation_strategy, 
																validation_target, validation_fraction, validation_k_fold, 
																validation_log, validation_dataset, validation_annotations,
																validation_filters, k, filters, pretrain, resume_training, 
																checkpoint, validate_on_batch, validation_steps, max_tiles_per_slide,
																min_tiles_per_slide, starting_epoch, self.FLAGS))
					process.start()
					log.empty(f"Spawning training process (PID: {process.pid})")
					process.join()

				# Perform training
				if k_fold:
					for k in valid_k:
						start_training_process(k)
						
				else:
					start_training_process(None)

				# Record results
				for mi in model_iterations:
					if mi not in results_dict:
						log.error(f"Training failed for model {model_name} for an unknown reason")
					else:
						sfutil.update_results_log(results_log_path, mi, results_dict[mi]['epochs'])
				log.complete(f"Training complete for model {model_name}, results saved to {sfutil.green(results_log_path)}")

			# Print summary of all models
			log.complete("Training complete; validation accuracies:", 0)
			for model in results_dict:
				try:
					last_epoch = max([int(e.split('epoch')[-1]) for e in results_dict[model]['epochs'].keys() if 'epoch' in e ])
					final_metrics = results_dict[model]['epochs'][f'epoch{last_epoch}']
					log.empty(f" - {sfutil.green(model)}: Train_Acc={str(final_metrics['train_acc'])}, " +
						f"Val_loss={final_metrics['val_loss']}, Val_Acc={final_metrics['val_acc']}" )
				except ValueError:
					pass

		return results_dict

	def update_manifest(self, force_update=False):
		'''Updates manifest file in the TFRecord directory, used to track number of records and verify annotations.
		
		Args:
			force_update:	If True, will re-validate contents of all TFRecords. If False, will only validate
								contents of TFRecords not yet in the manifest
		'''
		dataset = self.get_dataset()
		dataset.update_manifest()

	def update_tfrecords(self):
		log.header('Updating TFRecords...')
		working_dataset = self.get_dataset()
		for d in working_dataset.datasets:
			config = working_dataset.datasets[d]
			tfrecord_folder = join(config["tfrecords"], config["label"])
			num_updated = 0
			log.info(f"Updating TFRecords in {sfutil.green(tfrecord_folder)}...")
			num_updated += sfio.tfrecords.update_tfrecord_dir(tfrecord_folder, slide='case', image_raw='image_raw')
		log.complete(f"Updated {sfutil.bold(num_updated)} TFRecords files")
	
	def visualize_tiles(self, model, node, tfrecord_dict=None, directory=None, num_to_visualize=20, window=None):
		TV = TileVisualizer(model=model, 
							node=node,
							shape=[self.PROJECT['tile_px'], self.PROJECT['tile_px'], 3],
							tile_width=window)

		if tfrecord_dict:
			for tfrecord in tfrecord_dict:
				for tile_index in tfrecord_dict[tfrecord]:
					TV.visualize_tile(tfrecord=tfrecord, index=tile_index, save_dir=directory)

		else:
			tiles = [o for o in os.listdir(directory) if not isdir(join(directory, o))]
			tiles.sort(key=lambda x: int(x.split('-')[0]))
			tiles.reverse()
			for tile in tiles[:20]:
				tile_loc = join(directory, tile)
				TV.visualize_tile(image_jpg=tile_loc, save_dir=directory)