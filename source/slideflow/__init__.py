import argparse
import os
import sys
import shutil
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
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
import slideflow.trainer.model as sfmodel
import itertools
import multiprocessing
import slideflow.util as sfutil
from slideflow.util import datasets, tfrecords, TCGA, log
from slideflow.mosaic import Mosaic

__version__ = "1.0.1d"

SKIP_VERIFICATION = False
NUM_THREADS = 4
EVAL_BATCH_SIZE = 64
GPU_LOCK = None
NO_LABEL = 'no_label'
SOURCE_DIR = os.path.dirname(os.path.realpath(__file__))
VALIDATION_ID = ''.join(choice(ascii_lowercase) for i in range(10))
COMET_API_KEY = "A3VWRcPaHgqc4H5K0FoCtRXbp"
DEBUGGING = True

def set_logging_level(level):
	sfutil.LOGGING_LEVEL.INFO = level

def autoselect_gpu(number_available):
	global GPU_LOCK
	'''Automatically claims a free GPU and creates a lock file to prevent 
	other instances of slideflow from using the same GPU.'''
	for n in range(number_available):
		if not exists(join(SOURCE_DIR, f"gpu{n}.lock")):
			print(f"Requesting GPU #{n}")
			os.environ["CUDA_VISIBLE_DEVICES"]=str(n)
			open(join(SOURCE_DIR, f"gpu{n}.lock"), 'a').close()
			GPU_LOCK = n
			return
	log.error(f"No free GPUs detected; try deleting 'gpu[#].lock' files in the slideflow directory if GPUs are not in use.")

def select_gpu(number):
	global GPU_LOCK
	print(f"Requesting GPU #{number}")
	GPU_LOCK = number
	os.environ["CUDA_VISIBLE_DEVICES"]=str(number)

def release_gpu():
	global GPU_LOCK
	print("Cleaning up...")
	if GPU_LOCK != None and exists(join(SOURCE_DIR, f"gpu{GPU_LOCK}.lock")):
		print(f"Freeing GPU {GPU_LOCK}...")
		os.remove(join(SOURCE_DIR, f"gpu{GPU_LOCK}.lock"))
	
atexit.register(release_gpu)

class SlideFlowProject:
	MANIFEST = None

	def __init__(self, project_folder):
		'''Initializes project by creating project folder, prompting user for project settings, and project
		settings to "settings.json" within the project directory.'''
		os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
		log.header(f"Slideflow v{__version__}\n================")
		log.header("Loading project...")
		if project_folder and not os.path.exists(project_folder):
			if sfutil.yes_no_input(f'Directory "{project_folder}" does not exist. Create directory and set as project root? [Y/n] ', default='yes'):
				os.mkdir(project_folder)
			else:
				project_folder = sfutil.dir_input("Where is the project root directory? ", create_on_invalid=True, absolute=True)
		if not project_folder:
			project_folder = sfutil.dir_input("Where is the project root directory? ", create_on_invalid=True, absolute=True)
		sfutil.PROJECT_DIR = project_folder

		if exists(join(project_folder, "settings.json")):
			self.load_project(project_folder)
		else:
			self.create_project()
		
	def extract_tiles(self, filters=None, subfolder=None, skip_validation=False):
		'''Extract tiles from a group of slides; save a percentage of tiles for validation testing if the 
		validation target is 'per-slide'; and generate TFRecord files from the raw images.'''
		import slideflow.convoluter as convoluter

		log.header("Extracting image tiles...")
		subfolder = NO_LABEL if (not subfolder or subfolder=='') else subfolder
		convoluter.NUM_THREADS = NUM_THREADS
		slide_list = sfutil.get_filtered_slide_paths(self.PROJECT['slides_dir'], filters=filters)
		log.info(f"Extracting tiles from {len(slide_list)} slides", 1)

		save_folder = join(self.PROJECT['tiles_dir'], subfolder)
		if not os.path.exists(save_folder):
			os.makedirs(save_folder)

		c = convoluter.Convoluter(self.PROJECT['tile_px'], self.PROJECT['tile_um'], batch_size=None,
																					use_fp16=self.PROJECT['use_fp16'], 
																					stride_div=2,
																					save_folder=save_folder, 
																					roi_dir=self.PROJECT['roi_dir'])
		c.load_slides(slide_list)
		c.convolute_slides(export_tiles=True)

		if not skip_validation and self.PROJECT['validation_target'] == 'per-tile':
			if self.PROJECT['validation_target'] == 'per-tile':
				if self.PROJECT['validation_strategy'] == 'boostrap':
					log.warn("Validation bootstrapping is not supported when the validation target is per-tile; will generate random fixed validation target", 1)
				if self.PROJECT['validation_strategy'] in ('bootstrap', 'fixed'):
					# Split the extracted tiles into two groups
					datasets.split_tiles(save_folder, fraction=[-1, self.PROJECT['validation_fraction']], names=['training', 'validation'])
				if self.PROJECT['validation_strategy'] == 'k-fold':
					datasets.split_tiles(save_folder, fraction=[-1] * self.PROJECT['validation_k_fold'], names=[f'kfold-{i}' for i in range(self.PROJECT['validation_k_fold'])])

		self.generate_tfrecord(subfolder)

	def generate_tfrecord(self, subfolder=None):
		'''Create tfrecord files from a collection of raw images'''
		log.header('Writing TFRecord files...')
		subfolder = NO_LABEL if (not subfolder or subfolder=='') else subfolder
		tfrecord_dir = join(self.PROJECT['tfrecord_dir'], subfolder)
		tiles_dir = join(self.PROJECT['tiles_dir'], subfolder)

		# Check to see if subdirectories in the target folders are slide directories (contain images)
		#  or are further subdirectories (e.g. validation and training)
		log.info('Scanning tile directory structure...', 1)
		if sfutil.contains_nested_subdirs(tiles_dir):
			subdirs = [_dir for _dir in os.listdir(tiles_dir) if isdir(join(tiles_dir, _dir))]
			for subdir in subdirs:
				tfrecord_subdir = join(tfrecord_dir, subdir)
				tfrecords.write_tfrecords_multi(join(tiles_dir, subdir), tfrecord_subdir)
		else:
			tfrecords.write_tfrecords_multi(tiles_dir, tfrecord_dir)

		self.update_manifest()

		if self.PROJECT['delete_tiles']:
			shutil.rmtree(tiles_dir)		

	def delete_tiles(self, subfolder=None):
		'''Deletes all contents in the tiles directory (to be executed after TFRecords are generated).'''
		delete_folder = self.PROJECT['tiles_dir'] if not subfolder else join(self.PROJECT['tiles_dir'], subfolder)
		shutil.rmtree(delete_folder)
		log.info(f"Deleted tiles in folder {sfutil.green(delete_folder)}", 1)

	def initialize_model(self, model_name, train_tfrecords, validation_tfrecords, outcomes, model_type='categorical'):
		'''Prepares a Slideflow model using the provided outcome variable (category_header) 
		and a given set of training and validation tfrecords.'''
		# Using the project annotation file, assemble list of slides for training, as well as the slide annotations dictionary (output labels)
		model_dir = join(self.PROJECT['models_dir'], model_name)

		# Build a model using the slide list as input and the annotations dictionary as output labels
		SFM = sfmodel.SlideflowModel(model_dir, self.PROJECT['tile_px'], outcomes, train_tfrecords, validation_tfrecords,
																				manifest=sfutil.get_global_manifest(self.PROJECT['tfrecord_dir']),
																				use_fp16=self.PROJECT['use_fp16'],
																				model_type=model_type)
		return SFM

	def evaluate(self, model, category_header, model_type='categorical', filters=None, subfolder=None, checkpoint=None):
		'''Evaluates a saved model on a given set of tfrecords.'''
		log.header(f"Evaluating model {sfutil.bold(model)}...")
		subfolder = NO_LABEL if (not subfolder or subfolder=='') else subfolder
		model_name = model.split('/')[-1]
		tfrecord_path = join(self.PROJECT['tfrecord_dir'], subfolder)
		tfrecords = []

		# Check if given subfolder contains split data (tiles split into multiple TFRecords, likely for validation testing)
		# If true, can merge inputs and evaluate all data.
		subdirs = [sd for sd in os.listdir(tfrecord_path) if isdir(join(tfrecord_path, sd))]
		if len(subdirs):
			if sfutil.yes_no_input(f"Warning: TFRecord directory {sfutil.green(subfolder)} contains data split into sub-directories ({', '.join([sfutil.green(s) for s in subdirs])}); merge for evaluation? [y/N] ", default='no'):
				folders_to_search = [join(tfrecord_path, subdir) for subdir in subdirs]
			else:
				return
		else:
			folders_to_search = [tfrecord_path]
		for folder in folders_to_search:
			tfrecords += glob(os.path.join(folder, "*.tfrecords"))

		# Set up model for evaluation
		outcomes = sfutil.get_outcomes_from_annotations(category_header, filters=filters, use_float=(model_type=='linear'))
		SFM = self.initialize_model(f"eval-{model_name}", None, None, outcomes, model_type=model_type)
		log.info(f"Evaluating {sfutil.bold(len(SFM.SLIDES))} tfrecords", 1)
		model_dir = join(self.PROJECT['models_dir'], model, "trained_model.h5") if model[-3:] != ".h5" else model
		results = SFM.evaluate(tfrecords=tfrecords, hp=None, model=model_dir, model_type=model_type, checkpoint=checkpoint, batch_size=EVAL_BATCH_SIZE)
		print(results)
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

	def train(self, models=None, subfolder=NO_LABEL, category_header='category', filters=None, resume_training=None, 
				checkpoint=None, pretrain='imagenet', supervised=True, batch_file=None, model_type='categorical',
				validation_target=None, validation_strategy=None, validation_fraction=None, validation_k_fold=None, k_fold_iter=None):
		'''Train model(s) given configurations found in batch_train.tsv.

		Args:
			models				(optional) Either string representing a model name or an array of strings containing model names. 
									Will train models with these names in the batch_train.tsv config file.
									Defaults to None, which will train all models in the batch_train.tsv config file.
			subfolder			(optional) Which dataset to pull tfrecord files from (subdirectory in tfrecord_dir); defaults to 'no_label'
			category_header		(optional) Which header in the annotation file to use for the output category. Defaults to 'category'
			filters				(optional) Dictionary of column names mapping to column values by which to filter slides using the annotation file.
			resume_training		(optional) Path to .h5 model to continue training
			checkpoint			(optional) Path to cp.ckpt from which to load weights
			supervised			(optional) Whether to use verbose output and save training progress to Tensorboard
			batch_file			(optional) Manually specify batch file to use for a hyperparameter sweep. If not specified, will use project default.
			model_type			(optional) Type of output variable, either categorical (default) or linear.
			validation_target 	(optional) Whether to select validation data on a 'per-slide' or 'per-tile' basis. If not specified, will use project default.
			validation_strategy	(optional) Validation dataset selection strategy (bootstrap, k-fold, fixed, none). If not specified, will use project default.
			validation_fraction	(optional) Fraction of data to use for validation testing. If not specified, will use project default.
			validation_k_fold 	(optional) K, if using k-fold validation. If not specified, will use project default.
			k_fold_iter			(optional) Which iteration to train if using k-fold validation. Defaults to training all iterations.

		Returns:
			A dictionary containing model names mapped to train_acc, val_loss, and val_acc
		'''
		# Get list of slides for training and establish validation plan
		batch_train_file = self.PROJECT['batch_train_config'] if not batch_file else sfutil.global_path(batch_file)
		validation_target = self.PROJECT['validation_target'] if not validation_target else validation_target
		validation_strategy = self.PROJECT['validation_strategy'] if not validation_strategy else validation_strategy
		validation_fraction = self.PROJECT['validation_fraction'] if not validation_fraction else validation_fraction
		validation_k_fold = self.PROJECT['validation_k_fold'] if not validation_k_fold else validation_k_fold
		tfrecord_dir = join(self.PROJECT['tfrecord_dir'], subfolder)
		results_log_path = os.path.join(self.PROJECT['root'], "results_log.csv")
		k_fold_iter = [k_fold_iter] if (k_fold_iter != None and type(k_fold_iter) != list) else k_fold_iter
		k_fold = validation_k_fold if validation_strategy in ('k-fold', 'bootstrap') else 0
		valid_k = [] if not k_fold else [kf for kf in range(k_fold) if ((k_fold_iter and kf in k_fold_iter) or (not k_fold_iter))]
																						
		# Quickly scan for errors (duplicate model names) and prepare models to train
		models_to_train = self._get_valid_models(batch_train_file, models)
		log.header(f"Training {len(models_to_train)} models...")

		# Next, prepare the multiprocessing manager (needed to free VRAM after training)
		manager = multiprocessing.Manager()
		results_dict = manager.dict()

		# Load outcomes from annotations file
		outcomes = sfutil.get_outcomes_from_annotations(category_header, filters=filters, use_float=(model_type == 'linear'))
		print()

		# Create a worker that can execute one round of training
		def trainer(results_dict, model_name, hp, k_fold_i=None):
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
			training_tfrecords, validation_tfrecords = tfrecords.get_training_and_validation_tfrecords(tfrecord_dir, outcomes, model_type,
																										validation_target=validation_target,
																										validation_strategy=validation_strategy,
																										validation_fraction=validation_fraction,
																										validation_k_fold=validation_k_fold,
																										k_fold_iter=k_fold_i)
			# Initialize model
			SFM = self.initialize_model(full_model_name, training_tfrecords, validation_tfrecords, outcomes, model_type=model_type)

			with open(os.path.join(self.PROJECT['models_dir'], full_model_name, 'hyperparameters.log'), 'w') as hp_file:
				hp_text = f"Tile pixel size: {self.PROJECT['tile_px']}\n"
				hp_text += f"Tile micron size: {self.PROJECT['tile_um']}\n"
				hp_text += str(hp)
				for s in sfutil.FORMATTING_OPTIONS:
					hp_text = hp_text.replace(s, "")
				hp_file.write(hp_text)

			# Execute training
			try:
				results = SFM.train(hp, pretrain=pretrain, 
										resume_training=resume_training, 
										checkpoint=checkpoint,
										supervised=supervised)
				results_dict.update({full_model_name: results})
				logged_epochs = [int(e[5:]) for e in results.keys() if e[:5] == 'epoch']
				
				if not DEBUGGING: experiment.log_metrics(results[f'epoch{max(logged_epochs)}'])
				del(SFM)
			except tf.errors.ResourceExhaustedError:
				log.error(f"Training failed for {sfutil.bold(model_name)}, GPU memory exceeded.", 0)
				del(SFM)
				return

		# Assembling list of models and hyperparameters from batch_train.tsv file
		batch_train_rows = []
		with open(batch_train_file) as csv_file:
			reader = csv.reader(csv_file, delimiter='\t')
			header = next(reader)
			for row in reader:
				batch_train_rows += [row]
			
		for row in batch_train_rows:
			# Read hyperparameters
			hp, model_name = self._get_hp(row, header)
			if model_name not in models_to_train: continue
			model_iterations = [model_name] if not k_fold else [f"{model_name}-kfold{k+1}" for k in valid_k]

			# Perform training
			if k_fold:
				for k in valid_k:
					if DEBUGGING:
						trainer(results_dict, model_name, hp, k+1)
					else:
						process = multiprocessing.Process(target=trainer, args=(results_dict, model_name, hp, k+1))
						process.start()
						process.join()
			else:
				if DEBUGGING:
					trainer(results_dict, model_name, hp)
				else:
					process = multiprocessing.Process(target=trainer, args=(results_dict, model_name, hp))
					process.start()
					process.join()

			# Record results
			for mi in model_iterations:
				if mi not in results_dict:
					log.error(f"Training failed for model {model_name} for an unknown reason")
				else:
					self._update_results_log(results_log_path, mi, results_dict[mi])
			log.complete(f"Training complete for model {model_name}, results saved to {sfutil.green(results_log_path)}")
		
		# Print summary of all models
		log.complete("Training complete; validation accuracies:", 0)
		for model in results_dict:
			final_metrics = results_dict[model]['final']
			print(f" - {sfutil.green(model)}: Train_Acc={str(final_metrics['train_acc'])}, " +
				f"Val_loss={final_metrics['val_loss']}, Val_Acc={final_metrics['val_acc']}" )

	def generate_heatmaps(self, model_name, filters=None, resolution='medium'):
		'''Creates predictive heatmap overlays on a set of slides. 

		Args:
			model_name		Which model to use for generating predictions
			filter_header	Column name for filtering input slides based on the project annotations file. 
			filter_values	List of values to include when filtering slides according to filter_header.
			resolution		Heatmap resolution (determines stride of tile predictions). 
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
		slide_list = sfutil.get_filtered_slide_paths(self.PROJECT['slides_dir'], filters=filters)
		heatmaps_folder = os.path.join(self.PROJECT['root'], 'heatmaps')
		if not os.path.exists(heatmaps_folder): os.makedirs(heatmaps_folder)
		model_path = model_name if model_name[-3:] == ".h5" else join(self.PROJECT['models_dir'], model_name, 'trained_model.h5')

		c = convoluter.Convoluter(self.PROJECT['tile_px'], self.PROJECT['tile_um'], batch_size=64,
																					use_fp16=self.PROJECT['use_fp16'],
																					stride_div=stride_div,
																					save_folder=heatmaps_folder,
																					roi_dir=self.PROJECT['roi_dir'])
		c.load_slides(slide_list)
		c.build_model(model_path)
		c.convolute_slides(save_heatmaps=True, save_final_layer=True, export_tiles=False)

	def generate_mosaic(self, model=None, filters=None, focus_header=None, focus_values=None, subfolder=None, resolution="medium"):
		'''Generates a mosaic map with dimensionality reduction on penultimate layer weights. Tile data is extracted from the provided
		set of TFRecords and predictions are calculated using the specified model.'''
		
		log.header("Generating mosaic map...")
		subfolder = NO_LABEL if (not subfolder or subfolder=='') else subfolder
		slide_list = sfutil.get_filtered_tfrecords_paths(join(self.PROJECT['tfrecord_dir'], subfolder), filters=filters)
		if focus_header and focus_values:
			focus_list = sfutil.get_filtered_tfrecords_paths(join(self.PROJECT['tfrecord_dir'], subfolder), filters=filters)
		else:
			focus_list = None
		mosaic = Mosaic(save_dir=self.PROJECT['root'], resolution=resolution)
		mosaic.generate_from_tfrecords(slide_list, model=model, image_size=self.PROJECT['tile_px'], focus=focus_list)

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

	def create_blank_annotations_file(self, outfile=None, slides_dir=None):
		'''Creates an example blank annotations file.'''
		if not outfile: 
			outfile = self.PROJECT['annotations']
		if not slides_dir:
			slides_dir = self.PROJECT['slides_dir']

		with open(outfile, 'w') as csv_outfile:
			csv_writer = csv.writer(csv_outfile, delimiter=',')
			header = [TCGA.patient, 'dataset', 'category']
			csv_writer.writerow(header)

	def associate_slide_names(self):
		'''Experimental function used to automatically associated patient names with slide filenames in the annotations file.'''
		sfutil.verify_annotations_slides(self.PROJECT['slides_dir'])

	def update_manifest(self, force_update=False):
		'''Updates manifest file in the TFRecord directory, used to track number of records and verify annotations.
		
		Args:
			force_update	If True, will re-validate contents of all TFRecords. If False, will only validate
								contents of TFRecords not yet in the manifest
		'''
		sfutil.update_tfrecord_manifest(directory=self.PROJECT['tfrecord_dir'], 
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

		if not SKIP_VERIFICATION:
			log.header("Verifying Annotations...")
			sfutil.verify_annotations_slides(slides_dir=self.PROJECT['slides_dir'])
			log.header("Verifying TFRecord manifest...")
			self.update_manifest()

		# Load annotations
		sfutil.load_annotations(self.PROJECT['annotations'])

	def save_project(self):
		'''Saves current project configuration as "settings.json".'''
		sfutil.write_json(self.PROJECT, join(self.PROJECT['root'], 'settings.json'))

	def create_project(self):
		'''Prompts user to provide all relevant project configuration and saves configuration to "settings.json".'''
		# General setup and slide configuration
		project = {'root': sfutil.PROJECT_DIR}
		project['name'] = input("What is the project name? ")
		project['slides_dir'] = sfutil.dir_input("Where are the SVS slides stored? [./slides] ",
									default='./slides', create_on_invalid=True)
		project['roi_dir'] = sfutil.dir_input("Where are the ROI files (CSV) stored? [./slides] ",
									default='./slides', create_on_invalid=True)
		
		# Ask for annotations file location; if one has not been made, offer to create a blank template and then exit
		if not sfutil.yes_no_input("Has an annotations (CSV) file already been created? [y/N] ", default='no'):
			if sfutil.yes_no_input("Create a blank annotations file? [Y/n] ", default='yes'):
				project['annotations'] = sfutil.file_input("Where will the annotation file be located? [./annotations.csv] ", 
									default='./annotations.csv', filetype="csv", verify=False)
				self.create_blank_annotations_file(project['annotations'], project['slides_dir'])
		else:
			project['annotations'] = sfutil.file_input("Where is the project annotations (CSV) file located? [./annotations.csv] ", 
									default='./annotations.csv', filetype="csv")

		# Slide tessellation
		project['tiles_dir'] = sfutil.dir_input("Where will the tessellated image tiles be stored? (recommend SSD) [./tiles] ",
									default='./tiles', create_on_invalid=True)
		project['use_tfrecord'] = sfutil.yes_no_input("Store tiles in TFRecord format? (required for training) [Y/n] ", default='yes')
		if project['use_tfrecord']:
			project['delete_tiles'] = sfutil.yes_no_input("Should raw tile images be deleted after TFRecord storage? [Y/n] ", default='yes')
			project['tfrecord_dir'] = sfutil.dir_input("Where should the TFRecord files be stored? (recommend HDD) [./tfrecord] ",
									default='./tfrecord', create_on_invalid=True)
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
		project['validation_target'] = sfutil.choice_input("How should validation data be selected by default, per-tile or per-slide? [per-slide] ", valid_choices=['per-tile', 'per-slide'], default='per-slide')
		if project['validation_target'] == 'per-slide':
			project['validation_strategy'] = sfutil.choice_input("Which validation strategy should be used by default, k-fold, bootstrap, or fixed? ", valid_choices=['k-fold', 'bootstrap', 'fixed', 'none'])
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
		sys.exit()
