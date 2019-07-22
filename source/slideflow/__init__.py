import argparse
import os
import sys
import shutil
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf

from os.path import join, isfile, exists, isdir
from pathlib import Path
from glob import glob
from random import shuffle
import csv

import gc
import subprocess
import slideflow.trainer.model as sfmodel
import itertools
import multiprocessing
import slideflow.util as sfutil
from slideflow.util import datasets, tfrecords, TCGAAnnotations, log

# TODO: scan for duplicate SVS files (especially when converting TCGA long-names 
# 	to short-names, e.g. when multiple diagnostic slides are present)
# TODO: automatic loading of training configuration even when training one model

__version__ = "0.9.6"

SKIP_VERIFICATION = False
NUM_THREADS = 4
EVAL_BATCH_SIZE = 64
NO_LABEL = 'no_label'

def set_logging_level(level):
	sfutil.LOGGING_LEVEL.INFO = level

def select_gpu(number):
	os.environ["CUDA_VISIBLE_DEVICES"]=str(number)
	print(f"Requesting GPU #{number}")

class SlideFlowProject:
	MANIFEST = None
	USE_FP16 = True

	def __init__(self, project_folder):
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
		
	def extract_tiles(self, filter_header=None, filter_values=None, subfolder=None, skip_validation=False):
		import slideflow.convoluter as convoluter

		log.header("Extracting image tiles...")
		subfolder = NO_LABEL if (not subfolder or subfolder=='') else subfolder
		convoluter.NUM_THREADS = NUM_THREADS
		slide_list = sfutil.get_filtered_slide_paths(self.PROJECT['slides_dir'], self.PROJECT['annotations'], filter_header=filter_header,
																				  							  filter_values=filter_values)
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
			self.separate_validation_tiles(save_folder)

		self.generate_tfrecord(subfolder)

	def separate_validation_tiles(self, folder):
		# If validation is performed per-tile, separate tiles from each slide into 
		#  the necessary number of groups, before combining into tfrecords
		val_target = self.PROJECT['validation_target']
		val_strategy = self.PROJECT['validation_strategy']
		val_fraction = self.PROJECT['validation_fraction']
		k_fold = self.PROJECT['validation_k_fold']

		if val_target == 'per-tile':
			if val_strategy == 'boostrap':
				log.warn("Validation bootstrapping is not supported when the validation target is per-tile; will generate random fixed validation target", 1)
			if val_strategy in ('bootstrap', 'fixed'):
				# Split the extracted tiles into two groups
				datasets.split_tiles(folder, fraction=[-1, val_fraction], names=['training', 'validation'])
			if val_strategy == 'k-fold':
				datasets.split_tiles(folder, fraction=[-1] * k_fold, names=[f'kfold-{i}' for i in range(k_fold)])

	def generate_tfrecord(self, subfolder=None):
		'''Create tfrecord files from a collection of raw images'''
		log.header('Writing TFRecord files...')
		subfolder = NO_LABEL if (not subfolder or subfolder=='') else subfolder
		tfrecord_dir = join(self.PROJECT['tfrecord_dir'], subfolder)
		tiles_dir = join(self.PROJECT['tiles_dir'], subfolder)

		# Check to see if subdirectories in the target folders are case directories (contain images)
		#  or are further subdirectories (e.g. validation and training)
		log.info('Scanning tile directory structure...', 1)
		if sfutil.contains_nested_subdirs(tiles_dir):
			subdirs = [_dir for _dir in os.listdir(tiles_dir) if isdir(join(tiles_dir, _dir))]
			for subdir in subdirs:
				tfrecord_subdir = join(tfrecord_dir, subdir)
				tfrecords.write_tfrecords_multi(join(tiles_dir, subdir), tfrecord_subdir)
				self.update_manifest(tfrecord_subdir)
		else:
			tfrecords.write_tfrecords_multi(tiles_dir, tfrecord_dir)
			self.update_manifest(tfrecord_dir)

		if self.PROJECT['delete_tiles']:
			shutil.rmtree(tiles_dir)		

	def delete_tiles(self, subfolder=None):
		delete_folder = self.PROJECT['tiles_dir'] if not subfolder else join(self.PROJECT['tiles_dir'], subfolder)
		shutil.rmtree(delete_folder)
		log.info(f"Deleted tiles in folder {sfutil.green(delete_folder)}", 1)

	def checkpoint_to_h5(self, model_name):
		tfrecords.checkpoint_to_h5(self.PROJECT['models_dir'], model_name)

	def get_training_and_validation_tfrecords(self, subfolder, slide_list, validation_target=None, validation_strategy=None, 
												validation_fraction=None, validation_k_fold=None, k_fold_iter=None):
		'''From a specified subfolder within the project's main TFRecord folder, prepare a training set and validation set.
		Returns two arrays: an array of full paths to training tfrecords, and an array of paths to validation tfrecords.''' 
		tfrecord_dir = join(self.PROJECT['tfrecord_dir'], subfolder)
		subdirs = [sd for sd in os.listdir(tfrecord_dir) if isdir(join(tfrecord_dir, sd))]
		if k_fold_iter: k_fold_index = int(k_fold_iter)-1
		training_tfrecords = []
		validation_tfrecords = []
		validation_plan = {}

		val_target = validation_target if validation_target else self.PROJECT['validation_target']
		val_strategy = validation_strategy if validation_strategy else self.PROJECT['validation_strategy']
		val_fraction = validation_fraction if validation_fraction else self.PROJECT['validation_fraction']
		k_fold = validation_k_fold if validation_k_fold else self.PROJECT['validation_k_fold']

		# If validation is done per-tile, use pre-separated TFRecord files (validation separation done at time of TFRecord creation)
		if val_target == 'per-tile':
			log.info(f"Loading pre-separated TFRecords in {sfutil.green(subfolder)}", 1)
			if val_strategy == 'bootstrap':
				log.warn("Validation bootstrapping is not supported when the validation target is per-tile; using tfrecords in 'training' and 'validation' subdirectories", 1)
			if val_strategy in ('bootstrap', 'fixed'):
				# Load tfrecords from 'validation' and 'training' subdirectories
				if ('validation' not in subdirs) or ('training' not in subdirs):
					log.error(f"{sfutil.bold(val_strategy)} selected as validation strategy but tfrecords are not organized as such (unable to find 'training' or 'validation' subdirectories)")
					sys.exit()
				training_tfrecords += glob(join(tfrecord_dir, 'training', "*.tfrecords"))
				validation_tfrecords += glob(join(tfrecord_dir, 'validation', "*.tfrecords"))
			elif val_strategy == 'k-fold':
				log.warn("No k-fold iteration specified; assuming iteration #1", 1)
				if k_fold_iter > k_fold:
					log.error(f"K-fold iteration supplied ({k_fold_iter}) exceeds the project K-fold setting ({k_fold})", 1)
					sys.exit()
				for k in range(k_fold):
					if not exists(join(tfrecord_dir, f'kfold-{k}')):
						log.error(f"Unable to find kfold-{k} in {sfutil.green(tfrecord_dir)}", 1)
						sys.exit()
					if k == k_fold_index:
						validation_tfrecords += glob(join(tfrecord_dir, f'kfold-{k}', "*.tfrecords"))
					else:
						training_tfrecords += glob(join(tfrecord_dir, f'kfold-{k}', "*.tfrecords"))
			elif val_strategy == 'none':
				if len(subdirs):
					log.error(f"Validation strategy set as 'none' but the TFRecord directory has been configured for validation (contains subfolders {', '.join(subdirs)})", 1)
					sys.exit()
			# Remove tfrecords not specified in slide_list
			training_tfrecords = [tfr for tfr in training_tfrecords if tfr.split('/')[-1][:-10] in slide_list]
			validation_tfrecords = [tfr for tfr in validation_tfrecords if tfr.split('/')[-1][:-10] in slide_list]

		# If validation is done per-slide, create and log a validation subset
		elif val_target == 'per-slide':
			validation_log = join(tfrecord_dir, "validation_plan.json")
			tfrecords = glob(join(tfrecord_dir, "*.tfrecords"))
			tfrecords = [tfr for tfr in glob(join(tfrecord_dir, "*.tfrecords")) if tfr.split('/')[-1][:-10] in slide_list]
			shuffle(tfrecords)
			if len(subdirs):
				log.error(f"Validation target set to 'per-slide', but the TFRecord directory has validation configured per-tile (contains subfolders {', '.join(subdirs)}", 1)
				sys.exit()
			if val_strategy == 'bootstrap':
				num_val = int(val_fraction * len(tfrecords))
				log.info(f"Using boostrap validation: selecting {sfutil.bold(num_val)} slides at random to use for validation testing", 1)
				validation_tfrecords = tfrecords[0:num_val]
				training_tfrecords = tfrecords[num_val:]
			elif val_strategy == 'fixed':
				num_val = int(val_fraction * len(tfrecords))
				# Start by checking for a valid plan
				if not exists(validation_log):
					log.info(f"No validation log found; will log validation plan at {sfutil.green(validation_log)}", 1)
				else:
					validation_plan = sfutil.load_json(validation_log)
					if 'fixed' not in validation_plan:
						log.info(f"No fixed validation plan found in {sfutil.green(validation_log)}; will create new plan", 1)
					elif len(validation_plan['fixed']) != num_val:
						log.warn(f"Fixed validation plan detected at {sfutil.green(validation_log)}, but does not match provided slide set; will create new validation plan", 1)
					else:
						valid_plan = True
						for tf in validation_plan['fixed']:
							if tf[:-10] not in slide_list:
								log.warn(f"Fixed validation plan detected at {sfutil.green(validation_log)}, but contains tfrecords not in slide set; will create new validation plan", 1)
								valid_plan = False
						if valid_plan:
							# Use existing valid plan
							log.info(f"Using fixed validation plan detected at {sfutil.green(validation_log)}", 1)
							training_tfrecords = [tfr for tfr in tfrecords if tfr.split('/')[-1] not in validation_plan['fixed']] 
							validation_tfrecords = [tfr for tfr in tfrecords if tfr.split('/')[-1] in validation_plan['fixed']] 
							log.info(f"Using {sfutil.bold(len(training_tfrecords))} TFRecords for training, {sfutil.bold(len(validation_tfrecords))} for validation", 1)
							return training_tfrecords, validation_tfrecords
				# Create a new fixed validation plan and log plan results
				validation_tfrecords = tfrecords[0:num_val]
				training_tfrecords = tfrecords[num_val:]
				validation_plan['fixed'] = [tfr.split('/')[-1] for tfr in validation_tfrecords]
				sfutil.write_json(validation_plan, validation_log)
			elif val_strategy == 'k-fold':
				if not exists(validation_log):
					log.info(f"No validation log found; will log validation plan at {sfutil.green(validation_log)}", 1)
				else:
					validation_plan = sfutil.load_json(validation_log)
					if 'k-fold' not in validation_plan:
						log.info(f"No k-fold validation plan found in {sfutil.green(validation_log)}; will create new plan", 1)
					elif len(validation_plan['k-fold']) != k_fold:
						log.warn(f"K-fold validation plan detected at {sfutil.green(validation_log)}, but logged k ({len(validation_plan['k-fold'])}) does not match project setting ({k_fold}); will create new validation plan", 1)
					else:
						logged_cases = []
						for fold in range(len(validation_plan['k-fold'])):
							logged_cases += validation_plan['k-fold'][fold]
						if len(logged_cases) != len(tfrecords):
							log.warn(f"K-fold validation plan detected at {sfutil.green(validation_log)}, but number of cases do not match; will create new plan", 1)
						else:
							valid_plan = True
							for tf in logged_cases:
								if tf[:-10] not in slide_list:
									log.warn(f"K-fold validation plan detected at {sfutil.green(validation_log)}, but contains tfrecords not in slide set; will create new validation plan", 1)
									valid_plan = False
							if valid_plan:
								log.info(f"Using k-fold validation plan detected at {sfutil.green(validation_log)}", 1)
								training_tfrecords = [tfr for tfr in tfrecords if (tfr.split('/')[-1] not in validation_plan['k-fold'][k_fold_index]) and 
																				  (tfr.split('/')[-1] in logged_cases)]
								validation_tfrecords = [tfr for tfr in tfrecords if tfr.split('/')[-1] in validation_plan['k-fold'][k_fold_index]] 
								log.info(f"Using {sfutil.bold(len(training_tfrecords))} TFRecords for training, {sfutil.bold(len(validation_tfrecords))} for validation", 1)
								return training_tfrecords, validation_tfrecords
				# Create a new k-fold validation plan and log plan results
				validation_plan['k-fold'] = []

				def split(a, n):
					k, m = divmod(len(a), n)
					return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

				split_records = list(split(tfrecords, k_fold))
				for k in range(k_fold):
					if k == k_fold_index:
						validation_tfrecords = split_records[k]
					else:
						training_tfrecords += split_records[k]
					validation_plan['k-fold'] += [[tfr.split('/')[-1] for tfr in split_records[k]]]
				sfutil.write_json(validation_plan, validation_log)

			elif val_strategy == 'none':
				training_tfrecords += tfrecords

		log.info(f"Using {sfutil.bold(len(training_tfrecords))} TFRecords for training, {sfutil.bold(len(validation_tfrecords))} for validation", 1)
		return training_tfrecords, validation_tfrecords

	def initialize_model(self, model_name, train_tfrecords, validation_tfrecords, category_header, filter_header=None, filter_values=None, skip_validation=False, model_type='categorical'):
		# Assemble list of slides for training and annotations dictionary
		slide_to_category = sfutil.get_annotations_dict(self.PROJECT['annotations'], 'slide', category_header, 
																							filter_header=filter_header, 
																							filter_values=filter_values,
																							use_encode=(model_type=='categorical'),
																							use_float=(model_type=='linear'))

		model_dir = join(self.PROJECT['models_dir'], model_name)

		SFM = sfmodel.SlideflowModel(model_dir, self.PROJECT['tile_px'], slide_to_category, train_tfrecords, validation_tfrecords,
																				manifest=self.MANIFEST, 
																				use_fp16=self.PROJECT['use_fp16'],
																				model_type=model_type)
		return SFM

	def evaluate(self, model, subfolder, category_header, model_type='categorical', filter_header=None, filter_values=None, checkpoint=None):
		log.header(f"Evaluating model {sfutil.bold(model)}...")
		model_name = model.split('/')[-1]
		tfrecord_path = join(self.PROJECT['tfrecord_dir'], subfolder)
		tfrecords = []

		# Check if given subfolder contains split data
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
		log.info(f"Evaluating {sfutil.bold(len(tfrecords))} tfrecords", 1)

		# Perform evaluation
		SFM = self.initialize_model(f"eval-{model_name}", None, None, category_header, filter_header, filter_values, skip_validation=True, model_type=model_type)
		model_dir = join(self.PROJECT['models_dir'], model, "trained_model.h5") if model[-3:] != ".h5" else model
		results = SFM.evaluate(tfrecords=tfrecords, hp=None, model=model_dir, model_type=model_type, checkpoint=checkpoint, batch_size=EVAL_BATCH_SIZE)
		print(results)
		return results

	def train(self, models=None, subfolder=NO_LABEL, category_header='category', filter_header=None, filter_values=None, resume_training=None, 
				checkpoint=None, pretrain='imagenet', supervised=True, batch_file=None, model_type='categorical',
				validation_target=None, validation_strategy=None, validation_fraction=None, validation_k_fold=None, k_fold_iter=None):
		'''Train model(s) given configurations found in batch_train.tsv.
		Args:
			models			(optional) Either string representing a model name or an array of strings containing model names. 
								Will train models with these names in the batch_train.tsv config file.
								Defaults to None, which will train all models in the batch_train.tsv config file.
			subfolder		(optional) Which dataset to pull tfrecord files from (subdirectory in tfrecord_dir); defaults to 'no_label'
			category_header	(optional) Which header in the annotation file to use for the output category. Defaults to 'category'
			filter_header	(optional) Filter slides to inculde in training by this column
			filter_values	(optional) String or array of strings. Only train on slides with these values in the filter_header column.
			resume_training	(optional) Path to .h5 model to continue training
			checkpoint		(optional) Path to cp.ckpt from which to load weights
			supervised		(optional) Whether to use verbose output and save training progress to Tensorboard
		Returns:
			A dictionary containing model names mapped to train_acc, val_loss, and val_acc'''

		# Get list of slides for training and establish validation plan
		batch_train_file = self.PROJECT['batch_train_config'] if not batch_file else sfutil.global_path(batch_file)
		validation_target = self.PROJECT['validation_target'] if not validation_target else validation_target
		validation_strategy = self.PROJECT['validation_strategy'] if not validation_strategy else validation_strategy
		validation_fraction = self.PROJECT['validation_fraction'] if not validation_fraction else validation_fraction
		validation_k_fold = self.PROJECT['validation_k_fold'] if not validation_k_fold else validation_k_fold
		k_fold_iter = [k_fold_iter] if type(k_fold_iter) != list else k_fold_iter

		slide_to_category = sfutil.get_annotations_dict(self.PROJECT['annotations'], 'slide', category_header, 
																							filter_header=filter_header, 
																							filter_values=filter_values,
																							use_encode=(model_type=='categorical'))
		slide_list = slide_to_category.keys()

		# Quickly scan for errors (duplicate model names) and prepare models to train
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

		log.header(f"Training {len(models_to_train)} models...")

		# Next, prepare the multiprocessing manager (needed for Keras memory management)
		manager = multiprocessing.Manager()
		results_dict = manager.dict()

		# Create a worker that can execute one round of training
		def trainer(results_dict, model_name, hp, k_fold_i=None):
			if supervised:
				k_fold_msg = "" if not k_fold_i else f" ({validation_strategy} iteration #{k_fold_i})"
				log.empty(f"Training model {sfutil.bold(model_name)}{k_fold_msg}...", 1)
				log.info(hp, 1)
			full_model_name = model_name if not k_fold_i else model_name+f"-kfold{k_fold_i}"

			training_tfrecords, validation_tfrecords = self.get_training_and_validation_tfrecords(subfolder, slide_list, validation_target=validation_target,
																														 validation_strategy=validation_strategy,
																														 validation_fraction=validation_fraction,
																														 validation_k_fold=validation_k_fold,
																														 k_fold_iter=k_fold_i)

			SFM = self.initialize_model(full_model_name, training_tfrecords, validation_tfrecords, category_header, filter_header, filter_values, model_type=model_type)
			with open(os.path.join(self.PROJECT['models_dir'], full_model_name, 'hyperparameters.log'), 'w') as hp_file:
				hp_text = f"Tile pixel size: {self.PROJECT['tile_px']}\n"
				hp_text += f"Tile micron size: {self.PROJECT['tile_um']}\n"
				hp_text += str(hp)
				for s in sfutil.FORMATTING_OPTIONS:
					hp_text = hp_text.replace(s, "")
				hp_file.write(hp_text)
			try:
				train_acc, val_loss, val_acc = SFM.train(hp, pretrain=pretrain, 
															resume_training=resume_training, 
															checkpoint=checkpoint,
															supervised=supervised)

				results_dict.update({model_name: {'train_acc': max(train_acc),
													'val_loss': val_loss,
													'val_acc': val_acc } 			})
				del(SFM)

			except tf.errors.ResourceExhaustedError:
				log.error(f"Training failed for {sfutil.bold(model_name)}, GPU memory exceeded.", 0)
				del(SFM)
				return

		k_fold = validation_k_fold if validation_strategy in ('k-fold', 'bootstrap') else 0

		# Begin assembling models and hyperparameters from batch_train.tsv file
		with open(batch_train_file) as csv_file:
			reader = csv.reader(csv_file, delimiter='\t')
			header = next(reader)
			log_path = os.path.join(self.PROJECT['root'], "results_log.csv")
			already_started = os.path.exists(log_path)
			with open(log_path, "a") as results_file:
				writer = csv.writer(results_file)
				if not already_started:
					results_header = ['model', 'train_acc', 'val_loss', 'val_acc']
					writer.writerow(results_header)
			model_name_i = header.index('model_name')
			# Get all column headers except 'model_name'
			args = header[0:model_name_i] + header[model_name_i+1:]
			for row in reader:
				model_name = row[model_name_i]
				if model_name not in models_to_train: continue
				hp = sfmodel.HyperParameters()
				for arg in args:
					value = row[header.index(arg)]
					if arg in hp._get_args():
						if arg != 'finetune_epochs':
							arg_type = type(getattr(hp, arg))
							setattr(hp, arg, arg_type(value))
						else:
							epochs = [int(i) for i in value.split(',')]
							setattr(hp, arg, epochs)
					else:
						log.error(f"Unknown argument '{arg}' found in training config file.", 0)

				# Start the model training
				if k_fold:
					train_acc_list = []
					val_loss_list = []
					val_acc_list = []
					error_encountered = False
					for k in range(k_fold):
						if k not in k_fold_iter:
							continue
						p = multiprocessing.Process(target=trainer, args=(results_dict, model_name, hp, k+1))
						p.start()
						p.join()
						if model_name not in results_dict:
							log.error(f"Training failed for model {model_name} for an unknown reason.")
							error_encountered = True
							break
						train_acc_list += [results_dict[model_name]['train_acc']]
						val_loss_list += [results_dict[model_name]['val_loss']]
						val_acc_list += [results_dict[model_name]['val_acc']]
					if error_encountered: continue
					avg_train_acc = sum(train_acc_list) / len(train_acc_list)
					avg_val_loss = sum(val_loss_list) / len(val_loss_list)
					avg_val_acc = sum(val_acc_list) / len(val_acc_list)
					log.complete(f"Training complete for model {model_name}, validation accuracy {sfutil.info(str(avg_val_acc))} after {k_fold}-fold validation", 0)
					for k in range(k_fold):
						log.complete(f"Set {k+1} accuracy: {sfutil.info(str(val_acc_list[k]))}", 1)
					with open(os.path.join(self.PROJECT['root'], "results_log.csv"), "a") as results_file:
						writer = csv.writer(results_file)
						writer.writerow([model_name, avg_train_acc, 
													avg_val_loss,
													avg_val_acc])
				else:
					p = multiprocessing.Process(target=trainer, args=(results_dict, model_name, hp))
					p.start()
					p.join()
					if model_name not in results_dict:
						log.error(f"Training failed for model {model_name} for an unknown reason")
						continue
					# Record the model accuracies
					log.complete(f"Training complete for model {model_name}, validation accuracy {sfutil.info(str(results_dict[model_name]['val_acc']))}", 0)	
					with open(os.path.join(self.PROJECT['root'], "results_log.csv"), "a") as results_file:
						writer = csv.writer(results_file)
						writer.writerow([model_name, results_dict[model_name]['train_acc'], 
													results_dict[model_name]['val_loss'],
													results_dict[model_name]['val_acc']])					
		
		# Print final results
		log.complete("Training complete; validation accuracies:", 0)
		for model in results_dict:
			print(f" - {sfutil.green(model)}: Train_Acc={str(results_dict[model]['train_acc'])}, " +
				f"Val_loss={results_dict[model]['val_loss']}, Val_Acc={results_dict[model]['val_acc']}" )

	def generate_heatmaps(self, model_name, filter_header=None, filter_values=None, resolution='medium'):
		import slideflow.convoluter as convoluter
		
		log.header("Generating heatmaps...")
		resolutions = {'low': 1, 'medium': 2, 'high': 4}
		try:
			stride_div = resolutions[resolution]
		except KeyError:
			log.error(f"Invalid resolution '{resolution}': must be either 'low', 'medium', or 'high'.")
			return
		slide_list = sfutil.get_filtered_slide_paths(self.PROJECT['slides_dir'], self.PROJECT['annotations'], filter_header=filter_header,
																				  							  filter_values=filter_values)
		heatmaps_folder = os.path.join(self.PROJECT['root'], 'heatmaps')
		if not os.path.exists(heatmaps_folder): os.makedirs(heatmaps_folder)

		c = convoluter.Convoluter(self.PROJECT['tile_px'], self.PROJECT['tile_um'], batch_size=64,
																					use_fp16=self.PROJECT['use_fp16'],
																					stride_div=stride_div,
																					save_folder=heatmaps_folder,
																					roi_dir=self.PROJECT['roi_dir'])
		c.load_slides(slide_list)
		c.build_model(join(self.PROJECT['models_dir'], model_name, 'trained_model.h5'))
		c.convolute_slides(save_heatmaps=True, save_final_layer=True, export_tiles=False)

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

	def create_blank_annotations_file(self, outfile=None, slides_dir=None, scan_for_cases=False):
		case_header_name = TCGAAnnotations.case

		if not outfile: 
			outfile = self.PROJECT['annotations']
		if not slides_dir:
			slides_dir = self.PROJECT['slides_dir']

		with open(outfile, 'w') as csv_outfile:
			csv_writer = csv.writer(csv_outfile, delimiter=',')
			header = [case_header_name, 'dataset', 'category']
			csv_writer.writerow(header)

		if scan_for_cases:
			sfutil.verify_annotations(outfile, slides_dir=slides_dir)

	def associate_slide_names(self):
		sfutil.verify_annotations(self.PROJECT['annotations'], self.PROJECT['slides_dir'])

	def update_manifest(self, dataset_label, skip_verification=False):
		manifest_path = sfutil.global_path('manifest.json')
		if not os.path.exists(manifest_path):
			self.generate_manifest()
		else:
			input_dir = join(self.PROJECT['tfrecord_dir'], dataset_label)
			annotations = sfutil.get_annotations_dict(self.PROJECT['annotations'], key_name="slide", value_name="category")
			tfrecord_files = glob(os.path.join(input_dir, "*.tfrecords"))
			self.MANIFEST = sfutil.load_json(manifest_path)
			if not skip_verification:
				self.MANIFEST.update(sfutil.verify_tiles(annotations, tfrecord_files))
			sfutil.write_json(self.MANIFEST, manifest_path)

	def generate_manifest(self):
		self.MANIFEST = {}
		input_dir = self.PROJECT['tfrecord_dir']
		annotations = sfutil.get_annotations_dict(self.PROJECT['annotations'], key_name="slide", value_name="category")
		tfrecord_files = glob(os.path.join(input_dir, "**/*.tfrecords"))
		self.MANIFEST = sfutil.verify_tiles(annotations, tfrecord_files)
		sfutil.write_json(self.MANIFEST, sfutil.global_path("manifest.json"))

	def load_project(self, directory):
		if exists(join(directory, "settings.json")):
			self.PROJECT = sfutil.load_json(join(directory, "settings.json"))
			log.empty("Project configuration loaded.\n")
		else:
			raise OSError(f'Unable to locate settings.json at location "{directory}".')

		# Enable logging
		log.logfile = sfutil.global_path("log.log")

		if not SKIP_VERIFICATION:
			sfutil.verify_annotations(self.PROJECT['annotations'], slides_dir=self.PROJECT['slides_dir'])
		if os.path.exists(sfutil.global_path("manifest.json")):
			self.MANIFEST = sfutil.load_json(sfutil.global_path("manifest.json"))
		else:
			self.generate_manifest()

	def save_project(self):
		sfutil.write_json(self.PROJECT, join(self.PROJECT['root'], 'settings.json'))

	def create_project(self):
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
				self.create_blank_annotations_file(project['annotations'], project['slides_dir'], scan_for_cases=sfutil.yes_no_input("Scan slide folder for case names? [Y/n] ", default='yes'))
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
		project['validation_target'] = sfutil.choice_input("How should validation data be selected, per-tile or per-slide? [per-slide] ", valid_choices=['per-tile', 'per-slide'], default='per-slide')
		if project['validation_target'] == 'per-slide':
			project['validation_strategy'] = sfutil.choice_input("Which validation strategy should be used, k-fold, bootstrap, or fixed? ", valid_choices=['k-fold', 'bootstrap', 'fixed', 'none'])
		else:
			project['validation_strategy'] = sfutil.choice_input("Which validation strategy should be used, k-fold or fixed? ", valid_choices=['k-fold', 'fixed', 'none'])
		if project['validation_strategy'] == 'k-fold':
			project['validation_k_fold'] = sfutil.int_input("What is K? [3] ", default=3)
		elif project['validation_strategy'] == 'bootstrap':
			project['validation_k_fold'] = sfutil.int_input("How many iterations should be performed when bootstrapping? [3] ", default=3)
		else:
			project['validation_k_fold'] = 0

		sfutil.write_json(project, join(sfutil.PROJECT_DIR, 'settings.json'))
		self.PROJECT = project

		# Write a sample actions.py file
		with open('sample_actions.py', 'r') as sample_file:
			sample_actions = sample_file.read()
			with open(os.path.join(sfutil.PROJECT_DIR, 'actions.py'), 'w') as actions_file:
				actions_file.write(sample_actions)

		print("\nProject configuration saved.\n")
		sys.exit()
