import argparse
import os
import sys
import shutil
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf

from os.path import join, isfile, exists
from pathlib import Path
from glob import glob
import csv

import gc
import subprocess
import convoluter
import sfmodel
import itertools
import multiprocessing
from util import datasets, tfrecords, sfutil
from util.sfutil import TCGAAnnotations, log

# TODO: scan for duplicate SVS files (especially when converting TCGA long-names 
# 	to short-names, e.g. when multiple diagnostic slides are present)
# TODO: automatic loading of training configuration even when training one model

SKIP_VERIFICATION = False
NUM_THREADS = 4
EVAL_BATCH_SIZE = 64
NO_LABEL = 'no_label'

def set_logging_level(level):
	sfutil.LOGGING_LEVEL.INFO = level

def select_gpu(number):
	os.environ["CUDA_VISIBLE_DEVICES"]=str(number)

class SlideFlowProject:
	MANIFEST = None
	USE_FP16 = True

	def __init__(self, project_folder):
		os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
		log.header('''SlideFlow v0.9.3\n================''')
		log.header('''Loading project...''')
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
		
	def extract_tiles(self, filter_header=None, filter_values=None, dataset_label=None, skip_validation=False):
		'''filter is a dict whose keys correspond with a header label and whose value is a 
		list of acceptable values; all other cases will be ignored
		
		If a single case is supplied, extract tiles for just that case.'''
		log.header("Extracting image tiles...")
		dataset_label = NO_LABEL if not dataset_label else dataset_label
		convoluter.NUM_THREADS = NUM_THREADS
		slide_list = sfutil.get_filtered_slide_paths(self.PROJECT['slides_dir'], self.PROJECT['annotations'], filter_header=filter_header,
																				  							  filter_values=filter_values)
		log.info(f"Extracting tiles from {len(slide_list)} slides", 1)

		save_folder = join(self.PROJECT['tiles_dir'], dataset_label)
		if not os.path.exists(save_folder):
			os.makedirs(save_folder)

		c = convoluter.Convoluter(self.PROJECT['tile_px'], self.PROJECT['tile_um'], batch_size=None,
																					use_fp16=self.PROJECT['use_fp16'], 
																					stride_div=2,
																					save_folder=save_folder, 
																					roi_dir=self.PROJECT['roi_dir'])
		c.load_slides(slide_list)
		c.convolute_slides(export_tiles=True)

		if not skip_validation:
			self.separate_training_and_validation(dataset_label)
		self.generate_tfrecord(dataset_label)

	def separate_training_and_validation(self, dataset_label=None):
		'''Separate training and eval raw image sets. Assumes images are located in tile directory with subfolders according to slide.'''
		log.header('Separating training and validation datasets...')
		dataset_label = NO_LABEL if not dataset_label else dataset_label
		tiles_dir = join(self.PROJECT['tiles_dir'], dataset_label)
		if self.PROJECT['validation_strategy'] == 'per-tile':
			train_dir = join(tiles_dir, "train_data")
			validation_dir = join(tiles_dir, "validation_data")
			if not os.path.exists(train_dir): os.makedirs(train_dir)
			if not os.path.exists(validation_dir): os.makedirs(validation_dir)

			# Move tiles into the train_data subfolder
			tile_directories = [_dir for _dir in os.listdir(tiles_dir) if os.path.isdir(join(tiles_dir, _dir)) and _dir not in ["train_data", 'validation_data']]
			for directory in tile_directories:
				try:
					shutil.move(join(tiles_dir, directory), train_dir)
				except:
					log.error("Training and validation dataset separation has already been started, recommend cleaning tiles directory", 1)
					sys.exit()
			# Extract a fraction of tiles from train_data into validation_data
			datasets.build_validation(train_dir, validation_dir, fraction = self.PROJECT['validation_fraction'])

	def generate_tfrecord(self, dataset_label=None):
		'''Create tfrecord files from a collection of raw images'''
		# Note: this will not work as the write_tfrecords function expects a category directory
		# Will leave as is to manually test performance with category defined in the TFRecrod
		#  vs. dynamically assigning category via annotation metadata during training
		log.header('Writing TFRecord files...')
		dataset_label = NO_LABEL if not dataset_label else dataset_label
		tfrecord_dir = join(self.PROJECT['tfrecord_dir'], dataset_label)
		tiles_dir = join(self.PROJECT['tiles_dir'], dataset_label)

		if not exists(tfrecord_dir):
			os.makedirs(tfrecord_dir)

		if self.PROJECT['validation_strategy'] == 'per-tile':
			tfrecord_train_dir = join(tfrecord_dir, 'train')
			tfrecord_validation_dir = join(tfrecord_dir, 'validation')
			if not exists(tfrecord_train_dir):
				os.makedirs(tfrecord_train_dir)
			if not exists(tfrecord_validation_dir):
				os.makedirs(tfrecord_validation_dir)
			tfrecords.write_tfrecords_multi(join(tiles_dir, 'train_data'), tfrecord_train_dir, self.PROJECT['annotations'])
			tfrecords.write_tfrecords_multi(join(tiles_dir, 'validation_data'), tfrecord_validation_dir, self.PROJECT['annotations'])
			self.update_manifest(tfrecord_train_dir, skip_verification=True)
			self.update_manifest(tfrecord_validation_dir)
		else:
			tfrecords.write_tfrecords_multi(tiles_dir, tfrecord_dir, self.PROJECT['annotations'])
			self.update_manifest(tfrecord_dir)
		
		if self.PROJECT['delete_tiles']:
			shutil.rmtree(tiles_dir)
		

	def delete_tiles(self, subdir=None):
		delete_folder = self.PROJECT['tiles_dir'] if not subdir else join(self.PROJECT['tiles_dir'], subdir)
		shutil.rmtree(delete_folder)
		log.info(f"Deleted tiles in folder {sfutil.green(delete_folder)}", 1)

	def checkpoint_to_h5(self, model_name):
		tfrecords.checkpoint_to_h5(self.PROJECT['models_dir'], model_name)

	def initialize_model(self, model_name, input_dir, category_header, filter_header=None, filter_values=None):
		# Assemble list of slides for training and annotations dictionary
		slide_to_category = sfutil.get_annotations_dict(self.PROJECT['annotations'], 'slide', category_header, 
																							filter_header=filter_header, 
																							filter_values=filter_values,
																							use_encode=True)

		model_dir = join(self.PROJECT['models_dir'], model_name)
		
		validation_strategy == 'per-tile' if 'validation_strategy' not in self.PROJECT else self.PROJECT['validation_strategy']
		validation_fraction == 0.1 if 'validation_fraction' not in self.PROJECT else self.PROJECT['validation_fraction']

		SFM = sfmodel.SlideflowModel(model_dir, input_dir, self.PROJECT['tile_px'], slide_to_category,
																				validation_strategy=self.PROJECT['validation_strategy'],
																				validation_fraction=self.PROJECT['validation_fraction'],
																				manifest=self.MANIFEST, 
																				use_fp16=self.PROJECT['use_fp16'])
		return SFM

	def evaluate(self, model, dataset_label, category_header, filter_header=None, filter_values=None, checkpoint=None):
		log.header(f"Evaluating model {sfutil.bold(model)}...")
		SFM = self.initialize_model("evaluation", tfrecord_dir, category_header, filter_header, filter_values)
		model_dir = join(self.PROJECT['models_dir'], model, "trained_model.h5") if model[-3:] != ".h5" else model
		results = SFM.evaluate(subdir=dataset_label, hp=None, model=model_dir, checkpoint=checkpoint, batch_size=EVAL_BATCH_SIZE)
		print(results)

	def train(self, models=None, dataset_label='training', category_header='category', filter_header=None, filter_values=None, resume_training=None, checkpoint=None, supervised=True):
		'''Train model(s) given configurations found in batch_train.csv.
		Args:
			models			(optional) Either string representing a model name or an array of strings containing model names. 
								Will train models with these names in the batch_train.csv config file.
								Defaults to None, which will train all models in the batch_train.csv config file.
			dataset_label	(optional) Which dataset to pull tfrecord files from (subdirectory in tfrecord_dir); defaults to 'training'
			category_header	(optional) Which header in the annotation file to use for the output category. Defaults to 'category'
			filter_header	(optional) Filter slides to inculde in training by this column
			filter_values	(optional) String or array of strings. Only train on slides with these values in the filter_header column.
			resume_training	(optional) Path to .h5 model to continue training
			checkpoint		(optional) Path to cp.ckpt from which to load weights
			supervised		(optional) Whether to use verbose output and save training progress to Tensorboard
		Returns:
			A dictionary containing model names mapped to train_acc, val_loss, and val_acc'''
		tfrecord_dir = join(self.PROJECT['tfrecord_dir'], dataset_label)
		# First, quickly scan for errors (duplicate model names) and prepare models to train
		models_to_train, model_acc = [], {}
		with open(self.PROJECT['batch_train_config']) as csv_file:
			reader = csv.reader(csv_file)
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
						log.error(f"Duplicate model names found in {sfutil.green(self.PROJECT['batch_train_config'])}.", 0)
						sys.exit()
					models_to_train += [model_name]

		log.header(f"Training {len(models_to_train)} models...")

		# Next, prepare the multiprocessing manager (needed for Keras memory management)
		manager = multiprocessing.Manager()
		results_dict = manager.dict()

		# Create a worker that can execute one round of training
		def trainer (results_dict, model_name, hp, k_fold_iter=None):
			if supervised: 
				k_fold_msg = "" if not k_fold_iter else f" (k-fold iteration #{k_fold_iter}"
				log.empty(f"Training model {sfutil.bold(model_name)}{k_fold_msg}...", 1)
				log.info(hp, 1)
			model_name = model_name if not k_fold_iter else model_name+f"-kfold{k_fold_iter}"
			SFM = self.initialize_model(model_name, tfrecord_dir, category_header, filter_header, filter_values)
			try:
				train_acc, val_loss, val_acc = SFM.train(hp, pretrain=self.PROJECT['pretrain'], 
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

		k_fold = 0 if self.PROJECT['validation_strategy'] == 'per-tile' else self.PROJECT['validation_k_fold']
		
		# Now begin assembling models and hyperparameters from batch_train.csv file
		with open(self.PROJECT['batch_train_config']) as csv_file:
			reader = csv.reader(csv_file)
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
						arg_type = type(getattr(hp, arg))
						setattr(hp, arg, arg_type(value))
					else:
						log.error(f"Unknown argument '{arg}' found in training config file.", 0)

				# Start the model training
				if k_fold:
					train_acc_list = []
					val_loss_list = []
					val_acc_list = []
					error_encountered = False
					for k in range(k_fold):
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
			writer = csv.writer(csv_outfile, delimiter=',')
			# Create headers and first row
			header = ['model_name']
			firstrow = ['model1']
			default_hp = sfmodel.HyperParameters()
			for arg in default_hp._get_args():
				header += [arg]
				firstrow += [getattr(default_hp, arg)]
			writer.writerow(header)
			writer.writerow(firstrow)

	def create_hyperparameter_sweep(self, toplayer_epochs, finetune_epochs, model, pooling, loss, learning_rate, batch_size, hidden_layers,
									optimizer, early_stop, early_stop_patience, balanced_training, balanced_validation, augment, filename=None):
		'''Prepares a hyperparameter sweep using the batch train config file.'''
		log.header("Preparing hyperparameter sweep...")
		# Assemble all possible combinations of provided hyperparameters
		pdict = locals()
		del(pdict['self'])
		del(pdict['filename'])
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
			writer = csv.writer(csv_outfile, delimiter=',')
			# Create headers
			header = ['model_name']
			default_hp = sfmodel.HyperParameters()
			for arg in default_hp._get_args():
				header += [arg]
			writer.writerow(header)
			# Iterate through sweep
			for i, params in enumerate(sweep):
				row = [f'HPSweep{i}']
				hp = sfmodel.HyperParameters(*params)
				for arg in hp._get_args():
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
		if not sfutil.yes_no_input("Has an annotations (CSV) file already been created? [Y/n] ", default='yes'):
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
		if sfutil.yes_no_input("Will models utilize pre-training? [Y/n] ", default='yes'):
			if sfutil.yes_no_input("Use Imagenet pre-training? [Y/n] ", default='yes'):
				project['pretrain'] = 'imagenet'
			else:
				project['pretrain'] = sfutil.dir_input("Where is the pretrained model folder located? ", create_on_invalid=False)
		project['tile_um'] = sfutil.int_input("What is the tile width in microns? [280] ", default=280)
		project['tile_px'] = sfutil.int_input("What is the tile width in pixels? [224] ", default=224)
		project['use_fp16'] = sfutil.yes_no_input("Should FP16 be used instead of FP32? (recommended) [Y/n] ", default='yes')
		project['batch_train_config'] = sfutil.file_input("Location for the batch training CSV config file? [./batch_train.csv] ",
													default='./batch_train.csv', filetype='csv', verify=False)
		
		if not exists(project['batch_train_config']):
			print("Batch training file not found, creating blank")
			self.create_blank_train_config(project['batch_train_config'])
		
		# Validation strategy
		project['validation_strategy'] = sfutil.choice_input("Which validation strategy should be used, per-tile or per-slide? [per-slide] ", valid_choices=['per-tile', 'per-slide'], default='per-slide')
		if project['validation_strategy'] == 'per-slide':

			project['validation_k_fold'] = sfutil.int_input("If using K-fold cross-validation, what is K? (default, 1, is no cross-validation) [1] ", default=1)
		else:
			project['validation_k_fold'] = 1
		project['validation_fraction'] = sfutil.float_input("What fraction of training data should be used for validation testing? [0.2] ", valid_range=[0,1], default=0.2)

		sfutil.write_json(project, join(sfutil.PROJECT_DIR, 'settings.json'))
		self.PROJECT = project
		print("\nProject configuration saved.\n")
		sys.exit()
