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

import subprocess
import convoluter
import sfmodel
import itertools
from util import datasets, tfrecords, sfutil
from util.sfutil import TCGAAnnotations

# TODO: scan for duplicate SVS files (especially when converting TCGA long-names 
# 	to short-names, e.g. when multiple diagnostic slides are present)
# TODO: automatic loading of training configuration even when training one model

SKIP_VERIFICATION = False

class SlideFlowProject:
	MANIFEST = None
	NUM_THREADS = 4
	USE_FP16 = True

	def __init__(self, project_folder):
		os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
		print('''\nSlideFlow v0.9.2\n==============\n''')
		print('''Loading project...''')
		if project_folder and not os.path.exists(project_folder):
			if sfutil.yes_no_input(f'Directory "{project_folder}" does not exist. Create directory and set as project root? [Y/n] ', default='yes'):
				os.mkdir(project_folder)
			else:
				project_folder = sfutil.dir_input("Where is the project root directory? ", create_on_invalid=True)
		if not project_folder:
			project_folder = sfutil.dir_input("Where is the project root directory? ", create_on_invalid=True)
		sfutil.PROJECT_DIR = project_folder

		if exists(join(project_folder, "settings.json")):
			self.load_project(project_folder)
		else:
			self.create_project(project_folder)
		
	def extract_tiles(self, filter_header=None, filter_values=None):
		'''filter is a dict whose keys correspond with a header label and whose value is a 
		list of acceptable values; all other cases will be ignored
		
		If a single case is supplied, extract tiles for just that case.'''

		print("Extracting image tiles...")
		convoluter.NUM_THREADS = self.NUM_THREADS
		if not exists(join(self.PROJECT['tiles_dir'], "train_data")):
			datasets.make_dir(join(self.PROJECT['tiles_dir'], "train_data"))
		if not exists(join(self.PROJECT['tiles_dir'], "eval_data")):
			datasets.make_dir(join(self.PROJECT['tiles_dir'], "eval_data"))

		slide_list = sfutil.get_filtered_slide_paths(self.PROJECT['slides_dir'], self.PROJECT['annotations'], filter_header=filter_header,
																				  							  filter_values=filter_values)

		print(f" + [{sfutil.info('INFO')}] Extracting tiles from {len(slide_list)} slides")

		c = convoluter.Convoluter(self.PROJECT['tile_px'], self.PROJECT['tile_um'], batch_size=None,
																					use_fp16=self.PROJECT['use_fp16'], 
																					stride_div=2,
																					save_folder=join(self.PROJECT['tiles_dir'], "train_data"), 
																					roi_dir=self.PROJECT['roi_dir'])
		c.load_slides(slide_list)
		c.convolute_slides(export_tiles=True)

	def delete_tiles(self, case_list):
		if type(case_list) == str:
			case_list = [case_list]
		if not sfutil.yes_no_input(print(f"[{sfutil.warn('WARN')}] Delete tiles from {len(case_list)} cases? [y/N] "), default='no'):
			return
		if self.PROJECT['use_tfrecord']:
			print(f"[{sfutil.warn('WARN')}] Unable to delete tiles in TFRecord files.")
			return
		slides_with_tiles = os.listdir(join(self.PROJECT['tiles_dir'], "train_data"))
		slides_with_tiles.extend(os.listdir(join(self.PROJECT['tiles_dir'], "eval_data")))
		case_slide_dict = sfutil.get_annotations_dict(self.PROJECT['annotations'], TCGAAnnotations.case, TCGAAnnotations.slide, use_encode=False)	
		for case in case_list:
			try:
				if case in slides_with_tiles:
					shutil.rmtree(join(self.PROJECT['tiles_dir'], "train_data", case))
					shutil.rmtree(join(self.PROJECT['tiles_dir'], "eval_data", case))
				elif (case in case_slide_dict) and (case_slide_dict[case] in slides_with_tiles):
					shutil.rmtree(join(self.PROJECT['tiles_dir'], "train_data", case_slide_dict[case]))
					shutil.rmtree(join(self.PROJECT['tiles_dir'], "eval_data", case_slide_dict[case]))
				else:
					print(f"[{sfutil.fail('ERROR')}] Unable to delete tiles for case '{case}'.")
			except:
				print(f"[{sfutil.fail('ERROR')}] Unable to delete tiles for case '{case}'.")
					
	def separate_training_and_eval(self, fraction=0.1):
		'''Separate training and eval raw image sets. Assumes images are located in "train_data" directory.'''
		print('Separating training and eval datasets...')
		datasets.build_validation(join(self.PROJECT['tiles_dir'], "train_data"), join(self.PROJECT['tiles_dir'], "eval_data"), fraction = fraction)

	def generate_tfrecord(self):
		'''Create tfrecord files from a collection of raw images'''
		# Note: this will not work as the write_tfrecords function expects a category directory
		# Will leave as is to manually test performance with category defined in the TFRecrod
		#  vs. dynamically assigning category via annotation metadata during training
		tfrecord_train_dir = join(self.PROJECT['tfrecord_dir'], 'train')
		tfrecord_eval_dir = join(self.PROJECT['tfrecord_dir'], 'eval')
		if not exists(self.PROJECT['tfrecord_dir']):
			datasets.make_dir(self.PROJECT['tfrecord_dir'])
		if not exists(tfrecord_train_dir):
			os.mkdir(tfrecord_train_dir)
		if not exists(tfrecord_eval_dir):
			os.mkdir(tfrecord_eval_dir)

		print('Writing TFRecord files...')		
		tfrecords.write_tfrecords_multi(join(self.PROJECT['tiles_dir'], 'train_data'), tfrecord_train_dir, self.PROJECT['annotations'])
		tfrecords.write_tfrecords_multi(join(self.PROJECT['tiles_dir'], 'eval_data'), tfrecord_eval_dir, self.PROJECT['annotations'])
		if self.PROJECT['delete_tiles']:
			shutil.rmtree(join(self.PROJECT['tiles_dir'], "train_data"))
			shutil.rmtree(join(self.PROJECT['tiles_dir'], "eval_data"))
		self.generate_manifest()

	def checkpoint_to_h5(self, model_name):
		tfrecords.checkpoint_to_h5(self.PROJECT['models_dir'], model_name)

	def initialize_model(self, model_name, category_header, filter_header=None, filter_values=None):
		# Assemble list of slides for training and annotations dictionary
		slide_to_category = sfutil.get_annotations_dict(self.PROJECT['annotations'], 'slide', category_header, 
																							filter_header=filter_header, 
																							filter_values=filter_values,
																							use_encode=True)

		model_dir = join(self.PROJECT['models_dir'], model_name)
		input_dir = self.PROJECT['tfrecord_dir'] #if self.PROJECT['use_tfrecord'] else self.PROJECT['tiles_dir']

		SFM = sfmodel.SlideflowModel(model_dir, input_dir, self.PROJECT['tile_px'], slide_to_category,
																				manifest=self.MANIFEST, 
																				use_fp16=self.PROJECT['use_fp16'])
		return SFM

	def evaluate(self, model, category_header, filter_header=None, filter_values=None, checkpoint=None):
		SFM = self.initialize_model("evaluation", category_header, filter_header, filter_values)
		results = SFM.evaluate(model, checkpoint)
		print(results)

	def train(self, models=None, category_header='category', filter_header=None, filter_values=None, resume_training=None, checkpoint=None, supervised=True):
		'''Train model(s) given configurations found in batch_train.csv.
		Args:
			models			(optional) Either string representing a model name or an array of strings containing model names. 
								Will train models with these names in the batch_train.csv config file.
								Defaults to None, which will train all models in the batch_train.csv config file.
			category_header	(optional) Which header in the annotation file to use for the output category. Defaults to 'category'
			filter_header	(optional) Filter slides to inculde in training by this column
			filter_values	(optional) String or array of strings. Only train on slides with these values in the filter_header column.
			resume_training	(optional) Path to .h5 model to continue training
			checkpoint		(optional) Path to cp.ckpt from which to load weights
			supervised		(optional) Whether to use verbose output and save training progress to Tensorboard
		Returns:
			A dictionary containing model names mapped to train_acc, val_loss, and val_acc'''
		# First, quickly scan for errors (duplicate model names) and prepare models to train
		models_to_train, model_acc = [], {}
		with open(self.PROJECT['batch_train_config']) as csv_file:
			reader = csv.reader(csv_file)
			header = next(reader)
			try:
				model_name_i = header.index('model_name')
			except:
				print(f"[{sfutil.fail('ERROR')}] Unable to find column 'model_name' in the batch training config file.")
				sys.exit() 
			for row in reader:
				model_name = row[model_name_i]
				# First check if this row is a valid model
				if (not models) or (type(models)==str and model_name==models) or model_name in models:
					# Now verify there are no duplicate model names
					if model_name in models_to_train:
						print(f"[{sfutil.fail('ERROR')}] Duplicate model names found in {sfutil.green(self.PROJECT['batch_train_config'])}.")
						sys.exit()
					models_to_train += [model_name]

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
						print(f"[{sfutil.fail('ERROR')}] Unknown argument '{arg}' found in training config file.")

				print(f"Training model {sfutil.bold(model_name)}...")
				print(hp)
				SFM = self.initialize_model(model_name, category_header, filter_header, filter_values)
				train_acc, val_loss, val_acc = SFM.train(hp, pretrain=self.PROJECT['pretrain'], 
															resume_training=resume_training, 
															checkpoint=checkpoint,
															supervised=supervised)
			
				model_acc.update({model_name: {'train_acc': max(train_acc),
											'val_loss': val_loss,
											'val_acc': val_acc }
				})
				with open(os.path.join(self.PROJECT['root'], "results_log.csv"), "a") as results_file:
					writer = csv.writer(results_file)
					writer.writerow([model_name, max(train_acc), val_loss, val_acc])
				tf.keras.backend.clear_session()
				print(f"\n[{sfutil.header('Complete')}] Training complete for model {model_name}, max validation accuracy {sfutil.info(str(val_acc))}\n")
		print(f"\n[{sfutil.header('Complete')}] Training complete; validation accuracies:")
		for model in model_acc:
			print(f" - {sfutil.green(model)}: Train_Acc={str(model_acc[model]['train_acc'])}, " +
				f"Val_loss={model_acc[model]['val_loss']}, Val_Acc={model_acc[model]['val_acc']}" )

	def generate_heatmaps(self, model_name, filter_header=None, filter_values=None):
		slide_list = sfutil.get_filtered_slide_paths(self.PROJECT['slides_dir'], self.PROJECT['annotations'], filter_header=filter_header,
																				  							  filter_values=filter_values)
		heatmaps_folder = os.path.join(self.PROJECT['root'], 'heatmaps')
		if not os.path.exists(heatmaps_folder): os.makedirs(heatmaps_folder)

		c = convoluter.Convoluter(self.PROJECT['tile_px'], self.PROJECT['tile_um'], batch_size=64,
																					use_fp16=self.PROJECT['use_fp16'],
																					stride_div=2,
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

	def create_blank_annotations_file(self, scan_for_cases=False):
		case_header_name = TCGAAnnotations.case

		with open(self.PROJECT['annotations'], 'w') as csv_outfile:
			csv_writer = csv.writer(csv_outfile, delimiter=',')
			header = [case_header_name, 'dataset', 'category']
			csv_writer.writerow(header)

		if scan_for_cases:
			sfutil.verify_annotations(self.PROJECT['annotations'], slides_dir=self.PROJECT['slides_dir'])

	def generate_manifest(self):
		input_dir = self.PROJECT['tfrecord_dir'] #if self.PROJECT['use_tfrecord'] else self.PROJECT['tiles_dir']
		annotations = sfutil.get_annotations_dict(self.PROJECT['annotations'], key_name="slide", value_name="category")
		tfrecord_files = glob(os.path.join(input_dir, "*/*.tfrecords")) #if self.PROJECT['use_tfrecord'] else []
		self.MANIFEST = sfutil.verify_tiles(annotations, input_dir, tfrecord_files)
		sfutil.write_json(self.MANIFEST, sfutil.global_path("manifest.json"))

	def load_project(self, directory):
		if exists(join(directory, "settings.json")):
			self.PROJECT = sfutil.load_json(join(directory, "settings.json"))
			print("\nProject configuration loaded.\n")
		else:
			raise OSError(f'Unable to locate settings.json at location "{directory}".')
		if not SKIP_VERIFICATION:
			sfutil.verify_annotations(self.PROJECT['annotations'], slides_dir=self.PROJECT['slides_dir'])
		if os.path.exists(sfutil.global_path("manifest.json")):
			self.MANIFEST = sfutil.load_json(sfutil.global_path("manifest.json"))
		else:
			self.generate_manifest()

	def create_project(self, directory):
		# General setup and slide configuration
		project = {}
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
				self.create_blank_annotations_file(scan_for_cases=sfutil.yes_no_input("Scan slide folder for case names? [Y/n] ", default='yes'))
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
		project['tile_px'] = sfutil.int_input("What is the tile width in pixels? [512] ", default=512)
		project['use_fp16'] = sfutil.yes_no_input("Should FP16 be used instead of FP32? (recommended) [Y/n] ", default='yes')
		project['batch_train_config'] = sfutil.file_input("Location for the batch training CSV config file? [./batch_train.csv] ",
													default='./batch_train.csv', filetype='csv', verify=False)
		
		if not exists(project['batch_train_config']):
			print("Batch training file not found, creating blank")
			self.create_blank_train_config(project['batch_train_config'])

		sfutil.write_json(project, join(directory, 'settings.json'))
		self.PROJECT = project
		print("\nProject configuration saved.\n")
		sys.exit()
