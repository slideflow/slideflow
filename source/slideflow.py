import argparse
import os
import sys
import json
import shutil
import tensorflow as tf

from os.path import join, isfile, exists
from pathlib import Path
from glob import glob

import convoluter
import sfmodel
from  util import datasets, tfrecords, sfutil

class SlideFlowProject:
	PROJECT_DIR = ""
	NAME = None
	ANNOTATIONS_FILE = None
	SLIDES_DIR = None
	ROI_DIR = None
	TILES_DIR = None
	MODELS_DIR = None
	PRETRAIN_DIR = None
	USE_TFRECORD = False
	TFRECORD_DIR = None
	DELETE_TILES = False
	TILE_UM = None
	TILE_PX = None
	NUM_CLASSES = None

	EVAL_FRACTION = 0.1
	AUGMENTATION = convoluter.STRICT_AUGMENTATION
	NUM_THREADS = 6
	USE_FP16 = True

	def __init__(self, project_folder):
		os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
		tf.logging.set_verbosity(tf.logging.ERROR)
		print('''SlideFlow v1.0\n==============\n''')
		print('''Loading project...''')
		if project_folder and not os.path.exists(project_folder):
			if self.yes_no_input(f'Directory "{project_folder}" does not exist. Create directory and set as project root? [Y/n] ', default='yes'):
				os.mkdir(project_folder)
			else:
				project_folder = self.dir_input("Where is the project root directory? ", create_on_invalid=True)
		if not project_folder:
			project_folder = self.dir_input("Where is the project root directory? ", create_on_invalid=True)
		self.PROJECT_DIR = project_folder

		self.CONFIG = os.path.join(project_folder, "settings.json")
		if os.path.exists(self.CONFIG):
			self.load_project()
		else:
			self.create_project()

	def reset_tasks(self):
		print("Resetting task progress.")
		data = self.parse_config(self.CONFIG)
		for task in data['tasks'].keys():
			data['tasks'][task] = 'not started'
		self.write_config(data, self.CONFIG)

	def prepare_tiles(self):
		self.extract_tiles()
		self.separate_training_and_eval()
		if self.USE_TFRECORD:
			self.generate_tfrecord()

	def extract_tiles(self, ignore=None):
		if self.get_task('extract_tiles') == 'complete':
			print('Tile extraction already complete.')
			return
		elif self.get_task('extract_tiles') == 'in process':
			if not self.yes_no_input('Tile extraction already in process; restart? [y/N] ', default='no'):
				sys.exit()
		print("Extracting image tiles...")
		self.update_task('extract_tiles', 'in process')
		convoluter.NUM_THREADS = self.NUM_THREADS
		if not exists(join(self.TILES_DIR, "train_data")):
			datasets.make_dir(join(self.TILES_DIR, "train_data"))
		if not exists(join(self.TILES_DIR, "eval_data")):
			datasets.make_dir(join(self.TILES_DIR, "eval_data"))

		num_dir = len(self.SLIDES_DIR.split('/'))
		slide_list = [i for i in glob(join(self.SLIDES_DIR, '**/*.svs'))
					  if i.split('/')[num_dir] != 'thumbs']

		slide_list.extend( [i for i in glob(join(self.SLIDES_DIR, '**/*.jpg'))
					  		if i.split('/')[num_dir] != 'thumbs'] )

		slide_list.extend(glob(join(self.SLIDES_DIR, '*.svs')))
		slide_list.extend(glob(join(self.SLIDES_DIR, '*.jpg')))
		
		if ignore:
			ignore_counter = 0
			for i, slide in enumerate(slide_list):
				slide_name = slide.split('/')[-1][:-4]
				if slide_name in ignore:
					del slide_list[i]
					ignore_counter += 1
			print(f" + [{sfutil.info('INFO')}] Extracting tiles from {len(slide_list)} slides, ignored {ignore_counter}")

		c = convoluter.Convoluter(self.TILE_PX, self.TILE_UM, self.NUM_CLASSES, self.BATCH_SIZE, 
									self.USE_FP16, join(self.TILES_DIR, "train_data"), self.ROI_DIR, self.AUGMENTATION)
		c.load_slides(slide_list)
		c.convolute_slides(export_tiles=True)
		self.update_task('extract_tiles', 'complete')
	
	def separate_training_and_eval(self):
		if self.get_task('separate_training_and_eval') == 'complete':
			print('Training and eval dataset separation already complete.')
			return
		print('Separating training and eval datasets...')
		self.update_task('separate_training_and_eval', 'in process')
		datasets.build_validation(join(self.TILES_DIR, "train_data"), join(self.TILES_DIR, "eval_data"), fraction = self.EVAL_FRACTION)
		self.update_task('separate_training_and_eval', 'complete')

	def generate_tfrecord(self):
		# Note: this will not work as the write_tfrecords function expects a category directory
		# Will leave as is to manually test performance with category defined in the TFRecrod
		#  vs. dynamically assigning category via annotation metadata during training
		if self.get_task('generate_tfrecord') == 'complete':
			print('TFRecords already generated.')
			return
		print('Writing TFRecord files...')
		self.update_task('generate_tfrecord', 'in process')
		tfrecords.write_tfrecords(join(self.TILES_DIR, "train_data"), self.TFRECORD_DIR, "train", self.ANNOTATIONS_FILE)
		tfrecords.write_tfrecords(join(self.TILES_DIR, "eval_data"), self.TFRECORD_DIR, "eval", self.ANNOTATIONS_FILE)
		if self.DELETE_TILES:
			shutil.rmtree(join(self.TILES_DIR, "train_data"))
			shutil.rmtree(join(self.TILES_DIR, "eval_data"))
		self.update_task('generate_tfrecord', 'complete')

	def start_training(self, model_name):
		self.update_task('training', 'in process')
		print(f"Training model {model_name}...")
		model_dir = join(self.MODELS_DIR, model_name)
		tensorboard_dir = join(model_dir, 'logs/projector')
		if not exists(model_dir):
			datasets.make_dir(model_dir)
		if not exists(tensorboard_dir):
			datasets.make_dir(tensorboard_dir)
		input_dir = self.TFRECORD_DIR if self.USE_TFRECORD else self.TILES_DIR
		SFM = sfmodel.SlideflowModel(model_dir, input_dir, self.ANNOTATIONS_FILE)
		SFM.train(restore_checkpoint = self.PRETRAIN_DIR)
		os.system(f'tensorboard --logdir={tensorboard_dir}')

	def create_global_path(self, path_string):
		if path_string and (len(path_string) > 2) and path_string[:2] == "./":
			return os.path.join(self.PROJECT_DIR, path_string[2:])
		elif path_string and (path_string[0] != "/"):
			return os.path.join(self.PROJECT_DIR, path_string)
		else:
			return path_string

	def yes_no_input(self, prompt, default='no'):
		yes = ['yes','y']
		no = ['no', 'n']
		while True:
			response = input(f"{prompt}")
			if not response and default:
				return True if default in yes else False
			if response.lower() in yes:
				return True
			if response.lower() in no:
				return False
			print(f"Invalid response.")

	def dir_input(self, prompt, default=None, create_on_invalid=False):
		while True:
			response = self.create_global_path(input(f"{prompt}"))
			if not response and default:
				response = self.create_global_path(default)
			if not os.path.exists(response) and create_on_invalid:
				if self.yes_no_input(f'Directory "{response}" does not exist. Create directory? [Y/n] ', default=True):
					os.mkdir(response)
					return response
				else:
					continue
			elif not os.path.exists(response):
				print(f'Unable to locate directory "{response}"')
				continue
			return response

	def file_input(self, prompt, default=None, filetype=None):
		while True:
			response = self.create_global_path(input(f"{prompt}"))
			if not response and default:
				response = self.create_global_path(default)
			if not os.path.exists(response):
				print(f'Unable to locate file "{response}"')
				continue
			extension = response.split('.')[-1]
			if filetype and (extension != filetype):
				print(f'Incorrect filetype; provided file of type "{extension}", need type "{filetype}"')
				continue
			return response

	def int_input(self, prompt, default=None):
		while True:
			response = input(f"{prompt}")
			if not response and default:
				return default
			try:
				int_response = int(response)
			except ValueError:
				print("Please supply a valid number.")
				continue
			return int_response
	
	def parse_config(self, config_file):
		with open(config_file, 'r') as data_file:
			return json.load(data_file)

	def write_config(self, data, config_file):
		with open(config_file, "w") as data_file:
			json.dump(data, data_file)
		
	def update_task(self, task, status):
		data = self.parse_config(self.CONFIG)
		data['tasks'][task] = status
		self.write_config(data, self.CONFIG)

	def get_task(self, task):
		return self.parse_config(self.CONFIG)['tasks'][task]

	def load_project(self):
		if os.path.exists(self.CONFIG):
			data = self.parse_config(self.CONFIG)
			self.NAME = data['name']
			self.PROJECT_DIR = data['root']
			self.ANNOTATIONS_FILE = data['annotations']
			self.SLIDES_DIR = data['slides']
			self.ROI_DIR = data['ROI'] 
			self.TILES_DIR = data['tiles']
			self.MODELS_DIR = data['models']
			self.PRETRAIN_DIR = data['pretraining']
			self.TILE_UM = data['tile_um']
			self.TILE_PX = data['tile_px']
			self.NUM_CLASSES = data['num_classes']
			self.BATCH_SIZE = data['batch_size']
			self.USE_FP16 = data['use_fp16']
			self.USE_TFRECORD = data['use_tfrecord']
			self.TFRECORD_DIR = data['tfrecord_dir']
			self.DELETE_TILES = data['delete_tiles']
			print("\nProject configuration loaded.\n")
		else:
			raise OSError(f'Unable to locate project json at location "{self.CONFIG}".')

	def create_project(self):
		# General setup and slide configuration
		self.NAME = input("What is the project name? ")
		self.ANNOTATIONS_FILE = self.file_input("Where is the project annotations (CSV) file located? [./annotations.csv] ", 
									default='./annotations.csv', filetype="csv")
		self.SLIDES_DIR = self.dir_input("Where are the SVS slides stored? [./slides] ",
									default='./slides', create_on_invalid=True)
		self.ROI_DIR = self.dir_input("Where are the ROI files (CSV) stored? [./slides] ",
									default='./slides', create_on_invalid=True)

		# Slide tessellation
		self.TILES_DIR = self.dir_input("Where will the tessellated image tiles be stored? (recommend SSD) [./tiles] ",
									default='./tiles', create_on_invalid=True)
		self.USE_TFRECORD = self.yes_no_input("Store tiles in TFRecord format? [Y/n] ", default='yes')
		if self.USE_TFRECORD:
			self.DELETE_TILES = self.yes_no_input("Should raw tile images be deleted after TFRecord storage? [Y/n] ", default='yes')
			self.TFRECORD_DIR = self.dir_input("Where should the TFRecord files be stored? (recommend HDD) [./tfrecord] ",
									default='./tfrecord', create_on_invalid=True)
		# Training
		self.MODELS_DIR = self.dir_input("Where should the saved models be stored? [./models] ",
									default='./models', create_on_invalid=True)
		if self.yes_no_input("Will models utilize pre-training? [y/N] ", default='no'):
			self.PRETRAIN_DIR = self.dir_input("Where is the pretrained model folder located? ", create_on_invalid=False)
		self.TILE_UM = self.int_input("What is the tile width in microns? [280] ", default=280)
		self.TILE_PX = self.int_input("What is the tile width in pixels? [512] ", default=512)
		self.NUM_CLASSES = self.int_input("How many classes are there to be trained? ")
		self.BATCH_SIZE = self.int_input("What batch size should be used? [64] ", default=64)
		self.USE_FP16 = self.yes_no_input("Should FP16 be used instead of FP32? (recommended) [Y/n] ", default='yes')

		data = {}
		data['name'] = self.NAME
		data['root'] = self.PROJECT_DIR
		data['annotations'] = self.ANNOTATIONS_FILE
		data['slides'] = self.SLIDES_DIR
		data['ROI'] = self.ROI_DIR
		data['tiles'] = self.TILES_DIR
		data['models'] = self.MODELS_DIR
		data['pretraining'] = self.PRETRAIN_DIR
		data['tile_um'] = self.TILE_UM
		data['tile_px'] = self.TILE_PX
		data['num_classes'] = self.NUM_CLASSES
		data['batch_size'] = self.BATCH_SIZE
		data['use_fp16'] = self.USE_FP16
		data['use_tfrecord'] = self.USE_TFRECORD
		data['tfrecord_dir'] = self.TFRECORD_DIR
		data['delete_tiles'] = self.DELETE_TILES
		data['tasks'] = {
			'extract_tiles': 'not started',
			'separate_training_and_eval': 'not started',
			'generate_tfrecord': 'not started',
			'training': 'not started',
			'analytics': 'not started',
			'heatmaps': 'not started',
			'mosaic': 'not started'
		}
		self.write_config(data, self.CONFIG)
		print("\nProject configuration saved.\n")