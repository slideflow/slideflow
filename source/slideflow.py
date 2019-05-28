import argparse
import os
import json
import tensorflow as tf

from os.path import join, isfile, exists
from pathlib import Path
from glob import glob

import convoluter
import  data_utils as util

class SlideFlowProject:
	PROJECT_DIR = ""
	NAME = None
	ANNOTATIONS_FILE = None
	SLIDES_DIR = None
	ROI_DIR = None
	TILES_DIR = None
	MODELS_DIR = None
	TILE_UM = None
	TILE_PX = None
	NUM_CLASSES = None

	EVAL_FRACTION = 0.1
	AUGMENTATION = convoluter.STRICT_AUGMENTATION	
	NUM_THREADS = 8
	USE_FP16 = True

	def __init__(self, project_folder):
		os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
		tf.logging.set_verbosity(tf.logging.ERROR)
		print('''SlideFlow v1.0\n==============\n''')
		print('''Loading project...''')
		if project_folder and not os.path.exists(project_folder):
			if self.yes_no_input(f'Directory "{project_folder}" does not exist. Create directory and set as project root? [Y/n] ', default=True):
				os.mkdir(project_folder)
			else:
				project_folder = self.dir_input("Where is the project root directory? ", create_on_invalid=True)
		if not project_folder:
			project_folder = self.dir_input("Where is the project root directory? ", create_on_invalid=True)
		self.PROJECT_DIR = project_folder

		project_json = os.path.join(project_folder, "settings.json")
		if os.path.exists(project_json):
			self.load_project(project_json)
		else:
			self.create_project(project_json)

	def extract_tiles(self):
		convoluter.NUM_THREADS = self.NUM_THREADS
		if not exists(join(self.TILES_DIR, "train_data")):
			util.make_dir(join(self.TILES_DIR, "train_data"))
		if not exists(join(self.TILES_DIR, "eval_data")):
			util.make_dir(join(self.TILES_DIR, "eval_data"))

		c = convoluter.Convoluter(self.TILE_PX, self.TILE_UM, self.NUM_CLASSES, self.BATCH_SIZE, 
									self.USE_FP16, join(self.TILES_DIR, "train_data"), self.ROI_DIR, self.AUGMENTATION)

		slide_list = glob(join(self.SLIDES_DIR, '**/*.svs'))
		slide_list.extend(glob(join(self.SLIDES_DIR, '**/*.jpg')))
		c.load_slides(slide_list)
		c.convolute_slides(export_tiles=True)
	
	def separate_training_and_eval_data(self):
		util.build_validation(join(self.TILES_DIR, "train_data"), join(self.TILES_DIR, "eval_data"), fraction = self.EVAL_FRACTION)
		pass

	def create_global_path(self, path_string):
		if path_string and (len(path_string) > 2) and path_string[:2] == "./":
			return os.path.join(self.PROJECT_DIR, path_string[2:])
		elif path_string and (path_string[0] != "/"):
			return os.path.join(self.PROJECT_DIR, path_string)
		else:
			return path_string

	def yes_no_input(self, prompt, default=None):
		yes = ['yes','y']
		no = ['no', 'n']
		while True:
			response = input(f"{prompt}")
			if not response and default:
				return default
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

	def load_project(self, project_json):
		if os.path.exists(project_json):
			with open(project_json, 'r') as json_data_file:
				data = json.load(json_data_file)
			self.NAME = data['name']
			self.PROJECT_DIR = data['root']
			self.ANNOTATIONS_FILE = data['annotations']
			self.SLIDES_DIR = data['slides']
			self.ROI_DIR = data['ROI'] 
			self.TILES_DIR = data['tiles']
			self.MODELS_DIR = data['models']
			self.TILE_UM = data['tile_um']
			self.TILE_PX = data['tile_px']
			self.NUM_CLASSES = data['num_classes']
			self.BATCH_SIZE = data['batch_size']
			self.USE_FP16 = data['use_fp16']
			print("\nProject configuration loaded.\n")
		else:
			raise OSError(f'Unable to locate project json at location "{project_json}".')

	def create_project(self, project_json):
		self.NAME = input("What is the project name? ")
		self.ANNOTATIONS_FILE = self.file_input("Where is the project annotations (CSV) file located? [./annotations.csv] ", 
									default='./annotations.csv', filetype="csv")
		self.SLIDES_DIR = self.dir_input("Where are the SVS slides stored? [./slides] ",
									default='./slides', create_on_invalid=True)
		self.ROI_DIR = self.dir_input("Where are the ROI files (CSV) stored? [./slides] ",
									default='./slides', create_on_invalid=True)									
		self.TILES_DIR = self.dir_input("Where will the tessellated image tiles be stored? [./tiles] ",
									default='./tiles', create_on_invalid=True)
		self.MODELS_DIR = self.dir_input("Where are the saved models stored? [./models] ",
									default='./models', create_on_invalid=True)
		self.TILE_UM = self.int_input("What is the tile width in microns? [280] ", default=280)
		self.TILE_PX = self.int_input("What is the tile width in pixels? [512] ", default=512)
		self.NUM_CLASSES = self.int_input("How many classes are there to be trained? ")
		self.BATCH_SIZE = self.int_input("What batch size should be used? [64] ", default=64)
		self.USE_FP16 = self.yes_no_input("Should FP16 be used instead of FP32? (recommended) [Y/n] ", default=True)

		data = {}
		data['name'] = self.NAME
		data['root'] = self.PROJECT_DIR
		data['annotations'] = self.ANNOTATIONS_FILE
		data['slides'] = self.SLIDES_DIR
		data['ROI'] = self.ROI_DIR
		data['tiles'] = self.TILES_DIR
		data['models'] = self.MODELS_DIR
		data['tile_um'] = self.TILE_UM
		data['tile_px'] = self.TILE_PX
		data['num_classes'] = self.NUM_CLASSES
		data['batch_size'] = self.BATCH_SIZE
		data['use_fp16'] = self.USE_FP16

		with open(join(self.PROJECT_DIR, "settings.json"), "w") as json_data_file:
			json.dump(data, json_data_file)
		print("\nProject configuration saved.\n")