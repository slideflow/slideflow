import argparse
import os
import sys
import shutil
import tensorflow as tf

from os.path import join, isfile, exists
from pathlib import Path
from glob import glob
import csv

import subprocess
import convoluter
import sfmodel
from util import datasets, tfrecords, sfutil
from util.sfutil import TCGAAnnotations

# TODO: scan for duplicate SVS files (especially when converting TCGA long-names 
# 	to short-names, e.g. when multiple diagnostic slides are present)

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
	AUGMENTATION = convoluter.NO_AUGMENTATION
	NUM_THREADS = 4
	USE_FP16 = True

	def __init__(self, project_folder):
		os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
		tf.logging.set_verbosity(tf.logging.ERROR)
		print('''SlideFlow v1.0\n==============\n''')
		print('''Loading project...''')
		if project_folder and not os.path.exists(project_folder):
			if sfutil.yes_no_input(f'Directory "{project_folder}" does not exist. Create directory and set as project root? [Y/n] ', default='yes'):
				os.mkdir(project_folder)
			else:
				project_folder = sfutil.dir_input("Where is the project root directory? ", create_on_invalid=True)
		if not project_folder:
			project_folder = sfutil.dir_input("Where is the project root directory? ", create_on_invalid=True)
		self.PROJECT_DIR = project_folder
		sfutil.PROJECT_DIR = project_folder

		self.CONFIG = os.path.join(project_folder, "settings.json")
		if os.path.exists(self.CONFIG):
			self.load_project()
		else:
			self.create_project()

	def reset_tasks(self):
		print("Resetting task progress.")
		data = sfutil.parse_config(self.CONFIG)
		for task in data['tasks'].keys():
			data['tasks'][task] = 'not started'
		sfutil.write_config(data, self.CONFIG)

	def prepare_tiles(self):
		self.extract_tiles()
		self.separate_training_and_eval()
		if self.USE_TFRECORD:
			self.generate_tfrecord()

	def extract_tiles(self, ignore=None, slide_filters=None):
		'''filter is a dict whose keys correspond with a header label and whose value is a 
		list of acceptable values; all other cases will be ignored
		
		If a single case is supplied, extract tiles for just that case.'''
		
		if not slide_filters and not ignore:
			if self.get_task('extract_tiles') == 'complete':
				print('Tile extraction already complete.')
				return
			elif self.get_task('extract_tiles') == 'in process':
				if not sfutil.yes_no_input('Tile extraction already in process; restart? [y/N] ', default='no'):
					sys.exit()
			self.update_task('extract_tiles', 'in process')

		print("Extracting image tiles...")
		convoluter.NUM_THREADS = self.NUM_THREADS
		if not exists(join(self.TILES_DIR, "train_data")):
			datasets.make_dir(join(self.TILES_DIR, "train_data"))
		if not exists(join(self.TILES_DIR, "eval_data")):
			datasets.make_dir(join(self.TILES_DIR, "eval_data"))

		slide_list = sfutil.get_slide_paths(self.SLIDES_DIR)
		slide_case_dict = sfutil.get_annotations_dict(self.ANNOTATIONS_FILE, TCGAAnnotations.slide, TCGAAnnotations.case, use_encode=False)
		# Remove slides not in the annotation file
		to_remove = []
		for slide in slide_list:
			slide_name = slide.split('/')[-1][:-4]
			if slide_name not in slide_case_dict:
				print(f" + [{sfutil.warn('WARN')}] Slide {sfutil._shortname(slide_name)} not in annotation file, skipping")
				to_remove.extend([slide])
		for item in to_remove:
			slide_list.remove(item)
		to_remove = []

		if slide_filters:
			for key in slide_filters.keys():
				filter_dict = sfutil.get_annotations_dict(self.ANNOTATIONS_FILE, 'slide', key, use_encode=False)
				for slide in slide_list:
					slide_name = slide.split('/')[-1][:-4]							
					if filter_dict[slide_name] not in slide_filters[key]:
						to_remove.extend([slide])
		
		if ignore:
			for slide in slide_list:
				slide_name = slide.split('/')[-1][:-4]
				if slide_name in ignore:
					to_remove.extend([slide])

		to_remove = list(set(to_remove))
		num_removed = len(to_remove)
		for item in to_remove:
			slide_list.remove(item)
		message_tail = f", filtered {num_removed}" if num_removed > 0 else ""
		print(f" + [{sfutil.info('INFO')}] Extracting tiles from {len(slide_list)} slides{message_tail}")

		c = convoluter.Convoluter(self.TILE_PX, self.TILE_UM, self.NUM_CLASSES, self.BATCH_SIZE, 
									self.USE_FP16, join(self.TILES_DIR, "train_data"), self.ROI_DIR, self.AUGMENTATION)
		c.load_slides(slide_list)
		c.convolute_slides(export_tiles=True)
		self.update_task('extract_tiles', 'complete')

	def delete_tiles(self, case_list):
		if type(case_list) == str:
			case_list = [case_list]
		if not sfutil.yes_no_input(print(f"[{sfutil.warn('WARN')}] Delete tiles from {len(case_list)} cases? [y/N] "), default='no'):
			return
		if self.USE_TFRECORD:
			print(f"[{sfutil.warn('WARN')}] Unable to delete tiles in TFRecord files.")
			return
		slides_with_tiles = os.listdir(join(self.TILES_DIR, "train_data"))
		slides_with_tiles.extend(os.listdir(join(self.TILES_DIR, "eval_data")))
		case_slide_dict = sfutil.get_annotations_dict(self.ANNOTATIONS_FILE, TCGAAnnotations.case, TCGAAnnotations.slide, use_encode=False)	
		for case in case_list:
			try:
				if case in slides_with_tiles:
					shutil.rmtree(join(self.TILES_DIR, "train_data", case))
					shutil.rmtree(join(self.TILES_DIR, "eval_data", case))
				elif (case in case_slide_dict) and (case_slide_dict[case] in slides_with_tiles):
					shutil.rmtree(join(self.TILES_DIR, "train_data", case_slide_dict[case]))
					shutil.rmtree(join(self.TILES_DIR, "eval_data", case_slide_dict[case]))
				else:
					print(f"[{sfutil.fail('ERROR')}] Unable to delete tiles for case '{case}'.")
			except:
				print(f"[{sfutil.fail('ERROR')}] Unable to delete tiles for case '{case}'.")
					
	def separate_training_and_eval(self):
		if self.get_task('separate_training_and_eval') == 'complete':
			print('Warning: Training and eval dataset separation already complete.')
			#return
		print('Separating training and eval datasets...')
		self.update_task('separate_training_and_eval', 'in process')
		datasets.build_validation(join(self.TILES_DIR, "train_data"), join(self.TILES_DIR, "eval_data"), fraction = self.EVAL_FRACTION)
		self.update_task('separate_training_and_eval', 'complete')

	def generate_tfrecord(self):
		# Note: this will not work as the write_tfrecords function expects a category directory
		# Will leave as is to manually test performance with category defined in the TFRecrod
		#  vs. dynamically assigning category via annotation metadata during training
		if self.get_task('generate_tfrecord') == 'complete':
			print('Warning: TFRecords already generated.')
			#return
		if not exists(self.TFRECORD_DIR):
			datasets.make_dir(self.TFRECORD_DIR)
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

		devnull = open(os.devnull, 'w')
		#tensorboard_process = subprocess.Popen(['tensorboard', f'--logdir={model_dir}'], stdout=devnull)

		input_dir = self.TFRECORD_DIR if self.USE_TFRECORD else self.TILES_DIR
		SFM = sfmodel.SlideflowModel(model_dir, input_dir, self.ANNOTATIONS_FILE, self.TILE_PX, self.NUM_CLASSES, self.BATCH_SIZE, self.USE_FP16, augment=False)
		SFM.train(restore_checkpoint = self.PRETRAIN_DIR)

	def create_blank_annotations_file(self, scan_for_cases=False):
		case_header_name = TCGAAnnotations.case

		with open(self.ANNOTATIONS_FILE, 'w') as csv_outfile:
			csv_writer = csv.writer(csv_outfile, delimiter=',')
			header = [case_header_name]
			csv_writer.writerow(header)

		if scan_for_cases:
			pass
		
	def update_task(self, task, status):
		return
		#data = sfutil.parse_config(self.CONFIG)
		#data['tasks'][task] = status
		#sfutil.write_config(data, self.CONFIG)

	def get_task(self, task):
		return sfutil.parse_config(self.CONFIG)['tasks'][task]

	def load_project(self):
		if os.path.exists(self.CONFIG):
			data = sfutil.parse_config(self.CONFIG)
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
			sfutil.verify_annotations(self.ANNOTATIONS_FILE, slides_dir=self.SLIDES_DIR)

			# If tile extraction has already been started, verify all slides in the annotation file
			#  have corresponding image tiles
			if (self.get_task('extract_tiles') != "not started") and (not self.USE_TFRECORD or self.get_task('generate_tfrecord') == 'complete'):
				if sfutil.yes_no_input("Perform image tile verification? [Y/n] ", default='yes'):
					input_dir = self.TFRECORD_DIR if self.USE_TFRECORD else self.TILES_DIR
					annotations = sfutil.get_annotations_dict(self.ANNOTATIONS_FILE, key_name="slide", value_name="category")
					tfrecord_files = [os.path.join(input_dir, f"{x}.tfrecords") for x in ["train", "eval"]] if self.USE_TFRECORD else []
					sfutil.verify_tiles(annotations, input_dir, tfrecord_files)

			print("\nProject configuration loaded.\n")
		else:
			raise OSError(f'Unable to locate project json at location "{self.CONFIG}".')

	def create_project(self):
		# General setup and slide configuration
		self.NAME = input("What is the project name? ")
		self.SLIDES_DIR = sfutil.dir_input("Where are the SVS slides stored? [./slides] ",
									default='./slides', create_on_invalid=True)
		self.ROI_DIR = sfutil.dir_input("Where are the ROI files (CSV) stored? [./slides] ",
									default='./slides', create_on_invalid=True)
		
		# Ask for annotations file location; if one has not been made, offer to create a blank template and then exit
		if not sfutil.yes_no_input("Has an annotations (CSV) file already been created? [Y/n] ", default='yes'):
			if sfutil.yes_no_input("Create a blank annotations file? [Y/n] ", default='yes'):
				self.ANNOTATIONS_FILE = sfutil.file_input("Where will the annotation file be located? [./annotations.csv] ", 
									default='./annotations.csv', filetype="csv", verify=False)
				self.create_blank_annotations_file(scan_for_cases=sfutil.yes_no_input("Scan slide folder for case names? [Y/n] ", default='yes'))
		else:
			self.ANNOTATIONS_FILE = sfutil.file_input("Where is the project annotations (CSV) file located? [./annotations.csv] ", 
									default='./annotations.csv', filetype="csv")

		# Slide tessellation
		self.TILES_DIR = sfutil.dir_input("Where will the tessellated image tiles be stored? (recommend SSD) [./tiles] ",
									default='./tiles', create_on_invalid=True)
		self.USE_TFRECORD = sfutil.yes_no_input("Store tiles in TFRecord format? [Y/n] ", default='yes')
		if self.USE_TFRECORD:
			self.DELETE_TILES = sfutil.yes_no_input("Should raw tile images be deleted after TFRecord storage? [Y/n] ", default='yes')
			self.TFRECORD_DIR = sfutil.dir_input("Where should the TFRecord files be stored? (recommend HDD) [./tfrecord] ",
									default='./tfrecord', create_on_invalid=True)
		# Training
		self.MODELS_DIR = sfutil.dir_input("Where should the saved models be stored? [./models] ",
									default='./models', create_on_invalid=True)
		if sfutil.yes_no_input("Will models utilize pre-training? [y/N] ", default='no'):
			self.PRETRAIN_DIR = sfutil.dir_input("Where is the pretrained model folder located? ", create_on_invalid=False)
		self.TILE_UM = sfutil.int_input("What is the tile width in microns? [280] ", default=280)
		self.TILE_PX = sfutil.int_input("What is the tile width in pixels? [512] ", default=512)
		self.NUM_CLASSES = sfutil.int_input("How many classes are there to be trained? ")
		self.BATCH_SIZE = sfutil.int_input("What batch size should be used? [64] ", default=64)
		self.USE_FP16 = sfutil.yes_no_input("Should FP16 be used instead of FP32? (recommended) [Y/n] ", default='yes')

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
		sfutil.write_config(data, self.CONFIG)
		print("\nProject configuration saved.\n")