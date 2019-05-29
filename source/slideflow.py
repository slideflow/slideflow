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
import histcon
from  util import datasets, tfrecords

ignore = '''TCGA-BJ-A0ZG-01Z-00-DX1.99FBAFA8-F009-4291-8217-26C64A1A470B
TCGA-BJ-A45D-01Z-00-DX1.671AA845-0931-4830-B837-5E121339A7AB
TCGA-BJ-A45G-01Z-00-DX1.044854D2-C2A7-4011-97BD-F42214B9031B
TCGA-BJ-A45K-01Z-00-DX1.3074A445-BE1B-491D-918B-AB65B54DBAD4
TCGA-DE-A2OL-01Z-00-DX1.A7E6EC42-A184-415A-9531-D1E83436FAE2
TCGA-DJ-A13M-01Z-00-DX1.B26C53E3-F572-463B-B7EA-4A6AA8CF59DF
TCGA-DJ-A13R-01Z-00-DX1.B219EFD3-A2BC-46CA-B797-DA6ED873A14C
TCGA-DJ-A13S-01Z-00-DX1.D92F9C56-8477-4C3E-848F-3948AD224015
TCGA-DJ-A13W-01Z-00-DX1.02059A44-7DF1-420D-BA48-587D611F34F5
TCGA-DJ-A1QM-01Z-00-DX1.3A58434F-A1A0-4FF4-B126-214C02CCEC7A
TCGA-DJ-A2PP-01Z-00-DX1.5BC2A5F2-1918-44E9-9544-1972974BA7BC
TCGA-DJ-A2PX-01Z-00-DX1.3C7F4F1E-8D23-4F10-B6FA-FD557F608EBB
TCGA-DJ-A2Q2-01Z-00-DX1.3CA0FEAB-FFA1-46C6-9558-E98F7746B451
TCGA-DJ-A2QA-01Z-00-DX1.0CC34156-B3F5-43DD-B740-A6F51BF45693
TCGA-DJ-A2QB-01Z-00-DX1.3C514B32-31C5-4BFF-8DDD-9BA583B33E19
TCGA-DJ-A3UR-01Z-00-DX1.5A6B0168-DEB6-45FF-A4CC-E36172B741C7
TCGA-DJ-A3UT-01Z-00-DX1.781CB82B-24FE-468A-8087-C5CB4D71B94B
TCGA-DJ-A3VG-01Z-00-DX1.15F3F9D8-17F0-49C0-B794-E45D841910F7
TCGA-DJ-A3VK-01Z-00-DX1.27EC85CB-86A8-4E18-A6F6-0B1540E9B7F0
TCGA-DJ-A3VL-01Z-00-DX1.B4873C68-3405-4944-AFF4-C87AED853BDE
TCGA-DJ-A3VM-01Z-00-DX1.C2613B37-AE47-42CD-809C-28B9F2204247
TCGA-DJ-A4UR-01Z-00-DX1.F97A2EC3-3EB8-40F7-BA2F-A2C3DEA1B133
TCGA-DO-A1JZ-01Z-00-DX1.CD2D8D52-8E7A-4187-A5BA-DF3E0FA7984A
TCGA-DO-A1JZ-01Z-00-DX2.0AFF242D-24B6-4A84-BA67-9C6BF870DA3A
TCGA-E3-A3DY-01Z-00-DX1.5FFB8A1E-3AC8-494D-8112-5F1FBDC3F597
TCGA-E3-A3DZ-01Z-00-DX1.A7EF818F-31D1-4EFE-9220-4EF14A682F03
TCGA-E3-A3E2-01Z-00-DX1.E21626F3-0840-40ED-B853-E3E2992EBB24
TCGA-EL-A3CP-01Z-00-DX1.D0F4C535-AF19-4F30-9440-5B6656B0C3F0
TCGA-EM-A1YA-01Z-00-DX1.6CEACBCA-9D05-4BF4-BFF2-725F869ABFA8
TCGA-EM-A1YB-01Z-00-DX1.687B8D57-43FF-4673-BC3A-F4045F163DD7
TCGA-EM-A1YD-01Z-00-DX1.476300DE-BEE0-4B66-8128-2C06A2A25D0A
TCGA-EM-A22L-01Z-00-DX1.1EA5070D-EA82-4A9F-ACEA-0FC5F1120689
TCGA-EM-A22N-01Z-00-DX1.DB853528-DCF5-4463-B196-AFA8670D67FF
TCGA-EM-A22Q-01Z-00-DX1.C9DC2543-BAA0-4930-A871-0BAFE5A391F7
TCGA-EM-A2CL-01Z-00-DX1.32F1BCF4-7F07-4653-A859-056DE18DE6CE
TCGA-EM-A2CN-01Z-00-DX1.8A82A101-19A6-4809-B873-B67014DD9E0A
TCGA-EM-A2CO-01Z-00-DX1.D4390B8C-50DD-414C-906A-3338C97BAD10
TCGA-EM-A2CQ-01Z-00-DX1.DD22D5B4-3638-44B3-8127-03458B2ABC4D
TCGA-EM-A2CR-01Z-00-DX1.EC73EBFF-0EC3-4153-9F9B-CDEF918C89DF
TCGA-EM-A2CT-01Z-00-DX1.717B3037-9B23-42A2-BB3E-6710FCEF8D58
TCGA-EM-A2CU-01Z-00-DX1.7939ADB1-8224-47AC-900B-FFFCB294275D
TCGA-EM-A2OV-01Z-00-DX1.9E2596F8-5380-443C-888E-270809144429
TCGA-EM-A2OW-01Z-00-DX1.8495B3A9-2FD1-4ADF-868C-816C897FAD14
TCGA-EM-A2OY-01Z-00-DX1.A5DD274E-0EBD-4859-809E-AE5CC8D75016
TCGA-EM-A3AJ-01Z-00-DX1.20774305-2AB9-41A6-93C4-ACF1F8D55BAA
TCGA-EM-A3AL-01Z-00-DX1.BF94702E-FEE4-4684-A26E-3D405119B3D5
TCGA-EM-A3FL-01Z-00-DX1.8B0828AC-4EFF-4C9A-8B11-B249C97C6F59
TCGA-EM-A3FN-01Z-00-DX1.B93AB1DC-1F4A-46FE-A269-0BEC3256E76F
TCGA-EM-A3FP-01Z-00-DX1.AE63796E-37CE-41B8-B15C-FDCA8BA4DF2D
TCGA-EM-A3O6-01Z-00-DX1.D6CB7107-7A56-4C76-A65C-033FD214EBA5
TCGA-EM-A3O9-01Z-00-DX1.D875DF72-EBDE-43FD-9FAE-3F696D4C0F03
TCGA-EM-A3OA-01Z-00-DX1.6EDE47B4-8926-4598-AB68-439590DE7CA8
TCGA-EM-A3OB-01Z-00-DX1.CCD73BD9-C5C0-4429-9F74-701DC8B54860
TCGA-EM-A3SY-01Z-00-DX1.34A87FD3-A47E-448B-8558-24C4FE15C641
TCGA-EM-A4FH-01Z-00-DX1.6EB7BADA-ADC1-47A9-A915-4034BA055777
TCGA-EM-A4FK-01Z-00-DX1.3D8685DA-332D-4F12-A33E-F23E33E837D8
TCGA-EM-A4FQ-01Z-00-DX1.EDAA6AF4-8736-46C8-AF49-F4901AC5D0C2
TCGA-EM-A4FU-01Z-00-DX1.6195521B-D0CB-4658-95CC-8050A2929649
TCGA-EM-A4G1-01Z-00-DX1.285286C5-567C-48AE-B741-F5433241E096
TCGA-ET-A2N3-01Z-00-DX2.98964734-A316-4B21-AE31-5796B6FFE3F4
TCGA-ET-A2N4-01Z-00-DX1.726A5148-0065-4494-9352-3CBFB9E58536
TCGA-ET-A2N5-01Z-00-DX1.07BFF458-3765-4256-96DA-63DEA49A475D
TCGA-ET-A39I-01Z-00-DX1.1720ECA0-1C10-4796-96FA-261C32BA67F1
TCGA-ET-A39T-01Z-00-DX1.7006B750-108F-4C2F-A934-80A2434FA029
TCGA-ET-A3DQ-01Z-00-DX1.A9648057-5AC1-4A77-9A82-846BF4D5804E
TCGA-ET-A3DV-01Z-00-DX1.7871EE46-984D-432A-9CE4-FE98C253374E
TCGA-ET-A40P-01Z-00-DX1.BC1D4E35-CD10-44C9-8A0C-9818E4365D51
TCGA-ET-A4KQ-01Z-00-DX1.0FE00105-559A-4B26-8AD7-904FF418DC54
TCGA-FE-A3PA-01Z-00-DX1.C9930C2F-0D3F-44FB-B509-C62A3FCF8EE1
TCGA-FE-A3PD-01Z-00-DX1.92071070-3B46-499C-9721-FB6677F20182
TCGA-FY-A3I5-01Z-00-DX1.E14BD882-987B-4F24-9210-06E06C80779F
TCGA-FY-A3NP-01Z-00-DX1.AD8D020E-2B02-4A62-B9F2-1DBDCA8DB8A1
TCGA-FY-A3WA-01Z-00-DX1.1DFC59E4-7114-4EA5-8038-224E353F6499
TCGA-H2-A3RH-01Z-00-DX1.444D8191-ADFF-4A59-BC33-B08F246594AE
TCGA-IM-A41Z-01Z-00-DX1.BEB9E1F0-75E7-418D-921A-0B28268F52FE
TCGA-KS-A41I-01Z-00-DX1.84CF483E-44A6-47DC-A8B6-3F4FC77A886C
TCGA-KS-A41L-01Z-00-DX1.285904AC-86B5-4CFD-80B6-2AC9ABD5A3AA
TCGA-KS-A4ID-01Z-00-DX1.36A8DF83-3A9A-496D-92F2-C39379427FF4
TCGA-L6-A4ET-01Z-00-DX1.C1AFDAD4-C9D5-423D-9647-ADBBEB693611'''

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

	def prepare_tiles(self):
		self.extract_tiles()
		self.separate_training_and_eval()
		if self.USE_TFRECORD:
			self.generate_tfrecord()

	def extract_tiles(self):
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

		c = convoluter.Convoluter(self.TILE_PX, self.TILE_UM, self.NUM_CLASSES, self.BATCH_SIZE, 
									self.USE_FP16, join(self.TILES_DIR, "train_data"), self.ROI_DIR, self.AUGMENTATION)

		num_dir = len(self.SLIDES_DIR.split('/'))
		slide_list = [i for i in glob(join(self.SLIDES_DIR, '**/*.svs'))
					  if i.split('/')[num_dir] != 'thumbs']

		slide_list.extend( [i for i in glob(join(self.SLIDES_DIR, '**/*.jpg'))
					  		if i.split('/')[num_dir] != 'thumbs'] )

		slide_list.extend(glob(join(self.SLIDES_DIR, '*.svs')))
		slide_list.extend(glob(join(self.SLIDES_DIR, '*.jpg')))
		
		for i, slide in enumerate(slide_list):
			slide_name = slide.split('/')[-1][:-4]
			if slide_name in ignore.split('\n'):
				del slide_list[i]
		print(len(slide_list))
		sys.exit()
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
		tfrecords.write_tfrecords(join(self.TILES_DIR, "train_data"), self.TFRECORD_DIR, "train")
		tfrecords.write_tfrecords(join(self.TILES_DIR, "eval_data"), self.TFRECORD_DIR, "eval")
		if self.DELETE_TILES:
			shutil.rmtree(join(self.TILES_DIR, "train_data"))
			shutil.rmtree(join(self.TILES_DIR, "eval_data"))
		self.update_task('generate_tfrecord', 'complete')

	def start_training(self, model_name):
		self.update_task('generate_tfrecord', 'in process')
		print(f"Training model {model_name}...")
		model_dir = join(self.MODELS_DIR, model_name)
		tensorboard_dir = join(model_dir, 'logs/projector')
		if not exists(model_dir):
			datasets.make_dir(model_dir)
		if not exists(tensorboard_dir):
			datasets.make_dir(tensorboard_dir)
		input_dir = self.TFRECORD_DIR if self.USE_TFRECORD else self.TILES_DIR
		histcon_model = histcon.HistconModel(model_dir, input_dir)
		histcon_model.train(restore_checkpoint = self.PRETRAIN_DIR)
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