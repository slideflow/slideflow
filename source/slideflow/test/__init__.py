import tensorflow as tf
import slideflow as sf
import os
import csv

from slideflow.trainer import model as sfmodel
from slideflow.util import TCGA, log

from glob import glob
from os.path import join

''' Todo:
- JPG, TIFF, SVS
- Verify properties: dimensions, properties (dict), level_dimensions, level_count, level_downsamples
- Verify ROI (area, coordinates)
- Verify thumbnail
- Verify extracted tiles at different pixels and microns
- Verify logits and prelogits based on saved model
- Verify heatmaps
'''

# --- TEST suite configuration --------------------------------------------------------
TEST_DATASETS = {
	'TEST1': {
		'slides': '/home/shawarma/data/HNSC/train_slides',
		'roi': '/home/shawarma/data/HNSC/train_slides',
		'tiles': '/home/shawarma/data/test_project/tiles/TEST1',
		'tfrecords': '/home/shawarma/data/test_project/tfrecords/TEST1',
		'label': sf.NO_LABEL
	},
	'TEST2': {
		'slides': '/media/Backup/Other_files/Thyroid/SVS/PTC-follicular',
		'roi': '/media/Backup/Other_files/Thyroid/SVS/PTC-follicular',
		'tiles': '/home/shawarma/data/test_project/tiles/TEST2',
		'tfrecords': '/home/shawarma/data/test_project/tfrecords/TEST2',
		'label': sf.NO_LABEL
	}
}
PROJECT_CONFIG = {
	'root': '/home/shawarma/data/test_project',
	'name': 'TEST_PROJECT',
	'annotations': '/home/shawarma/data/test_project/annotations.csv',
	'dataset_config': '/home/shawarma/data/test_datasets.json',
	'datasets': ['TEST1', 'TEST2'],
	'delete_tiles': False,
	'models_dir': '/home/shawarma/data/test_project/models',
	'tile_um': 302,
	'tile_px': 299,
	'use_fp16': True,
	'batch_train_config': '/home/shawarma/data/test_project/batch_train.csv',
	'validation_fraction': 0.2,
	'validation_target': 'per-patient',
	'validation_strategy': 'k-fold',
	'validation_k_fold': 3,
}

ANNOTATIONS = [
	[TCGA.patient, 'dataset', 'category1', 'category2', 'linear1', 'linear2', 'slide'],
	['234839', 'TEST2', 'cat1a', 'cat2a', '1.1', '1.2', ''],
	['234834', 'TEST2', 'cat1b', 'cat2a', '2.1', '2.2', ''],
	['234832', 'TEST2', 'cat1a', 'cat2b', '4.3', '3.2', ''],
	['234840', 'TEST2', 'cat1b', 'cat2b', '2.8', '4.2', ''],
	['235551', 'TEST1', 'cat1a', 'cat2a', '0.9', '2.2', ''],
	['235552', 'TEST1', 'cat1b', 'cat2b', '5.1', '0.2', ''],
	['235553', 'TEST1', 'cat1a', 'cat2b', '3.1', '8.7', '']
]

# --------------------------------------------------------------------------------------

class TestSuite:
	'''Class to supervise standardized testing of slideflow pipeline.'''
	def __init__(self, reset=True):
		'''Initialize testing models.'''
		#sf.set_logging_level(sf.SILENT)

		# Force slideflow into testing mode
		sfmodel.TEST_MODE = True

		# Reset test progress
		if reset: self.reset()

		# Intiailize project
		self.SFP = sf.SlideflowProject(PROJECT_CONFIG['root'], interactive=False)
		self.configure_project()

		# Configure datasets (input)
		self.configure_datasets()
		self.configure_annotations()

		# Prepare batch training
		self.setup_hp("categorical")

	def reset(self):
		print("Resetting test project...")
		try:
			os.remove(PROJECT_CONFIG['dataset_config'])
		except:
			pass
		try:
			os.remove(PROJECT_CONFIG['root'])
		except:
			pass
		for dataset_name in TEST_DATASETS.keys():
			try:
				shutil.rmtree(TEST_DATASETS[dataset_name]['tiles'])
			except:
				pass
			try:
				shutil.rmtree(TEST_DATASETS[dataset_name]['tfrecords'])
			except:
				pass
		print("\t...DONE")

	def configure_project(self):
		print("Setting up initial project configuration...")
		self.SFP.PROJECT = PROJECT_CONFIG
		self.SFP.save_project()
		print("\t...DONE")

	def configure_datasets(self):
		print("Setting up test dataset configuration...")
		for dataset_name in TEST_DATASETS.keys():
			self.SFP.add_dataset(dataset_name, slides=TEST_DATASETS[dataset_name]['slides'],
											   roi=TEST_DATASETS[dataset_name]['roi'],
											   tiles=TEST_DATASETS[dataset_name]['tiles'],
											   tfrecords=TEST_DATASETS[dataset_name]['tfrecords'],
											   label=TEST_DATASETS[dataset_name]['label'],
											   path=PROJECT_CONFIG['dataset_config'])
		print("\t...DONE")

	def configure_annotations(self):
		print("Testing annotation configuration and slide name associations...")
		outfile = PROJECT_CONFIG['annotations']
		with open(outfile, 'w') as csv_outfile:
			csv_writer = csv.writer(csv_outfile, delimiter=',')
			for an in ANNOTATIONS:
				csv_writer.writerow(an)
		self.SFP.associate_slide_names()
		print("\t...OK")

	def setup_hp(self, model_type):
		# Remove old batch train file
		print("Setting up hyperparameter setup...")
		try:
			os.remove(PROJECT_CONFIG['batch_train_config'])
		except:
			pass
		# Setup loss function
		if model_type == 'categorical':
			loss = 'sparse_categorical_crossentropy'
		elif model_type == 'linear':
			loss = 'mean_squared_error'
		# Create batch train file
		self.SFP.create_hyperparameter_sweep(finetune_epochs=[1],
											 toplayer_epochs=[0],
											 model=["Xception"],
											 pooling=["max"],
											 loss=[loss],
											 learning_rate=[0.001],
											 batch_size=[16],
											 hidden_layers=[0],
											 optimizer=["Adam"],
											 early_stop=[False],
											 early_stop_patience=[15],
											 balanced_training=["BALANCE_BY_PATIENT"],
											 balanced_validation=["NO_BALANCE"],
											 augment=[True],
											 filename=PROJECT_CONFIG["batch_train_config"])
		print("\t...DONE")

	def test_convolution(self):
		# Test tile extraction, default parameters
		print("Testing convolution...")
		self.SFP.extract_tiles()
		print("\t...OK")

	'''def test_input_stream(self, outcome, balancing, batch_size=16, augment=True, filters=None, model_type='categorical'):
		dataset, dataset_with_slidenames, num_tiles = SFM.build_dataset_inputs(SFM.TRAIN_TFRECORDS, batch_size=batch_size, 
																									balance=balancing,
																									augment=augment,
																									finite=False,
																									include_slidenames=False)'''
	def test_training(self, categorical=True, linear=True):
		if categorical:
			# Test categorical outcome
			self.setup_hp('categorical')
			print("Testing single categorical outcome training...")
			self.SFP.train(outcome_header='category1')
			print("\t...OK")
			print("Testing multiple sequential categorical outcome training...")
			# Test multiple sequential categorical outcome models
			self.SFP.train(outcome_header=['category1', 'category2'])
			print("\t...OK")
		if linear:
			# Test single linear outcome
			self.setup_hp('linear')
			print("Testing single linear outcome training...")
			self.SFP.train(outcome_header='linear1', model_type='linear')
			print("\t...OK")
			# Test multiple linear outcome
			print("Testing multiple linear outcome training...")
			self.SFP.train(outcome_header=['linear1', 'linear2'], multi_outcome=True, model_type='linear')
			print("\t...OK")
		print("\t...OK")

	def test_heatmap(self):
		print("Testing heatmap generation...")
		self.SFP.generate_heatmaps('category1-HPSweep0-kfold1')
		print("\t...OK")

	def test_mosaic(self):
		print("Testing mosaic generation...")
		self.SFP.generate_mosaic('category1-HPSweep0-kfold1')
		print("\t...OK")

	def test(self):
		'''Perform and report results of all available testing.'''
		self.test_convolution()
		self.test_training()
		self.test_heatmap()
		self.test_mosaic()
		#self.test_input_stream()