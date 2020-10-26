import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import slideflow as sf
import tensorflow as tf
import csv
import shutil

from slideflow.io import Dataset
from slideflow.util import TCGA, log, ProgressBar
from slideflow.statistics import TFRecordMap

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
	},
	'TEST2': {
		'slides': '/media/Backup/SLIDES/THCA/UCH',
		'roi': '/media/Backup/SLIDES/THCA/UCH',
		'tiles': '/home/shawarma/data/test_project/tiles/TEST2',
		'tfrecords': '/home/shawarma/data/test_project/tfrecords/TEST2',
	},
	'TMA': {
		'slides': '/home/shawarma/data/test_slides',
		'roi': '/home/shawarma/data/test_slides',
		'tiles': '/home/shawarma/data/test_project/tiles/TMA',
		'tfrecords': '/home/shawarma/data/test_project/tfrecords/TMA',
	}
}
PROJECT_CONFIG = {
	'root': '/home/shawarma/data/test_project',
	'name': 'TEST_PROJECT',
	'annotations': '/home/shawarma/data/test_project/annotations.csv',
	'dataset_config': '/home/shawarma/data/test_datasets.json',
	'datasets': ['TEST2', 'TMA'],
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
	['234839', 'TEST2', 'PTC-follicular', 'BRAF', '1.1', '1.2', '234839'],
	['234834', 'TEST2', 'PTC-follicular', 'BRAF', '2.1', '2.2', '234834'],
	['234809', 'TEST2', 'PTC-follicular', 'BRAF', '2.2', '1.2', ''],
	['234840', 'TEST2', 'PTC-follicular', 'BRAF', '2.8', '4.2', '234840'],
	['234832', 'TEST2', 'PTC-follicular', 'Non-mutant', '4.3', '3.2', ''],
	['234803', 'TEST2', 'PTC-follicular', 'Non-mutant', '2.2', '1.2', ''],
	['234823', 'TEST2', 'PTC-follicular', 'Non-mutant', '0.2', '1.1', ''],
	['234833', 'TEST2', 'PTC-follicular', 'Non-mutant', '7.2', '4.2', ''],

	['234798', 'TEST2', 'NIFTP', 'cat2a', '2.8', '4.8', ''],
	['234808', 'TEST2', 'NIFTP', 'cat2b', '2.8', '4.7', ''],
	['234810', 'TEST2', 'NIFTP', 'cat2a', '3.8', '4.6', ''],
	['234829', 'TEST2', 'NIFTP', 'cat2b', '4.8', '4.5', ''],
	['234843', 'TEST2', 'NIFTP', 'cat2a', '5.8', '4.4', ''],
	['234851', 'TEST2', 'NIFTP', 'cat2b', '6.8', '4.2', ''],
	['234867', 'TEST2', 'NIFTP', 'cat2a', '7.8', '4.1', ''],

	['TMA_1185', 'TMA', 'PRAD', 'None', '7.8', '4.1', ''],
]

SLIDES_TO_VERIFY = ['234834', '234840']

SAVED_MODEL = join(PROJECT_CONFIG['models_dir'], 'category1-performance-kfold1', 'trained_model_epoch1.h5')

# --------------------------------------------------------------------------------------

class TestSuite:
	'''Class to supervise standardized testing of slideflow pipeline.'''
	def __init__(self, reset=True, silent=True, buffer=None):
		'''Initialize testing models.'''
			
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

		# Setup buffering
		self.buffer = buffer

	def reset(self):
		log.header("Resetting test project...")
		if os.path.exists(PROJECT_CONFIG['dataset_config']):
			os.remove(PROJECT_CONFIG['dataset_config'])
		if os.path.exists(PROJECT_CONFIG['root']):
			shutil.rmtree(PROJECT_CONFIG['root'])
		for dataset_name in TEST_DATASETS.keys():
			if os.path.exists(TEST_DATASETS[dataset_name]['tiles']):
				shutil.rmtree(TEST_DATASETS[dataset_name]['tiles'])
			if os.path.exists(TEST_DATASETS[dataset_name]['tfrecords']):
				print(f"Removing {TEST_DATASETS[dataset_name]['tfrecords']}")
				shutil.rmtree(TEST_DATASETS[dataset_name]['tfrecords'])
		print("\t...DONE")

	def configure_project(self):
		log.header("Setting up initial project configuration...")
		self.SFP.PROJECT = PROJECT_CONFIG
		self.SFP.save_project()
		print("\t...DONE")

	def configure_datasets(self):
		log.header("Setting up test dataset configuration...")
		for dataset_name in TEST_DATASETS.keys():
			self.SFP.add_dataset(dataset_name, slides=TEST_DATASETS[dataset_name]['slides'],
											   roi=TEST_DATASETS[dataset_name]['roi'],
											   tiles=TEST_DATASETS[dataset_name]['tiles'],
											   tfrecords=TEST_DATASETS[dataset_name]['tfrecords'],
											   path=PROJECT_CONFIG['dataset_config'])
		print("\t...DONE")

	def configure_annotations(self):
		log.header("Testing annotation configuration and slide name associations...")
		outfile = PROJECT_CONFIG['annotations']
		with open(outfile, 'w') as csv_outfile:
			csv_writer = csv.writer(csv_outfile, delimiter=',')
			for an in ANNOTATIONS:
				csv_writer.writerow(an)
		project_dataset = Dataset(tile_px=299, tile_um=302,
								  config_file=PROJECT_CONFIG['dataset_config'],
								  sources=PROJECT_CONFIG['datasets'],
								  annotations=PROJECT_CONFIG['annotations'])
		project_dataset.update_annotations_with_slidenames(PROJECT_CONFIG['annotations'])
		loaded_slides = project_dataset.get_slides()
		for slide in SLIDES_TO_VERIFY:
			if slide not in loaded_slides:
				log.error(f"Failed to correctly associate slide names ({slide}); please see annotations file below.")
				with open(outfile, 'r') as ann_read:
					print()
					print(ann_read.read())
				print("\t...FAILED")
				return
		print("\t...OK")

	def setup_hp(self, model_type):
		# Remove old batch train file
		log.header("Setting up hyperparameter setup...")
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
		self.SFP.create_hyperparameter_sweep(tile_px=299, tile_um=302,
											 finetune_epochs=[1],
											 toplayer_epochs=[0],
											 model=["InceptionV3"],
											 pooling=["max"],
											 loss=[loss],
											 learning_rate=[0.001],
											 batch_size=[64],
											 hidden_layers=[0,1],
											 optimizer=["Adam"],
											 early_stop=[False],
											 early_stop_patience=[15],
											 early_stop_method='loss',
											 hidden_layer_width=500,
											 trainable_layers=0,
											 L2_weight=0,
											 balanced_training=["BALANCE_BY_PATIENT"],
											 balanced_validation=["NO_BALANCE"],
											 augment=[True],
											 label='TEST',
											 filename=PROJECT_CONFIG["batch_train_config"])

		# Create single hyperparameter combination
		hp = sf.model.HyperParameters(finetune_epochs=1, toplayer_epochs=0, model='InceptionV3', pooling='max', loss=loss,
				learning_rate=0.001, batch_size=64, hidden_layers=1, optimizer='Adam', early_stop=False, 
				early_stop_patience=0, balanced_training='BALANCE_BY_PATIENT', balanced_validation='NO_BALANCE', 
				augment=True)
		print("\t...DONE")
		return hp

	def test_extraction(self, regular=True, tma=True):
		# Test tile extraction, default parameters, for regular slides
		if regular:
			log.header("Testing multiple slides extraction...")
			self.SFP.extract_tiles(tile_px=299, tile_um=302, buffer=self.buffer, dataset=['TEST2'])
		if tma:
			log.header("Testing Tumor Micro-array (TMA) extraction...")
			self.SFP.extract_tiles(tile_px=299, tile_um=302, buffer=self.buffer, dataset=['TMA'], tma=True)
		print("\t...OK")

	def test_single_extraction(self, buffer=True):
		log.header("Testing single slide extraction...")
		extracting_dataset = Dataset(tile_px=299, tile_um=302, config_file=self.SFP.PROJECT['dataset_config'], sources=self.SFP.PROJECT['datasets'])
		extracting_dataset.load_annotations(self.SFP.PROJECT['annotations'])
		dataset_name = self.SFP.PROJECT['datasets'][0]
		slide_list = extracting_dataset.get_slide_paths(dataset=dataset_name)
		roi_dir = extracting_dataset.datasets[dataset_name]['roi'] 
		tiles_dir = extracting_dataset.datasets[dataset_name]['tiles']
		pb = None#ProgressBar(bar_length=5, counter_text='tiles')
		whole_slide = sf.slide.SlideReader(slide_list[0], 299, 302, 1, enable_downsample=False, export_folder=tiles_dir, roi_dir=roi_dir, roi_list=None, buffer=buffer, pb=pb) 
		whole_slide.extract_tiles(normalizer='macenko')
		print("\t...OK")

	def test_realtime_normalizer(self):
		print("Testing realtime normalization, using Macenko")
		hp = self.setup_hp('categorical')
		self.SFP.train(outcome_header='category1', k_fold_iter=1, normalizer='macenko', normalizer_strategy='realtime')

	def test_training(self, categorical=True, linear=True, multi_input=True):
		if categorical:
			# Test categorical outcome
			hp = self.setup_hp('categorical')
			print("Training to single categorical outcome from specified hyperparameters...")
			results_dict = self.SFP.train(models = 'manual_hp', outcome_header='category1', hyperparameters=hp, k_fold_iter=1, validate_on_batch=50)
			
			if not results_dict or 'history' not in results_dict[results_dict.keys()[0]]:
				print("\tFAIL: Keras results object not received from training")
			else:
				print("\t...OK")

			print("Training to multiple sequential categorical outcomes from batch train file...")
			# Test multiple sequential categorical outcome models
			self.SFP.train(outcome_header=['category1', 'category2'], k_fold_iter=1)
			print("\t...OK")
		if linear:
			# Test single linear outcome
			hp = self.setup_hp('linear')
			# Test multiple linear outcome
			print("Training to multiple linear outcomes...")
			self.SFP.train(outcome_header=['linear1', 'linear2'], multi_outcome=True, k_fold_iter=1, validate_on_batch=50)
			print("\t...OK")
		if multi_input:
			hp = self.setup_hp('categorical')
			print("Training with multiple input types...")
			self.SFP.train(outcome_header='category1', input_header='category2', k_fold_iter=1, validate_on_batch=50)
		print("\t...OK")

	def test_training_performance(self):
		hp = self.setup_hp('categorical')
		hp.finetune_epochs = [1,3]
		log.header("Testing performance of training (single categorical outcome)...")
		results_dict = self.SFP.train(models='performance', outcome_header='category1', hyperparameters=hp, k_fold_iter=1)

	def test_evaluation(self):
		log.header("Testing evaluation of a saved model...")
		results = self.SFP.evaluate(outcome_header='category1', model=SAVED_MODEL)
		print('\t...OK')

	def test_heatmap(self):
		log.header("Testing heatmap generation...")
		self.SFP.generate_heatmaps(SAVED_MODEL, filters={TCGA.patient: ['234839']})
		print("\t...OK")

	def test_mosaic(self):
		log.header("Testing mosaic generation...")
		self.SFP.generate_mosaic(SAVED_MODEL)
		print("\t...OK")

	def test_activations(self):
		log.header("Testing activations analytics...")
		AV = self.SFP.generate_activations_analytics(model=SAVED_MODEL, 
													outcome_header='category1', 
													focus_nodes=[0])
		AV.generate_box_plots()
		umap = TFRecordMap.from_activations(AV)
		umap.save_2d_plot(join(PROJECT_CONFIG['root'], 'stats', '2d_umap.png'))
		top_nodes = AV.get_top_nodes_by_slide()
		for node in top_nodes[:5]:
			umap.save_3d_node_plot(node, join(PROJECT_CONFIG['root'], 'stats', f'3d_node{node}.png'))
		print("\t...OK")

	def test(self, extract=True, train=True, train_performance=True, evaluate=True, heatmap=True, mosaic=True, activations=True):
		'''Perform and report results of all available testing.'''
		if extract: 			self.test_extraction()
		if train:				self.test_training()
		if train_performance: 	self.test_training_performance()
		if evaluate:			self.test_evaluation()
		if heatmap:				self.test_heatmap()
		if mosaic:				self.test_mosaic()
		if activations:			self.test_activations()