import os
import sys
import csv
import shutil
import requests
import json
import re
import time
import logging

import slideflow as sf
import slideflow.util as sfutil
from slideflow.io import Dataset
from slideflow.util import log, ProgressBar
from slideflow.util.spinner import Spinner
from slideflow.statistics import TFRecordMap

from glob import glob
from os.path import join

# Todo:
#- JPG, TIFF, SVS
#- Verify properties: dimensions, properties (dict), level_dimensions, level_count, level_downsamples
#- Verify ROI (area, coordinates)
#- Verify thumbnail
#- Verify extracted tiles at different pixels and microns
#- Verify logits and prelogits based on saved model
#- Verify heatmaps

SKIP_DOWNLOAD = 'skip'
AUTO_DOWNLOAD = 'auto'

class TestConfigurator:
    def __init__(self, root, download='auto', tma=False):
        self.DATASETS = {
            'TEST': {
                'slides': 	join(root, 'slides'),
                'roi': 		join(root, 'roi'),
                'tiles': 	join(root, 'project', 'tiles', 'TEST'), 
                'tfrecords':join(root, 'project', 'tfrecords', 'TEST') 
            }
        }
        self.PROJECT = {
            'root': join(root, 'project'), 
            'name': 'TEST_PROJECT',
            'annotations': join(root, 'project', 'annotations.csv'),
            'dataset_config': join(root, 'datasets.json'),
            'datasets': ['TEST', 'TMA'],
            'models_dir': join(root, 'project', 'models'),
            'tile_um': 302,
            'tile_px': 299,
            'use_fp16': True,
            'batch_train_config': join(root, 'project', 'batch_train.csv'),
            'validation_fraction': 0.2,
            'validation_target': 'per-patient',
            'validation_strategy': 'k-fold',
            'validation_k_fold': 3,
        }
        self.SLIDES = {
            'TCGA-BJ-A2N9-01Z-00-DX1.CFCB1FA9-7890-4B1B-93AB-4066E160FBF5':'0b0b560d-f3e7-4103-9b1b-d4981e00c0e7',
            'TCGA-BJ-A3PT-01Z-00-DX1.A307F39F-AE85-42F4-B705-11AF06F391D9':'0eeb9df4-4cb0-4075-9e18-3861dea2ba05',
            'TCGA-BJ-A45J-01Z-00-DX1.F3646444-749B-4583-A45D-17C580FCB866':'0c376805-5f09-4687-8e29-ad36b2171577',
            'TCGA-DJ-A2PT-01Z-00-DX1.8C28F7F7-426A-4AAC-8AC6-D082F85C4D34':'1af4e340-38d3-4589-8a7b-6be3f207bc06',
            'TCGA-DJ-A4UQ-01Z-00-DX1.2F88113C-4F3B-4250-A7C3-5B01AB6ABE55':'0d0e4ddf-749c-44ba-aea9-989732e79d8d',
            'TCGA-DJ-A13W-01Z-00-DX1.02059A44-7DF1-420D-BA48-587D611F34F5':'0c5592d5-b51c-406a-9dd5-72778e982f13',
            'TCGA-DO-A1K0-01Z-00-DX1.5ED4011C-6AAA-4197-8044-1F69D55CEAEE':'0d78b583-ecf2-45f4-95a4-dc61057be898',
            'TCGA-E3-A3E5-01Z-00-DX1.E7E8AB8B-695F-4158-A3C0-E2B801E07D2A':'1a4242c5-495d-46f2-b87d-050acc6cef44',
            'TCGA-E8-A242-01Z-00-DX1.9DDBB5BB-696E-4C61-BF4A-464062403F04':'1bcfd879-c48b-4232-b6a7-ff1337be9914',
            'TCGA-EL-A3CO-01Z-00-DX1.7BF5F004-E7E6-4320-BA89-39D05657BBCB':'0ac4f9a9-32f8-40b5-be0e-52ceeef7dbbf',
            'TCGA-EM-A1YA-01Z-00-DX1.6CEACBCA-9D05-4BF4-BFF2-725F869ABFA8':'0b239138-5e5e-4080-89a3-e19052f4cb7d',
            'TCGA-EM-A2CN-01Z-00-DX1.8A82A101-19A6-4809-B873-B67014DD9E0A':'1ad7bea7-5279-4c28-9357-b74a52e6b4c0',
            'TCGA-EM-A3SY-01Z-00-DX1.34A87FD3-A47E-448B-8558-24C4FE15C641':'01bd4e78-0a78-46c8-a4d5-7028ac3c84f8',
            'TCGA-EM-A22M-01Z-00-DX1.3E4CFB51-A1A5-48B5-96BC-F3EE763C7C5A':'0d6306d5-c687-4d59-a9be-2af18b9a8a2e',
            'TCGA-H2-A3RH-01Z-00-DX1.444D8191-ADFF-4A59-BC33-B08F246594AE':'0c4a266b-708d-4f01-8b9b-ac6272ffefff',
            'TCGA-IM-A41Z-01Z-00-DX1.BEB9E1F0-75E7-418D-921A-0B28268F52FE':'1ac4a8aa-f370-4d0d-9a41-6f3eb2f324ca',
        }
        self.ANNOTATIONS = [
            [sfutil.TCGA.patient, 'dataset', 'category1', 'category2', 'linear1', 'linear2'],
            ['TCGA-BJ-A2N9', 'TEST', 'PTC-follicular', 'BRAF', '1.1', '1.2'],
            ['TCGA-BJ-A3PT', 'TEST', 'PTC-follicular', 'BRAF', '2.1', '2.2'],
            ['TCGA-BJ-A45J', 'TEST', 'PTC-follicular', 'BRAF', '2.2', '1.2'],
            ['TCGA-DJ-A2PT', 'TEST', 'PTC-follicular', 'BRAF', '2.8', '4.2'],
            ['TCGA-DJ-A4UQ', 'TEST', 'PTC-follicular', 'Non-mutant', '4.3', '3.2'],
            ['TCGA-DJ-A13W', 'TEST', 'PTC-follicular', 'Non-mutant', '2.2', '1.2'],
            ['TCGA-DO-A1K0', 'TEST', 'PTC-follicular', 'Non-mutant', '0.2', '1.1'],
            ['TCGA-E3-A3E5', 'TEST', 'PTC-follicular', 'Non-mutant', '7.2', '4.2'],

            ['TCGA-E8-A242', 'TEST', 'NIFTP', 'cat2a', '2.8', '4.8'],
            ['TCGA-EL-A3CO', 'TEST', 'NIFTP', 'cat2b', '2.8', '4.7'],
            ['TCGA-EM-A1YA', 'TEST', 'NIFTP', 'cat2a', '3.8', '4.6'],
            ['TCGA-EM-A2CN', 'TEST', 'NIFTP', 'cat2b', '4.8', '4.5'],
            ['TCGA-EM-A3SY', 'TEST', 'NIFTP', 'cat2a', '5.8', '4.4'],
            ['TCGA-EM-A22M', 'TEST', 'NIFTP', 'cat2b', '6.8', '4.2'],
            ['TCGA-IM-A41Z', 'TEST', 'NIFTP', 'cat2a', '7.8', '4.1'],
        ]

        if tma:
            self.ANNOTATIONS += [['TMA_1185', 'TMA', 'PRAD', 'None', '7.8', '4.1']]
            self.DATASETS.update({'TMA': {
                'slides': 	join(root, 'tma_slides'),
                'roi': 		join(root, 'tma_roi'),
                'tiles': 	join(root, 'project', 'tiles', 'TMA'),
                'tfrecords':join(root, 'project', 'tfrecords', 'TMA')
            }})

        self.SAVED_MODEL = join(self.PROJECT['models_dir'], 'category1-performance-kfold1', 'trained_model_epoch1.h5')
        self.REFERENCE_MODEL = None

        # Verify slides
        with TaskWrapper("Downloading slides...") as test:
            if not os.path.exists(join(root, 'slides')): os.makedirs(join(root, 'slides'))
            existing_slides = [sfutil.path_to_name(f) for f in os.listdir(join(root, 'slides')) if sfutil.path_to_ext(f).lower() == 'svs']

            for slide in self.SLIDES:
                if slide not in existing_slides and download == AUTO_DOWNLOAD:
                    self.download_tcga(uuid=self.SLIDES[slide], 
                                    dest=join(root, 'slides'),
                                    message=f"Downloading {sfutil.green(slide)} from TCGA...")
            if download != AUTO_DOWNLOAD:
                test.skip()
    
    def download_tcga(self, uuid, dest, message=None):
        params = {'ids': [uuid]}
        data_endpt = "https://api.gdc.cancer.gov/data"
        response = requests.post(data_endpt, data = json.dumps(params), headers = {"Content-Type": "application/json"}, stream=True)
        response_head_cd = response.headers["Content-Disposition"]
        block_size = 4096
        file_size = int(response.headers.get('Content-Length', None))
        pb = ProgressBar(file_size, leadtext=message)
        file_name = join(dest, re.findall("filename=(.+)", response_head_cd)[0])
        with open(file_name, "wb") as output_file:
            for i, chunk in enumerate(response.iter_content(chunk_size=block_size)):
                output_file.write(chunk)
                pb.increase_bar_value(block_size)
        pb.end()

class TaskWrapper:
    '''Test wrapper to assist with logging.'''
    def __init__(self, message):
        self.message = message
        self.failed = False
        self.skipped = False
        self.start = time.time()
        self.spinner = Spinner(message)

    def __enter__(self):
        self.spinner.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_traceback):
        duration = time.time() - self.start
        self.spinner.__exit__(exc_type, exc_val, exc_traceback)
        if self.failed:
            self._end_msg("FAIL", sfutil.fail, f' [{duration:.0f} s]')
        elif self.skipped:
            self._end_msg("SKIPPED", sfutil.warn, f' [{duration:.0f} s]')
        else:
            self._end_msg("DONE", sfutil.green, f' [{duration:.0f} s]')
        
    def _end_msg(self, end_str, color_func, trail, width=80):
        right_msg = f' {color_func(end_str)}{trail}'
        if len(self.message) > width: left_msg = self.message[:width]
        else: left_msg = self.message + " " * (width - len(self.message))
        sys.stdout.write(left_msg)
        sys.stdout.write('\b' * (len(end_str) + len(trail) + 1))
        sys.stdout.write(right_msg)
        sys.stdout.flush()
        print()

    def fail(self):
        self.failed = True

    def skip(self):
        self.skipped = True

class TestSuite:
    '''Class to supervise standardized testing of slideflow pipeline.'''
    def __init__(self, root, reset=True, buffer=None, num_threads=8, debug=False, download=AUTO_DOWNLOAD, include_tma=False):
        '''Initialize testing models.'''
        
        # Set logging level
        if debug:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"
            import tensorflow as tf
            tf.get_logger().setLevel("INFO")
        else:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
            logging.getLogger("tensorflow").setLevel(logging.ERROR)
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")

        # Check if GPU available
        with TaskWrapper("Checking GPU availability...") as gpu_test:
            if not tf.test.is_gpu_available():
                gpu_test.fail()

        # Configure testing environment
        self.config = TestConfigurator(root, download=download, tma=include_tma)

        # Reset test progress
        with TaskWrapper("Project reset...") as reset_test:
            if reset: 
                self.reset()
            else:
                reset_test.skip()

        # Intiailize project
        log_levels = {
            'info': 3 if debug else 0,
            'warn': 3,
            'error': 3,
            'complete': 3 if debug else 0,
            'silent': False if debug else True
        }
        flags = sf.DEFAULT_FLAGS
        flags['num_threads'] = num_threads
        flags['logging_levels'] = log_levels

        self.SFP = sf.SlideflowProject(self.config.PROJECT['root'], interactive=False, flags=flags)
        self.SFP.PROJECT = self.config.PROJECT
        self.SFP.save_project()

        # Configure datasets (input)
        self.configure_datasets()
        self.configure_annotations(include_tma=include_tma)

        # Setup buffering
        self.buffer = buffer

    def reset(self):
        if os.path.exists(self.config.PROJECT['dataset_config']):
            os.remove(self.config.PROJECT['dataset_config'])
        if os.path.exists(self.config.PROJECT['root']):
            shutil.rmtree(self.config.PROJECT['root'])
        for dataset_name in self.config.DATASETS.keys():
            if os.path.exists(self.config.DATASETS[dataset_name]['tiles']):
                shutil.rmtree(self.config.DATASETS[dataset_name]['tiles'])
            if os.path.exists(self.config.DATASETS[dataset_name]['tfrecords']):
                print(f"Removing {self.config.DATASETS[dataset_name]['tfrecords']}")
                shutil.rmtree(self.config.DATASETS[dataset_name]['tfrecords'])

    def configure_datasets(self):
        with TaskWrapper("Dataset configuration...") as test:
            for dataset_name in self.config.DATASETS.keys():
                self.SFP.add_dataset(dataset_name, slides=self.config.DATASETS[dataset_name]['slides'],
                                                roi=self.config.DATASETS[dataset_name]['roi'],
                                                tiles=self.config.DATASETS[dataset_name]['tiles'],
                                                tfrecords=self.config.DATASETS[dataset_name]['tfrecords'],
                                                path=self.SFP.PROJECT['dataset_config'])

    def configure_annotations(self, include_tma=False):
        with TaskWrapper("Annotation configuration...") as test:
            outfile = self.SFP.PROJECT['annotations']
            with open(outfile, 'w') as csv_outfile:
                csv_writer = csv.writer(csv_outfile, delimiter=',')
                for an in self.config.ANNOTATIONS:
                    csv_writer.writerow(an)
            project_dataset = Dataset(tile_px=299, tile_um=302,
                                    config_file=self.SFP.PROJECT['dataset_config'],
                                    sources=(self.SFP.PROJECT['datasets'] if include_tma else 'TEST'),
                                    annotations=self.SFP.PROJECT['annotations'])
            project_dataset.update_annotations_with_slidenames(self.SFP.PROJECT['annotations'])
            loaded_slides = project_dataset.get_slides()
            for slide in [row[0] for row in self.config.ANNOTATIONS[1:]]:
                if slide not in [sfutil._shortname(l) for l in loaded_slides]:
                    print()
                    log.error(f"Failed to correctly associate slide names ({slide}); please see annotations file below.")
                    with open(outfile, 'r') as ann_read:
                        print(ann_read.read())
                    test.fail()
                    return

    def setup_hp(self, model_type):
        # Remove old batch train file
        try:
            os.remove(self.SFP.PROJECT['batch_train_config'])
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
                                            filename=self.SFP.PROJECT["batch_train_config"])

        # Create single hyperparameter combination
        hp = sf.model.HyperParameters(finetune_epochs=1, toplayer_epochs=0, model='InceptionV3', pooling='max', loss=loss,
                learning_rate=0.001, batch_size=64, hidden_layers=1, optimizer='Adam', early_stop=False, 
                early_stop_patience=0, balanced_training='BALANCE_BY_PATIENT', balanced_validation='NO_BALANCE', 
                augment=True)
        return hp

    def test_extraction(self, regular=True, tma=False):
        # Test tile extraction, default parameters, for regular slides
        if regular:
            with TaskWrapper("Testing slide extraction...") as test: 
                self.SFP.extract_tiles(tile_px=299, tile_um=302, buffer=self.buffer, dataset=['TEST'])
        if tma:
            with TaskWrapper("Testing Tumor Micro-array (TMA) extraction...") as test:
                self.SFP.extract_tiles(tile_px=299, tile_um=302, buffer=self.buffer, dataset=['TMA'], tma=True)

    def test_single_extraction(self, buffer=True):
        with TaskWrapper("Testing single slide extraction...") as test:
            extracting_dataset = Dataset(tile_px=299, tile_um=302, config_file=self.SFP.PROJECT['dataset_config'], sources=self.SFP.PROJECT['datasets'])
            extracting_dataset.load_annotations(self.SFP.PROJECT['annotations'])
            dataset_name = self.SFP.PROJECT['datasets'][0]
            slide_list = extracting_dataset.get_slide_paths(dataset=dataset_name)
            roi_dir = extracting_dataset.datasets[dataset_name]['roi'] 
            tiles_dir = extracting_dataset.datasets[dataset_name]['tiles']
            pb = None#ProgressBar(bar_length=5, counter_text='tiles')
            whole_slide = sf.slide.SlideReader(slide_list[0], 299, 302, 1, enable_downsample=False, export_folder=tiles_dir, roi_dir=roi_dir, roi_list=None, buffer=buffer, pb=pb) 
            whole_slide.extract_tiles(normalizer='macenko')

    def test_realtime_normalizer(self):
        with TaskWrapper("Testing realtime normalization, using Macenko...") as test:
            hp = self.setup_hp('categorical')
            self.SFP.train(outcome_header='category1', k_fold_iter=1, normalizer='reinhard', normalizer_strategy='realtime', steps_per_epoch_override=5)

    def test_training(self, categorical=True, linear=True, multi_input=True):
        if categorical:
            # Test categorical outcome
            with TaskWrapper("Training to single categorical outcome from hyperparameters...") as test:
                hp = self.setup_hp('categorical')
                results_dict = self.SFP.train(models = 'manual_hp', outcome_header='category1', hyperparameters=hp, k_fold_iter=1, validate_on_batch=50, steps_per_epoch_override=5)
            
                if not results_dict or 'history' not in results_dict[results_dict.keys()[0]]:
                    print("\tKeras results object not received from training")
                    test.fail()

            # Test multiple sequential categorical outcome models
            with TaskWrapper("Training sequentially to multiple outcomes from batch train file...") as test:
                self.SFP.train(outcome_header=['category1', 'category2'], k_fold_iter=1, steps_per_epoch_override=5)

        if linear:
            # Test multiple linear outcome
            with TaskWrapper("Training to multiple linear outcomes...") as test:
                hp = self.setup_hp('linear')
                self.SFP.train(outcome_header=['linear1', 'linear2'], multi_outcome=True, k_fold_iter=1, validate_on_batch=50, steps_per_epoch_override=5)

        if multi_input:
            with TaskWrapper("Training with multiple input types...") as test:
                hp = self.setup_hp('categorical')
                self.SFP.train(outcome_header='category1', input_header='category2', k_fold_iter=1, validate_on_batch=50, steps_per_epoch_override=5)
            

    def test_training_performance(self):
        with TaskWrapper("Testing performance of training (single categorical outcome)...") as test:
            hp = self.setup_hp('categorical')
            hp.finetune_epochs = [1,3]
            results_dict = self.SFP.train(models='performance', outcome_header='category1', hyperparameters=hp, k_fold_iter=1)

    def test_evaluation(self):
        with TaskWrapper("Testing evaluation of a saved model...") as test:
            self.SFP.evaluate(outcome_header='category1', model=self.config.SAVED_MODEL)
        #print("Testing that evaluation matches known baseline...")
        #self.SFP.evaluate(outcome_header='category1', model=REFERENCE_MODEL, filters={'submitter_id': '234839'})
        # Code to lookup excel sheet of predictions and verify they match known baseline

    def test_heatmap(self, slide='auto'):
        with TaskWrapper("Testing heatmap generation...") as test:
            if slide.lower() == 'auto':
                sfdataset = self.SFP.get_dataset()
                slide_paths = sfdataset.get_slide_paths(dataset='TEST')
                patient_name = sfutil.path_to_name(slide_paths[0])
            self.SFP.generate_heatmaps(self.config.SAVED_MODEL, filters={sfutil.TCGA.patient: [patient_name]})

    def test_mosaic(self):
        with TaskWrapper("Testing mosaic generation...") as test:
            self.SFP.generate_mosaic(self.config.SAVED_MODEL, mosaic_filename="mosaic_test.png")

    def test_activations(self):
        with TaskWrapper("Testing activations analytics...") as test:
            AV = self.SFP.generate_activations(model=self.config.SAVED_MODEL, 
                                                        outcome_header='category1', 
                                                        focus_nodes=[0])
            AV.generate_box_plots()
            umap = TFRecordMap.from_activations(AV)
            umap.save_2d_plot(join(self.SFP.PROJECT['root'], 'stats', '2d_umap.png'))
            top_nodes = AV.get_top_nodes_by_slide()
            for node in top_nodes[:5]:
                umap.save_3d_plot(node=node, filename=join(self.SFP.PROJECT['root'], 'stats', f'3d_node{node}.png'))

    def test(self, extract=True, train=True, normalizer=True, train_performance=True, 
                evaluate=True,heatmap=True, mosaic=True, activations=True):
        '''Perform and report results of all available testing.'''
        if extract: 			self.test_extraction()
        if train:				self.test_training()
        if normalizer:			self.test_realtime_normalizer()
        if train_performance: 	self.test_training_performance()
        if evaluate:			self.test_evaluation()
        if heatmap:				self.test_heatmap()
        if mosaic:				self.test_mosaic()
        if activations:			self.test_activations()
        