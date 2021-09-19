import os
import sys
import csv
import shutil
import requests
import json
import re
import time
import logging
import random

import slideflow as sf
import slideflow.util as sfutil
from slideflow.io import Dataset
from slideflow.util import log, ProgressBar
from slideflow.util.spinner import Spinner
from slideflow.statistics import TFRecordMap
from os.path import join

#TODO:
#- JPG, TIFF, SVS
#- Verify properties: dimensions, properties (dict), level_dimensions, level_count, level_downsamples
#- Verify ROI (area, coordinates)
#- Verify thumbnail
#- Verify extracted tiles at different pixels and microns
#- Verify logits and prelogits based on saved model
#- Verify heatmaps
#- CPH model testing
#- CLAM testing

RANDOM_TCGA = 100

def get_random_tcga_slides():
    return {
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
        }

def download_from_tcga(uuid, dest, message=None):
    params = {'ids': [uuid]}
    data_endpt = "https://api.gdc.cancer.gov/data"
    response = requests.post(data_endpt,
                                data = json.dumps(params),
                                headers = {"Content-Type": "application/json"},
                                stream=True)
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

def random_annotations(slides_path):
    slides = [sfutil.path_to_name(f) for f in os.listdir(slides_path) 
                                        if sfutil.path_to_ext(f).lower() in sfutil.SUPPORTED_FORMATS][:10]
    annotations = [[sfutil.TCGA.patient, 'dataset', 'category1', 'category2', 'linear1', 'linear2']]
    for slide in slides:
        cat1 = random.choice(['A', 'B'])
        cat2 = random.choice(['C', 'D'])
        lin1 = random.random()
        lin2 = random.random()
        annotations += [[slide, 'TEST', cat1, cat2, lin1, lin2]]
    return annotations

class TestConfigurator:
    def __init__(self, path, slides=RANDOM_TCGA):
        '''Test Suite configuration.
        
        Args:
            path        Path to directory for test projects and data.
            slides      Specifies source of test slides. Either RANDOM_TCGA (default), or path to directory.
                            If RANDOM_TCGA, will download random sample of slides from TCGA for testing.
                            If path to directory containing slides, will use subset of slides at random for testing.
        '''
        random.seed(0)
        slides_path = join(path, 'slides') if slides == RANDOM_TCGA else slides
        if not os.path.exists(slides_path): os.makedirs(slides_path)
        self.datasets = {
            'TEST': {
                'slides': 	slides_path,
                'roi': 		join(path, 'roi'),
                'tiles': 	join(path, 'project', 'tiles', 'TEST'),
                'tfrecords':join(path, 'project', 'tfrecords', 'TEST')
            }
        }
        self.project_settings = {
            'root': join(path, 'project'),
            'name': 'TEST_PROJECT',
            'annotations': '$ROOT/annotations.csv',
            'dataset_config': join(path, 'datasets.json'),
            'datasets': ['TEST'],
            'models_dir': '$ROOT/models',
            'tile_um': 302,
            'tile_px': 299,
            'mixed_precision': True,
            'batch_train_config': '$ROOT/batch_train.csv',
        }
        if slides == RANDOM_TCGA:
            with TaskWrapper("Downloading slides..."):
                existing_slides = [sfutil.path_to_name(f) for f in os.listdir(slides_path)
                                                            if sfutil.path_to_ext(f).lower() in sfutil.SUPPORTED_FORMATS]
                for slide in [s for s in slides if s not in existing_slides]:
                    download_from_tcga(uuid=slides[slide],
                                       dest=slides_path,
                                       message=f"Downloading {sfutil.green(slide)} from TCGA...")

        self.annotations = random_annotations(slides_path)
        self.SAVED_MODEL = join(path, 'project', 'models', 'category1-performance-kfold1', 'trained_model_epoch1')
        self.REFERENCE_MODEL = None

class TaskWrapper:
    '''Test wrapper to assist with logging.'''
    VERBOSITY = logging.DEBUG

    def __init__(self, message):
        self.message = message
        self.failed = False
        self.skipped = False
        self.start = time.time()
        if self.VERBOSITY >= logging.WARNING:
            self.spinner = Spinner(message)

    def __enter__(self):
        if self.VERBOSITY >= logging.WARNING:
            self.spinner.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_traceback):
        duration = time.time() - self.start
        if self.VERBOSITY >= logging.WARNING:
            self.spinner.__exit__(exc_type, exc_val, exc_traceback)
        if self.failed:
            self._end_msg("FAIL", sfutil.red, f' [{duration:.0f} s]')
        elif self.skipped:
            self._end_msg("SKIPPED", sfutil.yellow, f' [{duration:.0f} s]')
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
    def __init__(self, root, slides=RANDOM_TCGA, reset=False, buffer=None, num_threads=8, debug=False, gpu=None):
        '''Initialize testing models.'''

        # Set logging level
        if debug:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"
            self.verbosity = logging.DEBUG
        else:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
            logging.getLogger("tensorflow").setLevel(logging.ERROR)
            self.verbosity = logging.WARNING

        TaskWrapper.VERBOSITY = self.verbosity

        # Configure testing environment
        self.config = TestConfigurator(root, slides=slides)

        # Reset test progress
        with TaskWrapper("Project reset...") as reset_test:
            if reset:
                self.reset()
            else:
                reset_test.skip()

        self.SFP = sf.SlideflowProject(self.config.project_settings['root'],
                                       interactive=False,
                                       gpu=gpu,
                                       verbosity=logging.DEBUG if debug else logging.ERROR,
                                       default_threads=num_threads)
        self.SFP._settings = self.config.project_settings
        self.SFP.save_project()

        # Check if GPU available
        import tensorflow as tf
        with TaskWrapper("Checking GPU availability...") as gpu_test:
            if not tf.config.list_physical_devices('GPU'):
                gpu_test.fail()

        # Finish logging settings after importing tensorflow
        if debug:
            tf.get_logger().setLevel("INFO")
        else:
            tf.get_logger().setLevel("ERROR")

        # Configure datasets (input)
        self.configure_datasets()
        self.configure_annotations()

        # Setup buffering
        self.buffer = buffer

    def reset(self):
        if os.path.exists(self.config.project_settings['dataset_config']):
            os.remove(self.config.project_settings['dataset_config'])
        if os.path.exists(self.config.project_settings['root']):
            shutil.rmtree(self.config.project_settings['root'])
        for dataset_name in self.config.datasets.keys():
            if os.path.exists(self.config.datasets[dataset_name]['tiles']):
                shutil.rmtree(self.config.datasets[dataset_name]['tiles'])
            if os.path.exists(self.config.datasets[dataset_name]['tfrecords']):
                print(f"Removing {self.config.datasets[dataset_name]['tfrecords']}")
                shutil.rmtree(self.config.datasets[dataset_name]['tfrecords'])

    def configure_datasets(self):
        with TaskWrapper("Dataset configuration...") as test:
            for dataset_name in self.config.datasets.keys():
                self.SFP.add_dataset(dataset_name, slides=self.config.datasets[dataset_name]['slides'],
                                                roi=self.config.datasets[dataset_name]['roi'],
                                                tiles=self.config.datasets[dataset_name]['tiles'],
                                                tfrecords=self.config.datasets[dataset_name]['tfrecords'],
                                                path=self.SFP.dataset_config)

    def configure_annotations(self):
        with TaskWrapper("Annotation configuration...") as test:
            outfile = self.SFP.annotations
            with open(outfile, 'w') as csv_outfile:
                csv_writer = csv.writer(csv_outfile, delimiter=',')
                for an in self.config.annotations:
                    csv_writer.writerow(an)
            project_dataset = Dataset(tile_px=299, 
                                      tile_um=302,
                                      sources='TEST',
                                      config_file=self.SFP.dataset_config,
                                      annotations=self.SFP.annotations)
            project_dataset.update_annotations_with_slidenames(self.SFP.annotations)
            loaded_slides = project_dataset.get_slides()
            for slide in [row[0] for row in self.config.annotations[1:]]:
                if slide not in loaded_slides:
                    print()
                    log.error(f"Failed to correctly associate slide names ({slide}); please see annotations file below.")
                    with open(outfile, 'r') as ann_read:
                        print(ann_read.read())
                    test.fail()
                    return

    def setup_hp(self, model_type):
        # Remove old batch train file
        try:
            os.remove(self.SFP.batch_train_config)
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
                                            L2_weight=0.1,
                                            dropout=0.1,
                                            balanced_training=["BALANCE_BY_PATIENT"],
                                            balanced_validation=["NO_BALANCE"],
                                            augment=[True],
                                            label='TEST',
                                            filename=self.SFP.batch_train_config)

        # Create single hyperparameter combination
        hp = sf.model.HyperParameters(finetune_epochs=1,
                                      toplayer_epochs=0,
                                      model='InceptionV3',
                                      pooling='max',
                                      loss=loss,
                                      learning_rate=0.001,
                                      batch_size=64,
                                      hidden_layers=1,
                                      optimizer='Adam',
                                      early_stop=False,
                                      dropout=0.1,
                                      L2_weight=0.1,
                                      early_stop_patience=0,
                                      balanced_training='BALANCE_BY_PATIENT',
                                      balanced_validation='NO_BALANCE',
                                      augment=True)
        return hp

    def test_extraction(self, **kwargs):
        # Test tile extraction, default parameters, for regular slides
        with TaskWrapper("Testing slide extraction...") as test:
            self.SFP.extract_tiles(tile_px=299, tile_um=302, buffer=self.buffer, dataset=['TEST'], skip_missing_roi=False, **kwargs)

    def test_realtime_normalizer(self, **train_kwargs):
        with TaskWrapper("Testing realtime normalization, using Macenko...") as test:
            hp = self.setup_hp('categorical')
            self.SFP.train(outcome_label_headers='category1',
                           validation_settings=sf.project.get_validation_settings(k_fold_iter=1),
                           normalizer='reinhard',
                           normalizer_strategy='realtime',
                           steps_per_epoch_override=5,
                           **train_kwargs)

    def test_training(self, categorical=True, linear=True, multi_input=True, **train_kwargs):
        val_settings = sf.project.get_validation_settings(k_fold_iter=1, validate_on_batch=50)
        if categorical:
            # Test categorical outcome
            with TaskWrapper("Training to single categorical outcome from hyperparameters...") as test:
                hp = self.setup_hp('categorical')
                results_dict = self.SFP.train(model_names = 'manual_hp',
                                              outcome_label_headers='category1',
                                              hyperparameters=hp,
                                              validation_settings=val_settings,
                                              steps_per_epoch_override=5,
                                              **train_kwargs)

                if not results_dict or 'history' not in results_dict[results_dict.keys()[0]]:
                    print("\tKeras results object not received from training")
                    test.fail()

            # Test multiple sequential categorical outcome models
            with TaskWrapper("Training to multiple outcomes...") as test:
                self.SFP.train(outcome_label_headers=['category1', 'category2'],
                               validation_settings=val_settings,
                               steps_per_epoch_override=5,
                               **train_kwargs)

        if linear:
            # Test multiple linear outcome
            with TaskWrapper("Training to multiple linear outcomes...") as test:
                hp = self.setup_hp('linear')
                self.SFP.train(outcome_label_headers=['linear1', 'linear2'],
                               validation_settings=val_settings,
                               steps_per_epoch_override=5,
                               **train_kwargs)

        if multi_input:
            with TaskWrapper("Training with multiple inputs (image + annotation feature)...") as test:
                hp = self.setup_hp('categorical')
                self.SFP.train(model_names='multi_input',
                               outcome_label_headers='category1',
                               input_header='category2',
                               hyperparameters=hp,
                               validation_settings=val_settings,
                               steps_per_epoch_override=5,
                               **train_kwargs)

    def test_training_performance(self, **train_kwargs):
        with TaskWrapper("Testing performance of training (single categorical outcome)...") as test:
            hp = self.setup_hp('categorical')
            hp.finetune_epochs = [1,3]
            results_dict = self.SFP.train(model_names='performance',
                                          outcome_label_headers='category1',
                                          hyperparameters=hp,
                                          validation_settings=sf.project.get_validation_settings(k_fold_iter=1),
                                          save_predictions=True,
                                          **train_kwargs)

    def test_evaluation(self, **eval_kwargs):
        with TaskWrapper("Testing evaluation of single categorical outcome model...") as test:
            self.SFP.evaluate(model=self.config.SAVED_MODEL,
                              outcome_label_headers='category1',
                              histogram=True,
                              save_predictions=True,
                              **eval_kwargs)

        with TaskWrapper("Testing evaluation of multi-categorical outcome model...") as test:
            self.SFP.evaluate(model=join(self.SFP.models_dir, 'category1-category2-TEST-HPSweep0-kfold1', 'trained_model_epoch1'),
                              outcome_label_headers=['category1', 'category2'],
                              histogram=True,
                              save_predictions=True,
                              **eval_kwargs)

        with TaskWrapper("Testing evaluation of multi-linear outcome model...") as test:
            self.SFP.evaluate(model=join(self.SFP.models_dir, 'linear1-linear2-TEST-HPSweep0-kfold1', 'trained_model_epoch1'),
                              outcome_label_headers=['linear1', 'linear2'],
                              histogram=True,
                              save_predictions=True,
                              **eval_kwargs)

        with TaskWrapper("Testing evaluation of multi-input (image + annotation feature) model...") as test:
            self.SFP.evaluate(model=join(self.SFP.models_dir, 'category1-multi_input-kfold1', 'trained_model_epoch1'),
                              outcome_label_headers='category1',
                              input_header='category2',
                              **eval_kwargs)

        #print("Testing that evaluation matches known baseline...")
        #self.SFP.evaluate(outcome_label_headers='category1', model=REFERENCE_MODEL, filters={'submitter_id': '234839'})
        # Code to lookup excel sheet of predictions and verify they match known baseline

    def test_heatmap(self, slide='auto', **heatmap_kwargs):
        with TaskWrapper("Testing heatmap generation...") as test:
            if slide.lower() == 'auto':
                sfdataset = self.SFP.get_dataset()
                slide_paths = sfdataset.get_slide_paths(dataset='TEST')
                patient_name = sfutil.path_to_name(slide_paths[0])
            self.SFP.generate_heatmaps(self.config.SAVED_MODEL, filters={sfutil.TCGA.patient: [patient_name]}, **heatmap_kwargs)

    def test_mosaic(self, **mosaic_kwargs):
        with TaskWrapper("Testing mosaic generation...") as test:
            self.SFP.generate_mosaic(self.config.SAVED_MODEL, mosaic_filename="mosaic_test.png", **mosaic_kwargs)

    def test_activations(self, **act_kwargs):
        with TaskWrapper("Testing activations analytics...") as test:
            AV = self.SFP.generate_activations(model=self.config.SAVED_MODEL,
                                               outcome_label_headers='category1',
                                               focus_nodes=[0],
                                               **act_kwargs)
            AV.generate_box_plots()
            umap = TFRecordMap.from_activations(AV)
            umap.save_2d_plot(join(self.SFP.root, 'stats', '2d_umap.png'))
            top_nodes = AV.get_top_nodes_by_slide()
            for node in top_nodes[:5]:
                umap.save_3d_plot(node=node, filename=join(self.SFP.root, 'stats', f'3d_node{node}.png'))

    def test_predict_wsi(self):
        with TaskWrapper("Testing WSI prediction...") as test:
            sfdataset = self.SFP.get_dataset()
            slide_paths = sfdataset.get_slide_paths(dataset='TEST')
            patient_name = sfutil.path_to_name(slide_paths[0])
            self.SFP.predict_wsi(self.config.SAVED_MODEL,
                                 299,
                                 302,
                                 join(self.SFP.root, 'wsi'),
                                 filters={sfutil.TCGA.patient: [patient_name]})

    def test(self, extract=True, train=True, normalizer=True, train_performance=True,
                evaluate=True,heatmap=True, mosaic=True, activations=True, predict_wsi=True):
        '''Perform and report results of all available testing.'''
        if extract: 			self.test_extraction()
        if train:				self.test_training()
        if normalizer:			self.test_realtime_normalizer()
        if train_performance: 	self.test_training_performance()
        if evaluate:			self.test_evaluation()
        if heatmap:				self.test_heatmap()
        if mosaic:				self.test_mosaic()
        if activations:			self.test_activations()
        if predict_wsi:			self.test_predict_wsi()
