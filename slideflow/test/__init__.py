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
import multiprocessing

import slideflow as sf
from slideflow.dataset import Dataset
from slideflow.util import log, ProgressBar
from slideflow.util.spinner import Spinner
from slideflow.statistics import SlideMap
from os.path import join
from functools import wraps
from tqdm import tqdm

def get_tcga_slides():
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
    slides = [sf.util.path_to_name(f) for f in os.listdir(slides_path)
                                        if sf.util.path_to_ext(f).lower() in sf.util.SUPPORTED_FORMATS][:10]
    if not slides:
        raise OSError(f'No slides found at {slides_path}')
    annotations = [[sf.util.TCGA.patient, 'dataset', 'category1', 'category2', 'linear1', 'linear2', 'time', 'event']]
    for s, slide in enumerate(slides):
        cat1 = ['A', 'B'][s % 2]
        cat2 = ['A', 'B'][s % 2]
        lin1 = random.random()
        lin2 = random.random()
        time = random.randint(0, 100)
        event = random.choice([0, 1])
        annotations += [[slide, 'TEST', cat1, cat2, lin1, lin2, time, event]]
    return annotations

# ---------------------------------------

def _evaluation_tester(project, verbosity, **kwargs):
    logging.getLogger("slideflow").setLevel(verbosity)
    project.evaluate(**kwargs)

def evaluation_tester(project, **kwargs):
    """Evaluation testing must happen in an isolated thread in order to free GPU memory
    after evaluation has completed, due to the need for sequential testing of multiple models."""

    ctx = multiprocessing.get_context('spawn')
    verbosity = logging.getLogger('slideflow').level
    process = ctx.Process(target=_evaluation_tester, args=(project, verbosity), kwargs=kwargs)
    process.start()
    process.join()

# -----------------------------------------

def _activations_tester(project, model, verbosity, **kwargs):
    logging.getLogger("slideflow").setLevel(verbosity)
    with TaskWrapper("Testing activations...") as test:
        dataset = project.dataset(299, 302)
        test_slide = dataset.slides()[0]

        df = project.generate_features(model=model, outcome_label_headers='category1', **kwargs)
        assert df.num_features == 2048
        assert df.num_logits == 2
        assert len(df.activations) == len(dataset.tfrecords())
        assert len(df.locations) == len(df.activations) == len(df.logits)
        assert all([len(df.activations[slide]) == len(df.logits[slide]) == len(df.locations[slide]) for slide in df.activations])
        assert len(df.activations_by_category(0)) == 2
        assert sum([len(a) for a in df.activations_by_category(0).values()]) == sum([len(df.activations[s]) for s in df.slides])
        lm = df.logits_mean()
        l_perc = df.logits_percent()
        l_pred = df.logits_predict()
        assert len(lm) == len(df.activations)
        assert len(lm[test_slide]) == df.num_logits
        assert len(l_perc) == len(df.activations)
        assert len(l_perc[test_slide]) == df.num_logits
        assert len(l_pred) == len(df.activations)

        umap = SlideMap.from_features(df)
        umap.save(join(project.root, 'stats', '2d_umap.png'))
        tile_stats, pt_stats, cat_stats = df.stats()
        top_features_by_tile = sorted(range(df.num_features), key=lambda f: tile_stats[f]['p'])
        for feature in top_features_by_tile[:5]:
            umap.save_3d_plot(join(project.root, 'stats', f'3d_feature{feature}.png'), feature=feature)
        df.box_plots(top_features_by_tile[:5], join(project.root, 'box_plots'))

    with TaskWrapper("Testing mosaic generation...") as test:
        mosaic = project.generate_mosaic(df)
        mosaic.save(os.path.join(project.root, "mosaic_test.png"))

def activations_tester(project, model, **kwargs):
    ctx = multiprocessing.get_context('spawn')
    verbosity = logging.getLogger('slideflow').level
    process = ctx.Process(target=_activations_tester, args=(project, model, verbosity), kwargs=kwargs)
    process.start()
    process.join()

# -----------------------------------------

def _wsi_prediction_tester(project, model, verbosity):
    logging.getLogger("slideflow").setLevel(verbosity)
    with TaskWrapper("Testing WSI prediction...") as test:
        dataset = project.dataset()
        slide_paths = dataset.slide_paths(source='TEST')
        patient_name = sf.util.path_to_name(slide_paths[0])
        project.predict_wsi(model,
                            join(project.root, 'wsi'),
                            filters={sf.util.TCGA.patient: [patient_name]})

def wsi_prediction_tester(project, model):
    ctx = multiprocessing.get_context('spawn')
    verbosity = logging.getLogger('slideflow').level
    process = ctx.Process(target=_wsi_prediction_tester, args=(project, model, verbosity))
    process.start()
    process.join()

# -----------------------------------------

def _clam_feature_generator(project, model, verbosity):
    logging.getLogger("slideflow").setLevel(verbosity)
    outdir = join(project.root, 'clam')
    project.generate_features_for_clam(model, outdir=outdir, force_regenerate=True)

def clam_feature_generator(project, model):
    ctx = multiprocessing.get_context('spawn')
    verbosity = logging.getLogger('slideflow').level
    process = ctx.Process(target=_clam_feature_generator, args=(project, model, verbosity))
    process.start()
    process.join()

# ----------------------------------------

def reader_tester(project):
    dataset = project.project.dataset(299, 302)
    tfrecords = dataset.tfrecords()
    batch_size = 128
    assert len(tfrecords)

    with TaskWrapper("Testing torch and tensorflow readers...") as test:
        try:
            import tensorflow as tf
            import torch
        except:
            log.warning("Unable to import both tensorflow and pytorch, will skip TFRecord reading test")
            test.skip()
            return

        # Torch backend
        from slideflow.io.torch import interleave_dataloader
        torch_results = []
        torch_dts = dataset.torch(labels=None, batch_size=batch_size, infinite=False, augment=False, standardize=False, num_workers=6, pin_memory=False)
        if project.verbosity < logging.WARNING: torch_dts = tqdm(torch_dts, leave=False, ncols=80, unit_scale=batch_size, total=dataset.num_tiles // batch_size)
        for images, labels in torch_dts:
            torch_results += [hash(str(img.numpy().transpose(1, 2, 0))) for img in images] # CWH -> WHC
        torch_results = sorted(torch_results)

        # Tensorflow backend
        from slideflow.io.tensorflow import interleave
        tf_results = []
        tf_dts = dataset.tensorflow(labels=None, batch_size=batch_size, infinite=False, augment=False, standardize=False)
        if project.verbosity < logging.WARNING: tf_dts = tqdm(tf_dts, leave=False, ncols=80, unit_scale=batch_size, total=dataset.num_tiles // batch_size)
        for images, labels in tf_dts:
            tf_results += [hash(str(img.numpy())) for img in images]
        tf_results = sorted(tf_results)

        assert len(torch_results)
        assert len(torch_results) == len(tf_results) == dataset.num_tiles
        assert torch_results == tf_results

class TestConfigurator:
    def __init__(self, path, slides):
        '''Test Suite configuration.

        Args:
            path        Path to directory for test projects and data.
            slides      Specifies source of test slides. Either path to directory, or 'download' to download a set of
                            slides for testing from TCGA.  If path to directory containing slides, will use subset of
                            slides at random for testing.
        '''
        random.seed(0)
        slides_path = join(path, 'slides') if slides == 'download' else slides
        if not os.path.exists(slides_path): os.makedirs(slides_path)
        self.sources = {
            'TEST': {
                'slides':     slides_path,
                'roi':         join(path, 'roi'),
                'tiles':     join(path, 'project', 'tiles', 'TEST'),
                'tfrecords':join(path, 'project', 'tfrecords', 'TEST')
            }
        }
        self.project_settings = {
            'name': 'TEST_PROJECT',
            'annotations': './annotations.csv',
            'dataset_config': join(path, 'datasets.json'),
            'sources': ['TEST'],
            'models_dir': './models',
            'eval_dir': './eval',
            'mixed_precision': True,
        }
        if slides == 'download':
            tcga_slides = get_tcga_slides()
            with TaskWrapper("Downloading slides..."):
                existing_slides = [sf.util.path_to_name(f) for f in os.listdir(slides_path)
                                                            if sf.util.path_to_ext(f).lower() in sf.util.SUPPORTED_FORMATS]
                for slide in [s for s in tcga_slides if s not in existing_slides]:
                    download_from_tcga(uuid=tcga_slides[slide],
                                       dest=slides_path,
                                       message=f"Downloading {sf.util.green(slide)} from TCGA...")

        self.annotations = random_annotations(slides_path)
        self.reference_model = None

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
        else:
            print(self.message)
        return self

    def __exit__(self, exc_type, exc_val, exc_traceback):
        duration = time.time() - self.start
        if self.VERBOSITY >= logging.WARNING:
            self.spinner.__exit__(exc_type, exc_val, exc_traceback)
        if self.failed or (exc_type is not None or exc_val is not None or exc_traceback is not None):
            self._end_msg("FAIL", sf.util.red, f' [{duration:.0f} s]')
        elif self.skipped:
            self._end_msg("SKIPPED", sf.util.yellow, f' [{duration:.0f} s]')
        else:
            self._end_msg("DONE", sf.util.green, f' [{duration:.0f} s]')
        if self.VERBOSITY < logging.WARNING:
            print()

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
    def __init__(self, root, slides, buffer=None, num_threads=8,
                 verbosity=logging.WARNING, reset=False, gpu=None):
        '''Initialize testing models.'''

        # Set logging level
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("slideflow").setLevel(verbosity)
        self.verbosity = verbosity
        TaskWrapper.VERBOSITY = verbosity

        # Configure testing environment
        self.test_root = root
        self.project_root = join(root, 'project')
        self.config = TestConfigurator(root, slides=slides)

        if os.path.exists(join(self.project_root, 'settings.json')) and reset:
            shutil.rmtree(self.project_root)
        if os.path.exists(join(self.project_root, 'settings.json')):
            self.project = sf.Project(self.project_root,
                                  gpu=gpu)
        else:
            self.project = sf.Project(self.project_root,
                                  gpu=gpu,
                                  **self.config.project_settings)
        self.project.save()

        # Check if GPU available

        with TaskWrapper("Checking GPU availability...") as gpu_test:
            if sf.backend() == 'tensorflow':
                import tensorflow as tf
                if not tf.config.list_physical_devices('GPU'):
                    gpu_test.fail()
            elif sf.backend() == 'torch':
                import torch
                if not torch.cuda.is_available():
                    gpu_test.fail()
            else:
                raise ValueError(f"Unrecognized backend {sf.backend()} (Must be 'tensorflow' or 'torch')")

        # Configure datasets (input)
        self.configure_sources()
        self.configure_annotations()

        # Setup buffering
        self.buffer = buffer

        # Rebuild tfrecord indices
        self.project.dataset(299, 302).build_index(True)

    def _get_model(self, name, epoch=1):
        prev_run_dirs = [x for x in os.listdir(self.project.models_dir) if os.path.isdir(join(self.project.models_dir, x))]
        for run in sorted(prev_run_dirs, reverse=True):
            if run[6:] == name:
                return join(self.project.models_dir, run, f'{name}_epoch{epoch}')
        raise OSError(f"Unable to find trained model {name}")

    def configure_sources(self):
        with TaskWrapper("Dataset configuration...") as test:
            for source in self.config.sources.keys():
                self.project.add_source(source, slides=self.config.sources[source]['slides'],
                                            roi=self.config.sources[source]['roi'],
                                            tiles=self.config.sources[source]['tiles'],
                                            tfrecords=self.config.sources[source]['tfrecords'],
                                            path=self.project.dataset_config)

    def configure_annotations(self):
        with TaskWrapper("Annotation configuration...") as test:
            outfile = self.project.annotations
            with open(outfile, 'w') as csv_outfile:
                csv_writer = csv.writer(csv_outfile, delimiter=',')
                for an in self.config.annotations:
                    csv_writer.writerow(an)
            project_dataset = Dataset(tile_px=299,
                                      tile_um=302,
                                      sources='TEST',
                                      config=self.project.dataset_config,
                                      annotations=self.project.annotations)
            project_dataset.update_annotations_with_slidenames(self.project.annotations)
            loaded_slides = project_dataset.slides()
            for slide in [row[0] for row in self.config.annotations[1:]]:
                if slide not in loaded_slides:
                    print()
                    log.error(f"Failed to correctly associate slide names ({slide}); please see annotations file below.")
                    with open(outfile, 'r') as ann_read:
                        print(ann_read.read())
                    test.fail()
                    return

    def setup_hp(self, model_type, sweep=False, normalizer=None):
        import slideflow.model

        # Setup loss function
        if model_type == 'categorical':
            loss = 'sparse_categorical_crossentropy' if sf.backend() == 'tensorflow' else 'CrossEntropy'
        elif model_type == 'linear':
            loss = 'mean_squared_error' if sf.backend() == 'tensorflow' else 'MSE'
        elif model_type == 'cph':
            loss = 'negative_log_likelihood' if sf.backend() == 'tensorflow' else 'NLL'

        # Create batch train file
        if sweep:
            self.project.create_hyperparameter_sweep(tile_px=299,
                                                 tile_um=302,
                                                 epochs=[1,2,3],
                                                 toplayer_epochs=[0],
                                                 model=["xception"],
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
                                                 training_balance=["category"],
                                                 validation_balance=["none"],
                                                 augment=[True],
                                                 normalizer=normalizer,
                                                 label='TEST',
                                                 filename='sweep.json')

        # Create single hyperparameter combination
        hp = sf.model.ModelParams(epochs=1,
                                  toplayer_epochs=0,
                                  model="xception",
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
                                  training_balance='patient',
                                  validation_balance='none',
                                  augment=True)
        return hp

    def test_extraction(self, enable_downsample=True, **kwargs):
        # Test tile extraction, default parameters, for regular slides
        with TaskWrapper("Testing slide extraction...") as test:
            self.project.extract_tiles(tile_px=299,
                                   tile_um=302,
                                   buffer=self.buffer,
                                   source=['TEST'],
                                   roi_method='ignore',
                                   skip_extracted=False,
                                   img_format='png',
                                   enable_downsample=enable_downsample,
                                   **kwargs)

    def test_realtime_normalizer(self, **train_kwargs):
        with TaskWrapper("Testing realtime normalization, using Reinhard...") as test:
            self.project.train(outcome_label_headers='category1',
                           val_k=1,
                           params=self.setup_hp('categorical', normalizer='reinhard'),
                           steps_per_epoch_override=5,
                           **train_kwargs)

    def test_readers(self):
        ctx = multiprocessing.get_context('spawn')
        process = ctx.Process(target=reader_tester, args=(self,))
        process.start()
        process.join()

    def train_perf(self, **train_kwargs):
        with TaskWrapper("Training to single categorical outcome from hyperparameter sweep...") as test:
            self.setup_hp('categorical', sweep=True)
            results_dict = self.project.train(exp_label='manual_hp',
                                          outcome_label_headers='category1',
                                          val_k=1,
                                          validate_on_batch=10,
                                          save_predictions=True,
                                          steps_per_epoch_override=20,
                                          params='sweep.json',
                                          **train_kwargs)

            if not results_dict:
                log.error("Results object not received from training")
                test.fail()

    def test_training(self, categorical=True, multi_categorical=True, linear=True, multi_input=True, cph=True, multi_cph=True, **train_kwargs):
        if categorical:
            # Test categorical outcome
            self.train_perf(**train_kwargs)

        if multi_categorical:
            # Test multiple sequential categorical outcome models
            with TaskWrapper("Training to multiple outcomes...") as test:
                self.project.train(outcome_label_headers=['category1', 'category2'],
                                   val_k=1,
                                   params=self.setup_hp('categorical'),
                                   validate_on_batch=10,
                                   steps_per_epoch_override=20,
                                   **train_kwargs)

        if linear:
            # Test multiple linear outcome
            with TaskWrapper("Training to multiple linear outcomes...") as test:
                self.project.train(outcome_label_headers=['linear1', 'linear2'],
                                   val_k=1,
                                   params=self.setup_hp('linear'),
                                   validate_on_batch=10,
                                   steps_per_epoch_override=20,
                                   **train_kwargs)

        if multi_input:
            with TaskWrapper("Training with multiple inputs (image + annotation feature)...") as test:
                self.project.train(exp_label='multi_input',
                                  outcome_label_headers='category1',
                                  input_header='category2',
                                  params=self.setup_hp('categorical'),
                                  val_k=1,
                                  validate_on_batch=10,
                                  steps_per_epoch_override=20,
                                  **train_kwargs)

        if cph:
            with TaskWrapper("Training a CPH model...") as test:
                if sf.backend() == 'tensorflow':
                    self.project.train(exp_label='cph',
                                       outcome_label_headers='time',
                                       input_header='event',
                                       params=self.setup_hp('cph'),
                                       val_k=1,
                                       validate_on_batch=10,
                                       steps_per_epoch_override=20,
                                       **train_kwargs)
                else:
                    test.skip()

        if multi_cph:
            with TaskWrapper("Training a multi-input CPH model...") as test:
                if sf.backend() == 'tensorflow':
                    self.project.train(exp_label='multi_cph',
                                       outcome_label_headers='time',
                                       input_header=['event', 'category1'],
                                       params=self.setup_hp('cph'),
                                       val_k=1,
                                       validate_on_batch=10,
                                       steps_per_epoch_override=20,
                                       **train_kwargs)
                else:
                    test.skip()
        else:
            print("Skipping CPH model testing [current backend is Pytorch]")

    def test_evaluation(self, **eval_kwargs):
        multi_cat_model = self._get_model('category1-category2-HP0-kfold1')
        multi_lin_model = self._get_model('linear1-linear2-HP0-kfold1')
        multi_inp_model = self._get_model('category1-multi_input-HP0-kfold1')
        perf_model = self._get_model('category1-manual_hp-TEST-HPSweep0-kfold1')

        # Performs evaluation in isolated thread to avoid OOM errors with sequential model loading/testing
        with TaskWrapper("Testing evaluation of single categorical outcome model...") as test:
            evaluation_tester(project=self.project,
                              model=perf_model,
                              outcome_label_headers='category1',
                              histogram=True,
                              save_predictions=True,
                              **eval_kwargs)

        with TaskWrapper("Testing evaluation of multi-categorical outcome model...") as test:
            evaluation_tester(project=self.project,
                              model=multi_cat_model,
                              outcome_label_headers=['category1', 'category2'],
                              histogram=True,
                              save_predictions=True,
                              **eval_kwargs)

        with TaskWrapper("Testing evaluation of multi-linear outcome model...") as test:
            evaluation_tester(project=self.project,
                              model=multi_lin_model,
                              outcome_label_headers=['linear1', 'linear2'],
                              histogram=True,
                              save_predictions=True,
                              **eval_kwargs)

        with TaskWrapper("Testing evaluation of multi-input (image + annotation feature) model...") as test:
            evaluation_tester(project=self.project,
                              model=multi_inp_model,
                              outcome_label_headers='category1',
                              input_header='category2',
                              **eval_kwargs)

        with TaskWrapper("Testing evaluation of CPH model...") as test:
            if sf.backend() == 'tensorflow':
                cph_model = self._get_model('time-cph-HP0-kfold1')
                evaluation_tester(project=self.project,
                                model=cph_model,
                                outcome_label_headers='time',
                                input_header='event',
                                **eval_kwargs)
            else:
                test.skip()

    def test_heatmap(self, slide='auto', **heatmap_kwargs):
        perf_model = self._get_model('category1-manual_hp-TEST-HPSweep0-kfold1') #self._get_model('category1-manual_hp-HP0-kfold1')
        assert os.path.exists(perf_model)

        with TaskWrapper("Testing heatmap generation...") as test:
            if slide.lower() == 'auto':
                dataset = self.project.dataset()
                slide_paths = dataset.slide_paths(source='TEST')
                patient_name = sf.util.path_to_name(slide_paths[0])
            self.project.generate_heatmaps(perf_model,
                                       filters={sf.util.TCGA.patient: [patient_name]},
                                       roi_method='ignore',
                                       **heatmap_kwargs)

    def test_activations_and_mosaic(self, **act_kwargs):
        perf_model = self._get_model('category1-manual_hp-TEST-HPSweep0-kfold1')
        assert os.path.exists(perf_model)
        activations_tester(project=self.project, model=perf_model)

    def test_predict_wsi(self):
        perf_model = self._get_model('category1-manual_hp-TEST-HPSweep0-kfold1') #self._get_model('category1-manual_hp-HP0-kfold1')
        assert os.path.exists(perf_model)
        wsi_prediction_tester(self.project, perf_model)

    def test_clam(self):
        perf_model = self._get_model('category1-manual_hp-TEST-HPSweep0-kfold1') #self._get_model('category1-manual_hp-HP0-kfold1')
        assert os.path.exists(perf_model)

        try:
            skip_test = False
            import torch
        except:
            log.warning("Unable to import both tensorflow and pytorch, will skip TFRecord reading test")
            skip_test = True

        with TaskWrapper("Testing CLAM feature export...") as test:
            if skip_test:
                test.skip()
            else:
                clam_feature_generator(self.project, perf_model)

        with TaskWrapper("Testing CLAM training...") as test:
            if skip_test:
                test.skip()
            else:
                dataset = self.project.dataset(299, 302)
                self.project.train_clam('TEST_CLAM', join(self.project.root, 'clam'), 'category1', dataset)

    def test(self, extract=True, reader=True, train=True, normalizer=True, evaluate=True, heatmap=True,
             activations=True, predict_wsi=True, clam=True):
        '''Perform and report results of all available testing.'''

        if extract:             self.test_extraction()
        if reader:              self.test_readers()
        if train:               self.test_training()
        if normalizer:          self.test_realtime_normalizer()
        if evaluate:            self.test_evaluation()
        if heatmap:             self.test_heatmap()
        if activations:         self.test_activations_and_mosaic()
        if predict_wsi:         self.test_predict_wsi()
        if clam:                self.test_clam()
