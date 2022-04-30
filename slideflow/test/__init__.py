import os
import csv
import time
import logging
import multiprocessing
import unittest
from os.path import join, exists
from tqdm import tqdm
from PIL import Image
from typing import Any, Tuple, Optional

import slideflow as sf
from slideflow.util import log, Path
from slideflow.util import colors as col
from slideflow.stats import SlideMap
from slideflow import errors
from slideflow.test import dataset_test
from slideflow.test.utils import TaskWrapper, TestConfig, _assert_valid_results


# ---------------------------------------

def _prediction_tester(project: sf.Project, verbosity: int, **kwargs) -> None:
    logging.getLogger("slideflow").setLevel(verbosity)
    project.predict(**kwargs)


def prediction_tester(project: sf.Project, **kwargs) -> None:
    """Prediction testing must happen in an isolated to free GPU memory
    after evaluation is done, due to the need for testing multiple models."""

    ctx = multiprocessing.get_context('spawn')
    verbosity = logging.getLogger('slideflow').level
    process = ctx.Process(
        target=_prediction_tester,
        args=(project, verbosity),
        kwargs=kwargs
    )
    process.start()
    process.join()


# ---------------------------------------

def _evaluation_tester(project: sf.Project, verbosity: int, **kwargs) -> None:
    logging.getLogger("slideflow").setLevel(verbosity)
    project.evaluate(**kwargs)


def evaluation_tester(project: sf.Project, **kwargs) -> None:
    """Evaluation testing must happen in an isolated to free GPU memory
    after evaluation is done, due to the need for testing multiple models."""

    ctx = multiprocessing.get_context('spawn')
    verbosity = logging.getLogger('slideflow').level
    process = ctx.Process(
        target=_evaluation_tester,
        args=(project, verbosity),
        kwargs=kwargs
    )
    process.start()
    process.join()


# -----------------------------------------

def _activations_tester(
    project: sf.Project,
    model: Path,
    verbosity: int,
    **kwargs
) -> None:
    logging.getLogger("slideflow").setLevel(verbosity)
    TaskWrapper.VERBOSITY = verbosity
    with TaskWrapper("Testing activations..."):
        dataset = project.dataset(71, 1208)
        test_slide = dataset.slides()[0]

        df = project.generate_features(
            model=model,
            outcomes='category1',
            **kwargs
        )
        act_by_cat = df.activations_by_category(0).values()
        assert df.num_features == 1280  # mobilenet_v2
        assert df.num_logits == 2
        assert len(df.activations) == len(dataset.tfrecords())
        assert len(df.locations) == len(df.activations) == len(df.logits)
        assert all([
            len(df.activations[s]) == len(df.logits[s]) == len(df.locations[s])
            for s in df.activations
        ])
        assert len(df.activations_by_category(0)) == 2
        assert (sum([len(a) for a in act_by_cat])
                == sum([len(df.activations[s]) for s in df.slides]))
        lm = df.logits_mean()
        l_perc = df.logits_percent()
        l_pred = df.logits_predict()
        assert len(lm) == len(df.activations)
        assert len(lm[test_slide]) == df.num_logits
        assert len(l_perc) == len(df.activations)
        assert len(l_perc[test_slide]) == df.num_logits
        assert len(l_pred) == len(df.activations)

        umap = SlideMap.from_features(df)
        if not exists(join(project.root, 'stats')):
            os.makedirs(join(project.root, 'stats'))
        umap.save(join(project.root, 'stats', '2d_umap.png'))
        tile_stats, pt_stats, cat_stats = df.stats()
        top_features_by_tile = sorted(
            range(df.num_features),
            key=lambda f: tile_stats[f]['p']
        )
        for feature in top_features_by_tile[:5]:
            umap.save_3d_plot(
                join(project.root, 'stats', f'3d_feature{feature}.png'),
                feature=feature
            )
        df.box_plots(
            top_features_by_tile[:5],
            join(project.root, 'box_plots')
        )

    with TaskWrapper("Testing mosaic generation..."):
        mosaic = project.generate_mosaic(df)
        mosaic.save(join(project.root, "mosaic_test.png"), resolution='low')


def activations_tester(project: sf.Project, model: Path, **kwargs) -> None:
    ctx = multiprocessing.get_context('spawn')
    verbosity = logging.getLogger('slideflow').level
    process = ctx.Process(
        target=_activations_tester,
        args=(project, model, verbosity),
        kwargs=kwargs
    )
    process.start()
    process.join()


# -----------------------------------------

def _wsi_prediction_tester(
    project: sf.Project,
    model: Path,
    verbosity: int
) -> None:
    logging.getLogger("slideflow").setLevel(verbosity)
    with TaskWrapper("Testing WSI prediction..."):
        dataset = project.dataset()
        slide_paths = dataset.slide_paths(source='TEST')
        patient_name = sf.util.path_to_name(slide_paths[0])
        project.predict_wsi(
            model,
            join(project.root, 'wsi'),
            filters={'patient': [patient_name]}
        )


def wsi_prediction_tester(project: sf.Project, model: Path) -> None:
    ctx = multiprocessing.get_context('spawn')
    verbosity = logging.getLogger('slideflow').level
    process = ctx.Process(
        target=_wsi_prediction_tester,
        args=(project, model, verbosity)
    )
    process.start()
    process.join()


def _clam_feature_generator(
    project: sf.Project,
    model: Path,
    verbosity: int
) -> None:
    logging.getLogger("slideflow").setLevel(verbosity)
    outdir = join(project.root, 'clam')
    project.generate_features_for_clam(
        model,
        outdir=outdir,
        force_regenerate=True
    )


def clam_feature_generator(project: sf.Project, model: Path) -> None:
    ctx = multiprocessing.get_context('spawn')
    verbosity = logging.getLogger('slideflow').level
    process = ctx.Process(
        target=_clam_feature_generator,
        args=(project, model, verbosity)
    )
    process.start()
    process.join()


# ----------------------------------------

def reader_tester(project: sf.Project, verbosity: int) -> None:
    dataset = project.dataset(71, 1208)
    tfrecords = dataset.tfrecords()
    batch_size = 128
    assert len(tfrecords)

    # Torch backend
    torch_results = []
    torch_dts = dataset.torch(
        labels=None,
        batch_size=batch_size,
        infinite=False,
        augment=False,
        standardize=False,
        num_workers=6,
        pin_memory=False
    )
    if verbosity < logging.WARNING:
        torch_dts = tqdm(
            torch_dts,
            leave=False,
            ncols=80,
            unit_scale=batch_size,
            total=dataset.num_tiles // batch_size
        )
    for images, labels in torch_dts:
        torch_results += [
            hash(str(img.numpy().transpose(1, 2, 0)))  # CWH -> WHC
            for img in images
        ]
    if verbosity < logging.WARNING:
        torch_dts.close()  # type: ignore
    torch_results = sorted(torch_results)

    # Tensorflow backend
    tf_results = []
    tf_dts = dataset.tensorflow(
        labels=None,
        batch_size=batch_size,
        infinite=False,
        augment=False,
        standardize=False
    )
    if verbosity < logging.WARNING:
        tf_dts = tqdm(
            tf_dts,
            leave=False,
            ncols=80,
            unit_scale=batch_size,
            total=dataset.num_tiles // batch_size
        )
    for images, labels in tf_dts:
        tf_results += [hash(str(img.numpy())) for img in images]
    if verbosity < logging.WARNING:
        tf_dts.close()
    tf_results = sorted(tf_results)

    assert len(torch_results) == len(tf_results) == dataset.num_tiles
    assert torch_results == tf_results


# -----------------------------------------------

def normalizer_tester(
    project: sf.Project,
    args: Tuple,
    single: bool,
    multi: bool,
    verbosity: Optional[int] = None
) -> None:
    if verbosity is not None:
        logging.getLogger("slideflow").setLevel(verbosity)
    if not len(args):
        methods = sf.norm.StainNormalizer.normalizers
    else:
        methods = args  # type: ignore
    dataset = project.dataset(71, 1208)
    prefix = '\r\033[kTesting '
    v = '(vectorized)'

    if single:
        with TaskWrapper("Testing normalization single-thread throughput..."):
            dts_kw = {'standardize': False, 'infinite': True}
            if sf.backend() == 'tensorflow':
                dts = dataset.tensorflow(None, None, **dts_kw)
                raw_img = next(iter(dts))[0].numpy()
            elif sf.backend() == 'torch':
                dts = dataset.torch(None, None, **dts_kw)
                raw_img = next(iter(dts))[0].permute(1, 2, 0).numpy()
            Image.fromarray(raw_img).save(join(project.root, 'raw_img.png'))
            for method in methods:
                gen_norm = sf.norm.autoselect(method, prefer_vectorized=False)
                vec_norm = sf.norm.autoselect(method, prefer_vectorized=True)
                st_msg = col.yellow('SINGLE-thread')
                print(f"{prefix}{method} [{st_msg}]...", end="")

                # Save example image
                img = Image.fromarray(gen_norm.rgb_to_rgb(raw_img))
                img.save(join(project.root, f'{method}.png'))

                gen_tpt = test_throughput(dts, gen_norm)
                dur = col.blue(f"[{gen_tpt:.1f} img/s]")
                print(f"{prefix}{method} [{st_msg}]... DONE " + dur)
                if type(vec_norm) != type(gen_norm):
                    print(f"{prefix}{method} {v} [{st_msg}]...", end="")

                    # Save example image
                    img = Image.fromarray(vec_norm.rgb_to_rgb(raw_img))
                    img.save(join(project.root, f'{method}_vectorized.png'))

                    vec_tpt = test_throughput(dts, vec_norm)
                    dur = col.blue(f"[{vec_tpt:.1f} img/s]")
                    print(f"{prefix}{method} {v} [{st_msg}]... DONE {dur}")

    if multi:
        with TaskWrapper("Testing normalization multi-thread throughput..."):
            for method in methods:
                gen_norm = sf.norm.autoselect(method, prefer_vectorized=False)
                vec_norm = sf.norm.autoselect(method, prefer_vectorized=True)
                mt_msg = col.purple('MULTI-thread')
                print(f"{prefix}{method} [{mt_msg}]...", end="")
                gen_tpt = test_multithread_throughput(dataset, gen_norm)
                dur = col.blue(f"[{gen_tpt:.1f} img/s]")
                print(f"{prefix}{method} [{mt_msg}]... DONE " + dur)
                if type(vec_norm) != type(gen_norm):
                    print(f"{prefix}{method} {v} [{mt_msg}]...", end="")
                    vec_tpt = test_multithread_throughput(dataset, vec_norm)
                    dur = col.blue(f"[{vec_tpt:.1f} img/s]")
                    print(f"{prefix}{method} {v} [{mt_msg}]... DONE " + dur)


# -----------------------------------------------

def test_throughput(
    dts: Any,
    normalizer: sf.norm.StainNormalizer = None,
    s: int = 5,
    step_size: int = 1
) -> float:
    '''Returns images / sec'''
    start = -1  # type: float
    count = 0
    total_time = 0  # type: float
    for img, slide in dts:
        if sf.backend() == 'torch':
            if len(img.shape) == 3:
                img = img.permute(1, 2, 0)
            else:
                img = img.permute(0, 2, 3, 1)
        img = img.numpy()
        if normalizer is not None:
            normalizer.rgb_to_rgb(img)
        if start == -1:
            start = time.time()
        else:
            count += step_size
        if time.time() - start > s:
            total_time = count / (time.time() - start)
            break
    return total_time


def test_multithread_throughput(
    dataset: Any,
    normalizer: sf.norm.StainNormalizer,
    s: int = 5,
    batch_size: int = 32
) -> float:
    if sf.backend() == 'tensorflow':
        dts = dataset.tensorflow(
            None,
            batch_size,
            standardize=False,
            infinite=True,
            normalizer=normalizer
        )
    elif sf.backend() == 'torch':
        dts = dataset.torch(
            None,
            batch_size,
            standardize=False,
            infinite=True,
            normalizer=normalizer,
        )
    step_size = 1 if batch_size is None else batch_size
    return test_throughput(dts, step_size=step_size, s=s)


# -----------------------------------------------


class TestSuite:
    '''Class to supervise standardized testing of slideflow pipeline.'''
    def __init__(
        self,
        root: str,
        slides: Optional[str],
        buffer: Optional[Path] = None,
        verbosity: int = logging.WARNING,
        reset: bool = False
    ) -> None:
        '''Initialize testing models.'''

        if slides is None:
            print(col.yellow("Path to slides not provided, unable to perform"
                             " functional tests."))
            self.project = None
        else:
            # Set logging level
            logging.getLogger("slideflow").setLevel(verbosity)
            # Set the tensorflow logger
            if logging.getLogger('slideflow').level == logging.DEBUG:
                logging.getLogger('tensorflow').setLevel(logging.DEBUG)
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
            else:
                logging.getLogger('tensorflow').setLevel(logging.ERROR)
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            self.verbosity = verbosity
            TaskWrapper.VERBOSITY = verbosity

            # Configure testing environment
            self.test_root = root
            self.project_root = join(root, 'project')
            self.slides_root = slides
            print(f'Setting up test project at {col.green(root)}')
            print(f'Testing using slides from {col.green(slides)}')
            self.config = TestConfig(root, slides=slides)
            self.project = self.config.create_project(self.project_root,
                                                    overwrite=reset)

            # Check if GPU available
            if sf.backend() == 'tensorflow':
                import tensorflow as tf
                if not tf.config.list_physical_devices('GPU'):
                    log.error("GPU unavailable - tests may fail.")
            elif sf.backend() == 'torch':
                import torch
                if not torch.cuda.is_available():
                    log.error("GPU unavailable - tests may fail.")
            else:
                raise errors.BackendError(
                    f"Unknown backend {sf.backend()} "
                    "Valid backends: 'tensorflow' or 'torch'"
                )

            # Configure datasets (input)
            self.buffer = buffer

            # Rebuild tfrecord indices
            self.project.dataset(71, 1208).build_index(True)

    def _get_model(self, name: str, epoch: int = 1) -> str:
        assert self.project is not None
        prev_run_dirs = [
            x for x in os.listdir(self.project.models_dir)
            if os.path.isdir(join(self.project.models_dir, x))
        ]
        for run in sorted(prev_run_dirs, reverse=True):
            if run[6:] == name:
                return join(
                    self.project.models_dir,
                    run,
                    f'{name}_epoch{epoch}'
                )
        raise OSError(f"Unable to find trained model {name}")

    def setup_hp(
        self,
        model_type: str,
        sweep: bool = False,
        normalizer: Optional[str] = None,
        uq: bool = False
    ) -> sf.ModelParams:
        """Set up hyperparameters."""

        assert self.project is not None
        if model_type == 'categorical':
            loss = ('sparse_categorical_crossentropy'
                    if sf.backend() == 'tensorflow'
                    else 'CrossEntropy')
        elif model_type == 'linear':
            loss = ('mean_squared_error'
                    if sf.backend() == 'tensorflow'
                    else 'MSE')
        elif model_type == 'cph':
            loss = ('negative_log_likelihood'
                    if sf.backend() == 'tensorflow'
                    else 'NLL')

        # Create batch train file
        if sweep:
            self.project.create_hp_sweep(
                tile_px=71,
                tile_um=1208,
                epochs=[1, 3],
                toplayer_epochs=[0],
                model=["mobilenet_v2"],
                loss=[loss],
                learning_rate=[0.001],
                batch_size=[16],
                hidden_layers=[0, 1],
                optimizer=["Adam"],
                early_stop=[False],
                early_stop_patience=[15],
                early_stop_method='loss',
                hidden_layer_width=500,
                trainable_layers=0,
                dropout=0.1,
                training_balance=["category"],
                validation_balance=["none"],
                augment=[True],
                normalizer=normalizer,
                label='TEST',
                uq=uq,
                filename='sweep.json'
            )

        # Create single hyperparameter combination
        hp = sf.model.ModelParams(
            tile_px=71,
            tile_um=1208,
            epochs=1,
            toplayer_epochs=0,
            model="mobilenet_v2",
            pooling='max',
            loss=loss,
            learning_rate=0.001,
            batch_size=16,
            hidden_layers=1,
            optimizer='Adam',
            early_stop=False,
            dropout=0.1,
            early_stop_patience=0,
            training_balance='patient',
            validation_balance='none',
            uq=uq,
            augment=True
        )
        return hp

    def test_extraction(self, enable_downsample: bool = True, **kwargs) -> None:
        # Test tile extraction, default parameters, for regular slides
        assert self.project is not None
        with TaskWrapper("Testing slide extraction..."):
            self.project.extract_tiles(
                tile_px=71,
                tile_um=1208,
                buffer=self.buffer,
                source=['TEST'],
                roi_method='ignore',
                skip_extracted=False,
                img_format='png',
                enable_downsample=enable_downsample,
                **kwargs
            )

    def test_normalizers(
        self,
        *args,
        single: bool = True,
        multi: bool = True,
    ) -> None:
        # Tests throughput of normalizers, save a single example image for each
        assert self.project is not None
        verbosity = logging.getLogger('slideflow').level
        ctx = multiprocessing.get_context('spawn')
        process = ctx.Process(
            target=normalizer_tester,
            args=(self.project, args, single, multi, verbosity)
        )
        process.start()
        process.join()

    def test_readers(self) -> None:
        assert self.project is not None
        with TaskWrapper("Testing torch and tensorflow readers...") as test:
            try:
                import tensorflow as tf  # noqa F401
                import torch  # noqa F401
            except ImportError:
                log.warning(
                    "Can't import tensorflow and pytorch, skipping TFRecord test"
                )
                test.skip()
                return

            ctx = multiprocessing.get_context('spawn')
            process = ctx.Process(target=reader_tester, args=(self.project,
                                                            self.verbosity))
            process.start()
            process.join()

    def train_perf(self, **train_kwargs) -> None:
        assert self.project is not None
        msg = "Training single categorical outcome from HP sweep..."
        with TaskWrapper(msg) as test:
            self.setup_hp(
                'categorical',
                sweep=True,
                normalizer='reinhard_fast',
                uq=False
            )
            results = self.project.train(
                exp_label='manual_hp',
                outcomes='category1',
                val_k=1,
                validate_on_batch=10,
                save_predictions=True,
                steps_per_epoch_override=20,
                params='sweep.json',
                pretrain=None,
                **train_kwargs
            )
            _assert_valid_results(results)

    def test_training(
        self,
        categorical: bool = True,
        uq: bool = True,
        multi_categorical: bool = True,
        linear: bool = True,
        multi_linear: bool = True,
        multi_input: bool = True,
        cph: bool = True,
        multi_cph: bool = True,
        **train_kwargs
    ) -> None:
        assert self.project is not None
        # Disable checkpoints for tensorflow backend, to save disk space
        if (sf.backend() == 'tensorflow'
           and 'save_checkpoints' not in train_kwargs):
            train_kwargs['save_checkpoints'] = False

        if categorical:
            # Test categorical outcome
            self.train_perf(**train_kwargs)

        if uq:
            # Test categorical outcome with UQ
            msg = "Training single categorical outcome with UQ..."
            with TaskWrapper(msg) as test:
                hp = self.setup_hp('categorical', sweep=False, uq=True)
                results = self.project.train(
                    exp_label='UQ',
                    outcomes='category1',
                    val_k=1,
                    params=hp,
                    validate_on_batch=10,
                    steps_per_epoch_override=20,
                    save_predictions=True,
                    pretrain=None,
                    **train_kwargs
                )
                _assert_valid_results(results)

        if multi_categorical:
            # Test multiple sequential categorical outcome models
            with TaskWrapper("Training to multiple outcomes...") as test:
                results = self.project.train(
                    outcomes=['category1', 'category2'],
                    val_k=1,
                    params=self.setup_hp('categorical'),
                    validate_on_batch=10,
                    steps_per_epoch_override=20,
                    save_predictions=True,
                    pretrain=None,
                    **train_kwargs
                )
                _assert_valid_results(results)

        if linear:
            # Test multiple linear outcome
            with TaskWrapper("Training multiple linear outcomes...") as test:
                results = self.project.train(
                    outcomes=['linear1', 'linear2'],
                    val_k=1,
                    params=self.setup_hp('linear'),
                    validate_on_batch=10,
                    steps_per_epoch_override=20,
                    save_predictions=True,
                    pretrain=None,
                    **train_kwargs
                )
                _assert_valid_results(results)

        if multi_linear:
            # Test multiple linear outcome
            with TaskWrapper("Training multiple linear outcomes...") as test:
                results = self.project.train(
                    outcomes=['linear1'],
                    val_k=1,
                    params=self.setup_hp('linear'),
                    validate_on_batch=10,
                    steps_per_epoch_override=20,
                    save_predictions=True,
                    pretrain=None,
                    **train_kwargs
                )
                _assert_valid_results(results)

        if multi_input:
            msg = 'Training with multiple inputs (image + slide feature)...'
            with TaskWrapper(msg) as test:
                results = self.project.train(
                    exp_label='multi_input',
                    outcomes='category1',
                    input_header='category2',
                    params=self.setup_hp('categorical'),
                    val_k=1,
                    validate_on_batch=10,
                    steps_per_epoch_override=20,
                    save_predictions=True,
                    pretrain=None,
                    **train_kwargs
                )
                _assert_valid_results(results)

        if cph:
            with TaskWrapper("Training a CPH model...") as test:
                if sf.backend() == 'tensorflow':
                    results = self.project.train(
                        exp_label='cph',
                        outcomes='time',
                        input_header='event',
                        params=self.setup_hp('cph'),
                        val_k=1,
                        validate_on_batch=10,
                        steps_per_epoch_override=20,
                        save_predictions=True,
                        pretrain=None,
                        **train_kwargs
                    )
                    _assert_valid_results(results)
                else:
                    test.skip()

        if multi_cph:
            with TaskWrapper("Training a multi-input CPH model...") as test:
                if sf.backend() == 'tensorflow':
                    results = self.project.train(
                        exp_label='multi_cph',
                        outcomes='time',
                        input_header=['event', 'category1'],
                        params=self.setup_hp('cph'),
                        val_k=1,
                        validate_on_batch=10,
                        steps_per_epoch_override=20,
                        save_predictions=True,
                        pretrain=None,
                        **train_kwargs
                    )
                    _assert_valid_results(results)
                else:
                    test.skip()
        else:
            print("Skipping CPH model testing [current backend is Pytorch]")

    def test_prediction(self, **predict_kwargs) -> None:
        assert self.project is not None
        model = self._get_model('category1-manual_hp-TEST-HPSweep0-kfold1')

        with TaskWrapper("Testing categorical model predictions..."):
            prediction_tester(
                project=self.project,
                model=model,
                **predict_kwargs
            )

    def test_evaluation(self, **eval_kwargs) -> None:
        assert self.project is not None
        multi_cat_model = self._get_model('category1-category2-HP0-kfold1')
        multi_lin_model = self._get_model('linear1-linear2-HP0-kfold1')
        multi_inp_model = self._get_model('category1-multi_input-HP0-kfold1')
        f_model = self._get_model('category1-manual_hp-TEST-HPSweep0-kfold1')

        # Performs evaluation in isolated thread to avoid OOM errors
        # with sequential model loading/testing
        with TaskWrapper("Testing categorical model evaluation..."):
            evaluation_tester(
                project=self.project,
                model=f_model,
                outcomes='category1',
                histogram=True,
                save_predictions=True,
                **eval_kwargs
            )

        with TaskWrapper("Testing categorical UQ model evaluation...") as test:
            uq_model = self._get_model('category1-UQ-HP0-kfold1')
            evaluation_tester(
                project=self.project,
                model=uq_model,
                outcomes='category1',
                histogram=True,
                save_predictions=True,
                **eval_kwargs
            )

        with TaskWrapper("Testing multi-categorical model evaluation..."):
            evaluation_tester(
                project=self.project,
                model=multi_cat_model,
                outcomes=['category1', 'category2'],
                histogram=True,
                save_predictions=True,
                **eval_kwargs
            )

        with TaskWrapper("Testing multi-linear model evaluation..."):
            evaluation_tester(
                project=self.project,
                model=multi_lin_model,
                outcomes=['linear1', 'linear2'],
                histogram=True,
                save_predictions=True,
                **eval_kwargs
            )

        with TaskWrapper("Testing multi-input model evaluation..."):
            evaluation_tester(
                project=self.project,
                model=multi_inp_model,
                outcomes='category1',
                input_header='category2',
                **eval_kwargs
            )

        with TaskWrapper("Testing CPH model evaluation...") as test:
            if sf.backend() == 'tensorflow':
                cph_model = self._get_model('time-cph-HP0-kfold1')
                evaluation_tester(
                    project=self.project,
                    model=cph_model,
                    outcomes='time',
                    input_header='event',
                    **eval_kwargs
                )
            else:
                test.skip()

    def test_heatmap(self, slide: str = 'auto', **heatmap_kwargs) -> None:
        assert self.project is not None
        model = self._get_model('category1-manual_hp-TEST-HPSweep0-kfold1')
        assert exists(model), "Model has not yet been trained."

        with TaskWrapper("Testing heatmap generation..."):
            if slide.lower() == 'auto':
                dataset = self.project.dataset()
                slide_paths = dataset.slide_paths(source='TEST')
                patient_name = sf.util.path_to_name(slide_paths[0])
            self.project.generate_heatmaps(
                model,
                filters={'patient': [patient_name]},
                roi_method='ignore',
                **heatmap_kwargs
            )

    def test_activations_and_mosaic(self, **act_kwargs) -> None:
        assert self.project is not None
        model = self._get_model('category1-manual_hp-TEST-HPSweep0-kfold1')
        assert exists(model), "Model has not yet been trained."
        activations_tester(project=self.project, model=model)

    def test_predict_wsi(self) -> None:
        assert self.project is not None
        model = self._get_model('category1-manual_hp-TEST-HPSweep0-kfold1')
        assert exists(model), "Model has not yet been trained."
        wsi_prediction_tester(self.project, model)

    def test_clam(self) -> None:
        assert self.project is not None
        model = self._get_model('category1-manual_hp-TEST-HPSweep0-kfold1')
        assert exists(model), "Model has not yet been trained."

        try:
            skip_test = False
            import torch  # noqa F401
        except ImportError:
            log.warning("Unable to import pytorch, skipping CLAM test")
            skip_test = True

        with TaskWrapper("Testing CLAM feature export...") as test:
            if skip_test:
                test.skip()
            else:
                clam_feature_generator(self.project, model)

        with TaskWrapper("Testing CLAM training...") as test:
            if skip_test:
                test.skip()
            else:
                dataset = self.project.dataset(71, 1208)
                self.project.train_clam(
                    'TEST_CLAM',
                    join(self.project.root, 'clam'),
                    'category1',
                    dataset
                )

    def test(
        self,
        extract: bool = True,
        reader: bool = True,
        train: bool = True,
        normalizer: bool = True,
        evaluate: bool = True,
        predict: bool = True,
        heatmap: bool = True,
        activations: bool = True,
        predict_wsi: bool = True,
        clam: bool = True
    ) -> None:
        '''Perform and report results of all available testing.'''

        start = time.time()
        self.unittests()
        if self.project is None:
            print(col.yellow("Slides not provided; unable to perform "
                             "functional testing."))
        else:
            if extract:
                self.test_extraction()
            if reader:
                self.test_readers()
            if train:
                self.test_training()
            if normalizer:
                self.test_normalizers()
            if evaluate:
                self.test_evaluation()
            if predict:
                self.test_prediction()
            if heatmap:
                self.test_heatmap()
            if activations:
                self.test_activations_and_mosaic()
            if predict_wsi:
                self.test_predict_wsi()
            if clam:
                self.test_clam()
        end = time.time()
        m, s = divmod(end-start, 60)
        print(f'Tests complete. Time: {int(m)} min, {s:.2f} sec')

    def unittests(self) -> None:
        print("Running unit tests...")
        runner = unittest.TextTestRunner()
        all_tests = [
            unittest.TestLoader().loadTestsFromModule(module)
            for module in (dataset_test, )
        ]
        suite = unittest.TestSuite(all_tests)
        runner.run(suite)