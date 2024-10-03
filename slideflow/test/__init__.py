
import logging
import os
import time
import traceback
import unittest
from os.path import exists, join
from typing import Optional
from rich import print

import slideflow as sf
import slideflow.test.functional
from slideflow import errors
from slideflow.test import (dataset_test, slide_test, stats_test, norm_test,
                            model_test)
from slideflow.test.utils import (TaskWrapper, TestConfig,
                                  _assert_valid_results, process_isolate)
from slideflow.util import log


class TestSuite:
    """Supervises functional testing of the Slideflow pipeline."""
    def __init__(
        self,
        root: str,
        slides: Optional[str] = None,
        buffer: Optional[str] = None,
        verbosity: int = logging.WARNING,
        reset: bool = False,
        tile_px: int = 71,
    ) -> None:
        """Prepare for functional and unit testing testing. Functional tests
        require example slides.

        Args:
            root (str): Root directory of test project.
            slides (str, optional): Path to folder containing test slides.
            buffer (str, optional): Buffer slides to this location for faster
                testing. Defaults to None.
            verbosity (int, optional): Logging level. Defaults to
                logging.WARNING.
            reset (bool, optional): Reset the test project folder before
                starting. Defaults to False.

        Raises:
            errors.UnrecognizedBackendError: If the environmental variable
                SF_BACKEND is something other than  "tensorflow" or "torch".
        """
        self.tile_px = tile_px

        if slides is None:
            print("[yellow]Path to slides not provided, unable to perform"
                  " functional tests.")
            self.project = None
            return
        else:
            detected_slides = [
                sf.util.path_to_name(f)
                for f in os.listdir(slides)
                if sf.util.is_slide(join(slides, f))
            ][:10]
            if not len(detected_slides):
                print(f"[yellow]No slides found at {slides}; "
                      "unable to perform functional tests.")
                self.project = None
                return

        # --- Set up project --------------------------------------------------

        sf.setLoggingLevel(verbosity)
        if verbosity == logging.DEBUG:
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
        print(f'Setting up test project at [green]{root}')
        print(f'Testing using slides from [green]{slides}')
        self.config = TestConfig(root, slides=slides)
        self.project = self.config.create_project(
            self.project_root,
            overwrite=reset
        )

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
            raise errors.UnrecognizedBackendError

        # Configure datasets (input)
        self.buffer = buffer

        # Rebuild tfrecord indices
        self.project.dataset(self.tile_px, 604).build_index(True)

        # Set up training keyword arguments.
        self.train_kwargs = dict(
            validate_on_batch=5,
            steps_per_epoch_override=50,
            save_predictions=True
        )

    def _get_model(self, name: str, epoch: int = 1) -> str:
        assert self.project is not None
        prev_run_dirs = [
            x for x in os.listdir(self.project.models_dir)
            if os.path.isdir(join(self.project.models_dir, x))
        ]
        tail = '' if sf.backend() == 'tensorflow' else '.zip'
        for run in sorted(prev_run_dirs, reverse=True):
            if run[6:] == name:
                model_name = join(
                    self.project.models_dir,
                    run,
                    f'{name}_epoch{epoch}'+tail,
                )
                if not exists(model_name):
                    raise OSError(f"Unable to find trained model {name}")
                else:
                    return model_name
        raise OSError(f"Unable to find trained model {name}")

    def setup_hp(
        self,
        model_type: str,
        sweep: bool = False,
        normalizer: Optional[str] = 'reinhard_fast',
        uq: bool = False,
        balance: Optional[str] = 'patient',
    ) -> sf.ModelParams:
        """Set up hyperparameters.

        Args:
            model_type (str): Type of model, ('classification', 'regression, 'survival').
            sweep (bool, optional): Set up HP sweep. Defaults to False.
            normalizer (str, optional): Normalizer strategy. Defaults to None.
            uq (bool, optional): Uncertainty quantification. Defaults to False.

        Returns:
            sf.ModelParams: Hyperparameter object.
        """

        assert self.project is not None
        if model_type == 'classification':
            loss = ('sparse_categorical_crossentropy'
                    if sf.backend() == 'tensorflow'
                    else 'CrossEntropy')
        elif model_type == 'regression':
            loss = ('mean_squared_error'
                    if sf.backend() == 'tensorflow'
                    else 'MSE')
        elif model_type == 'survival':
            loss = ('negative_log_likelihood'
                    if sf.backend() == 'tensorflow'
                    else 'NLL')

        # Create batch train file
        if sweep:
            self.project.create_hp_sweep(
                tile_px=self.tile_px,
                tile_um=604,
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
        hp = sf.ModelParams(
            tile_px=self.tile_px,
            tile_um=604,
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
            l2=1e-4,
            early_stop_patience=0,
            training_balance=balance,
            validation_balance='none',
            uq=uq,
            augment=True
        )
        return hp

    def test_extraction(self, enable_downsample: bool = True, **kwargs) -> None:
        """Test tile extraction.

        Args:
            enable_downsample (bool, optional): Enable using intermediate
                downsample layers in slides. Defaults to True.
        """
        assert self.project is not None
        with TaskWrapper("Testing slide extraction...") as test:
            try:
                self.project.extract_tiles(
                    tile_px=self.tile_px,
                    tile_um=604,
                    buffer=self.buffer,
                    source=['TEST'],
                    roi_method='ignore',
                    skip_extracted=False,
                    img_format='png',
                    enable_downsample=enable_downsample,
                    **kwargs
                )
                self.project.extract_tiles(
                    tile_px=self.tile_px,
                    tile_um="10x",
                    buffer=self.buffer,
                    source=['TEST'],
                    roi_method='ignore',
                    img_format='png',
                    enable_downsample=enable_downsample,
                    dry_run=True,
                    **kwargs
                )
            except Exception as e:
                log.error(traceback.format_exc())
                test.fail()

    def test_normalizers(
        self,
        *args,
        single: bool = True,
        multi: bool = True,
    ) -> None:
        """Test normalizer strategy and throughput, saving example image
        for each.

        Args:
            single (bool, optional): Perform single-thread tests.
                Defaults to True.
            multi (bool, optional): Perform multi-thread tests.
                Defaults to True.
        """
        assert self.project is not None
        if single:
            with TaskWrapper("Testing normalization single-thread throughput...") as test:
                passed = process_isolate(
                    sf.test.functional.single_thread_normalizer_tester,
                    project=self.project,
                    methods=args,
                    tile_px=self.tile_px
                )
                if not passed:
                    test.fail()
        if multi:
            with TaskWrapper("Testing normalization multi-thread throughput...") as test:
                passed = process_isolate(
                    sf.test.functional.multi_thread_normalizer_tester,
                    project=self.project,
                    methods=args,
                    tile_px=self.tile_px
                )
                if not passed:
                    test.fail()

    def test_readers(self) -> None:
        """Test TFRecord reading between backends (Tensorflow/PyTorch), ensuring
        that both yield identical results.
        """
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
            passed = process_isolate(
                sf.test.functional.reader_tester,
                project=self.project,
                tile_px=self.tile_px
            )
            if not passed:
                test.fail()

    def train_perf(self, **train_kwargs) -> None:
        """Test model training across multiple epochs."""

        assert self.project is not None
        msg = "Training single classification outcome from HP sweep..."
        with TaskWrapper(msg) as test:
            try:
                self.setup_hp(
                    'classification',
                    sweep=True,
                    normalizer='reinhard_fast',
                    uq=False
                )
                results = self.project.train(
                    exp_label='manual_hp',
                    outcomes='category1',
                    val_k=1,
                    params='sweep.json',
                    pretrain=None,
                    **train_kwargs
                )
                _assert_valid_results(results)
            except Exception as e:
                log.error(traceback.format_exc())
                test.fail()

    def test_training(
        self,
        classification: bool = True,
        resume: bool = True,
        uq: bool = True,
        multi_classification: bool = True,
        regression: bool = True,
        multi_regression: bool = True,
        multi_input: bool = True,
        survival: bool = True,
        multi_survival: bool = True,
        from_wsi: bool = True,
        **train_kwargs
    ) -> None:
        """Test model training using a variety of strategies.

        Models are trained for one epoch for only 20 steps.

        Args:
            classification (bool, optional): Test training a single outcome,
                multi-class classification model. Defaults to True.
            uq (bool, optional): Test training with UQ. Defaults to True.
            multi_classification (bool, optional): Test training a multi-outcome,
                multi-class classification model. Defaults to True.
            regression (bool, optional): Test training a continuous outcome.
                Defaults to True.
            multi_regression (bool, optional): Test training with multiple
                continuous outcomes. Defaults to True.
            multi_input (bool, optional): Test training with slide-level input
                in addition to image input. Defaults to True.
            survival (bool, optional): Test training a Cox-Proportional Hazards
                model. Defaults to True.
            multi_survival (bool, optional): Test training a survival model with
                additional slide-level input. Defaults to True.
        """
        assert self.project is not None
        for k in self.train_kwargs:
            if k not in train_kwargs:
                train_kwargs[k] = self.train_kwargs[k]
        # Disable checkpoints for tensorflow backend, to save disk space
        if (sf.backend() == 'tensorflow'
           and 'save_checkpoints' not in train_kwargs):
            train_kwargs['save_checkpoints'] = False

        if classification:
            # Test classification outcome
            self.train_perf(**train_kwargs)

        if resume:
            # Test resuming training
            with TaskWrapper("Training with resume...") as test:
                try:
                    to_resume = self._get_model('category1-manual_hp-TEST-HPSweep1-kfold1')
                    if sf.backend() == 'tensorflow':
                        resume_kw = dict(
                            resume_training=to_resume,
                        )
                    else:
                        resume_kw = dict(
                            checkpoint=to_resume
                        )
                except OSError:
                    log.warning("Could not find classification model for testing resume_training")
                    resume_kw = dict()
                    test.skip()
                else:
                    try:
                        hp = self.setup_hp('classification', sweep=False, uq=False)
                        results = self.project.train(
                            exp_label='resume',
                            outcomes='category1',
                            val_k=1,
                            params=hp,
                            pretrain=None,
                            **resume_kw,
                            **train_kwargs
                        )
                        _assert_valid_results(results)
                    except Exception as e:
                        log.error(traceback.format_exc())
                        test.fail()

        if uq:
            # Test classification outcome with UQ
            try:
                # Use pretrained model if possible, for testing
                to_resume = self._get_model('category1-manual_hp-TEST-HPSweep0-kfold1')
            except OSError:
                to_resume = None  # type: ignore
            msg = "Training single classification outcome with UQ..."
            with TaskWrapper(msg) as test:
                try:
                    hp = self.setup_hp('classification', sweep=False, uq=True)
                    results = self.project.train(
                        exp_label='UQ',
                        outcomes='category1',
                        val_k=1,
                        params=hp,
                        pretrain=to_resume,
                        **train_kwargs
                    )
                    _assert_valid_results(results)
                except Exception as e:
                    log.error(traceback.format_exc())
                    test.fail()

        if multi_classification:
            # Test multiple sequential categorical outcome models
            with TaskWrapper("Training to multiple outcomes...") as test:
                try:
                    results = self.project.train(
                        outcomes=['category1', 'category2'],
                        val_k=1,
                        params=self.setup_hp('classification'),
                        pretrain=None,
                        **train_kwargs
                    )
                    _assert_valid_results(results)
                except Exception as e:
                    log.error(traceback.format_exc())
                    test.fail()

        if regression:
            # Test single regression outcome
            with TaskWrapper("Training with single regression outcome...") as test:
                try:
                    results = self.project.train(
                        outcomes=['continuous1'],
                        val_k=1,
                        params=self.setup_hp('regression'),
                        pretrain=None,
                        **train_kwargs
                    )
                    _assert_valid_results(results)
                except Exception as e:
                    log.error(traceback.format_exc())
                    test.fail()

        if multi_regression:
            # Test multiple regression outcome
            with TaskWrapper("Training multiple regression outcomes...") as test:
                try:
                    results = self.project.train(
                        outcomes=['continuous1', 'continuous2'],
                        val_k=1,
                        params=self.setup_hp('regression'),
                        pretrain=None,
                        **train_kwargs
                    )
                    _assert_valid_results(results)
                except Exception as e:
                    log.error(traceback.format_exc())
                    test.fail()

        if multi_input:
            msg = 'Training with multiple inputs (image + slide feature)...'
            with TaskWrapper(msg) as test:
                try:
                    results = self.project.train(
                        exp_label='multi_input',
                        outcomes='category1',
                        input_header='category2',
                        params=self.setup_hp('classification'),
                        val_k=1,
                        pretrain=None,
                        **train_kwargs
                    )
                    _assert_valid_results(results)
                except Exception as e:
                    log.error(traceback.format_exc())
                    test.fail()

        if survival:
            with TaskWrapper("Training a survival model...") as test:
                if sf.backend() == 'tensorflow':
                    try:
                        results = self.project.train(
                            exp_label='survival',
                            outcomes='time',
                            input_header='event',
                            params=self.setup_hp('survival'),
                            val_k=1,
                            pretrain=None,
                            **train_kwargs
                        )
                        _assert_valid_results(results)
                    except Exception as e:
                        log.error(traceback.format_exc())
                        test.fail()
                else:
                    test.skip()

        if multi_survival:
            with TaskWrapper("Training a multi-input survival model...") as test:
                if sf.backend() == 'tensorflow':
                    try:
                        results = self.project.train(
                            exp_label='multi_survival',
                            outcomes='time',
                            input_header=['event', 'category1'],
                            params=self.setup_hp('survival'),
                            val_k=1,
                            pretrain=None,
                            **train_kwargs
                        )
                        _assert_valid_results(results)
                    except Exception as e:
                        log.error(traceback.format_exc())
                        test.fail()
                else:
                    test.skip()
        if from_wsi:
            # Test training from slides without TFRecords
            msg = "Training model directly from slides (from_wsi=True)..."
            with TaskWrapper(msg) as test:
                try:
                    hp = self.setup_hp('classification', sweep=False, balance=None)
                    results = self.project.train(
                        exp_label='from_wsi',
                        outcomes='category1',
                        val_k=1,
                        params=hp,
                        from_wsi=True,
                        pretrain=None,
                        **train_kwargs
                    )
                    _assert_valid_results(results)
                except Exception as e:
                    log.error(traceback.format_exc())
                    test.fail()

    def test_prediction(self, **predict_kwargs) -> None:
        """Test prediction generation using a previously trained model."""

        assert self.project is not None
        model = self._get_model('category1-manual_hp-TEST-HPSweep0-kfold1')

        with TaskWrapper("Testing classification model predictions...") as test:
            passed = process_isolate(
                sf.test.functional.prediction_tester,
                project=self.project,
                model=model,
                **predict_kwargs
            )
            if not passed:
                test.fail()

    def test_evaluation(self, **eval_kwargs) -> None:
        """Test evaluation of previously trained models."""

        assert self.project is not None
        multi_cat_model = self._get_model('category1-category2-HP0-kfold1')
        multi_lin_model = self._get_model('continuous1-continuous2-HP0-kfold1')
        multi_inp_model = self._get_model('category1-multi_input-HP0-kfold1')
        f_model = self._get_model('category1-manual_hp-TEST-HPSweep0-kfold1')

        # Performs evaluation in isolated thread to avoid OOM errors
        # with sequential model loading/testing
        with TaskWrapper("Testing classification model evaluation...") as test:
            passed = process_isolate(
                sf.test.functional.evaluation_tester,
                project=self.project,
                model=f_model,
                outcomes='category1',
                save_predictions=True,
                **eval_kwargs
            )
            if not passed:
                test.fail()

        with TaskWrapper("Testing classification UQ model evaluation...") as test:
            uq_model = self._get_model('category1-UQ-HP0-kfold1')
            passed = process_isolate(
                sf.test.functional.evaluation_tester,
                project=self.project,
                model=uq_model,
                outcomes='category1',
                save_predictions=True,
                **eval_kwargs
            )
            if not passed:
                test.fail()

        with TaskWrapper("Testing multi-classification model evaluation...") as test:
            passed = process_isolate(
                sf.test.functional.evaluation_tester,
                project=self.project,
                model=multi_cat_model,
                outcomes=['category1', 'category2'],
                save_predictions=True,
                **eval_kwargs
            )
            if not passed:
                test.fail()

        with TaskWrapper("Testing multi-outcome regression model evaluation...") as test:
            passed = process_isolate(
                sf.test.functional.evaluation_tester,
                project=self.project,
                model=multi_lin_model,
                outcomes=['continuous1', 'continuous2'],
                save_predictions=True,
                **eval_kwargs
            )
            if not passed:
                test.fail()

        with TaskWrapper("Testing multi-input model evaluation...") as test:
            passed = process_isolate(
                sf.test.functional.evaluation_tester,
                project=self.project,
                model=multi_inp_model,
                outcomes='category1',
                input_header='category2',
                **eval_kwargs
            )
            if not passed:
                test.fail()

        with TaskWrapper("Testing survival model evaluation...") as test:
            if sf.backend() == 'tensorflow':
                survival_model = self._get_model('time-survival-HP0-kfold1')
                passed = process_isolate(
                    sf.test.functional.evaluation_tester,
                    project=self.project,
                    model=survival_model,
                    outcomes='time',
                    input_header='event',
                    **eval_kwargs
                )
                if not passed:
                    test.fail()
            else:
                test.skip()

    def test_heatmap(self, slide: str = 'auto', **heatmap_kwargs) -> None:
        """Test heatmap generation using a previously trained model."""

        assert self.project is not None
        model = self._get_model('category1-manual_hp-TEST-HPSweep0-kfold1')
        assert exists(model), "Model has not yet been trained."

        with TaskWrapper("Testing heatmap generation...") as test:
            try:
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
            except Exception as e:
                log.error(traceback.format_exc())
                test.fail()

    def test_activations_and_mosaic(self, **act_kwargs) -> None:
        """Test calculation of final-layer activations & creation
        of a mosaic map.
        """
        assert self.project is not None
        model = self._get_model('category1-manual_hp-TEST-HPSweep0-kfold1')
        assert exists(model), "Model has not yet been trained."
        with TaskWrapper("Testing activations and mosaic...") as test:
            passed = process_isolate(
                sf.test.functional.activations_tester,
                project=self.project,
                model=model,
                tile_px=self.tile_px,
                **act_kwargs
            )
            if not passed:
                test.fail()

    def test_predict_wsi(self) -> None:
        """Test predictions for whole-slide images."""

        assert self.project is not None
        model = self._get_model('category1-manual_hp-TEST-HPSweep0-kfold1')
        assert exists(model), "Model has not yet been trained."
        with TaskWrapper("Testing WSI prediction...") as test:
            passed = process_isolate(
                sf.test.functional.wsi_prediction_tester,
                project=self.project,
                model=model
            )
            if not passed:
                test.fail()

    def test_mil(self) -> None:
        """Test the MIL submodule."""

        assert self.project is not None
        model = self._get_model('category1-manual_hp-TEST-HPSweep0-kfold1')
        assert exists(model), "Model has not yet been trained."

        try:
            skip_test = False
            import torch  # noqa F401
        except ImportError:
            log.warning("Unable to import pytorch, skipping MIL test")
            skip_test = True

        with TaskWrapper("Testing MIL feature export...") as test:
            if skip_test:
                test.skip()
            else:
                passed = process_isolate(
                    sf.test.functional.feature_generator_tester,
                    project=self.project,
                    model=model
                )
                if not passed:
                    test.fail()

        with TaskWrapper("Testing MIL training...") as test:
            if skip_test:
                test.skip()
            else:
                try:
                    dataset = self.project.dataset(self.tile_px, 604)
                    train_dts, val_dts = dataset.split(val_fraction=0.3)
                    import slideflow.mil
                    config = sf.mil.mil_config('attention_mil', epochs=5, lr=1e-4, drop_last=False)
                    self.project.train_mil(
                        config,
                        train_dts,
                        val_dts,
                        outcomes='category1',
                        bags=join(self.project.root, 'mil'),
                        attention_heatmaps=True
                    )
                except Exception as e:
                    log.error(traceback.format_exc())
                    test.fail()

    def test(
        self,
        unit: bool = True,
        extract: bool = True,
        reader: bool = True,
        train: bool = True,
        normalizer: bool = True,
        evaluate: bool = True,
        predict: bool = True,
        heatmap: bool = True,
        activations: bool = True,
        predict_wsi: bool = True,
        mil: bool = True,
        slide_threads: Optional[int] = None
    ) -> None:
        """Perform and report results of all available testing."""

        start = time.time()
        if unit:
            self.unittests()
        if self.project is None:
            print("[yellow]Slides not provided; unable to perform "
                  "functional or WSI testing.")
        else:
            if extract:
                self.test_extraction(num_threads=slide_threads)
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
            if mil:
                self.test_mil()
        end = time.time()
        m, s = divmod(end-start, 60)
        print(f'Tests complete. Time: {int(m)} min, {s:.2f} sec')

    def unittests(self) -> None:
        """Run unit tests."""

        try:
            import tensorflow as tf
            sf.util.allow_gpu_memory_growth()
        except ImportError:
            pass

        print("Running unit tests...")
        runner = unittest.TextTestRunner(verbosity=2)
        all_tests = [
            unittest.TestLoader().loadTestsFromModule(module)
            for module in (norm_test, dataset_test, stats_test, model_test)
        ]
        suite = unittest.TestSuite(all_tests)

        # Add WSI tests if slides are provided
        if self.project is not None:
            test_slide = self.project.dataset().slide_paths()[0]
            test_loader = unittest.TestLoader()
            test_names = test_loader.getTestCaseNames(slide_test.TestSlide)
            for test_name in test_names:
                suite.addTest(slide_test.TestSlide(test_name, test_slide))

        runner.run(suite)
