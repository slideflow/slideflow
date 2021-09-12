import os
import types
import shutil
import logging
import itertools
import csv
import queue
import threading
import time
import pickle
import numpy as np
import shapely.geometry as sg
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.colors as mcol

from os.path import join, exists, isdir, dirname
from glob import glob
from datetime import datetime
from statistics import mean, median
from tqdm import tqdm

import slideflow as sf
import slideflow.util as sfutil
import slideflow.io as sfio

from slideflow import project_utils
from slideflow.io import Dataset
from slideflow.statistics import TFRecordMap, calculate_centroid
from slideflow.util import TCGA, ProgressBar, log, StainNormalizer
from slideflow.project_utils import get_validation_settings

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class SlideflowProject:

    '''Class to assist with organizing datasets and executing pipeline functions.'''

    def __init__(self, project_folder, gpu=None, gpu_pool=None, reverse_select_gpu=True,
                 interactive=True, default_threads=4, verbosity=logging.DEBUG):

        """Initializes project by creating project folder, prompting user for project settings, and project
        settings to "settings.json" within the project directory.

        Args:
            project_folder (str):                 Path to project directory.
            gpu (int, optional):                  Manually assign GPU. Defaults to None.
            gpu_pool (int, optional):             List of ints indicating available GPUs. Defaults to None.
            reverse_select_gpu (bool, optional):  Select from gpu_pool in reverse order. Defaults to True.
            interactive (bool, optional):         Prompt user for settings if project not initialized. Defaults to True.
            default_threads (int, optional):      Default threads available for multithreaded functions. Defaults to 4.
            verbosity (str, optional):            Default project-wide logging verbosity. Defaults to logging.DEBUG.
        """
        log.setLevel(verbosity)
        self.verbosity = verbosity
        self.default_threads = default_threads

        if project_folder and not os.path.exists(project_folder):
            if interactive:
                print(f'Directory "{project_folder}" does not exist.')
                if sfutil.yes_no_input('Create direcotry and set as project root? [Y/n] ', default='yes'):
                    os.makedirs(project_folder)
                else:
                    project_folder = sfutil.dir_input('Where is the project root directory? ',
                                                      None,
                                                      create_on_invalid=True,
                                                      absolute=True)
            else:
                log.info(f'Project directory {project_folder} not found; will create.')
                os.makedirs(project_folder)
        if not project_folder:
            project_folder = sfutil.dir_input('Where is the project root directory? ',
                                              None,
                                              create_on_invalid=True,
                                              absolute=True)

        #log.configure(filename=join(project_folder, 'log.log'))

        if exists(join(project_folder, 'settings.json')):
            self.load_project(project_folder)
        elif interactive:
            self.create_project(project_folder)

        # Set up GPU
        if gpu is not None:
            self.select_gpu(gpu)
        elif gpu_pool:
            self.autoselect_gpu(gpu_pool, reverse=reverse_select_gpu)

    @property
    def root(self):
        '''Path root directory.'''
        return self._settings['root']

    @property
    def annotations(self):
        '''Path to annotations file.'''
        return self._read_relative_path(self._settings['annotations'])

    @property
    def dataset_config(self):
        '''Path to dataset configuration JSON file.'''
        return self._read_relative_path(self._settings['dataset_config'])

    @property
    def models_dir(self):
        return self._read_relative_path(self._settings['models_dir'])

    @property
    def batch_train_config(self):
        return self._read_relative_path(self._settings['batch_train_config'])

    @property
    def datasets(self):
        return self._settings['datasets']

    @property
    def mixed_precision(self):
        if 'mixed_precision' in self._settings:
            return self._settings['mixed_precision']
        elif 'use_fp16' in self._settings:
            log.warn("'mixed_precision' not found in project settings. Please update the settings.json file.")
            return self._settings['use_fp16']

    def _read_relative_path(self, path):
        if path[:6] == '$ROOT/':
            return join(self.root, path[6:])
        else:
            return path

    def autoselect_gpu(self, number_available, reverse=True):
        '''Automatically claims a free GPU.

        Args:
            number_available:	Total number of GPUs available to select from
            reverse:			Bool, if True, will select GPU from pool in reverse
        '''
        log.info('Selecting GPU...')

        if not number_available:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            log.warn(f'Disabling GPU access.')
        else:
            gpus = range(number_available) if not reverse else list(reversed(range(number_available)))
            gpu_selected = -1
            if len(gpus):
                gpu_selected = gpus[0]
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_selected)
                log.info(f'Using GPU {gpu_selected}')

    def select_gpu(self, gpu):
        '''Sets environmental variables such that the indicated GPU is used by CUDA/Tensorflow.'''
        if gpu == -1:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            log.warn(f'Disabling GPU access.')
        else:
            log.info(f'Using GPU #{gpu}')
            os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)

        import tensorflow as tf

    def add_dataset(self, name, slides, roi, tiles, tfrecords, path=None):
        '''Adds a dataset to the dataset configuration file.

        Args:
            name:		Dataset name.
            slides:		Path to directory containing slides.
            roi:		Path to directory containing CSV ROIs.
            tiles:		Path to directory in which to store extracted tiles.
            tfrecords:	Path to directory in which to store TFRecords of extracted tiles.
            path:		(optional) Path to dataset configuration file. If not provided, uses project default.
        '''

        if not path:
            path = self.dataset_config
        try:
            datasets_data = sfutil.load_json(path)
        except FileNotFoundError:
            datasets_data = {}
        datasets_data.update({name: {
            'slides': slides,
            'roi': roi,
            'tiles': tiles,
            'tfrecords': tfrecords,
        }})
        sfutil.write_json(datasets_data, path)
        log.info(f'Saved dataset {name} to {path}')

    def associate_slide_names(self):
        '''Funtion to automatically associate patient names with slide filenames in the annotations file.'''
        log.info('Associating slide names...')
        dataset = self.get_dataset(tile_px=0, tile_um=0, verification=None)
        dataset.update_annotations_with_slidenames(self.annotations)

    def create_blank_annotations_file(self, filename=None):
        '''Creates an example blank annotations file.'''
        if not filename:
            filename = self.annotations
        with open(filename, 'w') as csv_outfile:
            csv_writer = csv.writer(csv_outfile, delimiter=',')
            header = [TCGA.patient, 'dataset', 'category']
            csv_writer.writerow(header)

    def create_blank_train_config(self, filename=None):
        '''Creates a CSV file with the batch training hyperparameter structure.'''
        from slideflow.model import HyperParameters

        if not filename:
            filename = self.batch_train_config
        with open(filename, 'w') as csv_outfile:
            writer = csv.writer(csv_outfile, delimiter='\t')
            # Create headers and first row
            header = ['model_name']
            firstrow = ['model1']
            default_hp = HyperParameters()
            for arg in default_hp._get_args():
                header += [arg]
                firstrow += [getattr(default_hp, arg)]
            writer.writerow(header)
            writer.writerow(firstrow)

    def create_hyperparameter_sweep(self, tile_px, tile_um, finetune_epochs,
                                    label=None, filename=None, **kwargs):
        '''Prepares a hyperparameter sweep, saving to a batch train TSV file.'''
        log.info('Preparing hyperparameter sweep...')
        pdict = kwargs
        pdict.update({'tile_px': tile_px, 'tile_um': tile_um})

        args = list(pdict.keys())
        for arg in args:
            if not isinstance(pdict[arg], list):
                pdict[arg] = [pdict[arg]]
        argsv = list(pdict.values())
        sweep = list(itertools.product(*argsv))

        from slideflow.model import HyperParameters

        if not filename:
            filename = self.batch_train_config
        label = '' if not label else f'{label}-'
        with open(filename, 'w') as csv_outfile:
            writer = csv.writer(csv_outfile, delimiter='\t')
            # Create headers
            header = ['model_name', 'finetune_epochs']
            for arg in args:
                header += [arg]
            writer.writerow(header)
            # Iterate through sweep
            for i, params in enumerate(sweep):
                row = [f'{label}HPSweep{i}', ','.join([str(f) for f in finetune_epochs])]
                full_params = dict(zip(['finetune_epochs'] + args, [finetune_epochs] + list(params)))
                hp = HyperParameters(**full_params)
                for arg in args:
                    row += [getattr(hp, arg)]
                writer.writerow(row)
        log.info(f'Wrote {len(sweep)} combinations for sweep to {sfutil.green(filename)}')

    def create_project(self, project_folder):
        '''Prompts user to provide all relevant project configuration
            and saves configuration to "settings.json".'''
        # General setup and slide configuration
        project = {
            'root': project_folder,
            'slideflow_version': sf.__version__
        }
        project['name'] = input('What is the project name? ')

        # Ask for annotations file location; if one has not been made,
        # offer to create a blank template and then exit
        if not sfutil.yes_no_input('Has an annotations (CSV) file already been created? [y/N] ', default='no'):
            if sfutil.yes_no_input('Create a blank annotations file? [Y/n] ', default='yes'):
                project['annotations'] = sfutil.file_input('Annotations file location [./annotations.csv] ',
                                                            root=project['root'],
                                                            default='./annotations.csv',
                                                            filetype='csv',
                                                            verify=False)
                self.create_blank_annotations_file(project['annotations'])
        else:
            project['annotations'] = sfutil.file_input('Annotations file location [./annotations.csv] ',
                                                        root=project['root'],
                                                        default='./annotations.csv',
                                                        filetype='csv')

        # Dataset configuration
        project['dataset_config'] = sfutil.file_input('Dataset configuration file location [./datasets.json] ',
                                                      root=project['root'],
                                                      default='./datasets.json',
                                                      filetype='json',
                                                      verify=False)

        project['datasets'] = []
        while not project['datasets']:
            datasets_data, datasets_names = self.load_datasets(project['dataset_config'])

            print(sfutil.bold('Detected datasets:'))
            if not len(datasets_names):
                print(' [None]')
            else:
                for i, name in enumerate(datasets_names):
                    print(f' {i+1}. {name}')
                print(f' {len(datasets_names)+1}. ADD NEW')
                valid_dataset_choices = [str(l) for l in range(1, len(datasets_names)+2)]
                dataset_selection = sfutil.choice_input(f'Which datasets should be used? ',
                                                        valid_choices=valid_dataset_choices,
                                                        multi_choice=True)

            if not len(datasets_names) or str(len(datasets_names)+1) in dataset_selection:
                # Create new dataset
                print(f"{sfutil.bold('Creating new dataset')}")
                dataset_name = input('What is the dataset name? ')
                dataset_slides = sfutil.dir_input('Where are the slides stored? [./slides] ',
                                        root=project['root'], default='./slides', create_on_invalid=True)
                dataset_roi = sfutil.dir_input('Where are the ROI files (CSV) stored? [./slides] ',
                                        root=project['root'], default='./slides', create_on_invalid=True)
                dataset_tiles = sfutil.dir_input('Image tile storage location [./tiles] ',
                                        root=project['root'], default='./tiles', create_on_invalid=True)
                dataset_tfrecords = sfutil.dir_input('TFRecord storage location [./tfrecord] ',
                                        root=project['root'], default='./tfrecord', create_on_invalid=True)

                self.add_dataset(name=dataset_name,
                                 slides=dataset_slides,
                                 roi=dataset_roi,
                                 tiles=dataset_tiles,
                                 tfrecords=dataset_tfrecords,
                                 path=project['dataset_config'])

                print('Updated dataset configuration file.')
            else:
                try:
                    project['datasets'] = [datasets_names[int(j)-1] for j in dataset_selection]
                except TypeError:
                    print(f'Invalid selection: {dataset_selection}')
                    continue

        # Training
        project['models_dir'] = sfutil.dir_input('Where should the saved models be stored? [./models] ',
                                                  root=project['root'],
                                                  default='./models',
                                                  create_on_invalid=True)

        project['mixed_precision'] = sfutil.yes_no_input('Use mixed precision? [Y/n] ', default='yes')
        project['batch_train_config'] = sfutil.file_input('Batch training TSV location [./batch_train.tsv] ',
                                                          root=project['root'],
                                                          default='./batch_train.tsv',
                                                          filetype='tsv',
                                                          verify=False)

        if not exists(project['batch_train_config']):
            print('Batch training file not found, creating blank')
            self.create_blank_train_config(project['batch_train_config'])


        # Save settings as relative paths
        project['annotations'] = project['annotations'].replace(project_folder, '$ROOT')
        project['dataset_config'] = project['dataset_config'].replace(project_folder, '$ROOT')
        project['models_dir'] = project['models_dir'].replace(project_folder, '$ROOT')
        project['batch_train_config'] = project['batch_train_config'].replace(project_folder, '$ROOT')

        sfutil.write_json(project, join(project_folder, 'settings.json'))
        self._settings = project

        # Write a sample actions.py file
        with open(join(os.path.dirname(os.path.realpath(__file__)), 'sample_actions.py'), 'r') as sample_file:
            sample_actions = sample_file.read()
            with open(os.path.join(project_folder, 'actions.py'), 'w') as actions_file:
                actions_file.write(sample_actions)

        print('\nProject configuration saved.\n')
        self.load_project(project_folder)

    def evaluate(self,
                 model,
                 outcome_label_headers,
                 hyperparameters=None,
                 filters=None,
                 checkpoint=None,
                 eval_k_fold=None,
                 max_tiles_per_slide=0,
                 min_tiles_per_slide=0,
                 normalizer=None,
                 normalizer_source=None,
                 batch_size=64,
                 input_header=None,
                 permutation_importance=False,
                 histogram=False,
                 save_predictions=False):

        '''Evaluates a saved model on a given set of tfrecords.

        Args:
            model:					Path to Tensorflow model to evaluate.
            outcome_label_headers:	Annotation column header that specifies the outcome label(s).
            hyperparameters:		Path to model's hyperparameters.json file.
                                        If None (default), searches in the model directory.
            filters:				Filters to use when selecting tfrecords on which to perform evaluation.
            checkpoint:				Path to cp.ckpt file to load, if evaluating a saved checkpoint.
            eval_k_fold:			K-fold iteration number to evaluate.
                                        If None, will evaluate all tfrecords irrespective of K-fold.
            max_tiles_per_slide:	Will only use up to this many tiles from each slide for evaluation.
                                        If zero, will include all tiles.
            min_tiles_per_slide:	Minimum number of tiles a slide must have to be included in evaluation.
                                        Default is 0, but a minimum of at least 10 tiles per slide is recommended.
            normalizer:				Normalization strategy to use on image tiles.
            normalizer_source:		Path to normalizer source image.
            input_header:			Annotation column header to use as additional input to the model.
            permutation_importance:	Bool. True if you want to calculate the permutation feature importance
                                        Used to determine relative importance when using multiple model inputs.
            histogram:				Bool. If true, will create tile-level histograms to show
                                        prediction distributions for each class.
            save_predictions:		Either True, False, or any combination of 'tile', 'patient', or 'slide'.
                                        Will save tile-level, patient-level, and/or slide-level predictions.
                                        If True, will save all.
        '''
        log.info(f'Evaluating model {sfutil.green(model)}...')

        if (input_header is None) and permutation_importance:
            log.warn('Permutation feature importance is designed to be used with multimodal models. Turning off.')
            permutation_importance = False

        manager = multiprocessing.Manager()
        results_dict = manager.dict()
        ctx = multiprocessing.get_context('spawn')

        process = ctx.Process(target=project_utils.evaluator, args=(self,
                                                                    outcome_label_headers,
                                                                    model,
                                                                    results_dict,
                                                                    input_header,
                                                                    filters,
                                                                    hyperparameters,
                                                                    checkpoint,
                                                                    eval_k_fold,
                                                                    max_tiles_per_slide,
                                                                    min_tiles_per_slide,
                                                                    normalizer,
                                                                    normalizer_source,
                                                                    batch_size,
                                                                    permutation_importance,
                                                                    histogram,
                                                                    save_predictions))
        process.start()
        log.info(f'Spawning evaluation process (PID: {process.pid})')
        process.join()

        return results_dict

    def evaluate_clam(self, exp_name, pt_files, outcome_label_headers, tile_px, tile_um, eval_tag=None,
                        filters=None, filter_blank=None, attention_heatmaps=True):
        '''Evaluate CLAM model on saved feature activations.

        Args:
            exp_name:			Name of experiemnt to evaluate (directory in clam/ subfolder)
            pt_files:				Path to pt_files containing tile-level features.
            outcome_label_headers:	Name in annotation column which specifies the outcome label.
            tile_px:				Tile width in pixels.
            tile_um:				Tile width in microns.
            eval_tag:				Unique identifier for this evaluation.
            filters:				Dictionary of column names mapping to column values
                                        by which to filter slides using the annotation file.
                                        Used if train_slides and validation_slides are 'auto'.
            filter_blank:			List of annotations headers; slides blank in this column will be excluded.
                                        Used if train_slides and validation_slides are 'auto'.
            attention_heatmaps:		Bool. If true, will save attention heatmaps of validation dataset.

        Returns:
            None
        '''

        import slideflow.clam as clam
        from slideflow.clam.datasets.dataset_generic import Generic_MIL_Dataset
        from slideflow.clam.create_attention import export_attention

        # Detect source CLAM experiment which we are evaluating.
        # First, assume it lives in this project's clam folder
        if exists(join(self.root, 'clam', exp_name, 'experiment.json')):
            exp_name = join(self.root, 'clam', exp_name)
        elif exists(join(exp_name, 'experiment.json')):
            pass
        else:
            raise Exception(f"Unable to find the experiment '{exp_name}'")

        log.info(f'Loading trained experiment from {sfutil.green(exp_name)}')
        eval_dir = join(exp_name, 'eval')
        if not exists(eval_dir): os.makedirs(eval_dir)

        # Set up evaluation directory with unique evaluation tag
        existing_tags = [int(d) for d in os.listdir(eval_dir) if d.isdigit()]
        if eval_tag is None:
            eval_tag = '0' if not existing_tags else str(max(existing_tags))

        # Ensure evaluation tag will not overwrite existing results
        if eval_tag in existing_tags:
            unique, base_tag = 1, eval_tag
            eval_tag = f'{base_tag}_{unique}'
            while exists(join(eval_dir, eval_tag)):
                eval_tag = f'{base_tag}_{unique}'
                unique += 1
            log.info(f"Eval tag {base_tag} already exists, will save evaluation under 'eval_tag'")

        # Load trained model checkpoint
        ckpt_path = join(exp_name, 'results', 's_0_checkpoint.pt')
        eval_dir = join(eval_dir, eval_tag)
        if not exists(eval_dir): os.makedirs(eval_dir)
        args_dict = sfutil.load_json(join(exp_name, 'experiment.json'))
        args = types.SimpleNamespace(**args_dict)
        args.save_dir = eval_dir

        dataset = self.get_dataset(tile_px=tile_px,
                                   tile_um=tile_um,
                                   filters=filters,
                                   filter_blank=filter_blank)

        evaluation_slides = [s for s in dataset.get_slides() if exists(join(pt_files, s+'.pt'))]
        dataset.apply_filters({'slide': evaluation_slides})

        slide_labels, unique_labels = dataset.get_labels_from_annotations(outcome_label_headers,
                                                                          use_float=False,
                                                                          key='outcome_label')

        # Set up evaluation annotations file based off existing pt_files
        outcome_dict = dict(zip(range(len(unique_labels)), unique_labels))
        with open(join(eval_dir, 'eval_annotations.csv'), 'w') as eval_file:
            writer = csv.writer(eval_file)
            header = ['submitter_id', 'slide', outcome_label_headers]
            writer.writerow(header)
            for slide in evaluation_slides:
                row = [slide, slide, outcome_dict[slide_labels[slide]['outcome_label']]]
                writer.writerow(row)

        clam_dataset = Generic_MIL_Dataset(csv_path=join(eval_dir, 'eval_annotations.csv'),
                                           data_dir=pt_files,
                                           shuffle=False,
                                           seed=args.seed,
                                           print_info=True,
                                           label_col=outcome_label_headers,
                                           label_dict = dict(zip(unique_labels, range(len(unique_labels)))),
                                           patient_strat=False,
                                           ignore=[])

        clam.evaluate(ckpt_path, args, clam_dataset)

        # Get attention from trained model on validation set
        attention_tfrecords = dataset.get_tfrecords()
        attention_dir = join(eval_dir, 'attention')
        if not exists(attention_dir): os.makedirs(attention_dir)
        export_attention(args_dict,
                            ckpt_path=ckpt_path,
                            export_dir=attention_dir,
                            pt_files=pt_files,
                            slides=dataset.get_slides(),
                            reverse_label_dict = dict(zip(range(len(unique_labels)), unique_labels)),
                            slide_to_label = {s:slide_labels[s]['outcome_label'] for s in slide_labels})
        if attention_heatmaps:
            heatmaps_dir = join(eval_dir, 'attention_heatmaps')
            if not exists(heatmaps_dir): os.makedirs(heatmaps_dir)

            for tfr in attention_tfrecords:
                attention_dict = {}
                slide = sfutil.path_to_name(tfr)
                try:
                    with open(join(attention_dir, slide+'.csv'), 'r') as csv_file:
                        reader = csv.reader(csv_file)
                        for row in reader:
                            attention_dict.update({int(row[0]): float(row[1])})
                except FileNotFoundError:
                    print(f'Unable to find attention scores for slide {slide}, skipping')
                    continue
                self.generate_tfrecord_heatmap(tfr, attention_dict, heatmaps_dir, tile_px=tile_px, tile_um=tile_um)

    def extract_tiles(self, tile_px, tile_um, filters=None, filter_blank=None, stride_div=1,
                        tma=False, full_core=False, save_tiles=False, save_tfrecord=True,
                        enable_downsample=False, roi_method='inside', skip_missing_roi=True,
                        skip_extracted=True, dataset=None, validation_settings=None,
                        normalizer=None, normalizer_source=None, whitespace_fraction=1.0,
                        whitespace_threshold=230, grayspace_fraction=0.6,
                        grayspace_threshold=0.05, img_format='png', randomize_origin=False,
                        buffer=None, shuffle=True, num_workers=4, threads_per_worker=4):

        '''Extract tiles from a group of slides; save a percentage of tiles for validation testing if the
        validation target is 'per-patient'; and generate TFRecord files from the raw images.

        Args:
            tile_px:				Tile size in pixels.
            tile_um:				Tile size in microns.
            filters:				Dataset filters to use when selecting slides for tile extraction.
            stride_div:				Stride divisor to use when extracting tiles.
                                        A stride of 1 will extract non-overlapping tiles.
                                        A stride_div of 2 will extract overlapping tiles,
                                        with a stride equal to 50% of the tile width.
            tma:					Bool. If True, reads slides as Tumor Micro-Arrays (TMAs),
                                        detecting and extracting tumor cores.
            full_core:				Bool. Only used if extracting from TMA. If True, will save entire TMA core as image.
                                        Otherwise, will extract sub-images from each core
                                        using the given tile micron size.
            save_tiles:				Bool. If True, will save images of extracted tiles to a tile directory.
            save_tfrecord:			Bool. If True, will save compressed image data from extracted tiles
                                        into TFRecords in the corresponding TFRecord directory.
            enable_downsample:		Bool. If True, enables the use of downsampling while reading slide images.
                                        This may result in corrupted image tiles if downsampled slide layers
                                        are corrupted or incomplete. Recommend manual confirmation of tile integrity.
            roi_method:				Either 'inside', 'outside', or 'ignore'.
                                        Whether to extract tiles inside or outside the ROIs.
            skip_missing_roi:		Bool. If True, will skip slides that are missing ROIs
            skip_extracted:			Bool. If True, will skip slides that have already been fully extracted
            dataset:				Name of dataset from which to select slides for extraction.
                                        If not provided, will default to all datasets in project
            validation_settings:	Namespace of validation settings, provided by sf.project.get_validation_settings().
                                        Necessary if performing per-tile validation. If not provided, will ignore.
            normalizer:				Normalization strategy to use on image tiles
            normalizer_source:		Path to normalizer source image
            whitespace_fraction:	Float 0-1. Fraction of whitespace which causes a tile to be discarded.
                                        If 1, will not perform whitespace filtering.
            whitespace_threshold:	Int 0-255. Threshold above which a pixel (RGB average) is considered whitespace.
            grayspace_fraction:		Float 0-1. Fraction of grayspace which causes a tile to be discarded.
                                        If 1, will not perform grayspace filtering.
            grayspace_threshold:	Int 0-1. HSV (hue, saturation, value) is calculated for each pixel.
                                        If a pixel's saturation is below this threshold, it is considered grayspace.
            img_format:				'png' or 'jpg'. Format of images for internal storage in tfrecords.
                                        PNG (lossless) format recommended for fidelity, JPG (lossy) for efficiency.
            randomize_origin:		Bool. If true, will randomize the pixel starting position during extraction.
            buffer:					Path to directory. Slides will be copied to the here before extraction.
                                        Using an SSD or ramdisk buffer vastly improves tile extraction speed.
            shuffle:				Bool. If true (default), will shuffle tiles in tfrecords.
            num_workers:			Number of slides from which to be extracting tiles simultaneously.
            threads_per_worker:		Number of processes to allocate to each slide for tile extraction.
        '''

        import slideflow.slide as sfslide

        log.info('Extracting image tiles...')

        if not save_tiles and not save_tfrecord:
            log.error('Either save_tiles or save_tfrecord must be true to extract tiles.')
            return

        if dataset: datasets = [dataset] if not isinstance(dataset, list) else dataset
        else:		datasets = self.datasets

        # Load dataset for evaluation
        extracting_dts = self.get_dataset(filters=filters,
                                          filter_blank=filter_blank,
                                          tile_px=tile_px,
                                          tile_um=tile_um,
                                          verification='slides')

        # Prepare validation/training subsets if per-tile validation is being used
        if validation_settings and validation_settings.target == 'per-tile':
            if validation_settings.strategy == 'boostrap':
                raise ValueError('Validation bootstrapping is not supported when the validation target is per-tile.')
            if validation_settings.strategy in ('bootstrap', 'fixed'):
                # Split the extracted tiles into two groups
                split_fraction = [-1, validation_settings.fraction]
                split_names = ['training', 'validation']
            if validation_settings.strategy == 'k-fold':
                split_fraction = [-1] * validation_settings.k_fold
                split_names = [f'kfold-{i}' for i in range(validation_settings.k_fold)]
        else:
            split_fraction, split_names = None, None

        if normalizer: log.info(f'Extracting tiles using {sfutil.bold(normalizer)} normalization')
        if whitespace_fraction < 1:
            log.info('Filtering tiles by whitespace fraction')
            log.info(f'Whitespace defined as RGB avg > {whitespace_threshold})')
            log.info(f'(exclude if >={whitespace_fraction*100:.0f}% whitespace')

        for dataset_name in datasets:
            log.info(f'Working on dataset {sfutil.bold(dataset_name)}')

            tiles_dir = join(extracting_dts.datasets[dataset_name]['tiles'],
                                extracting_dts.datasets[dataset_name]['label'])
            roi_dir = extracting_dts.datasets[dataset_name]['roi']
            dataset_config = extracting_dts.datasets[dataset_name]
            tfrecord_dir = join(dataset_config['tfrecords'], dataset_config['label'])
            if save_tfrecord and not exists(tfrecord_dir):
                os.makedirs(tfrecord_dir)
            if save_tiles and not os.path.exists(tiles_dir):
                os.makedirs(tiles_dir)

            # Prepare list of slides for extraction
            slide_list = extracting_dts.get_slide_paths(dataset=dataset_name)

            # Check for interrupted or already-extracted tfrecords
            if skip_extracted and save_tfrecord:
                already_done = [sfutil.path_to_name(tfr) for tfr in extracting_dts.get_tfrecords(dataset=dataset_name)]
                interrupted = [sfutil.path_to_name(marker) for marker in glob(join((tfrecord_dir
                                                           if tfrecord_dir else tiles_dir), '*.unfinished'))]
                if len(interrupted):
                    log.info(f'Interrupted tile extraction in {len(interrupted)} tfrecords, will re-extract slides')
                    for interrupted_slide in interrupted:
                        log.info(interrupted_slide)
                        if interrupted_slide in already_done:
                            del already_done[already_done.index(interrupted_slide)]

                slide_list = [slide for slide in slide_list if sfutil.path_to_name(slide) not in already_done]
                if len(already_done):
                    log.info(f'Skipping {len(already_done)} slides; TFRecords already generated.')
            log.info(f'Extracting tiles from {len(slide_list)} slides ({tile_um} um, {tile_px} px)')

            # Verify slides and estimate total number of tiles
            log.info('Verifying slides...')
            total_tiles = 0
            for slide_path in tqdm(slide_list, leave=False):
                if tma:
                    slide = sfslide.TMAReader(slide_path, tile_px, tile_um, stride_div, silent=True)
                else:
                    slide = sfslide.SlideReader(slide_path,
                                                tile_px,
                                                tile_um,
                                                stride_div,
                                                roi_dir=roi_dir,
                                                roi_method=roi_method,
                                                skip_missing_roi=False,
                                                silent=True)
                log.info(f"Estimated tiles for slide {slide.name}: {slide.estimated_num_tiles}")
                total_tiles += slide.estimated_num_tiles
                del slide
            log.info(f'Verification complete. Total estimated tiles to extract: {total_tiles}')

            # Use multithreading if specified, extracting tiles from all slides in the filtered list
            if len(slide_list):
                q = queue.Queue()
                task_finished = False
                manager = multiprocessing.Manager()
                ctx = multiprocessing.get_context('spawn')
                reports = manager.dict()
                counter = manager.Value('i', 0)
                counter_lock = manager.Lock()

                if total_tiles:
                    pb = ProgressBar(total_tiles,
                                     counter_text='tiles',
                                     leadtext='Extracting tiles... ',
                                     show_counter=True,
                                     show_eta=True,
                                     mp_counter=counter,
                                     mp_lock=counter_lock)
                    pb.auto_refresh(0.1)
                else:
                    pb = None

                # Worker to grab slide path from queue and start tile extraction
                def worker():
                    while True:
                        try:
                            path = q.get()
                            process = ctx.Process(target=project_utils.tile_extractor, args=(path,
                                                                                             roi_dir,
                                                                                             roi_method,
                                                                                             skip_missing_roi,
                                                                                             randomize_origin,
                                                                                             img_format,
                                                                                             tma,
                                                                                             full_core,
                                                                                             shuffle,
                                                                                             tile_px,
                                                                                             tile_um,
                                                                                             stride_div,
                                                                                             enable_downsample,
                                                                                             whitespace_fraction,
                                                                                             whitespace_threshold,
                                                                                             grayspace_fraction,
                                                                                             grayspace_threshold,
                                                                                             normalizer,
                                                                                             normalizer_source,
                                                                                             split_fraction,
                                                                                             split_names,
                                                                                             self.root,
                                                                                             tfrecord_dir,
                                                                                             tiles_dir,
                                                                                             save_tiles,
                                                                                             save_tfrecord,
                                                                                             buffer,
                                                                                             threads_per_worker,
                                                                                             counter,
                                                                                             counter_lock))

                            process.start()
                            process.join()
                            if buffer and buffer != 'vmtouch':
                                os.remove(path)
                            q.task_done()
                        except queue.Empty:
                            if task_finished:
                                return

                # Start the worker threads
                threads = [threading.Thread(target=worker, daemon=True) for t in range(num_workers)]
                for thread in threads:
                    thread.start()

                # Put each slide path into queue
                for slide_path in slide_list:
                    warned = False
                    if buffer and buffer != 'vmtouch':
                        while True:
                            if q.qsize() < num_workers:
                                try:
                                    buffered_path = join(buffer, os.path.basename(slide_path))
                                    shutil.copy(slide_path, buffered_path)
                                    q.put(buffered_path)
                                    break
                                except OSError as e:
                                    if not warned:
                                        formatted_slide = sfutil._shortname(sfutil.path_to_name(slide_path))
                                        log.warn(f'OSError encountered for slide {formatted_slide}: buffer likely full')
                                        log.info(f'Q size: {q.qsize()}')
                                        warned = True
                                    time.sleep(1)
                            else:
                                time.sleep(1)
                    else:
                        q.put(slide_path)
                q.join()
                task_finished = True
                if pb: pb.end()
                log.info('Generating PDF (this may take some time)...', )
                pdf_report = sfslide.ExtractionReport(reports.values(), tile_px=tile_px, tile_um=tile_um)
                timestring = datetime.now().strftime('%Y%m%d-%H%M%S')
                pdf_report.save(join(self.root, f'tile_extraction_report-{timestring}.pdf'))

            # Update manifest
            extracting_dts.update_manifest()

    def extract_tiles_from_tfrecords(self, tile_px, tile_um, destination=None, filters=None):
        '''Extracts all tiles from a set of TFRecords.

        Args:
            tile_px:		Tile size in pixels
            tile_um:		Tile size in microns
            destination:	Destination folder in which to save tile images
            filters:		Dataset filters to use when selecting TFRecords
        '''
        log.info(f'Extracting tiles from TFRecords')
        to_extract_dataset = self.get_dataset(filters=filters,
                                              tile_px=tile_px,
                                              tile_um=tile_um)

        for dataset_name in self.datasets:
            to_extract_tfrecords = to_extract_dataset.get_tfrecords(dataset=dataset_name)
            if destination:
                tiles_dir = destination
            else:
                tiles_dir = join(to_extract_dataset.datasets[dataset_name]['tiles'],
                                 to_extract_dataset.datasets[dataset_name]['label'])
                if not exists(tiles_dir):
                    os.makedirs(tiles_dir)
            for tfr in to_extract_tfrecords:
                sfio.tfrecords.extract_tiles(tfr, tiles_dir)

    def generate_activations(self,
                             model,
                             outcome_label_headers=None,
                             layers=['postconv'],
                             filters=None,
                             filter_blank=None,
                             focus_nodes=None,
                             activations_export=None,
                             activations_cache=None,
                             normalizer=None,
                             normalizer_source=None,
                             max_tiles_per_slide=0,
                             min_tiles_per_slide=None,
                             include_logits=True,
                             batch_size=None,
                             torch_export=None,
                             isolated_process=False):

        '''Calculates final layer activations and displays information regarding
        the most significant final layer nodes. Note: GPU memory will remain in use,
        as the Keras model associated with the visualizer is active.

        Args:
            model:						Path to Tensorflow model
            outcome_label_headers:		(optional) Column header in annotations file.
                                            Used for category-level comparisons
            layers:						Layers from which to generate activations.
            filters:					Dataset filters for selecting TFRecords
            filter_blank:				List of label headers; slides that have blank entries in this label header
                                            in the annotations file will be excluded
            focus_nodes:				List of int, indicates which nodes are of interest for subsequent analysis
            activations_export:			Path to CSV file, if provided, will save activations in CSV format to this file
            activations_cache:			Either 'default' or path to 'PKL' file; will cache activations to this file.
            normalizer:					Normalization strategy to use on image tiles
            normalizer_source:			Path to normalizer source image
            max_tiles_per_slide:		Int. If > 0, will only take this many tiles per slide.
            min_tiles_per_slide:		Int. If > 0, will skip slides with fewer than this many tiles.
            include_logits:				If true, will also generate logit predictions along with layer activations.
            batch_size:					Batch size to use when calculating activations.
            torch_export:				Path. If true, exports activations to torch-compatible file in this location.
            isolated_process:			Bool. If true, will run in an isolated process (multiprocessing),
                                            allowing GPU memory to be freed after completion, and return None.
                                            If false, will return the ActivationsVisualizer object after completion.
        '''

        if isolated_process:
            manager = multiprocessing.Manager()
            results_dict = manager.dict()
            ctx = multiprocessing.get_context('spawn')

            process = ctx.Process(target=project_utils.activations_generator, args=(self,
                                                                                    model,
                                                                                    outcome_label_headers,
                                                                                    layers,
                                                                                    filters,
                                                                                    filter_blank,
                                                                                    focus_nodes,
                                                                                    activations_export,
                                                                                    activations_cache,
                                                                                    normalizer,
                                                                                    normalizer_source,
                                                                                    max_tiles_per_slide,
                                                                                    min_tiles_per_slide,
                                                                                    include_logits,
                                                                                    batch_size,
                                                                                    torch_export,
                                                                                    results_dict))
            process.start()
            log.info(f'Spawning activations process (PID: {process.pid})')
            process.join()
            return results_dict
        else:
            AV = project_utils.activations_generator(self,
                                                     model,
                                                     outcome_label_headers,
                                                     layers,
                                                     filters,
                                                     filter_blank,
                                                     focus_nodes,
                                                     activations_export,
                                                     activations_cache,
                                                     normalizer,
                                                     normalizer_source,
                                                     max_tiles_per_slide,
                                                     min_tiles_per_slide,
                                                     include_logits,
                                                     batch_size,
                                                     torch_export,
                                                     None)
            return AV

    def generate_features_for_clam(self,
                                   model,
                                   export_dir='auto',
                                   activation_layers=['postconv'],
                                   max_tiles_per_slide=0,
                                   min_tiles_per_slide=8,
                                   filters=None,
                                   filter_blank=None,
                                   force_regenerate=False):

        '''Using the specified model, generates tile-level features for slides for use with CLAM.

        Args:
            model:					Path to model from which to generate activations.
                                        May provide either this or "pt_files"
            export_dir:				Path in which to save exported activations in .pt format.
                                        If 'auto', will save in project directory.
            activation_layers:		Which model layer(s) from which to generate activations.
            max_tiles_per_slide:	Maximum number of tiles to take per slide
            min_tiles_per_slide:	Minimum number of tiles per slide. Will skip slides not meeting this threshold.
            filters:				Dictionary mapping annotation column names to values by which to filter slides.
            filter_blank:			List of annotations headers; slides blank in this column will be excluded.
            force_regenerate:		If true, will generate activations for all slides. If false, will skip slides that
                                        already have a .pt file in the export directory.
        Returns:
            Path to directory containing exported .pt files
        '''
        assert min_tiles_per_slide > 8, 'Slides must have at least 8 tiles to train CLAM.'

        # First, ensure the model is valid with a hyperparameters file
        hp_data = sfutil.get_model_hyperparameters(model)
        if not hp_data:
            raise Exception('Unable to find model hyperparameters file.')
        tile_px = hp_data['tile_px']
        tile_um = hp_data['tile_um']

        # Set up the pt_files directory for storing model activations
        if export_dir.lower() == 'auto':
            model_name_end = '' if 'k_fold_i' not in hp_data else f"_kfold{hp_data['k_fold_i']}"
            export_dir = join(self.root, 'pt_files', hp_data['model_name']+model_name_end)
        if not exists(export_dir):
            os.makedirs(export_dir)

        # Detect already generated pt files
        already_generated = [sfutil.path_to_name(f) for f in os.listdir(export_dir)
                                                    if sfutil.path_to_ext(join(export_dir, f)) == 'pt']
        if force_regenerate or not len(already_generated):
            activation_filters = filters
        else:
            pt_dataset = self.get_dataset(tile_px, tile_um, filters=filters, filter_blank=filter_blank)
            all_slides = pt_dataset.get_slides()
            slides_to_generate = [s for s in all_slides if s not in already_generated]
            activation_filters = filters.copy()
            activation_filters['slide'] = slides_to_generate
            filtered_dataset = self.get_dataset(tile_px, tile_um, filters=activation_filters, filter_blank=filter_blank)
            filtered_slides_to_generate = filtered_dataset.get_slides()
            log.info(f'Activations already generated for {len(already_generated)} files, will not regenerate.')
            log.info(f'Attempting to generate for {len(filtered_slides_to_generate)} slides')

        # Set up activations interface
        self.generate_activations(model,
                                  filters=activation_filters,
                                  filter_blank=filter_blank,
                                  layers=activation_layers,
                                  max_tiles_per_slide=max_tiles_per_slide,
                                  min_tiles_per_slide=min_tiles_per_slide,
                                  torch_export=export_dir,
                                  isolated_process=True,
                                  activations_cache=None)
        return export_dir

    def generate_heatmaps(self,
                          model,
                          filters=None,
                          filter_blank=None,
                          directory=None,
                          resolution='low',
                          interpolation='none',
                          show_roi=True,
                          roi_method='inside',
                          logit_cmap=None,
                          vmin=0,
                          vcenter=0.5,
                          vmax=1,
                          normalizer=None,
                          normalizer_source=None,
                          batch_size=64,
                          buffer=True,
                          isolated_process=True,
                          num_threads='auto'):

        '''Creates predictive heatmap overlays on a set of slides.

        Args:
            model:				Path to Tensorflow model with which predictions will be generated.
            filters:			Dataset filters to use when selecting slides for which to generate heatmaps.
            filter_blank:		List of label headers; slides that have blank entries in this label header
                                     in the annotations file will be excluded
            directory:			Directory in which to save heatmap images.
            resolution:			Heatmap resolution (determines stride of tile predictions).
                                    "low" uses a stride equal to tile width.
                                    "medium" uses a stride equal 1/2 tile width.
                                    "high" uses a stride equal to 1/4 tile width.
            interpolation:		Interpolation strategy for smoothing heatmap predictions
                                    (matplotlib imshow interpolation options).
            show_roi:			Bool. If True, will show ROI on heatmaps.
            roi_method:			'inside', 'outside', or 'none'. Determines where heatmap should be made
                                    with respect to annotated ROI.
            logit_cmap:			Either a function or a dictionary used to create heatmap colormap.
                                    If None (default), separate heatmaps will be generated for each label category,
                                    with color representing likelihood of category prediction.
                                    Each image tile will generate a list of predictions of length O,
                                    where O is the number of label categories.
                                    If logit_cmap is a function, then the logit predictions will be passed,
                                    and the function is expected to return [R, G, B] values which will be displayed.
                                    isolated_process must be true if a function is passed.
                                    If the logit_cmap is a dictionary, it should map 'r', 'g', and 'b' to label indices;
                                    The prediction for these label categories will be mapped to corresponding colors.
                                    Thus, the corresponding color will only reflect predictions of up to three labels.
                                        Example (this would map predictions for label 0 to red, 3 to green, etc):
                                        {'r': 0, 'g': 3, 'b': 1 }
            normalizer:			Normalization strategy to use on image tiles
            normalizer_source:	Path to normalizer source image
            buffer:				Path to directory. Slides will be copied to the directory as a buffer before extraction.
                                    This vastly improves extraction speed when using SSD or ramdisk buffer.
            isolated_process:	Bool. If True, will wrap function in separate process,
                                    allowing GPU memory to be freed after completion.
                                    If False, will perform as single thread (GPU memory not be freed after completion).
                                    Allows use for functions being passed to logit_cmap (functions are not pickleable).
        '''
        log.info('Generating heatmaps...')

        # Prepare dataset
        hp_data = sfutil.get_model_hyperparameters(model)
        heatmaps_dataset = self.get_dataset(filters=filters,
                                            filter_blank=filter_blank,
                                            tile_px=hp_data['hp']['tile_px'],
                                            tile_um=hp_data['hp']['tile_um'])
        slide_list = heatmaps_dataset.get_slide_paths()
        roi_list = heatmaps_dataset.get_rois()

        # Attempt to auto-detect supplied model name
        detected_model_name = os.path.basename(model)
        hp_data = sfutil.get_model_hyperparameters(model)
        if hp_data and 'model_name' in hp_data:
            detected_model_name = hp_data['model_name']

        # Make output directory
        heatmaps_folder = directory if directory else os.path.join(self.root, 'heatmaps', detected_model_name)
        if not exists(heatmaps_folder): os.makedirs(heatmaps_folder)

        # Heatmap processes
        ctx = multiprocessing.get_context('spawn')
        for slide in slide_list:
            if isolated_process:
                process = ctx.Process(target=project_utils.heatmap_generator, args=(self,
                                                                                    slide,
                                                                                    model,
                                                                                    heatmaps_folder,
                                                                                    roi_list,
                                                                                    show_roi,
                                                                                    roi_method,
                                                                                    resolution,
                                                                                    interpolation,
                                                                                    logit_cmap,
                                                                                    vmin,
                                                                                    vcenter,
                                                                                    vmax,
                                                                                    buffer,
                                                                                    normalizer,
                                                                                    normalizer_source,
                                                                                    batch_size,
                                                                                    num_threads))
                process.start()
                log.info(f'Spawning heatmaps process (PID: {process.pid})')
                process.join()
            else:
                project_utils.heatmap_generator(self,
                                                slide,
                                                model,
                                                heatmaps_folder,
                                                roi_list,
                                                show_roi,
                                                roi_method,
                                                resolution,
                                                interpolation,
                                                logit_cmap,
                                                vmin,
                                                vcenter,
                                                vmax,
                                                buffer,
                                                normalizer,
                                                normalizer_source,
                                                batch_size,
                                                num_threads)

    def generate_mosaic(self,
                        model,
                        mosaic_filename=None,
                        umap_filename=None,
                        outcome_label_headers=None,
                        layers=['postconv'],
                        filters=None,
                        filter_blank=None,
                        focus_filters=None,
                        resolution='low',
                        num_tiles_x=50,
                        max_tiles_per_slide=100,
                        expanded=False,
                        map_slide=None,
                        show_prediction=None,
                        restrict_pred=None,
                        predict_on_axes=None,
                        include_logits=True,
                        label_names=None,
                        cmap=None,
                        model_type=None,
                        umap_cache='default',
                        activations_cache='default',
                        activations_export=None,
                        umap_export=None,
                        use_float=False,
                        normalizer=None,
                        normalizer_source=None,
                        batch_size=64,
                        low_memory=False):

        '''Generates a mosaic map by overlaying images onto a set of mapped tiles.
            Image tiles are extracted from the provided set of TFRecords, and predictions + post-convolutional
            node activations are calculated using the specified model. Tiles are mapped either with dimensionality
            reduction on post-convolutional layer activations (default behavior), or by using outcome predictions
            for two categories, mapped to X- and Y-axis (via predict_on_axes).

        Args:
            model:					Path to Tensorflow model to use when generating layer activations.
            mosaic_filename:		Filename for mosaic image. If not provided, mosaic will not be calculated or saved.
                                        Will be saved in project mosaic directory.
            umap_filename:			Filename for UMAP plot image. If not provided, plot will not be saved.
                                        Will be saved in project stats directory.
            outcome_label_headers:	Column name in annotations file from which to read category labels.
            filters:				Dataset filters to use when selecting slides to include the mosaic.
            filter_blank:			List of label headers; slides that have blank entries in this label header
                                         in the annotations file will be excluded
            focus_filters:			Dataset filters to use when selecting slides to highlight on the mosaic.
            resolution:				Resolution of the mosaic map. Low, medium, or high.
            num_tiles_x:			Specifies the size of the mosaic map grid.
            max_tiles_per_slide:	Limits the number of tiles taken from each slide.
                                        Too high of a number may introduce memory issues.
            expanded:				Bool. If False, will limit tile assignment to the each grid space (strict display).
                                        If True, allows for display of nearby tiles if a given grid is empty.
            map_slide:				None (default), 'centroid', or 'average'.
                                        If provided, will map slides using slide-level calculations, either mapping
                                        centroid tiles if 'centroid', or calculating node averages across all tiles
                                        in a slide and mapping slide-level node averages, if 'average'
            show_prediction:		May be either int or string, corresponding to label category.
                                        Predictions for this category will be displayed on the exported UMAP plot.
            restrict_pred:	List of int, if provided, will restrict predictions to only these categories
                                        (final tile-level prediction is made by choosing category with highest logit)
            predict_on_axes:		(int, int). Each int corresponds to an label category id.
                                        If provided, predictions are generated for these two labels categories;
                                        tiles are then mapped with these predictions with the pattern (x, y)
                                        and the mosaic is generated from this map. This replaces the default
                                        dimensionality reduction mapping.
            label_names:			Dict mapping label id (int) to string names.
                                        Saved in the hyperparameters file as 'outcome_labels'
            cmap:					Colormap mapping labels to colors for display on UMAP plot
            model_type:				Indicates label type. 'categorical', 'linear', or 'cph' (Cox Proportional Hazards)
            umap_cache:				Either 'default' or path to PKL file in which to save/cache UMAP coordinates
            activations_cache:		Either 'default' or path to PKL file in which to save/cache nodal activations
            activations_export:		Filename for CSV export of activations. Will be saved in project stats directory.
            umap_export:			Filename for CSV export of UMAP coordinates, saved in project stats directory.
            use_float:				Bool, if True, assumes labels are float / linear (as opposed to categorical)
            normalizer:				Normalization strategy to use on image tiles
            normalizer_source:		Path to normalizer source image
            low_memory:				Bool, if True, will attempt to limit memory during UMAP calculations
                                        at the cost of increased computational complexity
        '''
        from slideflow.activations import ActivationsVisualizer
        from slideflow.mosaic import Mosaic

        log.info('Generating mosaic map...')

        # Set up paths
        stats_root = join(self.root, 'stats')
        mosaic_root = join(self.root, 'mosaic')
        if not exists(stats_root): os.makedirs(stats_root)
        if not exists(mosaic_root): os.makedirs(mosaic_root)
        if umap_cache and umap_cache == 'default':
            umap_cache = join(stats_root, 'umap_cache.pkl')
        elif umap_cache:
            umap_cache = join(stats_root, umap_cache)

        # Prepare dataset & model
        hp_data = sfutil.get_model_hyperparameters(model)
        mosaic_dataset = self.get_dataset(filters=filters,
                                          filter_blank=filter_blank,
                                          tile_px=hp_data['hp']['tile_px'],
                                          tile_um=hp_data['hp']['tile_um'])
        tfrecords_list = mosaic_dataset.get_tfrecords()
        if focus_filters:
            mosaic_dataset.apply_filters(focus_filters)
            focus_list = mosaic_dataset.get_tfrecords()
        else:
            focus_list = None
        num_focus = 0 if not focus_list else len(focus_list)
        log.info(f'Generating mosaic from {len(tfrecords_list)} slides, focus on {num_focus} slides.')

        # If a header category is supplied and we are not showing predictions,
        # then assign slide labels from annotations
        if model_type == 'linear': use_float = True
        if outcome_label_headers and (show_prediction is None):
            slide_labels = mosaic_dataset.slide_to_label(outcome_label_headers, use_float=use_float)
        else:
            slide_labels = {}

        # If showing predictions, try to automatically load prediction labels
        if (show_prediction is not None) and (not use_float) and (not label_names):
            model_hp = sfutil.get_model_hyperparameters(model)
            if model_hp:
                outcome_labels = model_hp['outcome_labels']
                model_type = model_type if model_type else model_hp['model_type']
                log.info(f'Automatically loaded prediction labels found at {sfutil.green(model)}')
            else:
                log.info(f'Unable to auto-detect prediction labels from model hyperparameters file')

        # Initialize mosaic, umap, and ActivationsVisualizer
        mosaic, umap = None, None
        if activations_export:
            activations_export = join(stats_root, activations_export)
        AV = ActivationsVisualizer(model=model,
                                   tfrecords=tfrecords_list,
                                   layers=layers,
                                   export_dir=join(self.root, 'stats'),
                                   image_size=hp_data['hp']['tile_px'],
                                   focus_nodes=None,
                                   normalizer=normalizer,
                                   normalizer_source=normalizer_source,
                                   batch_size=batch_size,
                                   include_logits=include_logits,
                                   activations_export=activations_export,
                                   max_tiles_per_slide=max_tiles_per_slide,
                                   cache=activations_cache,
                                   manifest=mosaic_dataset.get_manifest())

        if predict_on_axes:
            # Create mosaic using x- and y- axis corresponding to label predictions
            umap_x, umap_y, umap_meta = AV.map_to_predictions(predict_on_axes[0], predict_on_axes[1])
            umap = TFRecordMap.from_precalculated(tfrecords=mosaic_dataset.get_tfrecords(),
                                                  slides=mosaic_dataset.get_slides(),
                                                  x=umap_x,
                                                  y=umap_y,
                                                  meta=umap_meta)
        else:
            # Create mosaic map from dimensionality reduction on post-convolutional layer activations
            umap = TFRecordMap.from_activations(AV,
                                                map_slide=map_slide,
                                                prediction_filter=restrict_pred,
                                                cache=umap_cache,
                                                low_memory=low_memory,
                                                max_tiles_per_slide=max_tiles_per_slide)

        # If displaying centroid AND predictions, then show slide-level predictions rather than tile-level predictions
        if (map_slide=='centroid') and show_prediction is not None:
            log.info('Showing slide-level predictions at point of centroid')

            # If not model has not been assigned, assume categorical model
            model_type = model_type if model_type else 'categorical'

            # Get predictions
            if model_type == 'categorical':
                slide_pred, slide_percent = AV.get_slide_level_categorical_predictions(prediction_filter=restrict_pred)
            else:
                slide_pred = slide_percent = AV.get_slide_level_linear_predictions()

            # If show_prediction is provided (either a number or string),
            # then display ONLY the prediction for the provided category, as a colormap
            if type(show_prediction) == int:
                log.info(f'Showing prediction for label {show_prediction} as colormap')
                slide_labels = {k:v[show_prediction] for k, v in slide_percent.items()}
                show_prediction = None
                use_float = True
            elif type(show_prediction) == str:
                log.info(f'Showing prediction for label {show_prediction} as colormap')
                reversed_labels = {v:k for k, v in outcome_labels.items()}
                if show_prediction not in reversed_labels:
                    raise ValueError(f"Unknown label category '{show_prediction}'")
                slide_labels = {k:v[int(reversed_labels[show_prediction])] for k, v in slide_percent.items()}
                show_prediction = None
                use_float = True
            elif use_float:
                # Displaying linear predictions needs to be implemented here
                raise TypeError("If showing predictions & use_float=True, set 'show_prediction' \
                                    to category to be predicted.")
            # Otherwise, show_prediction is assumed to be just "True", in which case show categorical predictions
            else:
                try:
                    slide_labels = {k:outcome_labels[v] for (k,v) in slide_pred.items()}
                except KeyError:
                    # Try interpreting prediction label keys as strings
                    slide_labels = {k:outcome_labels[str(v)] for (k,v) in slide_pred.items()}

        if umap_filename:
            if slide_labels:
                umap.label_by_slide(slide_labels)
            if show_prediction and (map_slide != 'centroid'):
                umap.label_by_tile_meta('prediction', translation_dict=outcome_labels)
            umap.filter(mosaic_dataset.get_slides())
            umap.save_2d_plot(join(stats_root, umap_filename), cmap=cmap, use_float=use_float)
        if umap_export:
            umap.export_to_csv(join(stats_root, umap_export))

        if mosaic_filename:
            mosaic = Mosaic(umap,
                            leniency=1.5,
                            expanded=expanded,
                            tile_zoom=15,
                            num_tiles_x=num_tiles_x,
                            resolution=resolution,
                            normalizer=normalizer,
                            normalizer_source=normalizer_source)
            mosaic.focus(focus_list)
            mosaic.save(join(mosaic_root, mosaic_filename))
            mosaic.save_report(join(stats_root, sfutil.path_to_name(mosaic_filename)+'-mosaic_report.csv'))

        return AV, mosaic, umap

    def generate_mosaic_from_annotations(self, header_x, header_y, tile_px, tile_um, model=None,
                                            mosaic_filename=None, umap_filename=None, outcome_label_headers=None,
                                            filters=None, resolution='low', num_tiles_x=50, max_tiles_per_slide=100,
                                            expanded=False, use_optimal_tile=False, activations_cache='default',
                                            normalizer=None, normalizer_source=None, batch_size=64):
        '''Generates a mosaic map by overlaying images onto a set of mapped tiles.
            Slides are mapped with slide-level annotations, x-axis determined from header_x, y-axis from header_y.
            If use_optimal_tile is False and no model is provided, first image tile in each TFRecord is displayed.
            If optimal_tile is True, post-convolutional layer activations for all tiles in each slide are
            calculated using the provided model, and the tile nearest to centroid is used for display.

        Args:
            header_x:				Column name in annotations file from which to read X-axis coordinates.
            header_y:				Column name in annotations file from which to read Y-axis coordinates.
            tile_px:				Tile size in pixels.
            tile_um:				Tile size in microns.
            model:					Path to Tensorflow model to use when generating layer activations.
            mosaic_filename:		Filename for mosaic image. If not provided, mosaic will not be calculated or saved.
                                        Will be saved in project mosaic directory.
            umap_filename:			Filename for UMAP plot image. If not provided, plot will not be saved.
                                        Will be saved in project stats directory.
            outcome_label_headers:	Column name in annotations file from which to read category labels.
            filters:				Dataset filters to use when selecting slides to include the mosaic.
            focus_filters:			Dataset filters to use when selecting slides to highlight on the mosaic.
            resolution:				Resolution of the mosaic map. Impacts size of the final figure.
                                        Either low, medium, or high.
            num_tiles_x:			Specifies the size of the mosaic map grid.
            max_tiles_per_slide:	Limits the number of tiles taken from each slide.
                                        Too high of a number may introduce memory issues.
            expanded:				Bool. If False, will limit tile to the corresponding grid space (strict display).
                                        If True, allows for display of nearby tiles if a given grid is empty.
            use_optimal_tile:		Bool. If True, will use model to create layer activations for all tiles in
                                        each slide, and choosing tile nearest centroid for each slide for display.
            activations_cache:		Either 'default' or path to PKL file in which to save/cache nodal activations
            normalizer:				Normalization strategy to use on image tiles
            normalizer_source:		Path to normalizer source image
        '''
        from slideflow.activations import ActivationsVisualizer
        from slideflow.mosaic import Mosaic

        # Setup paths
        stats_root = join(self.root, 'stats')
        mosaic_root = join(self.root, 'mosaic')
        if not exists(stats_root): os.makedirs(stats_root)
        if not exists(mosaic_root): os.makedirs(mosaic_root)

        # Setup dataset
        dataset = self.get_dataset(filters=filters,
                                   filter_blank=[header_x, header_y],
                                   tile_px=tile_px,
                                   tile_um=tile_um)

        # We are assembling a list of slides from the TFRecords path list,
        # because we only want to use slides that have a corresponding TFRecord
        # (some slides did not have a large enough ROI for tile extraction,
        # and some slides may be in the annotations but are missing a slide image)
        slides = [sfutil.path_to_name(tfr) for tfr in dataset.get_tfrecords()]
        slide_labels_dict, _ = dataset.get_labels_from_annotations([header_x, header_y], use_float=True)
        slide_to_category = dataset.slide_to_label(outcome_label_headers)

        umap_x = np.array([slide_labels_dict[slide]['label'][0] for slide in slides])
        umap_y = np.array([slide_labels_dict[slide]['label'][1] for slide in slides])

        if use_optimal_tile and not model:
            log.error('Unable to calculate optimal tile if no model is specified.')
            return
        elif use_optimal_tile:
            # Calculate most representative tile in each slide/TFRecord for display
            AV = ActivationsVisualizer(model=model,
                                       tfrecords=dataset.get_tfrecords(),
                                       export_dir=join(self.root, 'stats'),
                                       image_size=tile_px,
                                       normalizer=normalizer,
                                       normalizer_source=normalizer_source,
                                       batch_size=batch_size,
                                       max_tiles_per_slide=max_tiles_per_slide,
                                       cache=activations_cache)

            optimal_slide_indices, _ = calculate_centroid(AV.slide_node_dict)

            # Restrict mosaic to only slides that had enough tiles to calculate an optimal index from centroid
            successful_slides = list(optimal_slide_indices.keys())
            num_warned = 0
            warn_threshold = 3
            for slide in slides:
                print_func = print if num_warned < warn_threshold else None
                if slide not in successful_slides:
                    log.warn(f'Unable to calculate optimal tile for {sfutil.green(slide)}; will skip')
                    num_warned += 1
            if num_warned >= warn_threshold:
                log.warn(f'...{num_warned} total warnings, see project log for details')

            umap_x = np.array([slide_labels_dict[slide]['label'][0] for slide in successful_slides])
            umap_y = np.array([slide_labels_dict[slide]['label'][1] for slide in successful_slides])
            umap_meta = [{'slide': slide, 'index': optimal_slide_indices[slide]} for slide in successful_slides]
        else:
            # Take the first tile from each slide/TFRecord
            umap_meta = [{'slide': slide, 'index': 0} for slide in slides]

        umap = TFRecordMap.from_precalculated(tfrecords=dataset.get_tfrecords(),
                                               slides=slides,
                                               x=umap_x,
                                               y=umap_y,
                                               meta=umap_meta)

        mosaic_map = Mosaic(umap,
                            leniency=1.5,
                            expanded=expanded,
                            tile_zoom=15,
                            num_tiles_x=num_tiles_x,
                            tile_select='centroid' if use_optimal_tile else 'nearest',
                            resolution=resolution,
                            normalizer=normalizer,
                            normalizer_source=normalizer_source)
        if mosaic_filename:
            mosaic_map.save(join(mosaic_root, mosaic_filename))
            mosaic_map.save_report(join(stats_root, sfutil.path_to_name(mosaic_filename)+'-mosaic_report.csv'))
        if umap_filename:
            umap.label_by_slide(slide_to_category)
            umap.save_2d_plot(join(stats_root, umap_filename))

    def generate_thumbnails(self, size=512, filters=None, filter_blank=None, roi=False, enable_downsample=False):
        '''Generates square slide thumbnails with black box borders of a fixed size, and saves to project folder.

        Args:
            size:				Int. Width/height of thumbnail in pixels.
            filters:			Dataset filters.
            filter_blank:		Header columns in annotations by which to filter slides,
                                    if the slides are blank in this column.
            roi:				Bool. If True, will include ROI in the thumbnail images.
            enable_downsample:	Bool. If True and a thumbnail is not embedded in the slide file,
                                    downsampling is permitted in order to accelerate thumbnail calculation.
        '''
        import slideflow.slide as sfslide
        log.info('Generating thumbnails...')

        thumb_folder = join(self.root, 'thumbs')
        if not exists(thumb_folder): os.makedirs(thumb_folder)
        dataset = self.get_dataset(filters=filters, filter_blank=filter_blank, tile_px=0, tile_um=0)
        slide_list = dataset.get_slide_paths()
        roi_list = dataset.get_rois()
        log.info(f'Saving thumbnails to {sfutil.green(thumb_folder)}')

        for slide_path in slide_list:
            print(f'\r\033[KWorking on {sfutil.green(sfutil.path_to_name(slide_path))}...', end='')
            whole_slide = sfslide.SlideReader(slide_path,
                                              tile_px=1000,
                                              tile_um=1000,
                                              stride_div=1,
                                              enable_downsample=enable_downsample,
                                              roi_list=roi_list,
                                              skip_missing_roi=roi,
                                              buffer=None,
                                              silent=True)
            if roi:
                thumb = whole_slide.annotated_thumb()
            else:
                thumb = whole_slide.square_thumb(size)
            thumb.save(join(thumb_folder, f'{whole_slide.name}.png'))
        print('\r\033[KThumbnail generation complete.')

    def generate_tfrecords_from_tiles(self, tile_px, tile_um, delete_tiles=True):
        '''Create tfrecord files from a collection of raw images, as stored in project tiles directory'''
        log.info('Writing TFRecord files...')

        # Load dataset for evaluation
        working_dataset = Dataset(config_file=self.dataset_config,
                                  sources=self.datasets,
                                  tile_px=tile_px,
                                  tile_um=tile_um)

        for d in working_dataset.datasets:
            log.info(f'Working on dataset {d}')
            config = working_dataset.datasets[d]
            tfrecord_dir = join(config['tfrecords'], config['label'])
            tiles_dir = join(config['tiles'], config['label'])
            if not exists(tiles_dir):
                log.warn(f'No tiles found for dataset {sfutil.bold(d)}')
                continue

            # Check to see if subdirectories in the target folders are slide directories (contain images)
            #  or are further subdirectories (e.g. validation and training)
            log.info('Scanning tile directory structure...')
            if sfutil.contains_nested_subdirs(tiles_dir):
                subdirs = [_dir for _dir in os.listdir(tiles_dir) if isdir(join(tiles_dir, _dir))]
                for subdir in subdirs:
                    tfrecord_subdir = join(tfrecord_dir, subdir)
                    sfio.tfrecords.write_tfrecords_multi(join(tiles_dir, subdir), tfrecord_subdir)
            else:
                sfio.tfrecords.write_tfrecords_multi(tiles_dir, tfrecord_dir)

            working_dataset.update_manifest()

            if delete_tiles:
                shutil.rmtree(tiles_dir)

    def generate_tfrecord_heatmap(self, tfrecord, tile_dict, export_dir, tile_px, tile_um):
        '''Creates a tfrecord-based WSI heatmap using a dictionary of tile values for heatmap display.

        Args:
            tfrecord:		Path to tfrecord
            tile_dict:		Dictionary mapping tfrecord indices to a tile-level value for display in heatmap format
            export_dir:		Path to directory in which to save images
            tile_px:		Tile width in pixels
            tile_um:		Tile width in microns

        Returns:
            Dictionary mapping slide names to dict of statistics (mean, median, above_0, and above_1)'''

        from slideflow.io.tfrecords import get_locations_from_tfrecord
        from slideflow.slide import SlideReader

        slide_name = sfutil.path_to_name(tfrecord)
        loc_dict = get_locations_from_tfrecord(tfrecord)
        dataset = self.get_dataset(tile_px=tile_px, tile_um=tile_um)
        slide_paths = {sfutil.path_to_name(sp):sp for sp in dataset.get_slide_paths()}

        try:
            slide_path = slide_paths[slide_name]
        except KeyError:
            raise Exception(f'Unable to locate slide {slide_name}')

        if tile_dict.keys() != loc_dict.keys():
            raise Exception(f'Length of provided tile_dict ({len(list(tile_dict.keys()))}) does not match \
                                number of tiles stored in the TFRecord ({len(list(loc_dict.keys()))}).')

        print(f'Generating TFRecord heatmap for {sfutil.green(tfrecord)}...')
        slide = SlideReader(slide_path, tile_px, tile_um, skip_missing_roi=False)

        stats = {}

        # Loaded CSV coordinates:
        x = [int(loc_dict[l][0]) for l in loc_dict]
        y = [int(loc_dict[l][1]) for l in loc_dict]
        vals = [tile_dict[l] for l in loc_dict]

        stats.update({
            slide_name: {
                'mean':mean(vals),
                'median':median(vals),
                'above_0':len([v for v in vals if v > 0]),
                'above_1':len([v for v in vals if v > 1]),
            }
        })

        print('\nLoaded tile values')
        print(f'Min: {min(vals)}\t Max:{max(vals)}')

        scaled_x = [(xi * slide.ROI_SCALE) - slide.full_extract_px/2 for xi in x]
        scaled_y = [(yi * slide.ROI_SCALE) - slide.full_extract_px/2 for yi in y]

        print('\nLoaded CSV coordinates:')
        print(f'Min x: {min(x)}\t Max x: {max(x)}')
        print(f'Min y: {min(y)}\t Max y: {max(y)}')

        print('\nScaled CSV coordinates:')
        print(f'Min x: {min(scaled_x)}\t Max x: {max(scaled_x)}')
        print(f'Min y: {min(scaled_y)}\t Max y: {max(scaled_y)}')

        print('\nSlide properties:')
        print(f'Raw size (x): {slide.full_shape[0]}\t Raw size (y): {slide.full_shape[1]}')

        # Slide coordinate information
        max_coord_x = max([c[0] for c in slide.coord])
        max_coord_y = max([c[1] for c in slide.coord])
        num_x = len(set([c[0] for c in slide.coord]))
        num_y = len(set([c[1] for c in slide.coord]))

        print('\nSlide tile grid:')
        print(f'Number of tiles (x): {num_x}\t Max coord (x): {max_coord_x}')
        print(f'Number of tiles (y): {num_y}\t Max coord (y): {max_coord_y}')

        # Calculate dead space (un-extracted tiles) in x and y axes
        dead_x = slide.full_shape[0] - max_coord_x
        dead_y = slide.full_shape[1] - max_coord_y
        fraction_dead_x = dead_x / slide.full_shape[0]
        fraction_dead_y = dead_y / slide.full_shape[1]

        print('\nSlide dead space')
        print(f'x: {dead_x}\t y:{dead_y}')

        # Work on grid
        x_grid_scale = max_coord_x / (num_x-1)
        y_grid_scale = max_coord_y / (num_y-1)

        print('\nCoordinate grid scale:')
        print(f'x: {x_grid_scale}\t y: {y_grid_scale}')

        grid = np.zeros((num_y, num_x))

        indexed_x = [round(xi / x_grid_scale) for xi in scaled_x]
        indexed_y = [round(yi / y_grid_scale) for yi in scaled_y]

        for i, (xi,yi,v) in enumerate(zip(indexed_x,indexed_y,vals)):
            grid[yi][xi] = v

        fig = plt.figure(figsize=(18, 16))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(bottom = 0.25, top=0.95)
        gca = plt.gca()
        gca.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)

        print('Generating thumbnail...')
        thumb = slide.thumb(mpp=5)
        print('Saving thumbnail....')
        thumb.save(join(export_dir, f'{slide_name}' + '.png'))
        print('Generating figure...')
        implot = ax.imshow(thumb, zorder=0)

        extent = implot.get_extent()
        extent_x = extent[1]
        extent_y = extent[2]
        grid_extent = (extent[0], extent_x * (1-fraction_dead_x), extent_y * (1-fraction_dead_y), extent[3])

        print('\nImage extent:')
        print(extent)
        print('\nGrid extent:')
        print(grid_extent)

        divnorm=mcol.TwoSlopeNorm(vmin=min(-0.01, min(vals)), vcenter=0, vmax=max(0.01, max(vals)))
        heatmap = ax.imshow(grid,
                            zorder=10,
                            alpha=0.6,
                            extent=grid_extent,
                            interpolation='bicubic',
                            cmap='coolwarm',
                            norm=divnorm)

        print('Saving figure...')
        plt.savefig(join(export_dir, f'{slide_name}_attn.png'), bbox_inches='tight')

        # Clean up
        print('Cleaning up...')
        plt.clf()
        del slide
        del thumb

        return stats

    def get_dataset(self, tile_px=None, tile_um=None, filters=None, filter_blank=None, verification='both'):
        '''Returns slideflow.io.Dataset object using project settings.

        Args:
            tile_px:		Tile size in pixels
            tile_um:		Tile size in microns
            filters:		Dictionary of annotations filters to use when selecting slides to include in dataset
            filter_blank:	List of label headers; will only include slides that are not blank in these headers
            verification:	'tfrecords', 'slides', or 'both'.
                                If 'slides', will verify all annotations are mapped to slides.
                                If 'tfrecords', will check that TFRecords exist and update manifest
        '''
        try:
            dataset = Dataset(config_file=self.dataset_config,
                              sources=self.datasets,
                              tile_px=tile_px,
                              tile_um=tile_um,
                              annotations=self.annotations,
                              filters=filters,
                              filter_blank=filter_blank)

        except FileNotFoundError:
            log.warn('No datasets configured.')

        if verification in ('both', 'slides'):
            log.info("Verifying slide annotations...")
            dataset.verify_annotations_slides()
        if verification in ('both', 'tfrecords'):
            log.info("Verifying tfrecords...")
            dataset.update_manifest()

        return dataset

    def load_datasets(self, path):
        '''Loads datasets dictionaries from a given datasets.json file.'''
        try:
            datasets_data = sfutil.load_json(path)
            datasets_names = list(datasets_data.keys())
            datasets_names.sort()
        except FileNotFoundError:
            datasets_data = {}
            datasets_names = []
        return datasets_data, datasets_names

    def load_project(self, directory):
        '''Loads a saved and pre-configured project from the specified directory.'''
        if exists(join(directory, 'settings.json')):
            self._settings = sfutil.load_json(join(directory, 'settings.json'))
        else:
            raise OSError(f'Unable to locate settings.json at location "{directory}".')

        # Enable logging
        #log.logfile = join(self.root, 'log.log')

    def predict_wsi(self,
                    model_path,
                    tile_px,
                    tile_um,
                    export_dir,
                    filters=None,
                    filter_blank=None,
                    stride_div=1,
                    enable_downsample=False,
                    roi_method='inside',
                    skip_missing_roi=False,
                    dataset=None,
                    normalizer=None,
                    normalizer_source=None,
                    whitespace_fraction=1.0,
                    whitespace_threshold=230,
                    grayspace_fraction=0.6,
                    grayspace_threshold=0.05,
                    randomize_origin=False,
                    buffer=None,
                    num_threads=-1):

        '''Using a given model, generates a spatial map of tile-level predictions for a whole-slide image (WSI)
            and dumps prediction arrays into pkl files for later use.

        Args:
            model_path:				Path to model from which to generate predictions.
            tile_px:				Tile size in pixels.
            tile_um:				Tile size in microns.
            export_dir:				Path to export directory in which to save .pkl files.
            filters:				Annotations filters to use when selecting slides.
            filter_blank:			Filter out slides blank in this annotations column.
            stride_div:				Stride divisor when extracting tiles from a whole slide image.
            enable_downsample:		Bool. If true, enables tile extraction from downsampled pyramids in slide images.
                                        May result in corrupted images for some slides.
            roi_method:				None, 'inside', or 'outside'. How to extract tiles wrt. annoted ROIs.
            skip_missing_roi:		Bool. If true, will skip slides with missing ROIs.
            dataset:				Name of dataset from which to get slides. If None, will use project default.
            normalizer:				Name of normalizer to use for tiles.
            normalizer_source:		Path to image tile to use as reference for normalizer.
                                        If None, will use internal default reference tile.
            whitespace_fraction:	Float 0-1. Fraction of whitespace which causes a tile to be discarded.
                                        If 1, will not perform whitespace filtering.
            whitespace_threshold:	Int 0-255. Threshold above which a pixel (RGB average) is considered whitespace.
            grayspace_fraction:		Float 0-1. Fraction of grayspace which causes a tile to be discarded.
                                        If 1, will not perform grayspace filtering.
            grayspace_threshold:	Int 0-1. HSV (hue, saturation, value) is calculated for each pixel.
                                        If a pixel's saturation is below this threshold, it is considered grayspace.
            randomize_origin:		Bool. If true, will randomize the origin for the extraction grid on each slide.
            buffer:					Path to buffer directory in which to copy slides prior to extraction.
                                        Using a ramdisk buffer greatly improves tile extraction speed
                                        for slides stored on disks with slow random access.
            num_threads:			Number of processes to use when generating predictions.
                                        If 1, will not use multiprocessing.

        Returns:
            None'''
        import slideflow.slide as sfslide

        log.info('Generating WSI prediction / activation maps...')
        if not exists(export_dir):
            os.makedirs(export_dir)
        if dataset: datasets = [dataset] if not isinstance(dataset, list) else dataset
        else:		datasets = self.datasets

        # Load dataset for evaluation
        extracting_dataset = self.get_dataset(filters=filters,
                                              filter_blank=filter_blank,
                                              tile_px=tile_px,
                                              tile_um=tile_um,
                                              verification='slides')
        # Info logging
        if normalizer: log.info(f'Using {sfutil.bold(normalizer)} normalization')
        if whitespace_fraction < 1:
            log.info('Filtering tiles by whitespace fraction')
            log.info(f'Whitespace defined as RGB avg > {whitespace_threshold})')
            log.info(f'(exclude if >={whitespace_fraction*100:.0f}% whitespace')

        for dataset_name in datasets:
            log.info(f'Working on dataset {sfutil.bold(dataset_name)}')
            roi_dir = extracting_dataset.datasets[dataset_name]['roi']

            # Prepare list of slides for extraction
            slide_list = extracting_dataset.get_slide_paths(dataset=dataset_name)
            log.info(f'Generating predictions for {len(slide_list)} slides ({tile_um} um, {tile_px} px)')

            # Verify slides and estimate total number of tiles
            log.info('Verifying slides...')
            total_tiles = 0
            for slide_path in tqdm(slide_list, leave=False):
                slide = sfslide.SlideReader(slide_path,
                                            tile_px,
                                            tile_um,
                                            stride_div,
                                            roi_dir=roi_dir,
                                            roi_method=roi_method,
                                            skip_missing_roi=False,
                                            silent=True,
                                            buffer=None)
                log.info(f"Estimated tiles for slide {slide.name}: {slide.estimated_num_tiles}")
                total_tiles += slide.estimated_num_tiles
                del slide
            log.info(f'Verification complete. Total estimated tiles to extract: {total_tiles}')

            if total_tiles:
                pb = ProgressBar(total_tiles,
                                counter_text='tiles',
                                leadtext='Extracting tiles... ',
                                show_counter=True,
                                show_eta=True)
                pb.auto_refresh()
                pb_counter = pb.get_counter()
                pb_lock = pb.get_lock()
                print_fn = pb.print
            else:
                pb_counter = pb_lock = print_fn = None

            # Function to extract tiles from a slide
            def predict_wsi_from_slide(slide_path, downsample):
                print_func = print if not pb else pb.print
                log.info(f'Working on slide {sfutil.path_to_name(slide_path)}')
                whole_slide = sfslide.SlideReader(slide_path,
                                                  tile_px,
                                                  tile_um,
                                                  stride_div,
                                                  enable_downsample=downsample,
                                                  roi_dir=roi_dir,
                                                  roi_method=roi_method,
                                                  randomize_origin=randomize_origin,
                                                  skip_missing_roi=skip_missing_roi,
                                                  buffer=buffer,
                                                  pb_counter=pb_counter,
                                                  counter_lock=pb_lock,
                                                  print_fn=print_fn)

                if not whole_slide.loaded_correctly():
                    return

                try:
                    wsi_grid = whole_slide.predict(model=model_path,
                                                   normalizer=normalizer,
                                                   normalizer_source=normalizer_source,
                                                   whitespace_fraction=whitespace_fraction,
                                                   whitespace_threshold=whitespace_threshold,
                                                   grayspace_fraction=grayspace_fraction,
                                                   grayspace_threshold=grayspace_threshold)

                    with open (join(export_dir, whole_slide.name+'.pkl'), 'wb') as pkl_file:
                        pickle.dump(wsi_grid, pkl_file)

                except sfslide.TileCorruptionError:
                    if downsample:
                        log.warn(f'Corrupt tile in {sfutil.green(sfutil.path_to_name(slide_path))}; will try \
                                    re-extraction with downsampling disabled')
                        predict_wsi_from_slide(slide_path, downsample=False)
                    else:
                        formatted_slide = sfutil.green(sfutil.path_to_name(slide_path))
                        log.error(f'Corrupt tile in {formatted_slide}; skipping slide')
                        return None

            # Use multithreading if specified, extracting tiles from all slides in the filtered list
            if num_threads == -1: num_threads = self.default_threads
            if num_threads > 1 and len(slide_list):
                q = queue.Queue()
                task_finished = False

                def worker():
                    while True:
                        try:
                            path = q.get()
                            if buffer and buffer != 'vmtouch':
                                buffered_path = join(buffer, os.path.basename(path))
                                predict_wsi_from_slide(buffered_path, enable_downsample)
                                os.remove(buffered_path)
                            else:
                                predict_wsi_from_slide(path, enable_downsample)
                            q.task_done()
                        except queue.Empty:
                            if task_finished:
                                return

                threads = [threading.Thread(target=worker, daemon=True) for t in range(num_threads)]
                for thread in threads:
                    thread.start()

                for slide_path in slide_list:
                    if buffer and buffer != 'vmtouch':
                        while True:
                            try:
                                shutil.copyfile(slide_path, join(buffer, os.path.basename(slide_path)))
                                q.put(slide_path)
                                break
                            except OSError:
                                time.sleep(5)
                    else:
                        q.put(slide_path)
                q.join()
                task_finished = True
                if pb: pb.end()
            else:
                for slide_path in slide_list:
                    predict_wsi_from_slide(slide_path, enable_downsample)
                if pb: pb.end()

    def resize_tfrecords(self, source_tile_px, source_tile_um, dest_tile_px, filters=None):
        '''Resizes images in a set of TFRecords to a given pixel size.

        Args:
            source_tile_px:		Pixel size of source images. Used to select source TFRecords.
            source_tile_um:		Micron size of source images. Used to select source TFRecords.
            dest_tile_px:		Pixel size of resized images.
            filters:			Dictionary of dataset filters to use for selecting TFRecords for resizing.
        '''
        log.info(f'Resizing TFRecord tiles to ({dest_tile_px}, {dest_tile_px})')
        resize_dataset = self.get_dataset(filters=filters,
                                          tile_px=source_tile_px,
                                          tile_um=source_tile_um)
        tfrecords_list = resize_dataset.get_tfrecords()
        log.info(f'Resizing {len(tfrecords_list)} tfrecords')

        for tfr in tfrecords_list:
            sfio.tfrecords.transform_tfrecord(tfr, tfr+'.transformed', resize=dest_tile_px)

    def save_project(self):
        '''Saves current project configuration as "settings.json".'''
        sfutil.write_json(self._settings, join(self.root, 'settings.json'))

    def slide_report(self, tile_px, tile_um, filters=None, filter_blank=None, dataset=None,
                        stride_div=1, destination='auto', tma=False, enable_downsample=False,
                        roi_method='inside', skip_missing_roi=True, normalizer=None, normalizer_source=None):
        '''Creates a PDF report of slides, including images of 10 example extracted tiles.

        Args:
            tile_px:				Tile width in pixels
            tile_um:				Tile width in microns
            filters:				Dataset filters to use for selecting TFRecords
            filter_blank:			List of label headers; slides that have blank entries in this label header
                                         in the annotations file will be excluded
            dataset:				Name of dataset from which to select TFRecords.
                                        If not provided, will use all project datasets
            stride_div:				Stride divisor for tile extraction
            destination:			Either 'auto' or explicit filename at which to save the PDF report
            tma:					Bool, if True, interprets slides to be TMA (tumor microarrays)
            enable_downsample:		Bool, if True, enables downsampling during tile extraction
            roi_method:				Either 'inside', 'outside', or 'ignore'.
                                        Determines how ROIs will guide tile extraction
            skip_missing_roi:		Bool, if True, will skip tiles that are missing ROIs
            normalizer:				Normalization strategy to use on image tiles
            normalizer_source:		Path to normalizer source image
        '''
        import slideflow.slide as sfslide

        if dataset: datasets = [dataset] if not isinstance(dataset, list) else dataset
        else:		datasets = self.datasets

        extracting_dataset = self.get_dataset(filters=filters,
                                              filter_blank=filter_blank,
                                              tile_px=tile_px,
                                              tile_um=tile_um)

        log.info('Generating slide report...')
        reports = []
        for dataset_name in datasets:
            roi_dir = extracting_dataset.datasets[dataset_name]['roi']
            slide_list = extracting_dataset.get_slide_paths(dataset=dataset_name)

            # Function to extract tiles from a slide
            def get_slide_report(slide_path):
                print(f'\r\033[KGenerating report for slide {sfutil.green(sfutil.path_to_name(slide_path))}...', end='')

                if tma:
                    whole_slide = sfslide.TMAReader(slide_path,
                                                    tile_px,
                                                    tile_um,
                                                    stride_div,
                                                    enable_downsample=enable_downsample,
                                                    silent=True)
                else:
                    whole_slide = sfslide.SlideReader(slide_path,
                                                      tile_px,
                                                      tile_um,
                                                      stride_div,
                                                      enable_downsample=enable_downsample,
                                                      roi_dir=roi_dir,
                                                      roi_method=roi_method,
                                                      silent=True,
                                                      skip_missing_roi=skip_missing_roi)

                if not whole_slide.loaded_correctly():
                    return

                report = whole_slide.extract_tiles(normalizer=normalizer, normalizer_source=normalizer_source)
                return report

            for slide_path in slide_list:
                report = get_slide_report(slide_path)
                reports += [report]
        print('\r\033[K', end='')
        log.info('Generating PDF (this may take some time)...', )
        pdf_report = sfslide.ExtractionReport(reports, tile_px=tile_px, tile_um=tile_um)
        timestring = datetime.now().strftime('%Y%m%d-%H%M%S')
        filename = destination if destination != 'auto' else join(self.root, f'tile_extraction_report-{timestring}.pdf')
        pdf_report.save(filename)
        log.info(f'Slide report saved to {sfutil.green(filename)}')

    def tfrecord_report(self, tile_px, tile_um, source_directory=None, filters=None, filter_blank=None, dataset=None,
                         destination='auto', normalizer=None, normalizer_source=None):
        '''Creates a PDF report of TFRecords, including 10 example tiles per TFRecord.

        Args:
            tile_px:				Tile width in pixels
            tile_um:				Tile width in microns
            filters:				Dataset filters to use for selecting TFRecords
            filter_blank:			List of label headers; slides that have blank entries in this label header
                                         in the annotations file will be excluded
            dataset:				Optional. Name of dataset from which to generate reports.
                                        Defaults to all project datasets.
            destination:			Either 'auto' or explicit filename at which to save the PDF report
            normalizer:				Normalization strategy to use on image tiles
            normalizer_source:		Path to normalizer source image
        '''
        from slideflow.slide import ExtractionReport, SlideReport
        import tensorflow as tf

        if dataset: datasets = [dataset] if not isinstance(dataset, list) else dataset
        else:		datasets = self.datasets

        if normalizer: log.info(f'Using realtime {normalizer} normalization')
        normalizer = None if not normalizer else StainNormalizer(method=normalizer, source=normalizer_source)

        if source_directory:
            tfrecord_list = [join(source_directory, t) for t in os.listdir(source_directory)
                                                       if sfutil.path_to_ext(t) == 'tfrecords']
        else:
            tfrecord_list = []
            tfrecord_dataset = self.get_dataset(filters=filters,
                                                filter_blank=filter_blank,
                                                tile_px=tile_px,
                                                tile_um=tile_um)
            for dataset_name in datasets:
                tfrecord_list += tfrecord_dataset.get_tfrecords(dataset=dataset_name)
        reports = []
        log.info('Generating TFRecords report...')
        for tfr in tfrecord_list:
            print(f'\r\033[KGenerating report for tfrecord {sfutil.green(sfutil.path_to_name(tfr))}...', end='')
            dataset = tf.data.TFRecordDataset(tfr)
            parser = sfio.tfrecords.get_tfrecord_parser(tfr, ('image_raw',), to_numpy=True, decode_images=False)
            if not parser: continue
            sample_tiles = []
            for i, record in enumerate(dataset):
                if i > 9: break
                image_raw_data = parser(record)[0]
                if normalizer:
                    image_raw_data = normalizer.jpeg_to_jpeg(image_raw_data)
                sample_tiles += [image_raw_data]
            reports += [SlideReport(sample_tiles, tfr)]

        print('\r\033[K', end='')
        log.info('Generating PDF (this may take some time)...')
        pdf_report = ExtractionReport(reports, tile_px=tile_px, tile_um=tile_um)
        timestring = datetime.now().strftime('%Y%m%d-%H%M%S')
        filename = destination if destination != 'auto' else join(self.root, f'tfrecord_report-{timestring}.pdf')
        pdf_report.save(filename)
        log.info(f'TFRecord report saved to {sfutil.green(filename)}')

    def train(self,
              outcome_label_headers,
              model_names=None,
              input_header=None,
              filters=None,
              filter_blank=None,
              resume_training=None,
              checkpoint=None,
              pretrain='imagenet',
              batch_file=None,
              hyperparameters=None,
              validation_settings=None,
              max_tiles_per_slide=0,
              min_tiles_per_slide=0,
              starting_epoch=0,
              steps_per_epoch_override=None,
              auto_extract=False,
              normalizer=None,
              normalizer_source=None,
              normalizer_strategy='tfrecord',
              use_tensorboard=False,
              multi_gpu=False,
              save_predictions=False,
              skip_metrics=False):

        '''Train model(s).

        Args:
            model_names:			Either a string model name, or an array of strings with multiple names.
                                        Required if training to a single hyperparameter combination
                                        with the "hyperparameters" argument. If performing a hyperparameter sweep,
                                        will only train models with these names in the batch_train.tsv config file.
                                        May supply None if performing a hyperparameter sweep, in which case
                                        all models in the batch_train.tsv config file will be trained.
            outcome_label_headers:	String or list. Specifies which header(s) in the annotation file to use
                                        for the output category. If a list is provided, will loop through all outcomes
                                        and perform HP sweep on each.
            filters:				Dictionary of column names mapping to column values by which to filter slides.
            resume_training:		Path to Tensorflow model to continue training
            checkpoint:				Path to cp.ckpt from which to load weights
            pretrain:				Pretrained weights to load. Default is imagenet.
                                        May supply a compatible Tensorflow model from which to load weights.
            batch_file:				Manually specify batch file to use for a hyperparameter sweep.
                                        If not specified, will use project default.
            hyperparameters:		Manually specify hyperparameter combination to use for training.
                                        If specified, will ignore batch training file.
            validation_settings:	Namespace of validation settings, provided by sf.project.get_validation_settings()
            max_tiles_per_slide:	Will only use up to this many tiles from each slide for training.
                                        If zero, will include all tiles.
            min_tiles_per_slide:	Minimum number of tiles a slide must have to be included in training.
            starting_epoch:			Starts training at the specified epoch
            steps_per_epoch_override:	If provided, will manually set the number of steps in an epoch
                                        (default epoch length is the number of total tiles)
            auto_extract:			Bool. If True, will automatically extract tiles as needed for training,
                                        without needing to explicitly call extract_tiles()
            normalizer:				Normalization strategy to use on image tiles
            normalizer_source:		Path to normalizer source image
            normalizer_strategy:	Either 'tfrecord' or 'realtime'.
                                        If 'tfrecord' & auto_extract = True, will extract tiles to TFRecords normalized.
                                        If 'realtime', normalization is performed during training.
            use_tensorboard:		Bool. If True, will add tensorboard callback for realtime training monitoring.
            multi_gpu:				Bool. If True, will train using multiple GPUs using Keras MirroredStrategy.
            save_predicitons:		Bool. If True, will save predictions with each validation.
                                        May increase validation time for large projects.
            skip_metrics:			Bool. If True, will skip metrics (ROC, AP, F1) during validation,
                                        which may improve training time for large projects.

        Returns:
            A dictionary containing model names mapped to train_acc, val_loss, and val_acc
        '''
        from slideflow.model import get_hyperparameter_combinations

        # Reconcile provided arguments with project defaults
        batch_train_file = self.batch_train_config if not batch_file else join(self.root, batch_file)
        validation_log = join(self.root, 'validation_plans.json')

        # Get default validation settings if none provided
        val_settings = validation_settings if validation_settings else get_validation_settings()

        # Quickly scan for errors (duplicate model names in batch training file) and prepare models to train
        if hyperparameters and not model_names:
            log.error("If specifying hyperparameters, 'model_names' must be supplied. ")
            return

        if normalizer and normalizer_strategy not in ('tfrecord', 'realtime'):
            log.error(f"Unknown normalizer strategy {normalizer_strategy}, must be either 'tfrecord' or 'realtime'")
            return

        if (val_settings.strategy in ('k-fold-manual', 'k-fold-preserved-site', 'k-fold', 'bootstrap')
            and val_settings.dataset):

            log.error(f'Unable to use {val_settings.strategy} if validation_dataset has been provided.')
            return

        # Setup normalization
        tfrecord_normalizer = normalizer if (normalizer and normalizer_strategy == 'tfrecord') else None
        tfrecord_normalizer_source = normalizer_source if (normalizer and normalizer_strategy == 'tfrecord') else None
        train_normalizer = normalizer if (normalizer and normalizer_strategy == 'realtime') else None
        train_normalizer_source = normalizer_source if (normalizer and normalizer_strategy == 'realtime') else None

        # Prepare hyperparameters
        log.info('Performing hyperparameter sweep...')

        hyperparameter_list = get_hyperparameter_combinations(hyperparameters, model_names, batch_train_file)

        if not isinstance(outcome_label_headers, list):
            outcome_label_headers = [outcome_label_headers]
        if len(outcome_label_headers) > 1:
            num_h = len(hyperparameter_list)
            num_o = len(outcome_label_headers)
            log.info(f'Training ({num_h} models) with {num_o} variables as simultaneous outcomes:')
            for label in outcome_label_headers:
                log.info(label)
            if log.getEffectiveLevel() <= 20: print()

        # Next, prepare the multiprocessing manager (needed to free VRAM after training and keep track of results)
        manager = multiprocessing.Manager()
        results_dict = manager.dict()
        ctx = multiprocessing.get_context('spawn')

        # For each hyperparameter combination, perform training
        for hp, hp_model_name in hyperparameter_list:

            # Prepare k-fold validation configuration
            results_log_path = os.path.join(self.root, 'results_log.csv')
            k_iter = val_settings.k_fold_iter
            k_iter = [k_iter] if (k_iter != None and not isinstance(k_iter, list)) else k_iter

            if val_settings.strategy == 'k-fold-manual':
                training_dataset = self.get_dataset(tile_px=hp.tile_px,
                                                    tile_um=hp.tile_um,
                                                    filters=filters,
                                                    filter_blank=filter_blank)

                k_fold_slide_labels, valid_k = training_dataset.slide_to_label(val_settings.k_fold_header,
                                                                               return_unique=True,
                                                                               verbose=False)
                k_fold = len(valid_k)
                log.info(f"Manual K-fold iterations detected: {', '.join(valid_k)}")
                if k_iter:
                    valid_k = [kf for kf in valid_k if (int(kf) in k_iter or kf in k_iter)]
            elif val_settings.strategy in ('k-fold', 'k-fold-preserved-site', 'bootstrap'):
                k_fold = val_settings.k_fold
                valid_k = [kf for kf in range(1, k_fold+1) if ((k_iter and kf in k_iter) or (not k_iter))]
                k_fold_slide_labels = None
            else:
                k_fold = 0
                valid_k = []
                k_fold_slide_labels = None

            if hp.model_type() != 'linear' and len(outcome_label_headers) > 1:
                #raise Exception("Multiple outcome labels only supported for linear outcome labels.")
                log.info('Using experimental multi-outcome approach for categorical outcome')

            # Auto-extract tiles if requested
            if auto_extract:
                self.extract_tiles(hp.tile_px,
                                    hp.tile_um,
                                    filters=filters,
                                    filter_blank=filter_blank,
                                    normalizer=tfrecord_normalizer,
                                    normalizer_source=tfrecord_normalizer_source)

            label_string = '-'.join(outcome_label_headers)
            model_name = f'{label_string}-{hp_model_name}'
            model_iterations = [model_name] if not k_fold else [f'{model_name}-kfold{k}' for k in valid_k]

            def start_training_process(k):
                # Using a separate process ensures memory is freed once training has completed
                process = ctx.Process(target=project_utils.trainer, args=(self,
                                                                          outcome_label_headers,
                                                                          model_name,
                                                                          results_dict,
                                                                          hp,
                                                                          val_settings,
                                                                          validation_log,
                                                                          k,
                                                                          k_fold_slide_labels,
                                                                          input_header,
                                                                          filters,
                                                                          filter_blank,
                                                                          pretrain,
                                                                          resume_training,
                                                                          checkpoint,
                                                                          max_tiles_per_slide,
                                                                          min_tiles_per_slide,
                                                                          starting_epoch,
                                                                          steps_per_epoch_override,
                                                                          train_normalizer,
                                                                          train_normalizer_source,
                                                                          use_tensorboard,
                                                                          multi_gpu,
                                                                          save_predictions,
                                                                          skip_metrics))
                process.start()
                log.info(f'Spawning training process (PID: {process.pid})')
                process.join()

            # Perform training
            log.info('Training model...')
            if k_fold:
                for k in valid_k:
                    start_training_process(k)

            else:
                start_training_process(None)

            # Record results
            for mi in model_iterations:
                if mi not in results_dict:
                    log.error(f'Training failed for model {model_name}')
                else:
                    sfutil.update_results_log(results_log_path, mi, results_dict[mi]['epochs'])
            log.info(f'Training complete for model {model_name}, results saved to {sfutil.green(results_log_path)}')

        # Print summary of all models
        log.info('Training complete; validation accuracies:')
        for model in results_dict:
            try:
                last_epoch = max([int(e.split('epoch')[-1]) for e in results_dict[model]['epochs'].keys()
                                                            if 'epoch' in e ])
                final_train_metrics = results_dict[model]['epochs'][f'epoch{last_epoch}']['train_metrics']
                final_val_metrics = results_dict[model]['epochs'][f'epoch{last_epoch}']['val_metrics']
                log.info(f'{sfutil.green(model)} training metrics:')
                for m in final_train_metrics:
                    log.info(f'{m}: {final_train_metrics[m]}')
                log.info(f'{sfutil.green(model)} validation metrics:')
                for m in final_val_metrics:
                    log.info(f'{m}: {final_val_metrics[m]}')
            except ValueError:
                pass

        return results_dict

    def train_clam(self,
                   exp_name,
                   pt_files,
                   outcome_label_headers,
                   tile_px,
                   tile_um,
                   train_slides='auto',
                   validation_slides='auto',
                   filters=None,
                   filter_blank=None,
                   clam_args=None,
                   attention_heatmaps=True):

        '''Using a trained model, generate feature activations and train a CLAM model.

        Args:
            exp_name:				Name of experiment. Will make clam/{exp_name} subfolder.
            pt_files:				Path to pt_files containing tile-level features.
            outcome_label_headers:	Name in annotation column which specifies the outcome label.
            tile_px:				Tile width in pixels.
            tile_um:				Tile width in microns.
            train_slides:			List of slide names for training.
                                        If 'auto' (default), will auto-generate training/validation split.
            validation_slides:		List of slide names for training. If 'auto' (default),
                                        will auto-generate training/validation split.
            filters:				Dictionary of column names mapping to column values by which to filter slides.
                                        Used if train_slides and validation_slides are 'auto'.
            filter_blank:			List of annotations headers; slides blank in this column will be excluded.
                                        Used if train_slides and validation_slides are 'auto'.
            clam_args:				Dictionary with clam arguments, as provided by sf.clam.get_args()
            attention_heatmaps:		Bool. If true, will save attention heatmaps of validation dataset.

        Returns:
            None
        '''

        import slideflow.clam as clam
        from slideflow.clam.datasets.dataset_generic import Generic_MIL_Dataset
        from slideflow.clam.create_attention import export_attention

        # Set up CLAM experiment data directory
        clam_dir = join(self.root, 'clam', exp_name)
        results_dir = join(clam_dir, 'results')
        if not exists(results_dir): os.makedirs(results_dir)

        # Detect number of features automatically from saved pt_files
        pt_file_paths = [p for p in os.listdir(pt_files) if sfutil.path_to_ext(join(pt_files, p)) == 'pt']
        num_features = clam.detect_num_features(pt_file_paths[0])

        # Set up outcomes for CLAM model
        dataset = self.get_dataset(tile_px=tile_px,
                                   tile_um=tile_um,
                                   filters=filters,
                                   filter_blank=filter_blank)

        # Note: CLAM only supports categorical outcomes
        slide_labels, unique_labels = dataset.get_labels_from_annotations(outcome_label_headers,
                                                                          use_float=False,
                                                                          key='outcome_label')

        if train_slides == validation_slides == 'auto':
            validation_log = join(self.root, 'validation_plans.json')
            train_tfrecords, eval_tfrecords = sfio.tfrecords.get_train_and_val_tfrecords(dataset,
                                                                                 validation_log,
                                                                                 'categorical',
                                                                                 slide_labels,
                                                                                 outcome_key='outcome_label',
                                                                                 val_target='per-patient',
                                                                                 val_strategy='k-fold',
                                                                                 val_k_fold=clam_args.k,
                                                                                 k_fold_iter=0) # TODO fix this
            train_slides = [sfutil.path_to_name(t) for t in train_tfrecords]
            validation_slides = [sfutil.path_to_name(v) for v in eval_tfrecords]

        # Remove slides without associated .pt files
        num_supplied_slides = len(train_slides) + len(validation_slides)
        train_slides = [s for s in train_slides if exists(join(pt_files, s+'.pt'))]
        validation_slides = [s for s in validation_slides if exists(join(pt_files, s+'.pt'))]
        if len(train_slides) + len(validation_slides) != num_supplied_slides:
            num_skipped = num_supplied_slides - (len(train_slides) + len(validation_slides))
            log.warn(f'Skipping {num_skipped} slides missing associated .pt files.')

        # Set up training/validation splits (mirror base model)
        split_dir = join(clam_dir, 'splits')
        if not exists(split_dir): os.makedirs(split_dir)

        header = ['','train','val','test']
        with open(join(split_dir, 'splits_0.csv'), 'w') as splits_file:
            writer = csv.writer(splits_file)
            writer.writerow(header)
            # Currently, the below sets the validation & test sets to be the same
            for i in range(max(len(train_slides), len(validation_slides))):
                row = [i]
                if i < len(train_slides): 		row += [train_slides[i]]
                else: 							row += ['']
                if i < len(validation_slides):	row += [validation_slides[i], validation_slides[i]]
                else:							row += ['', '']
                writer.writerow(row)

        # Set up CLAM args/settings
        if not clam_args:
            clam_args = clam.get_args()

        # Assign CLAM settings based on this project
        clam_args.model_size = [num_features, 256, 128]
        clam_args.results_dir = results_dir
        clam_args.n_classes = len(unique_labels)
        clam_args.split_dir = split_dir
        clam_args.data_root_dir = pt_files

        # Save clam settings
        sfutil.write_json(clam_args, join(clam_dir, 'experiment.json'))

        # Create CLAM dataset
        clam_dataset = Generic_MIL_Dataset(csv_path=self.annotations,
                                           data_dir=pt_files,
                                           shuffle=False,
                                           seed=clam_args.seed,
                                           print_info=True,
                                           label_col = outcome_label_headers,
                                           label_dict = dict(zip(unique_labels, range(len(unique_labels)))),
                                           patient_strat=False,
                                           ignore=[])

        # Run CLAM
        clam.main(clam_args, clam_dataset)

        # Get attention from trained model on validation set
        attention_tfrecords = [tfr for tfr in dataset.get_tfrecords() if sfutil.path_to_name(tfr) in validation_slides]
        for ki in range(clam_args.k):
            attention_dir = join(clam_dir, 'attention', str(ki))
            if not exists(attention_dir): os.makedirs(attention_dir)
            export_attention(clam_args,
                             ckpt_path=join(results_dir, f's_{ki}_checkpoint.pt'),
                             export_dir=attention_dir,
                             pt_files=pt_files,
                             slides=validation_slides,
                             reverse_label_dict = dict(zip(range(len(unique_labels)), unique_labels)),
                             slide_to_label = {s:slide_labels[s]['outcome_label'] for s in slide_labels})
            if attention_heatmaps:
                heatmaps_dir = join(clam_dir, 'attention_heatmaps', str(ki))
                if not exists(heatmaps_dir): os.makedirs(heatmaps_dir)

                for tfr in attention_tfrecords:
                    attention_dict = {}
                    slide = sfutil.path_to_name(tfr)
                    try:
                        with open(join(attention_dir, slide+'.csv'), 'r') as csv_file:
                            reader = csv.reader(csv_file)
                            for row in reader:
                                attention_dict.update({int(row[0]): float(row[1])})
                    except FileNotFoundError:
                        print(f'Unable to find attention scores for slide {slide}, skipping')
                        continue
                    self.generate_tfrecord_heatmap(tfr, attention_dict, heatmaps_dir, tile_px=tile_px, tile_um=tile_um)

    def split_tfrecords_by_roi(self, tile_px, tile_um, destination, filters=None, filter_blank=None):
        from slideflow.slide import SlideReader
        import slideflow.io.tfrecords
        import tensorflow as tf

        dataset = self.get_dataset(tile_px, tile_um, filters=filters, filter_blank=filter_blank)
        tfrecords = dataset.get_tfrecords()
        slides = {sfutil.path_to_name(s):s for s in dataset.get_slide_paths()}
        rois = dataset.get_rois()
        manifest = dataset.get_manifest()

        for tfr in tfrecords:
            slidename = sfutil.path_to_name(tfr)
            if slidename not in slides:
                continue
            slide = SlideReader(slides[slidename], tile_px, tile_um, roi_list=rois)
            if slide.load_error:
                continue
            feature_description, _ = sf.io.tfrecords.detect_tfrecord_format(tfr)
            parser = sf.io.tfrecords.get_tfrecord_parser(tfr, ('loc_x', 'loc_y'), to_numpy=True)
            reader = tf.data.TFRecordDataset(tfr)
            if not exists(join(destination, 'inside')):
                os.makedirs(join(destination, 'inside'))
            if not exists(join(destination, 'outside')):
                os.makedirs(join(destination, 'outside'))
            inside_roi_writer = tf.io.TFRecordWriter(join(destination, 'inside', f'{slidename}.tfrecords'))
            outside_roi_writer = tf.io.TFRecordWriter(join(destination, 'outside', f'{slidename}.tfrecords'))
            for record in tqdm(reader, total=manifest[tfr]['total']):
                loc_x, loc_y = parser(record)
                tile_in_roi = any([annPoly.contains(sg.Point(loc_x, loc_y)) for annPoly in slide.annPolys])
                record_bytes = sf.io.tfrecords._read_and_return_record(record, feature_description)
                if tile_in_roi:
                    inside_roi_writer.write(record_bytes)
                else:
                    outside_roi_writer.write(record_bytes)
            inside_roi_writer.close()
            outside_roi_writer.close()

    def visualize_tiles(self, model, node, tfrecord_dict=None, directory=None, mask_width=None,
                        normalizer=None, normalizer_source=None):
        '''Visualizes node activations across a set of image tiles through progressive convolutional masking.

        Args:
            model:				Path to Tensorflow model
            node:				Int, node to analyze
            tfrecord_dict:		Dictionary mapping tfrecord paths to tile indices.
                                    Visualization will be performed on these tiles.
            directory:			Directory in which to save images.
            mask_width:			Width of mask to convolutionally apply. Defaults to 1/6 of tile_px
            normalizer:				Normalization strategy to use on image tiles.
            normalizer_source:		Path to normalizer source image.
        '''
        from slideflow.activations import TileVisualizer

        hp_data = sfutil.get_model_hyperparameters(model)
        tile_px = hp_data['hp']['tile_px']
        TV = TileVisualizer(model=model,
                            node=node,
                            tile_px=tile_px,
                            mask_width=mask_width,
                            normalizer=normalizer,
                            normalizer_source=normalizer_source)

        if tfrecord_dict:
            for tfrecord in tfrecord_dict:
                for tile_index in tfrecord_dict[tfrecord]:
                    TV.visualize_tile(tfrecord=tfrecord, index=tile_index, export_folder=directory)

        else:
            tiles = [o for o in os.listdir(directory) if not isdir(join(directory, o))]
            tiles.sort(key=lambda x: int(x.split('-')[0]))
            tiles.reverse()
            for tile in tiles[:20]:
                tile_loc = join(directory, tile)
                TV.visualize_tile(image_jpg=tile_loc, export_folder=directory)
