import os
import types
import re
import json
import logging
import itertools
import csv
import copy
import pickle
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.colors as mcol

from os.path import join, exists, isdir, basename
from statistics import mean, median
from tqdm import tqdm

import slideflow as sf
import slideflow.util.neptune_utils

from slideflow import project_utils
from slideflow.dataset import Dataset
from slideflow.util import log
from slideflow.project_utils import get_validation_settings

try:
    import git
except ImportError: # git is not needed for pypi distribution
    git = None

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Project:
    """Assists with project / dataset organization and execution of pipeline functions.

    Standard instantiation with __init__ assumes a project already exists at a given directory,
    or that project configuration will be supplied via kwargs. Alternatively, a project may be instantiated
    using :meth:`from_prompt`, which interactively guides users through configuration.

    *Interactive instantiation:*

    .. code-block:: python

        >>> import slideflow as sf
        >>> P = sf.Project.from_prompt('/project/path')
        What is the project name?

    *Manual configuration:*

    .. code-block:: python

        >>> import slideflow as sf
        >>> P = sf.Project('/project/path', name=..., ...)

    """

    def __init__(self, root, gpu=None, use_neptune=False, **project_kwargs):
        """Initializes project at the specified project folder, creating a new project using
        the specified kwargs if one does not already exist. Will create a blank annotations file with
        dataset slide names if one does not exist.

        Args:
            root (str): Path to project directory.
            gpu (str, optional): Manually assign GPU. Comma separated int. Defaults to None.

        Keyword Args:
            name (str): Project name. Defaults to 'MyProject'.
            annotations (str): Path to annotations CSV file. Defaults to './annotations.csv'
            dataset_config (str): Path to dataset configuration JSON file. Defaults to './datasets.json'.
            sources (list(str)): List of dataset sources to include in project. Defaults to 'source1'.
            models_dir (str): Path to directory in which to save models. Defaults to './models'.
            eval_dir (str): Path to directory in which to save evaluations. Defaults to './eval'.
            mixed_precision (bool): Use mixed precision for training. Defaults to True.

        Raises:
            slideflow.util.UserError: if the project folder does not exist, or the folder exists but
                kwargs are provided.
        """

        self.root = root

        if exists(join(root, 'settings.json')) and project_kwargs:
            raise sf.util.UserError(f"Project already exists at {root}. " + \
                                   f"Unable to override user-provided settings: {', '.join(project_kwargs.keys())}")
        elif exists(join(root, 'settings.json')):
            self.load_project(root)
        elif project_kwargs:
            log.info(f"Creating project at {root}...")
            self._settings = project_utils._project_config(**project_kwargs)
            if not exists(root):
                os.makedirs(root)
            self.save()
        else:
            raise sf.util.UserError(f"Project folder {root} does not exist.")

        # Create directories, if not already made
        if not exists(self.models_dir):
            os.makedirs(self.models_dir)
        if not exists(self.eval_dir):
            os.makedirs(self.eval_dir)

        # Create blank annotations file if one does not exist
        if not exists(self.annotations) and exists(self.dataset_config):
            self.create_blank_annotations()

        # Set up logging
        logger = logging.getLogger('slideflow')
        self.verbosity = logger.getEffectiveLevel()
        fh = logging.FileHandler(join(root, 'log.txt'))
        fh.setFormatter(sf.util.FileFormatter())
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        # Neptune
        self.use_neptune = use_neptune

        if gpu is not None:
            self.select_gpu(gpu)

    @classmethod
    def from_prompt(cls, root, **kwargs):
        """Initializes project by creating project folder, prompting user for project settings, and
        saves settings to "settings.json" within the project directory.

        Args:
            root (str): Path to project directory.
            gpu (str, optional): Manually assign GPU. Comma separated int. Defaults to None.
        """

        if not exists(join(root, 'settings.json')):
            log.info(f'Project at "{root}" does not exist; will set up new project.')
            project_utils.interactive_project_setup(root)
        obj = cls(root, **kwargs)
        return obj

    def __repr__(self):
        with_neptune = '' if not self.use_neptune else ", use_neptune={!r}".format(self.use_neptune)
        return "Project(root={!r}{})".format(self.root, with_neptune)

    @property
    def annotations(self):
        """Path to annotations file."""
        return self._read_relative_path(self._settings['annotations'])

    @annotations.setter
    def annotations(self, val):
        if not isinstance(val, str):
            raise sf.util.UserError("'annotations' must be a str (path to annotations file)")
        self._settings['annotations'] = val

    @property
    def dataset_config(self):
        """Path to dataset configuration JSON file."""
        return self._read_relative_path(self._settings['dataset_config'])

    @dataset_config.setter
    def dataset_config(self, val):
        if not isinstance(val, str):
            raise sf.util.UserError("'dataset_config' must be a str (path to dataset config JSON file)")
        self._settings['dataset_config'] = val

    @property
    def eval_dir(self):
        """Path to evaluation directory."""
        if 'eval_dir' not in self._settings:
            log.warning("Please add eval_dir to project settings.json. Assuming ./eval")
            return self._read_relative_path('./eval')
        else:
            return self._read_relative_path(self._settings['eval_dir'])

    @eval_dir.setter
    def eval_dir(self, val):
        if not isinstance(val, str):
            raise sf.util.UserError("'eval_dir' must be a str (path to evaluation directory)")
        self._settings['eval_dir'] = val

    @property
    def mixed_precision(self):
        """Returns bool indicating whether mixed precision should be used for this project."""
        if 'mixed_precision' in self._settings:
            return self._settings['mixed_precision']
        elif 'use_fp16' in self._settings:
            log.warn("'mixed_precision' not found in project settings. Please update the settings.json file.")
            return self._settings['use_fp16']

    @mixed_precision.setter
    def mixed_precision(self, val):
        if not isinstance(val, bool):
            raise sf.util.UserError("'mixed_precision' must be a bool")
        self._settings['mixed_precision'] = val

    @property
    def models_dir(self):
        """Path to models directory."""
        return self._read_relative_path(self._settings['models_dir'])

    @models_dir.setter
    def models_dir(self, val):
        if not isinstance(val, str):
            raise sf.util.UserError("'models_dir' must be a str (path to models directory)")
        self._settings['models_dir'] = val

    @property
    def name(self):
        """Descriptive project name."""
        return self._settings['name']

    @name.setter
    def name(self, val):
        if not isinstance(val, str):
            raise sf.util.UserError("'name' must be a str")
        self._settings['name'] = val

    @property
    def neptune_workspace(self):
        """Neptune workspace name."""

        if 'neptune_workspace' in self._settings:
            return self._settings['neptune_workspace']
        elif 'NEPTUNE_WORKSPACE' in os.environ:
            return os.environ['NEPTUNE_WORKSPACE']
        else:
            return None

    @neptune_workspace.setter
    def neptune_workspace(self, name):
        """Neptune workspace name."""

        if not isinstance(name, str):
            raise sf.util.UserError('Neptune workspace name must be a string.')
        self._settings['neptune_workspace'] = name

    @property
    def neptune_api(self):
        """Neptune API token."""

        if 'neptune_api' in self._settings:
            return self._settings['neptune_api']
        elif 'NEPTUNE_API_TOKEN' in os.environ:
            return os.environ['NEPTUNE_API_TOKEN']
        else:
            return None

    @neptune_api.setter
    def neptune_api(self, api_token):
        """Neptune API token."""

        if not isinstance(api_token, str):
            raise sf.util.UserError('API token must be a string.')
        self._settings['neptune_api'] = api_token

    @property
    def sources(self):
        """Returns list of dataset sources active in this project."""
        if 'sources' in self._settings:
            return self._settings['sources']
        elif 'datasets' in self._settings:
            log.warn("'sources' not found in project settings. Please update settings.json (previously 'datasets').")
            return self._settings['datasets']

    @sources.setter
    def sources(self, val):
        if not isinstance(val, list) or any([not isinstance(v, str) for v in val]):
            raise sf.util.UserError("'sources' must be a list of str")
        self._settings['sources'] = val

    def _read_relative_path(self, path):
        """Converts relative path within project directory to global path."""
        return sf.util.relative_path(path, self.root)

    def select_gpu(self, gpu):
        """Sets CUDA_VISIBLE_DEVICES in order to restrict GPU access to the given devices.

        Args:
            gpu (str): String indicating what the CUDA_VISIBLE_DEVICES should be set to.

        Raises:
            ValueError: if gpu is not a string
        """

        if not isinstance(gpu, str):
            raise ValueError(f'Invalid option {gpu}; must supply string (e.g. "0", "0,1", "-1")')
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        if gpu == '-1':
            log.warn(f'Disabling GPU access.')
        else:
            log.info(f'Using GPU: {gpu}')

    def add_source(self, name, slides, roi, tiles, tfrecords, path=None):
        """Adds a dataset source to the dataset configuration file.

        Args:
            name (str): Dataset source name.
            slides (str): Path to directory containing slides.
            roi (str): Path to directory containing CSV ROIs.
            tiles (str): Path to directory in which to store extracted tiles.
            tfrecords (str): Path to directory in which to store TFRecords of extracted tiles.
            path (str, optional): Path to dataset configuration file. Defaults to None.
                If not provided, uses project default.
        """

        if not path:
            path = self.dataset_config
        project_utils.add_source(name, slides, roi, tiles, tfrecords, path)
        if name not in self.sources:
            self.sources += [name]
        self.save()

    def associate_slide_names(self):
        """Automatically associate patient names with slide filenames in the annotations file."""
        dataset = self.dataset(tile_px=0, tile_um=0, verification=None)
        dataset.update_annotations_with_slidenames(self.annotations)

    def create_blank_annotations(self, filename=None):
        """Creates an empty annotations file with slide names from project settings, assuming one slide per patient.

        Args:
            filename (str): Annotations file destination. If not provided, will use project default.
        """

        if filename is None:
            filename = self.annotations
        if exists(filename):
            raise sf.util.UserError(f"Unable to create blank annotations file at {filename}; file already exists.")
        if not exists(self.dataset_config):
            raise sf.util.UserError("Unable to create blank annotations file, dataset configuration file " +
                                    f"{self.dataset_config} does not exist.")

        dataset = Dataset(config=self.dataset_config,
                          sources=self.sources,
                          tile_px=None,
                          tile_um=None,
                          annotations=None)

        slides = [sf.util.path_to_name(s) for s in dataset.slide_paths(apply_filters=False)]
        with open(filename, 'w') as csv_outfile:
            csv_writer = csv.writer(csv_outfile, delimiter=',')
            header = [sf.util.TCGA.patient, 'dataset', 'category']
            csv_writer.writerow(header)
            for slide in slides:
                csv_writer.writerow([slide, '', ''])
        log.info(f"Wrote blank annotations file to {sf.util.green(filename)}")

    def create_hyperparameter_sweep(self, filename='sweep.json', label=None, **kwargs):
        """Prepares a hyperparameter sweep, saving to a batch train TSV file.

        Args:
            label (str, optional): Label to use when naming models in sweep. Defaults to None.
            filename (str, optional): Filename for hyperparameter sweep. Overwrites existing files. Saves in project
                root directory. Defaults to "sweep.json".
        """

        pdict = copy.deepcopy({k:v for k,v in kwargs.items() if k != 'epochs'})
        args = list(pdict.keys())
        for arg in args:
            if not isinstance(pdict[arg], list):
                pdict[arg] = [pdict[arg]]
        argsv = list(pdict.values())
        sweep = list(itertools.product(*argsv))

        label = '' if not label else f'{label}-'
        hp_list = []
        for i, params in enumerate(sweep):
            full_params = dict(zip(args, list(params)))
            if 'epochs' in kwargs:
                full_params['epochs'] = kwargs['epochs']
            mp = sf.model.ModelParams(**full_params)
            hp_list += [{ f'{label}HPSweep{i}': mp.get_dict() }]
        sf.util.write_json(hp_list, os.path.join(self.root, filename))
        log.info(f'Wrote {len(sweep)} combinations for sweep to {sf.util.green(filename)}')

    def evaluate(self, model, outcome_label_headers, dataset=None, filters=None, checkpoint=None, model_config=None,
                 eval_k_fold=None, splits="splits.json", max_tiles=0, min_tiles=0, batch_size=64, input_header=None,
                 permutation_importance=False, histogram=False, save_predictions=False):

        """Evaluates a saved model on a given set of tfrecords.

        Args:
            model (str): Path to model to evaluate.
            outcome_label_headers (str): Str or list of str. Annotation column header specifying the outcome label(s).
            dataset (:class:`slideflow.dataset.Dataset`, optional): Dataset object from which to generate activations.
                If not supplied, will calculate activations for all project tfrecords at the tile_px/tile_um
                matching the supplied model, optionally using provided filters and filter_blank.
            filters (dict, optional): Filters dict to use when selecting tfrecords. Defaults to None.
                See :meth:`get_dataset` documentation for more information on filtering.
            checkpoint (str, optional): Path to cp.ckpt file, if evaluating a saved checkpoint. Defaults to None.
            model_config (str, optional): Path to model's config.json file. Defaults to None.
                If None (default), searches in the model directory.
            eval_k_fold (int, optional): K-fold iteration number to evaluate. Defaults to None.
                If None, will evaluate all tfrecords irrespective of K-fold.
            splits (str, optional): Filename of JSON file in which to log training/validation splits. Looks for
                filename in project root directory. Defaults to "splits.json".
            max_tiles (int, optional): Maximum number of tiles from each slide to evaluate. Defaults to 0.
                If zero, will include all tiles.
            min_tiles (int, optional): Minimum number of tiles a slide must have to be included in evaluation.
                Defaults to 0. Recommend considering a minimum of at least 10 tiles per slide.
            input_header (str, optional): Annotation column header to use as additional input. Defaults to None.
            permutation_importance (bool, optional): Calculate the permutation feature importance.  Determine relative
                importance when using multiple model inputs. Only available for Tensorflow backend. Defaults to False.
            histogram (bool, optional): Create tile-level histograms for each class. Defaults to False.
            save_predictions (bool or str, optional): Either True, False, or any combination of 'tile', 'patient',
                or 'slide', either as string or list of strings. Save tile-level, patient-level, and/or
                slide-level predictions. If True, will save all.

        Returns:
            Dict: Dictionary of keras training results, nested by epoch.
        """

        import slideflow.model
        log.info(f'Evaluating model {sf.util.green(model)}...')

        if not isinstance(outcome_label_headers, list):
            outcome_label_headers = [outcome_label_headers]

        if (input_header is None) and permutation_importance:
            log.warn('Permutation feature importance is designed to be used with multimodal models. Turning off.')
            permutation_importance = False

        log.setLevel(self.verbosity)

        # Load hyperparameters from saved model
        if model_config:
            config = sf.util.load_json(model_config)
        else:
            config = sf.util.get_model_config(model)
            if not config:
                raise OSError(f"Unable to find configuration file for model {model}.")
        hp = sf.model.ModelParams()
        hp.load_dict(config['hp'])
        model_name = f"eval-{basename(model)}"
        splits_file = join(self.root, splits)

        # Filter out slides that are blank in the outcome label, or blank in any of the input_header categories
        filter_blank = [o for o in outcome_label_headers]
        if input_header:
            input_header = [input_header] if not isinstance(input_header, list) else input_header
            filter_blank += input_header

        # Load dataset and annotations for evaluation
        if dataset is None:
            dataset = self.dataset(tile_px=hp.tile_px,
                                        tile_um=hp.tile_um,
                                        filters=filters,
                                        filter_blank=filter_blank,
                                        min_tiles=min_tiles)
        else:
            if dataset.tile_px != hp.tile_px or dataset.tile_um != hp.tile_um:
                raise ValueError(f"Dataset tile size ({dataset.tile_px}px, {dataset.tile_um}um) does not match " + \
                                 f"model ({hp.tile_px}px, {hp.tile_um}um)")
            dataset.filter_blank = filter_blank

        # Set up outcome labels
        if hp.model_type() == 'categorical':
            if len(outcome_label_headers) == 1 and outcome_label_headers[0] not in config['outcome_labels']:
                outcome_label_to_int = {outcome_label_headers[0]: {v: int(k) for k, v in config['outcome_labels'].items()}}
            else:
                outcome_label_to_int = {o:
                                            { v: int(k) for k, v in config['outcome_labels'][o].items() }
                                        for o in config['outcome_labels']}
        else:
            outcome_label_to_int = None

        use_float = (hp.model_type() in ['linear', 'cph'])
        labels, unique_labels = dataset.labels(outcome_label_headers,
                                               use_float=use_float,
                                               assigned_labels=outcome_label_to_int)

        if hp.model_type() == 'categorical' and len(outcome_label_headers) > 1:
            def process_outcome_label(v):
                return '-'.join(map(str, v)) if isinstance(v, list) else v
            labels_for_split = {k: process_outcome_label(v) for k,v in labels.items()}
        else:
            labels_for_split = labels

        # If using a specific k-fold, load validation plan
        if eval_k_fold:
            log.info(f"Using {sf.util.bold('k-fold iteration ' + str(eval_k_fold))}")
            _, eval_dts = dataset.training_validation_split(hp.model_type(),
                                                            labels_for_split,
                                                            val_strategy=config['validation_strategy'],
                                                            splits=splits_file,
                                                            val_fraction=config['validation_fraction'],
                                                            val_k_fold=config['validation_k_fold'],
                                                            k_fold_iter=eval_k_fold)
        # Otherwise use all TFRecords
        else:
            eval_dts = dataset

        # Set max tiles
        eval_dts = eval_dts.clip(max_tiles)

        # Prepare additional slide-level input
        model_inputs = {}
        if input_header:
            input_header = [input_header] if not isinstance(input_header, list) else input_header
            feature_len_dict = {}   # Dict mapping input_vars to total number of different labels for each input header
            input_labels_dict = {}  # Dict mapping input_vars to nested dictionaries,
                                    #    which map category ID to category label names (for categorical variables)
                                    #     or mapping to 'float' for float variables
            for slide in labels:
                model_inputs[slide] = []

            for input_var in input_header:
                # Check if variable can be converted to float (default). If not, will assume categorical.
                try:
                    dataset.labels(input_var, use_float=True)
                    is_float = True
                except TypeError:
                    is_float = False
                log.info(f"Adding input variable {sf.util.blue(input_var)} as {'float' if is_float else 'categorical'}")

                if is_float:
                    input_labels, _ = dataset.labels(input_var, use_float=is_float)
                    for slide in input_labels:
                        model_inputs[slide] += input_labels[slide]
                    input_labels_dict[input_var] = 'float'
                    feature_len_dict[input_var] = 1
                else:
                    # Read categorical variable assignments from hyperparameter file
                    input_label_to_int = {v: int(k) for k, v in config['input_feature_labels'][input_var].items()}
                    input_labels, _ = dataset.labels(input_var,
                                                     use_float=is_float,
                                                     assigned_labels=input_label_to_int)
                    feature_len_dict[input_var] = len(input_label_to_int)
                    input_labels_dict[input_var] = config['input_feature_labels'][input_var]

                    for slide in labels:
                        onehot_label = sf.util.to_onehot(input_labels[slide], feature_len_dict[input_var]).tolist()
                        model_inputs[slide] += onehot_label

            feature_sizes = [feature_len_dict[i] for i in input_header]

        else:
            input_labels_dict = None
            feature_sizes = None

        if feature_sizes and (sum(feature_sizes) != sum(config['input_feature_sizes'])):
            #TODO: consider using training matrix
            raise Exception(f'Patient-level feature matrix (size {sum(feature_sizes)}) not equal to what was used ' + \
                            f'for model training (size {sum(config["input_feature_sizes"])}).')
            #feature_sizes = config['feature_sizes']
            #feature_names = config['feature_names']
            #num_slide_features = sum(config['feature_sizes'])

        # Set up model for evaluation
        # Using the project annotation file, assemble list of slides for training,
        # as well as the slide annotations dictionary (output labels)
        prev_run_dirs = [x for x in os.listdir(self.eval_dir) if isdir(join(self.eval_dir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        model_dir = join(self.eval_dir, f'{cur_run_id:05d}-{model_name}')
        assert not exists(model_dir)
        os.makedirs(model_dir)

        # Log model settings and hyperparameters
        outcome_labels = None if hp.model_type() != 'categorical' else dict(zip(range(len(unique_labels)), unique_labels))

        # Check git commit if running from source
        git_commit = None
        if git is not None:
            try:
                git_commit = git.Repo(search_parent_directories=True).head.object.hexsha
            except:
                pass

        hp_file = join(model_dir, 'params.json')

        config = {
            'slideflow_version': sf.__version__,
            'project': self.name,
            'git_commit': git_commit,
            'model_name': model_name,
            'model_path': model,
            'stage': 'evaluation',
            'tile_px': hp.tile_px,
            'tile_um': hp.tile_um,
            'model_type': hp.model_type(),
            'outcome_label_headers': outcome_label_headers,
            'input_features': input_header,
            'input_feature_sizes': feature_sizes,
            'input_feature_labels': input_labels_dict,
            'outcome_labels': outcome_labels,
            'dataset_config': self.dataset_config,
            'sources': self.sources,
            'annotations': self.annotations,
            'validation_strategy': config['validation_strategy'],
            'validation_fraction': config['validation_fraction'],
            'validation_k_fold': config['validation_k_fold'],
            'k_fold_i': eval_k_fold,
            'filters': filters,
            'pretrain': None,
            'resume_training': None,
            'checkpoint': checkpoint,
            'hp': hp.get_dict(),
            'max_tiles': max_tiles,
            'min_tiles': min_tiles,
        }
        sf.util.write_json(config, hp_file)

        # Perform evaluation
        log.info(f'Evaluating {sf.util.bold(len(eval_dts.tfrecords()))} tfrecords')

        # Build a model using the slide list as input and the annotations dictionary as output labels
        trainer = sf.model.trainer_from_hp(hp,
                                           outdir=model_dir,
                                           labels=labels,
                                           patients=dataset.patients(),
                                           slide_input=model_inputs,
                                           manifest=dataset.manifest(),
                                           mixed_precision=self.mixed_precision,
                                           feature_names=input_header,
                                           feature_sizes=feature_sizes,
                                           outcome_names=outcome_label_headers,
                                           use_neptune=self.use_neptune,
                                           neptune_api=self.neptune_api,
                                           neptune_workspace=self.neptune_workspace,)
        if isinstance(model, str):
            trainer.load(model)
        if checkpoint:
            trainer.model = hp.build_model(labels=labels, num_slide_features=(0 if not feature_sizes else sum(feature_sizes)))
            trainer.model.load_weights(checkpoint)

        results = trainer.evaluate(dataset=eval_dts,
                                   batch_size=batch_size,
                                   permutation_importance=permutation_importance,
                                   histogram=histogram,
                                   save_predictions=save_predictions)

        return results

    def evaluate_clam(self, exp_name, pt_files, outcome_label_headers, tile_px, tile_um, k=0, eval_tag=None,
                        filters=None, filter_blank=None, attention_heatmaps=True):
        """Evaluate CLAM model on saved feature activations and export attention heatmaps.

        Args:
            exp_name (str): Name of experiment to evaluate (directory in clam/ subfolder)
            pt_files (str): Path to pt_files containing tile-level features.
            outcome_label_headers (str or list): Name in annotation column which specifies the outcome label.
            tile_px (int): Tile width in pixels.
            tile_um (int): Tile width in microns.
            k (int, optional): K-fold / split iteration to evaluate. Defaults to 0.
                Evaluates the model saved as s_{k}_checkpoint.pt in the CLAM results folder.
            eval_tag (str, optional): Unique identifier for this evaluation. Defaults to None
            filters (dict, optional): Filters dict to use when selecting tfrecords. Defaults to None.
                See :meth:`get_dataset` documentation for more information on filtering.
            filter_blank (list, optional): Slides blank in these columns will be excluded. Defaults to None.
            attention_heatmaps (bool, optional): Save attention heatmaps of validation dataset. Defaults to True.

        Returns:
            None
        """

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

        log.info(f'Loading trained experiment from {sf.util.green(exp_name)}, k={k}')
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
        ckpt_path = join(exp_name, 'results', f's_{k}_checkpoint.pt')
        eval_dir = join(eval_dir, eval_tag)
        if not exists(eval_dir): os.makedirs(eval_dir)
        args_dict = sf.util.load_json(join(exp_name, 'experiment.json'))
        args = types.SimpleNamespace(**args_dict)
        args.save_dir = eval_dir

        dataset = self.dataset(tile_px=tile_px,
                                   tile_um=tile_um,
                                   filters=filters,
                                   filter_blank=filter_blank)

        evaluation_slides = [s for s in dataset.slides() if exists(join(pt_files, s+'.pt'))]
        dataset = dataset.filter(filters={'slide': evaluation_slides})

        slide_labels, unique_labels = dataset.labels(outcome_label_headers, use_float=False)

        # Set up evaluation annotations file based off existing pt_files
        outcome_dict = dict(zip(range(len(unique_labels)), unique_labels))
        with open(join(eval_dir, 'eval_annotations.csv'), 'w') as eval_file:
            writer = csv.writer(eval_file)
            header = [sf.util.TCGA.patient, sf.util.TCGA.slide, outcome_label_headers]
            writer.writerow(header)
            for slide in evaluation_slides:
                row = [slide, slide, outcome_dict[slide_labels[slide]]]
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
        attention_tfrecords = dataset.tfrecords()
        attention_dir = join(eval_dir, 'attention')
        if not exists(attention_dir): os.makedirs(attention_dir)
        export_attention(args_dict,
                         ckpt_path=ckpt_path,
                         outdir=attention_dir,
                         pt_files=pt_files,
                         slides=dataset.slides(),
                         reverse_labels=dict(zip(range(len(unique_labels)), unique_labels)),
                         labels=slide_labels)
        if attention_heatmaps:
            heatmaps_dir = join(eval_dir, 'attention_heatmaps')
            if not exists(heatmaps_dir): os.makedirs(heatmaps_dir)

            for tfr in attention_tfrecords:
                attention_dict = {}
                slide = sf.util.path_to_name(tfr)
                try:
                    with open(join(attention_dir, slide+'.csv'), 'r') as csv_file:
                        reader = csv.reader(csv_file)
                        for row in reader:
                            attention_dict.update({int(row[0]): float(row[1])})
                except FileNotFoundError:
                    print(f'Unable to find attention scores for slide {slide}, skipping')
                    continue
                self.generate_tfrecord_heatmap(tfr, attention_dict, heatmaps_dir, tile_px=tile_px, tile_um=tile_um)

    def extract_tiles(self, tile_px, tile_um, filters=None, filter_blank=None, **kwargs):
        """Extracts tiles from slides. Compatibility function. Preferred use is calling
        :func:`slideflow.dataset.Dataset.extract_tiles` on a :class:`slideflow.dataset.Dataset` directly.

        Args:
            save_tiles (bool, optional): Save images of extracted tiles to project tile directory. Defaults to False.
            save_tfrecords (bool, optional): Save compressed image data from extracted tiles into TFRecords
                in the corresponding TFRecord directory. Defaults to True.
            source (str, optional): Name of dataset source from which to select slides for extraction. Defaults to None.
                If not provided, will default to all sources in project.
            stride_div (int, optional): Stride divisor to use when extracting tiles. Defaults to 1.
                A stride of 1 will extract non-overlapping tiles.
                A stride_div of 2 will extract overlapping tiles, with a stride equal to 50% of the tile width.
            enable_downsample (bool, optional): Enable downsampling when reading slide images. Defaults to True.
                This may result in corrupted image tiles if downsampled slide layers are corrupted or incomplete.
                Recommend manual confirmation of tile integrity.
            roi_method (str, optional): Either 'inside', 'outside', or 'ignore'. Defaults to 'inside'.
                Indicates whether tiles are extracted inside or outside ROIs, or if ROIs are ignored entirely.
            skip_missing_roi (bool, optional): Skip slides that are missing ROIs. Defaults to False.
            skip_extracted (bool, optional): Skip slides that have already been extracted. Defaults to True.
            tma (bool, optional): Reads slides as Tumor Micro-Arrays (TMAs), detecting and extracting tumor cores.
                Defaults to False. Experimental function with limited testing.
            randomize_origin (bool, optional): Randomize pixel starting position during extraction. Defaults to False.
            buffer (str, optional): Slides will be copied to this directory before extraction. Defaults to None.
                Using an SSD or ramdisk buffer vastly improves tile extraction speed.
            num_workers (int, optional): Extract tiles from this many slides simultaneously. Defaults to 4.
            q_size (int, optional): Size of queue when using a buffer. Defaults to 4.
            qc (str, optional): 'otsu', 'blur', 'both', or None. Perform blur detection quality control - discarding
                tiles with detected out-of-focus regions or artifact - and/or otsu's method. Increases tile extraction
                time. Defaults to None.
            report (bool, optional): Save a PDF report of tile extraction. Defaults to True.

        Keyword Args:
            normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image. Defaults to None.
                If None but using a normalizer, will use an internal tile for normalization.
                Internal default tile can be found at slideflow.slide.norm_tile.jpg
            whitespace_fraction (float, optional): Range 0-1. Defaults to 1.
                Discard tiles with this fraction of whitespace. If 1, will not perform whitespace filtering.
            whitespace_threshold (int, optional): Range 0-255. Defaults to 230.
                Threshold above which a pixel (RGB average) is considered whitespace.
            grayspace_fraction (float, optional): Range 0-1. Defaults to 0.6.
                Discard tiles with this fraction of grayspace. If 1, will not perform grayspace filtering.
            grayspace_threshold (float, optional): Range 0-1. Defaults to 0.05.
                Pixels in HSV format with saturation below this threshold are considered grayspace.
            img_format (str, optional): 'png' or 'jpg'. Defaults to 'jpg'. Image format to use in tfrecords.
                PNG (lossless) format recommended for fidelity, JPG (lossy) for efficiency.
            full_core (bool, optional): Only used if extracting from TMA. If True, will save entire TMA core as image.
                Otherwise, will extract sub-images from each core using the given tile micron size. Defaults to False.
            shuffle (bool, optional): Shuffle tiles prior to storage in tfrecords. Defaults to True.
            num_threads (int, optional): Number of workers threads for each tile extractor. Defaults to 4.
            qc_blur_radius (int, optional): Quality control blur radius for out-of-focus area detection. Only used if
                qc=True. Defaults to 3.
            qc_blur_threshold (float, optional): Quality control blur threshold for detecting out-of-focus areas.
                Only used if qc=True. Defaults to 0.1
            qc_filter_threshold (float, optional): Float between 0-1. Tiles with more than this proportion of blur
                will be discarded. Only used if qc=True. Defaults to 0.6.
            qc_mpp (float, optional): Microns-per-pixel indicating image magnification level at which quality control
                is performed. Defaults to mpp=4 (effective magnification 2.5 X)
            dry_run (bool, optional): Determine tiles that would be extracted, but do not export any images.
                Defaults to None.
        """

        dataset = self.dataset(tile_px, tile_um, filters=filters, filter_blank=filter_blank, verification='slides')
        dataset.extract_tiles(**kwargs)

    def extract_tiles_from_tfrecords(self, tile_px, tile_um, filters=None, filter_blank=None, dest=None):
        """Extracts tiles from a set of TFRecords. Compatibility function. Preferred use is calling
        :meth:`slideflow.dataset.Dataset.extract_tiles_from_tfrecords` on a :class:`slideflow.dataset.Dataset` directly.

        Args:
            tile_px (int): Tile size in pixels
            tile_um (int): Tile size in microns
            filters (dict, optional): Filters dict to use when selecting tfrecords. Defaults to None.
                See :meth:`get_dataset` documentation for more information on filtering.
            filter_blank (list, optional): Slides blank in these columns will be excluded. Defaults to None.
            dest (str): Path to directory in which to save tile images. Defaults to None. If None, uses dataset default.
        """
        dataset = self.dataset(tile_px, tile_um, filters=filters, filter_blank=filter_blank)
        dataset.extract_tiles_from_tfrecords(dest)

    def generate_features(self, model, dataset=None, filters=None, filter_blank=None, min_tiles=0, max_tiles=0,
                         outcome_label_headers=None, torch_export=None, **kwargs):

        """Calculate layer features / activations and provide interface for calculating statistics.

        Args:
            model (str): Path to model
            dataset (:class:`slideflow.dataset.Dataset`, optional): Dataset object from which to generate activations.
                If not supplied, will calculate activations for all project tfrecords at the tile_px/tile_um
                matching the supplied model, optionally using provided filters and filter_blank.
            filters (dict, optional): Filters dict to use when selecting tfrecords. Defaults to None.
                Ignored if dataset is supplied.
                See :meth:`get_dataset` documentation for more information on filtering.
            filter_blank (list, optional): Slides blank in these columns will be excluded. Defaults to None.
                Ignored if dataset is supplied.
            min_tiles (int, optional): Only include slides with this minimum number of tiles. Defaults to 0.
            max_tiles (int, optional): Only include maximum of this many tiles per slide. Defaults to 0 (all tiles).
            outcome_label_headers (list, optional): Column header(s) in annotations file. Defaults to None.
                Used for category-level comparisons
            torch_export (str, optional): Path. Export activations to torch-compatible file at this location.
                Defaults to None.

        Keyword Args:
            layers (list(str)): Layers from which to generate activations. Defaults to 'postconv'.
            export (str): Path to CSV file. Save activations in CSV format to this file. Defaults to None.
            cache (str): Path to PKL file. Cache activations at this location. Defaults to None.
            include_logits (bool): Generate and store logit predictions along with layer activations.
            batch_size (int): Batch size to use when calculating activations. Defaults to 32.

        Returns:
            :class:`slideflow.model.DatasetFeatures`:
        """

        # Setup directories
        stats_root = join(self.root, 'stats')
        if not exists(stats_root): os.makedirs(stats_root)

        # Load dataset for evaluation
        config = sf.util.get_model_config(model)
        if dataset is None:
            tile_px = config['hp']['tile_px']
            tile_um = config['hp']['tile_um']
            dataset = self.dataset(tile_px=tile_px,
                                       tile_um=tile_um,
                                       filters=filters,
                                       filter_blank=filter_blank)
        else:
            if config and (dataset.tile_px != config['hp']['tile_px'] or dataset.tile_um != config['hp']['tile_um']):
                raise ValueError(f"Dataset tile size ({dataset.tile_px}px, {dataset.tile_um}um) does not match " + \
                                 f"model ({config['hp']['tile_px']}px, {config['hp']['tile_um']}um)")
            if filters is not None or filter_blank is not None:
                log.warning("Dataset supplied; ignoring provided filters and filter_blank")
            tile_px = dataset.tile_px

        dataset = dataset.filter(min_tiles=min_tiles).clip(max_tiles)

        outcome_annotations = dataset.labels(outcome_label_headers, format='name')[0] if outcome_label_headers else None
        log.info(f'Visualizing activations from {len(dataset.tfrecords())} slides')

        df = sf.model.DatasetFeatures(model=model,
                                      dataset=dataset,
                                      annotations=outcome_annotations,
                                      manifest=dataset.manifest(),
                                      **kwargs)
        if torch_export:
            df.export_to_torch(torch_export)

        return df

    def generate_features_for_clam(self, model, outdir='auto', layers=['postconv'], max_tiles=0, min_tiles=16,
                                   filters=None, filter_blank=None, force_regenerate=False):

        """Using the specified model, generates tile-level features for slides for use with CLAM.

        Args:
            model (str): Path to model from which to generate activations.
                May provide either this or "pt_files"
            outdir (str, optional): Path in which to save exported activations in .pt format. Defaults to 'auto'.
                If 'auto', will save in project directory.
            layers (list, optional): Which model layer(s) generate activations. Defaults to 'postconv'.
            max_tiles (int, optional): Maximum number of tiles to take per slide. Defaults to 0.
            min_tiles (int, optional): Minimum number of tiles per slide. Defaults to 8.
                Will skip slides not meeting this threshold.
            filters (dict, optional): Filters dict to use when selecting tfrecords. Defaults to None.
                Ignored if dataset is supplied.
                See :meth:`get_dataset` documentation for more information on filtering.
            filter_blank (list, optional): Slides blank in these columns will be excluded. Defaults to None.
                Ignored if dataset is supplied.
            force_regenerate (bool, optional): Generate activations for all slides. Defaults to False.
                If False, will skip slides that already have a .pt file in the export directory.

        Returns:
            Path to directory containing exported .pt files
        """

        assert min_tiles >= 8, 'Slides must have at least 8 tiles to train CLAM.'

        # First, ensure the model is valid with a hyperparameters file
        config = sf.util.get_model_config(model)
        if not config:
            raise Exception('Unable to find model hyperparameters file.')
        tile_px = config['tile_px']
        tile_um = config['tile_um']

        # Set up the pt_files directory for storing model activations
        if outdir.lower() == 'auto':
            model_name_end = '' if 'k_fold_i' not in config else f"_kfold{config['k_fold_i']}"
            outdir = join(self.root, 'pt_files', config['model_name']+model_name_end)
        if not exists(outdir):
            os.makedirs(outdir)

        # Detect already generated pt files
        already_generated = [sf.util.path_to_name(f) for f in os.listdir(outdir)
                                                    if sf.util.path_to_ext(join(outdir, f)) == 'pt']
        if force_regenerate or not len(already_generated):
            activation_filters = filters
        else:
            pt_dataset = self.dataset(tile_px, tile_um, filters=filters, filter_blank=filter_blank)
            all_slides = pt_dataset.slides()
            slides_to_generate = [s for s in all_slides if s not in already_generated]
            if len(slides_to_generate) != len(all_slides):
                to_skip = len(all_slides) - len(slides_to_generate)
                log.info(f"Skipping {to_skip} of {len(all_slides)} slides, which already have features generated. " + \
                         "To override, pass force_regenerate=True.")
            if not slides_to_generate:
                log.warn("No slides to generate CLAM features.")
                return outdir
            activation_filters = {} if filters is None else filters.copy()
            activation_filters['slide'] = slides_to_generate
            filtered_dataset = self.dataset(tile_px, tile_um, filters=activation_filters, filter_blank=filter_blank)
            filtered_slides_to_generate = filtered_dataset.slides()
            log.info(f'Features already generated for {len(already_generated)} files, will not regenerate.')
            log.info(f'Attempting to generate for {len(filtered_slides_to_generate)} slides')

        # Set up activations interface
        self.generate_features(model,
                               filters=activation_filters,
                               filter_blank=filter_blank,
                               layers=layers,
                               max_tiles=max_tiles,
                               min_tiles=min_tiles,
                               torch_export=outdir,
                               include_logits=False,
                               cache=None)
        return outdir

    def generate_heatmaps(self, model, filters=None, filter_blank=None, outdir=None, resolution='low', batch_size=64,
                          roi_method='inside', buffer=None, num_threads=8, skip_completed=False, **kwargs):

        """Creates predictive heatmap overlays on a set of slides.

        Args:
            model (str): Path to Tensorflow model with which predictions will be generated.
            filters (dict, optional): Filters dict to use when selecting tfrecords. Defaults to None.
                See :meth:`get_dataset` documentation for more information on filtering.
            filter_blank (list, optional): Slides blank in these columns will be excluded. Defaults to None.
            outdir (path, optional): Directory in which to save heatmap images.
            resolution (str, optional): Heatmap resolution (determines stride of tile predictions). Defaults to 'low'.
                "low" uses a stride equal to tile width.
                "medium" uses a stride equal 1/2 tile width.
                "high" uses a stride equal to 1/4 tile width.
            batch_size (int, optional): Batch size when calculating logits for heatmap. Defaults to 64.
            roi_method (str, optional): 'inside', 'outside', or 'none'. Defaults to 'inside'.
                Determines where heatmap should be made with respect to annotated ROI.
            buffer (str, optional): Path to which slides are copied prior to heatmap generation. Defaults to None.
                This vastly improves extraction speed when using SSD or ramdisk buffer.
            num_threads (int, optional): Number of threads to assign to tile extraction. Defaults to 8.
                Performance improvements can be seen by increasing this number in highly multi-core systems.
            skip_completed (bool, optional): Skip heatmaps for slides that already have heatmaps in target directory.

        Keyword args:
            show_roi (bool): Show ROI on heatmaps.
            interpolation (str): Interpolation strategy for smoothing heatmap predictions. Defaults to None.
                Includes all matplotlib imshow interpolation options.
            logit_cmap: Either a function or a dictionary used to create heatmap colormap.
                If None (default), separate heatmaps will be generated for each label category,
                with color representing likelihood of category prediction.
                Each image tile will generate a list of predictions of length O,
                If logit_cmap is a function, then the logit predictions will be passed,
                where O is the number of label categories.
                and the function is expected to return [R, G, B] values which will be displayed.
                If the logit_cmap is a dictionary, it should map 'r', 'g', and 'b' to label indices;
                The prediction for these label categories will be mapped to corresponding colors.
                Thus, the corresponding color will only reflect predictions of up to three labels.
                Example (this would map predictions for label 0 to red, 3 to green, etc): {'r': 0, 'g': 3, 'b': 1 }
            vmin (float): Minimimum value to display on heatmap. Defaults to 0.
            vcenter (float): Center value for color display on heatmap. Defaults to 0.5.
            vmax (float): Maximum value to display on heatmap. Defaults to 1.
        """

        # Prepare arguments for subprocess
        heatmap_args = types.SimpleNamespace(**locals())
        del heatmap_args.self

        # Prepare dataset1
        config = sf.util.get_model_config(model)
        heatmaps_dataset = self.dataset(filters=filters,
                                            filter_blank=filter_blank,
                                            tile_px=config['hp']['tile_px'],
                                            tile_um=config['hp']['tile_um'])
        slide_list = heatmaps_dataset.slide_paths()
        heatmap_args.rois = heatmaps_dataset.rois()

        # Set resolution / stride
        resolutions = {'low': 1, 'medium': 2, 'high': 4}
        try:
            stride_div = resolutions[resolution]
        except KeyError:
            log.error(f"Invalid resolution '{resolution}': must be either 'low', 'medium', or 'high'.")
            return
        heatmap_args.stride_div = stride_div

        # Attempt to auto-detect supplied model name
        config = sf.util.get_model_config(model)
        detected_model_name = os.path.basename(model)
        config = sf.util.get_model_config(model)
        if config and 'model_name' in config:
            detected_model_name = config['model_name']

        # Make output directory
        outdir = outdir if outdir else join(self.root, 'heatmaps', detected_model_name)
        if not exists(outdir): os.makedirs(outdir)
        heatmap_args.outdir = outdir

        # Any function loading a slide must be kept in an isolated process, as loading more than one slide
        # in a single process causes instability / hangs. I suspect this is a libvips or openslide issue but
        # I haven't been able to identify the root cause. Isolating processes when multiple slides are to be processed
        # sequentially is a functional workaround.
        for slide in slide_list:
            if skip_completed and exists(join(outdir, f'{sf.util.path_to_name(slide)}-custom.png')):
                log.info(f'Skipping already-completed heatmap for slide {sf.util.path_to_name(slide)}')
                return

            ctx = multiprocessing.get_context('spawn')
            process = ctx.Process(target=project_utils._heatmap_worker, args=(slide, heatmap_args, kwargs))
            process.start()
            process.join()

    def generate_mosaic(self, df, dataset=None, filters=None, filter_blank=None, outcome_label_headers=None,
                        map_slide=None, show_prediction=None, restrict_pred=None, predict_on_axes=None, max_tiles=0,
                        umap_cache=None, use_float=False, low_memory=False, **kwargs):

        """Generates a mosaic map by overlaying images onto a set of mapped tiles.
            Image tiles are extracted from the provided set of TFRecords, and predictions + features from
            post-convolutional layer activations are calculated using the specified model. Tiles are mapped either
            with dimensionality reduction on post-convolutional layer activations (default behavior), or by using
            outcome predictions for two categories, mapped to X- and Y-axis (via predict_on_axes).

        Args:
            df (:class:`slideflow.model.DatasetFeatures`): DatasetFeatures containing model features.
            dataset (:class:`slideflow.dataset.Dataset`, optional): Dataset object from which to generate mosaic.
                If not supplied, will generate mosaic for all project tfrecords at the tile_px/tile_um
                matching the supplied model, optionally using provided filters and filter_blank.
            filters (dict, optional): Filters dict to use when selecting tfrecords. Defaults to None.
                Ignored if dataset is supplied.
                See :meth:`get_dataset` documentation for more information on filtering.
            filter_blank (list, optional): Slides blank in these columns will be excluded. Defaults to None.
                Ignored if dataset is supplied.
            outcome_label_headers (list, optional): Column name in annotations file from which to read category labels.
            map_slide (str, optional): None (default), 'centroid', or 'average'.
                If provided, will map slides using slide-level calculations, either mapping
                centroid tiles if 'centroid', or calculating node averages across all tiles
                in a slide and mapping slide-level node averages, if 'average'
            show_prediction (int or str, optional): May be either int or string, corresponding to label category.
                Predictions for this category will be displayed on the exported UMAP plot.
            restrict_pred (list, optional): List of int, if provided, will restrict predictions to only these categories
                Final tile-level prediction is made by choosing category with highest logit.
            predict_on_axes (list, optional): (int, int). Each int corresponds to an label category id.
                If provided, predictions are generated for these two labels categories; tiles are then mapped
                with these predictions with the pattern (x, y) and the mosaic is generated from this map.
                This replaces the default dimensionality reduction mapping.
            max_tiles (int, optional): Limits the number of tiles taken from each slide. Defaults to 0.
            umap_cache (str, optional): Path to PKL file in which to save/cache UMAP coordinates. Defaults to None.
            use_float (bool, optional): Assume labels are float / linear (as opposed to categorical). Defaults to False.
            low_memory (bool, optional): Limit memory during UMAP calculations. Defaults to False.

        Keyword Args:
            resolution (str): Resolution of the mosaic map. Low, medium, or high.
            num_tiles_x (int): Specifies the size of the mosaic map grid.
            expanded (bool): If False, limits tile assignment to the each grid space (strict display).
                If True, allows for display of nearby tiles if a given grid is empty.
                Defaults to False.
            leniency (float): UMAP leniency. Defaults to 1.5.
            tile_zoom (int): Tile zoom level. Defaults to 15.

        Returns:
            :class:`slideflow.mosaic.Mosaic`: Mosaic object.
        """

        # Set up paths
        stats_root = join(self.root, 'stats')
        mosaic_root = join(self.root, 'mosaic')
        if not exists(stats_root): os.makedirs(stats_root)
        if not exists(mosaic_root): os.makedirs(mosaic_root)

        # Prepare dataset & model
        config = sf.util.get_model_config(df.model)
        if dataset is None:
            tile_px, tile_um = config['hp']['tile_px'], config['hp']['tile_um']
            dataset = self.dataset(tile_px=tile_px,
                                       tile_um=tile_um,
                                       filters=filters,
                                       filter_blank=filter_blank)
        else:
            if config and (dataset.tile_px != config['hp']['tile_px'] or dataset.tile_um != config['hp']['tile_um']):
                raise ValueError(f"Dataset tile size ({dataset.tile_px}px, {dataset.tile_um}um) does not match " + \
                                 f"model ({config['hp']['tile_px']}px, {config['hp']['tile_um']}um)")
            if filters is not None or filter_blank is not None:
                log.warning("Dataset supplied; ignoring provided filters and filter_blank")
            tile_px = dataset.tile_px

        # Clip to max tiles
        dataset = dataset.clip(max_tiles)

        # Get TFrecords, and prepare a list for focus, if requested
        tfrecords_list = dataset.tfrecords()
        log.info(f'Generating mosaic from {len(tfrecords_list)} slides')

        # If a header category is supplied and we are not showing predictions,
        # then assign slide labels from annotations
        model_type = config['model_type']
        if model_type == 'linear':
            use_float = True
        if outcome_label_headers and (show_prediction is None):
            slide_labels, _ = dataset.labels(outcome_label_headers, use_float=use_float, format='name')
        else:
            slide_labels = {}

        # If showing predictions, try to automatically load prediction labels
        if (show_prediction is not None) and (not use_float):
            model_hp = sf.util.get_model_config(df.model)
            if model_hp:
                outcome_labels = model_hp['outcome_labels']
                model_type = model_type if model_type else model_hp['model_type']
                log.info(f'Automatically loaded prediction labels found at {sf.util.green(df.model)}')
            else:
                log.info(f'Unable to auto-detect prediction labels from model hyperparameters file')

        # Initialize mosaic, umap
        mosaic, umap = None, None

        if predict_on_axes:
            # Create mosaic using x- and y- axis corresponding to label predictions
            umap_x, umap_y, umap_meta = df.map_to_predictions(predict_on_axes[0], predict_on_axes[1])
            umap = sf.statistics.SlideMap.from_precalculated(slides=dataset.slides(),
                                                             x=umap_x,
                                                             y=umap_y,
                                                             meta=umap_meta)
        else:
            # Create mosaic map from dimensionality reduction on post-convolutional layer activations
            umap = sf.statistics.SlideMap.from_features(df,
                                                        map_slide=map_slide,
                                                        prediction_filter=restrict_pred,
                                                        cache=umap_cache,
                                                        low_memory=low_memory)

        # If displaying centroid AND predictions, then show slide-level predictions rather than tile-level predictions
        if (map_slide=='centroid') and show_prediction is not None:
            log.info('Showing slide-level predictions at point of centroid')

            # If not model has not been assigned, assume categorical model
            model_type = model_type if model_type else 'categorical'

            # Get predictions
            if model_type == 'categorical':
                slide_pred = df.logits_predict(restrict_pred)
                slide_percent = df.logits_percent(restrict_pred)
            else:
                slide_pred = slide_percent = df.logits_mean()

            # If show_prediction is provided (either a number or string),
            # then display ONLY the prediction for the provided category, as a colormap
            if type(show_prediction) == int:
                log.info(f'Showing prediction for label {show_prediction} as colormap')
                slide_labels = {k:v[show_prediction] for k, v in slide_percent.items()}
                show_prediction = None
            elif type(show_prediction) == str:
                log.info(f'Showing prediction for label {show_prediction} as colormap')
                reversed_labels = {v:k for k, v in outcome_labels.items()}
                if show_prediction not in reversed_labels:
                    raise ValueError(f"Unknown label category '{show_prediction}'")
                slide_labels = {k:v[int(reversed_labels[show_prediction])] for k, v in slide_percent.items()}
                show_prediction = None
            elif use_float:
                # Displaying linear predictions needs to be implemented here
                raise TypeError("If showing predictions & use_float=True, set 'show_prediction' " + \
                                    "to category to be predicted.")
            # Otherwise, show_prediction is assumed to be just "True", in which case show categorical predictions
            else:
                try:
                    slide_labels = {k:outcome_labels[v] for (k,v) in slide_pred.items()}
                except KeyError:
                    # Try interpreting prediction label keys as strings
                    slide_labels = {k:outcome_labels[str(v)] for (k,v) in slide_pred.items()}

        if slide_labels:
            umap.label_by_slide(slide_labels)
        if show_prediction and (map_slide != 'centroid'):
            umap.label_by_tile_meta('prediction', translation_dict=outcome_labels)
        umap.filter(dataset.slides())

        mosaic = sf.Mosaic(umap,
                           dataset.tfrecords(),
                           normalizer=df.normalizer,
                           normalizer_source=df.normalizer_source,
                           **kwargs)
        return mosaic

    def generate_mosaic_from_annotations(self, header_x, header_y, dataset, model=None, mosaic_filename=None,
                                         umap_filename=None, outcome_label_headers=None, max_tiles=100,
                                         use_optimal_tile=False, activations_cache=None, normalizer=None,
                                         normalizer_source=None, batch_size=64, **kwargs):

        """Generates a mosaic map by overlaying images onto a set of mapped tiles.
            Slides are mapped with slide-level annotations, x-axis determined from header_x, y-axis from header_y.
            If use_optimal_tile is False and no model is provided, first image tile in each TFRecord is displayed.
            If optimal_tile is True, post-convolutional layer activations for all tiles in each slide are
            calculated using the provided model, and the tile nearest to centroid is used for display.

        Args:
            header_x (str): Column name in annotations file from which to read X-axis coordinates.
            header_y (str): Column name in annotations file from which to read Y-axis coordinates.
            dataset (:class:`slideflow.dataset.Dataset`): Dataset object.
            model (str, optional): Path to Tensorflow model to use when generating layer activations.
            mosaic_filename (str, optional): Filename for mosaic image. Defaults to None.
                If not provided, mosaic will not be calculated or saved. If provided, saved in project mosaic directory.
            umap_filename (str, optional): Filename for UMAP plot image. Defaults to None.
                If not provided, plot will not be saved. Will be saved in project stats directory.
            outcome_label_headers (list(str)): Column name(s) in annotations file from which to read category labels.
            filters (dict, optional): Filters dict to use when selecting tfrecords. Defaults to None.
                Ignored if dataset is supplied.
                See :meth:`get_dataset` documentation for more information on filtering.
            max_tiles (int, optional): Limits the number of tiles taken from each slide. Defaults to 0.
            use_optimal_tile (bool, optional): Use model to create layer activations for all tiles in
                each slide, and choosing tile nearest centroid for each slide for display.
            activations_cache (str, optional): Path to PKL file in which to cache nodal activations. Defaults to None.
            normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image. Defaults to None.
                If None but using a normalizer, will use an internal tile for normalization.
                Internal default tile can be found at slideflow.slide.norm_tile.jpg
            batch_size (int, optional): Batch size for model. Defaults to 64.

        Keyword Args:
            resolution (str): Resolution of the mosaic map. Low, medium, or high.
            num_tiles_x (int): Specifies the size of the mosaic map grid.
            expanded (bool): If False, limits tile assignment to the each grid space (strict display).
                If True, allows for display of nearby tiles if a given grid is empty.
                Defaults to False.
            leniency (float): UMAP leniency. Defaults to 1.5.
            tile_zoom (int): Tile zoom level. Defaults to 15.
        """

        # Setup paths
        stats_root = join(self.root, 'stats')
        mosaic_root = join(self.root, 'mosaic')
        if not exists(stats_root): os.makedirs(stats_root)
        if not exists(mosaic_root): os.makedirs(mosaic_root)

        # Filter dataset to exclude slides blank in the x and y header columns
        dataset = dataset.filter(filter_blank=[header_x, header_y]).clip(max_tiles)

        # We are assembling a list of slides from the TFRecords path list,
        # because we only want to use slides that have a corresponding TFRecord
        # (some slides did not have a large enough ROI for tile extraction,
        # and some slides may be in the annotations but are missing a slide image)
        slides = [sf.util.path_to_name(tfr) for tfr in dataset.tfrecords()]
        labels, _ = dataset.labels([header_x, header_y], use_float=True)
        slide_to_category, _ = dataset.labels(outcome_label_headers, format='name')

        umap_x = np.array([labels[slide][0] for slide in slides])
        umap_y = np.array([labels[slide][1] for slide in slides])

        # Ensure normalization strategy is consistent
        if model is not None and sf.util.get_model_config['hp']['normalizer'] != normalizer:
            m_normalizer = sf.util.get_model_config['hp']['normalizer']
            log.warning(f'Model trained with {m_normalizer} normalizer, but normalizer set to "{normalizer}"!')

        if use_optimal_tile and not model:
            log.error('Unable to calculate optimal tile if no model is specified.')
            return
        elif use_optimal_tile:
            # Calculate most representative tile in each slide/TFRecord for display
            df = sf.model.DatasetFeatures(model=model,
                                                dataset=dataset,
                                                batch_size=batch_size,
                                                cache=activations_cache)

            optimal_slide_indices, _ = sf.statistics.calculate_centroid(df.activations)

            # Restrict mosaic to only slides that had enough tiles to calculate an optimal index from centroid
            successful_slides = list(optimal_slide_indices.keys())
            num_warned = 0
            warn_threshold = 3
            for slide in slides:
                if slide not in successful_slides:
                    log.warn(f'Unable to calculate optimal tile for {sf.util.green(slide)}; will skip')
                    num_warned += 1
            if num_warned >= warn_threshold:
                log.warn(f'...{num_warned} total warnings, see project log for details')

            umap_x = np.array([labels[slide][0] for slide in successful_slides])
            umap_y = np.array([labels[slide][1] for slide in successful_slides])
            umap_meta = [{'slide': slide, 'index': optimal_slide_indices[slide]} for slide in successful_slides]
        else:
            # Take the first tile from each slide/TFRecord
            umap_meta = [{'slide': slide, 'index': 0} for slide in slides]

        umap = sf.statistics.SlideMap.from_precalculated(slides=slides,
                                                         x=umap_x,
                                                         y=umap_y,
                                                         meta=umap_meta)

        mosaic_map = sf.Mosaic(umap,
                               dataset.tfrecords(),
                               tile_select='centroid' if use_optimal_tile else 'nearest',
                               normalizer=normalizer,
                               normalizer_source=normalizer_source,
                               **kwargs)
        if mosaic_filename:
            mosaic_map.save(join(mosaic_root, mosaic_filename))
            mosaic_map.save_report(join(stats_root, sf.util.path_to_name(mosaic_filename)+'-mosaic_report.csv'))
        if umap_filename:
            umap.label_by_slide(slide_to_category)
            umap.save_2d_plot(join(stats_root, umap_filename))

    def generate_thumbnails(self, size=512, dataset=None, filters=None, filter_blank=None,
                            roi=False, enable_downsample=True):

        """Generates square slide thumbnails with black box borders of a fixed size, and saves to project folder.

        Args:
            size (int, optional): Width/height of thumbnail in pixels. Defaults to 512.
            dataset (:class:`slideflow.dataset.Dataset`, optional): Dataset object from which to generate activations.
                If not supplied, will calculate activations for all project tfrecords at the tile_px/tile_um
                matching the supplied model, optionally using provided filters and filter_blank.
            filters (dict, optional): Filters dict to use when selecting tfrecords. Defaults to None.
                Ignored if dataset is supplied.
                See :meth:`get_dataset` documentation for more information on filtering.
            filter_blank (list, optional): Slides blank in these columns will be excluded. Defaults to None.
                Ignored if dataset is supplied.
            roi (bool, optional): Include ROI in the thumbnail images. Defaults to False.
            enable_downsample (bool, optional): If True and a thumbnail is not embedded in the slide file,
                downsampling is permitted in order to accelerate thumbnail calculation.
        """

        from slideflow.slide import WSI
        log.info('Generating thumbnails...')

        thumb_folder = join(self.root, 'thumbs')
        if not exists(thumb_folder): os.makedirs(thumb_folder)
        if dataset is None:
            dataset = self.dataset(filters=filters, filter_blank=filter_blank, tile_px=0, tile_um=0)
        slide_list = dataset.slide_paths()
        rois = dataset.rois()
        log.info(f'Saving thumbnails to {sf.util.green(thumb_folder)}')

        for slide_path in slide_list:
            print(f'\r\033[KWorking on {sf.util.green(sf.util.path_to_name(slide_path))}...', end='')
            whole_slide = WSI(slide_path,
                              tile_px=1000,
                              tile_um=1000,
                              stride_div=1,
                              enable_downsample=enable_downsample,
                              rois=rois,
                              roi_method='inside',
                              skip_missing_roi=roi,
                              buffer=None)
            if roi:
                thumb = whole_slide.thumb(rois=True)
            else:
                thumb = whole_slide.square_thumb(size)
            thumb.save(join(thumb_folder, f'{whole_slide.name}.png'))
        print('\r\033[KThumbnail generation complete.')

    def generate_tfrecord_heatmap(self, tfrecord, tile_dict, outdir, tile_px, tile_um):
        """Creates a tfrecord-based WSI heatmap using a dictionary of tile values for heatmap display.

        Args:
            tfrecord (str): Path to tfrecord
            tile_dict (dict): Dictionary mapping tfrecord indices to a tile-level value for display in heatmap format
            outdir (str): Path to directory in which to save images
            tile_px (int): Tile width in pixels
            tile_um (int): Tile width in microns

        Returns:
            Dictionary mapping slide names to dict of statistics (mean, median, above_0, and above_1)
        """

        slide_name = sf.util.path_to_name(tfrecord)
        loc_dict = sf.io.get_locations_from_tfrecord(tfrecord)
        dataset = self.dataset(tile_px=tile_px, tile_um=tile_um)
        slide_paths = {sf.util.path_to_name(sp):sp for sp in dataset.slide_paths()}

        try:
            slide_path = slide_paths[slide_name]
        except KeyError:
            raise Exception(f'Unable to locate slide {slide_name}')

        if tile_dict.keys() != loc_dict.keys():
            raise Exception(f'Length of provided tile_dict ({len(list(tile_dict.keys()))}) does not match ' + \
                                f'number of tiles stored in the TFRecord ({len(list(loc_dict.keys()))}).')

        print(f'Generating TFRecord heatmap for {sf.util.green(tfrecord)}...')
        slide = sf.slide.WSI(slide_path, tile_px, tile_um, skip_missing_roi=False)

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

        scaled_x = [(xi * slide.roi_scale) - slide.full_extract_px/2 for xi in x]
        scaled_y = [(yi * slide.roi_scale) - slide.full_extract_px/2 for yi in y]

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
        thumb.save(join(outdir, f'{slide_name}' + '.png'))
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
        plt.savefig(join(outdir, f'{slide_name}_attn.png'), bbox_inches='tight')

        # Clean up
        print('Cleaning up...')
        plt.clf()
        del slide
        del thumb
        return stats

    def dataset(self, tile_px=None, tile_um=None, filters=None, filter_blank=None, min_tiles=0, verification='both'):
        """Returns :class:`slideflow.Dataset` object using project settings.

        Args:
            tile_px (int): Tile size in pixels
            tile_um (int): Tile size in microns
            filters (dict, optional): Filters dict to use when selecting tfrecords. Defaults to None.
            filter_blank (list, optional): Slides blank in these columns will be excluded. Defaults to None.
            verification (str, optional): 'tfrecords', 'slides', or 'both'. Defaults to 'both'.
                If 'slides', will verify all annotations are mapped to slides.
                If 'tfrecords', will check that TFRecords exist and update manifest
        """

        try:
            if self.annotations and exists(self.annotations):
                annotations = self.annotations
            else:
                self.annotations = None
            dataset = Dataset(config=self.dataset_config,
                              sources=self.sources,
                              tile_px=tile_px,
                              tile_um=tile_um,
                              annotations=annotations,
                              filters=filters,
                              filter_blank=filter_blank,
                              min_tiles=min_tiles)

        except FileNotFoundError:
            log.error('No datasets configured.')
            return

        if verification in ('both', 'slides'):
            log.debug("Verifying slide annotations...")
            dataset.verify_annotations_slides()
        if verification in ('both', 'tfrecords'):
            log.debug("Verifying tfrecords...")
            dataset.update_manifest()

        return dataset

    def load_project(self, path):
        """Loads a saved and pre-configured project from the specified path."""

        if exists(join(path, 'settings.json')):
            self._settings = sf.util.load_json(join(path, 'settings.json'))
        else:
            raise OSError(f'Unable to locate settings.json at location "{path}".')
        # Enable logging
        #log.logfile = join(self.root, 'log.log')

    def predict_wsi(self, model, outdir, dataset=None, filters=None, filter_blank=None, stride_div=1,
                    enable_downsample=True, roi_method='inside', skip_missing_roi=False, source=None,
                    randomize_origin=False, buffer=None, **kwargs):

        """Using a given model, generates a spatial map of tile-level predictions for a whole-slide image (WSI)
            and dumps prediction arrays into pkl files for later use.

        Args:
            model (str): Path to model from which to generate predictions.
            outdir (str): Path to directory in which to save WSI predictions in .pkl format.
            dataset (:class:`slideflow.dataset.Dataset`, optional): Dataset object from which to generate activations.
                If not supplied, will calculate activations for all project tfrecords at the tile_px/tile_um
                matching the supplied model, optionally using provided filters and filter_blank.
            filters (dict, optional): Filters dict to use when selecting tfrecords. Defaults to None.
                Ignored if dataset is supplied.
                See :meth:`get_dataset` documentation for more information on filtering.
            filter_blank (list, optional): Slides blank in these columns will be excluded. Defaults to None.
                Ignored if dataset is supplied.
            stride_div (int, optional): Stride divisor to use when extracting tiles. Defaults to 1.
                A stride of 1 will extract non-overlapping tiles.
                A stride_div of 2 will extract overlapping tiles, with a stride equal to 50% of the tile width.
            enable_downsample (bool, optional): Enable downsampling when reading slide images. Defaults to True.
                This may result in corrupted image tiles if downsampled slide layers are corrupted or incomplete.
                Recommend manual confirmation of tile integrity.
            roi_method (str, optional): Either 'inside', 'outside', or 'ignore'. Defaults to 'inside'.
                Indicates whether tiles are extracted inside or outside ROIs, or if ROIs are ignored entirely.
            skip_missing_roi (bool, optional): Skip slides that are missing ROIs. Defaults to True.
            source (list, optional): Name(s) of dataset sources from which to get slides. If None, will use all.
            randomize_origin (bool, optional): Randomize pixel starting position during extraction. Defaults to False.
            buffer (str, optional): Slides will be copied to this directory before extraction. Defaults to None.
                Using an SSD or ramdisk buffer vastly improves tile extraction speed.

        Keyword Args:
            whitespace_fraction (float, optional): Range 0-1. Defaults to 1.
                Discard tiles with this fraction of whitespace. If 1, will not perform whitespace filtering.
            whitespace_threshold (int, optional): Range 0-255. Defaults to 230.
                Threshold above which a pixel (RGB average) is considered whitespace.
            grayspace_fraction (float, optional): Range 0-1. Defaults to 0.6.
                Discard tiles with this fraction of grayspace. If 1, will not perform grayspace filtering.
            grayspace_threshold (float, optional): Range 0-1. Defaults to 0.05.
                Pixels in HSV format with saturation below this threshold are considered grayspace.
        """

        log.info('Generating WSI prediction / activation maps...')
        if not exists(outdir):
            os.makedirs(outdir)

        if source:   sources = [source] if not isinstance(source, list) else source
        else:        sources = self.sources

        # Prepare dataset & model
        config = sf.util.get_model_config(model)
        if dataset is None:
            tile_px, tile_um = config['hp']['tile_px'], config['hp']['tile_um']
            dataset = self.dataset(tile_px=tile_px,
                                       tile_um=tile_um,
                                       filters=filters,
                                       filter_blank=filter_blank,
                                       verification='slides')
        else:
            if config and (dataset.tile_px != config['hp']['tile_px'] or dataset.tile_um != config['hp']['tile_um']):
                raise ValueError(f"Dataset tile size ({dataset.tile_px}px, {dataset.tile_um}um) does not match " + \
                                 f"model ({config['hp']['tile_px']}px, {config['hp']['tile_um']}um)")
            if filters is not None or filter_blank is not None:
                log.warning("Dataset supplied; ignoring provided filters and filter_blank")
            tile_px = dataset.tile_px

        # Log extraction parameters
        sf.slide.log_extraction_params(**kwargs)

        for source in sources:
            log.info(f'Working on dataset source {sf.util.bold(source)}')
            roi_dir = dataset.sources[source]['roi']

            # Prepare list of slides for extraction
            slide_list = dataset.slide_paths(source=source)
            log.info(f'Generating predictions for {len(slide_list)} slides ({tile_um} um, {tile_px} px)')

            # Verify slides and estimate total number of tiles
            log.info('Verifying slides...')
            total_tiles = 0
            for slide_path in tqdm(slide_list, leave=False):
                slide = sf.slide.WSI(slide_path,
                                    tile_px,
                                    tile_um,
                                    stride_div,
                                    roi_dir=roi_dir,
                                    roi_method=roi_method,
                                    skip_missing_roi=False,
                                    buffer=None)
                log.debug(f"Estimated tiles for slide {slide.name}: {slide.estimated_num_tiles}")
                total_tiles += slide.estimated_num_tiles
                del slide
            log.info(f'Verification complete. Total estimated tiles to predict: {total_tiles}')

            for slide_path in slide_list:
                log.info(f'Working on slide {sf.util.path_to_name(slide_path)}')
                whole_slide = sf.slide.WSI(slide_path,
                                        tile_px,
                                        tile_um,
                                        stride_div,
                                        enable_downsample=enable_downsample,
                                        roi_dir=roi_dir,
                                        roi_method=roi_method,
                                        randomize_origin=randomize_origin,
                                        skip_missing_roi=skip_missing_roi,
                                        buffer=buffer)

                if not whole_slide.loaded_correctly():
                    continue

                try:
                    interface = sf.model.Features(model, include_logits=False)
                    wsi_grid = interface(whole_slide, num_threads=12)

                    with open (join(outdir, whole_slide.name+'.pkl'), 'wb') as pkl_file:
                        pickle.dump(wsi_grid, pkl_file)

                except sf.slide.TileCorruptionError:
                    formatted_slide = sf.util.green(sf.util.path_to_name(slide_path))
                    if enable_downsample:
                        log.warn(f'Corrupt tile in {formatted_slide}; consider disabling downsampling')
                    else:
                        log.error(f'Corrupt tile in {formatted_slide}; skipping slide')
                    continue

    def resize_tfrecords(self, *args, **kwargs):
        """Function moved to :meth:slideflow.dataset.Dataset.resize_tfrecords"""

        raise DeprecationWarning("Function moved to slideflow.dataset.Dataset.resize_tfrecords()")

    def save(self):
        """Saves current project configuration as "settings.json"."""
        sf.util.write_json(self._settings, join(self.root, 'settings.json'))

    def slide_report(self, *args, **kwargs):

        """Function moved to :meth:slideflow.dataset.Dataset.slide_report"""

        raise DeprecationWarning("Function moved to slideflow.dataset.Dataset.slide_report()")

    def tfrecord_report(self, *args, **kwargs):

        """Function moved to :meth:slideflow.dataset.Dataset.tfrecord_report"""

        raise DeprecationWarning("Function moved to slideflow.dataset.Dataset.tfrecord_report()")

    def train(self, outcome_label_headers, params, exp_label=None, filters=None, filter_blank=None,
              input_header=None, resume_training=None, checkpoint=None, pretrain='imagenet', min_tiles=0, max_tiles=0,
              multi_gpu=False, splits="splits.json", **training_kwargs):

        """Train model(s) using a given set of hyperparameters, outcomes, and inputs.

        Args:
            outcome_label_headers (str): Str or list of str. Annotation column header specifying the outcome label(s).
            params (:class:`slideflow.model.ModelParams`, list, dict, or str): Defaults to 'sweep'.
                If 'sweep', will use project batch train configuration file to sweep through all HP combinations.
                Additionally, a :class:`slideflow.model.ModelParams` may be provided, either individually,
                as a list, or as a dictionary mapping model names to HP objects. Please see examples below for use.
            exp_label (str, optional): Optional experiment label to add to each model name.
            filters (dict, optional): Filters dict to use when selecting tfrecords. Defaults to None.
                See :meth:`get_dataset` documentation for more information on filtering.
            filter_blank (list, optional): Slides blank in these columns will be excluded. Defaults to None.
            input_header (list, optional): List of annotation column headers to use as model input. Defaults to None.
            resume_training (str, optional): Path to Tensorflow model to continue training. Defaults to None.
            checkpoint (str, optional): Path to cp.ckpt from which to load weights. Defaults to None.
            pretrain (str, optional): Either 'imagenet' or path to Tensorflow model from which to load weights.
                Defaults to 'imagenet'.
            min_tiles (int): Minimum number of tiles a slide must have to include in training. Defaults to 0.
            max_tiles (int): Only use up to this many tiles from each slide for training. Defaults to 0.
                If zero, will include all tiles.
            multi_gpu (bool): Train using multiple GPUs using Keras MirroredStrategy when available. Defaults to True.
            splits (str, optional): Filename of JSON file in which to log training/validation splits. Looks for
                filename in project root directory. Defaults to "splits.json".

        Keyword Args:
            val_strategy (str): Validation dataset selection strategy. Defaults to 'k-fold'.
                Options include bootstrap, k-fold, k-fold-manual, k-fold-preserved-site, fixed, and none.
            val_k_fold (int): Total number of K if using K-fold validation. Defaults to 3.
            val_k (int): Iteration of K-fold to train, starting at 1. Defaults to None (training all k-folds).
            val_k_fold_header (str): Annotations file header column for manually specifying k-fold or for preserved-site
                cross validation. Only used if validation strategy is 'k-fold-manual' or 'k-fold-preserved-site'.
                Defaults to None for k-fold-manual and 'site' for k-fold-preserved-site.
            val_fraction (float): Fraction of dataset to use for validation testing, if strategy is 'fixed'.
            val_source (str): Dataset source to use for validation. Defaults to None (same as training).
            val_annotations (str): Path to annotations file for validation dataset. Defaults to None (same as training).
            val_filters (dict): Filters dictionary to use for validation dataset. Defaults to None (same as training).

            starting_epoch (int): Start training at the specified epoch. Defaults to 0.
            steps_per_epoch_override (int): If provided, will manually set the number of steps in an epoch
                Default epoch length is the number of total tiles.
            save_predicitons (bool): Save predictions with each validation. Defaults to False.
                May increase validation time for large projects.
            validate_on_batch (int): Validation will be performed every N batches. Defaults to 512.
            validation_batch_size (int): Validation dataset batch size. Defaults to 32.
            use_tensorboard (bool): Add tensorboard callback for realtime training monitoring. Defaults to False.
            validation_steps (int): Number of steps of validation to perform each time doing a validation check.
                Defaults to 200.

        Returns:
            A dictionary containing model names mapped to train_acc, val_loss, and val_acc

        Examples
            Method 1 (hyperparameter sweep from a configuration file):

                >>> import slideflow.model
                >>> P.train('outcome', params='sweep.json', ...)

            Method 2 (manually specified hyperparameters):

                >>> from slideflow.model import ModelParams
                >>> hp = ModelParams(...)
                >>> P.train('outcome', params=hp, ...)

            Method 3 (list of hyperparameters):

                >>> from slideflow.model import ModelParams
                >>> hp = [ModelParams(...), ModelParams(...)]
                >>> P.train('outcome', params=hp, ...)

            Method 4 (dict of hyperparameters):

                >>> from slideflow.model import ModelParams
                >>> hp = {'HP0': ModelParams(...), 'HP1': ModelParams(...)}
                >>> P.train('outcome', params=hp, ...)

        """

        import slideflow.model

        # Prepare outcome_label_headers
        if not isinstance(outcome_label_headers, list):
            outcome_label_headers = [outcome_label_headers]
        if len(outcome_label_headers) > 1:
            num_o = len(outcome_label_headers)
            log.info(f'Training with {num_o} variables as simultaneous outcomes: {", ".join(outcome_label_headers)}')

        # Prepare hyperparameters
        if isinstance(params, str):
            if exists(params):
                hp_dict = sf.model.get_hp_from_batch_file(params)
            elif exists(join(self.root, params)):
                hp_dict = sf.model.get_hp_from_batch_file(join(self.root, params))
            else:
                raise OSError(f"Unable to find hyperparameters file {params}")
        elif isinstance(params, sf.model.ModelParams):
            hp_dict = {'HP0': params}
        elif isinstance(params, list):
            if not all([isinstance(hp, sf.model.ModelParams) for hp in params]):
                raise sf.util.UserError('If supplying list of hyperparameters, items must be sf.model.ModelParams')
            hp_dict = {f'HP{i}':hp for i,hp in enumerate(params)}
        elif isinstance(params, dict):
            if not all([isinstance(hp, str) for hp in params.keys()]):
                raise sf.util.UserError('If supplying dict of hyperparameters, keys must be of type str')
            if not all([isinstance(hp, sf.model.ModelParams) for hp in params.values()]):
                raise sf.util.UserError('If supplying dict of hyperparameters, values must be sf.model.ModelParams')
            hp_dict = params

        # Get default validation settings from kwargs
        val_kwargs = {k[4:]:v for k,v in training_kwargs.items() if k[:4] == 'val_'}
        training_kwargs = {k:v for k,v in training_kwargs.items() if k[:4] != 'val_'}
        val_settings = get_validation_settings(**val_kwargs)
        splits_file = join(self.root, splits)
        if (val_settings.strategy in ('k-fold-manual', 'k-fold-preserved-site', 'k-fold', 'bootstrap')
            and val_settings.source):

            log.error(f'Unable to use {val_settings.strategy} if validation_dataset has been provided.')
            return

        # Next, prepare the multiprocessing manager (needed to free VRAM after training and keep track of results)
        manager = multiprocessing.Manager()
        results_dict = manager.dict()
        ctx = multiprocessing.get_context('spawn')

        # === Train with a set of hyperparameters =====================================================================
        for hp_model_name, hp in hp_dict.items():
            if exp_label:
                hp_model_name = f'{exp_label}-{hp_model_name}'

            # --- Prepare dataset -------------------------------------------------------------------------------------
            # Filter out slides that are blank in the outcome label,
            # or blank in any of the input_header categories
            if filter_blank: filter_blank += [o for o in outcome_label_headers]
            else: filter_blank = [o for o in outcome_label_headers]
            if input_header:
                input_header = [input_header] if not isinstance(input_header, list) else input_header
                filter_blank += input_header
            dataset = self.dataset(hp.tile_px, hp.tile_um)
            dataset = dataset.filter(filters=filters, filter_blank=filter_blank, min_tiles=min_tiles)

            # --- Load labels -----------------------------------------------------------------------------------------
            use_float = (hp.model_type() in ['linear', 'cph'])
            labels, unique_labels = dataset.labels(outcome_label_headers, use_float=use_float)
            if hp.model_type() == 'categorical' and len(outcome_label_headers) == 1:
                outcome_labels = dict(zip(range(len(unique_labels)), unique_labels))
            elif hp.model_type() == 'categorical':
                outcome_labels = {k:dict(zip(range(len(ul)), ul)) for k, ul in unique_labels.items()}
            else:
                outcome_labels = dict(zip(range(len(outcome_label_headers)), outcome_label_headers))
            if hp.model_type() != 'linear' and len(outcome_label_headers) > 1:
                log.info('Using experimental multi-outcome approach for categorical outcome')

            # If multiple categorical outcomes are used, create a merged variable for k-fold splitting
            if hp.model_type() == 'categorical' and len(outcome_label_headers) > 1:
                labels_for_split = {k:'-'.join(map(str, v)) for k,v in labels.items()}
            else:
                labels_for_split = labels

            # --- Prepare k-fold validation configuration -------------------------------------------------------------
            results_log_path = os.path.join(self.root, 'results_log.csv')
            k_iter = val_settings.k
            k_iter = [k_iter] if (k_iter != None and not isinstance(k_iter, list)) else k_iter

            if val_settings.strategy == 'k-fold-manual':
                _, valid_k = dataset.labels(val_settings.k_fold_header, verbose=False, format='name')
                k_fold = len(valid_k)
                log.info(f"Manual K-fold iterations detected: {', '.join(valid_k)}")
                if k_iter:
                    valid_k = [kf for kf in valid_k if (int(kf) in k_iter or kf in k_iter)]
            elif val_settings.strategy in ('k-fold', 'k-fold-preserved-site', 'bootstrap'):
                k_fold = val_settings.k_fold
                valid_k = [kf for kf in range(1, k_fold+1) if ((k_iter and kf in k_iter) or (not k_iter))]
            else:
                k_fold = 0
                valid_k = [None]
            # Create model labels
            label_string = '-'.join(outcome_label_headers)
            model_name = f'{label_string}-{hp_model_name}'
            model_iterations = [model_name] if not k_fold else [f'{model_name}-kfold{k}' for k in valid_k]

            # --- Train on a specific K-fold ------------ -------------------------------------------------------------
            for k in valid_k:
                # Log current model name and k-fold iteration, if applicable
                k_fold_msg = '' if not k else f' ({val_settings.strategy} iteration {k})'
                if log.getEffectiveLevel() <= 20: print()
                log.info(f'Training model {sf.util.bold(model_name)}{k_fold_msg}...')
                log.info(f'Hyperparameters: {hp}')
                log.info(f'Validation settings: {json.dumps(vars(val_settings), indent=2)}')

                # --- Set up validation data --------------------------------------------------------------------------
                manifest = dataset.manifest()

                # Use an external validation dataset if supplied
                if val_settings.source:
                    train_dts = dataset
                    val_dts = Dataset(tile_px=hp.tile_px,
                                      tile_um=hp.tile_um,
                                      config=self.dataset_config,
                                      sources=val_settings.source,
                                      annotations=val_settings.annotations,
                                      filters=val_settings.filters,
                                      filter_blank=val_settings.filter_blank)

                    validation_labels, _ = val_dts.labels(outcome_label_headers, use_float=use_float)
                    labels.update(validation_labels)

                # Use manual k-fold assignments if indicated
                elif val_settings.strategy == 'k-fold-manual':
                    train_dts = dataset.filter(filters={val_settings.k_fold_header: [ki for ki in valid_k if ki != k]})
                    val_dts = dataset.filter(filters={val_settings.k_fold_header: [k]})

                elif val_settings.strategy == 'none':
                    train_dts = dataset
                    val_dts = None

                # Otherwise, calculate k-fold splits
                else:
                    if val_settings.strategy == 'k-fold-preserved-site':
                        site_labels, _ = dataset.labels(val_settings.k_fold_header, verbose=False, format='name')
                    else:
                        site_labels = None
                    train_dts, val_dts = dataset.training_validation_split(hp.model_type(),
                                                                           labels_for_split,
                                                                           val_strategy=val_settings.strategy,
                                                                           splits=splits_file,
                                                                           val_fraction=val_settings.fraction,
                                                                           val_k_fold=val_settings.k_fold,
                                                                           k_fold_iter=k,
                                                                           site_labels=site_labels)

                # ---- Balance and clip datasets ---------------------------------------------------------------------
                train_dts = train_dts.balance(outcome_label_headers, hp.training_balance).clip(max_tiles)
                if val_dts:
                    val_dts = val_dts.balance(outcome_label_headers, hp.validation_balance).clip(max_tiles)

                num_train = len(train_dts.tfrecords())
                num_val = 0 if not val_dts else len(val_dts.tfrecords())
                log.info(f'Using {num_train} TFRecords for training, {num_val} for validation')

                # --- Prepare additional slide-level input -----------------------------------------------------------
                model_inputs = {}
                if input_header:
                    input_header = [input_header] if not isinstance(input_header, list) else input_header
                    feature_len_dict = {}   # Dict mapping input_vars to num of different labels for each input header
                    input_labels_dict = {}  # Dict mapping input_vars to nested dictionaries which map category ID
                                            #   to category label names (for categorical variables)
                                            #   or mapping to 'float' for float variables
                    for slide in labels:
                        model_inputs[slide] = []

                    for input_var in input_header:
                        # Check if variable can be converted to float (default). If not, will assume categorical.
                        try:
                            dataset.labels(input_var, use_float=True)
                            if val_settings.source:
                                val_dts.labels(input_var, use_float=True)
                            inp_is_float = True
                        except TypeError:
                            inp_is_float = False
                        log.info(f"Adding input variable {sf.util.blue(input_var)} as {'float' if inp_is_float else 'categorical'}")

                        # Next, if this is a categorical variable, harmonize categories in training and validation datasets
                        if (not inp_is_float) and val_settings.source:
                            _, unique_train_input_labels = dataset.labels(input_var, use_float=inp_is_float)
                            _, unique_val_input_labels = val_dts.labels(input_var, use_float=inp_is_float)

                            unique_inp_labels = sorted(list(set(unique_train_input_labels + unique_val_input_labels)))
                            input_label_to_int = dict(zip(unique_inp_labels, range(len(unique_inp_labels))))
                            inp_labels_dict, _ = dataset.labels(input_var, assigned_labels=input_label_to_int)
                            val_input_labels, _ = val_dts.labels(input_var, assigned_labels=input_label_to_int)
                            inp_labels_dict.update(val_input_labels)
                        else:
                            inp_labels_dict, unique_inp_labels = dataset.labels(input_var, use_float=inp_is_float)

                        # Assign features to 'input' key of the slide-level annotations dict
                        if inp_is_float:
                            feature_len_dict[input_var] = num_features = 1
                            for slide in labels:
                                model_inputs[slide] += inp_labels_dict[slide]
                            input_labels_dict[input_var] = 'float'
                        else:
                            feature_len_dict[input_var] = num_features = len(unique_inp_labels)
                            for slide in labels:
                                onehot_label = sf.util.to_onehot(inp_labels_dict[slide], num_features).tolist()
                                model_inputs[slide] += onehot_label # We are concatenating the onehot labels together
                            input_labels_dict[input_var] = dict(zip(range(len(unique_inp_labels)), unique_inp_labels))

                    feature_sizes = [feature_len_dict[i] for i in input_header]
                else:
                    input_labels_dict = None
                    feature_sizes = None

                # --- Initialize model -------------------------------------------------------------------------------
                # Using the project annotation file, assemble list of slides for training,
                # as well as the slide annotations dictionary (output labels)
                full_model_name = model_name if not k else model_name+f'-kfold{k}'
                prev_run_dirs = [x for x in os.listdir(self.models_dir) if isdir(join(self.models_dir, x))]
                prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
                prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
                cur_run_id = max(prev_run_ids, default=-1) + 1
                model_dir = os.path.join(self.models_dir, f'{cur_run_id:05d}-{full_model_name}')
                assert not os.path.exists(model_dir)
                os.makedirs(model_dir)

                # Check git commit if running from source
                git_commit = None
                if git is not None:
                    try:
                        git_commit = git.Repo(search_parent_directories=True).head.object.hexsha
                    except:
                        pass

                # Log model settings and hyperparameters
                config_file = join(model_dir, 'params.json')
                config = {
                    'slideflow_version': sf.__version__,
                    'project': self.name,
                    'git_commit': git_commit,
                    'model_name': model_name,
                    'full_model_name': full_model_name,
                    'stage': 'training',
                    'tile_px': hp.tile_px,
                    'tile_um': hp.tile_um,
                    'max_tiles': max_tiles,
                    'min_tiles': min_tiles,
                    'model_type': hp.model_type(),
                    'outcome_label_headers': outcome_label_headers,
                    'input_features': input_header,
                    'input_feature_sizes': feature_sizes,
                    'input_feature_labels': input_labels_dict,
                    'outcome_labels': outcome_labels,
                    'dataset_config': self.dataset_config,
                    'sources': self.sources,
                    'annotations': self.annotations,
                    'validation_strategy': val_settings.strategy,
                    'validation_fraction': val_settings.fraction,
                    'validation_k_fold': val_settings.k_fold,
                    'k_fold_i': k,
                    'filters': filters,
                    'pretrain': pretrain,
                    'resume_training': resume_training,
                    'checkpoint': checkpoint,
                    'hp': hp.get_dict(),
                }
                sf.util.write_json(config, config_file)

                training_args = types.SimpleNamespace(
                    model_dir=model_dir,
                    hp=hp,
                    config=config,
                    labels=labels,
                    patients=dataset.patients(),
                    slide_input=model_inputs,
                    train_dts=train_dts,
                    val_dts=val_dts,
                    verbosity=self.verbosity,
                    pretrain=pretrain,
                    resume_training=resume_training,
                    multi_gpu=multi_gpu,
                    checkpoint=checkpoint,
                    use_neptune=self.use_neptune,
                    neptune_api=self.neptune_api,
                    neptune_workspace=self.neptune_workspace
                )

                model_kwargs = {
                    'name': full_model_name,
                    'manifest': manifest,
                    'mixed_precision': self.mixed_precision,
                    'feature_names': input_header,
                    'feature_sizes': feature_sizes,
                    'outcome_names': outcome_label_headers
                }

                process = ctx.Process(target=project_utils._train_worker, args=(training_args,
                                                                                model_kwargs,
                                                                                training_kwargs,
                                                                                results_dict))
                process.start()
                log.debug(f'Spawning training process (PID: {process.pid})')
                process.join()

            # Record results
            for mi in model_iterations:
                if mi not in results_dict or 'epochs' not in results_dict[mi]:
                    log.error(f'Training failed for model {model_name}')
                    return {}
                else:
                    sf.util.update_results_log(results_log_path, mi, results_dict[mi]['epochs'])
            log.info(f'Training complete, results saved to {sf.util.green(results_log_path)}')

        # Print summary of all models
        log.info('Training complete; validation accuracies:')
        for model in results_dict:
            try:
                last_epoch = max([int(e.split('epoch')[-1]) for e in results_dict[model]['epochs'].keys()
                                                            if 'epoch' in e ])
                final_train_metrics = results_dict[model]['epochs'][f'epoch{last_epoch}']['train_metrics']
                log.info(f'{sf.util.green(model)} training metrics:')
                for m in final_train_metrics:
                    log.info(f'{m}: {final_train_metrics[m]}')

                if 'val_metrics' in results_dict[model]['epochs'][f'epoch{last_epoch}']:
                    final_val_metrics = results_dict[model]['epochs'][f'epoch{last_epoch}']['val_metrics']
                    log.info(f'{sf.util.green(model)} validation metrics:')
                    for m in final_val_metrics:
                        log.info(f'{m}: {final_val_metrics[m]}')
            except ValueError:
                pass

        return results_dict

    def train_clam(self, exp_name, pt_files, outcome_label_headers, dataset, train_slides='auto',
                   validation_slides='auto', splits='splits.json', clam_args=None, attention_heatmaps=True):

        """Train a CLAM model from layer activations exported with :meth:`slideflow.project.generate_features_for_clam`.

        Args:
            exp_name (str): Name of experiment. Will make clam/{exp_name} subfolder.
            pt_files (str): Path to pt_files containing tile-level features.
            outcome_label_headers (str): Name in annotation column which specifies the outcome label.
            dataset (:class:`slideflow.dataset.Dataset`): Dataset object from which to generate activations.
            train_slides (str, optional): List of slide names for training. Defaults to 'auto'.
                If 'auto' (default), will auto-generate training/validation split.
            validation_slides (str, optional): List of slide names for validation. Defaults to 'auto'.
                If 'auto' (default), will auto-generate training/validation split.
            splits (str, optional): Filename of JSON file in which to log training/validation splits. Looks for
                filename in project root directory. Defaults to "splits.json".
            clam_args (optional): Namespace with clam arguments, as provided by :func:`slideflow.clam.get_args`.
            attention_heatmaps (bool, optional): Save attention heatmaps of validation dataset.

        Returns:
            None

        Examples
            Train with basic settings:

                >>> dataset = P.dataset(tile_px=299, tile_um=302)
                >>> P.generate_features_for_clam('/model/', outdir='/pt_files/')
                >>> P.train_clam('NAME', '/pt_files', 'category1', dataset)

            Specify a specific layer from which to generate activations:

                >>> P.generate_features_for_clam(..., layers=['postconv'])

            Manually configure CLAM settings, with 5-fold validation and SVM bag loss:

                >>> import slideflow.clam as clam
                >>> clam_args = clam.get_args(k=5, bag_loss='svm')
                >>> P.generate_features_for_clam(...)
                >>> P.train_clam(..., clam_args=clam_args)
        """

        import slideflow.clam as clam
        from slideflow.clam.datasets.dataset_generic import Generic_MIL_Dataset
        from slideflow.clam.create_attention import export_attention

        # Set up CLAM experiment data directory
        clam_dir = join(self.root, 'clam', exp_name)
        results_dir = join(clam_dir, 'results')
        splits_file = join(self.root, splits)
        if not exists(results_dir): os.makedirs(results_dir)

        # Get base CLAM args/settings if not provided.
        if not clam_args:
            clam_args = clam.get_args()

        # Detect number of features automatically from saved pt_files
        pt_file_paths = [join(pt_files, p) for p in os.listdir(pt_files)
                                           if sf.util.path_to_ext(join(pt_files, p)) == 'pt']
        num_features = clam.detect_num_features(pt_file_paths[0])

        # Note: CLAM only supports categorical outcomes
        labels, unique_labels = dataset.labels(outcome_label_headers, use_float=False)

        if train_slides == validation_slides == 'auto':
            train_slides, validation_slides = {}, {}
            for k in range(clam_args.k):
                train_dts, val_dts = dataset.training_validation_split('categorical',
                                                                        labels,
                                                                        val_strategy='k-fold',
                                                                        splits=splits_file,
                                                                        val_k_fold=clam_args.k,
                                                                        k_fold_iter=k+1)

                train_slides[k] = [sf.util.path_to_name(t) for t in train_dts.tfrecords()]
                validation_slides[k] = [sf.util.path_to_name(t) for t in val_dts.tfrecords()]
        else:
            train_slides = {0: train_slides}
            validation_slides = {0: validation_slides}

        # Remove slides without associated .pt files
        num_skipped = 0
        for k in train_slides:
            num_supplied_slides = len(train_slides[k]) + len(validation_slides[k])
            train_slides[k] = [s for s in train_slides[k] if exists(join(pt_files, s+'.pt'))]
            validation_slides[k] = [s for s in validation_slides[k] if exists(join(pt_files, s+'.pt'))]
            num_skipped += num_supplied_slides - (len(train_slides[k]) + len(validation_slides[k]))

        if num_skipped:
            log.warn(f'Skipping {num_skipped} slides missing associated .pt files.')

        # Set up training/validation splits (mirror base model)
        split_dir = join(clam_dir, 'splits')
        if not exists(split_dir): os.makedirs(split_dir)
        header = ['','train','val','test']
        for k in range(clam_args.k):
            with open(join(split_dir, f'splits_{k}.csv'), 'w') as splits_file:
                writer = csv.writer(splits_file)
                writer.writerow(header)
                # Currently, the below sets the validation & test sets to be the same
                for i in range(max(len(train_slides[k]), len(validation_slides[k]))):
                    row = [i]
                    if i < len(train_slides[k]):        row += [train_slides[k][i]]
                    else:                               row += ['']
                    if i < len(validation_slides[k]):   row += [validation_slides[k][i], validation_slides[k][i]]
                    else:                               row += ['', '']
                    writer.writerow(row)

        # Assign CLAM settings based on this project
        clam_args.model_size = [num_features, 256, 128]
        clam_args.results_dir = results_dir
        clam_args.n_classes = len(unique_labels)
        clam_args.split_dir = split_dir
        clam_args.data_root_dir = pt_files

        # Save clam settings
        sf.util.write_json(vars(clam_args), join(clam_dir, 'experiment.json'))

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

        # Get attention from trained model on validation set(s)
        for k in validation_slides:
            attention_tfrecords = [tfr for tfr in dataset.tfrecords() if sf.util.path_to_name(tfr) in validation_slides[k]]
            attention_dir = join(clam_dir, 'attention', str(k))
            if not exists(attention_dir): os.makedirs(attention_dir)
            export_attention(vars(clam_args),
                             ckpt_path=join(results_dir, f's_{k}_checkpoint.pt'),
                             outdir=attention_dir,
                             pt_files=pt_files,
                             slides=validation_slides[k],
                             reverse_labels=dict(zip(range(len(unique_labels)), unique_labels)),
                             labels=labels)
            if attention_heatmaps:
                heatmaps_dir = join(clam_dir, 'attention_heatmaps', str(k))
                if not exists(heatmaps_dir): os.makedirs(heatmaps_dir)

                for tfr in attention_tfrecords:
                    attention_dict = {}
                    slide = sf.util.path_to_name(tfr)
                    try:
                        with open(join(attention_dir, slide+'.csv'), 'r') as csv_file:
                            reader = csv.reader(csv_file)
                            for row in reader:
                                attention_dict.update({int(row[0]): float(row[1])})
                    except FileNotFoundError:
                        print(f'Unable to find attention scores for slide {slide}, skipping')
                        continue
                    self.generate_tfrecord_heatmap(tfr,
                                                attention_dict,
                                                heatmaps_dir,
                                                tile_px=dataset.tile_px,
                                                tile_um=dataset.tile_um)
