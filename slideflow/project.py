import os
import types
import json
import logging
import itertools
import csv
import copy
import pickle
import numpy as np
import multiprocessing
from os.path import join, exists, basename
from tqdm import tqdm

import slideflow as sf
from slideflow import project_utils, errors
from slideflow.dataset import Dataset
from slideflow.model import ModelParams
from slideflow.util import log, path_to_name
from slideflow.util import colors as col
from slideflow.project_utils import get_validation_settings


class Project:
    """Assists with project organization and execution of pipeline functions.

    Standard instantiation with __init__ assumes a project already exists at
    a given directory, or that configuration will be supplied via kwargs.
    Alternatively, a project may be instantiated using :meth:`from_prompt`,
    which interactively guides users through configuration.

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

    def __init__(self, root, use_neptune=False, **project_kwargs):
        """Initializes project at the specified project folder, creating a new
        project using the specified kwargs if one does not already exist.
        Will create a blank annotations with slide names if one does not exist.

        Args:
            root (str): Path to project directory.

        Keyword Args:
            name (str): Project name. Defaults to 'MyProject'.
            annotations (str): Path to annotations CSV file.
                Defaults to './annotations.csv'
            dataset_config (str): Path to dataset configuration JSON file.
                Defaults to './datasets.json'.
            sources (list(str)): List of dataset sources to include in project.
                Defaults to 'source1'.
            models_dir (str): Path to directory in which to save models.
                Defaults to './models'.
            eval_dir (str): Path to directory in which to save evaluations.
                Defaults to './eval'.

        Raises:
            slideflow.errors.ProjectError: if project folder does not exist,
                or the folder exists but kwargs are provided.
        """

        self.root = root

        if exists(join(root, 'settings.json')) and project_kwargs:
            raise errors.ProjectError(f"Project already exists at {root}")
        elif exists(join(root, 'settings.json')):
            self.load_project(root)
        elif project_kwargs:
            log.info(f"Creating project at {root}...")
            self._settings = project_utils._project_config(**project_kwargs)
            if not exists(root):
                os.makedirs(root)
            self.save()
        else:
            raise errors.ProjectError(f"Project folder {root} does not exist.")

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
        fh = logging.FileHandler(join(root, 'log.txt'))
        fh.setFormatter(sf.util.FileFormatter())
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        # Neptune
        self.use_neptune = use_neptune

    @classmethod
    def from_prompt(cls, root, **kwargs):
        """Initializes project by creating project folder, prompting user for
        project settings, and saving to "settings.json" in project directory.

        Args:
            root (str): Path to project directory.
        """

        if not exists(join(root, 'settings.json')):
            log.info(f'Setting up new project at "{root}"')
            project_utils.interactive_project_setup(root)
        obj = cls(root, **kwargs)
        return obj

    def __repr__(self):
        if self.use_neptune:
            tail = ", use_neptune={!r}".format(self.use_neptune)
        else:
            tail = ''
        return "Project(root={!r}{})".format(self.root, tail)

    @property
    def verbosity(self):
        return logging.getLogger('slideflow').getEffectiveLevel()

    @property
    def annotations(self):
        """Path to annotations file."""
        return self._read_relative_path(self._settings['annotations'])

    @annotations.setter
    def annotations(self, val):
        if not isinstance(val, str):
            raise errors.ProjectError("'annotations' must be a path.")
        self._settings['annotations'] = val

    @property
    def dataset_config(self):
        """Path to dataset configuration JSON file."""
        return self._read_relative_path(self._settings['dataset_config'])

    @dataset_config.setter
    def dataset_config(self, val):
        if not isinstance(val, str):
            raise errors.ProjectError("'dataset_config' must be path to JSON.")
        self._settings['dataset_config'] = val

    @property
    def eval_dir(self):
        """Path to evaluation directory."""
        if 'eval_dir' not in self._settings:
            log.debug("Missing eval_dir in project settings, Assuming ./eval")
            return self._read_relative_path('./eval')
        else:
            return self._read_relative_path(self._settings['eval_dir'])

    @eval_dir.setter
    def eval_dir(self, val):
        if not isinstance(val, str):
            raise errors.ProjectError("'eval_dir' must be a path")
        self._settings['eval_dir'] = val

    @property
    def models_dir(self):
        """Path to models directory."""
        return self._read_relative_path(self._settings['models_dir'])

    @models_dir.setter
    def models_dir(self, val):
        if not isinstance(val, str):
            raise errors.ProjectError("'models_dir' must be a path")
        self._settings['models_dir'] = val

    @property
    def name(self):
        """Descriptive project name."""
        return self._settings['name']

    @name.setter
    def name(self, val):
        if not isinstance(val, str):
            raise errors.ProjectError("'name' must be a str")
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
            raise errors.ProjectError('Neptune workspace must be a string.')
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
            raise errors.ProjectError('API token must be a string.')
        self._settings['neptune_api'] = api_token

    @property
    def sources(self):
        """Returns list of dataset sources active in this project."""
        if 'sources' in self._settings:
            return self._settings['sources']
        elif 'datasets' in self._settings:
            log.debug("'sources' misnamed 'datasets' in project settings.")
            return self._settings['datasets']

    @sources.setter
    def sources(self, v):
        if not isinstance(v, list) or any([not isinstance(v, str) for v in v]):
            raise errors.ProjectError("'sources' must be a list of str")
        self._settings['sources'] = v

    def _read_relative_path(self, path):
        """Converts relative path within project directory to global path."""
        return sf.util.relative_path(path, self.root)

    def _setup_labels(self, dataset, hp, outcomes, config, splits,
                      eval_k_fold=None):
        '''Prepares dataset and labels.'''

        # Assign labels into int
        conf_labels = config['outcome_labels']
        if hp.model_type() == 'categorical':
            if len(outcomes) == 1 and outcomes[0] not in conf_labels:
                outcome_label_to_int = {
                    outcomes[0]: {
                        v: int(k) for k, v in conf_labels.items()
                    }
                }
            else:
                outcome_label_to_int = {
                    o: {
                        v: int(k) for k, v in conf_labels[o].items()
                    } for o in conf_labels
                }
        else:
            outcome_label_to_int = None

        # Get patient-level labels
        use_float = (hp.model_type() in ['linear', 'cph'])
        labels, unique = dataset.labels(
            outcomes,
            use_float=use_float,
            assign=outcome_label_to_int
        )
        # Prepare labels for validation splitting
        if hp.model_type() == 'categorical' and len(outcomes) > 1:
            def process_label(v):
                return '-'.join(map(str, v)) if isinstance(v, list) else v
            split_labels = {k: process_label(v) for k, v in labels.items()}
        else:
            split_labels = labels

        # If using a specific k-fold, load validation plan
        if eval_k_fold:
            log.info(f"Using k-fold iteration {eval_k_fold}")
            _, eval_dts = dataset.train_val_split(
                hp.model_type(),
                split_labels,
                val_strategy=config['validation_strategy'],
                splits=join(self.root, splits),
                val_fraction=config['validation_fraction'],
                val_k_fold=config['validation_k_fold'],
                k_fold_iter=eval_k_fold
            )
            return eval_dts, labels, unique

        # Otherwise use all TFRecords
        else:
            return dataset, labels, unique

    def _prepare_trainer(self, model, outcomes=None, dataset=None,
                         filters=None, checkpoint=None, eval_k_fold=None,
                         splits="splits.json", max_tiles=0, min_tiles=0,
                         input_header=None, mixed_precision=True):

        """Prepares a :class:`slideflow.model.Trainer` for eval or prediction.

        Args:
            model (str): Path to model to evaluate.
            outcomes (str): Str or list of str. Annotation column
                header specifying the outcome label(s).
            dataset (:class:`slideflow.dataset.Dataset`, optional): Dataset
                from which to generate activations. If not supplied, uses all
                data available in a project.
            filters (dict, optional): Filters to use when selecting tfrecords.
                Defaults to None.
            checkpoint (str, optional): Path to cp.ckpt file, if evaluating
                saved checkpoint. Defaults to None.
            eval_k_fold (int, optional): K-fold iteration number to evaluate.
                Defaults to None. If None, evaluate all tfrecords.
            splits (str, optional): Filename of JSON file in which to log
                training/validation splits. Looks for filename in project root.
                Defaults to "splits.json".
            max_tiles (int, optional): Maximum number of tiles from each slide
                to evaluate. Defaults to 0 (include all tiles).
            min_tiles (int, optional): Min number of tiles a slide must have
                to be included in evaluation. Defaults to 0.
            input_header (str, optional): Annotation column header to use as
                additional input. Defaults to None.
            permutation_importance (bool, optional): Determines relative
                importance when using multiple model inputs. Only available for
                Tensorflow backend. Defaults to False.
            mixed_precision (bool, optional): Enable mixed precision.
                Defaults to True.

        Returns:
            Trainer (:class:`slideflow.model.Trainer`),
            dataset (:class:`slideflow.Dataset`)
        """

        if eval_k_fold is not None and outcomes is None:
            raise ValueError('`eval_k_fold` invalid when predicting.')

        # Load hyperparameters from saved model
        config = sf.util.get_model_config(model)
        hp = ModelParams()
        hp.load_dict(config['hp'])
        model_name = f"eval-{basename(model)}"

        # If not provided, detect outcomes from model config
        predicting = (outcomes is None)
        if predicting:
            outcomes = config['outcomes']

        if not isinstance(outcomes, list):
            outcomes = [outcomes]

        # Filter out slides that are blank in the outcome label,
        # or blank in any of the input_header categories
        filter_blank = [o for o in outcomes]
        if input_header is not None and not isinstance(input_header, list):
            input_header = [input_header]
        if input_header is not None:
            filter_blank += input_header

        # Load dataset and annotations for evaluation
        if dataset is None:
            dataset = self.dataset(tile_px=hp.tile_px, tile_um=hp.tile_um)
            dataset = dataset.filter(filters=filters, min_tiles=min_tiles)
        else:
            dataset._assert_size_matches_hp(hp)

        # Set up outcome labels
        if not predicting:
            dataset = dataset.filter(filter_blank=filter_blank)
            eval_dts, labels, unique = self._setup_labels(
                dataset, hp, outcomes, config, splits, eval_k_fold=eval_k_fold
            )
        else:
            eval_dts = dataset
            if sf.backend() == 'torch':
                labels = config['outcome_labels']
            else:
                labels = {}
            unique = list(config['outcome_labels'].values())

        # Set max tiles
        eval_dts = eval_dts.clip(max_tiles)

        # Prepare additional slide-level input
        if input_header:
            _res = project_utils._setup_input_labels(eval_dts, input_header)
            inpt_labels, feature_sizes, slide_inp = _res
        else:
            inpt_labels = None
            feature_sizes = None
            slide_inp = {}

        n_feat = 0 if feature_sizes is None else sum(feature_sizes)
        if feature_sizes and n_feat != sum(config['input_feature_sizes']):
            n_model_feat = sum(config['input_feature_sizes'])
            msg = f'Patient feature matrix (size {n_feat}) '
            msg += f'is different from model (size {n_model_feat}).'
            raise Exception(msg)

        # Log model settings and hyperparameters
        if hp.model_type() == 'categorical':
            outcome_labels = dict(zip(range(len(unique)), unique))
        else:
            outcome_labels = None

        git_commit = sf.util.detect_git_commit()
        model_dir = sf.util.get_new_model_dir(self.eval_dir, model_name)

        # Set missing validation keys to NA
        for v_end in ('strategy', 'fraction', 'k_fold'):
            val_key = f'validation_{v_end}'
            if val_key not in config:
                config[val_key] = 'NA'

        eval_config = {
            'slideflow_version': sf.__version__,
            'project': self.name,
            'backend': sf.backend(),
            'git_commit': git_commit,
            'model_name': model_name,
            'model_path': model,
            'stage': 'evaluation',
            'tile_px': hp.tile_px,
            'tile_um': hp.tile_um,
            'model_type': hp.model_type(),
            'outcomes': outcomes,
            'input_features': input_header,
            'input_feature_sizes': feature_sizes,
            'input_feature_labels': inpt_labels,
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
        if 'norm_fit' in config:
            eval_config.update({'norm_fit': config['norm_fit']})

        # Build a model using the slide list as input
        # and the annotations dictionary as output labels
        trainer = sf.model.trainer_from_hp(
            hp,
            outdir=model_dir,
            labels=labels,
            config=eval_config,
            patients=dataset.patients(),
            slide_input=slide_inp,
            manifest=dataset.manifest(),
            mixed_precision=mixed_precision,
            feature_names=input_header,
            feature_sizes=feature_sizes,
            outcome_names=outcomes,
            use_neptune=self.use_neptune,
            neptune_api=self.neptune_api,
            neptune_workspace=self.neptune_workspace
        )
        if isinstance(model, str):
            trainer.load(model)
        if checkpoint:
            n_features = 0 if not feature_sizes else sum(feature_sizes)
            trainer.model = hp.build_model(
                labels=labels,
                num_slide_features=n_features
            )
            trainer.model.load_weights(checkpoint)

        return trainer, eval_dts

    def _train_hp(self, hp_name, hp, outcomes, val_settings, ctx, filters,
                  filter_blank, input_header, min_tiles, max_tiles,
                  mixed_precision, splits, balance_headers, results_dict,
                  training_kwargs):
        '''Trains a model(s) using the specified hyperparameters.

        Args:
            hp_name (str): Name of hyperparameter combination being run.
            hp (:class:`slideflow.ModelParams`): Model parameters.
            outcomes (str or list(str)): Annotation outcome headers.
            val_settings (:class:`types.SimpleNamspace`): Validation settings.
            ctx (multiprocessing.Context): Multiprocessing context for sharing
                results from isolated training processes.
            filters (dict): Dataset filters.
            filter_blank (list): Excludes slides blank in this annotation col.
            input_header (str or list(str)): Annotation col of additional
                slide-level input.
            min_tiles (int): Only includes tfrecords with >= min_tiles
            max_tiles (int): Cap maximum tiles per tfrecord.
            mixed_precision (bool): Train with mixed precision.
            splits (str): Location of splits file for logging/reading splits.
            balance_headers (str, list(str)): Annotation col headers for
                mini-batch balancing.
            results_dict (dict): Multiprocessing-friendly dict for sending
                results from isolated training processes
            training_kwargs (dict): Keyword arguments for training function.
        '''

        # --- Prepare dataset ---------------------------------------------
        # Filter out slides that are blank in the outcome label,
        # or blank in any of the input_header categories
        if filter_blank:
            filter_blank += [o for o in outcomes]
        else:
            filter_blank = [o for o in outcomes]
        if input_header is not None and not isinstance(input_header, list):
            input_header = [input_header]
        if input_header is not None:
            filter_blank += input_header
        dataset = self.dataset(hp.tile_px, hp.tile_um)
        dataset = dataset.filter(
            filters=filters,
            filter_blank=filter_blank,
            min_tiles=min_tiles
        )
        # --- Load labels -------------------------------------------------
        use_float = (hp.model_type() in ['linear', 'cph'])
        labels, unique = dataset.labels(outcomes, use_float=use_float)
        if hp.model_type() == 'categorical' and len(outcomes) == 1:
            outcome_labels = dict(zip(range(len(unique)), unique))
        elif hp.model_type() == 'categorical':
            outcome_labels = {
                k: dict(zip(range(len(ul)), ul)) for k, ul in unique.items()
            }
        else:
            outcome_labels = dict(zip(range(len(outcomes)), outcomes))
        if hp.model_type() != 'linear' and len(outcomes) > 1:
            log.info('Using multi-outcome approach for categorical outcome')

        # If multiple categorical outcomes are used,
        # create a merged variable for k-fold splitting
        if hp.model_type() == 'categorical' and len(outcomes) > 1:
            split_labels = {
                k: '-'.join(map(str, v)) for k, v in labels.items()
            }
        else:
            split_labels = labels

        # --- Prepare k-fold validation configuration ---------------------
        results_log_path = os.path.join(self.root, 'results_log.csv')
        k_header = val_settings.k_fold_header
        if val_settings.k is not None and not isinstance(val_settings.k, list):
            val_settings.k = [val_settings.k]
        if val_settings.strategy == 'k-fold-manual':
            _, valid_k = dataset.labels(k_header, format='name')
            valid_k = [int(kf) for kf in valid_k]
            k_fold = len(valid_k)
            log.info(f"Manual k-folds detected: {', '.join(valid_k)}")
            if val_settings.k:
                valid_k = [kf for kf in valid_k if kf in val_settings.k]
        elif val_settings.strategy in ('k-fold',
                                       'k-fold-preserved-site',
                                       'bootstrap'):
            k_fold = val_settings.k_fold
            if val_settings.k is None:
                valid_k = list(range(1, k_fold+1))
            else:
                valid_k = [
                    kf for kf in range(1, k_fold+1) if kf in val_settings.k
                ]
        else:
            k_fold = 0
            valid_k = [None]

        # Create model labels
        label_string = '-'.join(outcomes)
        model_name = f'{label_string}-{hp_name}'
        if k_fold is None:
            model_iterations = [model_name]
        else:
            model_iterations = [f'{model_name}-kfold{k}' for k in valid_k]

        s_args = types.SimpleNamespace(
            model_name=model_name,
            outcomes=outcomes,
            k_header=k_header,
            valid_k=valid_k,
            split_labels=split_labels,
            splits=splits,
            labels=labels,
            min_tiles=min_tiles,
            max_tiles=max_tiles,
            outcome_labels=outcome_labels,
            filters=filters,
            training_kwargs=training_kwargs,
            mixed_precision=mixed_precision,
            ctx=ctx,
            results_dict=results_dict,
            bal_headers=balance_headers,
            input_header=input_header
        )

        # --- Train on a specific K-fold ------------ -------------------------
        for k in valid_k:
            s_args.k = k
            self._train_split(dataset, hp, val_settings, s_args)

        # Record results
        for mi in model_iterations:
            if mi not in results_dict or 'epochs' not in results_dict[mi]:
                log.error(f'Training failed for model {model_name}')
                return {}
            else:
                sf.util.update_results_log(
                    results_log_path,
                    mi,
                    results_dict[mi]['epochs']
                )
        log.info(f'Training results saved to {col.green(results_log_path)}')

    def _train_split(self, dataset, hp, val_settings, s_args):
        '''Trains a model for a given training/validation split.

        Args:
            dataset (:class:`slideflow.Dataset`): Dataset to split into
                training and validation.
            hp (:class:`slideflow.ModelParams`): Model parameters.
            val_settings (:class:`types.SimpleNamspace`): Validation settings.
            s_args (:class:`types.SimpleNamspace`): Training settings.
        '''

        # Log current model name and k-fold iteration, if applicable
        k_msg = ''
        if s_args.k is not None:
            k_msg = f' ({val_settings.strategy} #{s_args.k})'
        if log.getEffectiveLevel() <= 20:
            print()
        log.info(f'Training model {col.bold(s_args.model_name)}{k_msg}...')
        log.info(f'Hyperparameters: {hp}')
        log.info(f'Val settings: {json.dumps(vars(val_settings), indent=2)}')

        # --- Set up validation data ------------------------------------------
        manifest = dataset.manifest()

        # Use an external validation dataset if supplied
        if val_settings.source:
            train_dts = dataset
            val_dts = Dataset(
                tile_px=hp.tile_px,
                tile_um=hp.tile_um,
                config=self.dataset_config,
                sources=val_settings.source,
                annotations=val_settings.annotations,
                filters=val_settings.filters,
                filter_blank=val_settings.filter_blank
            )
            is_float = (hp.model_type() in ['linear', 'cph'])
            val_labels, _ = val_dts.labels(s_args.outcomes, use_float=is_float)
            s_args.labels.update(val_labels)
        # Use manual k-fold assignments if indicated
        elif val_settings.strategy == 'k-fold-manual':
            t_filters = {
                s_args.k_header: [j for j in s_args.valid_k if j != s_args.k]
            }
            train_dts = dataset.filter(t_filters)
            val_dts = dataset.filter(filters={s_args.k_header: [s_args.k]})
        # No validation
        elif val_settings.strategy == 'none':
            train_dts = dataset
            val_dts = None
        # Otherwise, calculate k-fold splits
        else:
            if val_settings.strategy == 'k-fold-preserved-site':
                site_labels, _ = dataset.labels(s_args.k_header, format='name')
            else:
                site_labels = None
            train_dts, val_dts = dataset.train_val_split(
                hp.model_type(),
                s_args.split_labels,
                val_strategy=val_settings.strategy,
                splits=join(self.root, s_args.splits),
                val_fraction=val_settings.fraction,
                val_k_fold=val_settings.k_fold,
                k_fold_iter=s_args.k,
                site_labels=site_labels
            )

        # ---- Balance and clip datasets --------------------------------------
        if s_args.bal_headers is None:
            s_args.bal_headers = s_args.outcomes
        train_dts = train_dts.balance(s_args.bal_headers, hp.training_balance)
        train_dts = train_dts.clip(s_args.max_tiles)
        if val_dts:
            val_dts = val_dts.balance(
                s_args.bal_headers,
                hp.validation_balance
            )
            val_dts = val_dts.clip(s_args.max_tiles)
        num_train = len(train_dts.tfrecords())
        num_val = 0 if not val_dts else len(val_dts.tfrecords())
        log.info(f'Using {num_train} training TFRecords, {num_val} validation')

        # --- Prepare additional slide-level input ----------------------------
        if s_args.input_header:
            _res = project_utils._setup_input_labels(
                dataset,
                s_args.input_header,
                val_dts=val_dts
            )
            inpt_labels, feature_sizes, slide_inp = _res
        else:
            inpt_labels = None
            feature_sizes = None
            slide_inp = None

        # --- Initialize model ------------------------------------------------
        # Using the project annotation file, assemble slides for training,
        # as well as the slide annotations dictionary (output labels)
        full_name = s_args.model_name
        if s_args.k is not None:
            full_name += f'-kfold{s_args.k}'
        git_commit = sf.util.detect_git_commit()
        model_dir = sf.util.get_new_model_dir(self.models_dir, full_name)

        # Log model settings and hyperparameters
        config = {
            'slideflow_version': sf.__version__,
            'project': self.name,
            'backend': sf.backend(),
            'git_commit': git_commit,
            'model_name': s_args.model_name,
            'full_model_name': full_name,
            'stage': 'training',
            'img_format': train_dts.img_format,
            'tile_px': hp.tile_px,
            'tile_um': hp.tile_um,
            'max_tiles': s_args.max_tiles,
            'min_tiles': s_args.min_tiles,
            'model_type': hp.model_type(),
            'outcomes': s_args.outcomes,
            'input_features': s_args.input_header,
            'input_feature_sizes': feature_sizes,
            'input_feature_labels': inpt_labels,
            'outcome_labels': s_args.outcome_labels,
            'dataset_config': self.dataset_config,
            'sources': self.sources,
            'annotations': self.annotations,
            'validation_strategy': val_settings.strategy,
            'validation_fraction': val_settings.fraction,
            'validation_k_fold': val_settings.k_fold,
            'k_fold_i': s_args.k,
            'filters': s_args.filters,
            'hp': hp.get_dict(),
            'training_kwargs': s_args.training_kwargs,
        }
        model_kwargs = {
            'hp': hp,
            'name': full_name,
            'manifest': manifest,
            'feature_names': s_args.input_header,
            'feature_sizes': feature_sizes,
            'outcome_names': s_args.outcomes,
            'outdir': model_dir,
            'config': config,
            'patients': dataset.patients(),
            'slide_input': slide_inp,
            'labels': s_args.labels,
            'mixed_precision': s_args.mixed_precision,
            'use_neptune': self.use_neptune,
            'neptune_api': self.neptune_api,
            'neptune_workspace': self.neptune_workspace,
        }
        process = s_args.ctx.Process(target=project_utils._train_worker,
                                     args=((train_dts, val_dts),
                                           model_kwargs,
                                           s_args.training_kwargs,
                                           s_args.results_dict,
                                           self.verbosity))
        process.start()
        log.debug(f'Spawning training process (PID: {process.pid})')
        process.join()

    def add_source(self, name, slides, roi, tiles, tfrecords, path=None):
        """Adds a dataset source to the dataset configuration file.

        Args:
            name (str): Dataset source name.
            slides (str): Path to directory containing slides.
            roi (str): Path to directory containing CSV ROIs.
            tiles (str): Path to directory for storing extracted tiles.
            tfrecords (str): Path to directory for storing TFRecords of tiles.
            path (str, optional): Path to dataset configuration file.
                Defaults to None. If not provided, uses project default.
        """

        if not path:
            path = self.dataset_config
        project_utils.add_source(name, slides, roi, tiles, tfrecords, path)
        if name not in self.sources:
            self.sources += [name]
        self.save()

    def associate_slide_names(self):
        """Automatically associate patients with slides in the annotations."""
        dataset = self.dataset(tile_px=0, tile_um=0, verification=None)
        dataset.update_annotations_with_slidenames(self.annotations)

    def create_blank_annotations(self, filename=None):
        """Creates an empty annotations file.

        Args:
            filename (str): Annotations file destination. If not provided,
                will use project default.
        """

        if filename is None:
            filename = self.annotations
        if exists(filename):
            msg = f"Error creating annotations {filename}; file already exists"
            raise errors.AnnotationsError(msg)
        if not exists(self.dataset_config):
            msg = f"Dataset config {self.dataset_config} missing."
            raise errors.AnnotationsError(msg)

        dataset = Dataset(
            config=self.dataset_config,
            sources=self.sources,
            tile_px=None,
            tile_um=None,
            annotations=None
        )

        all_paths = dataset.slide_paths(apply_filters=False)
        slides = [path_to_name(s) for s in all_paths]
        with open(filename, 'w') as csv_outfile:
            csv_writer = csv.writer(csv_outfile, delimiter=',')
            header = [sf.util.TCGA.patient, 'dataset', 'category']
            csv_writer.writerow(header)
            for slide in slides:
                csv_writer.writerow([slide, '', ''])
        log.info(f"Wrote annotations file to {col.green(filename)}")

    def create_hp_sweep(self, filename='sweep.json', label=None, **kwargs):
        """Prepares a hyperparameter sweep, saving to a batch train TSV file.

        Args:
            label (str, optional): Label to use when naming models in sweep.
                Defaults to None.
            filename (str, optional): Filename for hyperparameter sweep.
                Overwrites existing files. Saves in project root directory.
                Defaults to "sweep.json".
        """

        non_epoch_kwargs = {k: v for k, v in kwargs.items() if k != 'epochs'}
        pdict = copy.deepcopy(non_epoch_kwargs)
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
            mp = ModelParams(**full_params)
            hp_list += [{f'{label}HPSweep{i}': mp.get_dict()}]
        sf.util.write_json(hp_list, os.path.join(self.root, filename))
        log.info(f'Wrote hp sweep (len {len(sweep)}) to {col.green(filename)}')

    def create_hyperparameter_sweep(self, *args, **kwargs):
        log.warn("Project.create_hyperparameter_sweep() -> create_hp_sweep()")
        self.create_hp_sweep(*args, **kwargs)

    def evaluate(self, model, outcomes, dataset=None, filters=None,
                 checkpoint=None, eval_k_fold=None, splits="splits.json",
                 max_tiles=0, min_tiles=0, input_header=None,
                 mixed_precision=True, **kwargs):

        """Evaluates a saved model on a given set of tfrecords.

        Args:
            model (str): Path to model to evaluate.
            outcomes (str): Str or list of str. Annotation column
                header specifying the outcome label(s).
            dataset (:class:`slideflow.dataset.Dataset`, optional): Dataset
                from which to generate activations. If not supplied, will
                calculate activations for all project tfrecords at the
                tile_px/tile_um matching the supplied model, optionally using
                provided filters and filter_blank.
            filters (dict, optional): Filters dict to use when selecting
                tfrecords. Defaults to None. See :meth:`get_dataset`
                documentation for more information on filtering.
            checkpoint (str, optional): Path to cp.ckpt file, if evaluating a
                saved checkpoint. Defaults to None.
            eval_k_fold (int, optional): K-fold iteration number to evaluate.
                Defaults to None. If None, will evaluate all tfrecords
                irrespective of K-fold.
            splits (str, optional): Filename of JSON file in which to log
                train/val splits. Looks for filename in project root directory.
                Defaults to "splits.json".
            max_tiles (int, optional): Maximum number of tiles from each slide
                to evaluate. Defaults to 0. If zero, will include all tiles.
            min_tiles (int, optional): Minimum number of tiles a slide must
                have to be included in evaluation. Defaults to 0.
            input_header (str, optional): Annotation column header to use as
                additional input. Defaults to None.
            mixed_precision (bool, optional): Enable mixed precision.
                Defaults to True.

        Keyword args:
            save_predictions (bool or str, optional): Either True, False, or
                any combination of 'tile', 'patient', or 'slide', as string
                or list of strings. Save tile-level, patient-level, and/or
                slide-level predictions. If True, will save all.
            histogram (bool, optional): Create tile-level histograms for each
                class. Defaults to False.
            permutation_importance (bool, optional): Calculate the permutation
                feature importance.  Determine relative importance when using
                multiple model inputs. Only available for Tensorflow backend.
                Defaults to False.

        Returns:
            Dict: Dictionary of keras training results, nested by epoch.
        """

        trainer, eval_dts = self._prepare_trainer(
            model=model,
            outcomes=outcomes,
            dataset=dataset,
            filters=filters,
            checkpoint=checkpoint,
            eval_k_fold=eval_k_fold,
            splits=splits,
            max_tiles=max_tiles,
            min_tiles=min_tiles,
            input_header=input_header,
            mixed_precision=mixed_precision
        )
        # Perform evaluation
        log.info(f'Evaluating {len(eval_dts.tfrecords())} tfrecords')
        return trainer.evaluate(eval_dts, **kwargs)

    def evaluate_clam(self, exp_name, pt_files, outcomes, tile_px, tile_um,
                      k=0, eval_tag=None, filters=None, filter_blank=None,
                      attention_heatmaps=True):
        """Evaluate CLAM model on saved activations & export attention heatmaps.

        Args:
            exp_name (str): Name of experiment to evaluate (subfolder in clam/)
            pt_files (str): Path to pt_files containing tile-level features.
            outcomes (str or list): Annotation column that specifies labels.
            tile_px (int): Tile width in pixels.
            tile_um (int): Tile width in microns.
            k (int, optional): K-fold / split iteration to evaluate. Evaluates
                the model saved as s_{k}_checkpoint.pt. Defaults to 0.
            eval_tag (str, optional): Unique identifier for this evaluation.
                Defaults to None
            filters (dict, optional): Filters to use when selecting tfrecords.
                Defaults to None.
            filter_blank (list, optional): Exclude slides blank in these cols.
                Defaults to None.
            attention_heatmaps (bool, optional): Save attention heatmaps of
                validation dataset. Defaults to True.

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
            raise errors.CLAMError(f"Unable to find experiment '{exp_name}'")

        log.info(f'Loading experiment from {col.green(exp_name)}, k={k}')
        eval_dir = join(exp_name, 'eval')
        if not exists(eval_dir):
            os.makedirs(eval_dir)

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
            log.info(f"Tag {base_tag} already exists, using tag 'eval_tag'")

        # Load trained model checkpoint
        ckpt_path = join(exp_name, 'results', f's_{k}_checkpoint.pt')
        eval_dir = join(eval_dir, eval_tag)
        if not exists(eval_dir):
            os.makedirs(eval_dir)
        args_dict = sf.util.load_json(join(exp_name, 'experiment.json'))
        args = types.SimpleNamespace(**args_dict)
        args.save_dir = eval_dir

        dataset = self.dataset(
            tile_px=tile_px,
            tile_um=tile_um,
            filters=filters,
            filter_blank=filter_blank
        )
        slides = dataset.slides()
        eval_slides = [s for s in slides if exists(join(pt_files, s+'.pt'))]
        dataset = dataset.filter(filters={'slide': eval_slides})
        slide_labels, unique_labels = dataset.labels(outcomes, use_float=False)

        # Set up evaluation annotations file based off existing pt_files
        outcome_dict = dict(zip(range(len(unique_labels)), unique_labels))
        with open(join(eval_dir, 'eval_annotations.csv'), 'w') as eval_file:
            writer = csv.writer(eval_file)
            header = [sf.util.TCGA.patient, sf.util.TCGA.slide, outcomes]
            writer.writerow(header)
            for slide in eval_slides:
                row = [slide, slide, outcome_dict[slide_labels[slide]]]
                writer.writerow(row)

        clam_dataset = Generic_MIL_Dataset(
            csv_path=join(eval_dir, 'eval_annotations.csv'),
            data_dir=pt_files,
            shuffle=False,
            seed=args.seed,
            print_info=True,
            label_col=outcomes,
            label_dict=dict(zip(unique_labels, range(len(unique_labels)))),
            patient_strat=False,
            ignore=[]
        )

        clam.evaluate(ckpt_path, args, clam_dataset)

        # Get attention from trained model on validation set
        attention_tfrecords = dataset.tfrecords()
        attention_dir = join(eval_dir, 'attention')
        if not exists(attention_dir):
            os.makedirs(attention_dir)
        reverse_labels = dict(zip(range(len(unique_labels)), unique_labels))
        export_attention(
            args_dict,
            ckpt_path=ckpt_path,
            outdir=attention_dir,
            pt_files=pt_files,
            slides=dataset.slides(),
            reverse_labels=reverse_labels,
            labels=slide_labels
        )
        if attention_heatmaps:
            heatmaps_dir = join(eval_dir, 'attention_heatmaps')
            if not exists(heatmaps_dir):
                os.makedirs(heatmaps_dir)

            for tfr in attention_tfrecords:
                attention_dict = {}
                slide = path_to_name(tfr)
                try:
                    with open(join(attention_dir, slide+'.csv'), 'r') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            attention_dict.update({int(row[0]): float(row[1])})
                except FileNotFoundError:
                    print(f'No attention scores for slide {slide}, skipping')
                    continue
                self.generate_tfrecord_heatmap(
                    tfr,
                    tile_px=tile_px,
                    tile_um=tile_um,
                    tile_dict=attention_dict,
                    outdir=heatmaps_dir
                )

    def extract_tiles(self, tile_px, tile_um, filters=None, filter_blank=None,
                      **kwargs):
        """Extracts tiles from slides. Preferred use is calling
        :func:`slideflow.dataset.Dataset.extract_tiles` on a
        :class:`slideflow.dataset.Dataset` directly.

        Args:
            save_tiles (bool, optional): Save tile images in loose format.
                Defaults to False.
            save_tfrecords (bool, optional): Save tile images as TFRecords.
                Defaults to True.
            source (str, optional): Process slides only from this source.
                Defaults to None (all slides in project).
            stride_div (int, optional): Stride divisor. Defaults to 1.
                A stride of 1 will extract non-overlapping tiles.
                A stride_div of 2 will extract overlapping tiles with a stride
                equal to 50% of the tile width.
            enable_downsample (bool, optional): Enable downsampling when
                reading slides. Defaults to True. This may result in corrupted
                image tiles if downsampled slide layers are corrupted or
                incomplete. Recommend manual confirmation of tile integrity.
            roi_method (str, optional): Either 'inside', 'outside' or 'ignore'.
                Indicates whether tiles are extracted inside or outside ROIs,
                or if ROIs are ignored entirely. Defaults to 'inside'.
            skip_missing_roi (bool, optional): Skip slides that missing ROIs.
                Defaults to False.
            skip_extracted (bool, optional): Skip already extracted slides.
                Defaults to True.
            tma (bool, optional): Reads slides as Tumor Micro-Arrays (TMAs),
                detecting and extracting tumor cores. Defaults to False.
            randomize_origin (bool, optional): Randomize pixel starting
                position during extraction. Defaults to False.
            buffer (str, optional): Copy slides here before extraction.
                Improves processing speed if using an SSD/ramdisk buffer.
                Defaults to None.
            num_workers (int, optional): Extract tiles from this many slides
                simultaneously. Defaults to 1.
            q_size (int, optional): Queue size for buffer. Defaults to 4.
            qc (str, optional): 'otsu', 'blur', 'both', or None. Perform blur
                detection quality control - discarding tiles with detected
                out-of-focus regions or artifact - and/or otsu's method.
                Defaults to None.
            report (bool, optional): Save a PDF report of tile extraction.
                Defaults to True.

        Keyword Args:
            normalizer (str, optional): Normalization strategy.
                Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image.
                Defaults to None (use internal image at slide.norm_tile.jpg).
            whitespace_fraction (float, optional): Range 0-1. Defaults to 1.
                Discard tiles with this fraction of whitespace. If 1, will not
                perform whitespace filtering.
            whitespace_threshold (int, optional): Range 0-255. Threshold above
                which a pixel (RGB average) is considered whitespace.
                Defaults to 230.
            grayspace_fraction (float, optional): Range 0-1. Discard tiles with
                this fraction of grayspace. If 1, will not perform grayspace
                filtering. Defaults to 0.6.
            grayspace_threshold (float, optional): Range 0-1. Pixels in HSV
                format with saturation below this are considered grayspace.
                Defaults to 0.05.
            img_format (str, optional): 'png' or 'jpg'. Defaults to 'jpg'.
                Image format to use in tfrecords. PNG (lossless) for
                fidelity, JPG (lossy) for efficiency.
            full_core (bool, optional): Only used if extracting from TMA. Save
                entire TMA core as image. Otherwise, will extract sub-images
                from each core at the tile micron size. Defaults to False.
            shuffle (bool, optional): Shuffle tiles before tfrecords storage.
                Defaults to True.
            num_threads (int, optional): Threads for each tile extractor.
                Defaults to 4.
            qc_blur_radius (int, optional): Blur radius for out-of-focus area
                detection. Used if qc=True. Defaults to 3.
            qc_blur_threshold (float, optional): Blur threshold for detecting
                out-of-focus areas. Used if qc=True. Defaults to 0.1.
            qc_filter_threshold (float, optional): Float between 0-1.
                Tiles with more than this proportion of blur will be discarded.
                Used if qc=True. Defaults to 0.6.
            qc_mpp (float, optional): Microns-per-pixel indicating image
                magnification level at which quality control is performed.
                Defaults to mpp=4 (effective magnification 2.5 X)
            dry_run (bool, optional): Determine tiles that would be extracted,
                but do not export any images. Defaults to None.
        """

        dataset = self.dataset(
            tile_px,
            tile_um,
            filters=filters,
            filter_blank=filter_blank,
            verification='slides'
        )
        dataset.extract_tiles(**kwargs)

    def generate_features(self, model, dataset=None, filters=None,
                          filter_blank=None, min_tiles=0, max_tiles=0,
                          outcomes=None, torch_export=None, **kwargs):
        """Calculate layer features / activations.

        Args:
            model (str): Path to model
            dataset (:class:`slideflow.dataset.Dataset`, optional): Dataset
                from which to generate activations. If not supplied, calculate
                activations for all tfrecords compatible with the model,
                optionally using provided filters and filter_blank.
            filters (dict, optional): Filters to use when selecting tfrecords.
                 Defaults to None.
            filter_blank (list, optional): Slides blank in these columns will
                be excluded. Defaults to None.
            min_tiles (int, optional): Only include slides with this minimum
                number of tiles. Defaults to 0.
            max_tiles (int, optional): Only include maximum of this many tiles
                per slide. Defaults to 0 (all tiles).
            outcomes (list, optional): Column header(s) in annotations file.
                Used for category-level comparisons. Defaults to None.
            torch_export (str, optional): Path. Export activations to
                torch-compatible file at this location. Defaults to None.

        Keyword Args:
            layers (list(str)): Layers from which to generate activations.
                Defaults to 'postconv'.
            export (str): Path to CSV file. Save activations in CSV format.
                Defaults to None.
            cache (str): Path to PKL file. Cache activations at this location.
                Defaults to None.
            include_logits (bool): Generate and store logit predictions along
                with layer activations. Defaults to True.
            batch_size (int): Batch size to use when calculating activations.
                Defaults to 32.

        Returns:
            :class:`slideflow.model.DatasetFeatures`:
        """

        # Setup directories
        stats_root = join(self.root, 'stats')
        if not exists(stats_root):
            os.makedirs(stats_root)

        # Load dataset for evaluation
        config = sf.util.get_model_config(model)
        if dataset is None:
            dataset = self.dataset(
                tile_px=config['hp']['tile_px'],
                tile_um=config['hp']['tile_um'],
                filters=filters,
                filter_blank=filter_blank
            )
        else:
            dataset._assert_size_matches_hp(config['hp'])

        # Prepare dataset and annotations
        dataset = dataset.filter(
            filters=filters,
            filter_blank=filter_blank,
            min_tiles=min_tiles
        )
        dataset = dataset.clip(max_tiles)
        if outcomes is not None:
            outcome_annotations = dataset.labels(outcomes, format='name')[0]
        else:
            outcome_annotations = None

        df = sf.model.DatasetFeatures(model=model,
                                      dataset=dataset,
                                      annotations=outcome_annotations,
                                      manifest=dataset.manifest(),
                                      **kwargs)
        if torch_export:
            df.export_to_torch(torch_export)
        return df

    def generate_features_for_clam(self, model, outdir='auto',
                                   layers='postconv', max_tiles=0,
                                   min_tiles=16, filters=None,
                                   filter_blank=None, force_regenerate=False):

        """Generate tile-level features for slides for use with CLAM.

        Args:
            model (str): Path to model from which to generate activations.
                May provide either this or "pt_files"
            outdir (str, optional): Save exported activations in .pt format.
                Defaults to 'auto' (project directory).
            layers (list, optional): Which model layer(s) generate activations.
                Defaults to 'postconv'.
            max_tiles (int, optional): Maximum tiles to take per slide.
                Defaults to 0.
            min_tiles (int, optional): Minimum tiles per slide. Skip slides
                not meeting this threshold. Defaults to 8.
            filters (dict, optional): Filters to use when selecting tfrecords.
                Defaults to None.
            filter_blank (list, optional): Slides blank in these columns will
                be excluded. Defaults to None.
            force_regenerate (bool, optional): Forcibly regenerate activations
                for all slides even if .pt file exists. Defaults to False.

        Returns:
            Path to directory containing exported .pt files
        """

        if min_tiles < 8:
            raise ValueError('Slides must have at >=8 tiles to train CLAM.')

        # First, ensure the model is valid with a hyperparameters file
        config = sf.util.get_model_config(model)
        tile_px = config['tile_px']
        tile_um = config['tile_um']

        # Set up the pt_files directory for storing model activations
        if outdir.lower() == 'auto':
            if 'k_fold_i' in config:
                _end = f"_kfold{config['k_fold_i']}"
            else:
                _end = ''
            outdir = join(self.root, 'pt_files', config['model_name'] + _end)
        if not exists(outdir):
            os.makedirs(outdir)

        # Detect already generated pt files
        done = [
            path_to_name(f) for f in os.listdir(outdir)
            if sf.util.path_to_ext(join(outdir, f)) == 'pt'
        ]

        if force_regenerate or not len(done):
            activation_filters = filters
        else:
            pt_dataset = self.dataset(
                tile_px,
                tile_um,
                filters=filters,
                filter_blank=filter_blank
            )
            all_slides = pt_dataset.slides()
            slides_to_generate = [s for s in all_slides if s not in done]
            if len(slides_to_generate) != len(all_slides):
                to_skip = len(all_slides) - len(slides_to_generate)
                skip_p = f'{to_skip}/{len(all_slides)}'
                log.info(f"Skipping {skip_p} finished slides.")
            if not slides_to_generate:
                log.warn("No slides to generate CLAM features.")
                return outdir
            activation_filters = {} if filters is None else filters.copy()
            activation_filters['slide'] = slides_to_generate
            filtered_dataset = self.dataset(
                tile_px,
                tile_um,
                filters=activation_filters,
                filter_blank=filter_blank
            )
            filtered_slides_to_generate = filtered_dataset.slides()
            log.info(f'Skipping {len(done)} files already done.')
            log.info(f'Working on {len(filtered_slides_to_generate)} slides')

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

    def generate_heatmaps(self, model, filters=None, filter_blank=None,
                          outdir=None, resolution='low', batch_size=32,
                          roi_method='inside', buffer=None, num_threads=None,
                          skip_completed=False, **kwargs):

        """Creates predictive heatmap overlays on a set of slides.

        Args:
            model (str): Path to Tensorflow model.
            filters (dict, optional): Filters to use when selecting tfrecords.
                Defaults to None.
            filter_blank (list, optional): Exclude slides blank in these cols.
                Defaults to None.
            outdir (path, optional): Directory in which to save heatmap images.
            resolution (str, optional): Heatmap resolution. Defaults to 'low'.
                "low" uses a stride equal to tile width.
                "medium" uses a stride equal 1/2 tile width.
                "high" uses a stride equal to 1/4 tile width.
            batch_size (int, optional): Batch size during heatmap calculation.
                Defaults to 64.
            roi_method (str, optional): 'inside', 'outside', or 'none'.
                Determines where heatmap should be made with respect to ROI.
                Defaults to 'inside'.
            buffer (str, optional): Path to which slides are copied prior to
                heatmap generation. Defaults to None.
            num_threads (int, optional): Number of threads for tile extraction.
                Defaults to CPU core count.
            skip_completed (bool, optional): Skip heatmaps for slides that
                already have heatmaps in target directory.

        Keyword args:
            show_roi (bool): Show ROI on heatmaps.
            interpolation (str): Interpolation strategy for predictions.
                Defaults to None.
                Includes all matplotlib imshow interpolation options.
            logit_cmap: Function or a dict used to create heatmap colormap.
                If None (default), separate heatmaps are generated for each
                category, with color representing category prediction.
                Each image tile will generate a list of preds of length O,
                If logit_cmap is a function, then the logit predictions will
                be passed, where O is the number of label categories.
                and the function is expected to return [R, G, B] values.
                If the logit_cmap is a dictionary, it should map 'r', 'g', and
                'b' to label indices; the prediction for these label categories
                will be mapped to corresponding colors. Thus, the corresponding
                color will only reflect predictions of up to three labels.
                Example (this would map predictions for label 0 to red, 3 to
                green, etc): {'r': 0, 'g': 3, 'b': 1 }
            vmin (float): Minimimum value to display on heatmap. Defaults to 0.
            vcenter (float): Center value for color display on heatmap.
                Defaults to 0.5.
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
            raise ValueError(f"Invalid resolution '{resolution}'.")
        heatmap_args.stride_div = stride_div
        heatmap_args.verbosity = self.verbosity

        # Attempt to auto-detect supplied model name
        model_name = os.path.basename(model)
        if 'model_name' in config:
            model_name = config['model_name']

        # Make output directory
        outdir = outdir if outdir else join(self.root, 'heatmaps', model_name)
        if not exists(outdir):
            os.makedirs(outdir)
        heatmap_args.outdir = outdir

        # Any function loading a slide must be kept in an isolated process,
        # as loading >1 slide in a single process causes instability.
        # I suspect this is a libvips or openslide issue but I haven't been
        # able to identify the root cause. Isolating processes when multiple
        # slides are to be processed sequentially is a functional workaround.
        for slide in slide_list:
            name = path_to_name(slide)
            if (skip_completed and exists(join(outdir, f'{name}-custom.png'))):
                log.info(f'Skipping completed heatmap for slide {name}')
                return

            ctx = multiprocessing.get_context('spawn')
            process = ctx.Process(target=project_utils._heatmap_worker,
                                  args=(slide, heatmap_args, kwargs))
            process.start()
            process.join()

    def generate_mosaic(self, df, dataset=None, filters=None,
                        filter_blank=None, outcomes=None, map_slide=None,
                        show_prediction=None, restrict_pred=None,
                        predict_on_axes=None, max_tiles=0, umap_cache=None,
                        use_float=False, low_memory=False, use_norm=True,
                        **kwargs):

        """Generates a mosaic map by overlaying images onto mapped tiles.
            Image tiles are extracted from the provided set of TFRecords, and
            predictions + features from layer activations are calculated using
            the specified model. Tiles are mapped either with UMAP of layer
            activations (default behavior), or by using outcome predictions for
            two categories, mapped to X- and Y-axis (via predict_on_axes).

        Args:
            df (:class:`slideflow.model.DatasetFeatures`): Dataset.
            dataset (:class:`slideflow.dataset.Dataset`, optional): Dataset
                from which to generate mosaic. If not supplied, will generate
                mosaic for all tfrecords at the tile_px/tile_um matching
                the supplied model, optionally using filters/filter_blank.
            filters (dict, optional): Filters dict to use when selecting
                tfrecords. Defaults to None.
            filter_blank (list, optional): Slides blank in these columns will
                be excluded. Defaults to None.
            outcomes (list, optional): Column name in annotations file from
                which to read category labels.
            map_slide (str, optional): None (default), 'centroid' or 'average'.
                If provided, will map slides using slide-level calculations,
                either mapping centroid tiles if 'centroid', or calculating
                node averages across tiles in a slide and mapping slide-level
                node averages, if 'average'.
            show_prediction (int or str, optional): May be either int or str,
                corresponding to label category. Predictions for this category
                will be displayed on the exported UMAP plot.
            restrict_pred (list, optional): List of int, if provided, restrict
                predictions to these categories. Final tile-level prediction
                is made by choosing category with highest logit.
            predict_on_axes (list, optional): (int, int). Each int corresponds
                to an label category id. If provided, predictions are generated
                for these two labels categories; tiles are then mapped with
                these predictions with the pattern (x, y) and the mosaic is
                generated from this map. This replaces the default UMAP.
            max_tiles (int, optional): Limits tiles taken from each slide.
                Defaults to 0.
            umap_cache (str, optional): Path to PKL file in which to save/cache
                UMAP coordinates. Defaults to None.
            use_float (bool, optional): Interpret labels as continuous instead
                of categorical. Defaults to False.
            low_memory (bool, optional): Limit memory during UMAP calculations.
                Defaults to False.
            use_norm (bool, optional): Display image tiles using the normalizer
                used during model training (if applicable). Detected from
                a model's metadata file (params.json). Defaults to True.

        Keyword Args:
            resolution (str): Mosaic map resolution. Low, medium, or high.
            num_tiles_x (int): Specifies the size of the mosaic map grid.
            expanded (bool): Controls tile assignment on grid spaces.
                If False, tile assignment is strict.
                If True, allows displaying nearby tiles if a grid is empty.
                Defaults to False.
            leniency (float): UMAP leniency. Defaults to 1.5.

        Returns:
            :class:`slideflow.mosaic.Mosaic`: Mosaic object.
        """

        # Set up paths
        stats_root = join(self.root, 'stats')
        mosaic_root = join(self.root, 'mosaic')
        if not exists(stats_root):
            os.makedirs(stats_root)
        if not exists(mosaic_root):
            os.makedirs(mosaic_root)

        # Prepare dataset & model
        config = sf.util.get_model_config(df.model)
        if dataset is None:
            tile_px, tile_um = config['hp']['tile_px'], config['hp']['tile_um']
            dataset = self.dataset(tile_px=tile_px, tile_um=tile_um)
        else:
            dataset._assert_size_matches_hp(config['hp'])
            tile_px = dataset.tile_px

        # Filter and clip dataset
        dataset = dataset.filter(filters=filters, filter_blank=filter_blank)
        dataset = dataset.clip(max_tiles)

        # Get TFrecords, and prepare a list for focus, if requested
        tfr = dataset.tfrecords()
        n_slides = len([t for t in tfr if path_to_name(t) in df.slides])
        log.info(f'Generating mosaic from {n_slides} slides')

        # If a header category is supplied and we are not showing predictions,
        # then assign slide labels from annotations
        model_type = config['model_type']
        if model_type == 'linear':
            use_float = True
        if outcomes and (show_prediction is None):
            labels, _ = dataset.labels(outcomes,
                                       use_float=use_float,
                                       format='name')
        else:
            labels = {}

        # If showing predictions, try to automatically load prediction labels
        if (show_prediction is not None) and (not use_float):
            outcome_labels = config['outcome_labels']
            model_type = model_type if model_type else config['model_type']
            log.info(f'Loaded pred labels found at {col.green(df.model)}')

        if predict_on_axes:
            # Create mosaic with x- and y-axis corresponding to predictions
            umap_x, umap_y, umap_meta = df.map_to_predictions(
                predict_on_axes[0],
                predict_on_axes[1]
            )
            umap = sf.SlideMap.from_precalculated(slides=dataset.slides(),
                                                  x=umap_x,
                                                  y=umap_y,
                                                  meta=umap_meta)
        else:
            # Create mosaic map from UMAP of layer activations
            umap = sf.SlideMap.from_features(df,
                                             map_slide=map_slide,
                                             prediction_filter=restrict_pred,
                                             cache=umap_cache,
                                             low_memory=low_memory)

        # If displaying centroid AND predictions, show slide-level predictions
        # rather than tile-level predictions
        if (map_slide == 'centroid') and show_prediction is not None:
            log.info('Showing slide-level predictions at point of centroid')

            # If not model has not been assigned, assume categorical model
            model_type = model_type if model_type else 'categorical'

            # Get predictions
            if model_type == 'categorical':
                s_pred = df.logits_predict(restrict_pred)
                s_perc = df.logits_percent(restrict_pred)
            else:
                s_pred = s_perc = df.logits_mean()

            # If show_prediction is provided (either a number or string),
            # then display ONLY the prediction for the provided category
            if type(show_prediction) == int:
                log.info(f'Showing preds for {show_prediction} as colormap')
                labels = {
                    k: v[show_prediction] for k, v in s_perc.items()
                }
                show_prediction = None
            elif type(show_prediction) == str:
                log.info(f'Showing preds for {show_prediction} as colormap')
                reversed_labels = {v: k for k, v in outcome_labels.items()}
                if show_prediction not in reversed_labels:
                    raise ValueError(f"Unknown category '{show_prediction}'")
                labels = {
                    k: v[int(reversed_labels[show_prediction])]
                    for k, v in s_perc.items()
                }
                show_prediction = None
            elif use_float:
                # Displaying linear predictions needs to be implemented here
                msg = "Showing slide preds not supported for linear outcomes."
                raise NotImplementedError(msg)
            # Otherwise, show_prediction is assumed to be just "True",
            # in which case show categorical predictions
            else:
                try:
                    labels = {
                        k: outcome_labels[v] for k, v in s_pred.items()
                    }
                except KeyError:
                    # Try interpreting prediction label keys as strings
                    labels = {
                        k: outcome_labels[str(v)] for k, v in s_pred.items()
                    }

        if labels:
            umap.label_by_slide(labels)
        if show_prediction and (map_slide != 'centroid'):
            umap.label_by_meta('prediction', translation_dict=outcome_labels)
        umap.filter(dataset.slides())

        mosaic = sf.Mosaic(
            umap,
            dataset.tfrecords(),
            normalizer=(df.normalizer if use_norm else None),
            **kwargs
        )
        return mosaic

    def generate_mosaic_from_annotations(self, header_x, header_y, dataset,
                                         model=None, outcomes=None,
                                         max_tiles=100, use_optimal_tile=False,
                                         cache=None, batch_size=32, **kwargs):

        """Generates mosaic map by overlaying images onto a set of mapped tiles.
            Slides are mapped with slide-level annotations, x-axis determined
            from header_x, y-axis from header_y. If use_optimal_tile is False
            and no model is provided, tje first image tile in each TFRecord
            will be displayed. If optimal_tile is True, layer
            activations for all tiles in each slide are calculated using the
            provided model, and the tile nearest to centroid is used.

        Args:
            header_x (str): Annotations file header with X-axis coords.
            header_y (str): Annotations file header with Y-axis coords.
            dataset (:class:`slideflow.dataset.Dataset`): Dataset object.
            model (str, optional): Path to Tensorflow model to use when
                generating layer activations.
            Defaults to None.
                If not provided, mosaic will not be calculated or saved.
                If provided, saved in project mosaic directory.
            outcomes (list(str)): Column name(s) in annotations file from which
                to read category labels.
            filters (dict, optional): Filters to use when selecting tfrecords.
                Defaults to None.
            max_tiles (int, optional): Limits the number of tiles taken from
                each slide. Defaults to 0.
            use_optimal_tile (bool, optional): Use model to calculate layer
                activations for all tiles in each slide, and choosing tile
                nearest centroid for each slide for display.
            cache (str, optional): Path to PKL file to cache node
                activations. Defaults to None.
            batch_size (int, optional): Batch size for model. Defaults to 64.

        Keyword Args:
            resolution (str): Resolution of the mosaic. Low, medium, or high.
            num_tiles_x (int): Specifies the size of the mosaic map grid.
            expanded (bool): Controls tile assignment on grid spaces.
                If False, tile assignment is strict.
                If True, allows displaying nearby tiles if a grid is empty.
                Defaults to False.
            leniency (float): UMAP leniency. Defaults to 1.5.
        """

        # Setup paths
        stats_root = join(self.root, 'stats')
        mosaic_root = join(self.root, 'mosaic')
        if not exists(stats_root):
            os.makedirs(stats_root)
        if not exists(mosaic_root):
            os.makedirs(mosaic_root)

        # Filter dataset to exclude slides blank in the x and y header columns
        dataset = dataset.filter(filter_blank=[header_x, header_y])
        dataset = dataset.clip(max_tiles)

        # We are assembling a list of slides from the TFRecords path list,
        # because we only want to use slides that have a corresponding TFRecord
        # (some slides did not have a large enough ROI for tile extraction
        # & some slides may be in the annotations but are missing a slide)
        slides = [path_to_name(tfr) for tfr in dataset.tfrecords()]
        labels, _ = dataset.labels([header_x, header_y], use_float=True)
        slide_to_category, _ = dataset.labels(outcomes, format='name')

        umap_x = np.array([labels[slide][0] for slide in slides])
        umap_y = np.array([labels[slide][1] for slide in slides])

        if use_optimal_tile and not model:
            raise ValueError("Optimal tile calculation requires a model.")
        elif use_optimal_tile:
            # Calculate most representative tile in each TFRecord for display
            df = sf.model.DatasetFeatures(model=model,
                                          dataset=dataset,
                                          batch_size=batch_size,
                                          cache=cache)
            opt_ind, _ = sf.stats.calculate_centroid(df.activations)

            # Restrict mosaic to only slides that had enough tiles to
            # calculate an optimal index from centroid
            success_slides = list(opt_ind.keys())
            sf.util.multi_warn(
                slides,
                lambda x: x not in success_slides,
                'Unable to calculate optimal tile for {}, skipping'
            )
            umap_x = np.array([labels[slide][0] for slide in success_slides])
            umap_y = np.array([labels[slide][1] for slide in success_slides])
            umap_meta = [{
                            'slide': slide,
                            'index': opt_ind[slide]
                         } for slide in success_slides]
        else:
            # Take the first tile from each slide/TFRecord
            umap_meta = [{'slide': slide, 'index': 0} for slide in slides]

        umap = sf.SlideMap.from_precalculated(slides=slides,
                                              x=umap_x,
                                              y=umap_y,
                                              meta=umap_meta)
        umap.label_by_slide(slide_to_category)
        mosaic = sf.Mosaic(
            umap,
            dataset.tfrecords(),
            tile_select='centroid' if use_optimal_tile else 'nearest',
            **kwargs
        )
        return mosaic

    def generate_thumbnails(self, size=512, dataset=None, filters=None,
                            filter_blank=None, roi=False,
                            enable_downsample=True):
        """Generates square slide thumbnails with black borders of fixed size,
        and saves to project folder.

        Args:
            size (int, optional): Width/height of thumbnail in pixels.
                Defaults to 512.
            dataset (:class:`slideflow.dataset.Dataset`, optional): Dataset
                from which to generate activations. If not supplied, will
                calculate activations for all tfrecords at the tile_px/tile_um
                matching the supplied model, optionally using provided filters
                and filter_blank.
            filters (dict, optional): Filters to use when selecting tfrecords.
                Defaults to None.
            filter_blank (list, optional): Exclude slides blank in these cols.
                Defaults to None.
            roi (bool, optional): Include ROI in the thumbnail images.
                Defaults to False.
            enable_downsample (bool, optional): If True and a thumbnail is not
                embedded in the slide file, downsampling is permitted to
                accelerate thumbnail calculation.
        """

        from slideflow.slide import WSI
        log.info('Generating thumbnails...')

        thumb_folder = join(self.root, 'thumbs')
        if not exists(thumb_folder):
            os.makedirs(thumb_folder)
        if dataset is None:
            dataset = self.dataset(tile_px=0, tile_um=0)
        dataset = dataset.filter(filters=filters, filter_blank=filter_blank)
        slide_list = dataset.slide_paths()
        rois = dataset.rois()
        log.info(f'Saving thumbnails to {col.green(thumb_folder)}')

        for slide_path in slide_list:
            fmt_name = col.green(path_to_name(slide_path))
            log.info(f'Working on {fmt_name}...', end='')
            whole_slide = WSI(slide_path,
                              tile_px=1000,
                              tile_um=1000,
                              stride_div=1,
                              enable_downsample=enable_downsample,
                              rois=rois,
                              roi_method='inside',
                              skip_missing_roi=roi)
            if roi:
                thumb = whole_slide.thumb(rois=True)
            else:
                thumb = whole_slide.square_thumb(size)
            thumb.save(join(thumb_folder, f'{whole_slide.name}.png'))
        log.info('Thumbnail generation complete.')

    def generate_tfrecord_heatmap(self, tfrecord, tile_px, tile_um, tile_dict,
                                  outdir=None):
        """Creates a tfrecord-based WSI heatmap using a dictionary of tile
        values for heatmap display, saving to project root directory.

        Args:
            tfrecord (str): Path to tfrecord
            tile_dict (dict): Dictionary mapping tfrecord indices to a
                tile-level value for display in heatmap format
            tile_px (int): Tile width in pixels
            tile_um (int): Tile width in microns

        Returns:
            Dictionary mapping slide names to dict of statistics
                (mean, median, above_0, and above_1)
        """

        name = path_to_name(tfrecord)
        dataset = self.dataset(tile_px=tile_px, tile_um=tile_um)
        slide_paths = {
            path_to_name(sp): sp for sp in dataset.slide_paths()
        }
        if outdir is None:
            outdir = self.root
        try:
            slide_path = slide_paths[name]
        except KeyError:
            raise errors.SlideNotFoundError(f'Unable to find slide {name}')
        sf.util.tfrecord_heatmap(
            tfrecord=tfrecord,
            slide=slide_path,
            tile_px=tile_px,
            tile_um=tile_um,
            tile_dict=tile_dict,
            outdir=outdir
        )

    def dataset(self, tile_px=None, tile_um=None, verification='both',
                **kwargs):
        """Returns :class:`slideflow.Dataset` object using project settings.

        Args:
            tile_px (int): Tile size in pixels
            tile_um (int): Tile size in microns

        Keyword Args:
            filters (dict, optional): Filters for selecting tfrecords.
                Defaults to None.
            filter_blank (list, optional): Exclude slides blank in these cols.
                Defaults to None.
            min_tiles (int, optional): Min tiles a slide must have.
                Defaults to 0.
            config (str, optional): Path to dataset configuration JSON file.
                Defaults to project default.
            sources (str, list(str), optional): Dataset sources to use from
                configuration. Defaults to project default.
            verification (str, optional): 'tfrecords', 'slides', or 'both'.
                If 'slides', verify all annotations are mapped to slides.
                If 'tfrecords', check that TFRecords exist and update manifest.
                Defaults to 'both'.
        """

        if 'config' not in kwargs:
            kwargs['config'] = self.dataset_config
        if 'sources' not in kwargs:
            kwargs['sources'] = self.sources
        try:
            if self.annotations and exists(self.annotations):
                annotations = self.annotations
            else:
                self.annotations = None
            dataset = Dataset(
                tile_px=tile_px,
                tile_um=tile_um,
                annotations=annotations,
                **kwargs
            )
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

        # Enable logging
        if exists(join(path, 'settings.json')):
            self._settings = sf.util.load_json(join(path, 'settings.json'))
        else:
            raise errors.ProjectError('Unable to find settings.json.')

    def predict(self, model, dataset=None, filters=None, checkpoint=None,
                eval_k_fold=None, splits="splits.json", max_tiles=0,
                min_tiles=0, batch_size=32, input_header=None, format='csv',
                mixed_precision=True, **kwargs):
        """Evaluates a saved model on a given set of tfrecords.

        Args:
            model (str): Path to model to evaluate.
            outcomes (str): Str or list of str. Annotation header specifying
                outcome label(s).
            dataset (:class:`slideflow.dataset.Dataset`, optional): Dataset
                from which to generate activations. If not supplied, will
                calculate activations for all project tfrecords at the
                tile_px/tile_um matching the model, optionally using provided
                filters and filter_blank.
            filters (dict, optional): Filters to use when selecting tfrecords.
                Defaults to None.
            checkpoint (str, optional): Path to cp.ckpt file, if evaluating a
                saved checkpoint. Defaults to None.
            eval_k_fold (int, optional): K-fold iteration number to evaluate.
                If None, will evaluate all tfrecords irrespective of K-fold.
                Defaults to None.
            splits (str, optional): Filename of JSON file in which to log
                training/validation splits. Looks for filename in project root
                directory. Defaults to "splits.json".
            max_tiles (int, optional): Maximum number of tiles from each slide
                to evaluate. If zero, will include all tiles. Defaults to 0.
            min_tiles (int, optional): Min tiles a slide must have
                to be included in evaluation. Defaults to 0.
            input_header (str, optional): Annotation column header to use as
                additional input. Defaults to None.
            format (str, optional): Format in which to save predictions.
                Either 'csv' or 'feather'. Defaults to 'csv'.
            mixed_precision (bool, optional): Enable mixed precision.
                Defaults to True.

        Returns:
            pandas.DataFrame of tile-level predictions.
        """

        # Perform evaluation
        log.info('Predicting model results')
        trainer, eval_dts = self._prepare_trainer(
            model=model,
            dataset=dataset,
            filters=filters,
            checkpoint=checkpoint,
            eval_k_fold=eval_k_fold,
            splits=splits,
            max_tiles=max_tiles,
            min_tiles=min_tiles,
            input_header=input_header,
            mixed_precision=mixed_precision
        )
        results = trainer.predict(
            dataset=eval_dts,
            batch_size=batch_size,
            format=format,
            **kwargs
        )
        return results

    def predict_wsi(self, model, outdir, dataset=None, filters=None,
                    filter_blank=None, stride_div=1, enable_downsample=True,
                    roi_method='inside', skip_missing_roi=False, source=None,
                    randomize_origin=False, img_format='auto', **kwargs):

        """Using a given model, generates a map of tile-level predictions for a
            whole-slide image (WSI), dumping prediction arrays into pkl files
            for later use.

        Args:
            model (str): Path to model from which to generate predictions.
            outdir (str): Directory for saving WSI predictions in .pkl format.
            dataset (:class:`slideflow.dataset.Dataset`, optional): Dataset
                from which to generate activations. If not supplied, will
                calculate activations for all tfrecords at the tile_px/tile_um
                matching the supplied model.
            filters (dict, optional): Filters to use when selecting tfrecords.
                Defaults to None.
            filter_blank (list, optional): Exclude slides blank in these cols.
                Defaults to None.
            stride_div (int, optional): Stride divisor for extracting tiles.
                A stride of 1 will extract non-overlapping tiles.
                A stride_div of 2 will extract overlapping tiles, with a stride
                equal to 50% of the tile width. Defaults to 1.
            enable_downsample (bool, optional): Enable downsampling for slides.
                This may result in corrupted image tiles if downsampled slide
                layers are corrupted or incomplete. Defaults to True.
            roi_method (str, optional): Either 'inside', 'outside' or 'ignore'.
                Indicates whether tiles are extracted inside or outside ROIs or
                if ROIs are ignored entirely. Defaults to 'inside'.
            skip_missing_roi (bool, optional): Skip slides missing ROIs.
                Defaults to True.
            source (list, optional): Name(s) of dataset sources from which to
                get slides. If None, will use all.
            randomize_origin (bool, optional): Randomize pixel starting
                position during extraction. Defaults to False.
            img_format (str, optional): Image format (png, jpg) to use when
                extracting tiles from slide. Must match the image format
                the model was trained on. If 'auto', will use the format
                logged in the model params.json.

        Keyword Args:
            whitespace_fraction (float, optional): Range 0-1. Defaults to 1.
                Discard tiles with this fraction of whitespace.
                If 1, will not perform whitespace filtering.
            whitespace_threshold (int, optional): Range 0-255. Defaults to 230.
                Threshold above which a pixel (RGB average) is whitespace.
            grayspace_fraction (float, optional): Range 0-1. Defaults to 0.6.
                Discard tiles with this fraction of grayspace.
                If 1, will not perform grayspace filtering.
            grayspace_threshold (float, optional): Range 0-1. Defaults to 0.05.
                Pixels in HSV format with saturation below this are grayspace.
        """

        log.info('Generating WSI prediction / activation maps...')
        if not exists(outdir):
            os.makedirs(outdir)

        if source:
            sources = [source] if not isinstance(source, list) else source
        else:
            sources = self.sources

        # Prepare dataset & model
        config = sf.util.get_model_config(model)
        if dataset is None:
            tile_px, tile_um = config['hp']['tile_px'], config['hp']['tile_um']
            dataset = self.dataset(tile_px, tile_um, verification='slides')
        else:
            dataset._assert_size_matches_hp(config['hp'])
            tile_px = dataset.tile_px
        dataset = dataset.filter(filters=filters, filter_blank=filter_blank)
        if img_format == 'auto':
            img_format = config['img_format']

        # Log extraction parameters
        sf.slide.log_extraction_params(**kwargs)

        for source in sources:
            log.info(f'Working on dataset source {col.bold(source)}')
            roi_dir = dataset.sources[source]['roi']

            # Prepare list of slides for extraction
            slide_list = dataset.slide_paths(source=source)
            log.info(f'Generating predictions for {len(slide_list)} slides')

            # Verify slides and estimate total number of tiles
            log.info('Verifying slides...')
            total_tiles = 0
            for slide_path in tqdm(slide_list, leave=False):
                slide = sf.WSI(slide_path,
                               tile_px,
                               tile_um,
                               stride_div,
                               roi_dir=roi_dir,
                               roi_method=roi_method,
                               skip_missing_roi=False)
                n_est = slide.estimated_num_tiles
                log.debug(f"Estimated tiles for slide {slide.name}: {n_est}")
                total_tiles += n_est
                del slide
            log.info(f'Total estimated tiles: {total_tiles}')

            # Predict for each WSI
            for slide_path in slide_list:
                log.info(f'Working on slide {path_to_name(slide_path)}')
                wsi = sf.WSI(slide_path,
                             tile_px,
                             tile_um,
                             stride_div,
                             enable_downsample=enable_downsample,
                             roi_dir=roi_dir,
                             roi_method=roi_method,
                             randomize_origin=randomize_origin,
                             skip_missing_roi=skip_missing_roi)
                if not wsi.loaded_correctly():
                    continue
                try:
                    interface = sf.model.Features(model, include_logits=False)
                    wsi_grid = interface(wsi, img_format=img_format)

                    with open(join(outdir, wsi.name+'.pkl'), 'wb') as file:
                        pickle.dump(wsi_grid, file)

                except errors.TileCorruptionError:
                    fmt_slide = col.green(path_to_name(slide_path))
                    log.error(f'{fmt_slide} is corrupt; skipping slide')
                    continue

    def save(self):
        """Saves current project configuration as "settings.json"."""
        sf.util.write_json(self._settings, join(self.root, 'settings.json'))

    def train(self, outcomes, params, exp_label=None, filters=None,
              filter_blank=None, input_header=None,
              min_tiles=0, max_tiles=0,
              splits="splits.json", balance_headers=None,
              mixed_precision=True, **training_kwargs):

        """Train model(s) using a given set of parameters, outcomes, and inputs.

        Args:
            outcomes (str or list(str)): Outcome label annotation header(s).
            params (:class:`slideflow.model.ModelParams`, list, dict, or str):
                Model parameters for training. May provide one `ModelParams`,
                a list, or dict mapping model names to params. If multiple
                params are provided, will train models for each. If JSON file
                is provided, will interpret as a hyperparameter sweep. See
                examples below for use.
            exp_label (str, optional): Experiment label to add model names.
            filters (dict, optional): Filters to use when selecting tfrecords.
                Defaults to None.
            filter_blank (list, optional): Exclude slides blank in these cols.
                Defaults to None.
            input_header (list, optional): List of annotation column headers to
                use as additional slide-level model input. Defaults to None.
            min_tiles (int): Minimum number of tiles a slide must have to
                include in training. Defaults to 0.
            max_tiles (int): Only use up to this many tiles from each slide for
                training. Defaults to 0 (include all tiles).
            splits (str, optional): Filename of JSON file in which to log
                train/val splits. Looks for filename in project root directory.
                Defaults to "splits.json".
            balance_headers (str or list(str)): Annotation header(s) specifying
                labels on which to perform mini-batch balancing. If performing
                category-level balancing and this is set to None, will default
                to balancing on outcomes. Defaults to None.
            mixed_precision (bool, optional): Enable mixed precision.
                Defaults to True.

        Keyword Args:
            val_strategy (str): Validation dataset selection strategy. Options
                include bootstrap, k-fold, k-fold-manual,
                k-fold-preserved-site, fixed, and none. Defaults to 'k-fold'.
            val_k_fold (int): Total number of K if using K-fold validation.
                Defaults to 3.
            val_k (int): Iteration of K-fold to train, starting at 1. Defaults
                to None (training all k-folds).
            val_k_fold_header (str): Annotations file header column for
                manually specifying k-fold or for preserved-site cross
                validation. Only used if validation strategy is 'k-fold-manual'
                or 'k-fold-preserved-site'. Defaults to None for k-fold-manual
                and 'site' for k-fold-preserved-site.
            val_fraction (float): Fraction of dataset to use for validation
                testing, if strategy is 'fixed'.
            val_source (str): Dataset source to use for validation. Defaults to
                None (same as training).
            val_annotations (str): Path to annotations file for validation
                dataset. Defaults to None (same as training).
            val_filters (dict): Filters to use for validation dataset.
                Defaults to None (same as training).

            checkpoint (str, optional): Path to cp.ckpt from which to load
                weights. Defaults to None.
            pretrain (str, optional): Either 'imagenet' or path to Tensorflow
                model from which to load weights. Defaults to 'imagenet'.
            multi_gpu (bool): Train using multiple GPUs when available.
                Defaults to False.
            resume_training (str, optional): Path to Tensorflow model to
                continue training. Defaults to None.
            starting_epoch (int): Start training at the specified epoch.
                Defaults to 0.
            steps_per_epoch_override (int): If provided, will manually set the
                number of steps in an epoch. Default epoch length is the number
                of total tiles.
            save_predicitons (bool): Save predictions with each validation.
                Defaults to False.
            save_model (bool, optional): Save models when evaluating at
                specified epochs. Defaults to True.
            validate_on_batch (int): Perform validation every N batches.
                Defaults to 0 (only at epoch end).
            validation_batch_size (int): Validation dataset batch size.
                Defaults to 32.
            use_tensorboard (bool): Add tensorboard callback for realtime
                training monitoring. Defaults to False.
            validation_steps (int): Number of steps of validation to perform
                each time doing a mid-epoch validation check. Defaults to 200.

        Returns:
            Dict with model names mapped to train_acc, val_loss, and val_acc

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

        # Prepare outcomes
        if not isinstance(outcomes, list):
            outcomes = [outcomes]
        if len(outcomes) > 1:
            log.info(f'Training with {len(outcomes)} outcomes')
            log.info(f'Outcomes: {", ".join(outcomes)}')

        # Prepare hyperparameters
        if isinstance(params, str):
            if exists(params):
                hp_dict = sf.model.read_hp_sweep(params)
            elif exists(join(self.root, params)):
                hp_dict = sf.model.read_hp_sweep(join(self.root, params))
            else:
                raise errors.ModelParamsError(f"Unable to find file {params}")
        elif isinstance(params, ModelParams):
            hp_dict = {'HP0': params}
        elif isinstance(params, list):
            if not all([isinstance(hp, ModelParams) for hp in params]):
                msg = 'If params is a list, items must be sf.model.ModelParams'
                raise errors.ModelParamsError(msg)
            hp_dict = {f'HP{i}': hp for i, hp in enumerate(params)}
        elif isinstance(params, dict):
            if not all([isinstance(hp, str) for hp in params.keys()]):
                msg = 'If params is a dict, keys must be of type str'
                raise errors.ModelParamsError(msg)
            all_hp = params.values()
            if not all([isinstance(hp, ModelParams) for hp in all_hp]):
                msg = 'If params is a dict, values must be sf.ModelParams'
                raise errors.ModelParamsError(msg)
            hp_dict = params
        else:
            raise ValueError(f"Unable to interprest params value {params}")

        # Get default validation settings from kwargs
        val_kwargs = {
            k[4:]: v for k, v in training_kwargs.items() if k[:4] == 'val_'
        }
        training_kwargs = {
            k: v for k, v in training_kwargs.items() if k[:4] != 'val_'
        }
        val_settings = get_validation_settings(**val_kwargs)
        _invalid = (
            'k-fold-manual',
            'k-fold-preserved-site',
            'k-fold',
            'bootstrap'
        )
        if (val_settings.strategy in _invalid) and val_settings.source:
            _m = f'{val_settings.strategy} invalid with val_source != None'
            raise ValueError(_m)

        # Next, prepare the multiprocessing manager (needed to free VRAM after
        # training and keep track of results)
        manager = multiprocessing.Manager()
        results_dict = manager.dict()
        ctx = multiprocessing.get_context('spawn')

        # === Train with a set of hyperparameters =============================
        for hp_name, hp in hp_dict.items():
            if exp_label:
                hp_name = f'{exp_label}-{hp_name}'
            self._train_hp(
                hp_name=hp_name,
                hp=hp,
                outcomes=outcomes,
                val_settings=val_settings,
                ctx=ctx,
                filters=filters,
                filter_blank=filter_blank,
                input_header=input_header,
                min_tiles=min_tiles,
                max_tiles=max_tiles,
                mixed_precision=mixed_precision,
                splits=splits,
                balance_headers=balance_headers,
                training_kwargs=training_kwargs,
                results_dict=results_dict
            )
        # Print summary of all models
        log.info('Training complete; validation accuracies:')
        for model in results_dict:
            if 'epochs' not in results_dict[model]:
                continue
            ep_res = results_dict[model]['epochs']
            epochs = [e for e in ep_res if 'epoch' in ep_res.keys()]
            try:
                last = max([int(e.split('epoch')[-1]) for e in epochs])
                final_train_metrics = ep_res[f'epoch{last}']['train_metrics']
            except ValueError:
                pass
            else:
                log.info(f'{col.green(model)} training metrics:')
                for m in final_train_metrics:
                    log.info(f'{m}: {final_train_metrics[m]}')
                if 'val_metrics' in ep_res[f'epoch{last}']:
                    final_val_metrics = ep_res[f'epoch{last}']['val_metrics']
                    log.info(f'{col.green(model)} validation metrics:')
                    for m in final_val_metrics:
                        log.info(f'{m}: {final_val_metrics[m]}')
        return results_dict

    def train_clam(self, exp_name, pt_files, outcomes, dataset,
                   train_slides='auto', val_slides='auto',
                   splits='splits.json', clam_args=None,
                   attention_heatmaps=True):
        """Train a CLAM model from layer activations exported with
        :meth:`slideflow.project.generate_features_for_clam`.

        Args:
            exp_name (str): Name of experiment. Makes clam/{exp_name} folder.
            pt_files (str): Path to pt_files containing tile-level features.
            outcomes (str): Annotation column which specifies the outcome.
            dataset (:class:`slideflow.dataset.Dataset`): Dataset object from
                which to generate activations.
            train_slides (str, optional): List of slide names for training.
                If 'auto' (default), will auto-generate training/val split.
            validation_slides (str, optional): List of slides for validation.
                If 'auto' (default), will auto-generate training/val split.
            splits (str, optional): Filename of JSON file in which to log
                training/val splits. Looks for filename in project root
                directory. Defaults to "splits.json".
            clam_args (optional): Namespace with clam arguments, as provided
                by :func:`slideflow.clam.get_args`.
            attention_heatmaps (bool, optional): Save attention heatmaps of
                validation dataset.

        Returns:
            None

        Examples
            Train with basic settings:

                >>> dataset = P.dataset(tile_px=299, tile_um=302)
                >>> P.generate_features_for_clam('/model', outdir='/pt_files')
                >>> P.train_clam('NAME', '/pt_files', 'category1', dataset)

            Specify a specific layer from which to generate activations:

                >>> P.generate_features_for_clam(..., layers=['postconv'])

            Manually configure CLAM, with 5-fold validation and SVM bag loss:

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
        if not exists(results_dir):
            os.makedirs(results_dir)

        # Get base CLAM args/settings if not provided.
        if not clam_args:
            clam_args = clam.get_args()

        # Detect number of features automatically from saved pt_files
        pt_file_paths = [
            join(pt_files, p) for p in os.listdir(pt_files)
            if sf.util.path_to_ext(join(pt_files, p)) == 'pt'
        ]
        num_features = clam.detect_num_features(pt_file_paths[0])

        # Note: CLAM only supports categorical outcomes
        labels, unique_labels = dataset.labels(outcomes, use_float=False)

        if train_slides == val_slides == 'auto':
            train_slides, val_slides = {}, {}
            for k in range(clam_args.k):
                train_dts, val_dts = dataset.train_val_split(
                    'categorical',
                    labels,
                    val_strategy='k-fold',
                    splits=splits_file,
                    val_k_fold=clam_args.k,
                    k_fold_iter=k+1
                )
                train_slides[k] = [
                    path_to_name(t) for t in train_dts.tfrecords()
                ]
                val_slides[k] = [
                    path_to_name(t) for t in val_dts.tfrecords()
                ]
        else:
            train_slides = {0: train_slides}
            val_slides = {0: val_slides}

        # Remove slides without associated .pt files
        num_skipped = 0
        for k in train_slides:
            n_supplied = len(train_slides[k]) + len(val_slides[k])
            train_slides[k] = [
                s for s in train_slides[k] if exists(join(pt_files, s+'.pt'))
            ]
            val_slides[k] = [
                s for s in val_slides[k] if exists(join(pt_files, s+'.pt'))
            ]
            n_train = len(train_slides[k])
            n_val = len(val_slides[k])
            num_skipped += n_supplied - (n_train + n_val)
        if num_skipped:
            log.warn(f'Skipping {num_skipped} slides missing .pt files.')

        # Set up training/validation splits (mirror base model)
        split_dir = join(clam_dir, 'splits')
        if not exists(split_dir):
            os.makedirs(split_dir)
        header = ['', 'train', 'val', 'test']
        for k in range(clam_args.k):
            with open(join(split_dir, f'splits_{k}.csv'), 'w') as splits_file:
                writer = csv.writer(splits_file)
                writer.writerow(header)
                # Currently, the below sets the val & test sets to be the same
                for i in range(max(len(train_slides[k]), len(val_slides[k]))):
                    row = [i]
                    if i < len(train_slides[k]):
                        row += [train_slides[k][i]]
                    else:
                        row += ['']
                    if i < len(val_slides[k]):
                        row += [val_slides[k][i], val_slides[k][i]]
                    else:
                        row += ['', '']
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
        clam_dataset = Generic_MIL_Dataset(
            csv_path=self.annotations,
            data_dir=pt_files,
            shuffle=False,
            seed=clam_args.seed,
            print_info=True,
            label_col=outcomes,
            label_dict=dict(zip(unique_labels, range(len(unique_labels)))),
            patient_strat=False,
            ignore=[]
        )
        # Run CLAM
        clam.main(clam_args, clam_dataset)

        # Get attention from trained model on validation set(s)
        for k in val_slides:
            tfr = dataset.tfrecords()
            attention_tfrecords = [
                t for t in tfr if path_to_name(t) in val_slides[k]
            ]
            attention_dir = join(clam_dir, 'attention', str(k))
            if not exists(attention_dir):
                os.makedirs(attention_dir)
            rev_labels = dict(zip(range(len(unique_labels)), unique_labels))
            export_attention(
                vars(clam_args),
                ckpt_path=join(results_dir, f's_{k}_checkpoint.pt'),
                outdir=attention_dir,
                pt_files=pt_files,
                slides=val_slides[k],
                reverse_labels=rev_labels,
                labels=labels
            )
            if attention_heatmaps:
                heatmaps_dir = join(clam_dir, 'attention_heatmaps', str(k))
                if not exists(heatmaps_dir):
                    os.makedirs(heatmaps_dir)

                for tfr in attention_tfrecords:
                    attention_dict = {}
                    slide = path_to_name(tfr)
                    try:
                        with open(join(attention_dir, slide+'.csv'), 'r') as f:
                            reader = csv.reader(f)
                            for row in reader:
                                attention_dict.update({
                                    int(row[0]): float(row[1])
                                })
                    except FileNotFoundError:
                        print(f'Attention scores for slide {slide} not found')
                        continue
                    self.generate_tfrecord_heatmap(
                        tfr,
                        tile_px=dataset.tile_px,
                        tile_um=dataset.tile_um,
                        tile_dict=attention_dict,
                        outdir=heatmaps_dir
                    )
