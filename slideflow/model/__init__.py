'''Submodule that includes tools for intermediate layer activations.

Supports both PyTorch and Tensorflow backends, importing either model.tensorflow
or model.pytorch based on the environmental variable SF_BACKEND.
'''

import csv
import os
import pickle
import queue
import sys
import threading
import time
import warnings
from collections import defaultdict
from math import isnan
from os.path import exists, join
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import slideflow as sf
from rich.progress import track, Progress
from slideflow import errors
from slideflow.util import log, Labels, ImgBatchSpeedColumn
from .base import BaseFeatureExtractor
from .extractors import get_feature_extractor


if TYPE_CHECKING:
    import tensorflow as tf
    import torch

# --- Backend-specific imports ------------------------------------------------

if sf.backend() == 'tensorflow':
    from slideflow.model.tensorflow import (CPHTrainer, Features, load, # noqa F401
                                            LinearTrainer, ModelParams,
                                            Trainer, UncertaintyInterface)
elif sf.backend() == 'torch':
    from slideflow.model.torch import (CPHTrainer, Features, load, # noqa F401
                                       LinearTrainer, ModelParams,
                                       Trainer, UncertaintyInterface)
else:
    raise errors.UnrecognizedBackendError

# -----------------------------------------------------------------------------


def is_tensorflow_tensor(arg: Any) -> bool:
    """Checks if the given object is a Tensorflow Tensor."""
    if sf.util.tf_available:
        import tensorflow as tf
        return isinstance(arg, tf.Tensor)
    else:
        return False


def is_torch_tensor(arg: Any) -> bool:
    """Checks if the given object is a Tensorflow Tensor."""
    if sf.util.torch_available:
        import torch
        return isinstance(arg, torch.Tensor)
    else:
        return False


def is_tensorflow_model(arg: Any) -> bool:
    """Checks if the object is a Tensorflow Model or path to Tensorflow model."""
    if isinstance(arg, str):
        return sf.util.is_tensorflow_model_path(arg)
    elif sf.util.tf_available:
        import tensorflow as tf
        return isinstance(arg, tf.keras.models.Model)
    else:
        return False


def is_torch_model(arg: Any) -> bool:
    """Checks if the object is a PyTorch Module or path to PyTorch model."""
    if isinstance(arg, str):
        return sf.util.is_torch_model_path(arg)
    elif sf.util.torch_available:
        import torch
        return isinstance(arg, torch.nn.Module)
    else:
        return False


def trainer_from_hp(*args, **kwargs):
    warnings.warn(
        "sf.model.trainer_from_hp() is deprecated. Please use "
        "sf.model.build_trainer().",
        DeprecationWarning
    )
    return build_trainer(*args, **kwargs)


def build_trainer(
    hp: "ModelParams",
    outdir: str,
    labels: Dict[str, Any],
    **kwargs
) -> Trainer:
    """From the given :class:`slideflow.model.ModelParams` object, returns
    the appropriate instance of :class:`slideflow.model.Trainer`.

    Args:
        hp (:class:`slideflow.model.ModelParams`): ModelParams object.
        outdir (str): Path for event logs and checkpoints.
        labels (dict): Dict mapping slide names to outcome labels (int or
            float format).

    Keyword Args:
        slide_input (dict): Dict mapping slide names to additional
            slide-level input, concatenated after post-conv.
        name (str, optional): Optional name describing the model, used for
            model saving. Defaults to 'Trainer'.
        feature_sizes (list, optional): List of sizes of input features.
            Required if providing additional input features as input to
            the model.
        feature_names (list, optional): List of names for input features.
            Used when permuting feature importance.
        outcome_names (list, optional): Name of each outcome. Defaults to
            "Outcome {X}" for each outcome.
        mixed_precision (bool, optional): Use FP16 mixed precision (rather
            than FP32). Defaults to True.
        allow_tf32 (bool): Allow internal use of Tensorfloat-32 format.
                Defaults to False.
        config (dict, optional): Training configuration dictionary, used
            for logging. Defaults to None.
        use_neptune (bool, optional): Use Neptune API logging.
            Defaults to False
        neptune_api (str, optional): Neptune API token, used for logging.
            Defaults to None.
        neptune_workspace (str, optional): Neptune workspace.
            Defaults to None.
        load_method (str): Either 'full' or 'weights'. Method to use
                when loading a Tensorflow model. If 'full', loads the model with
                ``tf.keras.models.load_model()``. If 'weights', will read the
                ``params.json``configuration file, build the model architecture,
                and then load weights from the given model with
                ``Model.load_weights()``. Loading with 'full' may improve
                compatibility across Slideflow versions. Loading with 'weights'
                may improve compatibility across hardware & environments.
        custom_objects (dict, Optional): Dictionary mapping names
                (strings) to custom classes or functions. Defaults to None.
    """
    if hp.model_type() == 'categorical':
        return Trainer(hp, outdir, labels, **kwargs)
    if hp.model_type() == 'linear':
        return LinearTrainer(hp, outdir, labels, **kwargs)
    if hp.model_type() == 'cph':
        return CPHTrainer(hp, outdir, labels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {hp.model_type()}")


def read_hp_sweep(
    filename: str,
    models: List[str] = None
) -> Dict[str, "ModelParams"]:
    """Organizes a list of hyperparameters ojects and associated models names.

    Args:
        filename (str): Path to hyperparameter sweep JSON file.
        models (list(str)): List of model names. Defaults to None.
            If not supplied, returns all valid models from batch file.

    Returns:
        List of (Hyperparameter, model_name) for each HP combination
    """
    if models is not None and not isinstance(models, list):
        raise ValueError("If supplying models, must be list(str) "
                         "with model names.")
    if isinstance(models, list) and not list(set(models)) == models:
        raise ValueError("Duplicate model names provided.")

    hp_list = sf.util.load_json(filename)

    # First, ensure all indicated models are in the batch train file
    if models:
        valid_models = []
        for hp_dict in hp_list:
            model_name = list(hp_dict.keys())[0]
            if ((not models)
               or (isinstance(models, str) and model_name == models)
               or model_name in models):
                valid_models += [model_name]
        missing = [m for m in models if m not in valid_models]
        if missing:
            raise ValueError(f"Unable to find models {', '.join(missing)}")
    else:
        valid_models = [list(hp_dict.keys())[0] for hp_dict in hp_list]

    # Read the batch train file and generate HyperParameter objects
    # from the given configurations
    loaded = {}
    for hp_dict in hp_list:
        name = list(hp_dict.keys())[0]
        if name in valid_models:
            loaded.update({
                name: ModelParams.from_dict(hp_dict[name])
            })
    return loaded  # type: ignore


class DatasetFeatures:

    """Loads annotations, saved layer activations / features, and prepares
    output saving directories. Will also read/write processed features to a
    PKL cache file to save time in future iterations.

    Note:
        Storing predictions along with layer features is optional, to offer the user
        reduced memory footprint. For example, saving predictions for a 10,000 slide
        dataset with 1000 categorical outcomes would require:

        4 bytes/float32-logit
        * 1000 predictions/slide
        * 3000 tiles/slide
        * 10000 slides
        ~= 112 GB
    """

    def __init__(
        self,
        model: Union[str, "tf.keras.models.Model", "torch.nn.Module"],
        dataset: "sf.Dataset",
        *,
        labels: Optional[Labels] = None,
        cache: Optional[str] = None,
        annotations: Optional[Labels] = None,
        **kwargs: Any
    ) -> None:

        """Calculates features / layer activations from model, storing to
        internal parameters ``self.activations``, and ``self.predictions``,
        ``self.locations``, dictionaries mapping slides to arrays of activations,
        predictions, and locations for each tiles' constituent tiles.

        Args:
            model (str): Path to model from which to calculate activations.
            dataset (:class:`slideflow.Dataset`): Dataset from which to
                generate activations.
            labels (dict, optional): Dict mapping slide names to outcome
                categories.
            cache (str, optional): File for PKL cache.
            annotations: Deprecated.

        Keyword Args:
            layers (str): Model layer(s) from which to calculate activations.
                Defaults to 'postconv'.
            batch_size (int): Batch size for activations calculations.
                Defaults to 32.
            include_preds (bool): Calculate and store predictions.
                Defaults to True.
        """
        self.activations = defaultdict(list)  # type: Dict[str, Any]
        self.predictions = defaultdict(list)  # type: Dict[str, Any]
        self.uncertainty = defaultdict(list)  # type: Dict[str, Any]
        self.locations = defaultdict(list)  # type: Dict[str, Any]
        self.num_features = 0
        self.num_classes = 0
        self.manifest = dataset.manifest()
        self.model = model
        self.dataset = dataset
        self.tile_px = dataset.tile_px
        self.tfrecords = np.array(dataset.tfrecords())
        self.slides = sorted([sf.util.path_to_name(t) for t in self.tfrecords])

        if labels is not None and annotations is not None:
            raise DeprecationWarning(
                'Cannot supply both "labels" and "annotations" to sf.DatasetFeatures. '
                '"annotations" is deprecated and has been replaced with "labels".'
            )
        elif annotations is not None:
            warnings.warn(
                'The "annotations" argument to sf.DatasetFeatures is deprecated.'
                'Please use the argument "labels" instead.',
                DeprecationWarning
            )
            self.labels = annotations
        else:
            self.labels = labels

        # Load configuration if model is path to a saved model
        if isinstance(model, BaseFeatureExtractor):
            self.uq = model.num_uncertainty > 0
            self.normalizer = model.normalizer
        elif isinstance(model, str) and sf.util.is_simclr_model_path(model):
            self.uq = False
            self.normalizer = None
        elif isinstance(model, str):
            model_config = sf.util.get_model_config(model)
            hp = ModelParams.from_dict(model_config['hp'])
            self.uq = hp.uq
            self.normalizer = hp.get_normalizer()
            if self.normalizer:
                log.info(f'Using realtime {self.normalizer.method} normalization')
                if 'norm_fit' in model_config:
                    self.normalizer.set_fit(**model_config['norm_fit'])
        else:
            self.normalizer = None
            self.uq = False

        if self.labels:
            self.categories = list(set(self.labels.values()))
            if self.activations:
                for slide in self.slides:
                    try:
                        if self.activations[slide]:
                            used = (self.used_categories
                                    + [self.labels[slide]])
                            self.used_categories = list(set(used))  # type: List[Union[str, int, List[float]]]
                            self.used_categories.sort()
                    except KeyError:
                        raise KeyError(f"Slide {slide} not in labels.")
                total = len(self.used_categories)
                cat_list = ", ".join([str(c) for c in self.used_categories])
                log.debug(f'Observed categories (total: {total}): {cat_list}')
        else:
            self.categories = []
            self.used_categories = []

        # Load from PKL (cache) if present
        if cache and exists(cache):
            self.load_cache(cache)

        # Otherwise will need to generate new activations from a given model
        else:
            self._generate_from_model(model, cache=cache, **kwargs)

        # Now delete slides not included in our filtered TFRecord list
        loaded_slides = list(self.activations.keys())
        for loaded_slide in loaded_slides:
            if loaded_slide not in self.slides:
                log.debug(
                    f'Removing activations from slide {loaded_slide} '
                    'slide not in the filtered tfrecords list'
                )
                self.remove_slide(loaded_slide)

        # Now screen for missing slides in activations
        missing = []
        for slide in self.slides:
            if slide not in self.activations:
                missing += [slide]
            elif self.activations[slide] == []:
                missing += [slide]
        num_loaded = len(self.slides)-len(missing)
        log.debug(
            f'Loaded activations from {num_loaded}/{len(self.slides)} '
            f'slides ({len(missing)} missing)'
        )
        if missing:
            log.warning(f'Activations missing for {len(missing)} slides')

        # Record which categories have been included in the specified tfrecords
        if self.categories and self.labels:
            self.used_categories = list(set([
                self.labels[slide]
                for slide in self.slides
            ]))
            self.used_categories.sort()

        total = len(self.used_categories)
        cat_list = ", ".join([str(c) for c in self.used_categories])
        log.debug(f'Observed categories (total: {total}): {cat_list}')

        # Show total number of features
        if self.num_features is None:
            self.num_features = self.activations[self.slides[0]].shape[-1]
        log.debug(f'Number of activation features: {self.num_features}')

    def _generate_from_model(
        self,
        model: Union[str, "tf.keras.models.Model", "torch.nn.Module"],
        layers: Union[str, List[str]] = 'postconv',
        include_preds: Optional[bool] = None,
        include_uncertainty: bool = True,
        batch_size: int = 32,
        cache: Optional[str] = None,
        **kwargs
    ) -> None:

        """Calculates activations from a given model, saving to self.activations

        Args:
            model (str): Path to Tensorflow model from which to calculate final
                layer activations.
            layers (str, optional): Layers from which to generate activations.
                Defaults to 'postconv'.
            include_preds (bool, optional): Include logit predictions.
                Defaults to True.
            include_uncertainty (bool, optional): Include uncertainty
                estimation if UQ enabled. Defaults to True.
            batch_size (int, optional): Batch size to use during activations
                calculations. Defaults to 32.
            cache (str, optional): File in which to store PKL cache.
        """

        # Rename tfrecord_array to tfrecords
        log.info(f'Calculating activations for {self.tfrecords.shape[0]} '
                 f'tfrecords (layers={layers})')
        log.info(f'Generating from [green]{model}')

        layers = sf.util.as_list(layers)
        is_extractor = isinstance(model, BaseFeatureExtractor)
        is_simclr = sf.util.is_simclr_model_path(model)
        is_torch = sf.model.is_torch_model(model)
        is_tf = sf.model.is_tensorflow_model(model)

        if is_extractor and include_preds is None:
            include_preds = model.num_classes > 0
        elif include_preds is None:
            include_preds = True

        # Build the combined (feature +/- prediction) model.
        feat_kw = dict(
            layers=layers,
            include_preds=include_preds,
            **kwargs
        )
        if isinstance(model, BaseFeatureExtractor):
            is_torch = model.is_torch()
            is_tf = model.is_tensorflow()
            combined_model = model
        elif self.uq and include_uncertainty:
            combined_model = sf.model.UncertaintyInterface(
                model,
                layers=layers
            )
        elif is_simclr:
            from slideflow import simclr
            simclr_args = simclr.load_model_args(model)
            combined_model = simclr.load(model)
            combined_model.num_features = simclr_args.proj_out_dim
            combined_model.num_classes = simclr_args.num_classes
        elif isinstance(model, str):
            combined_model = sf.model.Features(model, **feat_kw)
        elif is_tf:
            combined_model = sf.model.Features.from_model(model, **feat_kw)
        elif is_torch:
            combined_model = sf.model.Features.from_model(
                model,
                tile_px=self.tile_px,
                **feat_kw
            ).to('cuda')
        else:
            raise ValueError(f'Unrecognized model {model}')

        self.num_features = combined_model.num_features
        self.num_classes = 0 if not include_preds else combined_model.num_classes

        # Calculate final layer activations for each tfrecord
        fla_start_time = time.time()

        # Interleave tfrecord datasets
        estimated_tiles = self.dataset.num_tiles

        # Prepare preprocessing
        dataset_kwargs = {
            'infinite': False,
            'batch_size': batch_size,
            'augment': False,
            'incl_slidenames': True,
            'incl_loc': True,
            'normalizer': self.normalizer
        }
        if is_extractor:
            dataset_kwargs.update(model.preprocess_kwargs)

        # Get backend-specific dataset
        if is_simclr:
            builder = simclr.DatasetBuilder(
                val_dts=self.dataset,
                dataset_kwargs=dict(
                    incl_slidenames=True,
                    incl_loc=True,
                    normalizer=self.normalizer
                )
            )
            dataloader = builder.build_dataset(
                batch_size,
                is_training=False,
                simclr_args=simclr_args
            )
        elif is_tf:
            dataloader = self.dataset.tensorflow(
                None,
                num_parallel_reads=None,
                deterministic=True,
                **dataset_kwargs  # type: ignore
            )
        elif is_torch:
            dataloader = self.dataset.torch(
                None,
                num_workers=1,
                **dataset_kwargs  # type: ignore
            )
        else:
            raise ValueError(f"Unrecognized model type: {type(model)}")

        # Worker to process activations/predictions, for more efficient throughput
        q = queue.Queue()  # type: queue.Queue

        def batch_worker():
            while True:
                model_out, batch_slides, batch_loc = q.get()
                if model_out is None:
                    return
                model_out = sf.util.as_list(model_out)

                if is_simclr or is_tf:
                    decoded_slides = [
                        bs.decode('utf-8')
                        for bs in batch_slides.numpy()
                    ]
                    model_out = [
                        m.numpy() if not isinstance(m, (list, tuple)) else m
                        for m in model_out
                    ]
                    if batch_loc[0] is not None:
                        batch_loc = np.stack([
                            batch_loc[0].numpy(),
                            batch_loc[1].numpy()
                        ], axis=1)
                    else:
                        batch_loc = None
                elif is_torch:
                    decoded_slides = batch_slides
                    model_out = [
                        m.cpu().numpy() if not isinstance(m, list) else m
                        for m in model_out
                    ]
                    if batch_loc[0] is not None:
                        batch_loc = np.stack([batch_loc[0], batch_loc[1]], axis=1)
                    else:
                        batch_loc = None

                # Process model outputs
                if is_simclr:
                    model_out = model_out[0]
                if self.uq and include_uncertainty:
                    uncertainty = model_out[-1]
                    model_out = model_out[:-1]
                else:
                    uncertainty = None
                if include_preds:
                    predictions = model_out[-1]
                    activations = model_out[:-1]
                else:
                    activations = model_out

                # Concatenate activations if we have activations from >`` layer
                if layers:
                    batch_act = np.concatenate(activations)

                for d, slide in enumerate(decoded_slides):
                    if layers:
                        self.activations[slide].append(batch_act[d])
                    if include_preds and predictions is not None:
                        self.predictions[slide].append(predictions[d])
                    if self.uq and include_uncertainty:
                        self.uncertainty[slide].append(uncertainty[d])
                    if batch_loc is not None:
                        self.locations[slide].append(batch_loc[d])

        batch_proc_thread = threading.Thread(target=batch_worker, daemon=True)
        batch_proc_thread.start()

        pb = Progress(*Progress.get_default_columns(),
                      ImgBatchSpeedColumn(),
                      transient=sf.getLoggingLevel()>20)
        task = pb.add_task("Generating...", total=estimated_tiles)
        pb.start()
        for batch_img, _, batch_slides, batch_loc_x, batch_loc_y in dataloader:
            call_kw = dict(training=False) if is_simclr else dict()
            if is_torch:
                import torch
                with torch.no_grad():
                    model_output = combined_model(batch_img.to('cuda'), **call_kw)
            else:
                model_output = combined_model(batch_img, **call_kw)
            q.put((model_output, batch_slides, (batch_loc_x, batch_loc_y)))
            pb.advance(task, batch_size)
        pb.stop()
        q.put((None, None, None))
        batch_proc_thread.join()

        self.activations = {s: np.stack(v) for s, v in self.activations.items()}
        self.predictions = {s: np.stack(v) for s, v in self.predictions.items()}
        self.locations = {s: np.stack(v) for s, v in self.locations.items()}
        self.uncertainty = {s: np.stack(v) for s, v in self.uncertainty.items()}

        fla_calc_time = time.time()
        log.debug(f'Calculation time: {fla_calc_time-fla_start_time:.0f} sec')
        log.debug(f'Number of activation features: {self.num_features}')

        if cache:
            self.save_cache(cache)

    def activations_by_category(
        self,
        idx: int
    ) -> Dict[Union[str, int, List[float]], np.ndarray]:
        """For each outcome category, calculates activations of a given
        feature across all tiles in the category. Requires annotations to
        have been provided.

        Args:
            idx (int): Index of activations layer to return, stratified by
                outcome category.

        Returns:
            dict: Dict mapping categories to feature activations for all
            tiles in the category.
        """

        if not self.categories:
            raise errors.FeaturesError(
                'Unable to calculate by category; annotations not provided.'
            )

        def act_by_cat(c):
            return np.concatenate([
                self.activations[pt][:, idx]
                for pt in self.slides
                if self.labels[pt] == c
            ])
        return {c: act_by_cat(c) for c in self.used_categories}

    def box_plots(self, features: List[int], outdir: str) -> None:
        """Generates plots comparing node activations at slide- and tile-level.

        Args:
            features (list(int)): List of feature indices for which to
                generate box plots.
            outdir (str): Path to directory in which to save box plots.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if not isinstance(features, list):
            raise ValueError("'features' must be a list of int.")
        if not self.categories:
            log.warning('Unable to generate box plots; no annotations loaded.')
            return
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        _, _, category_stats = self.stats()

        log.info('Generating box plots...')
        for f in features:
            # Display tile-level box plots & stats
            plt.clf()
            boxplot_data = list(self.activations_by_category(f).values())
            snsbox = sns.boxplot(data=boxplot_data)
            title = f'{f} (tile-level)'
            snsbox.set_title(title)
            snsbox.set(xlabel='Category', ylabel='Activation')
            plt.xticks(plt.xticks()[0], self.used_categories)
            boxplot_filename = join(outdir, f'boxplot_{title}.png')
            plt.gcf().canvas.start_event_loop(sys.float_info.min)
            plt.savefig(boxplot_filename, bbox_inches='tight')

            # Print slide_level box plots & stats
            plt.clf()
            snsbox = sns.boxplot(data=[c[:, f] for c in category_stats])
            title = f'{f} (slide-level)'
            snsbox.set_title(title)
            snsbox.set(xlabel='Category', ylabel='Average tile activation')
            plt.xticks(plt.xticks()[0], self.used_categories)
            boxplot_filename = join(outdir, f'boxplot_{title}.png')
            plt.gcf().canvas.start_event_loop(sys.float_info.min)
            plt.savefig(boxplot_filename, bbox_inches='tight')

    def export_to_torch(self, *args, **kwargs):
        """Deprecated function; please use `.to_torch()`"""
        warnings.warn(
            "Deprecation warning: DatasetFeatures.export_to_torch() will"
            " be removed in a future version. Use .to_torch() instead.",
            DeprecationWarning
        )
        self.to_torch(*args, **kwargs)

    def save_cache(self, path: str):
        """Cache calculated activations to file.

        Args:
            path (str): Path to pkl.
        """
        with open(path, 'wb') as pt_pkl_file:
            pickle.dump(
                [self.activations,
                 self.predictions,
                 self.uncertainty,
                 self.locations],
                pt_pkl_file
            )
        log.info(f'Data cached to [green]{path}')

    def to_csv(
        self,
        filename: str,
        level: str = 'tile',
        method: str = 'mean',
        slides: Optional[List[str]] = None
    ):
        """Exports calculated activations to csv.

        Args:
            filename (str): Path to CSV file for export.
            level (str): 'tile' or 'slide'. Indicates whether tile or
                slide-level activations are saved. Defaults to 'tile'.
            method (str): Method of summarizing slide-level results. Either
                'mean' or 'median'. Defaults to 'mean'.
            slides (list(str)): Slides to export. If None, exports all slides.
                Defaults to None.
        """
        if level not in ('tile', 'slide'):
            raise errors.FeaturesError(f"Export error: unknown level {level}")

        meth_fn = {'mean': np.mean, 'median': np.median}
        slides = self.slides if not slides else slides

        with open(filename, 'w') as outfile:
            csvwriter = csv.writer(outfile)
            logit_header = [f'Class_{log}' for log in range(self.num_classes)]
            feature_header = [f'Feature_{f}' for f in range(self.num_features)]
            header = ['Slide'] + logit_header + feature_header
            csvwriter.writerow(header)
            for slide in track(slides):
                if level == 'tile':
                    for i, tile_act in enumerate(self.activations[slide]):
                        if self.predictions[slide] != []:
                            csvwriter.writerow(
                                [slide]
                                + self.predictions[slide][i].tolist()
                                + tile_act.tolist()
                            )
                        else:
                            csvwriter.writerow([slide] + tile_act.tolist())
                else:
                    act = meth_fn[method](
                        self.activations[slide],
                        axis=0
                    ).tolist()
                    if self.predictions[slide] != []:
                        logit = meth_fn[method](
                            self.predictions[slide],
                            axis=0
                        ).tolist()
                        csvwriter.writerow([slide] + logit + act)
                    else:
                        csvwriter.writerow([slide] + act)
        log.debug(f'Activations saved to [green]{filename}')

    def to_torch(
        self,
        outdir: str,
        slides: Optional[List[str]] = None
    ) -> None:
        """Export activations in torch format to .pt files in the directory.

        Used for training CLAM models.

        Args:
            outdir (str): Path to directory in which to save .pt files.
        """

        import torch
        if not exists(outdir):
            os.makedirs(outdir)
        slides = self.slides if not slides else slides
        for slide in track(slides):
            if self.activations[slide] == []:
                log.info(f'Skipping empty slide [green]{slide}')
                continue
            slide_activations = torch.from_numpy(
                self.activations[slide].astype(np.float32)
            )
            torch.save(slide_activations, join(outdir, f'{slide}.pt'))
        args = {
            'model': self.model if isinstance(self.model, str) else '<NA>',
            'num_features': self.num_features
        }
        sf.util.write_json(args, join(outdir, 'settings.json'))
        log.info(f'Activations exported in Torch format to {outdir}')

    def to_df(
        self
    ) -> pd.core.frame.DataFrame:
        """Export activations, predictions, uncertainty, and locations to
        a pandas DataFrame.

        Returns:
            pd.core.frame.DataFrame: Dataframe with columns 'activations',
            'predictions', 'uncertainty', and 'locations'.
        """

        index = [s for s in self.slides
                   for _ in range(len(self.locations[s]))]
        df_dict = {}
        df_dict.update({
            'locations': pd.Series([
                self.locations[s][i]
                for s in self.slides
                for i in range(len(self.locations[s]))], index=index)
        })
        if self.activations:
            df_dict.update({
                'activations': pd.Series([
                    self.activations[s][i]
                    for s in self.slides
                    for i in range(len(self.activations[s]))], index=index)
            })
        if self.predictions:
            df_dict.update({
                'predictions': pd.Series([
                    self.predictions[s][i]
                    for s in self.slides
                    for i in range(len(self.predictions[s]))], index=index)
            })
        if self.uncertainty:
            df_dict.update({
                'uncertainty': pd.Series([
                    self.uncertainty[s][i]
                    for s in self.slides
                    for i in range(len(self.uncertainty[s]))], index=index)
            })
        return pd.DataFrame(df_dict)

    def load_cache(self, path: str):
        """Load cached activations from PKL.

        Args:
            path (str): Path to pkl cache.
        """
        log.info(f'Loading from cache [green]{path}...')
        with open(path, 'rb') as pt_pkl_file:
            loaded_pkl = pickle.load(pt_pkl_file)
            self.activations = loaded_pkl[0]
            self.predictions = loaded_pkl[1]
            self.uncertainty = loaded_pkl[2]
            self.locations = loaded_pkl[3]
            if self.activations:
                self.num_features = self.activations[self.slides[0]].shape[-1]
            if self.predictions:
                self.num_classes = self.predictions[self.slides[0]].shape[-1]

    def stats(
        self,
        outdir: Optional[str] = None,
        method: str = 'mean',
        threshold: float = 0.5
    ) -> Tuple[Dict[int, Dict[str, float]],
               Dict[int, Dict[str, float]],
               List[np.ndarray]]:
        """Calculates activation averages across categories, as well as
        tile-level and patient-level statistics, using ANOVA, exporting to
        CSV if desired.

        Args:
            outdir (str, optional): Path to directory in which CSV file will
                be saved. Defaults to None.
            method (str, optional): Indicates method of aggregating tile-level
                data into slide-level data. Either 'mean' (default) or
                'threshold'. If mean, slide-level feature data is calculated by
                averaging feature activations across all tiles. If threshold,
                slide-level feature data is calculated by counting the number
                of tiles with feature activations > threshold and dividing by
                the total number of tiles. Defaults to 'mean'.
            threshold (float, optional): Threshold if using 'threshold' method.

        Returns:
            A tuple containing

                dict: Dict mapping slides to dict of slide-level features;

                dict: Dict mapping features to tile-level statistics ('p', 'f');

                dict: Dict mapping features to slide-level statistics ('p', 'f');
        """

        if not self.categories:
            raise errors.FeaturesError('No annotations loaded')
        if method not in ('mean', 'threshold'):
            raise errors.FeaturesError(f"Stats method {method} unknown")
        if not self.labels:
            raise errors.FeaturesError("No annotations provided, unable"
                                       "to calculate feature stats.")

        log.info('Calculating activation averages & stats across features...')

        tile_stats = {}
        pt_stats = {}
        category_stats = []
        activation_stats = {}
        for slide in self.slides:
            if method == 'mean':
                # Mean of each feature across tiles
                summarized = np.mean(self.activations[slide], axis=0)
            elif method == 'threshold':
                # For each feature, count number of tiles with value above
                # threshold, divided by number of tiles
                act_sum = np.sum((self.activations[slide] > threshold), axis=0)
                summarized = act_sum / self.activations[slide].shape[-1]
            activation_stats[slide] = summarized
        for c in self.used_categories:
            category_stats += [np.array([
                activation_stats[slide]
                for slide in self.slides
                if self.labels[slide] == c
            ])]

        for f in range(self.num_features):
            # Tile-level ANOVA
            stats_vals = list(self.activations_by_category(f).values())
            with warnings.catch_warnings():
                if hasattr(stats, "F_onewayConstantInputWarning"):
                    warnings.simplefilter(
                        "ignore",
                        category=stats.F_onewayConstantInputWarning)
                elif hasattr(stats, "ConstantInputWarning"):
                    warnings.simplefilter(
                        "ignore",
                        category=stats.ConstantInputWarning)
                fvalue, pvalue = stats.f_oneway(*stats_vals)
                if not isnan(fvalue) and not isnan(pvalue):
                    tile_stats.update({f: {'f': fvalue,
                                        'p': pvalue}})
                else:
                    tile_stats.update({f: {'f': -1,
                                        'p': 1}})
                # Patient-level ANOVA
                fvalue, pvalue = stats.f_oneway(*[c[:, f] for c in category_stats])
                if not isnan(fvalue) and not isnan(pvalue):
                    pt_stats.update({f: {'f': fvalue,
                                        'p': pvalue}})
                else:
                    pt_stats.update({f: {'f': -1,
                                        'p': 1}})
        try:
            pt_sorted_ft = sorted(
                range(self.num_features),
                key=lambda f: pt_stats[f]['p']
            )
        except Exception:
            log.warning('No stats calculated; unable to sort features.')

        for f in range(self.num_features):
            try:
                log.debug(f"Tile-level P-value ({f}): {tile_stats[f]['p']}")
                log.debug(f"Patient-level P-value: ({f}): {pt_stats[f]['p']}")
            except Exception:
                log.warning(f'No stats calculated for feature {f}')

        # Export results
        if outdir:
            if not exists(outdir):
                os.makedirs(outdir)
            filename = join(outdir, 'slide_level_summary.csv')
            log.info(f'Writing results to [green]{filename}[/]...')
            with open(filename, 'w') as outfile:
                csv_writer = csv.writer(outfile)
                header = (['slide', 'category']
                          + [f'Feature_{n}' for n in pt_sorted_ft])
                csv_writer.writerow(header)
                for slide in self.slides:
                    category = self.labels[slide]
                    row = ([slide, category]
                           + list(activation_stats[slide][pt_sorted_ft]))
                    csv_writer.writerow(row)
                if tile_stats:
                    csv_writer.writerow(
                        ['Tile statistic', 'ANOVA P-value']
                        + [tile_stats[n]['p'] for n in pt_sorted_ft]
                    )
                    csv_writer.writerow(
                        ['Tile statistic', 'ANOVA F-value']
                        + [tile_stats[n]['f'] for n in pt_sorted_ft]
                    )
                if pt_stats:
                    csv_writer.writerow(
                        ['Slide statistic', 'ANOVA P-value']
                        + [pt_stats[n]['p'] for n in pt_sorted_ft]
                    )
                    csv_writer.writerow(
                        ['Slide statistic', 'ANOVA F-value']
                        + [pt_stats[n]['f'] for n in pt_sorted_ft]
                    )
        return tile_stats, pt_stats, category_stats

    def softmax_mean(self) -> Dict[str, np.ndarray]:
        """Calculates the mean prediction vector (post-softmax) across
        all tiles in each slide.

        Returns:
            dict:  This is a dictionary mapping slides to the mean logits
            array for all tiles in each slide.
        """

        return {s: np.mean(v, axis=0) for s, v in self.predictions.items()}

    def softmax_percent(
        self,
        prediction_filter: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """Returns dictionary mapping slides to a vector of length num_classes
        with the percent of tiles in each slide predicted to be each outcome.

        Args:
            prediction_filter:  (optional) List of int. If provided, will
                restrict predictions to only these categories, with final
                prediction being based based on highest logit among these
                categories.

        Returns:
            dict:  This is a dictionary mapping slides to an array of
            percentages for each logit, of length num_classes
        """

        if prediction_filter:
            assert isinstance(prediction_filter, list) and all([
                isinstance(i, int)
                for i in prediction_filter
            ])
            assert max(prediction_filter) <= self.num_classes
        else:
            prediction_filter = list(range(self.num_classes))

        slide_percentages = {}
        for slide in self.predictions:
            # Find the index of the highest prediction for each tile, only for
            # logits within prediction_filter
            tile_pred = np.argmax(
                self.predictions[slide][:, prediction_filter],
                axis=1
            )
            slide_perc = np.array([
                np.count_nonzero(tile_pred == logit) / len(tile_pred)
                for logit in range(self.num_classes)
            ])
            slide_percentages.update({slide: slide_perc})
        return slide_percentages

    def softmax_predict(
        self,
        prediction_filter: Optional[List[int]] = None
    ) -> Dict[str, int]:
        """Returns slide-level predictions, assuming the model is predicting a
        categorical outcome, by generating a prediction for each individual
        tile, and making a slide-level prediction by finding the most
        frequently predicted outcome among its constituent tiles.

        Args:
            prediction_filter:  (optional) List of int. If provided, will
                restrict predictions to only these categories, with final
                prediction based based on highest logit among these categories.

        Returns:
            dict:  Dictionary mapping slide names to slide-level predictions.
        """
        if prediction_filter:
            assert isinstance(prediction_filter, list)
            assert all([isinstance(i, int) for i in prediction_filter])
            assert max(prediction_filter) <= self.num_classes
        else:
            prediction_filter = list(range(self.num_classes))

        slide_predictions = {}
        for slide in self.predictions:
            # Find the index of the highest prediction for each tile, only for
            # logits within prediction_filter
            tile_pred = np.argmax(
                self.predictions[slide][:, prediction_filter],
                axis=1
            )
            slide_perc = np.array([
                np.count_nonzero(tile_pred == logit) / len(tile_pred)
                for logit in range(self.num_classes)
            ])
            slide_predictions.update({slide: int(np.argmax(slide_perc))})
        return slide_predictions

    def map_activations(self, **kwargs) -> "sf.SlideMap":
        """Map activations with UMAP.

        Keyword args:
            ...

        Returns:
            sf.SlideMap

        """
        return sf.SlideMap.from_features(self, **kwargs)

    def map_predictions(
        self,
        x: int = 0,
        y: int = 0,
        **kwargs
    ) -> "sf.SlideMap":
        """Map tile predictions onto x/y coordinate space.

        Args:
            x (int, optional): Outcome category id for which predictions will
                be mapped to the X-axis. Defaults to 0.
            y (int, optional): Outcome category id for which predictions will
                be mapped to the Y-axis. Defaults to 0.

        Keyword args:
            cache (str, optional): Path to parquet file to cache coordinates.
                Defaults to None (caching disabled).

        Returns:
            sf.SlideMap

        """
        all_x, all_y, all_slides, all_tfr_idx = [], [], [], []
        for slide in self.slides:
            all_x.append(self.predictions[slide].values[:, x])
            all_y.append(self.predictions[slide].values[:, y])
            all_slides.append([slide for _ in range(self.predictions[slide].shape[0])])
            all_tfr_idx.append(np.arange(self.predictions[slide].shape[0]))
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        all_slides = np.concatenate(all_slides)
        all_tfr_idx = np.concatenate(all_tfr_idx)

        return sf.SlideMap.from_xy(
            x=all_x,
            y=all_y,
            slides=all_slides,
            tfr_index=all_tfr_idx,
            **kwargs
        )

    def merge(self, df: "DatasetFeatures") -> None:
        '''Merges with another DatasetFeatures.

        Args:
            df (slideflow.DatasetFeatures): TargetDatasetFeatures
                to merge with.

        Returns:
            None
        '''

        self.activations.update(df.activations)
        self.predictions.update(df.predictions)
        self.uncertainty.update(df.uncertainty)
        self.locations.update(df.locations)
        self.tfrecords = np.concatenate([self.tfrecords, df.tfrecords])
        self.slides = list(self.activations.keys())

    def remove_slide(self, slide: str) -> None:
        """Removes slide from internally cached activations."""
        del self.activations[slide]
        del self.predictions[slide]
        del self.uncertainty[slide]
        del self.locations[slide]
        self.tfrecords = np.array([
            t for t in self.tfrecords
            if sf.util.path_to_name(t) != slide
        ])
        try:
            self.slides.remove(slide)
        except ValueError:
            pass

    def save_example_tiles(
        self,
        features: List[int],
        outdir: str,
        slides: Optional[List[str]] = None,
        tiles_per_feature: int = 100
    ) -> None:
        """For a set of activation features, saves image tiles named according
        to their corresponding activations.

        Duplicate image tiles will be saved for each feature, organized into
        subfolders named according to feature.

        Args:
            features (list(int)): Features to evaluate.
            outdir (str):  Path to folder in which to save examples tiles.
            slides (list, optional): List of slide names. If provided, will
                only include tiles from these slides. Defaults to None.
            tiles_per_feature (int, optional): Number of tiles to include as
                examples for each feature. Defaults to 100. Will evenly sample
                this many tiles across the activation gradient.
        """

        if not isinstance(features, list):
            raise ValueError("'features' must be a list of int.")

        if not slides:
            slides = self.slides
        for f in features:
            if not exists(join(outdir, str(f))):
                os.makedirs(join(outdir, str(f)))

            gradient_list = []
            for slide in slides:
                for i, val in enumerate(self.activations[slide][:, f]):
                    gradient_list += [{
                                    'val': val,
                                    'slide': slide,
                                    'index': i
                    }]
            gradient = np.array(sorted(gradient_list, key=lambda k: k['val']))
            sample_idx = np.linspace(
                0,
                gradient.shape[0]-1,
                num=tiles_per_feature,
                dtype=int
            )
            for i, g in track(enumerate(gradient[sample_idx]),
                             total=tiles_per_feature,
                             description=f"Feature {f}"):
                for tfr in self.tfrecords:
                    if sf.util.path_to_name(tfr) == g['slide']:
                        tfr_dir = tfr
                if not tfr_dir:
                    log.warning("TFRecord location not found for "
                                f"slide {g['slide']}")
                slide, image = sf.io.get_tfrecord_by_index(
                    tfr_dir,
                    g['index'],
                    decode=False
                )
                tile_filename = (f"{i}-tfrecord{g['slide']}-{g['index']}"
                                 + f"-{g['val']:.2f}.jpg")
                image_string = open(join(outdir, str(f), tile_filename), 'wb')
                image_string.write(image.numpy())
                image_string.close()

    # --- Deprecated functions ----------------------------------------------------

    def logits_mean(self):
        warnings.warn(
            "DatasetFeatures.logits_mean() is deprecated. Please use "
            "DatasetFeatures.softmax_mean()", DeprecationWarning
        )
        return self.softmax_mean()

    def logits_percent(self, *args, **kwargs):
        warnings.warn(
            "DatasetFeatures.logits_percent() is deprecated. Please use "
            "DatasetFeatures.softmax_percent()", DeprecationWarning
        )
        return self.softmax_percent(*args, **kwargs)

    def logits_predict(self, *args, **kwargs):
        warnings.warn(
            "DatasetFeatures.logits_predict() is deprecated. Please use "
            "DatasetFeatures.softmax_predict()", DeprecationWarning
        )
        return self.softmax_predict(*args, **kwargs)
