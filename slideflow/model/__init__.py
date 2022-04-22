'''Submodule that includes models, trainers, and tools for intermediate layer activations.

Supports both PyTorch and Tensorflow backends, importing either model.tensorflow or model.pytorch based on
the active backend given by the environmental variable SF_BACKEND.
'''

import sys
import os
import csv
import pickle
import time
import queue
import threading
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from collections import defaultdict
from os.path import join, exists
from math import isnan
from tqdm import tqdm

import slideflow as sf
from slideflow.util import log
from slideflow.util import colors as col
from slideflow import errors

# --- Backend-specific imports ------------------------------------------------

if os.environ['SF_BACKEND'] == 'tensorflow':
    from slideflow.model.tensorflow import ModelParams, Trainer, LinearTrainer, CPHTrainer, Features  # noqa F401
elif os.environ['SF_BACKEND'] == 'torch':
    from slideflow.model.torch import ModelParams, Trainer, LinearTrainer, CPHTrainer, Features  # noqa F401
else:
    raise errors.BackendError(f"Unknown backend {os.environ['SF_BACKEND']}")

# -----------------------------------------------------------------------------


def trainer_from_hp(hp, **kwargs):
    """From the given :class:`slideflow.model.ModelParams` object, returns
    the appropriate instance of :class:`slideflow.model.Model`.

    Args:
        hp (:class:`slideflow.model.ModelParams`): ModelParams object.

    Keyword Args:
        outdir (str): Target location for event logs and checkpoints.
        annotations (dict): Nested dict, mapping slide names to a dict with
            patient name (key 'patient'), outcome labels (key 'outcome_label'),
            and any additional slide-level inputs (key 'input').
        name (str, optional): Optional name describing the model, used for
            model saving. Defaults to None.
        manifest (dict, optional): Manifest dictionary mapping TFRecords to
            number of tiles. Defaults to None.
        model_type (str, optional): Type of model outcome, 'categorical' or
            'linear'. Defaults to 'categorical'.
        feature_sizes (list, optional): List of sizes of input features.
            Required if providing additional input features as model input.
        feature_names (list, optional): List of names for input features.
            Used when permuting feature importance.
        outcome_names (list, optional): Name of each outcome. Defaults to
            "Outcome {X}" for each outcome.
        mixed_precision (bool, optional): Use FP16 mixed precision
            (rather than FP32). Defaults to True.
    """
    if hp.model_type() == 'categorical':
        return Trainer(hp=hp, **kwargs)
    if hp.model_type() == 'linear':
        return LinearTrainer(hp=hp, **kwargs)
    if hp.model_type() == 'cph':
        return CPHTrainer(hp=hp, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {hp.model_type()}")


def read_hp_sweep(filename, models=None):
    """Organizes a list of hyperparameters ojects and associated models names.

    Args:
        filename (str): Path to hyperparameter sweep JSON file.
        models (list(str)): List of model names. Defaults to None.
            If not supplied, returns all valid models from batch file.

    Returns:
        List of (Hyperparameter, model_name) for each HP combination
    """
    if models is not None and not isinstance(models, list):
        raise ValueError("If supplying models, must be list(str) with model names.")
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
                name: sf.model.ModelParams.from_dict(hp_dict[name])
            })
    return loaded


class DatasetFeatures:

    """Loads annotations, saved layer activations / features, and prepares
    output saving directories. Will also read/write processed features to a
    PKL cache file to save time in future iterations.

    Note:
        Storing logits along with layer features is optional, to offer the user
        reduced memory footprint. For example, saving logits for a 10,000 slide
        dataset with 1000 categorical outcomes would require:

        4 bytes/float32-logit
        * 1000 logits/slide
        * 3000 tiles/slide
        * 10000 slides
        ~= 112 GB
    """

    def __init__(self, model, dataset, annotations=None, cache=None,
                 manifest=None, **kwargs):

        """Calculates features / layer activations from model, storing to
        internal parameters `self.activations`, and `self.logits`,
        `self.locations`, dictionaries mapping slides to arrays of activations,
        logits, and locations for each tiles' constituent tiles.

        Args:
            model (str): Path to model from which to calculate activations.
            dataset (:class:`slideflow.Dataset`): Dataset from which to
                generate activations.
            annotations (dict, optional): Dict mapping slide names to outcome
                categories.
            cache (str, optional): File for PKL cache.
            manifest (dict, optional): Dict mapping tfrecords to number of
                tiles contained. Used for progress bars.

        Keyword Args:
            layers (str): Model layer(s) from which to calculate activations.
                Defaults to 'postconv'.
            batch_size (int): Batch size for activations calculations.
                Defaults to 32.
            include_logits (bool): Calculate and store logits.
                Defaults to True.
        """
        self.activations = defaultdict(list)
        self.logits = defaultdict(list)
        self.uncertainty = defaultdict(list)
        self.locations = defaultdict(list)
        self.num_features = 0
        self.num_logits = 0
        self.manifest = manifest
        self.annotations = annotations
        self.model = model
        self.dataset = dataset
        self.tfrecords = np.array(dataset.tfrecords())
        self.slides = sorted([sf.util.path_to_name(t) for t in self.tfrecords])
        model_config = sf.util.get_model_config(model)
        self.tile_px = model_config['tile_px']
        self.hp = ModelParams.from_dict(model_config['hp'])
        self.normalizer = self.hp.get_normalizer()
        if self.normalizer:
            log.info(f'Using realtime {self.normalizer.method} normalization')
            if 'norm_fit' in model_config:
                self.normalizer.fit(**model_config['norm_fit'])

        if annotations:
            self.categories = list(set(self.annotations.values()))
            if self.activations:
                for slide in self.slides:
                    try:
                        if self.activations[slide]:
                            used = (self.used_categories
                                    + [self.annotations[slide]])
                            self.used_categories = list(set(used))
                            self.used_categories.sort()
                    except KeyError:
                        raise KeyError(f"Slide {slide} not in annotations.")
                total = len(self.used_categories)
                cat_list = ", ".join([str(c) for c in self.used_categories])
                log.debug(f'Observed categories (total: {total}): {cat_list}')
        else:
            self.categories = []
            self.used_categories = []

        # Load activations
        # Load from PKL (cache) if present
        if cache and exists(cache):
            # Load saved PKL cache
            log.info(f'Loading from cache {col.green(cache)}...')
            with open(cache, 'rb') as pt_pkl_file:
                self.activations, self.logits, self.uncertainty, self.locations = pickle.load(pt_pkl_file)
                self.num_features = self.activations[self.slides[0]].shape[-1]
                self.num_logits = self.logits[self.slides[0]].shape[-1]

        # Otherwise will need to generate new activations from a given model
        else:
            self._generate_from_model(model, cache=cache, **kwargs)

        # Now delete slides not included in our filtered TFRecord list
        loaded_slides = list(self.activations.keys())
        for loaded_slide in loaded_slides:
            if loaded_slide not in self.slides:
                msg = f'Removing activations from slide {loaded_slide}'
                msg += 'slide not in the filtered tfrecords list'
                log.debug(msg)
                self.remove_slide(loaded_slide)

        # Now screen for missing slides in activations
        missing = []
        for slide in self.slides:
            if slide not in self.activations:
                missing += [slide]
            elif self.activations[slide] == []:
                missing += [slide]
        num_loaded = len(self.slides)-len(missing)
        msg = f'Loaded activations from {num_loaded}/{len(self.slides)}'
        msg += f'slides ({len(missing)} missing)'
        log.debug(msg)
        if missing:
            log.warning(f'Activations missing for {len(missing)} slides')

        # Record which categories have been included in the specified tfrecords
        if self.categories:
            self.used_categories = list(set([
                self.annotations[slide]
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

    def _generate_from_model(self, model, layers='postconv',
                             include_logits=True, include_uncertainty=True,
                             batch_size=32, cache=None):

        """Calculates activations from a given model, saving to self.activations

        Args:
            model (str): Path to Tensorflow model from which to calculate final
                layer activations.
            layers (str, optional): Layers from which to generate activations.
                Defaults to 'postconv'.
            include_logits (bool, optional): Include logit predictions.
                Defaults to True.
            include_uncertainty (bool, optional): Include uncertainty
                estimation if UQ enabled. Defaults to True.
            batch_size (int, optional): Batch size to use during activations
                calculations. Defaults to 32.
            cache (str, optional): File in which to store PKL cache.
        """

        # Rename tfrecord_array to tfrecords
        log.info(f'Calculating activations for {self.tfrecords.shape[0]} tfrecords (layers={layers})')
        log.info(f'Generating from {col.green(model)}')
        if not isinstance(layers, list):
            layers = [layers]

        # Load model
        if self.hp.uq and include_uncertainty:
            combined_model = sf.model.tensorflow.UncertaintyInterface(model, layers=layers)
        else:
            combined_model = sf.model.Features(
                model,
                layers=layers,
                include_logits=include_logits
            )
        self.num_features = combined_model.num_features
        self.num_logits = 0 if not include_logits else combined_model.num_logits

        # Calculate final layer activations for each tfrecord
        fla_start_time = time.time()

        # Interleave tfrecord datasets
        estimated_tiles = self.dataset.num_tiles

        # Get backend-specific dataloader/dataset
        dataset_kwargs = {
            'infinite': False,
            'batch_size': batch_size,
            'augment': False,
            'incl_slidenames': True,
            'incl_loc': True,
            'normalizer': self.normalizer
        }
        if sf.backend() == 'tensorflow':
            dataloader = self.dataset.tensorflow(
                None,
                num_parallel_reads=None,
                deterministic=True,
                **dataset_kwargs
            )
        elif sf.backend() == 'torch':
            dataloader = self.dataset.torch(
                None,
                num_workers=1,
                **dataset_kwargs
            )

        # Worker to process activations/logits, for more efficient throughput
        q = queue.Queue()

        def batch_worker():
            while True:
                model_out, batch_slides, batch_loc = q.get()
                if model_out is None:
                    return
                if not isinstance(model_out, list):
                    model_out = [model_out]

                if sf.backend() == 'tensorflow':
                    decoded_slides = [
                        bs.decode('utf-8')
                        for bs in batch_slides.numpy()
                    ]
                    model_out = [
                        m.numpy() if not isinstance(m, list) else m
                        for m in model_out
                    ]
                    batch_loc = np.stack([
                        batch_loc[0].numpy(),
                        batch_loc[1].numpy()
                    ], axis=1)
                elif sf.backend() == 'torch':
                    decoded_slides = batch_slides
                    model_out = [
                        m.cpu().numpy() if not isinstance(m, list) else m
                        for m in model_out
                    ]
                    batch_loc = np.stack([batch_loc[0], batch_loc[1]], axis=1)

                # Process model outputs
                if self.hp.uq and include_uncertainty:
                    uncertainty = model_out[-1]
                    model_out = model_out[:-1]
                else:
                    uncertainty = None
                if include_logits:
                    logits = model_out[-1]
                    activations = model_out[:-1]
                else:
                    activations = model_out

                # Concatenate activations if we have activations from >`` layer
                if layers:
                    batch_act = np.concatenate(activations)

                for d, slide in enumerate(decoded_slides):
                    if layers:
                        self.activations[slide].append(batch_act[d])
                    if include_logits:
                        self.logits[slide].append(logits[d])
                    if self.hp.uq and include_uncertainty:
                        self.uncertainty[slide].append(uncertainty[d])
                    self.locations[slide].append(batch_loc[d])

        batch_proc_thread = threading.Thread(target=batch_worker, daemon=True)
        batch_proc_thread.start()

        pb = tqdm(total=estimated_tiles, ncols=80, leave=False)
        for batch_img, _, batch_slides, batch_loc_x, batch_loc_y in dataloader:
            model_output = combined_model(batch_img)
            q.put((model_output, batch_slides, (batch_loc_x, batch_loc_y)))
            pb.update(batch_size)
        pb.close()
        q.put((None, None, None))
        batch_proc_thread.join()

        self.activations = {s: np.stack(v) for s, v in self.activations.items()}
        self.logits = {s: np.stack(v) for s, v in self.logits.items()}
        self.locations = {s: np.stack(v) for s, v in self.locations.items()}
        self.uncertainty = {s: np.stack(v) for s, v in self.uncertainty.items()}

        fla_calc_time = time.time()
        log.debug(f'Calculation time: {fla_calc_time-fla_start_time:.0f} sec')
        log.debug(f'Number of activation features: {self.num_features}')

        # Dump PKL dictionary to file
        if cache:
            with open(cache, 'wb') as pt_pkl_file:
                pickle.dump([self.activations, self.logits, self.uncertainty, self.locations], pt_pkl_file)
            log.info(f'Data cached to {col.green(cache)}')

    def activations_by_category(self, idx):
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
            msg = 'Unable to calculate by category; annotations not provided.'
            raise errors.FeaturesError(msg)

        def act_by_cat(c):
            return np.concatenate([
                self.activations[pt][:, idx]
                for pt in self.slides
                if self.annotations[pt] == c
            ])

        return {c: act_by_cat(c) for c in self.used_categories}

    def box_plots(self, features, outdir):
        """Generates plots comparing node activations at slide- and tile-level.

        Args:
            features (list(int)): List of feature indices for which to
                generate box plots.
            outdir (str): Path to directory in which to save box plots.
        """
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

    def export_to_csv(self, filename, level='tile', method='mean',
                      slides=None):
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
            logit_header = [f'Logit_{log}' for log in range(self.num_logits)]
            feature_header = [f'Feature_{f}' for f in range(self.num_features)]
            header = ['Slide'] + logit_header + feature_header
            csvwriter.writerow(header)
            for slide in tqdm(slides, ncols=80, leave=False):
                if level == 'tile':
                    for i, tile_act in enumerate(self.activations[slide]):
                        if self.logits[slide] != []:
                            csvwriter.writerow([slide] + self.logits[slide][i].tolist() + tile_act.tolist())
                        else:
                            csvwriter.writerow([slide] + tile_act.tolist())
                else:
                    act = meth_fn[method](self.activations[slide], axis=0).tolist()
                    if self.logits[slide] != []:
                        logit = meth_fn[method](self.logits[slide], axis=0).tolist()
                        csvwriter.writerow([slide] + logit + act)
                    else:
                        csvwriter.writerow([slide] + act)
        log.debug(f'Activations saved to {col.green(filename)}')

    def export_to_torch(self, outdir, slides=None):
        """Export activations in torch format to .pt files in the given directory.

        Used for training CLAM models.

        Args:
            outdir (str): Path to directory in which to save .pt files.
        """

        import torch
        if not exists(outdir):
            os.makedirs(outdir)
        slides = self.slides if not slides else slides
        for slide in tqdm(slides, ncols=80, leave=False):
            if self.activations[slide] == []:
                log.info(f'Skipping empty slide {col.green(slide)}')
                continue
            slide_activations = torch.from_numpy(self.activations[slide].astype(np.float32))
            torch.save(slide_activations, join(outdir, f'{slide}.pt'))
        args = {
            'model': self.model,
            'num_features': self.num_features
        }
        sf.util.write_json(args, join(outdir, 'settings.json'))
        log.info('Activations exported in Torch format.')

    def stats(self, outdir=None, method='mean', threshold=0.5):
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
            dict: Dict mapping slides to dict of slide-level features;
            dict: Dict mapping features to tile-level statistics ('p', 'f');
            dict: Dict mapping features to slide-level statistics ('p', 'f');
        """

        if not self.categories:
            raise errors.FeaturesError('No annotations loaded')
        if method not in ('mean', 'threshold'):
            raise errors.FeaturesError(f"Stats method {method} unknown")

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
                if self.annotations[slide] == c
            ])]

        for f in range(self.num_features):
            # Tile-level ANOVA
            stats_vals = list(self.activations_by_category(f).values())
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
            log.info(f'Writing results to {col.green(filename)}...')
            with open(filename, 'w') as outfile:
                csv_writer = csv.writer(outfile)
                header = ['slide', 'category'] + [f'Feature_{n}' for n in pt_sorted_ft]
                csv_writer.writerow(header)
                for slide in self.slides:
                    category = self.annotations[slide]
                    row = [slide, category] + list(activation_stats[slide][pt_sorted_ft])
                    csv_writer.writerow(row)
                if tile_stats:
                    csv_writer.writerow(['Tile statistic', 'ANOVA P-value'] + [tile_stats[n]['p']
                                                                               for n in pt_sorted_ft])
                    csv_writer.writerow(['Tile statistic', 'ANOVA F-value'] + [tile_stats[n]['f']
                                                                               for n in pt_sorted_ft])
                if pt_stats:
                    csv_writer.writerow(['Slide statistic', 'ANOVA P-value'] + [pt_stats[n]['p']
                                                                                for n in pt_sorted_ft])
                    csv_writer.writerow(['Slide statistic', 'ANOVA F-value'] + [pt_stats[n]['f']
                                                                                for n in pt_sorted_ft])
        return tile_stats, pt_stats, category_stats

    def logits_mean(self):
        """Calculates the mean logits vector across all tiles in each slide.

        Returns:
            dict:  This is a dictionary mapping slides to the mean logits
                array for all tiles in each slide.
        """

        return {s: np.mean(v, axis=0) for s, v in self.logits.items()}

    def logits_percent(self, prediction_filter=None):
        """Returns dictionary mapping slides to a vector of length num_logits
        with the percent of tiles in each slide predicted to be each outcome.

        Args:
            prediction_filter:  (optional) List of int. If provided, will
                restrict predictions to only these categories, with final
                prediction being based based on highest logit among these
                categories.

        Returns:
            dict:  This is a dictionary mapping slides to an array of
                percentages for each logit, of length num_logits
        """

        if prediction_filter:
            assert isinstance(prediction_filter, list) and all([
                isinstance(i, int)
                for i in prediction_filter
            ])
            assert max(prediction_filter) <= self.num_logits
        else:
            prediction_filter = range(self.num_logits)

        slide_percentages = {}
        for slide in self.logits:
            # Find the index of the highest prediction for each tile, only for
            # logits within prediction_filter
            tile_pred = np.argmax(self.logits[slide][:, prediction_filter], axis=1)
            slide_perc = np.array([
                np.count_nonzero(tile_pred == logit) / len(tile_pred)
                for logit in range(self.num_logits)
            ])
            slide_percentages.update({slide: slide_perc})
        return slide_percentages

    def logits_predict(self, prediction_filter=None):
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
            assert max(prediction_filter) <= self.num_logits
        else:
            prediction_filter = range(self.num_logits)

        slide_predictions = {}
        for slide in self.logits:
            # Find the index of the highest prediction for each tile, only for
            # logits within prediction_filter
            tile_pred = np.argmax(self.logits[slide][:, prediction_filter], axis=1)
            slide_perc = np.array([
                np.count_nonzero(tile_pred == logit) / len(tile_pred)
                for logit in range(self.num_logits)
            ])
            slide_predictions.update({slide: np.argmax(slide_perc)})
        return slide_predictions

    def map_to_predictions(self, x=0, y=0):
        """Returns coordinates and metadata for tile-level predictions for all tiles,
        which can be used to create a SlideMap.

        Args:
            x (int, optional): Outcome category id for which predictions will
                be mapped to the X-axis. Defaults to 0.
            y (int, optional): Outcome category id for which predictions will
                be mapped to the Y-axis. Defaults to 0.

        Returns:
            list:   List of x-axis coordinates (preds for the category 'x')
            list:   List of y-axis coordinates (preds for the category 'y')
            list:   List of dict containing tile-level metadata (for SlideMap)
        """

        umap_x, umap_y, umap_meta = [], [], []
        for slide in self.slides:
            for tile_index in range(self.logits[slide].shape[0]):
                umap_x += [self.logits[slide][tile_index][x]]
                umap_y += [self.logits[slide][tile_index][y]]
                umap_meta += [{
                    'slide': slide,
                    'index': tile_index
                }]
        return np.array(umap_x), np.array(umap_y), umap_meta

    def merge(self, df):
        '''Merges with another DatasetFeatures.

        Args:
            df (slideflow.model.DatasetFeatures): TargetDatasetFeatures
                to merge with.

        Returns:
            None
        '''

        self.activations.update(df.activations)
        self.logits.update(df.logits)
        self.uncertainty.update(df.uncertainty)
        self.locations.update(df.locations)
        self.tfrecords = np.concatenate([self.tfrecords, df.tfrecords])
        self.slides = list(self.activations.keys())

    def remove_slide(self, slide):
        """Removes slide from internally cached activations."""
        del self.activations[slide]
        del self.logits[slide]
        del self.uncertainty[slide]
        del self.locations[slide]
        self.tfrecords = [
            t for t in self.tfrecords
            if sf.util.path_to_name(t) != slide
        ]
        try:
            self.slides.remove(slide)
        except ValueError:
            pass

    def save_example_tiles(self, features, outdir, slides=None,
                           tiles_per_feature=100):
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

            gradient = []
            for slide in slides:
                for i, val in enumerate(self.activations[slide][:, f]):
                    gradient += [{
                                    'val': val,
                                    'slide': slide,
                                    'index': i
                    }]
            gradient = np.array(sorted(gradient, key=lambda k: k['val']))
            sample_idx = np.linspace(
                0,
                gradient.shape[0]-1,
                num=tiles_per_feature,
                dtype=np.int
            )
            for i, g in tqdm(enumerate(gradient[sample_idx]),
                             ncols=80,
                             leave=False,
                             total=tiles_per_feature,
                             desc=f"Feature {f}"):
                for tfr in self.tfrecords:
                    if sf.util.path_to_name(tfr) == g['slide']:
                        tfr_dir = tfr
                if not tfr_dir:
                    log.warning(f"TFRecord location not found for slide {g['slide']}")
                slide, image = sf.io.get_tfrecord_by_index(tfr_dir, g['index'], decode=False)
                tile_filename = f"{i}-tfrecord{g['slide']}-{g['index']}-{g['val']:.2f}.jpg"
                image_string = open(join(outdir, str(f), tile_filename), 'wb')
                image_string.write(image.numpy())
                image_string.close()
