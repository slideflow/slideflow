import sys
import os
import shutil
import csv
import pickle
import time
import logging
import queue
import threading
from collections import defaultdict

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import slideflow as sf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import seaborn as sns
import scipy.stats as stats
import shapely.geometry as sg
import slideflow.slide

from slideflow.util import log
from slideflow.util.fastim import FastImshow
from os.path import join, exists
from math import isnan
from matplotlib.widgets import Slider
from functools import partial
from multiprocessing.dummy import Process as DProcess
from tqdm import tqdm

class ActivationsError(Exception):
    pass

class ActivationsInterface:
    """Interface for obtaining logits and intermediate layer activations from Slideflow models.

    Use by calling on either a batch of images (returning outputs for a single batch), or by calling on a
    :class:`slideflow.slide.WSI` object, which will generate an array of spatially-mapped activations matching
    the slide.

    Examples
        *Calling on batch of images:*

        .. code-block:: python

            interface = ActivationsInterface('/model/path', layers='postconv')
            for image_batch in train_data:
                # Return shape: (batch_size, num_features)
                batch_activations = interface(image_batch)

        *Calling on a slide:*

        .. code-block:: python

            slide = sf.slide.WSI(...)
            interface = ActivationsInterface('/model/path', layers='postconv')
            # Return shape: (slide.grid.shape[0], slide.grid.shape[1], num_features):
            activations_grid = interface(slide)

    """

    def __init__(self, path, layers='postconv', include_logits=False):
        """Creates an activations interface from a saved slideflow model which outputs feature activations
        at the designated layers.

        Intermediate layers are returned in the order of layers. Logits are returned last.

        Args:
            path (str): Path to saved Slideflow model.
            layers (list(str), optional): Layers from which to generate activations.  The post-convolution activation layer
                is accessed via 'postconv'. Defaults to 'postconv'.
            include_logits (bool, optional): Include logits in output. Will be returned last. Defaults to False.
        """

        if layers and not isinstance(layers, list): layers = [layers]
        self.path = path
        try:
            self.hp = sf.util.get_model_params(path)
        except:
            self.hp = None
        self.num_logits = 0
        self.num_features = 0
        self._model = tf.keras.models.load_model(self.path)
        self._build(layers=layers, include_logits=include_logits)

    def from_model(self, model, layers='postconv', include_logits=False):
        """Creates an activations interface from a loaded slideflow model which outputs feature activations
        at the designated layers.

        Intermediate layers are returned in the order of layers. Logits are returned last.

        Args:
            model (:class:`tensorflow.keras.models.Model` or :class:`slideflow.model.Trainer`): Loaded model.
            layers (list(str), optional): Layers from which to generate activations.  The post-convolution activation layer
                is accessed via 'postconv'. Defaults to 'postconv'.
            include_logits (bool, optional): Include logits in output. Will be returned last. Defaults to False.
        """
        if isinstance(model, tf.keras.models.Model):
            self._model = model
        elif isinstance(model, sf.model.Trainer):
            if not model.model:
                raise sf.util.UserError("Provided model has not yet been built or loaded.")
            self._model = model.model
        else:
            raise TypeError("Provided model is not a valid Tensorflow model.")
        self.hp = None
        self.num_logits = 0
        self.num_features = 0
        self._build(layers=layers, include_logits=include_logits)

    def __call__(self, inp, **kwargs):
        """Process a given input and return activations and/or logits. Expects either a batch of images or
        a :class:`slideflow.slide.WSI` object."""

        if isinstance(inp, sf.slide.WSI):
            return self._predict_slide(inp, **kwargs)
        else:
            return self._predict(inp)

    def _predict_slide(self, slide, batch_size=128, dtype=np.float16, **kwargs):
        """Generate activations from slide => activation grid array."""
        total_out = self.num_features + self.num_logits
        activations_grid = np.zeros((slide.grid.shape[1], slide.grid.shape[0], total_out), dtype=dtype)
        generator = slide.build_generator(shuffle=False, include_loc='grid', show_progress=True, **kwargs)

        if not generator:
            log.error(f"No tiles extracted from slide {sf.util.green(slide.name)}")
            return

        def _parse_function(record):
            image = record['image']
            loc = record['loc']
            parsed_image = tf.image.per_image_standardization(image)
            parsed_image.set_shape([slide.tile_px, slide.tile_px, 3])
            return parsed_image, loc

        # Generate dataset from the generator
        with tf.name_scope('dataset_input'):
            output_signature={'image':tf.TensorSpec(shape=(slide.tile_px,slide.tile_px,3), dtype=tf.uint8),
                              'loc':tf.TensorSpec(shape=(2), dtype=tf.uint32)}
            tile_dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
            tile_dataset = tile_dataset.map(_parse_function, num_parallel_calls=8)
            tile_dataset = tile_dataset.batch(batch_size, drop_remainder=False)
            tile_dataset = tile_dataset.prefetch(8)

        act_arr = []
        loc_arr = []
        for i, (batch_images, batch_loc) in enumerate(tile_dataset):
            model_out = self._predict(batch_images)
            if not isinstance(model_out, list): model_out = [model_out]
            concatenated_arr = np.concatenate([m.numpy() for m in model_out])
            act_arr += [concatenated_arr]
            loc_arr += [batch_loc.numpy()]

        act_arr = np.concatenate(act_arr)
        loc_arr = np.concatenate(loc_arr)

        for i, act in enumerate(act_arr):
            xi = loc_arr[i][0]
            yi = loc_arr[i][1]
            activations_grid[yi][xi] = act

        return activations_grid

    @tf.function
    def _predict(self, inp):
        """Return activations for a single batch of images."""
        return self.model(inp, training=False)

    def _build(self, layers, include_logits=True):
        """Builds the interface model that outputs feature activations at the designated layers and/or logits.
            Intermediate layers are returned in the order of layers. Logits are returned last."""

        if layers:
            log.debug(f"Setting up interface to return activations from layers {', '.join(layers)}")
            other_layers = [l for l in layers if l != 'postconv']
        else:
            other_layers = []
        outputs = {}
        if layers:
            intermediate_core = tf.keras.models.Model(inputs=self._model.layers[1].input,
                                                      outputs=[self._model.layers[1].get_layer(l).output for l in other_layers])
            if len(other_layers) > 1:
                int_out = intermediate_core(self._model.input)
                for l, layer in enumerate(other_layers):
                    outputs[layer] = int_out[l]
            elif len(other_layers):
                outputs[other_layers[0]] = intermediate_core(self._model.input)
            if 'postconv' in layers:
                outputs['postconv'] = self._model.layers[1].get_output_at(0)
        outputs_list = [] if not layers else [outputs[l] for l in layers]
        if include_logits:
            outputs_list += [self._model.output]
        self.model = tf.keras.models.Model(inputs=self._model.input, outputs=outputs_list)
        self.num_features = sum([outputs[o].shape[1] for o in outputs])
        if isinstance(self._model.output, list):
            log.warning("Multi-categorical outcomes not yet supported for this interface.")
            self.num_logits = 0
        else:
            self.num_logits = 0 if not include_logits else self._model.output.shape[1]
        if include_logits:
            log.debug(f'Number of logits: {self.num_logits}')
        log.debug(f'Number of activation features: {self.num_features}')

class ActivationsVisualizer:

    """Loads annotations, saved layer activations, and prepares output saving directories.
    Will also read/write processed activations to a PKL cache file to save time in future iterations.

    Note:
        Storing logits is optional in order to offer the user reduced memory footprint. For example, generating
        logits for a 10,000 slide dataset with 1000 categorical outcomes would generate:
        4 bytes/float32-logit * 1000 logits/slide * 3000 tiles/slide * 10000 slides ~= 112 GB

    """

    def __init__(self, model, dataset, annotations=None, cache=None, manifest=None, **kwargs):

        """Calculates activations from model, storing to internal parameters `self.activations`, and `self.logits`,
        `self.locations`, dictionaries mapping slides to arrays of activations, logits, and locations for each tiles'
        constituent tiles.

        Args:
            model (str): Path to model from which to calculate activations.
            dataset (:class:`slideflow.dataset.Dataset`): Dataset from which to generate activations.
            annotations (dict, optional): Dict mapping slide names to outcome categories.
            cache (str, optional): File in which to store activations PKL cache.
            manifest (dict, optional): Dict mapping tfrecords to number of tiles contained. Used for progress bars.

        Keyword Args:
            layers (str): Model layer(s) from which to calculate activations. Defaults to 'postconv'.
            batch_size (int): Batch size to use during activations calculations. Defaults to 32.
            include_logits (bool): Calculate and store logits. Defaults to True.
            normalizer (str): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str): Path to normalizer source image. Defaults to None.
                If None but using a normalizer, will use an internal tile for normalization.
                Internal default tile can be found at slideflow.util.norm_tile.jpg
        """

        self.activations = defaultdict(list)
        self.logits = defaultdict(list)
        self.locations = defaultdict(list)
        self.num_features = 0
        self.num_logits = 0
        self.manifest = manifest
        self.annotations = annotations
        self.model = model
        self.dataset = dataset
        self.tfrecords = np.array(dataset.tfrecords())
        self.slides = sorted([sf.util.path_to_name(tfr) for tfr in self.tfrecords])
        self.tile_px = sf.util.get_model_params(model)['tile_px']

        if annotations:
            self.categories = list(set(self.annotations.values()))
            if self.activations:
                for slide in self.slides:
                    try:
                        if self.activations[slide]:
                            self.used_categories = list(set(self.used_categories + [self.annotations[slide]]))
                            self.used_categories.sort()
                    except KeyError:
                        raise KeyError(f"Slide {slide} not found in provided annotations.")
                log.debug(f'Observed categories (total: {len(self.used_categories)}): {", ".join(self.used_categories)}')
        else:
            self.categories = []
            self.used_categories = []

        # Load activations
        # Load from PKL (cache) if present
        if cache and exists(cache):
            # Load saved PKL cache
            log.info(f'Loading pre-calculated predictions and activations from {sf.util.green(cache)}...')
            with open(cache, 'rb') as pt_pkl_file:
                self.activations, self.logits, self.locations = pickle.load(pt_pkl_file)
                self.num_features = self.activations[self.slides[0]].shape[-1]
                self.num_logits = self.logits[self.slides[0]].shape[-1]

        # Otherwise will need to generate new activations from a given model
        else:
            self._generate_from_model(model, cache=cache, **kwargs)

        # Now delete slides not included in our filtered TFRecord list
        loaded_slides = list(self.activations.keys())
        for loaded_slide in loaded_slides:
            if loaded_slide not in self.slides:
                self.remove_slide(loaded_slide)

        # Now screen for missing slides in activations
        missing_slides = []
        for slide in self.slides:
            if slide not in self.activations:
                missing_slides += [slide]
            elif self.activations[slide] == []:
                missing_slides += [slide]
        num_loaded = len(self.slides)-len(missing_slides)
        log.info(f'Loaded activations from {num_loaded}/{len(self.slides)} slides ({len(missing_slides)} missing)')
        if missing_slides:
            log.warning(f'Activations missing for {len(missing_slides)} slides')

        # Record which categories have been included in the specified tfrecords
        if self.categories:
            self.used_categories = list(set([self.annotations[slide] for slide in self.slides]))
            self.used_categories.sort()
        log.debug(f'Observed categories (total: {len(self.used_categories)}): {", ".join(self.used_categories)}')

        # Show total number of features
        if self.num_features is None:
            self.num_features = self.activations[self.slides[0]].shape[-1]
        log.debug(f'Number of activation features: {self.num_features}')

    def _generate_from_model(self, model, layers='postconv', normalizer=None, normalizer_source=None,
                            include_logits=True, batch_size=32, cache=None):

        """Calculates activations from a given model, saving to self.activations

        Args:
            model (str): Path to Tensorflow model from which to calculate final layer activations.
            layers (str, optional): Layers from which to generate activations. Defaults to 'postconv'.
            normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image. Defaults to None.
                If None but using a normalizer, will use an internal tile for normalization.
                Internal default tile can be found at slideflow.util.norm_tile.jpg
            include_logits (bool, optional): Include logit predictions. Defaults to True.
            batch_size (int, optional): Batch size to use during activations calculations. Defaults to 32.
            cache (str, optional): File in which to store activations PKL cache.
        """

        # Rename tfrecord_array to tfrecords
        log.info(f'Calculating activations from {sf.util.green(model)}')
        if not isinstance(layers, list): layers = [layers]

        # Load model
        combined_model = ActivationsInterface(model, layers=layers, include_logits=include_logits)
        self.num_features = combined_model.num_features
        self.num_logits = 0 if not include_logits else combined_model.num_logits

        # Prepare normalizer
        if normalizer:
            log.info(f'Using realtime {normalizer} normalization')
            normalizer = sf.util.StainNormalizer(method=normalizer, source=normalizer_source)

        # Calculate final layer activations for each tfrecord
        fla_start_time = time.time()
        include_tfrecord_loc = True

        # Interleave tfrecord datasets
        estimated_tiles = self.dataset.num_tiles
        tf_dataset = self.dataset.tensorflow(label_parser=None,
                                             infinite=False,
                                             batch_size=batch_size,
                                             augment=False,
                                             incl_slidenames=True,
                                             incl_loc=True)

        # Worker to process activations/logits, for more efficient GPU throughput
        q = queue.Queue()
        def batch_worker():
            while True:
                model_out, batch_slides, batch_loc = q.get()
                if model_out == None:
                    return
                decoded_slides = [bs.decode('utf-8') for bs in batch_slides.numpy()]
                if not isinstance(model_out, list):
                    model_out = [model_out]
                model_out = [m.numpy() if not isinstance(m, list) else m for m in model_out]

                if include_logits:
                    logits = model_out[-1]
                    activations = model_out[:-1]
                else:
                    activations = model_out
                # Concatenate activations if we have activations from more than one layer
                batch_act = np.concatenate(activations)

                if include_tfrecord_loc:
                    batch_loc = np.stack([batch_loc[0].numpy(), batch_loc[1].numpy()], axis=1)
                for d, slide in enumerate(decoded_slides):
                    self.activations[slide].append(batch_act[d])
                    if include_logits:
                        self.logits[slide].append(logits[d])
                    if include_tfrecord_loc:
                        self.locations[slide].append(batch_loc[d])

        batch_processing_thread = threading.Thread(target=batch_worker, daemon=True)
        batch_processing_thread.start()

        pb = tqdm(total=estimated_tiles, ncols=80, leave=False)
        for i, (batch_img, _, batch_slides, batch_loc_x, batch_loc_y) in enumerate(tf_dataset):
            model_output = combined_model(batch_img)
            q.put((model_output, batch_slides, (batch_loc_x, batch_loc_y)))
            pb.update(batch_size)
        pb.close()
        q.put((None, None, None))
        batch_processing_thread.join()

        self.activations = {s:np.stack(v) for s,v in self.activations.items()}
        self.logits = {s:np.stack(v) for s,v in self.logits.items()}
        self.locations = {s:np.stack(v) for s,v in self.locations.items()}

        fla_calc_time = time.time()
        log.debug(f'Activation calculation time: {fla_calc_time-fla_start_time:.0f} sec')
        log.debug(f'Number of activation features: {self.num_features}')

        # Dump PKL dictionary to file
        if cache:
            with open(cache, 'wb') as pt_pkl_file:
                pickle.dump([self.activations, self.logits, self.locations], pt_pkl_file)
            log.info(f'Predictions and activations cached to {sf.util.green(cache)}')

    def activations_by_category(self, idx):
        """For each outcome category, calculates activations of a given feature across all tiles in the category.
        Requires annotations to have been provided.

        Args:
            idx (int): Index of activations layer to return, stratified by outcome category.

        Returns:
            dict: Dict mapping categories to feature activations for all tiles in the category.
        """

        if not self.categories:
            raise sf.util.UserError('Unable to calculate activations by category; annotations not provided.')

        def activations_by_single_category(c):
            return np.concatenate([self.activations[pt][:,idx] for pt in self.slides if self.annotations[pt] == c])

        return {c: activations_by_single_category(c) for c in self.used_categories}

    def box_plots(self, features, outdir):
        """Generates box plots comparing nodal activations at the slide-level and tile-level.

        Args:
            features (list(int)): List of feature indices for which to generate box plots.
            outdir (str): Path to directory in which to save box plots.
        """
        if not isinstance(features, list): raise sf.util.UserError("'features' must be a list of int.")
        if not self.categories:
            log.warning('Unable to generate box plots; annotations not loaded. Please load with load_annotations().')
            return
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        _, _, category_stats = self.feature_stats()

        log.info('Generating box plots...')
        for f in features:
            # Display tile-level box plots & stats
            plt.clf()
            snsbox = sns.boxplot(data=list(self.activations_by_category(f).values()))
            title = f'{f} (tile-level)'
            snsbox.set_title(title)
            snsbox.set(xlabel='Category', ylabel='Activation')
            plt.xticks(plt.xticks()[0], self.used_categories)
            boxplot_filename = join(outdir, f'boxplot_{title}.png')
            plt.gcf().canvas.start_event_loop(sys.float_info.min)
            plt.savefig(boxplot_filename, bbox_inches='tight')

            # Print slide_level box plots & stats
            plt.clf()
            snsbox = sns.boxplot(data=[c[:,f] for c in category_stats])
            title = f'{f} (slide-level)'
            snsbox.set_title(title)
            snsbox.set(xlabel='Category',ylabel='Average tile activation')
            plt.xticks(plt.xticks()[0], self.used_categories)
            boxplot_filename = join(outdir, f'boxplot_{title}.png')
            plt.gcf().canvas.start_event_loop(sys.float_info.min)
            plt.savefig(boxplot_filename, bbox_inches='tight')

    def export_to_csv(self, filename, level='tile', method='mean', slides=None):
        """Exports calculated activations to csv.

        Args:
            filename (str): Path to CSV file for export.
            level (str): 'tile' or 'slide'. Indicates whether tile or slide-level activations are saved.
                Defaults to 'tile'.
            method (str): Method of summarizing slide-level results. Either 'mean' or 'median'. Defaults to 'mean'.
            slides (list(str)): Slides to export. If None, exports all slides. Defaults to None.
        """
        if level not in ('tile', 'slide'):
            raise sf.util.UserError(f"Unknown level {level}, must be either 'tile' or 'slide'.")

        meth_fn = {'mean': np.mean, 'median': np.median}
        slides = self.slides if not slides else slides

        with open(filename, 'w') as outfile:
            csvwriter = csv.writer(outfile)
            logit_header = [f'Logit_{l}' for l in range(self.num_logits)]
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
        log.debug(f'Activations saved to {sf.util.green(filename)}')

    def export_to_torch(self, outdir, slides=None):
        """Export activations in torch format to .pt files in the given directory.

        Used for training CLAM models.

        Args:
            outdir (str): Path to directory in which to save .pt files.
        """

        import torch

        slides = self.slides if not slides else slides
        for slide in tqdm(slides, ncols=80, leave=False):
            if self.activations[slide] == []:
                log.info(f'Skipping empty slide {sf.util.green(slide)}')
                continue
            slide_activations = torch.from_numpy(self.activations[slide].astype(np.float32))
            torch.save(slide_activations, join(outdir, f'{slide}.pt'))
        args = {
            'model': self.model,
            'num_features': self.num_features
        }
        sf.util.write_json(args, join(outdir, 'settings.json'))
        log.info('Activations exported in Torch format.')

    def feature_stats(self, outdir=None, method='mean', threshold=0.5):
        """Calculates activation averages across categories, as well as tile-level and patient-level statistics,
            using ANOVA, exporting to CSV if desired.

        Args:
            outdir (str, optional): Path to directory in which CSV file will be saved. Defaults to None.
            method (str, optional): Indicates method of aggregating tile-level data into slide-level data.
                Either 'mean' (default) or 'threshold'. If mean, slide-level feature data is calculated by averaging
                feature activations across all tiles. If threshold, slide-level feature data is calculated by counting
                the number of tiles with feature activations > threshold and dividing by the total number of tiles.
                Defaults to 'mean'.
            threshold (float, optional): Threshold if using 'threshold' method.

        Returns:
            dict: Dict mapping slides to dict of features mapping to slide-level feature values;
            dict: Dict mapping features to tile-level dict of statistics ('p', 'f');
            dict: Dict mapping features to slide-level dict of statistics ('p', 'f');
        """

        if not self.categories:
            raise ActivationsError('Unable to calculate activations statistics; Please load annotations with load_annotations().')
        if method not in ('mean', 'threshold'):
            raise ActivationsError(f"'method' must be either 'mean' or 'threshold', not {method}")

        log.info('Calculating activation averages & stats across features...')

        tile_feature_stats = {}
        pt_feature_stats = {}
        category_stats = []
        activation_stats = {}
        for slide in self.slides:
            if method == 'mean':
                # Mean of each feature across tiles
                summarized = np.mean(self.activations[slide], axis=0)
            elif method == 'threshold':
                # For each feature, count number of tiles with value above threshold, divided by number of tiles
                summarized = np.sum((self.activations[slide] > threshold), axis=0) / self.activations[slide].shape[-1]
            activation_stats[slide] = summarized
        for c in self.used_categories:
            category_stats += [np.array([activation_stats[slide] for slide in self.slides if self.annotations[slide] == c])]

        for f in range(self.num_features):
            # Tile-level ANOVA
            fvalue, pvalue = stats.f_oneway(*list(self.activations_by_category(f).values()))
            if not isnan(fvalue) and not isnan(pvalue):
                tile_feature_stats.update({f: {'f': fvalue,
                                               'p': pvalue} })
            else:
                tile_feature_stats.update({f: {'f': -1,
                                               'p': 1} })
            # Patient-level ANOVA
            fvalue, pvalue = stats.f_oneway(*[c[:,f] for c in category_stats])
            if not isnan(fvalue) and not isnan(pvalue):
                pt_feature_stats.update({f: {'f': fvalue,
                                             'p': pvalue} })
            else:
                pt_feature_stats.update({f: {'f': -1,
                                             'p': 1} })

        try:
            pt_sorted_features = sorted(range(self.num_features), key=lambda f: pt_feature_stats[f]['p'])
        except:
            log.warning('No stats calculated; unable to sort features.')

        for f in range(self.num_features):
            try:
                log.debug(f"Tile-level P-value ({f}): {tile_feature_stats[f]['p']}")
                log.debug(f"Patient-level P-value: ({f}): {pt_feature_stats[f]['p']}")
            except:
                log.warning(f'No stats calculated for feature {f}')

        # Export results
        if outdir:
            if not exists(outdir): os.makedirs(outdir)
            filename=join(outdir, 'slide_level_summary.csv')
            log.info(f'Writing results to {sf.util.green(filename)}...')
            with open(filename, 'w') as outfile:
                csv_writer = csv.writer(outfile)
                header = ['slide', 'category'] + [f'Feature_{n}' for n in pt_sorted_features]
                csv_writer.writerow(header)
                for slide in self.slides:
                    category = self.annotations[slide]
                    row = [slide, category] + list(activation_stats[slide][pt_sorted_features])
                    csv_writer.writerow(row)
                if tile_feature_stats:
                    csv_writer.writerow(['Tile statistic', 'ANOVA P-value'] + [tile_feature_stats[n]['p'] for n in pt_sorted_features])
                    csv_writer.writerow(['Tile statistic', 'ANOVA F-value'] + [tile_feature_stats[n]['f'] for n in pt_sorted_features])
                if pt_feature_stats:
                    csv_writer.writerow(['Slide statistic', 'ANOVA P-value'] + [pt_feature_stats[n]['p'] for n in pt_sorted_features])
                    csv_writer.writerow(['Slide statistic', 'ANOVA F-value'] + [pt_feature_stats[n]['f'] for n in pt_sorted_features])

        return tile_feature_stats, pt_feature_stats, category_stats

    def logits_mean(self):
        """Calculates the mean logits vector across all tiles in each slide.

        Returns:
            dict:  This is a dictionary mapping slides to the mean logits array for all tiles in each slide.
        """

        return {s: np.mean(v, axis=0) for s,v in self.logits.items()}

    def logits_percent(self, prediction_filter=None):
        """Returns dictionary mapping slides to a vector of length num_logits with the percent of tiles in each
        slide predicted to be each outcome.

        Args:
            prediction_filter:  (optional) List of int. If provided, will restrict predictions to only these
                categories, with final prediction being based based on highest logit
                among these categories.

        Returns:
            dict:  This is a dictionary mapping slides to an array of percentages for each logit, of length num_logits
        """

        if prediction_filter:
            assert isinstance(prediction_filter, list) and all([isinstance(i, int) for i in prediction_filter])
            assert max(prediction_filter) <= self.num_logits
        else:
            prediction_filter = range(self.num_logits)

        slide_percentages = {}
        for slide in self.logits:
            # Find the index of the highest prediction for each tile, only for logits within prediction_filter
            tile_pred = np.argmax(self.logits[slide][:,prediction_filter], axis=1)
            slide_perc = np.array([np.count_nonzero(tile_pred==l)/len(tile_pred) for l in range(self.num_logits)])
            slide_percentages.update({slide: slide_perc})
        return slide_percentages

    def logits_predict(self, prediction_filter=None):
        """Returns slide-level predictions, assuming the model is predicting a categorical outcome, by generating
        a prediction for each individual tile, and making a slide-level prediction by finding the mostly frequently
        predicted outcome among its constituent tiles.

        Args:
            prediction_filter:  (optional) List of int. If provided, will restrict predictions to only these
                categories, with final prediction being based based on highest logit among these categories.

        Returns:
            dict:  Dictionary mapping slide names to final slide-level predictions.
        """

        if prediction_filter:
            assert isinstance(prediction_filter, list) and all([isinstance(i, int) for i in prediction_filter])
            assert max(prediction_filter) <= self.num_logits
        else:
            prediction_filter = range(self.num_logits)

        slide_predictions = {}
        for slide in self.logits:
            # Find the index of the highest prediction for each tile, only for logits within prediction_filter
            tile_pred = np.argmax(self.logits[slide][:,prediction_filter], axis=1)
            slide_perc = np.array([np.count_nonzero(tile_pred==l)/len(tile_pred) for l in range(self.num_logits)])
            slide_predictions.update({slide: np.argmax(slide_perc)})
        return slide_predictions

    def map_to_predictions(self, x=0, y=0):
        """Returns coordinates and metadata for tile-level predictions for all tiles,
        which can be used to create a SlideMap.

        Args:
            x (int, optional): Outcome category id for which predictions will be mapped to the X-axis. Defaults to 0.
            y (int, optional): Outcome category id for which predictions will be mapped to the Y-axis. Defaults to 0.

        Returns:
            list:   List of x-axis coordinates (predictions for the category 'x')
            list:   List of y-axis coordinates (predictions for the category 'y')
            list:   List of dictionaries containing tile-level metadata (used for SlideMap)
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

    def remove_slide(self, slide):
        """Removes slide from internally cached activations."""
        del self.activations[slide]
        del self.logits[slide]
        del self.locations[slide]
        self.tfrecords = [t for t in self.tfrecords if sf.util.path_to_name(t) != slide]
        try:
            self.slides.remove(slide)
        except ValueError:
            pass

    def save_example_tiles(self, features, outdir, slides=None, tiles_per_feature=100):
        """For a set of activation features, saves image tiles named according to their corresponding activations.

        Duplicate image tiles will be saved for each feature, organized into subfolders named according to feature.

        Args:
            features (list(int)): Features to evaluate.
            outdir (str):  Path to folder in which to save examples tiles.
            slides (list, optional): List of slide names. If provided, will only include tiles from these slides.
                Defaults to None.
            tiles_per_feature (int, optional): Number of tiles to include as examples for each feature. Defaults to 100.
                Will evenly sample this many tiles across the activation gradient.
        """

        if not isinstance(features, list): raise sf.util.UserError("'features' must be a list of int.")

        if not slides:
            slides = self.slides
        for f in features:
            if not exists(join(outdir, str(f))):
                os.makedirs(join(outdir, str(f)))

            gradient = []
            for slide in slides:
                for i, val in enumerate(self.activations[slide][:,f]):
                    gradient += [{
                                    'val': val,
                                    'slide': slide,
                                    'index': i
                    }]
            gradient = np.array(sorted(gradient, key=lambda k: k['val']))
            sample_idx = np.linspace(0, gradient.shape[0]-1, num=tiles_per_feature, dtype=np.int)
            for i, g in tqdm(enumerate(gradient[sample_idx]), ncols=80, leave=False, total=tiles_per_feature, desc=f"Feature {f}"):
                for tfr in self.tfrecords:
                    if sf.util.path_to_name(tfr) == g['slide']:
                        tfr_dir = tfr
                if not tfr_dir:
                    log.warning(f"TFRecord location not found for slide {g['slide']}")
                slide, image = sf.io.tensorflow.get_tfrecord_by_index(tfr_dir, g['index'], decode=False)
                tile_filename = f"{i}-tfrecord{g['slide']}-{g['index']}-{g['val']:.2f}.jpg"
                image_string = open(join(outdir, str(f), tile_filename), 'wb')
                image_string.write(image.numpy())
                image_string.close()

class Heatmap:
    """Generates heatmap by calculating predictions from a sliding scale window across a slide."""

    def __init__(self, slide, model, stride_div=2, roi_dir=None, roi_list=None, roi_method='inside',
                 normalizer=None, normalizer_source=None, batch_size=32, num_threads=8, buffer=None):

        """Convolutes across a whole slide, calculating logits and saving predictions internally for later use.

        Args:
            slide (str): Path to slide.
            model (str): Path to Tensorflow model.
            stride_div (int, optional): Divisor for stride when convoluting across slide. Defaults to 2.
            roi_dir (str, optional): Directory in which slide ROI is contained. Defaults to None.
            roi_list (list, optional): List of paths to slide ROIs. Defaults to None. Alternative to providing roi_dir.
            roi_method (str, optional): Either 'inside', 'outside', or 'ignore'. Defaults to 'inside'.
                If inside, tiles will be extracted inside ROI region.
                If outside, tiles will be extracted outside ROI region.
            normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image. Defaults to None.
                If None but using a normalizer, will use an internal tile for normalization.
                Internal default tile can be found at slideflow.util.norm_tile.jpg
            batch_size (int, optional): Batch size when calculating predictions. Defaults to 32.
            num_threads (int, optional): Number of tile extraction worker threads. Defaults to 8.
            buffer (str): Either 'vmtouch' or path to directory to use for buffering slides. Defaults to None.
                Significantly improves performance for slides on HDDs.
        """

        from slideflow.slide import WSI

        self.logits = None
        if (roi_dir is None and roi_list is None) and roi_method != 'ignore':
            log.info("No ROIs provided; will generate whole-slide heatmap")
            roi_method = 'ignore'

        interface = ActivationsInterface(model, layers=None, include_logits=True)
        model_hyperparameters = sf.util.get_model_params(model)
        self.tile_px = model_hyperparameters['tile_px']
        self.tile_um = model_hyperparameters['tile_um']
        self.num_classes = interface.num_logits
        self.num_features = interface.num_features

        # Create slide buffer
        if buffer and os.path.isdir(buffer):
            new_path = os.path.join(buffer, os.path.basename(slide))
            shutil.copy(slide, new_path)
            slide = new_path
            buffered_slide = True
        else:
            buffered_slide = False

        # Load the slide
        self.slide = WSI(slide,
                         self.tile_px,
                         self.tile_um,
                         stride_div,
                         enable_downsample=False,
                         roi_dir=roi_dir,
                         roi_list=roi_list,
                         roi_method=roi_method,
                         buffer=buffer,
                         skip_missing_roi=(roi_method == 'inside'))

        if not self.slide.loaded_correctly():
            raise ActivationsError(f'Unable to load slide {self.slide.name} for heatmap generation')

        self.logits = interface(self.slide,
                                normalizer=normalizer,
                                normalizer_source=normalizer_source,
                                num_threads=num_threads,
                                dtype=np.float32)

        log.info(f"Heatmap complete for {sf.util.green(self.slide.name)}")

        if buffered_slide:
            os.remove(new_path)

    def _prepare_figure(self, show_roi=True):
        self.fig = plt.figure(figsize=(18, 16))
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(bottom = 0.25, top=0.95)
        gca = plt.gca()
        gca.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)
        # Plot ROIs
        if show_roi:
            print('\r\033[KPlotting ROIs...', end='')
            ROI_SCALE = self.slide.full_shape[0]/2048
            annPolys = [sg.Polygon(annotation.scaled_area(ROI_SCALE)) for annotation in self.slide.rois]
            for poly in annPolys:
                x,y = poly.exterior.xy
                plt.plot(x, y, zorder=20, color='k', linewidth=5)

    def display(self, show_roi=True, interpolation='none', logit_cmap=None):
        """Interactively displays calculated logits as a heatmap.

        Args:
            show_roi (bool, optional): Overlay ROIs onto heatmap image. Defaults to True.
            interpolation (str, optional): Interpolation strategy to use for smoothing heatmap. Defaults to 'none'.
            logit_cmap (obj, optional): Either function or a dictionary use to create heatmap colormap.
                Each image tile will generate a list of predictions of length O, where O is the number of outcomes.
                If logit_cmap is a function, then the logit prediction list will be passed to the function,
                and the function is expected to return [R, G, B] values which will be displayed.
                If the logit_cmap is a dictionary, it should map 'r', 'g', and 'b' to indices; the prediction for
                these outcome indices will be mapped to the RGB colors. Thus, the corresponding color will only
                reflect up to three outcomes. Example mapping prediction for outcome 0 to the red colorspace, 3
                to green, etc: {'r': 0, 'g': 3, 'b': 1}
        """

        self._prepare_figure(show_roi=False)
        heatmap_dict = {}

        if show_roi: thumb = self.slide.annotated_thumb()
        else: thumb = self.slide.thumb()
        implot = FastImshow(thumb, self.ax, extent=None, tgt_res=1024)

        def slider_func(val):
            for h, s in heatmap_dict.values():
                h.set_alpha(s.val)

        if logit_cmap:
            if callable(logit_cmap):
                map_logit = logit_cmap
            else:
                def map_logit(l):
                    # Make heatmap with specific logit predictions mapped to r, g, and b
                    return (l[logit_cmap['r']], l[logit_cmap['g']], l[logit_cmap['b']])
            heatmap = self.ax.imshow([[map_logit(l) for l in row] for row in self.logits],
                                     extent=implot.extent,
                                     interpolation=interpolation,
                                     zorder=10)
        else:
            divnorm = mcol.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1.0)
            for i in range(self.num_classes):
                heatmap = self.ax.imshow(self.logits[:, :, i],
                                         extent=implot.extent,
                                         cmap='coolwarm',
                                         norm=divnorm,
                                         alpha = 0.0,
                                         interpolation=interpolation,
                                         zorder=10) #bicubic

                ax_slider = self.fig.add_axes([0.25, 0.2-(0.2/self.num_classes)*i, 0.5, 0.03], facecolor='lightgoldenrodyellow')
                slider = Slider(ax_slider, f'Class {i}', 0, 1, valinit = 0)
                heatmap_dict.update({f'Class{i}': [heatmap, slider]})
                slider.on_changed(slider_func)

        self.fig.canvas.set_window_title(self.slide.name)
        implot.show()
        plt.show()

    def save(self, outdir, show_roi=True, interpolation='none', logit_cmap=None, vmin=0, vmax=1, vcenter=0.5):
        """Saves calculated logits as heatmap overlays.

        Args:
            outdir (str): Path to directory in which to save heatmap images.
            show_roi (bool, optional): Overlay ROIs onto heatmap image. Defaults to True.
            interpolation (str, optional): Interpolation strategy to use for smoothing heatmap. Defaults to 'none'.
            logit_cmap (obj, optional): Either function or a dictionary use to create heatmap colormap.
                Each image tile will generate a list of predictions of length O, where O is the number of outcomes.
                If logit_cmap is a function, then the logit prediction list will be passed to the function,
                and the function is expected to return [R, G, B] values which will be displayed.
                If the logit_cmap is a dictionary, it should map 'r', 'g', and 'b' to indices; the prediction for
                these outcome indices will be mapped to the RGB colors. Thus, the corresponding color will only
                reflect up to three outcomes. Example mapping prediction for outcome 0 to the red colorspace, 3
                to green, etc: {'r': 0, 'g': 3, 'b': 1}
            vmin (float): Minimimum value to display on heatmap. Defaults to 0.
            vcenter (float): Center value for color display on heatmap. Defaults to 0.5.
            vmax (float): Maximum value to display on heatmap. Defaults to 1.
        """

        print('\r\033[KSaving base figures...', end='')

        # Save base thumbnail as separate figure
        self._prepare_figure(show_roi=False)
        self.ax.imshow(self.slide.thumb(width=2048), zorder=0)
        plt.savefig(os.path.join(outdir, f'{self.slide.name}-raw.png'), bbox_inches='tight')
        plt.clf()

        # Save thumbnail + ROI as separate figure
        self._prepare_figure(show_roi=False)
        self.ax.imshow(self.slide.annotated_thumb(width=2048), zorder=0)
        plt.savefig(os.path.join(outdir, f'{self.slide.name}-raw+roi.png'), bbox_inches='tight')
        plt.clf()

        # Now prepare base image for the the heatmap overlay
        self._prepare_figure(show_roi=False)
        thumb_func = self.slide.annotated_thumb if show_roi else self.slide.thumb
        implot = self.ax.imshow(thumb_func(width=2048), zorder=0)

        if logit_cmap:
            if callable(logit_cmap):
                map_logit = logit_cmap
            else:
                def map_logit(l):
                    # Make heatmap with specific logit predictions mapped to r, g, and b
                    return (l[logit_cmap['r']], l[logit_cmap['g']], l[logit_cmap['b']])

            heatmap = self.ax.imshow([[map_logit(l) for l in row] for row in self.logits],
                                     extent=implot.get_extent(),
                                     interpolation=interpolation,
                                     zorder=10)

            plt.savefig(os.path.join(outdir, f'{self.slide.name}-custom.png'), bbox_inches='tight')
        else:
            # Make heatmap plots and sliders for each outcome category
            for i in range(self.num_classes):
                print(f'\r\033[KMaking heatmap {i+1} of {self.num_classes}...', end='')
                divnorm = mcol.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                heatmap = self.ax.imshow(self.logits[:, :, i],
                                         extent=implot.get_extent(),
                                         cmap='coolwarm',
                                         norm=divnorm,
                                         alpha=0.6,
                                         interpolation=interpolation, #bicubic
                                         zorder=10)
                plt.savefig(os.path.join(outdir, f'{self.slide.name}-{i}.png'), bbox_inches='tight')
                heatmap.set_alpha(1)
                plt.savefig(os.path.join(outdir, f'{self.slide.name}-{i}-solid.png'), bbox_inches='tight')
                heatmap.remove()

        plt.close()
        print('\r\033[K', end='')
        log.info(f'Saved heatmaps for {sf.util.green(self.slide.name)}')
