import csv
import os
import pickle
import queue
import sys
import threading
import time
import warnings
import multiprocessing as mp
from collections import defaultdict
from math import isnan
from os.path import exists, join
from typing import (
    TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Iterable, Callable
)

import numpy as np
import pandas as pd
import scipy.stats as stats
import slideflow as sf
from rich.progress import track, Progress
from slideflow import errors
from slideflow.util import log, Labels, ImgBatchSpeedColumn, tfrecord2idx
from .base import BaseFeatureExtractor


if TYPE_CHECKING:
    import tensorflow as tf
    import torch


# -----------------------------------------------------------------------------

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
        self.model = model
        self.dataset = dataset
        self.generator = None
        if dataset is not None:
            self.tile_px = dataset.tile_px
            self.manifest = dataset.manifest()
            self.tfrecords = np.array(dataset.tfrecords())
        else:
            # Used when creating via DatasetFeatures.from_df(),
            # otherwise dataset should not be None.
            self.tile_px = None
            self.manifest = dict()
            self.tfrecords = []
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
        elif model is not None:
            self._generate_features(cache=cache, **kwargs)

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

    @classmethod
    def from_df(cls, df: "pd.core.frame.DataFrame"):
        """Load DataFrame of features, as exported by :meth:`DatasetFeatures.to_df()`"""
        obj = cls(None, None)  # type: ignore
        obj.slides = df.slide.unique().tolist()
        if 'activations' in df.columns:
            obj.activations = {
                s: np.stack(df.loc[df.slide==s].activations.values)
                for s in obj.slides
            }
            obj.num_features = next(df.iterrows())[1].activations.shape[0]
        if 'locations' in df.columns:
            obj.locations = {
                s: np.stack(df.loc[df.slide==s].locations.values)
                for s in obj.slides
            }
        if 'uncertainty' in df.columns:
            obj.uncertainty = {
                s: np.stack(df.loc[df.slide==s].uncertainty.values)
                for s in obj.slides
            }
        if 'predictions' in df.columns:
            obj.uncertainty = {
                s: np.stack(df.loc[df.slide==s].predictions.values)
                for s in obj.slides
            }
            obj.num_classes = next(df.iterrows())[1].predictions.shape[0]
        return obj

    @classmethod
    def concat(
        cls,
        args: Iterable["DatasetFeatures"],
    ):
        """Concatenates activations from multiple DatasetFeatures together.

        For example, if ``df1`` is a DatasetFeatures object with 2048 features
        and ``df2`` is a DatasetFeatures object with 1024 features,
        then ``sf.DatasetFeatures.concat([df1, df2])`` would return an object
        with 3072.

        Vectors from DatasetFeatures objects are concatenated in the given order.
        During concatenation, predictions and uncertainty are dropped.

        If there are any tiles that do not have calculated features in both
        dataframes, these will be dropped.
        """
        assert len(args) > 1
        dfs = []
        for f, ftrs in enumerate(args):
            log.debug(f"Creating dataframe {f} from features...")
            dfs.append(ftrs.to_df())
        if not all([len(df) == len(dfs[0]) for df in dfs]):
            raise ValueError(
                "Unable to concatenate DatasetFeatures of different lengths "
                f"(got: {', '.join([str(len(_df)) for _df in dfs])})"
            )
        log.debug(f"Created {len(dfs)} dataframes")
        for i in range(len(dfs)):
            log.debug(f"Mapping tuples for df {i}")
            dfs[i]['locations'] = dfs[i]['locations'].map(tuple)
        for i in range(1, len(dfs)):
            log.debug(f"Merging dataframe {i}")
            dfs[0] = pd.merge(
                dfs[0],
                dfs[i],
                how='inner',
                left_on=['slide', 'locations', 'tfr_index'],
                right_on=['slide', 'locations', 'tfr_index'],
                suffixes=['_1', '_2']
            )
            log.debug("Dropping merged columns")
            to_drop = [c for c in dfs[0].columns
                       if ('predictions' in c or 'uncertainty' in c)]
            dfs[0].drop(columns=to_drop, inplace=True)
            log.debug("Concatenating activations")
            act1 = np.stack(dfs[0]['activations_1'].values)
            act2 = np.stack(dfs[0]['activations_2'].values)
            log.debug(f"Act 1 shape: {act1.shape}")
            log.debug(f"Act 2 shape: {act2.shape}")
            concatenated = np.concatenate((act1, act2), axis=1)
            as_list = [_c for _c in concatenated]
            dfs[0]['activations'] = as_list
            log.debug("Dropping old columns")
            dfs[0].drop(columns=['activations_1', 'activations_2'], inplace=True)
        log.debug("Sorting by TFRecord index")
        dfs[0].sort_values('tfr_index', inplace=True)
        log.debug("Creating DatasetFeatures object")
        return DatasetFeatures.from_df(dfs[0])

    @property
    def uq(self) -> bool:
        if self.feature_generator is None:
            return None
        else:
            return self.feature_generator.uq

    @property
    def normalizer(self):
        if self.feature_generator is None:
            return None
        else:
            return self.feature_generator.normalizer

    def _generate_features(self, cache: Optional[str] = None, **kwargs) -> None:

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

        fg = self.feature_generator = _FeatureGenerator(
            self.model,
            self.dataset,
            **kwargs
        )
        self.num_features = fg.num_features
        self.num_classes = fg.num_classes

        # Calculate final layer activations for each tfrecord
        fla_start_time = time.time()

        activations, predictions, locations, uncertainty = fg.generate()

        self.activations = {s: np.stack(v) for s, v in activations.items()}
        self.predictions = {s: np.stack(v) for s, v in predictions.items()}
        self.locations = {s: np.stack(v) for s, v in locations.items()}
        self.uncertainty = {s: np.stack(v) for s, v in uncertainty.items()}

        # Sort using TFRecord location information,
        # to ensure dictionary indices reflect TFRecord indices
        if fg.tfrecords_have_loc:
            slides_to_sort = [
                s for s in self.slides
                if (self.activations[s].size
                    or not self.predictions[s].size
                    or not self.locations[s].size
                    or not self.uncertainty[s].size)
            ]
            pool = mp.Pool(os.cpu_count())
            for i, true_locs in enumerate(track(pool.imap(self.dataset.get_tfrecord_locations, slides_to_sort),
                                                transient=False,
                                                total=len(slides_to_sort),
                                                description="Sorting...")):
                slide = slides_to_sort[i]
                # Get the order of locations stored in TFRecords,
                # and the corresponding indices for sorting
                cur_locs = self.locations[slide]
                idx = [true_locs.index(tuple(cur_locs[i])) for i in range(cur_locs.shape[0])]

                # Make sure that the TFRecord indices are continuous, otherwise
                # our sorted indices will be inaccurate
                assert max(idx)+1 == len(idx)

                # Final sorting
                sorted_idx = np.argsort(idx)
                if slide in self.activations:
                    self.activations[slide] = self.activations[slide][sorted_idx]
                if slide in self.predictions:
                    self.predictions[slide] = self.predictions[slide][sorted_idx]
                if slide in self.uncertainty:
                    self.uncertainty[slide] = self.uncertainty[slide][sorted_idx]
                self.locations[slide] = self.locations[slide][sorted_idx]
            pool.close()

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

        Used for training MIL models.

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
            tfrecord2idx.save_index(
                self.locations[slide],
                join(outdir, f'{slide}.index')
            )
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
        df_dict = dict()
        df_dict.update({
            'locations': pd.Series([
                self.locations[s][i]
                for s in self.slides
                for i in range(len(self.locations[s]))], index=index)
        })
        df_dict.update({
            'tfr_index': pd.Series([
                i
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
        df = pd.DataFrame(df_dict)
        df['slide'] = df.index
        return df

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
                slide, image = sf.io.get_tfrecord_by_index(tfr_dir, g['index'])
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

# -----------------------------------------------------------------------------

class _FeatureGenerator:
    """Provides common API for feature generator interfaces."""

    def __init__(
        self,
        model: Union[str, "BaseFeatureExtractor", "tf.keras.models.Model", "torch.nn.Module"],
        dataset: "sf.Dataset",
        *,
        layers: Union[str, List[str]] = 'postconv',
        include_preds: Optional[bool] = None,
        include_uncertainty: bool = True,
        batch_size: int = 32,
        **kwargs
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.layers = sf.util.as_list(layers)
        self.batch_size = batch_size
        self.simclr_args = None

        # Check if location information is stored in TFRecords
        self.tfrecords_have_loc = self.dataset.tfrecords_have_locations()
        if not self.tfrecords_have_loc:
            log.warning(
                "Some TFRecords do not have tile location information; "
                "dataset iteration speed may be affected."
            )

        if self.is_extractor() and include_preds is None:
            include_preds = self.model.num_classes > 0  # type: ignore
        elif include_preds is None:
            include_preds = True
        self.include_preds = include_preds
        self.include_uncertainty = include_uncertainty

        self._determine_uq_and_normalizer()
        self.generator = self._prepare_generator(**kwargs)
        self.num_features = self.generator.num_features
        self.num_classes = 0 if not include_preds else self.generator.num_classes
        self._prepare_dataset_kwargs()

    def _calculate_feature_batch(self, batch_img):
        """Calculate features from a batch of images."""

        # If a PyTorch generator, wrap in no_grad() and perform on CUDA
        if self.is_torch():
            import torch
            with torch.no_grad():
                batch_img = batch_img.to('cuda')
                if self.has_torch_gpu_normalizer():
                    batch_img = self.normalizer.preprocess(
                        batch_img,
                        standardize=self.standardize
                    )
                return self.generator(batch_img)
        elif self.is_simclr():
            return self.generator(batch_img, training=False)
        else:
            return self.generator(batch_img)

    def _process_out(self, model_out, batch_slides, batch_loc):
        model_out = sf.util.as_list(model_out)

        # Process data if the output is Tensorflow (SimCLR or Tensorflow model)
        if self.is_simclr() or self.is_tf():
            slides = [
                bs.decode('utf-8')
                for bs in batch_slides.numpy()
            ]
            model_out = [
                m.numpy() if not isinstance(m, (list, tuple)) else m
                for m in model_out
            ]
            if batch_loc[0] is not None:
                loc = np.stack([
                    batch_loc[0].numpy(),
                    batch_loc[1].numpy()
                ], axis=1)
            else:
                loc = None

        # Process data if the output is PyTorch
        elif self.is_torch():
            slides = batch_slides
            model_out = [
                m.cpu().numpy() if not isinstance(m, list) else m
                for m in model_out
            ]
            if batch_loc[0] is not None:
                loc = np.stack([batch_loc[0], batch_loc[1]], axis=1)
            else:
                loc = None

        # Final processing
        if self.is_simclr():
            model_out = model_out[0]
        if self.uq and self.include_uncertainty:
            uncertainty = model_out[-1]
            model_out = model_out[:-1]
        else:
            uncertainty = None
        if self.include_preds:
            predictions = model_out[-1]
            features = model_out[:-1]
        else:
            predictions = None
            features = model_out

        # Concatenate features if we have features from >`` layer
        if self.layers:
            features = np.concatenate(features)

        return features, predictions, uncertainty, slides, loc

    def _prepare_dataset_kwargs(self):
        """Prepare keyword arguments for Dataset.tensorflow() or .torch()."""

        dts_kw = {
            'infinite': False,
            'batch_size': self.batch_size,
            'augment': False,
            'incl_slidenames': True,
            'incl_loc': True,
        }

        # If this is a Feature Extractor, update the dataset kwargs
        # with any preprocessing instructions specified by the extractor
        if self.is_extractor():
            dts_kw.update(self.model.preprocess_kwargs)

        # Establish standardization.
        self.standardize = ('standardize' not in dts_kw or dts_kw['standardize'])

        # Check if normalization is happening on GPU with PyTorch.
        # If so, we will handle normalization and standardization
        # in the feature generation loop.
        if self.has_torch_gpu_normalizer():
            log.info("Using GPU for stain normalization")
            dts_kw['standardize'] = False
        else:
            # Otherwise, let the dataset handle normalization/standardization.
            dts_kw['normalizer'] = self.normalizer

        self.dts_kw = dts_kw

    def _determine_uq_and_normalizer(self):
        # Load configuration if model is path to a saved model
        if isinstance(self.model, BaseFeatureExtractor):
            self.uq = self.model.num_uncertainty > 0
            self.normalizer = self.model.normalizer
        elif isinstance(self.model, str) and sf.util.is_simclr_model_path(self.model):
            self.uq = False
            self.normalizer = None
        elif isinstance(self.model, str):
            model_config = sf.util.get_model_config(self.model)
            hp = sf.ModelParams.from_dict(model_config['hp'])
            self.uq = hp.uq
            self.normalizer = hp.get_normalizer()
            if self.normalizer:
                log.info(f'Using realtime {self.normalizer.method} normalization')
                if 'norm_fit' in model_config:
                    self.normalizer.set_fit(**model_config['norm_fit'])
        else:
            self.normalizer = None
            self.uq = False

    def _prepare_generator(self, **kwargs) -> Callable:
        """Prepare the feature generator."""

        # Generator is a Feature Extractor
        if self.is_extractor():
            if kwargs:
                raise ValueError(
                    f"Invalid keyword arguments: {', '.join(list(kwargs.keys()))}"
                )
            return self.model

        # Generator is a model, and we're using UQ
        elif self.uq and self.include_uncertainty:
            return sf.model.UncertaintyInterface(
                self.model,
                layers=self.layers,
                **kwargs
            )

        # Generator is a SimCLR path
        elif self.is_simclr():
            from slideflow import simclr
            self.simclr_args = simclr.load_model_args(self.model)
            generator = simclr.load(self.model)
            generator.num_features = self.simclr_args.proj_out_dim
            generator.num_classes = self.simclr_args.num_classes
            if 'normalizer' in kwargs:
                self.normalizer = kwargs['normalizer']
                del kwargs['normalizer']
                log.info(f"Normalizing with {self.normalizer.method}")
            if kwargs:
                raise ValueError(
                    f"Invalid keyword arguments: {', '.join(list(kwargs.keys()))}"
                )
            return generator

        # Generator is a path to a trained Slideflow model
        elif self.is_model_path():
            return sf.model.Features(
                self.model,
                layers=self.layers,
                include_preds=self.include_preds,
                **kwargs
            )

        # Generator is a loaded Tensorflow model
        elif self.is_tf():
            return sf.model.Features.from_model(
                self.model,
                layers=self.layers,
                include_preds=self.include_preds,
                **kwargs
            )

        # Generator is a loaded torch.nn.Module
        elif self.is_torch():
            return sf.model.Features.from_model(
                self.model.to('cuda'),
                tile_px=self.tile_px,
                layers=self.layers,
                include_preds=self.include_preds,
                **kwargs
            )

        # Unrecognized feature generator
        else:
            raise ValueError(f'Unrecognized model {self.model}')

    def is_model_path(self):
        return isinstance(self.model, str) and (self.is_tf() or self.is_torch())

    def is_extractor(self):
        return isinstance(self.model, BaseFeatureExtractor)

    def is_simclr(self):
        return sf.util.is_simclr_model_path(self.model)

    def is_torch(self):
        if self.is_extractor():
            return self.model.is_torch()
        else:
            return sf.model.is_torch_model(self.model)

    def is_tf(self):
        if self.is_extractor():
            return self.model.is_tensorflow()
        else:
            return sf.model.is_tensorflow_model(self.model)

    def has_torch_gpu_normalizer(self):
        return (
            isinstance(self.normalizer, sf.norm.StainNormalizer)
            and self.normalizer.__class__.__name__ == 'TorchStainNormalizer'
            and self.normalizer.device != 'cpu'
        )

    def build_dataset(self):
        """Build a dataloader."""

        # Generator is a SimCLR model.
        if self.is_simclr():
            log.debug("Setting up Tensorflow/SimCLR dataset iterator")
            from slideflow import simclr
            builder = simclr.DatasetBuilder(
                val_dts=self.dataset,
                dataset_kwargs=dict(
                    incl_slidenames=True,
                    incl_loc=True,
                    normalizer=self.normalizer
                )
            )
            return builder.build_dataset(
                self.batch_size,
                is_training=False,
                simclr_args=self.simclr_args
            )

        # Generator is a Tensorflow model.
        elif self.is_tf():
            log.debug(
                "Setting up Tensorflow dataset iterator (num_parallel_reads="
                f"None, deterministic={not self.tfrecords_have_loc})"
            )
            return self.dataset.tensorflow(
                None,
                num_parallel_reads=None,
                deterministic=(not self.tfrecords_have_loc),
                **self.dts_kw  # type: ignore
            )

        # Generator is a PyTorch model.
        elif self.is_torch():
            n_workers = (4 if self.tfrecords_have_loc else 1)
            log.debug(
                "Setting up PyTorch dataset iterator (num_workers="
                f"{n_workers}, chunk_size=8)"
            )
            return self.dataset.torch(
                None,
                num_workers=n_workers,
                chunk_size=8,
                **self.dts_kw  # type: ignore
            )

        # Unrecognized feature generator.
        else:
            raise ValueError(f"Unrecognized model type: {type(self.model)}")

    def generate(self):

        # Get the dataloader for iterating through tfrecords
        dataset = self.build_dataset()

        # Rename tfrecord_array to tfrecords
        log.info(f'Calculating activations for {len(self.dataset.tfrecords())} '
                 f'tfrecords (layers={self.layers})')
        log.info(f'Generating from [green]{self.model}')

        # Interleave tfrecord datasets
        estimated_tiles = self.dataset.num_tiles

        activations = defaultdict(list)  # type: Dict[str, Any]
        predictions = defaultdict(list)  # type: Dict[str, Any]
        uncertainty = defaultdict(list)  # type: Dict[str, Any]
        locations = defaultdict(list)  # type: Dict[str, Any]

        # Worker to process activations/predictions, for more efficient throughput
        q = queue.Queue()  # type: queue.Queue

        def batch_worker():
            while True:
                model_out, batch_slides, batch_loc = q.get()
                if model_out is None:
                    return
                features, preds, unc, slides, loc = self._process_out(
                    model_out, batch_slides, batch_loc
                )

                for d, slide in enumerate(slides):
                    if self.layers:
                        activations[slide].append(features[d])
                    if self.include_preds and preds is not None:
                        predictions[slide].append(preds[d])
                    if self.uq and self.include_uncertainty:
                        uncertainty[slide].append(unc[d])
                    if loc is not None:
                        locations[slide].append(loc[d])

        batch_proc_thread = threading.Thread(target=batch_worker, daemon=True)
        batch_proc_thread.start()

        pb = Progress(*Progress.get_default_columns(),
                      ImgBatchSpeedColumn(),
                      transient=sf.getLoggingLevel()>20)
        task = pb.add_task("Generating...", total=estimated_tiles)
        pb.start()
        with sf.util.cleanup_progress(pb):
            for batch_img, _, batch_slides, batch_loc_x, batch_loc_y in dataset:
                model_output = self._calculate_feature_batch(batch_img)
                q.put((model_output, batch_slides, (batch_loc_x, batch_loc_y)))
                pb.advance(task, self.batch_size)
        q.put((None, None, None))
        batch_proc_thread.join()
        if hasattr(dataset, 'close'):
            dataset.close()

        return activations, predictions, locations, uncertainty
