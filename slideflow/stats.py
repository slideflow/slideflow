import csv
import multiprocessing as mp
import os
import pickle
import sys
import time
from functools import partial
from os.path import join
from random import sample
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index as c_index
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.special import softmax
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm

import slideflow as sf
from slideflow import errors
from slideflow.util import Path, ProgressBar, as_list
from slideflow.util import colors as col
from slideflow.util import log, to_onehot

if TYPE_CHECKING:
    import neptune.new as neptune
    import tensorflow as tf
    import torch

    from slideflow.model import DatasetFeatures

# TODO: remove 'hidden_0' reference as this may not be present
# if the model does not have hidden layers
# TODO: refactor all this x /y /meta /values stuff to a pd.DataFrame


class SlideMap:
    """Two-dimensional slide map for visualization & backend for mosaic maps.

    Slides are mapped in 2D either explicitly with pre-specified coordinates,
    or with dimensionality reduction from post-convolutional layer weights,
    provided from :class:`slideflow.DatasetFeatures`.
    """

    def __init__(
        self,
        slides: List[str],
        cache: Optional[Path] = None
    ) -> None:
        """Backend for mapping slides into two dimensional space. Can use a
        DatasetFeatures object to map slides according to UMAP of features, or
        map according to pre-specified coordinates.

        Args:
            slides (list(str)): List of slide names
            cache (str, optional): Path to PKL file to cache activations.
                Defaults to None (caching disabled).
        """

        self.slides = slides
        self.cache = cache
        self.df = None  # type: Optional[DatasetFeatures]
        self.x = np.array([])
        self.y = np.array([])
        self.point_meta = np.array([])
        self.labels = np.array([])
        self.map_meta = {}  # type: Dict[str, Any]
        if self.cache:
            self.load_cache()

    @classmethod
    def from_precalculated(
        cls,
        slides: List[str],
        x: Union[np.ndarray, List[int]],
        y: Union[np.ndarray, List[int]],
        meta: Union[np.ndarray, List[Dict]],
        labels: Optional[Union[np.ndarray, List[str]]] = None,
        cache: Optional[Path] = None
    ) -> "SlideMap":
        """Initializes map from precalculated coordinates.

        Args:
            slides (list(str)): List of slide names.
            x (list(int)): List of X coordinates for tfrecords.
            y (list(int)): List of Y coordinates for tfrecords.
            meta (list(dict)): List of dicts containing metadata for each point
                 on the map (representing a single tfrecord).
            labels (list(str)): Labels assigned to each tfrecord, used for
                coloring TFRecords according to labels.
            cache (str, optional): Path to PKL file to cache coordinates.
                Defaults to None (caching disabled).
        """

        obj = cls(slides)
        obj.cache = cache
        obj.x = np.array(x) if not isinstance(x, np.ndarray) else x
        obj.y = np.array(y) if not isinstance(y, np.ndarray) else y
        if not isinstance(meta, np.ndarray):
            obj.point_meta = np.array(meta)
        else:
            obj.point_meta = meta
        if not isinstance(labels, np.ndarray):
            obj.labels = np.array(labels)
        else:
            obj.labels = labels
        if obj.labels == []:
            obj.labels = np.array(['None' for i in range(len(obj.point_meta))])
        obj.save_cache()
        return obj

    @classmethod
    def from_features(
        cls,
        df: "DatasetFeatures",
        exclude_slides: Optional[List[str]] = None,
        prediction_filter: Optional[List[int]] = None,
        recalculate: bool = False,
        map_slide: Optional[str] = None,
        cache: Optional[Path] = None,
        low_memory: bool = False,
        umap_dim: int = 2,
        **umap_kwargs: Any
    ) -> "SlideMap":
        """Initializes map from dataset features.

        Args:
            df (:class:`slideflow.DatasetFeatures`): DatasetFeatures.
            exclude_slides (list, optional): List of slides to exclude.
            prediction_filter (list, optional) Restrict outcome predictions to
                only these provided categories.
            recalculate (bool, optional):  Force recalculation of umap despite
                presence of cache.
            use_centroid (bool, optional): Calculate/map centroid activations.
            map_slide (str, optional): Either None, 'centroid', or 'average'.
                If None, will map all tiles from each slide. Defaults to None.
            cache (str, optional): Path to PKL file to cache coordinates.
                Defaults to None (caching disabled).
        """

        if map_slide is not None and map_slide not in ('centroid', 'average'):
            raise errors.SlideMapError(
                "map_slide must be None, 'centroid' or 'average', (got "
                f"{map_slide})"
            )

        if not exclude_slides:
            slides = df.slides
        else:
            slides = [s for s in df.slides if s not in exclude_slides]

        obj = cls(slides, cache=cache)
        obj.df = df
        if map_slide:
            obj._calculate_from_slides(
                method=map_slide,
                prediction_filter=prediction_filter,
                recalculate=recalculate,
                low_memory=low_memory,
                **umap_kwargs
            )
        else:
            obj._calculate_from_tiles(
                prediction_filter=prediction_filter,
                recalculate=recalculate,
                low_memory=low_memory,
                dim=umap_dim,
                **umap_kwargs
            )
        return obj

    def _calculate_from_tiles(
        self,
        prediction_filter: Optional[List[int]] = None,
        recalculate: bool = False,
        **umap_kwargs: Any
    ) -> None:
        """Internal function to guide calculation of UMAP from final layer
        features / activations, as provided by DatasetFeatures.

        Args:
            prediction_filter (list, optional): Restrict predictions to this
                list of logits. Default is None.
            recalculate (bool, optional): Recalculate of UMAP despite loading
                from cache. Defaults to False.

        Keyword Args:
            dim (int): Number of dimensions for UMAP. Defaults to 2.
            n_neighbors (int): Number of neighbors for UMAP. Defaults to 50.
            min_dist (float): Minimum distance for UMAP. Defaults to 0.1.
            metric (str): UMAP metric. Defaults to 'cosine'.
            low_memory (bool). Operate UMAP in low memory mode.
                Defaults to False.
        """

        if prediction_filter:
            log.info("Masking UMAP logits through a prediction filter.")

        assert self.df is not None
        if self.x.size and self.y.size and not recalculate:
            log.info("UMAP loaded from cache, will not recalculate")

            # First, filter out slides not included in provided activations
            filtered_idx = np.array([
                i for i, x in enumerate(self.point_meta)
                if x['slide'] in self.df.slides
            ])
            self.x = self.x[filtered_idx]
            self.y = self.y[filtered_idx]
            self.point_meta = self.point_meta[filtered_idx]

            # If UMAP already calculated, update predictions
            # if prediction filter is provided
            for i in range(len(self.point_meta)):
                slide = self.point_meta[i]['slide']
                tile_index = self.point_meta[i]['index']
                logits = self.df.logits[slide][tile_index]
                prediction = filtered_prediction(logits, prediction_filter)
                self.point_meta[i]['logits'] = logits
                self.point_meta[i]['prediction'] = prediction
            return

        # Calculate UMAP
        node_activations = np.concatenate([
            self.df.activations[slide] for slide in self.slides
        ])
        self.map_meta['num_features'] = self.df.num_features
        log.info("Calculating UMAP...")
        running_pm = []
        for slide in self.slides:
            for i in range(self.df.activations[slide].shape[0]):
                location = self.df.locations[slide][i]
                logits = self.df.logits[slide][i]
                if self.df.logits[slide] != []:
                    pred = filtered_prediction(logits, prediction_filter)
                else:
                    pred = None
                pm = {
                    'slide': slide,
                    'index': i,
                    'prediction': pred,
                    'logits': logits,
                    'loc': location
                }
                if self.df.hp.uq and self.df.uncertainty != {}:  # type: ignore
                    pm.update({'uncertainty': self.df.uncertainty[slide][i]})
                running_pm += [pm]
        self.point_meta = np.array(running_pm)
        coordinates = gen_umap(node_activations, **umap_kwargs)

        self.x = np.array([c[0] for c in coordinates])
        if umap_kwargs['dim'] > 1:
            self.y = np.array([c[1] for c in coordinates])
        else:
            self.y = np.array([0 for i in range(len(self.x))])

        assert self.x.shape[0] == self.y.shape[0]
        assert self.x.shape[0] == len(self.point_meta)

        self.save_cache()

    def _calculate_from_slides(
        self, method: str = 'centroid',
        prediction_filter: Optional[List[int]] = None,
        recalculate: bool = False,
        **umap_kwargs: Any
    ) -> None:
        """ Internal function to guide calculation of UMAP from final layer
            activations for each tile, as provided via DatasetFeatures, and
            then map only the centroid tile for each slide.

        Args:
            method (str, optional): 'centroid' or 'average'. If centroid, will
                calculate UMAP only from centroid tiles for each slide.
                If average, will calculate UMAP based on average node
                activations across all tiles within the slide, then display the
                centroid tile for each slide.
            prediction_filter (list, optional): List of int. If provided, will
                restrict predictions to these categories.
            recalculate (bool, optional): Recalculate of UMAP despite loading
            from cache. Defaults to False.
            low_memory (bool, optional): Calculate UMAP in low-memory mode.
                Defaults to False.

        Keyword Args:
            dim (int): Number of dimensions for UMAP. Defaults to 2.
            n_neighbors (int): Number of neighbors for UMAP. Defaults to 50.
            min_dist (float): Minimum distance for UMAP. Defaults to 0.1.
            metric (str): UMAP metric. Defaults to 'cosine'.
            low_memory (bool). Operate UMAP in low memory mode.
                Defaults to False.
        """

        if method not in ('centroid', 'average'):
            _m = f'Method must be either "centroid" or "average", not {method}'
            raise errors.SlideMapError(_m)
        assert self.df is not None

        # Calculate optimal slide indices and centroid activations
        log.info("Calculating centroid indices...")
        opt_idx, centroid_activations = calculate_centroid(self.df.activations)

        # Restrict mosaic to only slides that had enough tiles to calculate
        # an optimal index from centroid
        successful_slides = list(opt_idx.keys())
        num_warned = 0
        for slide in self.df.slides:
            if slide not in successful_slides:
                log.debug(f"No centroid for {col.green(slide)}; skipping")
        if num_warned:
            log.warning(f"No centroid for {num_warned} slides.")
        if self.x.size and self.y.size and not recalculate:
            log.info("UMAP loaded from cache.")
            log.debug("Filtering UMAP to include only provided tiles")
            new_x, new_y, new_meta = [], [], []
            for i in range(len(self.point_meta)):
                slide = self.point_meta[i]['slide']
                idx = self.point_meta[i]['index']
                if (slide in opt_idx and idx == opt_idx[slide]):
                    new_x += [self.x[i]]
                    new_y += [self.y[i]]
                    if prediction_filter:
                        logits = self.df.logits[slide][idx]
                        prediction = filtered_prediction(
                            logits, prediction_filter
                        )
                        meta = {
                            'slide': slide,
                            'index': idx,
                            'logits': logits,
                            'prediction': prediction,
                        }
                    else:
                        meta = self.point_meta[i]
                    new_meta += [meta]
            self.x = np.array(new_x)
            self.y = np.array(new_y)
            self.point_meta = np.array(new_meta)
        else:
            log.info(f"Calculating UMAP from slide-level {method}...")
            umap_input = []
            running_pm = []
            for slide in self.slides:
                if method == 'centroid':
                    umap_input += [centroid_activations[slide]]
                elif method == 'average':
                    activation_averages = np.mean(self.df.activations[slide])
                    umap_input += [activation_averages]
                running_pm += [{
                    'slide': slide,
                    'index': opt_idx[slide],
                    'logits': [],
                    'prediction': 0
                }]
            coordinates = gen_umap(np.array(umap_input), **umap_kwargs)
            self.x = np.array([c[0] for c in coordinates])
            self.y = np.array([c[1] for c in coordinates])
            self.point_meta = np.array(running_pm)
            self.save_cache()

    def cluster(self, n_clusters: int) -> np.ndarray:
        """Performs clustering on data and adds to metadata labels.
        Requires a DatasetFeatures backend.

        Clusters are saved to self.point_meta[i]['cluster'].

        Args:
            n_clusters (int): Number of clusters for K means clustering.

        Returns:
            ndarray: Array with cluster labels corresponding to tiles
            in self.point_meta.
        """

        if self.df is None:
            raise errors.SlideMapError(
                "Unable to cluster; no DatasetFeatures provided"
            )
        activations = [
            self.df.activations[pm['slide']][pm['index']]
            for pm in self.point_meta
        ]
        log.info(f"Calculating K-means clustering (n={n_clusters})")
        kmeans = KMeans(n_clusters=n_clusters).fit(activations)
        labels = kmeans.labels_
        for i, label in enumerate(labels):
            self.point_meta[i]['cluster'] = label
        return np.array([p['cluster'] for p in self.point_meta])

    def export_to_csv(self, filename: Path) -> None:
        """Exports calculated UMAP coordinates in csv format.

        Args:
            filename (str): Path to CSV file in which to save coordinates.
        """
        with open(filename, 'w') as outfile:
            csvwriter = csv.writer(outfile)
            header = ['slide', 'index', 'x', 'y']
            csvwriter.writerow(header)
            for index in range(len(self.point_meta)):
                x = self.x[index]
                y = self.y[index]
                meta = self.point_meta[index]
                slide = meta['slide']
                index = meta['index']
                row = [slide, index, x, y]
                csvwriter.writerow(row)

    def filter_index(self, idx: Union[int, List[int], np.ndarray]) -> None:
        self.x = self.x[idx]
        self.y = self.y[idx]
        self.point_meta = self.point_meta[idx]
        self.labels = self.labels[idx]

    def neighbors(
        self,
        slide_categories: Optional[Dict] = None,
        algorithm: str = 'kd_tree',
        method: str = 'map',
        pca_dim: int = 100
    ) -> None:
        """Calculates neighbors among tiles in this map, assigning neighboring
            statistics to tile metadata 'num_unique_neighbors' and
            'percent_matching_categories'.

        Args:
            slide_categories (dict, optional): Maps slides to categories.
                Defaults to None. If provided, will be used to calculate
                'percent_matching_categories' statistic.
            algorithm (str, optional): NearestNeighbor algorithm, either
                'kd_tree', 'ball_tree', or 'brute'. Defaults to 'kd_tree'.
            method (str, optional): Either 'map', 'pca', or 'features'. How
                neighbors are determined. If 'map', calculates neighbors based
                on UMAP coordinates. If 'features', calculates neighbors on the
                full feature space. If 'pca', reduces features into `pca_dim`
                space. Defaults to 'map'.
        """
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
        if self.df is None:
            raise errors.SlideMapError(
                "Unable perform neighbor search; no DatasetFeatures provided"
            )
        log.info(f"Initializing neighbor search (method={method})...")
        if method == 'map':
            X = np.stack((self.x, self.y), axis=-1)
        elif method == 'features':
            X = np.array([
                self.df.activations[pm['slide']][pm['index']]
                for pm in self.point_meta
            ])
        elif method == 'pca':
            log.info(f"Reducing dimensionality with PCA (dim={pca_dim})...")
            pca = PCA(n_components=pca_dim)
            features = np.array([
                self.df.activations[pm['slide']][pm['index']]
                for pm in self.point_meta
            ])
            pca.fit(features)
            X = pca.transform(features)

        else:
            raise ValueError(f'Unknown neighbor method {method}.')
        nbrs = NearestNeighbors(
            n_neighbors=100,
            algorithm=algorithm,
            n_jobs=-1
        ).fit(X)
        log.info("Calculating nearest neighbors...")
        _, indices = nbrs.kneighbors(X)

        def cat_match(i, j):
            a1 = slide_categories[self.point_meta[i]['slide']]
            a2 = slide_categories[self.point_meta[j]['slide']]
            return a1 == a2

        log.info('Matching neighbors...')
        for i, ind in enumerate(indices):
            ind_pm = [self.point_meta[_i]['slide'] for _i in ind]
            num_unique_slides = len(list(set(ind_pm)))
            self.point_meta[i]['num_unique_neighbors'] = num_unique_slides
            if slide_categories:
                matching_categories = [_i for _i in ind if cat_match(_i, i)]
                perc = len(matching_categories)/len(ind)
                self.point_meta[i]['percent_matching_categories'] = perc

    def filter(self, slides: List[str]) -> None:
        """Filters map to only show tiles from the given slides.

        Args:
            slides (list(str)): List of slide names.
        """

        if not hasattr(self, 'full_x'):
            # Store full coordinates
            self.full_x = self.x
            self.full_y = self.y
            self.full_meta = self.point_meta
        else:
            # Restore full coordinates
            self.x = self.full_x
            self.y = self.full_y
            self.point_meta = self.full_meta
        self.x = np.array([
            self.x[xi] for xi in range(len(self.x))
            if self.point_meta[xi]['slide'] in slides
        ])
        self.y = np.array([
            self.y[yi] for yi in range(len(self.y))
            if self.point_meta[yi]['slide'] in slides
        ])
        self.point_meta = np.array([
            pm for pm in self.point_meta
            if pm['slide'] in slides
        ])

    def label_by_uncertainty(self, index: int = 0) -> None:
        """Labels each point with the tile-level uncertainty, if available.

        Args:
            index (int, optional): Uncertainty index. Defaults to 0.
        """
        if self.df is None:
            raise errors.SlideMapError("DatasetFeatures not provided.")
        if not self.df.hp.uq or self.df.uncertainty == {}:  # type: ignore
            raise errors.DatasetError(
                'Unable to label by uncertainty; UQ estimates not available.'
            )
        else:
            self.labels = np.array([
                m['uncertainty'][index] for m in self.point_meta
            ])

    def label_by_logits(self, index: int) -> None:
        """Displays each point with label equal to the logits (linear from 0-1)

        Args:
            index (int): Logit index.
        """
        self.labels = np.array([m['logits'][index] for m in self.point_meta])

    def label_by_slide(self, slide_labels: Optional[Dict] = None) -> None:
        """Displays each point as the name of the corresponding slide.
            If slide_labels is provided, will use this dict to label slides.

        Args:
            slide_labels (dict, optional): Dict mapping slide names to labels.
        """

        if slide_labels:
            self.labels = np.array([
                slide_labels[m['slide']] for m in self.point_meta
            ])
        else:
            self.labels = np.array([m['slide'] for m in self.point_meta])

    def label_by_meta(self, meta: str, translation_dict: Dict = None) -> None:
        """Displays each point labeled by tile metadata (e.g. 'prediction')

        Args:
            meta (str): Key to metadata from which to read
            translation_dict (dict, optional): If provided, will translate the
                read metadata through this dictionary.
        """
        if translation_dict:
            try:
                self.labels = np.array([
                    translation_dict[m[meta]] for m in self.point_meta
                ])
            except KeyError:
                # Try by converting metadata to string
                self.labels = np.array([
                    translation_dict[str(m[meta])] for m in self.point_meta
                ])
        else:
            self.labels = np.array([m[meta] for m in self.point_meta])

    def save_2d_plot(self, *args: Any, **kwargs: Any) -> None:
        """Deprecated function; please use `save`."""

        log.warning("save_2d_plot() is deprecated, please use save()")
        self.save(*args, **kwargs)

    def save(
        self,
        filename: str,
        subsample: Optional[int] = None,
        title: Optional[str] = None,
        cmap: Optional[Dict] = None,
        xlim: Tuple[float, float] = (-0.05, 1.05),
        ylim: Tuple[float, float] = (-0.05, 1.05),
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend: Optional[str] = None,
        dpi: int = 300,
        **scatter_kwargs: Any
    ) -> None:
        """Saves plot of data to a provided filename.

        Args:
            filename (str): File path to save the image.
            subsample (int, optional): Subsample to only include this many
                tiles on plot. Defaults to None.
            title (str, optional): Title for plot.
            cmap (dict, optional): Dict mapping labels to colors.
            xlim (list, optional): List of float indicating limit for x-axis.
                Defaults to (-0.05, 1.05).
            ylim (list, optional): List of float indicating limit for y-axis.
                Defaults to (-0.05, 1.05).
            xlabel (str, optional): Label for x axis. Defaults to None.
            ylabel (str, optional): Label for y axis. Defaults to None.
            legend (str, optional): Title for legend. Defaults to None.
            dpi (int, optional): DPI for final image. Defaults to 300.
        """
        import seaborn as sns
        from matplotlib import pyplot as plt

        # Subsampling
        if subsample:
            ri = sample(range(len(self.x)), min(len(self.x), subsample))
        else:
            ri = list(range(len(self.x)))

        df = pd.DataFrame()
        x = self.x[ri]
        y = self.y[ri]
        df['umap_x'] = x
        df['umap_y'] = y

        if len(self.labels):
            labels = self.labels[ri]

            # Check for categorical labels
            if not np.issubdtype(self.labels.dtype, np.floating):
                log.debug("Interpreting labels as categorical")
                df['category'] = pd.Series(labels, dtype='category')
                unique = list(set(labels))
                unique.sort()
                if len(unique) >= 12:
                    sns_pal = sns.color_palette("Paired", len(unique))
                else:
                    sns_pal = sns.color_palette('hls', len(unique))
                if cmap is None:
                    cmap = {unique[i]: sns_pal[i] for i in range(len(unique))}
            else:
                log.debug("Interpreting labels as continuous")
                df['category'] = labels
        else:
            labels = ['NA']
            df['category'] = 'NA'

        # Make plot
        plt.clf()
        umap_2d = sns.scatterplot(
            x=x,
            y=y,
            data=df,
            hue='category',
            palette=cmap,
            **scatter_kwargs
        )
        plt.gca().set_ylim(*((None, None) if not ylim else ylim))
        plt.gca().set_xlim(*((None, None) if not xlim else xlim))
        umap_2d.legend(
            loc='center left',
            bbox_to_anchor=(1.25, 0.5),
            ncol=1,
            title=legend
        )
        umap_2d.set(xlabel=xlabel, ylabel=ylabel)
        umap_figure = umap_2d.get_figure()
        umap_figure.set_size_inches(6, 4.5)
        if title:
            umap_figure.axes[0].set_title(title)
        umap_figure.savefig(filename, bbox_inches='tight', dpi=dpi)
        log.info(f"Saved 2D UMAP to {col.green(filename)}")

    def save_3d_plot(
        self,
        filename: str,
        z: Optional[np.ndarray] = None,
        feature: Optional[int] = None,
        subsample: Optional[int] = None
    ) -> None:
        """Saves a plot of a 3D umap, with the 3rd dimension representing
        values provided by argument "z".

        Args:
            filename (str): Filename to save image of plot.
            z (list, optional): Values for z axis. Must supply z or feature.
                Defaults to None.
            feature (int, optional): Int, feature to plot on 3rd axis.
                Must supply z or feature. Defaults to None.
            subsample (int, optional): Subsample to only include this many
                tiles on plot. Defaults to None.
        """
        from matplotlib import pyplot as plt

        title = f"UMAP with feature {feature} focus"
        if self.df is None:
            raise errors.SlideMapError("DatasetFeatures not provided.")
        if not filename:
            filename = "3d_plot.png"
        if (z is None) and (feature is None):
            raise errors.SlideMapError("Must supply either 'z' or 'feature'.")
        # Get feature activations for 3rd dimension
        if z is None:
            z = np.array([
                self.df.activations[m['slide']][m['index']][feature]
                for m in self.point_meta
            ])
        # Subsampling
        if subsample:
            ri = sample(range(len(self.x)), min(len(self.x), subsample))
        else:
            ri = list(range(len(self.x)))
        x = self.x[ri]
        y = self.y[ri]
        z = z[ri]

        # Plot tiles on a 3D coordinate space with 2 coordinates from UMAP
        # and 3rd from the value of the excluded feature
        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5, edgecolor="k")
        ax.set_title(title)
        log.info(f"Saving 3D UMAP to {col.green(filename)}...")
        plt.savefig(filename, bbox_inches='tight')

    def get_tiles_in_area(
        self,
        x_lower: float,
        x_upper: float,
        y_lower: float,
        y_upper: float
    ) -> Dict[str, List[int]]:

        """Returns dictionary of slide names mapping to tile indices,
            or tiles that fall within the specified location on the umap.

        Args:
            x_lower (int): X-axis lower limit.
            x_upper (int): X-axis upper limit.
            y_lower (int): Y-axis lower limit.
            y_upper (int): Y-axis upper limit.

        Returns:
            dict: Dict mapping slide names to matching tile indices
        """

        # Find tiles that meet UMAP location criteria
        filtered_tiles = {}
        num_selected = 0
        for i in range(len(self.point_meta)):
            x_in_range = (x_lower < self.x[i] < x_upper)
            y_in_range = (y_lower < self.y[i] < y_upper)
            if x_in_range and y_in_range:
                slide = self.point_meta[i]['slide']
                tile_index = self.point_meta[i]['index']
                if slide not in filtered_tiles:
                    filtered_tiles.update({slide: [tile_index]})
                else:
                    filtered_tiles[slide] += [tile_index]
                num_selected += 1
        log.info(f"Selected {num_selected} tiles by filter criteria.")
        return filtered_tiles

    def save_cache(self) -> None:
        """Save cache of coordinates to PKL file."""
        if self.cache:
            try:
                with open(self.cache, 'wb') as cache_file:
                    pickle.dump(
                        [self.x, self.y, self.point_meta, self.map_meta],
                        cache_file
                    )
                    log.info(f"Wrote UMAP cache to {col.green(self.cache)}")
            except Exception:
                log.info(f"Error writing cache to {col.green(self.cache)}")

    def load_cache(self) -> bool:
        """Load coordinates from PKL cache.

        Returns:
            bool: If successfully loaded from cache.
        """
        if self.cache is None:
            raise errors.SlideMapError("Unable to load cache; none set.")
        try:
            with open(self.cache, 'rb') as f:
                self.x, self.y, self.point_meta, self.map_meta = pickle.load(f)
                log.info(f"Loaded UMAP cache from {col.green(self.cache)}")
                return True
        except FileNotFoundError:
            log.info(f"No UMAP cache found at {col.green(self.cache)}")
        return False


def _generate_tile_roc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_dir: str,
    label: str,
    histogram: bool = False,
    neptune_run: Optional["neptune.Run"] = None
) -> Tuple[float, float, float]:
    """Generate tile-level ROC. Defined separately for multiprocessing."""
    auc, ap, thresh = generate_roc(
        y_true,
        y_pred,
        data_dir,
        label.format('ROC'),
        neptune_run
    )
    if histogram:
        save_histogram(
            y_true,
            y_pred,
            outdir=data_dir,
            name=label.format('histogram'),
            neptune_run=neptune_run
        )
    return auc, ap, thresh  # ROC AUC, Average Precision, Optimal Threshold


def _cph_metrics(args: SimpleNamespace) -> None:
    """Internal function to calculate tile, slide, and patient-level metrics
    for a CPH outcome.
    """
    args.c_index['tile'] = concordance_index(
        args.df['time-y_true'].values,
        args.df[['time-y_pred', 'event-y_true']].values,
    )
    for group in ('patient', 'slide'):
        group_df = args.df.groupby(group).mean()
        if args.should_save_predictions(group):
            group_df.to_csv(join(
                args.data_dir,
                f"{group}_predictions{args.label_end}.csv",
            ))
        args.c_index['slide'] = concordance_index(
            group_df['time-y_true'].values,
            group_df[['time-y_pred', 'event-y_true']].values,
        )


def _linear_metrics(args: SimpleNamespace) -> None:
    """Internal function to calculate tile, slide, and patient-level metrics
    for a linear outcome.
    """
    y_pred_cols = [f'{o}-y_pred' for o in args.outcome_names]
    y_true_cols = [f'{o}-y_true' for o in args.outcome_names]

    args.r_squared['tile'] = generate_scatter(
        args.df[y_true_cols].values,
        args.df[y_pred_cols].values,
        args.data_dir,
        args.label_end,
        plot=args.plot,
        neptune_run=args.neptune_run
    )
    for group in ('patient', 'slide'):
        group_df = args.df.groupby(group).mean()
        if args.should_save_predictions(group):
            group_df.to_csv(join(
                args.data_dir,
                f"{group}_predictions{args.label_end}.csv"
            ))
        args.r_squared[group] = generate_scatter(
            group_df[y_true_cols].values,
            group_df[y_pred_cols].values,
            args.data_dir,
            f"{args.label_end}_by_{group}",
            neptune_run=args.neptune_run
        )


def _categorical_metrics(args: SimpleNamespace, outcome_name: str) -> None:
    """Internal function to calculate tile, slide, and patient level metrics
    for a categorical outcome.
    """
    y_pred_cols = [c for c in args.df.columns if c.startswith('y_pred')]
    original_cols = [c for c in args.df.columns if c not in ('patient', 'slide')]
    num_cat = args.df.y_true.max()+1
    num_pred = len(y_pred_cols)
    if num_cat != num_pred:
        raise errors.StatsError(
            "Model predictions have a different number of outcome "
            f"categories ({num_pred}) "
            f"than provided annotations ({num_cat})"
        )

    # Convert to one-hot encoding
    for i in range(num_cat):
        args.df[f'y_true_onehot{i}'] = (args.df.y_true == i).astype(int)
        args.df[f'y_pred_onehot{i}'] = (args.df[f'y_pred{i}'] == args.df[y_pred_cols].values.max(1)).astype(int)

    for level in ('tile', 'slide', 'patient'):
        args.auc[level][outcome_name] = []
        args.ap[level][outcome_name] = []

    #ctx = mp.get_context('spawn')
    #with ctx.Pool(processes=8) as p:
    def _gen_roc(i):
        return _generate_tile_roc(
            y_true=args.df[f'y_true_onehot{i}'],
            y_pred=args.df[f'y_pred{i}'],
            data_dir=args.data_dir,
            label=str(args.label_start + outcome_name) + "_tile_{}" + str(i),
            histogram=args.histogram
        )
    try:
        for i, (auc, ap, thresh) in enumerate(map(_gen_roc, range(num_cat))): #p.imap
            args.auc['tile'][outcome_name] += [auc]
            args.ap['tile'][outcome_name] += [ap]
            log.info(
                f"Tile-level AUC (cat #{i:>2}): {auc:.3f} "
                f"AP: {ap:.3f} (opt. threshold: {thresh:.3f})"
            )
    except ValueError as e:
        # Occurs when predictions contain NaN
        log.error(f'Error encountered when generating AUC: {e}')
        for level in ('tile', 'slide', 'patient'):
            args.auc[level][outcome_name] = -1
            args.ap[level][outcome_name] = -1
        return

    # Calculate tile-level accuracy.
    # Category-level accuracy is determined by comparing
    # one-hot predictions to one-hot y_true.
    for i in range(num_cat):
        try:
            yt_in_cat = args.df[f'y_true_onehot{i}']
            n_in_cat = yt_in_cat.sum()
            correct = args.df.loc[args.df[f'y_true_onehot{i}'] > 0][f'y_pred_onehot{i}'].sum()
            category_accuracy = correct / n_in_cat
            perc = category_accuracy * 100
            log.info(f"Category {i} acc: {perc:.1f}% ({correct}/{n_in_cat})")
        except IndexError:
            log.warning(f"Error with category accuracy for cat # {i}")

    # Calculate patient- and slide-level metrics
    for group in ('patient', 'slide'):
        group_df = args.df.groupby(group).mean()
        if args.should_save_predictions(group):
            group_df[original_cols].to_csv(join(
                args.data_dir,
                f"{group}_predictions_{outcome_name}{args.label_end}.csv",
            ))
        for i in range(num_cat):
            roc_auc, ap, thresh = generate_roc(
                group_df[f'y_true_onehot{i}'],
                group_df[f'y_pred_onehot{i}'],
                args.data_dir,
                f'{args.label_start}{outcome_name}_{group}_ROC{i}',
                neptune_run=args.neptune_run
            )
            args.auc[group][outcome_name] += [roc_auc]
            args.ap[group][outcome_name] += [ap]
            log.info(
                f"{group}-level AUC (cat #{i:>2}): {roc_auc:.3f}"
                f", AP: {ap:.3f} (opt. threshold: {thresh:.3f})"
            )


def filtered_prediction(
    logits: np.ndarray,
    filter: Optional[list] = None
) -> int:
    """Generates a prediction from a logits vector masked by a given filter.

    Args:
        filter (list(int)): List of logit indices to include when generating
            a prediction. All other logits will be masked.

    Returns:
        int: index of prediction.
    """
    if filter is None:
        prediction_mask = np.zeros(logits.shape, dtype=int)
    else:
        prediction_mask = np.ones(logits.shape, dtype=int)
        prediction_mask[filter] = 0
    masked_logits = np.ma.masked_array(logits, mask=prediction_mask)  # type: ignore
    return int(np.argmax(masked_logits))


def get_centroid_index(arr: np.ndarray) -> int:
    """Calculate index nearest to centroid from a given 2D input array."""
    km = KMeans(n_clusters=1).fit(arr)
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, arr)
    return closest[0]


def calculate_centroid(
    act: Dict[str, np.ndarray]
) -> Tuple[Dict[str, int], Dict[str, np.ndarray]]:
    """Calcultes slide-level centroid indices for a provided activations dict.

    Args:
        activations (dict): Dict mapping slide names to ndarray of activations
            across tiles, of shape (n_tiles, n_features)

    Returns:
        dict: Dict mapping slides to index of tile nearest to centroid
        dict: Dict mapping slides to activations of tile nearest to centroid
    """

    optimal_indices = {}
    centroid_activations = {}
    for slide in act:
        if not len(act[slide]):
            continue
        km = KMeans(n_clusters=1).fit(act[slide])
        closest, _ = pairwise_distances_argmin_min(
            km.cluster_centers_,
            act[slide]
        )
        closest_index = closest[0]
        closest_activations = act[slide][closest_index]
        optimal_indices.update({slide: closest_index})
        centroid_activations.update({slide: closest_activations})
    return optimal_indices, centroid_activations


def normalize_layout(
    layout: np.ndarray,
    min_percentile: int = 1,
    max_percentile: int = 99,
    relative_margin: float = 0.1
) -> np.ndarray:
    """Removes outliers and scales layout to between [0,1]."""

    # Compute percentiles
    mins = np.percentile(layout, min_percentile, axis=(0))
    maxs = np.percentile(layout, max_percentile, axis=(0))
    # Add margins
    mins -= relative_margin * (maxs - mins)
    maxs += relative_margin * (maxs - mins)
    # `clip` broadcasts, `[None]`s added only for readability
    clipped = np.clip(layout, mins, maxs)
    # embed within [0,1] along both axes
    clipped -= clipped.min(axis=0)
    clipped /= clipped.max(axis=0)
    return clipped


def gen_umap(
    array: np.ndarray,
    dim: int = 2,
    n_neighbors: int = 50,
    min_dist: float = 0.1,
    metric: str = 'cosine',
    **kwargs: Any
) -> np.ndarray:
    """Generates and returns a umap from a given array, using umap.UMAP"""

    import umap  # Imported in this function due to long import time
    if not len(array):
        raise errors.StatsError("Unable to perform UMAP on empty array.")
    layout = umap.UMAP(n_components=dim,
                       verbose=(log.getEffectiveLevel() <= 20),
                       n_neighbors=n_neighbors,
                       min_dist=min_dist,
                       metric=metric,
                       **kwargs).fit_transform(array)
    return normalize_layout(layout)


def save_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    outdir: str,
    name: str = 'histogram',
    subsample: int = 500,
    neptune_run: Optional["neptune.Run"] = None
) -> None:
    """Generates histogram of y_pred, labeled by y_true, saving to outdir."""
    import seaborn as sns
    from matplotlib import pyplot as plt

    # Subsample
    if subsample and y_pred.shape[0] > subsample:
        idx = np.arange(y_pred.shape[0])
        idx = np.random.choice(idx, subsample)
        y_pred = y_pred[idx]
        y_true = y_true[idx]

    cat_false = y_pred[y_true == 0]
    cat_true = y_pred[y_true == 1]
    plt.clf()
    plt.title('Tile-level Predictions')
    plot_kw = {
        'bins': 30,
        'kde': True,
        'stat': 'density',
        'linewidth': 0
    }
    try:
        sns.histplot(cat_false, color="skyblue", label="Negative", **plot_kw)
        sns.histplot(cat_true, color="red", label="Positive", **plot_kw)
    except np.linalg.LinAlgError:
        log.warning("Unable to generate histogram, insufficient data")
    plt.legend()
    plt.savefig(os.path.join(outdir, f'{name}.png'))
    if neptune_run:
        neptune_run[f'results/graphs/{name}'].upload(
            os.path.join(outdir, f'{name}.png')
        )


def generate_roc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: Optional[str] = None,
    name: str = 'ROC',
    neptune_run: Optional["neptune.Run"] = None
) -> Tuple[float, float, float]:
    """Generates and saves an ROC with a given set of y_true, y_pred values."""
    # ROC
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    # Precision recall
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    ap = metrics.average_precision_score(y_true, y_pred)

    # Calculate optimal cutoff via maximizing Youden's J statistic
    # (sens+spec-1, or TPR - FPR)
    try:
        max_youden = max(zip(tpr, fpr), key=lambda x: x[0]-x[1])
        opt_thresh_index = list(zip(tpr, fpr)).index(max_youden)
        opt_thresh = threshold[opt_thresh_index]
    except Exception:
        opt_thresh = -1
    if save_dir:
        from matplotlib import pyplot as plt

        # ROC
        plt.clf()
        plt.title('ROC Curve')
        plt.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc:.2f}')
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.savefig(os.path.join(save_dir, f'{name}.png'))
        # Precision recall
        plt.clf()
        plt.title('Precision-Recall Curve')
        plt.plot(precision, recall, 'b', label=f'AP = {ap:.2f}')
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Recall')
        plt.xlabel('Precision')
        plt.savefig(os.path.join(save_dir, f'{name}-PRC.png'))
        if neptune_run:
            neptune_run[f'results/graphs/{name}'].upload(
                os.path.join(save_dir, f'{name}.png')
            )
            neptune_run[f'results/graphs/{name}-PRC'].upload(
                os.path.join(save_dir, f'{name}-PRC.png')
            )
    return roc_auc, ap, opt_thresh


def generate_combined_roc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str,
    labels: List[Union[str, int]],
    name: str = 'ROC',
    neptune_run: Optional["neptune.Run"] = None
) -> List[float]:
    """Generates and saves overlapping ROCs."""
    from matplotlib import pyplot as plt

    plt.clf()
    plt.title(name)
    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    rocs = []
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        fpr, tpr, threshold = metrics.roc_curve(yt, yp)
        roc_auc = metrics.auc(fpr, tpr)
        rocs += [roc_auc]
        label = f'{labels[i]} (AUC: {roc_auc:.2f})'
        plt.plot(fpr, tpr, colors[i % len(colors)], label=label)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.savefig(os.path.join(save_dir, f'{name}.png'))
    if neptune_run:
        neptune_run[f'results/graphs/{name}'].upload(
            os.path.join(save_dir, f'{name}.png')
        )
    return rocs


def read_predictions(
    path: Path,
    level: str
) -> Dict[str, Dict[str, List[float]]]:
    """Reads predictions from a previously saved CSV file."""
    predictions = {}  # type: Dict[str, Dict[str, List[float]]]
    if level in ('patient', 'slide'):
        y_pred_label = f'y_pred_{level}'
    else:
        y_pred_label = 'y_pred'
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        prediction_labels = [
            h.split('y_true')[-1] for h in header if "y_true" in h
        ]
        for label in prediction_labels:
            predictions.update({label: {
                'y_true': [],
                'y_pred': []
            }})
        for row in reader:
            for label in prediction_labels:
                yti = header.index(f'y_true{label}')
                ypi = header.index(f'{y_pred_label}{label}')
                predictions[label]['y_true'] += [int(row[yti])]
                predictions[label]['y_pred'] += [float(row[ypi])]
    return predictions


def generate_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_dir: str,
    name: str = '_plot',
    plot: bool = True,
    neptune_run: Optional["neptune.Run"] = None
) -> List[float]:
    """Generate and save scatter plots and calculate R2 for each outcome.

        Args:
            y_true (np.ndarray): 2D array of labels. Observations are in first
                dimension, second dim is the outcome.
            y_pred (np.ndarray): 2D array of predictions.
            data_dir (str): Path to directory in which to save plots.
            name (str, optional): Label for filename. Defaults to '_plot'.
            plot (bool, optional): Save scatter plots.
            neptune_run (optional): Neptune Run. If provided, will upload plot.

        Returns:
            R squared.
    """
    import seaborn as sns
    from matplotlib import pyplot as plt

    if y_true.shape != y_pred.shape:
        m = f"Shape mismatch: y_true {y_true.shape} y_pred: {y_pred.shape}"
        raise errors.StatsError(m)
    if y_true.shape[0] < 2:
        raise errors.StatsError("Only one observation provided, need >1")
    r_squared = []
    # Perform scatter for each outcome
    for i in range(y_true.shape[1]):
        # y = mx + b
        m, b, r, p_val, err = stats.linregress(y_true[:, i], y_pred[:, i])
        r_squared += [r ** 2]
        if plot:
            p = sns.jointplot(x=y_true[:, i], y=y_pred[:, i], kind="reg")
            p.set_axis_labels('y_true', 'y_pred')
            plt.savefig(os.path.join(data_dir, f'Scatter{name}-{i}.png'))
            if neptune_run:
                neptune_run[f'results/graphs/Scatter{name}-{i}'].upload(
                    os.path.join(data_dir, f'Scatter{name}-{i}.png')
                )
    return r_squared


def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    '''Generates metrics, including sensitivity, specificity, and accuracy.'''
    assert(len(y_true) == len(y_pred))
    assert([y in (0, 1) for y in y_true])
    assert([y in (0, 1) for y in y_pred])

    TP = 0  # True positive
    TN = 0  # True negative
    FP = 0  # False positive
    FN = 0  # False negative

    for i, yt in enumerate(y_true):
        yp = y_pred[i]
        if yt == 1 and yp == 1:
            TP += 1
        elif yt == 1 and yp == 0:
            FN += 1
        elif yt == 0 and yp == 1:
            FP += 1
        elif yt == 0 and yp == 0:
            TN += 1

    results = {}
    results['accuracy'] = (TP + TN) / (TP + TN + FP + FN)
    results['sensitivity'] = TP / (TP + FN)
    results['specificity'] = TN / (TN + FP)
    results['precision'] = metrics.precision_score(y_true, y_pred)
    results['recall'] = metrics.recall_score(y_true, y_pred)
    results['f1_score'] = metrics.f1_score(y_true, y_pred)
    results['kappa'] = metrics.cohen_kappa_score(y_true, y_pred)
    return results


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    '''Calculates concordance index from a given y_true and y_pred.'''
    E = y_pred[:, -1]
    y_pred = y_pred[:, :-1]
    y_pred = y_pred.flatten()
    E = E.flatten()
    y_true = y_true.flatten()
    # Need -1 * concordance index, since these are log hazard ratios
    y_pred = - y_pred
    return c_index(y_true, y_pred, E)


def df_from_pred(y_true, y_pred, y_std, tile_to_slides):
    series = {
        'slide': pd.Series(tile_to_slides)
    }
    for oi in range(len(y_pred)):
        # Add y_pred columns
        series.update({
            f'out{oi}-y_pred{n}': y_pred[0][:, n]
            for n in range(y_pred[0].shape[1])
        })
        # Add y_true columns
        if y_true is not None:
            if len(y_true[0].shape) == 1:
                series.update({
                    f'out{oi}-y_true': y_true[0]
                })
            else:
                series.update({
                    f'out{oi}-y_true{n}': y_true[0][:, n]
                    for n in range(y_true[0].shape[1])
                })
        # Add uncertainty columns
        if y_std is not None:
            series.update({
                f'out{oi}-uncertainty{n}': y_std[0][:, n]
                for n in range(y_std[0].shape[1])
            })
    return pd.DataFrame(series)


def metrics_from_pred(
    df: pd.core.frame.DataFrame,
    patients: Dict[str, str],
    model_type: str,
    categorical_reduce: str = 'raw',
    outcome_names: Optional[List[str]] = None,
    label: str = '',
    data_dir: str = '',
    save_predictions: bool = True,
    histogram: bool = False,
    plot: bool = True,
    neptune_run: Optional["neptune.Run"] = None
) -> Dict[str, Dict[str, float]]:
    """Generates metrics from a set of predictions.

    For multiple outcomes, y_true and y_pred are expected to be a list of
    numpy arrays (each array corresponding to whole-dataset predictions
    for a single outcome)

    Args:
        y_true (ndarray): True labels for the dataset.
        y_pred (ndarray): Predicted labels for the dataset.
        tile_to_slides (list(str)): List of length y_true of slide names.
        labels (dict): Dictionary mapping slidenames to outcomes.
        patients (dict): Dictionary mapping slidenames to patients.
        model_type (str): Either 'linear', 'categorical', or 'cph'.

    Keyword args:
        y_std (np.ndarray, optional): Std. deviation (uncertainty) for dataset.
        categorical_reduce (str, optional): Reduction strategy for calculating
            slide-level and patient-level predictions for categorical outcomes.
            Either 'raw' or 'onehot'. If 'raw', will reduce with average of
            each logit across tiles. If 'onehot', will convert tile predictions
            into onehot encoding via `np.argmax`, then reduce by averaging
            these onehot values. Defaults to 'raw'.
        outcome_names (list, optional): List of str, names for outcomes.
            Defaults to None.
        label (str, optional): Label prefix/suffix for saving.
            Defaults to None.
        min_tiles (int, optional): Min tiles per slide to include in metrics.
            Defaults to 0.
        data_dir (str, optional): Path to data directory for saving.
            Defaults to None.
        save_predictions (bool, optional): Save tile, slide, and patient-level
            predictions to CSV. Defaults to True.
        histogram (bool, optional): Write histograms to data_dir.
            Defaults to False.
        plot (bool, optional): Save scatterplot for linear outcomes.
            Defaults to True.
        neptune_run (:class:`neptune.Run`, optional): Neptune run in which to
            log results. Defaults to None.
    """
    if categorical_reduce not in ('raw', 'onehot'):
        raise ValueError(
            f"Unrecognized reduction strategy {categorical_reduce}; "
            "must be either 'raw' or 'onehot'."
        )
    label_end = "" if label == '' else f"_{label}"
    label_start = "" if label == '' else f"{label}_"
    df['patient'] = df['slide'].map(patients)

    # Verify patient outcomes are consistent if multiples slides are present
    # for each patient
    #patient_error = False
    #for slide in labels:
    #    patient = patients[slide]
    #    if y_true_slide[slide] != y_true_patient[patient]:
    #        log.error(
    #            "Data integrity failure; patient assigned to multiple "
    #            "slides with different outcomes"
    #        )
    #        patient_error = True
    # if not len(outcome_names) == len(set(outcome_names)):
    #    raise ValueError("Duplicate outcome names found; all must be unique.")

    # Function to determine which predictions should be exported to CSV
    def should_save_predictions(grp):
        return (
            save_predictions is True
            or (isinstance(save_predictions, str) and save_predictions == grp)
            or (isinstance(save_predictions, list) and grp in save_predictions)
        )
    metric_args = SimpleNamespace(
        label_start=label_start,
        label_end=label_end,
        should_save_predictions=should_save_predictions,
        data_dir=data_dir,
        r_squared={'tile': None, 'slide': None, 'patient': None},
        c_index={'tile': None, 'slide': None, 'patient': None},
        auc={'tile': {}, 'slide': {}, 'patient': {}},
        ap={'tile': {}, 'slide': {}, 'patient': {}},
        plot=plot,
        histogram=histogram,
        neptune_run=neptune_run
    )

    # Detect the number of outcomes and confirm that the number of outcomes
    # match the provided outcome names
    if model_type == 'linear':
        n_outcomes = len([c for c in df.columns if c.startswith('out0-y_pred')])
    else:
        n_outcomes = len([c for c in df.columns if c.endswith('-y_pred0')])
    if not outcome_names:
        outcome_names = [f"Outcome {i}" for i in range(n_outcomes)]
    elif len(outcome_names) != n_outcomes:
        raise errors.StatsError(
            f"Number of outcome names {len(outcome_names)} does not "
            f"match y_true {n_outcomes}"
        )

    if model_type == 'categorical':
        # Update dataframe column names with outcome names
        outcome_cols_to_replace = {}
        for oi, outcome in enumerate(outcome_names):
            outcome_cols_to_replace.update({
                c: c.replace(f'out{oi}', outcome)
                for c in df.columns
                if c.startswith(f'out{oi}-')
            })
        df = df.rename(columns=outcome_cols_to_replace)

        # Perform analysis separately for each outcome column
        for oi, outcome in enumerate(outcome_names):
            outcome_cols = [c for c in df.columns if c.startswith(f'{outcome}-')]
            metric_args.df = df[['slide', 'patient'] + outcome_cols].rename(
                columns={
                    orig_col: orig_col.replace(f'{outcome}-', '', 1)
                    for orig_col in outcome_cols
                }
            )
            metric_args.cat_reduce = categorical_reduce
            log.info(f"Validation metrics for outcome {col.green(outcome)}:")
            _categorical_metrics(metric_args, outcome)

    elif model_type == 'linear':
        outcome_cols_to_replace = {}
        for oi, outcome in enumerate(outcome_names):
            outcome_cols_to_replace.update({
                c: f'{outcome}-y_pred'
                for c in df.columns
                if c.startswith('out0-y_pred') and c.endswith(str(oi))
            })
            outcome_cols_to_replace.update({
                c: f'{outcome}-y_true'
                for c in df.columns
                if c.startswith('out0-y_true') and c.endswith(str(oi))
            })
        df = df.rename(columns=outcome_cols_to_replace)

        # Calculate metrics
        metric_args.df = df
        metric_args.outcome_names = outcome_names
        _linear_metrics(metric_args)

    elif model_type == 'cph':
        assert len(outcome_names) == 1
        df = df.rename(columns={
            'out0-y_pred0': 'time-y_pred',
            'out0-y_pred1': 'event-y_true',
            'out0-y_true0': 'time-y_true',

        })
        # Calculate metrics
        metric_args.df = df
        _cph_metrics(metric_args)

    if should_save_predictions('tile'):
        df[[c for c in df.columns if c != 'patient']].to_csv(
            os.path.join(data_dir, f"tile_predictions{label_end}.csv"),
            index=False
        )
        log.debug(f"Predictions saved to {col.green(data_dir)}")

    combined_metrics = {
        'auc': metric_args.auc,
        'ap': metric_args.ap,
        'r_squared': metric_args.r_squared,
        'c_index': metric_args.c_index
    }
    return combined_metrics


def predict_from_torch(
    model: "torch.nn.Module",
    dataset: "torch.utils.data.DataLoader",
    model_type: str,
    pred_args: SimpleNamespace,
    uq_n: int = 30,
) -> pd.core.frame.DataFrame:
    """Generates predictions (y_true, y_pred, tile_to_slide) from
    a given PyTorch model and dataset.

    Args:
        model (torch.nn.Module): PyTorch model.
        dataset (torch.utils.data.DatatLoader): PyTorch dataloader.
        pred_args (namespace): Namespace containing slide_input,
            update_corrects, and update_loss functions.
        model_type (str, optional): 'categorical', 'linear', or 'cph'.
            If multiple linear outcomes are present, y_true is stacked into
            a single vector for each image. Defaults to 'categorical'.

    Returns:
        y_pred, y_std, tile_to_slides
    """
    import torch

    from slideflow.model.torch_utils import get_uq_predictions

    # Get predictions and performance metrics
    log.debug("Generating predictions from torch model")
    y_pred, tile_to_slides = [], []
    y_std = [] if pred_args.uq else None  # type: ignore
    num_outcomes = 0
    model.eval()
    device = torch.device('cuda:0')
    pb = tqdm(
        desc='Predicting...',
        total=dataset.num_tiles,  # type: ignore
        ncols=80,
        unit='img',
        leave=False
    )
    for img, yt, slide in dataset:  # TODO: support not needing to supply yt
        img = img.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # Slide-level features
                if pred_args.num_slide_features:
                    slide_inp = torch.tensor([
                        pred_args.slide_input[s] for s in slide
                    ])
                    inp = (img, slide_inp.to(device))
                else:
                    inp = (img,)  # type: ignore
                if pred_args.uq:
                    res, yp_std, num_outcomes = get_uq_predictions(
                        inp, model, num_outcomes, uq_n
                    )
                    if isinstance(yp_std, list):
                        yp_std = [y.cpu().numpy().copy() for y in yp_std]
                    else:
                        yp_std = yp_std.cpu().numpy().copy()
                    y_std += [yp_std]  # type: ignore
                else:
                    res = model(*inp)
                if isinstance(res, list):
                    res = [r.cpu().numpy().copy() for r in res]
                else:
                    res = res.cpu().numpy().copy()
                y_pred += [res]
        tile_to_slides += slide
        pb.update(img.shape[0])

    # Concatenate predictions for each outcome
    if type(y_pred[0]) == list:
        y_pred = [np.concatenate(yp) for yp in zip(*y_pred)]
        if pred_args.uq:
            y_std = [np.concatenate(ys) for ys in zip(*y_std)]  # type: ignore
    else:
        y_pred = [np.concatenate(y_pred)]
        if pred_args.uq:
            y_std = [np.concatenate(y_std)]

    # We will need to enforce softmax encoding for tile-level statistics.
    if model_type == 'categorical':
        y_pred = [softmax(yp, axis=1) for yp in y_pred]

    # Create pandas DataFrame from arrays
    df = df_from_pred(None, y_pred, y_std, tile_to_slides)

    log.debug("Prediction complete.")
    return df


def eval_from_torch(
    model: "torch.nn.Module",
    dataset: "torch.utils.data.DataLoader",
    model_type: str,
    pred_args: SimpleNamespace,
    uq_n: int = 30,
) -> Tuple[pd.core.frame.DataFrame, float, float]:
    """Generates predictions (y_true, y_pred, tile_to_slide) from
    a given PyTorch model and dataset.

    Args:
        model (str): Path to PyTorch model.
        dataset (tf.data.Dataset): PyTorch dataloader.
        pred_args (namespace): Namespace containing slide_input,
            update_corrects, and update_loss functions.
        model_type (str, optional): 'categorical', 'linear', or 'cph'. If
            multiple linear outcomes are present, y_true is stacked into a
            single vector for each image. Defaults to 'categorical'.

    Returns:
        y_true, y_pred, y_std, tile_to_slides, accuracy, loss
    """

    import torch

    from slideflow.model.torch_utils import get_uq_predictions
    y_true, y_pred, tile_to_slides = [], [], []
    y_std = [] if pred_args.uq else None  # type: ignore
    corrects = pred_args.running_corrects
    losses = 0
    total = 0
    num_outcomes = 0

    log.debug("Evaluating torch model")

    model.eval()
    device = torch.device('cuda:0')
    pb = tqdm(
        desc='Evaluating...',
        total=dataset.num_tiles,  # type: ignore
        ncols=80,
        unit='img',
        leave=False
    )
    for img, yt, slide in dataset:
        img = img.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # Slide-level features
                if pred_args.num_slide_features:
                    slide_inp = torch.tensor([
                        pred_args.slide_input[s] for s in slide
                    ])
                    inp = (img, slide_inp.to(device))
                else:
                    inp = (img,)  # type: ignore
                if pred_args.uq:
                    res, yp_std, num_outcomes = get_uq_predictions(
                        inp, model, num_outcomes, uq_n
                    )
                    if isinstance(yp_std, list):
                        yp_std = [y.cpu().numpy().copy() for y in yp_std]
                    else:
                        yp_std = yp_std.cpu().numpy().copy()
                    y_std += [yp_std]  # type: ignore
                else:
                    res = model(*inp)
                corrects = pred_args.update_corrects(res, yt, corrects)
                losses = pred_args.update_loss(res, yt, losses, img.size(0))
                if isinstance(res, list):
                    res = [r.cpu().numpy().copy() for r in res]
                else:
                    res = res.cpu().numpy().copy()
                y_pred += [res]
        if type(yt) == dict:
            y_true += [[yt[f'out-{o}'] for o in range(len(yt))]]
        else:
            yt = yt.detach().numpy().copy()
            y_true += [yt]
        tile_to_slides += slide
        total += img.shape[0]
        pb.update(img.shape[0])


    # Concatenate predictions for each outcome.
    if type(y_pred[0]) == list:
        y_pred = [np.concatenate(yp) for yp in zip(*y_pred)]
        if pred_args.uq:
            y_std = [np.concatenate(ys) for ys in zip(*y_std)]  # type: ignore
    else:
        y_pred = [np.concatenate(y_pred)]
        if pred_args.uq:
            y_std = [np.concatenate(y_std)]

    # Concatenate y_true for each outcome
    if type(y_true[0]) == list:
        y_true = [np.concatenate(yt) for yt in zip(*y_true)]

        # Merge multiple linear outcomes into a single vector
        if model_type == 'linear':
            y_true = [np.stack(y_true, axis=1)]  # type: ignore
    else:
        y_true = [np.concatenate(y_true)]

    # We will need to enforce softmax encoding for tile-level statistics.
    if model_type == 'categorical':
        y_pred = [softmax(yp, axis=1) for yp in y_pred]

    # Calculate final accuracy and loss
    loss = losses / total
    if isinstance(corrects, dict):
        acc = {k: v.cpu().numpy()/total for k, v in corrects.items()}
    elif isinstance(corrects, (int, float)):
        acc = corrects / total  # type: ignore
    else:
        acc = corrects.cpu().numpy() / total
    if log.getEffectiveLevel() <= 20:
        sf.util.clear_console()

    # Create pandas DataFrame from arrays
    df = df_from_pred(y_true, y_pred, y_std, tile_to_slides)

    log.debug("Evaluation complete.")
    return df, acc, loss  # type: ignore


def predict_from_tensorflow(
    model: "tf.keras.Model",
    dataset: "tf.data.Dataset",
    model_type: str,
    pred_args: SimpleNamespace,
    num_tiles: int = 0,
    uq_n: int = 30
) -> pd.core.frame.DataFrame:
    """Generates predictions (y_true, y_pred, tile_to_slide) from a given
    Tensorflow model and dataset.

    Args:
        model (str): Path to Tensorflow model.
        dataset (tf.data.Dataset): Tensorflow dataset.
        model_type (str, optional): 'categorical', 'linear', or 'cph'.
            Will not attempt to calculate accuracy for non-categorical models.
            Defaults to 'categorical'.
        pred_args (namespace): Namespace containing the property `loss`, loss
            function used to calculate loss.
        num_tiles (int, optional): Used for progress bar. Defaults to 0.
        uq_n (int, optional): Number of per-tile inferences to perform is
            calculating uncertainty via dropout.
        evaluate (bool, optional): Calculate and return accuracy and loss.
            Dataset must also return y_true.

    Returns:
        y_true, y_pred, tile_to_slides, accuracy, loss
    """
    import tensorflow as tf

    from slideflow.model.tensorflow_utils import get_uq_predictions

    @tf.function
    def get_predictions(img, training=False):
        return model(img, training=training)

    y_pred, tile_to_slides = [], []
    y_std = [] if pred_args.uq else None  # type: ignore
    num_vals, num_batches, num_outcomes = 0, 0, 0

    pb = tqdm(total=num_tiles, desc='Predicting...', ncols=80, leave=False)
    for img, yt, slide in dataset:  # TODO: support not needing to supply yt
        pb.update(slide.shape[0])
        tile_to_slides += [_bytes.decode('utf-8') for _bytes in slide.numpy()]
        num_vals += slide.shape[0]
        num_batches += 1
        if pred_args.uq:
            yp_mean, yp_std, num_outcomes = get_uq_predictions(
                img, get_predictions, num_outcomes, uq_n
            )
            y_pred += [yp_mean]
            y_std += [yp_std]  # type: ignore
        else:
            yp = get_predictions(img, training=False)
            y_pred += [yp]
    pb.close()

    if type(y_pred[0]) == list:
        # Concatenate predictions for each outcome
        y_pred = [np.concatenate(yp) for yp in zip(*y_pred)]
        if pred_args.uq:
            y_std = [np.concatenate(ys) for ys in zip(*y_std)]  # type: ignore
    else:
        y_pred = [np.concatenate(y_pred)]
        if pred_args.uq:
            y_std = [np.concatenate(y_std)]

    # Create pandas DataFrame from arrays
    df = df_from_pred(None, y_pred, y_std, tile_to_slides)

    log.debug("Prediction complete.")
    return df


def eval_from_tensorflow(
    model: "tf.keras.Model",
    dataset: "tf.data.Dataset",
    model_type: str,
    pred_args: SimpleNamespace,
    num_tiles: int = 0,
    uq_n: int = 30
) -> Tuple[pd.core.frame.DataFrame, float, float]:
    """Generates predictions (y_true, y_pred, tile_to_slide) from a given
    Tensorflow model and dataset.

    Args:
        model (str): Path to Tensorflow model.
        dataset (tf.data.Dataset): Tensorflow dataset.
        model_type (str, optional): 'categorical', 'linear', or 'cph'.
            Will not attempt to calculate accuracy for non-categorical models.
            Defaults to 'categorical'.
        pred_args (namespace): Namespace containing the property `loss`, loss
            function used to calculate loss.
        num_tiles (int, optional): Used for progress bar. Defaults to 0.
        uq_n (int, optional): Number of per-tile inferences to perform is
            calculating uncertainty via dropout.
        evaluate (bool, optional): Calculate and return accuracy and loss.
            Dataset must also return y_true.

    Returns:
        y_true, y_pred, tile_to_slides, accuracy, loss
    """

    import tensorflow as tf

    from slideflow.model.tensorflow_utils import get_uq_predictions

    @tf.function
    def get_predictions(img, training=False):
        return model(img, training=training)

    y_true, y_pred, tile_to_slides = [], [], []
    y_std = [] if pred_args.uq else None  # type: ignore
    num_vals, num_batches, num_outcomes, running_loss = 0, 0, 0, 0
    is_cat = (model_type == 'categorical')
    if not is_cat:
        acc = None

    pb = tqdm(total=num_tiles, desc='Evaluating...', ncols=80, leave=False)
    for img, yt, slide in dataset:
        pb.update(slide.shape[0])
        tile_to_slides += [_byte.decode('utf-8') for _byte in slide.numpy()]
        num_vals += slide.shape[0]
        num_batches += 1

        if pred_args.uq:
            yp, yp_std, num_outcomes = get_uq_predictions(
                img, get_predictions, num_outcomes, uq_n
            )
            y_pred += [yp]
            y_std += [yp_std]  # type: ignore
        else:
            yp = get_predictions(img, training=False)
            y_pred += [yp]
        if type(yt) == dict:
            y_true += [[yt[f'out-{o}'].numpy() for o in range(len(yt))]]
            yt = [yt[f'out-{o}'] for o in range(len(yt))]
        else:
            y_true += [yt.numpy()]
        loss = pred_args.loss(yt, yp)
        running_loss += tf.math.reduce_sum(loss).numpy() * slide.shape[0]
    pb.close()

    if type(y_pred[0]) == list:
        # Concatenate predictions for each outcome
        y_pred = [np.concatenate(yp) for yp in zip(*y_pred)]
        if pred_args.uq:
            y_std = [np.concatenate(ys) for ys in zip(*y_std)]  # type: ignore
    else:
        y_pred = [np.concatenate(y_pred)]
        if pred_args.uq:
            y_std = [np.concatenate(y_std)]

    if type(y_true[0]) == list:
        # Concatenate y_true for each outcome
        y_true = [np.concatenate(yt) for yt in zip(*y_true)]
        if is_cat:
            acc = [
                np.sum(y_true[i] == np.argmax(y_pred[i], axis=1)) / num_vals
                for i in range(len(y_true))
            ]
    else:
        y_true = [np.concatenate(y_true)]
        if is_cat:
            acc = np.sum(y_true == np.argmax(y_pred, axis=1)) / num_vals

    # Create pandas DataFrame from arrays
    df = df_from_pred(y_true, y_pred, y_std, tile_to_slides)

    # Note that Keras loss during training includes regularization losses,
    # so this loss will not match validation loss calculated during training
    loss = running_loss / num_vals
    log.debug("Evaluation complete.")
    return df, acc, loss  # type: ignore


def predict_from_dataset(
    model: Union["tf.keras.Model", "torch.nn.Module"],
    dataset: Union["tf.data.Dataset", "torch.utils.data.DataLoader"],
    model_type: str,
    pred_args: SimpleNamespace,
    num_tiles: int = 0,
    uq_n: int = 30
) -> pd.core.frame.DataFrame:
    """Generates predictions (y_pred, tile_to_slide) from model and dataset.

    Args:
        model (str): Path to PyTorch model.
        dataset (tf.data.Dataset): PyTorch dataloader.
        pred_args (namespace): Namespace containing slide_input,
            update_corrects, and update_loss functions.
        model_type (str, optional): 'categorical', 'linear', or 'cph'.
            If multiple linear outcomes are present, y_true is stacked into a
            single vector for each image. Defaults to 'categorical'.
        num_tiles (int, optional): Used for progress bar with Tensorflow.
            Defaults to 0.

    Returns:
        y_pred, tile_to_slides
    """
    if sf.backend() == 'tensorflow':
        return predict_from_tensorflow(
            model,
            dataset,
            model_type,
            pred_args,
            num_tiles=num_tiles,
            uq_n=uq_n
        )
    else:
        return predict_from_torch(
            model,
            dataset,
            model_type,
            pred_args,
            uq_n=uq_n
        )


def eval_from_dataset(
    model: Union["tf.keras.Model", "torch.nn.Module"],
    dataset: Union["tf.data.Dataset", "torch.utils.data.DataLoader"],
    model_type: str,
    pred_args: SimpleNamespace,
    num_tiles: int = 0,
    uq_n: int = 30
) -> Tuple[pd.core.frame.DataFrame, float, float]:
    """Generates predictions (y_true, y_pred, tile_to_slide) and accuracy/loss
    from a given model and dataset.

    Args:
        model (str): Path to PyTorch model.
        dataset (tf.data.Dataset): PyTorch dataloader.
        pred_args (namespace): Namespace containing slide_input,
            update_corrects, and update_loss functions.
        model_type (str, optional): 'categorical', 'linear', or 'cph'.
            If multiple linear outcomes are present, y_true is stacked into a
            single vector for each image. Defaults to 'categorical'.
        num_tiles (int, optional): Used for progress bar with Tensorflow.
            Defaults to 0.

    Returns:
        y_true, y_pred, tile_to_slides, accuracy, loss
    """

    if sf.backend() == 'tensorflow':
        return eval_from_tensorflow(
            model,
            dataset,
            model_type,
            pred_args,
            num_tiles=num_tiles,
            uq_n=uq_n
        )
    else:
        return eval_from_torch(
            model,
            dataset,
            model_type,
            pred_args,
            uq_n=uq_n
        )


def metrics_from_dataset(
    model: Union["tf.keras.Model", "torch.nn.Module"],
    model_type: str,
    patients: Dict[str, str],
    dataset: Union["tf.data.Dataset", "torch.utils.data.DataLoader"],
    pred_args: SimpleNamespace,
    num_tiles: int = 0,
    **kwargs
) -> Tuple[Dict, float, float]:

    """Evaluate performance of a given model on a given TFRecord dataset,
    generating a variety of statistical outcomes and graphs.

    Args:
        model (tf.keras.Model or torch.nn.Module): Keras/Torch model to eval.
        model_type (str): 'categorical', 'linear', or 'cph'.
        labels (dict): Dictionary mapping slidenames to outcomes.
        patients (dict): Dictionary mapping slidenames to patients.
        dataset (tf.data.Dataset or torch.utils.data.DataLoader): Dataset.
        num_tiles (int, optional): Number of total tiles expected in dataset.
            Used for progress bar. Defaults to 0.
        neptune_run (:class:`neptune.Run`, optional): Neptune run in which to
            log results. Defaults to None.
        pred_args (namespace, optional): Additional arguments to tensorflow and
            torch backends.

    Keyword args:
        categorical_reduce (str, optional): Reduction strategy for calculating
            slide-level and patient-level predictions for categorical outcomes.
            Either 'raw' or 'onehot'. If 'raw', will reduce with average of
            each logit across tiles. If 'onehot', will convert tile predictions
            into onehot encoding via `np.argmax`, then reduce by averaging
            these onehot values. Defaults to 'raw'.
        label (str, optional): Label prefix/suffix for saving.
            Defaults to None.
        outcome_names (list, optional): List of str, names for outcomes.
            Defaults to None.
        data_dir (str): Path to data directory for saving.
            Defaults to empty string (current directory).
        histogram (bool, optional): Write histograms to data_dir.
            Defaults to False.
        save_predictions (bool, optional): Save tile, slide, and patient-level
            predictions to CSV. Defaults to True.
        neptune_run (:class:`neptune.Run`, optional): Neptune run in which to
            log results. Defaults to None.

    Returns:
        metrics [dict], accuracy [float], loss [float]
    """

    df, acc, loss = eval_from_dataset(  # yt, yp, y_std, t_s
        model,
        dataset,
        model_type,
        pred_args,
        num_tiles=num_tiles
    )
    before_metrics = time.time()
    metrics = metrics_from_pred(
        df=df,
        patients=patients,
        model_type=model_type,
        plot=True,
        **kwargs
    )
    after_metrics = time.time()
    log.debug(f'Metrics generated ({after_metrics - before_metrics:.2f} s)')
    return metrics, acc, loss
