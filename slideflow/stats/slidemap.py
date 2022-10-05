from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import pickle
import numpy as np
import pandas as pd
import slideflow as sf
from os.path import join
from mpl_toolkits.mplot3d import Axes3D
from pandas.core.frame import DataFrame
from sklearn.cluster import KMeans
from slideflow import errors
from slideflow.stats.stats_utils import calculate_centroid, normalize_layout
from slideflow.util import log

if TYPE_CHECKING:
    import umap
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from slideflow.model import DatasetFeatures


class SlideMap:
    """Two-dimensional slide map for visualization & backend for mosaic maps.

    Slides are mapped in 2D either explicitly with pre-specified coordinates,
    or with dimensionality reduction from post-convolutional layer weights,
    provided from :class:`slideflow.DatasetFeatures`.
    """

    def __init__(
        self,
        slides: Optional[List[str]] = None,
        cache: Optional[str] = None,
        parametric_umap: bool = False
    ) -> None:
        """Backend for mapping slides into two dimensional space. Can use a
        DatasetFeatures object to map slides according to UMAP of features, or
        map according to pre-specified coordinates.

        Args:
            slides (list(str)): List of slide names
            cache (str, optional): Path to PKL file to cache activations.
                Defaults to None (caching disabled).
        """
        if slides is None and cache is None:
            raise ValueError("Argument `slides` required if cache not provided")

        self.data = None  # type: DataFrame
        self.cache = cache
        if self.cache:
            if self.load_cache() and slides is None:
                slides = self.data.slide.unique()
            elif slides is None:
                raise ValueError(f"Unable to load from cache: {cache}.")
        self.slides = slides
        self.df = None  # type: Optional[DatasetFeatures]
        self.parametric_umap = parametric_umap
        self._umap_normalized_range = None
        self.map_meta = {}  # type: Dict[str, Any]


    @classmethod
    def from_precalculated(
        cls,
        x: Union[np.ndarray, List[int], str],
        y: Union[np.ndarray, List[int], str],
        slides: Union[np.ndarray, List[str], str],
        tfr_index: Union[np.ndarray, List[int], str],
        data: Optional[DataFrame] = None,
        cache: Optional[str] = None,
        parametric_umap: bool = False
    ) -> "SlideMap":
        """Initializes map from precalculated coordinates.

        Args:
            slides (list(str)): List of slide names.
            x (list(int)): X coordinates for each point on the map. Can either
                be a list of int, or the name of a column in the DataFrame
                provided to the argument 'data'.
            y (list(int)): Y coordinates for tfrecords. Can either
                be a list of int, or the name of a column in the DataFrame
                provided to the argument 'data'.
            slides (list(str)): Slide names for each point on the map. Can
                either be a list of str, or the name of a column in the
                DataFrame provided to the argument 'data'.
            tfr_index (list(int)): TFRecord indicies for each point on
                the map. Can either be a list of int, or the name of a column
                in the DataFrame provided to the argument 'data'.
            data (DataFrame, optional): Optional DataFrame which can be used
                to supply the 'x', 'y', 'slides', and 'tfr_index' data.
            cache (str, optional): Path to PKL file to cache coordinates.
                Defaults to None (caching disabled).
        """
        # Read and verify provided input
        cols = {'x': x, 'y': y, 'slides': slides, 'tfr_index': tfr_index}
        for col, col_val in cols.items():
            if isinstance(col_val, str) and data is None:
                raise ValueError(
                    f"Could not interpret input {col_val} for arg {col}. "
                    "Did you mean to supply a DataFrame via 'data'?")
            elif data is not None:
                if isinstance(col_val, str) and col_val not in data.columns:
                    raise ValueError(f"Could not find column {col_val}.")
                elif isinstance(col_val, str):
                    cols[col] = data[col_val].values
            else:
                cols[col] = col_val

        # Verify lengths of provided input
        if not all(len(cols[c]) == len(cols['x']) for c in cols):
            raise ValueError(
                "Length of x, y, slides, and tfr_index must all be equal."
            )

        obj_data = pd.DataFrame({
            'x': pd.Series(cols['x']),
            'y': pd.Series(cols['y']),
            'slide': pd.Series(cols['slides']),
            'tfr_index': pd.Series(cols['tfr_index'])
        })
        obj = cls(obj_data.slide.unique())
        obj.data = obj_data
        obj.cache = cache
        obj.parametric_umap = parametric_umap
        obj.save_cache()
        return obj

    @classmethod
    def from_features(
        cls,
        df: "DatasetFeatures",
        exclude_slides: Optional[List[str]] = None,
        recalculate: bool = False,
        map_slide: Optional[str] = None,
        cache: Optional[str] = None,
        parametric_umap: bool = False,
        umap_dim: int = 2,
        umap: Optional[Any] = None,
        **umap_kwargs: Any
    ) -> "SlideMap":
        """Initializes map from dataset features.

        Args:
            df (:class:`slideflow.DatasetFeatures`): DatasetFeatures.
            exclude_slides (list, optional): List of slides to exclude.
            recalculate (bool, optional):  Force recalculation of umap despite
                presence of cache.
            map_slide (str, optional): Either None, 'centroid', or 'average'.
                If None, will map all tiles from each slide. Defaults to None.
            cache (str, optional): Path to PKL file to cache coordinates.
                Defaults to None (caching disabled).
            umap_dim (int, optional): Number of dimensions for UMAP. Defaults
                to 2.
            umap (umap.UMAP, optional): Fit UMAP, to be used instead of fitting
                a new UMAP.
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
        obj.umap = umap  # type: ignore
        obj.parametric_umap = parametric_umap
        if map_slide:
            obj._calculate_from_slides(
                method=map_slide,
                recalculate=recalculate,
                **umap_kwargs
            )
        else:
            obj._calculate_from_tiles(
                recalculate=recalculate,
                dim=umap_dim,
                **umap_kwargs
            )
        return obj

    @property
    def x(self):
        """X coordinates of map."""
        return self.data.x.values

    @property
    def y(self):
        """Y coordinates of map."""
        return self.data.y.values

    def _calculate_from_tiles(
        self,
        recalculate: bool = False,
        **umap_kwargs: Any
    ) -> None:
        """Internal function to guide calculation of UMAP from final layer
        features / activations, as provided by DatasetFeatures.

        Args:
            recalculate (bool, optional): Recalculate of UMAP despite loading
                from cache. Defaults to False.

        Keyword Args:
            dim (int): Number of dimensions for UMAP. Defaults to 2.
            n_neighbors (int): Number of neighbors for UMAP. Defaults to 50.
            min_dist (float): Minimum distance for UMAP. Defaults to 0.1.
            metric (str): UMAP metric. Defaults to 'cosine'.
            **umap_kwargs (optional): Additional keyword arguments for the
                UMAP function.
        """
        assert self.df is not None
        if self.data is not None and not recalculate:
            log.info("Data loaded from cache, will not recalculate")

            # First, filter out slides not included in provided activations
            self.data = self.data.loc[self.data.slide.isin(self.df.slides)]

            # If UMAP already calculated, update predictions
            # if prediction filter is provided
            if self.df.logits:
                logits = [
                    self.df.logits[row.slide][row.tfr_index]
                    for row in self.data.itertuples()
                ]
            predictions = np.argmax(np.array(logits), axis=1)
            self.data['logits'] = pd.Series(logits)
            self.data['predictions'] = pd.Series([p for p in predictions])
            return

        # Calculate UMAP
        node_activations = np.concatenate([
            self.df.activations[slide] for slide in self.slides
        ])

        self.map_meta['num_features'] = self.df.num_features
        log.info("Calculating UMAP...")

        coordinates = self.umap_transform(node_activations, **umap_kwargs)

        # Assemble dataframe
        locations = np.concatenate([
            self.df.locations[slide] for slide in self.slides
        ])
        tfrecord_indices = np.concatenate([
            np.arange(self.df.locations[slide].shape[0])
            for slide in self.slides
        ])
        slides = np.array([
            slide
            for slide in self.slides
            for _ in range(self.df.locations[slide].shape[0])
        ])
        data_dict = {
            'slide': pd.Series(slides),
            'x': pd.Series(coordinates[:, 0]),
            'tfr_index': pd.Series(tfrecord_indices),
            'location': pd.Series([l for l in locations]).astype(object)
        }
        if self.df.logits:
            logits = np.concatenate([
                self.df.logits[slide] for slide in self.slides
            ])
            data_dict.update({
                'prediction': pd.Series(np.argmax(logits, axis=1)),
                'logits': pd.Series([l for l in logits]).astype(object),
            })
        if self.df.uq and self.df.uncertainty != {}:  # type: ignore
            uncertainty = np.concatenate([
                self.df.uncertainty[slide] for slide in self.slides
            ])
            data_dict.update({
                'uncertainty': pd.Series(
                    [u for u in uncertainty]
                ).astype(object)
            })
        if 'dim' not in umap_kwargs or umap_kwargs['dim'] > 1:
            data_dict.update({
                'y': pd.Series(coordinates[:, 1]),
            })
        self.data = pd.DataFrame(data_dict)
        self.save_cache()

    def _calculate_from_slides(
        self,
        method: str = 'centroid',
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
            recalculate (bool, optional): Recalculate of UMAP despite loading
                from cache. Defaults to False.

        Keyword Args:
            dim (int): Number of dimensions for UMAP. Defaults to 2.
            n_neighbors (int): Number of neighbors for UMAP. Defaults to 50.
            min_dist (float): Minimum distance for UMAP. Defaults to 0.1.
            metric (str): UMAP metric. Defaults to 'cosine'.
            **umap_kwargs (optional): Additional keyword arguments for the
                UMAP function.
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
                log.debug(f"No centroid for [green]{slide}[/]; skipping")
        if num_warned:
            log.warning(f"No centroid for {num_warned} slides.")
        if self.data is not None and not recalculate:
            log.info("Slide map loaded from cache.")
            log.debug("Filtering to include only provided tiles")

            def is_opt(row):
                return ((row['slide'] in opt_idx)
                        and (row['tfr_index'] == opt_idx[row['slide']]))

            self.data = self.data.loc[
                self.data.apply(lambda row : is_opt(row), axis=1)
            ]
        else:
            log.info(f"Calculating UMAP from slide-level {method}...")

            if method == 'centroid':
                umap_input = np.array([
                    centroid_activations[slide] for slide in self.slides
                ])
            elif method == 'average':
                umap_input = np.array([
                    np.mean(self.df.activations[slide], axis=0)
                    for slide in self.slides
                ])

            # Calculate UMAP
            coordinates = self.umap_transform(
                umap_input,
                **umap_kwargs
            )

            # Create dataframe
            locations = np.stack([
                self.df.locations[slide][opt_idx[slide]] for slide in self.slides
            ])
            data_dict = {
                'slide': pd.Series(self.slides),
                'x': pd.Series(coordinates[:, 0]),
                'tfr_index': pd.Series(opt_idx[slide] for slide in self.slides),
                'location': pd.Series([l for l in locations]).astype(object)
            }
            if self.df.logits:
                logits = np.stack([
                    self.df.logits[slide][opt_idx[slide]] for slide in self.slides
                ])
                data_dict.update({
                    'logits': pd.Series([l for l in logits]).astype(object),
                    'prediction': pd.Series(np.argmax(logits, axis=1)),
                })
            if self.df.uq and self.df.uncertainty != {}:  # type: ignore
                uncertainty = np.stack([
                    self.df.uncertainty[slide][opt_idx[slide]]
                    for slide in self.slides
                ])
                data_dict.update({
                    'uncertainty': pd.Series(
                        [u for u in uncertainty]
                    ).astype(object)
                })
            if 'dim' not in umap_kwargs or umap_kwargs['dim'] > 1:
                data_dict.update({
                    'y': pd.Series(coordinates[:, 1]),
                })
            self.data = pd.DataFrame(data_dict)
            self.save_cache()

    def activations(self) -> np.ndarray:
        """Return associated DatasetFeatures activations as a numpy array
        corresponding to the points on this SlideMap."""
        if self.df is None:
            raise ValueError(
                "No associated DatasetFeatures object for reading activations."
            )
        return np.array([
            self.df.activations[row.slide][row.tfr_index]
            for row in self.data.itertuples()
        ])

    def cluster(self, n_clusters: int) -> None:
        """Performs K-means clustering on data and adds to metadata labels.

        Clusters are saved to self.data['cluster']. Requires a DatasetFeatures
        backend.

        Args:
            n_clusters (int): Number of clusters for K means clustering.
        """

        if self.df is None:
            raise errors.SlideMapError(
                "Unable to cluster; no DatasetFeatures provided"
            )
        activations = [
            self.df.activations[row.slide][row.tfr_index]
            for row in self.data.itertuples()
        ]
        log.info(f"Calculating K-means clustering (n={n_clusters})")
        kmeans = KMeans(n_clusters=n_clusters).fit(activations)
        self.data['cluster'] = kmeans.labels_

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
            X = np.stack((self.data.x.values, self.data.y.values), axis=-1)
        elif method == 'features':
            X = self.activations()
        elif method == 'pca':
            log.info(f"Reducing dimensionality with PCA (dim={pca_dim})...")
            pca = PCA(n_components=pca_dim)
            features = self.activations()
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

        def num_category_matching(idx_list, idx):
            list_cat = np.array([
                slide_categories[self.data.loc[_i].slide] for _i in idx_list
            ])
            idx_cat = slide_categories[self.data.loc[idx].slide]
            return (list_cat == idx_cat).sum()

        log.info('Matching neighbors...')
        #TODO: accelerate this step with multiprocessing
        self.data['num_unique_neighbors'] = [
            len(self.data.loc[ind].slide.unique())
            for ind in indices
        ]
        if slide_categories:
            self.data['percent_matching_categories'] = [
                num_category_matching(ind, i) / len(ind)
                for i, ind in enumerate(indices)
            ]

    def filter(self, slides: List[str]) -> None:
        """Filters map to only show tiles from the given slides.

        Args:
            slides (list(str)): List of slide names.
        """

        self.data = self.data.loc[self.data.slide.isin(slides)]

    def umap_transform(
        self,
        array: np.ndarray,
        *,
        dim: int = 2,
        n_neighbors: int = 50,
        min_dist: float = 0.1,
        metric: str = 'cosine',
        **kwargs: Any
    ) -> np.ndarray:
        """Generates and returns a umap from a given array, using umap.UMAP

        Args:
            array (np.ndarray): Array to transform with UMAP dimensionality
                reduction.

        Keyword Args:
            dim (int, optional): Number of dimensions for UMAP. Defaults to 2.
            n_neighbors (int, optional): Number of neighbors for UMAP
                algorithm. Defaults to 50.
            min_dist (float, optional): Minimum distance argument for UMAP
                algorithm. Defaults to 0.1.
            metric (str, optional): Metric for UMAP algorithm. Defaults to
                'cosine'.
            **kwargs (optional): Additional keyword arguments for the
                UMAP function.
        """
        import umap  # Imported in this function due to long import time
        if not len(array):
            raise errors.StatsError("Unable to perform UMAP on empty array.")
        if self.umap is None:  # type: ignore
            fn = umap.UMAP if not self.parametric_umap else umap.ParametricUMAP
            self.umap = fn(
                n_components=dim,
                verbose=(sf.getLoggingLevel() <= 20),
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                **kwargs
            )
        layout = self.umap.fit_transform(array)  # type: ignore
        (normalized,
         self._umap_normalized_range,
         self._umap_normalized_clip) = normalize_layout(layout)
        return normalized

    def label_by_uncertainty(self, index: int = 0) -> None:
        """Labels each point with the tile-level uncertainty, if available.

        Args:
            index (int, optional): Uncertainty index. Defaults to 0.
        """
        if 'label' in self.data.columns:
            self.data.drop(columns='label')
        if self.df is None:
            raise errors.SlideMapError("DatasetFeatures not provided.")
        if not self.df.uq or self.df.uncertainty == {}:  # type: ignore
            raise errors.DatasetError(
                'Unable to label by uncertainty; UQ estimates not available.'
            )
        else:
            uq_labels = np.stack(self.data['uncertainty'].values)[:, index]
            self.data['label'] = uq_labels

    def label_by_logits(self, index: int) -> None:
        """Displays each point with label equal to the logits (linear from 0-1)

        Args:
            index (int): Logit index.
        """
        if 'label' in self.data.columns:
            self.data.drop(columns='label')
        self.data['label'] = np.stack(self.data['logits'].values)[:, index]

    def label_by_slide(self, slide_labels: Optional[Dict] = None) -> None:
        """Displays each point as the name of the corresponding slide.
            If slide_labels is provided, will use this dict to label slides.

        Args:
            slide_labels (dict, optional): Dict mapping slide names to labels.
        """
        if 'label' in self.data.columns:
            self.data.drop(columns='label')
        if slide_labels:
            self.data['label'] = self.data.slide.map(slide_labels)
        else:
            self.data['label'] = self.data.slide.values

    def label(self, meta: str, translate: Optional[Dict] = None) -> None:
        """Displays each point labeled by tile metadata (e.g. 'prediction')

        Args:
            meta (str): Data column from which to assign labels.
            translate (dict, optional): If provided, will translate the
                read metadata through this dictionary.
        """
        if 'label' in self.data.columns:
            self.data.drop(columns='label')
        self.data['label'] = self.data[meta].values
        if translate:
            self.data['label'] = self.data['label'].map(translate)

    def plot(
        self,
        subsample: Optional[int] = None,
        title: Optional[str] = None,
        cmap: Optional[Dict] = None,
        xlim: Tuple[float, float] = (-0.05, 1.05),
        ylim: Tuple[float, float] = (-0.05, 1.05),
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend: Optional[str] = None,
        ax: Optional["Axes"] = None,
        categorical: Union[str, bool] = 'auto',
        **scatter_kwargs: Any
    ) -> None:
        """Plots calculated map.

        Args:
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
            ax (matplotlib.axes.Axes, optional): Figure axis. If not supplied,
                will prepare a new figure axis.
            categorical (str, optional): Specify whether labels are categorical.
                Determines the colormap.  Defaults to 'auto' (will attempt to
                automatically determine from the labels).
            **scatter_kwargs (optional): Additional keyword arguments to the
                seaborn scatterplot function.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Make plot
        if ax is None:
            fig = plt.figure(figsize=(6, 4.5))
            ax = fig.add_subplot(111)

        # Subsampling
        if subsample:
            plot_df = self.data.sample(subsample)
        else:
            plot_df = self.data

        x = plot_df.x
        y = plot_df.y

        if 'label' in self.data.columns:
            labels = plot_df.label

            # Check for categorical labels
            if (categorical is True
               or not pd.to_numeric(labels, errors='coerce').notnull().all()):

                log.debug("Interpreting labels as categorical")
                scatter_kwargs.update(
                    dict(hue=labels.astype('category'))
                )
                unique = list(labels.unique())
                unique.sort()
                if len(unique) >= 12:
                    sns_pal = sns.color_palette("Paired", len(unique))
                else:
                    sns_pal = sns.color_palette('hls', len(unique))
                if cmap is None:
                    cmap = {unique[i]: sns_pal[i] for i in range(len(unique))}
            else:
                log.debug("Interpreting labels as continuous")
                scatter_kwargs.update(dict(hue=labels))

        umap_2d = sns.scatterplot(
            x=x,
            y=y,
            palette=cmap,
            ax=ax,
            **scatter_kwargs
        )
        ax.set_ylim(*((None, None) if not ylim else ylim))
        ax.set_xlim(*((None, None) if not xlim else xlim))
        if 'hue' in scatter_kwargs:
            ax.legend(
                loc='center right',
                ncol=1,
                title=legend
            )
        umap_2d.set(xlabel=xlabel, ylabel=ylabel)
        if title:
            ax.set_title(title)

    def plot_3d(
        self,
        z: Optional[np.ndarray] = None,
        feature: Optional[int] = None,
        subsample: Optional[int] = None,
        fig: Optional["Figure"] = None,
    ) -> None:
        """Saves a plot of a 3D umap, with the 3rd dimension representing
        values provided by argument "z".

        Args:
            z (list, optional): Values for z axis. Must supply z or feature.
                Defaults to None.
            feature (int, optional): Int, feature to plot on 3rd axis.
                Must supply z or feature. Defaults to None.
            subsample (int, optional): Subsample to only include this many
                tiles on plot. Defaults to None.
            fig (matplotlib.figure.Figure, optional): Figure. If not supplied,
                will prepare a new figure.
        """
        import matplotlib.pyplot as plt

        if fig is None:
            fig = plt.figure()

        title = f"UMAP with feature {feature} focus"
        if self.df is None:
            raise errors.SlideMapError("DatasetFeatures not provided.")
        if (z is None) and (feature is None):
            raise errors.SlideMapError("Must supply either 'z' or 'feature'.")

        # Subsampling
        if subsample:
            plot_df = self.data.sample(subsample)
        else:
            plot_df = self.data

        # Get feature activations for 3rd dimension
        if z is None:
            z = np.array([
                self.df.activations[row.slide][row.tfr_index][feature]
                for row in plot_df.itertuples()
            ])

        # Plot tiles on a 3D coordinate space with 2 coordinates from UMAP
        # and 3rd from the value of the excluded feature
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        scatter_kw = dict(c=z, cmap='viridis', linewidth=0.5, edgecolor="k")
        ax.scatter(plot_df.x, plot_df.y, z, **scatter_kw)
        ax.set_title(title)

    def save(
        self,
        filename: str,
        dpi: int = 300,
        **kwargs
    ):
        """Save plot of slide map.

        Args:
            filename (str): File path to save the image.
            dpi (int, optional): DPI for final image. Defaults to 300.

        Keyword args:
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
            **scatter_kwargs (optional): Additional keyword arguments to the
                seaborn scatterplot function.

        """
        import matplotlib.pyplot as plt

        self.plot(**kwargs)
        plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        log.info(f"Saved 2D UMAP to [green]{filename}")

    def save_3d(
        self,
        filename: str,
        dpi: int = 300,
        **kwargs

    ):
        """Save 3D plot of slide map.

        Args:
            filename (str): _description_
            dpi (int, optional): _description_. Defaults to 300.

        Keyword args:
            z (list, optional): Values for z axis. Must supply z or feature.
                Defaults to None.
            feature (int, optional): Int, feature to plot on 3rd axis.
                Must supply z or feature. Defaults to None.
            subsample (int, optional): Subsample to only include this many
                tiles on plot. Defaults to None.

        """
        import matplotlib.pyplot as plt

        self.plot_3d(**kwargs)
        plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        log.info(f"Saved 3D UMAP to [green]{filename}")

    def save_cache(self, path: Optional[str] = None) -> None:
        """Save cache of coordinates to PKL file.

        Args:
            path (str, optional): Save cache to this location. If None,
                will use `self.cache`.
        """
        if path is None:
            path = self.cache
        if path:
            self.data.to_parquet(path)
            log.info(f"Wrote slide map cache to [green]{path}")

    def save_umap(self, path: str) -> None:
        """Save cache of UMAP to PKL file.

        Args:
            path (str, optional): Save cache to this location. If None,
                will use `self.cache`.
        """
        if self.parametric_umap:
            self.umap.save(path)
        else:
            with open(path, 'wb') as f:
                pickle.dump(self.umap, f)
                log.info(f"Wrote UMAP cache to [green]{path}")

    def save_encoder(self, path: str) -> None:
        """Save Parametric UMAP encoder."""
        if not self.parametric_umap:
            raise ValueError("SlideMap not built with Parametric UMAP.")
        self.umap.encoder.save(join(path, 'encoder'))
        self.save_cache(join(path, 'slidemap.parquet'))
        np.savez(
            join(path, 'range_clip.npz'),
            range=self._umap_normalized_range,
            clip=self._umap_normalized_clip)

    def load_umap(self, path: str) -> "umap.UMAP":
        if self.parametric_umap:
            from umap.parametric_umap import load_ParametricUMAP
            self.umap = load_ParametricUMAP(path)
        else:
            with open(path, 'rb') as f:
                self.umap = pickle.load(f)
                log.info(f"Loaded UMAP cache from [green]{path}")

    def load_cache(self, path: Optional[str] = None) -> bool:
        """Load coordinates from PKL cache.

        Args:
            path (str, optional): Load cache from this location. If None,
                will use `self.cache`.

        Returns:
            bool: If successfully loaded from cache.
        """
        if path is None:
            path = self.cache
        if path is None:
            raise errors.SlideMapError("No cache set or given.")
        try:
            self.data = pd.read_parquet(path)
            log.info(f"Loaded slide map cache from [green]{path}")
            return True
        except FileNotFoundError:
            log.info(f"No slide map cache found at [green]{path}")
        except Exception:
            log.error(
                f"Error loading slide map cache at [green]{path}[/], "
                "ensure it is a valid parquet-format dataframe."
            )
        return False
