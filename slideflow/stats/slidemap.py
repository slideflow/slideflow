from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import os
import pickle
import numpy as np
import pandas as pd
import slideflow as sf
import warnings
from os.path import join, exists, isdir
from pandas.core.frame import DataFrame
from sklearn.cluster import KMeans
from slideflow import errors
from slideflow.stats import stats_utils
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
        *,
        parametric_umap: bool = False
    ) -> None:
        """Backend for mapping slides into two dimensional space. Can use a
        DatasetFeatures object to map slides according to UMAP of features, or
        map according to pre-specified coordinates.

        Can be initialized with three methods: from precalculated X/Y
        coordinates, from a DatasetFeatures object, or from a saved map.

        Examples
            Build a SlideMap from a DatasetFeatures object

                .. code-block:: python

                    dts_ftrs = sf.DatasetFeatures(model, dataset)
                    slidemap = sf.SlideMap.from_features(dts_ftrs)

            Build a SlideMap from prespecified coordinates

                .. code-block:: python

                    x = np.array(...)
                    y = np.array(...)
                    slides = ['slide1', 'slide1', 'slide5', ...]
                    slidemap = sf.SlideMap.from_xy(
                        x=x, y=y, slides=slides
                    )

            Load a saved SlideMap

                .. code-block:: python

                    slidemap = sf.SlideMap.load('map.parquet')

        Args:
            slides (list(str)): List of slide names
        """
        assert isinstance(parametric_umap, bool), "Expected <bool> for argument 'parametric_umap'"
        self.data = None    # type: DataFrame
        self.ftrs = None    # type: Optional[DatasetFeatures]
        self.slides = None  # type: List[str]
        self.tfrecords = None  # type: List[str]
        self.parametric_umap = parametric_umap
        self._umap_normalized_range = None
        self.map_meta = {}  # type: Dict[str, Any]

    @classmethod
    def load(cls, path: str):
        """Load a previously saved SlideMap (UMAP and coordinates).

        Loads a ``SlideMap`` previously saved with ``SlideMap.save()``.

        Expects a directory with ``slidemap.parquet``, ``range_clip.npz``,
        and either ``umap.pkl`` (non-parametric models) or a folder named
        ``parametric_model``.

        Examples
            Save a SlideMap, then load it.

                .. code-block:: python

                    slidemap.save('/directory/')
                    new_slidemap = sf.SlideMap.load('/directory/')

        Args:
            path (str): Directory from which to load a previously saved UMAP.

        """
        log.debug(f"Loading SlideMap from {path}")
        obj = cls()
        if isdir(path):
            # Load coordinates
            if exists(join(path, 'slidemap.parquet')):
                obj.load_coordinates(join(path, 'slidemap.parquet'))
            else:
                log.warn("Could not find slidemap.parquet; no data loaded.")
            # Load UMAP
            if exists(join(path, 'parametric_model')):
                obj.parametric_umap = True
                obj.load_umap(path)
            elif exists(join(path, 'umap.pkl')):
                obj.load_umap(join(path, 'umap.pkl'))
            else:
                log.warn(f"Could not find a valid umap model at {path}. Ensure "
                         "the path is a valid directory with either 'parametric_umap' "
                         "subdirectory or a valid 'umap.pkl'.")
            # Load range/clip
            try:
                obj.load_range_clip(path)
            except FileNotFoundError:
                log.warn("Could not find range_clip.npz; results from "
                         "umap_transform() will not be normalized.")
            if exists(join(path, 'tfrecords.json')):
                obj.tfrecords = sf.util.load_json(join(path, 'tfrecords.json'))
        elif path.endswith('.parquet'):
            obj.load_coordinates(path)
        else:
            raise ValueError(
                f"Unable to determine how to load {path}. Expected "
                "a path to a directory, or a slidemap.parquet file."
            )
        obj.slides = obj.data.slide.unique()
        return obj

    @classmethod
    def from_xy(
        cls,
        x: Union[np.ndarray, List[int], str],
        y: Union[np.ndarray, List[int], str],
        slides: Union[np.ndarray, List[str], str],
        tfr_index: Union[np.ndarray, List[int], str],
        data: Optional[DataFrame] = None,
        parametric_umap: bool = False,
        cache: Optional[str] = None
    ) -> "SlideMap":
        """Initializes map from precalculated (x, y) coordinates.

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
            cache (str, optional): Deprecated
        """
        if cache is not None:
            warnings.warn(
                'Argument "cache" is deprecated for SlideMap. '
                'Instead of using/recalculating SlideMaps with cache, manually '
                'save and load maps with SlideMap.save() and SlideMap.load()',
                DeprecationWarning
            )
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
        obj = cls()
        obj.slides = obj_data.slide.unique()
        obj.data = obj_data
        obj.parametric_umap = parametric_umap
        return obj

    @classmethod
    def from_features(
        cls,
        ftrs: "DatasetFeatures",
        *,
        exclude_slides: Optional[List[str]] = None,
        map_slide: Optional[str] = None,
        parametric_umap: bool = False,
        umap_dim: int = 2,
        umap: Optional[Any] = None,
        recalculate: Optional[bool] = None, # Deprecated
        cache: Optional[str] = None,        # Deprecated
        **umap_kwargs: Any
    ) -> "SlideMap":
        """Initializes map from dataset features.

        Args:
            ftrs (:class:`slideflow.DatasetFeatures`): DatasetFeatures.
            exclude_slides (list, optional): List of slides to exclude.
            map_slide (str, optional): Either None, 'centroid', or 'average'.
                If None, will map all tiles from each slide. Defaults to None.
            umap_dim (int, optional): Number of dimensions for UMAP. Defaults
                to 2.
            umap (umap.UMAP, optional): Fit UMAP, to be used instead of fitting
                a new UMAP.
            cache (str, optional): Deprecated.
            recalculate (bool, optional): Deprecated
        """
        if recalculate or cache:
            warnings.warn(
                'Arguments "recalculate" and "cache" are deprecated for SlideMap. '
                'Instead of using/recalculating SlideMaps with cache, manually '
                'save and load maps with SlideMap.save() and SlideMap.load()',
                DeprecationWarning
            )
        if map_slide is not None and map_slide not in ('centroid', 'average'):
            raise errors.SlideMapError(
                "map_slide must be None, 'centroid' or 'average', (got "
                f"{map_slide})"
            )
        if not exclude_slides:
            slides = ftrs.slides
        else:
            slides = [s for s in ftrs.slides if s not in exclude_slides]

        obj = cls()
        obj.slides = slides
        obj.ftrs = ftrs
        obj.umap = umap  # type: ignore
        obj.parametric_umap = parametric_umap
        if map_slide:
            obj._calculate_from_slides(
                method=map_slide,
                **umap_kwargs
            )
        else:
            obj._calculate_from_tiles(
                dim=umap_dim,
                **umap_kwargs
            )
        return obj

    @classmethod
    def from_precalculated(cls, *args, **kwargs) -> "SlideMap":
        """Deprecated class initializer."""
        warnings.warn(
            "sf.SlideMap.from_precalculated() deprecated. Please use "
            "sf.SlideMap.from_xy() instead.",
            DeprecationWarning
        )
        return cls.from_xy(*args, **kwargs)

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
        **umap_kwargs: Any
    ) -> None:
        """Internal function to guide calculation of UMAP from final layer
        features / activations, as provided by DatasetFeatures.

        Keyword Args:
            dim (int): Number of dimensions for UMAP. Defaults to 2.
            n_neighbors (int): Number of neighbors for UMAP. Defaults to 50.
            min_dist (float): Minimum distance for UMAP. Defaults to 0.1.
            metric (str): UMAP metric. Defaults to 'cosine'.
            **umap_kwargs (optional): Additional keyword arguments for the
                UMAP function.
        """
        assert self.ftrs is not None

        # Calculate UMAP
        node_activations = np.concatenate([
            self.ftrs.activations[slide] for slide in self.slides
        ])

        self.map_meta['num_features'] = self.ftrs.num_features
        log.info("Calculating UMAP...")

        coordinates = self.umap_transform(node_activations, **umap_kwargs)

        # Assemble dataframe
        tfrecord_indices = np.concatenate([
            np.arange(self.ftrs.activations[slide].shape[0])
            for slide in self.slides
        ])
        slides = np.array([
            slide
            for slide in self.slides
            for _ in range(self.ftrs.activations[slide].shape[0])
        ])
        data_dict = {
            'slide': pd.Series(slides),
            'x': pd.Series(coordinates[:, 0]),
            'tfr_index': pd.Series(tfrecord_indices),
        }
        if self.ftrs.locations:
            locations = np.concatenate([
                self.ftrs.locations[slide] for slide in self.slides
            ])
            data_dict['location'] = pd.Series([l for l in locations]).astype(object)

        if self.ftrs.predictions and isinstance(self.ftrs, sf.DatasetFeatures):
            predictions = np.concatenate([
                self.ftrs.predictions[slide] for slide in self.slides
            ])
            data_dict.update({
                'predicted_class': pd.Series(np.argmax(predictions, axis=1)),
                'predictions': pd.Series([l for l in predictions]).astype(object),
            })
        if self.ftrs.uq and self.ftrs.uncertainty != {}:  # type: ignore
            uncertainty = np.concatenate([
                self.ftrs.uncertainty[slide] for slide in self.slides
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

    def _calculate_from_slides(
        self,
        method: str = 'centroid',
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
        assert self.ftrs is not None

        # Calculate optimal slide indices and centroid activations
        log.info("Calculating centroid indices...")
        opt_idx, centroid_activations = stats_utils.calculate_centroid(self.ftrs.activations)

        # Restrict mosaic to only slides that had enough tiles to calculate
        # an optimal index from centroid
        successful_slides = list(opt_idx.keys())
        num_warned = 0
        for slide in self.ftrs.slides:
            if slide not in successful_slides:
                log.debug(f"No centroid for [green]{slide}[/]; skipping")
        if num_warned:
            log.warning(f"No centroid for {num_warned} slides.")
        log.info(f"Calculating UMAP from slide-level {method}...")

        if method == 'centroid':
            umap_input = np.array([
                centroid_activations[slide] for slide in self.slides
            ])
        elif method == 'average':
            umap_input = np.array([
                np.mean(self.ftrs.activations[slide], axis=0)
                for slide in self.slides
            ])

        # Calculate UMAP
        coordinates = self.umap_transform(
            umap_input,
            **umap_kwargs
        )

        # Create dataframe
        locations = np.stack([
            self.ftrs.locations[slide][opt_idx[slide]] for slide in self.slides
        ])
        data_dict = {
            'slide': pd.Series(self.slides),
            'x': pd.Series(coordinates[:, 0]),
            'tfr_index': pd.Series(opt_idx[slide] for slide in self.slides),
            'location': pd.Series([l for l in locations]).astype(object)
        }
        if self.ftrs.predictions:
            predictions = np.stack([
                self.ftrs.predictions[slide][opt_idx[slide]] for slide in self.slides
            ])
            data_dict.update({
                'predictions': pd.Series([l for l in predictions]).astype(object),
                'predicted_class': pd.Series(np.argmax(predictions, axis=1)),
            })
        if self.ftrs.uq and self.ftrs.uncertainty != {}:  # type: ignore
            uncertainty = np.stack([
                self.ftrs.uncertainty[slide][opt_idx[slide]]
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

    def activations(self) -> np.ndarray:
        """Return associated DatasetFeatures activations as a numpy array
        corresponding to the points on this SlideMap."""
        if self.ftrs is None:
            raise ValueError(
                "No associated DatasetFeatures object for reading activations."
            )
        return np.array([
            self.ftrs.activations[row.slide][row.tfr_index]
            for row in self.data.itertuples()
        ])

    def build_mosaic(
        self,
        tfrecords: Optional[List[str]] = None,
        **kwargs
    ) -> "sf.Mosaic":
        """Build a mosaic map.

        Args:
            tfrecords (list(str), optional): List of tfrecord paths. If SlideMap
                was created using DatasetFeatures, this argument is not required.

        Keyword args:
            num_tiles_x (int, optional): Mosaic map grid size. Defaults to 50.
            tile_select (str, optional): 'first', 'nearest', or 'centroid'.
                Determines how to choose a tile for display on each grid space.
                If 'first', will display the first valid tile in a grid space
                (fastest; recommended). If 'nearest', will display tile nearest
                to center of grid space. If 'centroid', for each grid, will
                calculate which tile is nearest to centroid tile_meta.
                Defaults to 'nearest'.
            tile_meta (dict, optional): Tile metadata, used for tile_select.
                Dictionary should have slide names as keys, mapped to list of
                metadata (length of list = number of tiles in slide).
                Defaults to None.
            normalizer ((str or :class:`slideflow.norm.StainNormalizer`), optional):
                Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Stain normalization preset or
                path to a source image. Valid presets include 'v1', 'v2', and
                'v3'. If None, will use the default present ('v3').
                Defaults to None.

        """
        if self.ftrs is None and tfrecords is None:
            raise ValueError(
                "If SlideMap was not created using DatasetFeatures, then the "
                "`tfrecords` argument (list of TFRecord paths) must be supplied "
                "to `SlideMap.build_mosaic()`"
            )
        elif ((self.ftrs is not None and not len(self.ftrs.tfrecords))
               and tfrecords is None):
            raise ValueError(
                "The DatasetFeatures object used to create this SlideMap "
                "did not have paths to TFRecords stored. Please supply a list "
                "of TFRecord paths to the `tfrecords` argument "
                "of `SlideMap.build_mosaic()`"
            )
        elif (tfrecords is None
             and self.ftrs is not None
             and len(self.ftrs.tfrecords)):
            return sf.Mosaic(self, tfrecords=self.ftrs.tfrecords, **kwargs)
        else:
            return sf.Mosaic(self, tfrecords=tfrecords, **kwargs)

    def cluster(self, n_clusters: int) -> None:
        """Performs K-means clustering on data and adds to metadata labels.

        Clusters are saved to self.data['cluster']. Requires that SlideMap
        was generated via DatasetFeatures.

        Examples
            Perform K-means clustering and apply cluster labels.

                slidemap.cluster(n_clusters=5)
                slidemap.plot()

        Args:
            n_clusters (int): Number of clusters for K means clustering.
        """

        if self.ftrs is None:
            raise errors.SlideMapError(
                "Unable to cluster; no DatasetFeatures provided"
            )
        activations = [
            self.ftrs.activations[row.slide][row.tfr_index]
            for row in self.data.itertuples()
        ]
        log.info(f"Calculating K-means clustering (n={n_clusters})")
        kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(activations)
        self.data['cluster'] = kmeans.labels_
        self.label('cluster')

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
        if self.ftrs is None:
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
        """Transforms a given array using UMAP projection. If a UMAP has not
        yet been fit, this will fit a new UMAP on the given data.

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
             self._umap_normalized_clip) = stats_utils.normalize_layout(layout)
        else:
            layout = self.umap.transform(array)  # type: ignore
            if self._umap_normalized_range is not None:
                normalized = stats_utils.normalize(
                    layout,
                    norm_range=self._umap_normalized_range,
                    norm_clip=self._umap_normalized_clip)
            else:
                log.info("No range/clip information available; unable to "
                         "normalize UMAP output.")
                return layout

        return normalized

    def label_by_uncertainty(self, index: int = 0) -> None:
        """Labels each point with the tile-level uncertainty, if available.

        Args:
            index (int, optional): Uncertainty index. Defaults to 0.
        """
        if 'label' in self.data.columns:
            self.data.drop(columns='label', inplace=True)
        if self.ftrs is None:
            raise errors.SlideMapError("DatasetFeatures not provided.")
        if not self.ftrs.uq or self.ftrs.uncertainty == {}:  # type: ignore
            raise errors.DatasetError(
                'Unable to label by uncertainty; UQ estimates not available.'
            )
        else:
            uq_labels = np.stack(self.data['uncertainty'].values)[:, index]
            self.data['label'] = uq_labels

    def label_by_preds(self, index: int) -> None:
        """Displays each point with label equal to the prediction value (linear from 0-1)

        Args:
            index (int): Logit index.
        """
        if 'label' in self.data.columns:
            self.data.drop(columns='label', inplace=True)
        self.data['label'] = np.stack(self.data['predictions'].values)[:, index]

    def label_by_slide(self, slide_labels: Optional[Dict] = None) -> None:
        """Displays each point as the name of the corresponding slide.
            If slide_labels is provided, will use this dict to label slides.

        Args:
            slide_labels (dict, optional): Dict mapping slide names to labels.
        """
        if 'label' in self.data.columns:
            self.data.drop(columns='label', inplace=True)
        if slide_labels:
            self.data['label'] = self.data.slide.map(slide_labels)
        else:
            self.data['label'] = self.data.slide.values

    def label(self, meta: str, translate: Optional[Dict] = None) -> None:
        """Displays each point labeled by tile metadata (e.g. 'predicted_class')

        Args:
            meta (str): Data column from which to assign labels.
            translate (dict, optional): If provided, will translate the
                read metadata through this dictionary.
        """
        if 'label' in self.data.columns:
            self.data.drop(columns='label', inplace=True)
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
        loc: Optional[str] = 'center right',
        ncol: Optional[int] = 1,
        categorical: Union[str, bool] = 'auto',
        legend_kwargs: Optional[Dict] = None,
        **scatter_kwargs: Any,
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
            loc (str, optional): Location for legend, as defined by
                matplotlib.axes.Axes.legend(). Defaults to 'center right'.
            ncol (int, optional): Number of columns in legend, as defined
                by matplotlib.axes.Axes.legend(). Defaults to 1.
            categorical (str, optional): Specify whether labels are categorical.
                Determines the colormap.  Defaults to 'auto' (will attempt to
                automatically determine from the labels).
            legend_kwargs (dict, optional): Dictionary of additional keyword
                arguments to the matplotlib.axes.Axes.legend() function.
            **scatter_kwargs (optional): Additional keyword arguments to the
                 seaborn scatterplot function.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        if legend_kwargs is None:
            legend_kwargs = dict()

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
                try:
                    unique.sort()
                except TypeError:
                    log.error(
                        "Unable to sort categories; are some values NaN?"
                    )
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
                loc=loc,
                ncol=ncol,
                title=legend,
                **legend_kwargs
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
        from mpl_toolkits.mplot3d import Axes3D

        if fig is None:
            fig = plt.figure()

        title = f"UMAP with feature {feature} focus"
        if self.ftrs is None:
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
                self.ftrs.activations[row.slide][row.tfr_index][feature]
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
        path: str,
        dpi: int = 300,
        **kwargs,
    ):
        """Save UMAP, plot, coordinates, and normalization values to a directory.

        The UMAP, plot, coordinates, and normalization values can all be
        loaded from this directory after saving with ``sf.SlideMap.load(path)``.

        Args:
            path (str): Directory in which to save the plot and UMAP.
                The UMAP image will be saved with the filename "slidemap.png".
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
        if not exists(path):
            os.makedirs(path)
        if path.endswith('.png') or path.endswith('.jpg') or path.endswith('.jpeg'):
            log.warning(
                "Path provided to `SlideMap.save()` is a file name, "
                "not a directory. Will save the figure plot to this location, "
                "but will not save the associated UMAP. To save both plot and "
                "UMAP, provide a path to a directory instead."
            )
            self.save_plot(path, dpi=dpi, **kwargs)
        else:
            self.save_plot(join(path, "slidemap.png"), dpi=dpi, **kwargs)
            if self.umap is not None:
                self.save_umap(path)

    def save_plot(
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

    def save_coordinates(self, path: str) -> None:
        """Save coordinates only to parquet file.

        Args:
            path (str, optional): Save coordinates to this location.
        """
        self.data.to_parquet(path)
        log.info(f"Wrote slide map coordinates to [green]{path}")

    def save_umap(self, path: str) -> None:
        """Save UMAP, coordinates, and normalization information to a directory.

        Args:
            path (str, optional): Save UMAP and coordinates to this directory.
                Coordinates will be saved in this directory with the filename
                ``slidemap.parquet`` Model will be saved as umap.pkl (parametric)
                or model.pkl (parametric).
        """
        if self.parametric_umap:
            self.umap.save(path)
        else:
            with open(join(path, 'umap.pkl'), 'wb') as f:
                pickle.dump(self.umap, f)
                log.info(f"Wrote UMAP coordinates to [green]{path}")
        self.save_coordinates(join(path, 'slidemap.parquet'))
        self.save_range_clip(path)

    def save_encoder(self, path: str) -> None:
        """Save Parametric UMAP encoder only."""
        if not self.parametric_umap:
            raise ValueError("SlideMap not built with Parametric UMAP.")
        self.umap.encoder.save(join(path, 'encoder'))
        self.save_coordinates(join(path, 'slidemap.parquet'))
        self.save_range_clip(path)

    def save_range_clip(self, dest: str) -> None:
        """Save range/clip information.

        If ZIP saving is enabled, will save to range_clip.npz, with the
        attributes ``"range"`` and ``"clip"``.

        If ZIP saving is disabled (SF_ALLOW_ZIP=0, for databricks compatibility),
        will save these attributes to range.npy and clip.npy, separately.

        Args:
            dest (str): Destination directory.

        """
        if sf.util.zip_allowed():
            np.savez(
                dest + 'range_clip.npz',
                range=self._umap_normalized_range,
                clip=self._umap_normalized_clip
            )
        else:
            np.save(dest + 'range.npy', self._umap_normalized_range)
            np.save(dest + 'clip.npy', self._umap_normalized_clip)

    def load_range_clip(self, path: str) -> None:
        """Load a saved range_clip.npz file for normalizing raw UMAP output.

        Args:
            path (str): Path to numpy file (\*.npz) with 'clip' and 'range' keys
                as generated from ``SlideMap.save()``.

        """
        rc_path, r_path, c_path = None, None, None
        if exists(path) and path.endswith('.npz'):
            rc_path = path
        elif exists(join(path, 'range_clip.npz')):
            rc_path = join(path, 'range_clip.npz')
        elif exists(join(path, 'range.npy')) and exists(join(path, 'clip.npy')):
            r_path = join(path, 'range.npy')
            c_path = join(path, 'clip.npy')
        else:
            raise FileNotFoundError(
                f"Unable to find range/clip information at {path}."
            )
        if rc_path:
            loaded = np.load(path)
            if not ('range' in loaded and 'clip' in loaded):
                raise ValueError(f"Unable to load {path}; did not find values "
                                "'range' and 'clip'.")
            self._umap_normalized_clip = loaded['clip']
            self._umap_normalized_range = loaded['range']
        else:
            self._umap_normalized_clip = np.load(c_path)
            self._umap_normalized_range = np.load(r_path)
        log.info("Loaded range={}, clip={}".format(
            self._umap_normalized_range,
            self._umap_normalized_clip
        ))

    def load_umap(self, path: str) -> "umap.UMAP":
        """Load only a UMAP model and not slide coordinates or range_clip.npz.

        Args:
            path (str): Path to either umap.pkl or directory with saved
                parametric UMAP.

        """
        log.debug(f"Loading UMAP at {path}")
        if self.parametric_umap:
            from umap.parametric_umap import load_ParametricUMAP
            self.umap = load_ParametricUMAP(path)
        else:
            with open(path, 'rb') as f:
                self.umap = pickle.load(f)
                log.info(f"Loaded UMAP from [green]{path}")

    def load_coordinates(self, path: str) -> None:
        """Load coordinates from parquet file.

        Args:
            path (str, optional): Path to parquet file (.parquet) with SlideMap
                coordinates.

        """
        log.debug(f"Loading coordinates at {path}")
        self.data = pd.read_parquet(path)
        log.info(f"Loaded coordinates from [green]{path}")
