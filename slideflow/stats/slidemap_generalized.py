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
from slideflow.stats.dim_reducers import DimReducer, UMAPReducer

if TYPE_CHECKING:
    import umap
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from slideflow.model import DatasetFeatures


class SlideMapGeneralized:
    """Two-dimensional slide map for visualization & backend for mosaic maps.

    Slides are mapped in 2D either explicitly with pre-specified coordinates,
    or with dimensionality reduction from post-convolutional layer weights,
    provided from :class:`slideflow.DatasetFeatures`.
    """

    def __init__(
        self,
        *,
        reducer: Optional[DimReducer] = None,
        parametric_umap: bool = False
    ) -> None:
        """Backend for mapping slides into two dimensional space. Can use a
        DatasetFeatures object to map slides according to UMAP of features, or
        map according to pre-specified coordinates.

        Can be initialized with three methods: from precalculated X/Y
        coordinates, from a DatasetFeatures object, or from a saved map.

        Examples
            Build a SlideMapGeneralized from a DatasetFeatures object

                .. code-block:: python

                    dts_ftrs = sf.DatasetFeatures(model, dataset)
                    slidemap = sf.SlideMapGeneralized.from_features(dts_ftrs)

            Build a SlideMapGeneralized from prespecified coordinates

                .. code-block:: python

                    x = np.array(...)
                    y = np.array(...)
                    slides = ['slide1', 'slide1', 'slide5', ...]
                    slidemap = sf.SlideMapGeneralized.from_xy(
                        x=x, y=y, slides=slides
                    )

            Load a saved SlideMap

                .. code-block:: python

                    slidemap = sf.SlideMap.load('map.parquet')

        Args:
            slides (list(str)): List of slide names
        """
        if parametric_umap:
            warnings.warn(
                'The parametric_umap parameter is deprecated. Please use the reducer '
                'parameter with a UMAPReducer instance instead.',
                DeprecationWarning
            )
            assert isinstance(parametric_umap, bool), "Expected <bool> for argument 'parametric_umap'"
            self.reducer = UMAPReducer(parametric=True)
        else:
            self.reducer = reducer or UMAPReducer()
        
        self.data = None    # type: DataFrame
        self.ftrs = None    # type: Optional[DatasetFeatures]
        self.slides = None  # type: List[str]
        self.tfrecords = None  # type: List[str]
        # self.parametric_umap = parametric_umap
        # self._umap_normalized_range = None
        self.map_meta = {}  # type: Dict[str, Any]

    @classmethod
    def from_features(
        cls,
        ftrs: "DatasetFeatures",
        *,
        exclude_slides: Optional[List[str]] = None,
        map_slide: Optional[str] = None,
        parametric_umap: bool = False,  # For backwards compatibility
        umap_dim: int = 2,              # For backwards compatibility
        umap: Optional[Any] = None,     # For backwards compatibility
        recalculate: Optional[bool] = None, # Deprecated
        cache: Optional[str] = None,        # Deprecated
        reducer: Optional[DimReducer] = None,
        **reducer_kwargs: Any
    ) -> "SlideMapGeneralized":
        """Initializes map from dataset features.

        Args:
            ftrs (:class:`slideflow.DatasetFeatures`): DatasetFeatures.
            exclude_slides (list, optional): List of slides to exclude.
            map_slide (str, optional): Either None, 'centroid', or 'average'.
            reducer (DimReducer, optional): Dimensionality reduction method
            
            parametric_umap (bool): Deprecated. Use reducer parameter instead.
            umap_dim (int): Deprecated. Use reducer parameter instead.
            umap (umap.UMAP): Deprecated. Use reducer parameter instead.
            
            cache (str, optional): Deprecated.
            recalculate (bool, optional): Deprecated

            **reducer_kwargs: Additional arguments for the reducer
        """
        print('DEBUG: from_features')
        if recalculate or cache:
            warnings.warn(
                'Arguments "recalculate" and "cache" are deprecated for SlideMap. '
                'Instead of using/recalculating SlideMaps with cache, manually '
                'save and load maps with SlideMap.save() and SlideMap.load()',
                DeprecationWarning
            )
        umap_args = False
        # if reducer_kwargs contains n_neighbors or min_dist, then we need to warn
        if any([reducer_kwargs.get('n_neighbors') is not None, reducer_kwargs.get('min_dist') is not None]):
            warnings.warn(
                'Parameters n_neighbors and min_dist are deprecated. '
                'Please use the reducer parameter instead.',
                DeprecationWarning
            )
            umap_args = True

        if any([parametric_umap, umap_dim != 2, umap is not None, umap_args]):
            warnings.warn(
                'Parameters parametric_umap, umap_dim, and umap are deprecated. '
                'Please use the reducer parameter instead.',
                DeprecationWarning
            )
            reducer = UMAPReducer(
                dim=umap_dim,
                parametric=parametric_umap,
                **reducer_kwargs
            )
            if umap is not None:
                reducer.reducer = umap

        if map_slide is not None and map_slide not in ('centroid', 'average'):
            raise errors.SlideMapError(
                "map_slide must be None, 'centroid' or 'average', (got "
                f"{map_slide})"
            )
        if not exclude_slides:
            slides = ftrs.slides
        else:
            slides = [s for s in ftrs.slides if s not in exclude_slides]

        obj = cls(reducer=reducer)
        obj.slides = slides
        obj.ftrs = ftrs

        if map_slide:
            # not implemented
            print("not implemented")
            return None
        else:
            obj._calculate_from_tiles()
            
        return obj
    
    @property
    def x(self):
        """X coordinates of map."""
        return self.data.x.values

    @property
    def y(self):
        """Y coordinates of map."""
        return self.data.y.values

    def _calculate_from_tiles(self) -> None:
        """Internal function to calculate reduced dimensions from features."""
        assert self.ftrs is not None

        # Calculate reduced dimensions
        node_activations = np.concatenate([
            self.ftrs.activations[slide] for slide in self.slides
        ])

        self.map_meta['num_features'] = self.ftrs.num_features
        log.info("Calculating reduced dimensions...")

        coordinates = self.reducer.fit_transform(node_activations)

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
        
        # Add additional data if available
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
            
        if self.ftrs.uq and self.ftrs.uncertainty != {}:
            uncertainty = np.concatenate([
                self.ftrs.uncertainty[slide] for slide in self.slides
            ])
            data_dict.update({
                'uncertainty': pd.Series(
                    [u for u in uncertainty]
                ).astype(object)
            })
            
        if coordinates.shape[1] > 1:
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

    def filter(self, slides: List[str]) -> None:
        """Filters map to only show tiles from the given slides.

        Args:
            slides (list(str)): List of slide names.
        """

        self.data = self.data.loc[self.data.slide.isin(slides)]

    def umap_transform(self, array: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Deprecated. Use reducer.transform() instead."""
        warnings.warn(
            'umap_transform() is deprecated. Use reducer.transform() instead.',
            DeprecationWarning
        )
        return self.reducer.transform(array)

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
        """Displays each point with label equal to the prediction value (from 0-1)

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

        map_2d = sns.scatterplot(
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
        map_2d.set(xlabel=xlabel, ylabel=ylabel)
        if title:
            ax.set_title(title)

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

        with sf.util.matplotlib_backend('Agg'):
            self.plot(**kwargs)
            plt.savefig(filename, bbox_inches='tight', dpi=dpi)
            plt.close()
        log.info(f"Saved 2D MAP to [green]{filename}")

    def save_coordinates(self, path: str) -> None:
        """Save coordinates only to parquet file.

        Args:
            path (str, optional): Save coordinates to this location.
        """
        self.data.to_parquet(path)
        log.info(f"Wrote slide map coordinates to [green]{path}")

    def load_coordinates(self, path: str) -> None:
        """Load coordinates from parquet file.

        Args:
            path (str, optional): Path to parquet file (.parquet) with SlideMap
                coordinates.

        """
        log.debug(f"Loading coordinates at {path}")
        self.data = pd.read_parquet(path)
        log.info(f"Loaded coordinates from [green]{path}")
