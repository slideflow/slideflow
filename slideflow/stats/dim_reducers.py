from typing import Any, Optional, Dict
import numpy as np
import warnings

class DimReducer:
    """Base class for dimensionality reduction methods."""

    def __init__(self) -> None:
        self.reducer = None
        self._normalized_range = None
        self._normalized_clip = None

    def fit_transform(self, array: np.ndarray) -> np.ndarray:
        """Fits the reducer on the data and transforms it.
        
        Args:
            array (np.ndarray): Input array to reduce dimensions
            
        Returns:
            np.ndarray: Transformed array with reduced dimensions
        """
        raise NotImplementedError

    def transform(self, array: np.ndarray) -> np.ndarray:
        """Transforms new data using the fitted reducer.
        
        Args:
            array (np.ndarray): Input array to reduce dimensions
            
        Returns:
            np.ndarray: Transformed array with reduced dimensions
        """
        raise NotImplementedError

    @property
    def is_fitted(self) -> bool:
        """Returns whether the reducer has been fitted."""
        return self.reducer is not None

class UMAPReducer(DimReducer):
    """UMAP implementation of dimensionality reduction."""

    def __init__(
        self,
        dim: int = 2,
        n_neighbors: int = 50,
        min_dist: float = 0.1,
        metric: str = 'cosine',
        parametric: bool = False,
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """Initialize UMAP reducer.
        
        Args:
            dim (int): Output dimensions
            n_neighbors (int): Number of neighbors for UMAP
            min_dist (float): Minimum distance for UMAP
            metric (str): Distance metric for UMAP
            parametric (bool): Whether to use parametric UMAP
            verbose (bool): Whether to print progress
            **kwargs: Additional arguments passed to UMAP
        """
        super().__init__()
        self.params = {
            'n_components': dim,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'metric': metric,
            'verbose': verbose,
            **kwargs
        }
        self.parametric = parametric

    def _init_reducer(self):
        """Initialize the UMAP reducer with stored parameters."""
        import umap
        reducer_class = umap.ParametricUMAP if self.parametric else umap.UMAP
        self.reducer = reducer_class(**self.params)

    def fit_transform(self, array: np.ndarray) -> np.ndarray:
        """Fit UMAP to data and transform it.
        
        Args:
            array (np.ndarray): Input array
            
        Returns:
            np.ndarray: Transformed array
        """
        from slideflow.stats import stats_utils

        if not len(array):
            raise ValueError("Unable to perform reduction on empty array.")
            
        if self.reducer is None:
            self._init_reducer()
            
        layout = self.reducer.fit_transform(array)
        normalized, self._normalized_range, self._normalized_clip = stats_utils.normalize_layout(layout)
        return normalized

    def transform(self, array: np.ndarray) -> np.ndarray:
        """Transform new data using fitted UMAP.
        
        Args:
            array (np.ndarray): Input array
            
        Returns:
            np.ndarray: Transformed array
        """
        from slideflow.stats import stats_utils
        
        if not self.is_fitted:
            raise ValueError("Reducer must be fitted before transform can be called.")
            
        layout = self.reducer.transform(array)
        
        if self._normalized_range is not None:
            return stats_utils.normalize(
                layout,
                norm_range=self._normalized_range,
                norm_clip=self._normalized_clip
            )
        else:
            warnings.warn("No range/clip information available; unable to normalize output.")
            return layout