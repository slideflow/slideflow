from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def calculate_centroid(
    act: Dict[str, np.ndarray]
) -> Tuple[Dict[str, int], Dict[str, np.ndarray]]:
    """Calcultes slide-level centroid indices for a provided activations dict.

    Args:
        activations (dict): Dict mapping slide names to ndarray of activations
            across tiles, of shape (n_tiles, n_features)

    Returns:
        A tuple containing

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


def get_centroid_index(arr: np.ndarray) -> int:
    """Calculate index nearest to centroid from a given 2D input array."""
    km = KMeans(n_clusters=1).fit(arr)
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, arr)
    return closest[0]


def normalize_layout(
    layout: np.ndarray,
    min_percentile: int = 1,
    max_percentile: int = 99,
    relative_margin: float = 0.1
) -> np.ndarray:
    """Removes outliers and scales layout to between [0,1].

    Args:
        layout (np.ndarray): 2D array containing data to be scaled.
        min_percentile (int, optional): Percentile for scaling. Defaults to 1.
        max_percentile (int, optional): Percentile for scaling. Defaults to 99.
        relative_margin (float, optional): Add an additional margin (fraction
            of total plot width). Defaults to 0.1.

    Returns:
        np.ndarray: layout array, re-scaled and clipped.
    """

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

