"""Submodule for statistics, metrics, and related functions."""

from .metrics import (group_reduce, name_columns, df_from_pred, 
                      predict_from_dataset, eval_from_dataset, 
                      metrics_from_dataset)
from .slidemap import SlideMap
from .stats_utils import calculate_centroid, get_centroid_index