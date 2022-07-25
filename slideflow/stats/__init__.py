"""Submodule for statistics, metrics, and related functions."""

from . import metrics, plot
from .metrics import (df_from_pred, eval_from_dataset, eval_dataset,
                      group_reduce, metrics_from_dataset, name_columns,
                      predict_from_dataset, predict_dataset)
from .slidemap import SlideMap
from .stats_utils import calculate_centroid, get_centroid_index
