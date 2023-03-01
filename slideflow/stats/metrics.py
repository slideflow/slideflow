"""Categorical, linear, and CPH metrics for predictions."""

import math
import multiprocessing as mp
import warnings
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index as c_index
from pandas.core.frame import DataFrame
from sklearn import metrics
from os.path import join
from types import SimpleNamespace
from typing import (TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union,
                    Callable)


import slideflow as sf
from slideflow import errors
from slideflow.util import log
from .delong import delong_roc_variance

if TYPE_CHECKING:
    import neptune.new as neptune
    import tensorflow as tf
    import torch


class ClassifierMetrics:
    def __init__(self, y_true, y_pred, neptune_run=None, autofit=True):
        self.y_true = y_true
        self.y_pred = y_pred
        self.neptune_run = neptune_run

        self.fpr = None
        self.tpr = None
        self.threshold = None
        self.auroc = None
        self.precision = None
        self.recall = None
        self.ap = None

        if autofit:
            self.roc_fit()
            self.prc_fit()

    def roc_fit(self):
        self.fpr, self.tpr, self.threshold = metrics.roc_curve(
            self.y_true,
            self.y_pred
        )
        self.auroc = metrics.auc(self.fpr, self.tpr)
        try:
            max_youden = max(zip(self.tpr, self.fpr), key=lambda x: x[0]-x[1])
            opt_thresh_index = list(zip(self.tpr, self.fpr)).index(max_youden)
            self.opt_thresh = self.threshold[opt_thresh_index]
        except Exception:
            self.opt_thresh = None

    def auroc_ci(self, alpha=0.05):
        from scipy import stats
        delong_auc, auc_cov = delong_roc_variance(self.y_true, self.y_pred)
        auc_std = np.sqrt(auc_cov)
        lower_upper_q = np.abs(np.array([0, 1]) - alpha / 2)
        ci = stats.norm.ppf(lower_upper_q, loc=delong_auc, scale=auc_std)
        ci[ci > 1] = 1
        return tuple(ci)

    def auroc_pval(self, mu=0.5, alpha=0.05):
        from scipy.stats import norm
        lo, up = self.auroc_ci(alpha=alpha)
        se = (up - lo) / (2 * 1.96)
        z = (self.auroc - mu) / se
        return 2 * norm.cdf(-abs(z))

    def prc_fit(self):
        self.precision, self.recall, _ = metrics.precision_recall_curve(
            self.y_true,
            self.y_pred
        )
        self.ap = metrics.average_precision_score(self.y_true, self.y_pred)

    def save_roc(self, outdir, name):
        import matplotlib.pyplot as plt
        auroc_str = 'NA' if not self.auroc else f'{self.auroc:.2f}'
        sf.stats.plot.roc(self.fpr, self.tpr, f'AUC = {auroc_str}')
        full_path = join(outdir, f'{name}.png')
        plt.savefig(full_path)
        if self.neptune_run:
            self.neptune_run[f'results/graphs/{name}'].upload(full_path)

    def save_prc(self, outdir, name):
        import matplotlib.pyplot as plt
        ap_str = 'NA' if not self.ap else f'{self.ap:.2f}'
        sf.stats.plot.prc(self.precision, self.recall, label=f'AP = {ap_str}')
        full_path = join(outdir, f'{name}.png')
        plt.savefig(full_path)
        if self.neptune_run:
            self.neptune_run[f'results/graphs/{name}'].upload(full_path)


def _assert_model_type(model_type: str) -> None:
    """Raises a ValueError if the model type is invalid."""
    if model_type not in ('categorical', 'linear', 'cph'):
        raise ValueError(f"Unrecognized model_type {model_type}, must be "
                         "categorical, linear, or cph")


def _generate_tile_roc(
    yt_and_yp: Tuple[np.ndarray, np.ndarray],
    neptune_run: Optional["neptune.Run"] = None
) -> ClassifierMetrics:
    """Generate tile-level ROC. Defined separately for multiprocessing.

    Args:
        yt_and_yp (Tuple[np.ndarray, np.ndarray]): y_true and y_pred.
        neptune_run (neptune.Run, optional): Neptune run. Defaults to None.

    Returns:
        ClassifierMetrics: Contains metrics (AUROC, AP).
    """
    y_true, y_pred = yt_and_yp
    class_metrics = ClassifierMetrics(y_true, y_pred, neptune_run=neptune_run)
    return class_metrics


def _merge_metrics(metrics_by_level: Dict[str, Dict]) -> Dict[str, Dict]:
    """Merge dictionary of levels into a dictionary by metric.

    Function accepts a dictionary organized as such:

    {
        'tile':  {'auc': [...], 'ap': [...]},
        'slide': {'auc': [...], 'ap': [...]},
        ...
    }

    and converts it to:

    {
        'auc': {'tile': [...], 'slide': [...]},
        'ap':  {'tile': [...], 'slide': [...]},
        ...
    }
    """
    levels = list(metrics_by_level.keys())
    metrics = list(metrics_by_level[levels[0]].keys())
    return {
        metric: {
            level: metrics_by_level[level][metric]
            for level in levels
        } for metric in metrics
    }


def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Generates metrics, including sensitivity, specificity, and accuracy.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predictions.

    Returns:
        Dict[str, float]: Dict with metrics including accuracy, sensitivity,
        specificity, precision, recall, f1_score, and kappa.
    """
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


def categorical_metrics(
    df: DataFrame,
    label: str = '',
    level: str = 'tile',
    data_dir: str = '',
    neptune_run: Optional["neptune.Run"] = None
) -> Dict[str, Dict[str, float]]:
    """Generates categorical metrics (AUC/AP) from a set of predictions.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing labels, predictions,
            and optionally uncertainty, as returned by sf.stats.df_from_pred()

    Keyword args:
        label (str, optional): Label prefix/suffix for ROCs.
            Defaults to an empty string.
        level (str, optional): Group-level for the predictions. Used for
            labeling plots. Defaults to 'tile'.
        data_dir (str, optional): Path to data directory for saving plots.
            Defaults to None.

    Returns:
        Dict containing metrics, with the keys 'auc' and 'ap'.
    """

    label_start = "" if label == '' else f"{label}_"

    # Detect the number of outcomes and confirm that the number of outcomes
    # match the provided outcome names
    outcome_names = [c[:-8] for c in df.columns if c.endswith('-y_pred0')]

    if not len(outcome_names):
        raise errors.StatsError("No outcomes detected from dataframe.")

    all_auc = {outcome: [] for outcome in outcome_names}  # type: Dict
    all_ap = {outcome: [] for outcome in outcome_names}  # type: Dict

    def y_true_onehot(_df, i):
        return (_df.y_true == i).astype(int)

    def y_pred_onehot(_df, i):
        return (_df.y_pred_cat == i).astype(int)

    # Perform analysis separately for each outcome column
    for outcome in outcome_names:
        outcome_cols = [c for c in df.columns if c.startswith(f'{outcome}-')]

        # Remove the outcome name from the dataframe temporarily
        outcome_df = df[outcome_cols].rename(
            columns={
                orig_col: orig_col.replace(f'{outcome}-', '', 1)
                for orig_col in outcome_cols
            }
        )
        log.info(f"Validation metrics for outcome [green]{outcome}[/]:")
        y_pred_cols = [c for c in outcome_df.columns if c.startswith('y_pred')]
        num_cat = len(y_pred_cols)
        if not num_cat:
            raise errors.StatsError(
                f"Could not find predictions column for outcome {outcome}"
            )

        # Sort the prediction columns so that argmax will work as expected
        y_pred_cols = [f'y_pred{i}' for i in range(num_cat)]
        if len(y_pred_cols) != num_cat:
            raise errors.StatsError(
                "Malformed dataframe, unable to find all prediction columns"
            )
        if not all(col in outcome_df.columns for col in y_pred_cols):
            raise errors.StatsError("Malformed dataframe, invalid column names")

        # Convert to one-hot encoding
        outcome_df['y_pred_cat'] = outcome_df[y_pred_cols].values.argmax(1)

        log.debug(f"Calculating metrics with a thread pool")
        p = mp.dummy.Pool(8)
        yt_and_yp = [
            ((outcome_df.y_true == i).astype(int), outcome_df[f'y_pred{i}'])
            for i in range(num_cat)
        ]
        try:
            for i, fit in enumerate(p.imap(_generate_tile_roc, yt_and_yp)):
                fit.save_roc(data_dir, f"{label_start}{outcome}_{level}_ROC{i}")
                fit.save_prc(data_dir, f"{label_start}{outcome}_{level}_PRC{i}")
                all_auc[outcome] += [fit.auroc]
                all_ap[outcome] += [fit.ap]
                auroc_str = 'NA' if not fit.auroc else f'{fit.auroc:.3f}'
                ap_str = 'NA' if not fit.ap else f'{fit.ap:.3f}'
                thresh = 'NA' if not fit.opt_thresh else f'{fit.opt_thresh:.3f}'
                log.info(
                    f"{level}-level AUC (cat #{i:>2}): {auroc_str} "
                    f"AP: {ap_str} (opt. threshold: {thresh})"
                )
        except ValueError as e:
            # Occurs when predictions contain NaN
            log.error(f'Error encountered when generating AUC: {e}')
            all_auc[outcome] = -1
            all_ap[outcome] = -1
        p.close()

        # Calculate tile-level accuracy.
        # Category-level accuracy is determined by comparing
        # one-hot predictions to one-hot y_true.
        for i in range(num_cat):
            try:
                yt_in_cat =  y_true_onehot(outcome_df, i)
                n_in_cat = yt_in_cat.sum()
                correct = y_pred_onehot(outcome_df.loc[yt_in_cat == 1], i).sum()
                category_accuracy = correct / n_in_cat
                perc = category_accuracy * 100
                log.info(f"Category {i} acc: {perc:.1f}% ({correct}/{n_in_cat})")
            except IndexError:
                log.warning(f"Error with category accuracy for cat # {i}")
    return {
        'auc': all_auc,
        'ap': all_ap,
    }


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


def cph_metrics(
    df: DataFrame,
    level: str = 'tile',
    label: str = '',
    data_dir: str = '',
    neptune_run: Optional["neptune.Run"] = None
) -> Dict[str, float]:
    """Generates CPH metrics (concordance index) from a set of predictions.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing labels, predictions,
            and optionally uncertainty, as returned by sf.stats.df_from_pred().
            The dataframe columns should be appropriately named using
            sf.stats.name_columns().

    Keyword args:
        label (str, optional): Label prefix/suffix for ROCs.
            Defaults to an empty string.
        level (str, optional): Group-level for the predictions. Used for
            labeling plots. Defaults to 'tile'.
        data_dir (str, optional): Path to data directory for saving plots.
            Defaults to None.

    Returns:
        Dict containing metrics, with the key 'c_index'.
    """
    cph_cols = ('time-y_true', 'time-y_pred', 'event-y_true')
    if any(c not in df.columns for c in cph_cols):
        raise ValueError(
            "Improperly formatted dataframe to cph_metrics(), "
            f"must have columns {cph_cols}. Got: {list(df.columns)}"
        )

    # Calculate metrics
    try:
        c_index = concordance_index(
            df['time-y_true'].values,
            df[['time-y_pred', 'event-y_true']].values,
        )
        c_str = 'NA' if not c_index else f'{c_index:.3f}'
        log.info(f"C-index ({level}-level): {c_str}")
    except ZeroDivisionError as e:
        log.error(f"Error calculating concordance index: {e}")
        c_index = -1
    return {
        'c_index': c_index
    }


def df_from_pred(
    y_true: Optional[List[Any]],
    y_pred: List[Any],
    y_std: Optional[List[Any]],
    tile_to_slides: Union[List, np.ndarray],
    locations: Optional[Union[List, np.ndarray]] = None
) -> DataFrame:
    """Converts arrays of model predictions to a pandas dataframe.

    Args:
        y_true (list(np.ndarray)): List of y_true numpy arrays, one array for
            each outcome. For linear outcomes, the length of the outer
            list should be one, and the second shape dimension of the numpy
            array should be the number of linear outcomes.
        y_pred (list(np.ndarray)): List of y_pred numpy arrays, one array for
            each outcome. For linear outcomes, the length of the outer
            list should be one, and the second shape dimension of the numpy
            array should be the number of linear outcomes.
        y_std (list(np.ndarray)): List of uncertainty numpy arrays, formatted
            in the same way as y_pred.
        tile_to_slides (np.ndarray): Array of slide names for each tile. Length
            should match the numpy arrays in y_true, y_pred, and y_std.

    Returns:
        DataFrame: DataFrame of predictions.
    """
    len_err_msg = "{} must be a list of length equal to number of outcomes"
    if y_true is not None and not isinstance(y_true, (list, tuple)):
        raise ValueError(len_err_msg.format('y_true'))
    if y_true is not None and not len(y_true) == len(y_pred):
        raise ValueError('Length of y_pred and y_true must be equal')
    if not isinstance(y_pred, (list, tuple)):
        raise ValueError(len_err_msg.format('y_pred'))
    if y_std is not None and not isinstance(y_std, (list, tuple)):
        raise ValueError(len_err_msg.format('y_std'))
    if y_std is not None and len(y_std) != len(y_pred):
        raise ValueError('If y_std is provided, length must equal y_pred')
    if locations is not None and len(locations) != len(tile_to_slides):
        raise ValueError(
            'If locations is provided, length must equal tile_to_slides '
            f'(got: {len(locations)} and {len(tile_to_slides)})')

    n_outcomes = len(y_pred)
    series = {
        'slide': pd.Series(tile_to_slides)
    }
    if locations is not None:
        if not isinstance(locations, np.ndarray):
            locations = np.array(locations)
        series.update({
            'loc_x': locations[:, 0],
            'loc_y': locations[:, 1]
        })
    # Iterate through each outcome in y_pred
    for oi in range(n_outcomes):
        # Add y_pred columns
        series.update({
            f'out{oi}-y_pred{n}': y_pred[oi][:, n]
            for n in range(y_pred[oi].shape[1])
        })
        # Add y_true columns
        if y_true is not None:
            if len(y_true[oi].shape) == 1:
                series.update({
                    f'out{oi}-y_true': y_true[oi]
                })
            else:
                series.update({
                    f'out{oi}-y_true{n}': y_true[oi][:, n]
                    for n in range(y_true[oi].shape[1])
                })
        # Add uncertainty columns
        if y_std is not None:
            series.update({
                f'out{oi}-uncertainty{n}': y_std[oi][:, n]
                for n in range(y_std[oi].shape[1])
            })
    return pd.DataFrame(series)


def eval_from_dataset(*args, **kwargs):
    warnings.warning(
        "`sf.stats.metrics.eval_from_dataset() is deprecated. Please use "
        "`sf.stats.metrics.eval_dataset()` instead.",
        DeprecationWarning)
    return eval_dataset(*args, **kwargs)


def eval_dataset(
    model: Union["tf.keras.Model", "torch.nn.Module"],
    dataset: Union["tf.data.Dataset", "torch.utils.data.DataLoader"],
    model_type: str,
    num_tiles: int = 0,
    uq: bool = False,
    uq_n: int = 30,
    reduce_method: str = 'average',
    patients: Optional[Dict[str, str]] = None,
    outcome_names: Optional[List[str]] = None,
    loss: Optional[Callable] = None,
    torch_args: Optional[SimpleNamespace] = None,
) -> Tuple[DataFrame, float, float]:
    """Generates predictions and accuracy/loss from a given model and dataset.

    Args:
        model (str): Path to PyTorch model.
        dataset (tf.data.Dataset): PyTorch dataloader.
        model_type (str, optional): 'categorical', 'linear', or 'cph'.
            If multiple linear outcomes are present, y_true is stacked into a
            single vector for each image. Defaults to 'categorical'.
        num_tiles (int, optional): Used for progress bar with Tensorflow.
            Defaults to 0.
        uq_n (int, optional): Number of forward passes to perform
            when calculating MC Dropout uncertainty. Defaults to 30.
        reduce_method (str, optional): Reduction method for calculating
            slide-level and patient-level predictions for categorical outcomes.
            Either 'average' or 'proportion'. If 'average', will reduce with
            average of each logit across tiles. If 'proportion', will convert
            tile predictions into onehot encoding then reduce by averaging
            these onehot values. Defaults to 'average'.
        patients (dict, optional): Dictionary mapping slide names to patient
            names. Required for generating patient-level metrics.
        outcome_names (list, optional): List of str, names for outcomes.
            Defaults to None (outcomes will not be named).
        torch_args (namespace): Used for PyTorch models. Namespace containing
            num_slide_features, slide_input, update_corrects, and
            update_loss functions.

    Returns:
        pd.DataFrame, accuracy, loss
    """
    if model_type != 'categorical' and reduce_method != 'average':
        raise ValueError(
            f'Reduction method {reduce_method} incompatible with '
            f'model_type {model_type}'
        )
    if sf.model.is_tensorflow_model(model):
        from slideflow.model import tensorflow_utils
        df, acc, total_loss = tensorflow_utils.eval_from_model(
            model,
            dataset,
            model_type,
            loss=loss,
            num_tiles=num_tiles,
            uq=uq,
            uq_n=uq_n,
        )
    else:
        from slideflow.model import torch_utils
        df, acc, total_loss = torch_utils.eval_from_model(
            model,
            dataset,
            model_type,
            torch_args=torch_args,
            uq=uq,
            uq_n=uq_n,
        )

    if outcome_names or model_type == 'cph':
        df = name_columns(df, model_type, outcome_names)
    dfs = group_reduce(df, method=reduce_method, patients=patients)
    return dfs, acc, total_loss


def group_reduce(
    df: DataFrame,
    method: str = 'average',
    patients: Optional[Dict[str, str]] = None
) -> Dict[str, DataFrame]:
    """Reduces tile-level predictions to group-level predictions.

    Args:
        df (DataFrame): Tile-level predictions.
        method (str, optional): Reduction method for calculating
            slide-level and patient-level predictions for categorical outcomes.
            Either 'average' or 'proportion'. If 'average', will reduce with
            average of each logit across tiles. If 'proportion', will convert
            tile predictions into onehot encoding then reduce by averaging
            these onehot values. Defaults to 'average'.
        patients (dict, optional): Dictionary mapping slide names to patient
            names. Required for generating patient-level metrics.
    """
    if method not in ('proportion', 'average'):
        raise ValueError(f"Unknown method {method}")
    log.debug(f"Using reduce_method={method}")

    if patients is not None:
        df['patient'] = df['slide'].map(patients)
        groups = ['slide', 'patient']
    else:
        groups = ['slide']

    group_dfs = {
        'tile': df
    }
    _df = df[[c for c in df.columns if c not in ('loc_x', 'loc_y')]].copy()
    if method == 'proportion':
        outcome_names = [c[:-8] for c in df.columns if c.endswith('-y_pred0')]
        if not len(outcome_names):
            raise errors.StatsError("No outcomes detected from dataframe.")
        for outcome in outcome_names:
            y_pred_cols = [c for c in df.columns
                           if c.startswith(f"{outcome}-y_pred")]
            num_cat = len(y_pred_cols)
            if not num_cat:
                raise errors.StatsError(
                    f"Could not find predictions column for outcome {outcome}"
                )
            if num_cat != df[f'{outcome}-y_true'].max()+1:
                raise errors.StatsError(
                    "Model predictions have a different number of outcome "
                    f"categories ({df[f'{outcome}-y_true'].max()+1}) "
                    f"than provided annotations ({num_cat})"
                )
            y_pred_cols = [f'{outcome}-y_pred{i}' for i in range(num_cat)]
            if len(y_pred_cols) != num_cat:
                raise errors.StatsError(
                    "Malformed dataframe, unable to find all prediction columns"
                )
            if not all(col in df.columns for col in y_pred_cols):
                raise errors.StatsError(
                    "Malformed dataframe, invalid column names"
                )

            outcome_pred_cat = df[y_pred_cols].values.argmax(1)
            for i in range(num_cat):
                _df[f'{outcome}-y_pred{i}'] = (outcome_pred_cat == i).astype(int)

    for group in groups:
        group_dfs.update({
            group: _df.groupby(group, as_index=False).mean(numeric_only=True)
        })

    return group_dfs


def linear_metrics(
    df: DataFrame,
    label: str = '',
    level: str = 'tile',
    data_dir: str = '',
    neptune_run: Optional["neptune.Run"] = None
) -> Dict[str, List[float]]:
    """Generates metrics (R^2, coefficient of determination) from predictions.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing labels, predictions,
            and optionally uncertainty, as returned by sf.stats.df_from_pred()

    Keyword args:
        label (str, optional): Label prefix/suffix for ROCs.
            Defaults to an empty string.
        level (str, optional): Group-level for the predictions. Used for
            labeling plots. Defaults to 'tile'.
        data_dir (str, optional): Path to data directory for saving.
            Defaults to None.
        neptune_run (:class:`neptune.Run`, optional): Neptune run in which to
            log results. Defaults to None.

    Returns:
        Dict containing metrics, with the key 'r_squared'.
    """

    label_end = "" if label == '' else f"_{label}"

    # Detect the outcome names
    outcome_names = [c[:-7] for c in df.columns if c.endswith('-y_pred')]
    _outcomes_by_true = [c[:-7] for c in df.columns if c.endswith('-y_true')]
    if ((sorted(outcome_names) != sorted(_outcomes_by_true))
       or not len(outcome_names)):
        raise ValueError("Improperly formatted dataframe to linear_metrics(); "
                         "could not detect outcome names. Ensure that "
                         "prediction columns end in '-y_pred' and ground-truth "
                         "columns end in '-y_true'. Try setting column names "
                         "with slideflow.stats.name_columns(). "
                         f"DataFrame columns: {list(df.columns)}")

    # Calculate metrics
    y_pred_cols = [f'{o}-y_pred' for o in outcome_names]
    y_true_cols = [f'{o}-y_true' for o in outcome_names]
    r_squared = sf.stats.plot.scatter(
        df[y_true_cols].values,
        df[y_pred_cols].values,
        data_dir,
        f"{label_end}_by_{level}",
        neptune_run=neptune_run
    )

    # Show results
    for o, r in zip(outcome_names, r_squared):
        r_str = "NA" if not r else f'{r:.3f}'
        log.info(f"[green]{o}[/]: R-squared ({level}-level): {r_str}")

    return {
        'r_squared': r_squared,
    }


def metrics_from_dataset(
    model: Union["tf.keras.Model", "torch.nn.Module"],
    model_type: str,
    patients: Dict[str, str],
    dataset: Union["tf.data.Dataset", "torch.utils.data.DataLoader"],
    num_tiles: int = 0,
    outcome_names: Optional[List[str]] = None,
    reduce_method: str = 'average',
    label: str = '',
    save_predictions: Union[str, bool] = False,
    data_dir: str = '',
    uq: bool = False,
    loss: Optional[Callable] = None,
    torch_args: Optional[SimpleNamespace] = None,
    **kwargs
) -> Tuple[Dict, float, float]:

    """Evaluate performance of a given model on a given TFRecord dataset,
    generating a variety of statistical outcomes and graphs.

    Args:
        model (tf.keras.Model or torch.nn.Module): Keras/Torch model to eval.
        model_type (str): 'categorical', 'linear', or 'cph'.
        patients (dict): Dictionary mapping slidenames to patients.
        dataset (tf.data.Dataset or torch.utils.data.DataLoader): Dataset.
        num_tiles (int, optional): Number of total tiles expected in dataset.
            Used for progress bar. Defaults to 0.

    Keyword args:
        outcome_names (list, optional): List of str, names for outcomes.
            Defaults to None.
        reduce_method (str, optional): Reduction method for calculating
            slide-level and patient-level predictions for categorical outcomes.
            Either 'average' or 'proportion'. If 'average', will reduce with
            average of each logit across tiles. If 'proportion', will convert
            tile predictions into onehot encoding then reduce by averaging
            these onehot values. Defaults to 'average'.
        label (str, optional): Label prefix/suffix for saving.
            Defaults to None.
        save_predictions (bool, optional): Save tile, slide, and patient-level
            predictions to CSV. Defaults to True.
        data_dir (str): Path to data directory for saving.
            Defaults to empty string (current directory).
        neptune_run (:class:`neptune.Run`, optional): Neptune run in which to
            log results. Defaults to None.

    Returns:
        metrics [dict], accuracy [float], loss [float]
    """
    _assert_model_type(model_type)
    dfs, acc, total_loss = eval_dataset(
        model,
        dataset,
        model_type,
        uq=uq,
        loss=loss,
        num_tiles=num_tiles,
        patients=patients,
        outcome_names=outcome_names,
        reduce_method=reduce_method,
        torch_args=torch_args,
    )

    # Save predictions
    if save_predictions:
        if isinstance(save_predictions, str):
            fmt_kw = dict(format=save_predictions)
        else:
            fmt_kw = {}  # type: ignore
        save_dfs(dfs, outdir=data_dir, label=label, **fmt_kw)

    # Calculate metrics
    def metrics_by_level(metrics_function):
        return _merge_metrics({
            level: metrics_function(
                _df,
                level=level,
                data_dir=data_dir,
                label=label,
                **kwargs
            ) for level, _df in dfs.items()
        })

    if model_type == 'categorical':
        metrics = metrics_by_level(categorical_metrics)
    elif model_type == 'linear':
        metrics = metrics_by_level(linear_metrics)
    else:
        metrics = metrics_by_level(cph_metrics)

    log.debug(f'Metrics generation complete.')
    return metrics, acc, total_loss


def name_columns(
    df: DataFrame,
    model_type: str,
    outcome_names: Optional[List[str]] = None
):
    """Renames columns in a DataFrame to correspond to the given outcome names.

    Assumes the DataFrame supplied was generated by sf.stats.df_from_pred().

    Args:
        df (DataFrame): DataFrame from sf.stats.df_from_pred(), containing
            predictions and labels.
        model_type (str): Type of model ('categorical', 'linear', or 'cph').
        outcome_names (list(str)), optional): Outcome names to apply to the
            DataFrame. If this is from a CPH model, the standard names "time"
            and "event" will be used.

    Raises:
        ValueError: If outcome_names are not supplied and it is not a CPH model.
        errors.StatsError: If the length of outcome_names is incompatible
            with the DataFrame.

    Returns:
        DataFrame: DataFrame with renamed columns.
    """
    _assert_model_type(model_type)

    if outcome_names is None and model_type != 'cph':
        raise ValueError("Must supply outcome names for categorical "
                         "or linear models.")
    if (not isinstance(outcome_names, (list, tuple))
       and outcome_names is not None):
        outcome_names = [outcome_names]

    if model_type == 'categorical' and outcome_names is not None:
        # Update dataframe column names with outcome names
        outcome_cols_to_replace = {}
        for oi, outcome in enumerate(outcome_names):
            outcome_cols_to_replace.update({
                c: c.replace(f'out{oi}', outcome)
                for c in df.columns
                if c.startswith(f'out{oi}-')
            })
        df = df.rename(columns=outcome_cols_to_replace)

    elif model_type == 'linear':
        n_outcomes = len([c for c in df.columns if c.startswith('out0-y_pred')])
        if not outcome_names:
            outcome_names = [f"Outcome {i}" for i in range(n_outcomes)]
        elif len(outcome_names) != n_outcomes:
            raise errors.StatsError(
                f"Number of outcome names {len(outcome_names)} does not "
                f"match y_true {n_outcomes}"
            )

        # Rename columns
        outcome_cols_to_replace = {}
        def replace_dict(target, oi, ending_not_needed=False):
            return {
                c: f'{outcome}-{target}'
                for c in df.columns
                if c.startswith(f'out0-{target}') and (c.endswith(str(oi))
                                                        or ending_not_needed)
            }
        for oi, outcome in enumerate(outcome_names):
            outcome_cols_to_replace.update(replace_dict(
                'y_true', oi, ending_not_needed=(len(outcome_names) == 1)
            ))
            outcome_cols_to_replace.update(replace_dict('y_pred', oi))
            outcome_cols_to_replace.update(replace_dict('uncertainty', oi))
        df = df.rename(columns=outcome_cols_to_replace)

    else:
        df = df.rename(columns={
            'out0-y_pred0': 'time-y_pred',
            'out0-y_pred1': 'event-y_true',
            'out0-y_true0': 'time-y_true',

        })
    return df


def predict_from_dataset(*args, **kwargs):
    warnings.warning(
        "`sf.stats.metrics.predict_from_dataset() is deprecated. Please use "
        "`sf.stats.metrics.predict_dataset()` instead.",
        DeprecationWarning)
    return predict_dataset(*args, **kwargs)


def predict_dataset(
    model: Union["tf.keras.Model", "torch.nn.Module"],
    dataset: Union["tf.data.Dataset", "torch.utils.data.DataLoader"],
    model_type: str,
    num_tiles: int = 0,
    uq: bool = False,
    uq_n: int = 30,
    reduce_method: str = 'average',
    patients: Optional[Dict[str, str]] = None,
    outcome_names: Optional[List[str]] = None,
    torch_args: Optional[SimpleNamespace] = None,
) -> Dict[str, DataFrame]:
    """Generates predictions from model and dataset.

    Args:
        model (str): Path to PyTorch model.
        dataset (tf.data.Dataset): PyTorch dataloader.
        model_type (str, optional): 'categorical', 'linear', or 'cph'.
            If multiple linear outcomes are present, y_true is stacked into a
            single vector for each image. Defaults to 'categorical'.
        num_tiles (int, optional): Used for progress bar with Tensorflow.
            Defaults to 0.
        uq_n (int, optional): Number of forward passes to perform
            when calculating MC Dropout uncertainty. Defaults to 30.
        reduce_method (str, optional): Reduction method for calculating
            slide-level and patient-level predictions for categorical outcomes.
            Either 'average' or 'proportion'. If 'average', will reduce with
            average of each logit across tiles. If 'proportion', will convert
            tile predictions into onehot encoding then reduce by averaging
            these onehot values. Defaults to 'average'.
        patients (dict, optional): Dictionary mapping slide names to patient
            names. Required for generating patient-level metrics.
        outcome_names (list, optional): List of str, names for outcomes.
            Defaults to None (outcomes will not be named).
        torch_args (namespace): Used for PyTorch backend. Namespace containing
            num_slide_features and slide_input.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with keys 'tile', 'slide', and
        'patient', and values containing DataFrames with tile-, slide-,
        and patient-level predictions.
    """
    if model_type != 'categorical' and reduce_method != 'average':
        raise ValueError(
            f'Reduction method {reduce_method} incompatible with '
            f'model_type {model_type}'
        )

    if sf.model.is_tensorflow_model(model):
        from slideflow.model import tensorflow_utils
        df = tensorflow_utils.predict_from_model(
            model,
            dataset,
            num_tiles=num_tiles,
            uq=uq,
            uq_n=uq_n,
        )
    else:
        from slideflow.model import torch_utils
        df = torch_utils.predict_from_model(
            model,
            dataset,
            model_type,
            torch_args=torch_args,
            uq=uq,
            uq_n=uq_n,
        )
    if outcome_names is not None or model_type == 'cph':
        df = name_columns(df, model_type, outcome_names)
    return group_reduce(df, method=reduce_method, patients=patients)


def save_dfs(
    dfs: Dict[str, DataFrame],
    format: str = 'parquet',
    outdir: str = '',
    label: str = ''
) -> None:
    """Save DataFrames of predictions to files."""
    label_end = f'_{label}' if label else ''
    for level, _df in dfs.items():
        path = join(outdir, f"{level}_predictions{label_end}")

        # Convert half-floats to float32
        half_floats = _df.select_dtypes(include='float16')
        _df[half_floats.columns] = half_floats.astype('float32')

        if format == 'csv':
            _df.to_csv(path+'.csv')
        elif format == 'feather':
            import pyarrow.feather as feather
            feather.write_feather(_df, path+'.feather')
        else:
            _df.to_parquet(path+'.parquet.gzip', compression='gzip')
