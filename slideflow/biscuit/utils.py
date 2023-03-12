import os
from os.path import join
from statistics import mean, variance

import warnings
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import slideflow as sf
from scipy import stats
from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning

from .delong import delong_roc_variance
from .errors import ModelNotFoundError, MultipleModelsFoundError

# -----------------------------------------------------------------------------

def uncertainty_header(outcome, underscore=False):
    return str(outcome) + ('_' if underscore else '-') + 'uncertainty1'


def y_true_header(outcome, underscore=False):
    return str(outcome) + ('_' if underscore else '-') + 'y_true0'


def y_pred_header(outcome, underscore=False):
    return str(outcome) + ('_' if underscore else '-') + 'y_pred1'


def rename_cols(df, outcome, *, y_true=None, y_pred=None, uncertainty=None):
    """Renames columns of dataframe, in place."""
    # Support for using underscore or dashes
    if y_true is None:
        y_true = y_true_header(
            outcome,
            underscore=(y_true_header(outcome, underscore=True) in df.columns))
        if y_true not in df.columns:
            y_true = str(outcome) + '-y_true'
    if y_pred is None:
        y_pred = y_pred_header(
            outcome,
            underscore=(y_pred_header(outcome, underscore=True) in df.columns))
    if uncertainty is None:
        uncertainty = uncertainty_header(
            outcome,
            underscore=(uncertainty_header(outcome, underscore=True) in df.columns))
    new_cols = {
        y_true: 'y_true',
        y_pred: 'y_pred',
        uncertainty: 'uncertainty'
    }
    df.rename(columns=new_cols, inplace=True)

# --- General utility functions -----------------------------------------------

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Truncates matplotlib colormap."""

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def get_model_results(path, epoch, outcome):
    """Reads results/metrics from a trained model.

    Args:
        path (str): Path to model.
        outcome (str): Outcome name.

    Returns:
        Dict of results with the keys: pt_auc, pt_ap, slide_auc, slide_ap,
            tile_auc, tile_ap, opt_thresh
    """
    csv = pd.read_csv(join(path, 'results_log.csv'))
    result_rows = {}
    for i, row in csv.iterrows():
        try:
            row_epoch = int(row['model_name'].split('epoch')[-1])
        except ValueError:
            continue
        result_rows.update({
            row_epoch: row
        })
    if epoch not in result_rows:
        raise ModelNotFoundError(f"Unable to find results for epoch {epoch}")
    model_res = result_rows[epoch]
    pt_ap = mean(eval(model_res['patient_ap'])[outcome])
    pt_auc = eval(model_res['patient_auc'])[outcome][0]
    slide_ap = mean(eval(model_res['slide_ap'])[outcome])
    slide_auc = eval(model_res['slide_auc'])[outcome][0]
    tile_ap = mean(eval(model_res['tile_ap'])[outcome])
    tile_auc = eval(model_res['tile_auc'])[outcome][0]

    pred_path = join(
        path,
        f'patient_predictions_{outcome}_val_epoch{epoch}.csv'
    )
    if os.path.exists(pred_path):
        _, opt_thresh = auc_and_threshold(*read_group_predictions(pred_path))
    else:
        try:
            parquet_path = join(path, 'patient_predictions_val_epoch1.parquet.gzip')
            _, opt_thresh = auc_and_threshold(*read_group_predictions(parquet_path))
        except OSError:
            opt_thresh = None
    return {
        'pt_auc': pt_auc,
        'pt_ap': pt_ap,
        'slide_auc': slide_auc,
        'slide_ap': slide_ap,
        'tile_auc': tile_auc,
        'tile_ap': tile_ap,
        'opt_thresh': opt_thresh
    }


def get_eval_results(path, outcome):
    """Reads results/metrics from a trained model.

    Args:
        path (str): Path to model.
        outcome (str): Outcome name.

    Returns:
        Dict of results with the keys: pt_auc, pt_ap, slide_auc, slide_ap,
            tile_auc, tile_ap, opt_thresh
    """
    csv = pd.read_csv(join(path, 'results_log.csv'))
    for i, row in csv.iterrows():
        model_res = row
    pt_ap = mean(eval(model_res['patient_ap'])[outcome])
    pt_auc = eval(model_res['patient_auc'])[outcome][0]
    slide_ap = mean(eval(model_res['slide_ap'])[outcome])
    slide_auc = eval(model_res['slide_auc'])[outcome][0]
    tile_ap = mean(eval(model_res['tile_ap'])[outcome])
    tile_auc = eval(model_res['tile_auc'])[outcome][0]

    pred_path = join(
        path,
        f'patient_predictions_{outcome}_eval.csv'
    )
    if os.path.exists(pred_path):
        _, opt_thresh = auc_and_threshold(*read_group_predictions(pred_path))
    else:
        try:
            parquet_path = join(path, 'patient_predictions_eval.parquet.gzip')
            _, opt_thresh = auc_and_threshold(*read_group_predictions(parquet_path))
        except OSError:
            opt_thresh = None
    return {
        'pt_auc': pt_auc,
        'pt_ap': pt_ap,
        'slide_auc': slide_auc,
        'slide_ap': slide_ap,
        'tile_auc': tile_auc,
        'tile_ap': tile_ap,
        'opt_thresh': opt_thresh
    }


def find_cv_early_stop(project, label, outcome, k=3):
    """Detects early stop batch from cross-val trained models.

    Args:
        project (slideflow.Project): Project.
        label (str): Experimental label.
        k (int, optional): Number of k-fold iterations. Defaults to 3.
        outcome (str): Outcome name.

    Returns:
        int: Early stop batch.
    """
    cv_folders = find_cv(project, label, k=k, outcome=outcome)
    early_stop_batch = []
    for cv_folder in cv_folders:
        csv = pd.read_csv(join(cv_folder, 'results_log.csv'))
        model_res = next(csv.iterrows())[1]
        if 'early_stop_batch' in model_res:
            early_stop_batch += [model_res['early_stop_batch']]
    if len(early_stop_batch) == len(cv_folders):
        # Only returns early stop if it was triggered in all crossfolds
        return round(mean(early_stop_batch))
    else:
        return None


def df_from_cv(project, label, outcome, epoch=None, k=3, y_true=None,
               y_pred=None, uncertainty=None):
    """Loads tile predictions from cross-fold models & renames columns.

    Args:
        project (sf.Project): Slideflow project.
        label (str): Experimental label.
        epoch (int, optional): Epoch number of saved model. Defaults to None.
        k (int, optional): K-fold iteration. Defaults to 3.
        outcome (str, optional): Outcome name.
        y_true (str, optional): Column name for ground truth labels.
            Defaults to {outcome}_y_true0.
        y_pred (str, optional): Column name for predictions.
            Defaults to {outcome}_y_pred1.
        uncertainty (str, optional): Column name for uncertainty.
            Defaults to {outcome}_y_uncertainty1.

    Returns:
        list(DataFrame): DataFrame for each k-fold.
    """
    dfs = []
    model_folders = find_cv(project, label, epoch=epoch, k=k, outcome=outcome)
    patients = project.dataset().patients()
    e = '' if epoch is None else '../'

    for folder in model_folders:
        csv_path = join(folder, f'{e}tile_predictions_val_epoch1.csv')
        parquet_path = join(folder, f'{e}tile_predictions_val_epoch1.parquet.gzip')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        elif os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
        else:
            raise OSError(f"Could not find tile predictions file at {folder}")
        rename_cols(df, outcome, y_true=y_true, y_pred=y_pred, uncertainty=uncertainty)
        if 'patient' not in df.columns:
            df['patient'] = df['slide'].map(patients)
        dfs += [df]
    return dfs


# --- Utility functions for finding experiment models -------------------------

def find_model(project, label, outcome, epoch=None, kfold=None):
    """Searches for a model in a project model directory.

    Args:
        project (slideflow.Project): Project.
        label (str): Experimental label.
        outcome (str): Outcome name.
        epoch (int, optional): Epoch to search for. If not None, returns
            path to the saved model. If None, returns path to parent model
            folder. Defaults to None.
        kfold (int, optional): K-fold iteration. Defaults to None.


    Raises:
        MultipleModelsFoundError: If multiple potential matches are found.
        ModelNotFoundError: If no matching model is found.

    Returns:
        str: Path to matching model.
    """
    tail = '' if kfold is None else f'-kfold{kfold}'
    model_name = f'{outcome}-{label}-HP0{tail}'
    matching = [
        o for o in os.listdir(project.models_dir)
        if o[6:] == model_name
    ]
    if len(matching) > 1:
        raise MultipleModelsFoundError("Multiple matching models found "
                                       f"matching {model_name}")
    elif not len(matching):
        raise ModelNotFoundError("No matching model found matching "
                                 f"{model_name}.")
    elif epoch is not None:
        return join(
            project.models_dir,
            matching[0],
            f'{outcome}-{label}-HP0{tail}_epoch{epoch}'
        )
    else:
        return join(project.models_dir, matching[0])


def model_exists(project, label, outcome, epoch=None, kfold=None):
    """Check if matching model exists.

    Args:
        project (slideflow.Project): Project.
        label (str): Experimental label.
        outcome (str, optional): Outcome name.
        epoch (int, optional): Epoch number of saved model. Defaults to None.
        kfold (int, optional): K-fold iteration. Defaults to None.

    Returns:
        bool: If model exists
    """
    try:
        find_model(project, label, outcome, kfold=kfold, epoch=epoch)
        return True
    except ModelNotFoundError:
        return False


def find_cv(project, label, outcome, epoch=None, k=3):
    """Finds paths to cross-validation models.

    Args:
        project (slideflow.Project): Project.
        label (str): Experimental label.
        outcome (str, optional): Outcome name.
        epoch (int, optional): Epoch number of saved model. Defaults to None.
        kfold (int, optional): K-fold iteration. Defaults to None.

    Returns:
        list(str): Paths to cross-validation models.
    """
    return [
        find_model(project, label, outcome, epoch=epoch, kfold=_k)
        for _k in range(1, k+1)
    ]


def find_eval(project, label, outcome, epoch=1):
    """Finds matching eval directory.

    Args:
        project (slideflow.Project): Project.
        label (str): Experimental label.
        outcome (str, optional): Outcome name.
        epoch (int, optional): Epoch number of saved model. Defaults to None.


    Raises:
        MultipleModelsFoundError: If multiple matches are found.
        ModelNotFoundError: If no match is found.

    Returns:
        str: path to eval directory
    """
    matching = [
        o for o in os.listdir(project.eval_dir)
        if o[11:] == f'{outcome}-{label}-HP0_epoch{epoch}'
    ]
    if len(matching) > 1:
        raise MultipleModelsFoundError("Multiple matching eval experiments "
                                       f"found for label {label}")
    elif not len(matching):
        raise ModelNotFoundError(f"No matching eval found for label {label}")
    else:
        return join(project.eval_dir, matching[0])


def eval_exists(project, label, outcome, epoch=1):
    """Check if matching eval exists.

    Args:
        project (slideflow.Project): Project.
        label (str): Experimental label.
        epoch (int, optional): Epoch number of saved model. Defaults to None.

    Returns:
        bool: If eval exists
    """
    try:
        find_eval(project, label, outcome, epoch=epoch)
        return True
    except ModelNotFoundError:
        return False


# --- Thresholding and metrics functions --------------------------------------

def read_group_predictions(path):
    '''Reads patient- or slide-level predictions CSV or parquet file,
    returning y_true and y_pred.

    Expects a binary categorical outcome.

    Compatible with Slideflow 1.1 and 1.2.
    '''
    if not os.path.exists(path):
        raise OSError(f"Could not find predictions file at {path}")
    if sf.util.path_to_ext(path).lower() == 'csv':
        df = pd.read_csv(path)
    elif sf.util.path_to_ext(path).lower() in ('parquet', 'gzip'):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unrecognized extension for prediction file {path}")
    if 'y_true1' in df.columns:
        y_true = df['y_true1'].to_numpy()
    else:
        y_true_cols = [c for c in df.columns if c.endswith('y_true')]
        if len(y_true_cols) == 1:
            y_true = df[y_true_cols[0]].to_numpy()
        else:
            raise ValueError(f"Could not find y_true column at {path}")
    if 'percent_tiles_positive1' in df.columns:
        y_pred = df['percent_tiles_positive1'].to_numpy()
    else:
        y_pred_cols = [c for c in df.columns if 'y_pred' in c]
        if len(y_pred_cols) == 2:
            y_pred = df[y_pred_cols[1]].to_numpy()
        else:
            raise ValueError(f"Expected exactly 2 y_pred columns at {path}; "
                             f"got {len(y_pred_cols)}")
    return y_true, y_pred


def prediction_metrics(y_true, y_pred, threshold):
    """Calculate prediction metrics (AUC, sensitivity/specificity, etc)

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predictions.
        threshold (_type_): Prediction threshold.

    Returns:
        dict: Prediction metrics.
    """
    yt = y_true.astype(bool)
    yp = y_pred > threshold

    alpha = 0.05
    z = stats.norm.ppf((1 - alpha/2))
    tp = np.logical_and(yt, yp).sum()
    fp = np.logical_and(np.logical_not(yt), yp).sum()
    tn = np.logical_and(np.logical_not(yt), np.logical_not(yp)).sum()
    fn = np.logical_and(yt, np.logical_not(yp)).sum()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Youden's confidence interval, via BAC (bootstrap AC estimate)
    # Bootstrapping performed with sample size n = 100 and iterations B = 500
    all_jac = []
    for _ in range(500):
        bootstrap_i = np.random.choice(np.arange(yt.shape[0]), size=(150,))
        _yt = yt[bootstrap_i]
        _yp = yp[bootstrap_i]
        _tp = np.logical_and(_yt, _yp).sum()
        _fp = np.logical_and(np.logical_not(_yt), _yp).sum()
        _tn = np.logical_and(np.logical_not(_yt), np.logical_not(_yp)).sum()
        _fn = np.logical_and(_yt, np.logical_not(_yp)).sum()
        _jac = (((_tn + 0.5 * z**2) / (_tn + _fp + z**2))
                - ((_fn + 0.5 * z**2) / (_fn + _tp + z**2)))
        all_jac += [_jac]

    jac = mean(all_jac)
    jac_var = variance(all_jac)
    jac_low = jac - z * np.sqrt(jac_var)
    jac_high = jac + z * np.sqrt(jac_var)

    # AUC confidence intervals
    if not np.array_equal(np.unique(y_true), [0, 1]):
        sf.util.log.warn("Unable to calculate CI; NaNs exist")
        ci = [None, None]
    else:
        delong_auc, auc_cov = delong_roc_variance(y_true, y_pred)
        auc_std = np.sqrt(auc_cov)
        lower_upper_q = np.abs(np.array([0, 1]) - alpha / 2)
        ci = stats.norm.ppf(lower_upper_q, loc=delong_auc, scale=auc_std)
        ci[ci > 1] = 1

    return {
        'auc_low': ci[0],
        'auc_high': ci[1],
        'acc': acc,
        'sens': sensitivity,
        'spec': specificity,
        'youden': sensitivity + specificity - 1,
        'youden_low': jac_low,
        'youden_high': jac_high,
    }


def auc_and_threshold(y_true, y_pred):
    """Calculates AUC and optimal threshold (via Youden's J)

    Args:
        y_true (np.ndarray): Y true (labels).
        y_pred (np.ndarray): Y pred (predictions).

    Returns:
        float: AUC
        float: Optimal threshold
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
        roc_auc = metrics.auc(fpr, tpr)
    max_j = max(zip(tpr, fpr), key=lambda x: x[0]-x[1])
    optimal_threshold = threshold[list(zip(tpr, fpr)).index(max_j)]
    return roc_auc, optimal_threshold


def auc(y_true, y_pred):
    """Calculate Area Under Receiver Operator Curve (AUC / AUROC)

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predictions.

    Returns:
        Float: AUC
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        try:
            fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
            return metrics.auc(fpr, tpr)
        except ValueError:
            sf.util.log.warn("Unable to calculate ROC")
            return np.nan
