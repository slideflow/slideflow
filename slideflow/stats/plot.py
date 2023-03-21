import os
import warnings
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
from sklearn import metrics
from slideflow import errors
from slideflow.util import log

if TYPE_CHECKING:
    import neptune.new as neptune


def combined_roc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str,
    labels: List[Union[str, int]],
    name: str = 'ROC',
    neptune_run: Optional["neptune.Run"] = None
) -> List[float]:
    """Generates and saves overlapping ROCs.

    Args:
        y_true (np.ndarray): y_true array of shape = (n_curves, n_samples).
        y_pred (np.ndarray): y_pred array of shape = (n_curves, n_samples).
        save_dir (str, optional): Path in which to save ROC curves.
            Defaults to None.
        labels (list(str)): Labels for each plotted curve.
        name (str, optional): Name for plots. Defaults to 'ROC'.
        neptune_run (neptune.Run, optional): Neptune run for saving plots.
            Defaults to None.

    Returns:
        List[float]:  AUROC for each curve.
    """
    import matplotlib.pyplot as plt

    plt.clf()
    plt.title(name)
    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    aurocs = []
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        fpr, tpr, threshold = metrics.roc_curve(yt, yp)
        roc_auc = metrics.auc(fpr, tpr)
        aurocs += [roc_auc]
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
    return aurocs


def histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subsample: int = 500,
) -> None:
    """Generates histogram of y_pred, labeled by y_true, saving to outdir.

    Args:
        y_true (np.ndarray): y_true array.
        y_pred (np.ndarray): y_pred array.
        subsample (int, optional): Subsample data. Defaults to 500.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

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


def prc(
    precision: np.ndarray,
    recall: np.ndarray,
    label: Optional[str] = None
):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.title('Precision-Recall Curve')
    plt.plot(precision, recall, 'b', label=label)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Recall')
    plt.xlabel('Precision')


def roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    label: Optional[str] = None
):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label=label)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('TPR')
    plt.xlabel('FPR')


def scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_dir: str,
    name: str = '_plot',
    neptune_run: Optional["neptune.Run"] = None
) -> List[float]:
    """Generate and save scatter plots, and calculate R^2 (coefficient
    of determination) for each outcome.

    Args:
        y_true (np.ndarray): 2D array of labels. Observations are in first
            dimension, second dim is the outcome.
        y_pred (np.ndarray): 2D array of predictions.
        data_dir (str): Path to directory in which to save plots.
        name (str, optional): Label for filename. Defaults to '_plot'.
        neptune_run (optional): Neptune Run. If provided, will upload plot.

    Returns:
        List[float]:    R squared for each outcome.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    if y_true.shape != y_pred.shape:
        m = f"Shape mismatch: y_true {y_true.shape} y_pred: {y_pred.shape}"
        raise errors.StatsError(m)
    if y_true.shape[0] < 2:
        raise errors.StatsError("Only one observation provided, need >1")
    r_squared = []

    # Subsample to n=1000 for plotting
    if y_true.shape[0] > 1000:
        idx = np.random.choice(range(y_true.shape[0]), 1000)
        yt_sub = y_true[idx]
        yp_sub = y_pred[idx]
    else:
        yt_sub = y_true
        yp_sub = y_pred

    # Perform scatter for each outcome
    for i in range(y_true.shape[1]):
        r_squared += [metrics.r2_score(y_true[:, i], y_pred[:, i])]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            p = sns.jointplot(x=yt_sub[:, i], y=yp_sub[:, i], kind="reg")
        p.set_axis_labels('y_true', 'y_pred')
        plt.savefig(os.path.join(data_dir, f'Scatter{name}-{i}.png'))
        if neptune_run:
            neptune_run[f'results/graphs/Scatter{name}-{i}'].upload(
                os.path.join(data_dir, f'Scatter{name}-{i}.png')
            )
        plt.close()
    return r_squared
