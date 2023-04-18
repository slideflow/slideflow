import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning
from slideflow.util import log

from . import errors, utils


def plot_uncertainty(df, kind, threshold=None, title=None):
    """Plots figure of tile or slide-level predictions vs. uncertainty.

    Args:
        df (pandas.DataFrame): Processed dataframe containing columns
            'uncertainty', 'correct', 'y_pred'.
        kind (str): Kind of plot. If 'tile', subsample to only 1000 points.
            Included in title.
        threshold (float, optional): Uncertainty threshold.
            Defaults to None.
        title (str, optional): Title for plots. Defaults to None.

    Returns:
        None
    """
    try:
        from skmisc.loess import loess
    except ImportError:
        raise ImportError(
            "Uncertainty plots with loess estimation require scikit-misc, "
            "which is not installed."
        )

    # Subsample tile-level predictions
    if kind == 'tile':
        df = df.sample(n=1000)

    f, axes = plt.subplots(1, 3)
    f.set_size_inches(15, 5)
    palette = sns.color_palette("Set2")
    tf_pal = {True: palette[0], False: palette[1]}

    # Left figure - KDE -------------------------------------------------------
    kde = sns.kdeplot(
        x='uncertainty',
        hue='correct',
        data=df,
        fill=True,
        palette=tf_pal,
        ax=axes[0]
    )
    kde.set(xlabel='Uncertainty')
    axes[0].title.set_text(f'Uncertainty density ({kind}-level)')

    # Middle figure - Scatter -------------------------------------------------

    # - Above threshold
    if threshold is not None:
        axes[1].axhline(y=threshold, color='r', linestyle='--')
        at_df = df.loc[(df['uncertainty'] >= threshold)]
        c_a_df = at_df.loc[at_df['correct']]
        ic_a_df = at_df.loc[~at_df['correct']]
        axes[1].scatter(
            x=c_a_df['y_pred'],
            y=c_a_df['uncertainty'],
            marker='o',
            s=10,
            color='gray'
        )
        axes[1].scatter(
            x=ic_a_df['y_pred'],
            y=ic_a_df['uncertainty'],
            marker='x',
            color='#FC6D77'
        )
    # - Below threshold
    if threshold is not None:
        bt_df = df.loc[(df['uncertainty'] < threshold)]
    else:
        bt_df = df
    c_df = bt_df.loc[bt_df['correct']]
    ic_df = bt_df.loc[~bt_df['correct']]
    axes[1].scatter(
        x=c_df['y_pred'],
        y=c_df['uncertainty'],
        marker='o',
        s=10
    )
    axes[1].scatter(
        x=ic_df['y_pred'],
        y=ic_df['uncertainty'],
        marker='x',
        color='red'
    )
    if title is not None:
        axes[1].title.set_text(title)

    # Right figure - probability calibration ----------------------------------
    l_df = df[['uncertainty', 'correct']].sort_values(by=['uncertainty'])
    x = l_df['uncertainty'].to_numpy()
    y = l_df['correct'].astype(float).to_numpy()
    ol = loess(x, y)
    ol.fit()
    pred = ol.predict(x, stderror=True)
    conf = pred.confidence()
    z = pred.values
    ll = conf.lower
    ul = conf.upper
    axes[2].plot(x, y, '+', ms=6)
    axes[2].plot(x, z)
    axes[2].fill_between(x, ll, ul, alpha=.2)
    axes[2].tick_params(labelrotation=90)
    axes[2].set_ylim(-0.1, 1.1)
    if threshold is not None:
        axes[2].axvline(x=threshold, color='r', linestyle='--')

    # - Figure style
    for ax in (axes[1], axes[2]):
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['bottom'].set_color('black')
        ax.tick_params(axis='x', colors='black')
        ax.grid(visible=True, which='both', axis='both', color='white')
        ax.set_facecolor('#EAEAF2')


def process_tile_predictions(df, pred_thresh=0.5, patients=None):
    '''Load and process tile-level predictions from CSV.

    Args:
        df (pandas.DataFrame): Unprocessed DataFrame from reading tile-level
            predictions.
        pred_thresh (float or str, optional): Tile-level prediction threshold.
            If 'detect', will auto-detect via Youden's J. Defaults to 0.5.
        patients (dict, optional): Dict mapping slides to patients, used for
            patient-level thresholding. Defaults to None.

    Returns:
        pandas.DataFrame, tile prediction threshold
    '''

    # Tile-level AUC
    if np.isnan(df['y_pred'].to_numpy()).sum():
        raise errors.PredsContainNaNError
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        fpr, tpr, thresh = metrics.roc_curve(
            df['y_true'].to_numpy(),
            df['y_pred'].to_numpy()
        )
        tile_auc = metrics.auc(fpr, tpr)
    try:
        max_j = max(zip(tpr, fpr), key=lambda x: x[0]-x[1])
        opt_pred = thresh[list(zip(tpr, fpr)).index(max_j)]
    except ValueError:
        log.debug("Unable to calculate tile prediction threshold; using 0.5")
        opt_pred = 0.5

    if pred_thresh == 'detect':
        log.debug(f"Auto-detected tile prediction threshold: {opt_pred:.4f}")
        pred_thresh = opt_pred
    else:
        log.debug(f"Using tile prediction threshold: {pred_thresh:.4f}")

    if patients is not None:
        df['patient'] = df['slide'].map(patients)
    else:
        log.warn('Patients not provided; assuming 1:1 slide:patient mapping')

    log.debug(f'Tile AUC: {tile_auc:.4f}')
    # Calculate tile-level prediction accuracy
    df['error'] = abs(df['y_true'] - df['y_pred'])
    df['correct'] = (
        ((df['y_pred'] < pred_thresh) & (df['y_true'] == 0))
        | ((df['y_pred'] >= pred_thresh) & (df['y_true'] == 1))
    )
    df['incorrect'] = (~df['correct']).astype(int)
    df['y_pred_bin'] = (df['y_pred'] >= pred_thresh).astype(int)
    return df, pred_thresh


def process_group_predictions(df, pred_thresh, level):
    '''From a given dataframe of tile-level predictions, calculate group-level
    predictions and uncertainty.'''

    if any(c not in df.columns for c in ('y_true', 'y_pred', 'uncertainty')):
        raise ValueError('Missing columns. Expected y_true, y_pred, uncertainty.'
                         f'Got: {df.columns}')

    # Calculate group-level predictions
    log.debug(f'Calculating {level}-level means from {len(df)} predictions')
    levels = [l for l in pd.unique(df[level]) if l is not np.nan]
    reduced_df = df[[level, 'y_pred', 'y_true', 'uncertainty']]
    grouped = reduced_df.groupby(level, as_index=False).mean()
    yp = np.array([
        grouped.loc[grouped[level] == lev]['y_pred'].to_numpy()[0]
        for lev in levels
    ])
    yt = np.array([
        grouped.loc[grouped[level] == lev]['y_true'].to_numpy()[0]
        for lev in levels
    ], dtype=np.uint8)
    u = np.array([
        grouped.loc[grouped[level] == lev]['uncertainty'].to_numpy()[0]
        for lev in levels
    ])
    if not len(yt):
        raise errors.ROCFailedError("Unable to generate ROC; preds are empty.")

    # Slide-level AUC
    log.debug(f'Calculating {level}-level ROC')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        l_fpr, l_tpr, l_thresh = metrics.roc_curve(yt, yp)
        log.debug('Calculating AUC')
        level_auc = metrics.auc(l_fpr, l_tpr)
    log.debug('Calculating optimal threshold')

    if pred_thresh == 'detect':
        try:
            max_j = max(zip(l_tpr, l_fpr), key=lambda x: x[0]-x[1])
            pred_thresh = l_thresh[list(zip(l_tpr, l_fpr)).index(max_j)]
        except ValueError:
            raise errors.ROCFailedError(f"Unable to generate {level}-level ROC")
        log.debug(f"Using detected prediction threshold: {pred_thresh:.4f}")
    else:
        log.debug(f"Using {level} prediction threshold: {pred_thresh:.4f}")
    log.debug(f'{level} AUC: {level_auc:.4f}')

    correct = (((yp < pred_thresh) & (yt == 0))
               | ((yp >= pred_thresh) & (yt == 1)))
    incorrect = pd.Series(
        ((yp < pred_thresh) & (yt == 1))
        | ((yp >= pred_thresh) & (yt == 0))
    ).astype(int)

    l_df = pd.DataFrame({
        level: pd.Series(levels),
        'error': pd.Series(abs(yt - yp)),
        'uncertainty': pd.Series(u),
        'correct': correct,
        'incorrect': incorrect,
        'y_true': pd.Series(yt),
        'y_pred': pd.Series(yp),
        'y_pred_bin': pd.Series(yp >= pred_thresh).astype(int)
    })
    return l_df, pred_thresh


def apply(df, tile_uq, slide_uq, tile_pred=0.5,
          slide_pred=0.5, plot=False, keep='high_confidence',
          title=None, patients=None, level='slide'):

    '''Apply pre-calculcated tile- and group-level uncertainty thresholds.

    Args:
        df (pandas.DataFrame): Must contain columns 'y_true', 'y_pred',
            and 'uncertainty'.
        tile_uq (float): Tile-level uncertainty threshold.
        slide_uq (float): Slide-level uncertainty threshold.
        tile_pred (float, optional): Tile-level prediction threshold.
            Defaults to 0.5.
        slide_pred (float, optional): Slide-level prediction threshold.
            Defaults to 0.5.
        plot (bool, optional): Plot slide-level uncertainty. Defaults to False.
        keep (str, optional): Either 'high_confidence' or 'low_confidence'.
            Cohort to keep after thresholding. Defaults to 'high_confidence'.
        title (str, optional): Title for uncertainty plot. Defaults to None.
        patients (dict, optional): Dictionary mapping slides to patients. Adds
            a 'patient' column in the tile prediction dataframe, enabling
            patient-level thresholding. Defaults to None.
        level (str, optional): Either 'slide' or 'patient'. Level at which to
            apply threshold. If 'patient', requires patient dict be supplied.
            Defaults to 'slide'.

    Returns:
        Dictionary of results, with keys auc, percent_incl, accuracy,
            sensitivity, and specificity

        DataFrame of thresholded group-level predictions
    '''

    assert keep in ('high_confidence', 'low_confidence')
    assert not (level == 'patient' and patients is None)

    log.debug(f"Applying tile UQ threshold of {tile_uq:.5f}")
    if patients:
        df['patient'] = df['slide'].map(patients)
    log.debug(f"Number of {level}s before tile UQ filter: {pd.unique(df[level]).shape[0]}")
    log.debug(f"Number of tiles before tile-level filter: {len(df)}")

    df, _ = process_tile_predictions(
        df,
        pred_thresh=tile_pred,
        patients=patients
    )
    num_pre_filter = pd.unique(df[level]).shape[0]

    if tile_uq:
        df = df[df['uncertainty'] < tile_uq]

    log.debug(f"Number of {level}s after tile-level filter: {pd.unique(df[level]).shape[0]}")
    log.debug(f"Number of tiles after tile-level filter: {len(df)}")

    # Build group-level predictions
    try:
        s_df, _ = process_group_predictions(
            df,
            pred_thresh=slide_pred,
            level=level
        )
    except errors.ROCFailedError:
        log.error("Unable to process slide predictions")
        empty_results = {k: None for k in ['auc',
                                           'percent_incl',
                                           'acc',
                                           'sensitivity',
                                           'specificity']}
        return empty_results, None

    if plot:
        plot_uncertainty(s_df, threshold=slide_uq, kind=level, title=title)

    # Apply slide-level thresholds
    if slide_uq:
        log.debug(f"Using {level} uncertainty threshold of {slide_uq:.5f}")
        if keep == 'high_confidence':
            s_df = s_df.loc[s_df['uncertainty'] < slide_uq]
        elif keep == 'low_confidence':
            s_df = s_df.loc[s_df['uncertainty'] >= slide_uq]
        else:
            raise Exception(f"Unknown keep option {keep}")

    # Show post-filtering group-level predictions and AUC
    auc = utils.auc(s_df['y_true'].to_numpy(), s_df['y_pred'].to_numpy())
    num_post_filter = len(s_df)
    percent_incl = num_post_filter / num_pre_filter
    log.debug(f"Percent {level} included: {percent_incl*100:.2f}%")

    # Calculate post-thresholded sensitivity/specificity
    y_true = s_df['y_true'].to_numpy().astype(bool)
    y_pred = s_df['y_pred'].to_numpy() > slide_pred

    tp = np.logical_and(y_true, y_pred).sum()
    fp = np.logical_and(np.logical_not(y_true), y_pred).sum()
    tn = np.logical_and(np.logical_not(y_true), np.logical_not(y_pred)).sum()
    fn = np.logical_and(y_true, np.logical_not(y_pred)).sum()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    log.debug(f"Accuracy: {acc:.4f}")
    log.debug(f"Sensitivity: {sensitivity:.4f}")
    log.debug(f"Specificity: {specificity:.4f}")

    results = {
        'auc': auc,
        'percent_incl': percent_incl,
        'acc': acc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }
    return results, s_df


def detect(df, tile_uq='detect', slide_uq='detect', tile_pred='detect',
           slide_pred='detect', plot=False, patients=None):
    '''Detect optimal tile- and slide-level uncertainty thresholds.

    Args:
        df (pandas.DataFrame): Tile-level predictions. Must contain columns
            'y_true', 'y_pred', and 'uncertainty'.
        tile_uq (str or float): Either 'detect' or float. If 'detect',
            will detect tile-level uncertainty threshold. If float, will use
            the specified tile-level uncertainty threshold.
        slide_uq (str or float): Either 'detect' or float. If 'detect',
            will detect slide-level uncertainty threshold. If float, will use
            the specified slide-level uncertainty threshold.
        tile_pred (str or float): Either 'detect' or float. If 'detect',
            will detect tile-level prediction threshold. If float, will use the
            specified tile-level prediction threshold.
        slide_pred (str or float): Either 'detect' or float. If 'detect'
            will detect slide-level prediction threshold. If float, will use
            the specified slide-level prediction threshold.
        plot (bool, optional): Plot slide-level uncertainty. Defaults to False.
        patients (dict, optional): Dict mapping slides to patients. Required
            for patient-level thresholding.

    Returns:
        Dictionary with tile- and slide-level UQ and prediction threhsolds,
            with keys: 'tile_uq', 'tile_pred', 'slide_uq', 'slide_pred'

        Float: Slide-level AUROC
    '''

    log.debug("Detecting thresholds...")
    empty_thresh = {k: None
                    for k in ['tile_uq', 'slide_uq', 'tile_pred', 'slide_pred']}
    try:
        df, detected_tile_pred = process_tile_predictions(
            df,
            pred_thresh=tile_pred,
            patients=patients
        )
    except errors.PredsContainNaNError:
        log.error("Tile-level predictions contain NaNs; unable to process.")
        return empty_thresh, None

    if tile_pred == 'detect':
        tile_pred = detected_tile_pred

    # Tile-level ROC and Youden's J
    if isinstance(tile_uq, (float, np.float16, np.float32, np.float64)):
        df = df[df['uncertainty'] < tile_uq]
    elif tile_uq != 'detect':
        log.debug("Not performing tile-level uncertainty thresholding.")
        tile_uq = None
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            t_fpr, t_tpr, t_thresh = metrics.roc_curve(
                df['incorrect'].to_numpy(),
                df['uncertainty'].to_numpy()
            )
        max_j = max(zip(t_tpr, t_fpr), key=lambda x: x[0] - x[1])
        tile_uq = t_thresh[list(zip(t_tpr, t_fpr)).index(max_j)]
        log.debug(f"Tile-level optimal UQ threshold: {tile_uq:.4f}")
        df = df[df['uncertainty'] < tile_uq]

    slides = list(set(df['slide']))
    log.debug(f"Number of slides after filter: {len(slides)}")
    log.debug(f"Number of tiles after filter: {len(df)}")

    # Build slide-level predictions
    try:
        s_df, slide_pred = process_group_predictions(
            df,
            pred_thresh=slide_pred,
            level='slide'
        )
    except errors.ROCFailedError:
        log.error("Unable to process slide predictions")
        return empty_thresh, None

    # Slide-level thresholding
    if slide_uq == 'detect':
        if not s_df['incorrect'].to_numpy().sum():
            log.debug("Unable to calculate slide UQ threshold; no incorrect predictions made")
            slide_uq = None
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                s_fpr, s_tpr, s_thresh = metrics.roc_curve(
                    s_df['incorrect'],
                    s_df['uncertainty'].to_numpy()
                )
            max_j = max(zip(s_tpr, s_fpr), key=lambda x: x[0]-x[1])
            slide_uq = s_thresh[list(zip(s_tpr, s_fpr)).index(max_j)]
            log.debug(f"Slide-level optimal UQ threshold: {slide_uq:.4f}")
            if plot:
                plot_uncertainty(s_df, threshold=slide_uq, kind='slide')
            s_df = s_df[s_df['uncertainty'] < slide_uq]
    else:
        log.debug("Not performing slide-level uncertainty thresholding.")
        slide_uq = 0.5
        if plot:
            plot_uncertainty(s_df, threshold=slide_uq, kind='slide')

    # Show post-filtering slide predictions and AUC
    auc = utils.auc(s_df['y_true'].to_numpy(), s_df['y_pred'].to_numpy())
    thresholds = {
        'tile_uq': tile_uq,
        'slide_uq': slide_uq,
        'tile_pred': tile_pred,
        'slide_pred': slide_pred
    }
    return thresholds, auc


def from_cv(dfs, **kwargs):
    '''Finds the optimal tile and slide-level thresholds from a set of nested
    cross-validation experiments.

    Args:
        dfs (list(DataFrame)): List of DataFrames with tile predictions,
            containing headers 'y_true', 'y_pred', 'uncertainty', 'slide',
            and 'patient'.

    Keyword args:
        tile_uq (str or float): Either 'detect' or float. If 'detect',
            will detect tile-level uncertainty threshold. If float, will use
            the specified tile-level uncertainty threshold.
        slide_uq (str or float): Either 'detect' or float. If 'detect',
            will detect slide-level uncertainty threshold. If float, will use
            the specified slide-level uncertainty threshold.
        tile_pred (str or float): Either 'detect' or float. If 'detect',
            will detect tile-level prediction threshold. If float, will use the
            specified tile-level prediction threshold.
        slide_pred (str or float): Either 'detect' or float. If 'detect'
            will detect slide-level prediction threshold. If float, will use
            the specified slide-level prediction threshold.
        plot (bool, optional): Plot slide-level uncertainty. Defaults to False.
        patients (dict, optional): Dict mapping slides to patients. Required
            for patient-level thresholding.

    Returns:
        Dictionary with tile- and slide-level UQ and prediction threhsolds,
            with keys: 'tile_uq', 'tile_pred', 'slide_uq', 'slide_pred'
    '''

    required_cols = ('y_true', 'y_pred', 'uncertainty', 'slide', 'patient')
    k_tile_thresh, k_slide_thresh = [], []
    k_tile_pred_thresh, k_slide_pred_thresh = [], []
    k_auc = []
    skip_tile = ('tile_uq_thresh' in kwargs
                 and kwargs['tile_uq_thresh'] is None)
    skip_slide = ('slide_uq_thresh' in kwargs
                  and kwargs['slide_uq_thresh'] is None)

    for idx, df in enumerate(dfs):
        log.debug(f"Detecting thresholds from fold {idx}")
        if not all(col in df.columns for col in required_cols):
            raise ValueError(
                f"DataFrame missing columns, expected {required_cols}, got: "
                f"{', '.join(df.columns.tolist())}"
            )
        thresholds, auc = detect(df, **kwargs)
        if thresholds['tile_uq'] is None or thresholds['slide_uq'] is None:
            log.debug(f"Skipping CV #{idx}, unable to detect threshold")
            continue

        k_tile_pred_thresh += [thresholds['slide_pred']]
        k_slide_pred_thresh += [thresholds['tile_pred']]
        k_auc += [auc]

        if not skip_tile:
            k_tile_thresh += [thresholds['tile_uq']]
        if not skip_slide:
            k_slide_thresh += [thresholds['slide_uq']]

    if not skip_tile and not len(k_tile_thresh):
        raise errors.ThresholdError('Unable to detect tile UQ threshold.')
    if not skip_slide and not len(k_slide_thresh):
        raise errors.ThresholdError('Unable to detect slide UQ threshold.')

    k_slide_pred_thresh = np.mean(k_slide_pred_thresh)
    k_tile_pred_thresh = np.mean(k_tile_pred_thresh)

    if not skip_tile:
        k_tile_thresh = np.min(k_tile_thresh)
    if not skip_slide:
        k_slide_thresh = np.max(k_slide_thresh)

    return {
        'tile_uq': k_tile_thresh,
        'slide_uq': k_slide_thresh,
        'tile_pred': k_tile_pred_thresh,
        'slide_pred': k_slide_pred_thresh
    }
