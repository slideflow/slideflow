"""Training functions for various multi-instance learning (MIL) models."""

import os
import numpy as np
import slideflow as sf
import pandas as pd
from os.path import join, exists
from typing import Union, List, Optional, Dict, Tuple, TYPE_CHECKING
from slideflow import Dataset, log
from slideflow.util import path_to_name
from os.path import join, isdir
import logging

from .. import utils
from ..eval import predict_from_model, generate_attention_heatmaps, _export_attention
from .._params import (
    _TrainerConfig, TrainerConfigFastAI, TrainerConfigLightning
)

if TYPE_CHECKING:
    from fastai.learner import Learner

# -----------------------------------------------------------------------------

def train_mil(
    config: _TrainerConfig,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]],
    *args,
    outdir: str = 'mil',
    exp_label: Optional[str] = None,
    **kwargs
):

    mil_kwargs = dict(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        outcomes=outcomes,
        bags=bags,
        outdir=outdir,
        exp_label=exp_label,
        **kwargs
    )

    logging.info(f"Training MIL with config: {mil_kwargs}")
    
    return _train_mil(config, **mil_kwargs)



def _train_mil(
    config: _TrainerConfig,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]],
    *,
    outdir: str = 'mil',
    exp_label: Optional[str] = None,
    **kwargs
):
    """Train a multiple-instance learning (MIL) model.

    Args:
        config (:class:`slideflow.mil.TrainerConfigFastAI` or :class:`slideflow.mil.TrainerConfigLightning`):
            Trainer and model configuration.
        train_dataset (:class:`slideflow.Dataset`): Training dataset.
        val_dataset (:class:`slideflow.Dataset`): Validation dataset (defaults to train if None).
        outcomes (str or List[str]): Outcome column(s) to derive labels.
        bags (str or List[str]): Directory of *.pt files or list of *.pt paths.

    Keyword args:
        outdir (str): Directory for model/results.
        exp_label (str): Subdirectory label used inside `outdir`.
        **kwargs: Passed through to backend trainer.
    """
    # Select backend
    if isinstance(config, TrainerConfigFastAI):
        train_fn = train_fastai
        backend = "fastai"
    elif isinstance(config, TrainerConfigLightning):
        from ._lightning import train_lightning as train_fn
        backend = "lightning"
    else:
        raise ValueError(f"Unrecognized training configuration of type {type(config)}")

    # Default validation dataset to train_dataset if not provided
    if val_dataset is None:
        sf.log.info("Training without explicit validation; using training set for validation metrics.")
        val_dataset = train_dataset

    # Build experiment label
    if exp_label is None:
        try:
            exp_label = '{}-{}'.format(
                config.model_config.model,
                "-".join(outcomes if isinstance(outcomes, list) else [outcomes])
            )
        except Exception:
            exp_label = 'no_label'

    # Create output directory
    if outdir:
        if not exists(outdir):
            os.makedirs(outdir)
        outdir = sf.util.create_new_model_dir(outdir, exp_label)

    # FastAI path stays exactly as before — call and return
    if backend == "fastai":
        return train_fn(
            config,
            train_dataset,
            val_dataset,
            outcomes,
            bags,
            outdir=outdir,
            **kwargs
        )

    # -----------------------------
    # Lightning path: normalize data LIKE fastai does, then call with explicit kwargs
    # -----------------------------
    task = config.to_dict().get('task', 'classification')

    # 1) Labels & unique categories (mirror fastai build_learner)
    if task in ('classification', 'survival_discrete'):
        labels, unique_train = train_dataset.labels(outcomes, format='name', use_float=False)
        val_labels, unique_val = val_dataset.labels(outcomes, format='name', use_float=False)
    elif task in ('regression', 'survival'):
        labels, unique_train = train_dataset.labels(outcomes, format='value', use_float=True)
        val_labels, unique_val = val_dataset.labels(outcomes, format='value', use_float=True)
    else:
        raise ValueError(f"Unrecognized task {task} in config")

    labels.update(val_labels)
    if isinstance(unique_train, dict) and isinstance(unique_val, dict):
        unique_categories = np.unique(list(unique_train.values()) + list(unique_val.values()))
    else:
        unique_categories = np.unique(unique_train + unique_val)

    # 2) Collect .pt bag paths for train/val
    if isinstance(bags, str) or (isinstance(bags, list) and isdir(bags[0])):
        train_bags = train_dataset.pt_files(bags)
        if val_dataset is train_dataset:
            all_bags = train_bags
        else:
            val_bags = val_dataset.pt_files(bags)
            all_bags = np.concatenate((train_bags, val_bags))
    else:
        all_bags = np.array(bags)

    train_slides = train_dataset.slides()
    val_slides = val_dataset.slides()

    # 3) Aggregate (slide or patient) to produce bags/targets/train_idx/val_idx
    if config.aggregation_level == 'slide':
        bags_arr, targets, train_idx, val_idx = utils.aggregate_trainval_bags_by_slide(
            all_bags,
            labels,
            train_slides,
            val_slides,
            log_manifest=(join(outdir, 'slide_manifest.csv') if outdir else None)
        )
    elif config.aggregation_level == 'patient':
        slide_to_patient = {**train_dataset.patients(), **val_dataset.patients()}
        n_slide_bags = len(all_bags)
        bags_arr, targets, train_idx, val_idx = utils.aggregate_trainval_bags_by_patient(
            all_bags,
            labels,
            train_slides,
            val_slides,
            slide_to_patient=slide_to_patient,
            log_manifest=(join(outdir, 'slide_manifest.csv') if outdir else None)
        )
        log.info(f"Aggregated {n_slide_bags} slide bags to {len(bags_arr)} patient bags.")
    else:
        raise ValueError(f"Unknown aggregation_level: {config.aggregation_level}")

    log.info("Training dataset: {} merged bags (from {} possible slides)".format(
        len(train_idx), len(train_slides)))
    log.info("Validation dataset: {} merged bags (from {} possible slides)".format(
        len(val_idx), len(val_slides)))

    # 4) Ensure numpy arrays
    bags_arr   = np.array(bags_arr)
    targets    = np.array(targets)
    train_idx  = np.array(train_idx)
    val_idx    = np.array(val_idx)
    unique_categories = np.array(unique_categories)

    # 5) Finally call Lightning with the arguments it expects
    logging.info("Calling train_lightning with normalized arguments.")
    _lt = train_fn(
        config,
        bags=bags_arr,
        targets=targets,
        train_idx=train_idx,
        val_idx=val_idx,
        unique_categories=unique_categories,
        outdir=outdir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        outcomes=outcomes,
        **kwargs
    )

    # ---------- FASTAI-like outputs for Lightning ----------
    # Unpack Lightning return (support both dict and direct module returns)
    if isinstance(_lt, dict):
        module = _lt.get("module", None)
        n_in   = _lt.get("n_in", None)
        n_out  = _lt.get("n_out", None)
        if module is None:
            module = _lt  # fallback
    else:
        module = _lt
        n_in = n_out = None   # shapes not provided; mil_params will skip shapes

    # 1) mil_params.json (same structure as FastAI)
    unique_list = (unique_categories.tolist()
                if unique_categories is not None
                else None)
    if n_in is not None and n_out is not None:
        _log_mil_params(config, outcomes, unique_list, bags, n_in, n_out, outdir)
    else:
        log.warning("Lightning trainer did not return n_in/n_out; writing mil_params.json without shapes.")
        try:
            _log_mil_params(config, outcomes, unique_list, bags, None, None, outdir)
        except Exception:
            pass

    # 2) collect validation bags exactly like FastAI
    val_bags = _collect_val_bags_for_dataset(bags, val_dataset)

    # 3) run predictions + attention and save
    attention_heatmaps = kwargs.get("attention_heatmaps", False)
    uq = kwargs.get("uq", False)
    heatmap_kwargs = _extract_heatmap_kwargs(kwargs)

    _predict_and_save_outputs_like_fastai(
        getattr(module, "model", module),   # pass raw nn.Module; if LightningModule, use .model
        config,
        val_dataset,
        outcomes,
        val_bags,
        outdir,
        attention_heatmaps=attention_heatmaps,
        uq=uq,
        **heatmap_kwargs
    )

    # Mirror FastAI return style: FastAI returns a Learner; Lightning returns the module
    return module
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_fastai_learner(
    config: TrainerConfigFastAI,
    train_dataset: Dataset,
    val_dataset: Dataset,
    outcomes: Union[str, List[str]],
    bags: Union[str, np.ndarray, List[str]],
    *,
    outdir: str = 'mil',
    return_shape: bool = False,
    **kwargs
) -> "Learner":
    """Build a FastAI Learner for training an aMIL model.

    Args:
        train_dataset (:class:`slideflow.Dataset`): Training dataset.
        val_dataset (:class:`slideflow.Dataset`): Validation dataset.
        outcomes (str): Outcome column (annotation header) from which to
            derive category labels.
        bags (str): list of paths to individual \*.pt files. Each file should
            contain exported feature vectors, with each file containing all tile
            features for one patient.

    Keyword args:
        outdir (str): Directory in which to save model and results.
        exp_label (str): Experiment label, used for naming the subdirectory
            in the ``outdir`` folder, where training history
            and the model will be saved.
        lr (float): Learning rate, or maximum learning rate if
            ``fit_one_cycle=True``.
        epochs (int): Maximum epochs.

    Returns:
        fastai.learner.Learner
    """
    from . import _fastai

    task = config.to_dict()['task']
    
    #Treat labels as categorical...
    if task == 'classification' or task == 'survival_discrete':
        # Prepare labels and slides
        labels, unique_train = train_dataset.labels(outcomes, format='name', use_float=False)
        val_labels, unique_val = val_dataset.labels(outcomes, format='name', use_float=False)
        logging.debug(f"Unique float train labels: {unique_train}")
        logging.debug(f"Unique float val labels: {unique_val}")
    #Treat labels as float...
    elif task == 'regression' or task == 'survival':
        # Prepare labels and slides
        labels, unique_train = train_dataset.labels(outcomes, format='value', use_float=True)
        val_labels, unique_val = val_dataset.labels(outcomes, format='value', use_float=True)
        logging.debug(f"Unique float train labels: {unique_train}")
        logging.debug(f"Unique float val labels: {unique_val}")
    else:
        raise ValueError(f"Unrecognized task {task} in config")

    labels.update(val_labels)

    if isinstance(unique_train, dict) and isinstance(unique_val, dict):
        # Merge unique_train and unique_val dictionaries
        unique_categories = np.unique(list(unique_train.values()) + list(unique_val.values()))
    else:
        # Concatenate lists or arrays
        unique_categories = np.unique(unique_train + unique_val)

    # Prepare bags
    if isinstance(bags, str) or (isinstance(bags, list) and isdir(bags[0])):
        train_bags = train_dataset.pt_files(bags)
        if val_dataset is train_dataset:
            bags = train_bags
        else:
            val_bags = val_dataset.pt_files(bags)
            bags = np.concatenate((train_bags, val_bags))
    else:
        bags = np.array(bags)

    train_slides = train_dataset.slides()
    val_slides = val_dataset.slides()

    if config.aggregation_level == 'slide':
        # Aggregate feature bags across slides.

        bags, targets, train_idx, val_idx = utils.aggregate_trainval_bags_by_slide(
            bags,  # type: ignore
            labels,
            train_slides,
            val_slides,
            log_manifest=(join(outdir, 'slide_manifest.csv') if outdir else None)
        )

    elif config.aggregation_level == 'patient':
        # Associate patients and their slides.
        # This is a dictionary where each key is a slide name and each value
        # is a patient code. Multiple slides can match to the same patient.
        slide_to_patient = { **train_dataset.patients(),
                             **val_dataset.patients() }

        # Aggregate feature bags across patients.
        n_slide_bags = len(bags)
        bags, targets, train_idx, val_idx = utils.aggregate_trainval_bags_by_patient(
            bags,  # type: ignore
            labels,
            train_slides,
            val_slides,
            slide_to_patient=slide_to_patient,
            log_manifest=(join(outdir, 'slide_manifest.csv') if outdir else None)
        )
        log.info(f"Aggregated {n_slide_bags} slide bags to {len(bags)} patient bags.")

    log.info("Training dataset: {} merged bags (from {} possible slides)".format(
        len(train_idx), len(train_slides)))
    log.info("Validation dataset: {} merged bags (from {} possible slides)".format(
        len(val_idx), len(val_slides)))

    #Make sure targets are flattened
    logging.debug(f"Targets shape before building learner: {targets.shape}")

    # Build FastAI Learner
    learner, (n_in, n_out) = _fastai.build_learner(
        config,
        bags=bags,
        targets=targets,
        train_idx=train_idx,
        val_idx=val_idx,
        unique_categories=unique_categories,
        outdir=outdir,
        pin_memory=True,
        **kwargs
    )
    if return_shape:
        return learner, (n_in, n_out)
    else:
        return learner


def build_multimodal_learner(
    config: TrainerConfigFastAI,
    train_dataset: Dataset,
    val_dataset: Dataset,
    outcomes: Union[str, List[str]],
    bags: Union[str, np.ndarray, List[str]],
    *,
    outdir: str = 'mil',
    return_shape: bool = False,
    **kwargs
) -> "Learner":
    """Build a multi-magnification FastAI Learner for training an aMIL model."""

    from . import _fastai

    # Verify bags are in the correct format.
    if (not isinstance(bags, (tuple, list))
        or not all([isinstance(b, str) and isdir(b) for b in bags])):
        raise ValueError("Expected bags to be a list of paths, got {}".format(type(bags)))

    num_modes = len(bags)

    # Prepare labels and slides
    labels, unique_train = train_dataset.labels(outcomes, format='name')
    val_labels, unique_val = val_dataset.labels(outcomes, format='name')
    labels.update(val_labels)
    unique_categories = np.unique(unique_train + unique_val)

    # --- Prepare bags --------------------------------------------------------

    train_bags, train_slides = utils._get_nested_bags(train_dataset, bags)
    val_bags, val_slides = utils._get_nested_bags(val_dataset, bags)

    # --- Process bags and targets for training -------------------------------

    # Note: we are skipping patient-level bag aggregation for now.
    # TODO: implement patient-level bag aggregation for multi-modal MIL.

    # Concatenate training and validation bags.
    all_bags = np.concatenate((train_bags, val_bags)) # shape: (num_slides, num_modes)
    assert all_bags.shape[0] == len(train_slides) + len(val_slides)
    all_slides = train_slides + val_slides
    targets = np.array([labels[s] for s in all_slides])
    train_idx = np.arange(len(train_slides))
    val_idx = np.arange(len(train_slides), len(all_slides))

    # Write the slide manifest
    if outdir:
        sf.util.log_manifest(
            train_slides,
            val_slides,
            labels=labels,
            filename=join(outdir, 'slide_manifest.csv'),
            remove_extension=False
        )

    # Print a multi-modal dataset summary.
    log.info(
        "[bold]Multi-modal MIL training summary:[/]"
        + "\n  - [blue]Modes[/]: {}".format(num_modes)
        + "\n  - [blue]Slides with bags[/]: {}".format(len(np.unique(all_slides)))
        + "\n  - [blue]Multi-modal bags[/]: {}".format(all_bags.shape[0])
        + "\n  - [blue]Unique categories[/]: {}".format(len(unique_categories))
        + "\n  - [blue]Training multi-modal bags[/]: {}".format(len(train_idx))
        + "\n  - [blue]Training slides[/]: {}".format(len(np.unique(train_slides)))
        + "\n  - [blue]Validation multi-modal bags[/]: {}".format(len(val_idx))
        + "\n  - [blue]Validation slides[/]: {}".format(len(np.unique(val_slides)))
    )

    # Print a detailed summary of each mode.
    for i, mode in enumerate(bags):
        try:
            bags_config = sf.util.load_json(join(mode, 'bags_config.json'))
        except Exception:
            log.info(
                "Mode {i}: "
                + "\n  - Bags: {}".format(mode)
            )
        else:
            log.info(
                f"[bold]Mode {i+1}[/]: [green]{mode}[/]"
                + "\n  - Feature extractor: [purple]{}[/]".format(bags_config['extractor']['class'].split('.')[-1])
                + "\n  - Tile size (px): {}".format(bags_config['tile_px'])
                + "\n  - Tile size (um): {}".format(bags_config['tile_um'])
                + "\n  - Normalizer: {}".format(bags_config['normalizer'])
            )

    # --- Build FastAI Learner ------------------------------------------------

    # Build FastAI Learner
    learner, (n_in, n_out) = _fastai._build_multimodal_learner(
        config,
        all_bags,
        targets,
        train_idx,
        val_idx,
        unique_categories,
        num_modes,
        outdir=outdir,
        pin_memory=True,
        **kwargs
    )
    if return_shape:
        return learner, (n_in, n_out)
    else:
        return learner


def train_fastai(
    config: TrainerConfigFastAI,
    train_dataset: Dataset,
    val_dataset: Dataset,
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]],
    *,
    outdir: str = 'mil',
    attention_heatmaps: bool = False,
    uq: bool = False,
    **heatmap_kwargs
) -> None:
    """Train an aMIL model using FastAI.

    Args:
        train_dataset (:class:`slideflow.Dataset`): Training dataset.
        val_dataset (:class:`slideflow.Dataset`): Validation dataset.
        outcomes (str): Outcome column (annotation header) from which to
            derive category labels.
        bags (str): Either a path to directory with \*.pt files, or a list
            of paths to individual \*.pt files. Each file should contain
            exported feature vectors, with each file containing all tile
            features for one patient.

    Keyword args:
        outdir (str): Directory in which to save model and results.
        exp_label (str): Experiment label, used for naming the subdirectory
            in the ``{project root}/mil`` folder, where training history
            and the model will be saved.
        lr (float): Learning rate, or maximum learning rate if
            ``fit_one_cycle=True``.
        epochs (int): Maximum epochs.
        attention_heatmaps (bool): Generate attention heatmaps for slides.
            Defaults to False.
        interpolation (str, optional): Interpolation strategy for smoothing
            attention heatmaps. Defaults to 'bicubic'.
        cmap (str, optional): Matplotlib colormap for heatmap. Can be any
            valid matplotlib colormap. Defaults to 'inferno'.
        norm (str, optional): Normalization strategy for assigning heatmap
            values to colors. Either 'two_slope', or any other valid value
            for the ``norm`` argument of ``matplotlib.pyplot.imshow``.
            If 'two_slope', normalizes values less than 0 and greater than 0
            separately. Defaults to None.

    Returns:
        fastai.learner.Learner
    """
    from . import _fastai

    pb_config = heatmap_kwargs.get('pb_config', None)

    # Prepare validation bags.
    if isinstance(bags, str) or (isinstance(bags, list) and isdir(bags[0])):
        val_bags = val_dataset.pt_files(bags)
    else:
        val_bags = np.array([b for b in bags if sf.util.path_to_name(b) in val_dataset.slides()])

    # Build learner.
    learner, (n_in, n_out) = build_fastai_learner(
        config,
        train_dataset,
        val_dataset,
        outcomes,
        bags=bags,
        outdir=outdir,
        return_shape=True,
        **heatmap_kwargs
    )

    # Save MIL settings.
    # Attempt to read the unique categories from the learner.
    if not hasattr(learner.dls.train_ds, 'encoder'):
        unique = None
    else:
        encoder = learner.dls.train_ds.encoder
        if encoder is not None:
            unique = encoder.categories_[0].tolist()
        else:
            unique = None
    _log_mil_params(config, outcomes, unique, bags, n_in, n_out, outdir)

    # Train.
    if 'best_epoch_based_on' in pb_config['experiment'] and pb_config['experiment']['best_epoch_based_on'] != 'val_loss':
        _fastai.train(learner, config, pb_config)
    else:
        _fastai.train(learner, config)

    # Generate validation predictions.
    df, attention = predict_from_model(
        learner.model,
        config,
        dataset=val_dataset,
        outcomes=outcomes,
        bags=val_bags,
        attention=True,
        uq=uq,
        **heatmap_kwargs
    )
    if outdir:
        pred_out = join(outdir, 'predictions.parquet')
        df.to_parquet(pred_out)
        log.info(f"Predictions saved to [green]{pred_out}[/]")

    # Print categorical metrics, including per-category accuracy
    outcome_name = outcomes if isinstance(outcomes, str) else '-'.join(outcomes)
    df.rename(
        columns={c: f"{outcome_name}-{c}" for c in df.columns if c != 'slide'},
        inplace=True
    )
    sf.stats.metrics.categorical_metrics(df, level='slide')

    # Export attention to numpy arrays
    if attention and outdir:
        _export_attention(
            join(outdir, 'attention'),
            attention,
            [path_to_name(b) for b in val_bags]
        )

    # Attention heatmaps.
    if attention and attention_heatmaps and outdir:
        generate_attention_heatmaps(
            outdir=join(outdir, 'heatmaps'),
            dataset=val_dataset,
            bags=val_bags,
            attention=attention,
            **heatmap_kwargs
        )

    return learner

# ------------------------------------------------------------------------------

def _log_mil_params(config, outcomes, unique, bags, n_in, n_out, outdir=None):
    """Log MIL parameters to JSON."""
    mil_params = config.json_dump()
    mil_params['outcomes'] = outcomes
    if unique is not None:
        mil_params['outcome_labels'] = dict(zip(range(len(unique)), unique))
    else:
        mil_params['outcome_labels'] = None
    mil_params['bags'] = bags
    mil_params['input_shape'] = n_in
    mil_params['output_shape'] = n_out
    if isinstance(bags, str) and exists(join(bags, 'bags_config.json')):
        mil_params['bags_extractor'] = sf.util.load_json(
            join(bags, 'bags_config.json')
        )
    elif isinstance(bags, list):
        mil_params['bags_extractor'] = {}
        for b in bags:
            if isdir(b) and exists(join(b, 'bags_config.json')):
                mil_params['bags_extractor'][b] = sf.util.load_json(
                    join(b, 'bags_config.json')
                )
            else:
                mil_params['bags_extractor'][b] = None
    else:
        mil_params['bags_extractor'] = None
    if outdir:
        sf.util.write_json(mil_params, join(outdir, 'mil_params.json'))
    return mil_params

def _collect_val_bags_for_dataset(bags, val_dataset):
    """Return the list/array of .pt bags used for validation, mirroring FastAI behavior."""
    if isinstance(bags, str) or (isinstance(bags, list) and isdir(bags[0])):
        return val_dataset.pt_files(bags)
    # `bags` is a flat list/array of paths → keep only those belonging to val slides
    return np.array([b for b in bags if path_to_name(b) in val_dataset.slides()])


def _extract_heatmap_kwargs(kwargs: dict) -> dict:
    """Remove control flags that shouldn't be forwarded to heatmap/predict calls."""
    blocked = {"uq", "attention_heatmaps", "pb_config"}
    return {k: v for k, v in kwargs.items() if k not in blocked}


def _rename_outcome_columns_inplace(df: pd.DataFrame, outcomes: Union[str, List[str]]) -> None:
    """Prefix per-class probability columns to match FastAI logging style."""
    outcome_name = outcomes if isinstance(outcomes, str) else "-".join(outcomes)
    df.rename(columns={c: f"{outcome_name}-{c}" for c in df.columns if c != "slide"}, inplace=True)


def _predict_and_save_outputs_like_fastai(
    model,                       # nn.Module (e.g., module.model)
    config,
    val_dataset,
    outcomes,
    val_bags,
    outdir: Optional[str],
    *,
    attention_heatmaps: bool = False,
    uq: bool = False,
    **heatmap_kwargs
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Run predict_from_model, write predictions.parquet, log metrics, export attention, and heatmaps."""
    # 1) predictions + attention
    df, attention = predict_from_model(
        model,
        config,
        dataset=val_dataset,
        outcomes=outcomes,
        bags=val_bags,
        attention=True,
        uq=uq,
        **heatmap_kwargs
    )

    # 2) save predictions
    if outdir:
        pred_out = join(outdir, "predictions.parquet")
        df.to_parquet(pred_out)
        log.info(f"Predictions saved to [green]{pred_out}[/]")

    # 3) metrics (rename columns like FastAI)
    _rename_outcome_columns_inplace(df, outcomes)
    sf.stats.metrics.categorical_metrics(df, level="slide")

    # 4) export attention arrays
    if attention and outdir:
        _export_attention(
            join(outdir, "attention"),
            attention,
            [path_to_name(b) for b in val_bags]
        )

    # 5) optional attention heatmaps
    if attention and attention_heatmaps and outdir:
        generate_attention_heatmaps(
            outdir=join(outdir, "heatmaps"),
            dataset=val_dataset,
            bags=val_bags,
            attention=attention,
            **heatmap_kwargs
        )

    return df, attention