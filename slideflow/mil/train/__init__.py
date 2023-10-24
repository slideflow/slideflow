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

from .. import utils
from ..eval import predict_from_model, generate_attention_heatmaps, _export_attention
from .._params import (
    _TrainerConfig, TrainerConfigCLAM, TrainerConfigFastAI
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
    *,
    outdir: str = 'mil',
    exp_label: Optional[str] = None,
    **kwargs
):
    """Train a multiple-instance learning (MIL) model.

    Args:
        config (:class:`slideflow.mil.TrainerConfigFastAI` or :class:`slideflow.mil.TrainerConfigCLAM`):
            Trainer and model configuration.
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
        attention_heatmaps (bool): Generate attention heatmaps for slides.
            Not available for multi-modal MIL models. Defaults to False.
        interpolation (str, optional): Interpolation strategy for smoothing
            attention heatmaps. Defaults to 'bicubic'.
        cmap (str, optional): Matplotlib colormap for heatmap. Can be any
            valid matplotlib colormap. Defaults to 'inferno'.
        norm (str, optional): Normalization strategy for assigning heatmap
            values to colors. Either 'two_slope', or any other valid value
            for the ``norm`` argument of ``matplotlib.pyplot.imshow``.
            If 'two_slope', normalizes values less than 0 and greater than 0
            separately. Defaults to None.

    """
    mil_kwargs = dict(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        outcomes=outcomes,
        bags=bags,
        outdir=outdir,
        exp_label=exp_label,
        **kwargs
    )
    if config.is_multimodal:
        return _train_multimodal_mil(config, **mil_kwargs)  # type: ignore
    else:
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
        config (:class:`slideflow.mil.TrainerConfigFastAI` or :class:`slideflow.mil.TrainerConfigCLAM`):
            Trainer and model configuration.
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

    """
    log.info("Training FastAI MIL model with config:")
    log.info(f"{config}")
    if isinstance(config, TrainerConfigFastAI):
        train_fn = train_fastai
    elif isinstance(config, TrainerConfigCLAM):
        train_fn = train_clam
    else:
        raise ValueError(f"Unrecognized training configuration of type {type(config)}")
    if val_dataset is None:
        sf.log.info(
            "Training without validation; metrics will be calculated on training data."
        )
        val_dataset = train_dataset

    # Set up experiment label
    if exp_label is None:
        try:
            exp_label = '{}-{}'.format(
                config.model_config.model,
                "-".join(outcomes if isinstance(outcomes, list) else [outcomes])
            )
        except Exception:
            exp_label = 'no_label'

    # Set up output model directory
    if not exists(outdir):
        os.makedirs(outdir)
    outdir = sf.util.create_new_model_dir(outdir, exp_label)

    # Execute training.
    return train_fn(
        config,
        train_dataset,
        val_dataset,
        outcomes,
        bags,
        outdir=outdir,
        **kwargs
    )


def _train_multimodal_mil(
    config: TrainerConfigFastAI,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    outcomes: Union[str, List[str]],
    bags: List[str],
    *,
    outdir: str = 'mil',
    exp_label: Optional[str] = None,
    attention_heatmaps: bool = False,
):
    """Train a multi-modal (e.g. multi-magnification) MIL model."""

    from . import _fastai

    # Export attention & heatmaps.
    if attention_heatmaps:
        raise ValueError(
                "Attention heatmaps cannot yet be exported for multi-modal "
                "models. Please use Slideflow Studio for visualization of "
                "multi-modal attention."
            )

    log.info("Training FastAI Multi-modal MIL model with config:")
    log.info(f"{config}")
    if val_dataset is None:
        sf.log.info("Training without validation; metrics will be calculated on training data.")
        val_dataset = train_dataset

    # Set up experiment label
    if exp_label is None:
        try:
            exp_label = '{}-{}'.format(
                config.model_config.model,
                "-".join(outcomes if isinstance(outcomes, list) else [outcomes])
            )
        except Exception:
            exp_label = 'no_label'

    # Set up output model directory
    if not exists(outdir):
        os.makedirs(outdir)
    outdir = sf.util.create_new_model_dir(outdir, exp_label)

    # Prepare validation bags and targets.
    val_labels, val_unique = val_dataset.labels(outcomes, format='id')
    val_bags, val_slides = utils._get_nested_bags(val_dataset, bags)
    val_targets = np.array([val_labels[slide] for slide in val_slides])

    # Build learner.
    learner, (n_in, n_out) = build_multimodal_learner(
        config,
        train_dataset,
        val_dataset,
        outcomes,
        bags=bags,
        outdir=outdir,
        return_shape=True
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

    # Execute training.
    _fastai.train(learner, config)

    # Generate validation predictions
    y_pred, y_att = sf.mil.eval._predict_multimodal_mil(
        learner.model,
        val_bags,
        attention=True,
        use_lens=config.model_config.use_lens
    )

    # Combine to a dataframe.
    df_dict = dict(slide=val_slides, y_true=val_targets)
    for i in range(y_pred.shape[-1]):
        df_dict[f'y_pred{i}'] = y_pred[:, i]
    df = pd.DataFrame(df_dict)

    # Print categorical metrics, including per-category accuracy
    outcome_name = outcomes if isinstance(outcomes, str) else '-'.join(outcomes)
    df.rename(
        columns={c: f"{outcome_name}-{c}" for c in df.columns if c != 'slide'},
        inplace=True
    )
    sf.stats.metrics.categorical_metrics(df, level='slide')

    # Export predictions.
    pred_out = join(outdir, 'predictions.parquet')
    df.to_parquet(pred_out)
    log.info(f"Predictions saved to [green]{pred_out}[/]")

    # Export attention.
    if y_att:
        _export_attention(join(outdir, 'attention'), y_att, df.slide.values)

    return learner


# -----------------------------------------------------------------------------

def train_clam(
    config: TrainerConfigCLAM,
    train_dataset: Dataset,
    val_dataset: Dataset,
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]],
    *,
    outdir: str = 'mil',
    attention_heatmaps: bool = False,
    **heatmap_kwargs
) -> None:
    """Train a CLAM model from layer activations exported with
    :meth:`slideflow.project.generate_features_for_clam`.

    See :ref:`mil` for more information.

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
            in the ``outdir`` folder, where training history
            and the model will be saved.
        clam_args (optional): Namespace with clam arguments, as provided
            by :func:`slideflow.clam.get_args`.
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
        None

    """
    import slideflow.clam as clam
    from slideflow.clam import CLAM_Dataset

    # Set up results directory
    results_dir = join(outdir, 'results')
    if not exists(results_dir):
        os.makedirs(results_dir)

    # Set up labels.
    labels, unique_train = train_dataset.labels(outcomes, format='name', use_float=False)
    val_labels, unique_val = val_dataset.labels(outcomes, format='name', use_float=False)
    labels.update(val_labels)
    unique_labels = np.unique(unique_train + unique_val)
    label_dict = dict(zip(unique_labels, range(len(unique_labels))))

    # Prepare CLAM arguments.
    clam_args = config._to_clam_args()
    clam_args.results_dir = results_dir
    clam_args.n_classes = len(unique_labels)

    # Set up bags.
    if isinstance(bags, str):
        train_bags = train_dataset.pt_files(bags)
        val_bags = val_dataset.pt_files(bags)
    else:
        train_bags = val_bags = bags

    # Write slide/bag manifest
    sf.util.log_manifest(
        [b for b in train_bags],
        [b for b in val_bags],
        labels=labels,
        filename=join(outdir, 'slide_manifest.csv'),
    )

    # Set up datasets.
    train_mil_dataset = CLAM_Dataset(
        train_bags,
        annotations=train_dataset.filtered_annotations,
        label_col=outcomes,
        label_dict=label_dict
    )
    val_mil_dataset = CLAM_Dataset(
        val_bags,
        annotations=val_dataset.filtered_annotations,
        label_col=outcomes,
        label_dict=label_dict
    )

    # Get base CLAM args/settings if not provided.
    num_features = train_mil_dataset.detect_num_features()
    if isinstance(clam_args.model_size, str):
        model_size = config.model_fn.sizes[clam_args.model_size]
    else:
        model_size = clam_args.model_size
    if model_size[0] != num_features:
        _old_size = model_size[0]
        model_size[0] = num_features
        clam_args.model_size = model_size
        log.warn(
            f"First dimension of model size ({_old_size}) does not "
            f"match features ({num_features}). Updating model size to "
            f"{clam_args.model_size}. "
        )

    # Save clam settings
    sf.util.write_json(clam_args.to_dict(), join(outdir, 'experiment.json'))

    # Save MIL settings
    _log_mil_params(config, outcomes, unique_labels, bags, num_features, clam_args.n_classes, outdir)

    # Run CLAM
    datasets = (train_mil_dataset, val_mil_dataset, val_mil_dataset)
    model, results, test_auc, val_auc, test_acc, val_acc = clam.train(
        datasets, 0, clam_args
    )

    # Generate validation predictions
    df, attention = predict_from_model(
        model,
        config,
        dataset=val_dataset,
        outcomes=outcomes,
        bags=bags,
        attention=True
    )
    pred_out = join(outdir, 'results', 'predictions.parquet')
    df.to_parquet(pred_out)
    log.info(f"Predictions saved to [green]{pred_out}[/]")

    # Print categorical metrics, including per-category accuracy
    outcome_name = outcomes if isinstance(outcomes, str) else '-'.join(outcomes)
    df.rename(
        columns={c: f"{outcome_name}-{c}" for c in df.columns if c != 'slide'},
        inplace=True
    )
    sf.stats.metrics.categorical_metrics(df, level='slide')

    # Attention heatmaps
    if isinstance(bags, str):
        val_bags = val_dataset.pt_files(bags)
    else:
        val_bags = np.array(bags)

    # Export attention to numpy arrays
    if attention:
        _export_attention(
            join(outdir, 'attention'),
            attention,
            [path_to_name(b) for b in val_bags]
        )

    # Save attention heatmaps
    if attention and attention_heatmaps:
        assert len(val_bags) == len(attention)
        generate_attention_heatmaps(
            outdir=join(outdir, 'heatmaps'),
            dataset=val_dataset,
            bags=val_bags,
            attention=attention,
            **heatmap_kwargs
        )

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

    # Prepare labels and slides
    labels, unique_train = train_dataset.labels(outcomes, format='name')
    val_labels, unique_val = val_dataset.labels(outcomes, format='name')
    labels.update(val_labels)
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
            log_manifest=join(outdir, 'slide_manifest.csv')
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
            log_manifest=join(outdir, 'slide_manifest.csv')
        )
        log.info(f"Aggregated {n_slide_bags} slide bags to {len(bags)} patient bags.")

    log.info("Training dataset: {} merged bags (from {} possible slides)".format(
        len(train_idx), len(train_slides)))
    log.info("Validation dataset: {} merged bags (from {} possible slides)".format(
        len(val_idx), len(val_slides)))

    # Build FastAI Learner
    learner, (n_in, n_out) = _fastai.build_learner(
        config,
        bags=bags,
        targets=targets,
        train_idx=train_idx,
        val_idx=val_idx,
        unique_categories=unique_categories,
        outdir=outdir,
        pin_memory=True
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
        return_shape=True
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
    _fastai.train(learner, config)

    # Generate validation predictions.
    df, attention = predict_from_model(
        learner.model,
        config,
        dataset=val_dataset,
        outcomes=outcomes,
        bags=val_bags,
        attention=True
    )
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
    if attention:
        _export_attention(
            join(outdir, 'attention'),
            attention,
            [path_to_name(b) for b in val_bags]
        )

    # Attention heatmaps.
    if attention and attention_heatmaps:
        generate_attention_heatmaps(
            outdir=join(outdir, 'heatmaps'),
            dataset=val_dataset,
            bags=val_bags,
            attention=attention,
            **heatmap_kwargs
        )

    return learner

# ------------------------------------------------------------------------------

def _log_mil_params(config, outcomes, unique, bags, n_in, n_out, outdir):
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
    sf.util.write_json(mil_params, join(outdir, 'mil_params.json'))
    return mil_params