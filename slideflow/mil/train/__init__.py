"""Training functions for various multi-instance learning (MIL) models."""

import os
import numpy as np
import slideflow as sf
from os.path import join, exists
from typing import Union, List, Optional, TYPE_CHECKING
from slideflow import Dataset, log
from slideflow.util import path_to_name
from os.path import join

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

    See :ref:`clam_mil` for more information.

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
        filename=join(outdir, 'slide_manifest.csv')
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
        bags (str): Either a path to directory with \*.pt files, or a list
            of paths to individual \*.pt files. Each file should contain
            exported feature vectors, with each file containing all tile
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

    # Prepare bags and targets
    if isinstance(bags, str):
        train_bags = train_dataset.pt_files(bags)
        val_bags = val_dataset.pt_files(bags)
        bags = np.concatenate((train_bags, val_bags))
    else:
        bags = np.array(bags)
    targets = np.array([labels[path_to_name(f)] for f in bags])

    # Prepare training/validation indices
    train_slides = train_dataset.slides()
    train_idx = np.array([i for i, bag in enumerate(bags)
                            if path_to_name(bag) in train_slides])
    val_slides = val_dataset.slides()
    val_idx = np.array([i for i, bag in enumerate(bags)
                            if path_to_name(bag) in val_slides])

    log.info("Training dataset: {} bags (from {} slides)".format(len(train_idx), len(train_slides)))
    log.info("Validation dataset: {} bags (from {} slides)".format(len(val_idx), len(val_slides)))

    # Write slide/bag manifest
    sf.util.log_manifest(
        [bag for bag in bags if path_to_name(bag) in train_slides],
        [bag for bag in bags if path_to_name(bag) in val_slides],
        labels=labels,
        filename=join(outdir, 'slide_manifest.csv')
    )

    # Build FastAI Learner
    learner, (n_in, n_out) = _fastai.build_learner(
        config,
        bags=bags,
        targets=targets,
        train_idx=train_idx,
        val_idx=val_idx,
        unique_categories=unique_categories,
        outdir=outdir,
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

    # Prepare bags.
    if isinstance(bags, str):
        train_bags = train_dataset.pt_files(bags)
        val_bags = val_dataset.pt_files(bags)
        all_bags = np.concatenate((train_bags, val_bags))
    else:
        val_bags = np.array([b for b in bags if sf.util.path_to_name(b) in val_dataset.slides()])
        all_bags = np.array(bags)

    # Build learner.
    learner, (n_in, n_out) = build_fastai_learner(
        config,
        train_dataset,
        val_dataset,
        outcomes,
        bags=all_bags,
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
    else:
        mil_params['bags_extractor'] = None
    sf.util.write_json(mil_params, join(outdir, 'mil_params.json'))
    return mil_params