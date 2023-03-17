"""Training functions for various multi-instance learning (MIL) models."""

import os
import numpy as np
import slideflow as sf
from os.path import join, exists
from typing import Union, List, Optional, TYPE_CHECKING
from slideflow import Dataset, log
from slideflow.util import path_to_name
from .._params import TrainerConfig, TrainerConfigCLAM, TrainerConfigFastAI

if TYPE_CHECKING:
    from fastai.learner import Learner

# -----------------------------------------------------------------------------

def train_mil(
    config: TrainerConfig,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]],
    *,
    outdir: str = 'mil',
    **kwargs
):
    """Train a multi-instance learning model.

    Args:
        config (``TrainerConfig``): Trainer and model configuration.
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
    exp_label: str = "CLAM",
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
            in the ``{project root}/mil`` folder, where training history
            and the model will be saved.
        clam_args (optional): Namespace with clam arguments, as provided
            by :func:`slideflow.clam.get_args`.

    Returns:
        None

    """
    import slideflow.clam as clam
    from slideflow.clam import CLAM_Dataset

    # Set up output directory in project root
    if not exists(outdir):
        os.makedirs(outdir)
    outdir = sf.util.create_new_model_dir(outdir, exp_label)
    results_dir = join(outdir, 'results')
    if not exists(results_dir):
        os.makedirs(results_dir)

    # Export configuration to JSON.
    sf.util.write_json(config.json_dump(), join(outdir, 'mil_params.json'))

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

    # Run CLAM
    datasets = (train_mil_dataset, val_mil_dataset, val_mil_dataset)
    results, test_auc, val_auc, test_acc, val_acc = clam.train(
        datasets, 0, clam_args
    )
    return results

# -----------------------------------------------------------------------------

def build_fastai_learner(
    config: TrainerConfigFastAI,
    train_dataset: Dataset,
    val_dataset: Dataset,
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]],
    *,
    outdir: str = 'mil',
    exp_label: str = 'fastai',
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
            in the ``{project root}/mil`` folder, where training history
            and the model will be saved.
        lr_max (float): Maximum learning rate.
        epochs (int): Maximum epochs.

    Returns:
        fastai.learner.Learner
    """
    from . import _fastai

    # Set up output directory in project root
    if not exists(outdir):
        os.makedirs(outdir)
    outdir = sf.util.create_new_model_dir(outdir, exp_label)
    sf.util.write_json(config.json_dump(), join(outdir, 'mil_params.json'))

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

    # Build FastAI Learner
    learner = _fastai.build_learner(
        config,
        bags=bags,
        targets=targets,
        train_idx=train_idx,
        val_idx=val_idx,
        unique_categories=unique_categories,
        outdir=outdir,
    )
    return learner


def train_fastai(
    config: TrainerConfigFastAI,
    train_dataset: Dataset,
    val_dataset: Dataset,
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]],
    *,
    outdir: str = 'mil',
    exp_label: str = 'fastai',
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
        lr_max (float): Maximum learning rate.
        epochs (int): Maximum epochs.

    Returns:
        fastai.learner.Learner
    """
    from . import _fastai

    learner = build_fastai_learner(
        config,
        train_dataset,
        val_dataset,
        outcomes,
        bags=bags,
        outdir=outdir,
        exp_label=exp_label
    )
    return _fastai.train(learner, config)
