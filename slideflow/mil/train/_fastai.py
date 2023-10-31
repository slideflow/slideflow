import torch
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import List, Optional, Union, Tuple
from torch import nn
from sklearn.preprocessing import OneHotEncoder
from sklearn import __version__ as sklearn_version
from packaging import version
from fastai.vision.all import (
    DataLoader, DataLoaders, Learner, RocAuc, SaveModelCallback, CSVLogger, FetchPredsCallback
)

from slideflow import log
import slideflow.mil.data as data_utils
from slideflow.model import torch_utils
from .._params import TrainerConfigFastAI, ModelConfigCLAM

# -----------------------------------------------------------------------------

def train(learner, config, callbacks=None):
    """Train an attention-based multi-instance learning model with FastAI.

    Args:
        learner (``fastai.learner.Learner``): FastAI learner.
        config (``TrainerConfigFastAI``): Trainer and model configuration.

    Keyword args:
        callbacks (list(fastai.Callback)): FastAI callbacks. Defaults to None.
    """
    cbs = [
        SaveModelCallback(fname=f"best_valid"),
        CSVLogger(),
    ]
    if callbacks:
        cbs += callbacks
    if config.fit_one_cycle:
        if config.lr is None:
            lr = learner.lr_find().valley
            log.info(f"Using auto-detected learning rate: {lr}")
        else:
            lr = config.lr
        learner.fit_one_cycle(n_epoch=config.epochs, lr_max=lr, cbs=cbs)
    else:
        if config.lr is None:
            lr = learner.lr_find().valley
            log.info(f"Using auto-detected learning rate: {lr}")
        else:
            lr = config.lr
        learner.fit(n_epoch=config.epochs, lr=lr, wd=config.wd, cbs=cbs)
    return learner

# -----------------------------------------------------------------------------

def build_learner(config, *args, **kwargs) -> Tuple[Learner, Tuple[int, int]]:
    """Build a FastAI learner for training an MIL model.

    Args:
        config (``TrainerConfigFastAI``): Trainer and model configuration.
        bags (list(str)): Path to .pt files (bags) with features, one per patient.
        targets (np.ndarray): Category labels for each patient, in the same
            order as ``bags``.
        train_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the training set.
        val_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the validation set.
        unique_categories (np.ndarray(str)): Array of all unique categories
            in the targets. Used for one-hot encoding.
        outdir (str): Location in which to save training history and best model.
        device (torch.device or str): PyTorch device.

    Returns:
        fastai.learner.Learner, (int, int): FastAI learner and a tuple of the
            number of input features and output classes.

    """
    if isinstance(config.model_config, ModelConfigCLAM):
        return _build_clam_learner(config, *args, **kwargs)
    else:
        return _build_fastai_learner(config, *args, **kwargs)


def _build_clam_learner(
    config: TrainerConfigFastAI,
    bags: List[str],
    targets: npt.NDArray,
    train_idx: npt.NDArray[np.int_],
    val_idx: npt.NDArray[np.int_],
    unique_categories: npt.NDArray,
    outdir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    **dl_kwargs
) -> Tuple[Learner, Tuple[int, int]]:
    """Build a FastAI learner for a CLAM model.

    Args:
        config (``TrainerConfigFastAI``): Trainer and model configuration.
        bags (list(str)): Path to .pt files (bags) with features, one per patient.
        targets (np.ndarray): Category labels for each patient, in the same
            order as ``bags``.
        train_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the training set.
        val_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the validation set.
        unique_categories (np.ndarray(str)): Array of all unique categories
            in the targets. Used for one-hot encoding.
        outdir (str): Location in which to save training history and best model.
        device (torch.device or str): PyTorch device.

    Returns:
        FastAI Learner, (number of input features, number of classes).
    """
    from ..clam.utils import loss_utils

    # Prepare device.
    device = torch_utils.get_device(device)

    # Prepare data.
    # Set oh_kw to a dictionary of keyword arguments for OneHotEncoder,
    # using the argument sparse=False if the sklearn version is <1.2
    # and sparse_output=False if the sklearn version is >=1.2.
    if version.parse(sklearn_version) < version.parse("1.2"):
        oh_kw = {"sparse": False}
    else:
        oh_kw = {"sparse_output": False}
    encoder = OneHotEncoder(**oh_kw).fit(unique_categories.reshape(-1, 1))

    # Build dataloaders.
    train_dataset = data_utils.build_clam_dataset(
        bags[train_idx],
        targets[train_idx],
        encoder=encoder,
        bag_size=config.bag_size
    )
    train_dl = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=False,
        device=device,
        **dl_kwargs
    )
    val_dataset = data_utils.build_clam_dataset(
        bags[val_idx],
        targets[val_idx],
        encoder=encoder,
        bag_size=None
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
        device=device,
        **dl_kwargs
    )

    # Prepare model.
    batch = train_dl.one_batch()
    n_features = batch[0][0].shape[-1]
    n_classes = batch[-1].shape[-1]
    config_size = config.model_fn.sizes[config.model_config.model_size]
    model_size = [n_features] + config_size[1:]
    log.info(f"Training model [bold]{config.model_fn.__name__}[/] "
             f"(size={model_size}, loss={config.loss_fn.__name__})")
    model = config.build_model(size=model_size, n_classes=n_classes)

    model.relocate()

    # Loss should weigh inversely to class occurences.
    loss_func = config.loss_fn()

    # Create learning and fit.
    dls = DataLoaders(train_dl, val_dl)
    learner = Learner(dls, model, loss_func=loss_func, metrics=[loss_utils.RocAuc()], path=outdir)

    return learner, (n_features, n_classes)


def _build_fastai_learner(
    config: TrainerConfigFastAI,
    bags: List[str],
    targets: npt.NDArray,
    train_idx: npt.NDArray[np.int_],
    val_idx: npt.NDArray[np.int_],
    unique_categories: npt.NDArray,
    outdir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    **dl_kwargs
) -> Tuple[Learner, Tuple[int, int]]:
    """Build a FastAI learner for an MIL model.

    Args:
        config (``TrainerConfigFastAI``): Trainer and model configuration.
        bags (list(str)): Path to .pt files (bags) with features, one per patient.
        targets (np.ndarray): Category labels for each patient, in the same
            order as ``bags``.
        train_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the training set.
        val_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the validation set.
        unique_categories (np.ndarray(str)): Array of all unique categories
            in the targets. Used for one-hot encoding.
        outdir (str): Location in which to save training history and best model.
        device (torch.device or str): PyTorch device.

    Returns:

        FastAI Learner, (number of input features, number of classes).
    """
    # Prepare device.
    device = torch_utils.get_device(device)

    # Prepare data.
    # Set oh_kw to a dictionary of keyword arguments for OneHotEncoder,
    # using the argument sparse=False if the sklearn version is <1.2
    # and sparse_output=False if the sklearn version is >=1.2.
    if version.parse(sklearn_version) < version.parse("1.2"):
        oh_kw = {"sparse": False}
    else:
        oh_kw = {"sparse_output": False}
    encoder = OneHotEncoder(**oh_kw).fit(unique_categories.reshape(-1, 1))

    # Build dataloaders.
    train_dataset = data_utils.build_dataset(
        bags[train_idx],
        targets[train_idx],
        encoder=encoder,
        bag_size=config.bag_size,
        use_lens=config.model_config.use_lens
    )
    train_dl = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=False,
        device=device,
        **dl_kwargs
    )
    val_dataset = data_utils.build_dataset(
        bags[val_idx],
        targets[val_idx],
        encoder=encoder,
        bag_size=None,
        use_lens=config.model_config.use_lens
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
        device=device,
        **dl_kwargs
    )

    # Prepare model.
    batch = train_dl.one_batch()
    n_in, n_out = batch[0].shape[-1], batch[-1].shape[-1]
    log.info(f"Training model [bold]{config.model_fn.__name__}[/] "
             f"(in={n_in}, out={n_out}, loss={config.loss_fn.__name__})")
    model = config.build_model(n_in, n_out).to(device)
    if hasattr(model, 'relocate'):
        model.relocate()

    # Loss should weigh inversely to class occurences.
    counts = pd.value_counts(targets[train_idx])
    weight = counts.sum() / counts
    weight /= weight.sum()
    weight = torch.tensor(
        list(map(weight.get, encoder.categories_[0])), dtype=torch.float32
    ).to(device)
    loss_func = nn.CrossEntropyLoss(weight=weight)

    # Create learning and fit.
    dls = DataLoaders(train_dl, val_dl)
    learner = Learner(dls, model, loss_func=loss_func, metrics=[RocAuc()], path=outdir)

    return learner, (n_in, n_out)


def _build_multimodal_learner(
    config: TrainerConfigFastAI,
    bags: List[List[str]],
    targets: npt.NDArray,
    train_idx: npt.NDArray[np.int_],
    val_idx: npt.NDArray[np.int_],
    unique_categories: npt.NDArray,
    n_magnifications: int,
    *,
    outdir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    **dl_kwargs
) -> Tuple[Learner, Tuple[int, int]]:
    """Build a FastAI learner for an MIL model.

    Args:
        config (``TrainerConfigFastAI``): Trainer and model configuration.
        bags (list(str)): Path to .pt files (bags) with features, one per patient.
        targets (np.ndarray): Category labels for each patient, in the same
            order as ``bags``.
        train_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the training set.
        val_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the validation set.
        unique_categories (np.ndarray(str)): Array of all unique categories
            in the targets. Used for one-hot encoding.
        outdir (str): Location in which to save training history and best model.
        device (torch.device or str): PyTorch device.

    Returns:

        FastAI Learner, (number of input features, number of classes).
    """
    # Prepare device.
    device = torch_utils.get_device(device)

    # Prepare data.
    # Set oh_kw to a dictionary of keyword arguments for OneHotEncoder,
    # using the argument sparse=False if the sklearn version is <1.2
    # and sparse_output=False if the sklearn version is >=1.2.
    if version.parse(sklearn_version) < version.parse("1.2"):
        oh_kw = {"sparse": False}
    else:
        oh_kw = {"sparse_output": False}
    encoder = OneHotEncoder(**oh_kw).fit(unique_categories.reshape(-1, 1))

    # Build dataloaders.
    train_dataset = data_utils.build_multibag_dataset(
        bags[train_idx],
        targets[train_idx],
        encoder=encoder,
        bag_size=config.bag_size,
        n_bags=n_magnifications,
        use_lens=config.model_config.use_lens
    )
    train_dl = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=False,
        device=device,
        **dl_kwargs
    )
    val_dataset = data_utils.build_multibag_dataset(
        bags[val_idx],
        targets[val_idx],
        encoder=encoder,
        bag_size=None,
        n_bags=n_magnifications,
        use_lens=config.model_config.use_lens
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        device=device,
        **dl_kwargs
    )

    # Prepare model.
    batch = train_dl.one_batch()  # batch returns features, lens, and targets
    if config.model_config.use_lens:
        n_in = [b[0].shape[-1] for b in batch[:-1]]
    else:
        n_in = [b.shape[-1] for b in batch[:-1][0]]
    n_out = batch[-1].shape[-1]

    log.info(f"Training model [bold]{config.model_fn.__name__}[/] "
             f"(in={n_in}, out={n_out}, loss={config.loss_fn.__name__})")
    model = config.build_model(n_in, n_out).to(device)
    if hasattr(model, 'relocate'):
        model.relocate()

    # Loss should weigh inversely to class occurences.
    counts = pd.value_counts(targets[train_idx])
    weight = counts.sum() / counts
    weight /= weight.sum()
    weight = torch.tensor(
        list(map(weight.get, encoder.categories_[0])), dtype=torch.float32
    ).to(device)
    loss_func = nn.CrossEntropyLoss(weight=weight)

    # Create learning and fit.
    dls = DataLoaders(train_dl, val_dl)
    learner = Learner(dls, model, loss_func=loss_func, metrics=[RocAuc()], path=outdir)

    return learner, (n_in, n_out)
