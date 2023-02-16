import torch
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import List, Optional, Union
from torch import nn
from sklearn.preprocessing import OneHotEncoder
from fastai.vision.all import (
    DataLoader, DataLoaders, Learner, RocAuc, SaveModelCallback, CSVLogger
)

from slideflow import log
from .. import marugoto
from .._params import TrainerConfigFastAI, ModelConfigCLAM

# -----------------------------------------------------------------------------

def train(config, *args, **kwargs):
    """Train an attention-based multi-instance learning model.

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
    """

    if isinstance(config.model_config, ModelConfigCLAM):
        return _train_clam(config, *args, **kwargs)
    else:
        return _train_marugoto(config, *args, **kwargs)


def _train_clam(
    config: TrainerConfigFastAI,
    bags: List[str],
    targets: npt.NDArray,
    train_idx: npt.NDArray[np.int_],
    val_idx: npt.NDArray[np.int_],
    unique_categories: npt.NDArray,
    outdir: Optional[str] = None,
    device: Union[str, torch.device] = 'cuda',
) -> Learner:
    """Train an attention-based multi-instance learning model.

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
    """
    from ..clam.utils import loss_utils

    # Prepare device.
    if isinstance(device, str):
        device = torch.device('cuda')

    # Prepare data for Marugoto MIL
    encoder = OneHotEncoder(sparse=False).fit(unique_categories.reshape(-1, 1))

    # Build dataloaders.
    train_dataset = marugoto.data.build_clam_dataset(bags[train_idx], targets[train_idx], encoder=encoder, bag_size=config.bag_size)
    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False, device=device)
    val_dataset = marugoto.data.build_clam_dataset(bags[val_idx], targets[val_idx], encoder=encoder, bag_size=None)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, persistent_workers=True, device=device)

    # Prepare model.
    log.info(f"Training model {config.model_fn.__name__}, loss={config.loss_fn.__name__}")
    batch = train_dl.one_batch()
    model = config.model_fn(size=[batch[0][0].shape[-1], 256, 128], n_classes=batch[-1].shape[-1])
    model.relocate()

    # Loss should weigh inversely to class occurences.
    loss_func = config.loss_fn()

    # Create learning and fit.
    dls = DataLoaders(train_dl, val_dl)
    learn = Learner(dls, model, loss_func=loss_func, metrics=[loss_utils.RocAuc()], path=outdir)
    cbs = [
        SaveModelCallback(fname=f"best_valid"),
        CSVLogger(),
    ]
    learn.fit(n_epoch=config.epochs, lr=config.lr, wd=config.wd, cbs=cbs)
    del val_dl
    return learn


def _train_marugoto(
    config: TrainerConfigFastAI,
    bags: List[str],
    targets: npt.NDArray,
    train_idx: npt.NDArray[np.int_],
    val_idx: npt.NDArray[np.int_],
    unique_categories: npt.NDArray,
    outdir: Optional[str] = None,
    device: Union[str, torch.device] = 'cuda',
) -> Learner:
    """Train an attention-based multi-instance learning model.

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
    """
    # Prepare device.
    if isinstance(device, str):
        device = torch.device('cuda')

    # Prepare data for Marugoto MIL
    encoder = OneHotEncoder(sparse=False).fit(unique_categories.reshape(-1, 1))

    # Build dataloaders.
    train_dataset = marugoto.data.build_dataset(bags[train_idx], targets[train_idx], encoder=encoder, bag_size=config.bag_size)
    train_dl = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1, drop_last=True, device=device)
    val_dataset = marugoto.data.build_dataset(bags[val_idx], targets[val_idx], encoder=encoder, bag_size=None)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, persistent_workers=True, device=device)

    # Prepare model.
    log.info(f"Training model {config.model_fn.__name__}, loss={config.loss_fn.__name__}")
    batch = train_dl.one_batch()
    model = config.model_fn(batch[0].shape[-1], batch[-1].shape[-1]).to(device)

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
    learn = Learner(dls, model, loss_func=loss_func, metrics=[RocAuc()], path=outdir)
    cbs = [
        SaveModelCallback(fname=f"best_valid"),
        CSVLogger(),
    ]
    if config.fit_one_cycle:
        learn.fit_one_cycle(n_epoch=config.epochs, lr_max=config.lr_max, cbs=cbs)
    else:
        learn.fit(n_epoch=config.epochs, lr=config.lr, wd=config.wd, cbs=cbs)
    del val_dl
    return learn