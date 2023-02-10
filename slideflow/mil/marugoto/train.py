import os
import torch
import pandas as pd
import numpy as np
import numpy.typing as npt

from typing import List, Optional, Union
from torch import nn
from slideflow.mil import marugoto
from sklearn.preprocessing import OneHotEncoder
from fastai.vision.all import DataLoader, DataLoaders, Learner, RocAuc, SaveModelCallback, CSVLogger

from .model import Marugoto_MIL

def train_mil(
    bags: List[str],
    targets: npt.NDArray,
    train_idx: npt.NDArray[np.int_],
    val_idx: npt.NDArray[np.int_],
    unique_categories: npt.NDArray,
    epochs: int = 32,
    outdir: Optional[str] = None,
    device: Union[str, torch.device] = 'cuda',
    lr_max: float = 1e-4
) -> Learner:
    """Train an attention-based multi-instance learning model.

    Args:
        bags (list(str)): Path to .pt files (bags) with features, one per patient.
        targets (np.ndarray): Category labels for each patient, in the same
            order as ``bags``.
        train_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the training set.
        val_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the validation set.
        unique_categories (np.ndarray(str)): Array of all unique categories
            in the targets. Used for one-hot encoding.
        epochs (int): Number of epochs to train.
        outdir (str): Location in which to save training history and best model.
        device (torch.device or str): PyTorch device.
        lr_max (float): Max learning rate. Used for
            ``fasti.vision.Learner.fit_one_cycle(lr_max=...)``.
            Defaults to 1e-4.
    """
    # Prepare device.
    if isinstance(device, str):
        device = torch.device('cuda')

    # Prepare data for Marugoto MIL
    encoder = OneHotEncoder(sparse=False).fit(unique_categories.reshape(-1, 1))

    # Build dataloaders.
    train_dataset = marugoto.data.build_dataset(bags[train_idx], targets[train_idx], encoder=encoder, bag_size=512)
    train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1, drop_last=True, device=device)
    val_dataset = marugoto.data.build_dataset(bags[val_idx], targets[val_idx], encoder=encoder, bag_size=None)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count(), device=device)

    # Prepare model.
    batch = train_dl.one_batch()
    model = Marugoto_MIL(batch[0].shape[-1], batch[-1].shape[-1]).to(device)

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
    learn.fit_one_cycle(n_epoch=epochs, lr_max=lr_max, cbs=cbs)
    return learn