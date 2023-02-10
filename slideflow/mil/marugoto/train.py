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
    n_epoch: int = 32,
    outdir: Optional[str] = None,
    device: Union[str, torch.device] = 'cuda'
):
    #bags = list of .pt files
    #targets = is the int label

    if isinstance(device, str):
        device = torch.device('cuda')

    # Prepare data for Marugoto MIL
    encoder = OneHotEncoder(sparse=False).fit(unique_categories.reshape(-1, 1))

    train_dataset = marugoto.data.build_dataset(bags[train_idx], targets[train_idx], encoder=encoder, bag_size=512)
    train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1, drop_last=True, device=device)

    val_dataset = marugoto.data.build_dataset(bags[val_idx], targets[val_idx], encoder=encoder, bag_size=None)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count(), device=device)

    batch = train_dl.one_batch()

    model = Marugoto_MIL(batch[0].shape[-1], batch[-1].shape[-1]).to(device)

    # weigh inversely to class occurances
    counts = pd.value_counts(targets[train_idx])
    weight = counts.sum() / counts
    weight /= weight.sum()
    # reorder according to vocab
    weight = torch.tensor(
        list(map(weight.get, encoder.categories_[0])), dtype=torch.float32
    ).to(device)
    loss_func = nn.CrossEntropyLoss(weight=weight)

    dls = DataLoaders(train_dl, val_dl)
    learn = Learner(dls, model, loss_func=loss_func, metrics=[RocAuc()], path=outdir)

    cbs = [
        SaveModelCallback(fname=f"best_valid"),
        CSVLogger(),
    ]

    learn.fit_one_cycle(n_epoch=n_epoch, lr_max=1e-4, cbs=cbs)

    return learn