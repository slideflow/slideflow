import torch
import pandas as pd
import numpy as np
import numpy.typing as npt
import slideflow as sf
from typing import List, Optional, Union, Tuple
from torch import nn
from torch import Tensor
import fastai.optimizer as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn import __version__ as sklearn_version
from packaging import version
import multiprocessing as mp
from torch.utils.data import DataLoader as TorchDataLoader
from fastai.vision.all import (
    DataLoader, DataLoaders, Learner, RocAuc, SaveModelCallback, CSVLogger, FetchPredsCallback, Callback
)
from fastai.callback.schedule import ParamScheduler
from fastai.learner import Metric
from fastai.torch_core import to_detach, flatten_check
from fastai.metrics import mae
import fastai.metrics as fastai_metrics
from slideflow import log
import slideflow.mil.data as data_utils
from slideflow.model import torch_utils
from .._params import TrainerConfigFastAI
import logging
from functools import partial
import inspect

from lifelines.utils import concordance_index

#import custom losses and metrics
from pathbench import losses, metrics, callbacks
from pathbench.utils.metrics import ConcordanceIndex

# -----------------------------------------------------------------------------

"""
This function retrieves the optimizer class based on the optimizer name.
Optimizer can be any torch optimizer class.
"""
def retrieve_optimizer(optimizer_name):
    optimizer_class = getattr(optim, optimizer_name)
    return optimizer_class


"""
This function retrieves the custom loss class based on the loss name.
Loss can be any loss class as defined in pathbench/utils/losses.py
"""
def retrieve_custom_loss(loss_name):
    logging.info(f"Retrieving custom loss: {loss_name}")
    loss_class = getattr(losses, loss_name)
    return loss_class  # Return the loss class without instantiating


def retrieve_custom_callback(callback_name):
    """
    This function retrieves the custom callback class based on the callback name.
    Callback can be any callback class as defined in pathbench/utils/callbacks.py or any fastai callback.
    """
    logging.info(f"Retrieving custom callback: {callback_name}")
    # Check if the callback is a fastai callback
    if hasattr(Callback, callback_name):
        callback_class = getattr(Callback, callback_name)
    else:
        # Otherwise, assume it's a custom callback defined in pathbench/utils/callbacks.py
        callback_class = getattr(callbacks, callback_name)
    return callback_class()  # Instantiate the callback class

"""
This function retrieves the custom metric class based on the metric name.
Metric can be any metric class as defined in pathbench/utils/metrics.py or fastai.metrics
"""
def retrieve_custom_metric(metric_name):
    #Check if metric is in fastai.metrics
    if hasattr(fastai_metrics, metric_name):
        metric_class = getattr(fastai_metrics, metric_name)
    else:
        metric_class = getattr(metrics, metric_name)
    return metric_class()  # Instantiate the metric class




class PadToMinLength:
    """Pad or truncate each bag tensor in a batch to the minimum bag‐length in that batch."""
    def __call__(self, sample):
        # Find only the multi-dim tensors (your bags)
        tensor_items = [t for t in sample if isinstance(t, torch.Tensor) and t.dim()>0]
        if not tensor_items:
            return sample   # nothing to pad

        # Compute the shortest length along dim=0
        min_len = min(t.size(0) for t in tensor_items)

        new_sample = []
        for item in sample:
            # Only pad/truncate multi-dim tensors
            if isinstance(item, torch.Tensor) and item.dim()>0:
                L = item.size(0)
                if L > min_len:
                    item = item[:min_len]            # truncate
                elif L < min_len:
                    pad_amt = min_len - L
                    # For a [D, C, F] tensor, pad = (F_l, F_r, C_l, C_r, D_l, D_r)
                    item = F.pad(item, (0,0, 0,0, 0,pad_amt))
            new_sample.append(item)

        return new_sample

from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F

def slide_collate(samples):
    # samples is List[ (Tensor, label) ]
    xs, ys = zip(*samples)

    # 1) normalize each x → always 2-D (n_slides, feat_dim)
    proc_xs = []
    for x in xs:
        # if extra dims (e.g. [B, N, C, F]) flatten everything except last dim
        if x.dim() > 2:
            x = x.flatten(0, -2)
        # if it's 1-D (i.e. [F]), treat as single slide → [1, F]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        proc_xs.append(x)

    # 2) pad/truncate on the slide axis (dim=0) so all have same n_slides
    max_slides = max(x.shape[0] for x in proc_xs)
    padded = []
    for x in proc_xs:
        n, f = x.shape
        if n < max_slides:
            # pad only at end of slide axis
            x = F.pad(x, (0,0, 0, max_slides - n))
        else:
            x = x[:max_slides]
        padded.append(x)

    # 3) stack into [batch_size, n_slides, feat_dim]
    batch_x = torch.stack(padded, dim=0)
    batch_y = default_collate(ys)
    return batch_x, batch_y

def train(learner, config, pb_config=None, callbacks=None):
    """Train an attention-based multi-instance learning model with FastAI.

    Args:
        learner (``fastai.learner.Learner``): FastAI learner.
        config (``TrainerConfigFastAI``): Trainer and model configuration.
        pb_config (dict): PathBench configuration. Defaults to None.

    Keyword args:
        callbacks (list(fastai.Callback)): FastAI callbacks. Defaults to None.
    """
    if pb_config is not None:
        cbs = [
            SaveModelCallback(fname=f"best_valid", monitor=pb_config['experiment']['best_epoch_based_on']),
            CSVLogger(),
        ]
    else:
        cbs = [
            SaveModelCallback(fname=f"best_valid", monitor=config.save_monitor),
            CSVLogger(),
        ]
    if callbacks:
        cbs += callbacks

    #Check for override of learning parameters
    if pb_config is not None:
        #Overwrite learning rate if specified
        if 'lr' in pb_config['experiment']:
            logging.info(f"Overriding learning rate to {pb_config['experiment']['lr']}")
            lr = float(pb_config['experiment']['lr'])
            config.fit_one_cycle = False  # Disable fit_one_cycle if lr is specified
        else:
            if config.lr is None:
                try:
                    lr = learner.lr_find().valley
                    log.info(f"Using auto-detected learning rate: {lr}")
                except:
                    lr = 1e-3
                    log.info(f"Failed to find learning rate, using default: {lr}")
            else:
                lr = config.lr

        #Overwrite weight decay if specified
        if 'wd' in pb_config['experiment']:
            logging.info(f"Overriding weight decay to {pb_config['experiment']['wd']}")
            wd = float(pb_config['experiment']['wd'])
        else:
            wd = config.wd
        
        #Overwrite epochs if specified
        if 'epochs' in pb_config['experiment']:
            logging.info(f"Overriding epochs to {pb_config['experiment']['epochs']}")
            epochs = pb_config['experiment']['epochs']
        else:
            epochs = config.epochs

        #Add schedulers if specified
        if 'schedulers' in pb_config['experiment']:
            for scheduler in pb_config['experiment']['schedulers']:
                cbs.append(retrieve_custom_callback(scheduler))

        # Log the callbacks being used
        logging.info(f"Using callbacks: {[type(cb).__name__ for cb in cbs]}")

        learner.fit(n_epoch=epochs, lr=lr, wd=wd, cbs=cbs)
        return learner

    if config.fit_one_cycle:
        if config.lr is None:
            #Try lr.find to get the learning rate
            try:
                lr = learner.lr_find().valley
                log.info(f"Using auto-detected learning rate: {lr}")
            except:
                lr = 1e-3
                log.info(f"Failed to find learning rate, using default: {lr}")
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
    return _build_fastai_learner(config, *args, **kwargs)

    
def _build_fastai_learner(
    config,
    bags: List[str],
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    unique_categories: np.ndarray,
    outdir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    **dl_kwargs
) -> Tuple[Learner, Tuple[int, int]]:
    """Build a FastAI learner for an MIL model."""

    # Retrieve experiment config and number of workers
    pb_config = dl_kwargs.get("pb_config") or {}
    exp_cfg = pb_config.get('experiment', {})
    num_workers = exp_cfg.get('num_workers', dl_kwargs.get('num_workers', 0))
    persistent_workers = exp_cfg.get('persistent_workers', False)
    class_weighting = exp_cfg.get('class_weighting', False)

    config_dict = config.to_dict() # Convert to dictionary
    logging.info(f"Building FastAI learner with config: {config}")

    # Extract parameters from config
    encoder_layers = config_dict['encoder_layers']
    dropout_p = config_dict['dropout_p']
    z_dim = config_dict['z_dim']
    activation_function = config_dict['activation_function']
    problem_type = goal = config_dict['task']
    slide_level = config_dict['slide_level']

    if num_workers > 0:
        ctx = pb_config['experiment'].get('multiprocessing_context', 'spawn')
    else:
        ctx = None

    #Determine whether slide-level or bag-level training is required
    if slide_level:
        logging.info("Building slide-level FastAI learner....")
    else:
        logging.info("Building bag-level FastAI learner....")

    # Select the appropriate loss function based on the problem type
    if problem_type == "classification":
        default_loss_cls = nn.CrossEntropyLoss
    elif problem_type == "regression":
        default_loss_cls = nn.MSELoss
    elif problem_type == "survival":
        default_loss_cls = retrieve_custom_loss("CoxPHLoss")
    elif problem_type == 'survival_discrete':
        default_loss_cls = retrieve_custom_loss("NLLLogisticHazardLoss")
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")


    # Instantiate loss function, allowing override
    loss_name = dl_kwargs.get("loss")
    loss_cls = retrieve_custom_loss(loss_name) if loss_name else default_loss_cls
    if problem_type == 'classification' and class_weighting:
        counts = pd.value_counts(targets[train_idx])
        w = counts.sum() / counts
        w = (w / w.sum()).to_dict()
        class_weight = torch.tensor([w.get(c, 1.0) for c in unique_categories], dtype=torch.float32)
        loss_function = loss_cls(weight=class_weight)
    elif problem_type == 'survival' and class_weighting:
        durations = torch.tensor(targets[:, 0], dtype=torch.float32)
        events = torch.tensor(targets[:, 1], dtype=torch.int64)
        num_events = events.sum()
        num_censored = events.numel() - num_events
        event_weight = (num_censored / (num_events + num_censored)).item()
        censored_weight = (num_events / (num_events + num_censored)).item()
        loss_function = loss_cls(event_weight=event_weight, censored_weight=censored_weight)
    elif problem_type == 'survival_discrete' and class_weighting:
        # Compute weight per discrete time bin
        bins = targets[:, 0].astype(int)
        counts = pd.value_counts(bins[train_idx])
        w = counts.sum() / counts
        w = (w / w.sum()).to_dict()
        bin_weights = torch.tensor([w.get(b, 1.0) for b in np.unique(bins)], dtype=torch.float32)
        loss_function = loss_cls(weight=bin_weights)
    else:
        loss_function = loss_cls()

    # Prepare device.
    device = torch.device(device if device else 'cuda' if torch.cuda.is_available() else 'cpu')

    logging.debug(f"Problem type: {problem_type}")

    # === TARGETS & ENCODER PREPARATION ===
    encoder = None
    if slide_level:
        # Slide-level handling:
        if problem_type == "classification":
            encoder = OneHotEncoder(sparse_output=False).fit(unique_categories.reshape(-1, 1))
        elif problem_type in ["survival", "survival_discrete"]:
            #Make sure events are integer valued
            targets[:, 1] = targets[:, 1].astype(int)
            #Make sure durations are float valued in the case of survival
            if problem_type == "survival":
                targets[:, 0] = targets[:, 0].astype(float)
            elif problem_type == "survival_discrete":
                #Convert time bins to int
                targets[:, 0] = targets[:, 0].astype(int)
                # Use time bins to define the output dimension.
                encoder = OneHotEncoder(sparse_output=False).fit(targets[:, 0].reshape(-1, 1))
                logging.debug(f"Encoder categories: {encoder.categories_}")
            logging.debug(f"Events shape: {targets[:, 1].shape}, Events  dtype: {targets[:, 1].dtype}")
            logging.debug(f"Durations shape: {targets[:, 0].shape}, Durations dtype: {targets[:, 0].dtype}")
            #Check unique durations values
            unique_durations = np.unique(targets[:, 0])
            logging.debug(f"Unique durations: {unique_durations}")
        else:  # regression
            targets = np.array(targets, dtype=np.float32)
    else:
        # Bag-level handling.
        if problem_type == "classification":
            encoder = OneHotEncoder(sparse_output=False).fit(unique_categories.reshape(-1, 1))

        if problem_type == 'survival_discrete':
            time_bins = targets[:, 0].astype(int)
            logging.debug(f"Time bins shape: {time_bins.shape}")
            time_bin_centers = np.unique(time_bins)
            logging.debug(f"Unique time bins: {time_bin_centers}")
            encoder = OneHotEncoder(sparse_output=False).fit(time_bins.reshape(-1, 1))
            targets[:, 0] = targets[:, 0].astype(int)
            targets[:, 1] = targets[:, 1].astype(int)
            logging.debug("Encoder fitted for time bins")

        if problem_type in ["survival", "regression"]:
            targets = np.array(targets, dtype=np.float32)
            if problem_type == "survival":
                durations = targets[:, 0].astype(np.float32)
                events = targets[:, 1].astype(np.float32)
                durations = torch.tensor(durations, dtype=torch.float32)
                events = torch.tensor(events, dtype=torch.int64)
                logging.debug(f"Durations shape: {durations.shape}, Events shape: {events.shape}")
                logging.debug(f"Durations dtype: {durations.dtype}, Events dtype: {events.dtype}")
                # Check if events are binary (0 or 1)
                if not torch.all(torch.isin(events, torch.tensor([0, 1]))):
                    raise ValueError("Events must be binary (0 or 1) for survival analysis.")

            targets = torch.tensor(targets, dtype=torch.float32)

    # === DATASET & DATALOADER CREATION ===
    if slide_level:
        logging.info("Building slide-level datasets....")
        #Log encoder and targets
        if encoder is not None:
            logging.debug(f"Encoder categories: {encoder.categories_}")
        logging.debug(f"Targets shape: {targets.shape}")
        train_dataset = data_utils.build_slide_dataset(
            [bags[i] for i in train_idx],
            targets[train_idx],
            survival_discrete=(problem_type == "survival_discrete"),
            encoder=encoder,
            bag_size=None
        )
        val_dataset = data_utils.build_slide_dataset(
            [bags[i] for i in val_idx],
            targets[val_idx],
            survival_discrete=(problem_type == "survival_discrete"),
            encoder=encoder,
            bag_size=None
        )

        #Log one sample from the dataset
        logging.debug(f"Sample from slide-level train dataset: {train_dataset[0]}")
        # Dataloaders for slide-level (fixed-length feature vectors)
        train_dl = TorchDataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=dl_kwargs.get("num_workers", num_workers),
            drop_last=True,
            persistent_workers=persistent_workers,
            multiprocessing_context=ctx,
            collate_fn=slide_collate,  # Custom collate function to pad bags
        )
        val_dl = TorchDataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=dl_kwargs.get("num_workers", num_workers),
            multiprocessing_context=ctx,
            collate_fn=slide_collate,  # Custom collate function to pad bags
        )
        #Log one sample from the dataloader
        logging.debug(f"Sample from slide-level train dataloader: {next(iter(train_dl))}")
    else:
        # Bag-level datasets (each bag may have variable length and requires padding)
        train_dataset = data_utils.build_dataset(
            bags[train_idx],
            targets[train_idx],
            encoder=encoder,
            bag_size=config.bag_size,
            use_lens=config.model_config.use_lens,
            survival_discrete=(problem_type == "survival_discrete")
        )
        val_dataset = data_utils.build_dataset(
            bags[val_idx],
            targets[val_idx],
            encoder=encoder,
            bag_size=None,
            use_lens=config.model_config.use_lens,
            survival_discrete=(problem_type == "survival_discrete")
        )
        train_dl = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers= persistent_workers,
            drop_last=config.drop_last,
            device=device,
            multiprocessing_context=ctx,
            **dl_kwargs
        )
        val_dl = DataLoader(
            val_dataset,
            batch_size=1 if problem_type == "classification" else config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            device=device,
            after_item=PadToMinLength(),
            multiprocessing_context=ctx,
            **dl_kwargs
        )

    # Determine input/output dimensions
    sample = next(iter(train_dl))
    n_in = sample[0].shape[-1]
    if slide_level:
        if problem_type == 'classification': n_out = len(unique_categories)
        elif problem_type in ['regression', 'survival']: n_out = 1
        else: n_out = encoder.categories_[0].size
    else:
        n_out = sample[-1].shape[-1] if hasattr(sample[-1], 'shape') else 1
        if problem_type in ['regression', 'survival']: n_out = 1

    #Log the bag shape in the first sample
    logging.info(f"First bag shape: {sample[0].shape if isinstance(sample[0], torch.Tensor) else 'N/A'}")
    logging.info(f"Model dims: in={n_in}, out={n_out}, z_dim={z_dim}")


    logging.info(f"Training model {config.model_fn.__name__} (in={n_in}, out={n_out}, "
                    f"z_dim={z_dim}, encoder_layers={encoder_layers}, dropout_p={dropout_p})")

    model = config.build_model(
        n_in, n_out,
        z_dim=config.z_dim,
        encoder_layers=config.encoder_layers,
        dropout_p=config.dropout_p,
        activation_function=config.activation_function,
        goal=config.task
    ).to(device)

    sig = inspect.signature(model.forward)
    supports_attention = 'return_attention' in sig.parameters

    # Wrap the model's forward method to save attention weights if supported
    _orig_fwd = model.forward
    def _fwd_save_att(bags, *args, **kwargs):
        if supports_attention:
            preds, att = _orig_fwd(bags, return_attention=True, *args, **kwargs)
        else:
            # plain model → logits only
            preds = _orig_fwd(bags, *args, **kwargs)
            # uniform fallback attention
            B, N, _ = bags.shape
            att = torch.ones(B, N, device=bags.device) / N
        model._last_attention = att
        return preds
    model.forward = _fwd_save_att


    raw_loss_kwargs = {}
    if problem_type == 'classification' and class_weighting:
        raw_loss_kwargs['weight'] = class_weight
    elif problem_type == 'survival' and class_weighting:
        raw_loss_kwargs['event_weight']    = event_weight
        raw_loss_kwargs['censored_weight'] = censored_weight
    elif problem_type == 'survival_discrete' and class_weighting:
        raw_loss_kwargs['weight'] = bin_weights

    # Instantiate the “raw” loss (may or may not have require_attention=True)
    loss_cls = retrieve_custom_loss(loss_name) if loss_name else default_loss_cls
    raw_loss = loss_cls(**raw_loss_kwargs)

    #Wrap it so FastAI always calls loss(preds, targets)
    class LossWithOptionalAttention(nn.Module):
        def __init__(self, base_loss, model):
            super().__init__()
            self.base_loss = base_loss
            self.model     = model

        def forward(self, preds, targets):
            if getattr(self.base_loss, 'require_attention', False):
                return self.base_loss(
                    preds, targets,
                    attention_weights=self.model._last_attention
                )
            return self.base_loss(preds, targets)

    loss_function = LossWithOptionalAttention(raw_loss, model)

    # === METRICS ===
    if 'custom_metrics' in exp_cfg:
        metrics = [retrieve_custom_metric(m) for m in exp_cfg['custom_metrics']]
    else:
        if problem_type == 'classification': metrics = [RocAuc()]
        elif problem_type == 'regression': metrics = [mae]
        else: metrics = [ConcordanceIndex()]

    logging.debug(f"Targets shape: {targets.shape}")
    if targets.ndim > 1 and targets.shape[1] == 1:
        targets = targets.flatten()

    # === CREATE LEARNER ===
    dls = DataLoaders(train_dl, val_dl)
    opt_func = retrieve_optimizer(dl_kwargs['optimizer']) if 'optimizer' in dl_kwargs else optim.Adam
    learner = Learner(
        dls, model, loss_func=loss_function, metrics=metrics,
        path=outdir, opt_func=opt_func
    )
    return learner, (n_in, n_out)