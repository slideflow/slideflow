# slideflow/mil/train/_lightning.py
# -*- coding: utf-8 -*-
"""
Lightning trainer that mirrors the FastAI functionality in your codebase:

- Slide-level & bag-level training (variable-length bags supported)
- Dense OneHotEncoder for class labels / discrete-survival bins (sklearn >=1.2)
- Attention-aware losses (passes attention to custom losses when required)
- Custom PathBench losses (CrossEntropy variants, CoxPH, discrete survival, etc.)
- Class weighting for classification; event/censor weighting for CoxPH
- Metrics roughly equivalent to FastAI defaults:
    - classification: ROC AUC (binary/multiclass), accuracy
    - regression: MAE
    - survival: Concordance index (C-index)
- Trainer overrides from pb_config.experiment: lr, wd, epochs, optimizer,
  num_workers, persistent_workers, multiprocessing_context, best_epoch_based_on
- Optional schedulers: OneCycleLR, CosineAnnealingLR, StepLR, ExponentialLR,
  ReduceLROnPlateau (configurable via pb_config.experiment.schedulers)
- CSV logging and best-epoch checkpointing
- Backwards-compatible entrypoint `train_lightning(config, *args, **kwargs)`
"""



from __future__ import annotations
import os, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:true,max_split_size_mb:128")
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import inspect
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data._utils.collate import default_collate

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from slideflow import log
import slideflow.mil.data as data_utils

# Custom losses from PathBench
from pathbench import losses as pb_losses

# Metrics
from lifelines.utils import concordance_index as lifelines_cindex
from sklearn import metrics as sk_metrics
import sklearn
from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder
from packaging import version as _pkg_version


# ---------------------------------------------------------------------
# Utilities matching your FastAI pipeline
# ---------------------------------------------------------------------

def _make_ohe_dense() -> SkOneHotEncoder:
    """Return a dense OneHotEncoder irrespective of sklearn version."""
    if _pkg_version.parse(sklearn.__version__) >= _pkg_version.parse("1.2"):
        return SkOneHotEncoder(sparse_output=False, handle_unknown="ignore")
    return SkOneHotEncoder(sparse=False, handle_unknown="ignore")


def _normalize_feat_tensor(x: torch.Tensor) -> torch.Tensor:
    """Flatten all but last feature dim; ensure shape (N, D)."""
    if x.dim() > 2:
        x = x.flatten(0, -2)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x


def slide_collate(samples: Sequence[Any]) -> Tuple[torch.Tensor, Any]:
    """
    Slide-level: make robust to (x,y), (x,*,y), dicts, etc.
    Pads/truncates slides so all samples align, stacks to (B, N, D).
    """
    xs, ys = [], []
    for s in samples:
        if isinstance(s, dict):
            x = s.get("x") or s.get("bags") or s.get("feats") or s
            y = s.get("y")
        elif isinstance(s, (list, tuple)):
            x = s[0]
            y = s[-1]  # tolerate extra middle fields
        else:
            raise TypeError(f"Unexpected sample type in slide_collate: {type(s)}")

        x = _normalize_feat_tensor(x)
        xs.append(x)
        ys.append(y)

    max_n = max(x.shape[0] for x in xs)
    padded = []
    for x in xs:
        n, d = x.shape
        if n < max_n:
            x = F.pad(x, (0, 0, 0, max_n - n))
        padded.append(x[:max_n])

    batch_x = torch.stack(padded, dim=0)  # (B, N, D)
    batch_y = default_collate(ys)
    return batch_x, batch_y


def mil_collate(samples: Sequence[Any]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
    """
    Bag-level collation tolerant to:
      - (x, y)
      - (x, lens, y) or (x, y, lens)
      - dicts with keys x/bags/feats, y, lens/lengths/bag_lens
      - extra metadata tuples; first->x, last->y, auto-detect lens
    Returns: ((B, N, D), lens[B]), y
    """
    xs, ys, lens_list = [], [], []

    def _maybe_lens(obj):
        # Identify a bag-length vector or scalar-ish
        if torch.is_tensor(obj):
            if obj.ndim == 1 and obj.dtype in (torch.long, torch.int64, torch.int32):
                return obj
            if obj.ndim == 0 and obj.dtype in (torch.long, torch.int64, torch.int32):
                return obj
        if isinstance(obj, (list, tuple)) and all(isinstance(t, (int, np.integer)) for t in obj):
            return torch.tensor(obj, dtype=torch.long)
        if isinstance(obj, (int, np.integer)):
            return torch.tensor(int(obj), dtype=torch.long)
        return None

    for s in samples:
        x = y = lens = None

        if isinstance(s, dict):
            x = s.get("x") or s.get("bags") or s.get("feats") or s
            y = s.get("y")
            lens = s.get("lens") or s.get("lengths") or s.get("bag_lens")
        elif isinstance(s, (list, tuple)):
            # always take first as x, last as y; inspect middle for lens if present
            x = s[0]
            y = s[-1]
            if len(s) >= 3:
                mid = s[1]
                lens = _maybe_lens(mid)
                if lens is None and len(s) > 3:
                    # try any middle item
                    for mid_i in s[1:-1]:
                        lens = _maybe_lens(mid_i)
                        if lens is not None:
                            break
        else:
            raise TypeError(f"Unexpected sample type in mil_collate: {type(s)}")

        x = _normalize_feat_tensor(x)
        xs.append(x)
        ys.append(y)

        # If lens not provided, infer from x (#instances in the bag)
        if lens is None:
            lens_list.append(x.shape[0])
        else:
            # Normalize lens to a Python int (bag length) per sample
            if torch.is_tensor(lens):
                if lens.ndim == 0:
                    lens_list.append(int(lens.item()))
                elif lens.ndim == 1:
                    # Some datasets might pass the per-instance mask; use length
                    lens_list.append(int(lens.numel()))
                else:
                    lens_list.append(int(x.shape[0]))
            else:
                lens_list.append(int(lens))

    lens_tensor = torch.as_tensor(lens_list, dtype=torch.long)
    max_n = int(lens_tensor.max().item())

    padded = []
    for x in xs:
        n, d = x.shape
        if n < max_n:
            x = F.pad(x, (0, 0, 0, max_n - n))
        padded.append(x[:max_n])

    batch_x = torch.stack(padded, dim=0)  # (B, N, D)
    batch_y = default_collate(ys)
    return (batch_x, lens_tensor), batch_y


def _extract_x(sample: Any) -> torch.Tensor:
    """Robustly extract the feature tensor from a collated sample."""
    x = sample
    # DataLoader returns ((x, lens), y) for mil_collate, or (x, y) for slide_collate
    if isinstance(sample, (list, tuple)):
        x = sample[0]
    elif isinstance(sample, dict):
        x = sample.get("x") or sample.get("bags") or sample.get("feats") or sample
    if isinstance(x, (list, tuple)):
        x = x[0]  # pull out features from (x, lens)
    if not torch.is_tensor(x):
        raise TypeError(f"Expected feature tensor, got: {type(x)}")
    return x


def _class_indices_from_onehot(y: torch.Tensor) -> torch.Tensor:
    """Convert one-hot or dense prob target to class indices for CE."""
    if y.ndim > 1 and y.shape[-1] > 1:
        return y.argmax(dim=-1)
    return y.long().view(-1)


def _retrieve_custom_loss(loss_name: Optional[str], default_loss_cls: type) -> type | nn.Module:
    """Return a PathBench loss class/module by name, else default."""
    if loss_name:
        if hasattr(pb_losses, loss_name):
            return getattr(pb_losses, loss_name)
        raise ValueError(f"Unknown custom loss: {loss_name}")
    return default_loss_cls


def _retrieve_optimizer(optimizer_name: Optional[str]) -> type:
    """Map optimizer name to torch.optim class."""
    if optimizer_name is None:
        return torch.optim.Adam
    if hasattr(torch.optim, optimizer_name):
        return getattr(torch.optim, optimizer_name)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def _wrap_forward_to_save_attention(model: nn.Module) -> None:
    _orig_fwd = model.forward
    sig = inspect.signature(_orig_fwd)
    params = sig.parameters

    supports_attention = ('return_attention' in params)
    lens_aliases = ('lens', 'lengths', 'bag_lengths', 'bags_lens', 'mask', 'attention_mask')
    accepted_lens_kw = None
    for name in lens_aliases:
        if name in params:
            accepted_lens_kw = name
            break

    def _fwd_save_att(bags, *args, **kwargs):
        if len(args) > 0 and accepted_lens_kw is not None:
            kwargs.setdefault(accepted_lens_kw, args[0])
            args = args[1:]
        if 'lens' in kwargs and accepted_lens_kw is not None and accepted_lens_kw != 'lens':
            kwargs.setdefault(accepted_lens_kw, kwargs.pop('lens'))
        if accepted_lens_kw is None and 'lens' in kwargs:
            kwargs.pop('lens')

        if supports_attention and 'return_attention' not in kwargs:
            kwargs['return_attention'] = True

        out = _orig_fwd(bags, *args, **kwargs)

        if isinstance(out, tuple) and len(out) == 2:
            preds, att = out
        else:
            preds = out
            if bags.ndim >= 2:
                B = bags.shape[0]
                N = bags.shape[1] if bags.ndim >= 3 else 1
            else:
                B, N = 1, 1
            att = torch.ones(B, N, device=bags.device) / max(N, 1)

        model._last_attention = att
        return preds

    model.forward = _fwd_save_att

    # --- preserve the original signature for inspect.signature(...) ---
    try:
        _fwd_save_att.__signature__ = sig
    except Exception:
        pass
    # --- expose capability flags used elsewhere if needed  ---
    model._accepted_lens_kw = accepted_lens_kw
    model._supports_attention = supports_attention
    return model

def _wrap_collate_for_fp16(base_collate):
    def _collate_fp16(samples):
        bx, by = base_collate(samples)
        # slide_collate returns (x, y); mil_collate returns ((x, lens), y)
        if isinstance(bx, (list, tuple)):
            x, lens = bx
            x = x.to(torch.float16, copy=False)
            return (x, lens), by
        else:
            x = bx
            x = x.to(torch.float16, copy=False)
            return x, by
    return _collate_fp16

# ---------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------

class MILLightningModule(pl.LightningModule):
    """
    LightningModule replicating your FastAI training semantics:
    - Handles (x) or (x, lens)
    - Converts one-hot → indices for CE
    - Logs AUC/ACC (classification), MAE (regression), C-index (survival)
    - Supports attention-aware custom losses
    - Optional schedulers via pb_config.experiment.schedulers
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        *,
        task: str = "classification",
        optimizer_name: str = "Adam",
        lr: float = 1e-3,
        wd: float = 0.0,
        n_classes: Optional[int] = None,
        pb_config: Optional[Dict[str, Any]] = None,
        estimated_total_steps: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.task = task
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.wd = wd
        self.n_classes = n_classes
        self.pb_config = pb_config or {}
        self.estimated_total_steps = estimated_total_steps

        # Buffers for epoch-level metrics
        self._val_logits: List[torch.Tensor] = []
        self._val_targets: List[torch.Tensor] = []
        self._val_surv_durations: List[torch.Tensor] = []
        self._val_surv_events: List[torch.Tensor] = []

        self.save_hyperparameters(
            {
                "task": task,
                "optimizer_name": optimizer_name,
                "lr": lr,
                "wd": wd,
                "n_classes": n_classes,
            }
        )

    # ------------- batch parsing -------------

    def _extract_x_lens(self, x: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(x, dict):
            lens = x.get("lens") or x.get("lengths") or x.get("bag_lens")
            feats = x.get("x") or x.get("bags") or x.get("feats") or x.get("features") or x
            feats = feats if torch.is_tensor(feats) else feats[0] if isinstance(feats, (list, tuple)) else feats
            return feats, lens
        if isinstance(x, (list, tuple)):
            feats, lens = x[0], (x[1] if len(x) > 1 else None)
            return feats, lens
        return x, None

    def _prepare_xy(self, batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Return (features, lens, targets_for_loss).
        - For classification: ensure class indices (not one-hot strings)
        - For regression: pass float targets
        - For survival: pass (T, E) float/int tensors as provided by dataset
        """
        if isinstance(batch, dict):
            x = batch.get("x") or batch.get("bags") or batch
            y = batch.get("y")
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                x, y = batch
            elif len(batch) == 3:
                # Allow ((x, lens), y)
                x, lens, y = batch
                x = (x, lens)
            else:
                x, y = batch[0], batch[-1]
        else:
            x, y = batch, None

        feats, lens = self._extract_x_lens(x)

        if not torch.is_tensor(feats):
            raise TypeError(f"Expected tensor features, got {type(feats)}")

        # Normalize targets for each task
        if y is None:
            y_for_loss = None
        else:
            if self.task == "classification":
                if not torch.is_tensor(y):
                    y = torch.as_tensor(y)
                y_for_loss = _class_indices_from_onehot(y)
            elif self.task == "regression":
                y_for_loss = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
            elif self.task in ("survival", "survival_discrete"):
                y = torch.as_tensor(y)
                # Expect last dim at least 2: [time, event] (+ bins if discrete handled by dataset)
                y_for_loss = y
            else:
                y_for_loss = torch.as_tensor(y)

        return feats, lens, y_for_loss


    def _unpack_batch(self, batch):
        # supports (x, y) or (x, lens, y)
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                x, lens, y = batch
            elif len(batch) == 2:
                x, y = batch
                lens = None
            else:
                raise RuntimeError(f"Unexpected batch format with length {len(batch)}")
        else:
            raise RuntimeError(f"Unexpected batch type: {type(batch)}")


    def _forward_logits(self, x, lens):
        # Always pass lens keyword; wrapper will drop/rename if unsupported
        return self.model(x, lens=lens) if lens is not None else self.model(x)

    # ------------- steps -------------

    def training_step(self, batch, batch_idx):
        x, lens, y = self._prepare_xy(batch)
        logits = self._forward_logits(x, lens)
        loss = self.criterion(logits, y) if y is not None else logits.mean() * 0
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.task == "classification" and y is not None:
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == y.view_as(preds)).float().mean()
            self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, lens, y = self._prepare_xy(batch)
        logits = self._forward_logits(x, lens)
        loss = self.criterion(logits, y) if y is not None else logits.mean() * 0
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Accumulate for epoch-level metrics
        if y is not None:
            if self.task == "classification":
                self._val_logits.append(logits.detach().float().cpu())
                self._val_targets.append(y.detach().long().view(-1).cpu())
            elif self.task == "regression":
                self._val_logits.append(logits.detach().float().cpu())  # preds
                self._val_targets.append(y.detach().float().view_as(logits).cpu())
            elif self.task == "survival":
                # y: [T, E]
                y = y.float()
                self._val_logits.append(logits.detach().float().cpu().view(-1))  # risk score
                self._val_surv_durations.append(y[:, 0].detach().cpu().view(-1))
                self._val_surv_events.append(y[:, 1].detach().cpu().view(-1))
            elif self.task == "survival_discrete":
                # Approximate risk by sum of hazards (sigmoid over logits)
                probs = torch.sigmoid(logits).sum(dim=-1).detach().float().cpu().view(-1)
                y = y.float()
                self._val_logits.append(probs)
                self._val_surv_durations.append(y[:, 0].detach().cpu().view(-1))
                self._val_surv_events.append(y[:, 1].detach().cpu().view(-1))
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # Compute epoch metrics similar to FastAI defaults
        if self.task == "classification" and self._val_logits:
            logits = torch.cat(self._val_logits)
            targets = torch.cat(self._val_targets)

            # Accuracy
            preds = logits.argmax(dim=-1)
            acc = (preds == targets).float().mean().item()
            self.log("val/acc", acc, prog_bar=True)

            # ROC AUC (binary or multiclass)
            try:
                probs = torch.softmax(logits, dim=-1).numpy()
                y_true = targets.numpy()
                if probs.shape[1] == 2:
                    auc = sk_metrics.roc_auc_score(y_true, probs[:, 1])
                else:
                    auc = sk_metrics.roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
                self.log("val/auc", auc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                self.log("roc_auc_score", auc, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True) 

            except Exception as e:
                # Safeguard when only one class present in val, etc.
                logging.debug(f"Skipping ROC AUC: {e}")

        elif self.task == "regression" and self._val_logits:
            preds = torch.cat(self._val_logits).view(-1)
            targs = torch.cat(self._val_targets).view(-1)
            mae = torch.mean(torch.abs(preds - targs)).item()
            self.log("val/mae", mae, prog_bar=True)

        elif self.task in ("survival", "survival_discrete") and self._val_logits:
            risk = torch.cat(self._val_logits).numpy()
            durations = torch.cat(self._val_surv_durations).numpy()
            events = torch.cat(self._val_surv_events).numpy()
            try:
                c_index = lifelines_cindex(durations, -risk, events)  # lower risk ⇒ longer survival
                self.log("val/c_index", float(c_index), prog_bar=True)
                self.log("concordance_index", float(c_index), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
            except Exception as e:
                logging.debug(f"Skipping C-index: {e}")

        # Clear buffers
        self._val_logits.clear()
        self._val_targets.clear()
        self._val_surv_durations.clear()
        self._val_surv_events.clear()

    # ------------- optimizers & schedulers -------------

    def configure_optimizers(self):
        opt_cls = _retrieve_optimizer(self.optimizer_name)
        optimizer = opt_cls(self.parameters(), lr=self.lr, weight_decay=self.wd)

        exp_cfg = (self.pb_config or {}).get("experiment", {})
        sched_cfgs = exp_cfg.get("schedulers", None)

        if not sched_cfgs and exp_cfg.get("one_cycle", False):
            sched_cfgs = [{"name": "OneCycleLR"}]

        if not sched_cfgs:
            return optimizer

        sched_list = []
        for sc in sched_cfgs:
            if isinstance(sc, str):
                name, params = sc, {}
            else:
                name, params = sc.get("name"), sc.get("params", {})

            if name == "OneCycleLR":
                total_steps = self.estimated_total_steps
                if total_steps is None and self.trainer is not None:
                    total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
                if total_steps is None:
                    # Fallback to epochs * steps_per_epoch
                    max_epochs = int(exp_cfg.get("epochs", 1))
                    steps_per_epoch = getattr(self.trainer, "num_training_batches", None) or 1
                    total_steps = max(1, max_epochs * int(steps_per_epoch))

                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=params.get("max_lr", self.lr),
                    total_steps=int(total_steps),
                    pct_start=params.get("pct_start", 0.3),
                    anneal_strategy=params.get("anneal_strategy", "cos"),
                    div_factor=params.get("div_factor", 25.0),
                    final_div_factor=params.get("final_div_factor", 1e4),
                )
                sched_list.append(
                    {"scheduler": scheduler, "interval": "step"}
                )

            elif name == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=int(params.get("T_max", max(1, int(exp_cfg.get("epochs", 1))))),
                    eta_min=float(params.get("eta_min", 0.0)),
                )
                sched_list.append({"scheduler": scheduler, "interval": "epoch"})

            elif name == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=int(params.get("step_size", 10)),
                    gamma=float(params.get("gamma", 0.1)),
                )
                sched_list.append({"scheduler": scheduler, "interval": "epoch"})

            elif name == "ExponentialLR":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=float(params.get("gamma", 0.95))
                )
                sched_list.append({"scheduler": scheduler, "interval": "epoch"})

            elif name == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=params.get("mode", "min"),
                    factor=float(params.get("factor", 0.1)),
                    patience=int(params.get("patience", 10)),
                )
                sched_list.append({"scheduler": scheduler, "monitor": params.get("monitor", "val/loss")})

            else:
                logging.warning(f"Unknown scheduler '{name}', skipping.")



        if not sched_list:
            return optimizer

        # ✅ Lightning-safe returns:
        if len(sched_list) == 1:
            # single scheduler can use dict-return
            return {"optimizer": optimizer, "lr_scheduler": sched_list[0]}
        else:
            # multiple schedulers -> return as (optimizers, schedulers)
            return [optimizer], sched_list


# ---------------------------------------------------------------------
# Dataloaders & dims discovery (ports your FastAI builder)
# ---------------------------------------------------------------------

def _build_dataloaders(
    config,
    *,
    bags: List[str],
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    unique_categories: np.ndarray,
    device: Optional[Union[str, torch.device]] = None,
    pb_config: Optional[Dict[str, Any]] = None,
) -> Tuple[TorchDataLoader, TorchDataLoader, int, int, Optional[torch.Tensor], Optional[Dict[int, float]], Optional[SkOneHotEncoder]]:
    exp_cfg = (pb_config or {}).get("experiment", {})
    num_workers = exp_cfg.get("num_workers", 0)
    persistent_workers = exp_cfg.get("persistent_workers", True)
    class_weighting = exp_cfg.get("class_weighting", False)
    batch_size = getattr(config, "batch_size", 1)
    drop_last = getattr(config, "drop_last", False)
    slide_level = getattr(config, "slide_level", False)
    task = getattr(config, "task", "classification")
    bag_size = getattr(config, "bag_size", None)
    use_lens = getattr(getattr(config, "model_config", object), "use_lens", False)
    ctx = exp_cfg.get("multiprocessing_context", "spawn") if num_workers > 0 else None
    cast_inputs_to_half = bool(exp_cfg.get("cast_inputs_to_half", False))
    val_bag_size = exp_cfg.get("val_bag_size", bag_size)  # NEW
    pin_memory = bool(exp_cfg.get("pin_memory", False))
    prefetch_factor = int(exp_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None

    encoder: Optional[SkOneHotEncoder] = None
    targets_np = np.array(targets, copy=True)

    # --- target preparation & encoder creation (match your FastAI code) ---
    if slide_level:
        if task == "classification":
            encoder = _make_ohe_dense().fit(unique_categories.reshape(-1, 1))
        elif task in ("survival", "survival_discrete"):
            targets_np[:, 1] = targets_np[:, 1].astype(int)  # events
            if task == "survival":
                targets_np[:, 0] = targets_np[:, 0].astype(float)  # durations
            else:
                targets_np[:, 0] = targets_np[:, 0].astype(int)     # time bins
                encoder = _make_ohe_dense().fit(targets_np[:, 0].reshape(-1, 1))
        else:
            targets_np = targets_np.astype(np.float32)
    else:
        if task == "classification":
            encoder = _make_ohe_dense().fit(unique_categories.reshape(-1, 1))
        if task == "survival_discrete":
            time_bins = targets_np[:, 0].astype(int)
            encoder = _make_ohe_dense().fit(time_bins.reshape(-1, 1))
            targets_np[:, 0] = time_bins
            targets_np[:, 1] = targets_np[:, 1].astype(int)
        if task in ("survival", "regression"):
            targets_np = targets_np.astype(np.float32)

    slide_collate_fn = slide_collate
    mil_collate_fn = mil_collate
    if cast_inputs_to_half:
        slide_collate_fn = _wrap_collate_for_fp16(slide_collate)
        mil_collate_fn = _wrap_collate_for_fp16(mil_collate)

    # --- datasets & dataloaders ---
    if slide_level:
        train_dataset = data_utils.build_slide_dataset(
            [bags[i] for i in train_idx],
            targets_np[train_idx],
            survival_discrete=(task == "survival_discrete"),
            encoder=encoder,
            bag_size=None,
        )
        val_dataset = data_utils.build_slide_dataset(
            [bags[i] for i in val_idx],
            targets_np[val_idx],
            survival_discrete=(task == "survival_discrete"),
            encoder=encoder,
            bag_size=1,
        )
        train_dl = TorchDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=(persistent_workers and num_workers>0),
            multiprocessing_context=(ctx if num_workers>0 else None),
            collate_fn=slide_collate_fn,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )
        val_dl = TorchDataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=(persistent_workers and num_workers>0),
            multiprocessing_context=(ctx if num_workers>0 else None),
            collate_fn=slide_collate_fn,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )
    else:
        train_dataset = data_utils.build_dataset(
            [bags[i] for i in train_idx],
            targets_np[train_idx],
            encoder=encoder,
            bag_size=bag_size,
            use_lens=use_lens,
            survival_discrete=(task == "survival_discrete"),
        )
        val_dataset = data_utils.build_dataset(
            [bags[i] for i in val_idx],
            targets_np[val_idx],
            encoder=encoder,
            bag_size=1,
            use_lens=use_lens,
            survival_discrete=(task == "survival_discrete"),
        )
        train_dl = TorchDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=drop_last,
            persistent_workers=(persistent_workers and num_workers>0),
            multiprocessing_context=(ctx if num_workers>0 else None),
            collate_fn=mil_collate_fn,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )
        val_dl = TorchDataLoader(
            val_dataset,
            batch_size=1 if task == "classification" else batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=(persistent_workers and num_workers>0),
            multiprocessing_context=(ctx if num_workers>0 else None),
            collate_fn=mil_collate_fn,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )

    # --- infer dimensions from one batch ---
    sample = next(iter(train_dl))
    sample_x = _extract_x(sample)
    n_in = int(sample_x.shape[-1])
    del sample, sample_x
    gc.collect()

    if slide_level:
        if task == "classification":
            n_out = len(unique_categories)
        elif task in ("regression", "survival"):
            n_out = 1
        else:
            assert encoder is not None
            n_out = int(encoder.categories_[0].size)
    else:
        if task in ("regression", "survival"):
            n_out = 1
        elif task == "classification":
            n_out = len(unique_categories)
        else:
            assert encoder is not None
            n_out = int(encoder.categories_[0].size)

    # --- class/event weights like in FastAI block ---
    class_weight_tensor: Optional[torch.Tensor] = None
    survival_weight_dict: Optional[Dict[str, float]] = None

    if task == "classification" and class_weighting:
        train_labels = pd.Series(targets[train_idx])
        counts = train_labels.value_counts()
        w = counts.sum() / counts
        w = (w / w.sum()).to_dict()
        cats = unique_categories.tolist()
        class_weight_tensor = torch.tensor([float(w.get(c, 1.0)) for c in cats], dtype=torch.float32)

    if task == "survival" and class_weighting:
        t = targets[train_idx]
        events = torch.tensor(t[:, 1].astype(int))
        num_events = int(events.sum())
        num_total = int(events.numel())
        num_cens = num_total - num_events
        if num_total > 0:
            survival_weight_dict = {"event_weight": num_cens / num_total, "censored_weight": num_events / num_total}

    return train_dl, val_dl, n_in, n_out, class_weight_tensor, survival_weight_dict, encoder


# ---------------------------------------------------------------------
# Public training API (backwards compatible with positional args)
# ---------------------------------------------------------------------

def _train_lightning_impl(
    config,
    *,
    bags: List[str],
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    unique_categories: np.ndarray,
    outdir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,  # kept for API parity
    pb_config: Optional[Dict[str, Any]] = None,
    loss: Optional[str] = None,
    optimizer: Optional[str] = None,
    **kwargs,
):
    # ---- dataloaders, dims, weights
    (train_dl, val_dl, n_in, n_out,
     class_weight_tensor, survival_weight_dict, encoder) = _build_dataloaders(
        config,
        bags=bags, targets=targets, train_idx=train_idx, val_idx=val_idx,
        unique_categories=unique_categories, device=device, pb_config=pb_config,
    )

    # ---- model
    z_dim = getattr(config, "z_dim", 256)
    enc_layers = getattr(config, "encoder_layers", 1)
    dropout_p = getattr(config, "dropout_p", 0.0)
    act_fn = getattr(config, "activation_function", "ReLU")
    task = getattr(config, "task", "classification")
    log.info(f"Model dims: in={n_in}, out={n_out}, z_dim={z_dim}")

    model = config.build_model(
        n_in, n_out,
        z_dim=z_dim,
        encoder_layers=enc_layers,
        dropout_p=dropout_p,
        activation_function=act_fn,
        goal=task,
    )
    _wrap_forward_to_save_attention(model)

    # ---- loss
    if task == "classification":
        default_loss_cls = nn.CrossEntropyLoss
    elif task == "regression":
        default_loss_cls = nn.MSELoss
    elif task == "survival":
        default_loss_cls = getattr(pb_losses, "CoxPHLoss")
    elif task == "survival_discrete":
        default_loss_cls = getattr(pb_losses, "NLLLogisticHazardLoss")
    else:
        raise ValueError(f"Unsupported task: {task}")

    loss_cls = _retrieve_custom_loss(loss, default_loss_cls)
    loss_kwargs: Dict[str, Any] = {}
    if task == "classification" and class_weight_tensor is not None:
        loss_kwargs["weight"] = class_weight_tensor.to(torch.float32)
    if task == "survival" and survival_weight_dict is not None:
        loss_kwargs.update(survival_weight_dict)

    base_loss = loss_cls(**loss_kwargs) if inspect.isclass(loss_cls) else loss_cls
    criterion = _AttentionLossWrapper(base_loss, model)

    # ---- trainer settings (mirrors FastAI overrides)
    exp_cfg = (pb_config or {}).get("experiment", {})
    lr = float(exp_cfg.get("lr", getattr(config, "lr", 1e-3) or 1e-3))
    wd = float(exp_cfg.get("wd", getattr(config, "wd", 0.0) or 0.0))
    opt_name = optimizer or exp_cfg.get("optimizer", "Adam")
    max_epochs = int(exp_cfg.get("epochs", getattr(config, "epochs", 1) or 1))
    best_monitor = exp_cfg.get("best_epoch_based_on", "val/loss")

    # estimate steps (for OneCycle)
    estimated_total_steps = max_epochs * max(1, len(train_dl))

    module = MILLightningModule(
        model=model,
        criterion=criterion,
        task=task,
        optimizer_name=opt_name,
        lr=lr,
        wd=wd,
        n_classes=(n_out if task == "classification" else None),
        pb_config=pb_config,
        estimated_total_steps=estimated_total_steps,
    )

    # callbacks & loggers like SaveModelCallback + CSVLogger
    checkpoint_cb = ModelCheckpoint(
        dirpath=outdir, filename="best-epoch", save_top_k=1,
        monitor=best_monitor, mode="min" if "loss" in best_monitor.lower() else "max"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = CSVLogger(save_dir=outdir if outdir else ".", name="logs")

    callbacks = [checkpoint_cb, lr_monitor]
    cb_cfg = (pb_config or {}).get("experiment", {}).get("callbacks", {})  # NEW
    es_cfg = cb_cfg.get("early_stopping")  # e.g., {"monitor":"val/loss","mode":"min","patience":10}
    if es_cfg:
        callbacks.append(EarlyStopping(
            monitor=es_cfg.get("monitor", best_monitor),
            mode=es_cfg.get("mode", "min" if "loss" in best_monitor.lower() else "max"),
            patience=int(es_cfg.get("patience", 10)),
            min_delta=float(es_cfg.get("min_delta", 0.0)),
        ))

    log_every = max(1, min(10, len(train_dl)))
    trainer = pl.Trainer(
        default_root_dir=outdir,
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=log_every,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
    )

    trainer.fit(module, train_dl, val_dl)
    return {
        "trainer": trainer,
        "module": module,
        "model": model,
        "n_in": n_in,
        "n_out": n_out,
        "encoder": encoder,
        "best_ckpt": checkpoint_cb.best_model_path,
    }


class _AttentionLossWrapper(nn.Module):
    """
    Wrap a base loss so it can optionally receive attention weights
    saved by the model wrapper when `base_loss.require_attention` is True.
    """
    def __init__(self, base_loss: nn.Module, model: nn.Module):
        super().__init__()
        self.base_loss = base_loss
        self.model = model

    def forward(self, preds: torch.Tensor, targets: Optional[torch.Tensor]):
        if targets is None:
            return preds.mean() * 0  # valid 'no-op' loss
        if getattr(self.base_loss, "require_attention", False):
            return self.base_loss(preds, targets, attention_weights=getattr(self.model, "_last_attention", None))
        return self.base_loss(preds, targets)


def train_lightning(config, *args, **kwargs):
    """
    Backwards-compatible wrapper.

    Supports BOTH:
        train_lightning(config, bags, targets, train_idx, val_idx, unique_categories, **kw)
    and:
        train_lightning(config, *,
            bags=..., targets=..., train_idx=..., val_idx=..., unique_categories=..., **kw)
    """
    if len(args) >= 5:
        bags, targets, train_idx, val_idx, unique_categories, *rest = args
        if rest:
            # Accept and ignore extra historical positional args
            pass
        return _train_lightning_impl(
            config,
            bags=bags,
            targets=targets,
            train_idx=train_idx,
            val_idx=val_idx,
            unique_categories=unique_categories,
            **kwargs,
        )
    # keyword-style
    return _train_lightning_impl(config, **kwargs)
