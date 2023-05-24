"""Utility functions for calculating loss from CLAM module output.

Modification of https://github.com/mahmoodlab/CLAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.losses import CrossEntropyLossFlat
from fastai.metrics import AccumMetric, ActivationType, skm
import numpy as np
from sksurv.metrics import concordance_index_censored
# -----------------------------------------------------------------------------

class AccumMetricCLAM(AccumMetric):

    def accumulate(self, learn):
        "Store targs and preds from `learn`, using activation function and argmax as appropriate"
        pred, _ = learn.pred
        if self.activation in [ActivationType.Softmax, ActivationType.BinarySoftmax]:
            pred = F.softmax(pred, dim=self.dim_argmax)
            if self.activation == ActivationType.BinarySoftmax: pred = pred[:, -1]
        elif self.activation == ActivationType.Sigmoid: pred = torch.sigmoid(pred)
        elif self.dim_argmax: pred = pred.argmax(dim=self.dim_argmax)
        if self.thresh:  pred = (pred >= self.thresh)
        self.accum_values(pred,learn.y,learn)

def skm_to_fastai_clam(func, is_class=True, thresh=None, axis=-1, activation=None, **kwargs):
    "Convert `func` from sklearn.metrics to a fastai metric"
    dim_argmax = axis if is_class and thresh is None else None
    if activation is None:
        activation = ActivationType.Sigmoid if (is_class and thresh is not None) else ActivationType.No
    return AccumMetricCLAM(func, dim_argmax=dim_argmax, activation=activation, thresh=thresh,
                       to_np=True, invert_arg=True, **kwargs)

def RocAuc(axis=-1, average='macro', sample_weight=None, max_fpr=None, multi_class='ovr'):
    "Area Under the Receiver Operating Characteristic Curve for single-label multiclass classification problems"
    assert multi_class in ['ovr', 'ovo']
    return skm_to_fastai_clam(skm.roc_auc_score, axis=axis, activation=ActivationType.Softmax, flatten=False,
                         average=average, sample_weight=sample_weight, max_fpr=max_fpr, multi_class=multi_class)

# -----------------------------------------------------------------------------

class CrossEntropyLoss(CrossEntropyLossFlat):#nn.CrossEntropyLoss):
    """total_loss = bag_weight * loss + (1-bag_weight) * instance_loss"""

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, flatten=None, **kwargs)

    def __call__(self, output, target):
        logits, inst_loss_dict = output
        return super().__call__(logits, target)

    def activation(self, output):
        logits, inst_loss_dict = output
        return F.softmax(logits, dim=-1)

    def decodes(self, output):
        logits, inst_loss_dict = output
        return logits.argmax(dim=-1)


class CrossEntropyWithInstanceLoss(CrossEntropyLossFlat):#nn.CrossEntropyLoss):
    """bag_weight * cross_entropy_loss + (1-bag_weight) * instance_loss"""

    def __init__(self, *args, bag_weight=0.7, **kwargs):
        self.bag_weight = bag_weight
        return super().__init__(*args, flatten=None, **kwargs)

    def __call__(self, output, target):
        logits, inst_loss_dict = output
        instance_loss = inst_loss_dict['instance_loss']
        ce_loss = super().__call__(logits, target)
        return self.bag_weight * ce_loss + (1-self.bag_weight) * instance_loss

    def activation(self, output):
        logits, inst_loss_dict = output
        return F.softmax(logits, dim=-1)

    def decodes(self, output):
        logits, inst_loss_dict = output
        return logits.argmax(dim=-1)

class ConcordanceIndexCensored(AccumMetric):
    '''Fastai compatible metric for sksurv.metrics.concordance_index_censored'''
    def __init__(self, name='cindex', **kwargs):
        super().__init__(func=self._cindex, name=name, flatten=None, **kwargs)
        self.event = []
        self.targs = []
        self.preds = []

    def reset(self):
        super().reset()
        self.event = []
        self.targs = []
        self.preds = []

    def accumulate(self, learn):
        h = learn.pred[2]
        if h.shape[1] > 1:
            hazards = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            preds = -torch.sum(survival, dim=1)
        else:
            preds = h.squeeze()
        self.accum_values(preds, learn.y)    

    def accum_values(self, preds, targs):
        y_true, event = targs
        self.event.append(event.detach().cpu().numpy().squeeze())
        self.targs.append(y_true.argmax(dim=1).detach().cpu().numpy().squeeze())
        self.preds.append(preds.detach().cpu().numpy())

    @property
    def value(self):
        return self._cindex()

    def _cindex(self):
        event = np.asarray(self.event).astype(bool)
        targs = np.asarray(self.targs)
        preds = np.asarray(self.preds)
        return concordance_index_censored(event, targs, preds)[0]


class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Reduction of loss over batches. Must be one of ['mean', 'sum']
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, output, target):
        hazards, S, Y_hat, _, _ = output
        y_true, event = target 
        censor = 1 - event

        return self.nll_loss(
            h=hazards, 
            y=y_true.argmax(dim=1).unsqueeze(dim=1),
            c=censor.unsqueeze(dim=1),
        )

    def nll_loss(self, h, y, c):
        """
        The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
        Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y: (n_batches, 1)
            The true time bin index label.
        c: (n_batches, 1)
            The censoring status indicator.
        alpha: float
        eps: float
            Numerical constant; lower bound to avoid taking logs of tiny numbers.
        reduction: str
            Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
        References
        ----------
        Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
        """

        y = y.type(torch.int64)
        c = c.type(torch.int64)

        hazards = torch.sigmoid(h)
        S = torch.cumprod(1 - hazards, dim=1)

        S_padded = torch.cat([torch.ones_like(c), S], 1)
        s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=self.eps)
        h_this = torch.gather(hazards, dim=1, index=y).clamp(min=self.eps)
        s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=self.eps)

        uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
        censored_loss = - c * torch.log(s_this)

        neg_l = censored_loss + uncensored_loss
        if self.alpha is not None:
            loss = (1 - self.alpha) * neg_l + self.alpha * uncensored_loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            raise ValueError(f"Bad input for reduction: {self.reduction}")

        return loss