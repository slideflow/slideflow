"""Utility functions for calculating loss from CLAM module output.

Modification of https://github.com/mahmoodlab/CLAM
"""

import torch
import torch.nn.functional as F
from fastai.losses import CrossEntropyLossFlat
from fastai.metrics import AccumMetric, ActivationType, skm

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