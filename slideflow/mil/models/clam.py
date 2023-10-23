"""Modification of https://github.com/mahmoodlab/CLAM"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Optional, Callable

from slideflow.model.torch_utils import get_device
from ._utils import initialize_weights

# -----------------------------------------------------------------------------

class Attn_Net(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        self.module = [nn.Linear(L, D), nn.Tanh()]
        if dropout:
            self.module.append(nn.Dropout(0.25))
        self.module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

# -----------------------------------------------------------------------------


class _CLAM_Base(nn.Module):
    """
    args:
        size: config for network size
        dropout: whether to use dropout
        k_sample: number of positive/neg patches to sample for instance-level training
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
        instance_loss_fn: loss function to supervise instance-level training
        subtyping: whether it's a subtyping problem
        gate: whether to use gated attention network
    """

    sizes = {
        "small":      [1024, 512, 256],
        "big":        [1024, 512, 384],
        "multiscale": [2048, 512, 256]
    }

    def __init__(
        self,
        size: Union[str, List[int]] = "small",
        dropout: bool = False,
        k_sample: int = 8,
        n_classes: int = 2,
        instance_loss_fn: Optional[Callable] = None,
        subtyping: bool = False,
        gate: bool = True,
        multi_head_attention: bool = False
    ) -> None:
        super().__init__()

        if instance_loss_fn is None:
            instance_loss_fn = nn.CrossEntropyLoss()
        self.size = self.sizes[size] if isinstance(size, str) else size

        # Encoder
        fc = [nn.Linear(self.size[0], self.size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        # Attention net
        att_fn = Attn_Net_Gated if gate else Attn_Net
        n_att = 1 if not multi_head_attention else n_classes
        fc.append(
            att_fn(L=self.size[1], D=self.size[2], dropout=dropout, n_classes=n_att))
        self.attention_net = nn.Sequential(*fc)

        # Classifier head
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(self.size[1], 2) for _ in range(n_classes)]
        )
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    def relocate(self):
        device = get_device()
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()

    def _inst_eval(self, A, h, classifier, index=None):
        """Instance-level evaluation for in-the-class attention branch."""
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        try:
            top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        except RuntimeError as e:
            raise RuntimeError(
                f"Error selecting top_k from sample with shape {A.shape}. "
                f"Verify that all slides have at least {self.k_sample} tiles "
                f"(min_tiles={self.k_sample}). Error: {e}"
            )
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def _inst_eval_out(self, A, h, classifier, index=None):
        """Instance-level evaluation for out-of-the-class attention branch."""
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        try:
            top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        except RuntimeError as e:
            raise RuntimeError(
                f"Error selecting top_k from sample with shape {A.shape}. "
                f"Verify that all slides have at least {self.k_sample} tiles "
                f"(min_tiles={self.k_sample}). Error: {e}"
            )
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def _logits_from_m(self, M):
        """Calculate logits from M.

        Args:
            M: attention-weighted features

        Returns:
            logits: logits

        """
        return self.classifiers(M)

    def _process_inputs(self, h, label=None, instance_eval=False):
        """Process inputs to forward pass.

        Args:
            h: input features
            label: ground truth label
            instance_eval: whether to perform instance-level evaluation

        Returns:
            h: input features
            label: ground truth label
            instance_eval: whether to perform instance-level evaluation

        """
        if isinstance(h, (list, tuple)) and len(h) == 2:
            h, label = h
        elif isinstance(h, (list, tuple)) and len(h) == 3:
            h, label, instance_eval = h
        if h.ndim == 3:
            h = h.squeeze()
        if h.shape[1] != self.size[0]:
            raise RuntimeError(
                f"Input feature size ({h.shape[1]}) does not match size of "
                f"model first linear layer ({self.size[0]}). "
            )
        return h, label, instance_eval

    def instance_loss(self, A, h, label):
        """Calculate instance loss.

        Args:
            A: attention weights
            h: input features
            label: ground truth label

        Returns:
            total_inst_loss: total instance loss
            all_preds: all predictions
            all_targets: all targets

        """
        total_inst_loss = 0.0
        all_preds = []
        all_targets = []
        if label.ndim < 2 or label.shape[1] != self.n_classes:
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
        else:
            inst_labels = label[0]
        for i in range(len(self.instance_classifiers)):
            inst_label = inst_labels[i].item()
            classifier = self.instance_classifiers[i]
            if inst_label == 1: #in-the-class:
                instance_loss, preds, targets = self._inst_eval(A, h, classifier, i)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            else: #out-of-the-class
                if self.subtyping:
                    instance_loss, preds, targets = self._inst_eval_out(A, h, classifier, i)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    continue
            total_inst_loss += instance_loss

        if self.subtyping:
            total_inst_loss /= len(self.instance_classifiers)

        return total_inst_loss, np.array(all_preds), np.array(all_targets)

    def forward(
        self,
        h,
        label=None,
        instance_eval=False,
        return_attention=False,
    ):
        """Forward pass.

        Args:
            h: input features
            label: ground truth label
            instance_eval: whether to perform instance-level evaluation
            return_attention: whether to return attention weights

        Returns:
            logits: logits
            inst_loss_dict: instance loss dictionary
            A_raw: attention weights

        """
        # Process inputs
        h, label, instance_eval = self._process_inputs(h, label, instance_eval)

        # Attention net
        A, h = self.attention_net(h)  # NxK
        A = A_raw = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        # Instance-level evaluation
        if instance_eval:
            inst_loss, inst_targets, inst_preds = self.instance_loss(A, h, label)
            inst_loss_dict = {
                'instance_loss': inst_loss,
                'inst_labels': inst_targets,
                'inst_preds': inst_preds
            }
        else:
            inst_loss_dict = {}

        # Calculate logits
        M = torch.mm(A, h)
        logits = self._logits_from_m(M)

        if return_attention:
            return logits, A_raw, inst_loss_dict
        else:
            return logits, inst_loss_dict

    def calculate_attention(self, h):
        """Calculate attention weights.

        Args:
            h: input features
            label: ground truth label
            instance_eval: whether to perform instance-level evaluation

        Returns:
            A: attention weights

        """
        h, *_ = self._process_inputs(h, None, None)
        A, h = self.attention_net(h)  # NxK
        return torch.transpose(A, 1, 0)  # KxN

    def get_last_layer_activations(self, h):
        """Get activations from last layer.

        Args:
            h: input features

        Returns:
            activations: activations from last layer

        """
        h, *_ = self._process_inputs(h, None, None)
        A, h = self.attention_net(h)  # NxK
        A = F.softmax(A, dim=1)  # softmax over N
        A = torch.transpose(A, 1, 0)  # KxN
        M = torch.mm(A, h)
        return M, A


class CLAM_SB(_CLAM_Base):
    """
    args:
        size: config for network size
        dropout: whether to use dropout
        k_sample: number of positive/neg patches to sample for instance-level training
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
        instance_loss_fn: loss function to supervise instance-level training
        subtyping: whether it's a subtyping problem
        gate: whether to use gated attention network
    """

    sizes = {
        "small":          [1024, 512, 256],
        "big":            [1024, 512, 384],
        "multiscale":     [2048, 512, 256],
        "xception":       [2048, 256, 128],
        "xception_multi": [1880, 128, 64],
        "xception_3800":  [3800, 512, 256]
    }

    def __init__(
        self,
        size: Union[str, List[int]] = "small",
        dropout: bool = False,
        k_sample: int = 8,
        n_classes: int = 2,
        instance_loss_fn: Optional[Callable] = None,
        subtyping: bool = False,
        gate: bool = True,
    ) -> None:
        super().__init__(
            size=size,
            dropout=dropout,
            k_sample=k_sample,
            n_classes=n_classes,
            instance_loss_fn=instance_loss_fn,
            subtyping=subtyping,
            gate=gate,
            multi_head_attention=False
        )
        # Classifier head.
        self.classifiers = nn.Linear(self.size[1], n_classes)

        # Initialize weights.
        initialize_weights(self)


# -----------------------------------------------------------------------------

class CLAM_MB(_CLAM_Base):
    """
    args:
        size: config for network size
        dropout: whether to use dropout
        k_sample: number of positive/neg patches to sample for instance-level training
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
        instance_loss_fn: loss function to supervise instance-level training
        subtyping: whether it's a subtyping problem
        gate: whether to use gated attention network
    """

    sizes = {
        "small":      [1024, 512, 256],
        "big":        [1024, 512, 384],
        "multiscale": [2048, 512, 256]
    }

    def __init__(
        self,
        size: Union[str, List[int]] = "small",
        dropout: bool = False,
        k_sample: int = 8,
        n_classes: int = 2,
        instance_loss_fn: Optional[Callable] = None,
        subtyping: bool = False,
        gate: bool = True,
    ) -> None:
        super().__init__(
            size=size,
            dropout=dropout,
            k_sample=k_sample,
            n_classes=n_classes,
            instance_loss_fn=instance_loss_fn,
            subtyping=subtyping,
            gate=gate,
            multi_head_attention=True
        )
        # Classifier head
        # Use an independent linear layer to predict each class
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.size[1], 1) for _ in range(n_classes)]
        )
        # Initialize weights.
        initialize_weights(self)

    def _inst_eval(self, A, h, classifier, index):
        return super()._inst_eval(A[index], h, classifier)

    def _inst_eval_out(self, A, h, classifier, index):
        return super()._inst_eval(A[index], h, classifier)

    def _logits_from_m(self, M):
        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        return logits
