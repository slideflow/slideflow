"""Modification of https://github.com/mahmoodlab/CLAM"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List

from ._utils import initialize_weights

# -----------------------------------------------------------------------------

class MIL_fc(nn.Module):

    sizes = {
        "small": [1024, 512]
    }

    def __init__(
        self,
        size: Union[str, List[int]] = "small",
        dropout: bool = False,
        n_classes: int = 2,
        top_k: int = 1,
        gate: bool = True,
    ):
        super().__init__()
        assert n_classes == 2

        self.size = self.sizes[size] if isinstance(size, str) else size
        fc = [nn.Linear(self.size[0], self.size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        fc.append(nn.Linear(self.size[1], n_classes))
        self.classifier = nn.Sequential(*fc)
        initialize_weights(self)
        self.top_k=top_k

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(device)

    def forward(self, h):
        if isinstance(h, tuple) and len(h) == 2:
            h, label = h
        elif isinstance(h, tuple) and len(h) == 3:
            h, label, instance_eval = h
        if h.ndim == 3:
            h = h.squeeze()

        logits  = self.classifier(h) # K x 1
        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        results_dict = {}
        return top_instance, results_dict


class MIL_fc_mc(nn.Module):

    sizes = {
        "small": [1024, 512]
    }

    def __init__(
        self,
        size: Union[str, List[int]] = "small",
        dropout: bool = False,
        n_classes: int = 2,
        top_k: int = 1,
        gate: bool = True,
    ):
        super().__init__()
        if not n_classes > 2:
            raise ValueError(
                "The 'MIL_fc_mc' model is a multi-categorical model that "
                "requires more than two outcome categories. For binary outcomes "
                "with only 2 categories, use 'MIL_fc'."
            )

        self.size = self.sizes[size] if isinstance(size, str) else size

        fc = [nn.Linear(self.size[0], self.size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        self.fc = nn.Sequential(*fc)

        self.classifiers = nn.ModuleList([nn.Linear(self.size[1], 1) for i in range(n_classes)])
        initialize_weights(self)
        self.top_k=top_k
        self.n_classes = n_classes
        assert self.top_k == 1

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc = self.fc.to(device)
        self.classifiers = self.classifiers.to(device)

    def forward(self, h):
        if isinstance(h, tuple) and len(h) == 2:
            h, label = h
        elif isinstance(h, tuple) and len(h) == 3:
            h, label, instance_eval = h
        if h.ndim == 3:
            h = h.squeeze()

        device = h.device
        h = self.fc(h)
        logits = torch.empty(h.size(0), self.n_classes).float().to(device)

        for c in range(self.n_classes):
            if isinstance(self.classifiers, nn.DataParallel):
                logits[:, c] = self.classifiers.module[c](h).squeeze(1)
            else:
                logits[:, c] = self.classifiers[c](h).squeeze(1)

        y_probs = F.softmax(logits, dim = 1)
        m = y_probs.view(1, -1).argmax(1)
        top_indices = torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1).view(-1, 1)
        top_instance = logits[top_indices[0]]
        results_dict = {}
        return top_instance, results_dict



