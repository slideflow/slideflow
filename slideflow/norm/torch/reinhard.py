"""
Reinhard normalization based on method of:

E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.
"""

from __future__ import division

from typing import Tuple

import torch

from slideflow.norm.torch.reinhard_fast import fit as fit_fast
from slideflow.norm.torch.reinhard_fast import transform as transform_fast
from slideflow.norm.torch.reinhard_fast import ReinhardFastNormalizer


def standardize_brightness(I: torch.Tensor) -> torch.Tensor:
    """

    :param I:
    :return:
    """
    p = torch.quantile(I.float(), 0.9)  # p = np.percentile(I, 90)
    return torch.clip(I * 255.0 / p, 0, 255).to(torch.uint8)


def transform(
    I: torch.Tensor,
    tgt_mean: torch.Tensor,
    tgt_std: torch.Tensor
) -> torch.Tensor:
    I = standardize_brightness(I)
    return transform_fast(I, tgt_mean, tgt_std)


def fit(
    target: torch.Tensor,
    reduce: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    target = standardize_brightness(target)
    return fit_fast(target, reduce=reduce)


class ReinhardNormalizer(ReinhardFastNormalizer):
    """
    A stain normalization object
    """

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device('cuda')

    def fit(
        self,
        target: torch.Tensor,
        reduce: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(target.shape) == 3:
            target = torch.unsqueeze(target, dim=0)
        means, stds = fit(target, reduce=reduce)
        self.target_means = means
        self.target_stds = stds
        return means, stds

    def transform(self, I: torch.Tensor) -> torch.Tensor:
        if len(I.shape) == 3:
            return transform(torch.unsqueeze(I, dim=0), self.target_means, self.target_stds)[0]
        else:
            return transform(I, self.target_means, self.target_stds)