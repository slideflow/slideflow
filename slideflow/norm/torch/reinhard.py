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
    """Standardize image brightness to 90th percentile.

    Args:
        I (torch.Tensor): Image to standardize.

    Returns:
        torch.Tensor: Brightness-standardized image (uint8)
    """
    p = torch.quantile(I.float(), 0.9)
    return torch.clip(I * 255.0 / p, 0, 255).to(torch.uint8)


def transform(
    I: torch.Tensor,
    tgt_mean: torch.Tensor,
    tgt_std: torch.Tensor
) -> torch.Tensor:
    """Normalize an H&E image.

    Args:
        img (torch.Tensor): Image, uint8 with dimensions C, W, H.
        tgt_mean (torch.Tensor): Target channel means.
        tgt_std (torch.Tensor): Target channel standard deviations.

    Returns:
        torch.Tensor: Normalized image.
    """
    I = standardize_brightness(I)
    return transform_fast(I, tgt_mean, tgt_std)

class ReinhardNormalizer(ReinhardFastNormalizer):

    def __init__(self) -> None:
        """Reinhard H&E stain normalizer (PyTorch implementation).

        Normalizes an image as defined by:

        Reinhard, Erik, et al. "Color transfer between images." IEEE
        Computer graphics and applications 21.5 (2001): 34-41.

        """
        super().__init__()
        self.device = torch.device('cuda')

    def fit(
        self,
        target: torch.Tensor,
        reduce: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit normalizer to a target image.

        Args:
            img (torch.Tensor): Target image (RGB uint8) with dimensions
                C, W, H.
            reduce (bool, optional): Reduce fit parameters across a batch of
                images by average. Defaults to False.

        Returns:
            target_means (np.ndarray):  Channel means.

            target_stds (np.ndarray):   Channel standard deviations.
        """
        if len(target.shape) == 3:
            target = torch.unsqueeze(target, dim=0)
        target = standardize_brightness(target)
        means, stds = fit_fast(target, reduce=reduce)
        self.target_means = means
        self.target_stds = stds
        return means, stds

    def transform(self, I: torch.Tensor) -> torch.Tensor:
        """Normalize an H&E image.

        Args:
            img (torch.Tensor): Image, uint8 with dimensions C, W, H.

        Returns:
            torch.Tensor: Normalized image.
        """
        if len(I.shape) == 3:
            return transform(torch.unsqueeze(I, dim=0), self.target_means, self.target_stds)[0]
        else:
            return transform(I, self.target_means, self.target_stds)