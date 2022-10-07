"""
Reinhard normalization based on method of:

E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.
"""

from typing import Tuple, Dict, Optional, Union

import os
import torch
import torchvision
import numpy as np

import slideflow.norm.utils as ut
from slideflow.norm.torch import color
from slideflow.io.torch import cwh_to_whc


def standardize_brightness(I: torch.Tensor) -> torch.Tensor:
    """Standardize image brightness to 90th percentile.

    Args:
        I (torch.Tensor): Image to standardize.

    Returns:
        torch.Tensor: Brightness-standardized image (uint8)
    """
    p = torch.quantile(I.to(torch.float32), 0.9)
    return torch.clip(I * 255.0 / p, 0, 255).to(torch.uint8)


def lab_split(
    I: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert from RGB uint8 to LAB and split into channels

    Args:
        I (torch.Tensor): RGB uint8 image.

    Returns:
        A tuple containing

            torch.Tensor: I1, first channel (uint8).

            torch.Tensor: I2, first channel (uint8).

            torch.Tensor: I3, first channel (uint8).
    """

    I = I.to(torch.float32)
    I /= 255
    I = color.rgb_to_lab(I.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # BWHC -> BCWH -> BWHC
    I1, I2, I3 = torch.unbind(I, dim=-1)
    return I1, I2, I3


def merge_back(
    I1: torch.Tensor,
    I2: torch.Tensor,
    I3: torch.Tensor
) -> torch.Tensor:
    """Take seperate LAB channels and merge back to give RGB uint8

    Args:
        I1 (torch.Tensor): First channel (uint8).
        I2 (torch.Tensor): Second channel (uint8).
        I3 (torch.Tensor): Third channel (uint8).

    Returns:
        torch.Tensor: RGB uint8 image.
    """
    I = torch.stack((I1, I2, I3), dim=-1)
    I = color.lab_to_rgb(I.permute(0, 3, 1, 2), clip=False).permute(0, 2, 3, 1) * 255  # BWHC -> BCWH -> BWHC
    return I


def get_mean_std(
    I1: torch.Tensor,
    I2: torch.Tensor,
    I3: torch.Tensor,
    reduce: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get mean and standard deviation of each channel.

    Args:
        I1 (torch.Tensor): First channel (uint8).
        I2 (torch.Tensor): Second channel (uint8).
        I3 (torch.Tensor): Third channel (uint8).
        reduce (bool): Reduce batch to mean across images in the batch.

    Returns:
        torch.Tensor:     Channel means, shape = (3,)
        torch.Tensor:     Channel standard deviations, shape = (3,)
    """

    m1, sd1 = torch.mean(I1, dim=(1, 2)), torch.std(I1, dim=(1, 2))
    m2, sd2 = torch.mean(I2, dim=(1, 2)), torch.std(I2, dim=(1, 2))
    m3, sd3 = torch.mean(I3, dim=(1, 2)), torch.std(I3, dim=(1, 2))

    if reduce:
        m1, sd1 = torch.mean(m1), torch.mean(sd1)
        m2, sd2 = torch.mean(m2), torch.mean(sd2)
        m3, sd3 = torch.mean(m3), torch.mean(sd3)

    means = torch.stack([m1, m2, m3])
    stds = torch.stack([sd1, sd2, sd3])
    return means, stds


def transform(
    I: torch.Tensor,
    tgt_mean: torch.Tensor,
    tgt_std: torch.Tensor
) -> torch.Tensor:
    """Normalize an H&E image.

    Args:
        img (torch.Tensor): Batch of uint8 images (B x W x H x C).
        tgt_mean (torch.Tensor): Target channel means.
        tgt_std (torch.Tensor): Target channel standard deviations.

    Returns:
        torch.Tensor:   Stain normalized image.

    """
    I1, I2, I3 = lab_split(I)
    (I1_mean, I2_mean, I3_mean), (I1_std, I2_std, I3_std) = get_mean_std(I1, I2, I3)

    def norm(_I, _I_mean, _I_std, _tgt_std, _tgt_mean):
        # Equivalent to:
        #   norm1 = ((I1 - I1_mean) * (tgt_std / I1_std)) + tgt_mean[0]
        # But supports batches of images
        part1 = _I - _I_mean[:, None, None].expand(_I.shape)
        part2 = _tgt_std / _I_std
        part3 = part1 * part2[:, None, None].expand(part1.shape)
        return part3 + _tgt_mean

    norm1 = norm(I1, I1_mean, I1_std, tgt_std[0], tgt_mean[0])
    norm2 = norm(I2, I2_mean, I2_std, tgt_std[1], tgt_mean[1])
    norm3 = norm(I3, I3_mean, I3_std, tgt_std[2], tgt_mean[2])

    merged = merge_back(norm1, norm2, norm3)
    clipped = torch.clip(merged, min=0, max=255).to(torch.uint8)
    return clipped


def fit(
    target: torch.Tensor,
    reduce: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit a target image.

    Args:
        target (torch.Tensor): Batch of images to fit.
        reduce (bool, optional): Reduce the fit means/stds across the batch
            of images to a single mean/std array, reduced by average.
            Defaults to False (provides fit for each image in the batch).

    Returns:
        A tuple containing

            torch.Tensor: Fit means

            torch.Tensor: Fit stds
    """
    means, stds = get_mean_std(*lab_split(target), reduce=reduce)
    return means, stds


class ReinhardFastNormalizer:

    vectorized = True

    def __init__(self) -> None:
        """Modified Reinhard H&E stain normalizer without brightness
        standardization (PyTorch implementation).

        Normalizes an image as defined by:

        Reinhard, Erik, et al. "Color transfer between images." IEEE
        Computer graphics and applications 21.5 (2001): 34-41.

        This implementation does not include the brightness normalization step.
        """
        self.set_fit(**ut.fit_presets['reinhard_fast']['v1'])  # type: ignore

    def fit(
        self,
        target: torch.Tensor,
        reduce: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit normalizer to a target image.

        Args:
            img (torch.Tensor): Target image (RGB uint8) with dimensions
                W, H, C.
            reduce (bool, optional): Reduce fit parameters across a batch of
                images by average. Defaults to False.

        Returns:
            A tuple containing

                target_means (np.ndarray):  Channel means.

                target_stds (np.ndarray):   Channel standard deviations.
        """
        if len(target.shape) == 3:
            target = torch.unsqueeze(target, dim=0)
        means, stds = fit(target, reduce=reduce)
        self.target_means = means
        self.target_stds = stds
        return means, stds

    def fit_preset(self, preset: str) -> Dict[str, np.ndarray]:
        """Fit normalizer to a preset in sf.norm.utils.fit_presets.

        Args:
            preset (str): Preset.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping fit keys to their
            fitted values.
        """
        _fit = ut.fit_presets['reinhard_fast'][preset]
        self.set_fit(**_fit)
        return _fit

    def get_fit(self) -> Dict[str, Optional[np.ndarray]]:
        """Get the current normalizer fit.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping 'target_means'
            and 'target_stds' to their respective fit values.
        """
        return {
            'target_means': None if self.target_means is None else self.target_means.numpy(),
            'target_stds': None if self.target_stds is None else self.target_stds.numpy()
        }

    def set_fit(
        self,
        target_means: Union[np.ndarray, torch.Tensor],
        target_stds: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """Set the normalizer fit to the given values.

        Args:
            target_means (np.ndarray, torch.Tensor): Channel means. Must
                have the shape (3,).
            target_stds (np.ndarray, torch.Tensor): Channel standard deviations.
                Must have the shape (3,).
        """
        if not isinstance(target_means, torch.Tensor):
            target_means = torch.from_numpy(ut._as_numpy(target_means))
        if not isinstance(target_stds, torch.Tensor):
            target_stds = torch.from_numpy(ut._as_numpy(target_stds))
        self.target_means = target_means
        self.target_stds = target_stds

    def transform(self, I: torch.Tensor) -> torch.Tensor:
        """Normalize an H&E image.

        Args:
            img (torch.Tensor): Image, RGB uint8 with dimensions W, H, C.

        Returns:
            torch.Tensor: Normalized image (uint8)
        """
        if len(I.shape) == 3:
            return transform(torch.unsqueeze(I, dim=0), self.target_means, self.target_stds)[0]
        else:
            return transform(I, self.target_means, self.target_stds)


class ReinhardNormalizer(ReinhardFastNormalizer):

    def __init__(self) -> None:
        """Reinhard H&E stain normalizer (PyTorch implementation).

        Normalizes an image as defined by:

        Reinhard, Erik, et al. "Color transfer between images." IEEE
        Computer graphics and applications 21.5 (2001): 34-41.

        """
        super().__init__()
        self.set_fit(**ut.fit_presets['reinhard']['v1'])  # type: ignore

    def fit(
        self,
        target: torch.Tensor,
        reduce: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit normalizer to a target image.

        Args:
            img (torch.Tensor): Target image (RGB uint8) with dimensions
                W, H, C.
            reduce (bool, optional): Reduce fit parameters across a batch of
                images by average. Defaults to False.

        Returns:
            A tuple containing

                target_means (np.ndarray):  Channel means.

                target_stds (np.ndarray):   Channel standard deviations.
        """
        if len(target.shape) == 3:
            target = torch.unsqueeze(target, dim=0)
        target = standardize_brightness(target)
        means, stds = fit(target, reduce=reduce)
        self.target_means = means
        self.target_stds = stds
        return means, stds

    def fit_preset(self, preset: str) -> Dict[str, np.ndarray]:
        """Fit normalizer to a preset in sf.norm.utils.fit_presets.

        Args:
            preset (str): Preset.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping fit keys to their
            fitted values.
        """
        _fit = ut.fit_presets['reinhard'][preset]
        self.set_fit(**_fit)
        return _fit

    def transform(self, I: torch.Tensor) -> torch.Tensor:
        """Normalize an H&E image.

        Args:
            img (torch.Tensor): Image, uint8 with dimensions W, H, C.

        Returns:
            torch.Tensor: Normalized image.
        """
        if len(I.shape) == 3:
            return transform(standardize_brightness(torch.unsqueeze(I, dim=0)), self.target_means, self.target_stds)[0]
        else:
            return transform(standardize_brightness(I), self.target_means, self.target_stds)