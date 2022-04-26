import torch
from typing import Tuple
from slideflow.norm.torch import color

def lab_split(
    I: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Convert from RGB uint8 to LAB and split into channels'''

    I = I.float()
    I /= 255
    I = color.rgb_to_lab(I.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # BWHC -> BCWH -> BWHC
    I1, I2, I3 = torch.unbind(I, dim=-1)
    return I1, I2, I3

def merge_back(
    I1: torch.Tensor,
    I2: torch.Tensor,
    I3: torch.Tensor
) -> torch.Tensor:
    '''Take separate LAB channels and merge back to RGB uint8'''

    I = torch.stack((I1, I2, I3), dim=-1)
    I = color.lab_to_rgb(I.permute(0, 3, 1, 2), clip=False).permute(0, 2, 3, 1) * 255  # BWHC -> BCWH -> BWHC
    return I

def get_mean_std(
    I1: torch.Tensor,
    I2: torch.Tensor,
    I3: torch.Tensor,
    reduce: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Get mean and standard deviation of each channel.'''

    m1, sd1 = torch.mean(I1, dim=(1, 2)), torch.std(I1, dim=(1, 2))
    m2, sd2 = torch.mean(I2, dim=(1, 2)), torch.std(I2, dim=(1, 2))
    m3, sd3 = torch.mean(I3, dim=(1, 2)), torch.std(I3, dim=(1, 2))

    if reduce:
        m1, sd1 = torch.mean(m1), torch.mean(sd1)
        m2, sd2 = torch.mean(m2), torch.mean(sd2)
        m3, sd3 = torch.mean(m3), torch.mean(sd3)

    means = torch.Tensor([m1, m2, m3])
    stds = torch.Tensor([sd1, sd2, sd3])
    return means, stds

def transform(
    I: torch.Tensor,
    tgt_mean: torch.Tensor,
    tgt_std: torch.Tensor
) -> torch.Tensor:
    '''Where I = batch of images (WHC)'''

    I1, I2, I3 = lab_split(I)
    (I1_mean, I2_mean, I3_mean), (I1_std, I2_std, I3_std) = get_mean_std(I1, I2, I3)

    norm1 = ((I1 - I1_mean) * (tgt_std[0] / I1_std)) + tgt_mean[0]
    norm2 = ((I2 - I2_mean) * (tgt_std[1] / I2_std)) + tgt_mean[1]
    norm3 = ((I3 - I3_mean) * (tgt_std[2] / I3_std)) + tgt_mean[2]

    merged = merge_back(norm1, norm2, norm3).int()
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
        torch.Tensor: Fit means
        torch.Tensor: Fit stds
    """
    means, stds = get_mean_std(*lab_split(target), reduce=reduce)
    return means, stds
