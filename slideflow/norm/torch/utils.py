import torch
import torchvision
from typing import Optional
from slideflow.io.torch import cwh_to_whc

# ----------------------------------------------------------------------------0

def clip_size(I: torch.Tensor, max_size: int = 2048) -> torch.Tensor:
    # Brightness standardization uses torch.quantile(), which has an
    # unspecified maximum tensor size. Resizing to a max of (2048,2048)
    # for the brightness standardization step overcomes this issue.
    if len(I.shape) == 3:
        w, h = I.shape[0], I.shape[1]
    else:
        w, h = I.shape[1], I.shape[2]
    if w > max_size or h > max_size:
        if w > h:
            h = int((h / w) * max_size)
            w = max_size
        else:
            w = int((w / h) * max_size)
            h = max_size
        if I.shape[-1] == 3:
            from slideflow.io.torch import whc_to_cwh
            I = whc_to_cwh(I)
        I = torchvision.transforms.functional.resize(I, (w, h))
        I = cwh_to_whc(I)  # type: ignore
    return I


def brightness_percentile(I: torch.Tensor) -> torch.Tensor:
    return torch.quantile(I.to(torch.float32), 0.9)


def standardize_brightness(
    I: torch.Tensor,
    mask: bool = False
) -> torch.Tensor:
    """Standardize image brightness to 90th percentile.

    Args:
        I (torch.Tensor): Image to standardize.

    Returns:
        torch.Tensor: Brightness-standardized image (uint8)
    """
    if mask:
        ones = torch.all(I == 255, dim=len(I.shape)-1)
    bI = I if not mask else I[~ ones]
    p = brightness_percentile(bI)
    clipped = torch.clip(I * 255.0 / p, 0, 255).to(torch.uint8)
    if mask:
        clipped[ones] = 255
    return clipped