from typing import Any
import torch
from torchvision import transforms

def crop(
    img: torch.Tensor,
    gan_um: int,
    gan_px: int,
    target_um: int
) -> Any:
    """Process a batch of raw GAN output, converting to a Tensorflow tensor.

    Args:
        img (torch.Tensor): Raw batch of GAN images.
        gan_um (int, optional): Size of gan output images, in microns.
        gan_px (int, optional): Size of gan output images, in pixels.
        target_um (int, optional): Size of target images, in microns.
            Will crop image to meet this target.

    Returns:
        Cropped image.
    """

    # Calculate parameters for resize/crop.
    crop_factor = target_um / gan_um
    crop_width = int(crop_factor * gan_px)
    left = int(gan_px/2 - crop_width/2)
    upper = int(gan_px/2 - crop_width/2)

    # Perform crop/resize and convert to tensor
    return transforms.functional.crop(img, upper, left, crop_width, crop_width)
