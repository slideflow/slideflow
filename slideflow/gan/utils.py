import numpy as np
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

def crop(
    img: "torch.Tensor",
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
    from torchvision import transforms

    # Calculate parameters for resize/crop.
    crop_factor = target_um / gan_um
    crop_width = int(crop_factor * gan_px)
    left = int(gan_px/2 - crop_width/2)
    upper = int(gan_px/2 - crop_width/2)

    # Perform crop/resize and convert to tensor
    return transforms.functional.crop(img, upper, left, crop_width, crop_width)


def noise_tensor(seed: int, z_dim: int) -> "torch.Tensor":
    """Creates a noise tensor based on a given seed and dimension size.

    Args:
        seed (int): Seed.
        z_dim (int): Dimension of noise vector to create.

    Returns:
        torch.Tensor: Noise vector of shape (1, z_dim)
    """
    import torch
    return torch.from_numpy(np.random.RandomState(seed).randn(1, z_dim))