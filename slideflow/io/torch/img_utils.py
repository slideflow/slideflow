import torch
import torchvision
import numpy as np
from slideflow.io import convert_dtype
from typing import Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from slideflow.norm import StainNormalizer
    from torchvision.transforms import InterpolationMode

# -----------------------------------------------------------------------------

def is_cwh(img: torch.Tensor) -> bool:
    """Check if Tensor is in C x W x H format."""
    return (len(img.shape) == 3 and img.shape[0] == 3
            or (len(img.shape) == 4 and img.shape[1] == 3))


def is_whc(img: torch.Tensor) -> bool:
    """Check if Tensor is in W x H x C format."""
    return img.shape[-1] == 3


def as_cwh(img: torch.Tensor) -> torch.Tensor:
    """Convert torch tensor to C x W x H format."""
    if is_cwh(img):
        return img
    elif is_whc(img):
        return whc_to_cwh(img)
    else:
        raise ValueError(
            "Invalid shape for channel conversion. Expected 3 or 4 dims, "
            f"got {len(img.shape)} (shape={img.shape})")


def as_whc(img: torch.Tensor) -> torch.Tensor:
    """Convert torch tensor to W x H x C format."""
    if is_whc(img):
        return img
    elif is_cwh(img):
        return cwh_to_whc(img)
    else:
        raise ValueError(
            "Invalid shape for channel conversion. Expected 3 or 4 dims, "
            f"got {len(img.shape)} (shape={img.shape})")


def cwh_to_whc(img: torch.Tensor) -> torch.Tensor:
    """Convert torch tensor from C x W x H => W x H x C"""
    if len(img.shape) == 3:
        return img.permute(1, 2, 0)  # CWH -> WHC
    elif len(img.shape) == 4:
        return img.permute(0, 2, 3, 1)  # BCWH -> BWHC
    else:
        raise ValueError(
            "Invalid shape for channel conversion. Expected 3 or 4 dims, "
            f"got {len(img.shape)} (shape={img.shape})")


def whc_to_cwh(img: torch.Tensor) -> torch.Tensor:
    """Convert torch tensor from W x H x C => C x W x H"""
    if len(img.shape) == 3:
        return img.permute(2, 0, 1)  # WHC => CWH
    elif len(img.shape) == 4:
        return img.permute(0, 3, 1, 2)  # BWHC -> BCWH
    else:
        raise ValueError(
            "Invalid shape for channel conversion. Expected 3 or 4 dims, "
            f"got {len(img.shape)} (shape={img.shape})")

# -----------------------------------------------------------------------------


def preprocess_uint8(
    img: torch.Tensor,
    normalizer: Optional["StainNormalizer"] = None,
    standardize: bool = True,
    resize_px: Optional[int] = None,
    resize_method: Optional["InterpolationMode"] = None,
    resize_aa: bool = True,
) -> torch.Tensor:
    """Process batch of tensorflow images, resizing, normalizing,
    and standardizing.

    Args:
        img (tf.Tensor): Batch of tensorflow images (uint8).
        normalizer (sf.norm.StainNormalizer, optional): Normalizer.
            Defaults to None.
        standardize (bool, optional): Standardize images. Defaults to True.
        resize_px (Optional[int], optional): Resize images. Defaults to None.
        resize_method (str, optional): Interpolation mode for resizing. Must
            be a valid torchvision.transforms.InterpolationMode. Defaults to
            BICUBIC.
        resize_aa (bool, optional): Apply antialiasing during resizing.
            Defaults to True.

    Returns:
        Dict[str, tf.Tensor]: Processed image.
    """
    if resize_px is not None:
        if resize_method is None:
            resize_method = torchvision.transforms.InterpolationMode.BICUBIC
        img = torchvision.transforms.functional.resize(
            img,
            size=resize_px,
            interpolation=resize_method,
            antialias=resize_aa
        )
    if normalizer is not None:
        img = normalizer.torch_to_torch(img)  # type: ignore
    if standardize:
        img = convert_dtype(img, torch.float32)
    return img


def decode_image(
    image: Union[bytes, str, torch.Tensor],
    *,
    img_type: Optional[str] = None,
    device: Optional[torch.device] = None,
    transform: Optional[Any] = None,
) -> torch.Tensor:
    """Decodes image string/bytes to Tensor (W x H x C).

    Torch implementation; different than sf.io.tensorflow.

    Args:
        image (Union[bytes, str, torch.Tensor]): Image to decode.

    Keyword args:
        img_type (str, optional): Image type. Defaults to None.
        device (torch.device, optional): Device to move image to.
            Defaults to None.
        transform (Callable, optional): Arbitrary torchvision transform function.
            Performs transformation after augmentations but before standardization.
            Defaults to None.

    """
    if img_type != 'numpy':
        np_data = torch.from_numpy(np.fromstring(image, dtype=np.uint8))
        image = cwh_to_whc(torchvision.io.decode_image(np_data))
        # Alternative method using PIL decoding:
        # image = np.array(Image.open(BytesIO(img_string)))

    assert isinstance(image, torch.Tensor)

    if device is not None:
        image = image.to(device)

    if transform is not None:
        image = transform(image)

    return image
