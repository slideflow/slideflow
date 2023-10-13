import torch

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
