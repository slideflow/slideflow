import random
import torchvision
import torch

from torchvision import transforms
from typing import TYPE_CHECKING, Callable, List, Optional, Union

from .img_utils import cwh_to_whc, whc_to_cwh
from .color import generate_random_color_mapping, ColorProfile

if TYPE_CHECKING:
    from slideflow.norm import StainNormalizer

# -----------------------------------------------------------------------------

class RandomCardinalRotation:
    """Torchvision transform for random cardinal rotation."""
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


class RandomGaussianBlur:
    """Torchvision transform for random Gaussian blur."""
    def __init__(self, sigma: List[float], weights: List[float]):
        assert len(sigma) == len(weights)
        self.sigma = sigma
        self.weights = weights
        self.blur_fn = {
            s: (
                transforms.GaussianBlur(self.calc_kernel(s), sigma=s)
                if s else lambda x: x
            ) for s in self.sigma
        }

    @staticmethod
    def calc_kernel(sigma: float) -> int:
        sigma = 0.5
        opt_kernel = int((sigma * 4) + 1)
        if opt_kernel % 2 == 0:
            opt_kernel += 1
        return opt_kernel

    def __call__(self, x):
        s = random.choices(self.sigma, weights=self.weights)[0]
        return self.blur_fn[s](x)


class RandomJPEGCompression:
    """Torchvision transform for random JPEG compression."""
    def __init__(self, p: float = 0.5, q_min: int = 50, q_max: int = 100):
        self.p = p
        self.q_min = q_min
        self.q_max = q_max

    def __call__(self, x):
        return torch.where(
            torch.rand(1)[0] < self.p,
            random_jpeg_compression(x),
            x
        )


class RandomColorProfile:
    """Generate and apply a random histogram color profile."""

    def __init__(self):
        pass

    def __call__(self, x):
        mapping = generate_random_color_mapping(plot=False)
        profile = ColorProfile(mapping)
        return profile.apply(x)


class RandomColorDistortion:
    """Torchvision transform for random color distortion."""
    def __init__(self, s: float = 1.0):
        self.color_distort = compose_color_distortion(s=s)

    def __call__(self, x):
        return self.color_distort(x)


def random_jpeg_compression(
    img: torch.Tensor,
    q_min: int = 50,
    q_max: int = 100
):
    """Perform random JPEG compression on an image.

    Args:
        img (torch.Tensor): Image tensor, shape C x W x H.

    Returns:
        torch.Tensor: Transformed image (C x W x H).

    """
    q = (torch.rand(1)[0] * q_min) + (q_max - q_min)
    img = torchvision.io.encode_jpeg(img, quality=q)
    return torchvision.io.decode_image(img)


def compose_color_distortion(s=1.0):
    """Compose augmentation for random color distortion.

    Args:
        s (float): Strength of the distortion.

    Returns:
        Callable: PyTorch transform

    """
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0, 0)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose(
        [rnd_color_jitter, rnd_gray])
    return color_distort


def decode_augmentation_string(augment: str) -> List[Callable]:
    """Decode a string of augmentation characters into a list of
    augmentation functions.

    Args:
        augment (str): Augmentation string. Each character represents an
            augmentation function.

    Returns:
        List[Callable]: List of augmentation functions.

    """
    if not isinstance(augment, str):
        raise ValueError(f"Invalid argument: {augment}; expected a str")

    transformations = []  # type: List[Callable]
    for a in augment:
        if a == 'x':
            # Random x-flip.
            transformations.append(transforms.RandomHorizontalFlip(p=0.5))
        elif a == 'y':
            # Random y-flip.
            transformations.append(transforms.RandomVerticalFlip(p=0.5))
        elif a == 'r':
            # Random cardinal rotation.
            transformations.append(RandomCardinalRotation())
        elif a == 'd':
            # Random color distortion.
            transformations.append(RandomColorDistortion(s=1.0))
        elif a == 's':
            # Random sharpen
            transformations.append(transforms.RandomAdjustSharpness(sharpness_factor=2.0))
        elif a == 'p':
            # Random posterize
            transformations.append(transforms.RandomPosterize(bits=2))
        elif a == 'b':
            # Random Gaussian blur.
            transformations.append(
                RandomGaussianBlur(
                    sigma=[0, 0.5, 1.0, 1.5, 2.0],
                    weights=[0.9, 0.1, 0.05, 0.025, 0.0125]
                )
            )
        elif a == 'j':
            # Random JPEG compression.
            transformations.append(RandomJPEGCompression(p=0.5, q_min=30, q_max=100))
        elif a == 'c':
            # Random color profile.
            transformations.append(RandomColorProfile())
        elif a != 'n':
            raise ValueError(f"Invalid augmentation: {a}")
    return transformations


def compose_augmentations(
    augment: Union[str, bool] = False,
    *,
    standardize: bool = False,
    normalizer: Optional["StainNormalizer"] = None,
    transform: Optional[Callable] = None,
    whc: bool = False
):
    """Compose an augmentation pipeline for image processing.

    Args:
        augment (str or bool): Image augmentations to perform. Augmentations include:

            * ``'x'``: Random horizontal flip
            * ``'y'``: Random vertical flip
            * ``'r'``: Random 90-degree rotation
            * ``'j'``: Random JPEG compression (50% chance to compress with quality between 50-100)
            * ``'b'``: Random Gaussian blur (10% chance to blur with sigma between 0.5-2.0)
            * ``'d'``: Random color distortion (contrast and brightness)
            * ``'s'``: Random sharpen (sharpness_factor=2.0)
            * ``'p'``: Random posterize (bits=2)
            * ``'c'``: Random color profile

            Combine letters to define augmentations, such as ``'xyrj'``.
            A value of True will use ``'xyrjb'``.
            Note: this function does not support stain augmentation.

    Keyword args:
        standardize (bool, optional): Standardize images into the range (0,1)
            using img / (255/2) - 1. Defaults to False.
        normalizer (:class:`slideflow.norm.StainNormalizer`): Stain normalizer
            to use on images. Defaults to None.
        transform (Callable, optional): Arbitrary torchvision transform function.
            Performs transformation after augmentations but before standardization.
            Defaults to None.
        whc (bool): Images are in W x H x C format. Defaults to False.
    """

    transformations = []  # type: List[Callable]

    if augment is True:
        augment = 'xyrjb'

    # Stain normalization.
    if normalizer is not None:
        transformations.append(
            lambda img: normalizer.torch_to_torch(  # type: ignore
                img,
                augment=(isinstance(augment, str) and 'n' in augment)
            )
        )
    elif isinstance(augment, str) and 'n' in augment:
        raise ValueError(
            "Stain augmentation (n) requires a stain normalizer, which was not "
            "provided. Augmentation string: {}".format(augment)
        )

    # Assemble augmentation pipeline.
    if isinstance(augment, str):
        transformations += decode_augmentation_string(augment)
    elif callable(augment):
        transformations.append(augment)  # type: ignore

    # Arbitrary transformations via `transform` argument.
    if transform is not None:
        transformations.append(transform)

    # Image standardization.
    # Note: not the same as tensorflow's per_image_standardization
    # Convert back: image = (image + 1) * (255/2)
    if standardize:
        transformations.append(lambda img: img / (255/2) - 1)

    if transformations and whc:
        return transforms.Compose([whc_to_cwh] + transformations + [cwh_to_whc])
    else:
        return transforms.Compose(transformations)

# -----------------------------------------------------------------------------