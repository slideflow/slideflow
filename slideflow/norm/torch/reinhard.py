"""
Reinhard normalization based on method of:

E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.
"""

from typing import Tuple, Dict, Optional, Union

import torch
import numpy as np
from contextlib import contextmanager

import slideflow.norm.utils as ut
from slideflow.norm.torch import color
from .utils import clip_size, standardize_brightness

# -----------------------------------------------------------------------------

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


def get_masked_mean_std(I: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get mean and standard deviation of each channel, with white pixels masked.

    Args:
        I1 (torch.Tensor): First channel (uint8).
        I2 (torch.Tensor): Second channel (uint8).
        I3 (torch.Tensor): Third channel (uint8).
        reduce (bool): Reduce batch to mean across images in the batch.

    Returns:
        torch.Tensor:     Channel means, shape = (3,)
        torch.Tensor:     Channel standard deviations, shape = (3,)
    """
    ones = torch.all(I == 255, dim=3)
    I1, I2, I3 = lab_split(I)
    I1, I2, I3 = I1[~ones], I2[~ones], I3[~ones]

    m1, sd1 = torch.mean(I1), torch.std(I1)
    m2, sd2 = torch.mean(I2), torch.std(I2)
    m3, sd3 = torch.mean(I3), torch.std(I3)

    means = torch.unsqueeze(torch.stack([m1, m2, m3]), dim=1)
    stds = torch.unsqueeze(torch.stack([sd1, sd2, sd3]), dim=1)

    return means, stds


def get_mean_std(
    I1: torch.Tensor,
    I2: torch.Tensor,
    I3: torch.Tensor,
    reduce: bool = False,
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

# -----------------------------------------------------------------------------


def augmented_transform(
    I: torch.Tensor,
    tgt_mean: torch.Tensor,
    tgt_std: torch.Tensor,
    means_stdev: Optional[torch.Tensor] = None,
    stds_stdev: Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
    """Transform an image using a given target means & stds, with augmentation.

    Args:
        img (torch.Tensor): Batch of uint8 images (B x W x H x C).
        tgt_mean (torch.Tensor): Target channel means.
        tgt_std (torch.Tensor): Target channel standard deviations.
        means_stdev (torch.Tensor): Standard deviation of tgt_mean for
            augmentation.
        stds_stdev (torch.Tensor): Standard deviation of tgt_std for augmentation.

    Keyword args:
        ctx_mean (torch.Tensor, optional): Context channel means (e.g. from
            whole-slide image). If None, calculates means from the image.
            Defaults to None.
        ctx_std (torch.Tensor, optional): Context channel standard deviations
            (e.g. from whole-slide image). If None, calculates standard
            deviations from the image. Defaults to None.

    Returns:
        torch.Tensor:   Stain normalized image, with augmentation.

    """
    if means_stdev is None and stds_stdev is None:
        raise ValueError("Must supply either means_stdev and/or stds_stdev")
    if means_stdev is not None:
        tgt_mean = torch.normal(tgt_mean, means_stdev)
    if stds_stdev is not None:
        tgt_std = torch.normal(tgt_std, stds_stdev)
    return transform(I, tgt_mean, tgt_std, **kwargs)


def transform(
    I: torch.Tensor,
    tgt_mean: torch.Tensor,
    tgt_std: torch.Tensor,
    *,
    ctx_mean: Optional[torch.Tensor] = None,
    ctx_std: Optional[torch.Tensor] = None,
    mask_threshold: Optional[float] = None
) -> torch.Tensor:
    """Normalize an H&E image.

    Args:
        img (torch.Tensor): Batch of uint8 images (B x W x H x C).
        tgt_mean (torch.Tensor): Target channel means.
        tgt_std (torch.Tensor): Target channel standard deviations.

    Keyword args:
        ctx_mean (torch.Tensor, optional): Context channel means (e.g. from
            whole-slide image). If None, calculates means from the image.
            Defaults to None.
        ctx_std (torch.Tensor, optional): Context channel standard deviations
            (e.g. from whole-slide image). If None, calculates standard
            deviations from the image. Defaults to None.

    Returns:
        torch.Tensor:   Stain normalized image.

    """
    if ctx_mean is None and ctx_std is not None:
        raise ValueError(
        "If 'ctx_stds' is provided, 'ctx_means' must not be None"
    )
    if ctx_std is None and ctx_mean is not None:
        raise ValueError(
        "If 'ctx_means' is provided, 'ctx_stds' must not be None"
    )

    I1, I2, I3 = lab_split(I)

    if mask_threshold:
        mask = torch.unsqueeze(((I1 / 100) < mask_threshold), -1)

    if ctx_mean is not None and ctx_std is not None:
        I1_mean, I2_mean, I3_mean = ctx_mean[0], ctx_mean[1], ctx_mean[2]
        I1_std, I2_std, I3_std = ctx_std[0], ctx_std[1], ctx_std[2]
    else:
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
    if mask_threshold:
        return torch.where(mask, clipped, I)
    else:
        return clipped


def fit(
    target: torch.Tensor,
    reduce: bool = False,
    mask: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit a target image.

    Args:
        target (torch.Tensor): Batch of images to fit.
        reduce (bool, optional): Reduce the fit means/stds across the batch
            of images to a single mean/std array, reduced by average.
            Defaults to False (provides fit for each image in the batch).
        mask (bool): Ignore white pixels (255, 255, 255) when fitting.
                Defulats to False.

    Returns:
        A tuple containing

            torch.Tensor: Fit means

            torch.Tensor: Fit stds
    """
    if mask:
        return get_masked_mean_std(target)
    else:
        return get_mean_std(*lab_split(target), reduce=reduce)


class ReinhardFastNormalizer:

    vectorized = True
    preferred_device = 'gpu'
    preset_tag = 'reinhard_fast'

    def __init__(self) -> None:
        """Modified Reinhard H&E stain normalizer without brightness
        standardization (PyTorch implementation).

        Normalizes an image as defined by:

        Reinhard, Erik, et al. "Color transfer between images." IEEE
        Computer graphics and applications 21.5 (2001): 34-41.

        This implementation does not include the brightness normalization step.
        """
        self.threshold = None  # type: Optional[float]
        self._ctx_means = None  # type: Optional[torch.Tensor]
        self._ctx_stds = None  # type: Optional[torch.Tensor]
        self._augment_params = dict()  # type: Dict[str, torch.Tensor]
        self.set_fit(**ut.fit_presets[self.preset_tag]['v3'])  # type: ignore
        self.set_augment(**ut.augment_presets[self.preset_tag]['v1'])  # type: ignore

    def fit(
        self,
        target: torch.Tensor,
        reduce: bool = False,
        mask: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit normalizer to a target image.

        Args:
            img (torch.Tensor): Target image (RGB uint8) with dimensions
                W, H, C.
            reduce (bool, optional): Reduce fit parameters across a batch of
                images by average. Defaults to False.
            mask (bool): Ignore white pixels (255, 255, 255) when fitting.
                Defulats to False.

        Returns:
            A tuple containing

                target_means (np.ndarray):  Channel means.

                target_stds (np.ndarray):   Channel standard deviations.
        """
        if len(target.shape) == 3:
            target = torch.unsqueeze(target, dim=0)
        target = clip_size(target, 2048)
        means, stds = fit(target, reduce=reduce, mask=mask)
        self.target_means = means
        self.target_stds = stds
        return means, stds

    def augment_preset(self, preset: str) -> Dict[str, np.ndarray]:
        """Configure normalizer augmentation using a preset.

        Args:
            preset (str): Preset.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping fit keys to the
                augmentation values (standard deviations).
        """
        _aug = ut.augment_presets[self.preset_tag][preset]
        self.set_augment(**_aug)
        return _aug

    def fit_preset(self, preset: str) -> Dict[str, np.ndarray]:
        """Fit normalizer to a preset in sf.norm.utils.fit_presets.

        Args:
            preset (str): Preset.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping fit keys to their
            fitted values.
        """
        _fit = ut.fit_presets[self.preset_tag][preset]
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

    def _get_context_means(
        self,
        ctx_means: Optional[torch.Tensor] = None,
        ctx_stds: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self._ctx_means is not None and self._ctx_stds is not None:
            return self._ctx_means, self._ctx_stds
        else:
            return ctx_means, ctx_stds

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

    def set_augment(
        self,
        means_stdev: Optional[Union[np.ndarray, torch.Tensor]] = None,
        stds_stdev: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        """Set the normalizer augmentation to the given values.

        Args:
            means_stdev (np.ndarray, tf.Tensor): Standard deviation
                of the target means. Must have the shape (3,).
                Defaults to None (will not augment target means).
            stds_stdev (np.ndarray, tf.Tensor): Standard deviation
                of the target stds. Must have the shape (3,).
                Defaults to None (will not augment target stds).
        """
        if means_stdev is None and stds_stdev is None:
            raise ValueError(
                "One or both arguments 'means_stdev' and 'stds_stdev' are required."
            )
        if means_stdev is not None:
            self._augment_params['means_stdev'] = torch.from_numpy(ut._as_numpy(means_stdev))
        if stds_stdev is not None:
            self._augment_params['stds_stdev'] = torch.from_numpy(ut._as_numpy(stds_stdev))

    def transform(
        self,
        I: torch.Tensor,
        ctx_means: Optional[torch.Tensor] = None,
        ctx_stds: Optional[torch.Tensor] = None,
        *,
        augment: bool = False
    ) -> torch.Tensor:
        """Normalize an H&E image.

        Args:
            img (torch.Tensor): Image, RGB uint8 with dimensions W, H, C.
            ctx_means (tf.Tensor, optional): Context channel means (e.g. from
                whole-slide image). If None, calculates means from the image.
                Defaults to None.
            ctx_stds (tf.Tensor, optional): Context channel standard deviations
                (e.g. from whole-slide image). If None, calculates standard
                deviations from the image. Defaults to None.

        Keyword args:
            augment (bool): Transform using stain augmentation.
                Defaults to False.

        Returns:
            torch.Tensor: Normalized image (uint8)
        """
        if augment and not any(m in self._augment_params
                               for m in ('means_stdev', 'stds_stdev')):
            raise ValueError("Augmentation space not configured.")

        _I = torch.unsqueeze(I, dim=0) if len(I.shape) == 3 else I
        _ctx_means, _ctx_stds = self._get_context_means(ctx_means, ctx_stds)
        aug_kw = self._augment_params if augment else {}
        fn = augmented_transform if augment else transform
        transformed = fn(
            _I,
            self.target_means,
            self.target_stds,
            ctx_mean=_ctx_means,
            ctx_std=_ctx_stds,
            mask_threshold=self.threshold,
            **aug_kw
        )
        if len(I.shape) == 3:
            return transformed[0]
        else:
            return transformed

    @contextmanager
    def image_context(self, I: Union[np.ndarray, torch.Tensor]):
        """Set the whole-slide context for the stain normalizer.

        With contextual normalization, max concentrations are determined
        from the context (whole-slide image) rather than the image being
        normalized. This may improve stain normalization for sections of
        a slide that are predominantly eosin (e.g. necrosis or low cellularity).

        When calculating max concentrations from the image context,
        white pixels (255) will be masked.

        This function is a context manager used for temporarily setting the
        image context. For example:

        .. code-block:: python

            with normalizer.image_context(slide):
                normalizer.transform(target)

        Args:
            I (np.ndarray, torch.Tensor): Context to use for normalization, e.g.
                a whole-slide image thumbnail, optionally masked with masked
                areas set to (255, 255, 255).

        """
        self.set_context(I)
        yield
        self.clear_context()

    def set_context(self, I: Union[np.ndarray, torch.Tensor]):
        """Set the whole-slide context for the stain normalizer.

        With contextual normalization, max concentrations are determined
        from the context (whole-slide image) rather than the image being
        normalized. This may improve stain normalization for sections of
        a slide that are predominantly eosin (e.g. necrosis or low cellularity).

        When calculating max concentrations from the image context,
        white pixels (255) will be masked.

        Args:
            I (np.ndarray, torch.Tensor): Context to use for normalization, e.g.
                a whole-slide image thumbnail, optionally masked with masked
                areas set to (255, 255, 255).

        """
        if not isinstance(I, torch.Tensor):
            I = torch.from_numpy(ut._as_numpy(I))
        if len(I.shape) == 3:
            I = torch.unsqueeze(I, dim=0)
        I = clip_size(I, 2048)
        self._ctx_means, self._ctx_stds = get_masked_mean_std(I)

    def clear_context(self):
        """Remove any previously set stain normalizer context."""
        self._ctx_means, self._ctx_stds = None, None


class ReinhardFastMaskNormalizer(ReinhardFastNormalizer):

    preset_tag = 'reinhard_fast'

    def __init__(self, threshold: float = 0.93) -> None:
        """Modified Reinhard H&E stain normalizer only applied to
        non-whitepsace areas (PyTorch implementation).

        Normalizes an image as defined by:

        Reinhard, Erik, et al. "Color transfer between images." IEEE
        Computer graphics and applications 21.5 (2001): 34-41.

        This "masked" implementation only normalizes non-whitespace areas.

        This normalizer contains inspiration from StainTools by Peter Byfield
        (https://github.com/Peter554/StainTools).

        Args:
            threshold (float): Whitespace fraction threshold, above which
                pixels are masked and not normalized. Defaults to 0.93.
        """
        super().__init__()
        self.threshold = threshold


class ReinhardNormalizer(ReinhardFastNormalizer):

    preset_tag = 'reinhard'

    def __init__(self) -> None:
        """Reinhard H&E stain normalizer (PyTorch implementation).

        Normalizes an image as defined by:

        Reinhard, Erik, et al. "Color transfer between images." IEEE
        Computer graphics and applications 21.5 (2001): 34-41.

        """
        super().__init__()
        self.set_fit(**ut.fit_presets[self.preset_tag]['v3'])  # type: ignore

    def fit(
        self,
        target: torch.Tensor,
        reduce: bool = False,
        mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit normalizer to a target image.

        Args:
            img (torch.Tensor): Target image (RGB uint8) with dimensions
                W, H, C.
            reduce (bool, optional): Reduce fit parameters across a batch of
                images by average. Defaults to False.
            mask (bool): Ignore white pixels (255, 255, 255) when fitting.
                Defulats to False.

        Returns:
            A tuple containing

                target_means (np.ndarray):  Channel means.

                target_stds (np.ndarray):   Channel standard deviations.
        """
        if len(target.shape) == 3:
            target = torch.unsqueeze(target, dim=0)
        target = clip_size(target, 2048)
        target = standardize_brightness(target, mask=mask)
        means, stds = fit(target, reduce=reduce, mask=mask)
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
        _fit = ut.fit_presets[self.preset_tag][preset]
        self.set_fit(**_fit)
        return _fit

    def transform(
        self,
        I: torch.Tensor,
        ctx_means: Optional[torch.Tensor] = None,
        ctx_stds: Optional[torch.Tensor] = None,
        *,
        augment: bool = False
    ) -> torch.Tensor:
        """Normalize an H&E image.

        Args:
            img (torch.Tensor): Image, uint8 with dimensions W, H, C.
            ctx_means (tf.Tensor, optional): Context channel means (e.g. from
                whole-slide image). If None, calculates means from the image.
                Defaults to None.
            ctx_stds (tf.Tensor, optional): Context channel standard deviations
                (e.g. from whole-slide image). If None, calculates standard
                deviations from the image. Defaults to None.

        Keyword args:
            augment (bool): Transform using stain augmentation.
                Defaults to False.

        Returns:
            torch.Tensor: Normalized image.
        """
        if augment and not any(m in self._augment_params
                               for m in ('means_stdev', 'stds_stdev')):
            raise ValueError("Augmentation space not configured.")

        _I = torch.unsqueeze(I, dim=0) if len(I.shape) == 3 else I
        _I = standardize_brightness(_I)
        _ctx_means, _ctx_stds = self._get_context_means(ctx_means, ctx_stds)
        aug_kw = self._augment_params if augment else {}
        fn = augmented_transform if augment else transform
        transformed = fn(
            _I,
            self.target_means,
            self.target_stds,
            ctx_mean=_ctx_means,
            ctx_std=_ctx_stds,
            mask_threshold=self.threshold,
            **aug_kw
        )
        if len(I.shape) == 3:
            return transformed[0]
        else:
            return transformed

    def set_context(self, I: Union[np.ndarray, torch.Tensor]):
        """Set the whole-slide context for the stain normalizer.

        With contextual normalization, max concentrations are determined
        from the context (whole-slide image) rather than the image being
        normalized. This may improve stain normalization for sections of
        a slide that are predominantly eosin (e.g. necrosis or low cellularity).

        When calculating max concentrations from the image context,
        white pixels (255) will be masked.

        Args:
            I (np.ndarray, torch.Tensor): Context to use for normalization, e.g.
                a whole-slide image thumbnail, optionally masked with masked
                areas set to (255, 255, 255).

        """
        if not isinstance(I, torch.Tensor):
            I = torch.from_numpy(ut._as_numpy(I))
        if len(I.shape) == 3:
            I = torch.unsqueeze(I, dim=0)
        I = clip_size(I, 2048)
        I = standardize_brightness(I, mask=True)
        super().set_context(I)

    def clear_context(self):
        """Remove any previously set stain normalizer context."""
        super().clear_context()

class ReinhardMaskNormalizer(ReinhardNormalizer):

    preset_tag = 'reinhard'

    def __init__(self, threshold: float = 0.93) -> None:
        """Modified Reinhard H&E stain normalizer only applied to
        non-whitepsace areas (PyTorch implementation).

        Normalizes an image as defined by:

        Reinhard, Erik, et al. "Color transfer between images." IEEE
        Computer graphics and applications 21.5 (2001): 34-41.

        This "masked" implementation only normalizes non-whitespace areas.

        This normalizer contains inspiration from StainTools by Peter Byfield
        (https://github.com/Peter554/StainTools).

        Args:
            threshold (float): Whitespace fraction threshold, above which
                pixels are masked and not normalized. Defaults to 0.93.
        """
        super().__init__()
        self.threshold = threshold