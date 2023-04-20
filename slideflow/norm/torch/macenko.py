"""
Reinhard normalization based on method of:

E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional, Union
from contextlib import contextmanager

from slideflow import log
import slideflow.norm.utils as ut
from .utils import clip_size, standardize_brightness

# -----------------------------------------------------------------------------

def dot(a: torch.Tensor, b: torch.Tensor):
    """Equivalent to np.dot()."""
    if len(a.shape) == 0 or len(b.shape) == 0:
        return a * b
    if len(b.shape) == 1:
        return torch.tensordot(a, b, dims=[[-1], [-1]])
    else:
        return torch.tensordot(a, b, dims=[[-1], [-2]])


def T(a: torch.Tensor):
    return a.permute(*torch.arange(a.ndim - 1, -1, -1))

# -----------------------------------------------------------------------------

class MacenkoNormalizer:

    vectorized = False
    preferred_device = 'cpu'
    preset_tag = 'macenko'

    def __init__(
        self,
        Io: int = 255,
        alpha: float = 1,
        beta: float = 0.15,
    ) -> None:
        """Macenko H&E stain normalizer (PyTorch implementation).

        Normalizes an image as defined by:

        Macenko, Marc, et al. "A method for normalizing histology
        slides for quantitative analysis." 2009 IEEE International
        Symposium on Biomedical Imaging: From Nano to Macro. IEEE, 2009.

        Args:
            Io (int). Light transmission. Defaults to 255.
            alpha (float): Percentile of angular coordinates to be selected
                with respect to orthogonal eigenvectors. Defaults to 1.
            beta (float): Luminosity threshold. Pixels with luminance above
                this threshold will be ignored. Defaults to 0.15.

        Examples
            See :class:`slideflow.norm.StainNormalizer`
        """
        self.Io = Io
        self.alpha = alpha
        self.beta = beta
        self._ctx_maxC = None  # type: Optional[torch.Tensor]
        self._augment_params = dict()  # type: Dict[str, torch.Tensor]
        self.set_fit(**ut.fit_presets[self.preset_tag]['v3'])  # type: ignore
        self.set_augment(**ut.augment_presets[self.preset_tag]['v1'])  # type: ignore

    def fit(self, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit normalizer to a target image.

        Args:
            img (torch.Tensor): Target image (RGB uint8) with dimensions
                W, H, C.

        Returns:
            A tuple containing

                target_means (np.ndarray):  Channel means.

                target_stds (np.ndarray):   Channel standard deviations.
        """
        if len(target.shape) != 3:
            raise ValueError(
                f"Invalid shape for fit(): expected 3, got {target.shape}"
            )
        target = clip_size(target, 2048)
        HE, maxC, _ = self.matrix_and_concentrations(target)
        self.set_fit(HE, maxC)
        return HE, maxC

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

    def get_fit(self) -> Dict[str, Optional[np.ndarray]]:
        """Get the current normalizer fit.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping 'stain_matrix_target'
            and 'target_concentrations' to their respective fit values.
        """
        return {
            'stain_matrix_target': None if self.stain_matrix_target is None else self.stain_matrix_target.numpy(),
            'target_concentrations': None if self.target_concentrations is None else self.target_concentrations.numpy()
        }

    def set_fit(
        self,
        stain_matrix_target: Union[np.ndarray, torch.Tensor],
        target_concentrations: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """Set the normalizer fit to the given values.

        Args:
            stain_matrix_target (np.ndarray, torch.Tensor): Stain matrix target.
                Must have the shape (3, 2).
            target_concentrations (np.ndarray, torch.Tensor): Target
                concentrations. Must have the shape (2,).
        """
        if not isinstance(stain_matrix_target, torch.Tensor):
            stain_matrix_target = torch.from_numpy(ut._as_numpy(stain_matrix_target))
        if not isinstance(target_concentrations, torch.Tensor):
            target_concentrations = torch.from_numpy(ut._as_numpy(target_concentrations))
        self.stain_matrix_target = stain_matrix_target
        self.target_concentrations = target_concentrations

    def set_augment(
        self,
        matrix_stdev: Optional[Union[np.ndarray, torch.Tensor]] = None,
        concentrations_stdev: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        """Set the normalizer augmentation to the given values.

        Args:
            matrix_stdev (np.ndarray, tf.Tensor): Standard deviation
                of the stain matrix target. Must have the shape (3, 2).
                Defaults to None (will not augment stain matrix).
            concentrations_stdev (np.ndarray, tf.Tensor): Standard deviation
                of the target concentrations. Must have the shape (2,).
                Defaults to None (will not augment target concentrations).
        """
        if matrix_stdev is None and concentrations_stdev is None:
            raise ValueError(
                "One or both arguments 'matrix_stdev' and 'concentrations_stdev' are required."
            )
        if matrix_stdev is not None:
            self._augment_params['matrix_stdev'] = torch.from_numpy(ut._as_numpy(matrix_stdev))
        if concentrations_stdev is not None:
            self._augment_params['concentrations_stdev'] = torch.from_numpy(ut._as_numpy(concentrations_stdev))

    def _matrix_and_concentrations(
        self,
        img: torch.Tensor,
        mask: bool = False,
        standardize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets the H&E stain matrix and concentrations for a given image.

        Args:
            img (torch.Tensor): Image (RGB uint8) with dimensions W, H, C.
            mask (bool): Mask white pixels (255) during calculation.
                Defaults to False.
            standardize (bool): Perform brightness standardization.
                Defaults to True.

        Returns:
            A tuple containing

                torch.Tensor: H&E stain matrix, shape = (3, 2)

                torch.Tensor: Concentrations of individual stains
        """
        img = img.reshape((-1, 3))

        if mask:
            ones = torch.all(img == 255, dim=1)

        if standardize:
            img = standardize_brightness(img, mask=mask)

        # Calculate optical density.
        OD = -torch.log((img.to(torch.float32) + 1) / self.Io)

        # Remove transparent pixles.
        if mask:
            ODhat = OD[~ (torch.any(OD < self.beta, dim=1) | ones)]
        else:
            ODhat = OD[~torch.any(OD < self.beta, dim=1)]

        # Compute eigenvectors.
        eigvals, eigvecs = torch.linalg.eigh(torch.cov(ODhat.T))

        # Project on the plane spanned by the eigenvectors corresponding
        # to the two largest eigenvalues.
        That = ODhat.matmul(eigvecs[:, 1:3])

        phi = torch.atan2(That[:, 1], That[:, 0])

        minPhi = torch.quantile(phi, self.alpha / 100)
        maxPhi = torch.quantile(phi, 1 - self.alpha / 100)

        vMin = torch.tensordot(eigvecs[:, 1:3], T(torch.stack((torch.cos(minPhi), torch.sin(minPhi)))), dims=[[-1], [-1]])
        vMax = torch.tensordot(eigvecs[:, 1:3], T(torch.stack((torch.cos(maxPhi), torch.sin(maxPhi)))), dims=[[-1], [-1]])

        # Ensure the vector corresponding to hematoxylin is first, eosin second.
        if vMin[0] > vMax[0]:
            HE = torch.stack((vMin, vMax)).T
        else:
            HE = torch.stack((vMax, vMin)).T

        if mask:
            OD = OD[~ ones]

        # Rows correspond to channels (RGB), columns to OD values.
        Y = torch.reshape(OD, (-1, 3)).T

        # Determine concentrations of the individual stains.
        C = torch.linalg.lstsq(HE, Y, rcond=None)[0]

        return HE, C

    def matrix_and_concentrations(
        self,
        img: torch.Tensor,
        mask: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gets the H&E stain matrix and concentrations for a given image.

        Args:
            img (torch.Tensor): Image (RGB uint8) with dimensions W, H, C.
            mask (bool): Mask white pixels (255) during calculation.
                Defaults to False.

        Returns:
            A tuple containing

                torch.Tensor: H&E stain matrix, shape = (3, 2)

                torch.Tensor: Max concentrations, shape = (2,)

                torch.Tensor: Concentrations of individual stains
        """
        HE, C = self._matrix_and_concentrations(img, mask=mask)

        # Normalize stain concentrations.
        maxC = torch.stack((torch.quantile(C[0, :], 0.99), torch.quantile(C[1, :], 0.99)))

        return HE, maxC, C

    def transform(
        self,
        img: torch.Tensor,
        *,
        augment: bool = False,
        allow_errors: bool = True
    ) -> torch.Tensor:
        """Normalize an H&E image.

        Args:
            img (torch.Tensor): Image, RGB uint8 with dimensions W, H, C.

        Keyword args:
            augment (bool): Perform random stain augmentation.
                Defaults to False.

        Returns:
            torch.Tensor: Normalized image (uint8)
        """
        if len(img.shape) == 4:
            return torch.stack([
                self.transform(x_i) for x_i in torch.unbind(img, dim=0)
            ], dim=0)
        if len(img.shape) != 3:
            raise ValueError(
                f"Invalid shape for transform(): expected 3, got {img.shape}"
            )

        h, w, c = img.shape

        # Augmentation; optional
        if augment and not any(m in self._augment_params
                               for m in ('matrix_stdev', 'concentrations_stdev')):
            raise ValueError("Augmentation space not configured.")
        if augment and 'matrix_stdev' in self._augment_params:
            HERef = torch.normal(
                self.stain_matrix_target,
                self._augment_params['matrix_stdev']
            )
            HERef = HERef.to(img.device)
        else:
            HERef = self.stain_matrix_target.to(img.device)
        if augment and 'concentrations_stdev' in self._augment_params:
            maxCRef = torch.normal(
                self.target_concentrations,
                self._augment_params['concentrations_stdev']
            )
            maxCRef = maxCRef.to(img.device)
        else:
            maxCRef = self.target_concentrations.to(img.device)

        # Get stain matrix and concentrations from image.
        try:
            if self._ctx_maxC is not None:
                HE, C = self._matrix_and_concentrations(img)
                maxC = self._ctx_maxC
            else:
                HE, maxC, C = self.matrix_and_concentrations(img)
        except Exception as e:
            if allow_errors:
                log.debug(
                    "Error encountered during normalization. Returning "
                    f"original image. Error: {e}"
                )
                return img
            else:
                raise

        tmp = torch.divide(maxC, maxCRef)
        C2 = torch.divide(C, tmp[:, None])

        # Recreate the image using reference mixing matrix.
        Inorm = self.Io * torch.exp(-HERef.matmul(C2))
        Inorm = torch.clip(Inorm, 0, 255)
        Inorm = torch.reshape(Inorm.T, (h, w, 3)).to(torch.uint8)

        return Inorm

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
        I = clip_size(I, 2048)
        HE, maxC, C = self.matrix_and_concentrations(I, mask=True)
        self._ctx_maxC = maxC

    def clear_context(self):
        """Remove any previously set stain normalizer context."""
        self._ctx_maxC = None


class MacenkoFastNormalizer(MacenkoNormalizer):

    """Macenko H&E stain normalizer, with brightness standardization disabled."""

    preset_tag = 'macenko_fast'

    def _matrix_and_concentrations(
        self,
        img: torch.Tensor,
        mask: bool = False,
        standardize: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super()._matrix_and_concentrations(img, mask, standardize=False)