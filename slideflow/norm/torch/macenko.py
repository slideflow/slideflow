"""
Reinhard normalization based on method of:

E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional, Union
from contextlib import contextmanager

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
        self.set_fit(**ut.fit_presets[self.preset_tag]['v1'])  # type: ignore

    def fit(self, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _matrix_and_concentrations(
        self,
        img: torch.Tensor,
        mask: bool = False,
        standardize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets the H&E stain matrix and concentrations for a given image.

        Args:
            img (torch.Tensor): Image (RGB uint8) with dimensions W, H, C.

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
        OD = -torch.log((img.to(torch.float32) + 1) / 255)

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

    def transform(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize an H&E image.

        Args:
            img (torch.Tensor): Image, RGB uint8 with dimensions W, H, C.

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
        HERef = self.stain_matrix_target.to(img.device)
        maxCRef = self.target_concentrations.to(img.device)

        # Get stain matrix and concentrations from image.
        if self._ctx_maxC is not None:
            HE, C = self._matrix_and_concentrations(img)
            maxC = self._ctx_maxC
        else:
            HE, maxC, C = self.matrix_and_concentrations(img)

        tmp = torch.divide(maxC, maxCRef)
        C2 = torch.divide(C, tmp[:, None])

        # Recreate the image using reference mixing matrix.
        Inorm = 255 * torch.exp(-HERef.matmul(C2))
        Inorm = torch.clip(Inorm, 0, 255)
        Inorm = torch.reshape(Inorm.T, (h, w, 3)).to(torch.uint8)

        return Inorm

    @contextmanager
    def image_context(self, I: Union[np.ndarray, torch.Tensor]):
        self.set_context(I)
        yield
        self.clear_context()

    def set_context(self, I: Union[np.ndarray, torch.Tensor]):
        if not isinstance(I, torch.Tensor):
            I = torch.from_numpy(ut._as_numpy(I))
        I = clip_size(I, 2048)
        HE, maxC, C = self.matrix_and_concentrations(I, mask=True)
        self._ctx_maxC = maxC

    def clear_context(self):
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