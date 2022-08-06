"""Macenko H&E stain normalization."""

from __future__ import division

import numpy as np
from typing import Tuple, Dict

import slideflow.norm.utils as ut


class MacenkoNormalizer:

    def __init__(
        self,
        alpha: float = 1,
        beta: float = 0.15
    ) -> None:
        """Macenko H&E stain normalizer (numpy implementation).

        Normalizes an image as defined by:

        Macenko, Marc, et al. "A method for normalizing histology
        slides for quantitative analysis." 2009 IEEE International
        Symposium on Biomedical Imaging: From Nano to Macro. IEEE, 2009.

        This normalizer contains inspiration from StainTools by Peter Byfield
        (https://github.com/Peter554/StainTools).

        Args:
            alpha (float): Percentile of angular coordinates to be selected
                with respect to orthogonal eigenvectors. Defaults to 1.
            beta (float): Luminosity threshold. Pixels with luminance above
                this threshold will be ignored. Defaults to 0.15.

        Examples
            See :class:`slideflow.norm.StainNormalizer`
        """
        self.alpha = alpha
        self.beta = beta

        # Default fit.
        self.set_fit(**ut.fit_presets['macenko']['v1'])  # type: ignore

    def fit(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit normalizer to a target image.

        Calculates the stain matrix and concentrations for the given image,
        and sets these values as the normalizer target.

        Args:
            img (np.ndarray): Target image (RGB uint8) with dimensions W, H, C.

        Returns:
            A tuple containing

                np.ndarray:     Stain matrix target.

                np.ndarray:     Target concentrations.
        """
        HE, maxC, _ = self.matrix_and_concentrations(img)
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
        _fit = ut.fit_presets['macenko'][preset]
        self.set_fit(**_fit)
        return _fit

    def get_fit(self) -> Dict[str, np.ndarray]:
        """Get the current normalizer fit.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping 'stain_matrix_target'
            and 'target_concentrations' to their respective fit values.
        """
        return {
            'stain_matrix_target': self.stain_matrix_target,
            'target_concentrations': self.target_concentrations
        }

    def set_fit(
        self,
        stain_matrix_target: np.ndarray,
        target_concentrations: np.ndarray
    ) -> None:
        """Set the normalizer fit to the given values.

        Args:
            stain_matrix_target (np.ndarray): Stain matrix target. Must
                have the shape (3, 2).
            target_concentrations (np.ndarray): Target concentrations. Must
                have the shape (2,).
        """
        stain_matrix_target = ut._as_numpy(stain_matrix_target)
        target_concentrations = ut._as_numpy(target_concentrations)

        if stain_matrix_target.shape != (3, 2):
            raise ValueError("stain_matrix_target must have shape (3, 2) - "
                             f"got {stain_matrix_target.shape}")
        if target_concentrations.shape != (2,):
            raise ValueError("target_concentrations must have shape (2,) - "
                             f"got {target_concentrations.shape}")

        self.stain_matrix_target = stain_matrix_target
        self.target_concentrations = target_concentrations

    def matrix_and_concentrations(
        self,
        img: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gets the H&E stain matrix and concentrations for a given image.

        Args:
            img (np.ndarray): Image (RGB uint8) with dimensions W, H, C.

        Returns:
            A tuple containing

                np.ndarray: H&E stain matrix, shape = (3, 2)

                np.ndarray: Concentrations, shape = (2,)

                np.ndarray: Concentrations of individual stains
        """

        img = img.reshape((-1, 3))
        img = ut.standardize_brightness(img)

        # Calculate optical density.
        OD = -np.log((img.astype(float) + 1) / 255)

        # Remove transparent pixels.
        ODhat = OD[~np.any(OD < self.beta, axis=1)]

        # Compute eigenvectors.
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

        # Project on the plane spanned by the eigenvectors corresponding to
        # the two largest eigenvalues.
        That = ODhat.dot(eigvecs[:, 1:3])

        phi = np.arctan2(That[:, 1], That[:, 0])

        minPhi = np.percentile(phi, self.alpha)
        maxPhi = np.percentile(phi, 100 - self.alpha)

        vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

        # Ensure the vector corresponding to hematoxylin is first, eosin second.
        if vMin[0] > vMax[0]:
            HE = np.array((vMin[:, 0], vMax[:, 0])).T
        else:
            HE = np.array((vMax[:, 0], vMin[:, 0])).T

        # Rows correspond to channels (RGB), columns to OD values.
        Y = np.reshape(OD, (-1, 3)).T

        # Determine concentrations of the individual stains.
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]

        # Normalize stain concentrations.
        maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])

        return HE, maxC, C

    def transform(self, img: np.ndarray) -> np.ndarray:
        """Normalize an H&E image.

        Args:
            img (np.ndarray): Image, RGB uint8 with dimensions W, H, C.

        Returns:
            np.ndarray: Normalized image.
        """

        h, w, c = img.shape
        HERef = self.stain_matrix_target
        maxCRef = self.target_concentrations

        # Get stain matrix and concentrations from image.
        HE, maxC, C = self.matrix_and_concentrations(img)

        tmp = np.divide(maxC, maxCRef)
        C2 = np.divide(C, tmp[:, np.newaxis])

        # Recreate the image using reference mixing matrix.
        Inorm = np.multiply(255, np.exp(-HERef.dot(C2)))
        Inorm = np.clip(Inorm, 0, 255)
        Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

        return Inorm
