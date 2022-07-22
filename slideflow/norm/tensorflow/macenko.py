import numpy as np
import tensorflow as tf
from functools import partial
from typing import Tuple, Dict, Optional, Union

from tensorflow.experimental.numpy import dot
import tensorflow_probability as tfp

from slideflow.norm import utils as ut


@tf.function
def standardize_brightness(I: tf.Tensor) -> tf.Tensor:
    """Standardize image brightness to the 90th percentile.

    Args:
        I (tf.Tensor): Image, uint8.

    Returns:
        tf.Tensor: Brightness-standardized image (uint8)
    """
    p = tfp.stats.percentile(I, 90)  # p = np.percentile(I, 90)
    p = tf.cast(p, tf.float32)
    scaled = tf.cast(I, tf.float32) * tf.constant(255.0, dtype=tf.float32) / p
    scaled = tf.experimental.numpy.clip(scaled, 0, 255)
    return tf.cast(scaled, tf.uint8)


@tf.function
def matrix_and_concentrations(
    img: tf.Tensor,
    Io: int = 255,
    alpha: float = 1,
    beta: float = 0.15
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Gets the H&E stain matrix and concentrations for a given image.

    Args:
        img (tf.Tensor): Image (RGB uint8) with dimensions W, H, C.
        Io (int, optional). Light transmission. Defaults to 255.
        alpha (float): Percentile of angular coordinates to be selected
            with respect to orthogonal eigenvectors. Defaults to 1.
        beta (float): Luminosity threshold. Pixels with luminance above
            this threshold will be ignored. Defaults to 0.15.

    Returns:
        A tuple containing

            tf.Tensor: H&E stain matrix, shape = (3, 2)

            tf.Tensor: Concentrations, shape = (2,)

            tf.Tensor: Concentrations of individual stains
    """

    # reshape image
    img = tf.reshape(img, (-1, 3))

    img = standardize_brightness(img)

    # calculate optical density
    OD = -tf.math.log((tf.cast(img, tf.float32) + 1) / Io)

    # remove transparent pixels
    ODhat = OD[~ tf.math.reduce_any(OD < beta, axis=1)]

    # compute eigenvectors
    eigvals, eigvecs = tf.linalg.eigh(tfp.stats.covariance(ODhat))

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = dot(ODhat, eigvecs[:, 1:3])

    phi = tf.math.atan2(That[:, 1],That[:,0])

    minPhi = tfp.stats.percentile(phi, alpha)
    maxPhi = tfp.stats.percentile(phi, 100-alpha)

    vMin = dot(eigvecs[:, 1:3], tf.transpose(tf.stack((tf.math.cos(minPhi), tf.math.sin(minPhi)))[tf.newaxis, :]))
    vMax = dot(eigvecs[:, 1:3], tf.transpose(tf.stack((tf.math.cos(maxPhi), tf.math.sin(maxPhi)))[tf.newaxis, :]))

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = tf.transpose(tf.stack((vMin[:, 0], vMax[:, 0])))
    else:
        HE = tf.transpose(tf.stack((vMax[:, 0], vMin[:, 0])))

    # rows correspond to channels (RGB), columns to OD values
    Y = tf.transpose(tf.reshape(OD, (-1, 3)))

    # determine concentrations of the individual stains
    C = tf.linalg.lstsq(HE, Y)

    # normalize stain concentrations
    maxC = tf.stack([tfp.stats.percentile(C[0, :], 99), tfp.stats.percentile(C[1, :], 99)])

    return HE, maxC, C


@tf.function
def transform(
    img: tf.Tensor,
    stain_matrix_target: tf.Tensor,
    target_concentrations: tf.Tensor,
    Io: int = 255,
    alpha: float = 1,
    beta: float = 0.15
) -> tf.Tensor:
    """Normalize an image.

    Args:
        img (tf.Tensor): Image to transform.
        stain_matrix_target (tf.Tensor): Target stain matrix.
        target_concentrations (tf.Tensor): Target concentrations.
        Io (int, optional). Light transmission. Defaults to 255.
        alpha (float): Percentile of angular coordinates to be selected
            with respect to orthogonal eigenvectors. Defaults to 1.
        beta (float): Luminosity threshold. Pixels with luminance above
            this threshold will be ignored. Defaults to 0.15.

    Returns:
        tf.Tensor: Transformed image.
    """

    h, w, c = img.shape

    Io = tf.cast(Io, tf.float32)

    HERef = stain_matrix_target
    maxCRef = target_concentrations

    HE, maxC, C = matrix_and_concentrations(img, Io, alpha, beta)

    tmp = tf.math.divide(maxC, maxCRef)
    C2 = tf.math.divide(C, tmp[:, tf.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = tf.math.multiply(Io, tf.math.exp(dot(-HERef, C2)))
    Inorm = tf.experimental.numpy.clip(Inorm, 0, 255)
    Inorm = tf.cast(tf.reshape(tf.transpose(Inorm), (h, w, 3)), tf.uint8)

    # unmix hematoxylin and eosin
    H = tf.math.multiply(Io, tf.math.exp(dot(tf.expand_dims(-HERef[:, 0], axis=1), tf.expand_dims(C2[0, :], axis=0))))
    H = tf.experimental.numpy.clip(H, 0, 255)
    H = tf.cast(tf.reshape(tf.transpose(H), (h, w, 3)), tf.uint8)

    E = tf.math.multiply(Io, tf.math.exp(dot(tf.expand_dims(-HERef[:, 1], axis=1), tf.expand_dims(C2[1, :], axis=0))))
    E = tf.experimental.numpy.clip(E, 0, 255)
    E = tf.cast(tf.reshape(tf.transpose(E), (h, w, 3)), tf.uint8)

    return Inorm


@tf.function
def fit(
    img: tf.Tensor,
    Io: int = 255,
    alpha: float = 1,
    beta: float = 0.15,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Fit a target image.

    Args:
        img (torch.Tensor): Image to fit.
        Io (int, optional). Light transmission. Defaults to 255.
        alpha (float): Percentile of angular coordinates to be selected
            with respect to orthogonal eigenvectors. Defaults to 1.
        beta (float): Luminosity threshold. Pixels with luminance above
            this threshold will be ignored. Defaults to 0.15.

    Returns:
        A tuple containing

            tf.Tensor: Fit stain matrix target

            tf.Tensor: Fit target concentrations
    """
    HE, maxC, _ = matrix_and_concentrations(img, Io, alpha, beta)
    return HE, maxC


class MacenkoNormalizer:

    vectorized = False
    preferred_device = 'cpu'

    def __init__(
        self,
        Io: int = 255,
        alpha: float = 1,
        beta: float = 0.15
    ) -> None:
        """Macenko H&E stain normalizer (Tensorflow implementation).

        Normalizes an image as defined by:

        Macenko, Marc, et al. "A method for normalizing histology
        slides for quantitative analysis." 2009 IEEE International
        Symposium on Biomedical Imaging: From Nano to Macro. IEEE, 2009.

        Args:
            Io (int, optional). Light transmission. Defaults to 255.
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

        self.set_fit(**ut.fit_presets['macenko']['v1'])  # type: ignore

    def fit(
        self,
        target: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Fit normalizer to a target image.

        Calculates the stain matrix and concentrations for the given image,
        and sets these values as the normalizer target.

        Args:
            img (tf.Tensor): Target image (RGB uint8) with dimensions W, H, C.

        Returns:
            A tuple containing

                tf.Tensor:     Stain matrix target.

                tf.Tensor:     Target concentrations.
        """
        if len(target.shape) != 3:
            raise ValueError(
                f"Invalid shape for fit(): expected 3, got {target.shape}"
            )
        HE, maxC = fit(target, self.Io, self.alpha, self.beta)
        self.stain_matrix_target = HE
        self.target_concentrations = maxC
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

    def get_fit(self) -> Dict[str, Optional[np.ndarray]]:
        """Get the current normalizer fit.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping 'stain_matrix_target'
            and 'target_concentrations' to their respective fit values.
        """
        return {
            'stain_matrix_target': None if self.stain_matrix_target is None else self.stain_matrix_target.numpy(),  # type: ignore
            'target_concentrations': None if self.target_concentrations is None else self.target_concentrations.numpy()  # type: ignore
        }

    def set_fit(
        self,
        stain_matrix_target: Union[np.ndarray, tf.Tensor],
        target_concentrations: Union[np.ndarray, tf.Tensor]
    ) -> None:
        """Set the normalizer fit to the given values.

        Args:
            stain_matrix_target (np.ndarray, tf.Tensor): Stain matrix target.
                Must have the shape (3, 2).
            target_concentrations (np.ndarray, tf.Tensor): Target
                concentrations. Must have the shape (2,).
        """
        if not isinstance(stain_matrix_target, tf.Tensor):
            stain_matrix_target = tf.convert_to_tensor(ut._as_numpy(stain_matrix_target))
        if not isinstance(target_concentrations, tf.Tensor):
            target_concentrations = tf.convert_to_tensor(ut._as_numpy(target_concentrations))
        self.stain_matrix_target = stain_matrix_target
        self.target_concentrations = target_concentrations

    @tf.function
    def transform(self, I: tf.Tensor) -> tf.Tensor:
        """Normalize an H&E image.

        Args:
            img (tf.Tensor): Image, RGB uint8 with dimensions W, H, C.

        Returns:
            tf.Tensor: Normalized image.
        """
        if len(I.shape) != 3:
            raise ValueError(
                f"Invalid shape for transform(): expected 3, got {I.shape}"
            )
        return transform(I, self.stain_matrix_target, self.target_concentrations)