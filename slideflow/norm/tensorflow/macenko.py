import numpy as np
import tensorflow as tf
from functools import partial
from typing import Tuple, Dict, Optional, Union

from tensorflow.experimental.numpy import dot
import tensorflow_probability as tfp


@tf.function
def standardize_brightness(I: tf.Tensor) -> tf.Tensor:
    """

    :param I:
    :return:
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
    """Transform an image using a given target means & stds.

    Args:
        I (tf.Tensor): Image to transform
        tgt_mean (tf.Tensor): Target means.
        tgt_std (tf.Tensor): Target means.

    Raises:
        ValueError: If tgt_mean or tgt_std is None.

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
    reduce: bool = False
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Fit a target image.

    Args:
        target (torch.Tensor): Batch of images to fit.
        reduce (bool, optional): Reduce the fit means/stds across the batch
            of images to a single mean/std array, reduced by average.
            Defaults to False (provides fit for each image in the batch).

    Returns:
        tf.Tensor: Fit stain matrix target
        tf.Tensor: Fit target concentrations
    """
    HE, maxC, _ = matrix_and_concentrations(img, Io, alpha, beta)

    if reduce:
        HE = tf.math.reduce_mean(HE, axis=0)
        maxC = tf.math.reduce_mean(maxC, axis=0)

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

        self.Io = Io
        self.alpha = alpha
        self.beta = beta

        HERef = tf.constant([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]])
        maxCRef = tf.constant([1.9705, 1.0308])
        self.stain_matrix_target = HERef
        self.target_concentrations = maxCRef

    def fit(
        self,
        target: tf.Tensor,
        reduce: bool = True
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if len(target.shape) == 3:
            target = tf.expand_dims(target, axis=0)
        HE, maxC = fit(target, self.Io, self.alpha, self.beta, reduce=reduce)
        self.stain_matrix_target = HE
        self.target_concentrations = maxC
        return HE, maxC

    def get_fit(self) -> Dict[str, Optional[np.ndarray]]:
        return {
            'stain_matrix_target': None if self.stain_matrix_target is None else self.stain_matrix_target.numpy(),  # type: ignore
            'target_concentrations': None if self.target_concentrations is None else self.target_concentrations.numpy()  # type: ignore
        }

    def set_fit(
        self,
        stain_matrix_target: Union[np.ndarray, tf.Tensor],
        target_concentrations: Union[np.ndarray, tf.Tensor]
    ) -> None:
        if not isinstance(stain_matrix_target, tf.Tensor):
            stain_matrix_target = tf.convert_to_tensor(stain_matrix_target)
        if not isinstance(target_concentrations, tf.Tensor):
            target_concentrations = tf.convert_to_tensor(target_concentrations)
        self.stain_matrix_target = stain_matrix_target
        self.target_concentrations = target_concentrations

    @tf.function
    def transform(self, I: tf.Tensor) -> tf.Tensor:
        if len(I.shape) != 3:
            raise ValueError(
                f"Invalid shape for transform(): expected 3, got {I.shape}"
            )
        return transform(I, self.stain_matrix_target, self.target_concentrations)