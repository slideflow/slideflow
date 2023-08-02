import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.experimental.numpy import dot
from typing import Tuple, Dict, Optional, Union
from contextlib import contextmanager

from slideflow.norm import utils as ut
from .utils import clip_size, standardize_brightness

# -----------------------------------------------------------------------------

@tf.function
def is_self_adjoint(matrix):
    return tf.reduce_all(tf.math.equal(matrix, tf.linalg.adjoint(matrix)))

@tf.function
def normalize_c(C):
    return tf.stack([
        tfp.stats.percentile(C[0, :], 99),
        tfp.stats.percentile(C[1, :], 99)]
    )

@tf.function
def _matrix_and_concentrations(
    img: tf.Tensor,
    Io: int = 255,
    alpha: float = 1,
    beta: float = 0.15,
    mask: bool = False,
    standardize: bool = True
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Gets the H&E stain matrix and concentrations for a given image.

    Args:
        img (tf.Tensor): Image (RGB uint8) with dimensions W, H, C.
        Io (int, optional). Light transmission. Defaults to 255.
        alpha (float): Percentile of angular coordinates to be selected
            with respect to orthogonal eigenvectors. Defaults to 1.
        beta (float): Luminosity threshold. Pixels with luminance above
            this threshold will be ignored. Defaults to 0.15.
        mask (bool): Mask white pixels (255) during calculation.
            Defaults to False.
        standardize (bool): Perform brightness standardization.
            Defaults to True.

    Returns:
        A tuple containing

            tf.Tensor: H&E stain matrix, shape = (3, 2)

            tf.Tensor: Concentrations of individual stains
    """

    # reshape image
    img = tf.reshape(img, (-1, 3))

    if mask:
        ones = tf.math.reduce_all(img == 255, axis=1)

    if standardize:
        img = standardize_brightness(img, mask=mask)

    # calculate optical density
    OD = -tf.math.log((tf.cast(img, tf.float32) + 1) / Io)

    # remove transparent pixels
    if mask:
        ODhat = OD[~ (tf.math.reduce_any(OD < beta, axis=1) | ones)]
    else:
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
    OD = tf.reshape(OD, (-1, 3))

    if mask:
        OD = OD[~ ones]

    Y = tf.transpose(OD)

    # determine concentrations of the individual stains
    C = tf.linalg.lstsq(HE, Y)

    return HE, C


@tf.function
def matrix_and_concentrations(
    img: tf.Tensor,
    Io: int = 255,
    alpha: float = 1,
    beta: float = 0.15,
    mask: bool = False,
    standardize: bool = True
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Gets the H&E stain matrix and concentrations for a given image.

    Args:
        img (tf.Tensor): Image (RGB uint8) with dimensions W, H, C.
        Io (int, optional). Light transmission. Defaults to 255.
        alpha (float): Percentile of angular coordinates to be selected
            with respect to orthogonal eigenvectors. Defaults to 1.
        beta (float): Luminosity threshold. Pixels with luminance above
            this threshold will be ignored. Defaults to 0.15.
        mask (bool): Mask white pixels (255) during calculation.
            Defaults to False.
        standardize (bool): Perform brightness standardization.
            Defaults to True.

    Returns:
        A tuple containing

            tf.Tensor: H&E stain matrix, shape = (3, 2)

            tf.Tensor: Max concentrations, shape = (2,)

            tf.Tensor: Concentrations of individual stains
    """
    HE, C = _matrix_and_concentrations(
        img, Io, alpha, beta, mask=mask, standardize=standardize
    )

    # normalize stain concentrations
    maxC = normalize_c(C)

    return HE, maxC, C


@tf.function
def augmented_transform(
    img: tf.Tensor,
    stain_matrix_target: tf.Tensor,
    target_concentrations: tf.Tensor,
    matrix_stdev: Optional[tf.Tensor] = None,
    concentrations_stdev: Optional[tf.Tensor] = None,
    **kwargs
) -> tf.Tensor:
    """Normalize an image.

    Args:
        img (tf.Tensor): Image to transform.
        stain_matrix_target (tf.Tensor): Target stain matrix.
        target_concentrations (tf.Tensor): Target concentrations.
        matrix_stdev (np.ndarray, tf.Tensor): Standard deviation
            of the stain matrix target. Must have the shape (3, 2).
            Defaults to None (will not augment stain matrix).
        concentrations_stdev (np.ndarray, tf.Tensor): Standard deviation
            of the target concentrations. Must have the shape (2,).
            Defaults to None (will not augment target concentrations).

    Keyword args:
        Io (int, optional). Light transmission. Defaults to 255.
        alpha (float): Percentile of angular coordinates to be selected
            with respect to orthogonal eigenvectors. Defaults to 1.
        beta (float): Luminosity threshold. Pixels with luminance above
            this threshold will be ignored. Defaults to 0.15.

    Returns:
        tf.Tensor: Transformed image.

    """
    if matrix_stdev is None and concentrations_stdev is None:
        raise ValueError("Must supply either matrix_stdev and/or concentrations_stdev")
    if matrix_stdev is not None:
        stain_matrix_target = tf.random.normal([3, 2], mean=stain_matrix_target, stddev=matrix_stdev)
    if concentrations_stdev is not None:
        target_concentrations = tf.random.normal([2], mean=target_concentrations, stddev=concentrations_stdev)
    return transform(img, stain_matrix_target, target_concentrations, **kwargs)


@tf.function
def transform(
    img: tf.Tensor,
    stain_matrix_target: tf.Tensor,
    target_concentrations: tf.Tensor,
    *,
    Io: int = 255,
    alpha: float = 1,
    beta: float = 0.15,
    ctx_maxC: Optional[tf.Tensor] = None,
    standardize: bool = True,
    original_on_error: bool = True
) -> tf.Tensor:
    """Normalize an image.

    Args:
        img (tf.Tensor): Image to transform.
        stain_matrix_target (tf.Tensor): Target stain matrix.
        target_concentrations (tf.Tensor): Target concentrations.

    Keyword args:
        Io (int). Light transmission. Defaults to 255.
        alpha (float): Percentile of angular coordinates to be selected
            with respect to orthogonal eigenvectors. Defaults to 1.
        beta (float): Luminosity threshold. Pixels with luminance above
            this threshold will be ignored. Defaults to 0.15.
        standardize (bool): Perform brightness standardization.
            Defaults to True.
        ctx_maxC (tf.Tensor, optional): Max concentration from context
            (e.g. whole-slide image). If None, calculates max concentration
            from the target image. Defaults to None.

    Returns:
        tf.Tensor: Transformed image.
    """
    original_image = img
    h, w, c = img.shape

    Io = tf.cast(Io, tf.float32)

    HERef = stain_matrix_target
    maxCRef = target_concentrations

    # reshape image
    img = tf.reshape(img, (-1, 3))

    # -------------------------------------------------------------------------

    if standardize:
        img = standardize_brightness(img)

    # calculate optical density
    OD = -tf.math.log((tf.cast(img, tf.float32) + 1) / Io)

    # remove transparent pixels
    ODhat = OD[~ tf.math.reduce_any(OD < beta, axis=1)]

    # compute eigenvectors
    covar = tfp.stats.covariance(ODhat)
    if original_on_error and not is_self_adjoint(covar):
        return original_image
    eigvals, eigvecs = tf.linalg.eigh(covar)

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
    OD = tf.reshape(OD, (-1, 3))

    Y = tf.transpose(OD)

    # determine concentrations of the individual stains
    C = tf.linalg.lstsq(HE, Y)

    # -------------------------------------------------------------------------

    if ctx_maxC is not None:
        maxC = ctx_maxC
    else:
        maxC = normalize_c(C)

    tmp = tf.math.divide(maxC, maxCRef)
    C2 = tf.math.divide(C, tmp[:, tf.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = tf.math.multiply(Io, tf.math.exp(dot(-HERef, C2)))
    Inorm = tf.experimental.numpy.clip(Inorm, 0, 255)
    Inorm = tf.cast(tf.reshape(tf.transpose(Inorm), (h, w, 3)), tf.uint8)

    return Inorm


@tf.function
def fit(
    img: tf.Tensor,
    Io: int = 255,
    alpha: float = 1,
    beta: float = 0.15,
    standardize: bool = True
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Fit a target image.

    Args:
        img (torch.Tensor): Image to fit.
        Io (int, optional). Light transmission. Defaults to 255.
        alpha (float): Percentile of angular coordinates to be selected
            with respect to orthogonal eigenvectors. Defaults to 1.
        beta (float): Luminosity threshold. Pixels with luminance above
            this threshold will be ignored. Defaults to 0.15.
        standardize (bool): Perform brightness standardization.
            Defaults to True.

    Returns:
        A tuple containing

            tf.Tensor: Fit stain matrix target

            tf.Tensor: Fit target concentrations
    """
    HE, maxC, _ = matrix_and_concentrations(
        img, Io, alpha, beta, standardize=standardize
    )
    return HE, maxC


class MacenkoNormalizer:

    vectorized = False
    preferred_device = 'cpu'
    preset_tag = 'macenko'

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
        self._ctx_maxC = None  # type: Optional[tf.Tensor]
        self._augment_params = dict()  # type: Dict[str, tf.Tensor]
        self.set_fit(**ut.fit_presets[self.preset_tag]['v3'])  # type: ignore
        self.set_augment(**ut.augment_presets[self.preset_tag]['v1'])  # type: ignore

    def _fit(self, target: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return fit(target, self.Io, self.alpha, self.beta)

    def _transform(
        self,
        I: tf.Tensor,
        *,
        augment: bool = False,
        original_on_error: bool = True
    ) -> tf.Tensor:
        """Normalize an image."""
        if augment and not any(m in self._augment_params
                               for m in ('matrix_stdev', 'concentrations_stdev')):
            raise ValueError("Augmentation space not configured.")

        fn = augmented_transform if augment else transform
        aug_kw = self._augment_params if augment else {}
        return fn(
            I,
            self.stain_matrix_target,
            self.target_concentrations,
            ctx_maxC=self._ctx_maxC,
            original_on_error=original_on_error,
            **aug_kw
        )

    def fit(self, target: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
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
        target = clip_size(target, 2048)
        HE, maxC = self._fit(target)
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

    def set_augment(
        self,
        matrix_stdev: Optional[Union[np.ndarray, tf.Tensor]] = None,
        concentrations_stdev: Optional[Union[np.ndarray, tf.Tensor]] = None,
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
            self._augment_params['matrix_stdev'] = tf.convert_to_tensor(ut._as_numpy(matrix_stdev))
        if concentrations_stdev is not None:
            self._augment_params['concentrations_stdev'] = tf.convert_to_tensor(ut._as_numpy(concentrations_stdev))

    def transform(self, I: tf.Tensor, **kwargs) -> tf.Tensor:
        """Normalize an H&E image.

        Args:
            img (tf.Tensor): Image, RGB uint8 with dimensions W, H, C.

        Keyword args:
            augment (bool): Perform random stain augmentation.
                Defaults to False.

        Returns:
            tf.Tensor: Normalized image.
        """
        if len(I.shape) == 4:
            return tf.map_fn(self.transform, I)
        elif len(I.shape) == 3:
            return self._transform(I, **kwargs)
        else:
            raise ValueError(
                f"Invalid shape for transform(): expected 3 or 4, got {I.shape}"
            )

    @contextmanager
    def image_context(self, I: Union[np.ndarray, tf.Tensor]):
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
            I (np.ndarray, tf.Tensor): Context to use for normalization, e.g.
                a whole-slide image thumbnail, optionally masked with masked
                areas set to (255, 255, 255).

        """
        self.set_context(I)
        yield
        self.clear_context()

    def set_context(self, I: Union[np.ndarray, tf.Tensor]):
        """Set the whole-slide context for the stain normalizer.

        With contextual normalization, max concentrations are determined
        from the context (whole-slide image) rather than the image being
        normalized. This may improve stain normalization for sections of
        a slide that are predominantly eosin (e.g. necrosis or low cellularity).

        When calculating max concentrations from the image context,
        white pixels (255) will be masked.

        Args:
            I (np.ndarray, tf.Tensor): Context to use for normalization, e.g.
                a whole-slide image thumbnail, optionally masked with masked
                areas set to (255, 255, 255).

        """
        if not isinstance(I, tf.Tensor):
            I = tf.convert_to_tensor(ut._as_numpy(I))

        I = clip_size(I, 2048)
        HE, maxC, C = matrix_and_concentrations(I, mask=True)
        self._ctx_maxC = maxC

    def clear_context(self):
        """Remove any previously set stain normalizer context."""
        self._ctx_maxC = None


class MacenkoFastNormalizer(MacenkoNormalizer):

    """Macenko H&E stain normalizer, without brightness standardization."""

    preset_tag = 'macenko_fast'

    def _fit(self, target: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return fit(
            target, self.Io, self.alpha, self.beta,
            standardize=False
        )

    def _transform(
        self,
        I: tf.Tensor,
        *,
        augment: bool = False,
        original_on_error: bool = True
    ) -> tf.Tensor:
        """Normalize an image."""
        if augment and not any(m in self._augment_params
                               for m in ('matrix_stdev', 'concentrations_stdev')):
            raise ValueError("Augmentation space not configured.")

        fn = augmented_transform if augment else transform
        aug_kw = self._augment_params if augment else {}
        return fn(
            I,
            self.stain_matrix_target,
            self.target_concentrations,
            ctx_maxC=self._ctx_maxC,
            standardize=False,
            original_on_error=original_on_error,
            **aug_kw
        )

    def set_context(self, I: Union[np.ndarray, tf.Tensor]):
        """Set the whole-slide context for the stain normalizer.

        With contextual normalization, max concentrations are determined
        from the context (whole-slide image) rather than the image being
        normalized. This may improve stain normalization for sections of
        a slide that are predominantly eosin (e.g. necrosis or low cellularity).

        When calculating max concentrations from the image context,
        white pixels (255) will be masked.

        Args:
            I (np.ndarray, tf.Tensor): Context to use for normalization, e.g.
                a whole-slide image thumbnail, optionally masked with masked
                areas set to (255, 255, 255).

        """
        if not isinstance(I, tf.Tensor):
            I = tf.convert_to_tensor(ut._as_numpy(I))
        I = clip_size(I, 2048)
        HE, maxC, C = matrix_and_concentrations(
            I, mask=True, standardize=False
        )
        self._ctx_maxC = maxC
