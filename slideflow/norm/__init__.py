"""H&E stain normalization and augmentation tools."""

from __future__ import absolute_import

import os
import sys
import multiprocessing as mp
from io import BytesIO
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import slideflow as sf
from PIL import Image
from contextlib import contextmanager
from rich.progress import Progress
from slideflow import errors
from slideflow.dataset import Dataset
from slideflow.util import detuple, log, cleanup_progress
from slideflow.norm import (augment, macenko, reinhard, vahadane)

if TYPE_CHECKING:
    import tensorflow as tf
    import torch


class StainNormalizer:

    vectorized = False
    normalizers = {
        'macenko':  macenko.MacenkoNormalizer,
        'macenko_fast':  macenko.MacenkoFastNormalizer,
        'reinhard': reinhard.ReinhardNormalizer,
        'reinhard_fast': reinhard.ReinhardFastNormalizer,
        'reinhard_mask': reinhard.ReinhardMaskNormalizer,
        'reinhard_fast_mask': reinhard.ReinhardFastMaskNormalizer,
        'vahadane': vahadane.VahadaneSpamsNormalizer,
        'vahadane_sklearn': vahadane.VahadaneSklearnNormalizer,
        'vahadane_spams': vahadane.VahadaneSpamsNormalizer,
        'augment': augment.AugmentNormalizer
    }  # type: Dict[str, Any]

    def __init__(self, method: str, **kwargs) -> None:
        """H&E Stain normalizer supporting various normalization methods.

        The stain normalizer supports numpy images, PNG or JPG strings,
        Tensorflow tensors, and PyTorch tensors. The default ``.transform()``
        method will attempt to preserve the original image type while
        minimizing conversions to and from Tensors.

        Alternatively, you can manually specify the image conversion type
        by using the appropriate function. For example, to convert a Tensor
        to a normalized numpy RGB image, use ``.tf_to_rgb()``.

        Args:
            method (str): Normalization method. Options include 'macenko',
                'reinhard', 'reinhard_fast', 'reinhard_mask',
                'reinhard_fast_mask', 'vahadane', 'vahadane_spams',
                'vahadane_sklearn', and 'augment'.

        Keyword args:
            stain_matrix_target (np.ndarray, optional): Set the stain matrix
                target for the normalizer. May raise an error if the normalizer
                does not have a stain_matrix_target fit attribute.
            target_concentrations (np.ndarray, optional): Set the target
                concentrations for the normalizer. May raise an error if the
                normalizer does not have a target_concentrations fit attribute.
            target_means (np.ndarray, optional): Set the target means for the
                normalizer. May raise an error if the normalizer does not have
                a target_means fit attribute.
            target_stds (np.ndarray, optional): Set the target standard
                deviations for the normalizer. May raise an error if the
                normalizer does not have a target_stds fit attribute.

        Raises:
            ValueError: If the specified normalizer method is not available.

        Examples
            Normalize a numpy image using the default fit.

                >>> import slideflow as sf
                >>> macenko = sf.norm.StainNormalizer('macenko')
                >>> macenko.transform(image)

            Fit the normalizer to a target image (numpy or path).

                >>> macenko.fit(target_image)

            Fit the normalizer using a preset configuration.

                >>> macenko.fit('v2')

            Fit the normalizer to all images in a Dataset.

                >>> dataset = sf.Dataset(...)
                >>> macenko.fit(dataset)

            Normalize an image and convert from Tensor to numpy array (RGB).

                >>> macenko.tf_to_rgb(image)

            Normalize images during DataLoader pre-processing.

                >>> dataset = sf.Dataset(...)
                >>> dataloader = dataset.torch(..., normalizer=macenko)
                >>> dts = dataset.tensorflow(..., normalizer=macenko)

        """
        if method not in self.normalizers:
            raise ValueError(f"Unrecognized normalizer method {method}")

        self.method = method
        self.n = self.normalizers[method]()

        if kwargs:
            self.n.fit(**kwargs)

    def __repr__(self):
        base = "{}(\n".format(self.__class__.__name__)
        base += "  method = {!r},\n".format(self.method)
        for fit_param, fit_val in self.get_fit().items():
            base += "  {} = {!r},\n".format(fit_param, fit_val)
        base += ")"
        return base

    @property
    def device(self) -> str:
        return 'cpu'

    def _torch_transform(
        self,
        inp: "torch.Tensor",
        *,
        augment: bool = False
    ) -> "torch.Tensor":
        """Normalize a torch uint8 image (CWH).

        Normalization ocurs via intermediate conversion to WHC.

        Args:
            inp (torch.Tensor): Image, uint8. Images are normalized in
                W x H x C space. Images provided as C x W x H will be
                auto-converted and permuted back after normalization.

        Returns:
            torch.Tensor:   Image, uint8.

        """
        import torch
        from slideflow.io.torch import cwh_to_whc, whc_to_cwh, is_cwh

        if len(inp.shape) == 4:
            return torch.stack([self._torch_transform(img) for img in inp])
        elif is_cwh(inp):
            # Convert from CWH -> WHC (normalize) -> CWH
            return whc_to_cwh(
                torch.from_numpy(
                    self.rgb_to_rgb(
                        cwh_to_whc(inp).cpu().numpy(),
                        augment=augment
                    )
                )
            )
        else:
            return torch.from_numpy(self.rgb_to_rgb(inp.cpu().numpy()))

    def fit(
        self,
        arg1: Optional[Union[Dataset, np.ndarray, str]],
        batch_size: int = 64,
        num_threads: Union[str, int] = 'auto',
        **kwargs,
    ) -> "StainNormalizer":
        """Fit the normalizer to a target image or dataset of images.

        Args:
            arg1: (Dataset, np.ndarray, str): Target to fit. May be a str,
                numpy image array (uint8), path to an image, or a Slideflow
                Dataset. If this is a string, will fit to the corresponding
                preset fit (either 'v1', 'v2', or 'v3').
                If a Dataset is provided, will average fit values across
                all images in the dataset.
            batch_size (int, optional): Batch size during fitting, if fitting
                to dataset. Defaults to 64.
            num_threads (Union[str, int], optional): Number of threads to use
                during fitting, if fitting to a dataset. Defaults to 'auto'.
        """

        # Fit to a dataset
        if isinstance(arg1, Dataset):
            # Set up thread pool
            if num_threads == 'auto':
                num_threads = sf.util.num_cpu(default=8)  # type: ignore
            log.debug(f"Setting up pool (size={num_threads}) for norm fitting")
            log.debug(f"Using normalizer batch size of {batch_size}")
            pool = mp.dummy.Pool(num_threads)  # type: ignore

            dataset = arg1
            if sf.backend() == 'tensorflow':
                dts = dataset.tensorflow(
                    None,
                    batch_size,
                    standardize=False,
                    infinite=False
                )
            elif sf.backend() == 'torch':
                dts = dataset.torch(
                    None,
                    batch_size,
                    standardize=False,
                    infinite=False,
                    num_workers=8
                )
            all_fit_vals = []  # type: ignore
            pb = Progress(transient=True)
            task = pb.add_task('Fitting normalizer...', total=dataset.num_tiles)
            pb.start()
            with cleanup_progress(pb):
                for img_batch, slide in dts:
                    if sf.model.is_torch_tensor(img_batch):
                        img_batch = img_batch.permute(0, 2, 3, 1)  # BCWH -> BWHC

                    mapped = pool.imap(lambda x: self.n.fit(x.numpy()), img_batch)
                    for fit_vals in mapped:
                        if all_fit_vals == []:
                            all_fit_vals = [[] for _ in range(len(fit_vals))]
                        for v, val in enumerate(fit_vals):
                            all_fit_vals[v] += [np.squeeze(val)]
                    pb.advance(task, batch_size)
            self.n.set_fit(*[np.array(v).mean(axis=0) for v in all_fit_vals])
            pool.close()

        # Fit to numpy image
        elif isinstance(arg1, np.ndarray):
            self.n.fit(arg1, **kwargs)

        # Fit to a preset
        elif (isinstance(arg1, str)
              and arg1 in sf.norm.utils.fit_presets[self.n.preset_tag]):
            self.n.fit_preset(arg1, **kwargs)

        # Fit to a path to an image
        elif isinstance(arg1, str):
            self.src_img = cv2.cvtColor(cv2.imread(arg1), cv2.COLOR_BGR2RGB)
            self.n.fit(self.src_img, **kwargs)

        elif arg1 is None and kwargs:
            self.set_fit(**kwargs)

        else:
            raise ValueError(f'Unrecognized args for fit: {arg1}')

        log.debug('Fit normalizer: {}'.format(
            ', '.join([f"{fit_key} = {fit_val}"
            for fit_key, fit_val in self.get_fit().items()])
        ))
        return self

    def get_fit(self, as_list: bool = False):
        """Get the current normalizer fit.

        Args:
            as_list (bool). Convert the fit values (numpy arrays) to list
                format. Defaults to False.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping fit parameters (e.g.
            'target_concentrations') to their respective fit values.
        """
        _fit = self.n.get_fit()
        if as_list:
            return {k: v.tolist() for k, v in _fit.items()}
        else:
            return _fit

    def set_fit(self, **kwargs) -> None:
        """Set the normalizer fit to the given values.

        Keyword args:
            stain_matrix_target (np.ndarray, optional): Set the stain matrix
                target for the normalizer. May raise an error if the normalizer
                does not have a stain_matrix_target fit attribute.
            target_concentrations (np.ndarray, optional): Set the target
                concentrations for the normalizer. May raise an error if the
                normalizer does not have a target_concentrations fit attribute.
            target_means (np.ndarray, optional): Set the target means for the
                normalizer. May raise an error if the normalizer does not have
                a target_means fit attribute.
            target_stds (np.ndarray, optional): Set the target standard
                deviations for the normalizer. May raise an error if the
                normalizer does not have a target_stds fit attribute.
        """
        self.n.set_fit(**{k:v for k, v in kwargs.items() if v is not None})

    def set_augment(self, preset: Optional[str] = None, **kwargs) -> None:
        """Set the normalizer augmentation space.

        Args:
            preset (str, optional): Augmentation preset. Defaults to None.

        Keyword args:
            matrix_stdev (np.ndarray): Standard deviation
                of the stain matrix target. Must have the shape (3, 2).
                Used for Macenko normalizers.
                Defaults to None (will not augment stain matrix).
            concentrations_stdev (np.ndarray): Standard deviation
                of the target concentrations. Must have the shape (2,).
                Used for Macenko normalizers.
                Defaults to None (will not augment target concentrations).
            means_stdev (np.ndarray): Standard deviation
                of the target means. Must have the shape (3,).
                Used for Reinhard normalizers.
                Defaults to None (will not augment target means).
            stds_stdev (np.ndarray): Standard deviation
                of the target stds. Must have the shape (3,).
                Used for Reinhard normalizers.
                Defaults to None (will not augment target stds).

        """
        if preset is not None:
            return self.n.augment_preset(preset)
        if kwargs:
            self.n.set_augment(**{k:v for k, v in kwargs.items() if v is not None})

    def transform(
        self,
        image: Union[str, bytes, np.ndarray, "tf.Tensor", "torch.Tensor"],
        *,
        augment: bool = False
    ) -> Union[str, bytes, np.ndarray, "tf.Tensor", "torch.Tensor"]:
        """Normalize a target image, attempting to preserve the original type.

        Args:
            image (np.ndarray, tf.Tensor, or torch.Tensor): Image.

        Keyword args:
            augment (bool): Transform using stain aumentation.
                Defaults to False.

        Returns:
            Normalized image of the original type.
        """
        if isinstance(image, (str, bytes)):
            raise ValueError("Unable to auto-transform bytes or str; please "
                             "use .png_to_png() or .jpeg_to_jpeg().")
        if 'tensorflow' in sys.modules:
            import tensorflow as tf
            if isinstance(image, tf.Tensor):
                return self.tf_to_tf(image, augment=augment)
        if 'torch' in sys.modules:
            import torch
            if isinstance(image, torch.Tensor):
                return self.torch_to_torch(image, augment=augment)
        if isinstance(image, np.ndarray):
            return self.rgb_to_rgb(image, augment=augment)
        raise ValueError(f"Unrecognized image type {type(image)}; expected "
                         "np.ndarray, tf.Tensor, or torch.Tensor")

    def jpeg_to_jpeg(
        self,
        jpeg_string: Union[str, bytes],
        *,
        quality: int = 100,
        augment: bool = False
    ) -> bytes:
        """Normalize a JPEG image, returning a JPEG image.

        Args:
            jpeg_string (str, bytes): JPEG image data.

        Keyword args:
            augment (bool): Transform using stain aumentation.
                Defaults to False.
            quality (int, optional): Quality level for creating the resulting
                normalized JPEG image. Defaults to 100.

        Returns:
            bytes:  Normalized JPEG image.
        """
        cv_image = self.jpeg_to_rgb(jpeg_string, augment=augment)
        with BytesIO() as output:
            Image.fromarray(cv_image).save(
                output,
                format="JPEG",
                quality=quality
            )
            return output.getvalue()

    def jpeg_to_rgb(
        self,
        jpeg_string: Union[str, bytes],
        *,
        augment: bool = False
    ) -> np.ndarray:
        """Normalize a JPEG image, returning a numpy uint8 array.

        Args:
            jpeg_string (str, bytes): JPEG image data.

        Keyword args:
            augment (bool): Transform using stain aumentation.
                Defaults to False.

        Returns:
            np.ndarray: Normalized image, uint8, W x H x C.
        """
        cv_image = cv2.imdecode(
            np.fromstring(jpeg_string, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return self.rgb_to_rgb(cv_image, augment=augment)

    def png_to_png(
        self,
        png_string: Union[str, bytes],
        *,
        augment: bool = False
    ) -> bytes:
        """Normalize a PNG image, returning a PNG image.

        Args:
            png_string (str, bytes): PNG image data.

        Keyword args:
            augment (bool): Transform using stain aumentation.
                Defaults to False.

        Returns:
            bytes: Normalized PNG image.
        """
        cv_image = self.png_to_rgb(png_string, augment=augment)
        with BytesIO() as output:
            Image.fromarray(cv_image).save(output, format="PNG")
            return output.getvalue()

    def png_to_rgb(
        self,
        png_string: Union[str, bytes],
        *,
        augment: bool = False
    ) -> np.ndarray:
        """Normalize a PNG image, returning a numpy uint8 array.

        Args:
            png_string (str, bytes): PNG image data.

        Keyword args:
            augment (bool): Transform using stain aumentation.
                Defaults to False.

        Returns:
            np.ndarray: Normalized image, uint8, W x H x C.
        """
        return self.jpeg_to_rgb(png_string, augment=augment)  # It should auto-detect format

    def rgb_to_rgb(
        self,
        image: np.ndarray,
        *,
        augment: bool = False
    ) -> np.ndarray:
        """Normalize a numpy array (uint8), returning a numpy array (uint8).

        Args:
            image (np.ndarray): Image (uint8).

        Keyword args:
            augment (bool): Transform using stain aumentation.
                Defaults to False.

        Returns:
            np.ndarray: Normalized image, uint8, W x H x C.
        """
        return self.n.transform(image, augment=augment)

    def tf_to_rgb(
        self,
        image: "tf.Tensor",
        *,
        augment: bool = False
    ) -> np.ndarray:
        """Normalize a tf.Tensor (uint8), returning a numpy array (uint8).

        Args:
            image (tf.Tensor): Image (uint8).

        Keyword args:
            augment (bool): Transform using stain aumentation.
                Defaults to False.

        Returns:
            np.ndarray: Normalized image, uint8, W x H x C.
        """
        return self.rgb_to_rgb(np.array(image), augment=augment)

    def tf_to_tf(
        self,
        image: Union[Dict, "tf.Tensor"],
        *args: Any,
        augment: bool = False
    ) -> Tuple[Union[Dict, "tf.Tensor"], ...]:
        """Normalize a tf.Tensor (uint8), returning a numpy array (uint8).

        Args:
            image (tf.Tensor, Dict): Image (uint8) either as a raw Tensor,
                or a Dictionary with the image under the key 'tile_image'.
            args (Any, optional): Any additional arguments, which will be passed
                and returned unmodified.

        Keyword args:
            augment (bool): Transform using stain aumentation.
                Defaults to False.

        Returns:
            A tuple containing the normalized tf.Tensor image (uint8,
            W x H x C) and any additional arguments provided.
        """
        import tensorflow as tf

        if isinstance(image, dict):
            image['tile_image'] = tf.py_function(
                partial(self.tf_to_rgb, augment=augment),
                [image['tile_image']],
                tf.uint8
            )
        elif len(image.shape) == 4:
            image = tf.stack([self.tf_to_tf(_i, augment=augment) for _i in image])
        else:
            image = tf.py_function(
                partial(self.tf_to_rgb, augment=augment),
                [image],
                tf.uint8
            )
        return detuple(image, args)

    def torch_to_torch(
        self,
        image: Union[Dict, "torch.Tensor"],
        *args,
        augment: bool = False
    ) -> Tuple[Union[Dict, "torch.Tensor"], ...]:
        """Normalize a torch.Tensor (uint8), returning a numpy array (uint8).

        Args:
            image (torch.Tensor, Dict): Image (uint8) either as a raw Tensor,
                or a Dictionary with the image under the key 'tile_image'.
            args (Any, optional): Any additional arguments, which will be passed
                and returned unmodified.

        Keyword args:
            augment (bool): Transform using stain aumentation.
                Defaults to False.

        Returns:
            A tuple containing

                np.ndarray: Normalized tf.Tensor image, uint8, C x W x H.

                args (Any, optional): Any additional arguments provided, unmodified.
        """
        if isinstance(image, dict):
            to_return = {
                k: v for k, v in image.items()
                if k != 'tile_image'
            }
            to_return['tile_image'] = self._torch_transform(
                image['tile_image'],
                augment=augment
            )
            return detuple(to_return, args)
        else:
            return detuple(self._torch_transform(image, augment=augment), args)

    # --- Context management --------------------------------------------------

    @contextmanager
    def context(
        self,
        context: Union[str, "sf.WSI", np.ndarray, "tf.Tensor", "torch.Tensor"]
    ):
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

            with normalizer.context(slide):
                normalizer.transform(target)

        If a slide (``sf.WSI``) is used for context, any existing QC filters
        and regions of interest will be used to mask out background as white
        pixels, and the masked thumbnail will be used for creating the
        normalizer context. If no QC has been applied to the slide and the
        slide does not have any Regions of Interest, then both otsu's
        thresholding and Gaussian blur filtering will be applied
        to the thumbnail for masking.

        Args:
            I (np.ndarray, sf.WSI): Context to use for normalization, e.g.
                a whole-slide image thumbnail, optionally masked with masked
                areas set to (255, 255, 255).

        """
        self.set_context(context)
        yield
        self.clear_context()

    def set_context(
        self,
        context: Union[str, "sf.WSI", np.ndarray, "tf.Tensor", "torch.Tensor"]
    ) -> bool:
        """Set the whole-slide context for the stain normalizer.

        With contextual normalization, max concentrations are determined
        from the context (whole-slide image) rather than the image being
        normalized. This may improve stain normalization for sections of
        a slide that are predominantly eosin (e.g. necrosis or low cellularity).

        When calculating max concentrations from the image context,
        white pixels (255) will be masked.

        If a slide (``sf.WSI``) is used for context, any existing QC filters
        and regions of interest will be used to mask out background as white
        pixels, and the masked thumbnail will be used for creating the
        normalizer context. If no QC has been applied to the slide and the
        slide does not have any Regions of Interest, then both otsu's
        thresholding and Gaussian blur filtering will be applied
        to the thumbnail for masking.

        Args:
            I (np.ndarray, sf.WSI): Context to use for normalization, e.g.
                a whole-slide image thumbnail, optionally masked with masked
                areas set to (255, 255, 255).

        """
        if hasattr(self.n, 'set_context'):
            if isinstance(context, str):
                image = np.asarray(sf.WSI(context, 500, 500).thumb(mpp=4))
            elif isinstance(context, sf.WSI):
                image = context.masked_thumb(mpp=4, background='white')
            else:
                image = context  # type: ignore
            self.n.set_context(image)
            return True
        else:
            return False

    def clear_context(self) -> None:
        """Remove any previously set stain normalizer context."""
        if hasattr(self.n, 'clear_context'):
            self.n.clear_context()


def autoselect(
    method: str,
    source: Optional[str] = None,
    backend: Optional[str] = None,
    **kwargs
) -> StainNormalizer:
    """Select the best normalizer for a given method, and fit to a given source.

    If a normalizer method has a native implementation in the current backend
    (Tensorflow or PyTorch), the native normalizer will be used.
    If not, the default numpy implementation will be used.

    Currently, the PyTorch-native normalizers are NOT used by default, as they
    are slower than the numpy implementations. Thus, with the PyTorch backend,
    all normalizers will be the default numpy implementations.

    Args:
        method (str): Normalization method. Options include 'macenko',
            'reinhard', 'reinhard_fast', 'reinhard_mask', 'reinhard_fast_mask',
            'vahadane', 'vahadane_spams', 'vahadane_sklearn', and 'augment'.
        source (str, optional): Stain normalization preset or path to a source
            image. Valid presets include 'v1', 'v2', and 'v3'. If None, will
            use the default present ('v3'). Defaults to None.
        backend (str): Backend to use for native normalizers. Options include
            'tensorflow', 'torch', and 'opencv'. If None, will use the current
            backend, falling back to opencv/numpy if a native normalizer is
            not available. Defaults to None.

    Returns:
        StainNormalizer:    Initialized StainNormalizer.
    """
    if backend is None:
        backend = sf.backend()
    if backend == 'tensorflow':
        import slideflow.norm.tensorflow
        BackendNormalizer = sf.norm.tensorflow.TensorflowStainNormalizer
    elif backend == 'torch':
        import slideflow.norm.torch
        BackendNormalizer = sf.norm.torch.TorchStainNormalizer  # type: ignore
    elif backend == 'opencv':
        BackendNormalizer = StainNormalizer
    else:
        raise errors.UnrecognizedBackendError

    if method in BackendNormalizer.normalizers:
        normalizer = BackendNormalizer(method, **kwargs)
    else:
        normalizer = StainNormalizer(method, **kwargs)  # type: ignore

    if source is not None and source != 'dataset':
        normalizer.fit(source)

    return normalizer