from typing import Dict, Optional, Tuple, Union

import torch
import numpy as np
import torchvision
from rich.progress import Progress

from slideflow.dataset import Dataset
from slideflow.io.torch import cwh_to_whc, whc_to_cwh
from slideflow.norm import StainNormalizer
from slideflow.norm.torch import reinhard, macenko
from slideflow.util import detuple, log
from slideflow import errors


class TorchStainNormalizer(StainNormalizer):

    # Torch-specific vectorized normalizers disabled
    # as they are slower than the CV implementation

    normalizers = {
        'reinhard': reinhard.ReinhardNormalizer,
        'reinhard_fast': reinhard.ReinhardFastNormalizer,
        'macenko': macenko.MacenkoNormalizer
    }  # type: Dict

    def __init__(
        self,
        method: str,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> None:
        """PyTorch-native H&E Stain normalizer.

        The stain normalizer supports numpy images, PNG or JPG strings,
        Tensorflow tensors, and PyTorch tensors. The default `.transform()`
        method will attempt to preserve the original image type while minimizing
        conversions to and from Tensors.

        Alternatively, you can manually specify the image conversion type
        by using the appropriate function. For example, to convert a JPEG
        image to a normalized numpy RGB image, use `.jpeg_to_rgb()`.

        Args:
            method (str): Normalization method to use.

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
            Please see :class:`slideflow.norm.StainNormalizer` for examples.
        """

        super().__init__(method, **kwargs)
        self._device = device

    @property
    def vectorized(self) -> bool:  # type: ignore
        """Returns whether the associated normalizer is vectorized.

        Returns:
            bool: Normalizer is vectorized.
        """
        return self.n.vectorized

    def fit(
        self,
        arg1: Optional[Union[Dataset, np.ndarray, str]],
        batch_size: int = 64,
        num_threads: Union[str, int] = 'auto',
        **kwargs
    ):
        """Fit the normalizer to a target image or dataset of images.

        Args:
            arg1: (Dataset, np.ndarray, str): Target to fit. May be a numpy
                image array (uint8), path to an image, or a Slideflow Dataset.
                If a Dataset is provided, will average fit values across
                all images in the dataset.
            batch_size (int, optional): Batch size during fitting, if fitting
                to dataset. Defaults to 64.
        """
        if isinstance(arg1, Dataset):
            # Prime the normalizer
            dataset = arg1
            dts = dataset.tensorflow(
                None,
                batch_size,
                standardize=False,
                infinite=False
            )
            all_fit_vals = []  # type: ignore
            pb = Progress(transient=True)
            task = pb.add_task('Fitting normalizer...', total=dataset.num_tiles)
            pb.start()
            for i, slide in dts:
                if self.vectorized:
                    fit_vals = self.n.fit(i, reduce=True)
                else:
                    _img_fits = zip(*[self.n.fit(_i) for _i in i])
                    fit_vals = [torch.mean(torch.stack(v), dim=0) for v in _img_fits]
                if all_fit_vals == []:
                    all_fit_vals = [[] for _ in range(len(fit_vals))]
                for v, val in enumerate(fit_vals):
                    all_fit_vals[v] += [val]
                pb.advance(task, batch_size)
            pb.stop()
            self.n.set_fit(*[torch.mean(torch.stack(v), dim=0) for v in all_fit_vals])

        elif isinstance(arg1, np.ndarray):
            self.n.fit(torch.from_numpy(arg1))

        # Fit to a preset
        elif isinstance(arg1, str) and arg1 in ('v1', 'v2'):
            self.n.fit_preset(arg1)

        elif isinstance(arg1, str):
            self.src_img = torchvision.io.read_image(arg1)
            self.n.fit(self.src_img)

        elif arg1 is None and kwargs:
            self.set_fit(**kwargs)

        else:
            raise errors.NormalizerError(f'Unrecognized args for fit: {arg1}')

        log.debug('Fit normalizer: {}'.format(
            ', '.join([f"{fit_key} = {fit_val}"
            for fit_key, fit_val in self.get_fit().items()])
        ))

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

    def torch_to_torch(
        self,
        image: Union[Dict, torch.Tensor],
        *args
    ) -> Tuple[Union[Dict, torch.Tensor], ...]:
        """Normalize a torch.Tensor (uint8), returning a numpy array (uint8).

        Args:
            image (torch.Tensor, Dict): Image (uint8) either as a raw Tensor,
                or a Dictionary with the image under the key 'tile_image'.
            args (Any, optional): Any additional arguments, which will be passed
                and returned unmodified.

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
            to_return['tile_image'] = self.n.transform(image['tile_image'])
            return detuple(to_return, args)
        else:
            return detuple(self.n.transform(image), args)

    def rgb_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """Normalize a numpy array (uint8), returning a numpy array (uint8).

        Args:
            image (np.ndarray): Image (uint8), W x H x C.

        Returns:
            np.ndarray: Normalized image, uint8, W x H x C.
        """
        return self.n.transform(torch.from_numpy(image)).numpy()