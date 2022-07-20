from typing import Dict, Optional, Tuple, Union

import torch
import numpy as np
import torchvision
from tqdm import tqdm

from slideflow.dataset import Dataset
from slideflow.norm import StainNormalizer
from slideflow.norm.torch import reinhard, reinhard_fast
from slideflow.util import detuple, log
from slideflow import errors


class TorchStainNormalizer(StainNormalizer):

    # Torch-specific vectorized normalizers disabled
    # as they are slower than the CV implementation
    normalizers = {
        #'reinhard': reinhard.ReinhardNormalizer,
        #'reinhard_fast': reinhard_fast.ReinhardFastNormalizer,
    }  # type: Dict

    def __init__(
        self,
        method: str,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> None:

        super().__init__(method, **kwargs)
        self._device = device

    @property
    def vectorized(self) -> bool:  # type: ignore
        return self.n.vectorized

    def fit(
        self,
        arg1: Optional[Union[Dataset, np.ndarray, str]],
        batch_size: int = 64,
        num_threads: Union[str, int] = 'auto',
        **kwargs
    ):
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
            pb = tqdm(
                desc='Fitting normalizer...',
                ncols=80,
                total=dataset.num_tiles
            )
            for i, slide in dts:
                fit_vals = self.n.fit(i, reduce=True)
                if all_fit_vals == []:
                    all_fit_vals = [[] for _ in range(len(fit_vals))]
                for v, val in enumerate(fit_vals):
                    all_fit_vals[v] += [val]
                pb.update(batch_size)
            self.n.set_fit(*[torch.mean(torch.stack(v), dim=0) for v in all_fit_vals])

        elif isinstance(arg1, np.ndarray):
            self.n.fit(torch.from_numpy(arg1))

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
        _fit = self.n.get_fit()
        if as_list:
            return {k: v.tolist() for k, v in _fit.items()}
        else:
            return _fit

    def set_fit(self, **kwargs) -> None:
        self.n.set_fit(**kwargs)

    def torch_to_torch(
        self,
        image: Union[Dict, torch.Tensor],
        *args
    ) -> Tuple[Union[Dict, torch.Tensor], ...]:
        if isinstance(image, dict):
            to_return = {
                k: v for k, v in image.items()
                if k != 'tile_image'
            }
            to_return['tile_image'] = self.n.transform(image['tile_image'])
            return detuple(to_return, args)
        else:
            return detuple(self.n.transform(image), args)

    def torch_to_rgb(self, image: torch.Tensor) -> np.ndarray:
        return self.torch_to_torch(image).numpy()  # type: ignore

    def rgb_to_rgb(self, image: np.ndarray) -> np.ndarray:
        '''Non-normalized RGB numpy array -> normalized RGB numpy array'''
        return self.n.transform(torch.from_numpy(image)).numpy()

    def jpeg_to_rgb(self, jpeg_string: Union[str, bytes]) -> np.ndarray:
        '''Non-normalized compressed JPG string data -> normalized RGB numpy array'''
        return self.torch_to_rgb(torchvision.io.decode_image(jpeg_string))

    def png_to_rgb(self, png_string: Union[str, bytes]) -> np.ndarray:
        return self.torch_to_rgb(torchvision.io.decode_image(png_string))
