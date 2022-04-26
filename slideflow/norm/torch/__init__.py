import os
import torch
import torchvision
import numpy as np
from os.path import join
from typing import Dict, Optional, Tuple, Union, List

from slideflow.norm.torch import reinhard, reinhard_fast
from slideflow.norm import StainNormalizer


class TorchStainNormalizer(StainNormalizer):
    vectorized = True
    backend = 'torch'
    # Torch-specific vectorized normalizers disabled
    # as they are slower than the CV implementation
    normalizers = {}  # type: Dict

    def __init__(
        self,
        method: str = 'reinhard',
        source: Optional[str] = None
    ) -> None:

        self.method = method
        self.n = self.normalizers[method]
        self._source = source
        if not source:
            package_directory = os.path.dirname(os.path.abspath(__file__))
            source = join(package_directory, '../norm_tile.jpg')
        if source != 'dataset':
            self.src_img = torchvision.io.read_image(source).permute(1, 2, 0)  # CWH => WHC
            means, stds = self.n.fit(torch.unsqueeze(self.src_img, dim=0))
            self.target_means = torch.cat(means, 0).numpy()
            self.target_stds = torch.cat(stds, 0).numpy()
        else:
            self.target_means = None
            self.target_stds = None
        self.stain_matrix_target = None
        self.target_concentrations = None

    def __repr__(self) -> str:
        src = "" if not self._source else ", source={!r}".format(self._source)
        return "TorchStainNormalizer(method={!r}{})".format(self.method, src)

    @property
    def target_means(self) -> Optional[np.ndarray]:
        return self._target_means

    @target_means.setter
    def target_means(self, val: Optional[np.ndarray]):
        self._target_means = val

    @property
    def target_stds(self) -> Optional[np.ndarray]:
        return self._target_stds

    @target_stds.setter
    def target_stds(self, val: Optional[np.ndarray]):
        self._target_stds = val

    @property
    def stain_matrix_target(self) -> Optional[np.ndarray]:
        return self._stain_matrix_target

    @stain_matrix_target.setter
    def stain_matrix_target(self, val: Optional[np.ndarray]):
        self._stain_matrix_target = val

    @property
    def target_concentrations(self) -> Optional[np.ndarray]:
        return self._target_concentrations

    @target_concentrations.setter
    def target_concentrations(self, val: Optional[np.ndarray]):
        self._target_concentrations = val

    def fit(self, *args):
        return NotImplementedError()

    def get_fit(self) -> Dict[str, Optional[List[float]]]:
        return {
            'target_means': None if self.target_means is None else self.target_means.tolist(),
            'target_stds': None if self.target_stds is None else self.target_stds.tolist(),
            'stain_matrix_target': None if self.stain_matrix_target is None else self.stain_matrix_target.tolist(),
            'target_concentrations': None if self.target_concentrations is None else self.target_concentrations.tolist()
        }

    def batch_to_batch(
        self,
        batch: Union[dict, torch.Tensor],
        *args
    ) -> Tuple[Union[dict, torch.Tensor], ...]:
        if isinstance(batch, dict):
            to_return = {k: v for k, v in batch.items() if k != 'tile_image'}
            to_return['tile_image'] = self.torch_to_torch(batch['tile_image'])
            return tuple([to_return] + list(args))
        else:
            return tuple([self.torch_to_torch(batch)] + list(args))

    def torch_to_torch(self, image: torch.Tensor) -> torch.Tensor:
        if len(image.shape) == 3:
            return self.n.transform(torch.unsqueeze(image, dim=0), self.target_means, self.target_stds).squeeze()
        else:
            return self.n.transform(image, self.target_means, self.target_stds)

    def torch_to_rgb(self, image: torch.Tensor) -> np.ndarray:
        return self.torch_to_torch(image).numpy()

    def rgb_to_rgb(self, image: np.ndarray) -> np.ndarray:
        '''Non-normalized RGB numpy array -> normalized RGB numpy array'''
        return self.n.transform(
            torch.unsqueeze(torch.from_numpy(image), dim=0),
            self.target_means,
            self.target_stds
        ).squeeze().numpy()

    def jpeg_to_rgb(self, jpeg_string: Union[str, bytes]) -> np.ndarray:
        '''Non-normalized compressed JPG string data -> normalized RGB numpy array'''
        return self.torch_to_rgb(torchvision.io.decode_image(jpeg_string))

    def png_to_rgb(self, png_string: Union[str, bytes]) -> np.ndarray:
        return self.torch_to_rgb(torchvision.io.decode_image(png_string))
