"""Dataset utility functions for MIL."""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Callable, Union, Protocol
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------

def build_dataset(bags, targets, encoder, bag_size, use_lens=False,  max_bag_size=None, dtype=torch.float32):
    assert len(bags) == len(targets)

    def _zip(bag, targets):
        features, lengths = bag
        _targets = targets if encoder is None else targets.squeeze()
        if use_lens:
            return (features, lengths, _targets)
        else:
            return (features, _targets)

    dataset = MapDataset(
        _zip,
        BagDataset(bags, bag_size=bag_size, max_bag_size=max_bag_size, dtype=dtype),
        EncodedDataset(encoder, targets),
    )
    dataset.encoder = encoder
    return dataset

def build_multibag_dataset(bags, targets, encoder, bag_size, use_lens=False, max_bag_size=None, dtype=torch.float32):
    assert len(bags) == len(targets)
    n_bags = len(bags[0])

    def _zip(bags_and_lengths, targets):
        _targets = targets if encoder is None else targets.squeeze()
        if use_lens:
            return *bags_and_lengths, _targets
        else:
            return [b[0] for b in bags_and_lengths], _targets

    dataset = MapDataset(
        _zip,
        MultiBagDataset(bags, n_bags, bag_size=bag_size, max_bag_size=max_bag_size, dtype=dtype),
        EncodedDataset(encoder, targets),
    )
    dataset.encoder = encoder
    return dataset

# -----------------------------------------------------------------------------

def _to_fixed_size_bag(
    bag: torch.Tensor,
    bag_size: int = 512,
) -> Tuple[torch.Tensor, int]:
    # get up to bag_size elements
    bag_idxs = torch.randperm(bag.shape[0])[:bag_size]
    bag_samples = bag[bag_idxs]

    # zero-pad if we don't have enough samples
    zero_padded = torch.cat(
        (
            bag_samples,
            torch.zeros(bag_size - bag_samples.shape[0], bag_samples.shape[1]),
        )
    )
    return zero_padded, min(bag_size, len(bag))

def _to_clipped_bag(
    bag: torch.Tensor,
    max_bag_size: int = 512,
) -> Tuple[torch.Tensor, int]:
    if len(bag) > max_bag_size:
        bag_idxs = torch.randperm(bag.shape[0])[:max_bag_size]
        return bag[bag_idxs], max_bag_size
    else:
        return bag, len(bag)

# -----------------------------------------------------------------------------

@dataclass
class BagDataset(Dataset):

    def __init__(
        self,
        bags: Union[List[Path], List[np.ndarray], List[torch.Tensor], List[List[str]]],
        bag_size: Optional[int] = None,
        max_bag_size: Optional[int] = None,
        preload: bool = False,
        *,
        dtype: torch.dtype = torch.float32
    ):
        """A dataset of bags of instances.

        Args:

            bags (list(str), list(np.ndarray), list(torch.Tensor), list(list(str))):  Bags for each slide.
                This can either be a list of `.pt`  files, a list of numpy
                arrays, a list of Tensors, or a list of lists of strings (where
                each item in the list is a patient, and nested items are slides
                for that patient). Each bag consists of features taken from all
                images from a slide. Data should be of shape N x F, where N is
                the number of instances and F is the number of features per
                instance/slide.
            bag_size (int):  The number of instances in each bag. For bags
                containing more instances, a random sample of `bag_size`
                instances will be drawn.  Smaller bags are padded with zeros.
                If `bag_size` is None, all the samples will be used.

        """
        super().__init__()
        if bag_size and max_bag_size:
            raise ValueError("Cannot specify both bag_size and max_bag_size")
        self.bags = bags
        self.bag_size = bag_size
        self.max_bag_size = max_bag_size
        self.preload = preload
        self.dtype = dtype

        if self.preload:
            self.bags = [self._load(i) for i in range(len(self.bags))]

    def __len__(self):
        return len(self.bags)

    def _load(self, index: int):
        if isinstance(self.bags[index], str):
            feats = torch.load(self.bags[index]).to(self.dtype)
        elif isinstance(self.bags[index], np.ndarray):
            feats = torch.from_numpy(self.bags[index]).to(self.dtype)
        elif isinstance(self.bags[index], torch.Tensor):
            feats = self.bags[index].to(self.dtype)
        else:
            feats = torch.cat([
                torch.load(slide).to(self.dtype)
                for slide in self.bags[index]
            ])
        return feats

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # collect all the features
        if self.preload:
            feats = self.bags[index]
        else:
            feats = self._load(index)

        # sample a subset, if required
        if self.bag_size:
            return _to_fixed_size_bag(feats, bag_size=self.bag_size)
        elif self.max_bag_size:
            return _to_clipped_bag(feats, max_bag_size=self.max_bag_size)
        else:
            return feats, len(feats)

# -----------------------------------------------------------------------------

@dataclass
class MultiBagDataset(Dataset):
    """A dataset of bags of instances, with multiple bags per instance."""

    bags: List[Union[List[Path], List[np.ndarray], List[torch.Tensor], List[List[str]]]]
    """Bags for each slide.

    This can either be a list of `.pt` files, a list of numpy arrays, a list
    of Tensors, or a list of lists of strings (where each item in the list is
    a patient, and nested items are slides for that patient).

    Each bag consists of features taken from all images from a slide. Data
    should be of shape N x F, where N is the number of instances and F is the
    number of features per instance/slide.
    """

    n_bags: int
    """Number of bags per instance."""

    bag_size: Optional[int] = None
    """The number of instances in each bag.
    For bags containing more instances, a random sample of `bag_size`
    instances will be drawn.  Smaller bags are padded with zeros.  If
    `bag_size` is None, all the samples will be used.
    """

    max_bag_size: Optional[int] = None

    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if self.bag_size and self.max_bag_size:
            raise ValueError("Cannot specify both bag_size and max_bag_size")

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:

        bags = self.bags[index]
        assert len(bags) == self.n_bags

        # Load to tensors.
        loaded_bags = []
        for bag in bags:
            if isinstance(bag, str):
                loaded_bags.append(torch.load(bag).to(self.dtype))
            elif isinstance(self.bags[index], np.ndarray):
                loaded_bags.append(torch.from_numpy(bag, dtype=self.dtype))
            elif isinstance(self.bags[index], torch.Tensor):
                loaded_bags.append(bag.to(self.dtype))
            else:
                raise ValueError("Invalid bag type: {}".format(type(bag)))

        # Sample a subset, if required
        if self.bag_size:
            return [_to_fixed_size_bag(bag, bag_size=self.bag_size) for bag in loaded_bags]
        elif self.max_bag_size:
            return [_to_clipped_bag(bag, max_bag_size=self.max_bag_size) for bag in loaded_bags]
        else:
            return [(bag, len(bag)) for bag in loaded_bags]


# -----------------------------------------------------------------------------

class MapDataset(Dataset):
    def __init__(
        self,
        func: Callable,
        *datasets: Union[npt.NDArray, Dataset],
        strict: bool = True
    ) -> None:
        """A dataset mapping over a function over other datasets.
        Args:
            func:  Function to apply to the underlying datasets.  Has to accept
                `len(dataset)` arguments.
            datasets:  The datasets to map over.
            strict:  Enforce the datasets to have the same length.  If
                false, then all datasets will be truncated to the shortest
                dataset's length.
        """
        if strict:
            assert all(len(ds) == len(datasets[0]) for ds in datasets)  # type: ignore
            self._len = len(datasets[0])  # type: ignore
        elif datasets:
            self._len = min(len(ds) for ds in datasets)  # type: ignore
        else:
            self._len = 0

        self._datasets = datasets
        self.func = func
        self.encoder = None

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Any:
        return self.func(*[ds[index] for ds in self._datasets])

    def new_empty(self):
        # FIXME hack to appease fastai's export
        return self

# -----------------------------------------------------------------------------

class SKLearnEncoder(Protocol):
    """An sklearn-style encoder."""

    categories_: List[List[str]]

    def transform(self, x: List[List[Any]]):
        ...


# -----------------------------------------------------------------------------

class EncodedDataset(MapDataset):
    def __init__(self, encode: SKLearnEncoder, values: npt.NDArray):
        """A dataset which first encodes its input data.
        This class is can be useful with classes such as fastai, where the
        encoder is saved as part of the model.
        Args:
            encode:  an sklearn encoding to encode the data with.
            values:  data to encode.
        """
        super().__init__(self._unsqueeze_to_float32, values)
        self.encode = encode

    def _unsqueeze_to_float32(self, x):
        if self.encode is None:
            return torch.tensor(x, dtype=torch.float32)
        return torch.tensor(
            self.encode.transform(np.array(x).reshape(1, -1)), dtype=torch.float32
        )