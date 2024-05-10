"""Dataset utility functions for MIL."""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Callable, Union, Protocol
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------

def build_dataset(bags, targets, encoder, bag_size, use_lens=False, balanced=False):
    assert len(bags) == len(targets)

    def _zip(bag, targets):
        features, lengths = bag
        if use_lens:
            return (features, lengths, targets.squeeze())
        else:
            return (features, targets.squeeze())

    dataset = MapDataset(
        _zip,
        BagDataset(bags, bag_size=bag_size, balanced=balanced),
        EncodedDataset(encoder, targets),
    )
    dataset.encoder = encoder
    return dataset

def build_clam_dataset(bags, targets, encoder, bag_size, balanced=False):
    assert len(bags) == len(targets)

    def _zip(bag, targets):
        features, lengths = bag
        return (features, targets.squeeze(), True), targets.squeeze()

    dataset = MapDataset(
        _zip,
        BagDataset(bags, bag_size=bag_size, balanced=balanced),
        EncodedDataset(encoder, targets),
    )
    dataset.encoder = encoder
    return dataset

def build_multibag_dataset(bags, targets, encoder, bag_size, n_bags, use_lens=False):
    assert len(bags) == len(targets)

    def _zip(bags_and_lengths, targets):
        if use_lens:
            return *bags_and_lengths, targets.squeeze()
        else:
            return [b[0] for b in bags_and_lengths], targets.squeeze()

    dataset = MapDataset(
        _zip,
        MultiBagDataset(bags, n_bags, bag_size=bag_size),
        EncodedDataset(encoder, targets),
    )
    dataset.encoder = encoder
    return dataset

# -----------------------------------------------------------------------------

def _to_fixed_size_bag(
    bag: torch.Tensor,
    bag_size: int = 512,
    address: dict = dict(),
    balanced: bool = False
) -> Tuple[torch.Tensor, int]:
    '''
    Get up to bag_size elements
    Args:
        bag (torch.Tensor): feature
        bag_size (int): the maximize size of the returned features tensor. Defaults to 512
        address (dict): if bag was concatenated from multiple tensors, this address denotes which index belongs to which tensor
            if balanced is True, address must be always passed to this function and contains at least one element
        balanced (bool): if True and the bag was concatenated from multiple tensors, the returned features are drawn randomly with
            equal number of features from each tensor. Defaults to False
    '''
    if balanced:
        # The number of elements in address
        n_address = len(address)
        # If address is an empty dict
        if n_address == 0:
            raise ValueError("Address of features must contain at least 1 element.")
        
        # Ensure that address is in consistent format: {feats_index: [List of indexes]}
        min_bag_size = np.inf
        c_address = dict() # consistent_address
        for k, v in address.items():
            # v is a list of indexes
            if isinstance(v, list):
                # Each element must be an integer
                if not all([isinstance(ele, int) for ele in v]):
                    raise ValueError(f"Invalid value type of address. Expected intergers.")
                c_address[k] = v
                if len(v) < min_bag_size:
                    min_bag_size = len(v)
            # v is a tuple
            elif isinstance(v, tuple):
                # Each element must be an integer
                if not all([isinstance(ele, int) for ele in v]):
                    raise ValueError(f"Invalid value type of address. Expected intergers.")
                # v has format (start_index, end_index) or (end_index, start_index)
                if len(v) == 2:
                    start_index, end_index = min(v), max(v)
                    c_address[k] = np.arange(start_index, end_index + 1)
                # else, treat v as a list
                else:
                    c_address[k] = list(v)
                if len(v) < min_bag_size:
                    min_bag_size = len(v)
            # v is a number
            elif isinstance(v, int):
                c_address[k] = [v]
                if 1 < min_bag_size:
                    min_bag_size = 1
            else:
                raise ValueError(f"Invalid value type of address. Expected List, Tuple or Int, not {type(v)}")

        # The number of index to be chosen
        n_index = min(bag_size // n_address, min_bag_size)

        # Get the indexes from each tensor with equal number, randomly
        bag_idxs = []
        for add in c_address.values():
            bag_idxs.extend(np.random.choice(add, size=n_index, replace=False)) # replace = False means no repetition
    
    else:
        bag_idxs = torch.randperm(bag.shape[0])[:bag_size]

    bag_samples = bag[bag_idxs]

    # zero-pad if we don't have enough samples
    zero_padded = torch.cat(
        (
            bag_samples,
            torch.zeros(bag_size - bag_samples.shape[0], bag_samples.shape[1]),
        )
    )
    return zero_padded, min(bag_size, len(bag), len(bag_samples))

# -----------------------------------------------------------------------------

@dataclass
class BagDataset(Dataset):

    def __init__(
        self,
        bags: Union[List[Path], List[np.ndarray], List[torch.Tensor], List[List[str]]],
        bag_size: Optional[int] = None,
        preload: bool = False,
        balanced: bool = False,
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
            balanced (bool): if True and each bag inside bags was concatenated from multiple tensors, the returned features
                are drawn randomly with equal number of features from each tensor. Defaults to False

        """
        super().__init__()
        self.bags = bags
        self.bag_size = bag_size
        self.preload = preload
        self.balanced = balanced

        if self.preload:
            # self.bags becomes a list of tuples
            # each tuple is of format: (features (torch.Tensor), address(dict))
            self.bags = [self._load(i) for i in range(len(self.bags))]

    def __len__(self):
        return len(self.bags)

    def _load(self, index: int):
        # Address is a dictionary mapping the index of feats originating from a different slide
        address = dict()
        if isinstance(self.bags[index], str):
            feats = torch.load(self.bags[index]).to(torch.float32)
            address = {0: (0, len(feats)-1)}
        elif isinstance(self.bags[index], np.ndarray):
            feats = torch.from_numpy(self.bags[index]).to(torch.float32)
            address = {0: (0, len(feats)-1)}
        elif isinstance(self.bags[index], torch.Tensor):
            feats = self.bags[index]
            address = {0: (0, len(feats)-1)}
        else:
            # Load the first Tensor
            feats = torch.load(self.bags[index][0]).to(torch.float32)
            # Create the first address for Tensor at index 0
            address = {0: (0, len(feats)-1)}
            # iter_key is the key of the address dictionary
            iter_key = 0
            # For each Tensor from index 1
            for slide in self.bags[index][1:]:
                # Load Tensor
                next_feats = torch.load(slide).to(torch.float32)
                # Concat this Tensor to feats
                feats = torch.cat([feats, next_feats])
                # Create the address for this Tensor
                iter_key += 1
                start_index = address[iter_key-1][1] + 1
                address[iter_key] = (start_index, start_index + len(next_feats) - 1)
        
        return feats, address

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # collect all the features
        if self.preload:
            feats, address = self.bags[index]
        else:
            feats, address = self._load(index)

        # sample a subset, if required
        if self.bag_size:
            return _to_fixed_size_bag(feats, bag_size=self.bag_size, address=address, balanced=self.balanced)
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

    balanced: bool = False
    """
    balanced (bool): if True and each bag inside bags was concatenated from multiple tensors,
    the returned features are drawn randomly with equal number of features from each tensor.
    Defaults to False
    """

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:

        bags = self.bags[index]
        assert len(bags) == self.n_bags

        # Load to tensors.
        loaded_bags = []
        for bag in bags:
            if isinstance(bag, str):
                loaded_bags.append(torch.load(bag).to(torch.float32))
            elif isinstance(self.bags[index], np.ndarray):
                loaded_bags.append(torch.from_numpy(bag))
            elif isinstance(self.bags[index], torch.Tensor):
                loaded_bags.append(bag)
            else:
                raise ValueError("Invalid bag type: {}".format(type(bag)))

        # Sample a subset, if required
        if self.bag_size:
            return [_to_fixed_size_bag(bag, bag_size=self.bag_size, address={0: (0, len(bag) - 1)}, balanced=self.balanced) for bag in loaded_bags]
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
        return torch.tensor(
            self.encode.transform(np.array(x).reshape(1, -1)), dtype=torch.float32
        )