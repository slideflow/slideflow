import torch
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import List, Optional, Union, Tuple
from sklearn.preprocessing._encoders import _BaseEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import __version__ as sklearn_version
from packaging import version
from fastai.vision.all import (
    DataLoaders, Learner, SaveModelCallback, CSVLogger
)

from slideflow import log
from slideflow.model import torch_utils
from .._params import TrainerConfig

# -----------------------------------------------------------------------------

def train(learner, config, callbacks=None):
    """Train an attention-based multi-instance learning model with FastAI.

    Args:
        learner (``fastai.learner.Learner``): FastAI learner.
        config (``TrainerConfig``): Trainer and model configuration.

    Keyword args:
        callbacks (list(fastai.Callback)): FastAI callbacks. Defaults to None.
    """
    cbs = [
        SaveModelCallback(fname=f"best_valid", monitor=config.save_monitor),
        CSVLogger(),
    ]
    if callbacks:
        cbs += callbacks
    if config.fit_one_cycle:
        if config.lr is None:
            lr = learner.lr_find().valley
            log.info(f"Using auto-detected learning rate: {lr}")
        else:
            lr = config.lr
        learner.fit_one_cycle(n_epoch=config.epochs, lr_max=lr, cbs=cbs)
    else:
        if config.lr is None:
            lr = learner.lr_find().valley
            log.info(f"Using auto-detected learning rate: {lr}")
        else:
            lr = config.lr
        learner.fit(n_epoch=config.epochs, lr=lr, wd=config.wd, cbs=cbs)
    return learner

class OrdinalClassEncoder(_BaseEncoder):
    """Encode categorical features as ordinal numbers.
    
    For k classes, creates k-1 bits where:
    - First class is encoded as all zeros
    - Last class is encoded as all ones
    - Each class has one more '1' than the previous class
    
    Example for 4 classes:
    Class 1: [0, 0, 0]
    Class 2: [0, 0, 1]
    Class 3: [0, 1, 1]
    Class 4: [1, 1, 1]
    """

    def __init__(self):
        self.categories_ = None
        self.ordinal_map_ = None
    
    def fit(self, X):
        """Fit the OrdinalClassEncoder to X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.
        
        Returns
        -------
        self
        """
        X_list, n_samples, n_features = self._check_X(X)
        
        if n_features != 1:
            raise ValueError("X should have exactly one feature")
        
        # Get unique categories and sort them
        self.categories_ = [np.unique(X_list[0])]
        
        # Create ordinal mapping
        num_bits = len(self.categories_[0]) - 1
        self.ordinal_map_ = {}
        
        for i, category in enumerate(self.categories_[0]):
            # Create encoding where last i bits are 1 and rest are 0
            encoding = [1 if j >= (num_bits - i) else 0 for j in range(num_bits)]
            self.ordinal_map_[category] = encoding
            
        return self
    
    def transform(self, X):
        """Transform X using ordinal encoding.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.
            
        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed input.
        """
        X_list, n_samples, n_features = self._check_X(X)
        
        if n_features != 1:
            raise ValueError("X should have exactly one feature")
            
        # Convert to numpy array of ordinal encodings
        result = np.array([self.ordinal_map_[x] for x in X_list[0]])
        
        return result
    
class CustomClassEncoder8(_BaseEncoder):
    """Encode hierarchical classes into 8-dimensional vectors.
    
    Encoding scheme:
    [level1_As, level1_Bs, level1_TC, As_A, As_AB, Bs_B1, Bs_B2, Bs_B3]
    
    Examples:
    A  -> [1,0,0, 1,0, 0,0,0]  # As group, A subtype
    AB -> [1,0,0, 0,1, 0,0,0]  # As group, AB subtype
    B1 -> [0,1,0, 0,0, 1,0,0]  # Bs group, B1 subtype
    B2 -> [0,1,0, 0,0, 0,1,0]  # Bs group, B2 subtype
    B3 -> [0,1,0, 0,0, 0,0,1]  # Bs group, B3 subtype
    TC -> [0,0,1, 0,0, 0,0,0]  # TC group
    """
    def __init__(self):
        self.categories_ = None
        self.encoding_map_ = None

    def fit(self, X):
        """Fit the encoder to X."""
        X_list, n_samples, n_features = self._check_X(X)

        if n_features != 1:
            raise ValueError("X should have exactly one feature")

        # Get unique categories and create encoding map
        self.categories_ = [np.unique(X_list[0])]

        self.encoding_map_ = {
            'A':  [1,0,0, 1,0, 0,0,0],
            'AB': [1,0,0, 0,1, 0,0,0],
            'B1': [0,1,0, 0,0, 1,0,0],
            'B2': [0,1,0, 0,0, 0,1,0],
            'B3': [0,1,0, 0,0, 0,0,1],
            'TC': [0,0,1, 0,0, 0,0,0]
        }
        
        return self

    def transform(self, X):
        """Transform X using the encoding scheme."""
        X_list, n_samples, n_features = self._check_X(X)

        if n_features != 1:
            raise ValueError("X should have exactly one feature")

        # Convert to numpy array of encodings
        result = np.array([self.encoding_map_[x] for x in X_list[0]])

        return result

class CustomClassEncoder(_BaseEncoder):
    """Encode hierarchical classes into 7-dimensional vectors.
    
    Encoding scheme:
    [level1_As, level1_Bs, level1_TC, As_A, As_AB, Bs_B1, Bs_B2, Bs_B3]
    
    Examples:
    A  -> [1,0,0, 1,0, 0,0]  # As group, A subtype
    AB -> [1,0,0, 0,1, 0,0]  # As group, AB subtype
    B1 -> [0,1,0, 0,0, 1,0]  # Bs group, B1 subtype
    B2 -> [0,1,0, 0,0, 0,1]  # Bs group, B2 subtype
    B3 -> [0,1,0, 0,0, 1,1]  # Bs group, B3 subtype
    TC -> [0,0,1, 0,0, 0,0]  # TC group
    """
    def __init__(self):
        self.categories_ = None
        self.encoding_map_ = None
    
    def fit(self, X):
        """Fit the encoder to X."""
        X_list, n_samples, n_features = self._check_X(X)
        
        if n_features != 1:
            raise ValueError("X should have exactly one feature")
        
        # Get unique categories and create encoding map
        self.categories_ = [np.unique(X_list[0])]
        
        self.encoding_map_ = {
            'A':  [1,0,0, 1,0, 0,0],
            'AB': [1,0,0, 0,1, 0,0],
            'B1': [0,1,0, 0,0, 0,0],
            'B2': [0,1,0, 0,0, 0,1],
            'B3': [0,1,0, 0,0, 1,1],
            'TC': [0,0,1, 0,0, 0,0]
        }
            
        return self
    
    def transform(self, X):
        """Transform X using the encoding scheme."""
        X_list, n_samples, n_features = self._check_X(X)
        
        if n_features != 1:
            raise ValueError("X should have exactly one feature")
            
        # Convert to numpy array of encodings
        result = np.array([self.encoding_map_[x] for x in X_list[0]])
        
        return result

# -----------------------------------------------------------------------------

def build_learner(
    config: TrainerConfig,
    bags: List[str],
    targets: npt.NDArray,
    train_idx: npt.NDArray[np.int_],
    val_idx: npt.NDArray[np.int_],
    unique_categories: npt.NDArray,
    outdir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    **dl_kwargs
) -> Tuple[Learner, Tuple[int, int]]:
    """Build a FastAI learner for training an MIL model.

    Args:
        config (``TrainerConfig``): Trainer and model configuration.
        bags (list(str)): Path to .pt files (bags) with features, one per patient.
        targets (np.ndarray): Category labels for each patient, in the same
            order as ``bags``.
        train_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the training set.
        val_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the validation set.
        unique_categories (np.ndarray(str)): Array of all unique categories
            in the targets. Used for one-hot encoding.
        outdir (str): Location in which to save training history and best model.
        device (torch.device or str): PyTorch device.

    Returns:
        fastai.learner.Learner, (int, int): FastAI learner and a tuple of the
            number of input features and output classes.

    """
    log.debug("Building FastAI learner")

    # Prepare device.
    device = torch_utils.get_device(device)

    # Prepare data.
    # Set oh_kw to a dictionary of keyword arguments for OneHotEncoder,
    # using the argument sparse=False if the sklearn version is <1.2
    # and sparse_output=False if the sklearn version is >=1.2.
    if version.parse(sklearn_version) < version.parse("1.2"):
        oh_kw = {"sparse": False}
    else:
        oh_kw = {"sparse_output": False}

    # Choose encoder based on model type
    if config.model_type == 'hierarchical':
        encoder = CustomClassEncoder().fit(unique_categories.reshape(-1, 1))
    elif config.is_classification():
        encoder = OneHotEncoder(**oh_kw).fit(unique_categories.reshape(-1, 1))
    elif config.model_type == 'ordinal':
        encoder = OrdinalClassEncoder().fit(unique_categories.reshape(-1, 1))
    else:
        encoder = None

    # Build the dataloaders.
    train_dl = config.build_train_dataloader(
        bags[train_idx],
        targets[train_idx],
        encoder=encoder,
        dataloader_kwargs=dict(
            num_workers=1,
            device=device,
            pin_memory=True,
            **dl_kwargs
        )
    )
    val_dl = config.build_val_dataloader(
        bags[val_idx],
        targets[val_idx],
        encoder=encoder,
        dataloader_kwargs=dict(
            shufle=False,
            num_workers=8,
            persistent_workers=True,
            device=device,
            pin_memory=False,
            **dl_kwargs
        )
    )

    # Prepare model.
    batch = train_dl.one_batch()
    n_in, n_out = config.inspect_batch(batch)
    model = config.build_model(n_in, n_out).to(device)

    if hasattr(model, 'relocate'):
        model.relocate()

    # Loss should weigh inversely to class occurences.
    if config.is_classification() and config.weighted_loss:
        counts = pd.value_counts(targets[train_idx])
        weights = counts.sum() / counts
        weights /= weights.sum()
        weights = torch.tensor(
            list(map(weights.get, encoder.categories_[0])), dtype=torch.float32
        ).to(device)
        loss_kw = {"weight": weights}
    else:
        loss_kw = {}
    loss_func = config.loss_fn(**loss_kw)

    # Create learning and fit.
    dls = DataLoaders(train_dl, val_dl)
    learner = Learner(dls, model, loss_func=loss_func, metrics=config.get_metrics(), path=outdir)

    return learner, (n_in, n_out)
