'''PyTorch backend for the slideflow.model submodule.'''

import inspect
import json
import os
import types
import numpy as np
import multiprocessing as mp
import pandas as pd
import torch
import torchvision

from torch import Tensor
from torch.nn.functional import softmax
from packaging import version
from rich.progress import Progress, TimeElapsedColumn
from collections import defaultdict
from os.path import join
from pandas.api.types import is_float_dtype, is_integer_dtype
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple,
                    Union, Callable)

import slideflow as sf
import slideflow.util.neptune_utils
from slideflow import errors
from slideflow.model import base as _base
from slideflow.model import torch_utils
from slideflow.model.torch_utils import autocast
from slideflow.model.base import log_manifest, no_scope, BaseFeatureExtractor
from slideflow.util import log, NormFit, ImgBatchSpeedColumn

if TYPE_CHECKING:
    import pandas as pd
    from slideflow.norm import StainNormalizer


class LinearBlock(torch.nn.Module):
    '''Block module that includes a linear layer -> ReLU -> BatchNorm'''

    def __init__(
        self,
        in_ftrs: int,
        out_ftrs: int,
        dropout: Optional[float] = None
    ) -> None:
        super().__init__()
        self.in_ftrs = in_ftrs
        self.out_ftrs = out_ftrs
        self.linear = torch.nn.Linear(in_ftrs, out_ftrs)
        self.relu = torch.nn.ReLU(inplace=True)
        self.bn = torch.nn.BatchNorm1d(out_ftrs)
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.relu(x)
        x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ModelWrapper(torch.nn.Module):
    '''Wrapper for PyTorch modules to support multiple outcomes, clinical
    (patient-level) inputs, and additional hidden layers.'''

    def __init__(
        self,
        model: Any,
        n_classes: List[int],
        num_slide_features: int = 0,
        hidden_layers: Optional[List[int]] = None,
        drop_images: bool = False,
        dropout: Optional[float] = None,
        include_top: bool = True
    ) -> None:
        super().__init__()
        self.model = model
        self.n_classes = len(n_classes)
        self.drop_images = drop_images
        self.num_slide_features = num_slide_features
        self.num_hidden_layers = 0 if not hidden_layers else len(hidden_layers)
        self.has_aux = False
        log.debug(f'Model class name: {model.__class__.__name__}')
        if not drop_images:
            # Check for auxillary classifier
            if model.__class__.__name__ in ('Inception3',):
                log.debug("Auxillary classifier detected")
                self.has_aux = True

            # Get the last linear layer prior to the logits layer
            if model.__class__.__name__ in ('Xception', 'NASNetALarge'):
                num_ftrs = self.model.last_linear.in_features
                self.model.last_linear = torch.nn.Identity()
            elif model.__class__.__name__ in ('SqueezeNet'):
                num_ftrs = 1000
            elif hasattr(self.model, 'classifier'):
                children = list(self.model.classifier.named_children())
                if len(children):
                    # VGG, AlexNet
                    if include_top:
                        log.debug("Including existing fully-connected "
                                  "top classifier layers")
                        last_linear_name, last_linear = children[-1]
                        num_ftrs = last_linear.in_features
                        setattr(
                            self.model.classifier,
                            last_linear_name,
                            torch.nn.Identity()
                        )
                    elif model.__class__.__name__ in ('AlexNet',
                                                      'MobileNetV2',
                                                      'MNASNet'):
                        log.debug("Removing fully-connected classifier layers")
                        _, first_classifier = children[1]
                        num_ftrs = first_classifier.in_features
                        self.model.classifier = torch.nn.Identity()
                    elif model.__class__.__name__ in ('VGG', 'MobileNetV3'):
                        log.debug("Removing fully-connected classifier layers")
                        _, first_classifier = children[0]
                        num_ftrs = first_classifier.in_features
                        self.model.classifier = torch.nn.Identity()
                else:
                    num_ftrs = self.model.classifier.in_features
                    self.model.classifier = torch.nn.Identity()
            elif hasattr(self.model, 'fc'):
                num_ftrs = self.model.fc.in_features
                self.model.fc = torch.nn.Identity()
            elif hasattr(self.model, 'out_features'):
                num_ftrs = self.model.out_features
            elif hasattr(self.model, 'head'):
                num_ftrs = self.model.head.out_features
            else:
                print(self.model)
                raise errors.ModelError("Unable to find last linear layer for "
                                        f"model {model.__class__.__name__}")
        else:
            num_ftrs = 0

        # Add slide-level features
        num_ftrs += num_slide_features

        # Add hidden layers
        if hidden_layers:
            hl_ftrs = [num_ftrs] + hidden_layers
            for i in range(len(hidden_layers)):
                setattr(self, f'h{i}', LinearBlock(hl_ftrs[i],
                                                   hl_ftrs[i+1],
                                                   dropout=dropout))
            num_ftrs = hidden_layers[-1]

        # Add the outcome/logits layers for each outcome, if multiple outcomes
        for i, n in enumerate(n_classes):
            setattr(self, f'fc{i}', torch.nn.Linear(num_ftrs, n))

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            if name == 'model':
                raise e
            return getattr(self.model, name)

    def forward(
        self,
        img: Tensor,
        slide_features: Optional[Tensor] = None
    ):
        if slide_features is None and self.num_slide_features:
            raise ValueError("Expected 2 inputs, got 1")

        # Last linear of core convolutional model
        if not self.drop_images:
            x = self.model(img)

        # Discard auxillary classifier
        if self.has_aux and self.training:
            x = x.logits

        # Merging image data with any slide-level input data
        if self.num_slide_features and not self.drop_images:
            assert slide_features is not None
            x = torch.cat([x, slide_features], dim=1)
        elif self.num_slide_features:
            x = slide_features

        # Hidden layers
        if self.num_hidden_layers:
            x = self.h0(x)
        if self.num_hidden_layers > 1:
            for h in range(1, self.num_hidden_layers):
                x = getattr(self, f'h{h}')(x)

        # Return a list of outputs if we have multiple outcomes
        if self.n_classes > 1:
            out = [getattr(self, f'fc{i}')(x) for i in range(self.n_classes)]

        # Otherwise, return the single output
        else:
            out = self.fc0(x)

        return out  # , x


class ModelParams(_base._ModelParams):
    """Build a set of hyperparameters."""

    ModelDict = {
        'resnet18': torchvision.models.resnet18,
        'resnet50': torchvision.models.resnet50,
        'alexnet': torchvision.models.alexnet,
        'squeezenet': torchvision.models.squeezenet.squeezenet1_1,
        'densenet': torchvision.models.densenet161,
        'inception': torchvision.models.inception_v3,
        'googlenet': torchvision.models.googlenet,
        'shufflenet': torchvision.models.shufflenet_v2_x1_0,
        'resnext50_32x4d': torchvision.models.resnext50_32x4d,
        'vgg16': torchvision.models.vgg16,  # needs support added
        'mobilenet_v2': torchvision.models.mobilenet_v2,
        'mobilenet_v3_small': torchvision.models.mobilenet_v3_small,
        'mobilenet_v3_large': torchvision.models.mobilenet_v3_large,
        'wide_resnet50_2': torchvision.models.wide_resnet50_2,
        'mnasnet': torchvision.models.mnasnet1_0,
        'xception': torch_utils.xception,
        'nasnet_large': torch_utils.nasnetalarge
    }

    def __init__(self, *, loss: str = 'CrossEntropy', **kwargs) -> None:
        self.OptDict = {
            'Adadelta': torch.optim.Adadelta,
            'Adagrad': torch.optim.Adagrad,
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'SparseAdam': torch.optim.SparseAdam,
            'Adamax': torch.optim.Adamax,
            'ASGD': torch.optim.ASGD,
            'LBFGS': torch.optim.LBFGS,
            'RMSprop': torch.optim.RMSprop,
            'Rprop': torch.optim.Rprop,
            'SGD': torch.optim.SGD
        }
        self.LinearLossDict = {
            'L1': torch.nn.L1Loss,
            'MSE': torch.nn.MSELoss,
            'NLL': torch.nn.NLLLoss,  # negative log likelihood
            'HingeEmbedding': torch.nn.HingeEmbeddingLoss,
            'SmoothL1': torch.nn.SmoothL1Loss,
            'CosineEmbedding': torch.nn.CosineEmbeddingLoss,
        }
        self.AllLossDict = {
            'CrossEntropy': torch.nn.CrossEntropyLoss,
            'CTC': torch.nn.CTCLoss,
            'PoissonNLL': torch.nn.PoissonNLLLoss,
            'GaussianNLL': torch.nn.GaussianNLLLoss,
            'KLDiv': torch.nn.KLDivLoss,
            'BCE': torch.nn.BCELoss,
            'BCEWithLogits': torch.nn.BCEWithLogitsLoss,
            'MarginRanking': torch.nn.MarginRankingLoss,
            'MultiLabelMargin': torch.nn.MultiLabelMarginLoss,
            'Huber': torch.nn.HuberLoss,
            'SoftMargin': torch.nn.SoftMarginLoss,
            'MultiLabelSoftMargin': torch.nn.MultiLabelSoftMarginLoss,
            'MultiMargin': torch.nn.MultiMarginLoss,
            'TripletMargin': torch.nn.TripletMarginLoss,
            'TripletMarginWithDistance': torch.nn.TripletMarginWithDistanceLoss,
            'L1': torch.nn.L1Loss,
            'MSE': torch.nn.MSELoss,
            'NLL': torch.nn.NLLLoss,  # negative log likelihood
            'HingeEmbedding': torch.nn.HingeEmbeddingLoss,
            'SmoothL1': torch.nn.SmoothL1Loss,
            'CosineEmbedding': torch.nn.CosineEmbeddingLoss,
        }
        super().__init__(loss=loss, **kwargs)
        assert self.model in self.ModelDict.keys() or self.model.startswith('timm_')
        assert self.optimizer in self.OptDict.keys()
        assert self.loss in self.AllLossDict
        if self.model == 'inception':
            log.warn("Model 'inception' has an auxillary classifier, which "
                     "is currently ignored during training. Auxillary "
                     "classifier loss will be included during training "
                     "starting in version 1.3")


    def get_opt(self, params_to_update: Iterable) -> torch.optim.Optimizer:
        return self.OptDict[self.optimizer](
            params_to_update,
            lr=self.learning_rate,
            weight_decay=self.l2
        )

    def get_loss(self) -> torch.nn.modules.loss._Loss:
        return self.AllLossDict[self.loss]()

    def get_model_loader(self, model: str) -> Callable:
        if model in self.ModelDict:
            return self.ModelDict[model]
        elif model.startswith('timm_'):

            def loader(**kwargs):
                try:
                    import timm
                except ImportError:
                    raise ImportError(f"Unable to load model {model}; "
                                      "timm package not installed.")
                return timm.create_model(model[5:], **kwargs)

            return loader
        else:
            raise ValueError(f"Model {model} not found.")

    def build_model(
        self,
        labels: Optional[Dict] = None,
        num_classes: Optional[Union[int, Dict[Any, int]]] = None,
        num_slide_features: int = 0,
        pretrain: Optional[str] = None,
        checkpoint: Optional[str] = None
    ) -> torch.nn.Module:

        assert num_classes is not None or labels is not None
        if num_classes is None:
            assert labels is not None
            num_classes = self._detect_classes_from_labels(labels)
        if not isinstance(num_classes, dict):
            num_classes = {'out-0': num_classes}

        # Prepare custom model pretraining
        if pretrain:
            log.info(f"Using pretraining: [green]{pretrain}")
        if (isinstance(pretrain, str)
           and sf.util.path_to_ext(pretrain).lower() == 'zip'):
           _pretrained = pretrain
           pretrain = None
        else:
            _pretrained = None

        # Build base model
        if self.model in ('xception', 'nasnet_large'):
            _model = self.get_model_loader(self.model)(
                num_classes=1000,
                pretrained=pretrain
            )
        else:
            # Compatibility logic for prior versions of PyTorch
            model_fn = self.get_model_loader(self.model)
            model_fn_sig = inspect.signature(model_fn)
            model_kw = [
                param.name
                for param in model_fn_sig.parameters.values()
                if param.kind == param.POSITIONAL_OR_KEYWORD
            ]
            call_kw = {}
            if 'image_size' in model_kw:
                call_kw.update(dict(image_size=self.tile_px))
            if (version.parse(torchvision.__version__) >= version.parse("0.13")
               and not self.model.startswith('timm_')):
                # New Torchvision API
                w = 'DEFAULT' if pretrain == 'imagenet' else pretrain
                call_kw.update(dict(weights=w))  # type: ignore
            else:
                call_kw.update(dict(pretrained=pretrain))  # type: ignore
            _model = model_fn(**call_kw)

        # Add final layers to models
        hidden_layers = [
            self.hidden_layer_width
            for _ in range(self.hidden_layers)
        ]
        model = ModelWrapper(
            _model,
            list(num_classes.values()),
            num_slide_features,
            hidden_layers,
            self.drop_images,
            dropout=self.dropout,
            include_top=self.include_top
        )
        if _pretrained is not None:
            lazy_load_pretrained(model, _pretrained)
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint))
        return model

    def model_type(self) -> str:
        """Returns 'linear', 'categorical', or 'cph', reflecting the loss."""
        #check if loss is custom_[type] and returns type
        if self.loss.startswith('custom'):
            return self.loss[7:]
        elif self.loss == 'NLL':
            return 'cph'
        elif self.loss in self.LinearLossDict:
            return 'linear'
        else:
            return 'categorical'


class Trainer:
    """Base trainer class containing functionality for model building, input
    processing, training, and evaluation.

    This base class requires categorical outcome(s). Additional outcome types
    are supported by :class:`slideflow.model.LinearTrainer` and
    :class:`slideflow.model.CPHTrainer`.

    Slide-level (e.g. clinical) features can be used as additional model input
    by providing slide labels in the slide annotations dictionary, under
    the key 'input'.
    """

    _model_type = 'categorical'

    def __init__(
        self,
        hp: ModelParams,
        outdir: str,
        labels: Dict[str, Any],
        *,
        slide_input: Optional[Dict[str, Any]] = None,
        name: str = 'Trainer',
        feature_sizes: Optional[List[int]] = None,
        feature_names: Optional[List[str]] = None,
        outcome_names: Optional[List[str]] = None,
        mixed_precision: bool = True,
        allow_tf32: bool = False,
        config: Dict[str, Any] = None,
        use_neptune: bool = False,
        neptune_api: Optional[str] = None,
        neptune_workspace: Optional[str] = None,
        load_method: str = 'weights',
        custom_objects: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        transform: Optional[Union[Callable, Dict[str, Callable]]] = None
    ):
        """Sets base configuration, preparing model inputs and outputs.

        Args:
            hp (:class:`slideflow.ModelParams`): ModelParams object.
            outdir (str): Destination for event logs and checkpoints.
            labels (dict): Dict mapping slide names to outcome labels (int or
                float format).
            slide_input (dict): Dict mapping slide names to additional
                slide-level input, concatenated after post-conv.
            name (str, optional): Optional name describing the model, used for
                model saving. Defaults to None.
            feature_sizes (list, optional): List of sizes of input features.
                Required if providing additional input features as model input.
            feature_names (list, optional): List of names for input features.
                Used when permuting feature importance.
            outcome_names (list, optional): Name of each outcome. Defaults to
                "Outcome {X}" for each outcome.
            mixed_precision (bool, optional): Use FP16 mixed precision (rather
                than FP32). Defaults to True.
            allow_tf32 (bool): Allow internal use of Tensorfloat-32 format.
                Defaults to False.
            config (dict, optional): Training configuration dictionary, used
                for logging and image format verification. Defaults to None.
            use_neptune (bool, optional): Use Neptune API logging.
                Defaults to False
            neptune_api (str, optional): Neptune API token, used for logging.
                Defaults to None.
            neptune_workspace (str, optional): Neptune workspace.
                Defaults to None.
            load_method (str): Loading method to use when reading model.
                This argument is ignored in the PyTorch backend, as all models
                are loaded by first building the model with hyperparameters
                detected in ``params.json``, then loading weights with
                ``torch.nn.Module.load_state_dict()``. Defaults to
                'full' (ignored).
            transform (callable or dict, optional): Optional transform to
                apply to input images. If dict, must have the keys 'train'
                and/or 'val', mapping to callables that takes a single
                image Tensor as input and returns a single image Tensor.
                If None, no transform is applied. If a single callable is
                provided, it will be applied to both training and validation
                data. If a dict is provided, the 'train' transform will be
                applied to training data and the 'val' transform will be
                applied to validation data. If a dict is provided and either
                'train' or 'val' is None, no transform will be applied to
                that data. Defaults to None.
        """
        self.hp = hp
        self.outdir = outdir
        self.labels = labels
        self.patients = dict()  # type: Dict[str, str]
        self.name = name
        self.model = None  # type: Optional[torch.nn.Module]
        self.inference_model = None  # type: Optional[torch.nn.Module]
        self.mixed_precision = mixed_precision
        self.device = torch_utils.get_device(device)
        self.mid_train_val_dts: Optional[Iterable] = None
        self.loss_fn: torch.nn.modules.loss._Loss
        self.use_tensorboard: bool
        self.writer = None  # type: Optional[torch.utils.tensorboard.SummaryWriter]
        self._reset_training_params()

        if custom_objects is not None:
            log.warn("custom_objects argument ignored in PyTorch backend.")

        # Enable or disable Tensorflow-32
        # Allows PyTorch to internally use tf32 for matmul and convolutions
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32  # type: ignore
        self._allow_tf32 = allow_tf32

        # Slide-level input args
        if slide_input:
            self.slide_input = {
                k: [float(vi) for vi in v]
                for k, v in slide_input.items()
            }
        else:
            self.slide_input = None  # type: ignore
        self.feature_names = feature_names
        self.feature_sizes = feature_sizes
        self.num_slide_features = 0 if not feature_sizes else sum(feature_sizes)

        self.normalizer = self.hp.get_normalizer()
        if self.normalizer:
            log.info(f'Using realtime {self.hp.normalizer} normalization')

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        self._process_transforms(transform)
        self._process_outcome_labels(outcome_names)
        if isinstance(labels, pd.DataFrame):
            cat_assign = self._process_category_assignments()

        # Log parameters
        if config is None:
            config = {
                'slideflow_version': sf.__version__,
                'backend': sf.backend(),
                'git_commit': sf.__gitcommit__,
                'model_name': self.name,
                'full_model_name': self.name,
                'outcomes': self.outcome_names,
                'model_type': self.hp.model_type(),
                'img_format': None,
                'tile_px': self.hp.tile_px,
                'tile_um': self.hp.tile_um,
                'input_features': None,
                'input_feature_sizes': None,
                'input_feature_labels': None,
                'hp': self.hp.to_dict(),
            }
            if isinstance(labels, pd.DataFrame):
                config['outcome_labels'] = {str(k): v for k,v in cat_assign.items()}

        sf.util.write_json(config, join(self.outdir, 'params.json'))

        # Neptune logging
        self.config = config
        self.img_format = config['img_format'] if 'img_format' in config else None
        self.use_neptune = use_neptune
        self.neptune_run = None
        if self.use_neptune:
            if neptune_api is None or neptune_workspace is None:
                raise ValueError("If using Neptune, must supply neptune_api"
                                 " and neptune_workspace.")
            self.neptune_logger = sf.util.neptune_utils.NeptuneLog(
                neptune_api,
                neptune_workspace
            )

    @property
    def num_outcomes(self) -> int:
        if self.hp.model_type() == 'categorical':
            assert self.outcome_names is not None
            return len(self.outcome_names)
        else:
            return 1

    @property
    def multi_outcome(self) -> bool:
        return (self.num_outcomes > 1)

    def _process_category_assignments(self) -> Dict[int, str]:
        """Get category assignments for categorical outcome labels.

        Dataframes with integer labels are assumed to be categorical if
        if hp.model_type is 'categorical'.
        Dataframes with float labels are assumed to be linear.
        Dataframes with other labels are assumed to be categorical, and will
        be assigned an integer label based on the order of unique values.

        """
        if not isinstance(self.labels, pd.DataFrame):
            raise ValueError("Expected DataFrame with 'label' column.")
        if 'label' not in self.labels.columns:
            raise ValueError("Expected DataFrame with 'label' column.")
        if self.hp.model_type() == 'categorical':
            if is_integer_dtype(self.labels['label']) or is_float_dtype(self.labels['label']):
                return {i: str(i) for i in sorted(self.labels['label'].unique())}
            else:
                int_to_str = dict(enumerate(sorted(self.labels['label'].unique())))
                str_to_int = {v: k for k, v in int_to_str.items()}
                log.info("Assigned integer labels to categories:")
                log.info(str_to_int)
                self.labels['label'] = self.labels['label'].map(str_to_int)
                return int_to_str
        else:
            return {}


    def _process_transforms(
        self,
        transform: Optional[Union[Callable, Dict[str, Callable]]] = None
    ) -> None:
        """Process custom transformations for training and/or validation."""
        if not isinstance(transform, dict):
            transform = {'train': transform, 'val': transform}
        if any([t not in ('train', 'val') for t in transform]):
            raise ValueError("transform must be a callable or dict with keys "
                             "'train' and/or 'val'")
        if 'train' not in transform:
            transform['train'] = None
        if 'val' not in transform:
            transform['val'] = None
        self.transform = transform

    def _process_outcome_labels(
        self,
        outcome_names: Optional[List[str]] = None,
    ) -> None:
        """Process outcome labels to determine number of outcomes and names.

        Supports experimental tile-level labels provided via pandas DataFrame.

        Args:
            labels (dict): Dict mapping slide names to outcome labels (int or
                float format). Experimental funtionality: if labels is a
                pandas DataFrame, the 'label' column will be used as the
                outcome labels.
            outcome_names (list, optional): Name of each outcome. Defaults to
                "Outcome {X}" for each outcome.

        """
        # Process DataFrame tile-level labels
        if isinstance(self.labels, pd.DataFrame):
            if 'label' not in self.labels.columns:
                raise errors.ModelError("Expected DataFrame with 'label' "
                                        "column.")
            if outcome_names and len(outcome_names) > 1:
                raise errors.ModelError(
                    "Expected single outcome name for labels from a pandas dataframe."
                )
            self.outcome_names = outcome_names or ['Outcome 0']
            return

        # Process dictionary slide-level labels
        outcome_labels = np.array(list(self.labels.values()))
        if len(outcome_labels.shape) == 1:
            outcome_labels = np.expand_dims(outcome_labels, axis=1)
        if not outcome_names:
            self.outcome_names = [
                f'Outcome {i}'
                for i in range(outcome_labels.shape[1])
            ]
        else:
            self.outcome_names = outcome_names
        if not len(self.outcome_names) == outcome_labels.shape[1]:
            n_names = len(self.outcome_names)
            n_out = outcome_labels.shape[1]
            raise errors.ModelError(f"Number of outcome names ({n_names}) does"
                                    f" not match number of outcomes ({n_out})")

    def _reset_training_params(self) -> None:
        self.global_step = 0
        self.epoch = 0  # type: int
        self.step = 0  # type: int
        self.log_frequency = 0  # type: int
        self.early_stop = False  # type: bool
        self.moving_average = []  # type: List
        self.dataloaders = {}  # type: Dict[str, Any]
        self.validation_batch_size = None  # type: Optional[int]
        self.validate_on_batch = 0
        self.validation_steps = 0
        self.ema_observations = 0  # type: int
        self.ema_smoothing = 0
        self.last_ema = -1  # type: float
        self.ema_one_check_prior = -1  # type: float
        self.ema_two_checks_prior = -1  # type: float
        self.epoch_records = 0  # type: int
        self.running_loss = 0.0
        self.running_corrects = {}  # type: Union[Tensor, Dict[str, Tensor]]

    def _accuracy_as_numpy(
        self,
        acc: Union[Tensor, float, List[Tensor], List[float]]
    ) -> Union[float, List[float]]:
        if isinstance(acc, list):
            return [t.item() if isinstance(t, Tensor) else t for t in acc]
        else:
            return (acc.item() if isinstance(acc, Tensor) else acc)

    def _build_model(
        self,
        checkpoint: Optional[str] = None,
        pretrain: Optional[str] = None
    ) -> None:
        if checkpoint:
            log.info(f"Loading checkpoint at [green]{checkpoint}")
            self.load(checkpoint)
        else:
            self.model = self.hp.build_model(
                labels=self.labels,
                pretrain=pretrain,
                num_slide_features=self.num_slide_features
            )
        # Create an inference model before any multi-GPU parallelization
        # is applied to the self.model parameter
        self.inference_model = self.model

    def _calculate_accuracy(
        self,
        running_corrects: Union[Tensor, Dict[Any, Tensor]],
        num_records: int = 1
    ) -> Tuple[Union[Tensor, List[Tensor]], str]:
        '''Reports accuracy of each outcome.'''
        assert self.hp.model_type() == 'categorical'
        if self.num_outcomes > 1:
            if not isinstance(running_corrects, dict):
                raise ValueError("Expected running_corrects to be a dict:"
                                 " num_outcomes is > 1")
            acc_desc = ''
            acc_list = [running_corrects[r] / num_records
                        for r in running_corrects]
            for o in range(len(running_corrects)):
                _acc = running_corrects[f'out-{o}'] / num_records
                acc_desc += f"out-{o} acc: {_acc:.4f} "
            return acc_list, acc_desc
        else:
            assert not isinstance(running_corrects, dict)
            _acc = running_corrects / num_records
            return _acc, f'acc: {_acc:.4f}'

    def _calculate_loss(
        self,
        outputs: Union[Tensor, List[Tensor]],
        labels: Union[Tensor, Dict[Any, Tensor]],
        loss_fn: torch.nn.modules.loss._Loss
    ) -> Tensor:
        '''Calculates loss in a manner compatible with multiple outcomes.'''
        if self.num_outcomes > 1:
            if not isinstance(labels, dict):
                raise ValueError("Expected labels to be a dict: num_outcomes"
                                 " is > 1")
            loss = sum([
                loss_fn(out, labels[f'out-{o}'])
                for o, out in enumerate(outputs)
            ])
        else:
            loss = loss_fn(outputs, labels)
        return loss  # type: ignore

    def _check_early_stopping(
        self,
        val_acc: Optional[Union[float, List[float]]] = None,
        val_loss: Optional[float] = None
    ) -> str:
        if val_acc is None and val_loss is None:
            if (self.hp.early_stop
               and self.hp.early_stop_method == 'manual'
               and self.hp.manual_early_stop_epoch <= self.epoch  # type: ignore
               and self.hp.manual_early_stop_batch <= self.step):  # type: ignore
                log.info(f'Manual early stop triggered: epoch {self.epoch}, '
                         f'batch {self.step}')
                if self.epoch not in self.hp.epochs:
                    self.hp.epochs += [self.epoch]
                self.early_stop = True
        else:
            if self.hp.early_stop_method == 'accuracy':
                if self.num_outcomes > 1:
                    raise errors.ModelError(
                        "Early stopping method 'accuracy' not supported with"
                        " multiple outcomes; use 'loss'.")
                early_stop_val = val_acc
            else:
                early_stop_val = val_loss
            assert early_stop_val is not None
            assert isinstance(early_stop_val, float)

            self.moving_average += [early_stop_val]
            if len(self.moving_average) >= self.ema_observations:
                # Only keep track of the last [ema_observations]
                self.moving_average.pop(0)
                if self.last_ema == -1:
                    # Simple moving average
                    self.last_ema = (sum(self.moving_average)
                                     / len(self.moving_average))  # type: ignore
                    log_msg = f' (SMA: {self.last_ema:.3f})'
                else:
                    alpha = (self.ema_smoothing / (1 + self.ema_observations))
                    self.last_ema = (early_stop_val * alpha
                                     + (self.last_ema * (1 - alpha)))
                    log_msg = f' (EMA: {self.last_ema:.3f})'
                    if self.neptune_run and self.last_ema != -1:
                        neptune_dest = "metrics/val/batch/exp_moving_avg"
                        self.neptune_run[neptune_dest].log(self.last_ema)

                if (self.hp.early_stop
                   and self.ema_two_checks_prior != -1
                   and self.epoch > self.hp.early_stop_patience):

                    if ((self.hp.early_stop_method == 'accuracy'
                         and self.last_ema <= self.ema_two_checks_prior)
                       or (self.hp.early_stop_method == 'loss'
                           and self.last_ema >= self.ema_two_checks_prior)):

                        log.info(f'Early stop triggered: epoch {self.epoch}, '
                                 f'step {self.step}')
                        self._log_early_stop_to_neptune()
                        if self.epoch not in self.hp.epochs:
                            self.hp.epochs += [self.epoch]
                        self.early_stop = True
                        return log_msg

                self.ema_two_checks_prior = self.ema_one_check_prior
                self.ema_one_check_prior = self.last_ema
        return ''

    def _detect_patients(self, *args):
        self.patients = dict()
        for dataset in args:
            if dataset is None:
                continue
            dataset_patients = dataset.patients()
            if not dataset_patients:
                self.patients.update({s: s for s in self.slides})
            else:
                self.patients.update(dataset_patients)

    def _empty_corrects(self) -> Union[int, Dict[str, int]]:
        if self.multi_outcome:
            return {
                f'out-{o}': 0
                for o in range(self.num_outcomes)
            }
        else:
            return 0

    def _epoch_metrics(
        self,
        acc: Union[float, List[float]],
        loss: float,
        label: str
    ) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        epoch_metrics = {'loss': loss}  # type: Dict
        if self.hp.model_type() == 'categorical':
            epoch_metrics.update({'accuracy': acc})
        return {f'{label}_metrics': epoch_metrics}

    def _val_metrics(self, **kwargs) -> Dict[str, Dict[str, float]]:
        """Evaluate model and calculate metrics.

        Returns:
            Dict[str, Dict[str, float]]: Dict with validation metrics.
            Returns metrics in the form:
            ```
            {
                'val_metrics': {
                    'loss': ...,
                    'accuracy': ...,
                },
                'tile_auc': ...,
                'slide_auc': ...,
                ...
            }
            ```
        """
        if hasattr(self, 'optimizer'):
            self.optimizer.zero_grad()
        assert self.model is not None
        self.model.eval()
        results_log = os.path.join(self.outdir, 'results_log.csv')
        epoch_results = {}

        # Preparations for calculating accuracy/loss in metrics_from_dataset()
        def update_corrects(pred, labels, running_corrects=None):
            if running_corrects is None:
                running_corrects = self._empty_corrects()
            if self.hp.model_type() == 'categorical':
                labels = self._labels_to_device(labels, self.device)
                return self._update_corrects(pred, labels, running_corrects)
            else:
                return 0

        def update_loss(pred, labels, running_loss, size):
            labels = self._labels_to_device(labels, self.device)
            loss = self._calculate_loss(pred, labels, self.loss_fn)
            return running_loss + (loss.item() * size)

        torch_args = types.SimpleNamespace(
            update_corrects=update_corrects,
            update_loss=update_loss,
            num_slide_features=self.num_slide_features,
            slide_input=self.slide_input,
            normalizer=(self.normalizer if self._has_gpu_normalizer() else None),
        )
        # Calculate patient/slide/tile metrics (AUC, R-squared, C-index, etc)
        metrics, acc, loss = sf.stats.metrics_from_dataset(
            self.inference_model,
            model_type=self.hp.model_type(),
            patients=self.patients,
            dataset=self.dataloaders['val'],
            data_dir=self.outdir,
            outcome_names=self.outcome_names,
            neptune_run=self.neptune_run,
            torch_args=torch_args,
            uq=bool(self.hp.uq),
            **kwargs
        )
        loss_and_acc = {'loss': loss}
        if self.hp.model_type() == 'categorical':
            loss_and_acc.update({'accuracy': acc})
            self._log_epoch(
                'val',
                self.epoch,
                loss,
                self._calculate_accuracy(acc)[1]  # type: ignore
            )
        epoch_metrics = {'val_metrics': loss_and_acc}

        for metric in metrics:
            if metrics[metric]['tile'] is None:
                continue
            epoch_results[f'tile_{metric}'] = metrics[metric]['tile']
            epoch_results[f'slide_{metric}'] = metrics[metric]['slide']
            epoch_results[f'patient_{metric}'] = metrics[metric]['patient']
        epoch_metrics.update(epoch_results)
        sf.util.update_results_log(
            results_log,
            'trained_model',
            {f'epoch{self.epoch}': epoch_metrics}
        )
        self._log_eval_to_neptune(loss, acc, metrics, epoch_metrics)
        return epoch_metrics

    def _fit_normalizer(self, norm_fit: Optional[NormFit]) -> None:
        """Fit the Trainer normalizer using the specified fit, if applicable.

        Args:
            norm_fit (Optional[Dict[str, np.ndarray]]): Normalizer fit.
        """
        if norm_fit is not None and not self.normalizer:
            raise ValueError("norm_fit supplied, but model params do not"
                             "specify a normalizer.")
        if self.normalizer and norm_fit is not None:
            self.normalizer.set_fit(**norm_fit)  # type: ignore
        elif (self.normalizer
              and 'norm_fit' in self.config
              and self.config['norm_fit'] is not None):
            log.debug("Detecting normalizer fit from model config")
            self.normalizer.set_fit(**self.config['norm_fit'])

    def _has_gpu_normalizer(self) -> bool:
        import slideflow.norm.torch
        return (isinstance(self.normalizer, sf.norm.torch.TorchStainNormalizer)
                and self.normalizer.device != "cpu")

    def _labels_to_device(
        self,
        labels: Union[Dict[Any, Tensor], Tensor],
        device: torch.device
    ) -> Union[Dict[Any, Tensor], Tensor]:
        '''Moves a set of outcome labels to the given device.'''
        if self.num_outcomes > 1:
            if not isinstance(labels, dict):
                raise ValueError("Expected labels to be a dict: num_outcomes"
                                 " is > 1")
            labels = {
                k: la.to(device, non_blocking=True) for k, la in labels.items()
            }
        elif isinstance(labels, dict):
            labels = torch.stack(list(labels.values()), dim=1)
            return labels.to(device, non_blocking=True)
        else:
            labels = labels.to(device, non_blocking=True)
        return labels

    def _log_epoch(
        self,
        phase: str,
        epoch: int,
        loss: float,
        accuracy_desc: str,
    ) -> None:
        """Logs epoch description."""
        log.info(f'[bold blue]{phase}[/] Epoch {epoch} | loss:'
                 f' {loss:.4f} {accuracy_desc}')

    def _log_manifest(
        self,
        train_dts: Optional["sf.Dataset"],
        val_dts: Optional["sf.Dataset"],
        labels: Optional[Union[str, Dict]] = 'auto'
    ) -> None:
        """Log the tfrecord and label manifest to slide_manifest.csv

        Args:
            train_dts (sf.Dataset): Training dataset. May be None.
            val_dts (sf.Dataset): Validation dataset. May be None.
            labels (dict, optional): Labels dictionary. May be None.
                Defaults to 'auto' (read from self.labels).
        """
        if labels == 'auto':
            _labels = self.labels
        elif labels is None:
            _labels = None
        else:
            assert isinstance(labels, dict)
            _labels = labels
        log_manifest(
            (train_dts.tfrecords() if train_dts else None),
            (val_dts.tfrecords() if val_dts else None),
            labels=_labels,
            filename=join(self.outdir, 'slide_manifest.csv')
        )

    def _log_to_tensorboard(
        self,
        loss: float,
        acc: Union[float, List[float]],
        label: str
    ) -> None:
        self.writer.add_scalar(f'Loss/{label}', loss, self.global_step)
        if self.hp.model_type() == 'categorical':
            if self.num_outcomes > 1:
                assert isinstance(acc, list)
                for o, _acc in enumerate(acc):
                    self.writer.add_scalar(
                        f'Accuracy-{o}/{label}', _acc, self.global_step
                    )
            else:
                self.writer.add_scalar(
                    f'Accuracy/{label}', acc, self.global_step
                )

    def _log_to_neptune(
        self,
        loss: float,
        acc: Union[Tensor, List[Tensor]],
        label: str,
        phase: str
    ) -> None:
        """Logs epoch loss/accuracy to Neptune."""
        assert phase in ('batch', 'epoch')
        step = self.epoch if phase == 'epoch' else self.global_step
        if self.neptune_run:
            self.neptune_run[f"metrics/{label}/{phase}/loss"].log(loss,
                                                                  step=step)
            acc = self._accuracy_as_numpy(acc)
            if isinstance(acc, list):
                for a, _acc in enumerate(acc):
                    sf.util.neptune_utils.list_log(
                        run=self.neptune_run,
                        label=f'metrics/{label}/{phase}/accuracy-{a}',
                        val=_acc,
                        step=step
                    )
            else:
                sf.util.neptune_utils.list_log(
                    run=self.neptune_run,
                    label=f'metrics/{label}/{phase}/accuracy',
                    val=acc,
                    step=step
                )


    def _log_early_stop_to_neptune(self) -> None:
        # Log early stop to neptune
        if self.neptune_run:
            self.neptune_run["early_stop/early_stop_epoch"] = self.epoch
            self.neptune_run["early_stop/early_stop_batch"] = self.step
            self.neptune_run["early_stop/method"] = self.hp.early_stop_method
            self.neptune_run["sys/tags"].add("early_stopped")

    def _log_eval_to_neptune(
        self,
        loss: float,
        acc: float,
        metrics: Dict[str, Any],
        epoch_results: Dict[str, Any]
    ) -> None:
        if self.use_neptune:
            assert self.neptune_run is not None
            self.neptune_run['results'] = epoch_results

            # Validation epoch metrics
            self.neptune_run['metrics/val/epoch/loss'].log(loss,
                                                           step=self.epoch)
            sf.util.neptune_utils.list_log(
                self.neptune_run,
                'metrics/val/epoch/accuracy',
                acc,
                step=self.epoch
            )
            for metric in metrics:
                if metrics[metric]['tile'] is None:
                    continue
                for outcome in metrics[metric]['tile']:
                    # If only one outcome,
                    #   log to metrics/val/epoch/[metric].
                    # If more than one outcome,
                    #   log to metrics/val/epoch/[metric]/[outcome_name]
                    def metric_label(s):
                        if len(metrics[metric]['tile']) == 1:
                            return f'metrics/val/epoch/{s}_{metric}'
                        else:
                            return f'metrics/val/epoch/{s}_{metric}/{outcome}'

                    tile_metric = metrics[metric]['tile'][outcome]
                    slide_metric = metrics[metric]['slide'][outcome]
                    patient_metric = metrics[metric]['patient'][outcome]

                    # If only one value for a metric, log to .../[metric]
                    # If more than one value for a metric
                    #   (e.g. AUC for each category),
                    # log to .../[metric]/[i]
                    sf.util.neptune_utils.list_log(
                        self.neptune_run,
                        metric_label('tile'),
                        tile_metric,
                        step=self.epoch
                    )
                    sf.util.neptune_utils.list_log(
                        self.neptune_run,
                        metric_label('slide'),
                        slide_metric,
                        step=self.epoch
                    )
                    sf.util.neptune_utils.list_log(
                        self.neptune_run,
                        metric_label('patient'),
                        patient_metric,
                        step=self.epoch
                    )

    def _mid_training_validation(self) -> None:
        """Perform mid-epoch validation, if appropriate."""

        if not self.validate_on_batch:
            return
        elif not (
            'val' in self.dataloaders
            and self.step > 0
            and self.step % self.validate_on_batch == 0
        ):
            return

        if self.model is None or self.inference_model is None:
            raise errors.ModelError("Model not yet initialized.")
        self.model.eval()
        running_val_loss = 0
        num_val = 0
        running_val_correct = self._empty_corrects()

        for _ in range(self.validation_steps):
            val_img, val_label, slides, *_ = next(self.mid_train_val_dts)  # type:ignore
            val_img = val_img.to(self.device)
            val_img = val_img.to(memory_format=torch.channels_last)

            with torch.no_grad():
                _mp = (self.mixed_precision and self.device.type in ('cuda', 'cpu'))
                with autocast(self.device.type, mixed_precision=_mp):  # type: ignore

                    # GPU normalization, if specified.
                    if self._has_gpu_normalizer():
                        val_img = self.normalizer.preprocess(val_img)

                    if self.num_slide_features:
                        _slide_in = [self.slide_input[s] for s in slides]
                        inp = (val_img, Tensor(_slide_in).to(self.device))
                    else:
                        inp = (val_img,)  # type: ignore
                    val_outputs = self.inference_model(*inp)
                    val_label = self._labels_to_device(val_label, self.device)
                    val_batch_loss = self._calculate_loss(
                        val_outputs, val_label, self.loss_fn
                    )

            running_val_loss += val_batch_loss.item() * val_img.size(0)
            if self.hp.model_type() == 'categorical':
                running_val_correct = self._update_corrects(
                    val_outputs, val_label, running_val_correct  # type: ignore
                )
            num_val += val_img.size(0)
        val_loss = running_val_loss / num_val
        if self.hp.model_type() == 'categorical':
            val_acc, val_acc_desc = self._calculate_accuracy(
                running_val_correct, num_val  # type: ignore
            )
        else:
            val_acc, val_acc_desc = 0, ''  # type: ignore
        log_msg = f'Batch {self.step}: val loss: {val_loss:.4f} {val_acc_desc}'

        # Log validation metrics to neptune & check early stopping
        self._log_to_neptune(val_loss, val_acc, 'val', phase='batch')
        log_msg += self._check_early_stopping(
            self._accuracy_as_numpy(val_acc),
            val_loss
        )
        log.info(log_msg)

        # Log to tensorboard
        if self.use_tensorboard:
            if self.num_outcomes > 1:
                assert isinstance(running_val_correct, dict)
                _val_acc = [
                    running_val_correct[f'out-{o}'] / num_val
                    for o in range(len(val_outputs))
                ]
            else:
                assert not isinstance(running_val_correct, dict)
                _val_acc = running_val_correct / num_val  # type: ignore
            self._log_to_tensorboard(
                val_loss,
                self._accuracy_as_numpy(_val_acc),
                'test'
            )  # type: ignore

        # Put model back in training mode
        self.model.train()

    def _prepare_optimizers_and_loss(self) -> None:
        if self.model is None:
            raise ValueError("Model has not yet been initialized.")
        self.optimizer = self.hp.get_opt(self.model.parameters())
        if self.hp.learning_rate_decay:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.hp.learning_rate_decay
            )
            log.debug("Using exponentially decaying learning rate")
        else:
            self.scheduler = None  # type: ignore
        self.loss_fn = self.hp.get_loss()
        if self.mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()

    def _prepare_neptune_run(self, dataset: "sf.Dataset", label: str) -> None:
        if self.use_neptune:
            tags = [label]
            if 'k-fold' in self.config['validation_strategy']:
                tags += [f'k-fold{self.config["k_fold_i"]}']
            self.neptune_run = self.neptune_logger.start_run(
                self.name,
                self.config['project'],
                dataset,
                tags=tags
            )
            assert self.neptune_run is not None
            self.neptune_logger.log_config(self.config, label)
            self.neptune_run['data/slide_manifest'].upload(
                os.path.join(self.outdir, 'slide_manifest.csv')
            )
            try:
                config_path = join(self.outdir, 'params.json')
                config = sf.util.load_json(config_path)
                config['neptune_id'] = self.neptune_run['sys/id'].fetch()
            except Exception:
                log.info("Unable to log params (params.json) with Neptune.")

    def _print_model_summary(self, train_dts) -> None:
        """Prints model summary and logs to neptune."""
        if self.model is None:
            raise ValueError("Model has not yet been initialized.")
        empty_inp = [torch.empty(
            [self.hp.batch_size, 3, train_dts.tile_px, train_dts.tile_px]
        )]
        if self.num_slide_features:
            empty_inp += [
                torch.empty([self.hp.batch_size, self.num_slide_features])
            ]
        if sf.getLoggingLevel() <= 20:
            model_summary = torch_utils.print_module_summary(
                self.model, empty_inp
            )
            if self.neptune_run:
                self.neptune_run['summary'] = model_summary

    def _save_model(self) -> None:
        assert self.model is not None
        name = self.name if self.name else 'trained_model'
        save_path = os.path.join(self.outdir, f'{name}_epoch{self.epoch}.zip')
        torch.save(self.model.state_dict(), save_path)
        log.info(f"Model saved to [green]{save_path}")

    def _close_dataloaders(self):
        """Close dataloaders, ensuring threads have joined."""
        del self.mid_train_val_dts
        for name, d in self.dataloaders.items():
            if '_dataset' in dir(d):
                log.debug(f"Closing dataloader {name} via _dataset.close()")
                d._dataset.close()
            elif 'dataset' in dir(d):
                log.debug(f"Closing dataloader {name} via dataset.close()")
                d.dataset.close()

    def _setup_dataloaders(
        self,
        train_dts: Optional["sf.Dataset"],
        val_dts: Optional["sf.Dataset"],
        mid_train_val: bool = False,
        incl_labels: bool = True,
        from_wsi: bool = False,
        **kwargs
    ) -> None:
        """Prepare dataloaders from training and validation."""
        interleave_args = types.SimpleNamespace(
            rank=0,
            num_replicas=1,
            labels=(self.labels if incl_labels else None),
            chunk_size=8,
            pin_memory=True,
            num_workers=4 if not from_wsi else 0,
            onehot=False,
            incl_slidenames=True,
            from_wsi=from_wsi,
            **kwargs
        )
        # Use GPU stain normalization for PyTorch normalizers, if supported
        if self._has_gpu_normalizer():
            log.info("Using GPU for stain normalization")
            interleave_args.standardize = False
        else:
            interleave_args.normalizer = self.normalizer

        if train_dts is not None:
            self.dataloaders = {
                'train': iter(train_dts.torch(
                    infinite=True,
                    batch_size=self.hp.batch_size,
                    augment=self.hp.augment,
                    transform=self.transform['train'],
                    drop_last=True,
                    **vars(interleave_args)
                ))
            }
        else:
            self.dataloaders = {}
        if val_dts is not None:
            if not self.validation_batch_size:
                validation_batch_size = self.hp.batch_size
            self.dataloaders['val'] = val_dts.torch(
                infinite=False,
                batch_size=validation_batch_size,
                augment=False,
                transform=self.transform['val'],
                incl_loc=True,
                **vars(interleave_args)
            )
            # Mid-training validation dataset
            if mid_train_val:
                self.mid_train_val_dts = torch_utils.cycle(
                    self.dataloaders['val']
                )
            if not self.validate_on_batch:
                val_log_msg = ''
            else:
                val_log_msg = f'every {str(self.validate_on_batch)} steps and '
            log.debug(f'Validation during training: {val_log_msg}at epoch end')
            if self.validation_steps:
                num_samples = self.validation_steps * self.hp.batch_size
                log.debug(
                    f'Using {self.validation_steps} batches ({num_samples} '
                    'samples) each validation check'
                )
            else:
                log.debug('Using entire validation set each validation check')
        else:
            log.debug('Validation during training: None')

    def _training_step(self, pb: Progress) -> None:
        assert self.model is not None
        images, labels, slides = next(self.dataloaders['train'])
        images = images.to(self.device, non_blocking=True)
        images = images.to(memory_format=torch.channels_last)
        labels = self._labels_to_device(labels, self.device)
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            _mp = (self.mixed_precision and self.device.type in ('cuda', 'cpu'))
            with autocast(self.device.type, mixed_precision=_mp):  # type: ignore

                # GPU normalization, if specified.
                if self._has_gpu_normalizer():
                    images = self.normalizer.preprocess(
                        images,
                        augment=(isinstance(self.hp.augment, str)
                                 and 'n' in self.hp.augment)
                    )

                # Slide-level features
                if self.num_slide_features:
                    _slide_in = [self.slide_input[s] for s in slides]
                    inp = (images, Tensor(_slide_in).to(self.device))
                else:
                    inp = (images,)  # type: ignore
                outputs = self.model(*inp)
                loss = self._calculate_loss(outputs, labels, self.loss_fn)

            # Update weights
            if self.mixed_precision and self.device.type == 'cuda':
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Update learning rate if using a scheduler
            _lr_decay_steps = self.hp.learning_rate_decay_steps
            if self.scheduler and (self.global_step+1) % _lr_decay_steps == 0:
                log.debug("Stepping learning rate decay")
                self.scheduler.step()

        # Record accuracy and loss
        self.epoch_records += images.size(0)
        if self.hp.model_type() == 'categorical':
            self.running_corrects = self._update_corrects(
                outputs, labels, self.running_corrects
            )
            train_acc, acc_desc = self._calculate_accuracy(
                self.running_corrects, self.epoch_records
            )
        else:
            train_acc, acc_desc = 0, ''  # type: ignore
        self.running_loss += loss.item() * images.size(0)
        _loss = self.running_loss / self.epoch_records
        pb.update(task_id=0,  # type: ignore
                  description=(f'[bold blue]train[/] '
                               f'loss: {_loss:.4f} {acc_desc}'))
        pb.advance(task_id=0)  # type: ignore

        # Log to tensorboard
        if self.use_tensorboard and self.global_step % self.log_frequency == 0:
            if self.num_outcomes > 1:
                _train_acc = [
                    (self.running_corrects[f'out-{o}']  # type: ignore
                     / self.epoch_records)
                    for o in range(len(outputs))
                ]
            else:
                _train_acc = (self.running_corrects  # type: ignore
                              / self.epoch_records)
            self._log_to_tensorboard(
                loss.item(),
                self._accuracy_as_numpy(_train_acc),
                'train'
            )
        # Log to neptune & check early stopping
        self._log_to_neptune(loss.item(), train_acc, 'train', phase='batch')
        self._check_early_stopping(None, None)

    def _update_corrects(
        self,
        outputs: Union[Tensor, Dict[Any, Tensor]],
        labels: Union[Tensor, Dict[str, Tensor]],
        running_corrects: Union[Tensor, Dict[str, Tensor]],
    ) -> Union[Tensor, Dict[str, Tensor]]:
        '''Updates running accuracy in a manner compatible with >1 outcomes.'''
        assert self.hp.model_type() == 'categorical'
        if self.num_outcomes > 1:
            for o, out in enumerate(outputs):
                _, preds = torch.max(out, 1)
                running_corrects[f'out-{o}'] += torch.sum(  # type: ignore
                    preds == labels[f'out-{o}'].data  # type: ignore
                )
        else:
            _, preds = torch.max(outputs, 1)  # type: ignore
            running_corrects += torch.sum(preds == labels.data)  # type: ignore
        return running_corrects

    def _validate_early_stop(self) -> None:
        """Validates early stopping parameters."""

        if (self.hp.early_stop and self.hp.early_stop_method == 'accuracy' and
           self.hp.model_type() == 'categorical' and self.num_outcomes > 1):
            raise errors.ModelError("Cannot combine 'accuracy' early stopping "
                                    "with multiple categorical outcomes.")
        if (self.hp.early_stop_method == 'manual'
            and (self.hp.manual_early_stop_epoch is None
                 or self.hp.manual_early_stop_batch is None)):
            raise errors.ModelError(
                "Early stopping method 'manual' requires that both "
                "manual_early_stop_epoch and manual_early_stop_batch are set "
                "in model params."
            )

    def _verify_img_format(self, dataset, *datasets: Optional["sf.Dataset"]) -> str:
        """Verify that the image format of the dataset matches the model config.

        Args:
            dataset (sf.Dataset): Dataset to check.
            *datasets (sf.Dataset): Additional datasets to check. May be None.

        Returns:
            str: Image format, either 'png' or 'jpg', if a consistent image
                format was found, otherwise None.

        """
        # First, verify all datasets have the same image format
        img_formats = set([d.img_format for d in datasets if d])
        if len(img_formats) > 1:
            log.error("Multiple image formats detected: {}.".format(
                ', '.join(img_formats)
            ))
            return None
        elif self.img_format and not dataset.img_format:
            log.warning("Unable to verify image format (PNG/JPG) of dataset.")
            return None
        elif self.img_format and dataset.img_format != self.img_format:
            log.error(
                "Mismatched image formats. Expected '{}' per model config, "
                "but dataset has format '{}'.".format(
                    self.img_format,
                    dataset.img_format))
            return None
        else:
            return dataset.img_format

    def load(self, model: str, training=True) -> None:
        """Loads a state dict at the given model location. Requires that the
        Trainer's hyperparameters (Trainer.hp)
        match the hyperparameters of the model to be loaded."""

        if self.labels is not None:
            self.model = self.hp.build_model(
                labels=self.labels,
                num_slide_features=self.num_slide_features
            )
        else:
            self.model = self.hp.build_model(
                num_classes=len(self.outcome_names),
                num_slide_features=self.num_slide_features
            )
        self.model.load_state_dict(torch.load(model))
        self.inference_model = self.model

    def predict(
        self,
        dataset: "sf.Dataset",
        batch_size: Optional[int] = None,
        norm_fit: Optional[NormFit] = None,
        format: str = 'parquet',
        from_wsi: bool = False,
        roi_method: str = 'auto',
    ) -> Dict[str, "pd.DataFrame"]:
        """Perform inference on a model, saving predictions.

        Args:
            dataset (:class:`slideflow.dataset.Dataset`): Dataset containing
                TFRecords to evaluate.
            batch_size (int, optional): Evaluation batch size. Defaults to the
                same as training (per self.hp)
            norm_fit (Dict[str, np.ndarray]): Normalizer fit, mapping fit
                parameters (e.g. target_means, target_stds) to values
                (np.ndarray). If not provided, will fit normalizer using
                model params (if applicable). Defaults to None.
            format (str, optional): Format in which to save predictions. Either
                'csv', 'feather', or 'parquet'. Defaults to 'parquet'.
            from_wsi (bool): Generate predictions from tiles dynamically
                extracted from whole-slide images, rather than TFRecords.
                Defaults to False (use TFRecords).
            roi_method (str): ROI method to use if from_wsi=True (ignored if
                from_wsi=False).  Either 'inside', 'outside', 'auto', 'ignore'.
                If 'inside' or 'outside', will extract tiles in/out of an ROI,
                and raise errors.MissingROIError if an ROI is not available.
                If 'auto', will extract tiles inside an ROI if available,
                and across the whole-slide if no ROI is found.
                If 'ignore', will extract tiles across the whole-slide
                regardless of whether an ROI is available.
                Defaults to 'auto'.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary with keys 'tile', 'slide', and
            'patient', and values containing DataFrames with tile-, slide-,
            and patient-level predictions.
        """
        if format not in ('csv', 'feather', 'parquet'):
            raise ValueError(f"Unrecognized format {format}")

        self._detect_patients(dataset)

        # Verify image format
        self._verify_img_format(dataset)

        # Fit normalizer
        self._fit_normalizer(norm_fit)

        # Load and initialize model
        if not self.model:
            raise errors.ModelNotLoadedError
        self.model.to(self.device)
        self.model.eval()
        self._log_manifest(None, dataset, labels=None)

        if from_wsi and sf.slide_backend() == 'libvips':
            pool = mp.Pool(
                sf.util.num_cpu(default=8),
                initializer=sf.util.set_ignore_sigint
            )
        elif from_wsi:
            pool = mp.dummy.Pool(sf.util.num_cpu(default=8))
        else:
            pool = None
        if not batch_size:
            batch_size = self.hp.batch_size

        self._setup_dataloaders(
            train_dts=None,
            val_dts=dataset,
            incl_labels=False,
            from_wsi=from_wsi,
            roi_method=roi_method,
            pool=pool)

        log.info('Generating predictions...')
        torch_args = types.SimpleNamespace(
            num_slide_features=self.num_slide_features,
            slide_input=self.slide_input,
            normalizer=(self.normalizer if self._has_gpu_normalizer() else None),
        )
        dfs = sf.stats.predict_dataset(
            model=self.model,
            dataset=self.dataloaders['val'],
            model_type=self._model_type,
            torch_args=torch_args,
            outcome_names=self.outcome_names,
            uq=bool(self.hp.uq),
            patients=self.patients
        )
        # Save predictions
        sf.stats.metrics.save_dfs(dfs, format=format, outdir=self.outdir)
        self._close_dataloaders()
        if pool is not None:
            pool.close()
        return dfs

    def evaluate(
        self,
        dataset: "sf.Dataset",
        batch_size: Optional[int] = None,
        save_predictions: Union[bool, str] = 'parquet',
        reduce_method: str = 'average',
        norm_fit: Optional[NormFit] = None,
        uq: Union[bool, str] = 'auto',
        from_wsi: bool = False,
        roi_method: str = 'auto',
    ):
        """Evaluate model, saving metrics and predictions.

        Args:
            dataset (:class:`slideflow.dataset.Dataset`): Dataset to evaluate.
            batch_size (int, optional): Evaluation batch size. Defaults to the
                same as training (per self.hp)
            save_predictions (bool or str, optional): Save tile, slide, and
                patient-level predictions at each evaluation. May be 'csv',
                'feather', or 'parquet'. If False, will not save predictions.
                Defaults to 'parquet'.
            reduce_method (str, optional): Reduction method for calculating
                slide-level and patient-level predictions for categorical outcomes.
                Either 'average' or 'proportion'. If 'average', will reduce with
                average of each logit across tiles. If 'proportion', will convert
                tile predictions into onehot encoding then reduce by averaging
                these onehot values. Defaults to 'average'.
            norm_fit (Dict[str, np.ndarray]): Normalizer fit, mapping fit
                parameters (e.g. target_means, target_stds) to values
                (np.ndarray). If not provided, will fit normalizer using
                model params (if applicable). Defaults to None.
            uq (bool or str, optional): Enable UQ estimation (for
                applicable models). Defaults to 'auto'.
            from_wsi (bool): Generate predictions from tiles dynamically
                extracted from whole-slide images, rather than TFRecords.
                Defaults to False (use TFRecords).
            roi_method (str): ROI method to use if from_wsi=True (ignored if
                from_wsi=False).  Either 'inside', 'outside', 'auto', 'ignore'.
                If 'inside' or 'outside', will extract tiles in/out of an ROI,
                and raise errors.MissingROIError if an ROI is not available.
                If 'auto', will extract tiles inside an ROI if available,
                and across the whole-slide if no ROI is found.
                If 'ignore', will extract tiles across the whole-slide
                regardless of whether an ROI is available.
                Defaults to 'auto'.

        Returns:
            Dictionary of evaluation metrics.
        """
        if uq != 'auto':
            if not isinstance(uq, bool):
                raise ValueError(f"Unrecognized value {uq} for uq")
            self.hp.uq = uq
        if batch_size:
            self.validation_batch_size = batch_size
        if not self.model:
            raise errors.ModelNotLoadedError
        if from_wsi and sf.slide_backend() == 'libvips':
            pool = mp.Pool(
                sf.util.num_cpu(default=8),
                initializer=sf.util.set_ignore_sigint
            )
        elif from_wsi:
            pool = mp.dummy.Pool(sf.util.num_cpu(default=8))
        else:
            pool = None

        self._detect_patients(dataset)
        self._verify_img_format(dataset)
        self._fit_normalizer(norm_fit)
        self.model.to(self.device)
        self.model.eval()
        self.loss_fn = self.hp.get_loss()
        self._log_manifest(None, dataset)
        self._prepare_neptune_run(dataset, 'eval')
        self._setup_dataloaders(
            train_dts=None,
            val_dts=dataset,
            from_wsi=from_wsi,
            roi_method=roi_method,
            pool=pool)

        # Generate performance metrics
        log.info('Performing evaluation...')
        metrics = self._val_metrics(
            label='eval',
            reduce_method=reduce_method,
            save_predictions=save_predictions
        )
        results = {'eval': {
            k: v for k, v in metrics.items() if k != 'val_metrics'
        }}
        results['eval'].update(metrics['val_metrics'])  # type: ignore
        results_str = json.dumps(results['eval'], indent=2, sort_keys=True)
        log.info(f"Evaluation metrics: {results_str}")
        results_log = os.path.join(self.outdir, 'results_log.csv')
        sf.util.update_results_log(results_log, 'eval_model', results)

        if self.neptune_run:
            self.neptune_run['eval/results'] = results['eval']
            self.neptune_run.stop()
        self._close_dataloaders()
        if pool is not None:
            pool.close()
        return results

    def train(
        self,
        train_dts: "sf.Dataset",
        val_dts: "sf.Dataset",
        log_frequency: int = 20,
        validate_on_batch: int = 0,
        validation_batch_size: Optional[int] = None,
        validation_steps: int = 50,
        starting_epoch: int = 0,
        ema_observations: int = 20,
        ema_smoothing: int = 2,
        use_tensorboard: bool = True,
        steps_per_epoch_override: int = 0,
        save_predictions: Union[bool, str] = 'parquet',
        save_model: bool = True,
        resume_training: Optional[str] = None,
        pretrain: Optional[str] = 'imagenet',
        checkpoint: Optional[str] = None,
        save_checkpoints: bool = False,
        multi_gpu: bool = False,
        norm_fit: Optional[NormFit] = None,
        reduce_method: str = 'average',
        seed: int = 0,
        from_wsi: bool = False,
        roi_method: str = 'auto',
    ) -> Dict[str, Any]:
        """Builds and trains a model from hyperparameters.

        Args:
            train_dts (:class:`slideflow.dataset.Dataset`): Training dataset.
            val_dts (:class:`slideflow.dataset.Dataset`): Validation dataset.
            log_frequency (int, optional): How frequent to update Tensorboard
                logs, in batches. Defaults to 100.
            validate_on_batch (int, optional): Validation will be performed
                every N batches. Defaults to 0.
            validation_batch_size (int, optional): Validation batch size.
                Defaults to same as training (per self.hp).
            validation_steps (int, optional): Number of batches to use for each
                instance of validation. Defaults to 200.
            starting_epoch (int, optional): Starts training at this epoch.
                Defaults to 0.
            ema_observations (int, optional): Number of observations over which
                to perform exponential moving average smoothing.
                Defaults to 20.
            ema_smoothing (int, optional): Exponential average smoothing value.
                Defaults to 2.
            use_tensoboard (bool, optional): Enable tensorboard callbacks.
                Defaults to False.
            steps_per_epoch_override (int, optional): Manually set the number
                of steps per epoch. Defaults to None.
            save_predictions (bool or str, optional): Save tile, slide, and
                patient-level predictions at each evaluation. May be 'csv',
                'feather', or 'parquet'. If False, will not save predictions.
                Defaults to 'parquet'.
            save_model (bool, optional): Save models when evaluating at
                specified epochs. Defaults to False.
            resume_training (str, optional): Not applicable to PyTorch backend.
                Included as argument for compatibility with Tensorflow backend.
                Will raise NotImplementedError if supplied.
            pretrain (str, optional): Either 'imagenet' or path to Tensorflow
                model from which to load weights. Defaults to 'imagenet'.
            checkpoint (str, optional): Path to cp.ckpt from which to load
                weights. Defaults to None.
            norm_fit (Dict[str, np.ndarray]): Normalizer fit, mapping fit
                parameters (e.g. target_means, target_stds) to values
                (np.ndarray). If not provided, will fit normalizer using
                model params (if applicable). Defaults to None.
            reduce_method (str, optional): Reduction method for calculating
                slide-level and patient-level predictions for categorical outcomes.
                Either 'average' or 'proportion'. If 'average', will reduce with
                average of each logit across tiles. If 'proportion', will convert
                tile predictions into onehot encoding then reduce by averaging
                these onehot values. Defaults to 'average'.
            seed (int): Set numpy random seed. Defaults to 0.
            from_wsi (bool): Generate predictions from tiles dynamically
                extracted from whole-slide images, rather than TFRecords.
                Defaults to False (use TFRecords).
            roi_method (str): ROI method to use if from_wsi=True (ignored if
                from_wsi=False).  Either 'inside', 'outside', 'auto', 'ignore'.
                If 'inside' or 'outside', will extract tiles in/out of an ROI,
                and raise errors.MissingROIError if an ROI is not available.
                If 'auto', will extract tiles inside an ROI if available,
                and across the whole-slide if no ROI is found.
                If 'ignore', will extract tiles across the whole-slide
                regardless of whether an ROI is available.
                Defaults to 'auto'.

        Returns:
            Dict:   Nested dict containing metrics for each evaluated epoch.
        """
        if resume_training is not None:
            raise NotImplementedError(
                "PyTorch backend does not support `resume_training`; "
                "please use `checkpoint`"
            )
        if save_checkpoints:
            log.warning(
                "The argument save_checkpoints is ignored when training models "
                "in the PyTorch backend. To save a model throughout training, "
                "use the `epochs` hyperparameter."
            )
        results = {'epochs': defaultdict(dict)}  # type: Dict[str, Any]
        starting_epoch = max(starting_epoch, 1)
        self._detect_patients(train_dts, val_dts)
        self._reset_training_params()
        self.validation_batch_size = validation_batch_size
        self.validate_on_batch = validate_on_batch
        self.validation_steps = validation_steps
        self.ema_observations = ema_observations
        self.ema_smoothing = ema_smoothing
        self.log_frequency = log_frequency
        self.use_tensorboard = use_tensorboard

        # Verify image format across datasets.
        img_format = self._verify_img_format(train_dts, val_dts)
        if img_format and self.config['img_format'] is None:
            self.config['img_format'] = img_format
            sf.util.write_json(self.config, join(self.outdir, 'params.json'))

        if self.use_tensorboard:
            from google.protobuf import __version__ as protobuf_version
            if version.parse(protobuf_version) >= version.parse('3.21'):
                log.warning(
                    "Tensorboard is incompatible with protobuf >= 3.21."
                    "Downgrade protobuf to enable tensorboard logging."
                )
                self.use_tensorboard = False

        if from_wsi and sf.slide_backend() == 'libvips':
            pool = mp.Pool(
                sf.util.num_cpu(default=8),
                initializer=sf.util.set_ignore_sigint
            )
        elif from_wsi:
            pool = mp.dummy.Pool(sf.util.num_cpu(default=8))
        else:
            pool = None

        # Validate early stopping parameters
        self._validate_early_stop()

        # Fit normalizer to dataset, if applicable
        self._fit_normalizer(norm_fit)
        if self.normalizer and self.hp.normalizer_source == 'dataset':
            self.normalizer.fit(train_dts)

        if self.normalizer:
            config_path = join(self.outdir, 'params.json')
            if not os.path.exists(config_path):
                config = {
                    'slideflow_version': sf.__version__,
                    'hp': self.hp.to_dict(),
                    'backend': sf.backend()
                }
            else:
                config = sf.util.load_json(config_path)
            config['norm_fit'] = self.normalizer.get_fit(as_list=True)
            sf.util.write_json(config, config_path)

        # Training preparation
        if steps_per_epoch_override:
            self.steps_per_epoch = steps_per_epoch_override
            log.info(f"Setting steps per epoch = {steps_per_epoch_override}")
        else:
            self.steps_per_epoch = train_dts.num_tiles // self.hp.batch_size
            log.info(f"Steps per epoch = {self.steps_per_epoch}")
        if self.use_tensorboard:
            # Delayed import due to protobuf version conflicts.

            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.outdir, flush_secs=60)
        self._log_manifest(train_dts, val_dts)

        # Prepare neptune run
        self._prepare_neptune_run(train_dts, 'train')

        # Build model
        self._build_model(checkpoint, pretrain)
        assert self.model is not None

        # Print model summary
        self._print_model_summary(train_dts)

        # Multi-GPU
        if multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        # Setup dataloaders
        self._setup_dataloaders(
            train_dts=train_dts,
            val_dts=val_dts,
            mid_train_val=True,
            roi_method=roi_method,
            from_wsi=from_wsi,
            pool=pool)

        # Model parameters and optimizer
        self._prepare_optimizers_and_loss()

        # === Epoch loop ======================================================
        for self.epoch in range(starting_epoch, max(self.hp.epochs)+1):
            np.random.seed(seed+self.epoch)
            log.info(f'[bold]Epoch {self.epoch}/{max(self.hp.epochs)}')

            # Training loop ---------------------------------------------------
            self.epoch_records = 0
            self.running_loss = 0.0
            self.step = 1
            self.running_corrects = self._empty_corrects()  # type: ignore
            self.model.train()
            pb = Progress(
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
                ImgBatchSpeedColumn(self.hp.batch_size),
                transient=sf.getLoggingLevel()>20
            )
            task = pb.add_task("Training...", total=self.steps_per_epoch)
            pb.start()
            with sf.util.cleanup_progress(pb):
                while self.step <= self.steps_per_epoch:
                    self._training_step(pb)
                    if self.early_stop:
                        break
                    self._mid_training_validation()
                    self.step += 1
                    self.global_step += 1

            # Update and log epoch metrics ------------------------------------
            loss = self.running_loss / self.epoch_records
            epoch_metrics = {'train_metrics': {'loss': loss}}
            if self.hp.model_type() == 'categorical':
                acc, acc_desc = self._calculate_accuracy(
                    self.running_corrects, self.epoch_records
                )
                epoch_metrics['train_metrics'].update({
                    'accuracy': self._accuracy_as_numpy(acc)  # type: ignore
                })
            else:
                acc, acc_desc = 0, ''  # type: ignore
            results['epochs'][f'epoch{self.epoch}'].update(epoch_metrics)
            self._log_epoch('train', self.epoch, loss, acc_desc)
            self._log_to_neptune(loss, acc, 'train', 'epoch')
            if save_model and (self.epoch in self.hp.epochs or self.early_stop):
                self._save_model()

            # Full evaluation -------------------------------------------------
            # Perform full evaluation if the epoch is one of the
            # predetermined epochs at which to save/eval a model
            if 'val' in self.dataloaders and self.epoch in self.hp.epochs:
                epoch_res = self._val_metrics(
                    save_predictions=save_predictions,
                    reduce_method=reduce_method,
                    label=f'val_epoch{self.epoch}',
                )
                results['epochs'][f'epoch{self.epoch}'].update(epoch_res)

            # Early stopping --------------------------------------------------
            if self.early_stop:
                break

        # === [end epoch loop] ================================================

        if self.neptune_run:
            self.neptune_run['sys/tags'].add('training_complete')
            self.neptune_run.stop()
        self._close_dataloaders()
        if pool is not None:
            pool.close()
        return results


class LinearTrainer(Trainer):

    """Extends the base :class:`slideflow.model.Trainer` class to add support
    for linear outcomes. Requires that all outcomes be linear, with appropriate
    linear loss function. Uses R-squared as the evaluation metric, rather
    than AUROC.

    In this case, for the PyTorch backend, the linear outcomes support is
    already baked into the base Trainer class, so no additional modifications
    are required. This class is written to inherit the Trainer class without
    modification to maintain consistency with the Tensorflow backend.
    """

    _model_type = 'linear'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CPHTrainer(Trainer):

    """Cox proportional hazards (CPH) models are not yet implemented, but are
    planned for a future update."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

# -----------------------------------------------------------------------------

class Features(BaseFeatureExtractor):
    """Interface for obtaining predictions and features from intermediate layer
    activations from Slideflow models.

    Use by calling on either a batch of images (returning outputs for a single
    batch), or by calling on a :class:`slideflow.WSI` object, which will
    generate an array of spatially-mapped activations matching the slide.

    Examples
        *Calling on batch of images:*

        .. code-block:: python

            interface = Features('/model/path', layers='postconv')
            for image_batch in train_data:
                # Return shape: (batch_size, num_features)
                batch_features = interface(image_batch)

        *Calling on a slide:*

        .. code-block:: python

            slide = sf.slide.WSI(...)
            interface = Features('/model/path', layers='postconv')
            # Return shape:
            # (slide.grid.shape[0], slide.grid.shape[1], num_features)
            activations_grid = interface(slide)

    Note:
        When this interface is called on a batch of images, no image processing
        or stain normalization will be performed, as it is assumed that
        normalization will occur during data loader image processing. When the
        interface is called on a `slideflow.WSI`, the normalization strategy
        will be read from the model configuration file, and normalization will
        be performed on image tiles extracted from the WSI. If this interface
        was created from an existing model and there is no model configuration
        file to read, a slideflow.norm.StainNormalizer object may be passed
        during initialization via the argument `wsi_normalizer`.

    """

    def __init__(
        self,
        path: Optional[str],
        layers: Optional[Union[str, List[str]]] = 'postconv',
        *,
        include_preds: bool = False,
        mixed_precision: bool = True,
        device: Optional[torch.device] = None,
        apply_softmax: Optional[bool] = None,
        pooling: Optional[Any] = None,
        load_method: str = 'weights',
    ):
        """Creates an activations interface from a saved slideflow model which
        outputs feature activations at the designated layers.

        Intermediate layers are returned in the order of layers.
        predictions are returned last.

        Args:
            path (str): Path to saved Slideflow model.
            layers (list(str), optional): Layers from which to generate
                activations.  The post-convolution activation layer is accessed
                via 'postconv'. Defaults to 'postconv'.
            include_preds (bool, optional): Include predictions in output. Will be
                returned last. Defaults to False.
            mixed_precision (bool, optional): Use mixed precision.
                Defaults to True.
            device (:class:`torch.device`, optional): Device for model.
                Defaults to torch.device('cuda')
            apply_softmax (bool): Apply softmax transformation to model output.
                Defaults to True for categorical models, False for linear models.
            pooling (Callable or str, optional): PyTorch pooling function to use
                on feature layers. May be a string ('avg' or 'max') or a
                callable PyTorch function.
            load_method (str): Loading method to use when reading model.
                This argument is ignored in the PyTorch backend, as all models
                are loaded by first building the model with hyperparameters
                detected in ``params.json``, then loading weights with
                ``torch.nn.Module.load_state_dict()``. Defaults to
                'full' (ignored).
        """
        super().__init__('torch', include_preds=include_preds)
        if layers and isinstance(layers, str):
            layers = [layers]
        self.layers = layers
        self.path = path
        self.apply_softmax = apply_softmax
        self.mixed_precision = mixed_precision
        self._model = None
        self._pooling = None
        self._include_preds = None

        # Transformation for standardizing uint8 images to float32
        self.transform = torchvision.transforms.Lambda(lambda x: x / 127.5 - 1)

        # Hook for storing layer activations during model inference
        self.activation = {}  # type: Dict[Any, Tensor]

        # Configure device
        self.device = torch_utils.get_device(device)

        if path is not None:
            config = sf.util.get_model_config(path)
            if 'img_format' in config:
                self.img_format = config['img_format']
            self.hp = ModelParams()  # type: Optional[ModelParams]
            self.hp.load_dict(config['hp'])
            self.wsi_normalizer = self.hp.get_normalizer()
            if 'norm_fit' in config and config['norm_fit'] is not None:
                self.wsi_normalizer.set_fit(**config['norm_fit'])  # type: ignore
            self.tile_px = self.hp.tile_px
            self._model = self.hp.build_model(
                num_classes=len(config['outcome_labels'])
            )
            if apply_softmax is None:
                self.apply_softmax = True if config['model_type'] == 'categorical' else False
                log.debug(f"Using apply_softmax={self.apply_softmax}")
            self._model.load_state_dict(torch.load(path))
            self._model.to(self.device)
            self._model.eval()
            if self._model.__class__.__name__ == 'ModelWrapper':
                self.model_type = self._model.model.__class__.__name__
            else:
                self.model_type = self._model.__class__.__name__
            self._build(pooling=pooling)

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        tile_px: int,
        layers: Optional[Union[str, List[str]]] = 'postconv',
        *,
        include_preds: bool = False,
        mixed_precision: bool = True,
        wsi_normalizer: Optional["StainNormalizer"] = None,
        apply_softmax: bool = True,
        pooling: Optional[Any] = None
    ):
        """Creates an activations interface from a loaded slideflow model which
        outputs feature activations at the designated layers.

        Intermediate layers are returned in the order of layers.
        predictions are returned last.

        Args:
            model (:class:`tensorflow.keras.models.Model`): Loaded model.
            tile_px (int): Width/height of input image size.
            layers (list(str), optional): Layers from which to generate
                activations.  The post-convolution activation layer is accessed
                via 'postconv'. Defaults to 'postconv'.
            include_preds (bool, optional): Include predictions in output. Will be
                returned last. Defaults to False.
            mixed_precision (bool, optional): Use mixed precision.
                Defaults to True.
            wsi_normalizer (:class:`slideflow.norm.StainNormalizer`): Stain
                normalizer to use on whole-slide images. Is not used on
                individual tile datasets via __call__. Defaults to None.
            apply_softmax (bool): Apply softmax transformation to model output.
                Defaults to True.
            pooling (Callable or str, optional): PyTorch pooling function to use
                on feature layers. May be a string ('avg' or 'max') or a
                callable PyTorch function.
        """
        device = next(model.parameters()).device
        if include_preds is not None:
            kw = dict(include_preds=include_preds)
        else:
            kw = dict()
        obj = cls(
            None,
            layers,
            mixed_precision=mixed_precision,
            device=device,
            **kw
        )
        if isinstance(model, torch.nn.Module):
            obj._model = model
            obj._model.eval()
        else:
            raise errors.ModelError("Model is not a valid PyTorch model.")
        obj.hp = None
        if obj._model.__class__.__name__ == 'ModelWrapper':
            obj.model_type = obj._model.model.__class__.__name__
        else:
            obj.model_type = obj._model.__class__.__name__
        obj.tile_px = tile_px
        obj.wsi_normalizer = wsi_normalizer
        obj.apply_softmax = apply_softmax
        obj._build(pooling=pooling)
        return obj

    def __call__(
        self,
        inp: Union[Tensor, "sf.WSI"],
        **kwargs
    ) -> Optional[Union[List[Tensor], np.ndarray]]:
        """Process a given input and return activations and/or predictions. Expects
        either a batch of images or a :class:`slideflow.slide.WSI` object.

        When calling on a `WSI` object, keyword arguments are passed to
        :meth:`slideflow.WSI.build_generator()`.

        """
        if isinstance(inp, sf.slide.WSI):
            return self._predict_slide(inp, **kwargs)
        else:
            return self._predict(inp, **kwargs)

    def __repr__(self):
        return ("{}(\n".format(self.__class__.__name__) +
                "    path={!r},\n".format(self.path) +
                "    layers={!r},\n".format(self.layers) +
                "    include_preds={!r},\n".format(self.include_preds) +
                "    apply_softmax={!r},\n".format(self.apply_softmax) +
                "    pooling={!r},\n".format(self._pooling) +
                ")")

    def _predict_slide(
        self,
        slide: "sf.WSI",
        *,
        img_format: str = 'auto',
        batch_size: int = 32,
        dtype: type = np.float16,
        grid: Optional[np.ndarray] = None,
        shuffle: bool = False,
        show_progress: bool = True,
        callback: Optional[Callable] = None,
        normalizer: Optional[Union[str, "StainNormalizer"]] = None,
        normalizer_source: Optional[str] = None,
        **kwargs
    ) -> Optional[np.ndarray]:
        """Generate activations from slide => activation grid array."""

        # Check image format
        if img_format == 'auto' and self.img_format is None:
            raise ValueError(
                'Unable to auto-detect image format (png or jpg). Set the '
                'format by passing img_format=... to the call function.'
            )
        elif img_format == 'auto':
            assert self.img_format is not None
            img_format = self.img_format

        return sf.model.extractors.features_from_slide(
            self,
            slide,
            img_format=img_format,
            batch_size=batch_size,
            dtype=dtype,
            grid=grid,
            shuffle=shuffle,
            show_progress=show_progress,
            callback=callback,
            normalizer=(normalizer if normalizer else self.wsi_normalizer),
            normalizer_source=normalizer_source,
            preprocess_fn=self.transform,
            **kwargs
        )

    def _predict(self, inp: Tensor, no_grad: bool = True) -> List[Tensor]:
        """Return activations for a single batch of images."""
        assert torch.is_floating_point(inp), "Input tensor must be float"
        _mp = (self.mixed_precision and self.device.type in ('cuda', 'cpu'))
        with autocast(self.device.type, mixed_precision=_mp):  # type: ignore
            with torch.no_grad() if no_grad else no_scope():
                inp = inp.to(self.device).to(memory_format=torch.channels_last)
                logits = self._model(inp)
                if isinstance(logits, (tuple, list)) and self.apply_softmax:
                    logits = [softmax(l, dim=1) for l in logits]
                elif self.apply_softmax:
                    logits = softmax(logits, dim=1)

        layer_activations = []
        if self.layers:
            for la in self.layers:
                act = self.activation[la]
                if la == 'postconv':
                    act = self._postconv_processing(act)
                layer_activations.append(act)
        if self.include_preds:
            layer_activations += [logits]
        self.activation = {}
        return layer_activations

    def _get_postconv(self):
        """Returns post-convolutional layer."""

        if self.model_type == 'ViT':
            return self._model.to_latent
        if self.model_type in ('ResNet', 'Inception3', 'GoogLeNet'):
            return self._model.avgpool
        if self.model_type in ('AlexNet', 'SqueezeNet', 'VGG', 'MobileNetV2',
                               'MobileNetV3', 'MNASNet'):
            if self._model.classifier.__class__.__name__ == 'Identity':
                return self._model.classifier
            else:
                return next(self._model.classifier.children())
        if self.model_type == 'DenseNet':
            return self._model.features.norm5
        if self.model_type == 'ShuffleNetV2':
            return list(self._model.conv5.children())[1]
        if self.model_type == 'Xception':
            return self._model.bn4
        raise errors.FeaturesError(f"'postconv' layer not configured for "
                                   f"model type {self.model_type}")

    def _postconv_processing(self, output: Tensor) -> Tensor:
        """Applies processing (pooling, resizing) to post-conv outputs,
        to convert output to the shape (batch_size, num_features)"""

        def pool(x):
            return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))

        def squeeze(x):
            return x.view(x.size(0), -1)

        if self.model_type in ('ViT', 'AlexNet', 'VGG', 'MobileNetV2',
                               'MobileNetV3', 'MNASNet'):
            return output
        if self.model_type in ('ResNet', 'Inception3', 'GoogLeNet'):
            return squeeze(output)
        if self.model_type in ('SqueezeNet', 'DenseNet', 'ShuffleNetV2',
                               'Xception'):
            return squeeze(pool(output))
        return output

    def _build(self, pooling: Optional[Any] = None) -> None:
        """Builds the interface model that outputs feature activations at the
        designated layers and/or predictions. Intermediate layers are returned in
        the order of layers. predictions are returned last.

        Args:
            pooling (Callable or str, optional): PyTorch pooling function to use
                on feature layers. May be a string ('avg' or 'max') or a
                callable PyTorch function.
        """

        self._pooling = pooling

        if isinstance(pooling, str):
            if pooling == 'avg':
                pooling = lambda x: torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            elif pooling == 'max':
                pooling = lambda x: torch.nn.functional.adaptive_max_pool2d(x, (1, 1))
            else:
                raise ValueError(f"Unrecognized pooling value {pooling}. "
                                 "Expected 'avg', 'max', or custom Tensor op.")

        self.activation = {}

        def squeeze(x):
            return x.view(x.size(0), -1)

        def get_activation(name):
            def hook(model, input, output):
                if len(output.shape) == 4 and pooling is not None:
                    self.activation[name] = squeeze(pooling(output)).detach()
                else:
                    self.activation[name] = output.detach()
            return hook

        if isinstance(self.layers, list):
            for la in self.layers:
                if la == 'postconv':
                    self._get_postconv().register_forward_hook(
                        get_activation('postconv')
                    )
                else:
                    la_out = torch_utils.get_module_by_name(self._model, la)
                    la_out.register_forward_hook(
                        get_activation(la)
                    )
        elif self.layers is not None:
            raise errors.FeaturesError(f"Unrecognized type {type(self.layers)}"
                                       " for self.layers")

        # Calculate output and layer sizes
        rand_data = torch.rand(1, 3, self.tile_px, self.tile_px)
        output = self._model(rand_data.to(self.device))
        if isinstance(output, (tuple, list)) and self.include_preds:
            log.warning("Multi-categorical outcomes is experimental "
                        "for this interface.")
            self.num_classes = sum(o.shape[1] for o in output)
            self.num_outputs = len(output)
        elif self.include_preds:
            self.num_classes = output.shape[1]
            self.num_outputs = 1
        else:
            self.num_classes = 0
            self.num_outputs = 0
        self.num_features = sum([f.shape[1] for f in self.activation.values()])

        if self.include_preds:
            log.debug(f'Number of classes: {self.num_classes}')
        log.debug(f'Number of activation features: {self.num_features}')

    def dump_config(self):
        return {
            'class': 'slideflow.model.torch.Features',
            'kwargs': {
                'path': self.path,
                'layers': self.layers,
                'include_preds': self.include_preds,
                'apply_softmax': self.apply_softmax,
                'pooling': self._pooling
            }
        }


class UncertaintyInterface(Features):

    def __init__(
        self,
        path: Optional[str],
        layers: Optional[Union[str, List[str]]] = 'postconv',
        *,
        mixed_precision: bool = True,
        device: Optional[torch.device] = None,
        apply_softmax: Optional[bool] = None,
        pooling: Optional[Any] = None,
        load_method: str = 'weights',
    ) -> None:
        super().__init__(
            path,
            layers=layers,
            mixed_precision=mixed_precision,
            device=device,
            apply_softmax=apply_softmax,
            pooling=pooling,
            load_method=load_method,
            include_preds=True
        )
        if self._model is not None:
            torch_utils.enable_dropout(self._model)
        # TODO: As the below to-do suggests, this should be updated
        # for multi-class
        self.num_uncertainty = 1
        if self.num_classes > 2:
            log.warn("UncertaintyInterface not yet implemented for multi-class"
                     " models")

    @classmethod
    def from_model(cls, *args, **kwargs):
        if 'include_preds' in kwargs and not kwargs['include_preds']:
            raise ValueError("UncertaintyInterface requires include_preds=True")
        kwargs['include_preds'] = None
        obj = super().from_model(*args, **kwargs)
        torch_utils.enable_dropout(obj._model)
        return obj

    def __repr__(self):
        return ("{}(\n".format(self.__class__.__name__) +
                "    path={!r},\n".format(self.path) +
                "    layers={!r},\n".format(self.layers) +
                "    apply_softmax={!r},\n".format(self.apply_softmax) +
                "    pooling={!r},\n".format(self._pooling) +
                ")")

    def _predict(self, inp: Tensor, no_grad: bool = True) -> List[Tensor]:
        """Return activations (mean), predictions (mean), and uncertainty
        (stdev) for a single batch of images."""

        assert torch.is_floating_point(inp), "Input tensor must be float"
        _mp = (self.mixed_precision and self.device.type in ('cuda', 'cpu'))

        out_pred_drop = [[] for _ in range(self.num_outputs)]
        if self.layers:
            out_act_drop = [[] for _ in range(len(self.layers))]
        for _ in range(30):
            with autocast(self.device.type, mixed_precision=_mp):  # type: ignore
                with torch.no_grad() if no_grad else no_scope():
                    inp = inp.to(self.device)
                    inp = inp.to(memory_format=torch.channels_last)
                    logits = self._model(inp)
                    if isinstance(logits, (tuple, list)) and self.apply_softmax:
                        logits = [softmax(l, dim=1) for l in logits]
                    elif self.apply_softmax:
                        logits = softmax(logits, dim=1)
                    for n in range(self.num_outputs):
                        out_pred_drop[n] += [
                            (logits[n] if self.num_outputs > 1 else logits)
                        ]

            layer_activations = []
            if self.layers:
                for la in self.layers:
                    act = self.activation[la]
                    if la == 'postconv':
                        act = self._postconv_processing(act)
                    layer_activations.append(act)
                for n in range(len(self.layers)):
                    out_act_drop[n].append(layer_activations[n]
                    )
            self.activation = {}

        for n in range(self.num_outputs):
            out_pred_drop[n] = torch.stack(out_pred_drop[n], axis=0)
        predictions = torch.mean(torch.cat(out_pred_drop), dim=0)

        # TODO: Only takes STDEV from first outcome category which works for
        # outcomes with 2 categories, but a better solution is needed
        # for num_categories > 2
        uncertainty = torch.std(torch.cat(out_pred_drop), dim=0)[:, 0]
        uncertainty = torch.unsqueeze(uncertainty, axis=-1)

        if self.layers:
            for n in range(self.layers):
                out_act_drop[n] = torch.stack(out_act_drop[n], axis=0)
            reduced_activations = [
                torch.mean(out_act_drop[n], dim=0)
                for n in range(len(self.layers))
            ]
            return reduced_activations + [predictions, uncertainty]
        else:
            return predictions, uncertainty

    def dump_config(self):
        return {
            'class': 'slideflow.model.torch.UncertaintyInterface',
            'kwargs': {
                'path': self.path,
                'layers': self.layers,
                'apply_softmax': self.apply_softmax,
                'pooling': self._pooling
            }
        }

# -----------------------------------------------------------------------------

def load(path: str) -> torch.nn.Module:
    """Load a model trained with Slideflow.

    Args:
        path (str): Path to saved model. Must be a model trained in Slideflow.

    Returns:
        torch.nn.Module: Loaded model.
    """
    config = sf.util.get_model_config(path)
    hp = ModelParams.from_dict(config['hp'])
    if len(config['outcomes']) == 1 or config['model_type'] == 'linear':
        num_classes = len(list(config['outcome_labels'].keys()))
    else:
        num_classes = {
            outcome: len(list(config['outcome_labels'][outcome].keys()))
            for outcome in config['outcomes']
        }
    model = hp.build_model(
        num_classes=num_classes,
        num_slide_features=0 if not config['input_feature_sizes'] else sum(config['input_feature_sizes']),
        pretrain=None
    )
    if not torch.cuda.is_available():
        kw = dict(map_location=torch.device('cpu'))
    else:
        kw = dict()
    model.load_state_dict(torch.load(path, **kw))
    return model


def lazy_load_pretrained(
    module: torch.nn.Module,
    to_load: str
) -> None:
    """Loads pretrained model weights into an existing module, ignoring
    incompatible Tensors.

    Args:
        module (torch.nn.Module): Destination module for weights.
        to_load (str, torch.nn.Module): Module with weights to load. Either
            path to PyTorch Slideflow model, or an existing PyTorch module.

    Returns:
        None
    """
    # Get state dictionaries
    current_model_dict = module.state_dict()
    if isinstance(to_load, str):
        loaded_state_dict = torch.load(to_load)
    else:
        loaded_state_dict = to_load.state_dict()

    # Only transfer valid states
    new_state_dict = {k:v if v.size()==current_model_dict[k].size()
                          else  current_model_dict[k]
                      for k,v in zip(current_model_dict.keys(),
                                     loaded_state_dict.values())}
    n_states = len(list(new_state_dict.keys()))
    log.info(f"Loaded {n_states} Tensor states from "
             f"pretrained model [green] {to_load}")
    module.load_state_dict(new_state_dict, strict=False)
