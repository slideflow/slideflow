"""Model and trainer configuration for MIL models."""

from torch import nn
from typing import Optional, Union, Callable
from slideflow import log, errors


def mil_config(model: Union[str, Callable], trainer: str = 'fastai', **kwargs):
    """Create a multiple-instance learning (MIL) training configuration.

    All models by default are trained with the FastAI trainer. However, CLAM
    models can also be trained using the original, legacy CLAM trainer. This
    deprecated trainer has been kept for backwards compatibility; the
    FastAI trainer is preferred to all models, including CLAM.

    Args:
        model (str, Callable): Either the name of a model, or a custom torch
            module. Valid model names include ``"clam_sb"``, ``"clam_mb"``,
            ``"mil_fc"``, ``"mil_fc_mc"``, ``"attention_mil"``, and
            ``"transmil"``.
        trainer (str): Type of MIL trainer to use. Either 'fastai' or 'clam'.
            All models (including CLAM) can be trained with 'fastai'.
            The deprecated, legacy 'clam' trainer is only available for CLAM
            models, and has been kept for backwards compatibility.
            Defaults to 'fastai' (preferred).
        **kwargs: All additional keyword arguments are passed to either
            :class:`slideflow.mil.TrainerConfigCLAM` for CLAM models, or
            :class:`slideflow.mil.TrainerConfigFastAI` for all other models.

    """
    if trainer == 'fastai':
        return TrainerConfigFastAI(model=model, **kwargs)
    elif trainer == 'lightning':
        return TrainerConfigLightning(model=model, **kwargs)
    else:
        raise ValueError(f"Unrecognized trainer {trainer}, expected fastai.")

# -----------------------------------------------------------------------------

class DictConfig:
    def __init__(self):
        pass

    def to_dict(self):
        return {k:v for k,v in vars(self).items()
                if k not in (
                    'self',
                    'model_fn',
                    'loss_fn',
                    'build_model',
                    'is_multimodal'
                ) and not k.startswith('_')}



class _TrainerConfig(DictConfig):

    def __init__(self, *args, **kwargs):
        """Multiple-instance learning (MIL) training configuration.

        This configuration should not be created directly, but rather should
        be created through :func:`slideflow.mil.mil_config`, which will create
        and prepare an appropriate trainer configuration.

        """
        self.model_config = None
        super().__init__(*args, **kwargs)

    def __str__(self):
        out = f"{self.__class__.__name__}("
        for p, val in self.to_dict().items():
            if p != 'model_config':
                out += '\n  {}={!r}'.format(p, val)
        out += '\n)'
        return out

    @property
    def model_fn(self):
        """MIL model architecture (class/module)."""
        return self.model_config.model_fn

    @property
    def loss_fn(self):
        """MIL loss function."""
        return self.model_config.loss_fn

    @property
    def is_multimodal(self):
        """Whether the model is multimodal."""
        return self.model_config.is_multimodal

    def build_model(self, *args, **kwargs):
        """Build the mode."""
        if self.model_config.model_kwargs:
            model_kw = self.model_config.model_kwargs
        else:
            model_kw = dict()
        return self.model_config.model_fn(*args, **kwargs, **model_kw)

    def to_dict(self):
        """Converts this training configuration to a dictionary."""
        d = super().to_dict()
        if self.model_config is None:
            return d
        else:
            d.update(self.model_config.to_dict())
            del d['model_config']
            return d

    def json_dump(self):
        """Converts this training configuration to a JSON-compatible dict."""
        return dict(
            trainer='fastai',
            params=self.to_dict()
        )


# -----------------------------------------------------------------------------

class TrainerConfigLightning(_TrainerConfig):
    def __init__(
        self,
        model: Union[str, Callable] = 'attention_mil',
        *,
        aggregation_level: str = 'slide',
        lr: Optional[float] = None,
        wd: float = 1e-5,
        bag_size: int = 512,
        fit_one_cycle: bool = True,
        epochs: int = 32,
        batch_size: int = 64,
        drop_last: bool = True,
        save_monitor: str = 'val_loss',
        z_dim: int = 512,
        encoder_layers: int = 1,
        activation_function: str = 'ReLU',
        dropout_p: float = 0.2,
        task: str = 'classification',
        slide_level: bool = False,
        **kwargs,
    ):
        self.aggregation_level = aggregation_level
        self.lr = lr
        self.wd = wd
        self.bag_size = bag_size
        self.fit_one_cycle = fit_one_cycle
        self.epochs = epochs
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.save_monitor = save_monitor
        self.z_dim = z_dim
        self.encoder_layers = encoder_layers
        self.activation_function = activation_function
        self.dropout_p = dropout_p
        self.task = task
        self.slide_level = slide_level
        # keep same ModelConfigFastAI wiring
        from ._params import ModelConfigFastAI
        if self.task in ("classification", "survival_discrete"):
            self.model_config = ModelConfigFastAI(model=model, **kwargs)
        else:
            kwargs.pop('apply_softmax', None)
            self.model_config = ModelConfigFastAI(model=model, apply_softmax=False, **kwargs)

    def json_dump(self):
        return dict(trainer='lightning', params=self.to_dict())
# -----------------------------------------------------------------------------

class TrainerConfigFastAI(_TrainerConfig):
    def __init__(
        self,
        model: Union[str, Callable] = 'attention_mil',
        *,
        aggregation_level: str = 'slide',
        lr: Optional[float] = None,
        wd: float = 1e-5,
        bag_size: int = 512,
        fit_one_cycle: bool = True,
        epochs: int = 32,
        batch_size: int = 64,
        drop_last: bool = True,
        save_monitor: str = 'valid_loss',
        z_dim: int = 512,
        encoder_layers: int = 1,
        activation_function: str = 'ReLU',
        dropout_p: float = 0.2,
        task: str = 'classification',
        slide_level: bool = False,
        **kwargs
    ):
        r"""Training configuration for FastAI MIL models.

        This configuration should not be created directly, but rather should
        be created through :func:`slideflow.mil.mil_config`, which will create
        and prepare an appropriate trainer configuration.

        Args:
            model (str, Callable): Either the name of a model, or a custom torch
                module. Valid model names include ``"clam_sb"``, ``"clam_mb"``,
                ``"mil_fc"``, ``"mil_fc_mc"``, ``"attention_mil"``, and
                ``"transmil"``.

        Keyword args:
            aggregation_level (str): When equal to ``'slide'`` each bag
                contains tiles from a single slide. When equal to ``'patient'``
                tiles from all slides of a patient are grouped together.
            lr (float, optional): Learning rate. If ``fit_one_cycle=True``,
                this is the maximum learning rate. If None, uses the Leslie
                Smith `LR Range test <https://arxiv.org/abs/1506.01186>`_ to
                find an optimal learning rate. Defaults to None.
            wd (float): Weight decay. Only used if ``fit_one_cycle=False``.
                Defaults to 1e-5.
            bag_size (int): Bag size. Defaults to 512.
            fit_one_cycle (bool): Use `1cycle <https://sgugger.github.io/the-1cycle-policy.html>`_
                learning rate schedule. Defaults to True.
            epochs (int): Maximum number of epochs. Defaults to 32.
            batch_size (int): Batch size. Defaults to 64.
            drop_last (bool): Drop the last batch if it is smaller than the
                batch size. Defaults to True.
            **kwargs: All additional keyword arguments are passed to either
                :class:`slideflow.mil.ModelConfigCLAM` for CLAM models, or
                :class:`slideflow.mil.ModelConfigFastAI` for all other models.

        """
        self.aggregation_level = aggregation_level
        self.lr = lr
        self.wd = wd
        self.bag_size = bag_size
        self.fit_one_cycle = fit_one_cycle
        self.epochs = epochs
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.save_monitor = save_monitor
        self.z_dim = z_dim
        self.encoder_layers = encoder_layers
        self.activation_function = activation_function
        self.dropout_p = dropout_p
        self.task = task
        self.slide_level = slide_level

        #Check if task in kwargs
        if self.task is not None:
            # If not classification, apply_softmax=False
            if self.task == "classification" or self.task == 'survival_discrete':
                self.model_config = ModelConfigFastAI(model=model, **kwargs)
            else:
                #Drop apply softmax from kwargs
                kwargs.pop('apply_softmax', None)
                #If task is regression or survival, apply_softmax=False
                self.model_config = ModelConfigFastAI(model=model, apply_softmax=False, **kwargs)
        else:
            logging.warning("Task not specified. Assuming classification task.")
            self.model_config = ModelConfigFastAI(model=model, **kwargs)


class ModelConfigFastAI(DictConfig):

    valid_models = [
        'attention_mil',
        'transmil',
        'bistro.transformer',
        'mm_attention_mil',
    ]

    def __init__(
        self,
        model: Union[str, Callable] = 'attention_mil',
        *,
        use_lens: Optional[bool] = None,
        apply_softmax: bool = True,
        model_kwargs: Optional[dict] = None,
        validate: bool = False,
        **kwargs
    ) -> None:
        """Model configuration for a non-CLAM MIL model.

        Args:
            model (str, Callable): Either the name of a model, or a custom torch
                module. Valid model names include ``"attention_mil"`` and
                ``"transmil"``. Defaults to 'attention_mil'.

        Keyword args:
            use_lens (bool, optional): Whether the model expects a second
                argument to its ``.forward()`` function, an array with the
                bag size for each slide. If None, will default to True for
                ``'attention_mil'`` models and False otherwise.
                Defaults to None.

        """
        self.model = model
        self.apply_softmax = apply_softmax
        self.model_kwargs = model_kwargs

        if 'task' in kwargs:
            self.task = kwargs.pop('task', None)

        if use_lens is None and (hasattr(self.model_fn, 'use_lens')
                                 and self.model_fn.use_lens):
            self.use_lens = True
        elif use_lens is None:
            self.use_lens = False
        else:
            self.use_lens = use_lens
        if kwargs and validate:
            raise errors.UnrecognizedHyperparameterError("Unrecognized parameters: {}".format(
                ', '.join(list(kwargs.keys()))
            ))
        elif kwargs:
            log.warning("Ignoring unrecognized parameters: {}".format(
                ', '.join(list(kwargs.keys()))
            ))

    @property
    def model_fn(self):
        if not isinstance(self.model, str):
            return self.model
        elif self.model.lower() == 'attention_mil':
            from .models import Attention_MIL
            return Attention_MIL
        elif self.model.lower() == 'mm_attention_mil':
            from .models import MultiModal_Attention_MIL
            return MultiModal_Attention_MIL
        elif self.model.lower() == 'transmil':
            from .models import TransMIL
            return TransMIL
        elif self.model.lower() == 'bistro.transformer':
            from slideflow.mil.models.bistro import Transformer
            return Transformer
        else:
            from pathbench.models import aggregators
            # Look for the model in the aggregators module
            if hasattr(aggregators, self.model):
                #If so return the model class
                return getattr(aggregators, self.model)
            else:
                return self.model


    @property
    def loss_fn(self):
        return nn.CrossEntropyLoss

    @property
    def is_multimodal(self):
        if not isinstance(self.model, str):
            return False
        else:
            return (self.model.lower() == 'mm_attention_mil'
                or (hasattr(self.model_fn, 'is_multimodal')
                    and self.model_fn.is_multimodal))

    def to_dict(self):
        d = super().to_dict()
        if not isinstance(d['model'], str):
            d['model'] = d['model'].__name__
        return d
