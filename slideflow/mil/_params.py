"""Model and trainer configuration for MIL models."""

from torch import nn
from typing import Optional, Union, Callable
from slideflow.mil.models import Attention_MIL


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
    if model not in ModelConfigCLAM.valid_models and trainer == 'clam':
        raise ValueError(f"Model {model} incompatible with trainer {trainer}")
    if trainer == 'fastai':
        return TrainerConfigFastAI(model=model, **kwargs)
    elif trainer == 'clam':
        return TrainerConfigCLAM(model=model, **kwargs)
    else:
        raise ValueError(f"Unrecognized trainer {trainer}, expected fastai or clam.")

# -----------------------------------------------------------------------------

class DictConfig:
    def __init__(self):
        pass

    def to_dict(self):
        return {k:v for k,v in vars(self).items()
                if k not in ('self', 'model_fn', 'loss_fn') and not k.startswith('_')}


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
            trainer=('fastai' if isinstance(self, TrainerConfigFastAI) else 'clam'),
            params=self.to_dict()
        )

# -----------------------------------------------------------------------------

class TrainerConfigFastAI(_TrainerConfig):
    def __init__(
        self,
        model: Union[str, Callable] = 'attention_mil',
        *,
        lr: Optional[float] = None,
        wd: float = 1e-5,
        bag_size: int = 512,
        fit_one_cycle: bool = True,
        epochs: int = 32,
        batch_size: int = 64,
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
            **kwargs: All additional keyword arguments are passed to either
                :class:`slideflow.mil.ModelConfigCLAM` for CLAM models, or
                :class:`slideflow.mil.ModelConfigFastAI` for all other models.

        """
        self.lr = lr
        self.wd = wd
        self.bag_size = bag_size
        self.fit_one_cycle = fit_one_cycle
        self.epochs = epochs
        self.batch_size = batch_size
        if model in ModelConfigCLAM.valid_models:
            self.model_config = ModelConfigCLAM(model=model, **kwargs)
        else:
            self.model_config = ModelConfigFastAI(model=model, **kwargs)


class TrainerConfigCLAM(_TrainerConfig):
    def __init__(
        self,
        *,
        num_splits: int = 1,   # Unused; kept for backwards compatibility
        k: int = 3,
        k_start: int = -1,
        k_end: int = -1,
        max_epochs: int = 20,
        lr: float = 1e-4,
        reg: float = 1e-5,
        label_frac: float = 1,
        weighted_sample: bool = False,
        log_data: bool = False,
        testing: bool = False,
        early_stopping: bool = False,
        subtyping: bool = False,
        seed: int = 1,
        results_dir: Optional[str] = None,  # Unused; kept for compatibility
        n_classes: Optional[int] = None,
        split_dir=None,
        data_root_dir=None,
        micro_average=False,
        **kwargs
    ):
        """Training configuration for the legacy CLAM trainer.

        This configures the legacy CLAM trainer. The FastAI trainer is
        preferred for all models, including CLAM.

        The configuration options for the legacy CLAM trainer are identical to
        the options in the `original CLAM paper <https://arxiv.org/abs/2004.09666>`_.

        Keyword args:
            k (int): Number of cross-fold splits. Defaults to 3.
            k_start (int): Starting cross-fold. Defaults to first cross-fold.
            k_end (int): Ending cross-fold. Defaults to ending after last
                cross-fold is done.
            max_epochs (int): Number of epochs to train. Defaults to 20.
            lr (float): Learning rate. Defaults to 1e-4.
            reg (float): Weight decay. Defaults to 1e-5.
            weighted_sample (bool): Equally sample from all outcome classes.
                Defaults to False.
            log_data (bool): Log to tensorboard. Defaults to False.
            early_stopping (bool): Stop the training if validation loss doesn't
                improve after 5 epochs. Will not trigger early stopping
                until epoch 50. Defaults to False.
            subtyping (bool): Whether this is a subtyping problem.
                Defaults to False.
            seed (int): Set the random seed. Defaults to 1.
            n_classes (int): Number of outcome classes. Defaults to None.
            micro_average (bool): Use micro averaging when calculate AUROC.
            **kwargs: All additional keyword arguments are passed to
                :class:`slideflow.mil.ModelConfigCLAM`.
        """
        for argname, argval in dict(locals()).items():
            if argname != 'kwargs':
                setattr(self, argname, argval)
        self.model_config = ModelConfigCLAM(**kwargs)

    def _to_clam_args(self):
        """Convert into CLAM_Args format (legacy support)."""
        from ..clam import CLAM_Args
        all_kw = self.to_dict()
        all_kw.update(self.model_config.to_dict())
        all_kw['model_type'] = all_kw['model']
        all_kw['drop_out'] = all_kw['dropout']
        del all_kw['model']
        del all_kw['dropout']
        return CLAM_Args(**all_kw)

# -----------------------------------------------------------------------------

class ModelConfigCLAM(DictConfig):

    valid_models = ['clam_sb', 'clam_mb', 'mil_fc_mc', 'mil_fc']

    def __init__(
        self,
        *,
        model: str = 'clam_sb',
        model_size: str = 'small',
        bag_loss: str = 'ce',
        bag_weight: float = 0.7,
        dropout: bool = False,
        opt: str = 'adam',
        inst_loss: str = 'ce',
        no_inst_cluster: bool = False,
        B: int = 8,
    ):
        """Model configuration for CLAM models.

        These configuration options are identical to the options in the
        `original CLAM paper <https://arxiv.org/abs/2004.09666>`_.

        Keyword args:
            model (str): Model. Either ``'clam_sb'``, ``'clam_mb'``,
                ``'mil_fc'``, or ``'mil_fc_mc'``. Defaults to ``'clam_sb'``.
            model_size (str): Size of the model. Available sizes include:

                ``clam_sb``

                .. list-table::
                    :header-rows: 0

                    * - small
                      - [1024, 512, 256]
                    * - big
                      - [1024, 512, 384]
                    * - multiscale
                      - [2048, 512, 256]
                    * - xception
                      - [2048, 256, 128]
                    * - xception_multi
                      - [1880, 128, 64]
                    * - xception_3800
                      - [3800, 512, 256]

                ``clam_mb``

                .. list-table::
                    :header-rows: 0

                    * - small
                      - [1024, 512, 256]
                    * - big
                      - [1024, 512, 384]
                    * - multiscale
                      - [2048, 512, 256]

                ``mil_fc``

                .. list-table::
                    :header-rows: 0

                    * - small
                      - [1024, 512]

                ``mil_fc_mc``

                .. list-table::
                    :header-rows: 0

                    * - small
                      - [1024, 512]

            bag_loss (str): Primary loss function. Either 'ce' or 'svm'.
                If 'ce', the model loss function is a cross entropy loss.
                If 'svm', the model loss is topk.SmoothTop1SVM.
                Defaults to 'ce'.
            bag_weight (float): Weight of the bag loss. The total loss is
                defined0 as ``W * loss + (1 - W) * instance_loss``, where
                ``W`` is the bag weight. Defaults to 0.7
            dropout (bool): Add dropout (p=0.25) after the attention layers.
                Defaults to False.
            opt (str): Optimizer. Either 'adam' (Adam optimizer) or 'sgd'
                (Stochastic Gradient Descent). Defaults to 'adam'.
            inst_loss (str): Instance loss function. Either 'ce' or 'svm'.
                If 'ce', the instance loss is a cross entropy loss.
                If 'svm', the loss is topk.SmoothTop1SVM.
                Defaults to 'ce'.
            no_inst_cluster (bool): Disable instance-level clustering.
                Defaults to False.
            B (int): Number of positive/negative patches to sample for
                instance-level training. Defaults to 8.

        """
        for argname, argval in dict(locals()).items():
            setattr(self, argname, argval)

    @property
    def model_fn(self):
        from .models import CLAM_MB, CLAM_SB, MIL_fc_mc, MIL_fc
        model_dict = {
            'clam_sb': CLAM_SB,
            'clam_mb': CLAM_MB,
            'mil_fc_mc': MIL_fc_mc,
            'mil_fc': MIL_fc
        }
        return model_dict[self.model]

    @property
    def loss_fn(self):
        from .clam.utils import loss_utils
        if self.model.startswith('clam'):
            return loss_utils.CrossEntropyWithInstanceLoss
        else:
            return loss_utils.CrossEntropyLoss


class ModelConfigFastAI(DictConfig):

    valid_models = ['attention_mil', 'transmil']

    def __init__(
        self,
        model: Union[str, Callable] = 'attention_mil',
        *,
        use_lens: Optional[bool] = None
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
        if use_lens is None and (model == 'attention_mil' or model is Attention_MIL):
            self.use_lens = True
        elif use_lens is None:
            self.use_lens = False
        else:
            self.use_lens = use_lens

    @property
    def model_fn(self):
        if not isinstance(self.model, str):
            return self.model
        elif self.model.lower() == 'attention_mil':
            from .models import Attention_MIL
            return Attention_MIL
        elif self.model.lower() == 'transmil':
            from .models import TransMIL
            return TransMIL
        else:
            raise ValueError(f"Unrecognized model {self.model}")

    @property
    def loss_fn(self):
        return nn.CrossEntropyLoss

    def to_dict(self):
        d = super().to_dict()
        if not isinstance(d['model'], str):
            d['model'] = d['model'].__name__
        return d
