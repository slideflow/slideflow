"""Model and trainer configuration classes for MIL models."""

from torch import nn
from typing import Optional, Union, Callable
from slideflow.mil.models import Attention_MIL


def mil_config(model: Union[str, Callable], trainer: str = 'fastai', **kwargs):
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


class TrainerConfig(DictConfig):

    def __init__(self, *args, **kwargs):
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
        return self.model_config.model_fn

    @property
    def loss_fn(self):
        return self.model_config.loss_fn

    def to_dict(self):
        d = super().to_dict()
        if self.model_config is None:
            return d
        else:
            d.update(self.model_config.to_dict())
            del d['model_config']
            return d

    def json_dump(self):
        return dict(
            trainer=('fastai' if isinstance(self, TrainerConfigFastAI) else 'clam'),
            params=self.to_dict()
        )

# -----------------------------------------------------------------------------

class TrainerConfigFastAI(TrainerConfig):
    def __init__(
        self,
        model: Union[str, Callable] = 'attention_mil',
        lr: Optional[float] = None,
        lr_max: Optional[float] = None,
        wd: float = 1e-5,
        bag_size: int = 512,
        fit_one_cycle: bool = True,
        epochs: int = 32,
        batch_size: int = 64,
        **kwargs
    ):
        self.lr = lr
        self.lr_max = lr_max
        self.wd = wd
        self.bag_size = bag_size
        self.fit_one_cycle = fit_one_cycle
        self.epochs = epochs
        self.batch_size = batch_size
        if model in ModelConfigCLAM.valid_models:
            self.model_config = ModelConfigCLAM(model=model, **kwargs)
        else:
            self.model_config = ModelConfigFastAI(model=model, **kwargs)


class TrainerConfigCLAM(TrainerConfig):
    def __init__(
        self,
        num_splits=1,
        k=3,
        k_start=-1,
        k_end=-1,
        max_epochs=20,
        lr=1e-4,
        reg=1e-5,
        label_frac=1,
        weighted_sample=False,
        log_data=False,
        testing=False,
        early_stopping=False,
        subtyping=False,
        seed=1,
        results_dir=None,
        n_classes=None,
        split_dir=None,
        data_root_dir=None,
        micro_average=False,
        **kwargs
    ):
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
        bag_loss='ce',
        bag_weight=0.7,
        model='clam_sb',
        model_size='small',
        dropout=False,
        opt='adam',
        inst_loss=None,
        no_inst_cluster=False,
        B=8,
    ):
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
        use_lens: Optional[bool] = None
    ) -> None:
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
