from .train import train_mil, train_clam, train_fastai, build_fastai_learner
from .train._legacy import legacy_train_clam
from ._params import (
    build_config, TrainerConfig, TrainerConfigFastAI, TrainerConfigCLAM
)