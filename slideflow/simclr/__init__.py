from .simclr.tf2 import (
    data,
    data_util,
    lars_optimizer,
    metrics,
    model,
    objective,
    resnet,
    run,
    build_saved_model,
    load,
    perform_evaluation,
    run_simclr,
    save,
    try_restore_from_checkpoint)
from .simclr.tf2.model import SimCLR
from .simclr.tf2.data import (
    DatasetBuilder,
    build_distributed_dataset,
    get_preprocess_fn)
from .simclr.tf2.utils import load_model_args, get_args, SimCLR_Args

import tensorflow as tf
from typing import Union, Callable

# -----------------------------------------------------------------------------

class SimCLR_Generator:
    """Wrapper for SimCLR model to accelerate model inference.

    Improves model inference by wrapping with @tf.function.
    """
    def __init__(self, model: Union[str, Callable]):
        if isinstance(model, str):
            self.model = load(model)
        else:
            self.model = model

    @tf.function
    def generate(self, batch_images, **kwargs):
        return self.model(batch_images, **kwargs)

    def __call__(self, batch_images, **kwargs):
        return self.generate(batch_images, **kwargs)