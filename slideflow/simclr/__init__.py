import types

from .simclr.tf2 import (data_util, data, lars_optimizer, metrics, model,
                         objective, resnet, run)
from .simclr.tf2.model import SimCLR
from .simclr.tf2.data import (SlideflowBuilder, build_distributed_dataset,
                              get_preprocess_fn)
from .simclr.tf2.run import (run_simclr, save, build_saved_model,
                             perform_evaluation, try_restore_from_checkpoint)

def get_args(**kwargs):
    args_dict = {
        'dataset': 'imagenet2012', # FIXME: ?
        'data_dir': None, # FIXME: ?
        'train_split': 'train',
        'eval_split': 'validation',
        'eval_batch_size': 256
    }
    print('this version2')
    for k in kwargs:
        args_dict[k] = kwargs[k]
    args = types.SimpleNamespace(**args_dict)
    return args
