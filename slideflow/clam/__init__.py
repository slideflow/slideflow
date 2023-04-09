# Dummy submodule here for backwards compatibility.
# CLAM has been moved to slideflow.mil.clam

from ..mil.clam import (
    detect_num_features,
    evaluate,
    get_args,
    main,
    train,
    seed_torch,
    CLAM_Args,
    create_attention,
    datasets,
    utils
)
from ..mil.clam.create_attention import export_attention
from ..mil.clam.datasets import CLAM_Dataset
from ..mil.clam.datasets.dataset_generic import Generic_MIL_Dataset