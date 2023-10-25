"""Module for building pretrained feature extractors."""

from ._registry import (list_extractors, list_torch_extractors,
                        list_tensorflow_extractors, is_extractor,
                        is_torch_extractor, is_tensorflow_extractor,
                        register_tf, register_torch)
from ._factory import (build_feature_extractor, build_torch_feature_extractor,
                       build_tensorflow_feature_extractor, rebuild_extractor)
from ._factory_tensorflow import TensorflowImagenetLayerExtractor
from ._factory_torch import TorchImagenetLayerExtractor
from ._slide import features_from_slide