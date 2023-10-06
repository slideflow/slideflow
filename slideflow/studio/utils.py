"""Utilities for Slideflow Studio."""

from typing import Any, List

import os
import slideflow as sf
import numpy as np
from os.path import join, exists
from slideflow import log

if sf.util.tf_available:
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
if sf.util.torch_available:
    import slideflow.model.torch

#----------------------------------------------------------------------------

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


#----------------------------------------------------------------------------

def prediction_to_string(
    predictions: np.ndarray,
    outcomes: List[str],
    is_categorical: bool
) -> str:
    """Convert a prediction array to a human-readable string."""
    #TODO: support multi-outcome models
    if is_categorical:
        return f'{outcomes[str(np.argmax(predictions))]} ({np.max(predictions)*100:.1f}%)'
    else:
        return f'{predictions[0]:.2f}'


def _load_umap_encoders(path, model) -> EasyDict:
    import tensorflow as tf

    layers = [d for d in os.listdir(path) if os.path.isdir(join(path, d))]
    log.debug("Layers found at path {} in _load_umap_encoders: {}".format(path, layers))
    features = sf.model.Features.from_model(
        model,
        include_preds=True,
        layers=layers,
        pooling='avg'
    )

    outputs = []
    for i, layer in enumerate(layers):
        # Add outputs for each UMAP encoder
        encoder = tf.keras.models.load_model(join(path, layer, 'encoder'))
        encoder._name = f'{layer}_encoder'
        outputs += [encoder(features.model.outputs[i])]

    # Add the predictions output
    outputs += [features.model.outputs[-1]]

    # Build the encoder model for all layers
    encoder_model = tf.keras.models.Model(
        inputs=features.model.input,
        outputs=outputs
    )
    return EasyDict(
        encoder=encoder_model,
        layers=layers,
        range={
            layer: np.load(join(path, layer, 'range_clip.npz'))['range']
            for layer in layers
        },
        clip={
            layer: np.load(join(path, layer, 'range_clip.npz'))['clip']
            for layer in layers
        }
    )


def _load_model_and_saliency(model_path, device=None):
    log.debug("Loading model at {}...".format(model_path))
    _umap_encoders = None
    _saliency = None

    # Load a PyTorch model
    if sf.util.torch_available and sf.util.path_to_ext(model_path) == 'zip':
        import slideflow.model.torch
        _device = sf.model.torch.torch_utils.get_device()
        _model = sf.model.torch.load(model_path)
        _model.to(_device)
        _model.eval()
        if device is not None:
            _model = _model.to(device)
        _saliency = sf.grad.SaliencyMap(_model, class_idx=0)  #TODO: auto-update from heatmaps logit

    # Load a TFLite model
    elif sf.util.tf_available and sf.util.path_to_ext(model_path) == 'tflite':
        interpreter = tf.lite.Interpreter(model_path)
        _model = interpreter.get_signature_runner()

    # Load a Tensorflow model
    elif sf.util.tf_available:
        import slideflow.model.tensorflow
        _model = sf.model.tensorflow.load(model_path, method='weights')
        _saliency = sf.grad.SaliencyMap(_model, class_idx=0)  #TODO: auto-update from heatmaps logit
        if exists(join(model_path, 'umap_encoders')):
            _umap_encoders = _load_umap_encoders(join(model_path, 'umap_encoders'), _model)
    else:
        raise ValueError(f"Unable to interpret model {model_path}")
    return _model, _saliency, _umap_encoders
