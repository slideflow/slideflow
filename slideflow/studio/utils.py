"""Utilities for Slideflow Studio."""

from typing import Any, List

import imgui
import os
import slideflow as sf
import numpy as np
from os.path import join, exists
from slideflow import log
from typing import Tuple, Optional

if sf.util.tf_available:
    import tensorflow as tf
    sf.util.allow_gpu_memory_growth()
if sf.util.torch_available:
    import slideflow.model.torch

#----------------------------------------------------------------------------


LEFT_MOUSE_BUTTON = 0
RIGHT_MOUSE_BUTTON = 1

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
    is_classification: bool
) -> str:
    """Convert a prediction array to a human-readable string."""
    #TODO: support multi-outcome models
    if is_classification:
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

#----------------------------------------------------------------------------

class StatusMessage:
    """A class to manage status messages."""
    def __init__(
        self,
        viz: Any,
        message: str,
        description: Optional[str] = None,
        *,
        color: Tuple[float, float, float, float] = (0.7, 0, 0, 1),
        text_color: Tuple[float, float, float, float] = (1, 1, 1, 1),
        rounding: int = 0,
    ) -> None:
        self.viz = viz
        self.message = message
        self.description = description
        self.color = color
        self.text_color = text_color
        self.rounding = rounding


    def render(self):
        """Render the status message."""
        # Calculations.
        h = self.viz.status_bar_height
        r = self.viz.pixel_ratio
        y_pos = int((self.viz.content_frame_height - (h * r)) / r)
        size = imgui.calc_text_size(self.message)

        # Center the text.
        x_start = self.viz.content_width/2 - size.x/2
        imgui.same_line()
        imgui.set_cursor_pos_x(x_start)

        # Draw the background.
        draw_list = imgui.get_window_draw_list()
        pad = self.viz.spacing * 2
        draw_list.add_rect_filled(
            x_start - pad - 4,
            y_pos,
            x_start + size.x + pad,
            y_pos + h,
            imgui.get_color_u32_rgba(*self.color),
            rounding=self.rounding
        )

        # Draw the text.
        imgui.push_style_color(imgui.COLOR_TEXT, *self.text_color)
        imgui.text(self.message)
        imgui.pop_style_color(1)

        # Set the tooltip.
        if self.description is not None:
            if imgui.is_item_hovered():
                imgui.set_tooltip(self.description)
