"""Contains the DeepFocus algorithm, as published by Senaras et al:

    Senaras C, et al. DeepFocus: Detection of out-of-focus regions
    in whole slide digital images using deep
    learning. PLOS ONE 13(10): e0205387.

The published model / saved checkpoint was converted into TF2/Keras
format and is available at https://github.com/jamesdolezal/deepfocus.
This repository has been included as a submodule here for convenience.

The training dataset is available at
[https://doi.org/10.5281/zenodo.1134848](https://doi.org/10.5281/zenodo.1134848).
"""

import numpy as np
from packaging import version
from typing import Optional, Union

from .strided_dl import StridedDL

# -----------------------------------------------------------------------------

class DeepFocus(StridedDL):

    def __init__(
        self,
        ckpt: Optional[str] = None,
        *,
        buffer: int = 16,
        tile_um: Union[str, int] = '40x',
        **kwargs
    ):
        """Utilizes the DeepFocus QC algorithm, as published by Senaras et al:

            Senaras C, et al. DeepFocus: Detection of out-of-focus regions
            in whole slide digital images using deep
            learning. PLOS ONE 13(10): e0205387.

        The published model / saved checkpoint was converted into TF2/Keras
        format and is available at https://github.com/jamesdolezal/deepfocus.

        Args:
            ckpt (str, optional): Path to checkpoint. If not provided,
                will use the default, published 'ver5' checkpoint.
            tile_um (str or float): Tile size, in microns (int) or
                magnification (str). Defaults to '40x'.

        Keyword args:
            buffer (int): Number of tiles (width and height) to extract and
                process simultaneously. Extracted tile size (width/height)
                will be  ``tile_px * buffer``. Defaults to 8.
            grayspace_fraction (float): Grayspace fraction when extracting
                tiles from slides. Defaults to 1 (disables).
            verbose (bool): Show a progress bar during calculation.
            kwargs (Any): All remaining keyword arguments are passed to
                :meth:`slideflow.WSI.build_generator()`.
        """
        model = deepfocus_v3()
        self.enable_mixed_precision()
        load_checkpoint(model, ckpt)
        super().__init__(
            model=model,
            pred_idx=1,
            tile_px=64,
            tile_um=tile_um,
            buffer=buffer,
            **kwargs
        )
        self.ckpt = ckpt
        self.pb_msg = "Applying DeepFocus..."

    def __repr__(self):
        return "DeepFocus(tile_um={!r}, buffer={!r}, ckpt={!r})".format(
            self.tile_um, self.buffer, self.ckpt
        )

    def enable_mixed_precision(self):
        import tensorflow as tf
        _policy = 'mixed_float16'
        if version.parse(tf.__version__) > version.parse("2.8"):
            tf.keras.mixed_precision.set_global_policy(_policy)
        else:
            policy = tf.keras.mixed_precision.experimental.Policy(_policy)
            tf.keras.mixed_precision.experimental.set_policy(policy)

    def preprocess(self, image: np.ndarray):
        image = image.astype(np.float32) / 255.
        return image - np.mean(image)

# -----------------------------------------------------------------------------

def deepfocus_v3(
    filters = (32, 32, 64, 128, 128),
    kernel_size = (5, 3, 3, 3, 3),
    fc = (128, 64)
):
    """Build the DeepFocusV3 model architecture."""

    import tensorflow as tf
    from tensorflow.keras.regularizers import L2

    assert len(filters) == len(kernel_size)

    # Input.
    inp = tf.keras.layers.InputLayer(input_shape=(64, 64, 3), name='input')

    # Pre-processing.
    x = inp.output - tf.math.reduce_mean(inp.output, keepdims=True)

    # Convolutional layers.
    x = tf.keras.layers.Conv2D(filters[0], kernel_size[0], activation='relu', kernel_regularizer=L2, padding='same', name='Conv2D')(x)
    x = tf.keras.layers.BatchNormalization(name='BatchNormalization')(x)

    x = tf.keras.layers.Conv2D(filters[1], kernel_size[1], activation='relu', kernel_regularizer=L2, padding='same', name='Conv2D_1')(x)
    x = tf.keras.layers.BatchNormalization(name='BatchNormalization_1')(x)

    x = tf.keras.layers.Conv2D(filters[2], kernel_size[2], activation='relu', kernel_regularizer=L2, padding='same', name='Conv2D_2')(x)
    x = tf.keras.layers.BatchNormalization(name='BatchNormalization_2')(x)
    x = tf.keras.layers.MaxPool2D((2,2), padding='same')(x)

    x = tf.keras.layers.Conv2D(filters[3], kernel_size[3], activation='relu', kernel_regularizer=L2, padding='same', name='Conv2D_3')(x)
    x = tf.keras.layers.BatchNormalization(name='BatchNormalization_3')(x)
    x = tf.keras.layers.MaxPool2D((2,2), padding='same')(x)

    x = tf.keras.layers.Conv2D(filters[4], kernel_size[4], activation='relu', kernel_regularizer=L2, padding='same', name='Conv2D_4')(x)
    x = tf.keras.layers.BatchNormalization(name='BatchNormalization_4')(x)
    x = tf.keras.layers.MaxPool2D((2,2), padding='same')(x)

    # Fully connected layers.
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(fc[0], activation='relu', kernel_regularizer=L2, name='FullyConnected')(x)
    x = tf.keras.layers.BatchNormalization(name='BatchNormalization_5')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(fc[1], activation='relu', kernel_regularizer=L2, name='FullyConnected_1')(x)
    x = tf.keras.layers.BatchNormalization(name='BatchNormalization_6')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(2, activation='softmax', name='FullyConnected_2')(x)
    model = tf.keras.Model(inputs=inp.input, outputs=x)
    return model

#------------------------------------------------------------------------------

def transform_variable_name(name):
    """Transform a variable name from new (Keras) to old (TFLearn) format."""
    if name.endswith('kernel'):
        return name.split('kernel')[0] + 'W'
    if name.endswith('bias'):
        return name.split('bias')[0] + 'b'
    return name


def load_checkpoint(model, ckpt=None, verbose=False):
    """Load a manual checkpoint."""
    import tensorflow as tf

    if ckpt is None:
        ckpt = download_checkpoint()

    reader = tf.compat.v1.train.load_checkpoint(ckpt)
    ckpt_vars = tf.compat.v1.train.list_variables(ckpt)
    ckpt_layers = list(set([c[0].split('/')[0] for c in ckpt_vars]))
    if verbose:
        print("Loading {} variables ({} layers) from checkpoint {}.".format(
            len(ckpt_vars), len(ckpt_layers), ckpt)
        )

    if verbose:
        for name, tensor_shape in ckpt_vars:
            print("\t", name, tensor_shape)

    for layer_name in ckpt_layers:
        try:
            layer = model.get_layer(layer_name)
        except ValueError:
            if verbose:
                print("Skipping layer {}".format(layer_name))
        else:
            if verbose:
                print("Working on layer {}".format(layer_name))
            ckpt_vals = []
            for varname in [w.name for w in layer.weights]:
                varname = varname.split(':0')[0]
                varname = transform_variable_name(varname)
                ckpt_vals.append(reader.get_tensor(varname))
                if verbose:
                    print("\tWorking on varname", varname, ckpt_vals[-1].shape)
            layer.set_weights(ckpt_vals)
            if verbose:
                print("\tSet {} variables to layer {}".format(
                    len(layer.weights), layer_name)
                )
    return model


def download_checkpoint():
    """Download the pretrained checkpoint from HuggingFace."""
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(
        repo_id='jamesdolezal/deepfocus', filename='ver5.index'
    )
    hf_hub_download(
        repo_id='jamesdolezal/deepfocus', filename='ver5.meta'
    )
    hf_hub_download(
        repo_id='jamesdolezal/deepfocus', filename='ver5.data-00000-of-00001'
    )
    return ckpt_path[:-6]

# -----------------------------------------------------------------------------