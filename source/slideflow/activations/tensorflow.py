import os
import logging
import numpy as np

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import slideflow as sf
import slideflow.slide

from slideflow.util import log

class ActivationsInterface:
    """Interface for obtaining logits and intermediate layer activations from Slideflow models.

    Use by calling on either a batch of images (returning outputs for a single batch), or by calling on a
    :class:`slideflow.slide.WSI` object, which will generate an array of spatially-mapped activations matching
    the slide.

    Examples
        *Calling on batch of images:*

        .. code-block:: python

            interface = ActivationsInterface('/model/path', layers='postconv')
            for image_batch in train_data:
                # Return shape: (batch_size, num_features)
                batch_activations = interface(image_batch)

        *Calling on a slide:*

        .. code-block:: python

            slide = sf.slide.WSI(...)
            interface = ActivationsInterface('/model/path', layers='postconv')
            # Return shape: (slide.grid.shape[0], slide.grid.shape[1], num_features):
            activations_grid = interface(slide)

    """

    def __init__(self, path, layers='postconv', include_logits=False):
        """Creates an activations interface from a saved slideflow model which outputs feature activations
        at the designated layers.

        Intermediate layers are returned in the order of layers. Logits are returned last.

        Args:
            path (str): Path to saved Slideflow model.
            layers (list(str), optional): Layers from which to generate activations.  The post-convolution activation layer
                is accessed via 'postconv'. Defaults to 'postconv'.
            include_logits (bool, optional): Include logits in output. Will be returned last. Defaults to False.
        """

        if layers and not isinstance(layers, list): layers = [layers]
        self.path = path
        self.num_logits = 0
        self.num_features = 0
        if path is not None:
            self._model = tf.keras.models.load_model(self.path)
            self._build(layers=layers, include_logits=include_logits)

    @classmethod
    def from_model(cls, model, layers='postconv', include_logits=False):
        """Creates an activations interface from a loaded slideflow model which outputs feature activations
        at the designated layers.

        Intermediate layers are returned in the order of layers. Logits are returned last.

        Args:
            model (:class:`tensorflow.keras.models.Model`): Loaded model.
            layers (list(str), optional): Layers from which to generate activations.  The post-convolution activation layer
                is accessed via 'postconv'. Defaults to 'postconv'.
            include_logits (bool, optional): Include logits in output. Will be returned last. Defaults to False.
        """
        obj = cls(None, layers, include_logits)
        if isinstance(model, tf.keras.models.Model):
            obj._model = model
        else:
            raise TypeError("Provided model is not a valid Tensorflow model.")
        obj._build(layers=layers, include_logits=include_logits)
        return obj

    def __call__(self, inp, **kwargs):
        """Process a given input and return activations and/or logits. Expects either a batch of images or
        a :class:`slideflow.slide.WSI` object."""

        if isinstance(inp, sf.slide.WSI):
            return self._predict_slide(inp, **kwargs)
        else:
            return self._predict(inp)

    def _predict_slide(self, slide, batch_size=128, dtype=np.float16, **kwargs):
        """Generate activations from slide => activation grid array."""
        total_out = self.num_features + self.num_logits
        activations_grid = np.zeros((slide.grid.shape[1], slide.grid.shape[0], total_out), dtype=dtype)
        generator = slide.build_generator(shuffle=False, include_loc='grid', show_progress=True, **kwargs)

        if not generator:
            log.error(f"No tiles extracted from slide {sf.util.green(slide.name)}")
            return

        def _parse_function(record):
            image = record['image']
            loc = record['loc']
            parsed_image = tf.image.per_image_standardization(image)
            parsed_image.set_shape([slide.tile_px, slide.tile_px, 3])
            return parsed_image, loc

        # Generate dataset from the generator
        with tf.name_scope('dataset_input'):
            output_signature={'image':tf.TensorSpec(shape=(slide.tile_px,slide.tile_px,3), dtype=tf.uint8),
                              'loc':tf.TensorSpec(shape=(2), dtype=tf.uint32)}
            tile_dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
            tile_dataset = tile_dataset.map(_parse_function, num_parallel_calls=8)
            tile_dataset = tile_dataset.batch(batch_size, drop_remainder=False)
            tile_dataset = tile_dataset.prefetch(8)

        act_arr = []
        loc_arr = []
        for i, (batch_images, batch_loc) in enumerate(tile_dataset):
            model_out = self._predict(batch_images)
            if not isinstance(model_out, list): model_out = [model_out]
            act_arr += [np.concatenate([m.numpy() for m in model_out])]
            loc_arr += [batch_loc.numpy()]

        act_arr = np.concatenate(act_arr)
        loc_arr = np.concatenate(loc_arr)

        for i, act in enumerate(act_arr):
            xi = loc_arr[i][0]
            yi = loc_arr[i][1]
            activations_grid[yi][xi] = act

        return activations_grid

    @tf.function
    def _predict(self, inp):
        """Return activations for a single batch of images."""
        return self.model(inp, training=False)

    def _build(self, layers, include_logits=True):
        """Builds the interface model that outputs feature activations at the designated layers and/or logits.
            Intermediate layers are returned in the order of layers. Logits are returned last."""

        if layers:
            log.debug(f"Setting up interface to return activations from layers {', '.join(layers)}")
            other_layers = [l for l in layers if l != 'postconv']
        else:
            other_layers = []
        outputs = {}
        if layers:
            intermediate_core = tf.keras.models.Model(inputs=self._model.layers[1].input,
                                                      outputs=[self._model.layers[1].get_layer(l).output for l in other_layers])
            if len(other_layers) > 1:
                int_out = intermediate_core(self._model.input)
                for l, layer in enumerate(other_layers):
                    outputs[layer] = int_out[l]
            elif len(other_layers):
                outputs[other_layers[0]] = intermediate_core(self._model.input)
            if 'postconv' in layers:
                outputs['postconv'] = self._model.layers[1].get_output_at(0)
        outputs_list = [] if not layers else [outputs[l] for l in layers]
        if include_logits:
            outputs_list += [self._model.output]
        self.model = tf.keras.models.Model(inputs=self._model.input, outputs=outputs_list)
        self.num_features = sum([outputs[o].shape[1] for o in outputs])
        if isinstance(self._model.output, list):
            log.warning("Multi-categorical outcomes not yet supported for this interface.")
            self.num_logits = 0
        else:
            self.num_logits = 0 if not include_logits else self._model.output.shape[1]
        if include_logits:
            log.debug(f'Number of logits: {self.num_logits}')
        log.debug(f'Number of activation features: {self.num_features}')