.. currentmodule:: slideflow.grad

slideflow.grad
==============

This submodule contains tools for calculating and display pixel attribution, or
saliency, maps.

Saliency maps are calculated on preprocessed images, returning a grayscale
image. Below is an example of how to generate a single saliency map for one image.

.. code-block:: python

    import tensorflow as tf
    import slideflow as sf
    import slideflow.grad
    from PIL import Image

    # Get an example image
    P = sf.Project('/path/to/project')
    dataset = P.dataset(299, 302)
    tf_dts = dataset.tensorflow(None, batch_size=1, standardize=False)
    raw_image = next(iter(tf_dts))[0][0]

    # Load a model
    model_path = '/path/to/model'
    model = tf.keras.models.load_model(model_path)
    normalizer = sf.util.get_model_normalizer(model_path)

    # Process the image
    if normalizer:
        raw_image = normalizer.tf_to_tf(raw_image)
    processed_image = tf.image.per_image_standardization(raw_image)
    processed_image = processed_image.numpy()

    # Prepare the saliency map
    smap = sf.grad.SaliencyMap(model, class_idx=0)

    # Calculate gradients
    ig_map = smap.integrated_gradients(processed_image)

    # Display the gradients as an inferno heatmap
    ig_map = sf.grad.inferno(ig_map)
    Image.fromarray(ig_map).show()

There are also several utility functions that can be used to compare saliency maps
across multiple images, or compare multiple saliency map methods for a single image.
Documentation for these functions is given below.

.. autoclass:: SaliencyMap
    :inherited-members:

.. automodule:: slideflow.grad
   :members:

.. autofunction:: comparison_plot

.. autofunction:: inferno

.. autofunction:: multi_plot

.. autofunction:: oranges

.. autofunction:: overlay

.. autofunction:: saliency_map_comparison