.. _custom_extractors:

Custom Feature Extractors
=========================

Slideflow includes several :ref:`pretrained feature extractors <mil>` for converting image tiles into feature vectors as well as tools to assist with building your own feature extractor. In this note, we'll walk through the process of building a custom feature extractor from both a PyTorch and Tensorflow model.

PyTorch
*******

Feature extractors are implemented as a subclass of :class:`slideflow.model.extractors._factory_torch.TorchFeatureExtractor`. The base class provides core functionality and helper methods for generating features from image tiles (dtype uint8) or whole-slide images (type :class:`slideflow.WSI`).

The initializer should create the feature extraction model and move it to the appropriate device (*i.e.* GPU). The model should be a :class:`torch.nn.Module` that accepts an image tensor as input and returns a feature tensor as output.

.. code-block:: python

    # Import your custom torch.nn.Module,
    # which generates features from an image.
    from my_module import MyModel

    from slideflow.model.extractors._factory_torch import TorchFeatureExtractor

    class MyFeatureExtractor(TorchFeatureExtractor):

        tag = 'my_feature_extractor'  # Human-readable identifier

        def __init__(self):
            super().__init__()

            # Create the device, move to GPU, and set in evaluation mode.
            self.model = MyModel()
            self.model.to('cuda')
            self.model.eval()

Next, the initializer should set the number of features expected to be returned by the model.

.. code-block:: python

    ...

        def __init__(self):
            ...

            self.num_features = 1024

The initializer is also responsible for registering image preprocessing. The image preprocessing transformation, a function which converts a raw ``uint8`` image to a ``float32`` tensor for model input, should be stored in ``self.transform``. If the transformation standardizes the images, then the parameter ``self.preprocess_kwargs`` should be set to ``{'standardize': False}``, indicating that Slideflow should not perform any additional standardization. You can use the class method ``.build_transform()`` to use the standard preprocessing pipeline.

.. code-block:: python

    from torchvision import transforms

    ...

        def __init__(self):
            ...

            # Image preprocessing.
            self.transform = self.build_transform(img_size=256)
            # Disable Slideflow standardization,
            # as we are standardizing with transforms.Normalize
            self.preprocess_kwargs = {'standardize': False}

The final required method is ``.dump_config()``, which returns a dictionary of configuration parameters needed to regenerate this class. It should return a dictionary with ``"class"`` and ``"kwargs"`` attributes. This configuration is saved to a JSON configuration file when generating bags for MIL training.

.. code-block:: python

    ...

        def dump_config(self):
            return self._dump_config(
                class_name='my_module.MyFeatureExtractor'
            )

The final class should look like this:

.. code-block:: python

    from my_module import MyModel
    from slideflow.model.extractors._factory_torch import TorchFeatureExtractor
    from torchvision import transforms

    class MyFeatureExtractor(TorchFeatureExtractor):

        tag = 'my_feature_extractor'  # Human-readable identifier

        def __init__(self):
            super().__init__()

            # Create the device, move to GPU, and set in evaluation mode.
            self.model = MyModel()
            self.model.to('cuda')
            self.model.eval()
            self.num_features = 1024

            # Image preprocessing.
            self.transform = self.build_transform(img_size=256)
            # Disable Slideflow standardization,
            # as we are standardizing with transforms.Normalize
            self.preprocess_kwargs = {'standardize': False}

        def dump_config(self):
            return self._dump_config(
                class_name='my_module.MyFeatureExtractor'
            )

You can then use the feature extractor for generating bags for MIL training, as described in :ref:`mil`.

.. code-block:: python

    # Build the feature extractor.
    myfeatures = MyFeatureExtractor()

    # Load a dataset.
    project = slideflow.load_project(...)
    dataset = project.dataset(...)

    # Generate bags.
    project.generate_feature_bags(myfeatures, dataset)

You can also generate features across whole-slide images, returning a grid of features for each slide. The size of the returned grid reflects the slide's tile grid. For example, for a slide with 24 columns and 33 rows of tiles, the returned grid will have shape ``(24, 33, n_features)``.

.. code-block:: python

    >>> myfeatures = MyFeatureExtractor()
    >>> wsi = sf.WSI('path/to/wsi', tile_px=256, tile_um=302)
    >>> features = myfeatures(wsi)
    >>> features.shape
    (24, 33, 1024)

Finally, the feature extractor can also be used to perform latent space analysis and generate mosaic maps, as described in :ref:`activations`.

Slideflow includes a registration system for keeping track of all available feature extractors. To register your feature extractor, use the :func:`slideflow.model.extractors.register_torch` decorator.

.. code-block:: python

    from slideflow.model.extractors import register_torch

    @register_torch
    def my_feature_extractor(**kwargs):
        return MyFeatureExtractor(**kwargs)

Once registered, a feature extractor can be built by name:

.. code-block:: python

    import slideflow as sf
    extractor = sf.build_feature_extractor('my_feature_extractor')


Tensorflow
**********

Tensorflow feature extractors are implemented very similarly to PyTorch feature extractors, extended from :class:`slideflow.model.extractors._tensorflow_base.TensorflowFeatureExtractor`.

The initializer should create the model and set the expected number of features.

.. code-block:: python

    from my_module import MyModel
    from slideflow.model.extractors._tensorflow_base import TensorflowFeatureExtractor

    class MyFeatureExtractor(TensorflowFeatureExtractor):

        tag = 'my_feature_extractor'  # Unique identifier

        def __init__(self):
            super().__init__()

            # Create the model.
            self.model = MyModel()
            self.num_features = 1024

.. |per_image_standardization| replace:: ``tf.image.per_image_standardization``
.. _per_image_standardization: https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization


The initializer is also responsible for registering image preprocessing and transformations. Preprocessing steps are stored in the ``.preprocess_kwargs`` dictionary, which should have the keys ``standardize`` and ``transform``. If ``standardize=True``, images will be standardized using |per_image_standardization|_. If ``transform`` is not None, it should be a callable that accepts a single image tensor and returns a transformed image tensor.

For example, to only perform standardization and no further preprocessing:

.. code-block:: python

    ...

        def __init__(self):
            ...

            # Image preprocessing.
            self.preprocess_kwargs = {
                'standardize': True,
                'transform': None
            }

To perform standardization and resize images to 256x256:

.. code-block:: python

    import tensorflow as tf

    @tf.function
    def resize_256(x):
        return = tf.image.resize(x, (resize_px, resize_px))

    ...

        def __init__(self):
            ...

            # Image preprocessing.
            self.preprocess_kwargs = {
                'standardize': True,
                'transform': resize_256
            }

The ``.dump_config()`` method should then be set, which is expected to return a dictionary of configuration parameters needed to regenerate this class. It should return a dictionary with ``"class"`` and ``"kwargs"`` attributes. This configuration is saved to a JSON configuration file when generating bags for MIL training.

.. code-block:: python

    ...

        def dump_config(self):
            return {
                'class': 'MyFeatureExtractor',
                'kwargs': {}
            }

The final class should look like this:

.. code-block:: python

    from my_module import MyModel
    from slideflow.model.extractors._tensorflow_base import TensorflowFeatureExtractor

    class MyFeatureExtractor(TensorflowFeatureExtractor):

        tag = 'my_feature_extractor'  # Unique identifier

        def __init__(self):
            super().__init__()

            # Create the model.
            self.model = MyModel()
            self.num_features = 1024

            # Image preprocessing.
            self.preprocess_kwargs = {
                'standardize': True,
                'transform': None
            }

        def dump_config(self):
            return {
                'class': 'MyFeatureExtractor',
                'kwargs': {}
            }

As described above, this feature extractor can then be used to create bags for MIL training, generate features across whole-slide images, or perform feature space analysis across a dataset.

To register your feature extractor, use the :func:`slideflow.model.extractors.register_tensorflow` decorator.

.. code-block:: python

    from slideflow.model.extractors import register_tf

    @register_tf
    def my_feature_extractor(**kwargs):
        return MyFeatureExtractor(**kwargs)

...which will allow the feature extractor to be built by name:

.. code-block:: python

    import slideflow as sf
    extractor = sf.build_feature_extractor('my_feature_extractor')