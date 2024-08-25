.. _plugins:

Creating a Slideflow Plugin
===========================

Slideflow has been designed to be extensible, and we encourage users to contribute their own plugins to the Slideflow ecosystem. Plugins can be used to add new functionality to Slideflow, such as new feature extractors or new model architectures. This page provides an overview of how to create and use plugins with Slideflow.


MIL Model Registration
----------------------

As discussed in :ref:`custom_mil`, Slideflow supports the registration of custom MIL models. This is done by using the ``register_model`` decorator to register a custom MIL model.

For example, suppose you have a custom MIL model called ``MyMILModel`` that you want to register with Slideflow. You've already designed the model such that it meets Slideflow's MIL `requirements <custom_mil>`__. Now you want to make it available for use directly within Slideflow. You can accomplish this by using the ``register_model`` decorator:

.. code-block:: python

    from slideflow.model.mil import register_model

    @register_model
    def my_mil_model(**kwargs):
        from . import MyMILModel
        return MyMILModel(**kwargs)

Once this code is run, the custom MIL model will be available for use with Slideflow:

.. code-block:: python

    import slideflow as sf

    model = sf.build_mil_model("my_mil_model")


Feature Extractors
------------------

Similarly, Slideflow supports the integration of custom feature extractors via the ``register_torch`` and ``register_tf`` decorators. Please see our detailed `developer note <custom_extractors>`__ for more information on how to create and register custom extractors. Briefly, you can register a custom feature extractor with Slideflow as follows:

.. code-block:: python

    from slideflow.model.extractors import register_torch

    @register_torch
    def my_foundation_model(**kwargs):
        from . import MyFoundationModel
        return MyFoundationModel(**kwargs)


Creating a Plugin
-----------------

Once you have a custom MIL model or feature extractor that you want to integrate with Slideflow, you can create a plugin to make it available to other users.

Slideflow supports external plugins via standard Python entry points, allowing you to publish your own package that integrates with Slideflow.

In your package's ``setup.py`` file, use the "entry_points" key to connect with the Slideflow plugin interface:

.. code-block:: python

    ...,
    entry_points={
        'slideflow.plugins': [
            'extras = my_package:register_extras',
        ],
    },

Then, in your package's root ``__init__.py`` file, write a ``register_extras()`` function that does any preparation needed to initialize or import your model.

(in ``my_package/__init__.py``)

.. code-block:: python

    def register_extras():
        # Import the model, and do any other necessary preparation.
        # If my_module contains the @register_model decorator,
        # the model will be registered with Slideflow automatically.
        from . import my_module

    print("Registered MyFoundationModel")

You can then build and distribute your plugin, and once installed, the registration with Slideflow will happen automatically:

.. code-block:: bash

    pip install my_package


.. code-block:: python

    import slideflow as sf

    model = sf.build_feature_extractor("my_foundation_model")


For a complete example, head over to our `Slideflow-GPL <https://github.com/slideflow/slideflow-gpl>`_ and `Slideflow-NonCommercial <https://github.com/slideflow/slideflow-noncommercial>`_ repositories, which have been built using the plugin system described above.