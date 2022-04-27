.. currentmodule:: slideflow.model

slideflow.model
===============

This module provides the :class:`ModelParams` class to organize model and training
parameters/hyperparameters and assist with model building, as well as the :class:`Trainer` class that
executes model training and evaluation. :class:`LinearTrainer` and :class:`CPHTrainer`
are extensions of this class, supporting linear and Cox Proportional Hazards outcomes, respectively. The function
:func:`trainer_from_hp` can choose and return the correct model instance based on the provided
hyperparameters.

.. note::
    In order to support both Tensorflow and PyTorch backends, the :mod:`slideflow.model` module will import either
    :mod:`slideflow.model.tensorflow` or :mod:`slideflow.model.torch` according to the currently active backend,
    indicated by the environmental variable ``SF_BACKEND``.

Configuring and training models
*******************************

:class:`slideflow.model.ModelParams` will build models according to a set of model parameters and a given set of
outcome labels. To change the core image convolutional model to another architecture, set the ``model`` parameter
to the custom model class.

.. code-block:: python

    import CustomModel
    from slideflow.model import ModelParams

    mp = ModelParams(model=CustomModel, ...)

Working with layer activations
******************************

:class:`slideflow.model.Features` creates an interface to efficiently generate features/layer activations and logits
from either a batch of images (returning a batch of activations/logits) or a whole-slide image (returning a grid of
activations/logits).

:class:`slideflow.DatasetFeatures` calculates features and logits for an entire dataset, storing
result arrays into a dictionary mapping slide names to the generated activations. This buffer of whole-dataset
activations can then be used for functions requiring analysis of whole-dataset activations, including
:class:`slideflow.SlideMap` and :class:`slideflow.mosiac.Mosaic`.

.. automodule: slideflow.model

ModelParams
***********
.. autoclass:: ModelParams
    :inherited-members:

Trainer
***********
.. autoclass:: Trainer
    :inherited-members:

LinearTrainer
*************
.. autoclass:: LinearTrainer
    :inherited-members:

CPHTrainer
***********
.. autoclass:: CPHTrainer
    :inherited-members:

trainer_from_hp
***************
.. autofunction:: trainer_from_hp

Features
***********
.. autoclass:: Features
    :inherited-members:

DatasetFeatures
****************
.. autoclass:: DatasetFeatures
    :inherited-members: