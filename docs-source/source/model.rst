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

.. _balancing:

A Note on Input Balancing
*************************

During training, mini-batch balancing can be customized to assist with increasing representation of sparse outcomes or small slides. Five mini-batch balancing methods are available, set through the model parameters ``training_balance`` and ``validation_balance``. These are ``'tile'``, ``'category'``, ``'patient'``, ``'slide'``, and ``'none'``.

For the purposes of illustration, suppose you have five slides, labeled A through E. Slides A and B belong to category 1, while C, D, E belong to category 2. Let's suppose tumors in all the slides are roughly the same physical size, except for B which is three times as large. After tile extraction, all the patients except B produce roughly the same number of image tiles. If we train with a batch size of 32, how are those 32 images selected?

If **tile-level balancing** ("tile") is used, tiles will be selected randomly from the population of all extracted tiles. Because slide B has so many more tiles than the other slides, B will be over-represented in the batch. This means that the model may learn a bias towards patient B.

If **slide-based balancing** ("patient") is used, batches will contain equal representation of images from each slide. In the above example, category 1 (patients A and B) will have 13 tiles in the batch, whereas category 2 (patients C, D, and E) will have 19 tiles in the batch. With this type of balancing, models may learn bias towards categories with more slides (in this case category 2).

If **patient-based balancing** ("patient") is used, batches will balance image tiles across patients. The balancing is similar to slide-based balancing, except across patients (as each patient may have more than one slide).

If **category-based balancing** ("category") is used, batches will contain equal representation from each outcome category. In this example, there will be an equal number of tiles from category 1 and category 2, 16 from both.

If **no balancing** is performed, batches will be assembled by randomly selecting from TFRecords. This is equivalent to slide-based balancing if each slide has its own TFRecord (default behavior).


Working with layer activations
******************************

:class:`slideflow.model.Features` creates an interface to efficiently generate features/layer activations and predictions
from either a batch of images (returning a batch of activations/predictions) or a whole-slide image (returning a grid of
activations/predictions).

:class:`slideflow.DatasetFeatures` calculates features and predictions for an entire dataset, storing
result arrays into a dictionary mapping slide names to the generated activations. This buffer of whole-dataset
activations can then be used for functions requiring analysis of whole-dataset activations, including
:class:`slideflow.SlideMap` and :class:`slideflow.Mosaic`.

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