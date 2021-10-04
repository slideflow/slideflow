.. currentmodule:: slideflow.model

slideflow.model
===============

This module provides a :class:`slideflow.model.ModelParams` class to organize model and training
parameters/hyperparameters and assist with model building, as well as :class:`slideflow.model.Trainer` class that
executes model training and evaluation. The :class:`slideflow.model.LinearTrainer` and The
:class:`slideflow.model.CPHTrainer` are extensions of this class, supporting linear and Cox Proportional Hazards
outcomes, respectively. The function :func:`slideflow.model.trainer_from_hp` can choose and return the correct model
instance based on the provided hyperparameters.

:class:`slideflow.model.ModelParams` will build models according to a set of model parameters and a given set of
outcome labels. To change the core image convolutional model to another architecture, add the custom model class
to the ModelParams class variable ModelDict with a recognizable name. Then, change the instanced ModelParams object
"model" variable to this name. For example:

.. code-block:: python

    import CustomModel
    from slideflow.model import ModelParams

    ModelParams.ModelDict['custom'] = CustomModel
    mp = ModelParams(model='custom', ...)

To build a completely custom model without utilizing any of the automatic input/output setups according to your
outcome labels, write a class which inherits the ModelParams class and implements the function :func:`build_model`.

.. automodule: slideflow.model

ModelParams
---------------

.. autoclass:: ModelParams
    :inherited-members:

Trainer
--------
.. autoclass:: Trainer
    :inherited-members:

LinearTrainer
--------------
.. autoclass:: LinearTrainer
    :inherited-members:

CPHTrainer
-----------
.. autoclass:: CPHTrainer
    :inherited-members:

trainer_from_hp
---------------
.. autofunction:: trainer_from_hp