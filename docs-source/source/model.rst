.. currentmodule:: slideflow.model

slideflow.model
===============

This module provides a :class:`slideflow.model.Model` class that
assists with model building, training, and evaluation. The :class:`slideflow.model.LinearModel` and
The :class:`slideflow.model.CPHModels` are extensions of this class, supporting linear and Cox Proportional Hazards
outcomes, respectively. The function :func:`slideflow.model.model_from_hp` can choose and return the correct model
instance based on the provided hyperparameters.

.. automodule: slideflow.model

HyperParameters
---------------

.. autoclass:: HyperParameters
    :inherited-members:

Model
-----
.. autoclass:: Model
    :inherited-members:

LinearModel
-----------
.. autoclass:: LinearModel
    :inherited-members:

CPHModel
--------
.. autoclass:: CPHModel
    :inherited-members:

model_from_hp
-------------
.. autofunction:: model_from_hp