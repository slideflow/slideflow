.. currentmodule:: slideflow

.. _model_params:

slideflow.ModelParams
=====================

The :class:`ModelParams` class organizes model and training parameters/hyperparameters and assists with model building.

See :ref:`training` for a detailed look at how to train models.

ModelParams
***********
.. autoclass:: ModelParams
.. autofunction:: slideflow.ModelParams.to_dict
.. autofunction:: slideflow.ModelParams.get_normalizer
.. autofunction:: slideflow.ModelParams.validate
.. autofunction:: slideflow.ModelParams.model_type

Mini-batch balancing
********************

During training, mini-batch balancing can be customized to assist with increasing representation of sparse outcomes or small slides. Five mini-batch balancing methods are available when configuring :class:`slideflow.ModelParams`, set through the parameters ``training_balance`` and ``validation_balance``. These are ``'tile'``, ``'category'``, ``'patient'``, ``'slide'``, and ``'none'``.

If **tile-level balancing** ("tile") is used, tiles will be selected randomly from the population of all extracted tiles.

If **slide-based balancing** ("patient") is used, batches will contain equal representation of images from each slide.

If **patient-based balancing** ("patient") is used, batches will balance image tiles across patients. The balancing is similar to slide-based balancing, except across patients (as each patient may have more than one slide).

If **category-based balancing** ("category") is used, batches will contain equal representation from each outcome category.

If **no balancing** is performed, batches will be assembled by randomly selecting from TFRecords. This is equivalent to slide-based balancing if each slide has its own TFRecord (default behavior).

See :ref:`balancing` for more discussion on sampling and mini-batch balancing.

.. note::

    If you are :ref:`using a Trainer <training_with_trainer>` to train your models, you can further customize the mini-batch balancing strategy by using :meth:`slideflow.Dataset.balance` on your training and/or validation datasets.