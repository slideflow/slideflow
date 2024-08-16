.. currentmodule:: slideflow.mil

slideflow.mil
==============

This submodule contains tools for multiple-instance learning (MIL) model training and evaluation. See :ref:`mil` for more information.

Main functions
**************

.. autofunction:: mil_config
.. autofunction:: train_mil
.. autofunction:: train_fastai
.. autofunction:: train_multimodal_mil
.. autofunction:: build_fastai_learner
.. autofunction:: build_multimodal_learner
.. autofunction:: eval_mil
.. autofunction:: predict_mil
.. autofunction:: predict_multimodal_mil
.. autofunction:: predict_from_bags
.. autofunction:: predict_from_multimodal_bags
.. autofunction:: predict_slide
.. autofunction:: get_mil_tile_predictions
.. autofunction:: generate_mil_features

TrainerConfig
*************

.. autoclass:: slideflow.mil.TrainerConfig
.. autosummary::

    TrainerConfig.model_fn
    TrainerConfig.loss_fn

.. autofunction:: slideflow.mil.TrainerConfig.to_dict
.. autofunction:: slideflow.mil.TrainerConfig.json_dump


MILModelConfig
**************

.. autoclass:: MILModelConfig

CLAMModelConfig
***************

.. autoclass:: slideflow.clam.CLAMModelConfig

