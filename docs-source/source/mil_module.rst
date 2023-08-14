.. currentmodule:: slideflow.mil

slideflow.mil
==============

This submodule contains tools for multiple-instance learning (MIL) model training and evaluation. See :ref:`clam_mil` for more information.

Main functions
**************

.. autofunction:: mil_config
.. autofunction:: train_mil
.. autofunction:: train_clam
.. autofunction:: train_fastai
.. autofunction:: build_fastai_learner
.. autofunction:: eval_mil
.. autofunction:: predict_slide

TrainerConfigFastAI
*******************

.. autoclass:: slideflow.mil.TrainerConfigFastAI
.. autosummary::

    slideflow.mil.TrainerConfigFastAI.model_fn
    slideflow.mil.TrainerConfigFastAI.loss_fn

.. autofunction:: slideflow.mil.TrainerConfigFastAI.to_dict
.. autofunction:: slideflow.mil.TrainerConfigFastAI.json_dump

TrainerConfigCLAM
*****************

.. autoclass:: slideflow.mil.TrainerConfigCLAM
.. autosummary::

    slideflow.mil.TrainerConfigCLAM.model_fn
    slideflow.mil.TrainerConfigCLAM.loss_fn

.. autofunction:: slideflow.mil.TrainerConfigCLAM.to_dict
.. autofunction:: slideflow.mil.TrainerConfigCLAM.json_dump

ModelConfigFastAI
*****************

.. autoclass:: ModelConfigFastAI

ModelConfigCLAM
***************

.. autoclass:: ModelConfigCLAM

