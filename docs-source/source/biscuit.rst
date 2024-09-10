.. currentmodule:: slideflow.biscuit

slideflow.biscuit
=================

This module contains an official implementation of `BISCUIT <https://www.nature.com/articles/s41467-022-34025-x>`__, an uncertainty quantification and confidence thresholding algorithm for whole-slide images. The original implementation, which includes instructions for reproducing experimental results reported in the manuscript, is available on `GitHub <https://github.com/jamesdolezal/biscuit>`__.

This module is requires the ``slideflow-noncommercial`` package, which can be installed with:

.. code-block:: bash

    pip install slideflow-noncommercial

See :ref:`uncertainty` for more information.

.. autofunction:: find_cv
.. autofunction:: get_model_results

biscuit.Experiment
******************
.. autoclass:: Experiment
.. autofunction:: slideflow.biscuit.Experiment.display
.. autofunction:: slideflow.biscuit.Experiment.plot_uq_calibration
.. autofunction:: slideflow.biscuit.Experiment.results
.. autofunction:: slideflow.biscuit.Experiment.thresholds_from_nested_cv
.. autofunction:: slideflow.biscuit.Experiment.train
.. autofunction:: slideflow.biscuit.Experiment.train_nested_cv

biscuit.hp
**********

.. autofunction:: slideflow.biscuit.hp.nature2022

biscuit.threshold
*****************
.. autofunction:: slideflow.biscuit.threshold.apply
.. autofunction:: slideflow.biscuit.threshold.detect
.. autofunction:: slideflow.biscuit.threshold.from_cv
.. autofunction:: slideflow.biscuit.threshold.plot_uncertainty
.. autofunction:: slideflow.biscuit.threshold.process_group_predictions
.. autofunction:: slideflow.biscuit.threshold.process_tile_predictions

biscuit.utils
*************

.. autofunction:: slideflow.biscuit.utils.auc
.. autofunction:: slideflow.biscuit.utils.auc_and_threshold
.. autofunction:: slideflow.biscuit.utils.df_from_cv
.. autofunction:: slideflow.biscuit.utils.eval_exists
.. autofunction:: slideflow.biscuit.utils.find_cv
.. autofunction:: slideflow.biscuit.utils.find_cv_early_stop
.. autofunction:: slideflow.biscuit.utils.find_eval
.. autofunction:: slideflow.biscuit.utils.find_model
.. autofunction:: slideflow.biscuit.utils.get_model_results
.. autofunction:: slideflow.biscuit.utils.get_eval_results
.. autofunction:: slideflow.biscuit.utils.model_exists
.. autofunction:: slideflow.biscuit.utils.prediction_metrics
.. autofunction:: slideflow.biscuit.utils.read_group_predictions
.. autofunction:: slideflow.biscuit.utils.truncate_colormap

biscuit.delong
**************

.. autofunction:: slideflow.biscuit.delong.fastDeLong
.. autofunction:: slideflow.biscuit.delong.delong_roc_variance
.. autofunction:: slideflow.biscuit.delong.delong_roc_test
