.. _uncertainty:

Uncertainty Quantification
==========================

Several uncertainty quantification (UQ) methods have been developed for deep learning models and tested in digital histopathology, including MC Dropout, deep ensembles, hyper-deep ensembles, and test-time augmentation.

In verison 1.1, we implemented a dropout-based method of uncertainty estimation (`arXiv paper <https://arxiv.org/abs/2204.04516>`_). MC dropout UQ methods exploit the observation that neural networks with dropout approximate sampling of the Bayesian posterior. Images undergo multiple forward passes in a dropout-enabled network during inference, which results in a distribution of predictions. The standard deviation of such a distribution represents the uncertainty estimate.

Training with UQ
****************

Training models with UQ is straightforward, requiring only two hyperparameter settings. Dropout must be enabled (set to a nonzero value), and ``uq`` should be ``True``:

.. code-block:: python

    import slideflow as sf

    params = sf.ModelParams(
      tile_px=299,
      tile_um=302,
      ...,
      dropout=0.1,
      uq=True
    )

All predictions from this model will now involve 30 forward passes through the network, with dropout always enabled. Final tile-level predictions will be the average from each of the 30 forward passes, and tile-level uncertainty will be the standard deviation of the forward passes.

Evaluating with UQ
******************

Any pipeline function using a model trained with UQ will automatically estimate uncertainty, without any additional action from the user. When model predictions are saved during validation or evaluation, uncertainty estimates will be saved alongside predictions in the tile- and patient-level predictions files found in the model folder.

Uncertainty heatmaps
********************

If a model was trained with UQ enabled, the :meth:`slideflow.Project.generate_heatmaps()` function will automatically create uncertainty heatmaps alongside the prediction heatmaps.

Slide-level confidence & uncertainty thresholding
*************************************************

Uncertainty information can be exploited to separate slide- and patient-level predictions into low- and high-confidence. We developed an uncertainty thresholding algorithm (`BISCUIT <https://github.com/jamesdolezal/biscuit/>`_) to accomplish this task. Further details about slide-level confidence estimation and uncertainty thresholding can be found in our manuscript `detailing the method <https://arxiv.org/abs/2204.04516>`_.