.. _uncertainty:

Uncertainty Quantification
==========================

Several uncertainty quantification (UQ) methods have been developed for deep learning models and tested in digital histopathology, including MC Dropout, deep ensembles, hyper-deep ensembles, and test-time augmentation.

Slideflow includes a dropout-based method of uncertainty estimation. MC dropout UQ methods exploit the observation that neural networks with dropout approximate sampling of the Bayesian posterior. Images undergo multiple forward passes in a dropout-enabled network during inference, which results in a distribution of predictions. The standard deviation of such a distribution represents the uncertainty estimate.

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

Uncertainty thresholding
************************

Uncertainty information can be exploited to separate slide- and patient-level predictions into low- and high-confidence. We developed an uncertainty thresholding algorithm (`BISCUIT <https://github.com/jamesdolezal/biscuit/>`_) to accomplish this task, which is available in :mod:`slideflow.biscuit`. Algorithmic details and validation studies can be found in our `manuscript <https://www.nature.com/articles/s41467-022-34025-x>`_ detailing the method.

Here, we will run through an example of how to apply this UQ thresholding strategy for a weakly-supervised classification model. At present, ``biscuit`` only supports uncertainty estimation and confidence thresholding for binary classification.

Prepare an Experiment
---------------------

Start by creating a Slideflow project and then initializing a ``biscuit`` experiment, including the outcome target and the two classes.  We will be training models to predict ``"HPV_status"``, with the two classes ``"positive"`` and ``"negative"``.

.. code-block:: python

    import slideflow as sf
    from slideflow import biscuit

    # Create a Slideflow project
    P = sf.Project(...)

    # Initialize a biscuit experiment
    experiment = biscuit.Experiment(
        train_project=P,
        outcome='HPV_status',
        outcome1='negative',
        outcome2='positive'
    )

Next, prepare the model hyperparameters. Here, we will use the hyperparameters used in the original manuscript.

.. code-block:: python

    hp = biscuit.hp.nature2022()

Train with cross-validation
---------------------------

We'll start by training models in cross-validation on the full dataset. We'll use the default three-fold cross-validation strategy. We need to supply a label for experiment model tracking, which will be used for the rest of our experiments.

.. code-block:: python

    # Train outer cross-validation models.
    experiment.train(hp=hp, label='HPV')

Models will be saved in the project model folder.

Train inner cross-validation
----------------------------

Next, for each of the three cross-validation models trained, we will perform 5-fold nested cross-validation. Uncertainty thresholds are determined from nested cross-validation results.

.. code-block:: python

    # Train inner, nested cross-validation models.
    experiment.train_nested_cv(hp=hp, label='HPV')

Models will again be saved in the project model directory. We can view a summary of the results from these cross-validation studies using the :func:`biscuit.find_cv()` and :func:`biscuit.get_model_results()` functions.

.. code-block:: python

    from slideflow.biscuit import find_cv, get_model_results

    # Print results from outer cross-validation
    cv_models = find_cv(
        project=P,
        label='HPV',
        outcome='HPV_status'
    )
    for m in cv_models:
        results = get_model_results(m, outcome='HPV_status', epoch=1)
        print(m, results['pt_auc'])

Uncertainty thresholds are calculated using results from the inner cross-validation studies. :func:`biscuit.Experiment.thresholds_from_nested_cv` will calculate and return uncertainty and prediction thresholds.

.. code-block:: python

    # Calculate uncertainty thresholds
    df, thresh = experiment.thresholds_from_nested_cv(label='HPV')
    print(thresh)

.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    {'tile_uq': 0.02726791,
     'slide_uq': 0.0147878695,
     'tile_pred': 0.41621968,
     'slide_pred': 0.4756707}


Apply thresholds to test set
----------------------------

Finally, we can apply these thresholds to a held out test set. First, generate predictions for a held-out test set as described in :ref:`evaluation`. Locate the parquet file containing the saved tile-level predictions and load it into a DataFrame. Rename the columns in the dataframe so that ground-truth is ``y_true``, predictions are ``y_pred``, and uncertainty is ``uncertainty``.

.. code-block:: python

    import pandas as pd

    # Load tile-level predictions from a test set evaluation
    df = pd.read_parquet('/path/to/tile_predictions.parquet.gzip')

    # Rename the columns to y_true, y_pred, and uncertainty
    df.rename(columns={
        'HPV_status-y_true': 'y_true,
        'HPV_status-y_pred1': 'y_pred',
        'HPV_status-uncertainty1': 'uncertainty'
        '
    })

Use :func:`biscuit.threshold.apply` to apply the previously-determined thresholds to these predictions. This will return classifier metrics (AUROC, accuracy, sensitivity, specificity) for high-confidence predictions and a dataframe of slide-level high-confidence predictions. Slides with low-confidence predictions will be omitted. The percentage of slides with high-confidence predictions will be reported as ``'percent_incl'``.

.. code-block:: python

    # Calculate high-confidence slide-level predictions
    metrics, high_conf_df = biscuit.threshold.apply(
        df,           # Dataframe of tile-level predictions
        **thresh,     # Uncertainty thresholds
        level='slide' # We want slide-level predictions
    )
    print(metrics)

.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    {'auc': 0.9703296703296704,
     'percent_incl': 0.907051282051282,
     'acc': 0.9222614840989399,
     'sensitivity': 0.9230769230769231,
     'specificity': 0.9214285714285714}