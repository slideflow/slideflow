.. _validation_planning:

Validation Planning
===================

.. figure:: validation.png
    :width: 800px
    :align: center

An important first step in creating a new project is to determine the validation plan. Three groups of data are required:

1) **Training data** - data used for learning during training
2) **Validation data** - data used for testing during training, and early stopping (if applicable)
3) **Evaluation data** - data used for final evaluation once training has completed. Preferably an external cohort.

Validation data is used to assess model performance and generalizability during training. Once the model and parameters have been tuned with training/validation, the final model's performance is assessed on the held-out evaluation set.

Configuring a validation plan
*****************************

There are several ways you can plan to validate your data. The validation settings available include:

- **strategy**:  ``'bootstrap'``, ``'k-fold'``, ``'k-fold-manual'``, ``'k-fold-preserved-site'``, ``'fixed'``, and ``'none'``
- **fraction**:  (float between 0-1) [not used for k-fold validation]
- **k_fold**:  int

The default strategy is ``'k-fold'``, with k=3.

Validation strategy
^^^^^^^^^^^^^^^^^^^

The ``strategy`` option determines how the validation data is selected.

If **fixed**, a certain percentage of your training data is set aside for testing (determined by ``fraction``). The chosen validation subset is saved to a log file and will be re-used for all training iterations.

If **bootstrap**, validation data will be selected at random (percentage determined by ``fraction``), and all training iterations will be repeated a number of times equal to ``k_fold``. The saved and reported model training metrics will be an average of all bootstrap iterations.

If **k-fold**, training data will be automatically separated into *k* number of groups (where *k* is equal to ``k_fold``), and all training iterations will be repeated *k* number of times using k-fold cross validation. The saved and reported model training metrics will be an average of all k-fold iterations.

If you would like to manually separate your data into k-folds, you may do so with the **k-fold-manual** strategy. Assign each slide to a k-fold cohort in the annotations file, and designate the appropriate column header with ``k_fold_header``

The **k-fold-preserved-site** strategy is a cross-validation strategy that ensures site is preserved across the training/validation sets, in order to reduce bias from batch effect as described by `Howard, et al <https://www.nature.com/articles/s41467-021-24698-1>`_. This strategy is recommended when using data from The Cancer Genome Atlas (`TCGA <https://portal.gdc.cancer.gov/>`_).

.. note::
    Preserved-site cross-validation requires either `CPLEX <https://www.ibm.com/analytics/cplex-optimizer>`_ or `Pyomo/Bonmin <https://anaconda.org/conda-forge/coinbonmin>`_. The original implementation of the preserved-site cross-validation algorithm described by Howard et al can be found `on GitHub <https://github.com/fmhoward/PreservedSiteCV>`_.

If **none**, no validation testing will be performed.

Selecting an evaluation cohort
******************************

Designating an evaluation cohort is done using the project annotations file, with a column indicating whether a slide is set aside for evaluation.
The training and evaluation functions include a ``filter`` argument which will allow you to restrict your training or evaluation according to these annotations. This will be discussed in greater detail in subsequent sections.
