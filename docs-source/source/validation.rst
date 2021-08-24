.. _validation_planning:

Validation Planning
===================

An important first step in creating a new project is to determine the validation plan. Three groups of data are required:

1) **Training data** - data used for learning during training
2) **Validation data** - data used for performance testing during training
3) **Evaluation data** - data used for final evaluation once training has completed. Preferably an external cohort.

Validation data is used to assess model performance and generalizability during training. Once the model and hyperparameters have been tuned with training/validation, the final model's performance is assessed on the held-out evaluation set.

Configuring a validation plan
*****************************

There are several ways you can plan to validate your data. The validation settings available include:

- **target**:  *'per-patient'* or *'per-tile'*
- **strategy**:  *'bootstrap'*, *'k-fold'*, *k-fold-manual'*, *k-fold-preserved-site*, *'fixed'*, *'none'*
- **fraction**:  (float between 0-1)
- **k_fold**:  int

The default arguments are 'per-patient' target and 'k-fold' strategy, with K=3. If a different validation strategy is required, create a custom validation plan using :func:`slideflow.project.get_validation_settings` and passing the arguments to customize.

Validation target
^^^^^^^^^^^^^^^^^

The first consideration is whether you will be separating validation data on a **tile-level** (setting aside a certain % of tiles from every slide for validation) or **patient-level** (splitting patients, and their constituent slides, into training/validation datasets). Patient-level validation is highly recommended and used by default, due to risk of bias and overfitting when using tile-level validation.

*Note: if using tile-level validation, this must be configured at the time of tile extraction due to the way TFRecord tile data is stored. If you change validation_target mid-project, you may need to re-extract tiles.*

Validation strategy
^^^^^^^^^^^^^^^^^^^

The ``strategy`` option determines how the validation data is selected.

If **fixed**, a certain percentage of your training data is set aside for testing (determined by ``fraction``). The chosen validation subset is saved to a log file and will be re-used for all training iterations.

If **bootstrap**, validation data will be selected at random (percentage determined by ``fraction``), and all training iterations will be repeated a number of times equal to ``k_fold``. The saved and reported model training metrics will be an average of all bootstrap iterations. 

If **k-fold**, training data will be automatically separated into *k* number of groups (where *k* is equal to ``k_fold``), and all training iterations will be repeated *k* number of times using k-fold cross validation. The saved and reported model training metrics will be an average of all k-fold iterations. 

If you would like to manually separate your data into k-folds, you may do so with the **k-fold-manual** strategy, by indicating which k-fold each slide should be in using the annotations file, and designating the appropriate column header with ``k_fold_header``

The **k-fold-preserved-site** strategy is a cross-validation strategy that ensures site is preserved across the training/validation sets, in order to reduce bias from batch effect as described by `Howard, et al <https://www.nature.com/articles/s41467-021-24698-1>`_. This strategy is recommended when using data from The Cancer Genome Atlas (`TCGA <https://portal.gdc.cancer.gov/>`_).

If **none**, no validation testing will be performed.

Selecting an evaluation cohort
******************************

Designating an evaluation cohort is done using the project annotations file, with a column indicating whether a slide is set aside for evaluation.
The training and evaluation functions include a ``filter`` argument which will allow you to restrict your training or evaluationg according to these annotations. This will be discussed in greater detail in subsequent sections. 
