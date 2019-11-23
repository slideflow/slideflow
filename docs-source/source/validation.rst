.. _validation_planning:

Validation Planning
===================

An important first step in creating a new project is to determine the validation plan. When training neural networks, data is segregated into three groups:

1) **Training data** - data used for learning during training
2) **Validation data** - data used for performance testing during training
3) **Evaluation data** - data used for final evaluation once training has completed. Preferably an external cohort.

Validation data is used to assess model performance during training. Good performance on validation data generally indicates that your model is learning generalizable features. Poor performance on validation data despite good training performance is a sign of overfitting.

Over the course of a project, you will perform many model training iterations as you test different hyperparameter combinations. Throughout all of these training iterations, you will be using the same validation data to assess performance. Ideally, you will end up choosing hyperparameters which result in the best performance on your validation data. 

Once you have finished training models on all of your hyperparameter combinations, pick the best hyperparameters and train one last model on *both* your training data and validation data. Assess this final model's performance on your reserved evaluation dataset; this is your model's final performance.

Saving a subset of data for final evaluation testing reduces the risk of bias and ensures a more accurate assessment of model generalizability.

Configuring a validation plan
*****************************

There are several ways you can plan to validate your data. The project configuration options in slideflow are the following:

- **validation_target**:  *'per-patient'* or *'per-tile'*
- **validation_strategy**:  *'bootstrap'*, *'k-fold'*, *'fixed'*, *'none'*
- **validation_fraction**:  (float between 0-1)
- **validation_k_fold**:  int

validation_target
^^^^^^^^^^^^^^^^^

The first consideration is whether you will be separating validation data on a **tile-level** (setting aside a certain % of tiles from every slide for validation) or **slide-level** (setting aside a certain number of slides). In most instances, it is best to use slide-level validation separation, as this will offer the best insight into whether your model is generalizable on a different set of slides.

*Note: if using tile-level validation, this must be configured at the time of tile extraction due to the way TFRecord tile data is stored. If you change validation_target mid-project, you may need to re-extract tiles.*

validation_strategy
^^^^^^^^^^^^^^^^^^^

The ``validation_strategy`` option determines how the validation data is selected.

If **fixed**, a certain percentage of your training data is set aside for testing (determined by ``validation_fraction``). The chosen validation subset is saved to a log file and will be re-used for all training iterations.

If **bootstrap**, validation data will be selected at random (percentage determined by ``validation_fraction``), and all training iterations will be repeated a number of times equal to ``validation_k_fold``. The saved and reported model training metrics will be an average of all bootstrap iterations. 

If **k-fold**, training data will be separated into *k* number of groups (where *k* is equal to ``validation_k_fold``), and all training iterations will be repeated *k* number of times using k-fold cross validation. The saved and reported model training metrics will be an average of all k-fold iterations. 

If **none**, no validation testing will be performed.

Selecting an evaluation cohort
******************************

Unlike validation testing, selecting a final evaluation dataset is more straightforward. Ideally, your evaluation data will come from an external source; e.g. slides from another institution, or from an entirely different group of patients, or slides taken during a different time period. If you do not have an external evaluation dataset, you may instead choose to set aside a certain proportion of your original dataset. There is no magic number for how much data to set aside, but I generally recommend setting aside about 30% for final evaluation.
