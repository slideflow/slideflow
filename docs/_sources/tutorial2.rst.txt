.. _tutorial2:

Tutorial 2: Model training (advanced)
=======================================

In the first tutorial, we used :meth:`slideflow.Project.train` to execute training. This project function is useful in that it:

1) Configures outcome labels in a manner supporting multiple outcomes
2) Configures mini-batch balancing
3) Supports full cross-validation, as opposed to training a single model at a time
4) Supports hyperparameter sweeps
5) Prepares any additional slide-level model input from clinical annotations
6) Logs model parameters to the model directory

|

In this tutorial, we will walk through training a model using the :class:`slideflow.Datset` and :class:`slideflow.model.Trainer` classes directly in an interactive python session, rather than using the built-in :meth:`slideflow.Project.train` function. This tutorial will demonstrate how model training happens under the hood, in case you would like to customize any part of the model training pipeline.

Project Setup
*************

Using the same project configuration as the first tutorial, we will set up a new project:

.. code-block:: python

    >>> import slideflow as sf
    >>> P = sf.Project('/home/er_project', name="Breast_ER", annotations=...)

If you initialize a project with keywords, you will need to manually create a new dataset source with the :meth:`slideflow.Project.add_dataset` method:

.. code-block:: python

    >>> P.add_source(
    ...   name="NAME",
    ...   slides="/slides/directory",
    ...   roi="/roi/directory",
    ...   tiles="/tiles/directory",
    ...   tfrecords="/tfrecords/directory")
    ... )

As before, set up your annotations file, including columns "patient", "er_status_by_ihc", "dataset", and "slide".

Creating a Dataset
******************

Next, create a :class:`slideflow.Dataset` instance to indicate which slides we will be working with (again, we are working with 256 px tiles at 128 um). We only want to use our training set for now, and only include slides with an ER status annotation. For this, we will use the filters arguments.

.. code-block:: python

    >>> dataset = P.dataset(
    ...   tile_px=256,
    ...   tile_um=128,
    ...   filters={
    ...     'dataset': ['train'],
    ...     'er_status_by_ihc': ['Positive', 'Negative']
    ... })

To extract tiles from the slides in this dataset, use the :meth:`slideflow.Dataset.extract_tiles` method:

.. code-block:: python

    >>> dataset.extract_tiles()

We can see how many tiles there are in our dataset by inspecting the ``num_tiles`` attribute:

.. code-block:: python

    >>> dataset.num_tiles
    4503965

We can use the dataset to get our ER status labels. The :meth:`slideflow.Dataset.labels` method returns the dictionary mapping slides names to outcomes as the first parameter, and a list of unique outcomes as the second parameter (which is not required at this time).

.. code-block:: python

    >>> labels, _ = dataset.labels('er_status_by_ihc')
    2021-10-06 13:27:00 [INFO] - er_status_by_ihc 'Negative' assigned to value '0' [234 slides]
    2021-10-06 13:27:00 [INFO] - er_status_by_ihc 'Positive' assigned to value '1' [842 slides]

We can see the slideflow logs showing us that 234 slides with the outcome label "Negative" were assigned to the numerical outcome "0", and 842 "Positive" slides were assigned "1".

Next, we'll need to split this dataset into a training and validation set. We'll start by training on the first of 3 k-folds for cross-validated training. To split a dataset, use the :meth:`slideflow.Dataset.split` method. We'll need to provide our labels to ensure that the outcome categories are balanced in the training and validation sets.

.. code-block:: python

    >>> train_dts, val_dts = dataset.split(
    ...   model_type='classification',
    ...   labels=labels,
    ...   val_strategy='k-fold',
    ...   val_k_fold=3,
    ...   k_fold_iter=1
    ... )
    2021-10-06 13:27:39 [INFO] - No validation log provided; unable to save or load validation plans.
    2021-10-06 13:27:39 [INFO] - Category   0       1
    2021-10-06 13:27:39 [INFO] - K-fold-0   69      250
    2021-10-06 13:27:39 [INFO] - K-fold-1   69      250
    2021-10-06 13:27:39 [INFO] - K-fold-2   68      249
    2021-10-06 13:27:39 [INFO] - Using 636 TFRecords for training, 319 for validation

The first informational log tells us that no validation log was provided. We could have optionally provided a JSON file path to the argument ``splits``; this method can record splits to the provided file for automatic re-use later (helpful for hyperparameter sweeps). However, for the purposes of this tutorial, we have opted not to save our validation plan.

The rest of the log output shows us the distribution of our outcome categories among the k-folds, as well as the total number of slides for training and validation.

At this point, we can also add categorical balancing to our dataset (see :ref:`balancing`). Since we have nearly 4 times as many ER-positive samples as ER-negative, it may be helpful to balance each batch to have an equal proportion of positives and negatives. We can accomplish this with the :meth:`slideflow.Dataset.balance` method:

.. code-block:: python

    >>> train_dts = train_dts.balance('er_status_by_ihc')

Training
********

Now that our dataset is prepared, we can begin setting up our model and trainer. Our model training parameters are configured with :class:`slideflow.ModelParams`.

.. code-block:: python

    >>> hp = sf.ModelParams(
    ...   tile_px=256,
    ...   tile_um=128,
    ...   model='xception',
    ...   batch_size=32,
    ...   epochs=[3]
    ... )

In addition to the above model parameters, our trainer will need the outcome labels, patient list (dict mapping slide names to patient IDs, as some patients can have more than one slide), and the directory in which to save our models:

.. code-block:: python

    >>> trainer = sf.model.build_trainer(
    ...   hp=hp,
    ...   outdir='/some/directory',
    ...   labels=labels,
    ... )

Now we can start training. Pass the training and validation datasets to the :meth:`slideflow.model.Trainer.train` method of our trainer, assigning the output to a new variable ``results``

.. code-block:: python

    >>> results = trainer.train(train_dts, val_dts)

You'll see logs recording model structure, training progress across epochs, and metrics. The training and validation performance results are returned in dictionary format. ``results`` should have contents similar to the following (values will be different):

.. code-block:: json

    {
      "epochs": {
        "epoch3": {
          "train_metrics": {
            "loss": 0.497
            "accuracy": 0.806
            "val_loss": 0.719
            "val_accuracy": 0.778
          },
          "val_metrics": {
            "loss": 0.727
            "accuracy": 0.770
          },
          "tile": {
            "Outcome 0": [
              0.580
              0.580
            ]
          },
          "slide": {
            "Outcome 0": [
              0.658
              0.658
            ]
          },
          "patient": {
            "Outcome 0": [
              0.657
              0.657
            ]
          }
        }
      }
    }

Training results are separated with nested dictionaries according to epoch. The raw training metrics and validation metrics are stored with the keys ``"train_metrics"`` and ``"val_metrics"``, and tile-, slide-, and patient-level metrics (AUROC for classification, R-squared for regression outcomes, and concordance index for survival models) is reported under the ``"tile"``, ``"slide"``, and ``"patient"`` keys for each outcome, respectively.