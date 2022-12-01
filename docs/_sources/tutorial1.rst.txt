.. _tutorial1:

Tutorial 1: Model training (simple)
=====================================

In this first tutorial, we will walk through the steps needed to take an example project from start to finish, using
the bundled ``run_project.py`` script to execute pipeline functions. As with all of these tutorials, we will use
publicly available data from `The Cancer Genome Atlas (TCGA) <https://portal.gdc.cancer.gov>`_. In this first tutorial,
we will train a model to predict ER status from breast cancer slides.

Examples will be given assuming project files are in the directory ``/home/er_project`` and slides are in
``/home/brca_slides``, although you will need to customize these paths according to your needs.

Project Planning
****************

First, download slides and annotations for the TCGA-BRCA project using the `legacy GDC portal
<https://portal.gdc.cancer.gov/legacy-archive/search/f>`_. We should have a total of 1133 diagnostic slides across 1062
patients. Our outcome of interest is "er_status_by_ihc", of which 1011 have a documented result (either "Positive"
or "Negative"), giving us our final patient count of 1011.

To create a new project, use the ``run_project.py`` script:

.. code-block:: bash

    $ python3 run_project.py -p /home/er_project

We will then be taken through an interactive prompt asking for project settings. When prompted, use the
following settings (mostly defaults):

+-------------------------------+-------------------------------------------------------+
| **name**                      | Breast_ER                                             |
+-------------------------------+-------------------------------------------------------+
| **annotations**               | ./annotations.csv (default)                           |
+-------------------------------+-------------------------------------------------------+
| **dataset_config**            | ./datasets.json (default)                             |
+-------------------------------+-------------------------------------------------------+
| **sources**                   | BRCA                                                  |
+-------------------------------+-------------------------------------------------------+
| **models_dir**                | ./models (default)                                    |
+-------------------------------+-------------------------------------------------------+
| **eval_dir**                  | ./eval                                                |
+-------------------------------+-------------------------------------------------------+

After a blank datasets.json file is created, we will be prompted to add a new dataset source. Use the following
configuration for the added dataset source:

+-------------------------------+-------------------------------------------------------+
| **source**                    | BRCA                                                  |
+-------------------------------+-------------------------------------------------------+
| **slides**                    | /home/brca_slides                                     |
+-------------------------------+-------------------------------------------------------+
| **roi**                       | /home/brca_slides                                     |
+-------------------------------+-------------------------------------------------------+
| **tiles**                     | /home/er_project/tiles                                |
+-------------------------------+-------------------------------------------------------+
| **tfrecords**                 | /home/er_project/tfrecords                            |
+-------------------------------+-------------------------------------------------------+

For simplicity, we will not be using annotated tumor regions of interest (ROI), instead training on whole-slide images.

Setting up annotations
**********************

With our project initialized, we can set up our annotations file. Use the downloaded annotations file to create a new
CSV file, with a column "patient" indicating patient name (in the case of TCGA, these are in the format
TCGA-SS-XXXX, where SS indicates site of origin and XXXX is the patient identifier), and a column "er_status_by_ihc"
containing our outcome of interest. Add a third column "slide" containing the name of the slide associated with the
patient. If there are multiple slides per patient, list each slide on a separate row. Finally, add a column "dataset"
to indicate whether the slide should be used for training or evaluation. Set aside somewhere around 10-30% of the
dataset for evaluation.

.. note::

    If patient names are identical to the slide filenames, the "slide" column does not need to be manually added, as
    slideflow will auto-associate slides to patients.

Your annotations file should look something like:

+-----------------------+--------------------+-----------+-----------------------------------+
| *patient*             | *er_status_by_ihc* | *dataset* | *slide*                           |
+-----------------------+--------------------+-----------+-----------------------------------+
| TCGA-EL-A23A          | Positive           | train     | TCGA-EL-A3CO-01Z-00-DX1-7BF5F...  |
+-----------------------+--------------------+-----------+-----------------------------------+
| TCGA-EL-A24B          | Negative           | train     | TCGA-EL-A24B-01Z-00-DX1-7BF5F...  |
+-----------------------+--------------------+-----------+-----------------------------------+
| TCGA-EL-A25C          | Positive           | train     | TCGA-EL-A25C-01Z-00-DX1-7BF5F...  |
+-----------------------+--------------------+-----------+-----------------------------------+
| TCGA-EH-B31C          | Negative           | eval      | TCGA-EH-B31C-01Z-00-DX1-7BF5F...  |
+-----------------------+--------------------+-----------+-----------------------------------+
| ...                   | ...                | ...       | ...                               |
+-----------------------+--------------------+-----------+-----------------------------------+


Tile extraction
***************

The next step is to extract tiles from our slides. Find the sample ``actions.py`` file in the project folder, which we
will modify and use to execute our pipeline functions. Delete the commented-out examples in this file.

For this example, we will use a 256px x 256px tile size, at 0.5 µm/pixel (128 um). Add the following
to the project ``actions.py`` file:

.. code-block:: python

    def main(P):
        # Extract tiles at 256 pixels, 0.5 um/px
        P.extract_tiles(tile_px=256, tile_um=128)

.. hint::
    Tile extraction speed is greatly improved when slides are on an SSD or ramdisk; slides can be automatically
    buffered to an SSD or ramdisk directory by passing a directory to the argument ``buffer``.

    .. code-block:: python

        P.extract_tiles(256, 128, buffer='/mnt/ramdisk')

Training
********

After tiles are extracted, the dataset will be ready for training. We will train with a single set of manually defined
hyperparameters, which we can configure with :class:`slideflow.model.ModelParams`. We will use the
`Xception <https://arxiv.org/abs/1610.02357>`_ model with a batch size of 32, otherwise keeping defaults.

.. code-block:: python

    def main(P):
        from slideflow.model import ModelParams
        ...

        hp = ModelParams(
            tile_px=256,
            tile_um=128,
            model='xception',
            batch_size=32,
            epochs=[3]
        )

For training, we will use 5-fold cross-validation on the training dataset. To set up training, invoke the
:meth:`slideflow.Project.train` function with the outcome of interest, our hyperparameters, and our validation plan.
We will use the ``filters`` argument to limit our training to the "train" dataset, as well as limit the training
to only include patients with documented ER status (otherwise a blank "" would be marked as a third outcome).

.. code-block:: python

    def main(P):
        ...

        # Train with 5-fold cross-validation
        P.train(
            'er_status_by_ihc',
            params=hp,
            val_k_fold=5,
            filters={'dataset': ['train'],
                     'er_status_by_ihc': ['Positive', 'Negative']}
        )

After cross validation is complete, we will want to have a model trained across the entire dataset, so we can assess
performance on our held-out evaluation set. To train a model across the entire training dataset without validation,
we will set ``val_strategy`` to ``None``:

.. code-block:: python

    def main(P):
        ...

        # Train across the entire training dataset
        P.train(
            'er_status_by_ihc',
            params=hp,
            val_strategy='none',
            filters={'dataset': ['train'],
                     'er_status_by_ihc': ['Positive', 'Negative']}
        )

Now, it's time to start our pipeline. To review, our ``actions.py`` file at this point should look like:

.. code-block:: python

    def main(P):
        from slideflow.model import ModelParams

        # Extract tiles at 256 pixels, 0.5 um/px
        P.extract_tiles(tile_px=256, tile_um=128)

        hp = ModelParams(
            tile_px=256,
            tile_um=128,
            model='xception',
            batch_size=32,
            epochs=[3, 5, 10]
        )

        # Train with 5-fold cross-validation
        P.train(
            'er_status_by_ihc',
            params=hp,
            val_k_fold=5,
            filters={'dataset': ['train'],
                     'er_status_by_ihc': ['Positive', 'Negative']}
        )

        # Train across the entire training dataset
        P.train(
            'er_status_by_ihc',
            params=hp,
            val_strategy='none',
            filters={'dataset': ['train'],
                     'er_status_by_ihc': ['Positive', 'Negative']}
        )

To execute these functions, use the ``run_project.py`` script, passing the project directory with the ``-p`` flag.

.. code-block:: bash

    $ python3 run_project.py -p /home/er_project

The final training results should should show an average AUROC of around 0.87, with average AP around 0.83. Tile, slide,
and patient-level receiver operator curves are saved in the model folder, along with precision-recall curves (not shown):

.. list-table::

    * - .. figure:: er_roc_tile.png

        Tile-level receiver operator curve

      - .. figure:: er_roc_patient.png

        Patient-level receiver operator curve


Monitoring with Tensorboard
***************************

Tensorboard-formatted training and validation logs are saved the model directory. To monitor training with Tensorboard:

.. code-block:: bash

    $ tensorboard --logdir=/project_path/models/00001-outcome-HP0

Tensorboard can then be accessed by navigating to ``https://localhost:6006`` in a browser.

Monitoring with Neptune
***********************

Experiments can be automatically logged with `Neptune.ai <https://app.neptune.ai>`_. To enable logging, first locate your Neptune API token and workspace ID, and configure the environmental variables ``NEPTUNE_API_TOKEN`` and ``NEPTUNE_WORKSPACE``.

With the environmental variables set, Neptune logs are enabled either by passing a ``-n`` flag to the ``run_project.py`` script:

.. code-block:: bash

    $ python3 run_project.py -n -p /project_path/

or by passing ``use_neptune=True`` to the ``slideflow.Project`` class:

.. code-block:: python

    P = sf.Project('/project/path', use_neptune=True)