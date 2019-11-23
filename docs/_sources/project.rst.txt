Setting up a Project
====================

The easiest way to get the ``slideflow`` pipeline up and running is to use the bundled project management class, ``SlideflowProject``. In this section, we will examine how to set up a new project and then use the project to execute each of the pipeline steps. 

Before we start, make sure you have each of the following:

1.    A collection of slides in SVS format.
2.    A collection of ROIs in CSV format, generated using QuPath.
3.    A plan for which slides will be used for training and which will be used for final testing.

We will use the ``run_project.py`` script in the slideflow directory to both create and manage our project. To create a new project (or run commands on an existing project), use the following syntax:

.. code-block:: console

    james@example:~/slideflow/source$ python3 run_project.py -p /path/to/project/directory

...where the -p flag is used to designate the path to your project directory.

Project Configuration
*********************

Upon first executing the script, you will be asked a series of questions regarding your project. Default answers are given in brackets (if the question is a yes/no question, the default answer is the letter which is capitalized); if you press enter without typing anything, the default will be chosen. You can always change your answers later by editing ``settings.json`` in your project folder. Below is an overview of what you’ll be asked for.

+-------------------------------+-------------------------------------------------------+
| **Root**                      | Root project directory.                               |
+-------------------------------+-------------------------------------------------------+
| **Name**                      | Project name.                                         |
+-------------------------------+-------------------------------------------------------+
| **Annotations file**          | Path to CSV containing annotations.                   |
|                               | Each line is a unique patient and slide.              |
+-------------------------------+-------------------------------------------------------+
| **Dataset config**            | Path to JSON file containing dataset configuration.   |
+-------------------------------+-------------------------------------------------------+
| **Datasets**                  | Which dataset(s) to use as input.                     |
+-------------------------------+-------------------------------------------------------+
| **Delete tiles**              | Whether tiles should be deleted after extraction.     |
+-------------------------------+-------------------------------------------------------+
| **Models directory**          | Path to where model files and results should be saved.|
+-------------------------------+-------------------------------------------------------+
| **Tile microns**              | Size of extracted tiles, in microns.                  |
+-------------------------------+-------------------------------------------------------+
| **Tile pixels**               | Size of extracted tiles, in pixels.                   |
+-------------------------------+-------------------------------------------------------+
| **Use FP16**                  | Whether models should be trained using                |
|                               | 16-bit (vs. 32-bit) precision.                        |
+-------------------------------+-------------------------------------------------------+
| **Validation fraction**       | Fraction of data to save for validation testing.      |
|                               | Default is 20%.                                       |
+-------------------------------+-------------------------------------------------------+
| **Validation target**         | How to select validation data; by tile or by slide.   |
|                               | Default is 'per-patient'                              |
+-------------------------------+-------------------------------------------------------+
| **Validation strategy**       | Type of validation testing (K-fold, fixed plan, none) |
|                               | Default is 'k-fold'                                   |
+-------------------------------+-------------------------------------------------------+
| **Validation K-fold**         | If k-fold validation, how many folds should be used.  |
+-------------------------------+-------------------------------------------------------+

For more information about setting up a validation plan, see :ref:`validation_planning`.

Configuring Datasets
********************

Once initial project settings are established, you will then need to either create or load a dataset configuration, which will specify directory locations for slides, ROIs, tiles, and TFRecords for each group of slides. Each dataset has a name (e.g. BRCA) and may have an associated label (e.g. 604um).

Dataset configurations are saved in a JSON file with the below syntax. Dataset configuration files can be shared and used across multiple projects, or saved locally within a project directory. 

.. code-block:: json

    { 
        "DATASET_NAME": 
        {
            "slides": "./directory",
            "roi": "./directory",
            "tiles": "./directory",
            "tfrecords": "./directory",
            "label": "DATASET_LABEL"
        } 
    }

Setting up annotations
**********************

Your annotations CSV file is used to label patients and slides with clinical data and/or other outcome variables that will be used for training.
Each line in the annotations file should correspond to a unique patient/slide (*Note: v0.9.9 of slideflow currently supports only one slide per patient*).

The annotations file may contain as many columns as you would like, but it must contain the following headers at minimum:

- **submitter_id**: patient identifier
- **slide**: slide name (without the .jpg/.svs extension)
- **category**: some outcome variable

An example annotations file is given below:

+-----------------------+---------------+-----------+-----------------------------------+
| *submitter_id*        | *category*    | *dataset* | *slide*                           |
+-----------------------+---------------+-----------+-----------------------------------+
| TCGA-EL-A23A          | EGFR-mutant   | train     | TCGA-EL-A3CO-01Z-00-DX1-7BF5F     |
+-----------------------+---------------+-----------+-----------------------------------+
| TCGA-EL-A35B          | EGFR-mutant   | eval      | TCGA-EL-A35B-01Z-00-DX1-89FCD     |
+-----------------------+---------------+-----------+-----------------------------------+
| TCGA-EL-A26X          | non-mutant    | train     | TCGA-EL-A26X-01Z-00-DX1-4HA2C     |
+-----------------------+---------------+-----------+-----------------------------------+
| TCGA-EL-B83L          | non-mutant    | eval      | TCGA-EL-B83L-01Z-00-DX1-6BC5L     |
+-----------------------+---------------+-----------+-----------------------------------+

.. _execute:

Executing commands
******************

After the project has been setup, open the ``actions.py`` file located in the project directory. It should look something like this:

.. code-block:: python

    def main(SFP):
        #SFP.extract_tiles(filters = {'to_extract': 'yes'})
            
        #SFP.create_hyperparameter_sweep(finetune_epochs=[5], toplayer_epochs=0, model=['Xception'], pooling=['avg'], loss='sparse_categorical_crossentropy', 
        #                                learning_rate=[0.00001, 0.001], batch_size=64, hidden_layers=[1], optimizer='Adam', early_stop=True, early_stop_patience=15, balanced_training=['BALANCE_BY_CATEGORY'],
        #                                balanced_validation='NO_BALANCE', augment=True, filename=None)
        #SFP.train(
        #      outcome_header="category",
        #      filters = {
        #          'dataset': 'train',
        #          'category': ['negative', 'positive']
        #      },
        #      batch_file='batch_train.tsv')

        #SFP.evaluate(model='HPSweep0-kfold3', outcome_header="category", filters = {'dataset': 'eval'})
        #SFP.generate_heatmaps('HPSweep0')
        #SFP.generate_mosaic('HPSweep0')
        pass

The ``main()`` function contains several example commands, commented out with "#". These serve as examples to help remind you of arguments you can use when executing project functions.

To set up a project command, either uncomment an existing command or type a new command (specific commands will be discussed in more detail in the following sections).

To execute the commands you have prepared, save the ``actions.py`` file and go to your slideflow directory. The ``run_project.py`` will load the saved script in your project directory and begin execution.

.. code-block:: console

    james@example:~/slideflow/source$ python3 run_project.py -p /path/to/project/directory
