Setting up a Project
====================

The easiest way to use ``slideflow`` is through the bundled project management class, :class:`slideflow.Project`, which supports unified datasets, annotations, and project directory structure for all of the various pipeline functions. Projects are initialized at a directory, with settings saved in a ``settings.json`` file. Below are the configuration settings used to create a project:

+-------------------------------+-------------------------------------------------------+
| **name**                      | Project name.                                         |
+-------------------------------+-------------------------------------------------------+
| **annotations**               | Path to CSV containing annotations.                   |
|                               | Each line is a unique slide.                          |
+-------------------------------+-------------------------------------------------------+
| **dataset_config**            | Path to JSON file containing dataset configuration.   |
+-------------------------------+-------------------------------------------------------+
| **sources**                   | Names of dataset(s) to include in the project.        |
+-------------------------------+-------------------------------------------------------+
| **models_dir**                | Path, where model files and results are saved.        |
+-------------------------------+-------------------------------------------------------+
| **eval_dir**                  | Path, where model evaluation results are saved.       |
+-------------------------------+-------------------------------------------------------+

Project class
*************

To interactively create a new project, initialize a Project with the ``from_prompt`` initializer:

.. code-block:: python

    import slideflow as sf
    P = sf.Project.from_prompt('/path/to/project/directory')

You will then be prompted for each project configuration setting.

Alternatively, you may provide this configuration through keyword arguments:

.. code-block:: python

    import slideflow as sf
    P = sf.Project('/path/to/project/directory', name="MyProject", ...)

Once a project has been initialized at a directory, you may then load the project with the following syntax:

.. code-block:: python

    import slideflow as sf
    P = sf.Project('/path/to/project/directory')

Pipeline functions are then called on ``P``.

Alternatively, you can use the bundled ``run_project.py`` script to execute project functions stored in ``actions.py`` files in project directories. When ``run_project.py`` is run, it initializes a ``Project`` object at a given directory, then looks for and loads an ``actions.py`` file in this directory, executing functions contained therein.

To create a new project with this script, or execute functions on an existing project, use the following syntax:

.. code-block:: bash

    $ python3 run_project.py -p /path/to/project/directory

...where the -p flag is used to designate the path to your project directory. Other available flags can be seen by running ``python3 run_project.py --help``.

Configuring Datasets
********************

Once initial project settings are established, you will need to either create or load a dataset configuration, which will specify directory locations for slides, ROIs, tiles, and TFRecords for each group of slides.

Dataset configurations are saved in a JSON file with the below syntax. Dataset configuration files can be shared and used across multiple projects, or saved locally within a project directory.

.. code-block:: json

    {
        "SOURCE":
        {
            "slides": "/directory",
            "roi": "/directory",
            "tiles": "/directory",
            "tfrecords": "/directory",
        }
    }

Datasets are configured either interactively at the time of project initialization, or may be added by calling ``Project.add_dataset()``:

.. code-block:: python

    P.add_source(
        name="NAME",
        slides="/slides/directory",
        roi="/roi/directory",
        tiles="/tiles/directory",
        tfrecords="/tfrecords/directory"
    )

Setting up annotations
**********************

Your annotations CSV file is used to label patients and slides with clinical data and/or other outcome variables (or additional input variables) that will be used for training. Each line in the annotations file should correspond to a unique slide. Patients may have more than one slide.

The annotations file may contain as many columns as you would like, but it must contain the following headers at minimum:

- **patient**: patient identifier
- **slide**: slide name / identifier (without the file extension)

An example annotations file is given below:

+-----------------------+---------------+-----------+-----------------------------------+
| *patient*             | *category*    | *dataset* | *slide*                           |
+-----------------------+---------------+-----------+-----------------------------------+
| TCGA-EL-A23A          | EGFR-mutant   | train     | TCGA-EL-A3CO-01Z-00-DX1-7BF5F     |
+-----------------------+---------------+-----------+-----------------------------------+
| TCGA-EL-A35B          | EGFR-mutant   | eval      | TCGA-EL-A35B-01Z-00-DX1-89FCD     |
+-----------------------+---------------+-----------+-----------------------------------+
| TCGA-EL-A26X          | non-mutant    | train     | TCGA-EL-A26X-01Z-00-DX1-4HA2C     |
+-----------------------+---------------+-----------+-----------------------------------+
| TCGA-EL-B83L          | non-mutant    | eval      | TCGA-EL-B83L-01Z-00-DX1-6BC5L     |
+-----------------------+---------------+-----------+-----------------------------------+

An example annotations file is generated each time a new project is initialized. To manually generate an empty annotations file that contains all detected slides, use the bundled ``Project`` function:

.. code-block:: python

    P.create_blank_annotations()

Slide names do not need to be explicitly set in the annotations file by the user. Rather, once a dataset has been set up, slideflow will search through the linked slide directories and attempt to match slides to entries in the annotations file using **patient**. Entries that are blank in the **slide** column will be auto-populated with any detected and matching slides, if available.

.. _execute:

Executing commands
******************

If you plan to use ``run_project.py``, open the ``actions.py`` file located in the project directory. It should look something like this:

.. code-block:: python

    def main(P):
        #P.extract_tiles(tile_px=299, tile_um=302)

        #P.train(
        #      "category",
        #      filters = {
        #          'category': ['NEG', 'POS'],
        #          'dataset': 'train'
        #      },
        #)

        #model = '/path_to_model/'
        #P.evaluate(model, outcomes="category", filters={'dataset': 'eval'})
        #P.generate_heatmaps(model_to_evaluate)
        pass

The ``main()`` function contains several example functions. These serve as examples to help remind you of functions and arguments you can use on projects.

To execute the commands you have prepared in this file, execute the ``run_project.py`` script pointing to your project directory.

.. code-block:: bash

    $ python3 run_project.py -p /path/to/project/directory