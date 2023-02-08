.. _project_setup:

Setting up a Project
====================

Slideflow :ref:`Projects <project>` organize datasets, annotations, and results into a unified directory and provide a high-level API for common tasks.

Use :func:`slideflow.create_project` to create a new project, supplying an annotations file (with patient labels) and path to slides. A new dataset source (collection of slides and tfrecords) will be configured. Additional keyword arguments can be used to specify the location of trecords and saved models.

.. code-block:: python

    import slideflow as sf

    P = sf.create_project(
      root='project_path',
      annotations="./annotations.csv"
      slides='/path/to/slides/'
    )

Project settings are saved in a ``settings.json`` file in the root project directory. Each project will have the following settings:

+-------------------------------+-------------------------------------------------------+
| **name**                      | Project name.                                         |
|                               | Defaults to "MyProject".                              |
+-------------------------------+-------------------------------------------------------+
| **annotations**               | Path to CSV containing annotations.                   |
|                               | Each line is a unique slide.                          |
|                               | Defaults to "./annotations.csv"                       |
+-------------------------------+-------------------------------------------------------+
| **dataset_config**            | Path to JSON file containing dataset configuration.   |
|                               | Defaults to "./datasets.json"                         |
+-------------------------------+-------------------------------------------------------+
| **sources**                   | Names of dataset source(s) to include in the project. |
|                               | Defaults to an empty list.                            |
+-------------------------------+-------------------------------------------------------+
| **models_dir**                | Path, where model files and results are saved.        |
|                               | Defaults to "./models"                                |
+-------------------------------+-------------------------------------------------------+
| **eval_dir**                  | Path, where model evaluation results are saved.       |
|                               | Defaults to "./eval"                                  |
+-------------------------------+-------------------------------------------------------+

Once a project has been initialized at a directory, you may then load the project with the following syntax:

.. code-block:: python

    import slideflow as sf
    P = sf.load_project('/path/to/project/directory')

.. _dataset_sources:

Dataset Sources
***************

A :ref:`dataset source <datasets_and_validation>` is a collection of slides, Regions of Interest (ROI) annotations (if available), and extracted tiles. Sources are defined in the project dataset configuration file, which can be shared and used across multiple projects or saved locally within a project directory. These configuration files have the following format:

.. code-block:: bash

    {
      "SOURCE":
      {
        "slides": "/directory",
        "roi": "/directory",
        "tiles": "/directory",
        "tfrecords": "/directory",
      }
    }

When a project is created with :func:`slideflow.create_project`, a dataset source is automatically created. You can change where slides and extracted tiles are stored by editing the project's dataset configuration file.

It is possible for a project to have multiple dataset sources - for example, you may choose to organize data from multiple institutions into separate sources. You can add a new dataset source to a project with :meth:`Project.add_source`, which will update the project dataset configuration file accordingly.

.. code-block:: python

    P.add_source(
      name="SOURCE_NAME",
      slides="/slides/directory",
      roi="/roi/directory",
      tiles="/tiles/directory",
      tfrecords="/tfrecords/directory"
    )

Read more about :ref:`working with datasets <datasets_and_validation>`.

Annotations
***********

Your annotations file is used to label patients and slides with clinical data and/or other outcome variables that will be used for training. Each line in the annotations file should correspond to a unique slide. Patients may have more than one slide.

The annotations file may contain any number of columns, but it must contain the following headers at minimum:

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

The ``slide`` column may not need to be explicitly set in the annotations file by the user. Rather, once a dataset has been set up, slideflow will search through the linked slide directories and attempt to match slides to entries in the annotations file using **patient**. Entries that are blank in the **slide** column will be auto-populated with any detected and matching slides, if available.