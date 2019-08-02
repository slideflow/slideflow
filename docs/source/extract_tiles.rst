.. _filtering:

Tile extraction
===============

Once a validation plan has been established, our next step is tile extraction. To configure our project to extract tiles, uncomment the line with the ``extract_tiles()`` function. If you want to extract tiles for all of your slides, you may delete all of the suggested arguments and simply use the following syntax:

.. code-block:: python

	SFP.extract_tiles()

The documentation for the ``extract_tiles`` function is given below:

.. autofunction:: slideflow.SlideFlowProject.extract_tiles
   :noindex:

The optional ``filter_header`` and ``filter_values`` arguments are used to filter a subset of slides to act on, according to your annotations file. If not supplied, all valid slides will be used by default.

To filter according to a column in your annotations file, set ``filter_header`` to the column name. If you are filtering by multiple columns, supply names in a list. Next, set ``filter_values`` to a list containing all possible values you want to include. If you are filtering by multiple columns, this will be a nested list.

For example, to extract tiles only for slides that are labeled as "train" in the "dataset" column header in your annotations file, do:

.. code-block:: python

	SFP.extract_tiles(filter_header="dataset", filter_values=["train"])

To further filter by the annotation header "mutation_status", including only slides with the category "braf" or "ras", use nested lists:

.. code-block:: python

	SFP.extract_tiles(filter_header=["dataset", "mutation_status"], filter_values=[["train"], ["braf", "ras"]])

*Note: the arguments "filter_header" and "filter_values" can be used for filtering input slides in many slideflow functions, including train(), evaluate(), generate_heatmaps(), and generate_mosaic().*

To begin tile extraction, save the ``actions.py`` file and run your project as described in :ref:`execute`. 

Tiles will be extracted at the pixel and micron size specified in the project settings file, ``settings.json``. Tiles will be automatically stored in TFRecord format and separated into training and validation steps if required (necessary when validation data is generated on per-tile basis; see :ref:`validation_planning`).
