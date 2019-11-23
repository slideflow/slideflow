.. _filtering:

Tile extraction
===============

Once a validation plan has been established, our next step is tile extraction. To configure our project to extract tiles, uncomment the line with the ``extract_tiles()`` function. If you want to extract tiles for all of your slides, you may delete all of the suggested arguments and simply use the following syntax:

.. code-block:: python

	SFP.extract_tiles()

The documentation for the ``extract_tiles`` function is given below:

.. autofunction:: slideflow.SlideflowProject.extract_tiles
   :noindex:

The optional ``filters`` argument are used to filter a subset of slides to act on, according to your annotations file. If not supplied, all valid slides will be used by default.

To filter according to a columns in your annotations file, pass a dictionary with keys equal to column names and values to a list of all acceptable values you want to include. 

For example, to extract tiles only for slides that are labeled as "train" in the "dataset" column header in your annotations file, do:

.. code-block:: python

	SFP.extract_tiles(filters={"dataset": ["train"]})

To further filter by the annotation header "mutation_status", including only slides with the category "braf" or "ras", do:

.. code-block:: python

	SFP.extract_tiles(filters={"dataset": ["train"], "mutation_status": ["braf", "ras"]})

*Note: the "filters" argument can be also used for filtering input slides in many slideflow functions, including train(), evaluate(), generate_heatmaps(), and generate_mosaic().*

To begin tile extraction, save the ``actions.py`` file and run your project as described in :ref:`execute`. 

Tiles will be extracted at the pixel and micron size specified in the project settings file, ``settings.json``. Tiles will be automatically stored in TFRecord format and separated into training and validation steps if required (necessary when validation data is generated on per-tile basis; see :ref:`validation_planning`).
