.. _tile_labels:

Strong Supervision with Tile Labels
====================================

Pathology deep learning models are commonly trained with weak supervision, where the labels for individual image tiles are inherited from the parent slide. The end goal for such models is to predict the label for the entire slide, rather than individual tiles.

However, it is also possible to train models with strong supervision, where the labels for individual
image tiles are determined through :ref:`Region of Interest (ROI) <roi_labels>` labels. This note describes the process by which such labels are generated, and how they can be used to train a model. Training models with strong supervision requires PyTorch and is not supported in TensorFlow.

Labeling ROIs
*************

The first step is to create regions of interest (ROIs). The fastest way to create labeled ROIs is with :ref:`Slideflow Studio <studio_roi>`, which includes integrated tools for quickly assigning labels to both new and existing ROIs. However, it is also possible to create ROIs with other tools, such as QuPath or ImageScope (as described :ref:`here <roi_labels>`), and modify the generated ROI CSV file to add labels.

ROI CSV files are formatted with three required columns: "roi_name", "x_base", and "y_base". Each row is a single point in an ROI, with the "x_base" and "y_base" columns specifying the X/Y coordinates in the slide's lowest (base) dimension. Individual ROIs are grouped by the "roi_name" column, with each ROI having a unique name. An optional fourth column, "label", can be used to assign a label to each ROI. For example:

.. code-block:: csv

    roi_name,x_base,y_base,label
    1,100,100,tumor
    1,104,165,tumor
    1,532,133,tumor
    1,101,101,tumor
    2,200,200,stroma
    2,200,235,stroma
    2,222,267,stroma
    2,202,201,stroma

When ROIs are saved in Slideflow Studio, they are exported in this file format and saved in either the current working directory or, if a project is loaded, in the configured project directory .

Building tile labels
********************

Once ROIs have been generated, labeled, and saved in CSV format, the next step is to build a dataframe of tile labels. If not already done, start by :ref:`configuring a project <project_setup>` and ensuring that ROIs are in the correct directory. You can verify that the ROIs are in the right place by confirming that :meth:`slideflow.Dataset.rois` returns the number of slides with ROIs:

.. code-block:: python

    >>> import slideflow as sf
    >>> P = sf.load_project('/path/to/project')
    >>> dataset = P.dataset(tile_px=256, tile_um=256)
    >>> len(dataset.rois())
    941

Next, build a dataframe of tile labels with :meth:`slideflow.Dataset.get_tile_dataframe`. This will return a dataframe with tile coordinates (X/Y of tile center, in base dimension), slide grid index, and associated ROI name/label if the tile is in an ROI. For example:

.. code-block:: python

    >>> df = dataset.get_tile_dataframe()
    >>> df.head()
                    loc_x  loc_y  grid_x  grid_y roi_name roi_desc label    slide
    slide1-608-608  608    608    0       0      ROI_0    None     tumor    slide1
    slide1-608-864  608    864    0       1      ROI_0    None     tumor    slide1
    slide1-608-1120 608    1120   0       2      ROI_0    None     tumor    slide1
    ...

The index for this dataframe is the tile ID, a unique identifier built from a combination of the slide name and tile coordinates.

When training with supervised labels, we'll want to exclude tiles that are either not in an ROI or are in an unlabeled ROI. This can be done by filtering the dataframe to only include rows where the "label" column is not None:

.. code-block:: python

    >>> df = df.loc[df.label.notnull()]

Finally, we'll only need the "label" column and tile ID for training, so all other columns can be dropped. This step is optional but may reduce memory usage.

.. code-block:: python

    >>> df = df[['label']]
    >>> df.head()
                    label
    slide1-608-608  tumor
    slide1-608-864  tumor
    slide1-608-1120 tumor
    ...

This dataframe can now be used to train a model with strong supervision.

Training a model
****************

Training a model with strong supervision requires using a :class:`slideflow.model.Trainer`, as described in :ref:`tutorial2`. The only difference when training with strong supervision is that the trainer should be initialized with the tile dataframe for the labels:

.. code-block:: python

    >>> trainer = sf.model.build_trainer(..., labels=df)
    >>> trainer.train(...)

Once training has finished, the saved model can be used interchangeably with models trained with weak supervision for evaluation, inference, feature generation, etc.

Complete example
****************

Below is a complete example of training a model with strong supervision. This example assumes that a project has already been configured, tiles have been extracted, and ROIs have been generated and labeled.

.. code-block:: python

    import slideflow as sf

    # Load project and dataset
    P = sf.load_project('/path/to/project')
    dataset = P.dataset(tile_px=256, tile_um=256)

    # Build tile label dataframe, and filter
    # to only include tiles in an ROI.
    df = dataset.get_tile_dataframe()
    df = df.loc[df.label.notnull()]

    # Subsample our dataset to only include slides with ROI labels.
    dataset = dataset.filter({'slide': list(df.slide.unique())})

    # Split the dataset into training and validation.
    train, val = dataset.split(val_fraction=0.3)

    # Build model hyperparameters
    hp = sf.ModelParams(
        tile_px=256,
        tile_um=256,
        model='xception',
        batch_size=32
    )

    # Train model
    trainer = sf.model.build_trainer(
        hp=hp,
        outdir='/path/to/outdir',
        labels=df
    )
    trainer.train(train, val)
