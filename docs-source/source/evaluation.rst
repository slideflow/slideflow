.. _evaluation:

Evaluation
==========

In addition to cross-validation performance, model performance can also be assessed via evaluation on external datasets. Whole-slide image predictions can also be visualized in the form of a heatmap.

Model evaluation
****************

Once training and hyperparameter tuning is complete, you can test model performance on your held-out evaluation set using the ``evaluate`` function. Specify the path to the saved with the ``model`` argument.

.. code-block:: python

    P.evaluate(
      model="/path/to/trained_model_epoch1",
      outcomes="category",
      filters={"dataset": ["eval"]}
    )

.. autofunction:: slideflow.Project.evaluate
   :noindex:

Results are returned from the ``Project.evaluate()`` function as a dictionary and saved in the project evaluation directory. Tile-, slide-, and patient- level predictions are also saved in the corresponding evaluation folder.

Heatmaps
********

To generate a predictive heatmap for a set of slides, use the ``generate_heatmaps()`` function as below, which will automatically save heatmap images in your project directory:

.. code-block:: python

    P.generate_heatmaps(
      model="/path/to/trained_model_epoch1",
      filters={"dataset": ["eval"]}
    )

.. autofunction:: slideflow.Project.generate_heatmaps
   :noindex:

For more granular control, create a :class:`slideflow.Heatmap` object by providing paths to a slide and model:

.. code-block:: python

    heatmap = sf.Heatmap(
      slide='/path/to/slide.svs',
      model='/path/to/model'
    )

Regions of Interest (ROI) can be provided either through the ``roi_dir`` or ``rois`` method. The easiest way to use ROIs is through :class:`slideflow.Dataset`:

.. code-block:: python

    heatmap = sf.Heatmap(
        ...,
        rois=P.dataset().rois()
    )

Save heatmaps for each outcome category with :meth:`slideflow.Heatmap.save`. A heatmap displaying predictions from a single outcome class can be interactively displayed (ie. in a Jupyter notebook) with :meth:`slideflow.Heatmap.plot`:

.. code-block:: python

    heatmap.plot(class_idx=0, cmap='inferno')

Insets showing zoomed-in portions of the heatmap can be added with :meth:`slideflow.Heatmap.add_inset`:

.. code-block:: python

    heatmap.add_inset(zoom=20, x=(10000, 10500), y=(2500, 3000), loc=1, axes=False)
    heatmap.add_inset(zoom=20, x=(12000, 12500), y=(7500, 8000), loc=3, axes=False)
    heatmap.plot(class_idx=0, mpp=1)

.. image:: heatmap_inset.jpg

|

The spatial map of logits, as calculated across the input slide, can be accessed through ``heatmap.logits``. The spatial map of post-convolution, penultimate activations can be accessed through ``heatmap.postconv``.