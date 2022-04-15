Evaluation
==========

In addition to examining cross-validation training performance, model performance can be assessed with external dataset evaluation, and visualization of predictions across evaluation slides in the form of a heatmap.

Model evaluation
****************

Once training and hyperparameter tuning is complete, you can test model performance on your held-out evaluation set using the ``evaluate`` function. Specify the path to the saved with the ``model`` argument. For example:

.. code-block:: python

    P.evaluate(
      model="/path/to/trained_model_epoch1",
      outcomes="category",
      filters={"dataset": ["eval"]}
    )

.. autofunction:: slideflow.Project.evaluate
   :noindex:

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

If you would like to directly interact with the calculated heatmap data, create a :class:`slideflow.Heatmap` object by providing a path to a slide, a path to a model, and tile size information:

.. code-block:: python

    from slideflow import Heatmap

    heatmap = Heatmap(
      slide='/path/to/slide.svs',
      model='/path/to/model'
    )

The spatial map of logits, as calculated across the input slide, can be accessed through ``heatmap.logits``. The spatial map of post-convolution, penultimate activations can be accessed through ``heatmap.postconv``. The heatmap can be saved with ``heatmap.save('/path/')``.