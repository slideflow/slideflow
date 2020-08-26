Evaluation
==========

Once you have finished your hyperparameter selection and would like to test your model on a saved external evaluation dataset, you can perform a model evaluation using the ``evaluate`` function. Specify the saved model .h5 file with the ``model`` argument. For example:

.. code-block:: python

	SFP.evaluate(model="/path/to/trained_model.h5",
			outcome_header="category",
			filters={"dataset": ["evaluation"]})

.. autofunction:: slideflow.SlideflowProject.evaluate
   :noindex:

Generate heatmaps
*****************

To generate a predictive heatmap for a set of slides, use the ``generate_heatmaps()`` function as below, which will automatically save heatmap images in your project directory:

.. code-block:: python

	SFP.generate_heatmaps(model="/path/to/trained_model.h5",
		  		filters={"dataset": ["evaluation"]})

.. autofunction:: slideflow.SlideflowProject.generate_heatmaps
   :noindex:

If you would like to directly interact with the calculated heatmap data, create a ``sf.activations.Heatmap`` object by providing a path to a slide, a path to a model, and tile size information:

.. code-block:: python

	heatmap = sf.activations.Heatmap('/path/to/slide.svs', '/path/to/model.h5', size_px=299, size_um=302)

The spatial map of logits, as calculated across the input slide, can be accessed through ``heatmap.logits``. The spatial map of post-convolution, penultimate activations can be accessed through ``heatmap.postconv``. The heatmap can be saved with ``heatmap.save('/path/')``.