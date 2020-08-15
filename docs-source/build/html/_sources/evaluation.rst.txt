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

To generate a predictive heatmap for a set of slides, use the ``generate_heatmaps()`` function as below:

.. code-block:: python

	SFP.generate_heatmaps(model="/path/to/trained_model.h5",
		  		filters={"dataset": ["evaluation"]})

.. autofunction:: slideflow.SlideflowProject.generate_heatmaps
   :noindex:

Heatmaps will be saved in your project directory.