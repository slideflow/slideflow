Evaluation
==========

Once you have finished your hyperparameter selection and would like to test your model on a saved external evaluation dataset, you can perform a model evaluation using the ``evaluate`` function. Specify the model you want to test with the ``model`` argument.

For example, to evaluate performance of model "HPSweep0" on slides labeled as "evaluation" in the "dataset" column of our annotations file, use the following:

.. code-block:: python

	SFP.evaluate(model="HPSweep0",
		  category_header="category",
		  filters={"dataset": ["evaluation"]})

Generate heatmaps
*****************

If you would like to generate a predictive heatmap for a set of slides, use the ``generate_heatmaps()`` function as below:

.. code-block:: python

	SFP.generate_heatmaps(model="HPSweep0",
		  filters={"dataset": ["evaluation"]})

Heatmaps will be saved in your project directory.

Generate mosaic maps
********************

You can also generate mosaic maps using similar syntax to the above. In addition to simply supplyling a model name, you can also provide a saved \*.h5 model directly:

.. code-block:: python

	SFP.generate_mosaic(model="/path/to/saved/model.h5",
		  filters={"dataset": ["evaluation"]})

As with heatmaps, generated mosaic maps will be saved to your project directory.