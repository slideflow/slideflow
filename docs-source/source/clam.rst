CLAM
====

In addition to standard Tensorflow/Keras model applications, slideflow supports training models with `CLAM <https://github.com/mahmoodlab/CLAM>`_. A slightly modified version of CLAM which supports slideflow dataset and input pipelines is included in ``slideflow.clam``.

Creating slide activations
**************************

The first step in the CLAM pipeline is generating tile-level activations across whole-slide images. While the original `CLAM paper <https://arxiv.org/abs/2004.09666>`_ used features generated from an imagenet-trained model, we have found it useful to generate feature activations from models pretrained with histology images. To this end, the project function :func:`slideflow.project.Project.generate_features_for_clam` accepts any model as input and will generate feature vectors from the specified intermediate layers. For example:

.. code-block:: python

    SFP.generate_features_for_clam(
        model='/path/to/saved/model',
        outdir='/clam/path',
        layers=['postconv']
    )

Training
********

To train a CLAM model, use the project function :func:`slideflow.project.Project.train_clam`. Clam arguments are configured with :func:`slideflow.clam.get_args`:

.. code-block:: python

    dataset = SFP.dataset(tile_px=299, tile_um=302)
    SFP.generate_features_for_clam(..., outdir='/clam/path')

    clam_args = sf.clam.get_args(k=3, bag_loss='svm', ...)

    SFP.train_clam(
        exp_name='test_experiment',
        pt_files='/clam/path',
        outcome_label_headers='category1',
        dataset=dataset,
        clam_args=clam_args
    )

The training function will, by default, save heatmaps of the attention layers for each of the validation slides. This behavior can be disabled by passing ``attention_heatmaps=False``.

Evaluation
**********

To evaluate a saved CLAM model on an external dataset, first extract features from this dataset, then use the project function :func:`slideflow.project.Project.evaluate_clam`:

.. code-block:: python

    SFP.generate_features_for_clam(..., outdir='/eval/clam/path')

    SFP.evaluate_clam(
        exp_name='evaluation',
        pt_files='/eval/clam/path',
        outcome_label_headers='category1',
        tile_px=299,
        tile_um=302
    )