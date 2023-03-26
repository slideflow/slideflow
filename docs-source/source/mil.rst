.. _clam_mil:

Multiple-Instance Learning (MIL)
================================

In addition to standard tile-based neural networks, slideflow supports training multiple-instance learning (MIL) models. Several architectures are available, including `attention-based MIL <https://github.com/AMLab-Amsterdam/AttentionDeepMIL>`_ (``"Attention_MIL"``), `CLAM <https://github.com/mahmoodlab/CLAM>`_ (``"CLAM_SB",`` ``"CLAM_MB"``, ``"MIL_fc"``, ``"MIL_fc_mc"``), and `transformer MIL <https://github.com/szc19990412/TransMIL>`_ (``"TransMIL"``). Custom architectures can also be trained.

Generating features
*******************

The first step in MIL training is generating features from image tiles extracted from whole-slide images. First, determine the feature extractor that will be used. This can be an imagenet-pretrained model, a model finetuned in Slideflow, a pretrained feature extractor (such as CTransPath or RetCCL), or a fine-tuned SSL model.  In all cases, the :ref:`slideflow.DatasetFeatures <activations>` interface is used to generate and export features.

ImageNet Features
-----------------

To calculate features from an ImageNet-pretrained network, first build an imagenet feature extractor with :func:`slideflow.model.build_feature_extractor`. The first argument should be the name of an architecture followed by ``_imagnet``, and the expected tile size should be passed to the keyword argument ``tile_px``. You can optionally specify the layer from which to generate features with the ``layers`` argument; if not provided, it will default to calculating features from post-convolutional layer activations. For example, to build a ResNet50 feature extractor for images at 299 x 299 pixels:

.. code-block:: python

    from slideflow.model import build_feature_extractor

    resnet50 = build_feature_extractor(
        'resnet50_imagenet',
        tile_px=299
    )

Next, pass this feature extractor to the first argument of :class:`slideflow.DatasetFeatures`.  The second argument should be a :class:`slideflow.Dataset`.

.. code-block:: python

    ...

    # Load a project and dataset.
    P = sf.Project(...)
    dataset = P.dataset(tile_px=299, tile_um=302)

    # Calculate features for this dataset.
    features = sf.DatasetFeatures(resnet50, dataset)

Features from Finetuned Model
-----------------------------

You can also calculate features from any model trained in Slideflow. Set the first argument of :class:`slideflow.DatasetFeatures` equal to the path of the trained model, optionally specifying the layer name with ``layers`` to choose the layer at which features will be calculated (defaults to post-convolutional layer).

.. code-block:: python

    import slideflow as sf

    # Load a project and dataset.
    P = sf.Project(...)
    dataset = P.dataset(tile_px=299, tile_um=302)

    # Calculate features from trained model.
    features = sf.DatasetFeatures(
        '/path/to/model',
        dataset=dataset
        layers='sepconv3_bn'
    )

Pretrained Feature Extractor
----------------------------

Slideflow includes several pathology-specific pretrained feature extractors. Use :func:`slideflow.model.build_feature_extractor` to build a feature extractor by name, and then pass this extractor to the first argument of :class:`slideflow.DatasetFeatures`:

.. code-block:: python

    from slideflow.model import build_feature_extractor

    ctranspath = build_feature_extractor('ctranspath', tile_px=299)
    features = sf.DatasetFeatures(ctranspath, ...)

Self-Supervised Learning
------------------------

Finally, you can also generate features from a :ref:`self-supervised learning <simclr_ssl>` model. Simply pass the path to the saved model:

.. code-block:: python

    features = sf.DatasetFeatures('path/to/saved_model', ...)

Exporting Features
------------------

Once you have generated features for a dataset, export the feature "bags" to disk using :meth:`slideflow.DatasetFeatures.to_torch`:

.. code-block:: python

    features = sf.DatasetFeatures(...)
    features.to_torch('/path/to/bag_directory/')

This bag directory will then be used to train the MIL models.

Training
********


Model Configuration
-------------------

To train an MIL model on exported features, first prepare the MIL configuration using :func:`slideflow.mil.mil_config`.

The first argument to this function is the model architecture (which can be a name or a custom ``torch.nn.Module`` model), and the remaining arguments are used to configure the training process (including learning rate and epochs).

By default, training is executed using `FastAI <https://docs.fast.ai/>`_ with `1cycle learning rate scheduling <https://arxiv.org/pdf/1803.09820.pdf%E5%92%8CSylvain>`_.

.. code-block:: python

    import slideflow as sf
    from slideflow.mil import mil_config

    config = mil_config('attention_mil', lr_max=1e-3)


Legacy Trainer (CLAM)
---------------------

In addition to the FastAI trainer, CLAM models can be trained using the `original <https://github.com/mahmoodlab/CLAM>`_ CLAM training loop. This trainer has been modified, cleaned, and included as a submodule in Slideflow. This legacy trainer can be used for CLAM models by setting ``trainer='clam'`` for an MIL configuration:

.. code-block:: python

    config = mil_config(..., trainer='clam')


Training an MIL Model
---------------------

Next, prepare a :ref:`training and validation dataset <datasets_and_validation>` and use :func:`slideflow.Project.train_mil` to start training. For example, to train a model using three-fold cross-validation to the outcome "HPV_status":

.. code-block:: python

    ...

    # Prepare a project and dataset
    P = sf.Project(...)
    full_dataset = dataset = P.dataset(tile_px=299, tile_um=302)

    # Split the dataset using three-fold, site-preserved cross-validation
    splits = full_dataset.kfold_split(
        k=3,
        labels='HPV_status',
        preserved_site=True
    )

    # Train on each cross-fold
    with train, val in splits:
        P.train_mil(
            config=config,
            outcomes='HPV_status',
            train_dataset=train,
            val_dataset=val,
            bags='/path/to/bag_directory'
        )

Model training statistics, including validation performance (AUROC, AP) and predictions on the validation dataset, will be saved in an ``mil`` subfolder within the main project directory. For attention-based MIL models, heatmaps of the attention layers can be saved for each of the validation slides using the argument ``attention_heatmaps=True``.


Evaluation
**********

To evaluate a saved MIL model on an external dataset, first extract features from a dataset, then use :func:`slideflow.Project.evaluate_mil`:

.. code-block:: python

    import slideflow as sf
    from slideflow.model import build_feature_extractor

    # Prepare a project and dataset
    P = sf.Project(...)
    dataset = P.dataset(tile_px=299, tile_um=302)

    # Generate features using CTransPath
    ctranspath = build_feature_extractor('ctranspath', tile_px=299)
    features = sf.DatasetFeatures(ctranspath, dataset=dataset)
    features.to_torch('/path/to/bag_directory')

    # Evaluate a saved MIL model
    P.evaluate_mil(
        '/path/to/saved_model'
        outcomes='HPV_status',
        dataset=dataset,
        bags='/path/to/bag_directory',
    )

As with training, attention heatmaps can be generated for attention-based MIL models with the argument ``attention_heatmaps=True``.

.. image:: att_heatmap.jpg