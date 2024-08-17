.. _tutorial8:

Tutorial 8: Multiple-Instance Learning
======================================

In contrast with tutorials 1-4, which focused on training and evaluating traditional tile-based models, this tutorial provides an example of training a multiple-instance learning (MIL) model. MIL models are particularly useful for heterogeneous tumors, when only parts of a whole-slide image may carry a distinctive histological signature. In this tutorial, we'll train a MIL model to predict the ER status of breast cancer patients from whole slide images. Note: MIL models require PyTorch.

We'll start the same way as :ref:`tutorial1`, loading a project and preparing a dataset.

.. code-block:: python

    >>> import slideflow as sf
    >>> P = sf.load_project('/home/er_project')
    >>> dataset = P.dataset(
    ...   tile_px=256,
    ...   tile_um=128,
    ...   filters={
    ...     'er_status_by_ihc': ['Positive', 'Negative']
    ... })

If tiles have not yet been :ref:`extracted <filtering>` for this dataset, do that now.

.. code-block:: python

    >>> dataset.extract_tiles(qc='otsu')

Once a dataset has been prepared, the next step in training an MIL model is :ref:`converting images into features <mil>`. For this example, we'll use the pretrained `Virchow <https://huggingface.co/paige-ai/Virchow>`_ feature extractor, a vision transformer pretrained on 1.5M whole-slide images.  Virchow has an input size of 224x224, so our images will be resized to match.

.. code-block:: python

    >>> virchow = sf.build_feature_extractor('virchow', center_crop=True)
    >>> virchow.cite()
    @misc{vorontsov2024virchowmillionslidedigitalpathology,
        title={Virchow: A Million-Slide Digital Pathology Foundation Model},
        author={Eugene Vorontsov and Alican Bozkurt and Adam Casson and George Shaikovski and Michal Zelechowski and Siqi Liu and Kristen Severson and Eric Zimmermann and James Hall and Neil Tenenholtz and Nicolo Fusi and Philippe Mathieu and Alexander van Eck and Donghun Lee and Julian Viret and Eric Robert and Yi Kan Wang and Jeremy D. Kunz and Matthew C. H. Lee and Jan Bernhard and Ran A. Godrich and Gerard Oakley and Ewan Millar and Matthew Hanna and Juan Retamero and William A. Moye and Razik Yousfi and Christopher Kanan and David Klimstra and Brandon Rothrock and Thomas J. Fuchs},
        year={2024},
        eprint={2309.07778},
        archivePrefix={arXiv},
        primaryClass={eess.IV},
        url={https://arxiv.org/abs/2309.07778},
    }
    >>> virchow.num_features
    2560

The Virchow feature extractor produces a 2560-dimensional vector for each tile. We can generate and export :ref:`bags <bags>` of these features for all slides in our dataset using :func:`slideflow.Project.generate_feature_bags`.

.. code-block:: python

    >>> P.generate_feature_bags(
    ...     virchow,
    ...     dataset,
    ...     outdir='/bags/path'
    ... )

The output directory, ``/bags/path``, should look like:

.. code-block:: bash

    /bags/path
    ├── slide1.pt
    ├── slide1.indez.npz
    ├── slide2.pt
    ├── slide2.index.npz
    ├── ...
    └── bags_config.json

The ``*.pt`` files contain the feature vectors for tiles in each slide, and the ``*.index.npz`` files contain the corresponding X, Y coordinates for each tile.  The ``bags_config.json`` file contains the feature extractor configuration.

The next step is to create an MIL model configuration using :func:`slideflow.mil.mil_config`, specifying the architecture and relevant hyperparameters. For the architecture, we'll use :class:`slideflow.mil.models.Attention_MIL`. For the hyperparameters, we'll use a learning rate of 1e-4, a batch size of 32, 1cycle learning rate scheduling, and train for 10 epochs.

.. code-block:: python

    >>> from slideflow.mil import mil_config
    >>> config = mil_config(
    ...     model='attention_mil',
    ...     lr=1e-4,
    ...     batch_size=32,
    ...     epochs=10,
    ...     fit_one_cycle=True
    ... )

Finally, we can train the model using :func:`slideflow.mil.train_mil`. We'll split our dataset into 70% training and 30% validation, training to the outcome "er_status_by_ihc" and saving the model to ``/model/path``.

.. code-block:: python

    >>> from slideflow.mil import train_mil
    >>> train, val = dataset.split(labels='er_status_by_ihc', val_fraction=0.3)
    >>> train_mil(
    ...     config,
    ...     train_dataset=train,
    ...     val_dataset=val,
    ...     outcomes='er_status_by_ihc',
    ...     bags='/bags/path',
    ...     outdir='/model/path'
    ... )

During training, you'll see the training/validation loss and validation AUROC for each epoch. At the end of training, you'll see the validation metrics for each outcome.

.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    [18:51:01] INFO     Training FastAI MIL model with config:
               INFO     TrainerConfigFastAI(
                            aggregation_level='slide'
                            lr=0.0001
                            wd=1e-05
                            bag_size=512
                            fit_one_cycle=True
                            epochs=10
                            batch_size=32
                            model='attention_mil'
                            apply_softmax=True
                            model_kwargs=None
                            use_lens=True
                        )
    [18:51:02] INFO     Training dataset: 272 merged bags (from 272 possible slides)
               INFO     Validation dataset: 116 merged bags (from 116 possible slides)
    [18:51:04] INFO     Training model Attention_MIL (in=1024, out=2, loss=CrossEntropyLoss)
    epoch     train_loss  valid_loss  roc_auc_score  time
    0         0.328032    0.285096    0.580233       00:01
    Better model found at epoch 0 with valid_loss value: 0.2850962281227112.
    1         0.319219    0.266496    0.733721       00:01
    Better model found at epoch 1 with valid_loss value: 0.266496479511261.
    2         0.293969    0.230561    0.859690       00:01
    Better model found at epoch 2 with valid_loss value: 0.23056122660636902.
    3         0.266627    0.190546    0.927519       00:01
    Better model found at epoch 3 with valid_loss value: 0.1905461698770523.
    4         0.236985    0.165320    0.939147       00:01
    Better model found at epoch 4 with valid_loss value: 0.16532012820243835.
    5         0.215019    0.153572    0.946512       00:01
    Better model found at epoch 5 with valid_loss value: 0.153572216629982.
    6         0.199093    0.144464    0.948837       00:01
    Better model found at epoch 6 with valid_loss value: 0.1444639265537262.
    7         0.185597    0.141776    0.952326       00:01
    Better model found at epoch 7 with valid_loss value: 0.14177580177783966.
    8         0.173794    0.141409    0.951938       00:01
    Better model found at epoch 8 with valid_loss value: 0.14140936732292175.
    9         0.167547    0.140791    0.952713       00:01
    Better model found at epoch 9 with valid_loss value: 0.14079126715660095.
    [18:51:18] INFO     Predictions saved to {...}/predictions.parquet
               INFO     Validation metrics for outcome brs_class:
    [18:51:18] INFO     slide-level AUC (cat # 0): 0.953 AP: 0.984 (opt. threshold: 0.544)
               INFO     slide-level AUC (cat # 1): 0.953 AP: 0.874 (opt. threshold: 0.458)
               INFO     Category 0 acc: 88.4% (76/86)
               INFO     Category 1 acc: 83.3% (25/30)

After training has completed, the output directory, ``/model/path``, should look like:

.. code-block:: bash

    /model/path
    ├── attention
    │   ├── slide1_att.npz
    │   └── ...
    ├── models
    │   └── best_valid.pth
    ├── history.csv
    ├── mil_params.json
    ├── predictions.parquet
    └── slide_manifest.csv

The final model weights are saved in ``models/best_valid.pth``. Validation dataset predictions are saved in the "predictions.parquet" file. A manifest of training/validation data is saved in the "slide_manifest.csv" file, and training history is saved in the "history.csv" file. Attention values for all tiles in each slide are saved in the ``attention/`` directory.

The final saved model can be used for evaluation (:class:`slideflow.mil.eval_mil`) or inference (:class:`slideflow.mil.predict_slide` or :ref:`Slideflow Studio <studio_mil>`). The saved model path should be referenced by the parent directory (in this case, "/model/path") rather than the model file itself. For more information on MIL models, see :ref:`mil`.