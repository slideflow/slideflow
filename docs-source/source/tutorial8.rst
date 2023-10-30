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

Once a dataset has been prepared, the next step in training an MIL model is :ref:`converting images into features <mil>`. For this example, we'll use the pretrained `HistoSSL <https://github.com/owkin/HistoSSLscaling>`_ feature extractor, a vision transformer pretrained on 40 million histology images. HistoSSL was trained on tiles of size 224x224, so our images will be center-cropped to match.

.. code-block:: python

    >>> from slideflow.model import build_feature_extractor
    >>> histossl = build_feature_extractor('histossl', tile_px=256)

    This model is developed and licensed by Owkin, Inc. The license for use is
    provided in the LICENSE file in the same directory as this source file
    (slideflow/model/extractors/histossl/LICENSE), and is also available
    at https://github.com/owkin/HistoSSLscaling. By using this feature extractor,
    you agree to the terms of the license.

    >>> histossl.cite()

    @article{Filiot2023ScalingSSLforHistoWithMIM,
        author       = {Alexandre Filiot and Ridouane Ghermi and Antoine Olivier and Paul Jacob and Lucas Fidon and Alice Mac Kain and Charlie Saillard and Jean-Baptiste Schiratti},
        title        = {Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling},
        elocation-id = {2023.07.21.23292757},
        year         = {2023},
        doi          = {10.1101/2023.07.21.23292757},
        publisher    = {Cold Spring Harbor Laboratory Press},
        url          = {https://www.medrxiv.org/content/early/2023/07/26/2023.07.21.23292757},
        eprint       = {https://www.medrxiv.org/content/early/2023/07/26/2023.07.21.23292757.full.pdf},
        journal      = {medRxiv}
    }
    >>> histossl.num_features
    768

The HistoSSL feature extractor produces a 768-dimensional vector for each tile. We can generate and export :ref:`bags <bags>` of these features for all slides in our dataset using :func:`slideflow.Project.generate_feature_bags`.

.. code-block:: python

    >>> P.generate_feature_bags(
    ...     histossl,
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

The next step is to create an MIL model configuration using :func:`slideflow.mil.mil_config`, specifying the architecture and relevant hyperparameters. For the architecture, we'll use an :class:`slideflow.mil.models.Attention_MIL` model with a latent dimension size of 256. For the hyperparameters, we'll use a learning rate of 1e-4, a batch size of 32, 1cycle learning rate scheduling, and train for 10 epochs.

.. code-block:: python

    >>> from slideflow.mil import mil_config
    >>> config = mil_config(
    ...     model='Attention_MIL',
    ...     z_dim=256,
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