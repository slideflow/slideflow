Installation
============

.. figure:: https://github.com/user-attachments/assets/53d5c1f8-8fbc-4e0f-bd62-db16797492b0

Slideflow is tested on **Linux-based systems** (Ubuntu, CentOS, Red Hat, and Raspberry Pi OS) and **macOS** (Intel and Apple). Windows support is experimental.

Requirements
************

- Python >= 3.7 (<3.10 if using `cuCIM <https://docs.rapids.ai/api/cucim/stable/>`_)
- `PyTorch <https://pytorch.org/>`_ (1.9+) *or* `Tensorflow <https://www.tensorflow.org/>`_ (2.5-2.11)
    - Core functionality, including tile extraction, data processing, and tile-based model training, is supported for both PyTorch and Tensorflow. Additional advanced tools, such as Multiple-Instance Learning (MIL), GANs, and pretrained foundation models, require PyTorch.

Optional
--------

- `Libvips >= 8.9 <https://libvips.github.io/libvips/>`_ (alternative slide reader, adds support for \*.scn, \*.mrxs, \*.ndpi, \*.vms, and \*.vmu files)
- Linear solver (for site-preserved cross-validation):

  - `CPLEX 20.1.0 <https://www.ibm.com/docs/en/icos/12.10.0?topic=v12100-installing-cplex-optimization-studio>`_ with `Python API <https://www.ibm.com/docs/en/icos/12.10.0?topic=cplex-setting-up-python-api>`_
  - *or* `Pyomo <http://www.pyomo.org/installation>`_ with `Bonmin <https://anaconda.org/conda-forge/coinbonmin>`_ solver


Download with pip
*****************

Slideflow can be installed either with PyPI or as a Docker container. To install via pip:

.. code-block:: bash

    # Update to latest pip
    pip install --upgrade pip wheel

    # Current stable release, Tensorflow backend
    pip install slideflow[tf] cucim cupy-cuda11x

    # Alternatively, install with PyTorch backend
    pip install slideflow[torch] cucim cupy-cuda11x

The ``cupy`` package name depends on the installed CUDA version; `see here <https://docs.cupy.dev/en/stable/install.html#installing-cupy>`_ for installation instructions. ``cucim`` and ``cupy`` are not required if using Libvips.


Run a Docker container
**********************

Alternatively, pre-configured `docker images <https://hub.docker.com/repository/docker/jamesdolezal/slideflow>`_ are available with cuCIM, Libvips, and either PyTorch 1.11 or Tensorflow 2.9 pre-installed. Using a preconfigured `Docker <https://docs.docker.com/install/>`_ container is the easiest way to get started with compatible dependencies and GPU support.

To run a Docker container with the Tensorflow backend:

.. code-block:: bash

    docker pull jamesdolezal/slideflow:latest-tf
    docker run -it --gpus all jamesdolezal/slideflow:latest-tf

To run a Docker container with the PyTorch backend:

.. code-block:: bash

    docker pull jamesdolezal/slideflow:latest-torch
    docker run -it --shm-size=2g --gpus all jamesdolezal/slideflow:latest-torch

Build from source
*****************

To build Slideflow from source, clone the repository from the project `Github page <https://github.com/slideflow/slideflow>`_:

.. code-block:: bash

    git clone https://github.com/slideflow/slideflow
    cd slideflow
    conda env create -f environment.yml
    conda activate slideflow
    python setup.py bdist_wheel
    pip install dist/slideflow* cupy-cuda11x


Extensions
**********

The core Slideflow package is licensed under the **Apache-2.0** license. Additional functionality, such as pretrained foundation models, are distributed in separate packages according to their licensing terms. Available extensions include:

- **Slideflow-GPL**: GPL-3.0 licensed extensions (`GitHub <https://github.com/slideflow/slideflow-gpl>`__)
    - Includes: `RetCCL <https://www.sciencedirect.com/science/article/abs/pii/S1361841522002730>`__, `CTransPath <https://www.sciencedirect.com/science/article/abs/pii/S1361841522002043>`__, and `CLAM <https://www.nature.com/articles/s41551-020-00682-w>`__.
- **Slideflow-NonCommercial**: CC BY-NC 4.0 licensed extensions for non-commercial use (`GitHub <https://github.com/slideflow/slideflow-noncommercial>`__)
    - Includes: `HistoSSL <https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v2.full.pdf>`__, `PLIP <https://www.nature.com/articles/s41591-023-02504-3>`__, `GigaPath <https://aka.ms/gigapath>`__, `UNI <https://www.nature.com/articles/s41591-024-02857-3>`__, `BISCUIT <https://www.nature.com/articles/s41467-022-34025-x>`__, and `StyleGAN3 <https://nvlabs-fi-cdn.nvidia.com/stylegan3/stylegan3-paper.pdf>`__.

These extensions can be installed via pip. The GigaPath feature extractor has additional, more restrictive dependencies that must be installed separately.

.. code-block:: bash

    # Install Slideflow-GPL and Slideflow-NonCommercial
    pip install slideflow-gpl slideflow-noncommercial

    # Install GigaPath dependencies, if desired
    pip install slideflow-noncommercial[gigapath] git+ssh://git@github.com/prov-gigapath/prov-gigapath


.. note::
    The Slideflow-GPL and Slideflow-NonCommercial extensions are not included in the default Slideflow package due to their licensing terms. Please review the licensing terms of each extension before use.


PyTorch vs. Tensorflow
**********************

Slideflow supports both PyTorch and Tensorflow, with cross-compatible TFRecord storage. Slideflow will default to using PyTorch if both are available, but the backend can be manually specified using the environmental variable ``SF_BACKEND``. For example:

.. code-block:: bash

    export SF_BACKEND=tensorflow

.. _slide_backend:

cuCIM vs. Libvips
*****************

By default, Slideflow reads whole-slide images using `cuCIM <https://docs.rapids.ai/api/cucim/stable/>`_. Although much faster than other openslide-based frameworks, it supports fewer slide scanner formats. Slideflow also includes a `Libvips <https://libvips.github.io/libvips/>`_ backend, which adds support for \*.scn, \*.mrxs, \*.ndpi, \*.vms, and \*.vmu files. You can set the active slide backend with the environmental variable ``SF_SLIDE_BACKEND``:

.. code-block:: bash

    export SF_SLIDE_BACKEND=libvips


.. warning::
    A bug in the pixman library (version=0.38) will corrupt downsampled slide images, resulting in large black boxes across the slide. We have provided a patch for version 0.38 that has been tested for Ubuntu, which is provided in the project `Github page <https://github.com/slideflow/slideflow>`_ (``pixman_repair.sh``), although it may not be suitable for all environments and we make no guarantees regarding its use. The `Slideflow docker images <https://hub.docker.com/repository/docker/slideflow/slideflow>`_ already have this applied. If you are installing from source, have pixman version 0.38, and are unable to apply this patch, the use of downsampled image layers must be disabled to avoid corruption (pass ``enable_downsample=False`` to tile extraction functions).
