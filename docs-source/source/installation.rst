Installation
============

Slideflow has been tested and is supported on the following systems:

- Ubuntu 18.04, 20.04, and 22.04
- Centos 7, 8, and 8 Stream

Software Requirements
*********************

- Python 3.7 - 3.10
- `OpenSlide <https://openslide.org/download/>`_
- `Libvips 8.9+ <https://libvips.github.io/libvips/>`_
- `Tensorflow 2.5-2.9 <https://www.tensorflow.org/install>`_ or `PyTorch 1.9-1.12 <https://pytorch.org/get-started/locally/>`_
- `QuPath <https://qupath.github.io>`_ [*optional*] - used for ROI annotations
- Linear solver for site-preserved cross-validation [*optional*]:

  - `CPLEX 20.1.0 <https://www.ibm.com/docs/en/icos/12.10.0?topic=v12100-installing-cplex-optimization-studio>`_ with `Python API <https://www.ibm.com/docs/en/icos/12.10.0?topic=cplex-setting-up-python-api>`_
  - *or* `Pyomo <http://www.pyomo.org/installation>`_ with `Bonmin <https://anaconda.org/conda-forge/coinbonmin>`_ solver

Download with pip
*****************

Slideflow can be installed either with PyPI or as a Docker container. To install via pip:

.. code-block:: bash

    # Update to latest pip
    $ pip install --upgrade pip

    # Current stable release
    $ pip install slideflow

Run a Docker container
**********************

Alternatively, pre-configured `docker images <https://hub.docker.com/repository/docker/jamesdolezal/slideflow>`_ are available with OpenSlide, Libvips, and either PyTorch 1.11 or Tensorflow 2.8 pre-installed. Using a preconfigured `Docker <https://docs.docker.com/install/>`_ container is the easiest way to get started with compatible dependencies and GPU support.

To run a Docker container with the Tensorflow 2.8 backend:

.. code-block:: bash

    $ docker pull jamesdolezal/slideflow:latest-tf
    $ docker run -it --gpus all jamesdolezal/slideflow:latest-tf

To run a Docker container with the PyTorch 1.11 backend:

.. code-block:: bash

    $ docker pull jamesdolezal/slideflow:latest-torch
    $ docker run -it --shm-size=2g --gpus all jamesdolezal/slideflow:latest-torch

Build from source
*****************

To build Slideflow from source, clone the repository from the project `Github page <https://github.com/jamesdolezal/slideflow>`_:

.. code-block:: bash

    $ git clone https://github.com/jamesdolezal/slideflow
    $ cd slideflow
    $ pip install -r requirements.txt
    $ python setup.py bdist_wheel
    $ pip install dist/slideflow-1.X.X-py3-any.whl

.. warning::
    A bug in the pixman library (version=0.38) will corrupt downsampled slide images, resulting in large black boxes across the slide. We have provided a patch for version 0.38 that has been tested for Ubuntu, which is provided in the project `Github page <https://github.com/jamesdolezal/slideflow>`_ (``pixman_repair.sh``), although it may not be suitable for all environments and we make no guarantees regarding its use. The `Slideflow docker images <https://hub.docker.com/repository/docker/jamesdolezal/slideflow>`_ already have this applied. If you are installing from source, have pixman version 0.38, and are unable to apply this patch, the use of downsampled image layers must be disabled to avoid corruption (pass ``enable_downsample=False`` to tile extraction functions).

Changing backends
*****************

The default backend for this package is Tensorflow/Keras, but a full PyTorch backend is also included, with a dedicated TFRecord reader/writer that ensures saved image tiles can be served to both Tensorflow and PyTorch models in cross-compatible fashion.

If using the Tensorflow backend, PyTorch does not need to be installed; the reverse is true as well.

To switch backends, simply set the environmental variable ``SF_BACKEND`` equal to either ``torch`` or ``tensorflow``:

.. code-block:: console

    export SF_BACKEND=torch