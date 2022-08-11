![slideflow logo](https://github.com/jamesdolezal/slideflow/raw/master/docs-source/pytorch_sphinx_theme/images/slideflow-banner.png)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5703792.svg)](https://doi.org/10.5281/zenodo.5703792)
[![Python application](https://github.com/jamesdolezal/slideflow/actions/workflows/python-app.yml/badge.svg?branch=master)](https://github.com/jamesdolezal/slideflow/actions/workflows/python-app.yml)
[![PyPI version](https://badge.fury.io/py/slideflow.svg)](https://badge.fury.io/py/slideflow)

Slideflow provides a unified API for building and testing deep learning models for digital pathology, supporting both Tensorflow and PyTorch.

Slideflow includes tools for **whole-slide image processing** and tile extraction, **customizable deep learning model training** with dozens of supported architectures, **explainability tools** including heatmaps, mosaic maps, GANs, and saliency maps, **analysis of activations** from model layers, **uncertainty quantification**, and more. A variety of fast, optimized whole-slide image processing tools are included, including background filtering, blur/artifact detection, [stain normalization](https://slideflow.dev/norm.html), and efficient storage in `*.tfrecords` format. Model training is easy and highly configurable, with an easy drop-in API for training custom architectures. For external training loops, Slideflow can be used as an image processing backend, serving an optimized `tf.data.Dataset` or `torch.utils.data.DataLoader` to read and process slide images and perform real-time stain normalization.

Slideflow has been used by:

- [Dolezal et al](https://www.nature.com/articles/s41379-020-00724-3), _Modern Pathology_, 2020
- [Rosenberg et al](https://ascopubs.org/doi/10.1200/JCO.2020.38.15_suppl.e23529), _Journal of Clinical Oncology_ [abstract], 2020
- [Howard et al](https://www.nature.com/articles/s41467-021-24698-1), _Nature Communications_, 2021
- [Dolezal et al](https://arxiv.org/abs/2204.04516) [arXiv], 2022
- [Storozuk et al](https://www.nature.com/articles/s41379-022-01039-1.pdf), _Modern Pathology_ [abstract], 2022
- [Partin et al](https://arxiv.org/abs/2204.11678) [arXiv], 2022
- [Dolezal et al](https://ascopubs.org/doi/abs/10.1200/JCO.2022.40.16_suppl.8549) [abstract], 2022
- [Howard et al](https://www.biorxiv.org/content/10.1101/2022.07.07.499039v1) [bioRxiv], 2022

Full documentation with example tutorials can be found at [slideflow.dev](https://www.slideflow.dev/).

## Requirements
- Python >= 3.7
- [Libvips](https://libvips.github.io/libvips/) >= 8.9.
- [OpenSlide](https://openslide.org/download/)
- [Tensorflow](https://www.tensorflow.org/) 2.5-2.9 _or_ [PyTorch](https://pytorch.org/) 1.9-1.12
- [QuPath](https://qupath.github.io/) [_optional_] - Used for pathologist ROIs
- Linear solver [_optional_] - Used for preserved-site cross-validation
  - [CPLEX](https://www.ibm.com/docs/en/icos/12.10.0?topic=v12100-installing-cplex-optimization-studio) 20.1.0 with [Python API](https://www.ibm.com/docs/en/icos/12.10.0?topic=cplex-setting-up-python-api)
  - _or_ [Pyomo](http://www.pyomo.org/installation) with [Bonmin](https://anaconda.org/conda-forge/coinbonmin) solver

## Updates (1.2.4)
Please see the [Version 1.2.4 Release Notes](https://github.com/jamesdolezal/slideflow/releases/tag/1.2.4) for a summary of the latest updates and fixes.

## Installation
Slideflow can be installed either with PyPI or as a Docker container. To install via pip:

```
pip3 install --upgrade setuptools pip wheel
pip3 install slideflow
```

Alternatively, pre-configured [docker images](https://hub.docker.com/repository/docker/jamesdolezal/slideflow) are available with OpenSlide/Libvips and the latest version of either Tensorflow and PyTorch. To install with the Tensorflow backend:

```
docker pull jamesdolezal/slideflow:latest-tf
docker run -it --gpus all jamesdolezal/slideflow:latest-tf
```

To install with the PyTorch backend:

```
docker pull jamesdolezal/slideflow:latest-torch
docker run -it --shm-size=2g --gpus all jamesdolezal/slideflow:latest-torch
```

## Getting started
Slideflow experiments are organized into [Projects](https://slideflow.dev/project_setup.html), which supervise storage of whole-slide images, extracted tiles, and patient-level annotations. To create a new project, create an instance of the `slideflow.Project` class, supplying a pre-configured set of patient-level annotations in CSV format:

```python
import slideflow as sf
P = sf.Project(
  '/project/path',
  annotations="/patient/annotations.csv"
)
```

Once the project is created, add a new dataset source with paths to whole-slide images, tumor Region of Interest (ROI) files [if applicable], and paths to where extracted tiles/tfrecords should be stored. This will only need to be done once.

```python
P.add_source(
  name="TCGA",
  slides="/slides/directory",
  roi="/roi/directory",
  tiles="/tiles/directory",
  tfrecords="/tfrecords/directory"
)
```

This step should attempt to automatically associate slide names with the patient identifiers in your annotations file. After this step has completed, double check that the annotations file has a `slide` column for each annotation entry with the filename (without extension) of the corresponding slide.

## Extract tiles from slides

Next, whole-slide images are segmented into smaller image tiles and saved in `*.tfrecords` format. [Extract tiles](https://slideflow.dev/extract_tiles.html) from slides at a given magnification (width in microns size) and resolution (width in pixels) using `sf.Project.extract_tiles()`:

```python
P.extract_tiles(
  tile_px=299,  # Tile size, in pixels
  tile_um=302   # Tile size, in microns
)
```

If slides are on a network drive or a spinning HDD, tile extraction can be accelerated by buffering slides to a SSD or ramdisk:

```python
P.extract_tiles(
  ...,
  buffer="/mnt/ramdisk"
)
```

## Training models

Once tiles are extracted, models can be [trained](https://slideflow.dev/training.html). Start by configuring a set of [hyperparameters](https://slideflow.dev/model.html#modelparams):

```python
params = sf.ModelParams(
  tile_px=299,
  tile_um=302,
  batch_size=32,
  model='xception',
  learning_rate=0.0001,
  ...
)
```

Models can then be trained using these parameters. Models can be trained to categorical, multi-categorical, continuous, or time-series outcomes, and the training process is [highly configurable](https://slideflow.dev/training.html). For example, to train models in cross-validation to predict the outcome `'category1'` as stored in the project annotations file:

```python
P.train(
  'category1',
  params=params,
  save_predictions=True,
  multi_gpu=True
)
```

## Evaluation, heatmaps, mosaic maps, and more

Slideflow includes a host of additional tools, including model [evaluation](https://slideflow.dev/evaluation.html) and [prediction](https://slideflow.dev/project.html#slideflow.Project.predict), [heatmaps](https://slideflow.dev/project.html#slideflow.Project.generate_heatmaps), [mosaic maps](https://slideflow.dev/project.html#slideflow.Project.generate_mosaic), analysis of [layer activations](https://slideflow.dev/layer_activations.html), and more. See our [full documentation](https://slideflow.dev) for more details and tutorials.

## License
This code is made available under the GPLv3 License and is available for non-commercial academic purposes.

## Reference
The manuscript describing this protocol is in press. In the meantime, if you find our work useful for your research, or if you use parts of this code, please consider citing as follows:

James Dolezal, Sara Kochanny, & Frederick Howard. (2022). Slideflow: A Unified Deep Learning Pipeline for Digital Histology (1.1.0). Zenodo. https://doi.org/10.5281/zenodo.5703792

```
@software{james_dolezal_2022_5703792,
  author       = {James Dolezal and
                  Sara Kochanny and
                  Frederick Howard},
  title        = {{Slideflow: A Unified Deep Learning Pipeline for
                   Digital Histology}},
  month        = apr,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {1.1.0},
  doi          = {10.5281/zenodo.5703792},
  url          = {https://doi.org/10.5281/zenodo.5703792}
}
```
