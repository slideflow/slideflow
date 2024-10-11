<div align="center">
  <img src="https://github.com/user-attachments/assets/53d5c1f8-8fbc-4e0f-bd62-db16797492b0" alt="slideflow logo">

  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5703792.svg)](https://doi.org/10.5281/zenodo.5703792)
  [![Python application](https://github.com/slideflow/slideflow/actions/workflows/python-app.yml/badge.svg?branch=master)](https://github.com/slideflow/slideflow/actions/workflows/python-app.yml)
  [![PyPI version](https://badge.fury.io/py/slideflow.svg)](https://badge.fury.io/py/slideflow)

  [ArXiv](https://arxiv.org/abs/2304.04142) | [Docs](https://slideflow.dev) | [Slideflow Studio](https://slideflow.dev/studio/) | [Cite](#reference) | [✨ What's New in 3.0 ✨](https://github.com/slideflow/slideflow/releases/tag/3.0.0)

  ______________________________________________________________________

  ![Slideflow Studio: a visualization tool for interacting with models and whole-slide images.](https://github.com/slideflow/slideflow/assets/48372806/7f43d8cb-dc80-427d-84c4-3e5a35fa1472)

</div>

**Slideflow is a deep learning library for digital pathology, offering a user-friendly interface for model development.**
  
Designed for both medical researchers and AI enthusiasts, the goal of Slideflow is to provide an accessible, easy-to-use interface for developing state-of-the-art pathology models. Slideflow has been built with the future in mind, offering a scalable platform for digital biomarker development that bridges the gap between ever-evolving, sophisticated methods and the needs of a clinical researcher. For developers, Slideflow provides multiple endpoints for integration with other packages and external training paradigms, allowing you to leverage highly optimized, pathology-specific processes with the latest ML methodologies.



## 🚀 Features
- Easy-to-use, highly customizable training pipelines
- Robust **[slide processing](https://slideflow.dev/slide_processing) and [stain normalization](https://slideflow.dev/norm)** toolkit
- Support for training with **[weakly-supervised](https://slideflow.dev/training) or [strongly-supervised](https://slideflow.dev/tile_labels)** labels
- Built-in, state-of-the-art **[foundation models](https://slideflow.dev/features)**
- **[Multiple-instance learning (MIL)](https://slideflow.dev/mil)**
- **[Self-supervised learning (SSL)](https://slideflow.dev/ssl)**
- **[Generative adversarial networks (GANs)](https://slideflow.dev/training)**
- **Explainability tools**: [Heatmaps](https://slideflow.dev/evaluation/#heatmaps), [mosaic maps](https://slideflow.dev/posthoc/#mosaic-maps), [saliency maps](https://slideflow.dev/saliency/), [synthetic histology](https://slideflow.dev/stylegan)
- Robust **[layer activation analysis](https://slideflow.dev/posthoc)** tools
- **[Uncertainty quantification](https://slideflow.dev/uq)**
- **[Interactive user interface](https://slideflow.dev/studio)** for model deployment
- ... and more!

Full documentation with example tutorials can be found at [slideflow.dev](https://www.slideflow.dev/).

## Requirements
- Python >= 3.7 (<3.10 if using [cuCIM](https://docs.rapids.ai/api/cucim/stable/))
- [PyTorch](https://pytorch.org/) >= 1.9 _or_ [Tensorflow](https://www.tensorflow.org/) 2.5-2.11

### Optional
- [Libvips](https://libvips.github.io/libvips/) >= 8.9 (alternative slide reader, adds support for *.scn, *.mrxs, *.ndpi, *.vms, and *.vmu files).
- Linear solver (for preserved-site cross-validation)
  - [CPLEX](https://www.ibm.com/docs/en/icos/12.10.0?topic=v12100-installing-cplex-optimization-studio) 20.1.0 with [Python API](https://www.ibm.com/docs/en/icos/12.10.0?topic=cplex-setting-up-python-api)
  - _or_ [Pyomo](http://www.pyomo.org/installation) with [Bonmin](https://anaconda.org/conda-forge/coinbonmin) solver


## 📥 Installation
Slideflow can be installed with PyPI, as a Docker container, or run from source.

### Method 1: Install via pip

```
pip3 install --upgrade setuptools pip wheel
pip3 install slideflow[cucim] cupy-cuda11x
```

The `cupy` package name depends on the installed CUDA version; [see here](https://docs.cupy.dev/en/stable/install.html#installing-cupy) for installation instructions. `cupy` is not required if using Libvips.

### Method 2: Docker image

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

### Method 3: From source

To run from source, clone this repository, install the conda development environment, and build a wheel:

```
git clone https://github.com/slideflow/slideflow
conda env create -f slideflow/environment.yml
conda activate slideflow
pip install -e slideflow/ cupy-cuda11x
```

### Non-Commercial Add-ons

To add additional tools and pretrained models available under a non-commercial license, install `slideflow-gpl` and `slideflow-noncommercial`:

```
pip install slideflow-gpl slideflow-noncommercial
```

This will provide integrated access to 6 additional pretrained foundation models ([UNI](https://www.nature.com/articles/s41591-024-02857-3), [HistoSSL](https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v2.full.pdf), [GigaPath](https://aka.ms/gigapath), [PLIP](https://www.nature.com/articles/s41591-023-02504-3), [RetCCL](https://www.sciencedirect.com/science/article/abs/pii/S1361841522002730), and [CTransPath](https://www.sciencedirect.com/science/article/abs/pii/S1361841522002043)), the MIL architecture [CLAM](https://www.nature.com/articles/s41551-020-00682-w), the UQ algorithm [BISCUIT](https://www.nature.com/articles/s41467-022-34025-x), and the GAN framework [StyleGAN3](https://nvlabs-fi-cdn.nvidia.com/stylegan3/stylegan3-paper.pdf).

## ⚙️ Configuration

### Deep learning (PyTorch vs. Tensorflow)

Slideflow supports both PyTorch and Tensorflow, defaulting to PyTorch if both are available. You can specify the backend to use with the environmental variable `SF_BACKEND`. For example:

```
export SF_BACKEND=tensorflow
```

### Slide reading (cuCIM vs. Libvips)

By default, Slideflow reads whole-slide images using [cuCIM](https://docs.rapids.ai/api/cucim/stable/). Although much faster than other openslide-based frameworks, it supports fewer slide scanner formats. Slideflow also includes a [Libvips](https://libvips.github.io/libvips/) backend, which adds support for *.scn, *.mrxs, *.ndpi, *.vms, and *.vmu files. You can set the active slide backend with the environmental variable `SF_SLIDE_BACKEND`:

```
export SF_SLIDE_BACKEND=libvips
```


## Getting started
Slideflow experiments are organized into [Projects](https://slideflow.dev/project_setup), which supervise storage of whole-slide images, extracted tiles, and patient-level annotations. The fastest way to get started is to use one of our preconfigured projects, which will automatically download slides from the Genomic Data Commons:

```python
import slideflow as sf

P = sf.create_project(
    root='/project/destination',
    cfg=sf.project.LungAdenoSquam(),
    download=True
)
```

After the slides have been downloaded and verified, you can skip to [Extract tiles from slides](#extract-tiles-from-slides).

Alternatively, to create a new custom project, supply the location of patient-level annotations (CSV), slides, and a destination for TFRecords to be saved:

```python
import slideflow as sf
P = sf.create_project(
  '/project/path',
  annotations="/patient/annotations.csv",
  slides="/slides/directory",
  tfrecords="/tfrecords/directory"
)
```

Ensure that the annotations file has a `slide` column for each annotation entry with the filename (without extension) of the corresponding slide.

### Extract tiles from slides

Next, whole-slide images are segmented into smaller image tiles and saved in `*.tfrecords` format. [Extract tiles](https://slideflow.dev/slide_processing) from slides at a given magnification (width in microns size) and resolution (width in pixels) using `sf.Project.extract_tiles()`:

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

### Training models

Once tiles are extracted, models can be [trained](https://slideflow.dev/training). Start by configuring a set of [hyperparameters](https://slideflow.dev/model#modelparams):

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

Models can then be trained using these parameters. Models can be trained to categorical, multi-categorical, continuous, or time-series outcomes, and the training process is [highly configurable](https://slideflow.dev/training). For example, to train models in cross-validation to predict the outcome `'category1'` as stored in the project annotations file:

```python
P.train(
  'category1',
  params=params,
  save_predictions=True,
  multi_gpu=True
)
```

### Evaluation, heatmaps, mosaic maps, and more

Slideflow includes a host of additional tools, including model [evaluation and prediction](https://slideflow.dev/evaluation), [heatmaps](https://slideflow.dev/evaluation#heatmaps), analysis of [layer activations](https://slideflow.dev/posthoc), [mosaic maps](https://slideflow.dev/posthoc#mosaic-maps), and more. See our [full documentation](https://slideflow.dev) for more details and tutorials.

## 📚 Publications

Slideflow has been used by:

- [Dolezal et al](https://www.nature.com/articles/s41379-020-00724-3), _Modern Pathology_, 2020
- [Rosenberg et al](https://ascopubs.org/doi/10.1200/JCO.2020.38.15_suppl.e23529), _Journal of Clinical Oncology_ [abstract], 2020
- [Howard et al](https://www.nature.com/articles/s41467-021-24698-1), _Nature Communications_, 2021
- [Dolezal et al](https://www.nature.com/articles/s41467-022-34025-x) _Nature Communications_, 2022
- [Storozuk et al](https://www.nature.com/articles/s41379-022-01039-1.pdf), _Modern Pathology_ [abstract], 2022
- [Partin et al](https://doi.org/10.3389/fmed.2023.1058919) _Front Med_, 2022
- [Dolezal et al](https://ascopubs.org/doi/abs/10.1200/JCO.2022.40.16_suppl.8549) _Journal of Clinical Oncology_ [abstract], 2022
- [Dolezal et al](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9792820/) _Mediastinum_ [abstract], 2022
- [Howard et al](https://www.nature.com/articles/s41523-023-00530-5) _npj Breast Cancer_, 2023
- [Dolezal et al](https://www.nature.com/articles/s41698-023-00399-4) _npj Precision Oncology_, 2023
- [Hieromnimon et al](https://doi.org/10.1101/2023.03.22.533810) [bioRxiv], 2023
- [Carrillo-Perez et al](https://doi.org/10.1186/s40644-023-00586-3) _Cancer Imaging_, 2023

## 🔓 License
This code is made available under the Apache-2.0 license.

## 🔗 Reference
If you find our work useful for your research, or if you use parts of this code, please consider citing as follows:

Dolezal, J.M., Kochanny, S., Dyer, E. et al. Slideflow: deep learning for digital histopathology with real-time whole-slide visualization. BMC Bioinformatics 25, 134 (2024). https://doi.org/10.1186/s12859-024-05758-x

```
@Article{Dolezal2024,
    author={Dolezal, James M. and Kochanny, Sara and Dyer, Emma and Ramesh, Siddhi and Srisuwananukorn, Andrew and Sacco, Matteo and Howard, Frederick M. and Li, Anran and Mohan, Prajval and Pearson, Alexander T.},
    title={Slideflow: deep learning for digital histopathology with real-time whole-slide visualization},
    journal={BMC Bioinformatics},
    year={2024},
    month={Mar},
    day={27},
    volume={25},
    number={1},
    pages={134},
    doi={10.1186/s12859-024-05758-x},
    url={https://doi.org/10.1186/s12859-024-05758-x}
}
```
