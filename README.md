![slideflow logo](http://jmd172.bitbucket.io/_images/full_logo.png)
# Slideflow README

Slideflow is a Python package which aims to provide an easy and intuitive way of building and testing neural network models for use in histology image analysis. It is built using Keras (with Tensorflow backend) and supports many standard network architectures - e.g. InceptionV3, Resnet, and VGG16 - as well as custom architectures.
 
The overarching goal of the package is to provide tools to train and test models on histology slides, apply these models to new slides, and analyze performance by generating predictive heatmaps, ROCs, and mosaic maps.

Slideflow requires Python 3.6+ and [openslide](https://openslide.org/download/).

To get started, ensure you have the latest version of pip, setuptools, and wheel installed:

```
pip3 install --upgrade setuptools pip wheel
```

Install package requirements from source/requirements.txt:

```
pip3 install -r requirements.txt
```

Import the module in python and initialize a new project:

```python
import slideflow
SFP = slideflow.SlideflowProject("/path/to/project/directory")
```

You will be taken through a set of questions to configure your new project. Slideflow projects require an annotations file (CSV) associating patient names to outcome categories and slide names. If desired, a blank file will be created for you when you first setup a new project. Once the project is created, add rows to the annotations file with patient names and outcome categories. 

Next, you will be taken through a set of questions to configure your first dataset. Alternatively, you may manually add a dataset by calling:

```python
SFP.add_dataset( name="NAME",
                 slides="/slides/directory",
                 roi="/roi/directory",
                 tiles="/tiles/directory",
                 tfrecords="/tfrecords/directory",
                 label="LABEL" )
```

If your patient names and slide names follow standard naming conventions as used in [The Cancer Genome Atlas](https://portal.gdc.cancer.gov/), you can have slideflow search through your slides directory and attempt to match patients in your annotations files with their corresponding slides with the following function:

```python
SFP.associate_slide_names()
```

Once your annotations file has been set up, begin extracting tiles:

```python
SFP.extract_tiles()
```

Following tile extraction, you can begin model training:

```python
SFP.train()
```

Complete documentation of all functions and options can be found at [slideflow.dev](https://www.slideflow.dev/).
