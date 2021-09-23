from slideflow.project import Project, SlideflowProject # SlideflowProject is deprecated, to be removed

# +-----------------------------------------+
# | Written and maintained by James Dolezal |
# | james.dolezal@uchospitals.edu           |
# +-----------------------------------------+

__version__ = "1.12.0-dev1"

# Style information
# =================
# General style format should conform to Google Python best-practices
# (http://google.github.io/styleguide/pyguide.html), with the exception of a
# maximum line length of 120. Docstrings should also conform with Google Style.
# (https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
# A pylintrc file is included is the root directory to assist with formatting.

# Version planning (v1.12)
# - Moving toward unified use of Datasets as input to various project functions
# - Calling extract_tiles directly on a dataset
# - Docstring updates to Google format
# - Updated documentation with more details in pytorch style
# - Model saving changes - number format, description in model files
# - Remove per-tile validation options, as it should never be used

# Planned updates
# ===============
#TODO: unify slideflow model loading, even straight from directory,
#         into a SlideflowModel file, which
#         will auto-handle finding the hyperparameters.json file, etc
#         Should also be abstract to support torch or slideflow backend
#TODO: put tfrecord report in tfrecord directories & include information
#         on normalization, filtering, slideflow version, etc
#TODO: neptune integration
#TODO: multiprocessing logging
#TODO: improve realtime normalization speed
#TODO: auto-create annotations file from slides
#TODO: fix return_unique in dataset.slide_to_label()
#TODO: add predict() option in addition to evaluate()