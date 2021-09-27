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
# DONE - Moving toward unified use of Datasets as input to various project functions
# DONE - Calling extract_tiles directly on a dataset
# DONE - Docstring updates to Google format
# DONE - Refactoring entire codebase
# IN PROGRESS - Updated documentation with more details in pytorch style
# DONE - Remove per-tile validation options, as it should never be used

# Future updates
# ===============
#TODO: implement native TF normalizers to improve realtime normalization speed
#TODO: put tfrecord report in tfrecord directories & include information
#         on normalization, filtering, slideflow version, etc
#TODO: neptune integration