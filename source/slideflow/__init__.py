from slideflow.project import SlideflowProject

# Planned updates
#TODO: unify slideflow model loading, even straight from directory, into a SlideflowModel file, which
# 		will auto-handle finding the hyperparameters.json file, etc
#TODO: remove/rework SFP flags -> properties
#TODO: property tags for SlideflowProject rather than SFP.PROJECT
#TODO: put tfrecord report in tfrecord directories
#TODO: relative paths in projects, denoted by $ROOT/...

# Known Issues / Bugs
#TODO: use hyperparameters file in saved model folder, rather than the parent folder
#TODO: fix slide predictions saving to CSV with multiple outcomes
#TODO: eval model folder naming
#TODO: fix hyperparameters.json outcome_labels with evaluate()
#TODO: either fix or deprecate dual tile extraction

__version__ = "1.11.0-dev6"