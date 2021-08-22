from slideflow.project import SlideflowProject

#TODO: unify slideflow model loading, even straight from directory, into a SlideflowModel file, which
# 		will auto-handle finding the hyperparameters.json file, etc
#TODO: put tfrecord report in tfrecord directories
#TODO: use hyperparameters file in saved model folder, rather than the parent folder
#TODO: either fix or deprecate dual tile extraction
#TODO: relative paths in projects, denoted by $ROOT/...
#TODO: property tags for SlideflowProject rather than SFP.PROJECT
#TODO: remove/rework SFP flags -> properties
#TODO: fix slide predictions saving to CSV with multiple outcomes
#TODO: eval model folder naming
#TODO: fix hyperparameters.json outcome_labels with evaluate()

__version__ = "1.11.0-dev6"