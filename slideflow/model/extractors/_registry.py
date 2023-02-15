"""Feature extractor registry."""

_tf_extractors = dict()
_torch_extractors = dict()

__all__ = ['list_extractors', 'list_tensorflow_extractors', 'list_torch_extractors',
           'is_extractor', 'is_tensorflow_extractor', 'is_torch_extractor']

# -----------------------------------------------------------------------------

def list_extractors():
    """Return a list of all available feature extractors."""
    return list(set(list(_tf_extractors.keys()) + list(_torch_extractors.keys())))

def list_tensorflow_extractors():
    """Return a list of all Tensorflow feature extractors."""
    return list(_tf_extractors.keys())

def list_torch_extractors():
    """Return a list of all PyTorch feature extractors."""
    return list(_torch_extractors.keys())

def is_extractor(name):
    """Checks if a given name is a valid feature extractor."""
    _valid_extractors = list_extractors()
    return (name in _valid_extractors or name+'_imagenet' in _valid_extractors)

def is_tensorflow_extractor(name):
    """Checks if a given name is a valid Tensorflow feature extractor."""
    return name in _tf_extractors or name+'_imagenet' in _tf_extractors

def is_torch_extractor(name):
    """Checks if a given name is a valid PyTorch feature extractor."""
    return name in _torch_extractors or name+'_imagenet' in _torch_extractors

# -----------------------------------------------------------------------------

def register_tf(fn):
    _tf_extractors[fn.__name__] = fn
    return fn

def register_torch(fn):
    _torch_extractors[fn.__name__] = fn
    return fn

