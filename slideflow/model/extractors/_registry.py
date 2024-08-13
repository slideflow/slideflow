"""Feature extractor registry."""

_tf_extractors = dict()
_torch_extractors = dict()
_known_extras_packages = {
     'slideflow-gpl': ['retccl', 'ctranspath'],
     'slideflow-noncommercial': ['gigapath', 'gigapath.tile', 'gigapath.slide', 'histossl', 'plip']
}
_extras_extractors = {
    extractor: package 
    for package, extractors in _known_extras_packages.items() 
    for extractor in extractors
}

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

def register_torch(key_name=None):
    """Decorator to register a PyTorch feature extractor."""

    def decorator(fn):
        # Use the custom key name if provided, otherwise use the function's name
        name = key_name if isinstance(key_name, str) else fn.__name__
        _torch_extractors[name] = fn
        return fn

    # If the decorator is used without arguments, the key_name will be the function itself
    if callable(key_name):
        return decorator(key_name)

    return decorator

def register_tf(key_name=None):
    """Decorator to register a Tensorflow feature extractor."""

    def decorator(fn):
        # Use the custom key name if provided, otherwise use the function's name
        name = key_name if isinstance(key_name, str) else fn.__name__
        _tf_extractors[name] = fn
        return fn

    # If the decorator is used without arguments, the key_name will be the function itself
    if callable(key_name):
        return decorator(key_name)

    return decorator
