"""Feature extractor registry."""

_mil_trainers = dict()
_mil_models = dict()
_known_extras_trainers = {
    'slideflow-gpl': ['legacy_clam'],
}
_known_extras_models = {
    'slideflow-gpl': ['clam_mb', 'clam_sb', 'mil_fc', 'mil_fc_mc'],
}
_extras_trainers = {
    trainer: package 
    for package, trainers in _known_extras_trainers.items() 
    for trainer in trainers
}
_extras_models = {
    model: package 
    for package, models in _known_extras_models.items() 
    for model in models
}

__all__ = ['list_trainers', 'list_models', 'is_trainer', 'is_model']

# -----------------------------------------------------------------------------

def list_trainers():
    """List all available trainers."""
    return list(set(list(_mil_trainers.keys())))


def list_models():
    """List all available models."""
    return list(set(list(_mil_models.keys())))


def is_trainer(name):
    """Check if a trainer is available."""
    _valid_trainers = list_trainers()
    return (name in _valid_trainers)


def is_model(name):
    """Check if a model is available."""
    _valid_models = list_models()
    return (name in _valid_models)


def get_trainer(trainer_name):
    """Get a trainer by name."""
    if is_trainer(trainer_name): 
        return _mil_trainers[trainer_name]()
    if trainer_name in _extras_trainers:
        package = _extras_trainers[trainer_name]
        raise ValueError(f"Trainer '{trainer_name}' is part of the '{package}' package. Please install it to use this trainer.")
    raise ValueError(f"Unknown trainer '{trainer_name}'. Valid options are: {list_trainers()}")


def get_model(model_name):
    """Get a model by name."""
    if is_model(model_name):
        model_fn, config_class = _mil_models[model_name]
        return model_fn()
    if model_name in _extras_models:
        package = _extras_models[model_name]
        raise ValueError(f"Model '{model_name}' is part of the '{package}' package. Please install it to use this model.")
    raise ValueError(f"Unknown model '{model_name}'. Valid options are: {list_models()}")


def get_model_config_class(model_name):
    """Get the model configuration class."""
    if is_model(model_name):
        _, config_class = _mil_models[model_name]
        return config_class
    if model_name in _extras_models:
        package = _extras_models[model_name]
        raise ValueError(f"Model '{model_name}' is part of the '{package}' package. Please install it to use this model.")
    raise ValueError(f"Unknown model '{model_name}'. Valid options are: {list_models()}")



def build_model_config(model_name, **kwargs):
    """Get a model config by name."""
    if is_model(model_name):
        model_fn, config_class = _mil_models[model_name]
        return config_class(model_name, **kwargs)
    if model_name in _extras_models:
        package = _extras_models[model_name]
        raise ValueError(f"Model '{model_name}' is part of the '{package}' package. Please install it to use this model.")
    raise ValueError(f"Unknown model '{model_name}'. Valid options are: {list_models()}")


def register_trainer(tag=None):
    """Decorate to register a new trainer."""
    
    def decorator(fn):
        nonlocal tag
        name = tag or fn.__name__
        _mil_trainers[name] = fn
        return fn
    
    if callable(tag):
        fn = tag
        tag = None
        return decorator(fn)

    return decorator


def register_model(tag=None, config=None):
    """Decorator to register a PyTorch feature extractor."""

    def decorator(fn):
        nonlocal tag, config
        # Use the custom key name if provided, otherwise use the function's name
        from slideflow.mil import MILModelConfig

        name = tag or fn.__name__
        config = config or MILModelConfig
        _mil_models[name] = (fn, config)
        return fn
    
    if callable(tag):
        fn = tag
        tag = None
        return decorator(fn)

    return decorator
