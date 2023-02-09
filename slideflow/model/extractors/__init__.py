"""Module for building pretrained feature extractors."""

def get_feature_extractor(name, **kwargs):
    if name == 'ctranspath':
        from .ctranspath import CTransPathFeatures
        return CTransPathFeatures(**kwargs)
    else:
        raise ValueError(f"Unrecognized feature extractor {name}")