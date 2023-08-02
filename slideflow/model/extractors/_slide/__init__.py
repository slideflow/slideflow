"""Feature extraction from whole-slide images."""

# -----------------------------------------------------------------------------

def features_from_slide(extractor, slide, **kwargs):
    if extractor.is_torch():
        from ._torch import features_from_slide_torch
        return features_from_slide_torch(extractor, slide, **kwargs)
    else:
        from ._tf import features_from_slide_tf
        return features_from_slide_tf(extractor, slide, **kwargs)
