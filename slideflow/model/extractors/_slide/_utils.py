"""Utility functions for slide feature extraction."""

import numpy as np
import slideflow as sf
from slideflow import log

# -----------------------------------------------------------------------------

def _build_grid(extractor, slide, grid=None, dtype=np.float16):
    """Build a grid of features for a slide."""

    total_out = (extractor.num_features
                 + extractor.num_classes
                 + extractor.num_uncertainty)
    if grid is None:
        features_grid = np.ones((
                slide.grid.shape[1],
                slide.grid.shape[0],
                total_out),
            dtype=dtype)
        features_grid *= sf.heatmap.MASK
    else:
        assert grid.shape == (slide.grid.shape[1], slide.grid.shape[0], total_out)
        features_grid = grid
    return features_grid


def _log_normalizer(normalizer):
    """Log the stain normalizer being used."""
    if normalizer is None:
        log.debug("Calculating slide features without stain normalization")
    else:
        log.debug(
            f"Calculating slide features using stain normalizer: {normalizer}"
        )

def _use_numpy_if_png(img_format):
    """Use numpy image format instead of PNG."""
    if img_format == 'png':  # PNG is lossless; this is equivalent but faster
        log.debug("Using numpy image format instead of PNG")
        return 'numpy'
    return img_format
