"""Submodule used to interface with the PyTorch implementation of StyleGAN2
(maintained separately at https://github.com/jamesdolezal/stylegan2-slideflow).
"""

from .stylegan2 import stylegan2
from .stylegan3 import stylegan3
from .interpolate import StyleGAN2Interpolator