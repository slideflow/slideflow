"""
From https://github.com/wanghao14/Stain_Normalization
Normalize a patch stain to the target image using the method of:

E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.
"""

import slideflow.norm.utils as ut
from slideflow.norm.reinhard_fast import Normalizer as FastNormalizer

class Normalizer(FastNormalizer):
    """
    A stain normalization object
    """

    def __init__(self):
        super().__init__()

    def fit(self, target):
        target = ut.standardize_brightness(target)
        return super().fit(target)

    def transform(self, I):
        I = ut.standardize_brightness(I)
        return super().transform(I)