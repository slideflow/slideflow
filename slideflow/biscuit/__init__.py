"""Submodule for uncertainty quantification and confidence thresholding.

Bayesian Inference of Slide-level Confidence via Uncertainty Index Thresholding
(BISCUIT) is a uncertainty quantification and thresholding schema used to
separate deep learning classification predictions on whole-slide images (WSIs)
into low- and high-confidence. Uncertainty is estimated through dropout, which
approximates sampling of the Bayesian posterior, and thresholds are determined
on training data to mitigate data leakage during testing.

BISCUIT is available as a separate repository
`on GitHub <https://github.com/jamesdolezal/biscuit>`_. If you use this
uncertainty quantification approach in your research, please consider citing
as follows:

    Dolezal, J.M., Srisuwananukorn, A., Karpeyev, D. et al.
    Uncertainty-informed deep learning models enable high-confidence
    predictions for digital histopathology. Nat Commun 13, 6572 (2022).
    https://doi.org/10.1038/s41467-022-34025-x


"""

from . import hp
from . import experiment
from . import utils
from . import errors
from . import delong
from .utils import find_cv, get_model_results
from .experiment import Experiment

__version__ = '1.0.1'
