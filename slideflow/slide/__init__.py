'''This module includes tools to convolutionally section whole slide images
into tiles. These tessellated tiles can be exported as PNG or JPG as raw
images or stored in the binary format TFRecords, with or without augmentation.'''

import warnings

import slideflow.slide.qc
from slideflow.util import SUPPORTED_FORMATS  # noqa F401
from .report import ExtractionPDF  # noqa F401
from .report import ExtractionReport, SlideReport
from .utils import *
from .backends import tile_worker, wsi_reader, backend_formats
from .wsi import WSI

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 100000000000

# -----------------------------------------------------------------------
