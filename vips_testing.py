import pyvips as vips
import re
import cv2
import numpy as np
import slideflow as sf
from types import SimpleNamespace
from PIL import Image, UnidentifiedImageError
from typing import (Any, Dict, List, Optional, Tuple, Union)
from slideflow import errors
from slideflow.util import log, path_to_name, path_to_ext  # noqa F401
from slideflow.slide.utils import *

### DFCI Image
image = vips.Image.new_from_file("/mnt/labshare/SLIDES/DFCI_ACC/images/BS-10-E13153-B1.svs")
print("DFCI Image pages xres: ", image.get('xres'), "\n")
print("DFCI Image image-description: ", image.get('image-description'), "\n")
### UCH Bennett Image
b_image = vips.Image.new_from_file(r"/mnt/labshare/SLIDES/UCH_BENNETT_new/all_images_rois/images/JB_Research;41__20210503_100836.tiff")
#print("UCH Bennett Image pages resolution-unit: ", b_image.get('resolution-unit'), '\n')
#print("UCH Bennett Image pages image-description: ", b_image.get('image-description'), '\n')

### UCH ACC Image
u_image = vips.Image.new_from_file("/mnt/labshare/SLIDES/UCH_ACC/images/S09-24848.svs")
print('UCH ACC Image fields: ', vips.Image.new_from_file("/mnt/labshare/SLIDES/UCH_ACC/images/S09-24848.svs").get_fields())


### Following the slide loading process

# slide = sf.WSI("/mnt/labshare/SLIDES/DFCI_ACC/images/BS-10-E13153-B1.svs", tile_px=299, tile_um=302)

# This is creating an instance of the slide object
# wsi_reader(
#                 self.path,
#                 self._mpp_override,
#                 **self._reader_kwargs)