import os
import cv2
import numpy as np
from io import BytesIO
from os.path import join
from PIL import Image

from slideflow.slide import stainNorm_Macenko,  \
                            stainNorm_Reinhard, \
                            stainNorm_Vahadane, \
                            stainNorm_Augment,  \
                            stainNorm_Reinhard_Mask

class StainNormalizer:
    """Object to supervise stain normalization for images and efficiently convert between common image types."""

    normalizers = {
        'macenko':  stainNorm_Macenko.Normalizer,
        'reinhard': stainNorm_Reinhard.Normalizer,
        'reinhard_mask': stainNorm_Reinhard_Mask.Normalizer,
        'vahadane': stainNorm_Vahadane.Normalizer,
        'augment': stainNorm_Augment.Normalizer
    }

    def __init__(self, method='reinhard', source=None):
        """Initializer. Establishes normalization method.

        Args:
            method (str): Either 'macenko', 'reinhard', or 'vahadane'. Defaults to 'reinhard'.
            source (str): Path to source image for normalizer. If not provided, defaults to an internal example image.
        """

        self.method = method
        self._source = source
        if not source:
            package_directory = os.path.dirname(os.path.abspath(__file__))
            source = join(package_directory, 'norm_tile.jpg')
        self.n = self.normalizers[method]()
        self.n.fit(cv2.imread(source))

    def __repr__(self):
        src = "" if not self._source else ", source={!r}".format(self._source)
        return "StainNormalizer(method={!r}{})".format(self.method, src)

    def pil_to_pil(self, image):
        '''Non-normalized PIL.Image -> normalized PIL.Image'''
        cv_image = np.array(image.convert('RGB'))
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        cv_image = self.n.transform(cv_image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv_image)

    def tf_to_rgb(self, image):
        '''Non-normalized tensorflow image array -> normalized RGB numpy array'''
        return self.rgb_to_rgb(np.array(image))

    def rgb_to_rgb(self, image):
        '''Non-normalized RGB numpy array -> normalized RGB numpy array'''
        cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv_image = self.n.transform(cv_image)
        return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    def jpeg_to_rgb(self, jpeg_string):
        '''Non-normalized compressed JPG string data -> normalized RGB numpy array'''
        cv_image = cv2.imdecode(np.fromstring(jpeg_string, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv_image = self.n.transform(cv_image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return cv_image

    def png_to_rgb(self, png_string):
        '''Non-normalized compressed PNG string data -> normalized RGB numpy array'''
        return self.jpeg_to_rgb(png_string) # It should auto-detect format

    def jpeg_to_jpeg(self, jpeg_string):
        '''Non-normalized compressed JPG string data -> normalized compressed JPG string data'''
        cv_image = self.jpeg_to_rgb(jpeg_string)
        with BytesIO() as output:
            Image.fromarray(cv_image).save(output, format="JPEG", quality=75)
            return output.getvalue()