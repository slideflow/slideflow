import os
import torch
import torchvision

from io import BytesIO
from slideflow.slide import torch_reinhard
from os.path import join
from PIL import Image

class TorchStainNormalizer:
    vectorized = True
    normalizers = {
        'reinhard': torch_reinhard
    }

    def __init__(self, method='reinhard', source=None):

        self.method = method
        self.n = self.normalizers[method]
        self._source = source
        if not source:
            package_directory = os.path.dirname(os.path.abspath(__file__))
            source = join(package_directory, 'norm_tile.jpg')
        self.src_img = torchvision.io.read_image(source).permute(1, 2, 0) # CWH => WHC
        means, stds = self.n.fit(torch.unsqueeze(self.src_img, dim=0))
        self.target_means = torch.cat(means, 0)
        self.target_stds = torch.cat(stds, 0)

    def __repr__(self):
        src = "" if not self._source else ", source={!r}".format(self._source)
        return "TorchStainNormalizer(method={!r}{})".format(self.method, src)

    def fit(self, *args):
        return NotImplementedError()

    def batch_to_batch(self, batch, *args):
        device = torch.device("cuda")
        if isinstance(batch, dict):
            to_return = {k:v for k,v in batch.items() if k != 'tile_image'}
            to_return['tile_image'] = self.torch_to_torch(batch['tile_image'])
            return to_return, *args
        else:
            return self.torch_to_torch(batch), *args

    def torch_to_torch(self, image):
        if len(image.shape) == 3:
            return self.n.transform(torch.unsqueeze(image, dim=0), self.target_means, self.target_stds).squeeze()
        else:
            return self.n.transform(image, self.target_means, self.target_stds)

    def torch_to_rgb(self, image):
        return self.torch_to_torch(image).numpy()

    def rgb_to_rgb(self, image):
        '''Non-normalized RGB numpy array -> normalized RGB numpy array'''
        return self.n.transform(torch.from_numpy(image), self.target_means, self.target_stds).numpy()

    def jpeg_to_rgb(self, jpeg_string):
        '''Non-normalized compressed JPG string data -> normalized RGB numpy array'''
        return self.torch_to_rgb(torchvision.io.decode_image(jpeg_string))

    def png_to_rgb(self, png_string):
        return self.torch_to_rgb(torchvision.io.decode_image(png_string))

    def jpeg_to_jpeg(self, jpeg_string, quality=75):
        np_image = self.jpeg_to_rgb(jpeg_string)
        with BytesIO() as output:
            Image.fromarray(np_image).save(output, format="JPEG", quality=quality)
            return output.getvalue()

    def png_to_png(self, png_string):
        np_image = self.png_to_rgb(png_string)
        with BytesIO() as output:
            Image.fromarray(np_image).save(output, format="PNG")
            return output.getvalue()