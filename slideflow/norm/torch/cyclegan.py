"""CycleGAN-based stain normalization.

Modified from: https://github.com/Boehringer-Ingelheim/stain-transfer
Pretrained weights from: https://osf.io/byf27/
Original license:

BSD 2-Clause License

Copyright (c) 2023, Boehringer Ingelheim

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn as nn
import numpy as np
import functools
import torchvision.transforms as transforms
import slideflow as sf

from torchvision.transforms.functional import center_crop
from typing import Union, Optional, List, Tuple
from slideflow.io.torch import is_whc, as_cwh, as_whc
from slideflow.model.torch import autocast
from slideflow.model import torch_utils

# -----------------------------------------------------------------------------

def download_weights() -> Tuple[str, str]:
    """Download the pretrained checkpoint from HuggingFace."""
    from huggingface_hub import hf_hub_download

    sf.log.debug(
        "Using pretrained CycleGAN weights, available at https://osf.io/byf27/"
    )
    he2mt = hf_hub_download(
        repo_id='jamesdolezal/stain-transfer', filename='cyclegan_he2mt.pth'
    )
    mt2he = hf_hub_download(
        repo_id='jamesdolezal/stain-transfer', filename='cyclegan_mt2he.pth'
    )
    return he2mt, mt2he

# -----------------------------------------------------------------------------

def get_pad_layer(pad_type):
    if (pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif (pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif (pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


def get_filter(filt_size=3):
    if (filt_size == 1):
        a = np.array([1., ])
    elif (filt_size == 2):
        a = np.array([1., 1.])
    elif (filt_size == 3):
        a = np.array([1., 2., 1.])
    elif (filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif (filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif (filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif (filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """
        Initialize the Resnet block
        A resnet block is a conv block with skip connections. We construct a
        conv block with build_conv_block function, and implement skip connections
        in <forward> function. Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer,
                                                use_dropout, use_bias)

    def build_conv_block(self, dim: int, padding_type: str,
                         norm_layer: functools.partial, use_dropout: bool,
                         use_bias: bool):
        """
        Construct a convolutional block.

        :param dim: number of channels in the conv layer.
        :param padding_type: name of padding layer: reflect | replicate | zero
        :param norm_layer: normalization layer
        :param use_dropout: if use dropout layers.
        :param use_bias:    if the conv layer uses bias or not
        :return: a conv block (with a conv layer, a normalization layer, and a
        non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetGenerator(nn.Module):
    """ Resnet-based generator that consists of Resnet blocks between a few
    downsampling/upsampling operations."""

    def __init__(self, input_nc: int, output_nc: int, ngf: int = 64,
                 norm_layer: functools.partial = nn.BatchNorm2d,
                 use_dropout: bool = False, n_blocks: int = 6,
                 padding_type: str = 'reflect', no_antialias: bool = True,
                 no_antialias_up: bool = True):
        """Construct a Resnet-based generator

        :param input_nc: number of channels in input images
        :param output_nc: number of channels in output images
        :param ngf: number of filters in the last conv layer
        :param norm_layer: normalization layer
        :param use_dropout: if use dropout layers
        :param n_blocks: the number of ResNet blocks
        :param padding_type: the name of padding layer in conv layers:
        reflect | replicate | zero
        :param no_antialias: if true, use stride=2 convs instead of
        antialiased-downsampling
        :param no_antialias_up: if true, use [upconv(learned filter)] instead of
        [upconv(hard-coded [1,3,3,1] filter), conv]
        """

        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf), nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if (no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2), nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2), nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  norm_layer=norm_layer,
                                  use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2, padding=1,
                                             output_padding=1, bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1, padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input: torch.Tensor):
        """Standard forward"""
        return self.model(input)


class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride ** 2)
        self.register_buffer('filt', filt[None, None, :, :].repeat(
            (self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = nn.functional.conv_transpose2d(
            self.pad(inp), self.filt,
            stride=self.stride,
            padding=1 + self.pad_size,
            groups=inp.shape[1])[:, :, 1:, 1:]
        if (self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2,
                 pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2),
                          int(np.ceil(1. * (filt_size - 1) / 2)),
                          int(1. * (filt_size - 1) / 2),
                          int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat(
            (self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if (self.filt_size == 1):
            if (self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return nn.functional.conv2d(self.pad(inp), self.filt,
                                        stride=self.stride, groups=inp.shape[1])

# -----------------------------------------------------------------------------

class CycleGAN(nn.Module):
    """
    This class implements the CycleGAN model generator, for image-to-image
    translation inference. CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf.
    """

    no_antialias = True
    no_antialias_up = True

    def __init__(self, device=None):
        """
        Initializes model and create dataset loaders.

        :param conf: See BaseModel.
        :param no_antialias: leave as True, do not modify.
        :param no_antialias_up: leave as True, do not modify.
        """
        super().__init__()

        self.net = ResnetGenerator(
            3, 3, 64,
            norm_layer=functools.partial(
                nn.InstanceNorm2d,
                affine=False,
                track_running_stats=False
            ),
            use_dropout=False,
            n_blocks=9,
            no_antialias=self.no_antialias,
            no_antialias_up=self.no_antialias_up
        )
        self.device = torch_utils.get_device(device)
        self.net.to(self.device)
        self.transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def forward(self, t: torch.Tensor):
        """ Run forward pass."""
        return self.net(t.to(self.device))

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """ Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and (
                    key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and (
                    key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(
                state_dict, getattr(module, key), keys, i + 1)

    def load_weights(self, weights_path: str):
        state_dict = torch.load(weights_path, map_location=self.device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):
            # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, self.net,
                                                  key.split('.'))
        self.net.load_state_dict(state_dict)

    def to(self, device):
        self.device = device
        self.net.to(device)
        return self

# -----------------------------------------------------------------------------

class CycleGanStainTranslator:
    """Translates images from HE to MT, and vice versa, using two CycleGAN models."""

    def __init__(
        self,
        he2mt_weights: Optional[str] = None,
        mt2he_weights: Optional[str] = None,
        *,
        device = None,
        mixed_precision: bool = True,
    ) -> None:

        # Declare types.
        self.he2mt: CycleGAN
        self.mt2he: CycleGAN
        self.he2mt_weights: Optional[str] = None
        self.mt2he_weights: Optional[str] = None

        if he2mt_weights is None and mt2he_weights is None:
            try:
                he2mt_weights, mt2he_weights = download_weights()
            except Exception as e:
                sf.log.warning("Unable to download pretrained weights. Error: {}".format(e))

        self.device = device or torch_utils.get_device()
        self.mixed_precision = mixed_precision
        sf.log.debug("CycleGAN mixed_precision={}".format(self.mixed_precision))
        self.build_networks()
        self.load_weights(he2mt_weights, mt2he_weights)
        self.normalize = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.to_tensor = transforms.ToTensor()

    def _assert_loaded(self) -> None:
        if self.he2mt_weights is None or self.mt2he_weights is None:
            raise RuntimeError('Weights not loaded.')

    @property
    def _device_type(self):
        if isinstance(self.device, str):
            return self.device
        return self.device.type

    def build_networks(self) -> None:
        self.he2mt = CycleGAN(device=self.device)
        self.mt2he = CycleGAN(device=self.device)

    def load_weights(self, he2mt_weights: str, mt2he_weights: str) -> None:
        self.he2mt_weights = he2mt_weights
        self.mt2he_weights = mt2he_weights
        if he2mt_weights:
            self.he2mt.load_weights(he2mt_weights)
        if mt2he_weights:
            self.mt2he.load_weights(mt2he_weights)

    def to(self, device):
        self.device = device
        self.he2mt = self.he2mt.to(device)
        self.mt2he = self.mt2he.to(device)
        return self

    def crop_to(self, img: torch.Tensor, shape: List[int]) -> None:
        """Crop an image (or batch of images) to match the target shape."""
        # Convert to CWH format.
        convert_back_to_whc = is_whc(img)
        img = as_cwh(img)

        # Get width/height from the target shape.
        target_shape = shape if len(shape) == 3 else shape[1:]  # Remove batch dim.
        if target_shape[0] == 3:
            target_shape = target_shape[1:]  # If CWH, remove C.
        else:
            target_shape = target_shape[:2]  # If WHC, remove C.

        # Crop.
        cropped = center_crop(img, target_shape)

        # Convert back to WHC format, if needed.
        if convert_back_to_whc:
            return as_whc(cropped)
        return cropped

    def he_to_he(self, img: Union[np.ndarray, torch.Tensor]):
        """
        Translates an image from HE to HE.

        :param img: image to translate, in RGB format.
        :return: translated image, in RGB format.
        """
        with torch.no_grad():
            mt = self.he_to_mt(img, as_tensor=True)
            he = self.mt_to_he(mt, as_tensor=True)
            if he.shape != img.shape:
                he = self.crop_to(he, img.shape)
            if not isinstance(img, torch.Tensor):
                he = as_whc(he).cpu().numpy()

        return he

    def he_to_mt(
        self,
        img: Union[np.ndarray, torch.Tensor],
        as_tensor: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Translates an image from HE to MT.

        :param img: image to translate, in RGB format.
        :param as_tensor: if True, returns a tensor instead of a numpy array.
        :return: translated image, in RGB format.
        """
        self._assert_loaded()
        if isinstance(img, np.ndarray):
            img = self.to_tensor(img)
        whc = is_whc(img)
        with autocast(self._device_type, mixed_precision=self.mixed_precision):
            with torch.no_grad():
                img = self.normalize(as_cwh(img).float()).to(self.he2mt.device)
                mt = self.he2mt(img)
                mt = torch.clamp((mt + 1) / 2.0 * 255, 0, 255).to(torch.uint8)

        if as_tensor:
            return mt if not whc else as_whc(mt)
        else:
            return as_whc(mt).cpu().numpy()

    def mt_to_he(
        self,
        img: Union[np.ndarray, torch.Tensor],
        as_tensor: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Translates an image from MT to HE.

        :param img: image to translate, in RGB format.
        :param as_tensor: if True, returns a tensor instead of a numpy array.
        :return: translated image, in RGB format.
        """
        self._assert_loaded()
        if isinstance(img, np.ndarray):
            img = self.to_tensor(img)
        whc = is_whc(img)
        with autocast(self._device_type, mixed_precision=self.mixed_precision):
            with torch.no_grad():
                img = self.normalize(as_cwh(img).float()).to(self.mt2he.device)
                he = self.mt2he(img)
                he = torch.clamp((he + 1) / 2.0 * 255, 0, 255).to(torch.uint8)

        if as_tensor:
            return he if not whc else as_whc(he)
        else:
            return as_whc(he).cpu().numpy()

# -----------------------------------------------------------------------------

class CycleGanNormalizer(CycleGanStainTranslator):

    vectorized = True
    preferred_device = 'cuda'
    preset_tag = 'cyclegan'

    def __init__(self) -> None:
        """CycleGAN-based stain normalizer"""
        super().__init__()

    def _assert_loaded(self) -> None:
        if self.he2mt_weights is None or self.mt2he_weights is None:
            raise RuntimeError(
                'CycleGAN normalizer weights have not been loaded. '
                'Load weights by setting the normalizer fit parameters '
                '(he2mt_weights, mt2he_weights) to the path of the '
                '*.pth weights files.'
            )

    def fit(self, *args, **kwargs):
        self.set_fit(*args, **kwargs)

    def fit_preset(self, preset: Optional[str] = None):
        pass

    def get_fit(self) -> None:
        return {
            'he2mt_weights': self.he2mt_weights,
            'mt2he_weights': self.mt2he_weights,
        }

    def set_fit(self, he2mt_weights: str, mt2he_weights: str) -> None:
        """Set the normalizer fit to the given values."""
        self.load_weights(he2mt_weights, mt2he_weights)

    def transform(self, img: torch.Tensor, augment: bool = False) -> torch.Tensor:
        """Normalize a WxHxC H&E image (uint8)."""
        _img = torch.unsqueeze(img, 0) if len(img.shape) == 3 else img
        transformed = self.he_to_he(_img)
        if len(img.shape) == 3:
            transformed = torch.squeeze(transformed, 0)
        return transformed


class CycleGanReinhardNormalizer(CycleGanNormalizer):

    preset_tag = 'cyclegan_reinhard'

    def __init__(self) -> None:
        super().__init__()
        self.reinhard = sf.norm.autoselect(
            'reinhard_mask',
            backend='torch',
            device=self.device
        )

    def transform(self, img: torch.Tensor, augment: bool = False) -> torch.Tensor:
        """Normalize a WxHxC H&E image (uint8)."""
        from slideflow.io.torch.augment import RandomColorProfile
        img = self.reinhard.transform(img, augment=augment)
        img = super().transform(img, augment=augment)
        if augment:
            img = RandomColorProfile()(img)
        else:
            img = self.reinhard.transform(img, augment=False)
        return img


class ReinhardCycleGanNormalizer(CycleGanNormalizer):

    preset_tag = 'reinhard_cyclegan'

    def __init__(self) -> None:
        super().__init__()
        self.reinhard = sf.norm.autoselect(
            'reinhard_mask',
            backend='torch',
            device=self.device
        )

    def transform(self, img: torch.Tensor, augment: bool = False) -> torch.Tensor:
        """Normalize a WxHxC H&E image (uint8)."""
        from slideflow.io.torch.augment import RandomColorProfile
        img = self.reinhard.transform(img, augment=augment)
        img = super().transform(img, augment=augment)
        return img


class ReinhardCycleGanColorNormalizer(CycleGanNormalizer):

    preset_tag = 'reinhard_cyclegan_color'

    def __init__(self) -> None:
        super().__init__()
        self.reinhard = sf.norm.autoselect(
            'reinhard_mask',
            backend='torch',
            device=self.device
        )

    def transform(self, img: torch.Tensor, augment: bool = False) -> torch.Tensor:
        """Normalize a WxHxC H&E image (uint8)."""
        from slideflow.io.torch.augment import RandomColorProfile
        img = self.reinhard.transform(img, augment=augment)
        img = super().transform(img, augment=augment)
        if augment:
            img = RandomColorProfile()(img)
        return img