# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import time
import sys
import traceback
import numpy as np
from .utils import EasyDict

from rich import print
import slideflow as sf
from slideflow.util import model_backend

if sf.util.tf_available:
    import tensorflow as tf
    import slideflow.io.tensorflow
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception:
        pass
if sf.util.torch_available:
    import torch
    import torchvision
    import slideflow.io.torch

#----------------------------------------------------------------------------

class CapturedException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            _type, value, _traceback = sys.exc_info()
            assert value is not None
            if isinstance(value, CapturedException):
                msg = str(value)
            else:
                msg = traceback.format_exc()
        assert isinstance(msg, str)
        super().__init__(msg)

#----------------------------------------------------------------------------

class CaptureSuccess(Exception):
    def __init__(self, out):
        super().__init__()
        self.out = out

#----------------------------------------------------------------------------

def _reduce_dropout_preds_torch(yp_drop, num_outcomes, stack=True):
    if num_outcomes > 1:
        if stack:
            yp_drop = [torch.stack(yp_drop[n], dim=0) for n in range(num_outcomes)]
        else:
            yp_drop = [yp_drop[0] for n in range(num_outcomes)]
        yp_mean = [torch.mean(yp_drop[n], dim=0) for n in range(num_outcomes)]
        yp_std = [torch.std(yp_drop[n], dim=0) for n in range(num_outcomes)]
    else:
        if stack:
            yp_drop = torch.stack(yp_drop[0], dim=0)  # type: ignore
        else:
            yp_drop = yp_drop[0]
        yp_mean = torch.mean(yp_drop, dim=0)  # type: ignore
        yp_std = torch.std(yp_drop, dim=0)  # type: ignore
    return yp_mean, yp_std


def _reduce_dropout_preds_tf(yp_drop, num_outcomes, stack=True):
    if num_outcomes > 1:
        if stack:
            yp_drop = [tf.stack(yp_drop[n], axis=0) for n in range(num_outcomes)]
        else:
            yp_drop = [yp_drop[0] for n in range(num_outcomes)]
        yp_mean = [tf.math.reduce_mean(yp_drop[n], axis=0).numpy() for n in range(num_outcomes)]
        yp_std = [tf.math.reduce_std(yp_drop[n], axis=0).numpy() for n in range(num_outcomes)]
    else:
        if stack:
            yp_drop = tf.stack(yp_drop[0], axis=0)
        else:
            yp_drop = yp_drop[0]
        yp_mean = tf.math.reduce_mean(yp_drop, axis=0).numpy()
        yp_std = tf.math.reduce_std(yp_drop, axis=0).numpy()
    return yp_mean, yp_std


def _decode_jpeg(img, _model_type):
    if _model_type == 'tensorflow':
        return tf.image.decode_jpeg(img, channels=3)
    else:
        np_data = torch.from_numpy(np.fromstring(img, dtype=np.uint8))
        return torchvision.io.decode_image(np_data)


#----------------------------------------------------------------------------

class Renderer:
    def __init__(self, device=None):
        #self._device            = torch.device('cuda')
        self._pkl_data          = dict()    # {pkl: dict | CapturedException, ...}
        self._pinned_bufs       = dict()    # {(shape, dtype): torch.Tensor, ...}
        self._cmaps             = dict()    # {name: torch.Tensor, ...}
        self._is_timing         = False
        #self._start_event       = torch.cuda.Event(enable_timing=True)
        #self._end_event         = torch.cuda.Event(enable_timing=True)
        self._net_layers        = dict()
        self._uq_thread         = None
        self._model             = None
        self._saliency          = None
        self.device             = device

    @property
    def model_type(self):
        return model_backend(self._model)

    def to_numpy(self, x):
        if self.model_type == 'tensorflow':
            return x.numpy()
        else:
            return x.cpu().detach().numpy()

    def render(self, **args):
        self._is_timing = True
        #self._start_event.record(torch.cuda.current_stream(self._device))
        self._start_time = time.time()
        res = EasyDict()
        try:
            self._render_impl(res, **args)
        except:
            res.error = CapturedException()
        #self._end_event.record(torch.cuda.current_stream(self._device))
        if 'error' in res:
            res.error = str(res.error)
        if self._is_timing:
            #self._end_event.synchronize()
            res.render_time = time.time() - self._start_time #self._start_event.elapsed_time(self._end_event) * 1e-3
            self._is_timing = False
        return res

    def _ignore_timing(self):
        self._is_timing = False

    def _calc_preds_and_uncertainty(self, img, uq_n=30):

        if self.model_type == 'tensorflow':
            yp = self._model(tf.repeat(img, repeats=uq_n, axis=0), training=False)
            reduce_fn = _reduce_dropout_preds_tf
        else:
            yp = self._model(torch.repeat(img, repeats=uq_n, dim=0), training=False)
            reduce_fn = _reduce_dropout_preds_torch

        num_outcomes = 1 if not isinstance(yp, list) else len(yp)
        yp_drop = {n: [] for n in range(num_outcomes)}
        if num_outcomes > 1:
            for o in range(num_outcomes):
                yp_drop[o] = yp[o]
        else:
            yp_drop[0] = yp
        yp_mean, yp_std = reduce_fn(yp_drop, num_outcomes, stack=False)
        if num_outcomes > 1:
            uncertainty = [np.mean(s) for s in yp_std]
        else:
            uncertainty = np.mean(yp_std)
        predictions = yp_mean
        return predictions, uncertainty

    def _classify_img(self, img, use_uncertainty=False):
        """Classify an image, returning predictions and uncertainty.

        Args:
            img (_type_): _description_

        Returns:
            Predictions, Uncertainty
        """
        if self.model_type == 'tensorflow':
            img = tf.expand_dims(img, axis=0)
        elif self.model_type == 'torch':
            img = torch.unsqueeze(img, dim=0)

        if use_uncertainty:
            return self._calc_preds_and_uncertainty(img)
        else:
            preds = self._model(img)
            if isinstance(preds, list):
                preds = [self.to_numpy(p[0]) for p in preds]
            else:
                preds = self.to_numpy(preds[0])
            return preds, None

    def _render_impl(self, res,
        x                   = 0,
        y                   = 0,
        saliency_overlay    = False,
        saliency_method     = 0,
        img_format          = None,
        use_model           = False,
        use_uncertainty     = False,
        use_saliency        = False,
        normalizer          = None,
        wsi                 = None,
    ):
        if x is None or y is None:
            return
        assert wsi is not None

        res.predictions = None
        res.uncertainty = None

        import pyvips

        if self.model_type == 'tensorflow':
            dtype = tf.uint8
        elif self.model_type == 'torch':
            dtype = torch.uint8

        decode_jpeg = img_format is not None and img_format.lower() in ('jpg', 'jpeg')

        try:
            region = wsi.slide.read_region(
                (x, y),
                wsi.downsample_level,
                (wsi.extract_px, wsi.extract_px)
            )
            if region.bands == 4:
                region = region.flatten()  # removes alpha
            if int(wsi.tile_px) != int(wsi.extract_px):
                region = region.resize(wsi.tile_px/wsi.extract_px)
            if decode_jpeg:
                img = _decode_jpeg(region.jpegsave_buffer(), self.model_type)
                if self.model_type == 'torch':
                    res.image = sf.io.torch.cwh_to_whc(img).numpy()
                res.image = img.numpy()
                img = sf.io.convert_dtype(img, dtype)
            elif img_format in ('png', None):
                res.image = img = sf.slide.vips2numpy(region)
            else:
                raise ValueError(f"Unknown image format {img_format}")

        except pyvips.error.Error:
            print(f"Tile coordinates {x}, {y} are out of bounds, skipping")
        else:
            if use_model:
                if not self._model:
                    res.message = "Model not loaded"
                    return

                if self.model_type == 'tensorflow' and isinstance(img, np.ndarray):
                    proc_img = tf.convert_to_tensor(img)
                elif isinstance(img, np.ndarray):
                    proc_img = sf.io.torch.whc_to_cwh(torch.from_numpy(img)).to(self.device)
                else:
                    proc_img = img

                # Pre-process image.
                if normalizer:
                    proc_img = normalizer.transform(proc_img)
                    if not isinstance(proc_img, np.ndarray):
                        if self.model_type == 'torch':
                            res.normalized = sf.io.torch.cwh_to_whc(proc_img).numpy().astype(np.uint8)
                        else:
                            res.normalized = proc_img.numpy().astype(np.uint8)
                    else:
                        res.normalized = proc_img.astype(np.uint8)
                if self.model_type == 'tensorflow':
                    proc_img = sf.io.tensorflow.preprocess_uint8(proc_img, standardize=True)['tile_image']
                elif self.model_type == 'torch':
                    proc_img = sf.io.torch.preprocess_uint8(proc_img, standardize=True)
                    if self.device is not None:
                        proc_img = proc_img.to(self.device)

                # Saliency.
                if use_saliency:
                    mask = self._saliency.get(self.to_numpy(proc_img), method=saliency_method)
                    if saliency_overlay:
                        res.image = sf.grad.plot_utils.overlay(res.image, mask)
                    else:
                        res.image = sf.grad.plot_utils.inferno(mask)
                    if res.image.shape[-1] == 4:
                        res.image = res.image[:, :, 0:3]

                # Show predictions.
                predictions, uncertainty = self._classify_img(proc_img, use_uncertainty=use_uncertainty)
                res.predictions = predictions
                res.uncertainty = uncertainty

#----------------------------------------------------------------------------
