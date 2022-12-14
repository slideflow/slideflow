# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import cv2
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
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
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
    if _model_type in ('tensorflow', 'tflite'):
        return tf.image.decode_jpeg(img, channels=3)
    else:
        np_data = torch.from_numpy(np.fromstring(img, dtype=np.uint8))
        return torchvision.io.decode_image(np_data)


def _umap_normalize(vector, clip_min, clip_max, norm_min, norm_max):
    vector = np.clip(vector, clip_min, clip_max)
    vector -= norm_min
    vector /= (norm_max - norm_min)
    return vector

#----------------------------------------------------------------------------

class Renderer:
    def __init__(self, device=None, buffer=None, extract_px=None, tile_px=None):
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
        self._buffer            = buffer
        self.extract_px         = extract_px
        self.tile_px            = tile_px
        self.device             = device
        self._umap_encoders     = None
        self._addl_renderers    = []

    @property
    def model_type(self):
        if self._model is None:
            return None
        else:
            return model_backend(self._model)

    def process_tf_preds(self, preds):
        if isinstance(preds, list):
            return [self.to_numpy(p[0]) for p in preds]
        else:
            return self.to_numpy(preds[0])

    def to_numpy(self, x):
        if self.model_type in ('tensorflow', 'tflite'):
            return x.numpy()
        else:
            return x.cpu().detach().numpy()

    def add_renderer(self, renderer):
        self._addl_renderers += [renderer]

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

        if self.model_type in ('tensorflow', 'tflite'):
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
        if self.model_type in ('tensorflow', 'tflite'):
            img = tf.expand_dims(img, axis=0)
        elif self.model_type == 'torch':
            img = torch.unsqueeze(img, dim=0)

        if use_uncertainty:
            return self._calc_preds_and_uncertainty(img)
        elif self.model_type == 'tflite':
            sig_name = list(self._model._inputs.keys())[0]
            preds = self._model(**{sig_name: img})['output']
            if isinstance(preds, list):
                preds = [p[0] for p in preds]
            else:
                preds = preds[0]
        else:
            preds = self._model(img)
            preds = self.process_tf_preds(preds)
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
        viewer              = None,
        tile_px             = None,
        tile_um             = None,
        full_image          = None,
        use_umap_encoders   = False,
        **kwargs
    ):
        args = locals()
        del args['kwargs']
        del args['self']
        args.update(kwargs)
        args['use_model'] = False
        for renderer in self._addl_renderers:
            renderer._render_impl(**args)
        if (x is None or y is None) and 'image' not in res:
            return
        if full_image is not None:
            img = cv2.resize(full_image, (tile_px, tile_px))
            res.image = img
        elif viewer is not None:

            if not self._model:
                res.message = "Model not loaded"
                print(res.message)
                return

            res.predictions = None
            res.uncertainty = None

            if self.model_type in ('tensorflow', 'tflite'):
                dtype = tf.uint8
            elif self.model_type == 'torch':
                dtype = torch.uint8

            use_jpeg = use_model and img_format is not None and img_format.lower() in ('jpg', 'jpeg')
            img = viewer.read_tile(x, y, img_format='jpg' if use_jpeg else None, allow_errors=True)
            if img is None:
                res.message = "Invalid tile location."
                print(res.message)
                return
            if use_jpeg:
                img = _decode_jpeg(img, self.model_type)
                if self.model_type == 'torch':
                    res.image = sf.io.torch.cwh_to_whc(img).numpy()
                res.image = img.numpy()
                img = sf.io.convert_dtype(img, dtype)
            else:
                res.image = img
        elif 'image' in res and use_model:
            # Image was generated by one of the additional renderers.
            # Check if any additional pre-processing needs to be done.
            img = res.image
            for renderer in self._addl_renderers:
                if hasattr(renderer, 'preprocess'):
                    img = renderer.preprocess(img, tile_px=tile_px, tile_um=tile_um)

        if use_model:
            if self.model_type in ('tensorflow', 'tflite') and isinstance(img, np.ndarray):
                proc_img = tf.convert_to_tensor(img)
            elif isinstance(img, np.ndarray):
                proc_img = sf.io.torch.whc_to_cwh(torch.from_numpy(img)).to(self.device)
            else:
                proc_img = img

            # Pre-process image.
            if normalizer:
                _norm_start = time.time()
                proc_img = normalizer.transform(proc_img)
                if not isinstance(proc_img, np.ndarray):
                    if self.model_type == 'torch':
                        res.normalized = sf.io.torch.cwh_to_whc(proc_img).numpy().astype(np.uint8)
                    else:
                        res.normalized = proc_img.numpy().astype(np.uint8)
                else:
                    res.normalized = proc_img.astype(np.uint8)
                res.norm_time = time.time() - _norm_start
            if self.model_type in ('tensorflow', 'tflite'):
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
            _inference_start = time.time()
            if use_umap_encoders:
                u = self._umap_encoders
                encoder_out = u.encoder(tf.expand_dims(proc_img, axis=0))
                res.umap_coords = {}
                for i, layer in enumerate(u.layers):
                    res.umap_coords[layer] = _umap_normalize(
                        encoder_out[i],
                        clip_min=u.clip[layer][0],
                        clip_max=u.clip[layer][1],
                        norm_min=u.range[layer][0],
                        norm_max=u.range[layer][1]
                    )[0]
                res.predictions = self.process_tf_preds(encoder_out[-1])
                res.uncertainty = None
            else:
                res.predictions, res.uncertainty = self._classify_img(proc_img, use_uncertainty=use_uncertainty)
            res.inference_time = time.time() - _inference_start

#----------------------------------------------------------------------------
