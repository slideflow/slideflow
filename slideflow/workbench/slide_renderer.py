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

def _reduce_dropout_preds(yp_drop, num_outcomes, stack=True):
    if sf.backend() == 'tensorflow':
        import tensorflow as tf
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
    else:
        import torch
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


def _decode(img, img_format):
    if sf.backend() == 'tensorflow':
        import tensorflow as tf
        if img_format.lower() in ('jpeg', 'jpg'):
            return tf.image.decode_jpeg(img, channels=3)
        elif img_format.lower() == 'png':
            return tf.image.decode_png(img, channels=3)
    else:
        import torch
        import torchvision
        np_data = torch.from_numpy(np.fromstring(img, dtype=np.uint8))
        return torchvision.io.decode_image(np_data)


#----------------------------------------------------------------------------

class Renderer:
    def __init__(self, visualizer):
        self._visualizer        = visualizer
        #self._device            = torch.device('cuda')
        self._pkl_data          = dict()    # {pkl: dict | CapturedException, ...}
        self._pinned_bufs       = dict()    # {(shape, dtype): torch.Tensor, ...}
        self._cmaps             = dict()    # {name: torch.Tensor, ...}
        self._is_timing         = False
        #self._start_event       = torch.cuda.Event(enable_timing=True)
        #self._end_event         = torch.cuda.Event(enable_timing=True)
        self._net_layers        = dict()
        self._uq_thread         = None
        self._stop_uq_thread    = False
        self._stop_pred_thread  = False

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
        self._visualizer._uncertainty = None

        import tensorflow as tf
        yp = self._visualizer._model(tf.repeat(img, repeats=uq_n, axis=0), training=False)
        num_outcomes = 1 if not isinstance(yp, list) else len(yp)
        yp_drop = {n: [] for n in range(num_outcomes)}
        if num_outcomes > 1:
            for o in range(num_outcomes):
                yp_drop[o] = yp[o]
        else:
            yp_drop[0] = yp
        yp_mean, yp_std = _reduce_dropout_preds(yp_drop, num_outcomes, stack=False)
        if num_outcomes > 1:
            self._visualizer._uncertainty = [np.mean(s) for s in yp_std]
        else:
            self._visualizer._uncertainty = np.mean(yp_std)
        self._visualizer._predictions = yp_mean

    def _classify_img(self, img):
        def run_classification():
            nonlocal img
            if sf.backend() == 'tensorflow':
                import tensorflow as tf
                img = tf.expand_dims(img, axis=0)
                to_numpy = lambda x: x.numpy()
            elif sf.backend() == 'torch':
                import torch
                img = torch.unsqueeze(img, dim=0)
                to_numpy = lambda x: x.cpu().detach().numpy()

            if self._visualizer.has_uq() and self._visualizer._use_uncertainty:
                self._calc_preds_and_uncertainty(img)
            else:
                preds = self._visualizer._model(img)
                if isinstance(preds, list):
                    preds = [to_numpy(p[0]) for p in preds]
                else:
                    preds = to_numpy(preds[0])
                self._visualizer._predictions = preds

        run_classification()

    def _render_impl(self, res,
        x                   = 0,
        y                   = 0,
        show_saliency       = False,
        saliency_overlay    = False,
        saliency_method     = 0,
        img_format          = None,
    ):
        if x is None or y is None:
            return

        import pyvips
        # Stop UQ thread if running.
        self._stop_uq_thread = True
        viz = self._visualizer
        wsi = viz.wsi

        if sf.backend() == 'tensorflow':
            import tensorflow as tf
            dtype = tf.uint8
        elif sf.backend() == 'torch':
            import torch
            dtype = torch.uint8

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
            if img_format is not None and img_format.lower() in ('jpg', 'jpeg'):
                proc_img = img = _decode(region.jpegsave_buffer(), img_format)
            elif img_format == 'png':
                proc_img = img = _decode(region.pngsave_buffer(), img_format)
            elif img_format is None:
                res.image = img = sf.slide.vips2numpy(region)
            else:
                raise ValueError(f"Unknown image format {img_format}")
            if img_format is not None:
                if sf.backend() == 'torch':
                    res.image = sf.io.torch.cwh_to_whc(img).numpy()
                else:
                    res.image = img.numpy()
                proc_img = sf.io.convert_dtype(img, dtype)
            else:
                proc_img = img
        except pyvips.error.Error as e:
            print(f"Tile coordinates {x}, {y} are out of bounds, skipping")
        else:
            if viz._use_model:
                # Pre-process image.
                normalizer = viz._normalizer
                if normalizer:
                    proc_img = normalizer.transform(proc_img)
                    if not isinstance(proc_img, np.ndarray):
                        if sf.backend() == 'torch':
                            res.normalized = sf.io.torch.cwh_to_whc(proc_img).numpy().astype(np.uint8)
                        else:
                            res.normalized = proc_img.numpy().astype(np.uint8)
                    else:
                        res.normalized = proc_img.astype(np.uint8)
                if sf.backend() == 'tensorflow':
                    proc_img = sf.io.tensorflow.preprocess_uint8(proc_img, standardize=True)['tile_image']
                elif sf.backend() == 'torch':
                    proc_img = sf.io.torch.preprocess_uint8(proc_img, standardize=True)

                # Saliency.
                if show_saliency:
                    mask = viz.saliency.get(proc_img.numpy(), method=saliency_method)
                    if saliency_overlay:
                        res.image = sf.grad.plot_utils.overlay(img, mask)
                    else:
                        res.image = sf.grad.plot_utils.inferno(mask)
                    if res.image.shape[-1] == 4:
                        res.image = res.image[:, :, 0:3]

                # Show predictions.
                self._classify_img(proc_img)

#----------------------------------------------------------------------------
