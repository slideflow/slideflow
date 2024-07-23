import sys
import cv2
import time
import traceback
import numpy as np
import slideflow as sf
import slideflow.grad
from typing import Optional
from slideflow.util import model_backend
from rich import print

from .utils import EasyDict, _load_model_and_saliency

# -----------------------------------------------------------------------------

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

ABORT_RENDER = 10
ENABLE_EXPERIMENTAL_UQ = False

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
    return yp_mean.cpu().numpy(), yp_std.cpu().numpy()


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

def _prepare_args(args, kwargs):
    del args['kwargs']
    del args['self']
    args.update(kwargs)
    args['use_model'] = False
    return args

#----------------------------------------------------------------------------

class Renderer:
    def __init__(self, device=None, buffer=None, extract_px=None, tile_px=None):
        self._pkl_data          = dict()    # {pkl: dict | CapturedException, ...}
        self._pinned_bufs       = dict()    # {(shape, dtype): torch.Tensor, ...}
        self._cmaps             = dict()    # {name: torch.Tensor, ...}
        self._is_timing         = False
        self._net_layers        = dict()
        self._uq_thread         = None
        self._model             = None
        self._uq_model          = None
        self._deepfocus         = None
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

    def disable_deepfocus(self):
        print("Disabling deepfocus")
        del self._deepfocus
        self._deepfocus = None

    def enable_deepfocus(self):
        if self._deepfocus is None:
            print("Setting up DeepFocus model...")
            self._deepfocus = sf.slide.qc.DeepFocus().model
            print("...done")

    def _image_in_focus(self, img, method='laplacian') -> bool:
        """Predict whether an image is in-focus using DeepFocus."""
        if method == 'deepfocus':
            self.enable_deepfocus()
            img = tf.cast(tf.convert_to_tensor(img), tf.uint8)
            proc_img = tf.image.resize(img, (64, 64), method='lanczos3')
            # From what I can tell, DeepFocus was trained using the preprocessing steps:
            #   (img / 255.) - mean(img / 255.)
            # However, this does not work well with a live camera feed, probably because
            #   the brightness/contrast is lower.
            # Instead, I've found that tf.image.per_image_standardization() works better
            #   on live camera images, as it scales the variance of the image to be 1,
            #   which would be effectively similary to increasing contrast in the image.
            proc_img = tf.image.per_image_standardization(proc_img)
            proc_img = tf.expand_dims(proc_img, axis=0)
            _focus_pred = self._deepfocus(proc_img, training=False)[0][1]
            return _focus_pred > 0.5
        elif method == 'laplacian':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            return laplacian_variance > 80
        else:
            raise ValueError("Unrecognized focus method: {}".format(method))

    def process_tf_preds(self, preds):
        if isinstance(preds, list):
            return [self.to_numpy(p[0]) for p in preds]
        else:
            return self.to_numpy(preds[0])

    def to_numpy(self, x, as_whc=False):
        if self.model_type in ('tensorflow', 'tflite'):
            return x.numpy()
        else:
            if sf.io.torch.is_cwh(x):
                x = sf.io.torch.cwh_to_whc(x)
            return x.cpu().detach().numpy()

    def add_renderer(self, renderer):
        """Add a renderer to the rendering pipeline."""
        sf.log.debug(f"Adding renderer: {renderer}")
        self._addl_renderers += [renderer]

    def remove_renderer(self, renderer):
        """Remove a renderer from the rendering pipeline."""
        sf.log.debug(f"Removing renderer: {renderer}")
        idx = self._addl_renderers.index(renderer)
        del self._addl_renderers[idx]

    def render(self, **args):
        """Render predictions for an image and any post-processing."""
        self._is_timing = True
        self._start_time = time.time()
        res = EasyDict()
        try:
            self._render_impl(res, **args)
        except:
            res.error = CapturedException()
        if 'error' in res:
            res.error = str(res.error)
        if self._is_timing:
            res.render_time = time.time() - self._start_time
            self._is_timing = False
        return res

    def load_model(self, model_path: str, device: Optional[str] = None) -> None:
        """Load a model."""
        _model, _saliency, _umap_encoders = _load_model_and_saliency(model_path, device=device)
        self.set_model(_model, uq=sf.util.is_uq_model(model_path))
        self.set_saliency(_saliency)
        self.set_umap_encoders(_umap_encoders)

    def set_model(self, model, uq=False):
        """Set a loaded model to the active model."""
        self._model = model
        if uq and ENABLE_EXPERIMENTAL_UQ:
            self._uq_model = sf.model.tensorflow.build_uq_model(model)

    def set_saliency(self, saliency):
        """Set a loaded saliency model to the active saliency model."""
        self._saliency = saliency

    def set_umap_encoders(self, umap_encoders):
        """Set a loaded UMAP encoder to the active UMAP encoder."""
        self._umap_encoders = umap_encoders

    def _ignore_timing(self):
        self._is_timing = False

    def _calc_preds_and_uncertainty(self, img, uq_n=30):

        if self.model_type in ('tensorflow', 'tflite') and ENABLE_EXPERIMENTAL_UQ:
            yp_mean, yp_std = self._uq_model(img, training=False)
            yp_mean, yp_std = yp_mean.numpy()[0], yp_std.numpy()[0]
            num_outcomes = 1 if not isinstance(yp_std, list) else len(yp_std)
        elif self.model_type in ('tensorflow', 'tflite'):
            yp = self._model(tf.repeat(img, repeats=uq_n, axis=0), training=False)
            reduce_fn = _reduce_dropout_preds_tf
        else:
            import torch
            with torch.no_grad():
                yp = self._model(img.expand(uq_n, -1, -1, -1))
            reduce_fn = _reduce_dropout_preds_torch

        if self.model_type == 'torch' or not ENABLE_EXPERIMENTAL_UQ:
            # Previously for both tf & torch
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
            if self.model_type == 'torch':
                with torch.no_grad():
                    preds = self._model(img)
            else:
                preds = self._model(img, training=False)
            preds = self.process_tf_preds(preds)
        return preds, None

    def _read_tile_from_viewer(self, x, y, viewer, use_model, img_format):
        """Read a tile from the viewer at the given x/y coordinates."""
        # Determine the target type.
        if self.model_type in ('tensorflow', 'tflite'):
            dtype = tf.uint8
        elif self.model_type == 'torch':
            dtype = torch.uint8

        # Read the image tile.
        use_jpeg = use_model and img_format is not None and img_format.lower() in ('jpg', 'jpeg')
        img = viewer.read_tile(x, y, img_format='jpg' if use_jpeg else None, allow_errors=True)
        if img is None:
            return None, None

        # Convert types.
        if use_jpeg:
            img = _decode_jpeg(img, self.model_type)
            if self.model_type == 'torch':
                result_img = sf.io.torch.cwh_to_whc(img).cpu().numpy()
            else:
                result_img = img.numpy()
            img = sf.io.convert_dtype(img, dtype)
        else:
            result_img = img

        return img, result_img

    def _run_models(
        self,
        img,
        res,
        *,
        normalizer=None,
        use_saliency=False,
        saliency_method=None,
        saliency_overlay=None,
        use_umap_encoders=False,
        use_uncertainty=False,
        focus_img=None,
        assess_focus=None,
    ):
        """Run model pipelines on a rendered image."""

        # Check focus.
        if focus_img is not None:
            res.in_focus = self._image_in_focus(focus_img, method=assess_focus)
            if not res.in_focus:
                return

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
                    res.normalized = sf.io.torch.cwh_to_whc(proc_img).cpu().numpy().astype(np.uint8)
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
            mask = self._saliency.get(self.to_numpy(proc_img, as_whc=True), method=saliency_method)
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
            res.predictions, res.uncertainty = self._classify_img(
                proc_img,
                use_uncertainty=use_uncertainty
            )
        res.inference_time = time.time() - _inference_start

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
        assess_focus        = False,
        **kwargs
    ):

        # Trigger other renderers in the pipeline.
        renderer_args = _prepare_args(locals(), kwargs)
        for renderer in self._addl_renderers:
            exit_code = renderer._render_impl(**renderer_args)
            if exit_code == ABORT_RENDER:
                return

        # If coordinates are not provided and an image is not
        # already stored in `res`, then skip.
        if (x is None or y is None) and 'image' not in res:
            return

        # If full_image is provided, use this instead of looking up
        # a tile image from the viewer.
        focus_img, img = None, None
        if full_image is not None:
            if not tile_px:
                return
            img = cv2.resize(full_image, (tile_px, tile_px), interpolation=cv2.INTER_LANCZOS4)
            res.image = img

            if assess_focus:
                w = full_image.shape[0]
                if assess_focus == 'deepfocus':
                    # DeepFocus target size is tile_px=64 tile_um=256 (40X),
                    # but we'll work at 20X since it's more practical.
                    crop_ratio = 128. / tile_um
                    if crop_ratio > 1:
                        raise NotImplementedError
                    crop_w = crop_ratio * w
                else:
                    # Other focus methods use a 64x64 center crop
                    crop_w = 64
                focus_img = full_image[int(w/2-crop_w/2):int(w/2+crop_w/2),
                                       int(w/2-crop_w/2):int(w/2+crop_w/2)]


        # If image is in res, it was generated by one of the additional renderers.
        # Check if any additional pre-processing needs to be done.
        elif 'image' in res and use_model:
            img = res.image
            for renderer in self._addl_renderers:
                if hasattr(renderer, 'preprocess'):
                    img = renderer.preprocess(img, tile_px=tile_px, tile_um=tile_um)

        # Otherwise, use the viewer to find the tile image.
        elif viewer is not None:

            res.predictions = None
            res.uncertainty = None

            img, result_img = self._read_tile_from_viewer(
                x, y, viewer, use_model=use_model, img_format=img_format
            )
            if assess_focus:
                raise NotImplementedError

            if img is None:
                res.message = "Invalid tile location."
                print(res.message)
                return
            else:
                res.image = result_img

        if assess_focus and focus_img is None:
            focus_img = img

        # ---------------------------------------------------------------------

        if use_model and self._model:
            self._run_models(
                img,
                res,
                normalizer=normalizer,
                use_saliency=use_saliency,
                saliency_method=saliency_method,
                saliency_overlay=saliency_overlay,
                use_umap_encoders=use_umap_encoders,
                use_uncertainty=use_uncertainty,
                focus_img=(None if not assess_focus else focus_img),
                assess_focus=assess_focus
            )
