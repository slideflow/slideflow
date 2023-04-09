"""Submodule for calculating/displaying pixel attribution (saliency maps)."""

from typing import Any, Callable, Dict, Optional

import slideflow as sf
import numpy as np
import saliency.core as saliency
from functools import partial
from slideflow import errors
from slideflow.grad.plot_utils import (comparison_plot, inferno, multi_plot,
                                       oranges, overlay,
                                       saliency_map_comparison)

VANILLA = 0
VANILLA_SMOOTH = 1
INTEGRATED_GRADIENTS = 2
INTEGRATED_GRADIENTS_SMOOTH = 3
GUIDED_INTEGRATED_GRADIENTS = 4
GUIDED_INTEGRATED_GRADIENTS_SMOOTH = 5
BLUR_INTEGRATED_GRADIENTS = 6
BLUR_INTEGRATED_GRADIENTS_SMOOTH = 7
XRAI = 8
XRAI_FAST = 9

class SaliencyMap:

    def __init__(self, model: Callable, class_idx: int) -> None:
        """Class to assist with calculation and display of saliency maps.

        Args:
            model (Callable): Differentiable model from which saliency is
                calculated.
            class_idx (int): Index of class for backpropagating gradients.
        """
        if not callable(model):
            raise ValueError("'model' must be a differentiable model.")
        self.model = model
        self.feature_model = None
        self.feature_model_layer = None
        self.class_idx = class_idx
        self.gradients = saliency.GradientSaliency()
        self.gradcam_grads = saliency.GradCam()
        self.ig = saliency.IntegratedGradients()
        self.guided_ig = saliency.GuidedIG()
        self.blur_ig = saliency.BlurIG()
        self.xrai_grads = saliency.XRAI()
        self.fast_xrai_params = saliency.XRAIParameters()
        self.fast_xrai_params.algorithm = 'fast'
        self._fn_map = {
            VANILLA: self.vanilla,
            VANILLA_SMOOTH: partial(self.vanilla, smooth=True),
            INTEGRATED_GRADIENTS: self.integrated_gradients,
            INTEGRATED_GRADIENTS_SMOOTH: partial(self.integrated_gradients, smooth=True),
            GUIDED_INTEGRATED_GRADIENTS: self.guided_integrated_gradients,
            GUIDED_INTEGRATED_GRADIENTS_SMOOTH: partial(self.guided_integrated_gradients, smooth=True),
            BLUR_INTEGRATED_GRADIENTS: self.blur_integrated_gradients,
            BLUR_INTEGRATED_GRADIENTS_SMOOTH: partial(self.blur_integrated_gradients),
            XRAI: self.xrai,
            XRAI_FAST: self.xrai_fast
        }

    @property
    def model_backend(self):
        return sf.util.model_backend(self.model)

    @property
    def device(self):
        if self.model_backend == 'tensorflow':
            return None
        else:
            return next(self.model.parameters()).device

    def _update_feature_model(self, layer):
        if self.feature_model_layer == layer:
            return

        # Cleanup old model
        del self.feature_model
        self.feature_model = None
        self.feature_model_layer = layer

        if self.model_backend == 'tensorflow':
            import tensorflow as tf
            flattened = sf.model.tensorflow_utils.flatten(self.model)
            conv_layer = flattened.get_layer(layer)
            self.feature_model = tf.keras.models.Model([flattened.inputs], [conv_layer.output, flattened.output])
        else:
            import torch
            from slideflow.model import torch_utils
            conv_layer = torch_utils.get_module_by_name(self.model, layer)
            self._torch_conv_layer_outputs = {}
            def conv_layer_forward(m, i, o):
                # move the RGB dimension to the last dimension
                self._torch_conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).detach().numpy()
            def conv_layer_backward(m, i, o):
                # move the RGB dimension to the last dimension
                self._torch_conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, 3).detach().numpy()
            conv_layer.register_forward_hook(conv_layer_forward)
            conv_layer.register_full_backward_hook(conv_layer_backward)

    def _grad_fn_torch(
        self,
        image: np.ndarray,
        call_model_args: Any = None,
        expected_keys: Dict = None
    ) -> Any:
        """Calculate gradient attribution with PyTorch backend.

        Images are expected to be in W, H, C format.

        """
        import torch
        from slideflow.io.torch import whc_to_cwh
        image = torch.tensor(image, requires_grad=True).to(torch.float32).to(self.device)  # type: ignore
        output = self.model(whc_to_cwh(image))
        if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:  # type: ignore
            outputs = output[:, self.class_idx]
            grads = torch.autograd.grad(outputs, image, grad_outputs=torch.ones_like(outputs))  # type: ignore
            gradients = grads[0].cpu().detach().numpy()
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            # For Grad-CAM
            one_hot = torch.zeros_like(output)
            one_hot[:, self.class_idx] = 1
            self.model.zero_grad()  # type: ignore
            output.backward(gradient=one_hot, retain_graph=True)
            return self._torch_conv_layer_outputs

    def _grad_fn_tf(
        self,
        image: np.ndarray,
        call_model_args: Any = None,
        expected_keys: Dict = None
    ) -> Any:
        """Calculate gradient attribution with Tensorflow backend."""
        import tensorflow as tf

        image = tf.convert_to_tensor(image)
        with tf.GradientTape() as tape:
            if expected_keys == [saliency.base.INPUT_OUTPUT_GRADIENTS]:
                # For vanilla gradient, Integrated Gradients, XRAI
                tape.watch(image)
                output = self.model(image)[:, self.class_idx]
                gradients = tape.gradient(output, image)
                return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
            else:
                # For Grad-CAM
                conv_layer, output_layer = self.feature_model(image)
                gradients = np.array(tape.gradient(output_layer, conv_layer))
                return {saliency.base.CONVOLUTION_LAYER_VALUES: conv_layer,
                        saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}

    def _grad_fn(
        self,
        image: np.ndarray,
        call_model_args: Any = None,
        expected_keys: Dict = None
    ) -> Any:
        """Calculate gradient attribution."""
        if self.model_backend == 'tensorflow':
            return self._grad_fn_tf(image, call_model_args, expected_keys)
        elif self.model_backend == 'torch':
            return self._grad_fn_torch(image, call_model_args, expected_keys)
        else:
            raise errors.UnrecognizedBackendError

    def _apply_mask_fn(
        self,
        img: np.ndarray,
        grads: saliency.CoreSaliency,
        baseline: bool = False,
        smooth: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Applys a saliency masking function to a gradients map.

        Args:
            img (np.ndarray or list(np.ndarray)): Image or list of images.
            grads (saliency.CoreSaliency): Gradients for saliency.
            baseline (bool): Requires x_baseline argument.
            smooth (bool): Use a smoothed mask.

        Returns:
            np.ndarray: Saliency map.
        """
        mask_fn = grads.GetSmoothedMask if smooth else grads.GetMask

        def _get_mask(_img):
            if baseline:
                kwargs.update({'x_baseline': np.zeros(_img.shape)})
            out = mask_fn(_img, self._grad_fn, **kwargs)
            return out

        if isinstance(img, list):
            # Normalize together
            image_3d = list(map(_get_mask, img))
            v_maxes, v_mins = zip(*[max_min(img3d) for img3d in image_3d])
            vmax = max(v_maxes)
            vmin = min(v_mins)
            return [grayscale(img3d, vmax=vmax, vmin=vmin) for img3d in image_3d]
        else:
            return grayscale(_get_mask(img))

    def all(self, img: np.ndarray) -> Dict:
        """Calculate all saliency map methods.

        Args:
            img (np.ndarray): Pre-processed input image in W, H, C format.

        Returns:
            Dict: Dictionary mapping name of saliency method to saliency map.
        """
        return {
            'Vanilla': self.vanilla(img),
            'Vanilla (Smoothed)': self.vanilla(img, smooth=True),
            'Integrated Gradients': self.integrated_gradients(img),
            'Integrated Gradients (Smooth)': self.integrated_gradients(img, smooth=True),
            'Guided Integrated Gradients': self.guided_integrated_gradients(img),
            'Guided Integrated Gradients (Smooth)': self.guided_integrated_gradients(img, smooth=True),
            'Blur Integrated Gradients': self.blur_integrated_gradients(img),
            'Blur Integrated Gradients (Smooth)': self.blur_integrated_gradients(img, smooth=True),
        }

    def get(self, img: np.ndarray, method: int) -> np.ndarray:
        return self._fn_map[method](img)

    def vanilla(
        self,
        img: np.ndarray,
        smooth: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Calculate gradient-based saliency map.

        Args:
            img (np.ndarray): Pre-processed input image in W, H, C format.
            smooth (bool, optional): Smooth gradients. Defaults to False.

        Returns:
            np.ndarray: Saliency map.
        """
        return self._apply_mask_fn(
            img,
            self.gradients,
            smooth=smooth,
            **kwargs
        )

    def gradcam(
        self,
        img: np.ndarray,
        layer: str,
        smooth: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Calculate gradient-based saliency map.

        Args:
            img (np.ndarray): Pre-processed input image in W, H, C format.
            smooth (bool, optional): Smooth gradients. Defaults to False.

        Returns:
            np.ndarray: Saliency map.
        """
        self._update_feature_model(layer)
        return self._apply_mask_fn(
            img,
            self.gradcam_grads,
            smooth=smooth,
            **kwargs
        )

    def integrated_gradients(
        self,
        img: np.ndarray,
        x_steps: int = 25,
        batch_size: int = 20,
        smooth: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Calculate saliency map using integrated gradients.

        Args:
            img (np.ndarray): Pre-processed input image in W, H, C format.
            x_steps (int, optional): Steps for gradient calculation.
                Defaults to 25.
            max_dist (float, optional): Maximum distance for gradient
                calculation. Defaults to 1.0.
            smooth (bool, optional): Smooth gradients. Defaults to False.

        Returns:
            np.ndarray: Saliency map.
        """
        return self._apply_mask_fn(
            img,
            self.ig,
            smooth=smooth,
            x_steps=x_steps,
            batch_size=batch_size,
            baseline=True,
            **kwargs
        )

    def guided_integrated_gradients(
        self,
        img: np.ndarray,
        x_steps: int = 25,
        max_dist: float = 1.0,
        fraction: float = 0.5,
        smooth: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Calculate saliency map using guided integrated gradients.

        Args:
            img (np.ndarray): Pre-processed input image in W, H, C format.
            x_steps (int, optional): Steps for gradient calculation.
                Defaults to 25.
            max_dist (float, optional): Maximum distance for gradient
                calculation. Defaults to 1.0.
            fraction (float, optional): Fraction for gradient calculation.
                Defaults to 0.5.
            smooth (bool, optional): Smooth gradients. Defaults to False.

        Returns:
            np.ndarray: Saliency map.
        """
        return self._apply_mask_fn(
            img,
            self.guided_ig,
            x_steps=x_steps,
            max_dist=max_dist,
            fraction=fraction,
            smooth=smooth,
            baseline=True,
            **kwargs
        )

    def blur_integrated_gradients(
        self,
        img: np.ndarray,
        batch_size: int = 20,
        smooth: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Calculate saliency map using blur integrated gradients.

        Args:
            img (np.ndarray): Pre-processed input image in W, H, C format.
            batch_size (int, optional): Batch size. Defaults to 20.
            smooth (bool, optional): Smooth gradients. Defaults to False.

        Returns:
            np.ndarray: Saliency map.
        """
        return self._apply_mask_fn(
            img,
            self.blur_ig,
            smooth=smooth,
            batch_size=batch_size,
            **kwargs
        )

    def xrai(
        self,
        img: np.ndarray,
        batch_size: int = 20,
        **kwargs
    ) -> np.ndarray:
        """Calculate saliency map using XRAI.

        Args:
            img (np.ndarray): Pre-processed input image in W, H, C format.
            batch_size (int, optional): Batch size. Defaults to 20.

        Returns:
            np.ndarray: Saliency map.
        """
        mask = self.xrai_grads.GetMask(
            img,
            self._grad_fn,
            batch_size=batch_size,
            **kwargs
        )
        if isinstance(img, list):
            # Normalize together
            v_maxes, v_mins = zip(*[max_min(img3d) for img3d in mask])
            vmax = max(v_maxes)
            vmin = min(v_mins)
            return [normalize_xrai(img3d, vmax=vmax, vmin=vmin) for img3d in mask]
        else:
            return normalize_xrai(mask)

    def xrai_fast(
        self,
        img: np.ndarray,
        batch_size: int = 20,
        **kwargs
    ) -> np.ndarray:
        """Calculate saliency map using XRAI (fast implementation).

        Args:
            img (np.ndarray): Pre-processed input image in W, H, C format.
            batch_size (int, optional): Batch size. Defaults to 20.

        Returns:
            np.ndarray: Saliency map.
        """
        mask = self.xrai_grads.GetMask(
            img,
            self._grad_fn,
            batch_size=batch_size,
            extra_parameters=self.fast_xrai_params,
            **kwargs
        )
        if isinstance(img, list):
            # Normalize together
            v_maxes, v_mins = zip(*[max_min(img3d) for img3d in mask])
            vmax = max(v_maxes)
            vmin = min(v_mins)
            return [normalize_xrai(img3d, vmax=vmax, vmin=vmin) for img3d in mask]
        else:
            return normalize_xrai(mask)


def grayscale(image_3d, vmax=None, vmin=None, percentile=99):
    """Returns a 3D tensor as a grayscale 2D tensor.
    This method sums a 3D tensor across the absolute value of axis=2, and then
    clips values at a given percentile.
    """
    if vmax is None and vmin is None:
        vmax, vmin = max_min(image_3d, percentile=percentile)
    image_2d = np.sum(np.abs(image_3d), axis=2)
    return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)


def normalize_xrai(mask, percentile=99):
    vmax = np.percentile(mask, percentile)
    vmin = np.min(mask)
    return np.clip((mask - vmin) / (vmax - vmin), 0, 1)


def max_min(image_3d, percentile=99):
    image_2d = np.sum(np.abs(image_3d), axis=2)
    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)
    return vmax, vmin
