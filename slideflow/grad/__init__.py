from typing import Dict, Optional

import numpy as np
import saliency.core as saliency
import tensorflow as tf
from slideflow.grad.plot_utils import (comparison_plot, inferno, multi_plot,
                                       oranges, overlay,
                                       saliency_map_comparison)


class SaliencyMap:

    def __init__(self, model, class_idx):
        self.model = model
        self.class_idx = class_idx
        self.gradients = saliency.GradientSaliency()
        self.ig = saliency.IntegratedGradients()
        self.guided_ig = saliency.GuidedIG()
        self.blur_ig = saliency.BlurIG()
        self.xrai = saliency.XRAI()
        self.fast_xrai_params = saliency.XRAIParameters()
        self.fast_xrai_params.algorithm = 'fast'
        self.masks = {}

    def _grad_fn(self, image, call_model_args=None, expected_keys=None):
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
                raise ValueError

    def _apply_mask_fn(
        self,
        img: np.ndarray,
        grads: saliency.CoreSaliency,
        baseline: bool  =False,
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
            return mask_fn(_img, self._grad_fn, **kwargs)

        if isinstance(img, list):
            # Normalize together
            image_3d = [_get_mask(_img) for _img in img]
            v_maxes, v_mins = zip(*[max_min(img3d) for img3d in image_3d])
            vmax = max(v_maxes)
            vmin = min(v_mins)
            return [grayscale(img3d, vmax=vmax, vmin=vmin) for img3d in image_3d]
        else:
            return grayscale(_get_mask(img))

    def all(self, img: np.ndarray) -> Dict:
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

    def vanilla(
        self,
        img: np.ndarray,
        smooth: bool = False,
        **kwargs
    ) -> np.ndarray:
        return self._apply_mask_fn(
            img,
            self.gradients,
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
        return self.xrai.GetMask(
            img,
            self._grad_fn,
            batch_size=batch_size,
            **kwargs
        )

    def xrai_fast(
        self,
        img: np.ndarray,
        batch_size: int = 20,
        **kwargs
    ) -> np.ndarray:
        return self.xrai.GetMask(
            img,
            self._grad_fn,
            batch_size=batch_size,
            extra_parameters=self.fast_xrai_params,
            **kwargs
        )


def grayscale(image_3d, vmax=None, vmin=None, percentile=99):
    """Returns a 3D tensor as a grayscale 2D tensor.
    This method sums a 3D tensor across the absolute value of axis=2, and then
    clips values at a given percentile.
    """
    if vmax is None and vmin is None:
        vmax, vmin = max_min(image_3d, percentile=percentile)
    image_2d = np.sum(np.abs(image_3d), axis=2)
    return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)


def max_min(image_3d, percentile=99):
    image_2d = np.sum(np.abs(image_3d), axis=2)
    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)
    return vmax, vmin
