#from matplotlib.widgets import MultiCursor
from itertools import chain

import numpy as np
import saliency.core as saliency
import tensorflow as tf
from matplotlib import pyplot as plt
from slideflow.util import batch


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

    def grad_fn(self, image, call_model_args=None, expected_keys=None):
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

    def _calc_vanilla_map(self, img, mask_fn, **kwargs):
        mask_3d = mask_fn(img, self.grad_fn, **kwargs)
        return saliency.VisualizeImageGrayscale(mask_3d)

    def _calc_ig_map(self, img, mask_fn, x_steps=25, batch_size=20, **kwargs):
        mask_3d = mask_fn(
            img,
            self.grad_fn,
            x_baseline=np.zeros(img.shape),
            x_steps=x_steps,
            batch_size=batch_size
        )
        return saliency.VisualizeImageGrayscale(mask_3d)

    def _calc_guided_ig_map(self, img, mask_fn, x_steps=25, max_dist=1.0, fraction=0.5):
        mask_3d = mask_fn(
            img,
            self.grad_fn,
            x_baseline=np.zeros(img.shape),
            x_steps=x_steps,
            max_dist=max_dist,
            fraction=fraction
        )
        return saliency.VisualizeImageGrayscale(mask_3d)

    def _calc_blur_ig_map(self, img, mask_fn, batch_size=20):
        mask_3d = mask_fn(img, self.grad_fn, batch_size=batch_size)
        return saliency.VisualizeImageGrayscale(mask_3d)

    def all_maps(self, img):
        return {
            'Vanilla': self.vanilla_map(img),
            'Vanilla (Smoothed)': self.vanilla_map_smooth(img),
            'Integrated Gradients': self.integrated_gradients_map(img),
            'Integrated Gradients (Smooth)': self.integrated_gradients_map_smooth(img),
            'Guided Integrated Gradients': self.guided_integrated_gradients_map(img),
            'Guided Integrated Gradients (Smooth)': self.guided_integrated_gradients_map_smooth(img),
            'Blur Integrated Gradients': self.blur_integrated_gradients_map(img),
            'Blur Integrated Gradients (Smooth)': self.blur_integrated_gradients_map_smooth(img),
        }

    def vanilla_map(self, img, **kwargs):
        return self._calc_vanilla_map(img, self.gradients.GetMask, **kwargs)

    def vanilla_map_smooth(self, img, **kwargs):
        return self._calc_vanilla_map(img, self.gradients.GetSmoothedMask, **kwargs)

    def integrated_gradients_map(self, img, **kwargs):
        return self._calc_ig_map(img, self.ig.GetMask, **kwargs)

    def integrated_gradients_map_smooth(self, img, **kwargs):
        return self._calc_ig_map(img, self.ig.GetSmoothedMask, **kwargs)

    def guided_integrated_gradients_map(self, img, **kwargs):
        return self._calc_guided_ig_map(img, self.guided_ig.GetMask, **kwargs)

    def guided_integrated_gradients_map_smooth(self, img, **kwargs):
        return self._calc_guided_ig_map(img, self.guided_ig.GetSmoothedMask, **kwargs)

    def blur_integrated_gradients_map(self, img, **kwargs):
        return self._calc_blur_ig_map(img, self.blur_ig.GetMask, **kwargs)

    def blur_integrated_gradients_map_smooth(self, img, **kwargs):
        return self._calc_blur_ig_map(img, self.blur_ig.GetSmoothedMask, **kwargs)

    def xrai_map(self, img, batch_size=20, **kwargs):
        return self.xrai.GetMask(img, self.grad_fn, batch_size=batch_size, **kwargs)

    def xrai_map_fast(self, img, batch_size=20, **kwargs):
        return self.xrai.GetMask(
            img,
            self.grad_fn,
            batch_size=batch_size,
            extra_parameters=self.fast_xrai_params,
            **kwargs)


def comparison_plot(original, maps, cmap=plt.cm.gray):
    n_rows = 3
    n_cols = 3
    scale = 5
    ax_idx = [[i, j] for i in range(n_rows) for j in range(n_cols)]
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_rows * scale, n_cols * scale))
    #multi = MultiCursor(fig.canvas, list(chain.from_iterable(ax)), color='r', lw=1)

    ax[ax_idx[0][0], ax_idx[0][1]].axis('off')
    ax[ax_idx[0][0], ax_idx[0][1]].imshow(original)
    ax[ax_idx[0][0], ax_idx[0][1]].set_title('Original')

    for i, (map_name, map_img) in enumerate(maps.items()):
        ax[ax_idx[i+1][0], ax_idx[i+1][1]].axis('off')
        ax[ax_idx[i+1][0], ax_idx[i+1][1]].imshow(map_img, cmap=cmap, vmin=0, vmax=1)
        ax[ax_idx[i+1][0], ax_idx[i+1][1]].set_title(map_name)

    fig.subplots_adjust(wspace=0, hspace=0)

def multi_plot(originals, maps, method='adjacent', n_rows=4, cmap='inferno'):
    assert method in ('adjacent', 'overlay')
    orig_batch = list(batch(originals, n_rows))
    maps_batch = list(batch(maps, n_rows))
    n_cols = len(orig_batch) * 4
    scale = 5

    gridspec = dict(hspace=0.0, width_ratios=[1, 1, 1, 0.2]*len(orig_batch))
    figsize = (n_cols * scale * 0.79, n_rows * scale)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, gridspec_kw=gridspec)

    def remove_ticks(axis):
        axis.spines.right.set_visible(False)
        axis.spines.top.set_visible(False)
        axis.spines.left.set_visible(False)
        axis.spines.bottom.set_visible(False)
        axis.set_xticklabels([])
        axis.set_xticks([])
        axis.set_yticklabels([])
        axis.set_yticks([])

    for c in range(len(orig_batch)):
        for r in range(len(orig_batch[c])):
            _c = c * 4

            ax[r, _c+3].set_visible(False)

            remove_ticks(ax[r, _c])
            ax[r, _c].set_ylabel(f"{(c*4)+r}")
            ax[r, _c].imshow(orig_batch[c][r])
            ax[r, _c].set_aspect('equal')
            ax

            ax[r, _c+1].axis('off')
            ax[r, _c+1].imshow(orig_batch[c][r])
            ax[r, _c+1].set_aspect('equal')

            ax[r, _c+1].axis('off')
            ax[r, _c+1].imshow(maps_batch[c][r], cmap='Oranges', vmin=0, vmax=1, alpha=0.6)
            ax[r, _c+1].set_aspect('equal')

            ax[r, _c+2].axis('off')
            ax[r, _c+2].imshow(maps_batch[c][r], cmap=cmap, vmin=0, vmax=1)
            ax[r, _c+2].set_aspect('equal')

            if r == 0:
                ax[r, _c].set_title("Original")
                ax[r, _c+1].set_title("Overlay")
                ax[r, _c+2].set_title("Saliency Map")

    fig.subplots_adjust(wspace=0, hspace=0)
