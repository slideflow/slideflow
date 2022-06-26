#from matplotlib.widgets import MultiCursor
from typing import Callable, Iterable, List, Optional

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from slideflow.util import batch


def inferno(img):
    cmap = plt.get_cmap('inferno')
    return (cmap(img) * 255).astype(np.uint8)


def oranges(img):
    cmap = plt.get_cmap('Oranges')
    return (cmap(img) * 255).astype(np.uint8)


def overlay(image, mask):
    base = Image.fromarray(image)
    cmap = Image.fromarray(oranges(mask))
    cmap.putalpha(int(0.6*255))
    base.paste(cmap, mask=cmap)
    return np.array(base)


def remove_ticks(axis):
    axis.spines.right.set_visible(False)
    axis.spines.top.set_visible(False)
    axis.spines.left.set_visible(False)
    axis.spines.bottom.set_visible(False)
    axis.set_xticklabels([])
    axis.set_xticks([])
    axis.set_yticklabels([])
    axis.set_yticks([])


def comparison_plot(original, maps, cmap=plt.cm.gray):
    n_rows = 3
    n_cols = 3
    scale = 5
    ax_idx = [[i, j] for i in range(n_rows) for j in range(n_cols)]
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_rows * scale, n_cols * scale))

    ax[ax_idx[0][0], ax_idx[0][1]].axis('off')
    ax[ax_idx[0][0], ax_idx[0][1]].imshow(original)
    ax[ax_idx[0][0], ax_idx[0][1]].set_title('Original')

    for i, (map_name, map_img) in enumerate(maps.items()):
        ax[ax_idx[i+1][0], ax_idx[i+1][1]].axis('off')
        ax[ax_idx[i+1][0], ax_idx[i+1][1]].imshow(map_img, cmap=cmap, vmin=0, vmax=1)
        ax[ax_idx[i+1][0], ax_idx[i+1][1]].set_title(map_name)

    fig.subplots_adjust(wspace=0, hspace=0)


def multi_plot(
    raw_imgs: Iterable[np.ndarray],
    processed_imgs: Iterable[np.ndarray],
    method: Callable,
    cmap: str = 'inferno',
    xlabels: Optional[List[str]] = None,
    ylabels: Optional[List[str]]  =None,
    **kwargs
) -> None:
    """Creates a plot of saliency maps and overlays for a given set of images.

    The first row will be the raw images.
    The second row will be an overlay of the saliency map and the raw image.
    The third row will be the saliency maps.

    Args:
        raw_imgs (Iterable[np.ndarray]): Raw, unprocessed images.
        processed_imgs (Iterable[np.ndarray]): _description_
        method (Callable): _description_
        cmap (str, optional): _description_. Defaults to 'inferno'.
        xlabels (Optional[List[str]], optional): _description_. Defaults to None.
        ylabels (Optional[List[str]], optional): _description_. Defaults to None.

    Raises:
        ValueError: If length of raw_imgs, processed_imgs are not equal.
        ValueError: If xlabels is provided and not a list.
        ValueError: If ylabels is provided and not a list.
        ValueError: If xlabels is provided and length does not equal raw_imgs.
        ValueError: If ylabels is provided and length does not equal raw_imgs.
    """

    # Error checking
    if len(raw_imgs) != len(processed_imgs):
        raise ValueError(
            "Length of raw_imgs ({}) and processed_imgs ({}) unequal".format(
                len(raw_imgs),
                len(processed_imgs)
            )
        )
    if xlabels:
        if not isinstance(xlabels, list):
            raise ValueError("xlabels must be a list.")
        if len(xlabels) != len(raw_imgs):
            raise ValueError(
                "Length of raw_imgs ({}) and xlabels ({}) unequal".format(
                    len(raw_imgs),
                    len(xlabels)
                )
            )
    if ylabels:
        if not isinstance(ylabels, list):
            raise ValueError("ylabels must be a list of length 3.")
        if len(ylabels) != 3:
            raise ValueError(
                f"Unexpected length for ylabels; expected 3, got {len(ylabels)}"
            )

    # Calculate masks ans overlays
    masks = [method(p_img, **kwargs) for p_img in processed_imgs]
    overlays = [overlay(img, mask) for img, mask in zip(raw_imgs, masks)]

    # Initialize figure.
    figsize = (len(raw_imgs)*5, 15)
    fig, ax = plt.subplots(3, len(raw_imgs), figsize=figsize)

    # Plot labels if provided.
    if xlabels:
        for i in range(len(xlabels)):
            ax[0, i].set_title(xlabels[i], fontsize=22)
    if ylabels:
        for i in range(len(ylabels)):
            ax[i, 0].set_ylabel(ylabels[i], fontsize=22)

    # Plot the originals, overlays, and masks
    for i, img in enumerate(raw_imgs):
        remove_ticks(ax[0, i])
        ax[0, i].imshow(Image.fromarray(img))
    for i, ov in enumerate(overlays):
        remove_ticks(ax[1, i])
        ax[1, i].imshow(Image.fromarray(ov))
    for i, mask in enumerate(masks):
        remove_ticks(ax[2, i])
        ax[2, i].imshow(mask, cmap=cmap)

    fig.subplots_adjust(wspace=0, hspace=0)


def saliency_map_comparison(orig_imgs, saliency_fn, process_fn, saliency_labels=None, cmap='inferno', **kwargs):

    def apply_cmap(_img):
        cmap_fn = plt.get_cmap(cmap)
        return (cmap_fn(_img) * 255).astype(np.uint8)

    n_imgs = len(orig_imgs)
    n_saliency = len(saliency_fn)
    fig, ax = plt.subplots(n_imgs, n_saliency+1, figsize=((n_saliency+1)*5, n_imgs*5))

    if saliency_labels is None:
        saliency_labels = [f"Saliency{n}" for n in range(n_saliency)]
    assert len(saliency_labels) == len(saliency_fn)

    ax[0, 0].set_title("Original")
    for idx, orig in enumerate(orig_imgs):
        ax[idx, 0].axis('off')
        ax[idx, 0].imshow(orig)
        for s, s_fn in enumerate(saliency_fn):
            ax[0, s+1].set_title(saliency_labels[s])
            ax[idx, s+1].axis('off')
            ax[idx, s+1].imshow(apply_cmap(s_fn(process_fn(orig), **kwargs)))

    fig.subplots_adjust(wspace=0, hspace=0)