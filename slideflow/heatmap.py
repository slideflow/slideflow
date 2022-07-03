import os
import shutil
from collections import namedtuple
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Union)

import matplotlib.colors as mcol
import numpy as np
import shapely.geometry as sg
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

import slideflow as sf
from slideflow import errors
from slideflow.slide import WSI
from slideflow.util import Path
from slideflow.util import colors as col
from slideflow.util import log

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes


Inset = namedtuple("Inset", "x y zoom loc mark1 mark2 axes")


class Heatmap:
    """Generates heatmap by calculating predictions from a sliding scale
    window across a slide."""

    def __init__(
        self,
        slide: str,
        model: str,
        stride_div: int = 2,
        roi_dir: Optional[str] = None,
        rois: Optional[List[str]] = None,
        roi_method: str = 'auto',
        batch_size: int = 32,
        num_threads: Optional[int] = None,
        enable_downsample: bool = True,
        img_format: str = 'auto'
    ) -> None:
        """Convolutes across a whole slide, calculating logits and saving
        predictions internally for later use.

        Args:
            slide (str): Path to slide.
            model (str): Path to Tensorflow or PyTorch model.
            stride_div (int, optional): Divisor for stride when convoluting
                across slide. Defaults to 2.
            roi_dir (str, optional): Directory in which slide ROI is contained.
                Defaults to None.
            rois (list, optional): List of paths to slide ROIs. Alternative to
                providing roi_dir. Defaults to None.
            roi_method (str): Either 'inside', 'outside', 'auto', or 'ignore'.
                Determines how ROIs are used to extract tiles.
                If 'inside' or 'outside', will extract tiles in/out of an ROI,
                and raise errors.MissingROIError if an ROI is not available.
                If 'auto', will extract tiles inside an ROI if available,
                and across the whole-slide if no ROI is found.
                If 'ignore', will extract tiles across the whole-slide
                regardless of whether an ROI is available.
                Defaults to 'auto'.
            batch_size (int, optional): Batch size for calculating predictions.
                Defaults to 32.
            num_threads (int, optional): Number of tile worker threads.
                Defaults to CPU core count.
            enable_downsample (bool, optional): Enable the use of downsampled
                slide image layers. Defaults to True.
            img_format (str, optional): Image format (png, jpg) to use when
                extracting tiles from slide. Must match the image format
                the model was trained on. If 'auto', will use the format
                logged in the model params.json.
        """
        self.insets = []  # type: List[Inset]
        if (roi_dir is None and rois is None) and roi_method != 'ignore':
            log.info("No ROIs provided; will generate whole-slide heatmap")
            roi_method = 'ignore'

        model_config = sf.util.get_model_config(model)
        self.uq = model_config['hp']['uq']
        if img_format == 'auto' and 'img_format' not in model_config:
            raise errors.HeatmapError(
                f"Unable to auto-detect image format from model at {model}. "
                "Manually set to png or jpg with Heatmap(img_format=...)"
            )
        elif img_format == 'auto':
            img_format = model_config['img_format']
        if self.uq:
            interface = sf.model.UncertaintyInterface(model)  # type: Any
        else:
            interface = sf.model.Features(
                model,
                layers=None,
                include_logits=True
            )
        self.tile_px = model_config['tile_px']
        self.tile_um = model_config['tile_um']
        self.num_classes = interface.num_logits
        self.num_features = interface.num_features
        self.num_uncertainty = interface.num_uncertainty

        # Load the slide
        try:
            self.slide = WSI(
                slide,
                self.tile_px,
                self.tile_um,
                stride_div,
                enable_downsample=enable_downsample,
                roi_dir=roi_dir,
                rois=rois,
                roi_method=roi_method,
            )
        except errors.SlideLoadError:
            raise errors.HeatmapError(
                f'Error loading slide {self.slide.name} for heatmap'
            )
        out = interface(
            self.slide,
            num_threads=num_threads,
            batch_size=batch_size,
            img_format=img_format,
            dtype=np.float32
        )
        if self.uq:
            self.logits = out[:, :, :-(self.num_uncertainty)]
            self.uncertainty = out[:, :, -(self.num_uncertainty):]
        else:
            self.logits = out
            self.uncertainty = None
        log.info(f"Heatmap complete for {col.green(self.slide.name)}")

    @staticmethod
    def _prepare_ax(ax: Optional["Axes"] = None) -> "Axes":
        """Creates matplotlib figure and axis if one is not supplied,
        otherwise clears the axis contents.

        Args:
            ax (matplotlib.axes.Axes): Figure axis. If not supplied,
                will create a new figure and axis. Otherwise, clears axis
                contents. Defaults to None.

        Returns:
            matplotlib.axes.Axes: Figure axes.
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure(figsize=(18, 16))
            ax = fig.add_subplot(111)
            fig.subplots_adjust(bottom=0.25, top=0.95)
        else:
            ax.clear()
        return ax

    def _format_ax(
        self,
        ax: "Axes",
        thumb_size: Tuple[int, int],
        show_roi: bool = True,
        **kwargs
    ) -> None:
        """Formats matplotlib axis in preparation for heatmap plotting.

        Args:
            ax (matplotlib.axes.Axes): Figure axis.
            show_roi (bool, optional): Include ROI on heatmap. Defaults to True.
        """
        ax.tick_params(
            axis='x',
            top=True,
            labeltop=True,
            bottom=False,
            labelbottom=False
        )
        # Plot ROIs
        if show_roi:
            roi_scale = self.slide.dimensions[0] / thumb_size[0]
            annPolys = [
                sg.Polygon(annotation.scaled_area(roi_scale))
                for annotation in self.slide.rois
            ]
            for poly in annPolys:
                x, y = poly.exterior.xy
                ax.plot(x, y, zorder=20, **kwargs)

    def add_inset(
        self,
        x: Tuple[int, int],
        y: Tuple[int, int],
        zoom: int = 5,
        loc: int = 1,
        mark1: int = 2,
        mark2: int = 4,
        axes: bool = True
    ) -> Inset:
        """Adds a zoom inset to the heatmap."""
        _inset = Inset(
                x=x,
                y=y,
                zoom=zoom,
                loc=loc,
                mark1=mark1,
                mark2=mark2,
                axes=axes
        )
        self.insets += [_inset]
        return _inset

    def clear_insets(self) -> None:
        """Removes zoom insets."""
        self.insets = []

    def plot_thumbnail(
        self,
        show_roi: bool = False,
        roi_color: str = 'k',
        linewidth: int = 5,
        width: Optional[int] = None,
        mpp: Optional[float] = None,
        ax: Optional["Axes"] = None,
    ) -> "plt.image.AxesImage":
        """Plot a thumbnail of the slide, with or without ROI.

        Args:
            show_roi (bool, optional): Overlay ROIs onto heatmap image.
                Defaults to True.
            roi_color (str): ROI line color. Defaults to 'k' (black).
            linewidth (int): Width of ROI line. Defaults to 5.
            ax (matplotlib.axes.Axes, optional): Figure axis. If not supplied,
                will prepare a new figure axis.

        Returns:
            plt.image.AxesImage: Result from ax.imshow().
        """
        ax = self._prepare_ax(ax)
        if width is None and mpp is None:
            width = 2048
        thumb = self.slide.thumb(width=width, mpp=mpp)
        self._format_ax(
            ax,
            thumb_size=thumb.size,
            show_roi=show_roi,
            color=roi_color,
            linewidth=linewidth,
        )
        imshow_thumb = ax.imshow(thumb, zorder=0)

        for inset in self.insets:
            axins = zoomed_inset_axes(ax, inset.zoom, loc=inset.loc)
            axins.imshow(thumb)
            axins.set_xlim(inset.x[0], inset.x[1])
            axins.set_ylim(inset.y[0], inset.y[1])
            mark_inset(
                ax,
                axins,
                loc1=inset.mark1,
                loc2=inset.mark2,
                fc='none',
                ec='0',
                zorder=100
            )
            if not inset.axes:
                axins.get_xaxis().set_ticks([])
                axins.get_yaxis().set_ticks([])

        return imshow_thumb

    def plot_with_logit_cmap(
        self,
        logit_cmap: Union[Callable, Dict],
        interpolation: str = 'none',
        ax: Optional["Axes"] = None,
        **thumb_kwargs,
    ) -> None:
        """Plot a heatmap using a specified logit colormap.

        Args:
            logit_cmap (obj, optional): Either function or a dictionary use to
                create heatmap colormap. Each image tile will generate a list
                of predictions of length O, where O is the number of outcomes.
                If logit_cmap is a function, then the logit prediction list
                will be passed to the function, and the function is expected
                to return [R, G, B] values for display. If logit_cmap is a
                dictionary, it should map 'r', 'g', and 'b' to indices; the
                prediction for these outcome indices will be mapped to the RGB
                colors. Thus, the corresponding color will only reflect up to
                three outcomes. Example mapping prediction for outcome 0 to the
                red colorspace, 3 to green, etc: {'r': 0, 'g': 3, 'b': 1}
            interpolation (str, optional): Interpolation strategy to use for
                smoothing heatmap. Defaults to 'none'.
            ax (matplotlib.axes.Axes, optional): Figure axis. If not supplied,
                will prepare a new figure axis.

        Keyword args:
            show_roi (bool, optional): Overlay ROIs onto heatmap image.
                Defaults to True.
            roi_color (str): ROI line color. Defaults to 'k' (black).
            linewidth (int): Width of ROI line. Defaults to 5.
        """
        ax = self._prepare_ax(ax)
        implot = self.plot_thumbnail(ax=ax, **thumb_kwargs)
        ax.set_facecolor("black")
        if callable(logit_cmap):
            map_logit = logit_cmap
        else:
            # Make heatmap with specific logit predictions mapped
            # to r, g, and b
            def map_logit(logit):
                return (logit[logit_cmap['r']],
                        logit[logit_cmap['g']],
                        logit[logit_cmap['b']])
        ax.imshow(
            [[map_logit(logit) for logit in row] for row in self.logits],
            extent=implot.get_extent(),
            interpolation=interpolation,
            zorder=10
        )

    def plot_uncertainty(
        self,
        heatmap_alpha: float = 0.6,
        cmap: str = 'coowarm',
        interpolation: str = 'none',
        ax: Optional["Axes"] = None,
        **thumb_kwargs
    ):
        """Plot heatmap of uncertainty.

        Args:
            heatmap_alpha (float, optional): Alpha of heatmap overlay.
                Defaults to 0.6.
            cmap (str, optional): Matplotlib heatmap colormap.
                Defaults to 'coolwarm'.
            interpolation (str, optional): Interpolation strategy to use for
                smoothing heatmap. Defaults to 'none'.
            ax (matplotlib.axes.Axes, optional): Figure axis. If not supplied,
                will prepare a new figure axis.

        Keyword args:
            show_roi (bool, optional): Overlay ROIs onto heatmap image.
                Defaults to True.
            roi_color (str): ROI line color. Defaults to 'k' (black).
            linewidth (int): Width of ROI line. Defaults to 5.
        """
        ax = self._prepare_ax(ax)
        implot = self.plot_thumbnail(ax=ax, **thumb_kwargs)
        if heatmap_alpha == 1:
            implot.set_alpha(0)
        uqnorm = mcol.TwoSlopeNorm(
            vmin=0,
            vcenter=self.uncertainty.max()/2,
            vmax=self.uncertainty.max()
        )
        masked_uncertainty = np.ma.masked_where(
            self.uncertainty == -1,
            self.uncertainty
        )
        ax.imshow(
            masked_uncertainty,
            norm=uqnorm,
            extent=implot.get_extent(),
            cmap=cmap,
            alpha=heatmap_alpha,
            interpolation=interpolation,
            zorder=10
        )

    def plot(
        self,
        class_idx: int,
        heatmap_alpha: float = 0.6,
        cmap: str = 'coolwarm',
        interpolation: str = 'none',
        vmin: float = 0,
        vmax: float = 1,
        vcenter: float = 0.5,
        ax: Optional["Axes"] = None,
        **thumb_kwargs
    ) -> None:
        """Plot a predictive heatmap.

        Args:
            class_idx (int): Class index to plot.
            heatmap_alpha (float, optional): Alpha of heatmap overlay.
                Defaults to 0.6.
            show_roi (bool, optional): Overlay ROIs onto heatmap image.
                Defaults to True.
            cmap (str, optional): Matplotlib heatmap colormap.
                Defaults to 'coolwarm'.
            interpolation (str, optional): Interpolation strategy to use for
                smoothing heatmap. Defaults to 'none'.
            vmin (float): Minimimum value to display on heatmap.
                Defaults to 0.
            vcenter (float): Center value for color display on heatmap.
                Defaults to 0.5.
            vmax (float): Maximum value to display on heatmap.
                Defaults to 1.
            ax (matplotlib.axes.Axes, optional): Figure axis. If not supplied,
                will prepare a new figure axis.

        Keyword args:
            show_roi (bool, optional): Overlay ROIs onto heatmap image.
                Defaults to True.
            roi_color (str): ROI line color. Defaults to 'k' (black).
            linewidth (int): Width of ROI line. Defaults to 5.
        """

        ax = self._prepare_ax(ax)
        implot = self.plot_thumbnail(ax=ax, **thumb_kwargs)
        if heatmap_alpha == 1:
            implot.set_alpha(0)
        ax.set_facecolor("black")
        divnorm = mcol.TwoSlopeNorm(
            vmin=vmin,
            vcenter=vcenter,
            vmax=vmax
        )
        masked_arr = np.ma.masked_where(
            self.logits[:, :, class_idx] == -1,
            self.logits[:, :, class_idx]
        )
        ax.imshow(
            masked_arr,
            norm=divnorm,
            extent=implot.get_extent(),
            cmap=cmap,
            alpha=heatmap_alpha,
            interpolation=interpolation,
            zorder=10
        )

    def save(
        self,
        outdir: Path,
        show_roi: bool = True,
        interpolation: str = 'none',
        logit_cmap: Optional[Union[Callable, Dict]] = None,
        roi_color: str = 'k',
        linewidth: int = 5,
        **kwargs
    ) -> None:
        """Saves calculated logits as heatmap overlays.

        Args:
            outdir (str): Path to directory in which to save heatmap images.
            show_roi (bool, optional): Overlay ROIs onto heatmap image.
                Defaults to True.
            interpolation (str, optional): Interpolation strategy to use for
                smoothing heatmap. Defaults to 'none'.
            logit_cmap (obj, optional): Either function or a dictionary use to
                create heatmap colormap. Each image tile will generate a list
                of predictions of length O, where O is the number of outcomes.
                If logit_cmap is a function, then the logit prediction list
                will be passed to the function, and the function is expected
                to return [R, G, B] values for display. If logit_cmap is a
                dictionary, it should map 'r', 'g', and 'b' to indices; the
                prediction for these outcome indices will be mapped to the RGB
                colors. Thus, the corresponding color will only reflect up to
                three outcomes. Example mapping prediction for outcome 0 to the
                red colorspace, 3 to green, etc: {'r': 0, 'g': 3, 'b': 1}
            roi_color (str): ROI line color. Defaults to 'k' (black).
            linewidth (int): Width of ROI line. Defaults to 5.

        Keyword args:
            cmap (str, optional): Matplotlib heatmap colormap.
                Defaults to 'coolwarm'.
            vmin (float): Minimimum value to display on heatmap.
                Defaults to 0.
            vcenter (float): Center value for color display on heatmap.
                Defaults to 0.5.
            vmax (float): Maximum value to display on heatmap.
                Defaults to 1.

        """
        import matplotlib.pyplot as plt

        if self.logits is None:
            raise errors.HeatmapError("Logits not yet calculated.")

        def _savefig(label, bbox_inches='tight', **kwargs):
            plt.savefig(
                os.path.join(outdir, f'{self.slide.name}-{label}.png'),
                bbox_inches=bbox_inches,
                **kwargs
            )

        print('\r\033[KSaving base figures...', end='')

        # Prepare matplotlib figure
        ax = self._prepare_ax()

        thumb_kwargs = dict(roi_color=roi_color, linewidth=linewidth)

        # Save base thumbnail as separate figure
        self.plot_thumbnail(show_roi=False, ax=ax, **thumb_kwargs)
        _savefig('raw')

        # Save thumbnail + ROI as separate figure
        self.plot_thumbnail(show_roi=True, ax=ax, **thumb_kwargs)
        _savefig('raw+roi')

        if logit_cmap:
            self.plot_with_logit_cmap(logit_cmap, show_roi=show_roi, ax=ax)
            _savefig('custom')
        else:
            heatmap_kwargs = dict(
                show_roi=show_roi,
                interpolation=interpolation,
                **kwargs
            )
            save_kwargs = dict(
                bbox_inches='tight',
                facecolor=ax.get_facecolor(),
                edgecolor='none'
            )
            # Make heatmap plots and sliders for each outcome category
            for i in range(self.num_classes):
                print(f'\r\033[KMaking {i+1}/{self.num_classes}...', end='')
                self.plot(i, heatmap_alpha=0.6, ax=ax, **heatmap_kwargs)
                _savefig(str(i), **save_kwargs)

                self.plot(i, heatmap_alpha=1, ax=ax, **heatmap_kwargs)
                _savefig(f'{i}-solid', **save_kwargs)

            # Uncertainty map
            if self.uq:
                print('\r\033[KMaking uncertainty heatmap...', end='')
                self.plot_uncertainty(heatmap_alpha=0.6, ax=ax, **heatmap_kwargs)
                _savefig('UQ', **save_kwargs)

                self.plot_uncertainty(heatmap_alpha=1, ax=ax, **heatmap_kwargs)
                _savefig('UQ-solid', **save_kwargs)

        plt.close()
        print('\r\033[K', end='')
        log.info(f'Saved heatmaps for {col.green(self.slide.name)}')
