import os
import shutil
import slideflow as sf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import shapely.geometry as sg

from slideflow import errors
from slideflow.util import log
from slideflow.util import colors as col
from slideflow.slide import WSI


class Heatmap:
    """Generates heatmap by calculating predictions from a sliding scale
    window across a slide."""

    def __init__(self, slide, model, stride_div=2, roi_dir=None, rois=None,
                 roi_method='inside', batch_size=32, num_threads=None,
                 buffer=None, enable_downsample=True, img_format='auto'):
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
            roi_method (str, optional): 'inside', 'outside', or 'ignore'.
                If inside, tiles will be extracted inside ROI region.
                If outside, tiles will be extracted outside ROI region.
                Defaults to 'inside'.
            batch_size (int, optional): Batch size for calculating predictions.
                Defaults to 32.
            num_threads (int, optional): Number of tile worker threads.
                Defaults to CPU core count.
            buffer (str, optional): Path to use for buffering slides.
                Defaults to None.
            enable_downsample (bool, optional): Enable the use of downsampled
                slide image layers. Defaults to True.
            img_format (str, optional): Image format (png, jpg) to use when
                extracting tiles from slide. Must match the image format
                the model was trained on. If 'auto', will use the format
                logged in the model params.json.
        """

        self.logits = None
        if (roi_dir is None and rois is None) and roi_method != 'ignore':
            log.info("No ROIs provided; will generate whole-slide heatmap")
            roi_method = 'ignore'

        model_config = sf.util.get_model_config(model)
        self.uq = model_config['hp']['uq']
        if img_format == 'auto' and 'img_format' not in model_config:
            msg = f"Unable to auto-detect image format from model at {model}. "
            msg += "Manually set to png or jpg with Heatmap(img_format=...)"
            raise errors.HeatmapError(msg)
        elif img_format == 'auto':
            img_format = model_config['img_format']
        if self.uq:
            interface = sf.model.tensorflow.UncertaintyInterface(model)
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

        # Create slide buffer
        if buffer and os.path.isdir(buffer):
            new_path = os.path.join(buffer, os.path.basename(slide))
            shutil.copy(slide, new_path)
            slide = new_path
            buffered_slide = True
        else:
            buffered_slide = False

        # Load the slide
        self.slide = WSI(slide,
                         self.tile_px,
                         self.tile_um,
                         stride_div,
                         enable_downsample=enable_downsample,
                         roi_dir=roi_dir,
                         rois=rois,
                         roi_method=roi_method,
                         skip_missing_roi=False)

        if not self.slide.loaded_correctly():
            msg = f'Unable to load slide {self.slide.name} for heatmap'
            raise errors.HeatmapError(msg)

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
        if buffered_slide:
            os.remove(new_path)

    def _prepare_figure(self, show_roi=True):
        self.fig = plt.figure(figsize=(18, 16))
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(bottom=0.25, top=0.95)
        gca = plt.gca()
        gca.tick_params(
            axis='x',
            top=True,
            labeltop=True,
            bottom=False,
            labelbottom=False
        )
        # Plot ROIs
        if show_roi:
            print('\r\033[KPlotting ROIs...', end='')
            roi_scale = self.slide.full_shape[0] / 2048
            annPolys = [
                sg.Polygon(annotation.scaled_area(roi_scale))
                for annotation in self.slide.rois
            ]
            for poly in annPolys:
                x, y = poly.exterior.xy
                plt.plot(x, y, zorder=20, color='k', linewidth=5)

    def save(self, outdir, show_roi=True, interpolation='none',
             cmap='coolwarm', logit_cmap=None, vmin=0, vmax=1, vcenter=0.5):
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
            vmin (float): Minimimum value to display on heatmap.
                Defaults to 0.
            vcenter (float): Center value for color display on heatmap.
                Defaults to 0.5.
            vmax (float): Maximum value to display on heatmap.
                Defaults to 1.
        """

        print('\r\033[KSaving base figures...', end='')
        # Save base thumbnail as separate figure
        self._prepare_figure(show_roi=False)
        self.ax.imshow(self.slide.thumb(width=2048), zorder=0)
        plt.savefig(
            os.path.join(outdir, f'{self.slide.name}-raw.png'),
            bbox_inches='tight'
        )
        plt.clf()
        # Save thumbnail + ROI as separate figure
        self._prepare_figure(show_roi=False)
        self.ax.imshow(self.slide.thumb(width=2048, rois=True), zorder=0)
        plt.savefig(
            os.path.join(outdir, f'{self.slide.name}-raw+roi.png'),
            bbox_inches='tight'
        )
        plt.clf()
        # Now prepare base image for the the heatmap overlay
        self._prepare_figure(show_roi=False)
        implot = self.ax.imshow(
            self.slide.thumb(width=2048, rois=show_roi),
            zorder=0
        )
        self.ax.set_facecolor("black")
        if logit_cmap:
            if callable(logit_cmap):
                map_logit = logit_cmap
            else:
                # Make heatmap with specific logit predictions mapped
                # to r, g, and b
                def map_logit(logit):
                    return (logit[logit_cmap['r']],
                            logit[logit_cmap['g']],
                            logit[logit_cmap['b']])

            heatmap = self.ax.imshow(
                [[map_logit(logit) for logit in row] for row in self.logits],
                extent=implot.get_extent(),
                interpolation=interpolation,
                zorder=10
            )
            plt.savefig(
                os.path.join(outdir, f'{self.slide.name}-custom.png'),
                bbox_inches='tight'
            )
        else:
            heatmap_kwargs = {
                'extent': implot.get_extent(),
                'cmap': cmap,
                'alpha': 0.6,
                'interpolation': interpolation,
                'zorder': 10
            }
            save_kwargs = {
                'bbox_inches': 'tight',
                'facecolor': self.ax.get_facecolor(),
                'edgecolor': 'none'
            }
            # Make heatmap plots and sliders for each outcome category
            for i in range(self.num_classes):
                print(f'\r\033[KMaking {i+1}/{self.num_classes}...', end='')
                divnorm = mcol.TwoSlopeNorm(
                    vmin=vmin,
                    vcenter=vcenter,
                    vmax=vmax
                )
                masked_arr = np.ma.masked_where(
                    self.logits[:, :, i] == -1,
                    self.logits[:, :, i]
                )
                heatmap = self.ax.imshow(
                    masked_arr,
                    norm=divnorm,
                    **heatmap_kwargs
                )
                plt.savefig(
                    os.path.join(outdir, f'{self.slide.name}-{i}.png'),
                    **save_kwargs
                )
                heatmap.set_alpha(1)
                implot.set_alpha(0)
                plt.savefig(
                    os.path.join(outdir, f'{self.slide.name}-{i}-solid.png'),
                    **save_kwargs
                )
                heatmap.remove()
                implot.set_alpha(1)

            # Uncertainty map
            if self.uq:
                print('\r\033[KMaking uncertainty heatmap...', end='')
                uqnorm = mcol.TwoSlopeNorm(
                    vmin=0,
                    vcenter=self.uncertainty.max()/2,
                    vmax=self.uncertainty.max()
                )
                masked_uncertainty = np.ma.masked_where(
                    self.uncertainty == -1,
                    self.uncertainty
                )
                heatmap = self.ax.imshow(
                    masked_uncertainty,
                    norm=uqnorm,
                    **heatmap_kwargs
                )
                plt.savefig(
                    os.path.join(outdir, f'{self.slide.name}-UQ.png'),
                    **save_kwargs
                )
                heatmap.set_alpha(1)
                implot.set_alpha(0)
                plt.savefig(
                    os.path.join(outdir, f'{self.slide.name}-UQ-solid.png'),
                    **save_kwargs
                )
                heatmap.remove()
        plt.close()
        print('\r\033[K', end='')
        log.info(f'Saved heatmaps for {col.green(self.slide.name)}')
