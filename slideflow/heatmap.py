import os
import shutil
import slideflow as sf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import shapely.geometry as sg

from slideflow.util import log
from slideflow.util.fastim import FastImshow
from matplotlib.widgets import Slider

class HeatmapError(Exception):
    pass

class Heatmap:
    """Generates heatmap by calculating predictions from a sliding scale window across a slide."""

    def __init__(self, slide, model, stride_div=2, roi_dir=None, rois=None, roi_method='inside', batch_size=32,
                 num_threads=8, buffer=None, enable_downsample=True):

        """Convolutes across a whole slide, calculating logits and saving predictions internally for later use.

        Args:
            slide (str): Path to slide.
            model (str): Path to Tensorflow or PyTorch model.
            stride_div (int, optional): Divisor for stride when convoluting across slide. Defaults to 2.
            roi_dir (str, optional): Directory in which slide ROI is contained. Defaults to None.
            rois (list, optional): List of paths to slide ROIs. Defaults to None. Alternative to providing roi_dir.
            roi_method (str, optional): Either 'inside', 'outside', or 'ignore'. Defaults to 'inside'.
                If inside, tiles will be extracted inside ROI region.
                If outside, tiles will be extracted outside ROI region.
            batch_size (int, optional): Batch size when calculating predictions. Defaults to 32.
            num_threads (int, optional): Number of tile extraction worker threads. Defaults to 8.
            buffer (str, optional): Either 'vmtouch' or path to directory to use for buffering slides. Defaults to None.
                Significantly improves performance for slides on HDDs.
            enable_downsample (bool, optional): Enable the use of downsampled slide image layers. Defaults to True.
        """

        from slideflow.slide import WSI

        self.logits = None
        if (roi_dir is None and rois is None) and roi_method != 'ignore':
            log.info("No ROIs provided; will generate whole-slide heatmap")
            roi_method = 'ignore'

        interface = sf.model.Features(model, layers=None, include_logits=True)
        model_config = sf.util.get_model_config(model)
        self.tile_px = model_config['tile_px']
        self.tile_um = model_config['tile_um']
        self.num_classes = interface.num_logits
        self.num_features = interface.num_features

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
                         buffer=buffer,
                         skip_missing_roi=False)

        if not self.slide.loaded_correctly():
            raise HeatmapError(f'Unable to load slide {self.slide.name} for heatmap generation')

        self.logits = interface(self.slide,
                                normalizer=model_config['hp']['normalizer'],
                                normalizer_source=model_config['hp']['normalizer_source'],
                                num_threads=num_threads,
                                dtype=np.float32)

        log.info(f"Heatmap complete for {sf.util.green(self.slide.name)}")

        if buffered_slide:
            os.remove(new_path)

    def _prepare_figure(self, show_roi=True):
        self.fig = plt.figure(figsize=(18, 16))
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(bottom = 0.25, top=0.95)
        gca = plt.gca()
        gca.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)
        # Plot ROIs
        if show_roi:
            print('\r\033[KPlotting ROIs...', end='')
            roi_scale = self.slide.full_shape[0]/2048
            annPolys = [sg.Polygon(annotation.scaled_area(roi_scale)) for annotation in self.slide.rois]
            for poly in annPolys:
                x,y = poly.exterior.xy
                plt.plot(x, y, zorder=20, color='k', linewidth=5)

    def display(self, show_roi=True, interpolation='none', logit_cmap=None):
        """Interactively displays calculated logits as a heatmap.

        Args:
            show_roi (bool, optional): Overlay ROIs onto heatmap image. Defaults to True.
            interpolation (str, optional): Interpolation strategy to use for smoothing heatmap. Defaults to 'none'.
            logit_cmap (obj, optional): Either function or a dictionary use to create heatmap colormap.
                Each image tile will generate a list of predictions of length O, where O is the number of outcomes.
                If logit_cmap is a function, then the logit prediction list will be passed to the function,
                and the function is expected to return [R, G, B] values which will be displayed.
                If the logit_cmap is a dictionary, it should map 'r', 'g', and 'b' to indices; the prediction for
                these outcome indices will be mapped to the RGB colors. Thus, the corresponding color will only
                reflect up to three outcomes. Example mapping prediction for outcome 0 to the red colorspace, 3
                to green, etc: {'r': 0, 'g': 3, 'b': 1}
        """

        self._prepare_figure(show_roi=False)
        heatmap_dict = {}

        thumb = self.slide.thumb(rois=show_roi)
        implot = FastImshow(thumb, self.ax, extent=None, tgt_res=1024)

        def slider_func(val):
            for h, s in heatmap_dict.values():
                h.set_alpha(s.val)

        if logit_cmap:
            if callable(logit_cmap):
                map_logit = logit_cmap
            else:
                def map_logit(l):
                    # Make heatmap with specific logit predictions mapped to r, g, and b
                    return (l[logit_cmap['r']], l[logit_cmap['g']], l[logit_cmap['b']])
            heatmap = self.ax.imshow([[map_logit(l) for l in row] for row in self.logits],
                                     extent=implot.extent,
                                     interpolation=interpolation,
                                     zorder=10)
        else:
            divnorm = mcol.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1.0)
            for i in range(self.num_classes):
                heatmap = self.ax.imshow(self.logits[:, :, i],
                                         extent=implot.extent,
                                         cmap='coolwarm',
                                         norm=divnorm,
                                         alpha = 0.0,
                                         interpolation=interpolation,
                                         zorder=10) #bicubic

                ax_slider = self.fig.add_axes([0.25, 0.2-(0.2/self.num_classes)*i, 0.5, 0.03], facecolor='lightgoldenrodyellow')
                slider = Slider(ax_slider, f'Class {i}', 0, 1, valinit = 0)
                heatmap_dict.update({f'Class{i}': [heatmap, slider]})
                slider.on_changed(slider_func)

        self.fig.canvas.set_window_title(self.slide.name)
        implot.show()
        plt.show()

    def save(self, outdir, show_roi=True, interpolation='none', logit_cmap=None, vmin=0, vmax=1, vcenter=0.5):
        """Saves calculated logits as heatmap overlays.

        Args:
            outdir (str): Path to directory in which to save heatmap images.
            show_roi (bool, optional): Overlay ROIs onto heatmap image. Defaults to True.
            interpolation (str, optional): Interpolation strategy to use for smoothing heatmap. Defaults to 'none'.
            logit_cmap (obj, optional): Either function or a dictionary use to create heatmap colormap.
                Each image tile will generate a list of predictions of length O, where O is the number of outcomes.
                If logit_cmap is a function, then the logit prediction list will be passed to the function,
                and the function is expected to return [R, G, B] values which will be displayed.
                If the logit_cmap is a dictionary, it should map 'r', 'g', and 'b' to indices; the prediction for
                these outcome indices will be mapped to the RGB colors. Thus, the corresponding color will only
                reflect up to three outcomes. Example mapping prediction for outcome 0 to the red colorspace, 3
                to green, etc: {'r': 0, 'g': 3, 'b': 1}
            vmin (float): Minimimum value to display on heatmap. Defaults to 0.
            vcenter (float): Center value for color display on heatmap. Defaults to 0.5.
            vmax (float): Maximum value to display on heatmap. Defaults to 1.
        """

        print('\r\033[KSaving base figures...', end='')

        # Save base thumbnail as separate figure
        self._prepare_figure(show_roi=False)
        self.ax.imshow(self.slide.thumb(width=2048), zorder=0)
        plt.savefig(os.path.join(outdir, f'{self.slide.name}-raw.png'), bbox_inches='tight')
        plt.clf()

        # Save thumbnail + ROI as separate figure
        self._prepare_figure(show_roi=False)
        self.ax.imshow(self.slide.thumb(width=2048, rois=True), zorder=0)
        plt.savefig(os.path.join(outdir, f'{self.slide.name}-raw+roi.png'), bbox_inches='tight')
        plt.clf()

        # Now prepare base image for the the heatmap overlay
        self._prepare_figure(show_roi=False)
        implot = self.ax.imshow(self.slide.thumb(width=2048, rois=show_roi), zorder=0)

        if logit_cmap:
            if callable(logit_cmap):
                map_logit = logit_cmap
            else:
                def map_logit(l):
                    # Make heatmap with specific logit predictions mapped to r, g, and b
                    return (l[logit_cmap['r']], l[logit_cmap['g']], l[logit_cmap['b']])

            heatmap = self.ax.imshow([[map_logit(l) for l in row] for row in self.logits],
                                     extent=implot.get_extent(),
                                     interpolation=interpolation,
                                     zorder=10)

            plt.savefig(os.path.join(outdir, f'{self.slide.name}-custom.png'), bbox_inches='tight')
        else:
            # Make heatmap plots and sliders for each outcome category
            for i in range(self.num_classes):
                print(f'\r\033[KMaking heatmap {i+1} of {self.num_classes}...', end='')
                divnorm = mcol.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                heatmap = self.ax.imshow(self.logits[:, :, i],
                                         extent=implot.get_extent(),
                                         cmap='coolwarm',
                                         norm=divnorm,
                                         alpha=0.6,
                                         interpolation=interpolation, #bicubic
                                         zorder=10)
                plt.savefig(os.path.join(outdir, f'{self.slide.name}-{i}.png'), bbox_inches='tight')
                heatmap.set_alpha(1)
                plt.savefig(os.path.join(outdir, f'{self.slide.name}-{i}-solid.png'), bbox_inches='tight')
                heatmap.remove()

        plt.close()
        print('\r\033[K', end='')
        log.info(f'Saved heatmaps for {sf.util.green(self.slide.name)}')
