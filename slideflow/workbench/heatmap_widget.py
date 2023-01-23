import os
import numpy as np
import imgui
import threading
from array import array
from .gui_utils import imgui_utils

import slideflow as sf

#----------------------------------------------------------------------------

def _apply_cmap(img, cmap):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap)
    return (cmap(img) * 255).astype(np.uint8)

def process_grid(_heatmap, _grid):
    if _heatmap.uq:
        logits = _grid[:, :, :-(_heatmap.num_uncertainty)]
        uncertainty = _grid[:, :, -(_heatmap.num_uncertainty):]
        return logits, uncertainty
    else:
        return _grid, None

#----------------------------------------------------------------------------

class HeatmapWidget:
    def __init__(self, viz):
        import matplotlib.pyplot as plt

        self.viz                    = viz
        self.alpha                  = 0.5
        self.gain                   = 1.0
        self.show                   = True
        self.heatmap_logits         = 0
        self.heatmap_uncertainty    = 0
        self.use_logits             = True
        self.use_uncertainty        = False
        self.cmap_idx               = 0
        self.logits                 = None
        self.uncertainty            = None
        self.content_height         = 0
        self._generating            = False
        self._button_pressed        = False
        self._old_logits            = 0
        self._old_uncertainty       = 0
        self._logits_gain           = 1.0
        self._uq_gain               = 1.0
        self._heatmap_sum           = 0
        self._heatmap_grid          = None
        self._heatmap_thread        = None
        self._colormaps             = plt.colormaps()
        self._rendering_message     = "Calculating heatmap..."

    def _create_heatmap(self):
        viz = self.viz
        self.reset()
        if viz.low_memory or sf.slide_backend() == 'cucim':
            mp_kw = dict(num_threads=os.cpu_count())
        else:
            mp_kw = dict(num_processes=os.cpu_count())
        viz.heatmap = sf.heatmap.ModelHeatmap(
            viz.wsi,
            viz.model,
            img_format=viz._model_config['img_format'],
            generate=False,
            normalizer=viz._normalizer,
            uq=viz.has_uq(),
            **mp_kw
        )

    def drag_and_drop(self, path, ignore_errors=True):
        if path.endswith('npz'):
            try:
                self.load(path)
            except Exception:
                sf.log.debug(f"Unable to load {path} as heatmap.")
                if not ignore_errors:
                    raise

    def generate_heatmap(self):
        """Create and generate a heatmap asynchronously."""

        sw = self.viz.slide_widget
        self._create_heatmap()
        self._button_pressed = True
        self._generating = True
        self._heatmap_grid, self._heatmap_thread = self.viz.heatmap.generate(
            asynchronous=True,
            grayspace_fraction=sw.gs_fraction,
            grayspace_threshold=sw.gs_threshold,
            whitespace_fraction=sw.ws_fraction,
            whitespace_threshold=sw.ws_threshold,
        )

    def load(self, path):
        """Load a heatmap from a saved *.npz file."""

        if self.viz.heatmap is None:
            self._create_heatmap()
        self.viz.heatmap.load(path)
        self.logits = self.viz.heatmap.logits
        self.uncertainty = self.viz.heatmap.uncertainty
        self.render_heatmap()

    def refresh_generating_heatmap(self):
        """Refresh render of the asynchronously generating heatmap."""

        if self.viz.heatmap is not None and self._heatmap_grid is not None:
            logits, uncertainty = process_grid(self.viz.heatmap, self._heatmap_grid)
            _sum = np.sum(logits)
            if _sum != self._heatmap_sum:
                self.logits = logits
                self.uncertainty = uncertainty
                self.render_heatmap()
                self._heatmap_sum = _sum

        if self._heatmap_thread is not None and not self._heatmap_thread.is_alive():
            self._generating = False
            self._button_pressed = False
            self._heatmap_thread = None
            self.viz.clear_message(self._rendering_message)
            self.viz.create_toast("Heatmap complete.", icon="success")

    def render_heatmap(self):
        """Render the current heatmap."""

        self._old_logits = self.heatmap_logits
        self._old_uncertainty = self.heatmap_uncertainty
        if self.viz.heatmap is None:
            return
        if self.use_logits:
            heatmap_arr = self.logits[:, :, self.heatmap_logits]
            gain = self._logits_gain
        else:
            heatmap_arr = self.uncertainty[:, :, self.heatmap_uncertainty]
            gain = self._uq_gain
        self.viz.rendered_heatmap = _apply_cmap(heatmap_arr * gain, self._colormaps[self.cmap_idx])[:, :, 0:3]
        self.update_transparency()

    def reset(self):
        """Reset the heatmap display."""

        self.viz.rendered_heatmap   = None
        self.viz.overlay            = None
        self.viz.heatmap            = None
        self._heatmap_sum           = 0
        self._heatmap_grid          = None
        self._heatmap_thread        = None
        self._old_logits            = 0
        self._old_uncertainty       = 0
        self.logits                 = None
        self.uncertainty            = None
        self._generating            = False
        self.heatmap_logits         = 0
        self.heatmap_uncertainty    = 0

    def update_transparency(self):
        """Update transparency of the heatmap overlay."""

        if self.viz.rendered_heatmap is not None:
            alpha_channel = np.full(self.viz.rendered_heatmap.shape[0:2],
                                    int(self.alpha * 255),
                                    dtype=np.uint8)
            self.viz.overlay = np.dstack((self.viz.rendered_heatmap[:, :, 0:3],
                                                  alpha_channel))
            full_extract = int(self.viz.wsi.tile_um / self.viz.wsi.mpp)
            wsi_stride = int(full_extract / self.viz.wsi.stride_div)
            self.viz._overlay_wsi_dim = (wsi_stride * (self.viz.overlay.shape[1]),
                                         wsi_stride * (self.viz.overlay.shape[0]))
            self.viz._overlay_offset_wsi_dim = (full_extract/2 - wsi_stride/2, full_extract/2 - wsi_stride/2)


    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if self._generating:
            self.refresh_generating_heatmap()

        if show:
            self.content_height = imgui.get_text_line_height_with_spacing() * 13 + viz.spacing * 2
            _cmap_changed = False
            _uq_logits_switched = False
            bg_color = [0.16, 0.29, 0.48, 0.2]
            dim_color = list(imgui.get_style().colors[imgui.COLOR_TEXT])
            dim_color[-1] *= 0.5

            # Begin heatmap view.
            width = viz.font_size * 20
            height = imgui.get_text_line_height_with_spacing() * 13 + viz.spacing
            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, [0, 0])
            imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, *bg_color)
            imgui.push_style_color(imgui.COLOR_HEADER, 0, 0, 0, 0)
            imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, 0.16, 0.29, 0.48, 0.5)
            imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, 0.16, 0.29, 0.48, 0.9)
            imgui.begin_child('##heatmap_display', width=width, height=height, border=True)

            if viz.rendered_heatmap is not None:
                hw_ratio = (viz.rendered_heatmap.shape[0] / viz.rendered_heatmap.shape[1])
                max_width = min(width - viz.spacing*2, (height - viz.spacing*2) / hw_ratio)
                max_height = max_width * hw_ratio

                if viz._heatmap_tex_obj is not None:
                    imgui.same_line(int((width - max_width)/2))
                    imgui.image(viz._heatmap_tex_obj.gl_id, max_width, max_height)

            # End heatmap view.
            if viz.rendered_heatmap is None:
                imgui.text_colored('Heatmap not generated', *dim_color)
            imgui.end_child()
            imgui.pop_style_color(4)
            imgui.pop_style_var(1)

            imgui.same_line()
            imgui.begin_child('##heatmap_options', width=-1, height=height, border=False)

            # Heatmap options.
            with imgui_utils.grayed_out(viz._model_path is None or viz.wsi is None):

                # Colormap.
                with imgui_utils.item_width(viz.font_size * 6):
                    _clicked, self.cmap_idx = imgui.combo("##cmap", self.cmap_idx, self._colormaps)
                if _clicked:
                    _cmap_changed = True

                imgui.same_line(viz.font_size * 6 + viz.spacing)
                _clicked, self.show = imgui.checkbox('Show', self.show)
                if _clicked:
                    self.render_heatmap()
                    if self.show:
                        self.viz.slide_widget.show_slide_filter  = False
                        self.viz.slide_widget.show_tile_filter   = False

                imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.button_w)
                _button_text = ('Generate' if not self._button_pressed else "Working...")
                if imgui_utils.button(_button_text, width=viz.button_w, enabled=(not self._button_pressed)):
                    self.viz.set_message(self._rendering_message)
                    self.viz.create_toast(
                        title=self._rendering_message,
                        icon='info',
                    )
                    _thread = threading.Thread(target=self.generate_heatmap)
                    _thread.start()
                    self.show = True

                with imgui_utils.grayed_out(viz.heatmap is None):

                    # Alpha.
                    with imgui_utils.item_width(-1 - viz.button_w - viz.spacing):
                        _alpha_changed, self.alpha = imgui.slider_float('##alpha',
                                                                        self.alpha,
                                                                        min_value=0,
                                                                        max_value=1,
                                                                        format='Alpha %.2f')

                    imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.button_w)
                    if imgui_utils.button('Reset##alpha', width=-1, enabled=True):
                        self.alpha = 0.5
                        _alpha_changed = True

                    # Gain.
                    with imgui_utils.item_width(-1 - viz.button_w - viz.spacing):
                        _gain_changed, self.gain = imgui.slider_float('##gain',
                                                                      self.gain,
                                                                      min_value=0,
                                                                      max_value=10,
                                                                      format='Gain %.2f')

                    imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.button_w)
                    if imgui_utils.button('Reset##gain', width=-1, enabled=True):
                        if self.use_logits:
                            self.gain = 1.0
                        else:
                            self.gain = 1.0
                        _gain_changed = True

                # Logits.
                heatmap_logits_max = 0 if self.logits is None else self.logits.shape[2]-1
                self.heatmap_logits = min(max(self.heatmap_logits, 0), heatmap_logits_max)
                narrow_w = imgui.get_text_line_height_with_spacing()
                with imgui_utils.grayed_out(heatmap_logits_max == 0):
                    if imgui.radio_button('##logits_radio', self.use_logits):
                        if viz.has_uq():
                            _uq_logits_switched = True
                            self.use_logits = True

                    imgui.same_line()
                    with imgui_utils.item_width(-1 - viz.button_w - narrow_w * 2 - viz.spacing * 3):
                        _changed, self.heatmap_logits = imgui.drag_int('##heatmap_logits',
                                                                       self.heatmap_logits,
                                                                       change_speed=0.05,
                                                                       min_value=0,
                                                                       max_value=heatmap_logits_max,
                                                                       format=f'Logits %d/{heatmap_logits_max}')
                    imgui.same_line()
                    if imgui_utils.button('-##heatmap_logits', width=narrow_w):
                        self.heatmap_logits -= 1
                    imgui.same_line()
                    if imgui_utils.button('+##heatmap_logits', width=narrow_w):
                        self.heatmap_logits += 1
                    self.heatmap_logits = min(max(self.heatmap_logits, 0), heatmap_logits_max)
                    if heatmap_logits_max > 0:
                        imgui.same_line()
                        imgui.text(viz._model_config['outcome_labels'][str(self.heatmap_logits)])

                # Uncertainty.
                if viz.heatmap is None or self.uncertainty is None:
                    heatmap_uncertainty_max = 0
                else:
                    heatmap_uncertainty_max = self.uncertainty.shape[2] - 1

                self.heatmap_uncertainty = min(max(self.heatmap_uncertainty, 0), heatmap_uncertainty_max)
                narrow_w = imgui.get_text_line_height_with_spacing()
                with imgui_utils.grayed_out(viz.heatmap is None or not viz.has_uq()):
                    if imgui.radio_button('##uncertainty_radio', not self.use_logits):
                        if viz.has_uq():
                            _uq_logits_switched = True
                            self.use_logits = False

                    imgui.same_line()
                    with imgui_utils.item_width(-1 - viz.button_w - narrow_w * 2 - viz.spacing * 3):
                        _changed, self.heatmap_uncertainty = imgui.drag_int('##heatmap_uncertainty',
                                                                            self.heatmap_uncertainty,
                                                                            change_speed=0.05,
                                                                            min_value=0,
                                                                            max_value=heatmap_uncertainty_max,
                                                                            format=f'UQ %d/{heatmap_uncertainty_max}')
                    imgui.same_line()
                    if imgui_utils.button('-##heatmap_uncertainty', width=narrow_w):
                        self.heatmap_uncertainty -= 1
                    imgui.same_line()
                    if imgui_utils.button('+##heatmap_uncertainty', width=narrow_w):
                        self.heatmap_uncertainty += 1
                    self.heatmap_uncertainty = min(max(self.heatmap_uncertainty, 0), heatmap_uncertainty_max)

                _histogram_size = imgui.get_content_region_max()[0] - 1, viz.font_size * 4
                if viz.heatmap and self.logits is not None:
                    flattened = self.logits[:, :, self.heatmap_logits].flatten()
                    flattened = flattened[flattened >= 0]
                    _hist, _bin_edges = np.histogram(flattened, range=(0, 1))
                    if flattened.shape[0] > 0:
                        overlay_text = f"Predictions (avg: {np.mean(flattened):.2f})"
                        _hist_arr = array('f', _hist/np.sum(_hist))
                        scale_max = np.max(_hist/np.sum(_hist))
                    else:
                        overlay_text = "Predictions (avg: - )"
                        _hist_arr = array('f', [0])
                        scale_max = 1
                    imgui.separator()
                    imgui.core.plot_histogram('##heatmap_pred',
                                              _hist_arr,
                                              scale_min=0,
                                              overlay_text=overlay_text,
                                              scale_max=scale_max,
                                              graph_size=_histogram_size)

                if viz.heatmap and self.uncertainty is not None:
                    flattened = self.uncertainty[:, :, self.heatmap_uncertainty].flatten()
                    flattened = flattened[flattened >= 0]
                    _hist, _bin_edges = np.histogram(flattened)
                    if flattened.shape[0] > 0:
                        overlay_text = f"Uncertainty (avg: {np.mean(flattened):.2f})"
                        _hist_arr = array('f', _hist/np.sum(_hist))
                        scale_max = np.max(_hist/np.sum(_hist))
                    else:
                        overlay_text = "Uncertainty (avg: - )"
                        _hist_arr = array('f', [0])
                        scale_max = 1
                    imgui.separator()
                    imgui.core.plot_histogram('##heatmap_pred',
                                              _hist_arr,
                                              scale_min=0,
                                              overlay_text=overlay_text,
                                              scale_max=scale_max,
                                              graph_size=_histogram_size)
                    imgui.separator()
                elif not viz.has_uq():
                    imgui.text("Model not trained with uncertainty.")

            imgui.end_child()

            # Render heatmap.
            if _alpha_changed:
                self.update_transparency()
            if _gain_changed:
                if self.use_logits:
                    self._logits_gain = self.gain
                else:
                    self._uq_gain = self.gain
                self.render_heatmap()
            if _uq_logits_switched:
                self.gain = self._logits_gain if self.use_logits else self._uq_gain
                self.render_heatmap()
            if (_cmap_changed
               or (self.heatmap_logits != self._old_logits and self.use_logits)
               or (self.heatmap_uncertainty != self._old_uncertainty and self.use_uncertainty)
               or _uq_logits_switched):
                self.render_heatmap()
        else:
            self.content_height = 0
#----------------------------------------------------------------------------
