# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import imgui
import threading
import matplotlib.pyplot as plt
from .gui_utils import imgui_utils

import slideflow as sf

#----------------------------------------------------------------------------

def _apply_cmap(img, cmap):
        cmap = plt.get_cmap(cmap)
        return (cmap(img) * 255).astype(np.uint8)

#----------------------------------------------------------------------------

class HeatmapWidget:
    def __init__(self, viz):
        self.viz                    = viz
        self.alpha                  = 0.5
        self.gain                   = 1.0
        self.show                   = True
        self.heatmap_logits         = 0
        self.heatmap_uncertainty    = 0
        self.stride                 = 1
        self.use_logits             = True
        self.use_uncertainty        = False
        self.cmap_idx               = 0
        self._generating            = False
        self._old_logits            = 0
        self._old_uncertainty       = 0
        self._logits_gain           = 1.0
        self._uq_gain               = 1.0
        self._colormaps             = plt.colormaps()

    def update_transparency(self):
        if self.viz.rendered_heatmap is not None:
            alpha_channel = np.full(self.viz.rendered_heatmap.shape[0:2], int(self.alpha * 255), dtype=np.uint8)
            self.viz.overlay_heatmap = np.dstack((self.viz.rendered_heatmap[:, :, 0:3], alpha_channel))

    def render_heatmap(self):
        self._old_logits = self.heatmap_logits
        self._old_uncertainty = self.heatmap_uncertainty
        if self.viz.heatmap is None:
            return
        if self.use_logits:
            heatmap_arr = self.viz.heatmap.logits[:, :, self.heatmap_logits]
            gain = self._logits_gain
        else:
            heatmap_arr = self.viz.heatmap.uncertainty[:, :, self.heatmap_uncertainty]
            gain = self._uq_gain
        self.viz.rendered_heatmap = _apply_cmap(heatmap_arr * gain, self._colormaps[self.cmap_idx])[:, :, 0:3]
        self.update_transparency()

    def reset(self):
        self.viz.rendered_heatmap = None
        self.viz.overlay_heatmap = None
        self.viz.heatmap = None

    def generate_heatmap(self):
        self._generating = True
        viz = self.viz
        viz.heatmap = sf.Heatmap(viz.wsi.path, viz._model_path, stride_div=self.stride)
        self.render_heatmap()
        self.update_transparency()
        self._generating = False
        viz._show_overlay = self.show

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            _cmap_changed = False
            _uq_logits_switched = False
            bg_color = [0.16, 0.29, 0.48, 0.2]
            dim_color = list(imgui.get_style().colors[imgui.COLOR_TEXT])
            dim_color[-1] *= 0.5

            # Begin heatmap view.
            width = viz.font_size * 28
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
            with imgui_utils.grayed_out(viz._model is None or viz.wsi is None):

                with imgui_utils.item_width(viz.font_size * 5):
                    _alpha_changed, self.stride = imgui.slider_int('##stride', self.stride, min_value=1, max_value=16, format='Stride %d')

                imgui.same_line(viz.font_size * 5 + viz.spacing)
                _clicked, self.show = imgui.checkbox('Show', self.show)
                if _clicked:
                    viz._show_overlay = self.show

                imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.button_w)
                if imgui_utils.button(('Generate' if not self._generating else "Working..."), width=viz.button_w, enabled=(not self._generating)):
                    _thread = threading.Thread(target=self.generate_heatmap)
                    _thread.start()

                with imgui_utils.grayed_out(viz.heatmap is None):
                    # Colormap.
                    with imgui_utils.item_width(-1 - viz.button_w - viz.spacing):
                        _clicked, self.cmap_idx = imgui.combo("##cmap", self.cmap_idx, self._colormaps)
                    if _clicked:
                        _cmap_changed = True
                    imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.button_w)
                    if imgui_utils.button('Reset##cmap', width=-1, enabled=True):
                        self.cmap_idx = 0
                        _cmap_changed = True

                    # Alpha.
                    with imgui_utils.item_width(-1 - viz.button_w - viz.spacing):
                        _alpha_changed, self.alpha = imgui.slider_float('##alpha', self.alpha, min_value=0, max_value=1, format='Alpha %.2f')
                    imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.button_w)
                    if imgui_utils.button('Reset##alpha', width=-1, enabled=True):
                        self.alpha = 0.5
                        _alpha_changed = True

                    # Gain.
                    with imgui_utils.item_width(-1 - viz.button_w - viz.spacing):
                        _gain_changed, self.gain = imgui.slider_float('##gain', self.gain, min_value=0, max_value=10, format='Gain %.2f')
                    imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.button_w)
                    if imgui_utils.button('Reset##gain', width=-1, enabled=True):
                        if self.use_logits:
                            self.gain = 1.0
                        else:
                            self.gain = 1.0
                        _gain_changed = True

                # Logits.
                heatmap_logits_max = 0 if viz.heatmap is None else viz.heatmap.logits.shape[2]-1
                self.heatmap_logits = min(max(self.heatmap_logits, 0), heatmap_logits_max)
                narrow_w = imgui.get_text_line_height_with_spacing()
                with imgui_utils.grayed_out(heatmap_logits_max == 0):
                    if imgui.radio_button('##logits_radio', self.use_logits):
                        if viz.has_uq():
                            _uq_logits_switched = True
                            self.use_logits = True

                    imgui.same_line()
                    with imgui_utils.item_width(-1 - viz.button_w - narrow_w * 2 - viz.spacing * 3):
                        _changed, self.heatmap_logits = imgui.drag_int('##heatmap_logits', self.heatmap_logits, change_speed=0.05, min_value=0, max_value=heatmap_logits_max, format=f'Logits %d/{heatmap_logits_max}')
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

                # Uuncertainty.
                heatmap_uncertainty_max = 0 if (viz.heatmap is None or viz.heatmap.uncertainty is None) else viz.heatmap.uncertainty.shape[2]-1
                self.heatmap_uncertainty = min(max(self.heatmap_uncertainty, 0), heatmap_uncertainty_max)
                narrow_w = imgui.get_text_line_height_with_spacing()
                with imgui_utils.grayed_out(viz.heatmap is None or not viz.has_uq()):
                    if imgui.radio_button('##uncertainty_radio', not self.use_logits):
                        if viz.has_uq():
                            _uq_logits_switched = True
                            self.use_logits = False

                    imgui.same_line()
                    with imgui_utils.item_width(-1 - viz.button_w - narrow_w * 2 - viz.spacing * 3):
                        _changed, self.heatmap_uncertainty = imgui.drag_int('##heatmap_uncertainty', self.heatmap_uncertainty, change_speed=0.05, min_value=0, max_value=heatmap_uncertainty_max, format=f'UQ %d/{heatmap_uncertainty_max}')
                    imgui.same_line()
                    if imgui_utils.button('-##heatmap_uncertainty', width=narrow_w):
                        self.heatmap_uncertainty -= 1
                    imgui.same_line()
                    if imgui_utils.button('+##heatmap_uncertainty', width=narrow_w):
                        self.heatmap_uncertainty += 1
                    self.heatmap_uncertainty = min(max(self.heatmap_uncertainty, 0), heatmap_logits_max)

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
            if _cmap_changed or (self.heatmap_logits != self._old_logits and self.use_logits) or (self.heatmap_uncertainty != self._old_uncertainty and self.use_uncertainty) or _uq_logits_switched:
                self.render_heatmap()

#----------------------------------------------------------------------------
