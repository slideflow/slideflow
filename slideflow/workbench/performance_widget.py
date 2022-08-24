# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import array
import numpy as np
import imgui

from .gui_utils import imgui_utils

#----------------------------------------------------------------------------

class PerformanceWidget:
    def __init__(self, viz, low_memory=False):
        self.viz            = viz
        self.gui_times      = [float('nan')] * 60
        self.render_times   = [float('nan')] * 30
        self.fps_limit      = 60
        self.use_vsync      = False
        self.is_async       = False
        self.ignore_jpg     = False
        self.low_memory     = low_memory

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        self.gui_times = self.gui_times[1:] + [viz.frame_delta]
        if 'render_time' in viz.result:
            self.render_times = self.render_times[1:] + [viz.result.render_time]
            del viz.result.render_time

        if show:
            imgui.text('GUI')
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 8):
                imgui.plot_lines('##gui_times', array.array('f', self.gui_times), scale_min=0)
            imgui.same_line(viz.label_w + viz.font_size * 9)
            t = [x for x in self.gui_times if x > 0]
            t = np.mean(t) if len(t) > 0 else 0
            imgui.text(f'{t*1e3:.1f} ms' if t > 0 else 'N/A')
            imgui.same_line(viz.label_w + viz.font_size * 14)
            imgui.text(f'{1/t:.1f} FPS' if t > 0 else 'N/A')
            imgui.same_line(viz.label_w + viz.font_size * 18 + viz.spacing * 3)
            with imgui_utils.item_width(viz.font_size * 6):
                _changed, self.fps_limit = imgui.input_int('FPS limit', self.fps_limit, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                self.fps_limit = min(max(self.fps_limit, 5), 1000)
            imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.button_w * 2 - viz.spacing)
            _clicked, self.use_vsync = imgui.checkbox('Vertical sync', self.use_vsync)

        if show:
            imgui.text('Render')
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 8):
                imgui.plot_lines('##render_times', array.array('f', self.render_times), scale_min=0)
            imgui.same_line(viz.label_w + viz.font_size * 9)
            t = [x for x in self.render_times if x > 0]
            t = np.mean(t) if len(t) > 0 else 0
            imgui.text(f'{t*1e3:.1f} ms' if t > 0 else 'N/A')
            imgui.same_line(viz.label_w + viz.font_size * 14)
            imgui.text(f'{1/t:.1f} FPS' if t > 0 else 'N/A')
            imgui.same_line(viz.label_w + viz.font_size * 18 + viz.spacing * 3)
            #_clicked, self.is_async = imgui.checkbox('Separate process', self.is_async)
            _clicked, self.low_memory = imgui.checkbox('Low memory mode', self.low_memory)
            imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.button_w * 2 - viz.spacing)
            _clicked, self.ignore_jpg = imgui.checkbox('Ignore compression', self.ignore_jpg)

        viz.set_fps_limit(self.fps_limit)
        viz.set_vsync(self.use_vsync)
        viz.set_async(self.is_async)
        viz.set_low_memory(self.low_memory)
        viz._use_model_img_fmt = not self.ignore_jpg

#----------------------------------------------------------------------------
