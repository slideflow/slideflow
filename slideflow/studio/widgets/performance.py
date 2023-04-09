import array
import numpy as np
import imgui

from ..gui import imgui_utils

#----------------------------------------------------------------------------

class PerformanceWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.gui_times      = [float('nan')] * 60
        self.render_times   = [float('nan')] * 30
        self.norm_times     = [float('nan')] * 30
        self.predict_times  = [float('nan')] * 30
        self.fps_limit      = 60
        self.use_vsync      = True
        self.ignore_jpg     = viz._use_model_img_fmt
        self.low_memory     = viz.low_memory

        viz.set_fps_limit(self.fps_limit)
        viz.set_vsync(self.use_vsync)
        viz.low_memory = self.low_memory
        viz._use_model_img_fmt = not self.ignore_jpg

    def timing_text(self, times):
        viz = self.viz
        imgui.same_line(viz.label_w + viz.font_size * 7)
        t = [x for x in times if x > 0]
        t = np.mean(t) if len(t) > 0 else 0
        imgui.text(f'{t*1e3:.1f} ms' if t > 0 else 'N/A')
        if imgui.is_item_hovered():
            imgui.set_tooltip(f'{1/t:.1f} FPS' if t > 0 else 'N/A')

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        self.gui_times = self.gui_times[1:] + [viz.frame_delta]
        if 'render_time' in viz.result:
            self.render_times = self.render_times[1:] + [viz.result.render_time]
            del viz.result.render_time
        if 'norm_time' in viz.result:
            self.norm_times = self.norm_times[1:] + [viz.result.norm_time]
            del viz.result.norm_time
        if 'inference_time' in viz.result:
            self.predict_times = self.predict_times[1:] + [viz.result.inference_time]
            del viz.result.inference_time

        if show:

            viz.header("Performance")

            if viz.collapsing_header('Timing', default=True):
                # GUI times
                imgui.text_colored('GUI', *viz.theme.dim)
                imgui.same_line(viz.label_w)
                with imgui_utils.item_width(viz.font_size * 6):
                    imgui.plot_lines('##gui_times', array.array('f', self.gui_times), scale_min=0)
                self.timing_text(self.gui_times)
                #imgui.same_line(viz.label_w + viz.font_size * 18 + viz.spacing * 3)
                #with imgui_utils.item_width(viz.font_size * 6):
                #    _changed, self.fps_limit = imgui.input_int('FPS limit', self.fps_limit, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                #    self.fps_limit = min(max(self.fps_limit, 5), 1000)
                #    if _changed:
                #        viz.set_fps_limit(self.fps_limit)

                # Render
                imgui.text_colored('Render', *viz.theme.dim)
                imgui.same_line(viz.label_w)
                with imgui_utils.item_width(viz.font_size * 6):
                    imgui.plot_lines('##render_times', array.array('f', self.render_times), scale_min=0)
                self.timing_text(self.render_times)
                #imgui.same_line(viz.label_w + viz.font_size * 18 + viz.spacing * 3)
                #_clicked, self.low_memory = imgui.checkbox('Low memory mode', self.low_memory)
                #if _clicked:
                #    viz.low_memory = self.low_memory

                # Normalizer times
                imgui.text_colored('Normalize', *viz.theme.dim)
                imgui.same_line(viz.label_w)
                with imgui_utils.item_width(viz.font_size * 6):
                    imgui.plot_lines('##norm_times', array.array('f', self.norm_times), scale_min=0)
                self.timing_text(self.norm_times)
                #imgui.same_line(viz.label_w + viz.font_size * 18 + viz.spacing * 3)
                #_clicked, self.use_vsync = imgui.checkbox('Vertical sync', self.use_vsync)
                #if _clicked:
                #    viz.set_vsync(self.use_vsync)

                # Inference times
                imgui.text_colored('Predict', *viz.theme.dim)
                imgui.same_line(viz.label_w)
                with imgui_utils.item_width(viz.font_size * 6):
                    imgui.plot_lines('##predict_times', array.array('f', self.predict_times), scale_min=0)
                self.timing_text(self.predict_times)
                #imgui.same_line(viz.label_w + viz.font_size * 18 + viz.spacing * 3)
                #_clicked, self.ignore_jpg = imgui.checkbox('Ignore compression', self.ignore_jpg)
                #if _clicked:
                #    viz._use_model_img_fmt = not self.ignore_jpg

#----------------------------------------------------------------------------
