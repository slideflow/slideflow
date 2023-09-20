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

                # Render
                imgui.text_colored('Render', *viz.theme.dim)
                imgui.same_line(viz.label_w)
                with imgui_utils.item_width(viz.font_size * 6):
                    imgui.plot_lines('##render_times', array.array('f', self.render_times), scale_min=0)
                self.timing_text(self.render_times)

                # Normalizer times
                imgui.text_colored('Normalize', *viz.theme.dim)
                imgui.same_line(viz.label_w)
                with imgui_utils.item_width(viz.font_size * 6):
                    imgui.plot_lines('##norm_times', array.array('f', self.norm_times), scale_min=0)
                self.timing_text(self.norm_times)

                # Inference times
                imgui.text_colored('Predict', *viz.theme.dim)
                imgui.same_line(viz.label_w)
                with imgui_utils.item_width(viz.font_size * 6):
                    imgui.plot_lines('##predict_times', array.array('f', self.predict_times), scale_min=0)
                self.timing_text(self.predict_times)

#----------------------------------------------------------------------------
