import imgui
import numpy as np
from array import array

from ..gui import imgui_utils
from ..gui.annotator import AnnotationCapture

import slideflow.grad as grad


class ModelWidget:
    def __init__(self, viz, show_saliency=True):
        self.viz                = viz
        self.show_saliency      = show_saliency
        self.annotator          = AnnotationCapture(named=False)
        self.search_dirs        = []
        self.cur_model          = None
        self.user_model         = ''
        self.backend            = 'Unknown'
        self.recent_models      = []
        self.browse_cache       = dict()
        self.browse_refocus     = False
        self.use_model          = False
        self.use_uncertainty    = False
        self.enable_saliency    = False
        self.saliency_overlay   = False
        self.saliency_idx       = 0
        self.content_height     = 0
        self._clicking          = False
        self._show_params       = False
        self._show_popup        = False

        self._saliency_methods_all = {
            'Vanilla': grad.VANILLA,
            'Vanilla (Smoothed)': grad.VANILLA_SMOOTH,
            'Integrated Gradients': grad.INTEGRATED_GRADIENTS,
            'Integrated Gradients (Smooth)': grad.INTEGRATED_GRADIENTS_SMOOTH,
            'Guided Integrated Gradients': grad.GUIDED_INTEGRATED_GRADIENTS,
            'Guided Integrated Gradients (Smooth)': grad.GUIDED_INTEGRATED_GRADIENTS_SMOOTH,
            'Blur Integrated Gradients': grad.BLUR_INTEGRATED_GRADIENTS,
            'Blur Integrated Gradients (Smooth)': grad.BLUR_INTEGRATED_GRADIENTS_SMOOTH,
        }
        self._saliency_methods_names = list(self._saliency_methods_all.keys())

    def add_recent(self, model, ignore_errors=False):
        try:
            if model not in self.recent_models:
                self.recent_models.append(model)
        except:
            if not ignore_errors:
                raise

    def refresh_recent(self):
        if self.user_model in self.recent_models:
            self.recent_models.remove(self.user_model)
        self.recent_models.insert(0, self.user_model)

    def load(self, model, ignore_errors=False):
        self.viz.load_model(model, ignore_errors=ignore_errors)

    def draw_info(self):
        viz = self.viz
        config = viz._model_config

        if config is not None:
            if 'outcome_label_headers' in config:
                outcomes_list = config['outcome_label_headers']
            else:
                outcomes_list = config['outcomes']
            if len(outcomes_list) == 1:
                outcomes = outcomes_list[0]
            else:
                outcomes = ', '.join(outcomes_list)
            vals = [
                outcomes,
                str(config['tile_px']),
                str(config['tile_um']),
                "<unknown>" if 'img_format' not in config else config['img_format'],
                self.backend,
                str(config['slideflow_version']),
            ]
        else:
            vals = ["-" for _ in range(6)]
        rows = [
            #['Property',     'Value'],
            ['Outcomes',     vals[0]],
            ['Tile (px)',    vals[1]],
            ['Tile (um)',    vals[2]],
            ['Image format', vals[3]],
            ['Backend',      vals[4]],
            ['Version',      vals[5]],
        ]
        imgui.text_colored('Model name', *viz.theme.dim)
        imgui.same_line(viz.font_size * 6)
        with imgui_utils.clipped_with_tooltip(config['model_name'], 22):
            imgui.text(imgui_utils.ellipsis_clip(config['model_name'], 22))
        for y, cols in enumerate(rows):
            for x, col in enumerate(cols):
                if x != 0:
                    imgui.same_line(viz.font_size * (6 + (x - 1) * 6))
                if x == 0: # or y == 0:
                    imgui.text_colored(col, *viz.theme.dim)
                else:
                    imgui.text(col)

        with imgui_utils.grayed_out(viz._model_path is None):
            imgui.same_line(imgui.get_content_region_max()[0] - viz.font_size - viz.spacing * 2)
            if imgui.button("HP") and self.viz._model_config:
                self._show_params = not self._show_params

    def draw_params_popup(self):
        viz = self.viz
        hp = self.viz._model_config['hp']
        rows = list(zip(list(map(str, hp.keys())), list(map(str, hp.values()))))

        _, self._show_params = imgui.begin("Model parameters", closable=True, flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_SCROLLBAR)
        for y, cols in enumerate(rows):
            for x, col in enumerate(cols):
                if x != 0:
                    imgui.same_line(viz.font_size * 10)
                if x == 0:
                    imgui.text_colored(col, *viz.theme.dim)
                else:
                    imgui.text(col)
        imgui.end()

    def draw_predictions(self):
        viz = self.viz
        config = viz._model_config
        if config is not None:

            # Tile prediction =================================================
            if viz._use_model and viz._predictions is not None and isinstance(viz._predictions, list):
                for p, _pred_array in enumerate(viz._predictions):
                    self.content_height += (imgui.get_text_line_height_with_spacing() + viz.spacing)
                    imgui.text_colored(f'Pred {p}', *viz.theme.dim)
                    if 'outcomes' in config:
                        outcomes = config['outcomes'][p]
                    elif 'outcome_label_headers' in config:
                        outcomes = config['outcome_label_headers'][p]
                    ol = config['outcome_labels'][outcomes]
                    pred_str = ol[str(np.argmax(_pred_array))]
                    imgui.same_line(imgui.get_content_region_max()[0] - viz.spacing - imgui.calc_text_size(pred_str).x)
                    if viz._use_uncertainty and viz._uncertainty is not None:
                        pred_str += " (UQ: {:.4f})".format(viz._uncertainty[p])
                    imgui.text(pred_str)
                    imgui.core.plot_histogram('##pred', array('f', _pred_array), scale_min=0, scale_max=1)
            elif viz._use_model and viz._predictions is not None:
                self.content_height += (imgui.get_text_line_height_with_spacing() + viz.spacing)
                imgui.text_colored('Tile prediction', *viz.theme.dim)
                ol = config['outcome_labels']
                pred_str = ol[str(np.argmax(viz._predictions))]
                if viz._use_uncertainty and viz._uncertainty is not None:
                    pred_str += " (UQ: {:.4f})".format(viz._uncertainty)
                imgui.same_line(imgui.get_content_region_max()[0] - viz.spacing - imgui.calc_text_size(pred_str).x)
                imgui.text(pred_str)
                with imgui_utils.item_width(imgui.get_content_region_max()[0] - viz.spacing):
                    imgui.core.plot_histogram('##pred', array('f', viz._predictions), scale_min=0, scale_max=1)
            elif viz._use_model:
                imgui_utils.padded_text('Right click for a focal prediction.', vpad=[int(viz.font_size/2), int(viz.font_size)])
            else:
                imgui_utils.padded_text('Model not in use.', vpad=[int(viz.font_size/2), int(viz.font_size)])

            # Slide prediction ================================================
            txt = "Predict Slide" if (not viz.heatmap_widget._triggered) else "Predicting Slide..."
            if viz.sidebar.full_button(txt, enabled=(not viz.heatmap_widget._triggered and viz.wsi)):
                viz.heatmap_widget.generate()

            hw = viz.heatmap_widget
            _histogram_size = imgui.get_content_region_max()[0] - viz.spacing, viz.font_size * 4
            if viz.heatmap and hw.predictions is not None:
                flattened = hw.predictions[:, :, hw.heatmap_predictions].flatten()
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

            if viz.heatmap and hw.uncertainty is not None:
                flattened = hw.uncertainty[:, :, hw.heatmap_uncertainty].flatten()
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

            if viz.heatmap:
                viz.heatmap_widget.draw_outcome_selection(radio=False)

            imgui.text('')

    def draw_saliency(self):
        viz = self.viz
        with imgui_utils.grayed_out(viz._model_path is None), imgui_utils.item_width(viz.font_size * 5):
            _clicked, self.enable_saliency = imgui.checkbox('##saliency', self.enable_saliency)

        imgui.same_line()
        with imgui_utils.item_width(imgui.get_content_region_max()[0] - viz.font_size*1.8):
            _clicked, self.saliency_idx = imgui.combo("##method", self.saliency_idx, self._saliency_methods_names)

        if imgui.radio_button('Heatmap', not self.saliency_overlay):
            self.saliency_overlay = False
        imgui.same_line()
        if imgui.radio_button('Overlay', self.saliency_overlay):
            self.saliency_overlay = True

        viz._use_saliency = self.enable_saliency
        viz.args.saliency_method = self.saliency_idx
        viz.args.saliency_overlay = self.saliency_overlay

    def draw_config_popup(self):
        viz = self.viz
        has_model = viz._model_config is not None

        if self._show_popup:
            cx, cy = imgui.get_cursor_pos()
            imgui.set_next_window_position(viz.sidebar.full_width, cy)
            imgui.begin(
                '##model_popup',
                flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
            )
            if imgui.menu_item('Load model')[0]:
                viz.ask_load_model()
            if imgui.menu_item('Download model')[0]:
                print("Huggingface?")
            imgui.separator()
            if imgui.menu_item('Enable model', enabled=has_model, selected=self.use_model)[0]:
                self.use_model = not self.use_model
                viz._use_model = self.use_model
            if imgui.menu_item('Enable UQ', enabled=has_model, selected=self.use_uncertainty)[0]:
                self.use_uncertainty = not self.use_uncertainty
                viz._use_uncertainty = self.use_uncertainty
            imgui.separator()
            if imgui.menu_item('Show parameters', enabled=has_model)[0]:
                self._show_params = not self._show_params

            # Hide menu if we click elsewhere
            if imgui.is_mouse_down(0) and not imgui.is_window_hovered():
                self._clicking = True
            if self._clicking and imgui.is_mouse_released(0):
                self._clicking = False
                self._show_popup = False

            imgui.end()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        config = viz._model_config

        if show:
            with viz.sidebar.header_with_buttons("Model"):
                imgui.same_line(imgui.get_content_region_max()[0] - viz.font_size*1.5)
                cx, cy = imgui.get_cursor_pos()
                imgui.set_cursor_position((cx, cy-int(viz.font_size*0.25)))
                if viz.sidebar.small_button('gear'):
                    self._clicking = False
                    self._show_popup = not self._show_popup
                self.draw_config_popup()

        if show and not config:
            imgui_utils.padded_text('No model has been loaded.', vpad=[int(viz.font_size/2), int(viz.font_size)])
            if viz.sidebar.full_button("Load a Model"):
                viz.ask_load_model()
            if viz.sidebar.full_button("Download a Model"):
                print("Huggingface?")

        elif show:
            if viz.sidebar.collapsing_header('Info', default=True):
                self.draw_info()
            if viz.sidebar.collapsing_header('Predictions', default=True):
                self.draw_predictions()
            if viz.sidebar.collapsing_header('Saliency', default=False):
                self.draw_saliency()

        if self._show_params and self.viz._model_config:
            self.draw_params_popup()

