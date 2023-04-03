import imgui
import numpy as np
from PIL import Image
from os.path import join, dirname, abspath
from array import array
from collections import defaultdict

from ..gui import imgui_utils, gl_utils
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
        self.pred_idx           = defaultdict(int)
        self._clicking          = False
        self._show_params       = False
        self._show_popup        = False
        self._show_download     = False

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

    def reset(self):
        self.saliency_idx = 0
        self.pred_idx = defaultdict(int)

    def outcome_indices(self, outcome):
        config = self.viz._model_config
        _outcome_lengths = [
            len(config['outcome_labels'][o]) for o in config['outcomes']
        ]
        outcome_idx = config['outcomes'].index(outcome)
        start = sum([l for l in _outcome_lengths[:outcome_idx]])
        end = start + _outcome_lengths[outcome_idx]
        return list(range(start, end))

    def outcome_index_start(self, outcome):
        config = self.viz._model_config
        _outcome_lengths = [
            len(config['outcome_labels'][o]) for o in config['outcomes']
        ]
        outcome_idx = config['outcomes'].index(outcome)
        return sum([l for l in _outcome_lengths[:outcome_idx]])

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

    def download_popup(self):
        viz = self.viz
        if self._show_download:
            imgui.open_popup('download_popup')
            width = 200
            height = 315
            imgui.set_next_window_position(viz.content_width/2 - width/2, viz.content_height/2 - height/2)

            if imgui.begin_popup('download_popup'):
                with viz.bold_font():
                    viz.center_text('Coming Soon')
                imgui.separator()
                imgui.text(
                    "Automatic model downloads are coming soon.\n"
                    "In the meantime, you can find our public models\n"
                    "on HuggingFace: huggingface.co/jamesdolezal")
                imgui_utils.vertical_break()
                imgui.text('')
                imgui.same_line((imgui.get_content_region_max()[0])/2 - (self.viz.button_w/2) + self.viz.spacing)
                if imgui.button('Close', width=viz.button_w):
                    self._show_download = False
                imgui.end_popup()

    def load(self, model, ignore_errors=False):
        self.viz.load_model(model, ignore_errors=ignore_errors)

    # -------------------------------------------------------------------------

    def _draw_tile_pred_result(self, outcome, labels, pred_array, uq_array=None):
        viz = self.viz
        config = viz._model_config
        is_categorical = config['model_type'] == 'categorical'

        flattened = pred_array.flatten()
        flattened = flattened[flattened != -99]

        # Outcome name label
        imgui.text_colored(outcome, *viz.theme.dim)

        # Prediction string
        if is_categorical:
            pred_str = labels[str(np.argmax(pred_array))]
        else:
            pred_str = f'{pred_array:.3f}'
        if viz._use_uncertainty and uq_array is not None:
            pred_str += " (UQ: {:.4f})".format(uq_array)
        imgui.same_line(imgui.get_content_region_max()[0] - viz.spacing - imgui.calc_text_size(pred_str).x)
        imgui.text(pred_str)

        # Histogram
        if is_categorical:
            with imgui_utils.item_width(imgui.get_content_region_max()[0] - viz.spacing):
                imgui.core.plot_histogram('##pred', array('f', pred_array), scale_min=0, scale_max=1)

    def _draw_prediction_as_text(self, outcome, all_labels, pred_array):
        if pred_array is None:
            return
        masked = np.ma.masked_where(((pred_array == -99) | (pred_array == np.nan)), pred_array)
        viz = self.viz
        config = viz._model_config
        imgui.text_colored(outcome, *viz.theme.dim)
        if viz.heatmap_widget._triggered:
            pred_str = imgui_utils.spinner_text()
        elif config['model_type'] == 'categorical':
            preds = masked.mean(axis=(0,1)).filled()
            pred_str = all_labels[str(np.argmax(preds))]
        else:
            pred_str = f'{masked.mean():.3f}'
        imgui.same_line(imgui.get_content_region_max()[0] - viz.spacing - imgui.calc_text_size(pred_str).x)
        imgui.text(pred_str)

    def _draw_slide_histograms(self, outcome, all_labels, active_label, pred_array, uq_array=None):
        viz = self.viz
        config = viz._model_config
        hw = viz.heatmap_widget
        _histogram_size = imgui.get_content_region_max()[0] - viz.spacing, viz.font_size * 4

        # Slide prediction histogram and text
        if viz.heatmap and pred_array is not None:
            flattened = pred_array.flatten()
            flattened = flattened[flattened != -99]

            # Outcome selection
            imgui.separator()
            if uq_array is not None:
                imgui.text("Predictions")
            narrow_w = imgui.get_text_line_height_with_spacing()
            if imgui_utils.button(f'-##pred_idx{outcome}', width=narrow_w) and self.pred_idx[outcome] > 0:
                self.pred_idx[outcome] -= 1
            imgui.same_line()
            with imgui_utils.item_width(imgui.get_content_region_max()[0] - viz.spacing*3 - narrow_w*2):
                _, hpred = imgui.drag_int(
                    f'##pred_idx{outcome}',
                    self.pred_idx[outcome]+1,
                    min_value=1,
                    max_value=len(all_labels),
                    format=f'{active_label} (%d/{len(all_labels)})'
                )
            self.pred_idx[outcome] = hpred - 1
            imgui.same_line()
            if imgui_utils.button(f'+##pred_idx{outcome}', width=narrow_w) and self.pred_idx[outcome] < (len(all_labels)-1):
                self.pred_idx[outcome] += 1

            # Histogram
            _hist, _bin_edges = np.histogram(flattened, range=(0, 1))
            if flattened.shape[0] > 0:
                overlay_text = f"Average: {np.nanmean(flattened):.2f}"
                _hist_arr = array('f', _hist/np.sum(_hist))
                scale_max = np.max(_hist/np.sum(_hist))
            else:
                overlay_text = f"Average: - "
                _hist_arr = array('f', [0])
                scale_max = 1
            imgui.core.plot_histogram(
                f'##pred_histogram{outcome}',
                _hist_arr,
                scale_min=0,
                overlay_text=overlay_text,
                scale_max=scale_max,
                graph_size=_histogram_size
            )

        # Slide uncertainty histogram
        if viz.heatmap and uq_array is not None:
            flattened = uq_array.flatten()
            flattened = flattened[flattened != -99]

            _hist, _bin_edges = np.histogram(flattened)
            if flattened.shape[0] > 0:
                overlay_text = f"Average: {np.nanmean(flattened):.2f}"
                _hist_arr = array('f', _hist/np.sum(_hist))
                scale_max = np.max(_hist/np.sum(_hist))
            else:
                overlay_text = "Average: - "
                _hist_arr = array('f', [0])
                scale_max = 1
            imgui.separator()
            imgui.text("Uncertainty")
            imgui.core.plot_histogram(
                f'##uq_histogram{outcome}',
                _hist_arr,
                scale_min=0,
                overlay_text=overlay_text,
                scale_max=scale_max,
                graph_size=_histogram_size
            )
            imgui.separator()

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

    def draw_tile_predictions(self):
        viz = self.viz
        config = viz._model_config
        has_preds = viz._use_model and viz._predictions is not None
        is_categorical = config['model_type'] == 'categorical'

        if config is not None:

            # Multiple categorical outcomes
            if has_preds and isinstance(viz._predictions, list):
                for p, _pred_array in enumerate(viz._predictions):
                    self._draw_tile_pred_result(
                        outcome=config['outcomes'][p],
                        labels=config['outcome_labels'][config['outcomes'][p]],
                        pred_array=_pred_array,
                        uq_array=None if not (viz._use_uncertainty and viz._uncertainty is not None) else viz._uncertainty[p]
                    )

            # Single categorical outcome
            elif has_preds and is_categorical:
                self._draw_tile_pred_result(
                    outcome=config['outcomes'][0],
                    labels=config['outcome_labels'],
                    pred_array=viz._predictions,
                    uq_array=None if not (viz._use_uncertainty and viz._uncertainty is not None) else viz._uncertainty
                )

            # Linear outcome(s)
            elif has_preds:
                for o_idx, outcome in enumerate(config['outcomes']):
                    self._draw_tile_pred_result(
                        outcome=outcome,
                        labels=None,
                        pred_array=viz._predictions[o_idx],
                        uq_array=None if not (viz._use_uncertainty and viz._uncertainty is not None) else viz._uncertainty
                    )

            # Model not in use
            elif viz._use_model:
                imgui_utils.padded_text('Right click for a focal prediction.', vpad=[int(viz.font_size/2), int(viz.font_size)])
            else:
                imgui_utils.padded_text('Model not in use.', vpad=[int(viz.font_size/2), int(viz.font_size)])

            imgui_utils.vertical_break()

    def draw_slide_predictions(self):
        viz = self.viz
        config = viz._model_config
        hw = viz.heatmap_widget
        is_categorical = config['model_type'] == 'categorical'
        multiple_outcomes = len(config['outcomes']) > 1

        if config is not None and viz._use_model:

            # Multiple categorical outcomes
            if multiple_outcomes and is_categorical:
                for outcome in config['outcomes']:
                    # Render predictions as text
                    self._draw_prediction_as_text(
                        outcome=outcome,
                        all_labels=config['outcome_labels'][outcome],
                        pred_array=None if hw.predictions is None else hw.predictions[:, :, self.outcome_indices(outcome)],
                    )
                    # Draw histograms
                    self._draw_slide_histograms(
                        outcome=outcome,
                        all_labels=config['outcome_labels'][outcome],
                        active_label=config['outcome_labels'][outcome][str(self.pred_idx[outcome])],
                        pred_array=None if not (viz.heatmap and hw.predictions is not None) else hw.predictions[:, :, self.outcome_index_start(outcome)+self.pred_idx[outcome]],
                        uq_array=None if not (viz.heatmap and hw.uncertainty is not None) else hw.uncertainty[:, :, self.outcome_index_start(outcome)+self.pred_idx[outcome]]
                    )

            # Single categorical or linear outcome
            elif not multiple_outcomes:
                outcome = config['outcomes'][0]
                # Render predictions as text
                self._draw_prediction_as_text(
                    outcome=config['outcomes'][0],
                    all_labels=config['outcome_labels'],
                    pred_array=None if hw.predictions is None else hw.predictions,
                )
                # Draw histograms
                self._draw_slide_histograms(
                    outcome=config['outcomes'][0],
                    all_labels=config['outcome_labels'],
                    active_label=config['outcome_labels'][str(self.pred_idx[outcome])],
                    pred_array=None if not (viz.heatmap and hw.predictions is not None) else hw.predictions[:, :, self.pred_idx[config['outcomes'][0]]],
                    uq_array=None if not (viz.heatmap and hw.uncertainty is not None) else hw.uncertainty
                )
            # Multiple linear outcome(s)
            else:
                for o_idx, outcome in enumerate(config['outcomes']):
                    # Render predictions as text
                    self._draw_prediction_as_text(
                        outcome=outcome,
                        all_labels=None,
                        pred_array=None if hw.predictions is None else hw.predictions[:, :, o_idx],
                    )
                # Draw histograms
                self._draw_slide_histograms(
                    outcome='linear',
                    all_labels=config['outcomes'],
                    active_label=config['outcomes'][self.pred_idx['linear']],
                    pred_array=None if not (viz.heatmap and hw.predictions is not None) else hw.predictions[:, :, self.pred_idx['linear']],
                    uq_array=None if not (viz.heatmap and hw.uncertainty is not None) else hw.uncertainty[:, :, self.pred_idx['linear']]
                )

            imgui_utils.vertical_break()

        if config is not None:

            txt = "Predict Slide" if (not viz.heatmap_widget._triggered) else "Predicting Slide..."
            if viz.sidebar.full_button(txt, enabled=(not viz.heatmap_widget._triggered and viz.wsi and viz._use_model)):
                viz.heatmap_widget.generate()

            imgui_utils.vertical_break()

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
                self._show_download = True
            if imgui.menu_item('Close model')[0]:
                viz.close_model(True)
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
                self._show_download = True

        elif show:
            if viz.sidebar.collapsing_header('Info', default=True):
                self.draw_info()
            if viz.sidebar.collapsing_header('Tile Prediction', default=True):
                self.draw_tile_predictions()
            if viz.sidebar.collapsing_header('Slide Prediction', default=True):
                self.draw_slide_predictions()
            if viz.sidebar.collapsing_header('Saliency', default=False):
                self.draw_saliency()

        if self._show_params and self.viz._model_config:
            self.draw_params_popup()
        self.download_popup()

