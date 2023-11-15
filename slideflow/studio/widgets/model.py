import imgui
import numpy as np
import slideflow as sf
import slideflow.grad as grad
from typing import Union, List, Optional, Tuple
from array import array
from collections import defaultdict
from slideflow.util import isnumeric

from ..utils import EasyDict
from ..gui import imgui_utils
from ..gui.annotator import AnnotationCapture

# -----------------------------------------------------------------------------

def scale_uncertainty_bar(val, max_width, range=(0, 1)):
    """Scale a value to a given range and width."""
    _min, _max = range
    _normalized = (val - _min) / (_max - _min)
    _clipped = max(0, min(1, _normalized))
    return int(_clipped * max_width)


def _draw_tile_pred_result(
    viz,
    outcome: Union[str, List[str]],
    labels: List[str],
    is_categorical: bool,
    pred_array: np.ndarray,
    uq_array: Optional[np.ndarray] = None,
    *,
    uncertainty_color: Optional[List[int]] = None,
    uncertainty_range: Optional[Tuple[float, float]] = None,
    uncertainty_label: str = 'Uncertainty'
):
    """Render a tile prediction result with Imgui."""

    config = viz._model_config
    out_of_focus = hasattr(viz.result, 'in_focus') and not viz.result.in_focus

    if uncertainty_color is None:
        uncertainty_color = (0, 0, 0, 1)

    # Outcome name label
    imgui.text_colored(outcome, *viz.theme.dim)

    # Prediction string
    if is_categorical:
        pred_str = labels[str(np.argmax(pred_array))]
    else:
        pred_str = f'{pred_array:.3f}'
    imgui.same_line(imgui.get_content_region_max()[0] - viz.spacing - imgui.calc_text_size(pred_str).x)
    with viz.dim_text(out_of_focus):
        imgui.text(pred_str)

    # Histogram
    if is_categorical:
        if out_of_focus:
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, 0.5, 0.5, 0.5)
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM_HOVERED, 0.6, 0.6, 0.6)
        with imgui_utils.item_width(imgui.get_content_region_max()[0] - viz.spacing):
            _histogram_size = imgui.get_content_region_max()[0] - viz.spacing, viz.font_size * 3
            imgui.core.plot_histogram(
                '##pred',
                array('f', pred_array),
                scale_min=0,
                scale_max=1,
                graph_size=_histogram_size
            )
        if out_of_focus:
            imgui.pop_style_color(2)

    # Uncertainty bar
    if uq_array is not None:
        # Uncertainty bar
        draw_list = imgui.get_window_draw_list()
        w = imgui.get_content_region_max()[0]
        x, y = imgui.get_cursor_screen_position()
        x += int(viz.spacing / 2)
        w -= viz.spacing * 2
        y -= viz.spacing
        if uncertainty_range is None:
            if config is not None and 'thresholds' in config and 'tile_uq' in config['thresholds']:
                uncertainty_range = (0, config['thresholds']['tile_uq'] * 2)
            else:
                uncertainty_range = (0, 0.033)
        width = scale_uncertainty_bar(uq_array, max_width=w, range=uncertainty_range)
        draw_list.add_rect_filled(x, y, x+width, y+7, imgui.get_color_u32_rgba(*uncertainty_color))

        # Right-aligned text below bar
        uq_text = "{}: {:.4f}".format(uncertainty_label, uq_array)
        with viz.dim_text(out_of_focus):
            imgui.text(uq_text)


def draw_tile_predictions(
    viz,
    is_categorical: bool,
    config: "EasyDict" = None,
    has_preds: bool = None,
    using_model: bool = None,
    **kwargs
):
    """Render tile predictions with Imgui."""
    if config is None:
        config = viz._model_config
    if has_preds is None:
        has_preds = viz._use_model and viz._predictions is not None
    if using_model is None:
        using_model = viz._use_model


    if config is not None:

        if hasattr(viz.result, 'in_focus') and not viz.result.in_focus:
            imgui.text("Image out of focus.")

        # Process outcomes
        out_names = config['outcomes']
        out_names = [out_names] if not isinstance(out_names, list) else out_names

        # Multiple categorical outcomes
        if has_preds and isinstance(viz._predictions, list):
            for p, _pred_array in enumerate(viz._predictions):
                _draw_tile_pred_result(
                    viz,
                    outcome=out_names[p],
                    labels=config['outcome_labels'][out_names[p]],
                    is_categorical=is_categorical,
                    pred_array=_pred_array,
                    uq_array=viz._uncertainty,
                    **kwargs

                )

        # Single categorical outcome
        elif has_preds and is_categorical:
            _draw_tile_pred_result(
                viz,
                outcome=out_names[0],
                labels=config['outcome_labels'],
                is_categorical=is_categorical,
                pred_array=viz._predictions,
                uq_array=viz._uncertainty,
                **kwargs
            )

        # Linear outcome(s)
        elif has_preds:
            for o_idx, outcome in enumerate(out_names):
                _draw_tile_pred_result(
                    viz,
                    outcome=outcome,
                    labels=None,
                    is_categorical=is_categorical,
                    pred_array=viz._predictions[o_idx],
                    uq_array=viz._uncertainty,
                    **kwargs
                )

        # Model not in use
        elif using_model:
            imgui_utils.padded_text('Right click for a focal prediction.', vpad=[int(viz.font_size/2), int(viz.font_size)])
        else:
            imgui_utils.padded_text('Model not in use.', vpad=[int(viz.font_size/2), int(viz.font_size)])

        imgui_utils.vertical_break()

# -----------------------------------------------------------------------------

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
        self.pred_means         = defaultdict(lambda: None)
        self.pred_arrays        = defaultdict(lambda: None)
        self.pred_hist          = defaultdict(lambda: None)
        self.uq_hist            = defaultdict(lambda: None)
        self._clicking          = False
        self._show_params       = False
        self._show_popup        = False
        self._show_download     = False
        self._last_preds        = None
        self.uncertainty_color  = (0, 0, 0, 1)

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
        self.pred_idx    = defaultdict(int)
        self.pred_means  = defaultdict(lambda: None)
        self.pred_arrays = defaultdict(lambda: None)
        self.pred_hist   = defaultdict(lambda: None)
        self.uq_hist     = defaultdict(lambda: None)

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

    def is_categorical(self):
        """Check if model is a categorical model."""
        return self.viz._model_config['model_type'] == 'categorical'

    # -------------------------------------------------------------------------

    @staticmethod
    def _masked_histogram(arr):
        # Prediction histogram
        flattened = arr.flatten()
        flattened = flattened[flattened != sf.heatmap.MASK]
        hist, _ = np.histogram(flattened, range=(0, 1))
        if flattened.shape[0] > 0:
            hist_arr = array('f', hist/np.sum(hist))
            hist_scale_max = np.max(hist/np.sum(hist))
            hist_avg = np.nanmean(flattened)
        else:
            hist_arr = array('f', [0])
            hist_scale_max = 1
            hist_avg = None
        return (hist_avg, hist_arr, hist_scale_max)

    def _apply_pred_means(self, outcome, pred_array):
        masked = np.ma.masked_where(((pred_array == sf.heatmap.MASK) | (pred_array == np.nan)), pred_array)
        if self.is_categorical():
            self.pred_means[outcome] = masked.mean(axis=(0,1)).filled()
        else:
            self.pred_means[outcome] = masked.mean()

    def _apply_pred_histograms(self, outcome, pred_array, uq_array=None):
        self.pred_hist[outcome] = self._masked_histogram(pred_array)
        if uq_array is not None:
            self.uq_hist[outcome] = self._masked_histogram(uq_array)

    def _update_slide_preds(self):
        self._update_slide_pred_means()
        self._update_slide_pred_histograms()

    def _update_slide_pred_means(self):
        config = self.viz._model_config
        hw = self.viz.heatmap_widget
        multiple_outcomes = len(config['outcomes']) > 1

        if hw.predictions is None:
            return

        if config is not None and self.viz._use_model:
            # Multiple categorical outcomes
            if multiple_outcomes and self.is_categorical():
                for outcome in config['outcomes']:
                    pred_array = hw.predictions[self.outcome_indices(outcome)].grid
                    self._apply_pred_means(outcome, pred_array)

            # Single categorical or linear outcome
            elif not multiple_outcomes:
                outcome = config['outcomes'][0]
                self._apply_pred_means(outcome, np.dstack([overlay.grid for overlay in hw.predictions]))

            # Multiple linear outcome(s)
            else:
                for o_idx, outcome in enumerate(config['outcomes']):
                    self._apply_pred_means(outcome, hw.predictions[o_idx].grid)

    def _update_slide_pred_histograms(self):
        config = self.viz._model_config
        hw = self.viz.heatmap_widget
        multiple_outcomes = len(config['outcomes']) > 1

        if hw.predictions is None:
            return

        if config is not None and self.viz._use_model:
            # Multiple categorical outcomes
            if multiple_outcomes and self.is_categorical():
                for outcome in config['outcomes']:
                    if self.viz.heatmap:
                        pred_array = hw.predictions[self.outcome_index_start(outcome)+self.pred_idx[outcome]].grid
                        if hw.uncertainty is not None:
                            uq_array = hw.uncertainty[self.outcome_index_start(outcome)+self.pred_idx[outcome]].grid
                        else:
                            uq_array = None
                        self._apply_pred_histograms(outcome, pred_array, uq_array)

            # Single categorical or linear outcome
            elif not multiple_outcomes:
                outcome = config['outcomes'][0]
                if self.viz.heatmap:
                    pred_array = hw.predictions[self.pred_idx[config['outcomes'][0]]].grid
                    if hw.uncertainty is not None:
                        uq_array = hw.uncertainty[0].grid
                    else:
                        uq_array = None
                    self._apply_pred_histograms(outcome, pred_array, uq_array)

            # Multiple linear outcome(s)
            else:
                if self.viz.heatmap:
                    pred_array = hw.predictions[self.pred_idx['linear']].grid
                    if hw.uncertainty is not None:
                        uq_array = hw.uncertainty[self.pred_idx['linear']].grid
                    else:
                        uq_array = None
                    self._apply_pred_histograms('linear', pred_array, uq_array)

    def _draw_prediction_as_text(self, outcome, all_labels):
        viz = self.viz
        imgui.text_colored(outcome, *viz.theme.dim)
        if viz.heatmap_widget._triggered:
            pred_str = imgui_utils.spinner_text()
        elif self.is_categorical():
            pred_str = all_labels[str(np.argmax(self.pred_means[outcome]))]
        else:
            pred_str = f'{self.pred_means[outcome]:.3f}'
        imgui.same_line(imgui.get_content_region_max()[0] - viz.spacing - imgui.calc_text_size(pred_str).x)
        imgui.text(pred_str)

    def _draw_slide_histograms(self, outcome, all_labels, active_label):
        viz = self.viz
        _histogram_size = imgui.get_content_region_max()[0] - viz.spacing, viz.font_size * 4
        _old_idx = self.pred_idx[outcome]

        # Slide prediction histogram and text
        if viz.heatmap and self.pred_hist[outcome]:

            # Outcome selection
            imgui.separator()
            if self.uq_hist[outcome]:
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
            if hpred > 0 and hpred < len(all_labels):
                self.pred_idx[outcome] = hpred - 1
            imgui.same_line()
            if imgui_utils.button(f'+##pred_idx{outcome}', width=narrow_w) and self.pred_idx[outcome] < (len(all_labels)-1):
                self.pred_idx[outcome] += 1

            # Histogram
            hist_avg, hist_arr, hist_scale_max = self.pred_hist[outcome]
            if hist_avg is not None:
                overlay_text = f"Average: {hist_avg:.2f}"
            else:
                overlay_text = f"Average: - "
            imgui.core.plot_histogram(
                f'##pred_histogram{outcome}',
                hist_arr,
                scale_min=0,
                overlay_text=overlay_text,
                scale_max=hist_scale_max,
                graph_size=_histogram_size
            )

        # Slide uncertainty histogram
        if viz.heatmap and self.uq_hist[outcome]:
            hist_avg, hist_arr, hist_scale_max = self.uq_hist[outcome]
            if hist_avg is not None:
                overlay_text = f"Average: {hist_avg:.2f}"
            else:
                overlay_text = "Average: - "
            imgui.separator()
            imgui.text("Uncertainty")
            imgui.core.plot_histogram(
                f'##uq_histogram{outcome}',
                hist_arr,
                scale_min=0,
                overlay_text=overlay_text,
                scale_max=hist_scale_max,
                graph_size=_histogram_size
            )
            imgui.separator()

        if _old_idx != self.pred_idx[outcome]:
            self._update_slide_pred_histograms()

    # -------------------------------------------------------------------------

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
            img_format = "<unknown>" if 'img_format' not in config else config['img_format']
        rows = [
            ['Outcomes',     '-' if config is None else outcomes],
            ['Tile (px)',    '-' if config is None else str(config['tile_px'])],
            ['Tile (um)',    '-' if config is None else str(config['tile_um'])],
            ['Image format', '-' if config is None else img_format],
            ['Backend',      '-' if config is None else self.backend],
            ['Version',      '-' if config is None else str(config['slideflow_version'])],
        ]
        imgui.text_colored('Model name', *viz.theme.dim)
        imgui.same_line(viz.font_size * 6)
        with imgui_utils.clipped_with_tooltip(config['model_name'], 22):
            imgui.text(imgui_utils.ellipsis_clip(config['model_name'], 22))
        for y, cols in enumerate(rows):
            for x, col in enumerate(cols):
                if x != 0:
                    imgui.same_line(viz.font_size * (6 + (x - 1) * 6))
                if x == 0:
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

    def draw_slide_predictions(self):
        viz = self.viz
        config = viz._model_config
        multiple_outcomes = len(config['outcomes']) > 1

        if config is not None and viz._use_model:

            # Multiple categorical outcomes
            if multiple_outcomes and self.is_categorical():
                for outcome in config['outcomes']:
                    # Render predictions as text
                    if self.pred_means[outcome] is not None:
                        self._draw_prediction_as_text(
                            outcome=outcome,
                            all_labels=config['outcome_labels'][outcome],
                        )
                        # Draw histograms
                        self._draw_slide_histograms(
                            outcome=outcome,
                            all_labels=config['outcome_labels'][outcome],
                            active_label=config['outcome_labels'][outcome][str(self.pred_idx[outcome])],
                        )

            # Single categorical or linear outcome
            elif not multiple_outcomes and self.pred_means[config['outcomes'][0]] is not None:
                outcome = config['outcomes'][0]
                # Render predictions as text
                self._draw_prediction_as_text(
                    outcome=config['outcomes'][0],
                    all_labels=config['outcome_labels'],
                )
                # Draw histograms
                self._draw_slide_histograms(
                    outcome=config['outcomes'][0],
                    all_labels=config['outcome_labels'],
                    active_label=config['outcome_labels'][str(self.pred_idx[outcome])],
                )
            # Multiple linear outcome(s)
            else:
                for o_idx, outcome in enumerate(config['outcomes']):
                    if self.pred_means[outcome] is not None:
                        # Render predictions as text
                        self._draw_prediction_as_text(
                            outcome=outcome,
                            all_labels=None,
                        )
                if self.pred_hist['linear'] is not None:
                    # Draw histograms
                    self._draw_slide_histograms(
                        outcome='linear',
                        all_labels=config['outcomes'],
                        active_label=config['outcomes'][self.pred_idx['linear']],
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
                '##model_config_popup',
                flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
            )
            if imgui.menu_item('Load model')[0]:
                viz.ask_load_model()
            if imgui.menu_item('Download model')[0]:
                self._show_download = True
            if imgui.menu_item('Close model')[0]:
                viz.close_model()
            imgui.separator()
            if imgui.menu_item('Enable model', enabled=has_model, selected=self.use_model)[0]:
                self.use_model = not self.use_model
                viz._use_model = self.use_model
            if imgui.menu_item('Enable UQ', enabled=has_model, selected=self.use_uncertainty)[0]:
                self.use_uncertainty = not self.use_uncertainty
                viz._use_uncertainty = self.use_uncertainty
                if not self.use_uncertainty:
                    viz._uncertainty = None
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

    def update_uncertainty_color(self):
        viz = self.viz
        c = viz._model_config
        val = viz._uncertainty

        if not viz._model_config:
            return

        if hasattr(viz.result, 'in_focus') and not viz.result.in_focus:
            color = (0.5, 0.5, 0.5, 1)
        elif isnumeric(val):
            if 'thresholds' in c and 'tile_uq' in c['thresholds']:
                uq_thresh = c['thresholds']['tile_uq']
            else:
                uq_thresh = 0.033
            if val < uq_thresh:
                color = (0, 1, 0, 1)
            elif val < uq_thresh * 2:
                color = (1, 1, 0, 1)
            else:
                color = (1, 0, 0, 1)
        else:
            color = (1, 0, 0, 1)

        self.uncertainty_color = color
        viz._box_color = color[0:3]

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        config = viz._model_config

        # Color uncertainty
        self.update_uncertainty_color()

        if show:
            with viz.header_with_buttons("Model"):
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
            if viz.collapsing_header('Info', default=True):
                self.draw_info()
            if viz.collapsing_header('Tile Prediction', default=True):
                draw_tile_predictions(
                    viz,
                    self.is_categorical(),
                    uncertainty_color=self.uncertainty_color
                )
            if viz.collapsing_header('Slide Prediction', default=True):
                self.draw_slide_predictions()
            if viz.collapsing_header('Saliency', default=False):
                self.draw_saliency()

        if self._show_params and self.viz._model_config:
            self.draw_params_popup()
        self.download_popup()

