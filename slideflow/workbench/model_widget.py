# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import re
import imgui
import numpy as np
from array import array

from . import renderer
from .gui_utils import imgui_utils
from .utils import EasyDict

import slideflow as sf
import slideflow.grad as grad

#----------------------------------------------------------------------------

def _locate_results(pattern):
    return pattern

#----------------------------------------------------------------------------

class ModelWidget:
    def __init__(self, viz, show_preview=False, show_saliency=True):
        self.viz                = viz
        self.show_preview       = show_preview
        self.show_saliency      = show_saliency
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
        self._show_params       = False
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

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):

        viz = self.viz
        dim_color = list(imgui.get_style().colors[imgui.COLOR_TEXT])
        dim_color[-1] *= 0.5
        recent_models = [model for model in self.recent_models if model != self.user_model]

        if self._show_params and self.viz._model_config:
            hp = self.viz._model_config['hp']
            rows = list(zip(list(map(str, hp.keys())), list(map(str, hp.values()))))

            _, self._show_params = imgui.begin("Model parameters", closable=True, flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_SCROLLBAR)
            for y, cols in enumerate(rows):
                for x, col in enumerate(cols):
                    if x != 0:
                        imgui.same_line(viz.font_size * 10)
                    if x == 0:
                        imgui.text_colored(col, *dim_color)
                    else:
                        imgui.text(col)
            imgui.end()

        if show:
            self.content_height = imgui.get_text_line_height_with_spacing() * 8 + viz.spacing * 2
            imgui.text('Model')
            imgui.same_line(viz.label_w)
            changed, self.user_model = imgui_utils.input_text('##model', self.user_model, 1024,
                flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                width=(-1 - viz.button_w * 2 - viz.spacing * 2),
                help_text='<PATH>')
            if changed:
                self.load(self.user_model, ignore_errors=True)
            if imgui.is_item_hovered() and not imgui.is_item_active() and self.user_model != '':
                imgui.set_tooltip(self.user_model)
            imgui.same_line()
            if imgui_utils.button('Recent...', width=viz.button_w, enabled=(len(recent_models) != 0)):
                imgui.open_popup('recent_models_popup')
            imgui.same_line()
            if imgui_utils.button('Browse...', enabled=len(self.search_dirs) > 0, width=-1):
                imgui.open_popup('browse_models_popup')
                self.browse_cache.clear()
                self.browse_refocus = True

            # Predictions =====================================================
            config = viz._model_config
            if config is not None:
                if viz._use_model and viz._predictions is not None and isinstance(viz._predictions, list):
                    for p, _pred_array in enumerate(viz._predictions):
                        self.content_height += (imgui.get_text_line_height_with_spacing() + viz.spacing)
                        imgui.text(f'Pred {p}')
                        imgui.same_line(viz.label_w)
                        imgui.core.plot_histogram('##pred', array('f', _pred_array), scale_min=0, scale_max=1)
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
                elif viz._use_model and viz._predictions is not None:
                    self.content_height += (imgui.get_text_line_height_with_spacing() + viz.spacing)
                    imgui.text('Prediction')
                    imgui.same_line(viz.label_w)
                    imgui.core.plot_histogram('##pred', array('f', viz._predictions), scale_min=0, scale_max=1)
                    ol = config['outcome_labels']
                    pred_str = ol[str(np.argmax(viz._predictions))]
                    if viz._use_uncertainty and viz._uncertainty is not None:
                        pred_str += " (UQ: {:.4f})".format(viz._uncertainty)
                    imgui.same_line(imgui.get_content_region_max()[0] - viz.spacing - imgui.calc_text_size(pred_str).x)
                    imgui.text(pred_str)

                # Image preview ===================================================
                width = viz.font_size * 20
                height = imgui.get_text_line_height_with_spacing() * 7
                if self.show_preview:
                    imgui.begin_child('##pred_image', width=width, height=height, border=False)
                    if viz._tex_obj is not None and viz.tile_px:
                        imgui.image(viz._tex_obj.gl_id, viz.tile_px, viz.tile_px)
                    elif viz._model_path is not None:
                        imgui.text_colored('Right click to preview', *dim_color)
                    else:
                        imgui.text_colored('No model loaded', *dim_color)
                    imgui.same_line()
                    if self.use_model and viz._normalizer is not None and viz._norm_tex_obj is not None and viz.tile_px:
                        imgui.image(viz._norm_tex_obj.gl_id, viz.tile_px, viz.tile_px)
                    elif viz._tex_obj is not None and viz.tile_px:
                        imgui.text_colored('Normalizer not used', *dim_color)
                    imgui.end_child()
                    imgui.same_line()

                # Model info / options ============================================
                imgui.begin_child('##model_options', width=-1, height=height, border=False)

                # Model properties (sub-child). -----------------------------------
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
                height = imgui.get_text_line_height_with_spacing() * len(rows) + viz.spacing * 2
                imgui.begin_child('##model_properties', width=imgui.get_content_region_max()[0] / 2 - viz.spacing, height=height, border=True)
                for y, cols in enumerate(rows):
                    for x, col in enumerate(cols):
                        if x != 0:
                            imgui.same_line(viz.font_size * (6 + (x - 1) * 6))
                        if x == 0: # or y == 0:
                            imgui.text_colored(col, *dim_color)
                        else:
                            imgui.text(col)

                with imgui_utils.grayed_out(viz._model_path is None):
                    imgui.same_line(imgui.get_content_region_max()[0] - viz.font_size - viz.spacing * 2)
                    if imgui.button("HP") and self.viz._model_config:
                        self._show_params = not self._show_params
                imgui.end_child()

                # -----------------------------------------------------------------

                imgui.same_line(imgui.get_content_region_max()[0] / 2 + viz.spacing)
                imgui.begin_child('##model_settings', width=imgui.get_content_region_max()[0] / 2 - viz.spacing, height=height, border=False)
                imgui.text('Model')
                imgui.same_line(viz.label_w - viz.font_size)

                with imgui_utils.item_width(viz.font_size * 5), imgui_utils.grayed_out(viz._model_path is None):
                    _clicked, self.use_model = imgui.checkbox('Enable##model', self.use_model)
                    viz._use_model = self.use_model

                with imgui_utils.grayed_out(not viz.has_uq()):
                    imgui.same_line(viz.label_w - viz.font_size + viz.font_size * 5)
                    _clicked, _uq = imgui.checkbox('Enable UQ', self.use_uncertainty)
                    if _clicked and viz.has_uq():
                        viz._use_uncertainty = self.use_uncertainty = _uq

                # Saliency --------------------------------------------------------
                if self.show_saliency:
                    imgui.text('Saliency')
                    imgui.same_line(viz.label_w - viz.font_size)
                    with imgui_utils.grayed_out(viz._model_path is None), imgui_utils.item_width(viz.font_size * 5):
                        _clicked, self.enable_saliency = imgui.checkbox('Enable##saliency', self.enable_saliency)

                    imgui.same_line(viz.label_w - viz.font_size + viz.font_size * 5)
                    with imgui_utils.grayed_out(not self.enable_saliency):
                        _clicked, self.saliency_overlay = imgui.checkbox('Overlay', self.saliency_overlay)

                    imgui.text('')
                    imgui.same_line(viz.label_w - viz.font_size)
                    with imgui_utils.item_width(imgui.get_content_region_max()[0] - (viz.label_w - viz.font_size) - 1), imgui_utils.grayed_out(not self.enable_saliency):
                        _clicked, self.saliency_idx = imgui.combo("##method", self.saliency_idx, self._saliency_methods_names)

                imgui.end_child()
                imgui.end_child()
            else:
                self.content_height = imgui.get_text_line_height_with_spacing() + viz.spacing
        else:
            self.content_height = 0

        if imgui.begin_popup('recent_models_popup'):
            for model in recent_models:
                clicked, _state = imgui.menu_item(model)
                if clicked:
                    self.load(model, ignore_errors=True)
            imgui.end_popup()

        if imgui.begin_popup('browse_models_popup'):
            def recurse(parents):
                key = tuple(parents)
                items = self.browse_cache.get(key, None)
                if items is None:
                    items = self.list_runs_and_models(parents)
                    self.browse_cache[key] = items
                for item in items:
                    if item.type == 'run' and imgui.begin_menu(item.name):
                        recurse([item.path])
                        imgui.end_menu()
                    if item.type == 'model':
                        clicked, _state = imgui.menu_item(item.name)
                        if clicked:
                            self.load(item.path, ignore_errors=True)
                if len(items) == 0:
                    with imgui_utils.grayed_out():
                        imgui.menu_item('No results found')
            recurse(self.search_dirs)
            if self.browse_refocus:
                imgui.set_scroll_here()
                viz.skip_frame() # Focus will change on next frame.
                self.browse_refocus = False
            imgui.end_popup()

        viz._use_saliency = self.enable_saliency
        viz.args.saliency_method = self.saliency_idx
        viz.args.saliency_overlay = self.saliency_overlay

    def list_runs_and_models(self, parents):
        items = []
        run_regex = re.compile(r'\d+-.*')
        params_regex = re.compile(r'params\.json')
        zip_regex = re.compile(r'.*\.zip')
        for parent in set(parents):
            if os.path.isdir(parent):
                for entry in os.scandir(parent):
                    if entry.is_dir() and run_regex.fullmatch(entry.name):
                        items.append(EasyDict(type='run', name=entry.name, path=os.path.join(parent, entry.name)))
                    elif entry.is_dir():
                        for model_file in os.scandir(os.path.join(parent, entry.name)):
                            if model_file.is_file() and params_regex.fullmatch(model_file.name):
                                items.append(EasyDict(type='model', name=entry.name, path=os.path.join(parent, entry.name)))
                    elif entry.is_file() and zip_regex.fullmatch(entry.name):
                        items.append(EasyDict(type='model', name=entry.name, path=os.path.join(parent, entry.name)))

        items = sorted(items, key=lambda item: (item.name.replace('_', ' '), item.path))
        return items

#----------------------------------------------------------------------------
