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

from PIL import Image
from . import renderer
from .utils import EasyDict
from .gui_utils import imgui_utils

import slideflow as sf

#----------------------------------------------------------------------------

def _locate_results(pattern):
    return pattern

#----------------------------------------------------------------------------

class SlideWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.search_dirs    = []
        self.cur_slide        = None
        self.user_slide       = ''
        self.project_slides = []
        self.browse_cache   = dict()
        self.browse_refocus = False
        self.normalize_wsi  = False
        self.norm_idx       = 0
        self.qc_idx         = 0
        self.alpha          = 1.0
        self.show_qc        = False
        self._all_normalizer_methods = [
            'reinhard',
            'reinhard_fast',
            'reinhard_mask',
            'reinhard_fast_mask',
            'macenko',
            'vahadane_spams',
            'vahadane_sklearn',
            'augment']
        self._all_normalizer_methods_str = [
            'Reinhard',
            'Reinhard (Fast)',
            'Reinhard Mask',
            'Reinhard Mask (Fast)',
            'Macenko',
            'Vahadane (SPAMS)',
            'Vahadane (Sklearn)',
            'Augment']
        self._normalizer_methods = self._all_normalizer_methods
        self._normalizer_methods_str = self._all_normalizer_methods_str
        self._qc_methods_str    = ['Blur filter', "Otsu threshold", "Blur + Otsu"]
        self._qc_methods        = ['blur', 'otsu', 'both']
        self.load('', ignore_errors=True)

    def load(self, slide, ignore_errors=False):
        viz = self.viz
        viz.clear_result()
        viz.skip_frame() # The input field will change on next frame.
        try:
            name = slide.replace('\\', '/').split('/')[-1]
            self.cur_slide = slide
            self.user_slide = slide
            viz.result.message = f'Loading {name}...'
            viz.defer_rendering()
            tile_px = viz.tile_px if viz.tile_px else 100
            tile_um = viz.tile_um if viz.tile_um else 100
            viz._slide_path = slide
            viz.wsi = sf.WSI(slide, tile_px=tile_px, tile_um=tile_um, verbose=False)
            viz.reset_thumb(width=viz.content_width)
            viz.heatmap_widget.reset()

            # Generate WSI thumbnail.
            hw_ratio = (viz.wsi.dimensions[0] / viz.wsi.dimensions[1])
            max_width = int(min(800 - viz.spacing*2, (800 - viz.spacing*2) / hw_ratio))
            viz.wsi_thumb = np.asarray(viz.wsi.thumb(width=max_width))

        except:
            self.cur_slide = None
            self.user_slide = slide
            if slide == '':
                viz.result = EasyDict(message='No slide loaded')
            else:
                viz.result = EasyDict(error=renderer.CapturedException())
            if not ignore_errors:
                raise

    def hide_model_normalizer(self):
        self._normalizer_methods = self._all_normalizer_methods
        self._normalizer_methods_str = self._all_normalizer_methods_str
        self.norm_idx = min(self.norm_idx, len(self._normalizer_methods)-1)

    def show_model_normalizer(self):
        self._normalizer_methods = self._all_normalizer_methods + ['model']
        self._normalizer_methods_str = self._all_normalizer_methods_str + ['<Model>']

    def change_normalizer(self):
        method = self._normalizer_methods[self.norm_idx]
        if method == 'model':
            self.viz._normalizer = sf.util.get_model_normalizer(self.viz._model_path)
        else:
            self.viz._normalizer = sf.norm.autoselect(method, source='v2')

    def generate_qc(self):
        if self.show_qc and self.viz.wsi is not None:
            self.viz.wsi.remove_qc()
            qc_mask = np.asarray(self.viz.wsi.qc(self._qc_methods[self.qc_idx]))
            alpha_channel = (qc_mask).astype(np.uint8) * 255 * 255
            self.viz.rendered_qc = np.asarray(Image.fromarray(~qc_mask).convert('RGB')) * 255
            self.viz.overlay_heatmap = np.dstack((self.viz.rendered_qc[:, :, 0:3], alpha_channel))
        self.viz._show_overlay = self.show_qc

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            imgui.text('Slide')
            imgui.same_line(viz.label_w)
            changed, self.user_slide = imgui_utils.input_text('##slide', self.user_slide, 1024,
                flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                width=(-1 - viz.button_w * 2 - viz.spacing * 2),
                help_text='<PATH>.svs')
            if changed:
                self.load(self.user_slide, ignore_errors=True)
            if imgui.is_item_hovered() and not imgui.is_item_active() and self.user_slide != '':
                imgui.set_tooltip(self.user_slide)
            imgui.same_line()
            if imgui_utils.button('Project...', width=viz.button_w, enabled=(viz.project_widget.P is not None)):
                imgui.open_popup('project_slides_popup')
            imgui.same_line()
            if imgui_utils.button('Browse...', enabled=len(self.search_dirs) > 0, width=-1):
                imgui.open_popup('browse_slides_popup')
                self.browse_cache.clear()
                self.browse_refocus = True

            dim_color = list(imgui.get_style().colors[imgui.COLOR_TEXT])
            dim_color[-1] *= 0.5

            # WSI thumbnail ===================================================
            width = viz.font_size * 28
            height = imgui.get_text_line_height_with_spacing() * 12 + viz.spacing
            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, [0, 0])
            imgui.push_style_color(imgui.COLOR_HEADER, 0, 0, 0, 0)
            imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, 0.16, 0.29, 0.48, 0.5)
            imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, 0.16, 0.29, 0.48, 0.9)
            imgui.begin_child('##slide_thumb', width=width, height=height, border=True)

            if viz.wsi_thumb is not None:
                hw_ratio = (viz.wsi_thumb.shape[0] / viz.wsi_thumb.shape[1])
                max_width = min(width - viz.spacing*2, (height - viz.spacing*2) / hw_ratio)
                max_height = max_width * hw_ratio

                if viz._wsi_tex_obj is not None:
                    imgui.same_line(int((width - max_width)/2))
                    imgui.image(viz._wsi_tex_obj.gl_id, max_width, max_height)

            if viz.wsi_thumb is None:
                imgui.text_colored('Slide not loaded', *dim_color)
            imgui.end_child()
            imgui.pop_style_color(3)
            imgui.pop_style_var(1)
            # =================================================================

            # Slide options and properties (child) ============================
            imgui.same_line()
            imgui.begin_child('##slide_options', width=-1, height=height, border=False)

            # Slide properties (sub-child). -----------------------------------
            if viz.wsi is not None:
                prop = viz.wsi.properties
                vals = [
                    str(prop['width']),
                    str(prop['height']),
                    str(viz.wsi.mpp),
                    viz.wsi.vendor if viz.wsi.vendor is not None else '-',
                    f'{10 / (viz.wsi.slide.level_downsamples[0] * viz.wsi.mpp):.1f}x',
                    str(viz.wsi.slide.level_count),
                    '-' if viz._model is None else str(viz.wsi.estimated_num_tiles),
                    '-'
                ]
            else:
                vals = ["-" for _ in range(8)]
            rows = [
                ['Property',     'Value'],
                ['Width',         vals[0]],
                ['Height',        vals[1]],
                ['MPP',           vals[2]],
                ['Scanner',       vals[3]],
                ['Magnification', vals[4]],
                ['Levels',        vals[5]],
                ['Est. tiles',    vals[6]],
                ['ROIs',          vals[7]],
            ]
            height = imgui.get_text_line_height_with_spacing() * len(rows) + viz.spacing
            imgui.begin_child('##slide_properties', width=-1, height=height, border=True)
            for y, cols in enumerate(rows):
                for x, col in enumerate(cols):
                    if x != 0:
                        imgui.same_line(viz.font_size * (8 + (x - 1) * 6))
                    if x == 0 or y == 0:
                        imgui.text_colored(col, *dim_color)
                    else:
                        imgui.text(col)
            imgui.end_child()
            # -----------------------------------------------------------------

            _norm_clicked, self.normalize_wsi = imgui.checkbox('Normalize', self.normalize_wsi)
            viz._normalize_wsi = self.normalize_wsi
            if _norm_clicked:
                viz._refresh_thumb_flag = True
            imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.font_size*8)
            with imgui_utils.item_width(viz.font_size * 8), imgui_utils.grayed_out(not self.normalize_wsi):
                _norm_method_clicked, self.norm_idx = imgui.combo("##norm_method", self.norm_idx, self._normalizer_methods_str)
            if _norm_method_clicked:
                self.change_normalizer()
                viz._refresh_thumb_flag = True

            _qc_clicked, self.show_qc = imgui.checkbox('Show QC', self.show_qc)
            imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.font_size*8)
            with imgui_utils.item_width(viz.font_size * 8), imgui_utils.grayed_out(not self.show_qc):
                _qc_method_clicked, self.qc_idx = imgui.combo("##qc_method", self.qc_idx, self._qc_methods_str)
            if _qc_clicked or _qc_method_clicked:
                self.generate_qc()

            # End slide options and properties
            imgui.end_child()
            # =================================================================

        if imgui.begin_popup('project_slides_popup'):
            for slide in self.project_slides:
                clicked, _state = imgui.menu_item(slide)
                if clicked:
                    self.load(slide, ignore_errors=True)
            imgui.end_popup()

        if imgui.begin_popup('browse_slides_popup'):
            def recurse(parents):
                key = tuple(parents)
                items = self.browse_cache.get(key, None)
                if items is None:
                    items = self.list_runs_and_slides(parents)
                    self.browse_cache[key] = items
                for item in items:
                    if item.type == 'run' and imgui.begin_menu(item.name):
                        recurse([item.path])
                        imgui.end_menu()
                    if item.type == 'slide':
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

        paths = viz.pop_drag_and_drop_paths()
        if paths is not None and len(paths) >= 1:
            self.load(paths[0], ignore_errors=True)

    def list_runs_and_slides(self, parents):
        items = []
        run_regex = re.compile(r'\d+-.*')
        slide_regex = re.compile(r'network-snapshot-\d+\.slide')
        for parent in set(parents):
            if os.path.isdir(parent):
                for entry in os.scandir(parent):
                    if entry.is_dir() and run_regex.fullmatch(entry.name):
                        items.append(EasyDict(type='run', name=entry.name, path=os.path.join(parent, entry.name)))
                    if entry.is_file() and slide_regex.fullmatch(entry.name):
                        items.append(EasyDict(type='slide', name=entry.name, path=os.path.join(parent, entry.name)))

        items = sorted(items, key=lambda item: (item.name.replace('_', ' '), item.path))
        return items

#----------------------------------------------------------------------------
