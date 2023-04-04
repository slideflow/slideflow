import re
import os
import imgui
from os.path import basename

from .._renderer import CapturedException
from ..gui import imgui_utils
from ..utils import EasyDict

import slideflow as sf

#----------------------------------------------------------------------------

class ProjectWidget:
    def __init__(self, viz):
        self.viz                = viz
        self.search_dirs        = []
        self.project_path       = ''
        self.browse_cache       = dict()
        self.browse_refocus     = False
        self.P                  = None
        self.slide_paths        = []
        self.model_paths        = []
        self.slide_idx          = 0
        self.model_idx          = 0
        self.content_height     = 0
        self._show_welcome      = False

    def load(self, project, ignore_errors=False):
        viz = self.viz
        viz.clear_result()
        viz.skip_frame() # The input field will change on next frame.
        if project == '':
            viz.result = EasyDict(message='No project loaded')
            return
        try:
            self.project_path = project
            viz.defer_rendering()
            sf.log.debug("Loading project at {}...".format(project))
            self.P = sf.Project(project)
            self.slide_paths = sorted(self.P.dataset().slide_paths())
            self.viz.create_toast(f"Loaded project at {project}", icon="success")

        except Exception:
            self.project_path = project
            self.viz.create_toast(f"Unable to load project at {project}", icon="error")
            viz.result = EasyDict(error=CapturedException())
            if not ignore_errors:
                raise

    def recursive_model_scan(self):
        viz = self.viz

        def recurse(parents, dryrun=False):
            key = tuple(parents)
            items = self.browse_cache.get(key, None)
            if items is None:
                items = self._list_runs_and_models(parents)
                self.browse_cache[key] = items

            has_model = False
            recurse_checks = []

            for item in items:
                if item.type == 'run':
                    _recurse_has_models = recurse([item.path], dryrun=True)
                    recurse_checks.append(_recurse_has_models)
                    if _recurse_has_models and not dryrun and imgui.tree_node(item.name):
                        recurse([item.path])
                        imgui.tree_pop()
                if item.type == 'model':
                    has_model = True
                    if not dryrun:
                        clicked, _state = imgui.menu_item(item.name)
                        if clicked:
                            self.viz.load_model(item.path)

            return any(recurse_checks) or has_model

        result = recurse([self.P.models_dir])
        if self.browse_refocus:
            imgui.set_scroll_here()
            viz.skip_frame() # Focus will change on next frame.
            self.browse_refocus = False
        return result

    def _list_runs_and_models(self, parents):
        items = []
        run_regex = re.compile(r'\d+-.*')
        params_regex = re.compile(r'params\.json')
        zip_regex = re.compile(r'.*\.zip')
        for parent in set(parents):
            if os.path.isdir(parent):
                for entry in os.scandir(parent):
                    # Check if entry is a model training run
                    if entry.is_dir() and run_regex.fullmatch(entry.name):
                        items.append(EasyDict(type='run', name=entry.name, path=os.path.join(parent, entry.name)))

                    # Check if entry is a Tensorflow model (directory with a params.json inside)
                    elif entry.is_dir():
                        for model_file in os.scandir(os.path.join(parent, entry.name)):
                            if model_file.is_file() and params_regex.fullmatch(model_file.name):
                                items.append(EasyDict(type='model', name=entry.name, path=os.path.join(parent, entry.name)))

                    # Check if entry is a Torch model (*zip file)
                    elif entry.is_file() and zip_regex.fullmatch(entry.name):
                        items.append(EasyDict(type='model', name=entry.name, path=os.path.join(parent, entry.name)))

        items = sorted(items, key=lambda item: (item.name.replace('_', ' '), item.path))
        return items

    def draw_slide_list(self):
        for path in self.slide_paths:
            if imgui.menu_item(imgui_utils.ellipsis_clip(sf.util.path_to_name(path), 33))[0]:
                self.viz.load_slide(path)
            if imgui.is_item_hovered():
                imgui.set_tooltip(path)


    def draw_info(self):
        viz = self.viz
        config = viz._model_config

        imgui.text_colored('Name', *viz.theme.dim)
        imgui.same_line(viz.font_size * 6)
        with imgui_utils.clipped_with_tooltip(self.P.name, 22):
            imgui.text(imgui_utils.ellipsis_clip(self.P.name, 22))

        imgui.text_colored('Path', *viz.theme.dim)
        imgui.same_line(viz.font_size * 6)
        with imgui_utils.clipped_with_tooltip(self.P.root, 22):
            imgui.text(imgui_utils.ellipsis_clip(self.P.root, 22))

        imgui.text_colored('Annotations', *viz.theme.dim)
        imgui.same_line(viz.font_size * 6)
        imgui.text(imgui_utils.ellipsis_clip(basename(self.P.annotations), 22))
        if imgui.is_item_hovered():
            imgui.set_tooltip(self.P.annotations)

        imgui.text_colored('Dataset config', *viz.theme.dim)
        imgui.same_line(viz.font_size * 6)
        imgui.text(imgui_utils.ellipsis_clip(basename(self.P.dataset_config), 22))
        if imgui.is_item_hovered():
            imgui.set_tooltip(self.P.dataset_config)

        imgui.text_colored('Sources', *viz.theme.dim)
        imgui.same_line(viz.font_size * 6)
        source_str = str(self.P.sources)
        with imgui_utils.clipped_with_tooltip(source_str, 22):
            imgui.text(imgui_utils.ellipsis_clip(source_str, 22))

        imgui.text_colored('Slides', *viz.theme.dim)
        imgui.same_line(viz.font_size * 6)
        imgui.text(str(len(self.slide_paths)))

        imgui_utils.vertical_break()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            if self.P is None:
                viz.header("Project")
            if self.P is not None:
                with viz.header_with_buttons("Project"):
                    imgui.same_line(imgui.get_content_region_max()[0] - viz.font_size*1.5)
                    cx, cy = imgui.get_cursor_pos()
                    imgui.set_cursor_position((cx, cy-int(viz.font_size*0.25)))
                    if viz.sidebar.small_button('refresh'):
                        self.load(self.project_path)

        if show and self.P is None:
            imgui_utils.padded_text('No project has been loaded.', vpad=[int(viz.font_size/2), int(viz.font_size)])
            if viz.sidebar.full_button("Load a Project"):
                viz.ask_load_project()

        elif show:
            if viz.collapsing_header('Info', default=True):
                self.draw_info()

            if viz.collapsing_header('Slides', default=False):
                if not len(self.slide_paths):
                    imgui_utils.padded_text('No slides found.', vpad=[int(viz.font_size/2), int(viz.font_size)])
                else:
                    self.draw_slide_list()

            if viz.collapsing_header('Models', default=False):
                if not self.recursive_model_scan():
                    imgui_utils.padded_text('No models found.', vpad=[int(viz.font_size/2), int(viz.font_size)])