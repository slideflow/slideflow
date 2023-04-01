import imgui

from .._renderer import CapturedException
from ..gui import imgui_utils
from ..utils import EasyDict

import slideflow as sf

#----------------------------------------------------------------------------

class ProjectWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.search_dirs    = []
        self.project_path     = ''
        self.recent_projects  = []
        self.browse_cache   = dict()
        self.browse_refocus = False
        self.P              = None
        self.slide_paths    = []
        self.slide_idx      = 0
        self.content_height = 0
        self._show_welcome  = False

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

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            viz.sidebar.header("Project")

        if show and self.P is None:
            imgui_utils.padded_text('No project has been loaded.', vpad=[int(viz.font_size/2), int(viz.font_size)])
            if viz.sidebar.full_button("Load a Project"):
                viz.ask_load_project()

        elif show:
            imgui.text("Project loaded!")

#----------------------------------------------------------------------------
    def nothing(self):
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