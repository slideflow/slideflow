import imgui

from . import renderer
from .utils import EasyDict
from .gui_utils import imgui_utils

import slideflow as sf

#----------------------------------------------------------------------------

class ProjectWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.search_dirs    = []
        self.cur_project      = None
        self.user_project     = ''
        self.recent_projects  = []
        self.browse_cache   = dict()
        self.browse_refocus = False
        self.cur_project    = None
        self.P              = None
        self.slide_paths    = []
        self.slide_idx      = 0
        self.content_height = 0
        self._show_welcome  = False

    def add_recent(self, project, ignore_errors=False):
        try:
            if project not in self.recent_projects:
                self.recent_projects.append(project)
        except:
            if not ignore_errors:
                raise

    def disclaimer(self):
        if self._show_welcome:
            imgui.open_popup('disclaimer_popup')
            imgui.set_next_window_position(self.viz.content_width/2, self.viz.content_height/2)

            if imgui.begin_popup('disclaimer_popup'):

                imgui.text('Welcome to Workbench!')
                imgui.separator()
                imgui.text('This is an early preview under active development.\n'
                        'Please be aware there may be bugs or other issues. ')
                imgui.text('')
                imgui.same_line((imgui.get_content_region_max()[0])/2 - (self.viz.button_w/2) + self.viz.spacing)
                if imgui.button('Proceed'):
                    self._show_welcome = False
                imgui.end_popup()

    def load(self, project, ignore_errors=False):
        viz = self.viz
        viz.clear_result()
        viz.skip_frame() # The input field will change on next frame.
        if project == '':
            viz.result = EasyDict(message='No project loaded')
            return
        try:
            self.cur_project = project
            self.user_project = project

            viz.defer_rendering()
            if project in self.recent_projects:
                self.recent_projects.remove(project)
            self.recent_projects.insert(0, project)

            sf.log.debug("Loading project at {}...".format(project))
            self.P = sf.Project(project)
            self.slide_paths = sorted(self.P.dataset().slide_paths())
            viz.model_widget.search_dirs = [self.P.models_dir]
            viz.slide_widget.project_slides = self.slide_paths
            self.viz.create_toast(f"Loaded project at {project}", icon="success")

        except Exception:
            self.cur_project = None
            self.user_project = project
            self.viz.create_toast(f"Unable to load project at {project}", icon="error")
            viz.result = EasyDict(error=renderer.CapturedException())
            if not ignore_errors:
                raise

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        recent_projects = [project for project in self.recent_projects if project != self.user_project]
        self.disclaimer()
        if show:
            self.content_height = viz.font_size + viz.spacing
            dim_color = list(imgui.get_style().colors[imgui.COLOR_TEXT])
            dim_color[-1] *= 0.5

            imgui.text('Project')
            imgui.same_line(viz.label_w)
            changed, self.user_project = imgui_utils.input_text('##project', self.user_project, 1024,
                flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                width=(-1 - viz.button_w * 1 - viz.spacing * 1),
                help_text='<PATH>')
            if changed:
                self.load(self.user_project, ignore_errors=True)
            if imgui.is_item_hovered() and not imgui.is_item_active() and self.user_project != '':
                imgui.set_tooltip(self.user_project)
            imgui.same_line()
            if imgui_utils.button('Recent...', width=viz.button_w, enabled=(len(recent_projects) != 0)):
                imgui.open_popup('recent_projects_popup')
        else:
            self.content_height = 0

        if imgui.begin_popup('recent_projects_popup'):
            for project in recent_projects:
                clicked, _state = imgui.menu_item(project)
                if clicked:
                    self.load(project, ignore_errors=True)
            imgui.end_popup()

#----------------------------------------------------------------------------
