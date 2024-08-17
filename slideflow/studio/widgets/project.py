import re
import os
import imgui
import glfw
from os.path import basename
from tkinter.filedialog import askopenfilename, askdirectory

from .._renderer import CapturedException
from ..gui import imgui_utils
from ..utils import EasyDict, LEFT_MOUSE_BUTTON, RIGHT_MOUSE_BUTTON

import slideflow as sf

#----------------------------------------------------------------------------

class ProjectWidget:
    def __init__(self, viz):
        self.viz                    = viz
        self.search_dirs            = []
        self.project_path           = ''
        self.browse_cache           = dict()
        self.browse_refocus         = False
        self.P                      = None
        self.slide_paths            = []
        self.filtered_slide_paths   = []
        self.model_paths            = []
        self.content_height         = 0
        self.slide_search           = ''
        self._show_slide_filter_popup = False
        self._show_new_project      = False
        self._filter_by_has_roi     = None  # None = no filter, False = no ROIs, True = has ROIs
        self._clicking              = False
        self._new_project_path      = None
        self._new_annotations_path  = './annotations.csv'
        self._new_project_sources   = []
        self._new_project_n_slides  = 0
        self._dataset_config        = {}
        self._adding_source         = False
        self._add_source_name       = ''
        self._add_source_slides     = ''
        self._add_source_rois       = ''
        self._add_source_tfrecords  = ''
        self._add_source_n_slides   = 0

    @property
    def label_width(self) -> float:
        return self.viz.font_size * 5

    @property
    def button_width(self) -> float:
        return self.viz.font_size * 4

    @property
    def dataset_config(self) -> dict:
        config = {
            k: v for k, v in self._dataset_config.items()
            if k in self._new_project_sources
        }
        for source in config:
            if 'roi' not in config[source]:
                config[source]['roi'] = os.path.join(
                    sf.util.relative_path(
                        './roi', self._new_project_path
                    ),
                    source
                )
            if 'tfrecords' not in config[source]:
                config[source]['tfrecords'] = os.path.join(
                    sf.util.relative_path(
                        './tfrecords', self._new_project_path
                    ),
                    source
                )
        return config

    def end_create_project(self) -> None:
        """End the new project dialog."""
        if self._show_new_project:
            self.viz.resume_mouse_input_handling()
        self._show_new_project = False

    def ask_load_dataset_config(self) -> None:
        """Open a dialog to load a dataset configuration."""
        _path = askopenfilename(
            title="Dataset configuration", filetypes=[("JSON", "*.json",)]
        )
        if _path:
            try:
                self._dataset_config.update(sf.util.load_json(_path))
            except Exception as e:
                self.viz.create_toast(
                    f"Unable to load dataset configuration at {_path}",
                    icon="error"
                )
                sf.log.error(f"Unable to load dataset configuration at {_path}: {e}")
                raise

    def reset_new_project(self) -> None:
        """Reset the new project dialog."""
        self._new_project_path = None
        self._new_project_sources = []
        self._new_project_n_slides = 0
        self._new_annotations_path = './annotations.csv'
        self._dataset_config = {}

    def reset_add_source(self) -> None:
        """Reset the add source dialog."""
        self._add_source_name = ''
        self._add_source_slides = ''
        self._add_source_rois = ''
        self._add_source_tfrecords = ''
        self._add_source_n_slides = 0

    def new_project(self) -> None:
        """Open a dialog to create a new project."""
        self._show_new_project = True
        self.reset_new_project()

    def update_estimated_slides(self) -> None:
        """Recalculate and update the estimated patients/slides in the selected dataset source(s)."""
        self._new_project_n_slides = 0
        for source in self._new_project_sources:
            self._new_project_n_slides += len(sf.util.get_slide_paths(
                self._dataset_config[source]['slides']
            ))

    def keyboard_callback(self, key: int, action: int) -> None:
        """Handle keyboard events.

        Args:
            key (int): The key that was pressed. See ``glfw.KEY_*``.
            action (int): The action that was performed (e.g. ``glfw.PRESS``,
                ``glfw.RELEASE``, ``glfw.REPEAT``).

        """
        if self.viz._control_down and key == glfw.KEY_RIGHT and action == glfw.PRESS:
            # If the currently loaded slide is not in the filtered slide paths,
            # then start at the beginning.
            if self.viz.wsi is None or self.viz.wsi.path not in self.filtered_slide_paths:
                idx_to_load = 0 if len(self.filtered_slide_paths) else None
            # Otherwise, load the next slide in the filtered list.
            else:
                current_slide_idx = self.filtered_slide_paths.index(self.viz.wsi.path)
                idx_to_load = (current_slide_idx + 1) % len(self.filtered_slide_paths)
            if idx_to_load is not None:
                self.viz.load_slide(self.filtered_slide_paths[idx_to_load])
        if self.viz._control_down and key == glfw.KEY_LEFT and action == glfw.PRESS:
            # If the currently loaded slide is not in the filtered slide paths,
            # then start at the end.
            if self.viz.wsi is None or self.viz.wsi.path not in self.filtered_slide_paths:
                if len(self.filtered_slide_paths):
                    idx_to_load = len(self.filtered_slide_paths) - 1
                else:
                    idx_to_load = None
            # Otherwise, load the previous slide in the filtered list.
            else:
                current_slide_idx = self.filtered_slide_paths.index(self.viz.wsi.path)
                idx_to_load = (current_slide_idx - 1) % len(self.filtered_slide_paths)
            if idx_to_load is not None:
                self.viz.load_slide(self.filtered_slide_paths[idx_to_load])
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            if self._adding_source:
                self._adding_source = False
            else:
                self.end_create_project()

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
            self.filtered_slide_paths = self.slide_paths
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

    def get_path_input_width(self) -> float:
        return imgui.get_content_region_max()[0] - self.label_width - self.button_width - self.viz.spacing

    def draw_project_path(self) -> None:
        """Draw the project path input and browse button."""
        # Label.
        with imgui_utils.item_width(self.label_width):
            imgui.text('Path')
        # Input text.
        imgui.same_line(self.label_width)
        if not self._new_project_path:
            self._new_project_path = ''
        with imgui_utils.item_width(self.get_path_input_width()):
            _changed, self._new_project_path = imgui.input_text(
                '##new_project_path', self._new_project_path, 256
            )
        if not self._new_project_path and imgui.is_item_hovered() and not imgui.is_item_active():
            imgui.set_tooltip("Initialize a new project at this path.")
        if _changed:
            self._new_project_path = self._new_project_path.strip()
        # Browse button.
        imgui.same_line()
        if imgui_utils.button('Browse##project_path', width=self.button_width):
            self._new_project_path = askdirectory()

    def draw_dataset_sources(self) -> None:
        """Draw a selectable list of dataset sources."""
        changed = False
        with imgui.begin_list_box("##dataset_sources", -1, 100) as list_box:
            if list_box.opened:
                if not self._dataset_config:
                    with self.viz.dim_text():
                        imgui.text("No configured data sources.")
                else:
                    for source in self._dataset_config:
                        _clicked, _ = imgui.selectable(source, source in self._new_project_sources)
                        if _clicked:
                            changed = True
                            if source in self._new_project_sources:
                                self._new_project_sources.remove(source)
                            else:
                                self._new_project_sources.append(source)
        if changed:
            self.update_estimated_slides()

        # Show how many sources are selected.
        imgui.text("{} source{} selected.".format(
            len(self._new_project_sources),
            '' if len(self._new_project_sources) == 1 else 's'
        ))
        imgui.same_line(imgui.get_content_region_max()[0] - self.viz.font_size*6 - self.viz.spacing)
        if imgui_utils.button('Load##load_source', width=self.viz.font_size*3):
            self.ask_load_dataset_config()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Load preconfigured dataset(s).")
        imgui.same_line()
        if imgui_utils.button('Add##add_source', width=self.viz.font_size*3):
            self._adding_source = True
            self.reset_add_source()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Add a new dataset.")

        # Draw patient/slide counts.
        with self.viz.dim_text():
            imgui.text('Slides:')
            imgui.same_line()
        imgui.text(str(self._new_project_n_slides))


    def draw_annotations_path(self) -> None:
        """Draw the annotations path input and browse button."""
        # Label.
        with imgui_utils.item_width(self.label_width):
            imgui.text('Annotations')
        annotations_tooltip = ("(Optional) Path to a clinical annotations (CSV). "
                               "\nIf not provided, will create a blank file.")
        if imgui.is_item_hovered():
            imgui.set_tooltip(annotations_tooltip)
        # Input text.
        imgui.same_line(self.label_width)
        if not self._new_annotations_path:
            self._new_annotations_path = ''
        elif not self._new_annotations_path and self.P is not None:
            self._new_annotations_path = self.P.annotations or ''
        elif not self._new_annotations_path:
            self._new_annotations_path = ''
        with imgui_utils.item_width(self.get_path_input_width()):
            _changed, self._new_annotations_path = imgui.input_text(
                '##new_annotations_path', self._new_annotations_path, 256
            )
        if not self._new_annotations_path and imgui.is_item_hovered() and not imgui.is_item_active():
            imgui.set_tooltip(annotations_tooltip)
        if _changed:
            self._new_annotations_path = self._new_annotations_path.strip()
        # Browse button.
        imgui.same_line()
        if imgui_utils.button('Browse##annotations_path', width=self.button_width):
            _path = askopenfilename(
                title="Annotations", filetypes=[("CSV", "*.csv",)]
            )
            if _path:
                self._new_annotations_path = _path

    def draw_new_project_buttons(self) -> None:
        """Draw the 'Cancel' and 'Create' buttons for the new project dialog."""
        button_w = self.viz.font_size * 6
        if self.viz.sidebar.full_button2('Cancel', width=button_w):
            self.end_create_project()
        imgui.same_line(imgui.get_content_region_max()[0] - button_w)
        is_enabled = bool(
            self._new_project_path
            and self._new_project_sources
        )
        if self.viz.sidebar.full_button('Create', width=button_w, enabled=is_enabled):
            if self._new_project_path:
                try:
                    sf.create_project(
                        self._new_project_path,
                        annotations=(self._new_annotations_path or None),
                        sources=self._new_project_sources,
                        dataset_config=self.dataset_config
                    )
                    self.load(self._new_project_path)
                    self._show_new_project = False
                except Exception as e:
                    self.viz.create_toast(
                        f"Unable to create project at {self._new_project_path}",
                        icon="error"
                    )
                    sf.log.error(f"Unable to create project at {self._new_project_path}: {e}")
                    raise

    def draw_add_source_popup(self) -> None:
        """Draw the popup for adding a new dataset source."""
        imgui.open_popup('Add Dataset Source')
        imgui.set_next_window_size(self.viz.font_size * 20, 0)
        if imgui.begin_popup_modal('Add Dataset Source', None):

            # Name.
            with imgui_utils.item_width(self.label_width):
                imgui.text('Name')
            with imgui_utils.item_width(self.get_path_input_width()):
                imgui.same_line(self.label_width)
                _, self._add_source_name = imgui.input_text(
                    '##add_source_name', self._add_source_name, 256
                )

            # Slides.
            with imgui_utils.item_width(self.label_width):
                imgui.text('Slides')
            with imgui_utils.item_width(self.get_path_input_width()):
                imgui.same_line(self.label_width)
                _slides_changed, self._add_source_slides = imgui.input_text(
                    '##add_source_slides', self._add_source_slides, 256
                )
            imgui.same_line()
            if imgui_utils.button('Browse##add_source_slides', width=self.button_width):
                _path = askdirectory(title="Path to slides")
                if _path:
                    self._add_source_slides = _path
                    _slides_changed = True
            if _slides_changed and os.path.exists(self._add_source_slides) and os.path.isdir(self._add_source_slides):
                self._add_source_n_slides = len(sf.util.get_slide_paths(self._add_source_slides))

            # ROIs.
            with imgui_utils.item_width(self.label_width):
                imgui.text('ROIs')
            with imgui_utils.item_width(self.get_path_input_width()):
                imgui.same_line(self.label_width)
                _, self._add_source_rois = imgui.input_text(
                    '##add_source_rois', self._add_source_rois, 256
                )
            if imgui.is_item_hovered():
                imgui.set_tooltip("(Optional) Path to a directory containing ROIs.")
            imgui.same_line()
            if imgui_utils.button('Browse##add_source_rois', width=self.button_width):
                _path = askdirectory(title="Path to ROIs")
                if _path:
                    self._add_source_rois = _path

            # TFRecords.
            with imgui_utils.item_width(self.label_width):
                imgui.text('TFRecords')
            with imgui_utils.item_width(self.get_path_input_width()):
                imgui.same_line(self.label_width)
                _, self._add_source_tfrecords = imgui.input_text(
                    '##add_source_tfrecords', self._add_source_tfrecords, 256
                )
            if imgui.is_item_hovered():
                imgui.set_tooltip("(Optional) Path to destination TFRecords directory.")
            imgui.same_line()
            if imgui_utils.button('Browse##add_source_tfrecords', width=self.button_width):
                _path = askdirectory(title="Path to TFRecords")
                if _path:
                    self._add_source_tfrecords = _path

            imgui.text("Slides: {}".format(self._add_source_n_slides))

            imgui.separator()

            # Buttons.
            button_w = self.viz.font_size * 6
            if self.viz.sidebar.full_button2('Cancel', width=button_w):
                self._adding_source = False
                imgui.close_current_popup()
            imgui.same_line(imgui.get_content_region_max()[0] - button_w)
            is_enabled = bool(
                self._add_source_name
                and self._add_source_slides
            )
            if self.viz.sidebar.full_button('Add##add_source_button', width=button_w, enabled=is_enabled):
                source = dict()
                source['slides'] = self._add_source_slides
                if self._add_source_rois:
                    source['roi'] = self._add_source_rois
                if self._add_source_tfrecords:
                    source['tfrecords'] = self._add_source_tfrecords
                self._dataset_config[self._add_source_name] = source
                self._new_project_sources.append(self._add_source_name)
                self.update_estimated_slides()
                self._adding_source = False

            imgui.end_popup()

    def draw_new_project_dialog(self) -> None:
        """Draw the new project dialog."""
        self.viz.suspend_mouse_input_handling()
        imgui.open_popup('New Project')
        imgui.set_next_window_size(self.viz.font_size * 20, 0)
        if imgui.begin_popup_modal('New Project', None):
            self.draw_project_path()
            self.draw_annotations_path()
            self.draw_dataset_sources()
            imgui.separator()
            self.draw_new_project_buttons()
            if self._adding_source:
                self.draw_add_source_popup()
            imgui.end_popup()

    def draw_slide_search(self) -> None:
        """Draws the search bar for the list of slides."""
        viz = self.viz
        viz.icon('search')
        imgui.same_line()
        width = imgui.get_content_region_max()[0] - viz.font_size * 6
        with imgui_utils.item_width(width):
            _changed, self.slide_search = imgui.input_text('##slide_search', self.slide_search, 128)
        if _changed:
            self.slide_search = self.slide_search.strip()
            self.filtered_slide_paths = [path for path in self.slide_paths if self.slide_search.lower() in path.lower()]
        imgui.same_line()
        if imgui.button('Clear'):
            self.slide_search = ''
            self.filtered_slide_paths = self.slide_paths
        imgui.same_line()
        if viz.icon_button('filter'):
            self._show_slide_filter_popup = not self._show_slide_filter_popup

        self.draw_slide_filter_popup()

    def draw_slide_filter_popup(self) -> None:
        viz = self.viz

        if self._show_slide_filter_popup:
            cx, cy = imgui.get_cursor_pos()
            imgui.set_next_window_position(viz.sidebar.full_width, cy)
            imgui.begin(
                '##slide_filter_popup',
                flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
            )
            with viz.bold_font():
                imgui.text("Slide Filter")
            imgui.separator()
            updated = False
            if imgui.menu_item('Has ROIs', selected=(self._filter_by_has_roi == True))[0]:
                updated = True
                if self._filter_by_has_roi is None or self._filter_by_has_roi is False:
                    self._filter_by_has_roi = True
                else:
                    self._filter_by_has_roi = None
                # The below will close out the popup after this has been clicked,
                # which we don't want. Kept here for reference.
                # ----
                #self._clicking = False
                #self._show_slide_filter_popup = False
                # ----
            if imgui.menu_item('No ROIs', selected=(self._filter_by_has_roi == False))[0]:
                updated = True
                if self._filter_by_has_roi is None or self._filter_by_has_roi is True:
                    self._filter_by_has_roi = False
                else:
                    self._filter_by_has_roi = None

            # Update the filter
            if updated:
                rois = [sf.util.path_to_name(roi) for roi in self.P.dataset().rois()]
                self.filtered_slide_paths = [
                    path for path in self.slide_paths
                    if (self.slide_search.lower() in path.lower()
                        and (self._filter_by_has_roi is None
                             or (self._filter_by_has_roi is True and sf.util.path_to_name(path) in rois)
                             or (self._filter_by_has_roi is False and sf.util.path_to_name(path) not in rois)))
                ]

            # Hide menu if we click elsewhere
            if imgui.is_mouse_down(LEFT_MOUSE_BUTTON) and not imgui.is_window_hovered():
                self._clicking = True
            if self._clicking and imgui.is_mouse_released(LEFT_MOUSE_BUTTON):
                self._clicking = False
                self._show_slide_filter_popup = False

            imgui.end()

    def draw_slide_list(self) -> None:
        """Draws the list of slides in the project."""
        for path in self.filtered_slide_paths:
            with self.viz.bold_font(self.viz.wsi is not None and path == self.viz.wsi.path):
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

        if self._show_new_project:
            self.draw_new_project_dialog()

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
            if viz.sidebar.full_button("Create a Project"):
                self.new_project()

        elif show:
            if viz.collapsing_header('Info', default=True):
                self.draw_info()

            if viz.collapsing_header('Slides', default=False):
                if not len(self.slide_paths):
                    imgui_utils.padded_text('No slides found.', vpad=[int(viz.font_size/2), int(viz.font_size)])
                else:
                    self.draw_slide_search()
                    self.draw_slide_list()

            if viz.collapsing_header('Models', default=False):
                if not self.recursive_model_scan():
                    imgui_utils.padded_text('No models found.', vpad=[int(viz.font_size/2), int(viz.font_size)])