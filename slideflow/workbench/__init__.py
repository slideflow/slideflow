import time
import multiprocessing
import numpy as np
import imgui
import OpenGL.GL as gl

from slideflow.workbench.gui_utils import imgui_window
from slideflow.workbench.gui_utils import imgui_utils
from slideflow.workbench.gui_utils import gl_utils
from slideflow.workbench.gui_utils import text_utils
from slideflow.workbench.gui_utils import wsi_utils
from slideflow.workbench import slide_renderer as renderer
from slideflow.workbench import project_widget
from slideflow.workbench import slide_widget
from slideflow.workbench import model_widget
from slideflow.workbench import heatmap_widget
from slideflow.workbench import performance_widget
from slideflow.workbench import capture_widget

import slideflow as sf
import slideflow.grad
from slideflow.workbench.utils import EasyDict
from slideflow import log

if sf.util.tf_available:
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception:
        pass
if sf.util.torch_available:
    import slideflow.model.torch

#----------------------------------------------------------------------------

def _load_model_and_saliency(model_path, device=None):
    log.debug("Loading model at {}...".format(model_path))
    if sf.util.torch_available and sf.util.path_to_ext(model_path) == 'zip':
        _model = sf.model.torch.load(model_path)
        _model.eval()
        if device is not None:
            _model = _model.to(device)
        _saliency = sf.grad.SaliencyMap(_model, class_idx=0)  #TODO: auto-update from heatmaps logit
    elif sf.util.tf_available and sf.util.path_to_ext(model_path) == 'tflite':
        interpreter = tf.lite.Interpreter(model_path)
        _model = interpreter.get_signature_runner()
        _saliency = sf.grad.SaliencyMap(_model, class_idx=0)  #TODO: auto-update from heatmaps logit
    elif sf.util.tf_available:
        _model = sf.model.tensorflow.load(model_path, method='weights')
        _saliency = None
    else:
        raise ValueError(f"Unable to interpret model {model_path}")
    return _model, _saliency

#----------------------------------------------------------------------------

class Workbench(imgui_window.ImguiWindow):
    def __init__(self, low_memory=False, widgets=None):

        super().__init__(title=f'Slideflow Workbench')

        # Internals.
        self._dx                = 0
        self._dy                = 0
        self._last_error_print  = None
        self._async_renderer    = AsyncRenderer()
        self._defer_rendering   = 0
        self._tex_img           = None
        self._tex_obj           = None
        self._norm_tex_img      = None
        self._norm_tex_obj      = None
        self._heatmap_tex_img   = None
        self._heatmap_tex_obj   = None
        self._wsi_tex_obj       = None
        self._wsi_tex_img       = None
        self._overlay_tex_img   = None
        self._overlay_tex_obj   = None
        self._predictions       = None
        self._model_path        = None
        self._model_config      = None
        self._normalizer        = None
        self._normalize_wsi     = False
        self._uncertainty       = None
        self._content_width     = None
        self._content_height    = None
        self._pane_w            = None
        self._refresh_view      = False
        self._overlay_wsi_dim   = None
        self._overlay_offset_wsi_dim   = (0, 0)
        self._thumb_params      = None
        self._use_model         = None
        self._use_uncertainty   = None
        self._use_saliency      = None
        self._use_model_img_fmt = False
        self._tex_to_delete     = []
        self._low_memory        = low_memory
        self._defer_tile_refresh = None
        self._should_close_slide = False
        self._should_close_model = False

        # Interface.
        self._show_control      = True
        self._dock_control      = True
        self._show_performance  = False
        self._control_size      = None
        self._show_tile_preview = False
        self._tile_preview_is_new = True
        self._tile_preview_image_is_new = True

        # Widget interface.
        self.wsi                = None
        self.wsi_thumb          = None
        self.viewer             = None
        self.saliency           = None
        self.box_x              = None
        self.box_y              = None
        self.tile_px            = None
        self.tile_um            = None
        self.heatmap            = None
        self.rendered_heatmap   = None
        self.overlay            = None
        self.rendered_qc        = None
        self.overlay_qc         = None
        self.args               = EasyDict(use_model=False, use_uncertainty=False, use_saliency=False)
        self.result             = EasyDict(predictions=None, uncertainty=None)
        self.message            = None
        self.pane_w             = 0
        self.label_w            = 0
        self.button_w           = 0
        self.x                  = None
        self.y                  = None
        self.menu_bar_height    = self.font_size + self.spacing

        # Core widgets.
        self.project_widget     = project_widget.ProjectWidget(self)
        self.slide_widget       = slide_widget.SlideWidget(self)
        self.model_widget       = model_widget.ModelWidget(self)
        self.heatmap_widget     = heatmap_widget.HeatmapWidget(self)
        self.performance_widget = performance_widget.PerformanceWidget(self)
        self.capture_widget     = capture_widget.CaptureWidget(self)

        # User-defined widgets.
        self.widgets = []
        if widgets is None:
            widgets = self.get_default_widgets()
        for widget in widgets:
            self.widgets += [widget(self)]

        # Initialize window.
        self.set_position(0, 0)
        self._adjust_font_size()
        self.skip_frame() # Layout may change after first frame.
        self.load_slide('')

    @property
    def show_overlay(self):
        return self.slide_widget.show_overlay or self.heatmap_widget.show

    @property
    def model(self):
        return self._async_renderer._model

    @property
    def P(self):
        return self.project_widget.P

    @property
    def offset_x(self):
        return self.pane_w

    @property
    def offset_y(self):
        return self.menu_bar_height

    @staticmethod
    def get_default_widgets():
        return []

    def has_live_viewer(self):
        return (self.viewer is not None and self.viewer.live)

    def set_low_memory(self, low_memory):
        assert isinstance(low_memory, bool)
        self._low_memory = low_memory

    def reload_model(self):
        self._async_renderer.load_model(self._model_path)

    def close(self):
        super().close()
        if self._async_renderer is not None:
            self._async_renderer.close()
            self._async_renderer = None

    def set_message(self, msg):
        self.message = msg

    def clear_message(self, msg=None):
        if msg is None or self.message == msg:
            self.message = None

    def add_recent_slide(self, slide, ignore_errors=False):
        self.slide_widget.add_recent(slide, ignore_errors=ignore_errors)

    def load_project(self, project, ignore_errors=False):
        self.project_widget.load(project, ignore_errors=ignore_errors)

    def load_slide(self, slide, ignore_errors=False):
        self.slide_widget.load(slide, ignore_errors=ignore_errors)

    def _close_slide(self):
        self.wsi = None
        self.viewer = None
        self.wsi_thumb = None
        self.x = None
        self.y = None
        self.clear_result()
        self._wsi_tex_obj = None
        self._async_renderer._live_updates = False

    def close_slide(self, now=False):
        if now:
            self._close_slide()
            self._should_close_slide = False
        else:
            self._should_close_slide = True

    def _reload_wsi(self, path=None, stride=None, use_rois=True):
        if self.wsi is None and path is None:
            return
        if path is None:
            path = self.wsi.path
        if stride is None:
            stride = self.wsi.stride_div
        if self.P and use_rois:
            rois = self.P.dataset().rois()
        else:
            rois = None
        self.wsi = sf.WSI(
            path,
            tile_px=(self.tile_px if self.tile_px else 256),
            tile_um=(self.tile_um if self.tile_um else 512),
            stride_div=stride,
            rois=rois,
            vips_cache=dict(
                tile_width=512,
                tile_height=512,
                max_tiles=-1,
                threaded=True,
                persistent=True
            ),
            verbose=False)
        self.set_viewer(wsi_utils.SlideViewer(self.wsi, **self._viewer_kwargs()))

    def _viewer_kwargs(self):
        return dict(
            width=self.content_width - self.offset_x,
            height=self.content_height - self.offset_y,
            x_offset=self.offset_x,
            y_offset=self.offset_y,
            normalizer=(self._normalizer if self._normalize_wsi else None)
        )

    def reload_viewer(self):
        if self.viewer is not None:
            self.viewer.close()
            if isinstance(self.viewer, wsi_utils.SlideViewer):
                self.set_viewer(wsi_utils.SlideViewer(self.wsi, **self._viewer_kwargs()))
            else:
                self.viewer.reload(**self._viewer_kwargs())

    def set_viewer(self, viewer):
        log.debug("Setting viewer to {}".format(viewer))
        self.viewer = viewer
        self._async_renderer._live_updates = viewer.live
        self._async_renderer.set_async(viewer.live)

    def load_model(self, model, ignore_errors=False):
        log.debug("Loading model from workbench")
        self.close_model(True)
        log.debug("Model closed")
        self.clear_result()
        log.debug("Model result cleared")
        self.skip_frame() # The input field will change on next frame.
        self._async_renderer.get_result() # Flush prior result
        try:
            self.defer_rendering()
            self.model_widget.user_model = model

            # Read model configuration
            config = sf.util.get_model_config(model)
            normalizer = sf.util.get_model_normalizer(model)
            self.result.message = f'Loading {config["model_name"]}...'
            self.defer_rendering()
            self._use_model = True
            self._model_path = model
            self._model_config = config
            self._normalizer = normalizer
            self._predictions = None
            self._uncertainty = None
            self._use_uncertainty = 'uq' in config['hp'] and config['hp']['uq']
            self.tile_um = config['tile_um']
            self.tile_px = config['tile_px']
            self._async_renderer.load_model(model)
            if sf.util.torch_available and sf.util.path_to_ext(model) == 'zip':
                self.model_widget.backend = 'torch'
            else:
                self.model_widget.backend = 'tensorflow'

            # Update widgets
            log.debug("Updating widgets")
            self.model_widget.cur_model = model
            self.model_widget.use_model = True
            self.model_widget.use_uncertainty = 'uq' in config['hp'] and config['hp']['uq']
            self.model_widget.refresh_recent()
            if normalizer is not None and hasattr(self, 'slide_widget'):
                self.slide_widget.show_model_normalizer()
                self.slide_widget.norm_idx = len(self.slide_widget._normalizer_methods)-1
            if self.wsi:
                log.debug(f"Loading slide... tile_px={self.tile_px}, tile_um={self.tile_um}")
                self.slide_widget.load(self.wsi.path, ignore_errors=ignore_errors)
            if hasattr(self, 'heatmap_widget'):
                log.debug("Resetting heatmap")
                self.heatmap_widget.reset()

            # Update viewer
            self._show_tile_preview = True
            log.debug("Updating viewer with tile_px={}, tile_um={}".format(self.tile_px, self.tile_um))
            if self.viewer and not isinstance(self.viewer, wsi_utils.SlideViewer):
                self.viewer.set_tile_px(self.tile_px)
                self.viewer.set_tile_um(self.tile_um)

        except Exception as e:
            self.model_widget.cur_model = None
            if model == '':
                log.debug("Exception raised: no model loaded.")
                self.result = EasyDict(message='No model loaded')
            else:
                log.debug("Exception raised (ignore_errors={}): {}".format(ignore_errors, e))
                self.result = EasyDict(error=renderer.CapturedException())
            if not ignore_errors:
                raise
        log.debug("Model loading complete (path={})".format(self._model_path))

    def print_error(self, error):
        error = str(error)
        if error != self._last_error_print:
            print('\n' + error + '\n')
            self._last_error_print = error

    def defer_rendering(self, num_frames=1):
        self._defer_rendering = max(self._defer_rendering, num_frames)

    def clear_overlay(self):
        self._overlay_tex_img   = None
        self.overlay            = None

    def _close_model(self):
        self._async_renderer.clear_result()
        self._use_model   = False
        self._use_uncertainty   = False
        self._use_saliency      = False
        self._model_path        = None
        self._model_config      = None
        self._normalizer        = None
        self.tile_px            = None
        self.tile_um            = None
        self.heatmap            = None
        self.x                  = None
        self.y                  = None
        self._async_renderer.clear_model()
        self.clear_model_results()
        self.heatmap_widget.reset()

    def close_model(self, now=False):
        if now:
            self._close_model()
            self._should_close_model = False
        else:
            self._should_close_model = True

    def clear_result(self):
        self._async_renderer.clear_result()
        self._tex_img           = None
        self._norm_tex_img      = None
        self._heatmap_tex_img   = None
        self._wsi_tex_img       = None
        self.clear_model_results()
        self.clear_overlay()
        if self.viewer:
            self.viewer.clear()

    def clear_model_results(self):
        self._async_renderer.clear_result()
        self._predictions       = None
        self._tex_img           = None
        self._tex_obj           = None
        self._norm_tex_img      = None
        self._norm_tex_obj      = None
        self._heatmap_tex_img   = None
        self._heatmap_tex_obj   = None
        self._overlay_tex_img   = None
        self._overlay_tex_obj   = None

    def _adjust_font_size(self):
        old = self.font_size
        self.set_font_size(13)
        if self.font_size != old:
            self.skip_frame() # Layout changed.

    def has_uq(self):
        return (self._model_path is not None
                and self._model_config is not None
                and 'uq' in self._model_config['hp']
                and self._model_config['hp']['uq'])

    def _draw_menu_bar(self):
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu('File', True):
                imgui.menu_item('Open Project...', 'Ctrl+P')
                imgui.menu_item('Load Slide...', 'Ctrl+O')
                imgui.menu_item('Load Model...', 'Ctrl+M')
                imgui.separator()
                if imgui.begin_menu('Export...', True):
                    imgui.menu_item('Main view')
                    imgui.menu_item('Tile view')
                    imgui.menu_item('GUI view')
                    imgui.menu_item('Heatmap (PNG)')
                    imgui.menu_item('Heatmap (NPZ)')
                    imgui.end_menu()
                imgui.separator()
                if imgui.menu_item('Close Slide')[1]:
                    self.close_slide(True)
                if imgui.menu_item('Close Model')[1]:
                    self.close_model(True)
                imgui.separator()
                if imgui.menu_item('Exit', 'Ctrl+Q')[1]:
                    self._exit_trigger = True
                imgui.end_menu()

            if imgui.begin_menu('View', True):
                if imgui.menu_item('Fullscreen', 'Ctrl+F')[1]:
                    self.toggle_fullscreen()
                if imgui.menu_item('Dock Controls', 'Ctrl+Shift+D')[1]:
                    self._dock_control = not self._dock_control
                if imgui.menu_item('Toggle Controls', 'Ctrl+Shift+C')[1]:
                    self._show_control = not self._show_control
                if imgui.menu_item('Toggle Performance', 'Ctrl+Shift+P')[1]:
                    self._show_performance = not self._show_performance
                imgui.separator()
                if imgui.menu_item('Toggle tile preview')[1]:
                    self._show_tile_preview = not self._show_tile_preview
                imgui.menu_item('Toggle camera view')
                imgui.end_menu()

            if imgui.begin_menu('Help', True):
                imgui.menu_item('Get Started')
                imgui.menu_item('Documentation')
                imgui.separator()
                imgui.menu_item('Release Notes')
                imgui.menu_item('Report Issue')
                imgui.separator()
                imgui.menu_item('View License')
                imgui.menu_item('About')
                imgui.end_menu()

            version_text = f'slideflow {sf.__version__}'
            imgui.same_line(imgui.get_content_region_max()[0] - (imgui.calc_text_size(version_text)[0] + self.spacing))
            imgui.text(version_text)
            imgui.end_main_menu_bar()

    def _render_control_pane_contents(self):
        """Perform rendering of control panel contents, such as WSI thumbnails,
        widgets, and heatmaps."""

        # Render WSI thumbnail in the widget.
        if self.wsi_thumb is not None and self._show_control:
            if self._wsi_tex_img is not self.wsi_thumb:
                self._wsi_tex_img = self.wsi_thumb
                if self._wsi_tex_obj is None or not self._wsi_tex_obj.is_compatible(image=self._wsi_tex_img):
                    if self._wsi_tex_obj is not None:
                        self._tex_to_delete += [self._wsi_tex_obj]
                    self._wsi_tex_obj = gl_utils.Texture(image=self._wsi_tex_img, bilinear=True, mipmap=True)
                else:
                    self._wsi_tex_obj.update(self._wsi_tex_img)

        # Render user widgets.
        for widget in self.widgets:
            if hasattr(widget, 'render'):
                widget.render()

        # Display rendered (non-transparent) heatmap in widget.
        # Render overlay heatmap.
        if self.heatmap:
            if self._heatmap_tex_img is not self.rendered_heatmap:
                self._heatmap_tex_img = self.rendered_heatmap
                if self._heatmap_tex_obj is None or not self._heatmap_tex_obj.is_compatible(image=self._heatmap_tex_img):
                    if self._heatmap_tex_obj is not None:
                        self._tex_to_delete += [self._heatmap_tex_obj]
                    self._heatmap_tex_obj = gl_utils.Texture(image=self._heatmap_tex_img, bilinear=False, mipmap=False)
                else:
                    self._heatmap_tex_obj.update(self._heatmap_tex_img)

    def _draw_control_pane(self):
        """Draw the control pane with Imgui."""

        if self._dock_control and self._show_control:
            self.pane_w = self.font_size * 45
            imgui.set_next_window_position(0, self.menu_bar_height)
            imgui.set_next_window_size(self.pane_w, self.content_height - self.menu_bar_height)
            control_kw = dict(
                closable=False,
                flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))
        else:
            imgui.set_next_window_size(self.font_size * 45, self._control_size)
            self.pane_w = 0
            control_kw = dict(
                closable=True,
                flags=(imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_ALWAYS_AUTO_RESIZE)
            )

        if self._show_control:
            _, self._show_control = imgui.begin('Control Pane', **control_kw)

            # Core widgets.
            self._control_size = self.font_size * 4 + self.spacing * 11
            expanded, _visible = imgui_utils.collapsing_header('Slideflow project', default=True)
            self.project_widget(expanded)
            self._control_size += self.project_widget.content_height

            expanded, _visible = imgui_utils.collapsing_header('Whole-slide image', default=True)
            self.slide_widget(expanded)
            self._control_size += self.slide_widget.content_height

            expanded, _visible = imgui_utils.collapsing_header('Model & tile predictions', default=True)
            self.model_widget(expanded)
            self._control_size += self.model_widget.content_height

            expanded, _visible = imgui_utils.collapsing_header('Heatmap & slide prediction', default=True)
            self.heatmap_widget(expanded)
            self._control_size += self.heatmap_widget.content_height

            # User-defined widgets
            for widget in self.widgets:
                if hasattr(widget, 'collapsing_header'):
                    expanded, _visible = widget.collapsing_header()
                    self._control_size += (self.font_size + self.spacing * 3)
                widget(expanded)
                if hasattr(widget, 'content_height'):
                    self._control_size += widget.content_height

            self._render_control_pane_contents()
            imgui.end()

    def _draw_performance_pane(self):
        if self._show_performance:
            _, self._show_performance = imgui.begin('Performance & Capture', closable=True, flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE))
            self.performance_widget(True)
            self.capture_widget(True)
            imgui.end()

    def _draw_tile_view(self):
        if self._show_tile_preview:

            has_raw_image = self._tex_obj is not None and self.tile_px
            has_norm_image = self.model_widget.use_model and self._normalizer is not None and self._norm_tex_obj is not None and self.tile_px

            width = 0
            height = self.font_size * 2
            if has_raw_image:
                height += self.tile_px
                width += self.tile_px
            if has_norm_image:
                width += self.tile_px + self.spacing

            if not width:
                width = self.font_size * 8
                height = self.font_size * 3

            imgui.set_next_window_size(width, height)

            if self._tile_preview_is_new:
                imgui.set_next_window_position(self.content_width - width - self.spacing, self.content_height - height - self.spacing)
                self._tile_preview_is_new = False

            if self._tile_preview_image_is_new and (has_raw_image or has_norm_image):
                imgui.set_next_window_position(self.content_width - width - self.spacing, self.content_height - height - self.spacing)
                self._tile_preview_image_is_new = False

            _, self._show_tile_preview = imgui.begin("##tile view", closable=True, flags=(imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_SCROLLBAR))
            # Image preview ===================================================
            dim_color = list(imgui.get_style().colors[imgui.COLOR_TEXT])
            dim_color[-1] *= 0.5
            imgui.begin_child('##pred_image', border=False)
            if has_raw_image:
                imgui.image(self._tex_obj.gl_id, self.tile_px, self.tile_px)
            elif self._model_path is not None:
                imgui.text_colored('Right click to preview', *dim_color)
            else:
                imgui.text_colored('No model loaded', *dim_color)
            imgui.same_line()
            if has_norm_image:
                imgui.image(self._norm_tex_obj.gl_id, self.tile_px, self.tile_px)
            elif self._tex_obj is not None and self.tile_px:
                imgui.text_colored('Normalizer not used', *dim_color)
            imgui.end_child()
            imgui.same_line()
            imgui.end()

    def _draw_main_view(self, inp, window_changed):
        """Update the main window view.

        Draws the slide / picam view, overlay heatmap, overlay box, and ROIs.

        Args:
            inp (EasyDict): Dictionary of user input.
            window_changed (bool): Window size has changed (force refresh).
        """

        max_w = self.content_width - self.offset_x
        max_h = self.content_height - self.offset_y

        # Update the viewer in response to user input.
        if self.viewer and self.viewer.movable:
            # Update WSI focus location & zoom values
            # If shift-dragging or scrolling.
            dz = None
            if not inp.dragging:
                inp.dx, inp.dy = None, None
            if inp.wheel > 0:
                dz = 1/1.5
            if inp.wheel < 0:
                dz = 1.5
            if inp.wheel or inp.dragging or self._refresh_view:
                if inp.dx is not None:
                    self.viewer.move(inp.dx, inp.dy)
                if inp.wheel:
                    self.viewer.zoom(inp.cx, inp.cy, dz)
                if self._refresh_view and inp.dx is None and not inp.wheel:
                    self.viewer.refresh_view()
                    self._refresh_view = False

        # Render slide view.
        self.viewer.render(max_w, max_h)

        # Render overlay heatmap.
        if self.overlay is not None and self.show_overlay:
            if self._overlay_tex_img is not self.overlay:
                self._overlay_tex_img = self.overlay
                if self._overlay_tex_obj is None or not self._overlay_tex_obj.is_compatible(image=self._overlay_tex_img):
                    if self._overlay_tex_obj is not None:
                        self._tex_to_delete += [self._overlay_tex_obj]
                    self._overlay_tex_obj = gl_utils.Texture(image=self._overlay_tex_img, bilinear=False, mipmap=False)
                else:
                    self._overlay_tex_obj.update(self._overlay_tex_img)
            if self._overlay_wsi_dim is None:
                self._overlay_wsi_dim = self.viewer.dimensions
            h_zoom = (self._overlay_wsi_dim[0] / self.overlay.shape[1]) / self.viewer.view_zoom
            h_pos = self.viewer.wsi_coords_to_display_coords(*self._overlay_offset_wsi_dim)
            self._overlay_tex_obj.draw(pos=h_pos, zoom=h_zoom, align=0.5, rint=True, anchor='topleft')

        # Calculate location for model display.
        if (self._model_path
            and inp.clicking
            and not inp.dragging
            and self.viewer.is_in_view(inp.cx, inp.cy)):
            wsi_x, wsi_y = self.viewer.display_coords_to_wsi_coords(inp.cx, inp.cy, offset=False)
            self.x = wsi_x - (self.viewer.full_extract_px/2)
            self.y = wsi_y - (self.viewer.full_extract_px/2)

        # Update box location.
        if self.x is not None and self.y is not None:
            if inp.clicking or inp.dragging or inp.wheel or window_changed:
                self.box_x, self.box_y = self.viewer.wsi_coords_to_display_coords(self.x, self.y)
            tw = self.viewer.full_extract_px / self.viewer.view_zoom

            # Draw box on main display.
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            gl.glLineWidth(3)
            box_pos = np.array([self.box_x, self.box_y])
            gl_utils.draw_rect(pos=box_pos, size=np.array([tw, tw]), color=[1, 0, 0], mode=gl.GL_LINE_LOOP)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glLineWidth(1)

        # Render ROIs.
        self.viewer.late_render()

    def _handle_user_input(self):

        # Detect mouse dragging in the thumbnail display.
        clicking, cx, cy, wheel = imgui_utils.click_hidden_window(
            '##result_area',
            x=self.offset_x,
            y=self.offset_y,
            width=self.content_width - self.offset_x,
            height=self.content_height - self.offset_y,
            mouse_idx=1)
        dragging, dx, dy = imgui_utils.drag_hidden_window(
            '##result_area',
            x=self.offset_x,
            y=self.offset_y,
            width=self.content_width - self.offset_x,
            height=self.content_height - self.offset_y)
        return EasyDict(
            clicking=clicking,
            cx=cx,
            cy=cy,
            wheel=wheel,
            dragging=dragging,
            dx=dx,
            dy=dy
        )

    def _render_prediction_message(self, message):
        """Render a prediction string to below the tile bounding box."""
        max_w = self.content_width - self.offset_x
        max_h = self.content_height - self.offset_y
        tex = text_utils.get_texture(message, size=self.font_size, max_width=max_w, max_height=max_h, outline=2)
        box_w = self.viewer.full_extract_px / self.viewer.view_zoom
        text_pos = np.array([self.box_x + (box_w/2), self.box_y + box_w + self.font_size])
        tex.draw(pos=text_pos, align=0.5, rint=True, color=1)

    def draw_frame(self):
        """Main draw loop."""

        self.begin_frame()

        self.args = EasyDict(use_model=False, use_uncertainty=False, use_saliency=False)
        self.button_w = self.font_size * 5
        self.label_w = round(self.font_size * 4.5)
        self.menu_bar_height = self.font_size + self.spacing

        max_w = self.content_width - self.offset_x
        max_h = self.content_height - self.offset_y
        window_changed = (self._content_width != self.content_width
                          or self._content_height != self.content_height
                          or self._pane_w != self.pane_w)

        self._draw_menu_bar()
        self._draw_control_pane()
        self._draw_performance_pane()
        self._draw_tile_view()

        user_input = self._handle_user_input()

        # Re-generate WSI view if the window size changed, or if we don't
        # yet have a SlideViewer initialized.
        if window_changed:
            if self.viewer:
                self.reload_viewer()
            self._content_width  = self.content_width
            self._content_height = self.content_height
            self._pane_w = self.pane_w

            for widget in self.widgets:
                if hasattr(widget, '_on_window_change'):
                    widget._on_window_change()

        # Render black box behind the controls, if docked
        if self.pane_w:
            gl_utils.draw_rect(pos=np.array([0, self.offset_y]), size=np.array([self.offset_x, self.content_height - self.offset_y]), color=0, anchor='center')

        # Main display.
        if self.viewer:
            self._draw_main_view(user_input, window_changed)

        # --- Render arguments ------------------------------------------------
        self.args.x = self.x
        self.args.y = self.y
        if (self._model_config is not None
           and self._use_model
           and 'img_format' in self._model_config
           and self._use_model_img_fmt):
            self.args.img_format = self._model_config['img_format']
        self.args.use_model = self._use_model
        self.args.use_uncertainty =  (self.has_uq() and self._use_uncertainty)
        self.args.use_saliency = self._use_saliency
        self.args.normalizer = self._normalizer

        # Buffer tile view if using a live viewer.
        if self.has_live_viewer() and self.args.x and self.args.y:

            if self._async_renderer._args_queue.qsize() > 2:
                if self._defer_tile_refresh is None:
                    self._defer_tile_refresh = time.time()
                    self.defer_rendering()
                elif time.time() - self._defer_tile_refresh < 2:
                    self.defer_rendering()
                else:
                    self._defer_tile_refresh = None

            self.viewer.x = self.x
            self.viewer.y = self.y
            self.args.full_image = self.viewer.tile_view
            self.args.tile_px = self.viewer.tile_px

        if self.has_live_viewer():
            self.args.viewer = None
        else:
            self.args.viewer = self.viewer
        # ---------------------------------------------------------------------

        if self.is_skipping_frames():
            pass
        elif self._defer_rendering > 0:
            self._defer_rendering -= 1
        else:
            self._async_renderer.set_args(**self.args)
            result = self._async_renderer.get_result()
            if result is not None:
                self.result = result
                if 'predictions' in result:
                    self._predictions = result.predictions
                    self._uncertainty = result.uncertainty

        # Display input image.
        middle_pos = np.array([self.pane_w + max_w/2, max_h/2])
        if 'image' in self.result:
            if self._tex_img is not self.result.image:
                self._tex_img = self.result.image
                if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                    if self._tex_obj is not None:
                        self._tex_to_delete += [self._tex_obj]
                    self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=False, mipmap=False)
                else:
                    self._tex_obj.update(self._tex_img)
        if 'normalized' in self.result:
            if self._norm_tex_img is not self.result.normalized:
                self._norm_tex_img = self.result.normalized
                if self._norm_tex_obj is None or not self._norm_tex_obj.is_compatible(image=self._norm_tex_img):
                    if self._norm_tex_obj is not None:
                        self._tex_to_delete += [self._norm_tex_obj]
                    self._norm_tex_obj = gl_utils.Texture(image=self._norm_tex_img, bilinear=False, mipmap=False)
                else:
                    self._norm_tex_obj.update(self._norm_tex_img)
        if 'error' in self.result:
            self.print_error(self.result.error)
            if 'message' not in self.result:
                self.result.message = str(self.result.error)
        if 'message' in self.result or self.message:
            _msg = self.message if 'message' not in self.result else self.result['message']
            tex = text_utils.get_texture(_msg, size=self.font_size, max_width=max_w, max_height=max_h, outline=2)
            tex.draw(pos=middle_pos, align=0.5, rint=True, color=1)

        # Draw prediction message next to box on main display.
        if self._use_model and self._predictions is not None and not isinstance(self._predictions, list):
            #TODO: support multi-outcome models
            outcomes = self._model_config['outcome_labels']
            if self._model_config['model_type'] == 'categorical':
                pred_str = f'{outcomes[str(np.argmax(self._predictions))]} ({np.max(self._predictions)*100:.1f}%)'
            else:
                pred_str = f'{self._predictions[0]:.2f}'
            self._render_prediction_message(pred_str)

        # End frame.
        if self._should_close_model:
            self.close_model(True)
        if self._should_close_slide:
            self.close_slide(True)
        self._adjust_font_size()
        self.end_frame()

#----------------------------------------------------------------------------

class AsyncRenderer:
    def __init__(self):
        self._closed        = False
        self._is_async      = False
        self._cur_args      = None
        self._cur_result    = None
        self._cur_stamp     = 0
        self._renderer_obj  = None
        self._args_queue    = None
        self._result_queue  = None
        self._process       = None
        self._model_path    = None
        self._model         = None
        self._saliency      = None
        self._live_updates  = False
        self.tile_px        = None
        self.extract_px     = None

        if sf.util.torch_available:
            import torch
            self.device = torch.device('cuda')
        else:
            self.device = None

    def close(self):
        self._closed = True
        self._renderer_obj = None
        if self._process is not None:
            self._process.terminate()
        self._process = None
        self._args_queue = None
        self._result_queue = None

    @property
    def is_async(self):
        return self._is_async

    def set_async(self, is_async):
        self._is_async = is_async

    def set_args(self, **args):
        assert not self._closed
        if args != self._cur_args or self._live_updates:
            if self._is_async:
                self._set_args_async(**args)
            else:
                self._set_args_sync(**args)
            if not self._live_updates:
                self._cur_args = args

    def _set_args_async(self, **args):
        if self._process is None:
            ctx = multiprocessing.get_context('spawn')
            self._args_queue = ctx.Queue()
            self._result_queue = ctx.Queue()
            self._process = ctx.Process(target=self._process_fn,
                                        args=(self._args_queue, self._result_queue, self._model_path, self._live_updates),
                                        daemon=True)
            self._process.start()
        self._args_queue.put([args, self._cur_stamp])

    def _set_args_sync(self, **args):
        #if self._model is None and self._model_path:
        #    self._model, self._saliency = _load_model_and_saliency(self._model_path, device=self.device)
        if self._renderer_obj is None:
            self._renderer_obj = renderer.Renderer(device=self.device)
            self._renderer_obj._model = self._model
            self._renderer_obj._saliency = self._saliency
        self._cur_result = self._renderer_obj.render(**args)

    def get_result(self):
        assert not self._closed
        if self._result_queue is not None:
            while self._result_queue.qsize() > 0:
                result, stamp = self._result_queue.get()
                if stamp == self._cur_stamp:
                    self._cur_result = result
        return self._cur_result

    def clear_result(self):
        assert not self._closed
        self._cur_args = None
        self._cur_result = None
        self._cur_stamp += 1

    def load_model(self, model_path):
        if self._is_async:
            self._set_args_async(load_model=model_path)
        elif model_path != self._model_path:
            self._model_path = model_path
            if self._renderer_obj is None:
                self._renderer_obj = renderer.Renderer(device=self.device)
            self._model, self._saliency = _load_model_and_saliency(self._model_path, device=self.device)
            self._renderer_obj._model = self._model
            self._renderer_obj._saliency = self._saliency

    def clear_model(self):
        self._model_path = None
        self._model = None
        self._saliency = None

    @staticmethod
    def _process_fn(args_queue, result_queue, model_path, live_updates):
        if sf.util.torch_available:
            import torch
            device = torch.device('cuda')
        else:
            device = None
        renderer_obj = renderer.Renderer(device=device)
        if model_path:
            _model, _saliency = _load_model_and_saliency(model_path, device=device)
            renderer_obj._model = _model
            renderer_obj._saliency = _saliency
        cur_args = None
        cur_stamp = None
        while True:
            while args_queue.qsize() > 0:
                args, stamp = args_queue.get()
                if 'load_model' in args:
                    _model, _saliency = _load_model_and_saliency(args['load_model'], device=device)
                    renderer_obj._model = _model
                    renderer_obj._saliency = _saliency
                if 'quit' in args:
                    return
            # if ((args != cur_args or stamp != cur_stamp)
            if (live_updates and not result_queue.qsize()):
                result = renderer_obj.render(**args)
                if 'error' in result:
                    result.error = renderer.CapturedException(result.error)

                result_queue.put([result, stamp])
                cur_args = args
                cur_stamp = stamp
