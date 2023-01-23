import os
import time
import multiprocessing
import numpy as np
import webbrowser
import pyperclip
import imgui
import glfw
import OpenGL.GL as gl

from typing import List, Any, Optional, Dict, Tuple
from os.path import join, exists
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
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
from slideflow.workbench.utils import EasyDict, _load_model_and_saliency
from slideflow import log

#----------------------------------------------------------------------------

def stylegan_widgets(advanced: bool = True) -> List[Any]:
    from slideflow.gan.stylegan3.stylegan3.viz import (latent_widget,
                                                       pickle_widget,
                                                       stylemix_widget,
                                                       trunc_noise_widget,
                                                       equivariance_widget)
    widgets = [
        pickle_widget.PickleWidget,
        latent_widget.LatentWidget,
        stylemix_widget.StyleMixingWidget,
    ]
    if advanced:
        widgets += [trunc_noise_widget.TruncationNoiseWidget,
                    equivariance_widget.EquivarianceWidget]
    return widgets

#----------------------------------------------------------------------------

class Workbench(imgui_window.ImguiWindow):

    def __init__(
        self,
        low_memory: bool = False,
        widgets: Optional[List[Any]] = None
    ) -> None:
        """Create the main Workbench window.

        Args:
            low_memory (bool): Enable low memory mode, which uses thread pools
                instead of multiprocessing pools when applicable to reduce
                memory footprint, at the cost of decreased performance.
            widgets (list(Any), optional): Additional widgets to render.
        """

        # Initialize TK window in background (for file dialogs)
        Tk().withdraw()

        super().__init__(title=f'Slideflow Workbench')

        # Internals.
        self._dx                = 0
        self._dy                = 0
        self._last_error_print  = None
        self._async_renderer    = AsyncRenderer()
        self._addl_renderers    = dict()
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
        self._about_tex_obj     = None
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
        self._defer_tile_refresh = None
        self._should_close_slide = False
        self._should_close_model = False
        self._pane_w_div        = 37
        self.low_memory         = low_memory

        # Interface.
        self._show_about        = False
        self._show_control      = True
        self._dock_control      = True
        self._show_performance  = False
        self._control_size      = 0
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
        self.tile_zoom          = 1
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
        """An overlay (e.g. tile filter or heatmap) is currently being shown
        over the main view.
        """
        return self.slide_widget.show_overlay or self.heatmap_widget.show

    @property
    def model(self):
        """Tensorflow/PyTorch model currently in use."""
        return self._async_renderer._model

    @property
    def P(self):
        """Slideflow project currently in use."""
        return self.project_widget.P

    @property
    def offset_x(self):
        """Main window offset (x), in points."""
        return self.pane_w

    @property
    def offset_y(self):
        """Main window offset (y), in points."""
        return self.menu_bar_height

    @property
    def offset_x_pixels(self):
        """Main window offset (x), in pixels."""
        return int(self.offset_x * self.pixel_ratio)

    @property
    def offset_y_pixels(self):
        """Main window offset (y), in pixels."""
        return int(self.offset_y * self.pixel_ratio)

    @property
    def has_controls_to_render(self):
        return (
            self.P is not None
            or self.wsi is not None
            or self._model_path is not None
            or any(not hasattr(w, 'visible') or w.visible for w in self.widgets)
        )

    # --- Internals -----------------------------------------------------------

    def _adjust_font_size(self) -> None:
        """Change the interface font size."""
        old = self.font_size
        self.set_font_size(18)
        if self.font_size != old:
            self.skip_frame() # Layout changed.

    def _clear_textures(self) -> None:
        for tex in self._tex_to_delete:
            tex.delete()
        self._tex_to_delete = []

    def _close_model_now(self) -> None:
        """Close the currently loaded model now."""
        self._async_renderer.clear_result()
        self._use_model         = False
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

    def _close_slide_now(self) -> None:
        """Close the currently loaded slide now."""
        self.wsi = None
        self.viewer = None
        self.wsi_thumb = None
        self.x = None
        self.y = None
        self.clear_result()
        self._async_renderer._live_updates = False

    def _draw_about_dialog(self) -> None:
        """Draw the About dialog."""
        if self._show_about:
            import platform
            try:
                import pyvips
                from pyvips.base import version as lv
                libvips_version = f'{lv(0)}.{lv(1)}.{lv(2)}'
                pyvips_version = pyvips.__version__
            except Exception:
                libvips_version = 'NA'
                pyvips_version = 'NA'

            imgui.open_popup('about_popup')
            version_width = imgui.calc_text_size("Version: " + sf.__version__).x
            width = max(200, version_width + self.spacing)
            height = 315
            imgui.set_next_window_content_size(width, 0)
            imgui.set_next_window_position(self.content_width/2 - width/2, self.content_height/2 - height/2)

            about_text =  f"Version: {sf.__version__}\n"
            about_text += f"Python: {platform.python_version()}\n"
            about_text += f"Slide Backend: {sf.slide_backend()}\n"
            about_text += f"Libvips: {libvips_version}\n"
            about_text += f"Pyvips: {pyvips_version}\n"
            about_text += f"OS: {platform.system()} {platform.release()}\n"

            if imgui.begin_popup('about_popup'):

                if self._about_tex_obj is None:
                    about_img = text_utils.about_image()
                    self._about_tex_obj = gl_utils.Texture(image=about_img)
                imgui.text('')
                imgui.text('')
                imgui.same_line(imgui.get_content_region_max()[0]/2 - 32 + self.spacing)
                imgui.image(self._about_tex_obj.gl_id, 64, 64)

                imgui.text('')
                with self.bold_font():
                    self.center_text('Slideflow Workbench')
                imgui.text('')

                for line in about_text.split('\n'):
                    self.center_text(line)
                imgui.text('')
                imgui.same_line(self.spacing)
                if imgui_utils.button('Copy', width=self.button_w/2):
                    pyperclip.copy(about_text)
                imgui.same_line(imgui.get_content_region_max()[0] + self.spacing - self.button_w/2)
                if imgui_utils.button('Close', width=self.button_w/2):
                    self._show_about = False
                imgui.end_popup()

    def _draw_control_pane(self) -> None:
        """Draw the control pane and widgets."""

        if not self.has_controls_to_render:
            self.pane_w = 0
        else:
            _pane_w = self.font_size * self._pane_w_div
            if self._dock_control and self._show_control:
                self.pane_w = _pane_w
                imgui.set_next_window_position(0, self.menu_bar_height)
                imgui.set_next_window_size(self.pane_w, self.content_height - self.menu_bar_height)
                control_kw = dict(
                    closable=False,
                    flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE))
            else:
                imgui.set_next_window_size(_pane_w, self._control_size)
                self.pane_w = 0
                control_kw = dict(
                    closable=True,
                    flags=(imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_ALWAYS_AUTO_RESIZE)
                )

        # Hide control pane if there are no widgets to show
        if self._show_control and self.has_controls_to_render:
            drawing_control_pane = True
            _, self._show_control = imgui.begin('Control Pane', **control_kw)
        else:
            drawing_control_pane = False

        # --- Core widgets (always rendered, not always shown) ----------------
        header_height = self.font_size + (self.spacing * 2)
        self._control_size = self.spacing * 4

        # Slide widget
        if (self.P or self.wsi) and self._show_control:
            expanded, _visible = imgui_utils.collapsing_header('Whole-slide image', default=True)
            self._control_size += header_height
        else:
            expanded = False
        self.slide_widget(expanded and self._show_control)
        self._control_size += self.slide_widget.content_height

        # Project widget
        if self.P and self._show_control:
            expanded, _visible = imgui_utils.collapsing_header('Project', default=True)
            self._control_size += header_height
        else:
            expanded = False
        self.project_widget(expanded and self._show_control)
        self._control_size += self.project_widget.content_height

        # Model widget
        if (self.P or self._model_path) and self._show_control:
            expanded, _visible = imgui_utils.collapsing_header('Model & tile predictions', default=True)
            self._control_size += header_height
        else:
            expanded = False
        self.model_widget(expanded and self._show_control)
        self._control_size += self.model_widget.content_height

        # Heatmap / prediction widget
        if self.viewer is not None and self._model_config is not None and self._show_control:
            expanded, _visible = imgui_utils.collapsing_header('Heatmap & slide prediction', default=True)
            self._control_size += header_height
        else:
            expanded = False
        self.heatmap_widget(expanded and self._show_control)
        self._control_size += self.heatmap_widget.content_height

        # ---------------------------------------------------------------------

        # User-defined widgets
        for header, widgets in self._widgets_by_header():
            if header and self._show_control:
                expanded, _visible = imgui_utils.collapsing_header(header, default=True)
                self._control_size += header_height
            else:
                expanded = True
            for widget in widgets:
                widget(expanded and self._show_control)
                if hasattr(widget, 'content_height'):
                    self._control_size += widget.content_height

        # Render control panel contents, if the control pane is shown.
        if drawing_control_pane:
            imgui.end()

        if not self.has_controls_to_render:
            self.result.message = 'Load a slide with File -> "Open Slide..."'

    def _draw_main_view(self, inp: EasyDict, window_changed: bool) -> None:
        """Update the main window view.

        Draws the slide / picam view, overlay heatmap, overlay box, and ROIs.

        Args:
            inp (EasyDict): Dictionary of user input.
            window_changed (bool): Window size has changed (force refresh).
        """

        max_w = self.content_frame_width - self.offset_x_pixels
        max_h = self.content_frame_height - self.offset_y_pixels

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

    def _draw_menu_bar(self) -> None:
        """Draw the main menu bar (File, View, Help)"""

        if imgui.begin_main_menu_bar():
            # --- File --------------------------------------------------------
            if imgui.begin_menu('File', True):
                if imgui.menu_item('Open Project...', 'Ctrl+P')[1]:
                    self.ask_load_project()
                if imgui.menu_item('Open Slide...', 'Ctrl+O')[1]:
                    self.ask_load_slide()
                if imgui.menu_item('Load Model...', 'Ctrl+M')[1]:
                    self.ask_load_model()
                if imgui.menu_item('Load Heatmap...', 'Ctrl+H', enabled=self._model_path is not None)[1]:
                    self.ask_load_heatmap()
                imgui.separator()
                if imgui.begin_menu('Export...', True):
                    if imgui.menu_item('Main view')[1]:
                        self.capture_widget.save_view()
                    if imgui.menu_item('Tile view')[1]:
                        self.capture_widget.save_tile()
                    if imgui.menu_item('GUI view')[1]:
                        self.capture_widget.save_gui()
                    if imgui.menu_item('Heatmap (PNG)', enabled=(self.rendered_heatmap is not None))[0]:
                        h_img = Image.fromarray(self.rendered_heatmap)
                        h_img.resize(np.array(h_img.size) * 16, Image.NEAREST).save(f'{self.heatmap.slide.name}.png')
                        self.create_toast(f"Saved heatmap image to {self.heatmap.slide.name}.png", icon='success')
                    if imgui.menu_item('Heatmap (NPZ)', enabled=(self.heatmap is not None))[0]:
                        loc = self.heatmap.save_npz()
                        self.create_toast(f"Saved heatmap .npz to {loc}", icon='success')
                    imgui.end_menu()
                imgui.separator()
                if imgui.menu_item('Close Slide')[1]:
                    self.close_slide(True)
                if imgui.menu_item('Close Model')[1]:
                    self.close_model(True)

                # Widgets with "File" menu.
                for w in self.widgets:
                    if hasattr(w, 'file_menu_options'):
                        imgui.separator()
                        w.file_menu_options()

                imgui.separator()
                if imgui.menu_item('Exit', 'Ctrl+Q')[1]:
                    self._exit_trigger = True
                imgui.end_menu()

            # --- View --------------------------------------------------------
            has_wsi = self.viewer and isinstance(self.viewer, wsi_utils.SlideViewer)
            if imgui.begin_menu('View', True):
                if imgui.menu_item('Fullscreen', 'Ctrl+F')[0]:
                    self.toggle_fullscreen()
                if imgui.menu_item('Dock Controls', 'Ctrl+Shift+D')[0]:
                    self._dock_control = not self._dock_control
                imgui.separator()

                # --- Show sub-menu -------------------------------------------
                if imgui.begin_menu('Show', True):
                    if imgui.menu_item('Control Pane', 'Ctrl+Shift+C', selected=self._show_control)[0]:
                        self._show_control = not self._show_control
                    if imgui.menu_item('Performance', 'Ctrl+Shift+P', selected=self._show_performance)[0]:
                        self._show_performance = not self._show_performance
                    if imgui.menu_item('Tile Preview', 'Ctrl+Shift+T', selected=self._show_tile_preview)[0]:
                        self._show_tile_preview = not self._show_tile_preview
                    imgui.separator()
                    if imgui.menu_item('WSI Scale', selected=(has_wsi and self.viewer.show_scale), enabled=has_wsi)[0]:
                        self.viewer.show_scale = not self.viewer.show_scale
                    imgui.separator()
                    imgui.menu_item('Camera View', enabled=False)
                    imgui.end_menu()
                # -------------------------------------------------------------

                imgui.separator()
                if imgui.menu_item('Increase Font Size', 'Ctrl+=')[1]:
                    self.increase_font_size()
                if imgui.menu_item('Decrease Font Size', 'Ctrl+-')[1]:
                    self.decrease_font_size()

                imgui.separator()
                if imgui.menu_item('Increase Tile Zoom', 'Ctrl+]')[1]:
                    self.increase_tile_zoom()
                if imgui.menu_item('Decrease Tile Zoom', 'Ctrl+[')[1]:
                    self.decrease_tile_zoom()
                if imgui.menu_item('Reset Tile Zoom', 'Ctrl+\\')[1]:
                    self.reset_tile_zoom()

                # Widgets with "View" menu.
                for w in self.widgets:
                    if hasattr(w, 'view_menu_options'):
                        imgui.separator()
                        w.view_menu_options()

                imgui.end_menu()

            # --- Help --------------------------------------------------------
            if imgui.begin_menu('Help', True):
                if imgui.menu_item('Get Started')[1]:
                    webbrowser.open('https://slideflow.dev/workbench_tools.html')
                if imgui.menu_item('Documentation')[1]:
                    webbrowser.open('https://slideflow.dev')

                # Widgets with "Help" menu.
                for w in self.widgets:
                    if hasattr(w, 'help_menu_options'):
                        imgui.separator()
                        w.help_menu_options()

                imgui.separator()
                if imgui.menu_item('Release Notes')[1]:
                    webbrowser.open(join(sf.__github__, 'releases/tag', sf.__version__))
                if imgui.menu_item('Report Issue')[1]:
                    webbrowser.open(join(sf.__github__, 'issues'))
                imgui.separator()
                if imgui.menu_item('View License')[1]:
                    webbrowser.open(join(sf.__github__, 'blob/master/LICENSE'))
                if imgui.menu_item('About')[1]:
                    self._show_about = True
                imgui.end_menu()

            version_text = f'slideflow {sf.__version__}'
            imgui.same_line(imgui.get_content_region_max()[0] - (imgui.calc_text_size(version_text)[0] + self.spacing))
            imgui.text(version_text)
            imgui.end_main_menu_bar()

    def _draw_performance_pane(self):
        """Draw the performance and capture window."""

        if self._show_performance:
            _, self._show_performance = imgui.begin('Performance & Capture', closable=True, flags=(imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE))
            self.performance_widget(True)
            self.capture_widget(True)
            imgui.end()
        else:
            self.capture_widget(False)

    def _draw_tile_view(self):
        """Draw the tile view window, displaying the currently rendered tile(s).

        This window will show images rendered by a whole-slide viewer (image
        tile extracted at some x/y location from the slide), or potentially an
        image rendered via some other rendering mechanism as determined through
        renderes set via ``.add_to_render_pipeline()``. For example, images
        rendered by StyleGAN will be shown in this view. This view also shows
        a post-processed, post-normalized rendered image, if available.

        Rendered images are expected to be stored in the OpenGL objects
        ``.tex_obj`` and ``._norm_tex_obj``.
        """

        if self._show_tile_preview:
            has_raw_image = self._tex_obj is not None# and self.tile_px
            has_norm_image = self.model_widget.use_model and self._normalizer is not None and self._norm_tex_obj is not None and self.tile_px

            if not (has_raw_image or has_norm_image):
                width = self.font_size * 8
                height = self.font_size * 3
            else:
                raw_img_w = 0 if not has_raw_image else self._tex_img.shape[0] * self.tile_zoom
                norm_img_w = 0 if not has_norm_image else self._norm_tex_img.shape[0] * self.tile_zoom
                height = self.font_size * 2 + max(raw_img_w, norm_img_w)
                width = raw_img_w + norm_img_w + self.spacing

            imgui.set_next_window_size(width, height)

            if self._tile_preview_is_new:
                imgui.set_next_window_position(self.content_width - width - self.spacing, self.spacing + self.offset_y_pixels)
                self._tile_preview_is_new = False

            if self._tile_preview_image_is_new and (has_raw_image or has_norm_image):
                imgui.set_next_window_position(self.content_width - width - self.spacing, self.spacing + self.offset_y_pixels)
                self._tile_preview_image_is_new = False

            _, self._show_tile_preview = imgui.begin("##tile view", closable=True, flags=(imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_SCROLLBAR))

            # Image preview ===================================================
            dim_color = list(imgui.get_style().colors[imgui.COLOR_TEXT])
            dim_color[-1] *= 0.5
            imgui.begin_child('##pred_image', border=False)
            if has_raw_image:
                imgui.image(self._tex_obj.gl_id, raw_img_w, raw_img_w)
            elif self._model_path is not None:
                imgui.text_colored('Right click to preview', *dim_color)
            else:
                imgui.text_colored('No model loaded', *dim_color)
            imgui.same_line()
            if has_norm_image:
                imgui.image(self._norm_tex_obj.gl_id, norm_img_w, norm_img_w)
            elif self._tex_obj is not None and self.tile_px:
                imgui.text_colored('Normalizer not used', *dim_color)
            imgui.end_child()
            imgui.same_line()
            imgui.end()

    def _glfw_key_callback(self, _window, key, _scancode, action, _mods):
        """Callback for handling keyboard input."""
        super()._glfw_key_callback(_window, key, _scancode, action, _mods)
        if self._control_down and self._shift_down and action == glfw.PRESS and key == glfw.KEY_C:
            self._show_control = not self._show_control
        if self._control_down and self._shift_down and action == glfw.PRESS and key == glfw.KEY_D:
            self._dock_control = not self._dock_control
        if self._control_down and self._shift_down and action == glfw.PRESS and key == glfw.KEY_P:
            self._show_performance = not self._show_performance
        if self._control_down and self._shift_down and action == glfw.PRESS and key == glfw.KEY_T:
            self._show_tile_preview = not self._show_tile_preview
        if self._control_down and action == glfw.PRESS and key == glfw.KEY_Q:
            self._exit_trigger = True
        if self._control_down and action == glfw.PRESS and key == glfw.KEY_O:
            self.ask_load_slide()
        if self._control_down and not self._shift_down and action == glfw.PRESS and key == glfw.KEY_P:
            self.ask_load_project()
        if self._control_down and action == glfw.PRESS and key == glfw.KEY_M:
            self.ask_load_model()
        if self._control_down and action == glfw.PRESS and key == glfw.KEY_H:
            self.ask_load_heatmap()
        if self._control_down and action == glfw.PRESS and key == glfw.KEY_SPACE:
            self.heatmap_widget.show = True
        if self._control_down and action == glfw.RELEASE and key == glfw.KEY_SPACE:
            self.heatmap_widget.show = False
        if self._control_down and action == glfw.PRESS and key == glfw.KEY_LEFT_BRACKET:
            self.decrease_tile_zoom()
        if self._control_down and action == glfw.PRESS and key == glfw.KEY_RIGHT_BRACKET:
            self.increase_tile_zoom()
        if self._control_down and action == glfw.PRESS and key == glfw.KEY_BACKSLASH:
            self.reset_tile_zoom()

    def _handle_user_input(self):
        """Handle user input to support clicking/dragging the main viewer."""

        # Detect right mouse click in the main display.
        clicking, cx, cy, wheel = imgui_utils.click_hidden_window(
            '##result_area',
            x=self.offset_x,
            y=self.offset_y,
            width=self.content_width - self.offset_x,
            height=self.content_height - self.offset_y,
            mouse_idx=1)
        # Detect dragging with left mouse in the main display.
        dragging, dx, dy = imgui_utils.drag_hidden_window(
            '##result_area',
            x=self.offset_x,
            y=self.offset_y,
            width=self.content_width - self.offset_x,
            height=self.content_height - self.offset_y)
        return EasyDict(
            clicking=clicking,
            dragging=dragging,
            wheel=wheel,
            cx=int(cx * self.pixel_ratio),
            cy=int(cy * self.pixel_ratio),
            dx=int(dx * self.pixel_ratio),
            dy=int(dy * self.pixel_ratio)
        )

    def _reload_wsi(
        self,
        path: Optional[str] = None,
        stride: Optional[int] = None,
        use_rois: bool = True
    ) -> None:
        """Reload the currently loaded Whole-Slide Image.

        Args:
            path (str, optional): Path to the slide to reload. If not provided,
                will reload the currently loaded slide.
            stride (int, optional): Stride to use for the loaded slide. If not
                provided, will use the stride value from the currently loaded
                slide.
            use_rois (bool): Use ROIs from the loaded project, if available.
        """
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
        if sf.slide_backend() == 'cucim':
            reader_kwargs = dict(num_workers=os.cpu_count())
        else:
            reader_kwargs = {}
        self.wsi = sf.WSI(
            path,
            tile_px=(self.tile_px if self.tile_px else 256),
            tile_um=(self.tile_um if self.tile_um else 512),
            stride_div=stride,
            rois=rois,
            cache_kw=dict(
                tile_width=512,
                tile_height=512,
                max_tiles=-1,
                threaded=True,
                persistent=True
            ),
            verbose=False,
            **reader_kwargs)
        self.set_viewer(wsi_utils.SlideViewer(self.wsi, **self._viewer_kwargs()))

    def _render_prediction_message(self, message: str) -> None:
        """Render a prediction string to below the tile bounding box.

        Args:
            message (str): Message to render.
        """
        max_w = self.content_frame_width - self.offset_x_pixels
        max_h = self.content_frame_height - self.offset_y_pixels
        tex = text_utils.get_texture(message, size=self.gl_font_size, max_width=max_w, max_height=max_h, outline=2)
        box_w = self.viewer.full_extract_px / self.viewer.view_zoom
        text_pos = np.array([self.box_x + (box_w/2), self.box_y + box_w + self.font_size])
        tex.draw(pos=text_pos, align=0.5, rint=True, color=1)

    def _render_control_pane_contents(self) -> None:
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

    def _viewer_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments to use for loading a Viewer."""

        return dict(
            width=self.content_frame_width - self.offset_x_pixels,
            height=self.content_frame_height - self.offset_y_pixels,
            x_offset=self.offset_x_pixels,
            y_offset=self.offset_y_pixels,
            normalizer=(self._normalizer if self._normalize_wsi else None)
        )

    def _widgets_by_header(self) -> List[Tuple[str, List[Any]]]:
        """Return all unique widget headers and their corresponding widgets.

        Returns:
            List of tuple (header, List of widgets) for all unique headers.
        """
        def _get_header(w):
            return '' if not hasattr(w, 'header') else w.header

        headers = []
        for widget in self.widgets:
            _widget_header = _get_header(widget)
            if _widget_header not in headers:
                headers.append(_widget_header)

        return [
            (header, [w for w in self.widgets if _get_header(w) == header])
            for header in headers
        ]

    # --- Public methods ------------------------------------------------------

    def add_to_render_pipeline(
        self,
        renderer: Any,
        name: Optional[str] = None
    ) -> None:
        """Add a renderer to the rendering pipeline."""
        if name is not None:
            self._addl_renderers[name] = renderer
        self._async_renderer.add_to_render_pipeline(renderer)

    def ask_load_heatmap(self):
        """Prompt user for location of exported heatmap (*.npz) and load."""
        npz_path = askopenfilename(title="Load heatmap...", filetypes=[("*.npz", "*.npz")])
        if npz_path:
            self.load_heatmap(npz_path)

    def ask_load_model(self):
        """Prompt user for location of a model and load."""
        if sf.backend() == 'tensorflow':
            model_path = askdirectory(title="Load model (directory)...")
        else:
            model_path = askopenfilename(title="Load model...", filetypes=[("zip", ".zip"), ("All files", ".*")])
        if model_path:
            self.load_model(model_path, ignore_errors=True)

    def ask_load_project(self):
        """Prompt user for location of a project and load."""
        project_path = askdirectory(title="Load project (directory)...")
        if project_path:
            self.load_project(project_path, ignore_errors=True)

    def ask_load_slide(self):
        """Prompt user for location of a slide and load."""
        slide_path = askopenfilename(title="Load slide...", filetypes=[("Aperio ScanScope", ("*.svs", "*.svslide")),
                                                                       ("Hamamatsu", ("*.ndpi", "*.vms", "*.vmu")),
                                                                       ("Leica", "*.scn"),
                                                                       ("MIRAX", "*.mrxs"),
                                                                       ("Roche, Ventana", "*.bif"),
                                                                       ("Pyramid TIFF", ("*.tiff", "*.tif")),
                                                                       ("JPEG", (".jpg", "*.jpeg")),
                                                                       ("All files", ".*")])
        if slide_path:
            self.load_slide(slide_path, ignore_errors=True)

    def autoload(self, path, ignore_errors=False):
        """Automatically load a path, detecting if the path is a slide, model, or project."""
        sf.log.info(f"Attempting to load {path}")
        if sf.util.is_project(path):
            self.load_project(path, ignore_errors=ignore_errors)
        elif sf.util.is_slide(path):
            self.load_slide(path, ignore_errors=ignore_errors)
        elif sf.util.is_model(path):
            self.load_model(path, ignore_errors=ignore_errors)
        elif path.endswith('npz'):
            self.load_heatmap(path)
        else:
            # See if any widgets implement a drag_and_drop_hook() method
            for widget in self.widgets:
                sf.log.info(f"Attempting load with widget {widget}")
                if hasattr(widget, 'drag_and_drop_hook'):
                    widget.drag_and_drop_hook(path)

    def clear_overlay(self) -> None:
        """Remove the currently overlay image."""
        self._overlay_tex_img = None
        self.overlay = None

    def clear_result(self) -> None:
        """Clear all shown results and images."""
        self.clear_model_results()
        self.clear_overlay()
        self.result = EasyDict()
        self._wsi_tex_img = None
        if self.viewer:
            self.viewer.clear()

    def clear_message(self, msg: str = None) -> bool:
        """Clear a specific message from display, if the message is being shown.

        Args:
            msg (str): Message to clear.

        Returns:
            bool: Whether message was cleared from display.
        """
        if msg is None or self._message == msg:
            self._message = None
            return True
        return False

    def clear_model_results(self) -> None:
        """Clear all model results and associated images."""
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

    def close(self) -> None:
        """Close the application and renderer."""
        super().close()
        if self._async_renderer is not None:
            self._async_renderer.close()
            self._async_renderer = None

    def close_model(self, now: bool = False) -> None:
        """Close the currently loaded model.

        Args:
            now (bool): Close the model now, instead of at the end of the frame.
                Defaults to False (closes model at frame end).
        """
        if now:
            self._close_model_now()
            self._should_close_model = False
        else:
            self._should_close_model = True

    def close_slide(self, now: bool = False) -> None:
        """Close the currently loaded slide.

        Args:
            now (bool): Close the slide now, instead of at the end of the frame.
                Defaults to False (closes slide at frame end).
        """
        if now:
            self._close_slide_now()
            self._should_close_slide = False
        else:
            self._should_close_slide = True

    def defer_rendering(self, num_frames: int = 1) -> None:
        """Defer rendering for a number of frames."""
        self._defer_rendering = max(self._defer_rendering, num_frames)

    def draw_frame(self) -> None:
        """Main draw loop."""

        self.begin_frame()

        self.args = EasyDict(use_model=False, use_uncertainty=False, use_saliency=False)
        self.button_w = self.font_size * 5
        self.label_w = round(self.font_size * 4.5)
        self.menu_bar_height = self.font_size + self.spacing

        max_w = self.content_frame_width - self.offset_x_pixels
        max_h = self.content_frame_height - self.offset_y_pixels
        window_changed = (self._content_width != self.content_width
                          or self._content_height != self.content_height
                          or self._pane_w != self.pane_w)

        # Process drag-and-drop files
        paths = self.pop_drag_and_drop_paths()
        if paths is not None and len(paths) >= 1:
            self.autoload(paths[0], ignore_errors=True)

        self._clear_textures()
        self._draw_menu_bar()
        self._draw_control_pane()
        self._draw_performance_pane()
        self._draw_about_dialog()

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

        # Main display.
        if self.viewer:
            self._draw_main_view(user_input, window_changed)

        # --- Render arguments ------------------------------------------------
        self.args.x = self.x
        self.args.y = self.y
        if (self._model_config is not None and self._use_model):
            self.args.tile_px = self._model_config['tile_px']
            self.args.tile_um = self._model_config['tile_um']
            if 'img_format' in self._model_config and self._use_model_img_fmt:
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
            self.args.tile_um = self.viewer.tile_um

        if self.has_live_viewer():
            self.args.viewer = None
        else:
            self.args.viewer = self.viewer
        # ---------------------------------------------------------------------

        # Render control pane contents.
        if self._show_control and self.has_controls_to_render:
            self._render_control_pane_contents()

        # Render user widgets.
        for widget in self.widgets:
            if hasattr(widget, 'render'):
                widget.render()

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

        # Update input image textures (tile view).
        middle_pos = np.array([self.offset_x_pixels + max_w/2, max_h/2])
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
            tex = text_utils.get_texture(_msg, size=self.gl_font_size, max_width=max_w, max_height=max_h, outline=2)
            tex.draw(pos=middle_pos, align=0.5, rint=True, color=1)

        # Render the tile view.
        self._draw_tile_view()

        # Draw prediction message next to box on main display.
        if (self._use_model
           and self._predictions is not None
           and not isinstance(self._predictions, list)
           and self.viewer is not None):
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
        self.end_frame()

    @staticmethod
    def get_default_widgets() -> List[Any]:
        """Returns a list of the default non-mandatory widgets."""

        from slideflow.workbench.layer_umap_widget import LayerUMAPWidget
        return [LayerUMAPWidget]

    def get_renderer(self, name: str) -> Any:
        """Check for the given additional renderer in the rendering pipeline."""
        return self._addl_renderers[name]

    def has_live_viewer(self) -> bool:
        """Check if the current viewer is a live viewer (e.g. camera feed)."""
        return (self.viewer is not None and self.viewer.live)

    def has_uq(self) -> bool:
        """Check if the current model supports uncertainty quantification."""
        return (self._model_path is not None
                and self._model_config is not None
                and 'uq' in self._model_config['hp']
                and self._model_config['hp']['uq'])

    def increase_tile_zoom(self) -> None:
        """Increase zoom of tile view two-fold."""
        self.tile_zoom *= 2

    def decrease_tile_zoom(self) -> None:
        """Decrease zoom of tile view by half."""
        self.tile_zoom /= 2

    def reset_tile_zoom(self) -> None:
        """Reset tile zoom level."""
        self.tile_zoom = 1

    def load_heatmap(self, path: str) -> None:
        """Load a saved heatmap (*.npz).

        Args:
            path (str): Path to exported heatmap in *.npz format, as generated
                by Heatmap.save() or Heatmap.save_npz().
        """
        try:
            self.heatmap_widget.load(path)
            self.create_toast(f"Loaded heatmap at {path}", icon="success")

        except Exception as e:
            log.debug("Exception raised loading heatmap: {}".format(e))
            self.create_toast(f"Error loading heatmap at {path}", icon="error")

    def load_model(self, model: str, ignore_errors: bool = False) -> None:
        """Load the given model.

        Args:
            model (str): Path to Slideflow model (in either backend).
            ignore_errors (bool): Do not fail if an error is encountered.
                Defaults to False.
        """
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

            self.create_toast(f"Loaded model at {model}", icon="success")

        except Exception as e:
            self.model_widget.cur_model = None
            if model == '':
                log.debug("Exception raised: no model loaded.")
                self.result = EasyDict(message='No model loaded')
            else:
                log.debug("Exception raised (ignore_errors={}): {}".format(ignore_errors, e))
                self.create_toast(f"Error loading model at {model}", icon="error")
                self.result = EasyDict(error=renderer.CapturedException())
            if not ignore_errors:
                raise
        log.debug("Model loading complete (path={})".format(self._model_path))

    def load_project(self, project: str, ignore_errors: bool = False) -> None:
        """Load the given project.

        Args:
            project (str): Path to Slideflow project.
            ignore_errors (bool): Do not fail if an error is encountered.
                Defaults to False.
        """
        self.project_widget.load(project, ignore_errors=ignore_errors)

    def load_slide(self, slide: str, ignore_errors: bool = False) -> None:
        """Load the given slide.

        Args:
            slide (str): Path to whole-slide image.
            ignore_errors (bool): Do not fail if an error is encountered.
                Defaults to False.
        """
        self.slide_widget.load(slide, ignore_errors=ignore_errors)

    def print_error(self, error: str) -> None:
        """Print the given error message."""
        error = str(error)
        if error != self._last_error_print:
            print('\n' + error + '\n')
            self._last_error_print = error

    def reload_model(self) -> None:
        """Reload the current model."""
        self._async_renderer.load_model(self._model_path)

    def reload_viewer(self) -> None:
        """Reload the current main viewer."""
        if self.viewer is not None:
            self.viewer.close()
            if isinstance(self.viewer, wsi_utils.SlideViewer):
                self.set_viewer(wsi_utils.SlideViewer(self.wsi, **self._viewer_kwargs()))
            else:
                self.viewer.reload(**self._viewer_kwargs())

    def set_message(self, msg: str) -> None:
        """Set a message for display."""
        self._message = msg

    def set_viewer(self, viewer: Any) -> None:
        """Set the main viewer."""
        log.debug("Setting viewer to {}".format(viewer))
        self.viewer = viewer
        self._async_renderer._live_updates = viewer.live
        self._async_renderer.set_async(viewer.live)

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
        self._umap_encoders = None
        self._live_updates  = False
        self.tile_px        = None
        self.extract_px     = None
        self._addl_render   = []

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

    def add_to_render_pipeline(self, renderer):
        if self.is_async:
            raise ValueError("Cannot add to rendering pipeline when in "
                             "asynchronous mode.")
        self._addl_render += [renderer]
        if self._renderer_obj is not None:
            self._renderer_obj.add_renderer(renderer)


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
        if self._renderer_obj is None:
            self._renderer_obj = renderer.Renderer(device=self.device)
            for _renderer in self._addl_render:
                self._renderer_obj.add_renderer(_renderer)
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
                for _renderer in self._addl_render:
                    self._renderer_obj.add_renderer(_renderer)
            self._model, self._saliency, self._umap_encoders = _load_model_and_saliency(self._model_path, device=self.device)
            self._renderer_obj._model = self._model
            self._renderer_obj._saliency = self._saliency
            self._renderer_obj._umap_encoders = self._umap_encoders

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
            _model, _saliency, _umap_encoders = _load_model_and_saliency(model_path, device=device)
            renderer_obj._model = _model
            renderer_obj._saliency = _saliency
            renderer_obj._umap_encoders = _umap_encoders
        cur_args = None
        cur_stamp = None
        while True:
            while args_queue.qsize() > 0:
                args, stamp = args_queue.get()
                if 'load_model' in args:
                    _model, _saliency, _umap_encoders = _load_model_and_saliency(args['load_model'], device=device)
                    renderer_obj._model = _model
                    renderer_obj._saliency = _saliency
                    renderer_obj._umap_encoders = _umap_encoders
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
