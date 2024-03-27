import os
import time
import numpy as np
import webbrowser
import pyperclip
import imgui
import glfw
import OpenGL.GL as gl
from contextlib import contextmanager
from typing import List, Any, Optional, Dict, Union, Tuple
from os.path import join, dirname, abspath
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory

import slideflow as sf
from slideflow import log

from .gui import imgui_utils
from .gui import gl_utils
from .gui import text_utils
from .gui.theme import StudioTheme
from .gui.window import ImguiWindow
from .gui.viewer import SlideViewer
from .widgets import (
    ProjectWidget, SlideWidget, ModelWidget, HeatmapWidget, PerformanceWidget,
    CaptureWidget, SettingsWidget, ExtensionsWidget, Widget
)
from .utils import EasyDict, prediction_to_string, StatusMessage
from ._renderer import Renderer
from ._render_manager import AsyncRenderManager, Renderer, CapturedException

OVERLAY_GRID    = 0
OVERLAY_WSI     = 1
OVERLAY_VIEW    = 2

# -----------------------------------------------------------------------------

class Studio(ImguiWindow):

    def __init__(
        self,
        low_memory: bool = False,
        widgets: Optional[List[Any]] = None,
        skip_tk_init: bool = False,
        theme: Optional[StudioTheme] = None,
    ) -> None:
        """Create the main Studio window.

        Slideflow Studio is started by running the studio module.

        .. code-block:: bash

            python -m slideflow.studio

        Args:
            low_memory (bool): Enable low memory mode, which uses thread pools
                instead of multiprocessing pools when applicable to reduce
                memory footprint, at the cost of decreased performance.
            widgets (list(Any), optional): Additional widgets to render.
        """

        # Initialize TK window in background (for file dialogs)
        if not skip_tk_init:
            Tk().withdraw()

        if theme is None:
            theme = StudioTheme()

        super().__init__(
            title=f'Slideflow Studio',
            background=theme.main_background
        )

        # Internals.
        self._dx                = 0
        self._dy                = 0
        self._last_error_print  = None
        self._render_manager    = AsyncRenderManager()
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
        self._bg_logo           = None
        self._message           = None
        self._pred_message      = None
        self.low_memory         = low_memory
        self._suspend_mouse_input       = False
        self._suspend_keyboard_input    = False
        self._status_message            = None
        self._force_enable_tile_preview = False

        # Interface.
        self._show_about                = False
        self._show_tile_preview         = True
        self._tile_preview_is_new       = True
        self._tile_preview_image_is_new = True
        self._show_overlays             = True
        self._show_mpp_zoom_popup       = False
        self._input_mpp                 = 1.
        self._box_color                 = [1, 0, 0]
        self.theme                      = theme

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
        self.overlay_original   = None
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
        self.mouse_x            = None
        self.mouse_y            = None
        self._mouse_screen_x    = 0
        self._mouse_screen_y    = 0
        self.menu_bar_height    = self.font_size + self.spacing

        # Control sidebar.
        self.sidebar            = Sidebar(self)

        # Core widgets.
        self.project_widget     = ProjectWidget(self)
        self.slide_widget       = SlideWidget(self)
        self.model_widget       = ModelWidget(self)
        self.heatmap_widget     = HeatmapWidget(self)
        self.performance_widget = PerformanceWidget(self)
        self.capture_widget     = CaptureWidget(self)
        self.settings_widget    = SettingsWidget(self)

        # User-defined widgets.
        self.widgets = []
        if widgets is None:
            widgets = self.get_default_widgets()
        self.add_widgets(widgets)

        # Extensions widget.
        self.extensions_widget  = ExtensionsWidget(self)

        # Initialize window.
        self.set_window_icon(imgui_utils.logo_image())
        self.set_position(0, 0)
        self._update_window_limits()
        self._set_default_font_size()
        self.skip_frame() # Layout may change after first frame.
        self.load_slide('')

    @property
    def show_overlay(self):
        """An overlay (e.g. tile filter or heatmap) is currently being shown
        over the main view.
        """
        return ((self.slide_widget.show_overlay or self.heatmap_widget.show)
                and self._show_overlays)

    @property
    def model(self):
        """Tensorflow/PyTorch model currently in use."""
        return self._render_manager._model

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
    def status_bar_height(self):
        return self.font_size + self.spacing

    @property
    def mouse_is_over_viewer(self):
        """Mouse is currently over the main viewer."""
        cx, cy = imgui.get_mouse_pos()
        cx -= self.offset_x
        cy -= self.offset_y
        return (self.viewer is not None
                and self.viewer.is_in_view(cx, cy))

    @property
    def tile_preview_enabled(self):
        """Show a tile preview when right clicking."""
        return self._model_path or self._force_enable_tile_preview

    # --- Internals -----------------------------------------------------------

    def _set_default_font_size(self) -> None:
        """Change the interface font size."""
        old = self.font_size
        self.set_font_size(int(18 / self.pixel_ratio))
        if self.font_size != old:
            self.skip_frame()  # Layout changed.

    def _clear_textures(self) -> None:
        """Remove all textures."""
        for tex in self._tex_to_delete:
            tex.delete()
        self._tex_to_delete = []

    def _close_model_now(self) -> None:
        """Close the currently loaded model now."""
        self._render_manager.clear_result()
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
        self._render_manager.clear_model()
        self.clear_model_results()
        self.heatmap_widget.reset()

    def _close_slide_now(self) -> None:
        """Close the currently loaded slide now."""
        self.wsi = None
        self.viewer = None
        self.wsi_thumb = None
        self.x = None
        self.y = None
        self.mouse_x = None
        self.mouse_y = None
        self.clear_result()
        self._render_manager._live_updates = False
        self._heatmap_tex_img   = None
        self._heatmap_tex_obj   = None
        self.heatmap_widget.reset()
        self.set_title("Slideflow Studio")

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
                    about_img = imgui_utils.logo_image().resize((96, 96))
                    self._about_tex_obj = gl_utils.Texture(image=about_img)
                imgui.text('')
                imgui.text('')
                imgui.same_line(imgui.get_content_region_max()[0]/2 - 48 + self.spacing)
                imgui.image(self._about_tex_obj.gl_id, 96, 96)

                imgui.text('')
                with self.bold_font():
                    self.center_text('Slideflow Studio')
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

    def _draw_mpp_zoom_dialog(self):
        """Show a dialog that prompts the user to specify microns-per-pixel."""
        if not self._show_mpp_zoom_popup:
            return

        window_size = (self.font_size * 18, self.font_size * 7)
        self.center_next_window(*window_size)
        imgui.set_next_window_size(*window_size)
        _, opened = imgui.begin('Zoom to Microns-Per-Pixel (MPP)', closable=True, flags=imgui.WINDOW_NO_RESIZE)
        if not opened:
            self._show_mpp_zoom_popup = False

        imgui.text("Zoom the current view to a given MPP.")
        imgui.separator()
        imgui.text('')
        imgui.same_line(self.font_size*4)
        with imgui_utils.item_width(self.font_size*4):
            _changed, self._input_mpp = imgui.input_float('MPP##input_mpp', self._input_mpp, format='%.3f')
        imgui.same_line()
        if self._input_mpp:
            mag = f'{10/self._input_mpp:.1f}x'
        else:
            mag = '-'
        imgui.text(mag)
        if self.sidebar.full_button("Zoom", width=-1):
            self.viewer.zoom_to_mpp(window_size[0] / 2, window_size[1] / 2, self._input_mpp)
            self._show_mpp_zoom_popup = False
        imgui.end()

    def _draw_control_pane(self) -> None:
        """Draw the control pane and widgets."""
        self.sidebar.draw()

    def _draw_empty_background(self):
        """Render an empty background with the Studio logo."""
        if self._bg_logo is None:
            bg_path = join(dirname(abspath(__file__)), 'gui', 'logo_dark_outline.png')
            img = np.array(Image.open(bg_path))
            self._bg_logo = gl_utils.Texture(image=img, bilinear=True)
        self._bg_logo.draw(
            pos=(self.content_frame_width//2, self.content_frame_height//2),
            zoom=0.75,
            align=0.5,
            rint=True,
            anchor='center'
        )

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
            self.mouse_x, self.mouse_y = self.viewer.display_coords_to_wsi_coords(inp.cx, inp.cy, offset=False)

        # Render slide view.
        self.viewer.render(max_w, max_h)

        # Render overlay heatmap.
        if self.overlay is not None and self.show_overlay:
            self.viewer.render_overlay(
                self.overlay,
                dim=self._overlay_wsi_dim,
                offset=self._overlay_offset_wsi_dim)

        # Render overlay tooltip, if hovered.
        if self.overlay_original is not None and self.show_overlay:
            self.viewer.render_overlay_tooltip(self.overlay_original)

        # Calculate location for model display.
        if (self.tile_preview_enabled
            and inp.clicking
            and not inp.dragging
            and self.viewer.is_in_view(inp.cx, inp.cy)):

            wsi_x, wsi_y = self.viewer.display_coords_to_wsi_coords(inp.cx, inp.cy, offset=False)
            self.x = wsi_x - (self.viewer.full_extract_px/2)
            self.y = wsi_y - (self.viewer.full_extract_px/2)

        # Show box around location that a tile is being extracted for preview.
        if self.x is not None and self.y is not None:
            if inp.clicking or inp.dragging or inp.wheel or window_changed:
                self.box_x, self.box_y = self.viewer.wsi_coords_to_display_coords(self.x, self.y)
            tw = self.viewer.full_extract_px / self.viewer.view_zoom

            # Draw box on main display.
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            gl.glLineWidth(3)
            box_pos = np.array([self.box_x, self.box_y])
            gl_utils.draw_rect(pos=box_pos, size=np.array([tw, tw]), color=self._box_color, mode=gl.GL_LINE_LOOP)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glLineWidth(1)

        # Render ROIs.
        self.viewer.late_render()

    def _draw_menu_bar(self) -> None:
        """Draw the main menu bar (File, View, Help)"""

        if imgui.begin_main_menu_bar():
            # --- File --------------------------------------------------------
            if imgui.begin_menu('File', True):
                if imgui.menu_item('New Project...', 'Ctrl+N')[1]:
                    self.project_widget.new_project()
                imgui.separator()
                if imgui.menu_item('Open Project...', 'Ctrl+P')[1]:
                    self.ask_load_project()
                if imgui.menu_item('Open Slide...', 'Ctrl+O')[1]:
                    self.ask_load_slide()
                if imgui.menu_item('Load Model...', 'Ctrl+M')[1]:
                    self.ask_load_model()
                if imgui.menu_item('Load Heatmap...', 'Ctrl+H', enabled=self._model_path is not None)[1]:
                    self.ask_load_heatmap()

                # Widgets with "Open" menu options.
                for w in self.widgets:
                    if hasattr(w, 'open_menu_options'):
                        w.open_menu_options()

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
            has_wsi = self.viewer and isinstance(self.viewer, SlideViewer)
            if imgui.begin_menu('View', True):
                if imgui.menu_item('Fullscreen', 'Alt+Enter')[0]:
                    self.toggle_fullscreen()
                imgui.separator()

                # --- Show sub-menu -------------------------------------------
                if imgui.begin_menu('Show', True):
                    if imgui.menu_item('Tile Preview', 'Ctrl+Shift+T', selected=self._show_tile_preview)[0]:
                        self._show_tile_preview = not self._show_tile_preview
                    imgui.separator()
                    if imgui.menu_item('Thumbnail', selected=(has_wsi and self.viewer.show_thumbnail), enabled=has_wsi)[0]:
                        self.viewer.show_thumbnail = not self.viewer.show_thumbnail
                    if imgui.menu_item('Scale', selected=(has_wsi and self.viewer.show_scale), enabled=has_wsi)[0]:
                        self.viewer.show_scale = not self.viewer.show_scale

                    # Widgets with "View" menu.
                    for w in self.widgets:
                        if hasattr(w, 'show_menu_options'):
                            imgui.separator()
                            w.show_menu_options()

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
                if imgui.menu_item('Zoom to MPP', 'Ctrl+/')[1]:
                    self.ask_zoom_to_mpp()
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
                    webbrowser.open('https://slideflow.dev/studio')
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
            imgui_utils.right_aligned_text(version_text, spacing=self.spacing)
            imgui.end_main_menu_bar()

    def _draw_status_bar(self) -> None:
        """Draw the bottom status bar."""

        h = self.status_bar_height
        r = self.pixel_ratio
        y_pos = int((self.content_frame_height - (h * r)) / r)
        imgui.set_next_window_position(0-2, y_pos)
        imgui.set_next_window_size(self.content_width+4, h)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *self.theme.main_background)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, [10, 5])

        imgui.begin('Status bar', closable=True, flags=(imgui.WINDOW_NO_RESIZE
                                                        | imgui.WINDOW_NO_COLLAPSE
                                                        | imgui.WINDOW_NO_TITLE_BAR
                                                        | imgui.WINDOW_NO_MOVE
                                                        | imgui.WINDOW_NO_SCROLLBAR))

        # Backend
        backend = sf.slide_backend()
        if backend == 'cucim':
            tex = self.sidebar._button_tex[f'small_cucim'].gl_id
            imgui.image(tex, self.font_size, self.font_size)
            imgui.same_line()
            imgui.text_colored('cuCIM', 0.55, 1, 0.47, 1)
        elif backend == 'libvips':
            tex = self.sidebar._button_tex[f'small_vips'].gl_id
            imgui.image(tex, self.font_size, self.font_size)
            imgui.same_line()
            imgui.text_colored('VIPS', 0.47, 0.65, 1, 1)
        else:
            imgui.text(backend)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Slide backend")

        # Low memory mode
        if self.low_memory:
            tex = self.sidebar._button_tex[f'small_lowmem'].gl_id
            imgui.same_line()
            imgui.image(tex, self.font_size, self.font_size)
            imgui.same_line()
            imgui.text_colored("Low memory mode", 0.99, 0.75, 0.42, 1)

        # Status messages
        if self._status_message:
            self._status_message.render()

        # Location / MPP
        if self.viewer and hasattr(self.viewer, 'mpp') and self.mouse_x is not None:
            imgui_utils.right_aligned_text('x={:<8} y={:<8} mpp={:.3f}'.format(
                int(self.mouse_x), int(self.mouse_y), self.viewer.mpp)
            )
        elif self.viewer and self.mouse_x is not None:
            imgui_utils.right_aligned_text(
                'x={:<8} y={:<8}'.format(int(self.mouse_x), int(self.mouse_y))
            )

        imgui.end()
        imgui.pop_style_color(1)
        imgui.pop_style_var(1)

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
            has_raw_image = self._tex_obj is not None
            has_norm_image = 'normalized' in self.result and self._norm_tex_obj is not None  # self.model_widget.use_model and self._normalizer is not None and self._norm_tex_obj is not None and self.tile_px
            if not has_raw_image:
                return

            if not (has_raw_image or has_norm_image):
                width = self.font_size * 8
                height = self.font_size * 3
            else:
                raw_img_w = 0 if not has_raw_image else self._tex_img.shape[0] * self.tile_zoom
                norm_img_w = 0 if not has_norm_image else self._norm_tex_img.shape[0] * self.tile_zoom
                height = self.font_size * 2 + max(raw_img_w, norm_img_w)
                width = raw_img_w + norm_img_w + self.spacing*2

            imgui.set_next_window_size(width, height)

            if self._tile_preview_is_new:
                imgui.set_next_window_position(
                    self.content_width - width - self.spacing,
                    self.content_height - height - self.spacing - self.status_bar_height
                )
                self._tile_preview_is_new = False

            if self._tile_preview_image_is_new and (has_raw_image or has_norm_image):
                imgui.set_next_window_position(
                    self.content_width - width - self.spacing,
                    self.content_height - height - self.spacing - self.status_bar_height
                    )
                self._tile_preview_image_is_new = False

            _, self._show_tile_preview = imgui.begin(
                "##tile view",
                closable=True,
                flags=(imgui.WINDOW_NO_COLLAPSE
                       | imgui.WINDOW_NO_RESIZE
                       | imgui.WINDOW_NO_SCROLLBAR)
            )

            # Image preview ===================================================
            dim_color = list(imgui.get_style().colors[imgui.COLOR_TEXT])
            dim_color[-1] *= 0.5
            imgui.begin_child('##pred_image', border=False)
            imgui.image(self._tex_obj.gl_id, raw_img_w, raw_img_w)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Raw image")
            imgui.same_line()
            if has_norm_image:
                imgui.image(self._norm_tex_obj.gl_id, norm_img_w, norm_img_w)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Stain-normalized image")
            elif self._tex_obj is not None and self.tile_px:
                imgui.text_colored('Normalizer not used', *dim_color)
            imgui.end_child()
            imgui.same_line()
            imgui.end()

    def _glfw_key_callback(self, _window, key, _scancode, action, _mods):
        """Callback for handling keyboard input."""
        super()._glfw_key_callback(_window, key, _scancode, action, _mods)

        if self._suspend_keyboard_input:
            return

        if self._control_down and action == glfw.PRESS and key == glfw.KEY_N:
            self.project_widget.new_project()
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
        if self._control_down and action == glfw.PRESS and key == glfw.KEY_SLASH:
            self.ask_zoom_to_mpp()
        if self._control_down and action == glfw.PRESS and key == glfw.KEY_BACKSLASH:
            self.reset_tile_zoom()

        self.slide_widget.keyboard_callback(key, action)
        self.project_widget.keyboard_callback(key, action)
        for widget in self.widgets:
            if hasattr(widget, 'keyboard_callback'):
                widget.keyboard_callback(key, action)

    def suspend_mouse_input_handling(self):
        """Suspend mouse input handling."""
        self._suspend_mouse_input = True

    def resume_mouse_input_handling(self):
        """Resume mouse input handling."""
        self._suspend_mouse_input = False

    def suspend_keyboard_input(self) -> bool:
        """Suspend keyboard input handling."""
        self._suspend_keyboard_input = True

    def resume_keyboard_input(self) -> bool:
        """Resume keyboard input handling."""
        self._suspend_keyboard_input = False

    def mouse_input_is_suspended(self) -> bool:
        """Check if mouse input handling is suspended."""
        return self._suspend_mouse_input

    def is_mouse_down(self, mouse_idx: int = 0) -> bool:
        """Check if the mouse is currently down."""
        if self._suspend_mouse_input:
            return False
        return imgui.is_mouse_down(mouse_idx)

    def is_mouse_released(self, mouse_idx: int = 0) -> bool:
        """Check if the mouse was released."""
        if self._suspend_mouse_input:
            return False
        return imgui.is_mouse_released(mouse_idx)

    def _handle_user_input(self):
        """Handle user input to support clicking/dragging the main viewer."""

        self._mouse_screen_x, self._mouse_screen_y = imgui.get_mouse_pos()
        # Detect right mouse click in the main display.
        clicking, cx, cy, wheel = imgui_utils.click_hidden_window(
            '##result_area',
            x=self.offset_x,
            y=self.offset_y,
            width=self.content_width - self.offset_x,
            height=self.content_height - self.offset_y,
            mouse_idx=1)

        # Ignore right click if the slide widget
        # is capturing an ROI.
        if self.slide_widget.editing_rois:
            clicking = False

        # Detect dragging with left mouse in the main display.
        dragging, dx, dy = imgui_utils.drag_hidden_window(
            '##result_area',
            x=self.offset_x,
            y=self.offset_y,
            width=self.content_width - self.offset_x,
            height=self.content_height - self.offset_y)

        # Suspend mouse input handling if the user is interacting with a widget.
        if self._suspend_mouse_input:
            clicking, dragging, wheel = False, False, 0
            dx, dy = 0, 0

        return EasyDict(
            clicking=clicking,
            dragging=dragging,
            wheel=wheel,
            cx=int(cx * self.pixel_ratio),
            cy=int(cy * self.pixel_ratio),
            dx=int(dx * self.pixel_ratio),
            dy=int(dy * self.pixel_ratio)
        )

    def _reload_and_return_wsi(
        self,
        path: Optional[str] = None,
        stride: Optional[int] = None,
        use_rois: bool = True,
        tile_px: Optional[int] = None,
        tile_um: Optional[Union[str, int]] = None,
        **kwargs
    ) -> Optional[sf.WSI]:
        """Reload and return a Whole-Slide Image, with modified parameters.

        Args:
            path (str, optional): Path to the slide to reload. If not provided,
                will reload the currently loaded slide.
            stride (int, optional): Stride to use for the loaded slide. If not
                provided, will use the stride value from the currently loaded
                slide.
            use_rois (bool): Use ROIs from the loaded project, if available.

        Returns:
            slideflow.WSI: Reloaded slide.

        """
        if self.wsi is None and path is None:
            return None

        # Path to slide.
        if path is None:
            path = self.wsi.path

        # Stride.
        if stride is None and self.wsi is None:
            stride = 1
        elif stride is None:
            stride = self.wsi.stride_div

        # ROI filter method.
        if self.wsi is None:
            roi_filter_method = 'center'
        else:
            roi_filter_method = self.wsi.roi_filter_method

        # ROIs.
        if self.wsi is not None and path == self.wsi.path:
            roi_method = self.wsi.roi_method
            prior_rois = self.wsi.rois
            rois = None
        elif self.P and use_rois:
            rois = self.P.dataset().rois()
            roi_method, prior_rois = None, None
        else:
            roi_method, prior_rois, rois = None, None, None

        # Cap the number of workers in the CUCIM backend.
        if sf.slide_backend() == 'cucim':
            kwargs['num_workers'] = sf.util.num_cpu(default=4)

        # Tile size.
        if tile_px is None:
            tile_px = (self.tile_px if self.tile_px else 256)
        if tile_um is None:
            tile_um = (self.tile_um if self.tile_um else 512)

        # Pass through QC mask if the slide is already loaded.
        qc_mask = None if not self.wsi else self.wsi.get_qc_mask(roi=False)

        try:
            wsi = sf.WSI(
                path,
                tile_px=tile_px,
                tile_um=tile_um,
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
                mpp=self.slide_widget.manual_mpp,
                use_bounds=self.settings_widget.use_bounds,
                roi_filter_method=roi_filter_method,
                simplify_roi_tolerance=self.settings_widget.simplify_tolerance,
                **kwargs)
        except sf.errors.IncompatibleBackendError:
            self.create_toast(
                title="Incompatbile slide",
                message='Slide type "{}" is incompatible with the {} backend.'.format(
                    sf.util.path_to_ext(path), sf.slide_backend()),
                icon='error'
            )
            return None
        else:
            # Reapply QC
            if qc_mask is not None:
                wsi.apply_qc_mask(qc_mask)

            # Reapply ROIs
            if prior_rois is not None:
                wsi.rois = prior_rois
                wsi.roi_method = roi_method
                wsi.process_rois()
            return wsi

    def reload_wsi(
        self,
        path: Optional[str] = None,
        stride: Optional[int] = None,
        use_rois: bool = True,
        tile_px: Optional[int] = None,
        tile_um: Optional[Union[str, int]] = None,
        **kwargs
    ) -> bool:
        """Reload the currently loaded Whole-Slide Image.

        Args:
            path (str, optional): Path to the slide to reload. If not provided,
                will reload the currently loaded slide.
            stride (int, optional): Stride to use for the loaded slide. If not
                provided, will use the stride value from the currently loaded
                slide.
            use_rois (bool): Use ROIs from the loaded project, if available.

        Returns:
            bool: True if slide loaded successfully, False otherwise.

        """
        wsi = self._reload_and_return_wsi(
            path, stride, use_rois, tile_px, tile_um, **kwargs
        )
        if wsi:
            self.wsi = wsi
            old_viewer = self.viewer
            self.set_viewer(SlideViewer(wsi, **self._viewer_kwargs()))
            self.set_title(os.path.basename(wsi.path))
            if isinstance(old_viewer, SlideViewer):
                self.viewer.show_thumbnail = old_viewer.show_thumbnail
                self.viewer.show_scale = old_viewer.show_scale
            return True
        else:
            return False

    def _render_prediction_message(self, message: str) -> None:
        """Render a prediction string to below the tile bounding box.

        Args:
            message (str): Message to render.
        """
        max_w = self.content_frame_width - self.offset_x_pixels
        max_h = self.content_frame_height - self.offset_y_pixels
        tex = text_utils.get_texture(
            message,
            size=self.gl_font_size,
            max_width=max_w,
            max_height=max_h,
            outline=2
        )
        box_w = self.viewer.full_extract_px / self.viewer.view_zoom
        text_pos = np.array([self.box_x + (box_w/2), self.box_y + box_w + self.font_size])
        tex.draw(pos=text_pos, align=0.5, rint=True, color=1)

    def _render_control_pane_contents(self) -> None:
        """Perform rendering of control panel contents, such as WSI thumbnails,
        widgets, and heatmaps."""

        # Render WSI thumbnail in the widget.
        if self.wsi_thumb is not None:
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
            normalizer=(self._normalizer if self._normalize_wsi else None),
            viz=self
        )

    def _update_window_limits(self):
        """Update the minimum window size limits based on loaded widgets."""

        minheight = (((len(self.sidebar.navbuttons) + 3)
                       * (self.sidebar.navbutton_width / (self.font_size / 22)))
                     + self.status_bar_height
                     + self.menu_bar_height)

        glfw.set_window_size_limits(
            self._glfw_window,
            minwidth=int(self.sidebar.content_width+100),
            minheight=int(minheight),
            maxwidth=-1,
            maxheight=-1)

    # --- Imgui methods -------------------------------------------------------

    @contextmanager
    def dim_text(self, dim=True):
        """Render dim text.

        Examples
            Render dim text.

                .. code-block:: python

                    with studio.dim_text():
                        imgui.text('This is dim')

        """
        if dim:
            imgui.push_style_color(imgui.COLOR_TEXT, *self.theme.dim)
        yield
        if dim:
            imgui.pop_style_color(1)

    @contextmanager
    def highlighted(self, enable: bool = True):
        """Render highlighted text.

        Args:
            enable (bool): Whether to enable highlighting.

        Examples
            Render highlighted text.

                .. code-block:: python

                    with studio.highlighted(True):
                        imgui.text('This is highlighted')

        """
        if enable:
            imgui.push_style_color(imgui.COLOR_BUTTON, *self.theme.button_active)
        yield
        if enable:
            imgui.pop_style_color(1)

    def collapsing_header(self, text, **kwargs):
        """Render a collapsing header using the active theme.

        Examples
            Render a collapsing header that is open by default.

                .. code-block:: python

                    if viz.collapsing_header("Header", default=True):
                        imgui.text("Text underneath")

        Args:
            text (str): Header text.

        """
        imgui.push_style_color(imgui.COLOR_HEADER, *self.theme.header)
        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *self.theme.header_hovered)
        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *self.theme.header_hovered)
        imgui.push_style_color(imgui.COLOR_TEXT, *self.theme.header_text)
        expanded = imgui_utils.collapsing_header(text.upper(), **kwargs)[0]
        imgui.pop_style_color(4)
        return expanded

    def collapsing_header2(self, text, **kwargs):
        """Render a second-level collapsing header using the active theme.

        Examples
            Render a collapsing header that is open by default.

                .. code-block:: python

                    if viz.collapsing_header("Header", default=True):
                        imgui.text("Text underneath")

        Args:
            text (str): Header text.

        """
        imgui.push_style_color(imgui.COLOR_HEADER, *self.theme.header2)
        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *self.theme.header2_hovered)
        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *self.theme.header2_hovered)
        imgui.push_style_color(imgui.COLOR_TEXT, *self.theme.header2_text)
        expanded = imgui_utils.collapsing_header(text.upper(), **kwargs)[0]
        imgui.pop_style_color(4)
        return expanded

    def header(self, text):
        """Render a header using the active theme.

        Args:
            text (str): Text for the header. Text will be rendered in
                uppercase.

        """
        with imgui_utils.header(
            text.upper(),
            hpad=self.font_size,
            vpad=(int(self.font_size*0.4), int(self.font_size*0.75))
        ):
            pass

    @contextmanager
    def header_with_buttons(self, text):
        """Render a widget header with ability to add buttons.

        Examples
            Render a header with a gear icon.

                .. code-block:: python

                    with studio.header_with_buttons('Button'):
                        # Right align the button
                        x_width = imgui.get_content_region_max()[0]
                        imgui.same_line(x_width - 30)
                        cx, cy = imgui.get_cursor_pos()
                        imgui.set_cursor_position((cx, cy-5))

                        # Render the button
                        if sidebar.small_button('gear'):
                            do_something()

        Args:
            text (str): Text for the header. Text will be rendered in
                uppercase.

        """
        with imgui_utils.header(
            text.upper(),
            hpad=self.font_size,
            vpad=(int(self.font_size*0.4), int(self.font_size*0.75))
        ):
            yield

    def center_next_window(self, width, height):
        """Center the next imgui window.

        Args:
            width (int): Width of the next window.
            height (int): Height of the next window.

        """

        imgui.set_next_window_position(
            (self.content_width - width) / 2,
            (self.content_height - height - self.status_bar_height) / 2
        )

    # --- Public methods ------------------------------------------------------

    def reset_background(self):
        """Reset the Studio background to the default theme color."""
        self._background_color = self.theme.main_background

    def add_widgets(self, widgets: Widget) -> None:
        """Add widget extension(s).

        Add widgets to Studio and the sidebar. The ``.tag`` property is used
        as a unique identifier for the widget. The ``.icon`` property should
        be a path to an image file used for rendering the sidebar navigation
        icon. ``.icon_highlighted`` property should be a path to an image file
        used for rendering a hovered navigation icon.

        The widget should implement ``__call__()`` and ``.close()`` methods
        for rendering the imgui GUI and cleanup, respectively.

        Args:
            widgets (list(:class:`slideflow.studio.widgets.Widget`)): List of
                widgets to add as extensions. These should be classes, not
                instantiated objects.

        """
        if not isinstance(widgets, list):
            widgets = [widgets]
        for widget in widgets:
            self.widgets += [widget(self)]
        self.sidebar.add_widgets(widgets)
        self._update_window_limits()

    def remove_widget(self, widget: Widget) -> None:
        """Remove a widget from Studio.

        Args:
            widget (:class:`slideflow.studio.widgets.Widget`): Widget to remove.
                This should be a class, not an instantiated object.

        """
        widget_obj = None
        for w_idx, w in enumerate(self.widgets):
            if isinstance(w, widget):
                widget_obj = w
                self.widgets.remove(w)
                break
        if widget_obj is None:
            raise ValueError(f'Could not find widget "{widget}"')
        widget_obj.close()
        self.sidebar.remove_widget(widget_obj.tag)
        self._update_window_limits()

    def add_to_render_pipeline(
        self,
        renderer: Any,
        name: Optional[str] = None
    ) -> None:
        """Add a renderer to the rendering pipeline."""
        if name is not None:
            self._addl_renderers[name] = renderer
        self._render_manager.add_to_render_pipeline(renderer)

    def remove_from_render_pipeline(self, name: str):
        """Remove a renderer from the render pipeline.

        Remove a renderer added with ``.add_to_render_pipeline()``.

        Args:
            name (str): Name of the renderer to remove.

        """
        if name not in self._addl_renderers:
            raise ValueError(f'Could not find renderer "{name}"')
        renderer = self._addl_renderers[name]
        if self._render_manager is not None:
            self._render_manager.remove_from_render_pipeline(renderer)
        del self._addl_renderers[name]

    def ask_load_heatmap(self):
        """Prompt user for location of exported heatmap (\*.npz) and load."""
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
        slide_path = askopenfilename(title="Load slide...", filetypes=[("All files", ".*"),
                                                                       ("Aperio ScanScope", ("*.svs", "*.svslide")),
                                                                       ("Hamamatsu", ("*.ndpi", "*.vms", "*.vmu")),
                                                                       ("Leica", "*.scn"),
                                                                       ("MIRAX", "*.mrxs"),
                                                                       ("Roche, Ventana", "*.bif"),
                                                                       ("Pyramid TIFF", ("*.tiff", "*.tif")),
                                                                       ("JPEG", (".jpg", "*.jpeg"))])
        if slide_path:
            self.load_slide(slide_path, ignore_errors=True)

    def autoload(self, path, ignore_errors=False):
        """Automatically load a path, detecting the type of object to load.

        Supports slides, models, projects, and other items supported by
        widgets if the widget has implemented a `.drag_and_drop_hook` function.

        Args:
            path (str): Path to file to load.
            ignore_errors (bool): Gracefully handle errors.

        """
        sf.log.info(f"Auto-loading [green]{path}[/]")
        if sf.util.is_project(path):
            self.load_project(path, ignore_errors=ignore_errors)
        elif sf.util.is_slide(path):
            self.load_slide(path, ignore_errors=ignore_errors)
        elif sf.util.is_model(path) or path.endswith('tflite'):
            self.load_model(path, ignore_errors=ignore_errors)
        elif path.endswith('npz'):
            self.load_heatmap(path)
        else:
            # See if any widgets implement a drag_and_drop_hook() method
            handled = False
            for widget in self.widgets:
                sf.log.info(f"Attempting load with widget {widget}")
                if hasattr(widget, 'drag_and_drop_hook'):
                    if widget.drag_and_drop_hook(path):
                        handled = True
                        break
            if not handled:
                self.create_toast(f"No loading handler found for {path}", icon="error")

    def clear_overlay(self) -> None:
        """Remove the current overlay image, include heatmaps and masks."""
        self.overlay = None
        self.overlay_original = None
        if self.viewer is not None:
            self.viewer.clear_overlay()

    def clear_result(self) -> None:
        """Clear all shown results and images."""
        self.clear_model_results()
        self.clear_overlay()
        self.result = EasyDict()
        self.args = EasyDict()
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

    def clear_prediction_message(self) -> None:
        self._pred_message = None

    def clear_model_results(self) -> None:
        """Clear all model results and associated images."""
        if self._render_manager is not None:
            self._render_manager.clear_result()
        self._predictions       = None
        self._norm_tex_img      = None
        self._norm_tex_obj      = None
        self._heatmap_tex_img   = None
        self._heatmap_tex_obj   = None
        if self.viewer is not None:
            self.viewer.clear_overlay()

    def close(self) -> None:
        """Close the application and renderer."""
        super().close()
        if self._render_manager is not None:
            self._render_manager.close()
            self._render_manager = None
        if hasattr(self.viewer, 'close'):
            self.viewer.close()
        for w in self.widgets:
            if hasattr(w, 'close'):
                w.close()

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
        self.menu_bar_height = self.font_size + self.spacing/2

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
        self._draw_control_pane()
        self._draw_menu_bar()
        self._draw_about_dialog()
        self._draw_mpp_zoom_dialog()

        user_input = self._handle_user_input()

        # Re-generate WSI view if the window size changed, or if we don't
        # yet have a SlideViewer initialized.
        if window_changed:
            self._content_width  = self.content_width
            self._content_height = self.content_height
            self._pane_w = self.pane_w

            for widget in self.widgets:
                if hasattr(widget, '_on_window_change'):
                    widget._on_window_change()

        # Main display.
        if self.viewer:
            self.viewer.update(**self._viewer_kwargs())
            self._draw_main_view(user_input, window_changed)
        else:
            self._draw_empty_background()

        # --- Render arguments ------------------------------------------------
        self.args.x = self.x
        self.args.y = self.y
        self.args.tile_px = self.tile_px
        self.args.tile_um = self.tile_um
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

            if (self._render_manager.is_async
                and self._render_manager._args_queue.qsize() > 2):
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
            self.viewer.apply_args(self.args)

        if self.has_live_viewer():
            self.args.viewer = None
        else:
            self.args.viewer = self.viewer
        # ---------------------------------------------------------------------

        # Render control pane contents.
        self._render_control_pane_contents()


        if self.is_skipping_frames():
            pass
        elif self._defer_rendering > 0:
            self._defer_rendering -= 1
        else:
            self._render_manager.set_args(**self.args)
            result = self._render_manager.get_result()
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

        # Render user widgets.
        for widget in self.widgets:
            if hasattr(widget, 'render'):
                widget.render()

        # Render slide widget tile boxes (for tile extraction preview)
        self.slide_widget.early_render()

        # Render the tile view and status bar.
        self._draw_tile_view()
        self._draw_status_bar()

        # Draw prediction message next to box on main display.
        if self._pred_message and self.viewer is not None:
            self._render_prediction_message(self._pred_message)
        elif (self._use_model
           and self._predictions is not None
           and not isinstance(self._predictions, list)
           and self.viewer is not None):
            if not hasattr(self.result, 'in_focus') or self.result.in_focus:
                pred_str = prediction_to_string(
                    predictions=self._predictions,
                    outcomes=self._model_config['outcome_labels'],
                    is_categorical=(self._model_config['model_type'] == 'categorical')
                )
                self._render_prediction_message(pred_str)

        # End frame.
        if self._should_close_model:
            self.close_model(True)
        if self._should_close_slide:
            self.close_slide(True)

        self.end_frame()

    @staticmethod
    def get_default_widgets() -> List[Any]:
        """Returns a list of the default non-mandatory extension widgets."""
        return []

    def get_renderer(self, name: Optional[str] = None) -> Optional[Renderer]:
        """Check for the given additional renderer in the rendering pipeline.

        Args:
            name (str): Name of the renderer to check for. If None,
                returns the main renderer.

        Returns:
            Renderer if name is a recognized renderer, otherwise None

        """
        if name is None:
            if (self._render_manager is not None
                and self._render_manager._renderer_obj is not None):
                return self._render_manager._renderer_obj
            else:
                return None
        elif name in self._addl_renderers:
            return self._addl_renderers[name]
        else:
            return None

    def get_extension(self, tag: str) -> Optional[Widget]:
        """Returns a given widget (extension) by tag.

        Args:
            tag (str): Tag of the widget to search for.

        Returns:
            slideflow.studio.widgets.Widget if found, else None

        """
        for w in self.widgets:
            if w.tag == tag:
                return w
        return None

    def get_widget(self, name: str) -> Widget:
        """Returns a given widget by class name.

        Args:
            name (str): Name of the widget to search for.

        Returns:
            slideflow.studio.widgets.Widget

        Raises:
            ValueError: If the widget could not be found.

        """
        for w in self.widgets:
            if w.__class__.__name__ == name:
                return w
        raise ValueError(f"Unable to find widget with class name {name}")

    def has_live_viewer(self) -> bool:
        """Check if the current viewer is a live viewer (e.g. camera feed)."""
        return (self.viewer is not None and self.viewer.live)

    def has_uq(self) -> bool:
        """Check if the current model supports uncertainty quantification."""
        return (self._model_path is not None
                and self._model_config is not None
                and 'uq' in self._model_config['hp']
                and self._model_config['hp']['uq'])

    def ask_zoom_to_mpp(self) -> None:
        """Prompt the user to zoom to a specific microns-per-pixel (MPP)."""
        if self.viewer and isinstance(self.viewer, SlideViewer):
            self._show_mpp_zoom_popup = True

    def increase_tile_zoom(self) -> None:
        """Increase zoom of tile view two-fold."""
        self.tile_zoom *= 2

    def decrease_tile_zoom(self) -> None:
        """Decrease zoom of tile view by half."""
        self.tile_zoom /= 2

    def reset_tile_zoom(self) -> None:
        """Reset tile zoom level."""
        self.tile_zoom = 1

    def load_heatmap(self, path: Union[str, "sf.Heatmap"]) -> None:
        """Load a saved heatmap (\*.npz).

        Args:
            path (str): Path to exported heatmap in \*.npz format, as generated
                by Heatmap.save() or Heatmap.save_npz().

        """
        if self._model_config is None:
            self.create_toast(
                "Unable to load heatmap; model must also be loaded.",
                icon="error"
            )
            return
        try:
            self.heatmap_widget.load(path)
            self.create_toast(f"Loaded heatmap at {path}", icon="success")

        except Exception as e:
            log.warn("Exception raised loading heatmap: {}".format(e))
            self.create_toast(f"Error loading heatmap at {path}", icon="error")

    def load_model(self, model: str, ignore_errors: bool = False) -> None:
        """Load the given model.

        Args:
            model (str): Path to Slideflow model (in either backend).
            ignore_errors (bool): Do not fail if an error is encountered.
                Defaults to False.

        """
        log.debug("Loading model from Studio")
        self.close_model(True)
        log.debug("Model closed")
        self.clear_result()
        log.debug("Model result cleared")
        self.skip_frame() # The input field will change on next frame.
        self._render_manager.get_result() # Flush prior result
        self._render_manager.clear_result()
        try:

            # Trigger user widgets
            for widget in self.widgets:
                if hasattr(widget, '_before_model_load'):
                    widget._before_model_load()

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
            self._render_manager.load_model(model)
            if sf.util.torch_available and sf.util.path_to_ext(model) == 'zip':
                self.model_widget.backend = 'torch'
            else:
                self.model_widget.backend = 'tensorflow'

            # Update widgets
            log.debug("Updating widgets")
            self.model_widget.reset()
            self.model_widget.cur_model = model
            self.model_widget.use_model = True
            self.model_widget.use_uncertainty = 'uq' in config['hp'] and config['hp']['uq']
            if normalizer is not None and hasattr(self, 'slide_widget'):
                self.slide_widget.add_model_normalizer_option()
                self.slide_widget.norm_idx = len(self.slide_widget._normalizer_methods)-1
            if self.wsi:
                log.debug(f"Loading slide... tile_px={self.tile_px}, tile_um={self.tile_um}")
                self.slide_widget.load(
                    self.wsi.path,
                    mpp=self.slide_widget.manual_mpp,
                    ignore_errors=ignore_errors
                )
            if hasattr(self, 'heatmap_widget'):
                log.debug("Resetting heatmap")
                self.heatmap_widget.reset()
            if not self.sidebar.expanded:
                self.sidebar.selected = 'model'
                self.sidebar.expanded = True

            # Update viewer
            self._show_tile_preview = True
            log.debug("Updating viewer with tile_px={}, tile_um={}".format(self.tile_px, self.tile_um))
            if self.viewer and not isinstance(self.viewer, SlideViewer):
                self.viewer.set_tile_px(self.tile_px)
                self.viewer.set_tile_um(self.tile_um)

            # Trigger user widgets
            for widget in self.widgets:
                if hasattr(widget, '_on_model_load'):
                    widget._on_model_load()

            self.create_toast(f"Loaded model at {model}", icon="success")

        except Exception as e:
            self.model_widget.cur_model = None
            if model == '':
                log.debug("Exception raised: no model loaded.")
                self.result = EasyDict(message='No model loaded')
            else:
                log.warn("Exception raised (ignore_errors={}): {}".format(ignore_errors, e))
                self.create_toast(f"Error loading model at {model}", icon="error")
                self.result = EasyDict(error=CapturedException())
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

    def load_slide(self, slide: str, **kwargs) -> None:
        """Load the given slide.

        Args:
            slide (str): Path to whole-slide image.
            stride (int, optional): Stride for tiles. 1 is non-overlapping
                tiles, 2 is tiles with 50% overlap, etc. Defaults to 1.
            ignore_errors (bool): Do not fail if an error is encountered.
                Defaults to False.
        """
        self.slide_widget.load(slide, **kwargs)

        # Trigger user widgets
        for widget in self.widgets:
            if hasattr(widget, '_on_slide_load'):
                widget._on_slide_load()

    def print_error(self, error: str) -> None:
        """Print the given error message."""
        error = str(error)
        if error != self._last_error_print:
            print('\n' + error + '\n')
            self._last_error_print = error

    def reload_model(self) -> None:
        """Reload the current model."""
        self._render_manager.load_model(self._model_path)

    def reload_viewer(self) -> None:
        """Reload the current main viewer."""
        if self.viewer is not None:
            self.viewer.close()
            if isinstance(self.viewer, SlideViewer):
                old_viewer = self.viewer
                self.set_viewer(SlideViewer(self.wsi, **self._viewer_kwargs()))
                self.viewer.show_thumbnail = old_viewer.show_thumbnail
                self.viewer.show_scale = old_viewer.show_scale
            else:
                self.viewer.reload(**self._viewer_kwargs())

    def set_message(self, msg: str) -> None:
        """Set a message for display."""
        self._message = msg

    def set_status_message(
        self,
        message: str,
        description: Optional[str] = None,
        *,
        color: Optional[Tuple[float, float, float]] = (0.7, 0, 0, 1),
        text_color: Tuple[float, float, float, float] = (1, 1, 1, 1),
        rounding: int = 0
    ) -> None:
        """Set the status message to display in the status bar."""
        if not message:
            self.clear_status_message()
            return
        self._status_message = StatusMessage(
            self,
            message,
            description=description,
            color=color,
            text_color=text_color,
            rounding=rounding
        )

    def clear_status_message(self) -> None:
        """Clear the status message from the status bar."""
        self._status_message = None

    def set_prediction_message(self, msg: str) -> None:
        """Set the prediction message to display under the tile outline."""
        self._pred_message = msg

    def set_overlay(
        self,
        overlay: np.ndarray,
        method: int,
        *,
        original: Optional[np.ndarray] = None
    ) -> None:
        """Configure the overlay to be applied to the current view screen.

        Overlay is a numpy array, and method is a flag indicating the
        method to use when showing the overlay.

        If ``method`` is ``sf.studio.OVERLAY_WSI``, the array will be mapped
        to the entire whole-slide image, without offsets.

        If ``method`` is ``sf.studio.OVERLAY_GRID``, the array is interpreted
        as having been generated from the slide's grid, meaning that an offset
        will be applied to ensure that the overlay is aligned properly.

        If ``method`` is ``sf.studio.OVERLAY_VIEW``, the array is interpreted
        as an overlay that is applied only to the area of the slide
        currently in view.

        Args:
            overlay (np.ndarray): Overlay to render.
            method (int): Mapping method for linking the overlay to the
                whole-slide image.

        Keyword args:
            original (np.ndarray, optional): Original grid values before any
                colorization or other modifications. Used for displaying the
                tooltip when alt-hovering. Defaults to None.

        """
        if self.viewer is None:
            raise ValueError("Unable to set overlay; viewer not loaded.")
        if original is not None and original.shape != overlay.shape:
            raise ValueError("Unable to set grid overlay; original grid shape "
                             "does not match grid shape.")
        self.overlay = overlay
        self.overlay_original = original
        if method == OVERLAY_WSI:
            # Overlay maps to the entire whole-slide image,
            # with no offset needed.
            self._overlay_wsi_dim = self.wsi.dimensions
            self._overlay_offset_wsi_dim = (0, 0)
        elif method == OVERLAY_GRID:
            # Overlay was generated from the slide's grid, meaning
            # that we need to apply an offset to ensure the overlay
            # lines up apppropriately.
            self.set_grid_overlay(overlay)
        elif method == OVERLAY_VIEW:
            # Overlay should only apply to the area of the WSI
            # currently in view.
            self._overlay_wsi_dim = self.viewer.wsi_window_size
            self._overlay_offset_wsi_dim = self.viewer.origin
        else:
            raise ValueError(f"Unrecognized method {method}")

    def set_grid_overlay(
        self,
        grid: np.ndarray,
        *,
        tile_um: Optional[int] = None,
        stride_div: Optional[int] = None,
        mpp: Optional[float] = None,
        original: Optional[np.ndarray] = None
    ) -> None:
        """Set the grid overlay to the given grid.

        Args:
            grid (np.ndarray): Grid to render as an overlay.

        Keyword args:
            tile_um (int, optional): Tile size, in microns. If None, uses
                the tile size of the currently loaded slide.
            stride_div (int, optional): Stride divisor. If None, uses
                the stride divisor of the currently loaded slide.
            mpp (float, optional): Microns per pixel. If None, uses
                the MPP of the currently loaded slide.
            original (np.ndarray, optional): Original grid values before any
                colorization or other modifications. Used for displaying the
                tooltip when alt-hovering. Defaults to None.

        """
        if self.viewer is None:
            raise ValueError("Unable to set grid overlay; viewer not loaded.")
        if any(x is None for x in (tile_um, stride_div, mpp)) and self.wsi is None:
            raise ValueError("Unable to set grid overlay; no slide loaded.")
        if original is not None and original.shape[0:2] != grid.shape[0:2]:
            raise ValueError("Unable to set grid overlay; original grid shape "
                             "({}) does not match grid shape ({}).".format(
                                original.shape, grid.shape
                             ))

        self.overlay = grid
        self.overlay_original = original
        if tile_um is None:
            tile_um = self.wsi.tile_um
        if stride_div is None:
            stride_div = self.wsi.stride_div
        if mpp is None:
            mpp = self.wsi.mpp
        full_extract = int(tile_um / mpp)
        wsi_stride = int(full_extract / stride_div)
        self._overlay_wsi_dim = (wsi_stride * (grid.shape[1]),
                                 wsi_stride * (grid.shape[0]))
        self._overlay_offset_wsi_dim = (full_extract/2 - wsi_stride/2,
                                        full_extract/2 - wsi_stride/2)

    def set_viewer(self, viewer: Any) -> None:
        """Set the main viewer.

        Args:
            viewer (:class:`slideflow.studio.gui.viewer.Viewer`): Viewer to use.

        """
        log.debug("Setting viewer to {}".format(viewer))
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = viewer
        self._render_manager._live_updates = viewer.live
        self._render_manager.set_async(viewer.live)

# -----------------------------------------------------------------------------

class Sidebar:

    """Sidebar for Studio, rendering a navigation bar and widgets."""

    def __init__(self, viz: Studio):
        self.viz                = viz
        self.expanded           = False
        self.selected           = None
        self._buttonbar_width    = 72
        self._navbutton_width    = 70
        self._imagebutton_width  = 64
        self._button_tex        = dict()
        self._pane_w_div        = 15
        self.navbuttons         = ['project', 'slide', 'model', 'heatmap']

        self.add_widgets(viz.widgets)
        self._load_button_textures()

    @property
    def buttonbar_width(self):
        return int(self._buttonbar_width * (self.viz.font_size / 22))

    @property
    def navbutton_width(self):
        return int(self._navbutton_width * (self.viz.font_size / 22))

    @property
    def imagebutton_width(self):
        w = int(self._imagebutton_width * (self.viz.font_size / 22))
        return w

    @property
    def theme(self):
        """Active Studio theme."""
        return self.viz.theme

    @property
    def content_width(self):
        """Widget width."""
        return self.viz.font_size * self._pane_w_div

    @property
    def full_width(self):
        """Width of the expanded sidebar, including navigation and widgets."""
        return self.content_width + self.buttonbar_width

    # --- Internals -----------------------------------------------------------

    def _set_sidebar_style(self) -> None:
        """Start the Imgui sidebar style based on the active theme."""
        t = self.viz.theme
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *t.item_background)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *t.item_hover)
        imgui.push_style_color(imgui.COLOR_BORDER, *t.border)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *t.sidebar_background)
        imgui.push_style_color(imgui.COLOR_BUTTON, *t.button)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *t.button_hovered)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *t.button_active)
        imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *t.accent)
        imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM_HOVERED, *t.accent_hovered)

    def _end_sidebar_style(self) -> None:
        """End the Imgui sidebar style based on the active theme."""
        imgui.pop_style_color(9)

    def _set_sidebar_button_style(self) -> None:
        """Start the Imgui sidebar button style based on the active theme."""
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *self.theme.sidebar_background)
        imgui.push_style_color(imgui.COLOR_BUTTON, 0, 0, 0, 0)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0, 0, 0, 0)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0, 0, 0, 0)
        imgui.push_style_color(imgui.COLOR_BORDER, 0, 0, 0, 0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, [0, 0])
        imgui.push_style_var(imgui.STYLE_ITEM_SPACING, [0, 0])

    def _end_sidebar_button_style(self) -> None:
        """End the Imgui sidebar button style based on the active theme."""
        imgui.pop_style_color(5)
        imgui.pop_style_var(2)

    def _load_button_textures(self) -> None:
        """Reload textures for buttons."""
        button_dir = join(dirname(abspath(__file__)), 'gui', 'buttons')
        for bname in self.navbuttons + ['gear', 'circle_lightning', 'circle_plus', 'pencil', 'folder', 'floppy', 'model_loaded', 'extensions', 'add_freehand', 'add_polygon', 'add_point']:
            if bname in self._button_tex:
                continue
            self._button_tex[bname] = gl_utils.Texture(image=Image.open(join(button_dir, f'button_{bname}.png')), bilinear=True, mipmap=True)
            try:
                self._button_tex[f'{bname}_highlighted'] = gl_utils.Texture(image=Image.open(join(button_dir, f'button_{bname}_highlighted.png')), bilinear=True, mipmap=True)
            except FileNotFoundError:
                pass
        for name in ('vips', 'cucim', 'lowmem', 'ellipsis', 'gear', 'refresh'):
            if f"small_{name}" in self._button_tex:
                continue
            self._button_tex[f"small_{name}"] = gl_utils.Texture(image=Image.open(join(button_dir, f'small_button_{name}.png')), bilinear=True, mipmap=True, maxlevel=3)

    def _draw_navbar_button(self, name, start_px):
        """Draw a navigation bar button.

        Args:
            name (str): Name of the image associated with the button.
            start_px (int): Starting location of the button (y coordinates).

        """
        viz = self.viz

        if name == 'model' and viz._model_config is not None:
            tex_name = 'model_loaded'
        else:
            tex_name = name

        cx, cy = imgui.get_mouse_pos()
        cy -= viz.menu_bar_height
        end_px = start_px + self.navbutton_width
        if ((cx < 0 or cx > self.navbutton_width) or (cy < start_px or cy > end_px)) and self.selected != name:
            tex = self._button_tex[tex_name].gl_id
        else:
            tex = self._button_tex[f'{tex_name}_highlighted'].gl_id
        imgui.set_cursor_position((0, start_px))
        if imgui.image_button(tex, self.imagebutton_width, self.imagebutton_width):
            if name == self.selected or self.selected is None or not self.expanded:
                self.expanded = not self.expanded
            self.selected = name
        if self.selected == name:
            draw_list = imgui.get_window_draw_list()
            draw_list.add_line(2, viz.menu_bar_height+start_px, 2, viz.menu_bar_height+start_px+self.navbutton_width, imgui.get_color_u32_rgba(1,1,1,1), 2)

    def _draw_buttons(self) -> None:
        """Draw all navigation bar buttons."""
        viz = self.viz

        imgui.set_next_window_position(0, viz.menu_bar_height)
        buttonbar_height = viz.content_height - viz.menu_bar_height - viz.status_bar_height
        imgui.set_next_window_size(self.buttonbar_width, buttonbar_height)
        imgui.begin(
            'Sidebar',
            closable=False,
            flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS)
        )
        for b_id, b_name in enumerate(self.navbuttons):
            start_px = b_id * self.navbutton_width
            self._draw_navbar_button(b_name, start_px)

        self._draw_navbar_button('circle_lightning', buttonbar_height + viz.menu_bar_height - self.navbutton_width*3 - viz.status_bar_height)
        self._draw_navbar_button('extensions', buttonbar_height + viz.menu_bar_height - self.navbutton_width*2 - viz.status_bar_height)
        self._draw_navbar_button('gear', buttonbar_height + viz.menu_bar_height - self.navbutton_width - viz.status_bar_height)

        imgui.end()

    # --- Public methods ------------------------------------------------------

    def add_widgets(self, widgets):
        """Add widget extension(s).

        Add widgets to the navigation sidebar. The ``.tag`` property is used
        as a unique identifier for the widget. The ``.icon`` property should
        be a path to an image file used for rendering the sidebar navigation
        icon. ``.icon_highlighted`` property should be a path to an image file
        used for rendering a hovered navigation icon.

        The widget should implement ``__call__()`` and ``.close()`` methods
        for rendering the imgui GUI and cleanup, respectively.

        Args:
            widgets (list(:class:`slideflow.studio.widgets.Widget`)): List of
                widgets to add as extensions. These should be classes, not
                instantiated objects.

        """
        if not isinstance(widgets, list):
            widgets = [widgets]
        if not widgets:
            return
        for widget in widgets:
            self.navbuttons.append(widget.tag)
            self._button_tex[widget.tag] = gl_utils.Texture(image=Image.open(widget.icon), bilinear=True, mipmap=True)
            self._button_tex[f'{widget.tag}_highlighted'] = gl_utils.Texture(image=Image.open(widget.icon_highlighted), bilinear=True, mipmap=True)

    def remove_widget(self, tag):
        """Remove a widget from Studio.

        Args:
            widget (:class:`slideflow.studio.widgets.Widget`): Widget to remove.
                This should be a class, not an instantiated object.

        """
        if tag not in self.navbuttons:
            raise ValueError(f'No matching widget with tag "{tag}".')
        idx = self.navbuttons.index(tag)
        del self.navbuttons[idx]

    def full_button(self, text, width=None, **kwargs):
        """Render a button that spans the full width of the sidebar.

        The color of the button is determined through the loaded theme,
        ``bright_button`` properties.

        Args:
            text (str): Text of the button.
            width (int, optional): Width of the button. If not specified,
                uses a width that spans the width of the sidebar.

        Keyword args:
            enabled (bool): Whether the button is enabled.

        Returns:
            bool: Whether the button was clicked.

        """
        t = self.theme
        imgui.push_style_color(imgui.COLOR_BUTTON, *t.bright_button)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *t.bright_button_hovered)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *t.bright_button_active)
        imgui.push_style_color(imgui.COLOR_BORDER, 0, 0, 0, 0)
        imgui.push_style_color(imgui.COLOR_TEXT, 0, 0, 0, 1)
        if width is None:
            width = self.viz.sidebar.content_width - (self.viz.spacing * 2)
        result = imgui_utils.button(
            text,
            width=width,
            height=self.viz.font_size * 1.7,
            **kwargs
        )
        imgui.pop_style_color(5)
        return result

    def full_button2(self, text, width=None, **kwargs):
        """Render a button that spans the full width of the sidebar.

        The color of the button is determined through the loaded theme,
        ``bright_button2`` properties.

        Args:
            text (str): Text of the button.
            width (int, optional): Width of the button. If not specified,
                uses a width that spans the width of the sidebar.

        Keyword args:
            enabled (bool): Whether the button is enabled.

        Returns:
            bool: Whether the button was clicked.

        """
        t = self.theme
        imgui.push_style_color(imgui.COLOR_BUTTON, *t.bright_button2)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *t.bright_button2_hovered)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *t.bright_button2_active)
        imgui.push_style_color(imgui.COLOR_TEXT, *t.bright_button2_text)
        imgui.push_style_color(imgui.COLOR_BORDER, 0, 0, 0, 0)
        if width is None:
            width = self.viz.sidebar.content_width - (self.viz.spacing * 2)
        result = imgui_utils.button(
            text,
            width=width,
            height=self.viz.font_size * 1.7,
            **kwargs
        )
        imgui.pop_style_color(5)
        return result

    def small_button(self, image_name):
        """Render a small button for the sidebar.

        Args:
            image_name (str): Name of the image to render on the button.
                Valid names include 'vips', 'cucim', 'lowmem', 'ellipsis',
                'gear', and 'refresh'.

        Returns:
            bool: If the button was clicked.

        """
        viz = self.viz
        tex = self._button_tex[f'small_{image_name}'].gl_id
        imgui.push_style_color(imgui.COLOR_BUTTON, 0, 0, 0, 0)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *self.theme.button_hovered)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *self.theme.button_active)
        result = imgui.image_button(tex, viz.font_size, viz.font_size)
        imgui.pop_style_color(3)
        return result

    def large_image_button(self, image_name, size=None):
        """Render a small button for the sidebar.

        Args:
            image_name (str): Name of the image to render on the button.
                Valid names include 'gear', 'circle_lightning', 'circle_plus',
                'pencil', 'folder', 'floppy', 'model_loaded', 'extensions',
                'project', 'slide', 'model', and 'heatmap'.
            size (int): Simage button. Defaults to 64.

        Returns:
            bool: If the button was clicked.

        """
        if size is None:
            size = self.imagebutton_width
        tex = self._button_tex[f'{image_name}'].gl_id
        return imgui.image_button(tex, size, size)

    def draw(self):
        """Draw the sidebar and render all widgets."""
        viz = self.viz

        self._set_sidebar_button_style()
        self._draw_buttons()
        self._end_sidebar_button_style()
        self._set_sidebar_style()

        if self.expanded:
            drawing_control_pane = True

            viz.pane_w = self.full_width
            imgui.set_next_window_position(self.navbutton_width, viz.menu_bar_height)
            imgui.set_next_window_size(self.content_width, viz.content_height - viz.menu_bar_height - viz.status_bar_height)
            imgui.begin(
                'Control Pane',
                closable=False,
                flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS)
            )
        else:
            viz.pane_w = self.navbutton_width
            drawing_control_pane = False

        # --- Core widgets (always rendered, not always shown) ----------------

        # Slide widget
        viz.slide_widget(self.expanded and self.selected == 'slide')

        # Project widget
        viz.project_widget(self.expanded and self.selected == 'project')

        # Model widget
        viz.model_widget(self.expanded and self.selected == 'model')

        # Heatmap / prediction widget
        viz.heatmap_widget(self.expanded and self.selected == 'heatmap')

        # Performance / capture widget
        viz.performance_widget(self.expanded and self.selected == 'circle_lightning')
        viz.capture_widget(self.expanded and self.selected == 'circle_lightning')

        # Extensions widget
        viz.extensions_widget(self.expanded and self.selected == 'extensions')

        # Settings widget
        viz.settings_widget(self.expanded and self.selected == 'gear')

        # ---------------------------------------------------------------------

        # User-defined widgets
        for widget in viz.widgets:
            widget(self.expanded and self.selected == widget.tag)

        # Render control panel contents, if the control pane is shown.
        if drawing_control_pane:
            imgui.end()

        self._end_sidebar_style()

# -----------------------------------------------------------------------------