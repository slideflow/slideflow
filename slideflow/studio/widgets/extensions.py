import slideflow as sf
import traceback
import numpy as np
import imgui
import textwrap
from PIL import Image
from os.path import join, dirname, abspath
from typing import Union, Tuple

from ..gui import imgui_utils, gl_utils

#----------------------------------------------------------------------------

class ExtensionsWidget:

    tag = 'extensions'
    description = 'Extensions'
    icon = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_extensions.png')
    icon_highlighted = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_extensions_highlighted.png')

    def __init__(
        self,
        viz: "sf.studio.Studio",
        autoload: Union[bool, Tuple[str], str] = ('mil', 'segment')
    ) -> None:

        self.viz                = viz
        self._show_err_popup    = False

        self.stylegan = any([w.tag == 'stylegan' for w in viz.widgets])
        self.mosaic = any([w.tag == 'mosaic' for w in viz.widgets])
        self.cellseg = any([w.tag == 'cellseg' for w in viz.widgets])
        self.mil = any([w.tag == 'mil' for w in viz.widgets])
        self.segment = any([w.tag == 'segment' for w in viz.widgets])
        self.alignment = any([w.tag == 'alignment' for w in viz.widgets])

        _off_path = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'small_button_verified.png')
        self._official_tex      = gl_utils.Texture(
            image=np.array(Image.open(_off_path)), bilinear=True, mipmap=True
        )

        # Try to load all extensions at start.
        if autoload:
            if isinstance(autoload, bool) and autoload:
                autoload = ['stylegan', 'mosaic', 'cellseg', 'mil', 'segment']
            elif isinstance(autoload, str):
                autoload = [autoload]
            for ext in autoload:
                try:
                    if not getattr(self, f'{ext}'):
                        getattr(self, f'toggle_{ext}')()
                        setattr(self, f'{ext}', True)
                except Exception as e:
                    pass

    def toggle_stylegan(self):
        viz = self.viz
        from ..widgets.stylegan import StyleGANWidget
        if not any(isinstance(w, StyleGANWidget) for w in viz.widgets):
            viz.add_widgets(StyleGANWidget)
        else:
            viz.remove_widget(StyleGANWidget)

    def toggle_mosaic(self):
        viz = self.viz
        from ..widgets.mosaic import MosaicWidget
        if not any(isinstance(w, MosaicWidget) for w in viz.widgets):
            viz.add_widgets(MosaicWidget)
        else:
            viz.remove_widget(MosaicWidget)

    def toggle_cellseg(self):
        viz = self.viz
        from ..widgets.cellseg import CellSegWidget
        if not any(isinstance(w, CellSegWidget) for w in viz.widgets):
            viz.add_widgets(CellSegWidget)
        else:
            viz.remove_widget(CellSegWidget)

    def toggle_mil(self):
        viz = self.viz
        from ..widgets.mil import MILWidget
        if not any(isinstance(w, MILWidget) for w in viz.widgets):
            viz.add_widgets(MILWidget)
        else:
            viz.remove_widget(MILWidget)

    def toggle_segment(self):
        viz = self.viz
        from ..widgets.segment import TissueSegWidget
        if not any(isinstance(w, TissueSegWidget) for w in viz.widgets):
            viz.add_widgets(TissueSegWidget)
        else:
            viz.remove_widget(TissueSegWidget)

    def toggle_alignment(self):
        viz = self.viz
        from ..widgets.alignment import AlignmentWidget
        if not any(isinstance(w, AlignmentWidget) for w in viz.widgets):
            viz.add_widgets(AlignmentWidget)
        else:
            viz.remove_widget(AlignmentWidget)

    def extension_checkbox(self, title, description, check_value, official=False):
        viz = self.viz
        height = imgui.get_text_line_height_with_spacing() * 3
        imgui.begin_child(f'##{title}', height=height)
        with viz.bold_font():
            imgui.text(title)
        imgui.text_colored(description, *viz.theme.dim)
        if official:
            imgui.image(self._official_tex.gl_id, viz.font_size, viz.font_size)
            imgui.same_line(viz.font_size + viz.spacing/2)
            imgui.text("Official")
        else:
            imgui.text('')
        imgui.same_line(imgui.get_content_region_max()[0] - viz.font_size - viz.spacing * 1.5)
        result = imgui.checkbox(f'##{title}_checkbox', check_value)
        imgui.end_child()
        return result

    def show_extension_error(self, message, full_trace=None):
        self._show_err_popup = True
        self._err_msg = message
        if full_trace:
            print(full_trace)
        else:
            print(message)

    def draw_error_popup(self):
        """Show an error message that an extension failed to load."""
        wrapped = textwrap.wrap(self._err_msg, width=45)
        lh = imgui.get_text_line_height_with_spacing()
        window_size = (self.viz.font_size * 20, lh * len(wrapped) + self.viz.font_size * 4)
        self.viz.center_next_window(*window_size)
        imgui.set_next_window_size(*window_size)
        _, opened = imgui.begin('Error loading extension', closable=True, flags=imgui.WINDOW_NO_RESIZE)
        if not opened:
            self._show_err_popup = False

        for line in wrapped:
            imgui.text(line)

        if self.viz.sidebar.full_button("OK", width=-1):
            self._show_err_popup = False
        imgui.end()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            viz.header("Extensions")

            _c2, self.mosaic = self.extension_checkbox(
                'Mosaic Maps',
                description='Open and interact with Mosaic Maps.',
                check_value=self.mosaic,
                official=True
            )
            if _c2:
                self.toggle_mosaic()
            imgui.separator()

            _c1, self.stylegan = self.extension_checkbox(
                'StyleGAN',
                description='Generate images with StyleGAN.',
                check_value=self.stylegan,
                official=True
            )
            if _c1:
                try:
                    self.toggle_stylegan()
                except Exception as e:
                    self.show_extension_error(str(e), traceback.format_exc())
                    self.stylegan = False
            imgui.separator()

            _c3, self.cellseg = self.extension_checkbox(
                'Cell Segmentation',
                description='Segment cells with Cellpose.',
                check_value=self.cellseg,
                official=True
            )
            if _c3:
                try:
                    self.toggle_cellseg()
                except ImportError as e:
                    self.show_extension_error(
                        'Cellpose is not installed. Cellpose can be installed '
                        'with "pip install cellpose"'
                    )
                    self.cellseg = False
                except Exception as e:
                    self.show_extension_error(str(e), traceback.format_exc())
                    self.cellseg = False
            imgui.separator()

            _c5, self.segment = self.extension_checkbox(
                'Tissue Segmentation',
                description='Train and use segmentation models.',
                check_value=self.segment,
                official=True
            )
            if _c5:
                try:
                    self.toggle_segment()
                except Exception as e:
                    self.show_extension_error(str(e), traceback.format_exc())
                    self.segment = False
            imgui.separator()

            _c4, self.mil = self.extension_checkbox(
                'Multiple-Instance Learning',
                description='MIL support with attention heatmaps.',
                check_value=self.mil,
                official=True
            )
            if _c4:
                try:
                    self.toggle_mil()
                except Exception as e:
                    self.show_extension_error(str(e), traceback.format_exc())
                    self.mil = False

            _c6, self.alignment = self.extension_checkbox(
                'Alignment',
                description='Align slides.',
                check_value=self.alignment,
                official=True
            )
            if _c6:
                try:
                    self.toggle_alignment()
                except Exception as e:
                    self.show_extension_error(str(e), traceback.format_exc())
                    self.alignment = False


        if self._show_err_popup:
            self.draw_error_popup()