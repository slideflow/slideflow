import numpy as np
import imgui
from PIL import Image
from os.path import join, dirname, abspath

from ..gui import imgui_utils, gl_utils

#----------------------------------------------------------------------------

class ExtensionsWidget:

    tag = 'extensions'
    description = 'Extensions'
    icon = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_extensions.png')
    icon_highlighted = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_extensions_highlighted.png')

    def __init__(self, viz):
        self.viz                = viz

        self.stylegan = any([w.tag == 'stylegan' for w in viz.widgets])
        self.mosaic = any([w.tag == 'mosaic' for w in viz.widgets])
        self.segment = any([w.tag == 'segment' for w in viz.widgets])

        _off_path = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'small_button_verified.png')
        self._official_tex      = gl_utils.Texture(
            image=np.array(Image.open(_off_path)), bilinear=True, mipmap=True
        )

    def update_extensions(self):
        viz = self.viz

    def update_stylegan(self):
        viz = self.viz
        from ..widgets.stylegan import StyleGANWidget
        if not any(isinstance(w, StyleGANWidget) for w in viz.widgets):
            viz.add_widgets(StyleGANWidget)
        else:
            viz.remove_widget(StyleGANWidget)

    def update_mosaic(self):
        viz = self.viz
        from ..widgets.mosaic import MosaicWidget
        if not any(isinstance(w, MosaicWidget) for w in viz.widgets):
            viz.add_widgets(MosaicWidget)
        else:
            viz.remove_widget(MosaicWidget)

    def update_segment(self):
        viz = self.viz
        from ..widgets.segment import SegmentWidget
        if not any(isinstance(w, SegmentWidget) for w in viz.widgets):
            viz.add_widgets(SegmentWidget)
        else:
            viz.remove_widget(SegmentWidget)

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
                self.update_mosaic()
            imgui.separator()

            _c1, self.stylegan = self.extension_checkbox(
                'StyleGAN',
                description='Generate images with StyleGAN.',
                check_value=self.stylegan,
                official=True
            )
            if _c1:
                self.update_stylegan()
            imgui.separator()

            _c3, self.segment = self.extension_checkbox(
                'Cell Segmentation',
                description='Segment cells with Cellpose.',
                check_value=self.segment,
                official=True
            )
            if _c3:
                self.update_segment()