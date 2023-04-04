import imgui
from os.path import join, dirname, abspath

from ..gui import imgui_utils

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


    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            viz.sidebar.header("Extensions")

            imgui_utils.padded_text('No extensions found. ', vpad=[int(viz.font_size/2), int(viz.font_size)])

            _c2, self.mosaic = imgui.checkbox('Mosaic Maps', self.mosaic)
            if _c2:
                self.update_mosaic()
            _c1, self.stylegan = imgui.checkbox('StyleGAN', self.stylegan)
            if _c1:
                self.update_stylegan()
            _c3, self.segment = imgui.checkbox('Cell Segmentation', self.segment)
            if _c3:
                self.update_segment()