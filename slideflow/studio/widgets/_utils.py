"""Base Widget class to be extended."""

from os.path import join, dirname, abspath

from ..gui import imgui_utils

#----------------------------------------------------------------------------

class Widget:

    tag = 'Widget'
    description = 'Basic widget'
    icon = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_extensions.png')
    icon_highlighted = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_extensions_highlighted.png')

    def __init__(self, viz):
        self.viz = viz

    def close(self):
        pass

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

#----------------------------------------------------------------------------
