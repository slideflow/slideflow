from dataclasses import dataclass
from typing import Tuple

RGBA = Tuple[float, float, float, float]
__themes__ = {}

# -----------------------------------------------------------------------------

@dataclass
class StudioTheme:
    """Default theme.

    Dark theme with yellow 'bright' buttons
    and blue accents.
    """
    border: RGBA = (0, 0, 0, 0)
    main_background: RGBA = (0.11, 0.11, 0.11, 1)
    item_background: RGBA = (0.15, 0.15, 0.15, 1)
    item_hover: RGBA = (0.25, 0.25, 0.25, 1)
    button: RGBA = (0.15, 0.15, 0.15, 1)
    button_hovered: RGBA = (0.25, 0.25, 0.25, 1)
    button_active: RGBA = (0.3, 0.3, 0.3, 1)
    bright_button: RGBA = (1, 0.99, 0.47, 1)
    bright_button_hovered: RGBA = (1, 0.99, 0.55, 1)
    bright_button_active: RGBA = (0.7, 0.7, 0.33, 1)
    accent: RGBA = (0.47, 0.65, 1, 1)
    accent_hovered: RGBA = (0.56, 0.78, 1, 1)
    sidebar_background: RGBA = (0.09, 0.09, 0.09, 1)
    popup_background: RGBA = (0.11, 0.11, 0.11, 0.95)
    popup_border: RGBA = (0.3, 0.3, 0.3, 1)
    header: RGBA = (0.15, 0.15, 0.15, 1)
    header_hovered: RGBA = (0.25, 0.25, 0.25, 1)
    header_text: RGBA = (0.7, 0.7, 0.7, 1)
    header2: RGBA = (0.09, 0.09, 0.09, 1)
    header2_hovered: RGBA = (0.15, 0.15, 0.15, 1)
    header2_text: RGBA = (1, 1, 1, 1)
    dim: RGBA = (1, 1, 1, 0.5)

# -----------------------------------------------------------------------------

def register_theme(name):
    def decorator(function):
        __themes__[name] = function
        return function
    return decorator


def get_theme(name):
    if name not in __themes__:
        raise ValueError(f"Theme '{name}' not found.")
    return __themes__[name]()


def list_themes():
    return list(__themes__.keys())

# -----------------------------------------------------------------------------

@register_theme('Studio Dark')
def studio_dark() -> StudioTheme:
    return StudioTheme()


@register_theme('Studio Pink')
def pink() -> StudioTheme:
    return StudioTheme(
        bright_button=(0.99, 0.47, 1, 1),
        bright_button_hovered=(0.99, 0.55, 1, 1),
        bright_button_active=(0.7, 0.33, 0.7, 1),
    )


@register_theme('Studio Cyan')
def cyan() -> StudioTheme:
    return StudioTheme(
        bright_button=(0.47, 0.99, 1, 1),
        bright_button_hovered=(0.55, 0.99, 1, 1),
        bright_button_active=(0.33, 0.7, 0.7, 1),
    )
