from dataclasses import dataclass
from typing import Tuple


RGBA = Tuple[float, float, float, float]


@dataclass
class StudioTheme:
    border: RGBA
    main_background: RGBA
    item_background: RGBA
    item_hover: RGBA
    bright_button: RGBA
    bright_button_hovered: RGBA
    bright_button_active: RGBA
    button: RGBA
    button_hovered: RGBA
    button_active: RGBA
    accent: RGBA
    accent_hovered: RGBA
    sidebar_background: RGBA
    popup_background: RGBA
    popup_border: RGBA
    header: RGBA
    header_hovered: RGBA
    header_text: RGBA
    dim: RGBA


def monokai_fire() -> StudioTheme:
    return StudioTheme(
        border=(0, 0, 0, 0),
        main_background=(0.11, 0.11, 0.11, 1),
        item_background=(0.15, 0.15, 0.15, 1),
        item_hover=(0.25, 0.25, 0.25, 1),
        button=(0.15, 0.15, 0.15, 1),
        button_hovered=(0.25, 0.25, 0.25, 1),
        button_active=(0.3, 0.3, 0.3, 1),
        bright_button=(1, 0.99, 0.47, 1),
        bright_button_hovered=(1, 0.99, 0.55, 1),
        bright_button_active=(0.7, 0.7, 0.33, 1),
        accent=(0.47, 0.65, 1, 1),
        accent_hovered=(0.56, 0.78, 1, 1),
        sidebar_background=(0.09, 0.09, 0.09, 1),
        popup_background=(0.11, 0.11, 0.11, 0.95),
        popup_border=(0.3, 0.3, 0.3, 1),
        header=(0.15, 0.15, 0.15, 1),
        header_hovered=(0.25, 0.25, 0.25, 1),
        header_text=(0.7, 0.7, 0.7, 1),
        dim=(1, 1, 1, 0.5)
    )