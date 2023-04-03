"""Utility functions for Imgui applications."""

import time
import os
from PIL import Image
from os.path import join, dirname, abspath
import contextlib
import imgui

_SPINNER_ARRAY = ['.  ', '.. ', '...', ' ..', '  .', '   ']

#----------------------------------------------------------------------------

def logo_image():
    return Image.open(join(dirname(abspath(__file__)), 'icons', 'logo.png'))


def icons():
    icon_path = join(dirname(abspath(__file__)), 'icons')
    return {
        name.split('.')[0]: Image.open(join(icon_path, name))
        for name in os.listdir(icon_path)
    }

#----------------------------------------------------------------------------

def set_default_style(spacing=9, indent=23, scrollbar=27):
    s = imgui.get_style()
    s.window_padding        = [spacing, spacing]
    s.item_spacing          = [spacing, spacing]
    s.item_inner_spacing    = [spacing, spacing]
    s.columns_min_spacing   = spacing
    s.indent_spacing        = indent
    s.scrollbar_size        = scrollbar
    s.frame_padding         = [4, 3]
    s.window_border_size    = 1
    s.child_border_size     = 1
    s.popup_border_size     = 1
    s.frame_border_size     = 1
    s.window_rounding       = 0
    s.child_rounding        = 0
    s.popup_rounding        = 3
    s.frame_rounding        = 3
    s.scrollbar_rounding    = 3
    s.grab_rounding         = 3

#----------------------------------------------------------------------------

@contextlib.contextmanager
def header(text, color=0.4, hpad=20, vpad=15):
    if isinstance(vpad, (float, int)):
        vpad = [vpad, vpad]
    if isinstance(hpad, (float, int)):
        hpad = [hpad, hpad]
    line_height =  imgui.core.get_text_line_height()
    if isinstance(color, (float, int)):
        color = [color, color, color, 1]
    imgui.push_style_color(imgui.COLOR_TEXT, *color)
    cx, cy = imgui.get_cursor_position()
    imgui.set_cursor_position([cx+hpad[0], cy+vpad[0]])
    imgui.text(text)
    imgui.pop_style_color(1)
    yield
    imgui.set_cursor_position([cx+hpad[1], cy + line_height + vpad[0] + vpad[1]])
    imgui.separator()

#----------------------------------------------------------------------------

def vertical_break():
    cx, cy = imgui.get_cursor_position()
    imgui.set_cursor_position([cx, cy+10])

#----------------------------------------------------------------------------

def padded_text(text, hpad=0, vpad=0):
    if isinstance(vpad, (float, int)):
        vpad = [vpad, vpad]
    if isinstance(hpad, (float, int)):
        hpad = [hpad, hpad]
    line_height =  imgui.core.get_text_line_height()
    cx, cy = imgui.get_cursor_position()
    imgui.set_cursor_position([cx+hpad[0], cy+vpad[0]])
    imgui.text(text)
    imgui.set_cursor_position([cx+hpad[1], cy + line_height + vpad[0] + vpad[1]])

#----------------------------------------------------------------------------

def spinner():
    imgui.text(spinner_text())

def spinner_text():
    return _SPINNER_ARRAY[int(time.time()/0.05) % len(_SPINNER_ARRAY)]

#----------------------------------------------------------------------------

def ellipsis_clip(text, length):
    if len(text) > length:
        return text[:length-3] + '...'
    else:
        return text

@contextlib.contextmanager
def clipped_with_tooltip(text, length):
    clipped = ellipsis_clip(text, length)
    yield
    if clipped != text and imgui.is_item_hovered():
        imgui.set_tooltip(text)

#----------------------------------------------------------------------------

@contextlib.contextmanager
def grayed_out(cond=True):
    if cond:
        s = imgui.get_style()
        text = s.colors[imgui.COLOR_TEXT_DISABLED]
        grab = s.colors[imgui.COLOR_SCROLLBAR_GRAB]
        back = s.colors[imgui.COLOR_MENUBAR_BACKGROUND]
        imgui.push_style_color(imgui.COLOR_TEXT, *text)
        imgui.push_style_color(imgui.COLOR_CHECK_MARK, *grab)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *grab)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *grab)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *back)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *back)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *back)
        imgui.push_style_color(imgui.COLOR_BUTTON, *back)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *back)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *back)
        imgui.push_style_color(imgui.COLOR_HEADER, *back)
        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *back)
        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *back)
        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *back)
        yield
        imgui.pop_style_color(14)
    else:
        yield

#----------------------------------------------------------------------------

@contextlib.contextmanager
def item_width(width=None):
    if width is not None:
        imgui.push_item_width(width)
        yield
        imgui.pop_item_width()
    else:
        yield

def right_align(text, spacing=0):
    imgui.same_line(imgui.get_content_region_max()[0] - (imgui.calc_text_size(text)[0] + spacing))

def right_aligned_text(text, spacing=0):
    imgui.same_line(imgui.get_content_region_max()[0] - (imgui.calc_text_size(text)[0] + spacing))
    imgui.text(text)

#----------------------------------------------------------------------------

def scoped_by_object_id(method):
    def decorator(self, *args, **kwargs):
        imgui.push_id(str(id(self)))
        res = method(self, *args, **kwargs)
        imgui.pop_id()
        return res
    return decorator

#----------------------------------------------------------------------------

def button(label, width=0, height=0, enabled=True):
    with grayed_out(not enabled):
        clicked = imgui.button(label, width=width, height=height)
    clicked = clicked and enabled
    return clicked

#----------------------------------------------------------------------------

def collapsing_header(text, visible=None, flags=0, default=False, enabled=True, show=True):
    expanded = False
    if show:
        if default:
            flags |= imgui.TREE_NODE_DEFAULT_OPEN
        if not enabled:
            flags |= imgui.TREE_NODE_LEAF
        with grayed_out(not enabled):
            expanded, visible = imgui.collapsing_header(text, visible=visible, flags=flags)
        expanded = expanded and enabled
    return expanded, visible

#----------------------------------------------------------------------------

def popup_button(label, width=0, enabled=True):
    if button(label, width, enabled):
        imgui.open_popup(label)
    opened = imgui.begin_popup(label)
    return opened

#----------------------------------------------------------------------------

def input_text(label, value, buffer_length, flags, width=None, help_text=''):
    old_value = value
    color = list(imgui.get_style().colors[imgui.COLOR_TEXT])
    if value == '':
        color[-1] *= 0.5
    with item_width(width):
        imgui.push_style_color(imgui.COLOR_TEXT, *color)
        value = value if value != '' else help_text
        changed, value = imgui.input_text(label, value, buffer_length, flags)
        value = value if value != help_text else ''
        imgui.pop_style_color(1)
    if not flags & imgui.INPUT_TEXT_ENTER_RETURNS_TRUE:
        changed = (value != old_value)
    return changed, value

#----------------------------------------------------------------------------

def drag_previous_control(enabled=True):
    dragging = False
    dx = 0
    dy = 0
    if imgui.begin_drag_drop_source(imgui.DRAG_DROP_SOURCE_NO_PREVIEW_TOOLTIP):
        if enabled:
            dragging = True
            dx, dy = imgui.get_mouse_drag_delta()
            imgui.reset_mouse_drag_delta()
        imgui.end_drag_drop_source()
    return dragging, dx, dy

#----------------------------------------------------------------------------

def drag_button(label, width=0, enabled=True):
    clicked = button(label, width=width, enabled=enabled)
    dragging, dx, dy = drag_previous_control(enabled=enabled)
    return clicked, dragging, dx, dy

#----------------------------------------------------------------------------

def drag_hidden_window(label, x, y, width, height, enabled=True):
    imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0, 0, 0, 0)
    imgui.push_style_color(imgui.COLOR_BORDER, 0, 0, 0, 0)
    imgui.set_next_window_position(x, y)
    imgui.set_next_window_size(width, height)
    imgui.begin(label, closable=False, flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_FOCUS_ON_APPEARING | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
    dragging, dx, dy = drag_previous_control(enabled=enabled)
    imgui.end()
    imgui.pop_style_color(2)
    return dragging, dx, dy

#----------------------------------------------------------------------------

def click_previous_control(mouse_idx=0, enabled=True):
    clicking = False
    if imgui.is_mouse_down(mouse_idx) and enabled:
        clicking = True
    cx, cy = imgui.get_mouse_pos() # or position
    return clicking, cx, cy

#----------------------------------------------------------------------------

def click_hidden_window(label, x, y, width, height, enabled=True, mouse_idx=0):
    imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0, 0, 0, 0)
    imgui.push_style_color(imgui.COLOR_BORDER, 0, 0, 0, 0)
    imgui.set_next_window_position(x, y)
    imgui.set_next_window_size(width, height)
    imgui.begin(label, closable=False, flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_FOCUS_ON_APPEARING | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS))
    clicking, cx, cy = click_previous_control(mouse_idx=mouse_idx, enabled=enabled)
    if cx-x < 0 or cy-y < 0:
        clicking = False
        wheel = False
    else:
        wheel = imgui.get_io().mouse_wheel
    imgui.end()
    imgui.pop_style_color(2)
    return clicking, cx-x, cy-y, wheel
