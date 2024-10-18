#!/usr/bin/env python

import pkgutil
import threading
import click
import numpy as np
import glfw
import OpenGL.GL as gl
import OpenGL.GL.ARB.texture_float

from os.path import dirname, realpath, join
from PIL import Image, ImageFont
from contextlib import contextmanager
from functools import lru_cache
from os.path import join, dirname

__version__ = "3.0.2"

# -----------------------------------------------------------------------------

@click.command()
@click.argument('slide', metavar='PATH', required=False)
@click.option('--model', '-m', help='Classifier network for categorical predictions.', metavar='PATH')
@click.option('--project', '-p', help='Slideflow project.', metavar='PATH')
@click.option('--low_memory', '-l', is_flag=True, help='Low memory mode.', metavar=bool)
@click.option('--stylegan', '-g', is_flag=True, help='Enable StyleGAN support (requires PyTorch).', metavar=bool)
@click.option('--picam', '-pc', is_flag=True, help='Enable Picamera2 view (experimental).', metavar=bool)
@click.option('--camera', '-c', is_flag=True, help='Enable Camera (OpenCV) view (experimental).', metavar=bool)
@click.option('--cellpose', is_flag=True, help='Enable Cellpose segmentation (experimental).', metavar=bool)
def main(
    slide,
    model,
    project,
    low_memory,
    stylegan,
    picam,
    camera,
    cellpose
):
    """
    Whole-slide image viewer with deep learning model visualization tools.

    Optional PATH argument can be used specify which slide to initially load.
    """
    # Start the splash screen
    import_with_splash()

    from slideflow.studio import Studio

    if low_memory is None:
        low_memory = False

    # Load widgets
    widgets = Studio.get_default_widgets()
    if stylegan:
        from slideflow.studio.widgets.stylegan import StyleGANWidget
        widgets += [StyleGANWidget]

    if picam:
        from slideflow.studio.widgets.picam import PicamWidget
        widgets += [PicamWidget]

    if camera:
        from slideflow.studio.widgets.cvcam import CameraWidget
        widgets += [CameraWidget]

    if cellpose:
        from slideflow.studio.widgets.cellseg import CellSegWidget
        widgets += [CellSegWidget]

    viz = Studio(low_memory=low_memory, widgets=widgets)
    viz.project_widget.search_dirs += [dirname(realpath(__file__))]

    # Load model.
    if model is not None:
        viz.load_model(model)

    if project is not None:
        viz.load_project(project)

    # Load slide(s).
    if slide:
        viz.load_slide(slide)

    # Run.
    viz.run()

#----------------------------------------------------------------------------


def import_with_splash():

    _imported = False

    def _import_sildeflow():
        nonlocal _imported
        import slideflow.studio
        _imported = True

    # Start the import thread
    _thread = threading.Thread(target=_import_sildeflow)
    _thread.start()

    # Send Tk to the background (used for future file dialogs)
    from tkinter import Tk
    Tk().withdraw()

    # Load image
    sf_root = pkgutil.get_loader('slideflow').get_filename()
    splash_path = join(dirname(sf_root), 'studio', 'gui', 'splash.png')
    icon_path = join(dirname(sf_root), 'studio', 'gui', 'icons', 'logo.png')
    img = np.array(Image.open(splash_path))
    icon = np.array(Image.open(icon_path))

    # Start GLFW window
    if not glfw.init():
        return

    glfw.window_hint(glfw.DECORATED, False)
    height = img.shape[0]
    width = img.shape[1]

    # Center on screen
    _, _, mw, mh = glfw.get_monitor_workarea(glfw.get_primary_monitor())
    if not mw:
        vmode = glfw.get_video_mode(glfw.get_primary_monitor())
        mw, mh = vmode.size
        wscale, hscale = glfw.get_monitor_content_scale(glfw.get_primary_monitor())
    else:
        wscale, hscale = 1, 1
    window = glfw.create_window(int(width/wscale), int(height/hscale), "Slideflow Studio", None, None)
    glfw.set_window_pos(window, (mw - int(width/wscale)) // 2, (mh - int(height/hscale)) // 2)

    _tex_bg = None
    _tex_icon = None
    _version_text = None
    _first_frame = True

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    while not glfw.window_should_close(window):

        glfw.poll_events()

        # Initialize GL state.
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glTranslate(-1, 1, 0)
        gl.glScale(2 / max(width, 1), -2 / max(height, 1), 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA) # Pre-multiplied alpha.

        # Clear.
        gl.glClearColor(0, 0, 0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if _tex_bg is None:
            _tex_bg = Texture(image=img, bilinear=False)
        if _tex_icon is None:
            _tex_icon = Texture(image=icon, bilinear=True)
        if _version_text is None:
            _version_text = text_texture(__version__, size=22)
        _tex_bg.draw(pos=0, zoom=1, align=0.5, rint=True, anchor='topleft')
        _tex_icon.draw(pos=(width//2, int(height * 0.3)), zoom=0.25, align=0.5, rint=True, anchor='center')
        _version_text.draw(pos=(width//2, int(height * 0.7)), zoom=1, align=0.5, rint=True, anchor='center')

        if not _first_frame:
            glfw.show_window(window)
            _first_frame = False

        glfw.swap_buffers(window)

        if _imported:
            glfw.destroy_window(window)
            glfw.default_window_hints()
            break

    glfw.terminate()

# -----------------------------------------------------------------------------

class Texture:
    """Class to assist with creation and render of an OpenGL texture."""

    _texture_formats = {
        ('uint8',   1): dict(type=gl.GL_UNSIGNED_BYTE, format=gl.GL_LUMINANCE,       internalformat=gl.GL_LUMINANCE8),
        ('uint8',   2): dict(type=gl.GL_UNSIGNED_BYTE, format=gl.GL_LUMINANCE_ALPHA, internalformat=gl.GL_LUMINANCE8_ALPHA8),
        ('uint8',   3): dict(type=gl.GL_UNSIGNED_BYTE, format=gl.GL_RGB,             internalformat=gl.GL_RGB8),
        ('uint8',   4): dict(type=gl.GL_UNSIGNED_BYTE, format=gl.GL_RGBA,            internalformat=gl.GL_RGBA8),
        ('float32', 1): dict(type=gl.GL_FLOAT,         format=gl.GL_LUMINANCE,       internalformat=gl.ARB.texture_float.GL_LUMINANCE32F_ARB),
        ('float32', 2): dict(type=gl.GL_FLOAT,         format=gl.GL_LUMINANCE_ALPHA, internalformat=gl.ARB.texture_float.GL_LUMINANCE_ALPHA32F_ARB),
        ('float32', 3): dict(type=gl.GL_FLOAT,         format=gl.GL_RGB,             internalformat=gl.GL_RGB32F),
        ('float32', 4): dict(type=gl.GL_FLOAT,         format=gl.GL_RGBA,            internalformat=gl.GL_RGBA32F),
    }

    def __init__(
        self,
        *,
        image=None,
        width=None,
        height=None,
        channels=None,
        dtype=None,
        bilinear=True,
        mipmap=True
    ) -> None:

        self.gl_id = None
        self.bilinear = bilinear
        self.mipmap = mipmap

        # Determine size and dtype.
        if image is not None:
            image = np.asarray(image)
            if image.ndim == 2:
                image = image[:, :, np.newaxis]
            if image.dtype.name == 'float64':
                image = image.astype('float32')
            self.height, self.width, self.channels = image.shape
            self.dtype = image.dtype
        else:
            assert width is not None and height is not None
            self.width = width
            self.height = height
            self.channels = channels if channels is not None else 3
            self.dtype = np.dtype(dtype) if dtype is not None else np.uint8

        # Validate size and dtype.
        assert isinstance(self.width, int) and self.width >= 0
        assert isinstance(self.height, int) and self.height >= 0
        assert isinstance(self.channels, int) and self.channels >= 1

        # Create texture object.
        self.gl_id = gl.glGenTextures(1)
        with self.bind():
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR if self.bilinear else gl.GL_NEAREST)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR if self.mipmap else gl.GL_NEAREST)

        with self.bind():
            fmt = self._texture_formats[(np.dtype(self.dtype).name, int(self.channels))]
            gl.glPushClientAttrib(gl.GL_CLIENT_PIXEL_STORE_BIT)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, fmt['internalformat'], self.width, self.height, 0, fmt['format'], fmt['type'], image)
            if self.mipmap:
                gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
            gl.glPopClientAttrib()

    @contextmanager
    def bind(self):
        prev_id = gl.glGetInteger(gl.GL_TEXTURE_BINDING_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.gl_id)
        yield
        gl.glBindTexture(gl.GL_TEXTURE_2D, prev_id)

    def draw(self, *, pos=0, zoom=1, align=0, rint=False, color=1, alpha=1, rounding=0, anchor='center'):
        zoom = np.broadcast_to(np.asarray(zoom, dtype='float32'), [2])
        size = zoom * [self.width, self.height]
        with self.bind():
            gl.glPushAttrib(gl.GL_ENABLE_BIT)
            gl.glEnable(gl.GL_TEXTURE_2D)
            self._draw_rect(pos=pos, size=size, align=align, rint=rint, color=color, alpha=alpha, rounding=rounding, anchor=anchor)
            gl.glPopAttrib()

    def delete(self):
        if self.gl_id is not None:
            gl.glDeleteTextures([self.gl_id])
            self.gl_id = None

    def __del__(self):
        try:
            self.delete()
        except:
            pass

    def _draw_rect(self, *, pos=0, pos2=None, size=None, align=0, rint=False, color=1, alpha=1, rounding=0, mode=gl.GL_TRIANGLE_FAN, anchor='center'):
        assert pos2 is None or size is None
        assert anchor in ('center', 'topleft')
        pos = np.broadcast_to(np.asarray(pos, dtype='float32'), [2])
        pos2 = np.broadcast_to(np.asarray(pos2, dtype='float32'), [2]) if pos2 is not None else None
        size = np.broadcast_to(np.asarray(size, dtype='float32'), [2]) if size is not None else None
        size = size if size is not None else pos2 - pos if pos2 is not None else np.array([1, 1], dtype='float32')
        pos = pos - size * align
        if rint:
            pos = np.rint(pos)
        rounding = np.broadcast_to(np.asarray(rounding, dtype='float32'), [2])
        rounding = np.minimum(np.abs(rounding) / np.maximum(np.abs(size), 1e-8), 0.5)
        if np.min(rounding) == 0:
            rounding *= 0
        vertices = self._setup_center_rect(float(rounding[0]), float(rounding[1]))
        self._draw_shape(vertices, pos=pos, size=size, color=color, alpha=alpha, mode=mode, anchor=anchor)

    def _draw_shape(self, vertices, *, mode=gl.GL_TRIANGLE_FAN, pos=0, size=1, color=1, alpha=1, anchor='center'):
        assert vertices.ndim == 2 and vertices.shape[1] == 2
        assert anchor in ('center', 'topleft')
        pos = np.broadcast_to(np.asarray(pos, dtype='float32'), [2])
        size = np.broadcast_to(np.asarray(size, dtype='float32'), [2])
        color = np.broadcast_to(np.asarray(color, dtype='float32'), [3])
        alpha = np.clip(np.broadcast_to(np.asarray(alpha, dtype='float32'), []), 0, 1)

        gl.glPushClientAttrib(gl.GL_CLIENT_VERTEX_ARRAY_BIT)
        gl.glPushAttrib(gl.GL_CURRENT_BIT | gl.GL_TRANSFORM_BIT)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
        gl.glVertexPointer(2, gl.GL_FLOAT, 0, vertices)
        gl.glTexCoordPointer(2, gl.GL_FLOAT, 0, vertices)
        if anchor == 'center':
            gl.glTranslate(pos[0], pos[1], 0)
        else:
            gl.glTranslate(pos[0]+size[0]/2, pos[1]+size[1]/2, 0)
        gl.glScale(size[0], size[1], 1)
        gl.glColor4f(color[0] * alpha, color[1] * alpha, color[2] * alpha, alpha)
        gl.glDrawArrays(mode, 0, vertices.shape[0])

        gl.glPopMatrix()
        gl.glPopAttrib()
        gl.glPopClientAttrib()

    @lru_cache(maxsize=10000)
    def _setup_center_rect(self, rx, ry):
        t = np.linspace(0, np.pi / 2, 1 if max(rx, ry) == 0 else 64)
        s = 1 - np.sin(t); c = 1 - np.cos(t)
        x = [c * rx, 1 - s * rx, 1 - c * rx, s * rx]
        y = [s * ry, c * ry, 1 - s * ry, 1 - c * ry]
        v = np.stack([x, y], axis=-1).reshape(-1, 2)
        return v.astype('float32')


@lru_cache(maxsize=10000)
def _text_to_array(string, *, font=None, size=32, line_pad: int=None):

    if font is None:
        sf_root = pkgutil.get_loader('slideflow').get_filename()
        font = join(dirname(sf_root), 'studio', 'gui', 'fonts', 'DroidSans.ttf')
    pil_font = ImageFont.truetype(font=font, size=size)

    lines = [pil_font.getmask(line, 'L') for line in string.split('\n')]
    lines = [np.array(line, dtype=np.uint8).reshape([line.size[1], line.size[0]]) for line in lines]
    width = max(line.shape[1] for line in lines)
    lines = [np.pad(line, ((0, 0), (0, width - line.shape[1])), mode='constant') for line in lines]
    line_spacing = line_pad if line_pad is not None else size // 2
    lines = [np.pad(line, ((0, line_spacing), (0, 0)), mode='constant') for line in lines[:-1]] + lines[-1:]
    mask = np.concatenate(lines, axis=0)
    alpha = mask
    return np.stack([mask, alpha], axis=-1)


@lru_cache(maxsize=10000)
def text_texture(string, bilinear=False, mipmap=True, **kwargs):
    return Texture(image=_text_to_array(string, **kwargs), bilinear=bilinear, mipmap=mipmap)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
