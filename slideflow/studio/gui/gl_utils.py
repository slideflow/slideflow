"""OpenGL utility functions for rendering shapes and textures."""

import functools
import contextlib
import numpy as np
import OpenGL.GL as gl
import OpenGL.GL.ARB.texture_float

from ..utils import EasyDict

# -----------------------------------------------------------------------------

class Texture:
    """Class to assist with creation and render of an OpenGL texture."""

    def __init__(self, *, image=None, width=None, height=None, channels=None, dtype=None, bilinear=True, mipmap=True, maxlevel=None):
        self.gl_id = None
        self.bilinear = bilinear
        self.mipmap = mipmap

        # Determine size and dtype.
        if image is not None:
            image = prepare_texture(image)
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
        assert self.is_compatible(width=width, height=height, channels=channels, dtype=dtype)

        # Create texture object.
        self.gl_id = gl.glGenTextures(1)
        with self.bind():
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR if self.bilinear else gl.GL_NEAREST)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR if self.mipmap else gl.GL_NEAREST)
            if maxlevel:
                gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, maxlevel)
        self.update(image)

    def delete(self):
        if self.gl_id is not None:
            gl.glDeleteTextures([self.gl_id])
            self.gl_id = None

    def __del__(self):
        try:
            self.delete()
        except:
            pass

    @contextlib.contextmanager
    def bind(self):
        prev_id = gl.glGetInteger(gl.GL_TEXTURE_BINDING_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.gl_id)
        yield
        gl.glBindTexture(gl.GL_TEXTURE_2D, prev_id)

    def update(self, image):
        if image is not None:
            image = prepare_texture(image)
            assert self.is_compatible(image=image)
        with self.bind():
            fmt = get_texture_format(self.dtype, self.channels)
            gl.glPushClientAttrib(gl.GL_CLIENT_PIXEL_STORE_BIT)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, fmt.internalformat, self.width, self.height, 0, fmt.format, fmt.type, image)
            if self.mipmap:
                gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
            gl.glPopClientAttrib()

    def draw(self, *, pos=0, zoom=1, align=0, rint=False, color=1, alpha=1, rounding=0, anchor='center'):
        zoom = np.broadcast_to(np.asarray(zoom, dtype='float32'), [2])
        size = zoom * [self.width, self.height]
        with self.bind():
            gl.glPushAttrib(gl.GL_ENABLE_BIT)
            gl.glEnable(gl.GL_TEXTURE_2D)
            draw_rect(pos=pos, size=size, align=align, rint=rint, color=color, alpha=alpha, rounding=rounding, anchor=anchor)
            gl.glPopAttrib()

    def is_compatible(self, *, image=None, width=None, height=None, channels=None, dtype=None): # pylint: disable=too-many-return-statements
        if image is not None:
            if image.ndim != 3:
                return False
            ih, iw, ic = image.shape
            if not self.is_compatible(width=iw, height=ih, channels=ic, dtype=image.dtype):
                return False
        if width is not None and self.width != width:
            return False
        if height is not None and self.height != height:
            return False
        if channels is not None and self.channels != channels:
            return False
        if dtype is not None and self.dtype != dtype:
            return False
        return True

# -----------------------------------------------------------------------------

_texture_formats = {
    ('uint8',   1): EasyDict(type=gl.GL_UNSIGNED_BYTE, format=gl.GL_LUMINANCE,       internalformat=gl.GL_LUMINANCE8),
    ('uint8',   2): EasyDict(type=gl.GL_UNSIGNED_BYTE, format=gl.GL_LUMINANCE_ALPHA, internalformat=gl.GL_LUMINANCE8_ALPHA8),
    ('uint8',   3): EasyDict(type=gl.GL_UNSIGNED_BYTE, format=gl.GL_RGB,             internalformat=gl.GL_RGB8),
    ('uint8',   4): EasyDict(type=gl.GL_UNSIGNED_BYTE, format=gl.GL_RGBA,            internalformat=gl.GL_RGBA8),
    ('float32', 1): EasyDict(type=gl.GL_FLOAT,         format=gl.GL_LUMINANCE,       internalformat=OpenGL.GL.ARB.texture_float.GL_LUMINANCE32F_ARB),
    ('float32', 2): EasyDict(type=gl.GL_FLOAT,         format=gl.GL_LUMINANCE_ALPHA, internalformat=OpenGL.GL.ARB.texture_float.GL_LUMINANCE_ALPHA32F_ARB),
    ('float32', 3): EasyDict(type=gl.GL_FLOAT,         format=gl.GL_RGB,             internalformat=gl.GL_RGB32F),
    ('float32', 4): EasyDict(type=gl.GL_FLOAT,         format=gl.GL_RGBA,            internalformat=gl.GL_RGBA32F),
}

def get_texture_format(dtype, channels):
    return _texture_formats[(np.dtype(dtype).name, int(channels))]

# -----------------------------------------------------------------------------

def prepare_texture(image):
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if image.dtype.name == 'float64':
        image = image.astype('float32')
    return image

# -----------------------------------------------------------------------------

def pixel_capture(width, height, *, pos=0, dtype='uint8', channels=3):
    pos = np.broadcast_to(np.asarray(pos, dtype='float32'), [2])
    dtype = np.dtype(dtype)
    fmt = get_texture_format(dtype, channels)
    image = np.empty([height, width, channels], dtype=dtype)

    gl.glPushClientAttrib(gl.GL_CLIENT_PIXEL_STORE_BIT)
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    gl.glReadPixels(int(np.round(pos[0])), int(np.round(pos[1])), width, height, fmt.format, fmt.type, image)
    gl.glPopClientAttrib()
    return np.flipud(image)

# -----------------------------------------------------------------------------

def draw_shape(vertices, *, mode=gl.GL_TRIANGLE_FAN, pos=0, size=1, color=1, alpha=1, anchor='center'):
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


def create_buffer(vertices):
    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    return vbo


def draw_buffer(vbo, size):
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
    gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)

    gl.glMultiDrawArrays(gl.GL_LINE_LOOP, [i*4 for i in range(size)], [4 for _ in range(size)], size)

    gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)


def draw_rois(vertices, *, color=1, alpha=1, linewidth=2, vbo=None):
    """Draw multiple ROIs, reducing the number of OpenGL calls with VBO."""
    assert vertices.ndim == 3 and vertices.shape[2] == 2
    color = np.broadcast_to(np.asarray(color, dtype='float32'), [3])

    # Set the color and alpha
    gl.glColor4f(color[0] * alpha, color[1] * alpha, color[2] * alpha, alpha)
    gl.glLineWidth(linewidth)

    if vbo is not None:
        draw_buffer(vbo, size=vertices.shape[0])
    else:
        for i in range(vertices.shape[0]):
            draw_roi(vertices[i])

    gl.glLineWidth(1)


def draw_roi(vertices, *, color=1, alpha=1, linewidth=2, mode=gl.GL_LINE_STRIP):
    """Draw a single ROI using the fixed render pipeline."""
    assert vertices.ndim == 2 and vertices.shape[1] == 2
    color = np.broadcast_to(np.asarray(color, dtype='float32'), [3])
    gl.glLineWidth(linewidth)
    gl.glBegin(mode)
    gl.glColor4f(color[0] * alpha, color[1] * alpha, color[2] * alpha, alpha)
    for vertex in vertices:
        gl.glVertex2f(*vertex)
    gl.glVertex2f(*vertices[0])  # Close the ROI
    gl.glEnd()
    gl.glLineWidth(1)

    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

# -----------------------------------------------------------------------------

def draw_rect(*, pos=0, pos2=None, size=None, align=0, rint=False, color=1, alpha=1, rounding=0, mode=gl.GL_TRIANGLE_FAN, anchor='center'):
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
    vertices = _setup_center_rect(float(rounding[0]), float(rounding[1]))
    draw_shape(vertices, pos=pos, size=size, color=color, alpha=alpha, mode=mode, anchor=anchor)

draw_line = functools.partial(draw_rect, rounding=0, mode=gl.GL_LINE_STRIP)

@functools.lru_cache(maxsize=10000)
def _setup_center_rect(rx, ry):
    t = np.linspace(0, np.pi / 2, 1 if max(rx, ry) == 0 else 64)
    s = 1 - np.sin(t); c = 1 - np.cos(t)
    x = [c * rx, 1 - s * rx, 1 - c * rx, s * rx]
    y = [s * ry, c * ry, 1 - s * ry, 1 - c * ry]
    v = np.stack([x, y], axis=-1).reshape(-1, 2)
    return v.astype('float32')

# -----------------------------------------------------------------------------

def draw_circle(*, center=0, radius=100, hole=0, color=1, alpha=1):
    hole = np.broadcast_to(np.asarray(hole, dtype='float32'), [])
    vertices = _setup_circle(float(hole))
    draw_shape(vertices, mode=gl.GL_TRIANGLE_STRIP, pos=center, size=radius, color=color, alpha=alpha)


@functools.lru_cache(maxsize=10000)
def _setup_circle(hole):
    t = np.linspace(0, np.pi * 2, 128)
    s = np.sin(t); c = np.cos(t)
    v = np.stack([c, s, c * hole, s * hole], axis=-1).reshape(-1, 2)
    return v.astype('float32')

# -----------------------------------------------------------------------------

def draw_shadowed_line(verts, pos, linewidth=1, color=1, mode=gl.GL_LINE_LOOP, anchor='center'):
    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
    gl.glLineWidth(linewidth)
    draw_shape(verts, pos=pos, size=np.array([1, 1], dtype='float32'), color=color, mode=gl.GL_LINE_LOOP, anchor='center')
    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
    gl.glLineWidth(1)
