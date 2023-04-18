import imgui
from picamera2 import Picamera2
from picamera2.previews import NullPreview
from typing import Tuple

from ..gui import gl_utils, imgui_utils
from ..gui.viewer import Viewer

#----------------------------------------------------------------------------

class PicamPreview(NullPreview):

    def __init__(self, viewer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.viewer = viewer

    def handle_request(self, picam2):
        """Handle requests

        :param picam2: picamera2 object
        :type picam2: Picamera2
        """
        completed_request = picam2.process_requests()
        if completed_request:
            if not len(self.viewer.preview_images):
                self.viewer.preview_images += [completed_request.make_array("main")[:, :, 0:3]]
            completed_request.release()


class PicamViewer(Viewer):

    live        = True
    movable     = False

    def __init__(self,  um_width, width=800, height=600, **kwargs):
        super().__init__(width=width, height=height, **kwargs)
        self.picam2         = Picamera2()
        self.um_width       = um_width
        self.full_width     = 4056
        self.full_height    = 3040
        self.x              = None
        self.y              = None
        self._still_config  = self.picam2.create_still_configuration(
            main={"size": (self.full_width, self.full_height)},
        )
        self._initialize(width, height)

    @property
    def dimensions(self) -> Tuple[int, int]:
        return (self.full_width, self.full_height)

    @property
    def full_extract_px(self) -> int:
        return int(self.tile_um / self.mpp)

    @property
    def extract_px(self) -> int:
        return int(self.tile_um / self.preview_mpp)

    @property
    def mpp(self) -> float:
        return self.um_width / self.full_width

    @property
    def preview_mpp(self) -> float:
        return self.um_width / self.preview_width

    def _initialize(self, width, height):
        self.width = width
        self.height = height
        preview_ratio = self.full_width / self.full_height
        if preview_ratio < (width / height):
            width = int(self.full_width / (self.full_height / height))
        else:
            height = int(self.full_height / (self.full_width / width))

        config = self.picam2.create_preview_configuration(
            main={"size": (width, height)},
        )
        print("Initialized PicamViewer with p.width={} (window={}), p.height={} (window={})".format(
            width,
            height,
            self.width,
            self.height
        ))
        self.preview_width = width
        self.preview_height = height
        self.view_zoom = self.full_width / width
        self.picam2.configure(config)
        self.picam_preview = PicamPreview(self)
        self.picam_preview.start(self.picam2)
        self.picam2.have_event_loop = True
        self.picam2._preview = self.picam_preview
        self.picam2.start()
        self.camera_preview = None
        self.preview_images = []
        self._picam_tex_obj = None
        self._picam_tex_img = None

    def set_um_width(self, um_width):
        self.um_width = um_width

    def stop(self):
        self.picam_preview.stop()
        self.picam2.stop()

    def close(self):
        self.stop()

    def get_full_still(self):
        return self.picam2.switch_mode_and_capture_array(self._still_config)

    @property
    def tile_view(self):
        if self.x is None or self.y is None:
            return
        ratio = self.full_width / self.view.shape[1]
        x = int(self.x / ratio)
        y = int(self.y / ratio)
        return self.view[y:y+self.extract_px, x:x+self.extract_px, :]

    def is_in_view(*args, **kwargs):
        return True

    def reload(self, width=None, height=None, x_offset=None, y_offset=None, normalizer=None):
        self.stop()

        if x_offset is not None:
            self.x_offset = x_offset
        if y_offset is not None:
            self.y_offset = y_offset
        if normalizer is not None:
            self._normalizer = normalizer

        self._initialize(self.width if width is None else width,
                         self.height if height is None else height)

    def render(self, max_w, max_h):
        # Optional: capture high-quality still image.
        super().render()

        if len(self.preview_images):

            # Get the image.
            self._tex_img = self.view = self.preview_images.pop(0)

            # Normalize.
            if self._normalizer:
                self._tex_img = self._normalizer.transform(self._tex_img)

            # Update texture.
            if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                if self._tex_obj is not None:
                    self._tex_to_delete += [self._tex_obj]
                self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=True, mipmap=True)
            else:
                self._tex_obj.update(self._tex_img)

        # Calculate location and draw.
        img = self._tex_img
        if img is not None:
            off_x = int((max_w - img.shape[1]) / 2)
            off_y = int((max_h - img.shape[0]) / 2)
            h_pos = (self.x_offset + off_x, self.y_offset + off_y)
            self._tex_obj.draw(pos=h_pos, zoom=1, align=0.5, rint=True, anchor='topleft')

class PicamWidget:

    tag = 'camera'

    def __init__(self, viz):
        self.viz            = viz
        self.um_width       = 800
        self.content_height = 0

        viewer = PicamViewer(self.um_width, **viz._viewer_kwargs())
        viz.set_viewer(viewer)
        viz._use_model_img_fmt = False

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            self.content_height = viz.font_size + viz.spacing * 2
            imgui.text('Width (um)')
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 6):
                _changed, self.um_width = imgui.input_int('um_width', self.um_width, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                self.um_width = min(max(self.um_width, 1), 10000)
                if _changed:
                    viz.viewer.set_um_width(self.um_width)
        else:
            self.content_height = 0

#----------------------------------------------------------------------------
