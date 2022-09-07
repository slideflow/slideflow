from picamera2 import Picamera2
from picamera2.previews import NullPreview

from .gui_utils import gl_utils, imgui_utils

#----------------------------------------------------------------------------

class PicamPreview(NullPreview):

    def __init__(self, widget, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.widget = widget

    def handle_request(self, picam2):
        """Handle requests

        :param picam2: picamera2 object
        :type picam2: Picamera2
        """
        completed_request = picam2.process_requests()
        if completed_request:
            if not len(self.widget.preview_images):
                self.widget.preview_images += [completed_request.make_array("main")[:, :, 0:3]]
            completed_request.release()


class PicamWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.picam2         = Picamera2()
        self.picam_preview  = None
        self._still_config  = self.picam2.create_still_configuration(
            main={"size": (4056, 3040)},
        )
        self.start_picam_preview()

    def start_picam_preview(self):
        viz = self.viz
        if self.picam_preview is not None:
            self.picam_preview.stop()
        self.picam2.stop()

        # Draw on screen.
        max_w = viz.content_width - viz.pane_w
        max_h = viz.content_height
        width = 4056
        height = 3040
        preview_ratio = width / height
        if preview_ratio < (max_w / max_h):
            max_w = int(width / (height / max_h))
        else:
            max_h = int(height / (width / max_w))

        config = self.picam2.create_preview_configuration(
            main={"size": (max_w, max_h)},
        )
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

    def _on_window_change(self):
        self.start_picam_preview()

    def render(self):
        viz = self.viz

        # Optional: capture high-quality still image.
        #array = self.picam2.switch_mode_and_capture_array(self._still_config)
        #array = self.picam2.capture_array()
        #if array is not None:
        #   ...

        if len(self.preview_images):
            # Get the image.
            self._picam_tex_img = self.preview_images.pop(0)

            # Normalize.
            if viz._normalizer and viz._normalize_wsi:
                self._picam_tex_img = viz._normalizer.transform(self._picam_tex_img)

            # Update texture.
            if self._picam_tex_obj is None or not self._picam_tex_obj.is_compatible(image=self._picam_tex_img):
                if self._picam_tex_obj is not None:
                    self._tex_to_delete += [self._picam_tex_obj]
                self._picam_tex_obj = gl_utils.Texture(image=self._picam_tex_img, bilinear=True, mipmap=True)
            else:
                self._picam_tex_obj.update(self._picam_tex_img)

        # Calculate location and draw.
        img = self._picam_tex_img
        if img is not None:
            off_x = int(((viz.content_width - viz.pane_w) - img.shape[1]) / 2)
            off_y = int((viz.content_height - img.shape[0]) / 2)
            h_pos = (viz.pane_w + off_x, off_y)
            self._picam_tex_obj.draw(pos=h_pos, zoom=1, align=0.5, rint=True, anchor='topleft')

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        pass

#----------------------------------------------------------------------------
