import imgui
import cv2
import numpy as np
import threading
from typing import Tuple
from os.path import dirname, join, abspath

from ..gui import gl_utils, imgui_utils
from ..gui.viewer import Viewer

#----------------------------------------------------------------------------

def center_crop(image, width, height):
    width = min(image.shape[1], width)
    height = min(image.shape[0], height)
    x = int(image.shape[1]/2 - width/2)
    y = int(image.shape[0]/2 - height/2)
    return image[y:y+height, x:x+width]

#----------------------------------------------------------------------------

class OpenCVCamera:

    def __init__(self, width, height, pipeline='/dev/video0'):
        self.width              = width
        self.height             = height
        self.should_stop        = False
        self._active_frame      = None
        self._thread            = None
        self._alignment_thread  = None
        self._last_img          = None
        self._align_x           = 500
        self._align_y           = 500
        raw_width               = 1920
        raw_height              = 1080

        format_str = 'video/x-raw(memory:NVMM), ' \
                     f'width={raw_width}, ' \
                     f'height={raw_height}, ' \
                     'format=(string)NV12, ' \
                     'framerate=(fraction)20/1'

        jetson_pipeline = ('nvarguscamerasrc '
                           f'! {format_str} '
                           '! nvvidconv '
                           '! video/x-raw, format=(string)BGRx '
                           '! videoconvert '
                           '! video/x-raw, format=(string)BGR '
                           '! appsink')

        if pipeline == 'jetson':
            self.cap = cv2.VideoCapture(jetson_pipeline)
        else:
            self.cap = cv2.VideoCapture(pipeline)

    def _thread_runner(self):
        while not self.should_stop:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._active_frame = frame

    def _alignment_tracker(self):
        import slideflow as sf
        while not self.should_stop:
            if self._last_img is not None and self._active_frame is not None:
                last = center_crop(self._last_img, 512, 512)
                now = center_crop(self._active_frame, 512, 512)
                l_r = last.shape[0] / 256.
                n_r = now.shape[0] / 256.
                last = cv2.resize(last, (int(last.shape[1]/l_r), 256), interpolation=cv2.INTER_LANCZOS4)
                now = cv2.resize(now, (int(now.shape[1]/n_r), 256), interpolation=cv2.INTER_LANCZOS4)
                try:
                    alignment = sf.slide.utils.align_by_translation(now, last, h=50, search_window=53)
                    self._align_x += alignment[0]
                    self._align_y += alignment[1]
                except sf.errors.AlignmentError:
                    print("Unable to align.")
                    self._align_x = 500
                    self._align_y = 500
                else:
                    print(alignment)
            else:
                print("Skipping alignment.")
                if self._active_frame is not None:
                    self._last_img = self._active_frame

    def start(self):
        self._thread = threading.Thread(target=self._thread_runner)
        self._thread.start()

    def stop(self):
        self.should_stop = True
        self._thread.join()
        self.should_stop = False

    def capture_frame(self, width=None, height=None):
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        while self._active_frame is None and not self.should_stop:
            pass
        return center_crop(self._active_frame, width, height)


class CameraViewer(Viewer):

    live        = True
    movable     = False

    def __init__(self,  um_width, width=800, height=600, assess_focus='laplacian', pipeline='/dev/video0', **kwargs):
        super().__init__(width=width, height=height, **kwargs)
        self.um_width       = um_width
        self.full_width     = 4056
        self.full_height    = 3040
        self.x              = None
        self.y              = None
        self._initialize(width, height, pipeline)
        self._assess_focus  = assess_focus
        self.last_preds     = []
        self.autocapture    = False
        self.enhance_contrast = False

        self.clahe = None

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

    def apply_args(self, args):
        super().apply_args(args)
        args.assess_focus = self._assess_focus

    def _enhance_contrast(self, img):
        if self.clahe is None:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        cl = self.clahe.apply(l_channel)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def _initialize(self, width, height, pipeline):
        self.width = width
        self.height = height
        preview_ratio = self.full_width / self.full_height
        if preview_ratio < (width / height):
            width = int(self.full_width / (self.full_height / height))
        else:
            height = int(self.full_height / (self.full_width / width))
        self.preview_width = width
        self.preview_height = height
        self.view_zoom = self.full_width / width
        self.camera = OpenCVCamera(width=width, height=height, pipeline=pipeline)
        self.camera.start()
        print("Initialized OpenCVCamera[{}] with p.width={} (window={}), p.height={} (window={})".format(
            pipeline,
            width,
            height,
            self.width,
            self.height
        ))
        self.camera_preview = None

    def set_um_width(self, um_width):
        self.um_width = um_width

    def stop(self):
        self.camera.stop()

    def close(self):
        self.stop()

    def get_full_still(self):
        return self.camera.capture_frame()

    @property
    def tile_view(self):
        if self.x is None or self.y is None:
            return
        ratio = self.full_width / self.view.shape[1]
        x = int(self.x / ratio)
        y = int(self.y / ratio)
        view = self.view[y:y+self.extract_px, x:x+self.extract_px, :]
        if self.enhance_contrast:
            view = self._enhance_contrast(view)
        return view

    def is_in_view(*args, **kwargs):
        return True

    def reload(self, width=None, height=None, x_offset=None, y_offset=None, normalizer=None, pipeline=None):
        self.stop()

        if x_offset is not None:
            self.x_offset = x_offset
        if y_offset is not None:
            self.y_offset = y_offset
        if normalizer is not None:
            self._normalizer = normalizer

        self._initialize(self.width if width is None else width,
                         self.height if height is None else height,
                         self.pipeline if pipeline is None else pipeline)

    def render(self, max_w, max_h):
        # Optional: capture high-quality still image.
        super().render()
        viz = self.viz
        config = self.viz._model_config

        # Get the image.
        self._last_img = self.view
        self._tex_img = self.view = self.camera.capture_frame(max_w, max_h)

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

        # Track high-certainty images.
        # First, check that we have predicions and the image is in focus.
        if (self.autocapture
            and viz._use_model
            and viz._predictions is not None
            and viz._model_config is not None
            and viz._use_uncertainty
            and viz._uncertainty is not None
            and (not hasattr(viz.result, 'in_focus') or viz.result.in_focus)):

            # Establish predictions, UQ, and thresholds.
            uq = viz._uncertainty
            pred = viz._predictions
            if 'thresholds' in config and 'tile_uq' in config['thresholds']:
                uq_thresh = config['thresholds']['tile_uq']
            else:
                uq_thresh = 0.033

            # Only process if the predictions have been updated
            if len(self.last_preds) and viz._predictions is not self.last_preds[-1][1]:
                #TODO: expand for more than just single categorical outcome

                if len(self.last_preds) >= 8:
                    self.last_preds.pop(0)
                self.last_preds.append((uq, pred))
                if (uq < uq_thresh
                    and len([p for p in self.last_preds if p[0] < uq_thresh]) >= 5
                    and len([p for p in self.last_preds if p[0] >= uq_thresh*2]) == 0):

                    # Capture high-certainty prediction
                    from PIL import Image
                    Image.fromarray(self._last_img).save('high_certainty.png')
                    print("Captured high-certainty predictions!")
                    self.capture_animation()
                    self.viz.create_toast('Captured high-certainty image', icon='success')
                    self.last_preds = []
            elif not len(self.last_preds):
                self.last_preds.append((uq, pred))
        else:
            self.last_preds = []


class CameraWidget:

    tag = 'camera'
    description = 'Camera Viewer'
    icon = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_camera.png')
    icon_highlighted = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_camera_highlighted.png')

    def __init__(self, viz, pipeline='/dev/video0'):
        self.viz            = viz
        self.um_width       = 800
        self.content_height = 0
        self.assess_focus   = 'laplacian'
        self.focus_methods  = ['laplacian', 'deepfocus']
        self._focus_idx     = 0
        self.autocapture    = False
        self.enhance_contrast = False

        viewer = CameraViewer(
            self.um_width,
            assess_focus=self.assess_focus,
            pipeline=pipeline,
            **viz._viewer_kwargs()
        )
        viz.set_viewer(viewer)
        viz._use_model_img_fmt = False

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            viz.header("Camera")

            self.content_height = viz.font_size + viz.spacing * 2
            imgui.text('Width (um)')
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 6):
                _changed, self.um_width = imgui.input_int('um_width', self.um_width, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                self.um_width = min(max(self.um_width, 1), 10000)
                if _changed:
                    viz.viewer.set_um_width(self.um_width)

            _, self.assess_focus = imgui.checkbox("Assess Focus", self.assess_focus)
            imgui.same_line()
            with imgui_utils.item_width(viz.font_size * 8):
                _, self._focus_idx = imgui.combo("##focus", self._focus_idx, self.focus_methods)
            viz.viewer._assess_focus = self.focus_methods[self._focus_idx] if self.assess_focus else None
            _clicked, self.autocapture = imgui.checkbox("Auto-capture", self.autocapture)
            if _clicked:
                viz.viewer.autocapture = self.autocapture
            _clicked, self.enhance_contrast = imgui.checkbox("Enhance contrast", self.enhance_contrast)
            if _clicked:
                viz.viewer.enhance_contrast = self.enhance_contrast
        else:
            self.content_height = 0
