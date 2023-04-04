import os
import re
import numpy as np
import imgui
import PIL.Image
from ..gui import imgui_utils
from .._renderer import CapturedException

#----------------------------------------------------------------------------

class CaptureWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.path           = os.path.join(os.getcwd(), '_screenshots')
        self.dump_tile      = False
        self.dump_gui       = False
        self.dump_view      = False
        self._crop_next     = False
        self.defer_frames   = 0
        self.disabled_time  = 0

    def dump_png(self, image):
        viz = self.viz
        try:
            _height, _width, channels = image.shape
            assert channels in [1, 3]
            if image.dtype != np.uint8:
                try:
                    image = image.numpy()
                except Exception:
                    raise ValueError(f'Expected type np.uint8, got {image.dtype}')
            os.makedirs(self.path, exist_ok=True)
            file_id = 0
            for entry in os.scandir(self.path):
                if entry.is_file():
                    match = re.fullmatch(r'(\d+).*', entry.name)
                    if match:
                        file_id = max(file_id, int(match.group(1)) + 1)
            if channels == 1:
                pil_image = PIL.Image.fromarray(image[:, :, 0], 'L')
            else:
                pil_image = PIL.Image.fromarray(image, 'RGB')
            dest = os.path.join(self.path, f'{file_id:05d}.png')
            pil_image.save(dest)
            self.viz.create_toast(f"Saved captured image to {dest}", icon='success')
        except:
            self.viz.create_toast(f"An error occurred attempting to save captured image.", icon='error')
            viz.result.error = CapturedException()

    def save_tile(self):
        self.dump_tile = True
        self.defer_frames = 2
        self.disabled_time = 0.5

    def save_view(self):
        self.dump_view = True
        self.defer_frames = 2
        self.disabled_time = 0.5

    def save_gui(self):
        self.dump_gui = True
        self.defer_frames = 2
        self.disabled_time = 0.5

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            if viz.collapsing_header('Capture', default=True):
                with imgui_utils.grayed_out(self.disabled_time != 0):
                    _changed, self.path = imgui_utils.input_text('##path', self.path, 1024,
                        flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                        width=-1,
                        help_text='PATH')
                    if imgui.is_item_hovered() and not imgui.is_item_active() and self.path != '':
                        imgui.set_tooltip(self.path)
                    if imgui_utils.button('Save view', width=viz.button_w, enabled=(self.disabled_time == 0 and viz.viewer)):
                        self.save_view()
                    imgui.same_line()
                    if imgui_utils.button('Save tile', width=viz.button_w, enabled=(self.disabled_time == 0 and 'image' in viz.result)):
                        self.save_tile()
                    imgui.same_line()
                    if imgui_utils.button('Save GUI', width=-1, enabled=(self.disabled_time == 0)):
                        self.save_gui()

        self.disabled_time = max(self.disabled_time - viz.frame_delta, 0)
        if self.defer_frames > 0:
            self.defer_frames -= 1
        elif self.dump_tile:
            if 'image' in viz.result:
                self.dump_png(viz.result.image)
            self.dump_tile = False
        elif self.dump_gui or self.dump_view:
            viz.capture_next_frame()
            self._crop_next = self.dump_view
            self.dump_view = self.dump_gui = False
        captured_frame = viz.pop_captured_frame()
        if captured_frame is not None:
            if self._crop_next:
                captured_frame = captured_frame[self.viz.offset_y_pixels:, self.viz.offset_x_pixels:, :]
            self.dump_png(captured_frame)
            self._crop_next = False

#----------------------------------------------------------------------------
