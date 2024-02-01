import torch
import slideflow as sf
import imgui
import numpy as np
import segmentation_models_pytorch as smp
from os.path import join, dirname, abspath, exists
from threading import Thread
from tkinter.filedialog import askopenfilename
from slideflow.segment import TileMaskDataset
from slideflow.model.torch_utils import get_device

from ._utils import Widget
from ..gui import imgui_utils
from ..utils import LEFT_MOUSE_BUTTON, RIGHT_MOUSE_BUTTON
from .slide import stride_capture

from pytorch_lightning.callbacks import Callback

class ProgressCallback(Callback):

    def __init__(self, toast, max_epochs):
        super().__init__()
        self.toast = toast
        self.max_epochs = max_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        percent = (trainer.current_epoch + 1) / self.max_epochs
        self.toast.set_progress(min(percent, 1.))

# ----------------------------------------------------------------------------


class TissueSegWidget(Widget):

    tag = 'segment'
    description = 'Tissue Segmentation'
    #icon = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_segment.png')
    #icon_highlighted = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_segment_highlighted.png')

    def __init__(self, viz):
        self.viz                    = viz
        self._segment               = None
        self._thread                = None
        self._load_toast            = None
        self._working_toast         = None
        self._training_toast        = None
        self._show_params           = False
        self._rois_at_start         = 0
        self._need_to_refresh_rois  = False
        self._clicking              = False
        self._show_popup            = False

        # Parameters
        self._supported_archs       = ['FPN', 'DeepLabV3', 'DeepLabV3Plus', 'Linknet', 'MAnet', 'PAN', 'PSPNet', 'Unet', 'UnetPlusPlus']
        self._selected_arch         = 0
        self._supported_encoders    = smp.encoders.get_encoder_names()
        self._selected_encoder      = self._supported_encoders.index('resnet34')
        self._filter_methods        = ['otsu', 'roi']
        self._selected_filter_method = 0
        self._training_modes        = ['binary', 'multiclass', 'multilabel']
        self._selected_training_mode = 0
        self.max_epochs             = 20
        self.tile_px                = 1024
        self.tile_um                = 2048
        self.crop_margin            = 256
        self.stride                 = 1
        self._capturing_stride      = 1


    # --- Properties ---

    @property
    def cfg(self) -> sf.segment.SegmentConfig:
        seg = self._segment
        return None if seg is None else seg.cfg

    @property
    def arch(self) -> str:
        return self._supported_archs[self._selected_arch]

    @property
    def encoder(self) -> str:
        return self._supported_encoders[self._selected_encoder]

    @property
    def mpp(self) -> float:
        return self.tile_um / self.tile_px

    @property
    def filter_method(self) -> str:
        return self._filter_methods[self._selected_filter_method]

    @property
    def mode(self) -> str:
        return self._training_modes[self._selected_training_mode]

    # --- Internal ---

    def close(self):
        pass

    def is_thread_running(self):
        return self._thread is not None and self._thread.is_alive()

    def is_training(self):
        return self._training_toast is not None

    def drag_and_drop_hook(self, path, ignore_errors=False) -> bool:
        """Handle file paths provided via drag-and-drop."""
        if (sf.util.path_to_ext(path).lower() == 'pth'):
            if exists(join(dirname(path), 'segment_params.json')):
                self.load(path, ignore_errors=ignore_errors)
                return True
        return False

    # --- Model loading ---

    def ask_load_model(self) -> str:
        model_path = askopenfilename(
            title="Load model...",
            filetypes=[("pth", ".pth"), ("All files", ".*")]
        )
        if model_path:
            self.load(model_path)

    def load(self, path, ignore_errors=False):
        """Load a tissue segmentation model."""
        if self.is_thread_running():
            self._thread.join()
        self._load_toast = self.viz.create_toast(
            title=f"Loading segmentation model",
            icon='info',
            sticky=True,
            spinner=True)
        self._thread = Thread(target=self._load_model, args=(path, ignore_errors))
        self._thread.start()

    def _load_model(self, path, ignore_errors=False):
        try:
            self._segment = sf.slide.qc.Segment(path)
            self._segment.model.to(get_device())
        except Exception as e:
            if self._load_toast is not None:
                self._load_toast.done()
            sf.log.error(f"Error loading segment model: {e}")
            self.viz.create_toast(f"Error loading segment model: {e}", icon="error")
            self._segment = None
        else:
            if self._load_toast is not None:
                self._load_toast.done()
            self.viz.create_toast(
                f"Loaded model at {path}.",
                icon="success"
            )

    def close_model(self) -> None:
        self._segment = None

    def generate_rois(self):
        """Generate ROIs from the loaded segmentation model."""
        if self.is_thread_running():
            self.viz.create_toast("Failed to start thread.", icon="error")
            return
        self._rois_at_start = len(self.viz.wsi.rois)
        self._working_toast = self.viz.create_toast(
            title=f"Generating ROIs from segmentation model",
            icon='info',
            sticky=True,
            spinner=True)
        self._thread = Thread(target=self._generate_rois)
        self._thread.start()

    def _generate_rois(self):
        viz = self.viz
        self._segment.generate_rois(viz.wsi)
        self._need_to_refresh_rois = True
        if self._working_toast is not None:
            self._working_toast.done()
        viz.create_toast(
            "Generated {} ROIs.".format(
                len(self.viz.wsi.rois) - self._rois_at_start
            ),
            icon="success"
        )

    def train(self) -> None:
        """Train a segmentation model."""
        if self.is_thread_running():
            self.viz.create_toast("Failed to start thread.", icon="error")
            return

        # Create a progress toast.
        if self._training_toast is not None:
            self._training_toast.done()
        self._training_toast = self.viz.create_toast(
            title="Training segmentation model",
            icon='info',
            sticky=True,
            progress=True,
            spinner=True
        )
        self._thread = Thread(target=self._train)
        self._thread.start()

    def _train(self) -> None:
        """Train a segmentation model."""
        import pytorch_lightning as pl

        viz = self.viz

        # Prepare the dataset.
        dataset = viz.P.dataset(filters={'slide': [viz.wsi.name]})
        dts = TileMaskDataset(
            dataset,
            tile_px=self.tile_px,
            tile_um=self.tile_um,
            stride_div=self.stride,
            crop_margin=self.crop_margin,
            filter_method=self.filter_method
        )

        # Set the configuration.
        config = sf.segment.SegmentConfig(
            arch=self.arch,
            encoder_name=self.encoder,
            epochs=self.max_epochs,  # 100
            mpp=self.mpp,
            mode=self.mode,
        )

        # Create dataloader.
        train_dl = torch.utils.data.DataLoader(
            dts,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=4
        )

        # Build the model and trainer.
        model = config.build_model()
        trainer = pl.Trainer(
            max_epochs=config.epochs,
            devices=1,   # Distributed training not supported in a GUI.
            num_nodes=1, # Distributed training not supported in a GUI.
            callbacks=[ProgressCallback(self._training_toast, config.epochs)]
        )

        # Train the model.
        trainer.fit(model, train_dataloaders=train_dl)

        # Move model to eval & appropriate device.
        model.eval()
        model.to(get_device())

        # Create the segment object.
        self._segment = sf.slide.qc.Segment.from_model(model, config)

        # Cleanup.
        self._training_toast.done()
        self._training_toast = None
        self.viz.create_toast("Training complete.", icon="success")

    # --- Drawing ---

    def draw_info(self):
        """Draw information about the loaded model."""
        viz = self.viz

        rows = [
            ['Architecture', self.cfg.arch],
            ['Encoder',      self.cfg.encoder_name],
            ['Mode',         self.cfg.mode],
            ['Classes',      self.cfg.out_classes],
            ['MPP',          self.cfg.mpp, 'Microns per pixel (optical resolution)']
        ]
        imgui.text_colored('Model', *viz.theme.dim)
        imgui.same_line(viz.font_size * 6)
        model_path = self._segment.model_path or 'None'
        with imgui_utils.clipped_with_tooltip(model_path, 22):
            imgui.text(imgui_utils.ellipsis_clip(model_path, 22))
        for y, cols in enumerate(rows):
            for x, col in enumerate(cols):
                if x != 0:
                    imgui.same_line(viz.font_size * (6 + (x - 1) * 6))
                if x == 0:
                    imgui.text_colored(str(col), *viz.theme.dim)
                    if len(cols) == 3 and imgui.is_item_hovered():
                        imgui.set_tooltip(cols[2])
                elif x == 1:
                    imgui.text(str(col))

        imgui.same_line(imgui.get_content_region_max()[0] - viz.font_size - viz.spacing * 2)
        if imgui.button("HP"):
            self._show_params = not self._show_params

        imgui_utils.vertical_break()

    def draw_train_data_source(self) -> None:
        """Draw training data source options."""
        viz = self.viz

        # Slide sources
        imgui.text_colored('Source', *viz.theme.dim)
        imgui.same_line(viz.label_w)
        imgui.text("<this slide>")

        imgui_utils.vertical_break()

    def draw_train_data_processing(self) -> None:
        """Draw training data processing options."""
        viz = self.viz

        # Tile size.
        imgui.text_colored('Tile size', *viz.theme.dim)
        imgui.same_line(viz.label_w)
        with imgui_utils.item_width(viz.font_size * 3):
            _, self.tile_px = imgui.input_int(
                "##segment_tile_px",
                self.tile_px,
                step=0,
            )
        imgui.same_line()
        imgui.text('px')
        imgui.text('')
        imgui.same_line(viz.label_w)
        with imgui_utils.item_width(viz.font_size * 3):
            _, self.tile_um = imgui.input_int(
                "##segment_tile_um",
                self.tile_um,
                step=0,
            )
        imgui.same_line()
        imgui.text('um')
        imgui.same_line()
        imgui.text('(MPP={:.2f})'.format(self.mpp))

        # Crop margin.
        imgui.text_colored('Margin', *viz.theme.dim)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Margin for random cropping during training.")
        imgui.same_line(viz.label_w)
        with imgui_utils.item_width(viz.font_size * 6):
            _, self.crop_margin = imgui.input_int(
                "##segment_crop_margin",
                self.crop_margin,
                step=16,
            )
            self.crop_margin = max(0, self.crop_margin)
        imgui.same_line()
        imgui.text('px')

        # Stride.
        imgui.text_colored('Stride', *viz.theme.dim)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Stride for tiling the slide.")
        self.stride, self._capturing_stride, _ = stride_capture(
            viz,
            self.stride,
            self._capturing_stride,
            max_value=16,
            label='Stride',
            draw_label=False,
            offset=viz.label_w,
            width=imgui.get_content_region_max()[0] - viz.label_w - (viz.spacing)
        )

        # Filter method.
        imgui.text_colored('Filter', *viz.theme.dim)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Method for filtering tiles.\n"
                "If 'otsu', tiles are filtered using Otsu's thresholding.\n"
                "If 'roi', only tiles touching an ROI are used."
            )
        imgui.same_line(viz.label_w)
        _, self._selected_filter_method = imgui.combo(
            "##segment_filter_method",
            self._selected_filter_method,
            self._filter_methods
        )

        imgui_utils.vertical_break()

    def draw_train_params(self) -> None:
        """Draw training architecture & hyperparameter options."""
        viz = self.viz

        # === Architecture & training parameters ===
        # Architecture.
        imgui.text_colored('Arch', *viz.theme.dim)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Model architecture.")
        imgui.same_line(viz.label_w)
        _, self._selected_arch = imgui.combo(
            "##segment_arch",
            self._selected_arch,
            self._supported_archs
        )
        # Encoder.
        imgui.text_colored('Encoder', *viz.theme.dim)
        imgui.same_line(viz.label_w)
        _, self._selected_encoder = imgui.combo(
            "##segment_encoder",
            self._selected_encoder,
            self._supported_encoders
        )
        # Training mode.
        imgui.text_colored('Mode', *viz.theme.dim)
        imgui.same_line(viz.label_w)
        _, self._selected_training_mode = imgui.combo(
            "##segment_training_mode",
            self._selected_training_mode,
            self._training_modes
        )
        # Max epochs.
        imgui.text_colored('Epochs', *viz.theme.dim)
        imgui.same_line(viz.label_w)
        _, self.max_epochs = imgui.input_int(
            "##segment_max_epochs",
            self.max_epochs,
            step=1,
            step_fast=5
        )

    def draw_training_button(self) -> None:
        """Draw the training button."""
        viz = self.viz

        # Train button.
        _button_text = "Train" if not self.is_training() else "Training" + imgui_utils.spinner_text()
        if viz.sidebar.full_button(_button_text, enabled=(viz.P is not None and not self.is_training())):
            ...
            self.train()

        if imgui.is_item_hovered() and viz.P is None:
            imgui.set_tooltip("No project loaded. Load a project to train a model.")

    def draw_generate_rois_button(self) -> None:
        """Show a button prompting the user to generate ROIs."""
        viz = self.viz
        if viz.sidebar.full_button(
            'Generate ROIs',
            enabled=(
                not self.is_thread_running()
                and (self._segment is not None)
                and (viz.wsi is not None)
                and not self.is_training()
            )
        ):
            self.generate_rois()


    def draw_config_popup(self):
        viz = self.viz
        has_model = self._segment is not None

        if self._show_popup:
            cx, cy = imgui.get_cursor_pos()
            imgui.set_next_window_position(viz.sidebar.full_width, cy)
            imgui.begin(
                '##segment_config_popup',
                flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
            )
            if imgui.menu_item('Load model', enabled=(not self.is_training()))[0]:
                self.ask_load_model()
                self._clicking = False
                self._show_popup = False
            if imgui.menu_item('Close model', enabled=has_model)[0]:
                self.close_model()
                self._clicking = False
                self._show_popup = False

            # Hide menu if we click elsewhere
            if imgui.is_mouse_down(LEFT_MOUSE_BUTTON) and not imgui.is_window_hovered():
                self._clicking = True
            if self._clicking and imgui.is_mouse_released(LEFT_MOUSE_BUTTON):
                self._clicking = False
                self._show_popup = False

            imgui.end()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            with viz.header_with_buttons("Tissue Segmentation"):
                imgui.same_line(imgui.get_content_region_max()[0] - viz.font_size*1.5)
                cx, cy = imgui.get_cursor_pos()
                imgui.set_cursor_position((cx, cy-int(viz.font_size*0.25)))
                if viz.sidebar.small_button('gear'):
                    self._clicking = False
                    self._show_popup = not self._show_popup
                self.draw_config_popup()

        if show and self._segment is None:
            imgui_utils.padded_text(
                'Load or train a model.',
                vpad=[int(viz.font_size/2),
                      int(viz.font_size)]
            )
            if viz.sidebar.full_button("Load a Model", enabled=(not self.is_training())):
                self.ask_load_model()
            if imgui.is_item_hovered() and self.is_training():
                imgui.set_tooltip("Cannot load model while training.")
            imgui_utils.vertical_break()

        elif show:
            if viz.collapsing_header('Model Info', default=True):
                self.draw_info()

        if show:
            if viz.collapsing_header('Training', default=False):

                if viz.collapsing_header2('Data Source', default=False):
                    self.draw_train_data_source()

                if viz.collapsing_header2('Data Processing', default=False):
                    self.draw_train_data_processing()

                if viz.collapsing_header2('Arch & Params', default=False):
                    self.draw_train_params()

                imgui_utils.vertical_break()
                self.draw_training_button()

            imgui_utils.vertical_break()
            self.draw_generate_rois_button()

        # Refresh ROIs if necessary.
        # Must be in the main loop.
        if self._need_to_refresh_rois:
            self._need_to_refresh_rois = False
            viz.slide_widget.roi_widget.refresh_rois()