
# Standard
# stylemix_widget.StyleMixingWidget

# Advanced
# trunc_noise_widget.TruncationNoiseWidget
# equivariance_widget.EquivarianceWidget

# Custom
# .widgets.seed_map.SeedMapWidget

import slideflow as sf
import imgui
import numpy as np
import json
from os.path import join, dirname, abspath, basename, exists
from tkinter.filedialog import askopenfilename

from slideflow.gan.stylegan3.stylegan3.viz.renderer import (
    Renderer, CapturedException
)
from ._utils import Widget
from ..gui import imgui_utils
from ..utils import EasyDict


class StyleGANWidget(Widget):

    tag = 'stylegan'
    description = 'StyleGAN'
    icon = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_stylegan.png')
    icon_highlighted = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_stylegan_highlighted.png')

    def __init__(self, viz):
        self.viz            = viz

        self.pkl            = None
        self.opt            = None
        self.sf_opt         = None
        self._clicking      = False
        self._show_popup    = False
        self._show_layers   = False
        self.viz.close_gan  = self.close_gan

        # Latent variable factors
        self.latent     = EasyDict(x=0, y=0)
        self.latent_def = EasyDict(self.latent)
        self.class_idx  = -1
        self.step_y     = 100
        self.viz.latent_widget = self

        # Style mixing factors
        self.seed_def   = 1000
        self.seed       = self.seed_def
        self.mix_class  = -1
        self.enables    = []
        self.mix_class  = -1
        self.mix_frac   = 0.5
        self.enable_mix_class   = False
        self.enable_mix_seed    = False

        viz.add_to_render_pipeline(Renderer(), name='stylegan')

    @property
    def mixing(self):
        return self.enable_mix_class or self.enable_mix_seed

    def close(self):
        self.close_gan()
        self.viz.remove_from_render_pipeline('stylegan')

    def close_gan(self):
        self.pkl = None
        self.viz.pkl = None
        renderer = self.viz.get_renderer('stylegan')
        if renderer:
            renderer.reset()
        self.viz._tex_img           = None
        self.viz._tex_obj           = None
        self.viz.clear_result()
        self.viz._async_renderer.get_result()
        self.viz._async_renderer.clear_result()
        self.viz.skip_frame()

    def load(self, pkl, ignore_errors=False) -> bool:
        viz = self.viz
        success = False
        viz.clear_result()
        if hasattr(viz, 'close_slide'):
            viz.close_slide(now=False)
        viz.skip_frame() # The input field will change on next frame.
        try:
            self.viz.pkl = self.pkl = pkl
            viz.result.message = f'Loading {basename(pkl)}...'
            viz.defer_rendering()
            self.viz._show_tile_preview = True

            # Load the tile_px/tile_um parameters from the training options, if present
            training_options = join(dirname(self.pkl), 'training_options.json')
            gan_px = 0
            gan_um = 0
            if exists(training_options):
                with open(training_options, 'r') as f:
                    self.opt = json.load(f)
                if 'slideflow_kwargs' in self.opt:
                    self.sf_opt = self.opt['slideflow_kwargs']
                    gan_px = self.sf_opt['tile_px']
                    gan_um = self.sf_opt['tile_um']
                else:
                    self.sf_opt = None

            if gan_px or gan_um:
                renderer = self.viz.get_renderer('stylegan')
                renderer.gan_px = gan_px
                renderer.gan_um = gan_um
                if hasattr(self.viz, 'create_toast'):
                    self.viz.create_toast(f"Loaded GAN pkl at {pkl} (tile_px={gan_px}, tile_um={gan_um})", icon="success")
            elif hasattr(self.viz, 'create_toast'):
                self.viz.create_toast(f"Loaded GAN pkl at {pkl}; unable to detect tile_px and tile_um", icon="warn")
            success = True
        except:
            success = False
            if pkl == '':
                viz.result = EasyDict(message='No network pickle loaded')
            else:
                viz.result = EasyDict(error=CapturedException())
            if not ignore_errors:
                raise
        try:
            import slideflow as sf
            self.viz._gan_config = sf.util.get_gan_config(pkl)
        except Exception:
            self.viz._gan_config = None
        self.viz._tex_obj = None
        return success

    # -------------------------------------------------------------------------

    def file_menu_options(self):
        if imgui.menu_item('Load GAN...')[1]:
            self.ask_load_gan()
        if imgui.menu_item('Close GAN')[1]:
            self.close_gan()

    def drag_and_drop_hook(self, path, ignore_errors=False) -> bool:
        if path.endswith('pkl'):
            return self.load(path, ignore_errors=ignore_errors)
        return False

    # -------------------------------------------------------------------------

    def drag_latent(self, dx, dy):
        viz = self.viz
        self.latent.x += dx / viz.font_size * 4e-2
        self.latent.y += dy / viz.font_size * 4e-2

    def set_seed(self, seed):
        self.latent.x = seed
        self.latent.y = 0

    def set_class(self, class_idx):
        self.class_idx = class_idx

    def set_render_args(self):
        viz = self.viz
        viz.args.pkl = self.pkl
        viz.args.w0_seeds = [] # [[seed, weight], ...]
        num_ws = viz.result.get('num_ws', 0)

        # Latent
        viz.args.class_idx = self.class_idx
        for ofs_x, ofs_y in [[0, 0], [1, 0], [0, 1], [1, 1]]:
            seed_x = np.floor(self.latent.x) + ofs_x
            seed_y = np.floor(self.latent.y) + ofs_y
            seed = (int(seed_x) + int(seed_y) * self.step_y) & ((1 << 32) - 1)
            weight = (1 - abs(self.latent.x - seed_x)) * (1 - abs(self.latent.y - seed_y))
            if weight > 0:
                viz.args.w0_seeds.append([seed, weight])

        # Style mixing
        viz.args.mix_frac = self.mix_frac if self.mixing else 0
        viz.args.mix_class = self.mix_class if self.enable_mix_class else -1
        if any(self.enables[:num_ws]):
            viz.args.stylemix_idx = [idx for idx, enable in enumerate(self.enables) if enable]
            if self.enable_mix_seed:
                viz.args.stylemix_seed = self.seed & ((1 << 32) - 1)

    # -------------------------------------------------------------------------

    def ask_load_gan(self):
        pkl = askopenfilename()
        if pkl:
            self.load(pkl, ignore_errors=True)

    def class_selection(self, name, value):
        viz = self.viz
        return_val = None
        with imgui_utils.item_width(viz.font_size * 5):
            _changed, _idx = imgui.input_int(name, value, step=1, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
            if _changed and self.sf_opt and _idx >= 0 and str(_idx) not in self.sf_opt['outcome_labels']:
                viz.create_toast(f'Invalid class index: {_idx}', icon='warn')
            elif _changed:
                return_val = _idx
            if self.sf_opt and value >= 0:
                imgui.same_line()
                _outcome_label = self.sf_opt['outcome_labels'][str(value)]
                imgui.text(_outcome_label)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(_outcome_label)
        return return_val

    def draw_config_popup(self):
        viz = self.viz
        has_model = viz._model_config is not None

        if self._show_popup:
            cx, cy = imgui.get_cursor_pos()
            imgui.set_next_window_position(viz.sidebar.full_width, cy)
            imgui.begin(
                '##gan_config_popup',
                flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
            )
            if imgui.menu_item('Load GAN')[0]:
                pkl = askopenfilename()
                if pkl:
                    self.load(pkl, ignore_errors=True)
            if imgui.menu_item('Close GAN')[0]:
                self.close_gan()

            # Hide menu if we click elsewhere
            if imgui.is_mouse_down(0) and not imgui.is_window_hovered():
                self._clicking = True
            if self._clicking and imgui.is_mouse_released(0):
                self._clicking = False
                self._show_popup = False

            imgui.end()

    def draw_info(self):
        viz = self.viz

        imgui.text_colored('Path', *viz.theme.dim)
        imgui.same_line(viz.font_size * 6)
        with imgui_utils.clipped_with_tooltip(self.pkl, 22):
            imgui.text(imgui_utils.ellipsis_clip(self.pkl, 22))

        imgui.text_colored('Tile (px)', *viz.theme.dim)
        imgui.same_line(viz.font_size * 6)
        imgui.text(str('-' if not self.sf_opt else self.sf_opt['tile_px']))

        imgui.text_colored('Tile (um)', *viz.theme.dim)
        imgui.same_line(viz.font_size * 6)
        imgui.text(str('-' if not self.sf_opt else self.sf_opt['tile_um']))

        if self.sf_opt and 'outcomes' in self.sf_opt:
            outcomes = self.sf_opt['outcomes']
        elif self.sf_opt and 'outcome_label_headers' in self.sf_opt:
            outcomes = self.sf_opt['outcome_label_headers']
        else:
            outcomes = '-'
        if isinstance(outcomes, list):
            outcomes = ', '.join(outcomes)
        imgui.text_colored('Outcome', *viz.theme.dim)
        imgui.same_line(viz.font_size * 6)
        imgui.text(outcomes)

    def draw_latent(self):
        viz = self.viz

        # Class selection
        imgui.text('Class')
        imgui.same_line(viz.label_w)
        _class_idx = self.class_selection('##class_idx', self.class_idx)
        if _class_idx is not None:
            self.class_idx = _class_idx

        # Seed selection
        imgui.text('Seed')
        imgui.same_line(viz.label_w)
        seed = round(self.latent.x) + round(self.latent.y) * self.step_y
        seed_width = imgui.get_content_region_max()[0] - viz.label_w - viz.font_size * 5 - viz.spacing
        with imgui_utils.item_width(seed_width):
            changed, seed = imgui.input_int('##seed', seed, step=0)
            if changed:
                self.set_seed(seed)
        imgui.same_line(viz.label_w + seed_width + viz.spacing)
        frac_x = self.latent.x - round(self.latent.x)
        frac_y = self.latent.y - round(self.latent.y)
        with imgui_utils.item_width(viz.font_size * 5):
            changed, (new_frac_x, new_frac_y) = imgui.input_float2('##frac', frac_x, frac_y, format='%+.2f', flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
            if changed:
                self.latent.x += new_frac_x - frac_x
                self.latent.y += new_frac_y - frac_y
        _clicked, dragging, dx, dy = imgui_utils.drag_button('Drag', width=viz.label_w - viz.spacing*2)
        if dragging:
            self.drag_latent(dx, dy)

        imgui.same_line()
        snapped = EasyDict(self.latent, x=round(self.latent.x), y=round(self.latent.y))
        if imgui_utils.button('Snap', width=seed_width, enabled=(self.latent != snapped)):
            self.latent = snapped
        imgui.same_line()
        if imgui_utils.button('Reset', width=-1, enabled=(self.latent != self.latent_def)):
            self.latent = EasyDict(self.latent_def)

        imgui_utils.vertical_break()

    def draw_style_mixing(self):
        viz = self.viz
        num_ws = viz.result.get('num_ws', 0)
        num_enables = viz.result.get('num_ws', 18)
        self.enables += [True] * max(num_enables - len(self.enables), 0)

        # Class mixing
        with imgui_utils.grayed_out(num_ws == 0):
            _clicked, self.enable_mix_class = imgui.checkbox('Mix class', self.enable_mix_class)

        imgui.same_line(viz.font_size * 6 + viz.spacing)
        with imgui_utils.grayed_out(not self.enable_mix_class):
            _mix_class = self.class_selection('##mix_class', self.mix_class)
            if _mix_class is not None:
                self.mix_class = _mix_class

        # Seed mixing
        with imgui_utils.grayed_out(num_ws == 0):
            _clicked, self.enable_mix_seed = imgui.checkbox('Mix seed', self.enable_mix_seed)

        imgui.same_line(viz.font_size * 6 + viz.spacing)
        with imgui_utils.item_width(viz.font_size * 3), imgui_utils.grayed_out(not self.enable_mix_seed):
            _changed, self.seed = imgui.input_int('Seed', self.seed, step=0)

        imgui.same_line(imgui.get_content_region_max()[0] - viz.font_size - viz.spacing*1.5)
        if viz.sidebar.small_button('ellipsis'):
            self._show_layers = not self._show_layers
        if imgui.is_item_hovered():
            imgui.set_tooltip("Set mixing layers")

        # Mixing slider
        with imgui_utils.item_width(-1), imgui_utils.grayed_out(num_ws == 0 or not self.mixing):
            _changed, self.mix_frac = imgui.slider_float('##mix_fraction',
                                                self.mix_frac,
                                                min_value=0,
                                                max_value=1,
                                                format='Mix %.2f')

        imgui_utils.vertical_break()

    def draw_layers_popup(self):
        viz = self.viz
        num_ws = viz.result.get('num_ws', 0)
        num_enables = viz.result.get('num_ws', 18)

        _, _expanded = imgui.begin('Mixing Layers', closable=True, flags=imgui.WINDOW_NO_RESIZE)
        if not _expanded:
            self._show_layers = False
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, [0, 0])
        for idx in range(num_enables):
            if idx:
                imgui.same_line()
            if idx == 0:
                imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 3)
            with imgui_utils.grayed_out(num_ws == 0):
                _clicked, self.enables[idx] = imgui.checkbox(f'##{idx}', self.enables[idx])
            if imgui.is_item_hovered():
                imgui.set_tooltip(f'{idx}')
        imgui.pop_style_var(1)
        imgui.same_line()
        imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() - 3)
        with imgui_utils.grayed_out(num_ws == 0):
            if imgui_utils.button('All', width=viz.button_w, enabled=(num_ws != 0)):
                self.seed = self.seed_def
                self.enables = [True] * num_enables
            imgui.same_line()
            if imgui_utils.button('None', width=viz.button_w, enabled=(num_ws != 0)):
                self.seed = self.seed_def
                self.enables = [False] * num_enables
        imgui.end()

    def draw_prediction(self):
        viz = self.viz

        if not viz._model_config:
            imgui_utils.padded_text('No model has been loaded.', vpad=[int(viz.font_size/2), int(viz.font_size)])
            if viz.sidebar.full_button("Load a Model"):
                viz.ask_load_model()
        else:
            viz.model_widget.draw_tile_predictions()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            with viz.header_with_buttons("StyleGAN"):
                imgui.same_line(imgui.get_content_region_max()[0] - viz.font_size*1.5)
                cx, cy = imgui.get_cursor_pos()
                imgui.set_cursor_position((cx, cy-int(viz.font_size*0.25)))
                if viz.sidebar.small_button('gear'):
                    self._clicking = False
                    self._show_popup = not self._show_popup
                self.draw_config_popup()

        if show and self.pkl:
            if viz.collapsing_header('Info', default=True):
                self.draw_info()

            if viz.collapsing_header('Latent', default=True):
                self.draw_latent()

            if viz.collapsing_header('Style Mixing', default=False):
                self.draw_style_mixing()

            if viz.collapsing_header('Prediction', default=True):
                self.draw_prediction()

            if self._show_layers:
                self.draw_layers_popup()

        elif show:
            imgui_utils.padded_text('No GAN has been loaded.', vpad=[int(viz.font_size/2), int(viz.font_size)])
            if viz.sidebar.full_button("Load a GAN"):
                self.ask_load_gan()

        self.set_render_args()