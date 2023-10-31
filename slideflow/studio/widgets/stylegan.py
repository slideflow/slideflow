
# Standard
# stylemix_widget.StyleMixingWidget

# Advanced
# trunc_noise_widget.TruncationNoiseWidget
# equivariance_widget.EquivarianceWidget

# Custom
# .widgets.seed_map.SeedMapWidget

import time
import slideflow as sf
import imgui
import numpy as np
import json
import glfw
import csv
import re
from os.path import join, dirname, abspath, basename, exists
from tkinter.filedialog import askopenfilename
from .model import draw_tile_predictions
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
        self._pressing_left = False
        self._pressing_right= False
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
        self.saved_seeds = []
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
        if self.viz._render_manager is not None:
            self.viz._render_manager.get_result()
            self.viz._render_manager.clear_result()
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
                    if 'resize' in self.sf_opt:
                        gan_px = self.sf_opt['resize']
                    else:
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

        # Skip if this is a non-conditioned GAN
        if self.sf_opt['outcome_labels'] is None:
            return

        with imgui_utils.item_width(viz.font_size * 6):
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
        imgui.text(str(outcomes))

    def draw_latent(self):
        viz = self.viz

        # --- Class selection -------------------------------------------------
        label_w = viz.label_w - (viz.font_size*1.5)
        imgui.text('Class')
        imgui.same_line(label_w)
        _class_idx = self.class_selection('##class_idx', self.class_idx)
        if _class_idx is not None:
            self.class_idx = _class_idx

        # --- Seed selection --------------------------------------------------
        imgui.text('Seed')
        imgui.same_line(label_w)

        # Base seed selection
        seed = round(self.latent.x) + round(self.latent.y) * self.step_y
        seed_width = viz.font_size * 6  #imgui.get_content_region_max()[0] - viz.label_w - viz.font_size * 4 - viz.spacing
        with imgui_utils.item_width(seed_width):
            changed, seed = imgui.input_int('##seed', seed, step=1)
            if changed:
                self.set_seed(seed)

        # Seed fraction selection
        imgui.same_line(label_w + seed_width + viz.spacing)
        frac_x = self.latent.x - round(self.latent.x)
        frac_y = self.latent.y - round(self.latent.y)
        with imgui_utils.item_width(-1):
            changed, (new_frac_x, new_frac_y) = imgui.input_float2('##frac', frac_x, frac_y, format='%+.2f', flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
            if changed:
                self.latent.x += new_frac_x - frac_x
                self.latent.y += new_frac_y - frac_y


        # --- Buttons ---------------------------------------------------------
        button_w = (imgui.get_content_region_max()[0] - viz.spacing*4) / 4
        # Drag
        _clicked, dragging, dx, dy = imgui_utils.drag_button('Drag', width=button_w)
        if dragging:
            self.drag_latent(dx, dy)

        # Snap
        imgui.same_line()
        snapped = EasyDict(self.latent, x=round(self.latent.x), y=round(self.latent.y))
        if imgui_utils.button('Snap', width=button_w, enabled=(self.latent != snapped)):
            self.latent = snapped

        # Save
        imgui.same_line()
        if imgui_utils.button('Save', width=button_w):
            self.saved_seeds.append((self.latent.x, self.latent.y))

        # Reset
        imgui.same_line()
        if imgui_utils.button('Reset', width=button_w, enabled=(self.latent != self.latent_def)):
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

        #TODO: Hacky workaround, fix later
        if hasattr(viz, 'mil_widget') and viz.mil_widget.model is not None:
            draw_tile_predictions(
                viz,
                is_categorical=viz.mil_widget.is_categorical(),
                config=viz.mil_widget.mil_params,
                has_preds=(viz._predictions is not None),
                using_model=viz.mil_widget.model_loaded,
                uncertainty_color=viz.mil_widget.uncertainty_color,
                uncertainty_range=viz.mil_widget.uncertainty_range,
                uncertainty_label="Attention",
            )
        elif not viz._model_config:
            imgui_utils.padded_text('No model has been loaded.', vpad=[int(viz.font_size/2), int(viz.font_size)])
            if viz.sidebar.full_button("Load a Model"):
                viz.ask_load_model()
        else:
            draw_tile_predictions(viz, viz.model_widget.is_categorical())

        imgui_utils.vertical_break()

    def draw_saved_seeds(self):
        """Draw the saved seeds list box.

        The list box displays the saved seeds in the (seed, frac_x, frac_y) format.
        The seeds are saved in the (x, y) format.

        """
        viz = self.viz

        if not self.saved_seeds:
            imgui.text("No seeds have been saved.")
            imgui_utils.vertical_break()
            if viz.sidebar.full_button("Load Seeds"):
                self.load_seeds()
            return

        selected_idx = None
        with imgui.begin_list_box("##saved_seeds", -1, viz.font_size*10) as list_box:
            if list_box.opened:
                for idx, (x, y) in enumerate(self.saved_seeds):
                    seed = round(x) + round(y) * self.step_y
                    frac_x = x - round(x)
                    frac_y = y - round(y)
                    seed_str = f'{seed} ({frac_x:+.2f}, {frac_y:+.2f})'
                    selected = (self.latent.x == x) and (self.latent.y == y)
                    if selected:
                        selected_idx = idx
                    if imgui.selectable(seed_str, selected)[0]:
                        self.latent.x = x
                        self.latent.y = y

        button_w = (imgui.get_content_region_max()[0] - viz.spacing*4) / 4
        if imgui_utils.button('Export##export_seeds', width=button_w):
            self.export_seeds()
        imgui.same_line()
        if imgui_utils.button('Load##load_seeds', width=button_w):
            self.load_seeds()
        imgui.same_line()
        if imgui_utils.button('Remove##remove_seeds', width=button_w, enabled=(selected_idx is not None)):
            if selected_idx is not None:
                del self.saved_seeds[selected_idx]
        imgui.same_line()
        if imgui_utils.button('Reset##reset_button', width=button_w):
            self.saved_seeds = []

    def load_seeds(self):
        """Load seeds from CSV.

        Seeds should be in the (x, y) format, one seed per line.

        This differs from the display format, which is (seed, frac_x, frac_y).

        """
        path = askopenfilename(title="Load Seeds...", filetypes=[("CSV", "*.csv",)])
        if path:
            with open(path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                self.saved_seeds = []
                for row in reader:
                    self.saved_seeds.append((float(row[0]), float(row[1])))
            self.viz.create_toast(f'Loaded seeds from {path}', icon='success')

    def export_seeds(self):
        """Export seeds to CSV.

        Seeds are saved in (x, y) format.
        """

        # Find filenames in the current directory matching "saved_seeds_00000.csv"
        # and increment the number until we find a filename that doesn't exist.
        # Then save the seeds to that file.
        path = join(dirname(self.pkl), 'saved_seeds_00000.csv')
        while exists(path):
            match = re.match(r'saved_seeds_(\d{5}).csv', basename(path))
            if match:
                num = int(match.group(1))
                path = join(dirname(self.pkl), f'saved_seeds_{num+1:05}.csv')
            else:
                path = join(dirname(self.pkl), 'saved_seeds_00001.csv')

        # Save the seeds in CSV format to path.
        # All seeds are saved in (x, y) format, stored in the variable self.saved_seeds
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for x, y in self.saved_seeds:
                writer.writerow([x, y])

        self.viz.create_toast(f'Saved seeds to {path}', icon='success')

    # -------------------------------------------------------------------------

    def keyboard_callback(self, key, action):
        if action == glfw.PRESS and key == glfw.KEY_RIGHT:
            self.set_seed(round(self.latent.x) + round(self.latent.y) * self.step_y + 1)
            if not self._pressing_left:
                self._start_press = time.time()
                self._pressing_right = True
        if action == glfw.PRESS and key == glfw.KEY_LEFT:
            self.set_seed(round(self.latent.x) + round(self.latent.y) * self.step_y - 1)
            if not self._pressing_right:
                self._start_press = time.time()
                self._pressing_left = True
        if action == glfw.PRESS and key == glfw.KEY_S:
            self.saved_seeds.append((self.latent.x, self.latent.y))
        if action == glfw.RELEASE and key == glfw.KEY_RIGHT:
            self._pressing_right = False
        if action == glfw.RELEASE and key == glfw.KEY_LEFT:
            self._pressing_left = False

    def draw_saved_seeds(self):
        """Draw the saved seeds list box.

        The list box displays the saved seeds in the (seed, frac_x, frac_y) format.
        The seeds are saved in the (x, y) format.

        """
        viz = self.viz

        if not self.saved_seeds:
            imgui.text("No seeds have been saved.")
            imgui_utils.vertical_break()
            if viz.sidebar.full_button("Load Seeds"):
                self.load_seeds()
            return

        selected_idx = None
        with imgui.begin_list_box("##saved_seeds", -1, viz.font_size*10) as list_box:
            if list_box.opened:
                for idx, (x, y) in enumerate(self.saved_seeds):
                    seed = round(x) + round(y) * self.step_y
                    frac_x = x - round(x)
                    frac_y = y - round(y)
                    seed_str = f'{seed} ({frac_x:+.2f}, {frac_y:+.2f})'
                    selected = (self.latent.x == x) and (self.latent.y == y)
                    if selected:
                        selected_idx = idx
                    if imgui.selectable(seed_str, selected)[0]:
                        self.latent.x = x
                        self.latent.y = y

        button_w = (imgui.get_content_region_max()[0] - viz.spacing*4) / 4
        if imgui_utils.button('Export##export_seeds', width=button_w):
            self.export_seeds()
        imgui.same_line()
        if imgui_utils.button('Load##load_seeds', width=button_w):
            self.load_seeds()
        imgui.same_line()
        if imgui_utils.button('Remove##remove_seeds', width=button_w, enabled=(selected_idx is not None)):
            if selected_idx is not None:
                del self.saved_seeds[selected_idx]
        imgui.same_line()
        if imgui_utils.button('Reset##reset_button', width=button_w):
            self.saved_seeds = []

    def load_seeds(self):
        """Load seeds from CSV.

        Seeds should be in the (x, y) format, one seed per line.

        This differs from the display format, which is (seed, frac_x, frac_y).

        """
        path = askopenfilename(title="Load Seeds...", filetypes=[("CSV", "*.csv",)])
        if path:
            with open(path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                self.saved_seeds = []
                for row in reader:
                    self.saved_seeds.append((float(row[0]), float(row[1])))
            self.viz.create_toast(f'Loaded seeds from {path}', icon='success')

    def export_seeds(self):
        """Export seeds to CSV.

        Seeds are saved in (x, y) format.
        """

        # Find filenames in the current directory matching "saved_seeds_00000.csv"
        # and increment the number until we find a filename that doesn't exist.
        # Then save the seeds to that file.
        path = join(dirname(self.pkl), 'saved_seeds_00000.csv')
        while exists(path):
            match = re.match(r'saved_seeds_(\d{5}).csv', basename(path))
            if match:
                num = int(match.group(1))
                path = join(dirname(self.pkl), f'saved_seeds_{num+1:05}.csv')
            else:
                path = join(dirname(self.pkl), 'saved_seeds_00001.csv')

        # Save the seeds in CSV format to path.
        # All seeds are saved in (x, y) format, stored in the variable self.saved_seeds
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for x, y in self.saved_seeds:
                writer.writerow([x, y])

        self.viz.create_toast(f'Saved seeds to {path}', icon='success')

    # -------------------------------------------------------------------------

    def fast_scroll(self):
        if self._pressing_left and time.time() - self._start_press > 0.5:
            self.set_seed(round(self.latent.x) + round(self.latent.y) * self.step_y - 1)
        if self._pressing_right and time.time() - self._start_press > 0.5:
            self.set_seed(round(self.latent.x) + round(self.latent.y) * self.step_y + 1)

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

            if viz.collapsing_header('Saved Seeds', default=True):
                self.draw_saved_seeds()

            if self._show_layers:
                self.draw_layers_popup()

        elif show:
            imgui_utils.padded_text('No GAN has been loaded.', vpad=[int(viz.font_size/2), int(viz.font_size)])
            if viz.sidebar.full_button("Load a GAN"):
                self.ask_load_gan()

        self.fast_scroll()
        self.set_render_args()