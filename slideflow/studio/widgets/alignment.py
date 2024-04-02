import os
import cv2
import numpy as np
import glfw
import imgui
import slideflow as sf

from collections import defaultdict
from tkinter.filedialog import askopenfilename
from os.path import join, dirname, abspath, exists
from slideflow.slide import Alignment, best_fit_plane, z_on_plane
from slideflow.util import path_to_name
from ..gui.viewer import SlideViewer
from ..gui import imgui_utils, gl_utils

#----------------------------------------------------------------------------

class AlignmentWidget:

    tag = 'alignment'
    description = 'Whole-slide alignment'
    icon = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_mil.png')
    icon_highlighted = join(dirname(abspath(__file__)), '..', 'gui', 'buttons', 'button_mil_highlighted.png')

    def __init__(self, viz):
        self.viz            = viz
        self.rotations      = ["None", "90 (clockwise)", "180", "270 (clockwise)"]
        self.rot_int        = [
            None,
            sf.slide.ROTATE_90_CLOCKWISE,
            sf.slide.ROTATE_180_CLOCKWISE,
            sf.slide.ROTATE_270_CLOCKWISE
        ]
        self._rotation_idx  = 0
        self._reference_path = ''
        self._source_path = ''
        self._output_path = ''
        self._source_slide_paths = []
        self._source_slide_names = []
        self._reference_slide_paths = []
        self._reference_slide_names = []
        self._alignment_paths = []
        self.ref_viewer = None
        self.ref_wsi = None
        self.ref_preview_width = 800
        self.ref_preview_height = 600
        self.ref_x = None
        self.ref_y = None
        self.ref_tile = None
        self.aligned_coords = []
        self._ref_tile_tex = None
        self._last_aligned_coords = None
        self._last_mouse_drag_pos = None
        self._selected_coords = []
        self._last_wsi_tile = None
        self._last_ref_tile = None
        self._last_wsi = None
        self._last_project = None
        self._slide_to_patient = None
        self._patient_to_slides = None
        self._matching_patients = []
        self.mse = None
        self.batched_reference_load = False

        # Stain normalizer for alignment.
        self.normalizer = sf.norm.autoselect('reinhard_mask', backend='opencv')

        # Enable tile preview
        self.viz._force_enable_tile_preview = True

    def patient_is_known(self, slide):
        if self.viz.P is not None and (self._slide_to_patient is None or self._last_project is not self.viz.P):
            self._update_patient_dicts()
        return self._slide_to_patient is not None and slide in self._slide_to_patient

    def slide_to_patient(self, slide):
        if self._slide_to_patient is None or self._last_project is not self.viz.P:
            self._update_patient_dicts()
        return self._slide_to_patient[slide]

    def patient_to_slides(self, patient):
        if self._patient_to_slides is None or self._last_project is not self.viz.P:
            self._update_patient_dicts()
        return self._patient_to_slides[patient]

    def _update_patient_dicts(self):
        self._slide_to_patient = self.viz.P.dataset().patients()
        self._patient_to_slides = defaultdict(list)
        for slide, patient in self._slide_to_patient.items():
            self._patient_to_slides[patient].append(slide)
        self._last_project = self.viz.P

    def _on_slide_load(self):
        """Triggered when a slide is loaded."""
        viz = self.viz
        if self._rotation_idx != 0:
            rot = self.rot_int[self._rotation_idx]
            self.viz.reload_wsi(transforms=[rot])
        self.aligned_coords = []

    def both_tiles_shown(self):
        return self.viz._tex_img is not None and self.ref_tile is not None

    def keyboard_callback(self, key: int, action: int) -> None:
        """Handle keyboard events.

        Args:
            key (int): The key that was pressed. See ``glfw.KEY_*``.
            action (int): The action that was performed (e.g. ``glfw.PRESS``,
                ``glfw.RELEASE``, ``glfw.REPEAT``).

        """
        if self.viz.wsi is not None:
            dx = int(np.round(self.viz.wsi.full_extract_px / self.viz.wsi.tile_px))
        else:
            dx = 1
        if (key == glfw.KEY_RIGHT and action in (glfw.PRESS, glfw.REPEAT)):
            if self.viz.x is not None:
                self.viz.x += dx
        elif (key == glfw.KEY_LEFT and action in (glfw.PRESS, glfw.REPEAT)):
            if self.viz.x is not None:
                self.viz.x -= dx
        elif (key == glfw.KEY_UP and action in (glfw.PRESS, glfw.REPEAT)) and not self.viz._control_down:
            if self.viz.y is not None:
                self.viz.y -= dx
        elif (key == glfw.KEY_DOWN and action in (glfw.PRESS, glfw.REPEAT)) and not self.viz._control_down:
            if self.viz.y is not None:
                self.viz.y += dx
        elif key == glfw.KEY_SPACE and action == glfw.PRESS and self.both_tiles_shown() and not self.viz._control_down:
            self.align_tile()
        elif key == glfw.KEY_DOWN and action == glfw.PRESS and self.viz._control_down:
            self.open_next_patient()
        elif key == glfw.KEY_UP and action == glfw.PRESS and self.viz._control_down:
            self.open_previous_patient()
        elif key == glfw.KEY_ENTER and action == glfw.PRESS and self.viz._control_down:
            self.export_alignment()

    def open_next_patient(self) -> None:
        """Open the next patient in the list of matching patients."""
        if self._matching_patients and self.viz.wsi is not None:
            current_patient = self.slide_to_patient(self.viz.wsi.name)
            current_index = self._matching_patients.index(current_patient)
            next_index = (current_index + 1) % len(self._matching_patients)
            next_patient = self._matching_patients[next_index]
            self.load_patient(next_patient)

    def open_previous_patient(self) -> None:
        """Open the previous patient in the list of matching patients."""
        if self._matching_patients and self.viz.wsi is not None:
            current_patient = self.slide_to_patient(self.viz.wsi.name)
            current_index = self._matching_patients.index(current_patient)
            next_index = (current_index - 1) % len(self._matching_patients)
            next_patient = self._matching_patients[next_index]
            self.load_patient(next_patient)

    def load_patient(self, patient) -> None:
        """Load the given patient."""
        source_slide_path = [sp for sp in self._source_slide_paths if self.patient_is_known(path_to_name(sp)) and self.slide_to_patient(path_to_name(sp)) == patient][0]
        ref_slide_path = [sp for sp in self._reference_slide_paths if self.patient_is_known(path_to_name(sp)) and self.slide_to_patient(path_to_name(sp)) == patient][0]
        self.viz.load_slide(source_slide_path)
        self.load_reference_slide(ref_slide_path)

    def _update_reference_texture(self):
        """Update the reference slide texture with the whole-slide view."""
        if self.ref_viewer._tex_img is not self.ref_viewer.view:
            self.ref_viewer._update_texture()

    def _draw_reference_slide(self):
        """Draw the reference slide in the alignment widget."""
        imgui.image(
            self.ref_viewer._tex_obj.gl_id,
            self.ref_viewer.view.shape[1],
            self.ref_viewer.view.shape[0],
        )
        if imgui.is_item_hovered():
            self.viz.suspend_mouse_input_handling()
        else:
            self.viz.resume_mouse_input_handling()

    def set_reference_tile_img(self, img):
        """Set the reference tile image and update the texture."""
        self.ref_tile = img
        if img is None:
            self._ref_tile_tex = None
            return
        if self._ref_tile_tex is None or not self._ref_tile_tex.is_compatible(image=img):
            if self._ref_tile_tex is not None:
                self.viz._tex_to_delete += [self._ref_tile_tex]
            self._ref_tile_tex = gl_utils.Texture(image=img, bilinear=False, mipmap=False)  # type: ignore
        else:
            self._ref_tile_tex.update(img)

    def set_reference_tile_coords(self, x, y):
        """Set the reference tile coordinates and update the reference tile image."""
        self.ref_x, self.ref_y = x, y
        ref_tile = self.ref_viewer.read_tile(self.ref_x, self.ref_y, allow_errors=True)
        self.set_reference_tile_img(ref_tile)

    def mark_aligned(self):
        """Mark the currently loaded tile as aligned to the reference tile."""
        base_x, base_y = self.viz.wsi.slide.coord_to_raw(
            self.viz.x + (self.viz.wsi.full_extract_px // 2),
            self.viz.y + (self.viz.wsi.full_extract_px // 2)
        )
        self.aligned_coords.append(np.array([
            base_x,
            base_y,
            self.ref_x + (self.ref_wsi.full_extract_px // 2),
            self.ref_y + (self.ref_wsi.full_extract_px // 2)
        ]).astype(np.int))

    def get_aligned_coords(self, ref_x, ref_y):
        """Convert reference coordinates to aligned coordinates."""
        viz = self.viz
        alignment = viz.wsi.alignment

        # First, scale the reference coordinates to the loaded slide's MPP.
        ref_x = ref_x * (self.ref_wsi.mpp / viz.wsi.mpp)
        ref_y = ref_y * (self.ref_wsi.mpp / viz.wsi.mpp)

        # Then offset the reference coordinates by the origin of the alignment.
        origin = viz.wsi.slide.raw_to_coord(*alignment.origin)
        x = ref_x + origin[0]
        y = ref_y + origin[1]

        # If a plane of best fit was calculated, adjust the coordinates to be on that plane.
        if alignment.centroid is not None:
            x_centroid, y_centroid = alignment.centroid
            x_normal, y_normal = alignment.normal
            half_extract_px = int(np.round(viz.wsi.full_extract_px / 2))
            bx, by = viz.wsi.slide.coord_to_raw(x + half_extract_px, y + half_extract_px)
            adjust_x = int(np.round(z_on_plane(bx, by, x_centroid, x_normal)))
            adjust_y = int(np.round(z_on_plane(bx, by, y_centroid, y_normal)))
            x, y = viz.wsi.slide.raw_to_coord(bx + adjust_x, by + adjust_y)
            x -= half_extract_px
            y -= half_extract_px
        return x, y

    def set_viewer_tile_coords(self, x, y):
        """Update the viewer tile coordinates."""
        self.viz.x = x
        self.viz.y = y
        self.viz.box_x, self.viz.box_y = self.viz.viewer.wsi_coords_to_display_coords(x, y)

    def set_tile_to_reference(self):
        """Set the loaded slide tile to the reference slide tile."""
        x, y = self.get_aligned_coords(self.ref_x, self.ref_y)
        if self._last_aligned_coords != (x, y):
            self._last_aligned_coords = (x, y)
            self.set_viewer_tile_coords(x, y)

    def calculate_alignment(self, image, reference):
        """Calculate the alignment between two images."""
        if self.normalizer is not None:
            image = self.normalizer.transform(image[:, :, 0:3])
            reference = self.normalizer.transform(reference[:, :, 0:3])
        try:
            rough_alignment = sf.slide.utils._find_translation_matrix(reference, image, h=50, search_window=53)
        except cv2.error:
            sf.log.debug("Rough alignment failed.")
            rough_alignment = None
        try:
            return sf.slide.utils.align_by_translation(reference, image, round=True, warp_matrix=rough_alignment)
        except sf.errors.AlignmentError as e:
            self.viz.create_toast('Error aligning tiles.', icon='error')
            sf.log.error("Error aligning tiles: {}".format(e))
            return None

    def calculate_alignment_mse(self):
        """Calculate the MSE between the loaded tile and the reference tile."""
        if self.viz._tex_img is None or self.ref_tile is None:
            return None
        wsi_tile_gray = cv2.cvtColor(self.viz._tex_img, cv2.COLOR_BGR2GRAY)
        ref_tile_gray = cv2.cvtColor(self.ref_tile, cv2.COLOR_BGR2GRAY)
        return sf.slide.utils.compute_alignment_mse(
            wsi_tile_gray,
            ref_tile_gray,
            flatten=False
        )

    def align_tile(self):
        """Align the currently loaded tile view to the view from the reference slide."""
        viz = self.viz
        if not self.both_tiles_shown() or (viz.x is None or viz.y is None):
            viz.create_toast('Need to tiles to align.', icon='error')
            return


        # Calculate alignment between the tile images
        us = self.viz._tex_img
        them = self.ref_tile
        alignment = self.calculate_alignment(us, them)
        if alignment is None:
            return
        pixel_ratio = (viz.wsi.full_extract_px / viz.wsi.tile_px)
        x_adjust = int(np.round(alignment[0] * pixel_ratio))
        y_adjust = int(np.round(alignment[1] * pixel_ratio))
        print("Tile alignment complete. Adjustment: x={}, y={}".format(x_adjust, y_adjust))
        self.set_viewer_tile_coords(viz.x + x_adjust, viz.y + y_adjust)
        self.mark_aligned()
        self.align_slide()

    def align_slide(self):
        """Align the loaded slide to the reference slide."""
        viz = self.viz

        # Assemble coordinates of the reference pairs.
        aligned_coords = np.stack(self.aligned_coords, axis=0)
        loaded_slide_raw = aligned_coords[:, 0:2]
        cx, cy = viz.wsi.slide.raw_to_coord(loaded_slide_raw[:, 0], loaded_slide_raw[:, 1])
        loaded_slide_coords = np.column_stack((cx, cy))
        ref_coords = aligned_coords[:, 2:4]
        # Scale the reference slide coordinates to be in the same MPP scale as the loaded slide.
        ref_coords = ref_coords * (self.ref_wsi.mpp / viz.wsi.mpp)

        # Start by setting the origin, using the first aligned pair
        origin = np.mean(loaded_slide_coords - ref_coords, axis=0)

        if len(aligned_coords) < 3:
            print("Aligning slide from single tile (origin translation).")
            alignment = Alignment.from_translation(
                origin=viz.wsi.slide.coord_to_raw(*origin),
                scale=(self.ref_wsi.mpp / viz.wsi.mpp)
            )
            viz.wsi.apply_alignment(alignment)
        else:
            print("Aligning slide with {} aligned pairs, using plane of best fit.".format(len(aligned_coords)))
            # Now, calculate the plane of best fit for the remaining aligned pairs.

            # First, calculate the offset between the aligned pairs.
            # This is the distance the loaded slide needs to move to be aligned to the reference.
            origin_offset_coords = loaded_slide_coords - origin
            reference_offset_coords = origin_offset_coords - ref_coords

            # Next, translate our slide's loaded coordinates into base coordinates
            # (pre-rotation, pre-offset).
            x_base, y_base = viz.wsi.slide.coord_to_raw(origin_offset_coords[:, 0], origin_offset_coords[:, 1])
            x_base_adj, y_base_adj = viz.wsi.slide.coord_to_raw(
                origin_offset_coords[:, 0] + reference_offset_coords[:, 0],
                origin_offset_coords[:, 1] + reference_offset_coords[:, 1]
            )
            x_base_adjustment = x_base_adj - x_base
            y_base_adjustment = y_base_adj - y_base

            coord_raw = viz.wsi.slide.coord_to_raw(
                loaded_slide_coords[:, 0],
                loaded_slide_coords[:, 1]
            )

            x_adjust_coords = np.column_stack((
                coord_raw[0],
                coord_raw[1],
                x_base_adjustment
            ))
            y_adjust_coords = np.column_stack((
                coord_raw[0],
                coord_raw[1],
                y_base_adjustment
            ))
            x_centroid, x_normal = best_fit_plane(x_adjust_coords)
            y_centroid, y_normal = best_fit_plane(y_adjust_coords)
            alignment = Alignment.from_fit(
                origin=viz.wsi.slide.coord_to_raw(*origin),
                scale=(self.ref_wsi.mpp / viz.wsi.mpp),
                centroid=(x_centroid, y_centroid),
                normal=(x_normal, y_normal)
            )
            viz.wsi.apply_alignment(alignment)

        viz.create_toast('Slide aligned', icon='success')

    def _reference_coords_to_draw_coords(self, x, y, window_x, window_y):
        viz = self.viz
        window_x = window_x + viz.spacing
        window_y = window_y + viz.spacing

        t_w_ratio = self.ref_wsi.dimensions[0] / self.ref_viewer.view.shape[1]
        t_h_ratio = self.ref_wsi.dimensions[1] / self.ref_viewer.view.shape[0]

        rel_x = x / t_w_ratio
        rel_y = y / t_h_ratio

        return window_x + rel_x, window_y + rel_y

    def export_alignment(self, allow_errors=False):
        """Export the alignment to a file."""
        if self.viz.wsi is None or self.viz.wsi.alignment is None:
            if allow_errors:
                return
            else:
                self.viz.create_toast('No alignment to save.', icon='error')
                return
        dest_name = f'{self.viz.wsi.name}_{self.ref_wsi.name}_alignment.npz'
        dest = join(self._output_path, dest_name)
        self.viz.wsi.alignment.save(dest)
        self._alignment_paths = os.listdir(self._output_path)
        self.viz.create_toast(f'Alignment saved to {dest}', icon='success')

    def update_matching_patients(self):
        """From the given reference and source directory, find slides that belong to matching patients."""
        viz = self.viz
        if viz.P is None:
            return
        if not os.path.isdir(self._source_path):
            return
        if not os.path.isdir(self._reference_path):
            return
        src_patients = []
        ref_patients = []
        for f in self._source_slide_paths:
            slide = path_to_name(f)
            if self.patient_is_known(slide):
                src_patients.append(self.slide_to_patient(slide))
        for f in self._reference_slide_paths:
            slide = path_to_name(f)
            if self.patient_is_known(slide):
                ref_patients.append(self.slide_to_patient(slide))
        matching_patients = set(src_patients).intersection(set(ref_patients))
        self._matching_patients = sorted(list(matching_patients))

    def load_reference_slide(self, path):
        """Load the reference slide."""
        viz = self.viz
        try:
            self.ref_wsi = viz._reload_and_return_wsi(path)
        except Exception as e:
            viz.create_toast(f'Error loading slide at {path}', icon='error')
            sf.log.error('Error loading reference slide: {}'.format(e))

        else:
            self.ref_viewer = SlideViewer(self.ref_wsi, self.ref_preview_width, self.ref_preview_height)
            self.ref_x = None
            self.ref_y = None

    # === GUI ===

    def draw_reference_popup(self):
        """Draw the reference slide popup.

        Right-clicking on the reference slide will set the reference tile.

        """
        viz = self.viz
        if self.ref_viewer is not None:

            # Show reference slide.
            self._update_reference_texture()
            vs = viz.spacing * 2
            imgui.set_next_window_size(self.ref_viewer.view.shape[1] + vs, self.ref_viewer.view.shape[0] + vs)
            imgui.begin("Reference slide##popup_view", flags=(imgui.WINDOW_NO_TITLE_BAR))
            self._draw_reference_slide()

            window_x, window_y = imgui.get_window_position()

            # Handle right-click on the reference slide.
            if imgui.is_mouse_down(1) and imgui.is_item_hovered():
                mx, my = imgui.get_mouse_pos()

                # Show location overlay
                t_xo, t_yo = window_x, window_y
                t_xo = t_xo + viz.spacing
                t_yo = t_yo + viz.spacing
                rel_x = mx - t_xo
                rel_y = my - t_yo

                t_w_ratio = self.ref_wsi.dimensions[0] / self.ref_viewer.view.shape[1]
                t_h_ratio = self.ref_wsi.dimensions[1] / self.ref_viewer.view.shape[0]

                ref_x = int(rel_x * t_w_ratio) - (self.ref_wsi.full_extract_px // 2)
                ref_y = int(rel_y * t_h_ratio) - (self.ref_wsi.full_extract_px // 2)
                self.set_reference_tile_coords(ref_x, ref_y)

                if self.viz.wsi.alignment is not None:
                    self.set_tile_to_reference()

            # Draw a box at the current reference tile in view.
            draw_list = imgui.get_window_draw_list()
            if self.ref_x is not None:
                box_origin = self._reference_coords_to_draw_coords(self.ref_x, self.ref_y, window_x, window_y)
                box_end = self._reference_coords_to_draw_coords(
                    self.ref_x + self.ref_wsi.full_extract_px,
                    self.ref_y + self.ref_wsi.full_extract_px,
                    window_x,
                    window_y
                )
                draw_list.add_rect(*box_origin, *box_end, imgui.get_color_u32_rgba(0, 0, 0, 1), thickness=2)

            # Draw filled circles at the aligned pairs.
            for i, coord in enumerate(self.aligned_coords):
                x, y = self._reference_coords_to_draw_coords(coord[2], coord[3], window_x, window_y)
                draw_list.add_circle_filled(x, y, 5, imgui.get_color_u32_rgba(0, 0.88, 0, 1))
                draw_list.add_circle(x, y, 5, imgui.get_color_u32_rgba(0, 0, 0, 1))

            imgui.end()
        else:
            viz.resume_mouse_input_handling()

    def draw_reference_and_data(self):
        """Draw the reference slide and data loading controls."""
        viz = self.viz
        if imgui.radio_button('Single slide', not self.batched_reference_load):
            self.batched_reference_load = False
        imgui.same_line()
        if imgui.radio_button('Batch align', self.batched_reference_load):
            self.batched_reference_load = True

        # Reference path (slide or directory).
        imgui.text('Reference')
        if imgui.is_item_hovered():
            imgui.set_tooltip("Path to the reference slide(s).")
        imgui.same_line(viz.label_w)
        _ref_changed, self._reference_path = imgui.input_text(
            f"##reference_path",
            self._reference_path
        )

        if not self.batched_reference_load:
            if _ref_changed:
                self.load_reference_slide(self._reference_path)
        else:
            imgui.text('Source')
            if imgui.is_item_hovered():
                imgui.set_tooltip("Path to the directory containing reference slides.")
            imgui.same_line(viz.label_w)
            _src_changed, self._source_path = imgui.input_text(
                f"##source_path",
                self._source_path
            )
            if _src_changed:
                self._source_slide_paths = sf.util.get_slide_paths(self._source_path)
                self._source_slide_names = [path_to_name(sp) for sp in self._source_slide_paths]
            if _ref_changed:
                self._reference_slide_paths = sf.util.get_slide_paths(self._reference_path)
                self._reference_slide_names = [path_to_name(sp) for sp in self._reference_slide_paths]
            if _ref_changed or _src_changed:
                if viz.P is not None:
                    self.update_matching_patients()
                else:
                    viz.create_toast('Project must be loaded for batch aligning.', icon='error')
                if os.path.exists(self._output_path):
                    print("output path exists", self._output_path)
                    self._alignment_paths = os.listdir(self._output_path)
                else:
                    print("output path does NOT exist", self._output_path)
                    self._alignment_paths = []

        # Output directory configuration for saving alignments.
        imgui.text('Output')
        if imgui.is_item_hovered():
            imgui.set_tooltip("Directory to save alignment files.")
        imgui.same_line(viz.label_w)
        _out_changed, self._output_path = imgui.input_text(
            f"##output_path",
            self._output_path
        )

        # Show matching patients.
        with imgui.begin_list_box("##matching_patients", -1, viz.font_size * 5) as list_box:
            if list_box.opened:
                for i, patient in enumerate(self._matching_patients):
                    _selected = viz.wsi is not None and self.slide_to_patient(viz.wsi.name) == patient
                    patient_display = patient 
                    matching_slides = self.patient_to_slides(patient)
                    wsi_name = [name for name in matching_slides if name in self._source_slide_names][0]
                    ref_name = [name for name in matching_slides if name in self._reference_slide_names][0]
                    if self._output_path is not None and f'{wsi_name}_{ref_name}_alignment.npz' in self._alignment_paths:
                        patient_display += ' (aligned)'
                    _clicked, _sel = imgui.selectable(patient_display, selected=_selected)
                    if _clicked:
                        self.load_patient(patient)

        imgui_utils.vertical_break()

    def ask_load_alignment(self):
        """Prompt the user to load an alignment file."""
        path = askopenfilename(title="Load alignment...", filetypes=[("npz", ".npz"), ("All files", ".*")])
        if path:
            try:
                alignment = Alignment.load(path)
                self.viz.wsi.apply_alignment(alignment)
            except Exception as e:
                self.viz.create_toast(f'Error loading alignment at {path}', icon='error')
                sf.log.error('Error loading alignment: {}'.format(e))

    def draw_tile_images(self):
        """Draw the loaded tile image and the reference tile image."""
        viz = self.viz
        # Show loaded tile image
        if viz._tex_obj is not None:
            imgui.image(viz._tex_obj.gl_id, viz.wsi.tile_px, viz.wsi.tile_px)
            if imgui.is_item_hovered() or self._last_mouse_drag_pos is not None:
                if imgui.is_mouse_down(0):
                    mx, my = imgui.get_mouse_pos()
                    if self._last_mouse_drag_pos is not None:
                        dx, dy = mx - self._last_mouse_drag_pos[0], my - self._last_mouse_drag_pos[1]
                        self.set_viewer_tile_coords(viz.x + dx, viz.y + dy)
                    self._last_mouse_drag_pos = (mx, my)
                else:
                    imgui.set_tooltip("Current tile")
        # Show reference tile image
        if self._ref_tile_tex is not None:
            imgui.image(self._ref_tile_tex.gl_id, self.ref_wsi.tile_px, self.ref_wsi.tile_px)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Reference tile")
        # Check if the images have changed. If so, we'll calculate MSE.
        if self._last_wsi_tile is not viz._tex_img or self._last_ref_tile is not self.ref_tile:
            self._last_wsi_tile = viz._tex_img
            self._last_ref_tile = self.ref_tile
            self.mse = self.calculate_alignment_mse()
        if self.mse is not None:
            imgui.text("MSE: {:.2f}".format(self.mse))
        else:
            imgui.text("MSE: N/A")

    def draw_tile_alignment_control(self):
        """Draw the tile alignment control buttons."""
        viz = self.viz
        # Align the currently shown tile to the reference tile.
        half_button_width = (self.viz.sidebar.content_width - (self.viz.spacing * 3)) / 2
        if viz.sidebar.full_button2("Mark aligned", width=half_button_width, enabled=self.both_tiles_shown()):
            if self.ref_tile is not None:
                self.mark_aligned()
                self.align_slide()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Mark the shown tiles as aligned (enter)")
        imgui.same_line()
        if viz.sidebar.full_button("Align", width=half_button_width, enabled=self.both_tiles_shown()):
            self.align_tile()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Align the shown tiles (space)")

        # Show and manage the stored aligned coordinates.
        with imgui.begin_list_box("##aligned_coords", -1, viz.font_size * 5) as list_box:
            if list_box.opened:
                for i, coord in enumerate(self.aligned_coords):
                    _clicked, _sel = imgui.selectable("Aligned pair {}".format(i), selected=(i in self._selected_coords))
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("WSI: ({}, {}) | Reference: ({}, {})".format(*coord))
                    if _sel and i not in self._selected_coords:
                        if viz._shift_down:
                            self._selected_coords.append(i)
                        else:
                            self._selected_coords = [i]
                    elif _clicked and i in self._selected_coords:
                        self._selected_coords.remove(i)
                    if _clicked:
                        self.set_reference_tile_coords(
                            self.aligned_coords[i][2] - self.ref_wsi.full_extract_px // 2,
                            self.aligned_coords[i][3] - self.ref_wsi.full_extract_px // 2
                        )
                        viewer_coords = viz.wsi.slide.raw_to_coord(
                            self.aligned_coords[i][0],
                            self.aligned_coords[i][1]
                        )
                        self.set_viewer_tile_coords(
                            viewer_coords[0] - self.viz.wsi.full_extract_px // 2,
                            viewer_coords[1] - self.viz.wsi.full_extract_px // 2
                        )

        if imgui_utils.button('Delete', enabled=len(self._selected_coords) > 0):
            self.aligned_coords = [coord for i, coord in enumerate(self.aligned_coords) if i not in self._selected_coords]
            self._selected_coords = []
            if len(self.aligned_coords) > 0:
                self.align_slide()
            else:
                self.viz.wsi.alignment = None
        imgui.same_line()
        if imgui_utils.button('Reset', enabled=len(self.aligned_coords) > 0):
            self.aligned_coords = []
            self.viz.wsi.alignment = None

    def draw_settings(self):
        """Draw the settings for the alignment widget."""
        viz = self.viz
        imgui.text('Rotation')
        if imgui.is_item_hovered():
            imgui.set_tooltip("Rotate the loaded slide.")
        imgui.same_line(viz.label_w)
        with imgui_utils.item_width(viz.font_size*6):
            _clicked, self._rotation_idx = imgui.combo(
                "##rotation_combo",
                self._rotation_idx,
                self.rotations)
            if _clicked:
                rot = self.rot_int[self._rotation_idx]
                viz.reload_wsi(transforms=[rot])

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if not imgui.is_mouse_down(0):
            self._last_mouse_drag_pos = None

        if show:

            # Disable the slide thumbnail and tile extraction preview.
            if viz.viewer is not None:
                viz.viewer.show_thumbnail = False
            viz._show_tile_preview = False

            # Check if a new slide is loaded, and if so, auto-load alignment.
            if (self._last_wsi is not self.viz.wsi
                and self._output_path is not None
                and self.viz.wsi is not None
                and self.ref_wsi is not None
                and exists(join(self._output_path, f'{viz.wsi.name}_{self.ref_wsi.name}_alignment.npz'))):

                print("Loaded alignment for slide.")
                alignment = Alignment.load(join(self._output_path, f'{viz.wsi.name}_{self.ref_wsi.name}_alignment.npz'))
                viz.wsi.apply_alignment(alignment)
                self._last_wsi = self.viz.wsi

            viz.header("Slide Alignment")

            # === Control panel ===

            if viz.collapsing_header("Reference & Data", default=True):
                self.draw_reference_and_data()

            if viz.collapsing_header("Alignment", default=True):
                self.draw_tile_images()
                self.draw_tile_alignment_control()
                imgui.separator()
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Align the whole-slide image from the aligned pairs. (ctrl + space)")
                half_button_width = (self.viz.sidebar.content_width - (self.viz.spacing * 3)) / 2
                if viz.sidebar.full_button2("Load", width=half_button_width):
                    self.ask_load_alignment()
                imgui.same_line()
                if viz.sidebar.full_button("Export", width=half_button_width, enabled=(viz.wsi is not None and viz.wsi.alignment is not None)):
                    self.export_alignment()
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Export the alignment to a file (Ctrl+Enter).")
                imgui_utils.vertical_break()

            if viz.collapsing_header("Settings", default=False):
                # Rotation control for the loaded slide.
                self.draw_settings()

            self.draw_reference_popup()

#----------------------------------------------------------------------------
