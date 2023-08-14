from __future__ import absolute_import, division, print_function

import csv
import os
import sys
import time
import warnings
from functools import partial
from multiprocessing.dummy import Pool as DPool
from os.path import join
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from rich.progress import track

import slideflow as sf
from slideflow import errors
from slideflow.stats import SlideMap, get_centroid_index
from slideflow.util import log
from slideflow.stats import get_centroid_index

if TYPE_CHECKING:
    from slideflow.norm import StainNormalizer

# -----------------------------------------------------------------------------

def process_tile_image(args, decode_kwargs):
    if args is None:
        return None, None, None, None
    point_index, x, y, display_size, alpha, image = args
    if not point_index:
        return None, None, None, None
    if isinstance(image, tuple):
        tfr, tfr_idx = image
        image = sf.io.get_tfrecord_by_index(tfr, tfr_idx)['image_raw']
    if image is None:
        return point_index, None, None, None
    if sf.model.is_tensorflow_tensor(image):
        image = image.numpy()
    image = decode_image(image, **decode_kwargs)
    extent = [
        x - display_size/2,
        x + display_size/2,
        y - display_size/2,
        y + display_size/2
    ]
    return point_index, image, extent, alpha

def decode_image(
    image: Union[str, np.ndarray],
    normalizer: Optional["StainNormalizer"],
    img_format: str
) -> np.ndarray:
    """Internal method to convert an image string (as stored in TFRecords)
    to an RGB array."""

    if normalizer:
        try:
            if isinstance(image, np.ndarray):
                return normalizer.rgb_to_rgb(image)
            elif img_format in ('jpg', 'jpeg'):
                return normalizer.jpeg_to_rgb(image)
            elif img_format == 'png':
                return normalizer.png_to_rgb(image)
            else:
                return normalizer.transform(image)
        except Exception as e:
            log.error("Error encountered during image normalization, "
                        f"displaying image tile non-normalized. {e}")
    if isinstance(image, np.ndarray):
        return image
    else:
        image_arr = np.fromstring(image, np.uint8)
        tile_image_bgr = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(tile_image_bgr, cv2.COLOR_BGR2RGB)

def find_corresponding_points(row, points):
    return points.loc[((points.grid_x == row.x) & (points.grid_y == row.y))].index

# -----------------------------------------------------------------------------

class Mosaic:
    """Visualization of plotted image tiles."""

    def __init__(
        self,
        images: Union[SlideMap, List[np.ndarray], np.ndarray, List[Tuple[str, int]]],
        coords: Optional[Union[Tuple[int, int], np.ndarray]] = None,
        *,
        tfrecords: List[str] = None,
        normalizer: Optional[Union[str, "StainNormalizer"]] = None,
        normalizer_source: Optional[str] = None,
        **grid_kwargs
    ) -> None:
        """Generate a mosaic map, which visualizes plotted image tiles.

        Creating a mosaic map requires two components: a set of images and
        corresponding coordinates. Images and coordinates can either be manually
        provided, or the mosaic can dynamically read images from TFRecords as
        needed, reducing memory requirements.

        The first argument provides the images, and may be any of the following:

        - A list or array of images (np.ndarray, HxWxC)
        - A list of tuples, containing ``(slide_name, tfrecord_index)``
        - A ``slideflow.SlideMap`` object

        The second argument provides the coordinates, and may be any of:

        - A list or array of (x, y) coordinates for each image
        - None (if the first argument is a ``SlideMap``, which has coordinates)

        If images are to be read dynamically from tfrecords (with a ``SlideMap``,
        or by providing tfrecord indices directly), the keyword argument
        ``tfrecords`` must be specified with paths to tfrecords.

        Published examples:

        - Figure 4: https://doi.org/10.1038/s41379-020-00724-3
        - Figure 6: https://doi.org/10.1038/s41467-022-34025-x

        Examples
            Generate a mosaic map from a list of images and coordinates.

                .. code-block:: python

                    # Example data (images are HxWxC, np.ndarray)
                    images = [np.ndarray(...), ...]
                    coords = [(0.2, 0.9), ...]

                    # Generate the mosaic
                    mosaic = Mosaic(images, coordinates)

            Generate a mosaic map from tuples of TFRecord paths and indices.

                .. code-block:: python

                    # Example data
                    paths = ['/path/to/tfrecord.tfrecords', ...]
                    idx = [253, 112, ...]
                    coords = [(0.2, 0.9), ...]
                    tuples = [(tfr, idx) for tfr, i in zip(paths, idx)]

                    # Generate mosaic map
                    mosaic = sf.Mosaic(tuples, coords)

            Generate a mosaic map from a SlideMap and list of TFRecord paths.

                .. code-block:: python

                    # Prepare a SlideMap from a project
                    P = sf.Project('/project/path')
                    ftrs = P.generate_features('/path/to/model')
                    slide_map = sf.SlideMap.from_features(ftrs)

                    # Generate mosaic
                    mosaic = Mosaic(slide_map, tfrecords=ftrs.tfrecords)

        Args:
            images (list(np.ndarray), tuple, :class:`slideflow.SlideMap`):
                Images from which to generate the mosaic. May be a list or
                array of images (np.ndarray, HxWxC), a list of tuples,
                containing ``(slide_name, tfrecord_index)``, or a
                ``slideflow.SlideMap`` object.
            coords (list(str)): Coordinates for images. May be a list or array
                of (x, y) coordinates for each image (of same length
                as ``images``), or None (if ``images`` is a ``SlideMap`` object).

        Keyword args:
            tfrecords (list(str), optional): TFRecord paths. Required if
                ``images`` is either a ``SlideMap`` object or a list of tuples
                containing ``(slide_name, tfrecord_index)``. Defaults to None.
            num_tiles_x (int, optional): Mosaic map grid size. Defaults to 50.
            tile_select (str, optional): 'first', 'nearest', or 'centroid'.
                Determines how to choose a tile for display on each grid space.
                If 'first', will display the first valid tile in a grid space
                (fastest; recommended). If 'nearest', will display tile nearest
                to center of grid space. If 'centroid', for each grid, will
                calculate which tile is nearest to centroid tile_meta.
                Defaults to 'nearest'.
            tile_meta (dict, optional): Tile metadata, used for tile_select.
                Dictionary should have slide names as keys, mapped to list of
                metadata (length of list = number of tiles in slide).
                Defaults to None.
            normalizer ((str or :class:`slideflow.norm.StainNormalizer`), optional):
                Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Stain normalization preset or
                path to a source image. Valid presets include 'v1', 'v2', and
                'v3'. If None, will use the default present ('v3').
                Defaults to None.
        """
        self.tile_point_distances = []  # type: List[Dict]
        self.slide_map = None
        self.tfrecords = tfrecords
        self.grid_images = {}
        self.grid_coords = []   # type: np.ndarray
        self.grid_idx = []      # type: np.ndarray

        if isinstance(images, SlideMap):
            if tfrecords is None:
                raise ValueError("If building a Mosaic from a SlideMap, must "
                                 "provide paths to tfrecords via keyword arg "
                                 "tfrecords=...")
            elif isinstance(tfrecords, list) and not len(tfrecords):
                raise errors.TFRecordsNotFoundError()
            self._prepare_from_slidemap(images)
        elif isinstance(images[0], (tuple, list)) and isinstance(images[0][0], str):
            self._prepare_from_tuples(images, coords)  # type: ignore
        else:
            assert coords is not None
            assert len(images) == len(coords)
            self._prepare_from_coords(images, coords)  # type: ignore

        # ---------------------------------------------------------------------

        # Detect tfrecord image format
        if self.tfrecords is not None:
            _, self.img_format = sf.io.detect_tfrecord_format(self.tfrecords[0])
        else:
            self.img_format = 'numpy'

        # Setup normalization
        if isinstance(normalizer, str):
            log.info(f'Using realtime {normalizer} normalization')
            self.normalizer = sf.norm.autoselect(
                method=normalizer,
                source=normalizer_source
            )  # type: Optional[StainNormalizer]
        elif normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = None

        self.generate_grid(**grid_kwargs)

    def _prepare_from_coords(
        self,
        images: Union[List[np.ndarray], np.ndarray],
        coords: List[Union[Tuple[int, int], np.ndarray]]
    ) -> None:
        """Prepare the Mosaic map from a set of images and coordinates."""
        log.info('Loading coordinates and plotting points...')
        self.images = images
        self.mapped_tiles = []  # type: List[int]
        self.points = [{
            'coord': coords[i],
            'global_index': i,
            'category': 'none',
            'has_paired_tile': False,
        } for i in range(len(coords))]

    def _prepare_from_slidemap(
        self,
        slide_map: SlideMap,
        *,
        tile_meta: Optional[Dict] = None,
    ) -> None:
        """Prepare the Mosaic map from a ``SlideMap`` object."""
        log.info('Loading coordinates from SlideMap and plotting points...')
        self.slide_map = slide_map
        self.mapped_tiles = {}  # type: Dict[str, List[int]]
        self.points = slide_map.data.copy()
        self.points['has_paired_tile'] = False
        self.points['points_index'] = self.points.index
        self.points['alpha'] = 1.
        if tile_meta:
            self.points['meta'] = self.points.apply(lambda row: tile_meta[row.slide][row.tfr_index], axis=1)
        log.debug("Loading complete.")

    def _prepare_from_tuples(
        self,
        images: List[Tuple[str, int]],
        coords: List[Union[Tuple[int, int], np.ndarray]],
    ) -> None:
        """Prepare from a list of tuples with TFRecord names/indices."""
        log.info('Loading coordinates from SlideMap and plotting points...')
        self.mapped_tiles = {}  # type: Dict[str, List[int]]
        self.points = []
        for i, (tfr, idx) in enumerate(images):
            self.points.append({
                'coord': np.array(coords[i]),
                'global_index': i,
                'category': 'none',
                'slide': (tfr if self.tfrecords is not None
                          else sf.util.path_to_name(tfr)),
                'tfrecord': (tfr if self.tfrecords is None
                             else self._get_tfrecords_from_slide(tfr)),
                'tfrecord_index': idx,
                'has_paired_tile': None,
            })

    def _get_image_from_point(self, index):
        point = self.points.loc[index]
        if 'tfr_index' in point:
            tfr = self._get_tfrecords_from_slide(point.slide)
            tfr_idx = point.tfr_index
            if not tfr:
                log.error(f"TFRecord {tfr} not found in slide_map")
                return None
            image = sf.io.get_tfrecord_by_index(tfr, tfr_idx)['image_raw']
        else:
            image = self.images[index]
        return image

    def _get_tfrecords_from_slide(self, slide: str) -> Optional[str]:
        """Using the internal list of TFRecord paths, returns the path to a
        TFRecord for a given corresponding slide."""
        for tfr in self.tfrecords:
            if sf.util.path_to_name(tfr) == slide:
                return tfr
        log.error(f'Unable to find TFRecord path for slide [green]{slide}')
        return None

    def _initialize_figure(self, figsize, background):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=figsize)
        self.ax = fig.add_subplot(111, aspect='equal')
        self.ax.set_facecolor(background)
        fig.tight_layout()
        plt.subplots_adjust(
            left=0.02,
            bottom=0,
            right=0.98,
            top=1,
            wspace=0.1,
            hspace=0
        )
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

    def _plot_tile_image(self, image, extent, alpha=1):
        return self.ax.imshow(
            image,
            aspect='equal',
            origin='lower',
            extent=extent,
            zorder=99,
            alpha=alpha
        )

    def _finalize_figure(self):
        self.ax.autoscale(enable=True, tight=None)

    def _record_point(self, index):
        point = self.points.loc[index]
        if 'tfr_index' in point:
            tfr = self._get_tfrecords_from_slide(point.slide)
            if tfr is None:
                return
            if tfr in self.mapped_tiles:
                self.mapped_tiles[tfr] += [point.tfr_index]
            else:
                self.mapped_tiles[tfr] = [point.tfr_index]
        else:
            self.mapped_tiles += [index]

    @property
    def decode_kwargs(self):
        return dict(normalizer=self.normalizer, img_format=self.img_format)

    def points_at_grid_index(self, x, y):
        return self.points.loc[((self.points.grid_x == x) & (self.points.grid_y == y))]

    def selected_points(self):
        return self.points.loc[self.points.selected]

    def generate_grid(
        self,
        num_tiles_x: int = 50,
        tile_meta: Optional[Dict] = None,
        tile_select: str = 'first',
        max_dist: Optional[float] = None,
    ):
        """Generate the mosaic map grid.

        Args:
            num_tiles_x (int, optional): Mosaic map grid size. Defaults to 50.
            tile_meta (dict, optional): Tile metadata, used for tile_select.
                Dictionary should have slide names as keys, mapped to list of
                metadata (length of list = number of tiles in slide).
                Defaults to None.
            tile_select (str, optional): 'first', 'nearest', or 'centroid'.
                Determines how to choose a tile for display on each grid space.
                If 'first', will display the first valid tile in a grid space
                (fastest; recommended). If 'nearest', will display tile nearest
                to center of grid space. If 'centroid', for each grid, will
                calculate which tile is nearest to centroid tile_meta.
                Defaults to 'nearest'.
        """
        # Initial validation checks
        if tile_select not in ('nearest', 'centroid', 'first'):
            raise TypeError(f'Unknown tile selection method {tile_select}')
        else:
            log.debug(f'Tile selection method: {tile_select}')
        self.num_tiles_x = num_tiles_x
        self.grid_images = {}

        # Build the grid
        x_points = self.points.x.values
        y_points = self.points.y.values
        max_x = x_points.max()
        min_x = x_points.min()
        max_y = y_points.max()
        min_y = y_points.min()
        log.debug(f'Loaded {len(self.points)} points.')

        self.tile_size = (max_x - min_x) / self.num_tiles_x
        self.num_tiles_y = int((max_y - min_y) / self.tile_size)

        self.grid_idx = np.reshape(np.dstack(np.indices((self.num_tiles_x, self.num_tiles_y))), (self.num_tiles_x * self.num_tiles_y, 2))
        _grid_offset = np.array([(self.tile_size/2) + min_x, (self.tile_size/2) + min_y])
        self.grid_coords = (self.grid_idx * self.tile_size) + _grid_offset

        points_added = 0
        x_bins = np.arange(min_x, max_x, ((max_x - min_x) / self.num_tiles_x))[1:]
        y_bins = np.arange(min_y, max_y, ((max_y - min_y) / self.num_tiles_y))[1:]
        self.points['grid_x'] = np.digitize(self.points.x.values, x_bins, right=False)
        self.points['grid_y'] = np.digitize(self.points.y.values, y_bins, right=False)
        self.points['selected'] = False
        log.debug(f'{points_added} points added to grid')

        # Then, calculate distances from each point to each spot on the grid
        def select_nearest_points(idx):
            grid_x, grid_y = self.grid_idx[idx][0], self.grid_idx[idx][1]
            grid_coords = self.grid_coords[idx]
            # Calculate distance for each point within the grid tile from
            # center of the grid tile
            _points = self.points_at_grid_index(grid_x, grid_y)
            if not _points.empty:
                if tile_select == 'nearest':
                    point_coords = np.stack([_points.x.values, _points.y.values], axis=-1)
                    dist = np.linalg.norm(
                        point_coords - grid_coords,
                        ord=2,
                        axis=1.
                    )
                    if max_dist is not None:
                        masked_dist = np.ma.masked_array(dist, (dist >= (max_dist * self.tile_size)))
                        if masked_dist.count():
                            self.points.loc[_points.index[np.argmin(masked_dist)], 'selected'] = True
                    else:
                        self.points.loc[_points.index[np.argmin(dist)], 'selected'] = True
                elif not tile_meta:
                    raise errors.MosaicError(
                        'Mosaic centroid option requires tile_meta.'
                    )
                else:
                    centroid_index = get_centroid_index(_points.meta.values)
                    self.points.loc[_points.index[centroid_index], 'selected'] = True

        start = time.time()

        if tile_select == 'first':
            grid_group = self.points.groupby(['grid_x', 'grid_y'])
            first_indices = grid_group.nth(0).points_index.values
            self.points.loc[first_indices, 'selected'] = True
        elif tile_select in ('nearest', 'centroid'):
            self.points['selected'] = False
            dist_fn = partial(select_nearest_points)
            pool = DPool(sf.util.num_cpu())
            for i, _ in track(enumerate(pool.imap_unordered(dist_fn, range(len(self.grid_idx))), 1), total=len(self.grid_idx)):
                pass
            pool.close()
            pool.join()
        else:
            raise ValueError(
                f'Unrecognized value for tile_select: "{tile_select}"'
            )
        end = time.time()
        if sf.getLoggingLevel() <= 20:
            sys.stdout.write('\r\033[K')
        log.debug(f'Tile image selection complete ({end - start:.1f} sec)')

    def export(self, path: str) -> None:
        """Export SlideMap and configuration for later loading.

        Args:
            path (str): Directory in which to save configuration.

        """
        if self.slide_map is None:
            raise ValueError(
                "Mosaic.export() requires a Mosaic built from a SlideMap."
            )
        self.slide_map.save(path)
        if isinstance(self.tfrecords, list):
            tfr = self.tfrecords
        else:
            tfr = list(self.tfrecords)
        sf.util.write_json(tfr, join(path, 'tfrecords.json'))
        log.info(f"Mosaic configuration exported to {path}")

    def plot(
        self,
        figsize: Tuple[int, int] = (200, 200),
        focus: Optional[List[str]] = None,
        focus_slide: Optional[str] = None,
        background: str = '#dfdfdf',
        pool: Optional[Any] = None,
    ) -> None:
        """Initializes figures and places image tiles.

        If in a Jupyter notebook, the heatmap will be displayed in the cell
        output. If running via script or shell, the heatmap can then be
        shown on screen using matplotlib ``plt.show()``:

        .. code-block::

            import slideflow as sf
            import matplotlib.pyplot as plt

            heatmap = sf.Heatmap(...)
            heatmap.plot()
            plt.show()

        Args:
            figsize (Tuple[int, int], optional): Figure size. Defaults to
                (200, 200).
            focus (list, optional): List of tfrecords (paths) to highlight
                on the mosaic. Defaults to None.
            focus_slide (str, optional): Highlight tiles from this slide.
                Defaults to None.
        """
        if (focus is not None or focus_slide is not None) and self.tfrecords is None:
            raise ValueError("Unable to plot with focus; slides/tfrecords not configured.")

        log.debug("Initializing figure...")
        self._initialize_figure(figsize=figsize, background=background)

        # Reset alpha and display size
        if focus_slide:
            self.points['alpha'] = 1.
        self.points['display_size'] = self.tile_size

        if focus_slide:
            for idx in self.grid_idx:
                _points = self.points_at_grid_index(x=idx[0], y=idx[1])
                if not _points.empty and focus_slide:
                    n_matching = len(_points.loc[_points.slide == focus_slide])
                    self.points.loc[_points.index, 'alpha'] = n_matching / len(_points)

        # Then, pair grid tiles and points according to their distances
        log.info('Placing image tiles...')
        placed = 0
        start = time.time()
        to_map = []
        should_close_pool = False
        has_tfr = 'tfr_index' in self.points.columns
        selected_points = self.selected_points()

        for idx, point in selected_points.iterrows():
            if has_tfr:
                tfr = self._get_tfrecords_from_slide(point.slide)
                tfr_idx = point.tfr_index
                if tfr:
                    image = (tfr, tfr_idx)
                else:
                    log.error(f"TFRecord {tfr} not found in slide_map")
                    image = None
            else:
                image = self.images[idx]
            to_map.append((idx, point.grid_x * self.tile_size, point.grid_y * self.tile_size, point.display_size, point.alpha, image))

        if pool is None:
            pool = DPool(sf.util.num_cpu())
            should_close_pool = True
        for i, (point_idx, image, extent, alpha) in track(enumerate(pool.imap(partial(process_tile_image, decode_kwargs=self.decode_kwargs), to_map)), total=len(selected_points)):
            if point_idx is not None:
                self._record_point(point_idx)
                self._plot_tile_image(image, extent, alpha)
                point = self.points.loc[point_idx]
                self.grid_images[(point.grid_x, point.grid_y)] = image
                placed += 1

        if should_close_pool:
            pool.close()
            pool.join()
        log.debug(f'Tile images placed: {placed} ({time.time()-start:.2f}s)')
        if focus:
            self.focus(focus)
        self._finalize_figure()

    def save(self, filename: str, **kwargs: Any) -> None:
        """Saves the mosaic map figure to the given filename.

        Args:
            filename (str): Path at which to save the mosiac image.

        Keyword args:
            figsize (Tuple[int, int], optional): Figure size. Defaults to
                (200, 200).
            focus (list, optional): List of tfrecords (paths) to highlight on
                the mosaic.
        """
        import matplotlib.pyplot as plt

        self.plot(**kwargs)
        log.info('Exporting figure...')
        try:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
        except FileNotFoundError:
            pass
        plt.savefig(filename, bbox_inches='tight')
        log.info(f'Saved figure to [green]{filename}')
        plt.close()

    def save_report(self, filename: str) -> None:
        """Saves a report of which tiles (and their corresponding slide)
            were displayed on the Mosaic map, in CSV format."""
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['slide', 'index'])
            if isinstance(self.mapped_tiles, dict):
                for tfr in self.mapped_tiles:
                    for idx in self.mapped_tiles[tfr]:
                        writer.writerow([tfr, idx])
            else:
                for idx in self.mapped_tiles:
                        writer.writerow([idx])
        log.info(f'Mosaic report saved to [green]{filename}')

    def view(self, slides: List[str] = None) -> None:
        """Open Mosaic in Slideflow Studio.

        See :ref:`studio` for more information.

        Args:
            slides (list(str), optional): Path to whole-slide images. Used for
                displaying image tile context when hovering over a mosaic grid.
                Defaults to None.

        """
        from slideflow.studio.widgets import MosaicWidget
        from slideflow.studio import Studio

        studio = Studio(widgets=[MosaicWidget])
        mosaic = studio.get_widget('MosaicWidget')
        mosaic.load(
            self.slide_map,
            tfrecords=self.tfrecords,
            slides=slides,
            normalizer=self.normalizer
        )
        studio.run()
