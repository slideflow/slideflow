from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import cv2
import os
import sys
import csv

import numpy as np
import matplotlib.pyplot as plt
import slideflow as sf

from random import shuffle
from matplotlib import patches
from multiprocessing.dummy import Pool as DPool
from functools import partial
from tqdm import tqdm
from slideflow.util import log
from slideflow.slide import StainNormalizer
from slideflow.statistics import get_centroid_index

class MosaicError(Exception):
    pass

class Mosaic:
    """Visualization of tiles mapped using dimensionality reduction.

    .. _Published example (Figure 4):
        https://doi.org/10.1038/s41379-020-00724-3

    """

    def __init__(self, slide_map, tfrecords, focus=None, leniency=1.5, expanded=False, tile_zoom=15, num_tiles_x=50,
                 resolution='high', relative_size=False, tile_select='nearest', tile_meta=None, normalizer=None,
                 normalizer_source=None, focus_slide=None):

        """Generate a mosaic map.

        Args:
            slide_map (:class:`slideflow.statistics.SlideMap`): SlideMap object.
            tfrecords (list(str)): List of tfrecords paths.
            focus (list, optional): List of tfrecords (paths) to highlight on the mosaic.
            leniency (float, optional): UMAP leniency.
            expanded (bool, optional):If true, will try to fill in blank spots on the UMAP with nearby tiles.
                Takes exponentially longer to generate. Defaults to False.
            tile_zoom (int, optional): Factor which determines how large individual tiles appear. Defaults to 15.
            num_tiles_x (int, optional): Mosaic map grid size. Defaults to 50.
            resolution (str, optional): Resolution of exported figure; 'high', 'medium', or 'low'. Defaults to 'high'.
            relative_size (bool, optional): Physically size grid images in proportion to the number of tiles
                within the grid space. Defaults to False.
            tile_select (str, optional): Either 'nearest' or 'centroid'. Determines how to choose a tile for display
                on each grid space. If nearest, will display tile nearest to center of grid.
                If centroid, for each grid, will calculate which tile is nearest to centroid using data in tile_meta.
                Defaults to 'nearest'.
            tile_meta (dict, optional): Metadata for tiles, used if tile_select. Defaults to None.
                Dictionary should have slide names as keys, mapped to .ist of metadata
                (length of list = number of tiles in slide)
            normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image. Defaults to None.
                If None but using a normalizer, will use an internal tile for normalization.
                Internal default tile can be found at slideflow.slide.norm_tile.jpg
        """

        tile_point_distances = []
        max_distance_factor = leniency
        mapping_method = 'expanded' if expanded else 'strict'
        self.mapped_tiles = {}
        self.slide_map = slide_map
        self.num_tiles_x = num_tiles_x
        self.tfrecords = tfrecords

        # Detect tfrecord image format
        _, self.img_format = sf.io.detect_tfrecord_format(self.tfrecords[0])

        # Setup normalization
        if normalizer: log.info(f'Using realtime {normalizer} normalization')
        self.normalizer = None if not normalizer else StainNormalizer(method=normalizer, source=normalizer_source)

        # Initialize figure
        if resolution not in ('high', 'low'):
            log.warning(f"Unknown resolution option '{resolution}', defaulting to low resolution")
        if resolution == 'high':
            fig = plt.figure(figsize=(200,200))
            ax = fig.add_subplot(111, aspect='equal')
        else:
            fig = plt.figure(figsize=(24,18))
            ax = fig.add_subplot(121, aspect='equal')
        ax.set_facecolor('#dfdfdf')
        fig.tight_layout()
        plt.subplots_adjust(left=0.02, bottom=0, right=0.98, top=1, wspace=0.1, hspace=0)
        ax.set_aspect('equal', 'box')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # First, load UMAP coordinates
        log.info('Loading coordinates and plotting points...')
        self.points = []

        for i in range(len(slide_map.x)):
            slide = slide_map.point_meta[i]['slide']
            meta = tile_meta[slide][slide_map.point_meta[i]['index']] if tile_meta else None
            self.points.append({'coord':np.array((slide_map.x[i], slide_map.y[i])),
                                'global_index': i,
                                'category':'none',
                                'slide':slide,
                                'tfrecord':self._get_tfrecords_from_slide(slide),
                                'tfrecord_index':slide_map.point_meta[i]['index'],
                                'paired_tile':None,
                                'meta':meta})
        x_points = [p['coord'][0] for p in self.points]
        y_points = [p['coord'][1] for p in self.points]
        _x_width = max(x_points) - min(x_points)
        _y_width = max(y_points) - min(y_points)
        buffer = (_x_width + _y_width)/2 * 0.05
        max_x = max(x_points) + buffer
        min_x = min(x_points) - buffer
        max_y = max(y_points) + buffer
        min_y = min(y_points) - buffer
        log.debug(f'Loaded {len(self.points)} points.')

        tile_size = (max_x - min_x) / self.num_tiles_x
        self.num_tiles_y = int((max_y - min_y) / tile_size)
        max_distance = math.sqrt(2*((tile_size/2)**2)) * max_distance_factor

        # Initialize grid
        self.GRID = []
        for j in range(self.num_tiles_y):
            for i in range(self.num_tiles_x):
                x = ((tile_size/2) + min_x) + (tile_size * i)
                y = ((tile_size/2) + min_y) + (tile_size * j)
                self.GRID.append({    'coord': np.array((x, y)),
                                    'x_index': i,
                                    'y_index': j,
                                    'grid_index': len(self.GRID),
                                    'size': tile_size,
                                    'points':[],
                                    'nearest_index':[],
                                    'active': False,
                                    'image': None})

        # Add point indices to grid
        points_added = 0
        for point in self.points:
            x_index = int((point['coord'][0] - min_x) / tile_size)
            y_index = int((point['coord'][1] - min_y) / tile_size)
            for g in self.GRID:
                if g['x_index'] == x_index and g['y_index'] == y_index:
                    g['points'].append(point['global_index'])
                    points_added += 1
        for g in self.GRID:
            shuffle(g['points'])
        log.debug(f'{points_added} points added to grid')

        # Next, prepare mosaic grid by placing tile outlines
        log.info('Placing tile outlines...')
        max_grid_density = 1
        for g in self.GRID:
            max_grid_density = max(max_grid_density, len(g['points']))
        for grid_tile in self.GRID:
            rect_size = min((len(grid_tile['points']) / max_grid_density) * tile_zoom, 1) * tile_size

            tile = patches.Rectangle((grid_tile['coord'][0] - rect_size/2,
                                      grid_tile['coord'][1] - rect_size/2),
                                      rect_size,
                                      rect_size,
                                      fill=True,
                                      alpha=1,
                                      facecolor='white',
                                      edgecolor='#cccccc')
            ax.add_patch(tile)

            grid_tile['size'] = rect_size
            grid_tile['rectangle'] = tile
            grid_tile['paired_point'] = None

        # Then, calculate distances from each point to each spot on the grid
        if mapping_method not in ('strict', 'expanded'):
            raise TypeError('Unknown mapping method; must be strict or expanded')
        else:
            log.debug(f'Mapping method: {mapping_method}')

        if tile_select not in ('nearest', 'centroid'):
            raise TypeError('Unknown tile selection method; must be nearest or centroid')
        else:
            log.debug(f'Tile selection method: {tile_select}')

        def calc_distance(tile, global_point_coords):
            if mapping_method == 'strict':
                # Calculate distance for each point within the grid tile from center of the grid tile
                point_coords = np.asarray([self.points[global_index]['coord'] for global_index in tile['points']])
                if len(point_coords):
                    if tile_select == 'nearest':
                        distances = np.linalg.norm(point_coords - tile['coord'], ord=2, axis=1.)
                        tile['nearest_index'] = tile['points'][np.argmin(distances)]
                    elif not tile_meta:
                        raise MosaicError('Unable to calculate centroid for mosaic if tile_meta not provided.')
                    else:
                        meta_vals_from_points = [self.points[global_index]['meta'] for global_index in tile['points']]
                        centroid_index = get_centroid_index(meta_vals_from_points)
                        tile['nearest_index'] = tile['points'][centroid_index]
            elif mapping_method == 'expanded':
                # Calculate distance for each point within the entire grid from center of the grid tile
                distances = np.linalg.norm(global_point_coords - tile['coord'], ord=2, axis=1.)
                for i, distance in enumerate(distances):
                    if distance <= max_distance:
                        tile_point_distances.append({'distance': distance,
                                                    'grid_index':tile['grid_index'],
                                                    'point_index':self.points[i]['global_index']})

        log.info('Calculating tile-point distances...')
        tile_point_start = time.time()
        global_point_coords = np.asarray([p['coord'] for p in self.points])
        pool = DPool(8)
        dist_fn = partial(calc_distance, global_point_coords=global_point_coords)
        for i, _ in tqdm(enumerate(pool.imap_unordered(dist_fn, self.GRID), 1), total=len(self.GRID), ncols=80):
            if log.getEffectiveLevel() <= 20: sys.stderr.write(f'\rCompleted {i/len(self.GRID):.2%}')
        pool.close()
        pool.join()
        tile_point_end = time.time()
        if log.getEffectiveLevel() <= 20: sys.stdout.write('\r\033[K')
        log.debug(f'Calculations complete ({tile_point_end-tile_point_start:.0f} sec)')

        if mapping_method == 'expanded':
            tile_point_distances.sort(key=lambda d: d['distance'])

        # Then, pair grid tiles and points according to their distances
        log.info('Placing image tiles...')
        num_placed = 0
        if mapping_method == 'strict':
            for tile in self.GRID:
                if not len(tile['points']): continue
                closest_point = tile['nearest_index']
                point = self.points[closest_point]

                if not point['tfrecord']:
                    log.error(f"TFRecord {point['slide']} not found in slide_map; verify that the TFRecord exists.")
                    continue
                _, tile_image = sf.io.get_tfrecord_by_index(point['tfrecord'],
                                                            point['tfrecord_index'],
                                                            decode=False)
                if not tile_image: continue

                self.mapped_tiles.update({point['tfrecord']: point['tfrecord_index']})
                if sf.backend() == 'tensorflow':
                    tile_image = tile_image.numpy()
                tile_image = self._decode_image_string(tile_image)

                tile_alpha, num_slide, num_other = 1, 0, 0
                display_size = tile_size
                if relative_size:
                    if focus_slide and len(tile['points']):
                        for point_index in tile['points']:
                            point = self.points[point_index]
                            if point['slide'] == focus_slide:
                                num_slide += 1
                            else:
                                num_other += 1
                        fraction_slide = num_slide / (num_other + num_slide)
                        tile_alpha = fraction_slide
                    display_size = tile['size']
                image = ax.imshow(tile_image,
                                  aspect='equal',
                                  origin='lower',
                                  extent=[tile['coord'][0]-display_size/2,
                                  tile['coord'][0]+display_size/2,
                                  tile['coord'][1]-display_size/2,
                                  tile['coord'][1]+display_size/2],
                                  zorder=99,
                                  alpha=tile_alpha)
                tile['image'] = image
                num_placed += 1
        elif mapping_method == 'expanded':
            for i, distance_pair in tqdm(enumerate(tile_point_distances), total=len(tile_point_distances), ncols=80):
                # Attempt to place pair, skipping if unable (due to other prior pair)
                point = self.points[distance_pair['point_index']]
                tile = self.GRID[distance_pair['grid_index']]
                if not (point['paired_tile'] or tile['paired_point']):
                    _, tile_image = sf.io.get_tfrecord_by_index(point['tfrecord'],
                                                                point['tfrecord_index'],
                                                                decode=False)
                    if not tile_image: continue

                    point['paired_tile'] = True
                    tile['paired_point'] = True

                    self.mapped_tiles.update({point['tfrecord']: point['tfrecord_index']})
                    if sf.backend() == 'tensorflow':
                        tile_image = tile_image.numpy()
                    tile_image = self._decode_image_string(tile_image)

                    image = ax.imshow(tile_image,
                                      aspect='equal',
                                      origin='lower',
                                      extent=[tile['coord'][0]-tile_size/2,
                                      tile['coord'][0]+tile_size/2,
                                      tile['coord'][1]-tile_size/2,
                                      tile['coord'][1]+tile_size/2],
                                      zorder=99)
                    tile['image'] = image
                    num_placed += 1
        log.debug(f'Num placed: {num_placed}')

        # Focus on a subset of TFRecords if desired
        if focus: self.focus(focus)

        # Finally, finish the mosaic figure
        ax.autoscale(enable=True, tight=None)

    def _get_tfrecords_from_slide(self, slide):
        """Using the internal list of TFRecord paths, returns the path to a TFRecord for a given corresponding slide."""

        for tfr in self.tfrecords:
            if sf.util.path_to_name(tfr) == slide:
                return tfr
        log.error(f'Unable to find TFRecord path for slide {sf.util.green(slide)}')

    def _decode_image_string(self, string):
        """Internal method to convert an image string (as stored in TFRecords) to an RGB array."""

        if self.normalizer:
            if self.img_format in ('jpg', 'jpeg'):
                tile_image = self.normalizer.jpeg_to_rgb(string)
            elif self.img_format == 'png':
                tile_image = self.normalizer.png_to_rgb(string)
            else:
                raise MosaicError(f"Unknown image format in tfrecords: {self.img_format}")
        else:
            image_arr = np.fromstring(string, np.uint8)
            tile_image_bgr = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
            tile_image = cv2.cvtColor(tile_image_bgr, cv2.COLOR_BGR2RGB)
        return tile_image

    def focus(self, tfrecords):
        """Highlights certain tiles according to a focus list if list provided,
        or resets highlighting if no tfrecords provided."""

        if tfrecords:
            for tile in self.GRID:
                if not len(tile['points']) or not tile['image']: continue
                num_cat, num_other = 0, 0
                for point_index in tile['points']:
                    point = self.points[point_index]
                    if point['tfrecord'] in tfrecords:
                        num_cat += 1
                    else:
                        num_other += 1
                alpha = num_cat / (num_other + num_cat)
                tile['image'].set_alpha(alpha)
        else:
            for tile in self.GRID:
                if not len(tile['points']) or not tile['image']: continue
                tile['image'].set_alpha(1)

    def save(self, filename):
        """Saves the mosaic map figure to the given filename."""

        log.info('Exporting figure...')
        try:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
        except FileNotFoundError:
            pass
        plt.savefig(filename, bbox_inches='tight')
        log.info(f'Saved figure to {sf.util.green(filename)}')
        plt.close()

    def save_report(self, filename):
        """Saves a report of which tiles (and their corresponding slide)
            were displayed on the Mosaic map, in CSV format."""

        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['slide', 'index'])
            for tfr in self.mapped_tiles:
                writer.writerow([tfr, self.mapped_tiles[tfr]])
        log.info(f'Mosaic report saved to {sf.util.green(filename)}')

    def show(self):
        """Displays the mosaic map as an interactive matplotlib figure."""
        log.info('Displaying figure...')
        while True:
            try:
                plt.show()
            except UnicodeDecodeError:
                continue
            break