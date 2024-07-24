"""Utility functions and constants for slide reading."""

import slideflow as sf
import cv2
import csv
import io
import numpy as np
import shapely.validation as sv
import shapely.geometry as sg
import shapely.affinity as sa
import xml.etree.ElementTree as ET

from shapely.ops import unary_union, polygonize
from PIL import Image, ImageDraw
from slideflow import errors, log
from types import SimpleNamespace
from typing import Union, List, Tuple, Optional, Dict

# Constants
DEFAULT_JPG_MPP = 1
OPS_LEVEL_COUNT = 'openslide.level-count'
OPS_MPP_X = 'openslide.mpp-x'
OPS_VENDOR = 'openslide.vendor'
OPS_BOUNDS_HEIGHT = 'openslide.bounds-height'
OPS_BOUNDS_WIDTH = 'openslide.bounds-width'
OPS_BOUNDS_X = 'openslide.bounds-x'
OPS_BOUNDS_Y = 'openslide.bounds-y'
TIF_EXIF_KEY_MPP = 65326
OPS_WIDTH = 'width'
OPS_HEIGHT = 'height'
DEFAULT_WHITESPACE_THRESHOLD = 230
DEFAULT_WHITESPACE_FRACTION = 1.0
DEFAULT_GRAYSPACE_THRESHOLD = 0.05
DEFAULT_GRAYSPACE_FRACTION = 0.6
FORCE_CALCULATE_WHITESPACE = -1
FORCE_CALCULATE_GRAYSPACE = -1
ROTATE_90_CLOCKWISE = 1
ROTATE_180_CLOCKWISE = 2
ROTATE_270_CLOCKWISE = 3
FLIP_HORIZONTAL = 4
FLIP_VERTICAL = 5


def OPS_LEVEL_HEIGHT(level: int) -> str:
    return f'openslide.level[{level}].height'


def OPS_LEVEL_WIDTH(level: int) -> str:
    return f'openslide.level[{level}].width'


def OPS_LEVEL_DOWNSAMPLE(level: int) -> str:
    return f'openslide.level[{level}].downsample'

# -----------------------------------------------------------------------------
# Classes

class ROI:
    """Object container for an ROI polygon annotation."""

    def __init__(
        self,
        name: str,
        coordinates: Union[np.ndarray, List[Tuple[int, int]]],
        *,
        label: Optional[str] = None,
        holes: Optional[List["ROI"]] = None
    ) -> None:
        self.name = name
        self._label = label if label else None
        self.holes = holes if holes else {}
        self._poly = None
        self._triangles = None
        self.coordinates = np.array(coordinates)
        self.validate()

    def __repr__(self):
        return f"<ROI (coords={len(self.coordinates)} label={self.label})>"

    @property
    def description(self) -> str:
        """Return a description of the ROI."""
        if not self.holes:
            return self.name
        else:
            return self.name + ' (holes: {})'.format(', '.join(
                [h.name for h in self.holes.values()]
            ))

    @property
    def label(self) -> Optional[str]:
        """Return the label of the ROI."""
        return self._label

    @label.setter
    def label(self, label: str) -> None:
        """Set the label of the ROI."""
        self._label = label
        for h in self.holes.values():
            h.label = label

    # --- Polygons ------------------------------------------------------------

    @property
    def poly(self) -> sg.Polygon:
        """Return the shapely polygon object."""
        if self._poly is None:
            self.update_polygon()
        return self._poly

    @property
    def triangles(self) -> np.ndarray:
        """Return the triangulated mesh."""
        if self._triangles is None:
            self._triangles = self.create_triangles()
        return self._triangles

    def make_polygon(self) -> sg.Polygon:
        """Create a shapely polygon from the coordinates.

        Raises:
            ValueError: If the coordinates do not form a valid polygon.

        Returns:
            sg.Polygon: Shapely polygon object.

        """
        poly = sv.make_valid(sg.Polygon(self.coordinates))
        to_delete = []
        for h, hole in self.holes.items():
            if not poly.contains(hole.poly):
                # Hole is not contained within the polygon,
                # so remove it from the list of holes.
                to_delete.append(h)
            poly = poly.difference(hole.poly)
        for h in to_delete:
            del self.holes[h]
        return poly

    def update_polygon(self) -> None:
        """Update the shapely polygon object."""
        self._poly = self.make_polygon()
        self._triangles = None

    def scaled_poly(self, scale: float) -> sg.Polygon:
        """Create a scaled polygon."""
        poly = sv.make_valid(sg.Polygon(self.scaled_coords(scale)))
        for h in self.holes.values():
            poly = poly.difference(h.scaled_poly(scale))
        return poly

    def poly_coords(self) -> np.ndarray:
        """Return the coordinates of the polygon."""
        if self.poly.geom_type in ('MultiPolygon', 'GeometryCollection'):
            valid_polys = [p for p in self.poly.geoms if p.geom_type == 'Polygon']
            if not len(valid_polys):
                return np.array([])
            else:
                coords = np.concatenate([
                    np.stack(p.exterior.coords.xy, axis=-1)
                    for p in valid_polys
                ])
        elif self.poly.geom_type == 'Polygon':
            coords = np.stack(self.poly.exterior.coords.xy, axis=-1)
        else:
            # This should have been caught by the validate function
            raise errors.InvalidROIError(f"Unrecognized ROI polygon geometry: {self.poly.geom_type}")
        # Remove duplicate points
        coords = np.concatenate([
            # Take the first coordinate
            np.expand_dims(coords[0], 0),
            # Only take subsequent coordinates if they are not repeating
            coords[1:][~np.all(coords[:-1] == coords[1:], axis=-1)]
        ], axis=0)
        return coords

    def simplify(self, tolerance: float = 5) -> None:
        """Simplify the polygon."""
        if self.poly.geom_type in ('MultiPolygon', 'GeometryCollection'):
            poly_s = sg.MultiPolygon([p.simplify(tolerance) for p in self.poly.geoms if p.geom_type == 'Polygon'])
            if not len(poly_s.geoms):
                # Polygon is empty after simplification, and thus cannot be simplified.
                log.warning(f"ROI {self.name} is empty after simplification.")
                pass
            else:
                self.coordinates = np.concatenate([np.stack(p.exterior.coords.xy, axis=-1) for p in poly_s.geoms])
        elif self.poly.geom_type == 'Polygon':
            poly_s = self.poly.simplify(tolerance=tolerance)
            self.coordinates = np.stack(poly_s.exterior.coords.xy, axis=-1)
        else:
            # This should have been caught by the validate function
            raise errors.InvalidROIError(f"Unrecognized ROI polygon geometry: {self.poly.geom_type}")
        for hole in self.holes.values():
            hole.simplify(tolerance)
        self.update_polygon()

    def invert_roi(self, wsi_shape: Tuple[int, int]):
        '''
        Invert the ROI within the whole-slide image.
        ROI becomes a hole in the inverted ROI.
        Return the inverted ROI.
        '''
        # Ensure polygon is generated
        self.update_polygon()
        # Calculate polygon bounding box (whole-slide)
        width, height = wsi_shape
        roi_wsi_coords = np.array([[0., 0.], [0., height], [width, height], [width, 0.]])
        # Create the inverted ROI
        inverted_ROI = ROI(name=self.name, coordinates=roi_wsi_coords)
        # Add the hole to the ROI
        inverted_ROI.add_hole(self)
        return inverted_ROI

    def create_triangles(self) -> Optional[np.ndarray]:
        """Create a triangulated mesh from the polygon."""

        def as_open_array(array):
            if (array[0] == array[-1]).all():
                return array[:-1]
            else:
                return array

        # First, ensure the polygon is valid
        if not self.polygon_is_valid():
            sf.log.error(
                "Unable to create triangles; ROI polygon is invalid."
            )
            return None
        if self.poly.geom_type != 'Polygon' or any([h.poly.geom_type != 'Polygon' for h in self.holes.values()]):
            sf.log.error(
                "Unable to create triangles; ROI is not a simple polygon."
            )
            return None

        # Vertices of the hole boundaries
        hole_vertices = {
            h: as_open_array(hole.poly_coords())
            for h, hole in self.holes.items()
        }

        # Filter out holes that are too small
        valid_holes = [hole for h, hole in self.holes.items() if len(hole_vertices[h]) > 3]
        hole_vertices = [v for h, v in hole_vertices.items() if len(v) > 3]

        # Verify all holes are contained within the polygon
        for hole in valid_holes:
            poly = sv.make_valid(sg.Polygon(self.poly_coords()))
            if not poly.contains(hole.poly):
                # Hole is not contained within the polygon
                return None

        # Vertices of representative points within each hole
        hole_points = [
            hole.poly.representative_point().coords[0]
            for hole in valid_holes
        ]

        if not len(hole_vertices):
            hole_vertices = None
            hole_points = None

        # Build triangles.
        triangle_vertices = sf.util.create_triangles(
            as_open_array(self.poly_coords()),
            hole_vertices=hole_vertices,
            hole_points=hole_points
        )
        return triangle_vertices

    # --- Holes ---------------------------------------------------------------

    def add_hole(self, roi: "ROI") -> None:
        """Add a hole to the ROI."""
        hole_name = self.get_next_hole_name()
        self.holes[hole_name] = roi
        self.update_polygon()

    def remove_hole(self, roi: Union["ROI", str]) -> None:
        """Remove a hole from the ROI."""
        if isinstance(roi, str):
            roi = self.get_hole(roi)
        hole_idx = [h for h, r in self.holes.items() if r == roi]
        del self.holes[hole_idx]
        self.update_polygon()

    def get_hole(self, name: str) -> "ROI":
        """Get a hole by name."""
        for h in self.holes.values():
            if h.name == name:
                return h
        raise ValueError(f"No hole found with name {name}")

    def get_next_hole_name(self) -> str:
        """Get the next available hole name."""
        return len(self.holes)

    # --- Other functions -----------------------------------------------------

    def validate(self) -> None:
        """Validate the exterior coordinates form a valid polygon."""
        try:
            poly_coords = self.poly_coords()
        except ValueError as e:
            raise errors.InvalidROIError(f"Invalid ROI ({self.name}): {e}")
        if len(poly_coords) < 4:
            raise errors.InvalidROIError(f"Invalid ROI ({self.name}): ROI must contain at least 4 coordinates.")
        if self.poly.geom_type not in ('Polygon', 'MultiPolygon', 'GeometryCollection'):
            raise errors.InvalidROIError(
                f"Invalid ROI ({self.name}): ROI must either be a Polygon, or "
                "MultiPolygon/GeometryCollection containing at least one polygon."
            )
        if self.poly.geom_type in ('MultiPolygon', 'GeometryCollection'):
            if not len([p for p in self.poly.geoms if p.geom_type == 'Polygon']):
                raise errors.InvalidROIError(
                    f"Invalid ROI ({self.name}): ROI must contain at least one valid polygon."
                )

    def polygon_is_valid(self) -> bool:
        """Check if the polygon is valid."""
        poly = sg.Polygon(self.coordinates)
        if not len(list(polygonize(unary_union(poly)))):
            # Polygon is self-intersecting
            return False
        if not poly.is_valid:
            # Polygon is invalid
            return False
        return all([h.polygon_is_valid() for h in self.holes.values()])

    def scaled_coords(self, scale: float) -> np.ndarray:
        return np.multiply(self.coordinates, 1/scale)


class QCMask:

    def __init__(
        self,
        mask: np.ndarray,
        filter_threshold: float = 0.6,
        is_roi: bool = False
    ) -> None:

        if not 0 <= filter_threshold <= 1:
            raise ValueError('filter_threshold must be between 0 and 1')
        if not isinstance(mask, np.ndarray):
            raise ValueError('mask must be a numpy array')
        if not len(mask.shape) == 2:
            raise ValueError('mask must be a 2D array')
        if not mask.dtype == bool:
            raise ValueError('mask must be a boolean array')

        self.mask = mask
        self.is_roi = is_roi
        self.filter_threshold = filter_threshold

    def __repr__(self):
        return f"<QCMask (shape={self.shape}), filter_threshold={self.filter_threshold}, is_roi={self.is_roi}>"

    @property
    def shape(self):
        return self.mask.shape


class Alignment:

    def __init__(
        self,
        origin: Tuple[int, int],
        scale: float,
        coord: Optional[np.ndarray] = None
    ) -> None:
        self.origin = origin
        self.scale = scale
        self.coord = coord
        self.centroid = None  # type: Tuple[float, float]
        self.normal = None    # type: Tuple[float, float]

    def __repr__(self):
        return f"<Alignment (origin={self.origin}, coord={self.coord}, centroid={self.centroid}, normal={self.normal})>"

    @classmethod
    def from_fit(cls, origin, scale, centroid, normal):
        obj = cls(origin, scale, None)
        obj.centroid = centroid
        obj.normal = normal
        return obj

    @classmethod
    def from_translation(cls, origin, scale):
        return cls(origin, scale, None)

    @classmethod
    def from_coord(cls, origin, coord):
        return cls(origin, None, coord)

    def save(self, path):
        save_dict = dict(
            origin=np.array(self.origin),
            scale=np.array(self.scale)
        )
        if self.coord is not None:
            save_dict['coord'] = self.coord
        if self.centroid is not None:
            save_dict['centroid'] = np.array(self.centroid)
        if self.normal is not None:
            save_dict['normal'] = np.array(self.normal)
        np.savez(path, **save_dict)

    @classmethod
    def load(cls, path):
        load_dict = np.load(path, allow_pickle=True)
        origin = tuple(load_dict['origin'])
        scale = load_dict['scale']
        coord = load_dict['coord'] if 'coord' in load_dict else None
        centroid = load_dict['centroid'] if 'centroid' in load_dict else None
        normal = load_dict['normal'] if 'normal' in load_dict else None
        obj = cls(origin, scale, coord)
        obj.centroid = centroid
        obj.normal = normal
        return obj


# -----------------------------------------------------------------------------
# Functions

def numpy2jpg(img: np.ndarray) -> str:
    if img.shape[-1] == 4:
        img = img[:, :, 0:3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imencode(".jpg", img)[1].tobytes()   # Default quality = 95%


def numpy2png(img: np.ndarray) -> str:
    if img.shape[-1] == 4:
        img = img[:, :, 0:3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imencode(".png", img)[1].tobytes()


def predict(
    slide: str,
    model: str,
    *,
    stride_div: int = 1,
    **kwargs
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Generate a whole-slide prediction from a saved model.

    Args:
        slide (str): Path to slide.
        model (str): Path to saved model trained in Slideflow.

    Keyword args:
        stride_div (int, optional): Divisor for stride when convoluting
                across slide. Defaults to 1.
        roi_dir (str, optional): Directory in which slide ROI is contained.
            Defaults to None.
        rois (list, optional): List of paths to slide ROIs. Alternative to
            providing roi_dir. Defaults to None.
        roi_method (str): Either 'inside', 'outside', 'auto', or 'ignore'.
            Determines how ROIs are used to extract tiles.
            If 'inside' or 'outside', will extract tiles in/out of an ROI,
            and raise errors.MissingROIError if an ROI is not available.
            If 'auto', will extract tiles inside an ROI if available,
            and across the whole-slide if no ROI is found.
            If 'ignore', will extract tiles across the whole-slide
            regardless of whether an ROI is available.
            Defaults to 'auto'.
        batch_size (int, optional): Batch size for calculating predictions.
            Defaults to 32.
        num_threads (int, optional): Number of tile worker threads. Cannot
            supply both ``num_threads`` (uses thread pool) and
            ``num_processes`` (uses multiprocessing pool). Defaults to
            CPU core count.
        num_processes (int, optional): Number of child processes to spawn
            for multiprocessing pool. Defaults to None (does not use
            multiprocessing).
        enable_downsample (bool, optional): Enable the use of downsampled
            slide image layers. Defaults to True.
        img_format (str, optional): Image format (png, jpg) to use when
            extracting tiles from slide. Must match the image format
            the model was trained on. If 'auto', will use the format
            logged in the model params.json. Defaults to 'auto'.
        generator_kwargs (dict, optional): Keyword arguments passed to
            the :meth:`slideflow.WSI.build_generator()`.
        device (torch.device, optional): PyTorch device. Defaults to
            initializing a new CUDA device.

    Returns:
        np.ndarray: Predictions for each outcome, with shape = (num_classes, )

        np.ndarray, optional: Uncertainty for each outcome, if the model was
        trained with uncertainty, with shape = (num_classes,)

    """
    from slideflow import Heatmap
    log.info("Calculating whole-slide prediction...")
    heatmap = Heatmap(slide, model, generate=True, stride_div=stride_div, **kwargs)
    assert heatmap.predictions is not None
    preds = heatmap.predictions.reshape(-1, heatmap.predictions.shape[-1])
    preds = np.ma.masked_where(preds == sf.heatmap.MASK, preds).mean(axis=0).filled()
    if heatmap.uncertainty is not None:
        unc = heatmap.uncertainty.reshape(-1, heatmap.uncertainty.shape[-1])
        unc = np.ma.masked_where(unc == sf.heatmap.MASK, unc).mean(axis=0).filled()
        return preds, unc
    else:
        return preds


def log_extraction_params(**kwargs) -> None:
    """Log tile extraction parameters."""

    if 'whitespace_fraction' not in kwargs:
        ws_f = DEFAULT_WHITESPACE_FRACTION
    else:
        ws_f = kwargs['whitespace_fraction']
    if 'whitespace_threshold' not in kwargs:
        ws_t = DEFAULT_WHITESPACE_THRESHOLD
    else:
        ws_t = kwargs['whitespace_threshold']
    if 'grayspace_fraction' not in kwargs:
        gs_f = DEFAULT_GRAYSPACE_FRACTION
    else:
        gs_f = kwargs['grayspace_fraction']
    if 'grayspace_threshold' not in kwargs:
        gs_t = DEFAULT_GRAYSPACE_THRESHOLD
    else:
        gs_t = kwargs['grayspace_threshold']

    if 'normalizer' in kwargs:
        log.info(f'Extracting tiles using [magenta]{kwargs["normalizer"]}[/] '
                 'normalization')
    if ws_f < 1:
        log.info('Filtering tiles by whitespace fraction')
        excl = f'(exclude if >={ws_f*100:.0f}% whitespace)'
        log.debug(f'Whitespace defined as RGB avg > {ws_t} {excl}')
    if gs_f < 1:
        log.info('Filtering tiles by grayspace fraction')
        excl = f'(exclude if >={gs_f*100:.0f}% grayspace)'
        log.debug(f'Grayspace defined as HSV avg < {gs_t} {excl}')


def draw_roi(
    img: Union[np.ndarray, str],
    coords: List[List[int]],
    color: str = 'red',
    linewidth: int = 5
) -> np.ndarray:
    """Draw ROIs on image.

    Args:
        img (Union[np.ndarray, str]): Image.
        coords (List[List[int]]): ROI coordinates.

    Returns:
        np.ndarray: Image as numpy array.
    """
    annPolys = [sg.Polygon(b) for b in coords]
    if isinstance(img, np.ndarray):
        annotated_img = Image.fromarray(img)
    elif isinstance(img, str):
        annotated_img = Image.open(io.BytesIO(img))  # type: ignore
    else:
        raise ValueError("Expected img to be a numpy array or bytes, got: {}".format(
            type(img)
        ))
    draw = ImageDraw.Draw(annotated_img)
    for poly in annPolys:
        if poly.geom_type in ('MultiPolygon', 'GeometryCollection'):
            for p in poly.geoms:
                if p.is_empty or p.geom_type != 'Polygon':
                    continue
                x, y = p.exterior.coords.xy
                zipped = list(zip(x.tolist(), y.tolist()))
                draw.line(zipped, joint='curve', fill=color, width=linewidth)
        else:
            x, y = poly.exterior.coords.xy
            zipped = list(zip(x.tolist(), y.tolist()))
            draw.line(zipped, joint='curve', fill=color, width=linewidth)
    return np.asarray(annotated_img)


def roi_coords_from_image(
    c: List[int],
    args: SimpleNamespace
) -> Tuple[List[int], List[np.ndarray], List[List[int]]]:
    # Scale ROI according to downsample level
    extract_scale = (args.extract_px / args.full_extract_px)

    # Scale ROI according to image resizing
    resize_scale = (args.tile_px / args.extract_px)

    def proc_coords(coords):
        # Offset coordinates to extraction window
        coord = np.add(coord, np.array([-1 * c[0], -1 * c[1]]))
        # Rescale according to downsampling and resizing
        coord = np.multiply(coord, (extract_scale * resize_scale))
        return coord

    # Filter out ROIs not in this tile
    coords = []
    ll = np.array([0, 0])
    ur = np.array([args.tile_px, args.tile_px])
    for roi in args.rois:
        coord = proc_coords(roi.coordinates)
        idx = np.all(np.logical_and(ll <= coord, coord <= ur), axis=1)
        coords_in_tile = coord[idx]
        if len(coords_in_tile) > 3:
            coords += [coords_in_tile]
        for hole in roi.holes.values():
            hole_coord = proc_coords(hole.coordinates)
            hole_idx = np.all(np.logical_and(ll <= hole_coord, hole_coord <= ur), axis=1)
            hole_coords_in_tile = hole_coord[hole_idx]
            if len(hole_coords_in_tile) > 3:
                coords += [hole_coords_in_tile]

    # Convert outer ROI to bounding box that fits within tile
    boxes = []
    yolo_anns = []
    for coord in coords:
        max_vals = np.max(coord, axis=0)
        min_vals = np.min(coord, axis=0)
        max_x = min(max_vals[0], args.tile_px)
        max_y = min(max_vals[1], args.tile_px)
        min_x = max(min_vals[0], 0)
        min_y = max(0, min_vals[1])
        width = (max_x - min_x) / args.tile_px
        height = (max_y - min_y) / args.tile_px
        x_center = ((max_x + min_x) / 2) / args.tile_px
        y_center = ((max_y + min_y) / 2) / args.tile_px
        yolo_anns += [[x_center, y_center, width, height]]
        boxes += [np.array([
            [min_x, min_y],
            [min_x, max_y],
            [max_x, max_y],
            [max_x, min_y]
        ])]
    return coords, boxes, yolo_anns


def xml_to_csv(path: str) -> str:
    """Create a QuPath format CSV ROI file from an ImageScope-format XML.

    ImageScope-formatted XMLs are expected to have "Region" and "Vertex"
    attributes. The "Region" attribute should have an "ID" sub-attribute.

    Args:
        path (str): ImageScope XML ROI file path

    Returns:
        str: Path to new CSV file.

    Raises:
        slideflow.errors.ROIError: If the XML could not be converted.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    new_csv_file = path[:-4] + '.csv'
    required_attributes = ['.//Region', './/Vertex']
    if not all(root.findall(a) for a in required_attributes):
        raise errors.ROIError(
            f"No ROIs found in the XML file {path}. Check that the XML "
            "file attributes are named correctly named in ImageScope "
            "format with 'Region' and 'Vertex' tags."
        )
    with open(new_csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['ROI_name', 'X_base', 'Y_base'])
        for region in root.findall('.//Region'):
            id_tag = region.get('Id')
            if not id_tag:
                raise errors.ROIError(
                    "No ID attribute found for Region. Check xml file and "
                    "ensure it adheres to ImageScope format."
                )
            roi_name = 'ROI_' + str(id_tag)
            vertices = region.findall('.//Vertex')
            if not vertices:
                raise errors.ROIError(
                    "No Vertex found in ROI. Check xml file and ensure it "
                    "adheres to ImageScope format."
                )
            csvwriter.writerows([
                [roi_name, vertex.get('X'), vertex.get('Y')]
                for vertex in vertices
            ])
    return new_csv_file


def get_scaled_and_intersecting_polys(
    polys: "sg.Polygon",
    tile: "sg.Polygon",
    scale: float,
    origin: Tuple[int, int]
):
    """Get scaled and intersecting polygons for a given tile.

    Args:
        polys (sg.Polygon): Shapely polygon containing union of all ROIs, in
            base dimensions.
        tile (sg.Polygon): Shapely polygon representing the tile, in base
            dimensions.
        scale (float): Scale factor.
        full_stride (int): Full stride, indictating the number of pixels in
            between each tile.
        grid_idx (Tuple[int, int]): Grid index of the tile (x, y).

    Returns:
        sg.Polygon: ROI polygons intersecting the tile, scaled to the tile
            size and with the origin with reference to the tile.

    """
    topleft, topright = origin
    A = polys.intersection(tile)

    # Translate polygons so the intersection origin is at (0, 0)
    B = sa.translate(A, -topleft, -topright)

    # Scale to the target tile size
    C = sa.scale(B, xfact=scale, yfact=scale, origin=(0, 0))
    return C


def _align_to_matrix(im1: np.ndarray, im2: np.ndarray, warp_matrix: np.ndarray) -> np.ndarray:
    """Align an image to a warp matrix."""
    # Use the warpAffine function to apply the transformation
    return cv2.warpAffine(im1, warp_matrix, (im2.shape[1], im2.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)


def _find_translation_matrix(
    im1: np.ndarray,
    im2: np.ndarray,
    *,
    denoise: bool = True,
    h: float = 30,
    block_size: int = 7,
    search_window: int = 21,
    n_iterations: int = 10000,
    termination_eps = 1e-10,
    warp_matrix: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Align two images using only scaling and translation.

    :param im1: The image to be aligned.
    :param im2: The reference image.
    :return: Aligned image of im1.
    """
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # De-noising
    if denoise:
        im1_gray = cv2.fastNlMeansDenoising(im1_gray, None, h, block_size, search_window)
        im2_gray = cv2.fastNlMeansDenoising(im2_gray, None, h, block_size, search_window)

    # Transform images to normalize contrast
    im1_gray = cv2.equalizeHist(im1_gray)
    im2_gray = cv2.equalizeHist(im2_gray)

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 matrix to store the transformation
    if warp_matrix is None:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Set the number of iterations and termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iterations, termination_eps)

    # Use findTransformECC to compute the transformation
    _, warp_matrix = cv2.findTransformECC(im2_gray, im1_gray, warp_matrix, warp_mode, criteria)

    return warp_matrix  # type: ignore


def align_image(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    Align two images using only scaling and translation.

    :param im1: The image to be aligned.
    :param im2: The reference image.
    :return: Aligned image of im1.
    """
    warp_matrix = _find_translation_matrix(im1, im2)
    return _align_to_matrix(im1, im2, warp_matrix)


def align_by_translation(
    im1: np.ndarray,
    im2: np.ndarray,
    round: bool = False,
    calculate_mse: bool = False,
    **kwargs
) -> Union[Union[Tuple[float, float], Tuple[int, int]],
           Tuple[Union[Tuple[float, float], Tuple[int, int]], float]]:
    """
    Find the (x, y) translation that aligns im1 to im2.

    Args:
        im1 (np.ndarray): Target for alignment.
        im2 (np.ndarray): Image to align.
        round (bool): Round to the nearest int. Defaults to False.
        calculate_mse (bool): Return the mean squared error (MSE) of alignment.
            Defaults to False.

    """
    try:
        warp_matrix = _find_translation_matrix(im1, im2, **kwargs)
    except cv2.error:
        raise errors.AlignmentError(
            "Could not align images. Check that the images are the same "
            "size, that they are not rotated or flipped, and that they have "
            "overlapping regions."
        )
    alignment = -warp_matrix[0, 2], -warp_matrix[1, 2]
    if round:
        alignment = (int(np.round(alignment[0])), int(np.round(alignment[1])))

    if calculate_mse:
        aligned_im1 = _align_to_matrix(im1, im2, warp_matrix)
        mse = compute_alignment_mse(aligned_im1, im2)
        return alignment, mse
    else:
        return alignment


def compute_alignment_mse(
    imageA: np.ndarray,
    imageB: np.ndarray,
    flatten: bool = True
) -> float:
    """
    Compute the Mean Squared Error between two images in their overlapping region,
    excluding areas that are black (0, 0, 0) in either image.

    :param imageA: First image.
    :param imageB: Second image.
    :return: Mean Squared Error (MSE) between the images in the valid overlapping region.
    """
    # Remove the alpha channel from both images
    if flatten:
        imageA = imageA[:, :, 0:3]
        imageB = imageB[:, :, 0:3]

    assert imageA.shape == imageB.shape, "Image sizes must match."

    # Create a combined mask where neither of the images is black
    combined_mask = np.logical_not(np.logical_or(imageA == 0, imageB == 0))

    # Compute MSE only for valid regions
    diff = (imageA.astype("float") - imageB.astype("float")) ** 2
    err = np.sum(diff[combined_mask]) / np.sum(combined_mask)

    return err


def best_fit_plane(points):
    # Ensure the input is a numpy array
    points = np.array(points)

    # 1. Center the data
    centroid = points.mean(axis=0)
    centered_points = points - centroid

    # 2. Compute the covariance matrix
    cov_matrix = np.cov(centered_points, rowvar=False)

    # 3. Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 4. Get the eigenvector corresponding to the smallest eigenvalue
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

    # The equation of the plane is `normal_vector . (x - centroid) = 0`
    return centroid, normal_vector


def z_on_plane(x, y, centroid, normal):
    cx, cy, cz = centroid
    nx, ny, nz = normal

    if nz == 0:
        raise ValueError("Normal vector's Z component is zero. Can't compute Z value for the given X, Y.")

    z = cz + (nx * (cx - x) + ny * (cy - y)) / nz
    return z


def calc_alignment(c, us, them, n=None):
    idx, (x, y, xi, yi) = c
    our_tile = us[xi, yi]
    try:
        their_tile = them[xi, yi]
    except IndexError:
        return None, c
    if our_tile is None or their_tile is None:
        return None, c
    if n is not None:
        our_tile = n.transform(our_tile[:, :, 0:3])
        their_tile = n.transform(their_tile[:, :, 0:3])
    try:
        rough_alignment = sf.slide.utils._find_translation_matrix(their_tile, our_tile, h=50, search_window=53)
    except cv2.error:
        rough_alignment = None
        log.debug("Initial rough alignment failed at x={}, y={} (grid {}, {})".format(
            x, y, xi, yi
        ))
    else:
        log.debug("Initial rough alignment complete at x={}, y={} (grid {}, {}): {}".format(
            x, y, xi, yi, (int(np.round(-rough_alignment[0, 2])), int(np.round(-rough_alignment[1, 2])))
        ))
    try:
        return align_by_translation(their_tile, our_tile, round=True, warp_matrix=rough_alignment), c
    except errors.AlignmentError as e:
        return 'error', c

# -----------------------------------------------------------------------------
# Internals

def _update_kw_with_defaults(kwargs) -> Dict:
    """Updates a set of keyword arguments with default extraction values.
    for whitepsace/grayspace filtering.
    """
    if kwargs['whitespace_fraction'] is None:
        kwargs['whitespace_fraction'] = DEFAULT_WHITESPACE_FRACTION
    if kwargs['whitespace_threshold'] is None:
        kwargs['whitespace_threshold'] = DEFAULT_WHITESPACE_THRESHOLD
    if kwargs['grayspace_fraction'] is None:
        kwargs['grayspace_fraction'] = DEFAULT_GRAYSPACE_FRACTION
    if kwargs['grayspace_threshold'] is None:
        kwargs['grayspace_threshold'] = DEFAULT_GRAYSPACE_THRESHOLD
    if kwargs['img_format'] is None:
        kwargs['img_format'] = 'jpg'
    return kwargs


def _polyArea(x: List[float], y: List[float]) -> float:
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def _convert_img_to_format(image: np.ndarray, img_format: str) -> str:
    if img_format.lower() == 'png':
        return cv2.imencode(
            '.png',
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )[1].tobytes()
    elif img_format.lower() in ('jpg', 'jpeg'):
        return cv2.imencode(
            '.jpg',
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        )[1].tostring()
    else:
        raise ValueError(f"Unknown image format {img_format}")
