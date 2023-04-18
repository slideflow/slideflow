"""Utility functions and constants for slide reading."""

import csv
import io
import numpy as np
import shapely.geometry as sg
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw
from slideflow import errors
from types import SimpleNamespace
from typing import Union, List, Tuple

DEFAULT_JPG_MPP = 1
OPS_LEVEL_COUNT = 'openslide.level-count'
OPS_MPP_X = 'openslide.mpp-x'
OPS_VENDOR = 'openslide.vendor'
TIF_EXIF_KEY_MPP = 65326
OPS_WIDTH = 'width'
OPS_HEIGHT = 'height'
DEFAULT_WHITESPACE_THRESHOLD = 230
DEFAULT_WHITESPACE_FRACTION = 1.0
DEFAULT_GRAYSPACE_THRESHOLD = 0.05
DEFAULT_GRAYSPACE_FRACTION = 0.6
FORCE_CALCULATE_WHITESPACE = -1
FORCE_CALCULATE_GRAYSPACE = -1


def OPS_LEVEL_HEIGHT(level: int) -> str:
    return f'openslide.level[{level}].height'


def OPS_LEVEL_WIDTH(level: int) -> str:
    return f'openslide.level[{level}].width'


def OPS_LEVEL_DOWNSAMPLE(level: int) -> str:
    return f'openslide.level[{level}].downsample'


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
    draw = ImageDraw.Draw(annotated_img)
    for poly in annPolys:
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

    def proc_ann(ann):
        # Scale to full image size
        coord = ann.coordinates
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
        coord = proc_ann(roi)
        idx = np.all(np.logical_and(ll <= coord, coord <= ur), axis=1)
        coords_in_tile = coord[idx]
        if len(coords_in_tile) > 3:
            coords += [coords_in_tile]

    # Convert ROI to bounding box that fits within tile
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
