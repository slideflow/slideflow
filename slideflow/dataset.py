"""Module for the ``Dataset`` class and its associated functions.

The ``Dataset`` class handles management of collections of patients,
clinical annotations, slides, extracted tiles, and assembly of images
into torch DataLoader and tensorflow Dataset objects. The high-level
overview of the structure of ``Dataset`` is as follows:


 ──────────── Information Methods ───────────────────────────────
   Annotations      Slides        Settings         TFRecords
  ┌──────────────┐ ┌─────────┐   ┌──────────────┐ ┌──────────────┐
  │Patient       │ │Paths to │   │Tile size (px)│ | *.tfrecords  |
  │Slide         │ │ slides  │   │Tile size (um)│ |  (generated) |
  │Label(s)      │ └─────────┘   └──────────────┘ └──────────────┘
  │ - Categorical│  .slides()     .tile_px         .tfrecords()
  │ - Continuous │  .rois()       .tile_um         .manifest()
  │ - Time Series│  .slide_paths()                 .num_tiles
  └──────────────┘  .thumbnails()                  .img_format
    .patients()
    .rois()
    .labels()
    .harmonize_labels()
    .is_float()


 ─────── Filtering and Splitting Methods ──────────────────────
  ┌────────────────────────────┐
  │                            │
  │ ┌─────────┐                │ .filter()
  │ │Filtered │                │ .remove_filter()
  │ │ Dataset │                │ .clear_filters()
  │ └─────────┘                │ .split()
  │               Full Dataset │
  └────────────────────────────┘


 ───────── Summary of Image Data Flow ──────────────────────────
  ┌──────┐
  │Slides├─────────────┐
  └──┬───┘             │
     │                 │
     ▼                 │
  ┌─────────┐          │
  │TFRecords├──────────┤
  └──┬──────┘          │
     │                 │
     ▼                 ▼
  ┌────────────────┐ ┌─────────────┐
  │torch DataLoader│ │Loose images │
  │ / tf Dataset   │ │ (.png, .jpg)│
  └────────────────┘ └─────────────┘

 ──────── Slide Processing Methods ─────────────────────────────
  ┌──────┐
  │Slides├───────────────┐
  └──┬───┘               │
     │.extract_tiles()   │.extract_tiles(
     ▼                   │    save_tiles=True
  ┌─────────┐            │  )
  │TFRecords├────────────┤
  └─────────┘            │ .extract_tiles
                         │  _from_tfrecords()
                         ▼
                       ┌─────────────┐
                       │Loose images │
                       │ (.png, .jpg)│
                       └─────────────┘


 ─────────────── TFRecords Operations ─────────────────────────
                      ┌─────────┐
   ┌────────────┬─────┤TFRecords├──────────┐
   │            │     └─────┬───┘          │
   │.tfrecord   │.tfrecord  │ .balance()   │.resize_tfrecords()
   │  _heatmap()│  _report()│ .clip()      │.split_tfrecords
   │            │           │ .torch()     │  _by_roi()
   │            │           │ .tensorflow()│
   ▼            ▼           ▼              ▼
  ┌───────┐ ┌───────┐ ┌────────────────┐┌─────────┐
  │Heatmap│ │PDF    │ │torch DataLoader││TFRecords│
  └───────┘ │ Report│ │ / tf Dataset   │└─────────┘
            └───────┘ └────────────────┘
"""

import copy
import csv
import multiprocessing as mp
import os
import shutil
import threading
import time
import types
import tempfile
import warnings
from contextlib import contextmanager
from collections import defaultdict
from datetime import datetime
from glob import glob
from multiprocessing.dummy import Pool as DPool
from os.path import basename, dirname, exists, isdir, join
from queue import Queue
from random import shuffle
from tabulate import tabulate  # type: ignore[import]
from pprint import pformat
from functools import partial
from typing import (TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple,
                    Union, Callable)
import numpy as np
import pandas as pd
import shapely.geometry as sg
from rich.progress import track, Progress
from tqdm import tqdm

import slideflow as sf
from slideflow import errors
from slideflow.slide import WSI, ExtractionReport, SlideReport
from slideflow.util import (log, Labels, _shortname, path_to_name,
                            tfrecord2idx, TileExtractionProgress)

if TYPE_CHECKING:
    import tensorflow as tf
    import cellpose
    from slideflow.model import ModelParams
    from torch.utils.data import DataLoader
    from slideflow.norm import StainNormalizer

# -----------------------------------------------------------------------------


def _prepare_slide(
    path: str,
    report_dir: Optional[str],
    wsi_kwargs: Dict,
    qc: Optional[str],
    qc_kwargs: Dict,
) -> Optional["sf.WSI"]:

    try:
        slide = sf.WSI(path, **wsi_kwargs)
        if qc:
            slide.qc(method=qc, **qc_kwargs)
        return slide
    except errors.MissingROIError:
        log.debug(f'Missing ROI for slide {path}; skipping')
        return None
    except errors.IncompatibleBackendError:
        log.error('Slide {} has type {}, which is incompatible with the active '
                  'slide reading backend, {}. Consider using a different '
                  'backend, which can be set with the environmental variable '
                  'SF_SLIDE_BACKEND. See https://slideflow.dev/installation/#cucim-vs-libvips '
                  'for more information.'.format(
                    path,
                    sf.util.path_to_ext(path).upper(),
                    sf.slide_backend()
                  ))
    except errors.SlideLoadError as e:
        log.error(f'Error loading slide {path}: {e}. Skipping')
        return None
    except errors.QCError as e:
        log.error(e)
        return None
    except errors.TileCorruptionError:
        log.error(f'{path} corrupt; skipping')
        return None
    except (KeyboardInterrupt, SystemExit) as e:
        print('Exiting...')
        raise e
    except Exception as e:
        log.error(f'Error processing slide {path}: {e}. Skipping')
        return None


@contextmanager
def _handle_slide_errors(path: str):
    try:
        yield
    except errors.MissingROIError:
        log.info(f'Missing ROI for slide {path}; skipping')
    except errors.SlideLoadError as e:
        log.error(f'Error loading slide {path}: {e}. Skipping')
    except errors.QCError as e:
        log.error(e)
    except errors.TileCorruptionError:
        log.error(f'{path} corrupt; skipping')
    except (KeyboardInterrupt, SystemExit) as e:
        print('Exiting...')
        raise e


def _tile_extractor(
    path: str,
    tfrecord_dir: str,
    tiles_dir: str,
    reports: Dict,
    qc: str,
    wsi_kwargs: Dict,
    generator_kwargs: Dict,
    qc_kwargs: Dict,
    render_thumb: bool = True
) -> None:
    """Extract tiles. Internal function.

    Slide processing needs to be process-isolated when num_workers > 1 .

    Args:
        tfrecord_dir (str): Path to TFRecord directory.
        tiles_dir (str): Path to tiles directory (loose format).
        reports (dict): Multiprocessing-enabled dict.
        qc (bool): Quality control method.
        wsi_kwargs (dict): Keyword arguments for sf.WSI.
        generator_kwargs (dict): Keyword arguments for WSI.extract_tiles()
        qc_kwargs(dict): Keyword arguments for quality control.
    """
    with _handle_slide_errors(path):
        log.debug(f'Extracting tiles for {path_to_name(path)}')
        slide = _prepare_slide(
            path,
            report_dir=tfrecord_dir,
            wsi_kwargs=wsi_kwargs,
            qc=qc,
            qc_kwargs=qc_kwargs)
        if slide is not None:
            report = slide.extract_tiles(
                tfrecord_dir=tfrecord_dir,
                tiles_dir=tiles_dir,
                **generator_kwargs
            )
            if render_thumb and isinstance(report, SlideReport):
                _ = report.thumb
            reports.update({path: report})


def _buffer_slide(path: str, dest: str) -> str:
    """Buffer a slide to a path."""
    buffered = join(dest, basename(path))
    shutil.copy(path, buffered)

    # If this is an MRXS file, copy the associated folder.
    if path.lower().endswith('mrxs'):
        folder_path = join(dirname(path), path_to_name(path))
        if exists(folder_path):
            shutil.copytree(folder_path, join(dest, path_to_name(path)))
        else:
            log.debug("Could not find associated MRXS folder for slide buffer")

    return buffered


def _debuffer_slide(path: str) -> None:
    """De-buffer a slide."""
    os.remove(path)
    # If this is an MRXS file, remove the associated folder.
    if path.lower().endswith('mrxs'):
        folder_path = join(dirname(path), path_to_name(path))
        if exists(folder_path):
            shutil.rmtree(folder_path)
        else:
            log.debug("Could not find MRXS folder for slide debuffer")


def _fill_queue(
    slide_list: Sequence[str],
    q: Queue,
    q_size: int,
    buffer: Optional[str] = None
) -> None:
    """Fill a queue with slide paths, using an optional buffer."""
    for path in slide_list:
        warned = False
        if buffer:
            while True:
                if q.qsize() < q_size:
                    try:
                        q.put(_buffer_slide(path, buffer))
                        break
                    except OSError:
                        if not warned:
                            slide = _shortname(path_to_name(path))
                            log.debug(f'OSError for {slide}: buffer full?')
                            log.debug(f'Queue size: {q.qsize()}')
                            warned = True
                        time.sleep(1)
                else:
                    time.sleep(1)
        else:
            q.put(path)
    q.put(None)
    q.join()


def _count_otsu_tiles(wsi):
    wsi.qc('otsu')
    return wsi.estimated_num_tiles


def _create_index(tfrecord, force=False):
    index_name = join(
        dirname(tfrecord),
        path_to_name(tfrecord)+'.index'
    )
    if not tfrecord2idx.find_index(tfrecord) or force:
        tfrecord2idx.create_index(tfrecord, index_name)


def _get_tile_df(
    slide_path: str,
    tile_px: int,
    tile_um: Union[int, str],
    rois: Optional[List[str]],
    stride_div: int,
    roi_method: str
) -> pd.DataFrame:
    try:
        wsi = sf.WSI(
        slide_path,
        tile_px,
        tile_um,
        rois=rois,
        stride_div=stride_div,
        roi_method=roi_method,
        verbose=False
    )
    except Exception as e:
        log.warning("Skipping slide {}, error raised: {}".format(
            path_to_name(slide_path), e
        ))
        return None
    _df = wsi.get_tile_dataframe()
    _df['slide'] = wsi.name
    return _df

# -----------------------------------------------------------------------------

def split_patients_preserved_site(
    patients_dict: Dict[str, Dict],
    n: int,
    balance: Optional[str] = None,
    method: str = 'auto'
) -> List[List[str]]:
    """Split a dictionary of patients into n groups, with site balancing.

    Splits are balanced according to key "balance", while preserving site.

    Args:
        patients_dict (dict): Nested dictionary mapping patient names to
            dict of outcomes: labels
        n (int): Number of splits to generate.
        balance (str): Annotation header to balance splits across.
        method (str): Solver method. 'auto', 'cplex', or 'bonmin'. If 'auto',
            will use CPLEX if availabe, otherwise will default to pyomo/bonmin.

    Returns:
        List of patient splits
    """
    patient_list = list(patients_dict.keys())
    shuffle(patient_list)

    def flatten(arr):
        """Flatten an array."""
        return [y for x in arr for y in x]

    # Get patient outcome labels
    if balance is not None:
        patient_outcome_labels = [
            patients_dict[p][balance] for p in patient_list
        ]
    else:
        patient_outcome_labels = [1 for _ in patient_list]
    # Get unique outcomes
    unique_labels = list(set(patient_outcome_labels))
    n_unique = len(set(unique_labels))
    # Delayed import in case CPLEX not installed
    import slideflow.io.preservedsite.crossfolds as cv

    site_list = [patients_dict[p]['site'] for p in patient_list]
    df = pd.DataFrame(
        list(zip(patient_list, patient_outcome_labels, site_list)),
        columns=['patient', 'outcome_label', 'site']
    )
    df = cv.generate(
        df, 'outcome_label', k=n, target_column='CV', method=method
    )
    log.info("[bold]Train/val split with Preserved-Site Cross-Val")
    log.info("[bold]Category\t" + "\t".join(
        [str(cat) for cat in range(n_unique)]
    ))
    for k in range(n):
        def num_labels_matching(o):
            match = df[(df.CV == str(k+1)) & (df.outcome_label == o)]
            return str(len(match))
        matching = [num_labels_matching(o) for o in unique_labels]
        log.info(f"K-fold-{k}\t" + "\t".join(matching))
    splits = [
        df.loc[df.CV == str(ni+1), "patient"].tolist()
        for ni in range(n)
    ]
    return splits


def split_patients_balanced(
    patients_dict: Dict[str, Dict],
    n: int,
    balance: str
) -> List[List[str]]:
    """Split a dictionary of patients into n groups, balancing by outcome.

    Splits are balanced according to key "balance".

    Args:
        patients_dict (dict): Nested ditionary mapping patient names to
            dict of outcomes: labels
        n (int): Number of splits to generate.
        balance (str): Annotation header to balance splits across.

    Returns:
        List of patient splits
    """
    patient_list = list(patients_dict.keys())
    shuffle(patient_list)

    def flatten(arr):
        """Flatten an array."""
        return [y for x in arr for y in x]

    # Get patient outcome labels
    patient_outcome_labels = [
        patients_dict[p][balance] for p in patient_list
    ]
    # Get unique outcomes
    unique_labels = list(set(patient_outcome_labels))
    n_unique = len(set(unique_labels))

    # Now, split patient_list according to outcomes
    pt_by_outcome = [
        [p for p in patient_list if patients_dict[p][balance] == uo]
        for uo in unique_labels
    ]
    # Then, for each sublist, split into n components
    pt_by_outcome_by_n = [
        list(sf.util.split_list(sub_l, n)) for sub_l in pt_by_outcome
    ]
    # Print splitting as a table
    log.info(
        "[bold]Category\t" + "\t".join([str(cat) for cat in range(n_unique)])
    )
    for k in range(n):
        matching = [str(len(clist[k])) for clist in pt_by_outcome_by_n]
        log.info(f"K-fold-{k}\t" + "\t".join(matching))
    # Join sublists
    splits = [
        flatten([
            item[ni] for item in pt_by_outcome_by_n
        ]) for ni in range(n)
    ]
    return splits


def split_patients(patients_dict: Dict[str, Dict], n: int) -> List[List[str]]:
    """Split a dictionary of patients into n groups.

    Args:
        patients_dict (dict): Nested ditionary mapping patient names to
            dict of outcomes: labels
        n (int): Number of splits to generate.

    Returns:
        List of patient splits
    """
    patient_list = list(patients_dict.keys())
    shuffle(patient_list)
    return list(sf.util.split_list(patient_list, n))

# -----------------------------------------------------------------------------


class Dataset:
    """Supervises organization and processing of slides, tfrecords, and tiles.

    Datasets can be comprised of one or more sources, where a source is a
    combination of slides and any associated regions of interest (ROI) and
    extracted image tiles (stored as TFRecords or loose images).

    Datasets can be created in two ways: either by loading one dataset source,
    or by loading a dataset configuration that contains information about
    multiple dataset sources.

    For the first approach, the dataset source configuration is provided via
    keyword arguments (``tiles``, ``tfrecords``, ``slides``, and ``roi``).
    Each is a path to a directory containing the respective data.

    For the second approach, the first argument ``config`` is either a nested
    dictionary containing the configuration for multiple dataset sources, or
    a path to a JSON file with this information. The second argument is a list
    of dataset sources to load (keys from the ``config`` dictionary).

    With either approach, slide/patient-level annotations are provided through
    the ``annotations`` keyword argument, which can either be a path to a CSV
    file, or a pandas DataFrame, which must contain at minimum the column
    '`patient`'.
    """

    def __init__(
        self,
        config: Optional[Union[str, Dict[str, Dict[str, str]]]] = None,
        sources: Optional[Union[str, List[str]]] = None,
        tile_px: Optional[int] = None,
        tile_um: Optional[Union[str, int]] = None,
        *,
        tfrecords: Optional[str] = None,
        tiles: Optional[str] = None,
        roi: Optional[str] = None,
        slides: Optional[str] = None,
        filters: Optional[Dict] = None,
        filter_blank: Optional[Union[List[str], str]] = None,
        annotations: Optional[Union[str, pd.DataFrame]] = None,
        min_tiles: int = 0,
    ) -> None:
        """Initialize a Dataset to organize processed images.

        Examples
            Load a dataset via keyword arguments.

                .. code-block:: python

                    dataset = Dataset(
                        tfrecords='../path',
                        slides='../path',
                        annotations='../file.csv'
                    )

            Load a dataset configuration file and specify dataset source(s).

                .. code-block:: python

                    dataset = Dataset(
                        config='../path/to/config.json',
                        sources=['Lung_Adeno', 'Lung_Squam'],
                        annotations='../file.csv
                    )

        Args:
            config (str, dict): Either a dictionary or a path to a JSON file.
                If a dictionary, keys should be dataset source names, and
                values should be dictionaries containing the keys 'tiles',
                'tfrecords', 'roi', and/or 'slides', specifying directories for
                each dataset source. If `config` is a str, it should be a path
                to a JSON file containing a dictionary with the same
                formatting. If None, tiles, tfrecords, roi and/or slides should
                be manually provided via keyword arguments. Defaults to None.
            sources (List[str]): List of dataset sources to include from
                configuration. If not provided, will use all sources in the
                provided configuration. Defaults to None.
            tile_px (int): Tile size in pixels.
            tile_um (int or str): Tile size in microns (int) or magnification
                (str, e.g. "20x").

        Keyword args:
            filters (dict, optional): Dataset filters to use for
                selecting slides. See :meth:`slideflow.Dataset.filter` for
                more information. Defaults to None.
            filter_blank (list(str) or str, optional): Skip slides that have
                blank values in these patient annotation columns.
                Defaults to None.
            min_tiles (int, optional): Only include slides with this
                many tiles at minimum. Defaults to 0.
            annotations (str or pd.DataFrame, optional): Path
                to annotations file or pandas DataFrame with slide-level
                annotations. Defaults to None.

        Raises:
            errors.SourceNotFoundError: If provided source does not exist
                in the dataset config.
        """
        if isinstance(tile_um, str):
            sf.util.assert_is_mag(tile_um)
            tile_um = tile_um.lower()

        self.tile_px = tile_px
        self.tile_um = tile_um
        self._filters = filters if filters else {}
        if filter_blank is None:
            self._filter_blank = []
        else:
            self._filter_blank = sf.util.as_list(filter_blank)
        self._min_tiles = min_tiles
        self._clip = {}  # type: Dict[str, int]
        self.prob_weights = None  # type: Optional[Dict]
        self._annotations = None  # type: Optional[pd.DataFrame]
        self.annotations_file = None  # type: Optional[str]

        if (any(arg is not None for arg in (tfrecords, tiles, roi, slides))
           and (config is not None or sources is not None)):
            raise ValueError(
                "When initializing a Dataset object via keywords (tiles, "
                "tfrecords, slides, roi), the arguments 'config' and 'sources'"
                " are invalid."
            )
        elif any(arg is not None for arg in (tfrecords, tiles, roi, slides)):
            config = dict(dataset=dict(
                tfrecords=tfrecords, tiles=tiles, roi=roi, slides=slides
            ))
            sources = ['dataset']

        if isinstance(config, str):
            self._config = config
            loaded_config = sf.util.load_json(config)
        else:
            self._config = "<dict>"
            loaded_config = config

        # Read dataset sources from the configuration
        if sources is None:
            raise ValueError("Missing argument 'sources'")
        sources = sources if isinstance(sources, list) else [sources]
        try:
            self.sources = {
                k: v for k, v in loaded_config.items() if k in sources
            }
            self.sources_names = list(self.sources.keys())
        except KeyError:
            sources_list = ', '.join(sources)
            raise errors.SourceNotFoundError(sources_list, config)
        missing_sources = [s for s in sources if s not in self.sources]
        if len(missing_sources):
            log.warn(
                "The following sources were not found in the dataset "
                f"configuration: {', '.join(missing_sources)}"
            )
        # Create labels for each source based on tile size
        if (tile_px is not None) and (tile_um is not None):
            label = sf.util.tile_size_label(tile_px, tile_um)
        else:
            label = None
        for source in self.sources:
            self.sources[source]['label'] = label

        # Load annotations
        if annotations is not None:
            self.load_annotations(annotations)

    def __repr__(self) -> str:   # noqa D105
        _b = "Dataset(config={!r}, sources={!r}, tile_px={!r}, tile_um={!r})"
        return _b.format(
            self._config,
            self.sources_names,
            self.tile_px,
            self.tile_um
        )

    @property
    def annotations(self) -> Optional[pd.DataFrame]:
        """Pandas DataFrame of all loaded clinical annotations."""
        return self._annotations

    @property
    def num_tiles(self) -> int:
        """Number of tiles in tfrecords after filtering/clipping."""
        tfrecords = self.tfrecords()
        m = self.manifest()
        if not all([tfr in m for tfr in tfrecords]):
            self.update_manifest()
        n_tiles = [
            m[tfr]['total'] if 'clipped' not in m[tfr] else m[tfr]['clipped']
            for tfr in tfrecords
        ]
        return sum(n_tiles)

    @property
    def filters(self) -> Dict:
        """Returns the active filters, if any."""
        return self._filters

    @property
    def filter_blank(self) -> Union[str, List[str]]:
        """Returns the active filter_blank filter, if any."""
        return self._filter_blank

    @property
    def min_tiles(self) -> int:
        """Returns the active min_tiles filter, if any (defaults to 0)."""
        return self._min_tiles

    @property
    def filtered_annotations(self) -> pd.DataFrame:
        """Pandas DataFrame of clinical annotations, after filtering."""
        if self.annotations is not None:
            f_ann = self.annotations

            # Only return slides with annotation values specified in "filters"
            if self.filters:
                for filter_key in self.filters.keys():
                    if filter_key not in f_ann.columns:
                        raise IndexError(
                            f"Filter header {filter_key} not in annotations."
                        )
                    filter_vals = sf.util.as_list(self.filters[filter_key])
                    f_ann = f_ann.loc[f_ann[filter_key].isin(filter_vals)]

            # Filter out slides that are blank in a given annotation
            # column ("filter_blank")
            if self.filter_blank and self.filter_blank != [None]:
                for fb in self.filter_blank:
                    if fb not in f_ann.columns:
                        raise errors.DatasetFilterError(
                            f"Header {fb} not found in annotations."
                        )
                    f_ann = f_ann.loc[f_ann[fb].notna()]
                    f_ann = f_ann.loc[~f_ann[fb].isin(sf.util.EMPTY)]

            # Filter out slides that do not meet minimum number of tiles
            if self.min_tiles:
                manifest = self.manifest(key='name', filter=False)
                man_slides = [s for s in manifest
                              if manifest[s]['total'] >= self.min_tiles]
                f_ann = f_ann.loc[f_ann.slide.isin(man_slides)]

            return f_ann
        else:
            return None

    @property
    def img_format(self) -> Optional[str]:
        """Format of images stored in TFRecords (jpg/png).

        Verifies all tfrecords share the same image format.

        Returns:
            str: Image format of tfrecords (PNG or JPG), or None if no
            tfrecords have been extracted.
        """
        return self.verify_img_format(progress=False)

    def _tfrecords_set(self, source: str):
        if source not in self.sources:
            raise ValueError(f"Unrecognized dataset source {source}")
        config = self.sources[source]
        return 'tfrecords' in config and config['tfrecords']

    def _tiles_set(self, source: str):
        if source not in self.sources:
            raise ValueError(f"Unrecognized dataset source {source}")
        config = self.sources[source]
        return 'tiles' in config and config['tiles']

    def _roi_set(self, source: str):
        if source not in self.sources:
            raise ValueError(f"Unrecognized dataset source {source}")
        config = self.sources[source]
        return 'roi' in config and config['roi']

    def _slides_set(self, source: str):
        if source not in self.sources:
            raise ValueError(f"Unrecognized dataset source {source}")
        config = self.sources[source]
        return 'slides' in config and config['slides']

    def _assert_size_matches_hp(self, hp: Union[Dict, "ModelParams"]) -> None:
        """Check if dataset tile size (px/um) matches the given parameters."""

        if isinstance(hp, dict):
            hp_px = hp['tile_px']
            hp_um = hp['tile_um']
        elif isinstance(hp, sf.ModelParams):
            hp_px = hp.tile_px
            hp_um = hp.tile_um
        else:
            raise ValueError(f"Unrecognized hyperparameter type {type(hp)}")
        if self.tile_px != hp_px or self.tile_um != hp_um:
            d_sz = f'({self.tile_px}px, tile_um={self.tile_um})'
            m_sz = f'({hp_px}px, tile_um={hp_um})'
            raise ValueError(
                f"Dataset tile size {d_sz} does not match model {m_sz}"
            )

    def load_annotations(self, annotations: Union[str, pd.DataFrame]) -> None:
        """Load annotations.

        Args:
            annotations (Union[str, pd.DataFrame]): Either path to annotations
                in CSV format, or a pandas DataFrame.

        Raises:
            errors.AnnotationsError: If annotations are incorrectly formatted.
        """
        if isinstance(annotations, str):
            if not exists(annotations):
                raise errors.AnnotationsError(
                    f'Unable to find annotations file {annotations}'
                )
            try:
                ann_df = pd.read_csv(annotations, dtype=str)
                ann_df.fillna('', inplace=True)
                self._annotations = ann_df
                self.annotations_file = annotations
            except pd.errors.EmptyDataError:
                log.error(f"Unable to load empty annotations {annotations}")
        elif isinstance(annotations, pd.core.frame.DataFrame):
            annotations.fillna('', inplace=True)
            self._annotations = annotations
        else:
            raise errors.AnnotationsError(
                'Invalid annotations format; expected path or DataFrame'
            )

        # Check annotations
        assert self.annotations is not None
        if len(self.annotations.columns) == 1:
            raise errors.AnnotationsError(
                "Only one annotations column detected (is it in CSV format?)"
            )
        if len(self.annotations.columns) != len(set(self.annotations.columns)):
            raise errors.AnnotationsError(
                "Annotations file has duplicate headers; all must be unique"
            )
        if 'patient' not in self.annotations.columns:
            raise errors.AnnotationsError(
                "Patient identifier 'patient' missing in annotations."
            )
        if 'slide' not in self.annotations.columns:
            if isinstance(annotations, pd.DataFrame):
                raise errors.AnnotationsError(
                    "If loading annotations from a pandas DataFrame,"
                    " must include column 'slide' containing slide names."
                )
            log.info("Column 'slide' missing in annotations.")
            log.info("Attempting to associate patients with slides...")
            self.update_annotations_with_slidenames(annotations)
            self.load_annotations(annotations)

        # Check for duplicate slides
        ann = self.annotations.loc[self.annotations.slide.isin(self.slides())]
        if not ann.slide.is_unique:
            dup_slide_idx = ann.slide.duplicated()
            dup_slides = ann.loc[dup_slide_idx].slide.to_numpy().tolist()
            raise errors.DatasetError(
                f"Duplicate slides found in annotations: {dup_slides}."
            )

    def balance(
        self,
        headers: Optional[Union[str, List[str]]] = None,
        strategy: Optional[str] = 'category',
        *,
        force: bool = False,
    ) -> "Dataset":
        """Return a dataset with mini-batch balancing configured.

        Mini-batch balancing can be configured at tile, slide, patient, or
        category levels.

        Balancing information is saved to the attribute ``prob_weights``, which
        is used by the interleaving dataloaders when sampling from tfrecords
        to create a batch.

        Tile level balancing will create prob_weights reflective of the number
        of tiles per slide, thus causing the batch sampling to mirror random
        sampling from the entire population of  tiles (rather than randomly
        sampling from slides).

        Slide level balancing is the default behavior, where batches are
        assembled by randomly sampling from each slide/tfrecord with equal
        probability. This balancing behavior would be the same as no balancing.

        Patient level balancing is used to randomly sample from individual
        patients with equal probability. This is distinct from slide level
        balancing, as some patients may have multiple slides per patient.

        Category level balancing takes a list of annotation header(s) and
        generates prob_weights such that each category is sampled equally.
        This requires categorical outcomes.

        Args:
            headers (list of str, optional): List of annotation headers if
                balancing by category. Defaults to None.
            strategy (str, optional): 'tile', 'slide', 'patient' or 'category'.
                Create prob_weights used to balance dataset batches to evenly
                distribute slides, patients, or categories in a given batch.
                Tile-level balancing generates prob_weights reflective of the
                total number of tiles in a slide. Defaults to 'category.'
            force (bool, optional): If using category-level balancing,
                interpret all headers as categorical variables, even if the
                header appears to be a float.

        Returns:
            balanced :class:`slideflow.Dataset` object.
        """
        ret = copy.deepcopy(self)
        manifest = ret.manifest()
        tfrecords = ret.tfrecords()
        slides = [path_to_name(tfr) for tfr in tfrecords]
        totals = {
            tfr: (manifest[tfr]['total']
                  if 'clipped' not in manifest[tfr]
                  else manifest[tfr]['clipped'])
            for tfr in tfrecords
        }
        if not tfrecords:
            raise errors.DatasetBalanceError(
                "Unable to balance; no tfrecords found."
            )

        if strategy == 'none' or strategy is None:
            return self
        if strategy == 'tile':
            ret.prob_weights = {
                tfr: totals[tfr] / sum(totals.values()) for tfr in tfrecords
            }
        if strategy == 'slide':
            ret.prob_weights = {tfr: 1/len(tfrecords) for tfr in tfrecords}
        if strategy == 'patient':
            pts = ret.patients()  # Maps tfrecords to patients
            r_pts = {}  # Maps patients to list of tfrecords
            for slide in pts:
                if slide not in slides:
                    continue
                if pts[slide] not in r_pts:
                    r_pts[pts[slide]] = [slide]
                else:
                    r_pts[pts[slide]] += [slide]
            ret.prob_weights = {
                tfr: 1/(len(r_pts) * len(r_pts[pts[path_to_name(tfr)]]))
                for tfr in tfrecords
            }
        if strategy == 'category':
            if headers is None:
                raise ValueError('Category balancing requires headers.')
            # Ensure that header is not type 'float'
            headers = sf.util.as_list(headers)
            if any(ret.is_float(h) for h in headers) and not force:
                raise errors.DatasetBalanceError(
                    f"Headers {','.join(headers)} appear to be `float`. "
                    "Categorical outcomes required for balancing. "
                    "To force balancing with these outcomes, pass "
                    "`force=True` to Dataset.balance()"
                )
            labels, _ = ret.labels(headers, use_float=False)
            cats = {}  # type: Dict[str, Dict]
            cat_prob = {}
            tfr_cats = {}  # type: Dict[str, str]
            for tfrecord in tfrecords:
                slide = path_to_name(tfrecord)
                balance_cat = sf.util.as_list(labels[slide])
                balance_cat_str = '-'.join(map(str, balance_cat))
                tfr_cats[tfrecord] = balance_cat_str
                tiles = totals[tfrecord]
                if balance_cat_str not in cats:
                    cats.update({balance_cat_str: {
                        'num_slides': 1,
                        'num_tiles': tiles
                    }})
                else:
                    cats[balance_cat_str]['num_slides'] += 1
                    cats[balance_cat_str]['num_tiles'] += tiles
            for category in cats:
                min_cat_slides = min([
                    cats[i]['num_slides'] for i in cats
                ])
                slides_in_cat = cats[category]['num_slides']
                cat_prob[category] = min_cat_slides / slides_in_cat
            total_prob = sum([cat_prob[tfr_cats[tfr]] for tfr in tfrecords])
            ret.prob_weights = {
                tfr: cat_prob[tfr_cats[tfr]]/total_prob for tfr in tfrecords
            }
        return ret

    def build_index(self, force: bool = True) -> None:
        """Build index files for TFRecords.

        Args:
            force (bool): Force re-build existing indices.

        Returns:
            None
        """
        if force:
            index_to_update = self.tfrecords()
            # Remove existing indices
            for tfr in self.tfrecords():
                index = tfrecord2idx.find_index(tfr)
                if index:
                    os.remove(index)
        else:
            index_to_update = []
            for tfr in self.tfrecords():
                index = tfrecord2idx.find_index(tfr)
                if not index:
                    index_to_update.append(tfr)
                elif (not tfrecord2idx.index_has_locations(index)
                      and sf.io.tfrecord_has_locations(tfr)):
                    os.remove(index)
                    index_to_update.append(tfr)
            if not index_to_update:
                return

        index_fn = partial(_create_index, force=force)
        pool = mp.Pool(
            sf.util.num_cpu(),
            initializer=sf.util.set_ignore_sigint
        )
        for _ in track(pool.imap_unordered(index_fn, index_to_update),
                       description=f'Updating index files...',
                       total=len(index_to_update),
                       transient=True):
            pass
        pool.close()

    def cell_segmentation(
        self,
        diam_um: float,
        dest: str,
        *,
        model: Union["cellpose.models.Cellpose", str] = 'cyto2',
        window_size: int = 256,
        diam_mean: Optional[int] = None,
        qc: Optional[str] = None,
        qc_kwargs: Optional[dict] = None,
        buffer: Optional[str] = None,
        q_size: int = 2,
        force: bool = False,
        save_centroid: bool = True,
        save_flow: bool = False,
        **kwargs
    ) -> None:
        """Perform cell segmentation on slides, saving segmentation masks.

        Args:
            diam_um (int, optional): Cell segmentation diameter, in microns.
            dest (str): Destination in which to save cell segmentation masks.

        Keyword args:
            batch_size (int): Batch size for cell segmentation. Defaults to 8.
            cp_thresh (float): Cell probability threshold. All pixels with
                value above threshold kept for masks, decrease to find more and
                larger masks. Defaults to 0.
            diam_mean (int, optional): Cell diameter to detect, in pixels
                (without image resizing). If None, uses Cellpose defaults (17
                for the 'nuclei' model, 30 for all others).
            downscale (float): Factor by which to downscale generated masks
                after calculation. Defaults to None (keep masks at original
                size).
            flow_threshold (float): Flow error threshold (all cells with errors
                below threshold are kept). Defaults to 0.4.
            gpus (int, list(int)): GPUs to use for cell segmentation.
                Defaults to 0 (first GPU).
            interp (bool): Interpolate during 2D dynamics. Defaults to True.
            qc (str): Slide-level quality control method to use before
                performing cell segmentation. Defaults to "Otsu".
            model (str, :class:`cellpose.models.Cellpose`): Cellpose model to
                use for cell segmentation. May be any valid cellpose model.
                Defaults to 'cyto2'.
            mpp (float): Microns-per-pixel at which cells should be segmented.
                Defaults to 0.5.
            num_workers (int, optional): Number of workers.
                Defaults to 2 * num_gpus.
            save_centroid (bool): Save mask centroids. Increases memory
                utilization slightly. Defaults to True.
            save_flow (bool): Save flow values for the whole-slide image.
                Increases memory utilization. Defaults to False.
            sources (List[str]): List of dataset sources to include from
                configuration file.
            tile (bool): Tiles image to decrease GPU/CPU memory usage.
                Defaults to True.
            verbose (bool): Verbose log output at the INFO level.
                Defaults to True.
            window_size (int): Window size at which to segment cells across
                a whole-slide image. Defaults to 256.

        Returns:
            None
        """
        from slideflow.cellseg import segment_slide

        if qc_kwargs is None:
            qc_kwargs = {}

        slide_list = self.slide_paths()
        if not force:
            n_all = len(slide_list)
            slide_list = [
                s for s in slide_list
                if not exists(
                    join(dest, sf.util.path_to_name(s)+'-masks.zip')
                )
            ]
            n_skipped = n_all - len(slide_list)
            if n_skipped:
                log.info("Skipping {} slides (masks already generated)".format(
                    n_skipped
                ))
        if slide_list:
            log.info(f"Segmenting cells for {len(slide_list)} slides.")
        else:
            log.info("No slides found.")
            return

        if diam_mean is None:
            diam_mean = 30 if model != 'nuclei' else 17
        tile_um = int(window_size * (diam_um / diam_mean))
        pb = TileExtractionProgress()
        speed_task = pb.add_task(
            "Speed: ", progress_type="speed", total=None
        )
        slide_task = pb.add_task(
            "Slides: ", progress_type="slide_progress", total=len(slide_list)
        )
        q = Queue()  # type: Queue
        if buffer:
            thread = threading.Thread(
                target=_fill_queue,
                args=(slide_list, q, q_size, buffer))
            thread.start()

        pb.start()
        with sf.util.cleanup_progress(pb):
            while True:
                slide_path = q.get()
                if slide_path is None:
                    q.task_done()
                    break
                wsi = sf.WSI(
                    slide_path,
                    tile_px=window_size,
                    tile_um=tile_um,
                    verbose=False
                )
                if qc is not None:
                    wsi.qc(qc, **qc_kwargs)
                segment_task = pb.add_task(
                    "Segmenting... ",
                    progress_type="slide_progress",
                    total=wsi.estimated_num_tiles
                )
                # Perform segmentation and save
                segmentation = segment_slide(
                    wsi,
                    pb=pb,
                    pb_tasks=[speed_task, segment_task],
                    show_progress=False,
                    model=model,
                    diam_mean=diam_mean,
                    save_flow=save_flow,
                    **kwargs)
                mask_dest = dest if dest is not None else dirname(slide_path)
                segmentation.save(
                    join(mask_dest, f'{wsi.name}-masks.zip'),
                    flows=save_flow,
                    centroids=save_centroid)
                pb.advance(slide_task)
                pb.remove_task(segment_task)

                if buffer:
                    _debuffer_slide(slide_path)
                q.task_done()
        if buffer:
            thread.join()

    def check_duplicates(
        self,
        dataset: Optional["Dataset"] = None,
        px: int = 64,
        mse_thresh: int = 100
    ) -> List[Tuple[str, str]]:
        """Check for duplicate slides by comparing slide thumbnails.

        Args:
            dataset (`slideflow.Dataset`, optional): Also check for
                duplicate slides between this dataset and the provided dataset.
            px (int): Pixel size at which to compare thumbnails.
                Defaults to 64.
            mse_thresh (int): MSE threshold below which an image pair is
                considered duplicate. Defaults to 100.

        Returns:
            List[str], optional: List of path pairs of potential duplicates.
        """
        import cv2

        thumbs = {}
        dups = []

        def mse(A, B):
            """Calulate the mean squared error between two image matrices."""
            err = np.sum((A.astype("float") - B.astype("float")) ** 2)
            err /= float(A.shape[0] * A.shape[1])
            return err

        def img_from_path(path):
            """Read and resize an image."""
            img = cv2.imdecode(
                np.fromfile(path, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED)
            img = img[..., 0:3]
            return cv2.resize(img,
                              dsize=(px, px),
                              interpolation=cv2.INTER_CUBIC)

        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(join(temp_dir, 'this_dataset'))
            self.thumbnails(join(temp_dir, 'this_dataset'))
            if dataset:
                os.makedirs(join(temp_dir, 'other_dataset'))
                dataset.thumbnails(join(temp_dir, 'other_dataset'))
            for subdir in os.listdir(temp_dir):
                files = os.listdir(join(temp_dir, subdir))
                for file in tqdm(files, desc="Scanning for duplicates..."):
                    if dataset and subdir == 'other_dataset':
                        wsi_path = dataset.find_slide(slide=path_to_name(file))
                    else:
                        wsi_path = self.find_slide(slide=path_to_name(file))
                    assert wsi_path is not None
                    img = img_from_path(join(temp_dir, subdir, file))
                    thumbs[wsi_path] = img

                    # Check if this thumbnail has a duplicate
                    for existing_img in thumbs:
                        if wsi_path != existing_img:
                            img2 = thumbs[existing_img]
                            img_mse = mse(img, img2)
                            if img_mse < mse_thresh:
                                tqdm.write(
                                    'Possible duplicates: '
                                    '{} and {} (MSE: {})'.format(
                                        wsi_path,
                                        existing_img,
                                        mse(img, img2)
                                    )
                                )
                                dups += [(wsi_path, existing_img)]
        if not dups:
            log.info("No duplicates found.")
        else:
            log.info(f"{len(dups)} possible duplicates found.")
        return dups

    def clear_filters(self) -> "Dataset":
        """Return a dataset with all filters cleared.

        Returns:
            :class:`slideflow.Dataset` object.

        """
        ret = copy.deepcopy(self)
        ret._filters = {}
        ret._filter_blank = []
        ret._min_tiles = 0
        return ret

    def clip(
        self,
        max_tiles: int = 0,
        strategy: Optional[str] = None,
        headers: Optional[List[str]] = None
    ) -> "Dataset":
        """Return a dataset with TFRecords clipped to a max number of tiles.

        Clip the number of tiles per tfrecord to a given maximum value and/or
        to the min number of tiles per patient or category.

        Args:
            max_tiles (int, optional): Clip the maximum number of tiles per
                tfrecord to this number. Defaults to 0 (do not perform
                tfrecord-level clipping).
            strategy (str, optional): 'slide', 'patient', or 'category'.
                Clip the maximum number of tiles to the minimum tiles seen
                across slides, patients, or categories. If 'category', headers
                must be provided. Defaults to None (do not perform group-level
                clipping).
            headers (list of str, optional): List of annotation headers to use
                if clipping by minimum category count (strategy='category').
                Defaults to None.

        Returns:
            clipped :class:`slideflow.Dataset` object.

        """
        if strategy == 'category' and not headers:
            raise errors.DatasetClipError(
                "headers must be provided if clip strategy is 'category'."
            )
        if not max_tiles and strategy is None:
            return self.unclip()

        ret = copy.deepcopy(self)
        manifest = ret.manifest()
        tfrecords = ret.tfrecords()
        slides = [path_to_name(tfr) for tfr in tfrecords]
        totals = {tfr: manifest[tfr]['total'] for tfr in tfrecords}

        if not tfrecords:
            raise errors.DatasetClipError("No tfrecords found.")
        if strategy == 'slide':
            if max_tiles:
                clip = min(min(totals.values()), max_tiles)
            else:
                clip = min(totals.values())
            ret._clip = {
                tfr: (clip if totals[tfr] > clip else totals[tfr])
                for tfr in manifest
            }
        elif strategy == 'patient':
            patients = ret.patients()  # Maps slide name to patient
            rev_patients = {}  # Will map patients to list of slide names
            slide_totals = {path_to_name(tfr): t for tfr, t in totals.items()}
            for slide in patients:
                if slide not in slides:
                    continue
                if patients[slide] not in rev_patients:
                    rev_patients[patients[slide]] = [slide]
                else:
                    rev_patients[patients[slide]] += [slide]
            tiles_per_patient = {
                pt: sum([slide_totals[slide] for slide in slide_list])
                for pt, slide_list in rev_patients.items()
            }
            if max_tiles:
                clip = min(min(tiles_per_patient.values()), max_tiles)
            else:
                clip = min(tiles_per_patient.values())
            ret._clip = {
                tfr: (clip
                      if slide_totals[path_to_name(tfr)] > clip
                      else totals[tfr])
                for tfr in manifest
            }
        elif strategy == 'category':
            if headers is None:
                raise ValueError("Category clipping requires arg `headers`")
            labels, _ = ret.labels(headers, use_float=False)
            categories = {}
            cat_fraction = {}
            tfr_cats = {}
            for tfrecord in tfrecords:
                slide = path_to_name(tfrecord)
                balance_category = sf.util.as_list(labels[slide])
                balance_cat_str = '-'.join(map(str, balance_category))
                tfr_cats[tfrecord] = balance_cat_str
                tiles = totals[tfrecord]
                if balance_cat_str not in categories:
                    categories[balance_cat_str] = tiles
                else:
                    categories[balance_cat_str] += tiles

            for category in categories:
                min_cat_count = min([categories[i] for i in categories])
                cat_fraction[category] = min_cat_count / categories[category]
            ret._clip = {
                tfr: int(totals[tfr] * cat_fraction[tfr_cats[tfr]])
                for tfr in manifest
            }
        elif max_tiles:
            ret._clip = {
                tfr: (max_tiles if totals[tfr] > max_tiles else totals[tfr])
                for tfr in manifest
            }
        return ret

    def convert_xml_rois(self):
        """Convert ImageScope XML ROI files to QuPath format CSV ROI files."""
        n_converted = 0
        xml_list = []
        for source in self.sources:
            if self._roi_set(source):
                xml_list = glob(join(self.sources[source]['roi'], "*.xml"))
                if len(xml_list) == 0:
                    raise errors.DatasetError(
                        'No XML files found. Check dataset configuration.'
                    )
                for xml in xml_list:
                    try:
                        sf.slide.utils.xml_to_csv(xml)
                    except errors.ROIError as e:
                        log.warning(f"Failed to convert XML roi {xml}: {e}")
                    else:
                        n_converted += 1
        log.info(f'Converted {n_converted} XML ROIs -> CSV')

    def get_tile_dataframe(
        self,
        roi_method: str = 'auto',
        stride_div: int = 1,
    ) -> pd.DataFrame:
        """Generate a pandas dataframe with tile-level ROI labels.

        Returns:
            Pandas dataframe of all tiles, with the following columns:
            - ``loc_x``: X-coordinate of tile center
            - ``loc_y``: Y-coordinate of tile center
            - ``grid_x``: X grid index of the tile
            - ``grid_y``: Y grid index of the tile
            - ``roi_name``: Name of the ROI if tile is in an ROI, else None
            - ``roi_desc``: Description of the ROI if tile is in ROI, else None
            - ``label``: ROI label, if present.

        """
        df = None
        with mp.Pool(4, initializer=sf.util.set_ignore_sigint) as pool:
            fn = partial(
                _get_tile_df,
                tile_px=self.tile_px,
                tile_um=self.tile_um,
                rois=self.rois(),
                stride_div=stride_div,
                roi_method=roi_method
            )
            for _df in track(pool.imap_unordered(fn, self.slide_paths()),
                            description=f'Building...',
                            total=len(self.slide_paths()),
                            transient=True):
                if df is None:
                    df = _df
                else:
                    df = pd.concat([df, _df], axis=0, join='outer')

        return df

    def get_unique_roi_labels(self, allow_empty: bool = False) -> List[str]:
        """Get a list of unique ROI labels for all slides in this dataset."""

        # Get a list of unique labels.
        roi_unique_labels = []
        for roi in self.rois():
            _df = pd.read_csv(roi)
            if 'label' not in _df.columns:
                continue
            unique = [
                l for l in _df.label.unique().tolist()
                if (l not in roi_unique_labels)
            ]
            roi_unique_labels += unique
        without_nan = sorted([
            l for l in roi_unique_labels
            if (not isinstance(l, float) or not np.isnan(l))
        ])
        if allow_empty and (len(roi_unique_labels) > len(without_nan)):
                return without_nan + [None]
        else:
            return without_nan

    def extract_cells(
        self,
        masks_path: str,
        **kwargs
    ) -> Dict[str, SlideReport]:
        """Extract cell images from slides, with a tile at each cell centroid.

        Requires that cells have already been segmented with
        ``Dataset.cell_segmentation()``.

        Args:
            masks_path (str): Location of saved segmentation masks.

        Keyword Args:
            apply_masks (bool): Apply cell segmentation masks to the extracted
                tiles. Defaults to True.
            **kwargs: All other keyword arguments for
                :meth:`Dataset.extract_tiles()`.

        Returns:
            Dictionary mapping slide paths to each slide's SlideReport
            (:class:`slideflow.slide.report.SlideReport`)

        """
        from slideflow.cellseg.seg_utils import ApplySegmentation

        # Add WSI segmentation as slide-level transformation.
        qc = [] if 'qc' not in kwargs else kwargs['qc']
        if not isinstance(qc, list):
            qc = [qc]
        qc.append(ApplySegmentation(masks_path))
        kwargs['qc'] = qc

        # Extract tiles from segmentation centroids.
        return self.extract_tiles(
            from_centroids=True,
            **kwargs
        )

    def extract_tiles(
        self,
        *,
        save_tiles: bool = False,
        save_tfrecords: bool = True,
        source: Optional[str] = None,
        stride_div: int = 1,
        enable_downsample: bool = True,
        roi_method: str = 'auto',
        roi_filter_method: Union[str, float] = 'center',
        skip_extracted: bool = True,
        tma: bool = False,
        randomize_origin: bool = False,
        buffer: Optional[str] = None,
        q_size: int = 2,
        qc: Optional[Union[str, Callable, List[Callable]]] = None,
        report: bool = True,
        use_edge_tiles: bool = False,
        artifact_rois: Optional[Union[List[str], str]] = list(),
        mpp_override: Optional[float] = None,
        **kwargs: Any
    ) -> Dict[str, SlideReport]:
        r"""Extract tiles from a group of slides.

        Extracted tiles are saved either loose image or in TFRecord format.

        Extracted tiles are either saved in TFRecord format
        (``save_tfrecords=True``, default) or as loose \*.jpg / \*.png images
        (``save_tiles=True``). TFRecords or image tiles are saved in the
        the tfrecord and tile directories configured by
        :class:`slideflow.Dataset`.

        Keyword Args:
            save_tiles (bool, optional): Save tile images in loose format.
                Defaults to False.
            save_tfrecords (bool): Save compressed image data from
                extracted tiles into TFRecords in the corresponding TFRecord
                directory. Defaults to True.
            source (str, optional): Name of dataset source from which to select
                slides for extraction. Defaults to None. If not provided, will
                default to all sources in project.
            stride_div (int): Stride divisor for tile extraction.
                A stride of 1 will extract non-overlapping tiles.
                A stride_div of 2 will extract overlapping tiles, with a stride
                equal to 50% of the tile width. Defaults to 1.
            enable_downsample (bool): Enable downsampling for slides.
                This may result in corrupted image tiles if downsampled slide
                layers are corrupted or incomplete. Defaults to True.
            roi_method (str): Either 'inside', 'outside', 'auto', or 'ignore'.
                Determines how ROIs are used to extract tiles.
                If 'inside' or 'outside', will extract tiles in/out of an ROI,
                and skip the slide if an ROI is not available.
                If 'auto', will extract tiles inside an ROI if available,
                and across the whole-slide if no ROI is found.
                If 'ignore', will extract tiles across the whole-slide
                regardless of whether an ROI is available.
                Defaults to 'auto'.
            roi_filter_method (str or float): Method of filtering tiles with
                ROIs. Either 'center' or float (0-1). If 'center', tiles are
                filtered with ROIs based on the center of the tile. If float,
                tiles are filtered based on the proportion of the tile inside
                the ROI, and ``roi_filter_method`` is interpreted as a
                threshold. If the proportion of a tile inside the ROI is
                greater than this number, the tile is included. For example,
                if ``roi_filter_method=0.7``, a tile that is 80% inside of an
                ROI will be included, and a tile that is 50% inside of an ROI
                will be excluded. Defaults to 'center'.
            skip_extracted (bool): Skip slides that have already
                been extracted. Defaults to True.
            tma (bool): Reads slides as Tumor Micro-Arrays (TMAs).
                Deprecated argument; all slides are now read as standard WSIs.
            randomize_origin (bool): Randomize pixel starting
                position during extraction. Defaults to False.
            buffer (str, optional): Slides will be copied to this directory
                before extraction. Defaults to None. Using an SSD or ramdisk
                buffer vastly improves tile extraction speed.
            q_size (int): Size of queue when using a buffer.
                Defaults to 2.
            qc (str, optional): 'otsu', 'blur', 'both', or None. Perform blur
                detection quality control - discarding tiles with detected
                out-of-focus regions or artifact - and/or otsu's method.
                Increases tile extraction time. Defaults to None.
            report (bool): Save a PDF report of tile extraction.
                Defaults to True.
            normalizer (str, optional): Normalization strategy.
                Defaults to None.
            normalizer_source (str, optional): Stain normalization preset or
                path to a source image. Valid presets include 'v1', 'v2', and
                'v3'. If None, will use the default present ('v3').
                Defaults to None.
            whitespace_fraction (float, optional): Range 0-1. Discard tiles
                with this fraction of whitespace. If 1, will not perform
                whitespace filtering. Defaults to 1.
            whitespace_threshold (int, optional): Range 0-255. Defaults to 230.
                Threshold above which a pixel (RGB average) is whitespace.
            grayspace_fraction (float, optional): Range 0-1. Defaults to 0.6.
                Discard tiles with this fraction of grayspace.
                If 1, will not perform grayspace filtering.
            grayspace_threshold (float, optional): Range 0-1. Defaults to 0.05.
                Pixels in HSV format with saturation below this threshold are
                considered grayspace.
            img_format (str, optional): 'png' or 'jpg'. Defaults to 'jpg'.
                Image format to use in tfrecords. PNG (lossless) for fidelity,
                JPG (lossy) for efficiency.
            shuffle (bool, optional): Shuffle tiles prior to storage in
                tfrecords. Defaults to True.
            num_threads (int, optional): Number of worker processes for each
                tile extractor. When using cuCIM slide reading backend,
                defaults to the total number of available CPU cores, using the
                'fork' multiprocessing method. With Libvips, this defaults to
                the total number of available CPU cores or 32, whichever is
                lower, using 'spawn' multiprocessing.
            qc_blur_radius (int, optional): Quality control blur radius for
                out-of-focus area detection. Used if qc=True. Defaults to 3.
            qc_blur_threshold (float, optional): Quality control blur threshold
                for detecting out-of-focus areas. Only used if qc=True.
                Defaults to 0.1
            qc_filter_threshold (float, optional): Float between 0-1. Tiles
                with more than this proportion of blur will be discarded.
                Only used if qc=True. Defaults to 0.6.
            qc_mpp (float, optional): Microns-per-pixel indicating image
                magnification level at which quality control is performed.
                Defaults to mpp=4 (effective magnification 2.5 X)
            dry_run (bool, optional): Determine tiles that would be extracted,
                but do not export any images. Defaults to None.
            max_tiles (int, optional): Only extract this many tiles per slide.
                Defaults to None.
            use_edge_tiles (bool): Use edge tiles in extraction. Areas
                outside the slide will be padded white. Defaults to False.
            artifact_rois (list(str) or str, optional): List of ROI issue labels
                to treat as artifacts. Whenever this is not None, all the ROIs with
                referred label will be inverted with ROI.invert_roi().
                Defaults to an empty list.
            mpp_override (float, optional): Override the microns-per-pixel
                for each slide. If None, will auto-detect microns-per-pixel
                for all slides and raise an error if MPP is not found.
                Defaults to None.

        Returns:
            Dictionary mapping slide paths to each slide's SlideReport
            (:class:`slideflow.slide.report.SlideReport`)
        """
        if tma:
            warnings.warn(
                "tma=True is deprecated and will be removed in a future "
                "version. Tumor micro-arrays are read as standard slides. "
            )
        if not self.tile_px or not self.tile_um:
            raise errors.DatasetError(
                "Dataset tile_px and tile_um must be != 0 to extract tiles"
            )
        if source:
            sources = sf.util.as_list(source)  # type: List[str]
        else:
            sources = list(self.sources.keys())
        all_reports = []
        self.verify_annotations_slides()

        # Ensure self.artifact_rois is a list
        if isinstance(artifact_rois, str):
            artifact_rois = [artifact_rois]

        # Log the active slide reading backend
        col = 'green' if sf.slide_backend() == 'cucim' else 'cyan'
        log.info(f"Slide reading backend: [{col}]{sf.slide_backend()}[/]")

        # Set up kwargs for tile extraction generator and quality control
        qc_kwargs = {k[3:]: v for k, v in kwargs.items() if k[:3] == 'qc_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:3] != 'qc_'}
        sf.slide.log_extraction_params(**kwargs)

        for source in sources:
            log.info(f'Working on dataset source [bold]{source}[/]...')
            if self._roi_set(source):
                roi_dir = self.sources[source]['roi']
            else:
                roi_dir = None
            src_conf = self.sources[source]
            if 'dry_run' not in kwargs or not kwargs['dry_run']:
                if save_tfrecords and not self._tfrecords_set(source):
                    log.error(f"tfrecords path not set for source {source}")
                    continue
                elif save_tfrecords:
                    tfrecord_dir = join(
                        src_conf['tfrecords'],
                        src_conf['label']
                    )
                else:
                    tfrecord_dir = None
                if save_tiles and not self._tiles_set(source):
                    log.error(f"tiles path not set for source {source}")
                    continue
                elif save_tiles:
                    tiles_dir = join(src_conf['tiles'], src_conf['label'])
                else:
                    tiles_dir = None
                if save_tfrecords and not exists(tfrecord_dir):
                    os.makedirs(tfrecord_dir)
                if save_tiles and not exists(tiles_dir):
                    os.makedirs(tiles_dir)
            else:
                save_tfrecords, save_tiles = False, False
                tfrecord_dir, tiles_dir = None, None

            # Prepare list of slides for extraction
            slide_list = self.slide_paths(source=source)

            # Check for interrupted or already-extracted tfrecords
            if skip_extracted and save_tfrecords:
                done = [
                    path_to_name(tfr) for tfr in self.tfrecords(source=source)
                ]
                _dir = tfrecord_dir if tfrecord_dir else tiles_dir
                unfinished = glob(join((_dir), '*.unfinished'))
                interrupted = [path_to_name(marker) for marker in unfinished]
                if len(interrupted):
                    log.info(f'Re-extracting {len(interrupted)} interrupted:')
                    for interrupted_slide in interrupted:
                        log.info(interrupted_slide)
                        if interrupted_slide in done:
                            del done[done.index(interrupted_slide)]

                slide_list = [
                    s for s in slide_list if path_to_name(s) not in done
                ]
                if len(done):
                    log.info(f'Skipping {len(done)} slides; already done.')
            _tail = f"(tile_px={self.tile_px}, tile_um={self.tile_um})"
            log.info(f'Extracting tiles from {len(slide_list)} slides {_tail}')

            # Use multithreading if specified, extracting tiles
            # from all slides in the filtered list
            if len(slide_list):
                q = Queue()  # type: Queue
                # Forking incompatible with some libvips configurations
                ptype = 'spawn' if sf.slide_backend() == 'libvips' else 'fork'
                ctx = mp.get_context(ptype)
                manager = ctx.Manager()
                reports = manager.dict()
                kwargs['report'] = report

                # Use a single shared multiprocessing pool
                if 'num_threads' not in kwargs:
                    num_threads = sf.util.num_cpu()
                    if num_threads is None:
                        num_threads = 8
                    if sf.slide_backend() == 'libvips':
                        num_threads = min(num_threads, 32)
                else:
                    num_threads = kwargs['num_threads']
                if num_threads != 1:
                    pool = kwargs['pool'] = ctx.Pool(
                        num_threads,
                        initializer=sf.util.set_ignore_sigint
                    )
                    qc_kwargs['pool'] = pool
                else:
                    pool = None
                    ptype = None
                log.info(f'Using {num_threads} processes (pool={ptype})')

                # Set up the multiprocessing progress bar
                pb = TileExtractionProgress()
                pb.add_task(
                    "Speed: ",
                    progress_type="speed",
                    total=None)
                slide_task = pb.add_task(
                    "Extracting...",
                    progress_type="slide_progress",
                    total=len(slide_list))

                wsi_kwargs = {
                    'tile_px': self.tile_px,
                    'tile_um': self.tile_um,
                    'stride_div': stride_div,
                    'enable_downsample': enable_downsample,
                    'roi_dir': roi_dir,
                    'roi_method': roi_method,
                    'roi_filter_method': roi_filter_method,
                    'origin': 'random' if randomize_origin else (0, 0),
                    'pb': pb,
                    'use_edge_tiles': use_edge_tiles,
                    'artifact_rois': artifact_rois,
                    'mpp': mpp_override
                }
                extraction_kwargs = {
                    'tfrecord_dir': tfrecord_dir,
                    'tiles_dir': tiles_dir,
                    'reports': reports,
                    'qc': qc,
                    'generator_kwargs': kwargs,
                    'qc_kwargs': qc_kwargs,
                    'wsi_kwargs': wsi_kwargs,
                    'render_thumb': (buffer is not None)
                }
                pb.start()
                with sf.util.cleanup_progress(pb):
                    if buffer:
                        # Start the worker threads
                        thread = threading.Thread(
                            target=_fill_queue,
                            args=(slide_list, q, q_size, buffer))
                        thread.start()

                        # Grab slide path from queue and start extraction
                        while True:
                            path = q.get()
                            if path is None:
                                q.task_done()
                                break
                            _tile_extractor(path, **extraction_kwargs)
                            pb.advance(slide_task)
                            _debuffer_slide(path)
                            q.task_done()
                        thread.join()
                    else:
                        for slide in slide_list:
                            log.info(f'Extracting tiles from {os.path.basename(slide)}')
                            with _handle_slide_errors(slide):
                                wsi = _prepare_slide(
                                    slide,
                                    report_dir=tfrecord_dir,
                                    wsi_kwargs=wsi_kwargs,
                                    qc=qc,
                                    qc_kwargs=qc_kwargs)
                                if wsi is not None:
                                    log.debug(f'Extracting tiles for {wsi.name}')
                                    wsi_report = wsi.extract_tiles(
                                        tfrecord_dir=tfrecord_dir,
                                        tiles_dir=tiles_dir,
                                        **kwargs
                                    )
                                    reports.update({wsi.path: wsi_report})
                                    del wsi
                            pb.advance(slide_task)

                # Generate PDF report.
                if report:
                    log.info('Generating PDF (this may take some time)...', )
                    rep_vals = list(
                        reports.copy().values()
                    )  # type: List[SlideReport]
                    all_reports += rep_vals
                    num_slides = len(slide_list)
                    img_kwargs = defaultdict(lambda: None)  # type: Dict
                    img_kwargs.update(kwargs)
                    img_kwargs = sf.slide.utils._update_kw_with_defaults(img_kwargs)
                    report_meta = types.SimpleNamespace(
                        tile_px=self.tile_px,
                        tile_um=self.tile_um,
                        qc=qc,
                        total_slides=num_slides,
                        slides_skipped=len([r for r in rep_vals if r is None]),
                        roi_method=roi_method,
                        stride=stride_div,
                        gs_frac=img_kwargs['grayspace_fraction'],
                        gs_thresh=img_kwargs['grayspace_threshold'],
                        ws_frac=img_kwargs['whitespace_fraction'],
                        ws_thresh=img_kwargs['whitespace_threshold'],
                        normalizer=img_kwargs['normalizer'],
                        img_format=img_kwargs['img_format']
                    )
                    pdf_report = ExtractionReport(
                        [r for r in rep_vals if r is not None],
                        meta=report_meta,
                        pool=pool
                    )
                    _time = datetime.now().strftime('%Y%m%d-%H%M%S')
                    pdf_dir = tfrecord_dir if tfrecord_dir else ''
                    pdf_report.save(
                        join(pdf_dir, f'tile_extraction_report-{_time}.pdf')
                    )
                    pdf_report.update_csv(
                        join(pdf_dir, 'extraction_report.csv')
                    )
                    warn_path = join(pdf_dir, f'warn_report-{_time}.txt')
                    if pdf_report.warn_txt:
                        with open(warn_path, 'w') as warn_f:
                            warn_f.write(pdf_report.warn_txt)

                # Close the multiprocessing pool.
                if pool is not None:
                    pool.close()

        # Update manifest & rebuild indices
        self.update_manifest(force_update=True)
        self.build_index(True)
        all_reports = [r for r in all_reports if r is not None]
        return {report.path: report for report in all_reports}

    def extract_tiles_from_tfrecords(self, dest: str) -> None:
        """Extract tiles from a set of TFRecords.

        Args:
            dest (str): Path to directory in which to save tile images.
                If None, uses dataset default. Defaults to None.

        """
        for source in self.sources:
            to_extract_tfrecords = self.tfrecords(source=source)
            if dest:
                tiles_dir = dest
            elif self._tiles_set(source):
                tiles_dir = join(self.sources[source]['tiles'],
                                 self.sources[source]['label'])
                if not exists(tiles_dir):
                    os.makedirs(tiles_dir)
            else:
                log.error(f"tiles directory not set for source {source}")
                continue
            for tfr in to_extract_tfrecords:
                sf.io.extract_tiles(tfr, tiles_dir)

    def filter(self, *args: Any, **kwargs: Any) -> "Dataset":
        """Return a filtered dataset.

        This method can either accept a single argument (``filters``) or any
        combination of keyword arguments (``filters``, ``filter_blank``, or
        ``min_tiles``).

        Keyword Args:
            filters (dict, optional): Dictionary used for filtering
                the dataset. Dictionary keys should be column headers in the
                patient annotations, and the values should be the variable
                states to be included in the dataset. For example,
                ``filters={'HPV_status': ['negative', 'positive']}``
                would filter the dataset by the column ``HPV_status`` and only
                include slides with values of either ``'negative'`` or
                ``'positive'`` in this column.
                See `Filtering <https://slideflow.dev/datasets_and_val/#filtering>`_
                for further discussion. Defaults to None.
            filter_blank (list(str) or str, optional): Skip slides that have
                blank values in these patient annotation columns.
                Defaults to None.
            min_tiles (int): Filter out tfrecords that have less than this
                minimum number of tiles. Defaults to 0.

        Returns:
            :class:`slideflow.Dataset`: Dataset with filter added.
        """
        if len(args) == 1 and 'filters' not in kwargs:
            kwargs['filters'] = args[0]
        elif len(args):
            raise ValueError(
                "filter() accepts either one argument (filters), or any "
                "combination of keywords (filters, filter_blank, min_tiles)"
            )
        for kwarg in kwargs:
            if kwarg not in ('filters', 'filter_blank', 'min_tiles'):
                raise ValueError(f'Unknown filtering argument {kwarg}')
        ret = copy.deepcopy(self)
        if 'filters' in kwargs and kwargs['filters'] is not None:
            if not isinstance(kwargs['filters'], dict):
                raise TypeError("'filters' must be a dict.")
            ret._filters.update(kwargs['filters'])
        if 'filter_blank' in kwargs and kwargs['filter_blank'] is not None:
            if not isinstance(kwargs['filter_blank'], list):
                kwargs['filter_blank'] = [kwargs['filter_blank']]
            ret._filter_blank += kwargs['filter_blank']
        if 'min_tiles' in kwargs and kwargs['min_tiles'] is not None:
            if not isinstance(kwargs['min_tiles'], int):
                raise TypeError("'min_tiles' must be an int.")
            ret._min_tiles = kwargs['min_tiles']
        return ret

    def filter_bags_by_roi(
        self,
        bags_path: str,
        dest: str,
        *,
        tile_df: Optional[pd.DataFrame] = None
    ) -> None:
        """Filter bags by tiles in an ROI."""
        import torch

        #TODO: extend to tfrecords
        #TODO: accelerate with multiprocessing
        #TODO: save filtered indices
        #TODO: copy bags config

        if tile_df is None:
            tile_df = self.get_tile_dataframe()
        if not exists(dest):
            os.makedirs(dest)

        # Subset the dataframe to only include tiles with an ROI
        roi_df = tile_df.loc[tile_df.roi_name.notnull()]

        n_complete = 0
        for slide in tqdm(roi_df.slide.unique()):
            if not exists(join(bags_path, slide+'.pt')):
                continue

            # Get the bag
            bag = torch.load(join(bags_path, slide+'.pt'))
            bag_index = np.load(join(bags_path, slide+'.index.npz'))['arr_0']

            # Subset the ROI based on this slide
            slide_df = roi_df.loc[roi_df.slide == slide]

            # Get the common locations (in an ROI)
            bag_locs = {tuple(r) for r in bag_index}
            roi_locs = {tuple(r) for r in np.stack([slide_df.loc_x.values, slide_df.loc_y.values], axis=1)}
            common_locs = bag_locs.intersection(roi_locs)

            # Find indices in the bag that match the common locations (in an ROI)
            bag_i = [i for i, row in enumerate(bag_index) if tuple(row) in common_locs]

            if not len(bag_i):
                log.debug("No common locations found for {}".format(slide))
                continue

            # Subset and save the bag
            bag = bag[bag_i]
            torch.save(bag, join(dest, slide+'.pt'))
            log.debug("Subset size ({}): {} -> {}".format(slide, len(bag_index), len(bag)))
            n_complete += 1

        log.info("Bag filtering complete. {} bags filtered.".format(n_complete))

    def find_rois(self, slide: str) -> Optional[str]:
        """Find an ROI path from a given slide.

        Args:
            slide (str): Slide name.

        Returns:
            str: Matching path to ROI, if found. If not found, returns None
        """
        rois = self.rois()
        if not rois:
            return None
        for roi in rois:
            if path_to_name(roi) == slide:
                return roi
        return None

    def find_slide(
        self,
        *,
        slide: Optional[str] = None,
        patient: Optional[str] = None
    ) -> Optional[str]:
        """Find a slide path from a given slide or patient.

        Keyword args:
            slide (str): Find a tfrecord associated with this slide name.
            patient (str): Find a tfrecord associated with this patient.

        Returns:
            str: Matching path to slide, if found. If not found, returns None
        """
        if slide is None and patient is None:
            raise ValueError("Must supply either slide or patient.")
        if slide is not None and patient is not None:
            raise ValueError("Must supply either slide or patient, not both.")

        if slide is not None:
            filtered = self.filter({'slide': slide})
        if patient is not None:
            filtered = self.filter({'slide': patient})
        matching = filtered.slide_paths()
        if not len(matching):
            return None
        else:
            return matching[0]

    def find_tfrecord(
        self,
        *,
        slide: Optional[str] = None,
        patient: Optional[str] = None
    ) -> Optional[str]:
        """Find a TFRecord path from a given slide or patient.

        Keyword args:
            slide (str): Find a tfrecord associated with this slide name.
            patient (str): Find a tfrecord associated with this patient.

        Returns:
            str: Matching path to tfrecord, if found. Otherwise, returns None
        """
        if slide is None and patient is None:
            raise ValueError("Must supply either slide or patient.")
        if slide is not None and patient is not None:
            raise ValueError("Must supply either slide or patient, not both.")

        if slide is not None:
            filtered = self.filter({'slide': slide})
        if patient is not None:
            filtered = self.filter({'slide': patient})
        matching = filtered.tfrecords()
        if not len(matching):
            return None
        else:
            return matching[0]

    def generate_rois(
        self,
        model: str,
        *,
        overwrite: bool = False,
        dest: Optional[str] = None,
        **kwargs
    ):
        """Generate ROIs using a U-Net model.

        Args:
            model (str): Path to model (zip) or model configuration (json).

        Keyword args:
            overwrite (bool, optional): Overwrite existing ROIs. Defaults to False.
            dest (str, optional): Destination directory for generated ROIs.
                If not provided, uses the dataset's default ROI directory.
            sq_mm_threshold (float, optional): If not None, filter out ROIs with an area
                less than the given threshold (in square millimeters). Defaults to None.

        """

        # Load the model configuration.
        segment = sf.slide.qc.StridedSegment(model)

        for slide in track(self.slide_paths(), description='Generating...'):

            # Set the destination directory
            source = self.get_slide_source(slide)
            if 'roi' not in self.sources[source] and dest is None:
                raise errors.DatasetError(
                    "No ROI directory set. Please set an ROI directory in the "
                    "dataset configuration, or provide a destination directory "
                    "with the `dest` argument."
                )
            if dest is None:
                dest = self.sources[source]['roi']
            if not exists(dest):
                os.makedirs(dest)

            # Check if an ROI already exists.
            existing_rois = [path_to_name(f) for f in os.listdir(dest) if f.endswith('csv')]
            if path_to_name(slide) in existing_rois:
                if overwrite:
                    log.info(f"Overwriting ROI for slide {path_to_name(slide)} at {dest}")
                else:
                    log.info(f"ROI already exists for slide {path_to_name(slide)} at {dest}")
                    continue

            # Load the slide and remove any existing auto-loaded ROIs.
            wsi = sf.WSI(slide, 299, 512, verbose=False)
            wsi.rois = []

            # Generate and apply ROIs.
            segment.generate_rois(wsi, apply=True, **kwargs)

            # Export ROIs to CSV.
            wsi.export_rois(join(dest, wsi.name + '.csv'))


    def get_slide_source(self, slide: str) -> str:
        """Return the source of a given slide.

        Args:
            slide (str): Slide name.

        Returns:
            str: Source name.

        """
        for source in self.sources:
            paths = self.slide_paths(source=source)
            names = [path_to_name(path) for path in paths]
            if slide in paths or slide in names:
                return source
        raise errors.DatasetError(f"Could not find slide '{slide}'")

    def get_tfrecord_locations(self, slide: str) -> List[Tuple[int, int]]:
        """Return a list of locations stored in an associated TFRecord.

        Args:
            slide (str): Slide name.

        Returns:
            List of tuples of (x, y) coordinates.

        """
        tfr = self.find_tfrecord(slide=slide)
        if tfr is None:
            raise errors.TFRecordsError(
                f"Could not find associated TFRecord for slide '{slide}'"
            )
        tfr_idx = sf.util.tfrecord2idx.find_index(tfr)
        if not tfr_idx:
            _create_index(tfr)
        elif tfr_idx.endswith('index'):
            log.info(f"Updating index for {tfr}...")
            os.remove(tfr_idx)
            _create_index(tfr)
        return sf.io.get_locations_from_tfrecord(tfr)

    def harmonize_labels(
        self,
        *args: "Dataset",
        header: Optional[str] = None
    ) -> Dict[str, int]:
        """Harmonize labels with another dataset.

        Returns categorical label assignments converted to int, harmonized with
        another dataset to ensure label consistency between datasets.

        Args:
            *args (:class:`slideflow.Dataset`): Any number of Datasets.
            header (str): Categorical annotation header.

        Returns:
            Dict mapping slide names to categories.

        """
        if header is None:
            raise ValueError("Must supply kwarg 'header'")
        if not isinstance(header, str):
            raise ValueError('Harmonized labels require a single header.')

        _, my_unique = self.labels(header, use_float=False)
        other_uniques = [
            np.array(dts.labels(header, use_float=False)[1]) for dts in args
        ]
        other_uniques = other_uniques + [np.array(my_unique)]
        uniques_list = np.concatenate(other_uniques).tolist()
        all_unique = sorted(list(set(uniques_list)))
        labels_to_int = dict(zip(all_unique, range(len(all_unique))))
        return labels_to_int

    def is_float(self, header: str) -> bool:
        """Check if labels in the given header can all be converted to float.

        Args:
            header (str): Annotations column header.

        Returns:
            bool: If all values from header can be converted to float.

        """
        if self.annotations is None:
            raise errors.DatasetError("Annotations not loaded.")
        filtered_labels = self.filtered_annotations[header]
        try:
            filtered_labels = [float(o) for o in filtered_labels]
            return True
        except ValueError:
            return False

    def kfold_split(
        self,
        k: int,
        *,
        labels: Optional[Union[Dict, str]] = None,
        preserved_site: bool = False,
        site_labels: Optional[Union[str, Dict[str, str]]] = 'site',
        splits: Optional[str] = None,
        read_only: bool = False,
    ) -> Tuple[Tuple["Dataset", "Dataset"], ...]:
        """Split the dataset into k cross-folds.

        Args:
            k (int): Number of cross-folds.

        Keyword args:
            labels (dict or str, optional):  Either a dictionary mapping slides
                to labels, or an outcome label (``str``). Used for balancing
                outcome labels in training and validation cohorts. If None,
                will not balance k-fold splits by outcome labels. Defaults
                to None.
            preserved_site (bool): Split with site-preserved cross-validation.
                Defaults to False.
            site_labels (dict, optional): Dict mapping patients to site labels,
                or an outcome column with site labels. Only used for site
                preserved cross validation. Defaults to 'site'.
            splits (str, optional): Path to JSON file containing validation
                splits. Defaults to None.
            read_only (bool): Prevents writing validation splits to file.
                Defaults to False.

        """
        if splits is None:
            temp_dir = tempfile.TemporaryDirectory()
            splits = join(temp_dir.name, '_splits.json')
        else:
            temp_dir = None
        crossval_splits = []
        for k_fold_iter in range(k):
            split_kw = dict(
                labels=labels,
                val_strategy=('k-fold-preserved-site' if preserved_site
                              else 'k-fold'),
                val_k_fold=k,
                k_fold_iter=k_fold_iter+1,
                site_labels=site_labels,
                splits=splits,
                read_only=read_only
            )
            crossval_splits.append(self.split(**split_kw))
        if temp_dir is not None:
            temp_dir.cleanup()
        return tuple(crossval_splits)

    def labels(
        self,
        headers: Union[str, List[str]],
        use_float: Union[bool, Dict, str] = False,
        assign: Optional[Dict[str, Dict[str, int]]] = None,
        format: str = 'index'
    ) -> Tuple[Labels, Union[Dict[str, Union[List[str], List[float]]],
                             List[str],
                             List[float]]]:
        """Return a dict of slide names mapped to patient id and label(s).

        Args:
            headers (list(str)) Annotation header(s) that specifies label.
                May be a list or string.
            use_float (bool, optional) Either bool, dict, or 'auto'.
                If true, convert data into float; if unable, raise TypeError.
                If false, interpret all data as categorical.
                If a dict(bool), look up each header to determine type.
                If 'auto', will try to convert all data into float. For each
                header in which this fails, will interpret as categorical.
            assign (dict, optional):  Dictionary mapping label ids to
                label names. If not provided, will map ids to names by sorting
                alphabetically.
            format (str, optional): Either 'index' or 'name.' Indicates which
                format should be used for categorical outcomes when returning
                the label dictionary. If 'name', uses the string label name.
                If 'index', returns an int (index corresponding with the
                returned list of unique outcomes as str). Defaults to 'index'.

        Returns:
            A tuple containing

                **dict**: Dictionary mapping slides to outcome labels in
                numerical format (float for linear outcomes, int of outcome
                label id for categorical outcomes).

                **list**: List of unique labels. For categorical outcomes,
                this will be a list of str; indices correspond with the outcome
                label id.

        """
        if self.annotations is None:
            raise errors.DatasetError("Annotations not loaded.")
        if not len(self.filtered_annotations):
            raise errors.DatasetError(
                "Cannot generate labels: dataset is empty after filtering."
            )
        results = {}  # type: Dict
        headers = sf.util.as_list(headers)
        unique_labels = {}
        filtered_pts = self.filtered_annotations.patient
        filtered_slides = self.filtered_annotations.slide
        for header in headers:
            if assign and (len(headers) > 1 or header in assign):
                assigned_for_header = assign[header]
            elif assign is not None:
                raise errors.DatasetError(
                    f"Unable to read outcome assignments for header {header}"
                    f" (assign={assign})"
                )
            else:
                assigned_for_header = None
            unique_labels_for_this_header = []
            try:
                filtered_labels = self.filtered_annotations[header]
            except KeyError:
                raise errors.AnnotationsError(f"Missing column {header}.")

            # Determine whether values should be converted into float
            if isinstance(use_float, dict) and header not in use_float:
                raise ValueError(
                    f"use_float is dict, but header {header} is missing."
                )
            elif isinstance(use_float, dict):
                header_is_float = use_float[header]
            elif isinstance(use_float, bool):
                header_is_float = use_float
            elif use_float == 'auto':
                header_is_float = self.is_float(header)
            else:
                raise ValueError(f"Invalid use_float option {use_float}")

            # Ensure labels can be converted to desired type,
            # then assign values
            if header_is_float and not self.is_float(header):
                raise TypeError(
                    f"Unable to convert all labels of {header} into 'float' "
                    f"({','.join(filtered_labels)})."
                )
            elif header_is_float:
                log.debug(f'Interpreting column "{header}" as continuous')
                filtered_labels = filtered_labels.astype(float)
            else:
                log.debug(f'Interpreting column "{header}" as categorical')
                unique_labels_for_this_header = list(set(filtered_labels))
                unique_labels_for_this_header.sort()
                for i, ul in enumerate(unique_labels_for_this_header):
                    n_matching_filtered = sum(f == ul for f in filtered_labels)
                    if assigned_for_header and ul not in assigned_for_header:
                        raise KeyError(
                            f"assign was provided, but label {ul} missing"
                        )
                    elif assigned_for_header:
                        val_msg = assigned_for_header[ul]
                        n_s = str(n_matching_filtered)
                        log.debug(
                            f"{header} {ul} assigned {val_msg} [{n_s} slides]"
                        )
                    else:
                        n_s = str(n_matching_filtered)
                        log.debug(
                            f"{header} {ul} assigned {i} [{n_s} slides]"
                        )

            def _process_cat_label(o):
                if assigned_for_header:
                    return assigned_for_header[o]
                elif format == 'name':
                    return o
                else:
                    return unique_labels_for_this_header.index(o)

            # Check for multiple, different labels per patient and warn
            pt_assign = np.array(list(set(zip(filtered_pts, filtered_labels))))
            unique_pt, counts = np.unique(pt_assign[:, 0], return_counts=True)
            for pt in unique_pt[np.argwhere(counts > 1)][:, 0]:
                dup_vals = pt_assign[pt_assign[:, 0] == pt][:, 1]
                dups = ", ".join([str(d) for d in dup_vals])
                log.error(
                    f'Multiple labels for patient "{pt}" (header {header}): '
                    f'{dups}'
                )

            # Assemble results dictionary
            for slide, lbl in zip(filtered_slides, filtered_labels):
                if slide in sf.util.EMPTY:
                    continue
                if not header_is_float:
                    lbl = _process_cat_label(lbl)
                if slide in results:
                    results[slide] = sf.util.as_list(results[slide])
                    results[slide] += [lbl]
                elif header_is_float:
                    results[slide] = [lbl]
                else:
                    results[slide] = lbl
            unique_labels[header] = unique_labels_for_this_header
        if len(headers) == 1:
            return results, unique_labels[headers[0]]
        else:
            return results, unique_labels

    def load_indices(self, verbose=False) -> Dict[str, np.ndarray]:
        """Return TFRecord indices."""
        pool = DPool(8)
        tfrecords = self.tfrecords()
        indices = {}

        def load_index(tfr):
            tfr_name = path_to_name(tfr)
            index = tfrecord2idx.load_index(tfr)
            return tfr_name, index

        log.debug("Loading indices...")
        for tfr_name, index in pool.imap(load_index, tfrecords):
            indices[tfr_name] = index
        pool.close()
        return indices

    def manifest(
        self,
        key: str = 'path',
        filter: bool = True
    ) -> Dict[str, Dict[str, int]]:
        """Generate a manifest of all tfrecords.

        Args:
            key (str): Either 'path' (default) or 'name'. Determines key format
                in the manifest dictionary.
            filter (bool): Apply active filters to manifest.

        Returns:
            dict: Dict mapping key (path or slide name) to number of tiles.

        """
        if key not in ('path', 'name'):
            raise ValueError("'key' must be in ['path, 'name']")

        all_manifest = {}
        for source in self.sources:
            if self.sources[source]['label'] is None:
                continue
            if not self._tfrecords_set(source):
                log.warning(f"tfrecords path not set for source {source}")
                continue
            tfrecord_dir = join(
                self.sources[source]['tfrecords'],
                self.sources[source]['label']
            )
            manifest_path = join(tfrecord_dir, "manifest.json")
            if not exists(manifest_path):
                log.debug(f"No manifest at {tfrecord_dir}; creating now")
                sf.io.update_manifest_at_dir(tfrecord_dir)

            if exists(manifest_path):
                relative_manifest = sf.util.load_json(manifest_path)
            else:
                relative_manifest = {}
            global_manifest = {}
            for record in relative_manifest:
                k = join(tfrecord_dir, record)
                global_manifest.update({k: relative_manifest[record]})
            all_manifest.update(global_manifest)
        # Now filter out any tfrecords that would be excluded by filters
        if filter:
            filtered_tfrecords = self.tfrecords()
            manifest_tfrecords = list(all_manifest.keys())
            for tfr in manifest_tfrecords:
                if tfr not in filtered_tfrecords:
                    del all_manifest[tfr]
        # Log clipped tile totals if applicable
        for tfr in all_manifest:
            if tfr in self._clip:
                all_manifest[tfr]['clipped'] = min(self._clip[tfr],
                                                   all_manifest[tfr]['total'])
            else:
                all_manifest[tfr]['clipped'] = all_manifest[tfr]['total']
        if key == 'path':
            return all_manifest
        else:
            return {path_to_name(t): v for t, v in all_manifest.items()}

    def manifest_histogram(
        self,
        by: Optional[str] = None,
        binrange: Optional[Tuple[int, int]] = None
    ) -> None:
        """Plot histograms of tiles-per-slide.

        Example
            Create histograms of tiles-per-slide, stratified by site.

                .. code-block:: python

                    import matplotlib.pyplot as plt

                    dataset.manifest_histogram(by='site')
                    plt.show()

        Args:
            by (str, optional): Stratify histograms by this annotation column
                header. Defaults to None.
            binrange (tuple(int, int)): Histogram bin ranges. If None, uses
                full range. Defaults to None.

        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        if by is not None:
            _, unique_vals = self.labels(by, format='name')
            val_counts = [
                [
                    m['total']
                    for m in self.filter({by: val}).manifest().values()
                ]
                for val in unique_vals
            ]
            all_counts = [c for vc in val_counts for c in vc]
        else:
            unique_vals = ['']
            all_counts = [m['total'] for m in self.manifest().values()]
            val_counts = [all_counts]
        if binrange is None:
            max_count = (max(all_counts) // 20) * 20
            binrange = (0, max_count)

        fig, axes = plt.subplots(len(unique_vals), 1,
                                 figsize=(3, len(unique_vals)))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        fig.set_tight_layout({"pad": .0})
        for a, ax in enumerate(axes):
            sns.histplot(val_counts[a], bins=20, binrange=binrange, ax=ax)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.set_ylabel(unique_vals[a], rotation='horizontal', ha='right')
            ax.set_xlim(binrange)
            if a != (len(axes) - 1):
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.set(xlabel=None)
        ax.set(xlabel="Tiles per slide")

    def patients(self) -> Dict[str, str]:
        """Return a list of patient IDs from this dataset."""
        if self.annotations is None:
            raise errors.DatasetError("Annotations not loaded.")
        result = {}  # type: Dict[str, str]
        pairs = list(zip(
            self.filtered_annotations['slide'],
            self.filtered_annotations['patient']
        ))
        for slide, patient in pairs:
            if slide in result and result[slide] != patient:
                raise errors.AnnotationsError(
                    f'Slide "{slide}" assigned to multiple patients: '
                    f"({patient}, {result[slide]})"
                )
            else:
                if slide not in sf.util.EMPTY:
                    result[slide] = patient
        return result

    def pt_files(self, path, warn_missing=True):
        """Return list of all \*.pt files with slide names in this dataset.

        May return more than one \*.pt file for each slide.

        Args:
            path (str, list(str)): Directory(ies) to search for \*.pt files.
            warn_missing (bool): Raise a warning if any slides in this dataset
                do not have a \*.pt file.

        """
        slides = self.slides()
        if isinstance(path, str):
            path = [path]

        bags = []
        for p in path:
            if not exists(p):
                raise ValueError(f"Path {p} does not exist.")
            bags_at_path = np.array([
                join(p, f) for f in os.listdir(p)
                if f.endswith('.pt') and path_to_name(f) in slides
            ])
            bags.append(bags_at_path)
        bags = np.concatenate(bags)
        unique_slides_with_bags = np.unique([path_to_name(b) for b in bags])
        if (len(unique_slides_with_bags) != len(slides)) and warn_missing:
            log.warning(f"Bags missing for {len(slides) - len(unique_slides_with_bags)} slides.")
        return bags

    def read_tfrecord_by_location(
        self,
        slide: str,
        loc: Tuple[int, int],
        decode: Optional[bool] = None
    ) -> Any:
        """Read a record from a TFRecord, indexed by location.

        Finds the associated TFRecord for a slide, and returns the record
        inside which corresponds to a given tile location.

        Args:
            slide (str): Name of slide. Will search for the slide's associated
                TFRecord.
            loc ((int, int)): ``(x, y)`` tile location. Searches the TFRecord
                for the tile that corresponds to this location.
            decode (bool): Decode the associated record, returning Tensors.
                Defaults to True.

        Returns:
            Unprocessed raw TFRecord bytes if ``decode=False``, otherwise a
            tuple containing ``(slide, image)``, where ``image`` is a
            uint8 Tensor.

        """
        tfr = self.find_tfrecord(slide=slide)
        if tfr is None:
            raise errors.TFRecordsError(
                f"Could not find associated TFRecord for slide '{slide}'"
            )
        if decode is None:
            decode = True
        else:
            warnings.warn(
                "The 'decode' argument to `Dataset.read_tfrecord_by_location` "
                "is deprecated and will be removed in a future version. In the "
                "future, all records will be decoded."
            )
        return sf.io.get_tfrecord_by_location(tfr, loc, decode=decode)

    def remove_filter(self, **kwargs: Any) -> "Dataset":
        """Remove a specific filter from the active filters.

        Keyword Args:
            filters (list of str): Filter keys. Will remove filters with
                these keys.
            filter_blank (list of str): Will remove these headers stored in
                filter_blank.

        Returns:
            :class:`slideflow.Dataset`: Dataset with filter removed.

        """
        for kwarg in kwargs:
            if kwarg not in ('filters', 'filter_blank'):
                raise ValueError(f'Unknown filtering argument {kwarg}')
        ret = copy.deepcopy(self)
        if 'filters' in kwargs:
            if isinstance(kwargs['filters'], str):
                kwargs['filters'] = [kwargs['filters']]
            elif not isinstance(kwargs['filters'], list):
                raise TypeError("'filters' must be a list.")
            for f in kwargs['filters']:
                if f not in ret._filters:
                    raise errors.DatasetFilterError(
                        f"Filter {f} not found in dataset (active filters:"
                        f"{','.join(list(ret._filters.keys()))})"
                    )
                else:
                    del ret._filters[f]
        if 'filter_blank' in kwargs:
            kwargs['filter_blank'] = sf.util.as_list(kwargs['filter_blank'])
            for f in kwargs['filter_blank']:
                if f not in ret._filter_blank:
                    raise errors.DatasetFilterError(
                        f"Filter_blank {f} not found in dataset (active "
                        f"filter_blank: {','.join(ret._filter_blank)})"
                    )
                elif isinstance(ret._filter_blank, dict):
                    del ret._filter_blank[ret._filter_blank.index(f)]
        return ret

    def rebuild_index(self) -> None:
        """Rebuild index files for TFRecords.

        Equivalent to ``Dataset.build_index(force=True)``.

        Args:
            None

        Returns:
            None
        """
        self.build_index(force=True)

    def resize_tfrecords(self, tile_px: int) -> None:
        """Resize images in a set of TFRecords to a given pixel size.

        Args:
            tile_px (int): Target pixel size for resizing TFRecord images.

        """
        if not sf.util.tf_available:
            raise NotImplementedError(
                "Dataset.resize_tfrecords() requires Tensorflow, which is "
                "not installed.")

        log.info(f'Resizing TFRecord tiles to ({tile_px}, {tile_px})')
        tfrecords_list = self.tfrecords()
        log.info(f'Resizing {len(tfrecords_list)} tfrecords')
        for tfr in tfrecords_list:
            sf.io.tensorflow.transform_tfrecord(
                tfr,
                tfr+'.transformed',
                resize=tile_px
            )

    def rois(self) -> List[str]:
        """Return a list of all ROIs."""
        rois_list = []
        for source in self.sources:
            if self._roi_set(source):
                rois_list += glob(join(self.sources[source]['roi'], "*.csv"))
            else:
                log.warning(f"roi path not set for source {source}")
        slides = self.slides()
        return [r for r in list(set(rois_list)) if path_to_name(r) in slides]

    def slide_manifest(
        self,
        roi_method: str = 'auto',
        stride_div: int = 1,
        tma: bool = False,
        source: Optional[str] = None,
        low_memory: bool = False
    ) -> Dict[str, int]:
        """Return a dictionary of slide names and estimated number of tiles.

        Uses Otsu thresholding for background filtering, and the ROI strategy.

        Args:
            roi_method (str): Either 'inside', 'outside', 'auto', or 'ignore'.
                Determines how ROIs are used to extract tiles.
                If 'inside' or 'outside', will extract tiles in/out of an ROI,
                and skip a slide if an ROI is not available.
                If 'auto', will extract tiles inside an ROI if available,
                and across the whole-slide if no ROI is found.
                If 'ignore', will extract tiles across the whole-slide
                regardless of whether an ROI is available.
                Defaults to 'auto'.
            stride_div (int): Stride divisor for tile extraction.
                A stride of 1 will extract non-overlapping tiles.
                A stride_div of 2 will extract overlapping tiles, with a stride
                equal to 50% of the tile width. Defaults to 1.
            tma (bool): Deprecated argument. Tumor micro-arrays are read as
                standard slides. Defaults to False.
            source (str, optional): Dataset source name.
                Defaults to None (using all sources).
            low_memory (bool): Operate in low-memory mode at the cost of
                worse performance.

        Returns:
            Dict[str, int]: Dictionary mapping slide names to number of
            estimated non-background tiles in the slide.

        """
        if tma:
            warnings.warn(
                "tma=True is deprecated and will be removed in a future "
                "version. Tumor micro-arrays are read as standard slides. "
            )
        if self.tile_px is None or self.tile_um is None:
            raise errors.DatasetError(
                "tile_px and tile_um must be set to calculate a slide manifest"
            )
        paths = self.slide_paths(source=source)
        pb = Progress(transient=True)
        read_task = pb.add_task('Reading slides...', total=len(paths))
        if not low_memory:
            otsu_task = pb.add_task("Otsu thresholding...", total=len(paths))
        pb.start()
        pool = mp.Pool(
            sf.util.num_cpu(default=16),
            initializer=sf.util.set_ignore_sigint
        )
        wsi_list = []
        to_remove = []
        counts = []
        for path in paths:
            try:
                wsi = sf.WSI(
                    path,
                    self.tile_px,
                    self.tile_um,
                    rois=self.rois(),
                    stride_div=stride_div,
                    roi_method=roi_method,
                    verbose=False)
                if low_memory:
                    wsi.qc('otsu')
                    counts += [wsi.estimated_num_tiles]
                else:
                    wsi_list += [wsi]
                pb.advance(read_task)
            except errors.SlideLoadError as e:
                log.error(f"Error reading slide {path}: {e}")
                to_remove += [path]
        for path in to_remove:
            paths.remove(path)
        pb.update(read_task, total=len(paths))
        pb.update(otsu_task, total=len(paths))
        if not low_memory:
            for count in pool.imap(_count_otsu_tiles, wsi_list):
                counts += [count]
                pb.advance(otsu_task)
        pb.stop()
        pool.close()
        return {path: counts[p] for p, path in enumerate(paths)}

    def slide_paths(
        self,
        source: Optional[str] = None,
        apply_filters: bool = True
    ) -> List[str]:
        """Return a list of paths to slides.

        Either returns a list of paths to all slides, or slides only matching
        dataset filters.

        Args:
            source (str, optional): Dataset source name.
                Defaults to None (using all sources).
            filter (bool, optional): Return only slide paths meeting filter
                criteria. If False, return all slides. Defaults to True.

        Returns:
            list(str): List of slide paths.

        """
        if source and source not in self.sources.keys():
            raise errors.DatasetError(f"Dataset {source} not found.")
        # Get unfiltered paths
        if source:
            if not self._slides_set(source):
                log.warning(f"slides path not set for source {source}")
                return []
            else:
                paths = sf.util.get_slide_paths(self.sources[source]['slides'])
        else:
            paths = []
            for src in self.sources:
                if not self._slides_set(src):
                    log.warning(f"slides path not set for source {src}")
                else:
                    paths += sf.util.get_slide_paths(
                        self.sources[src]['slides']
                    )

        # Remove any duplicates from shared dataset paths
        paths = list(set(paths))
        # Filter paths
        if apply_filters:
            filtered_slides = self.slides()
            filtered_paths = [
                p for p in paths if path_to_name(p) in filtered_slides
            ]
            return filtered_paths
        else:
            return paths

    def slides(self) -> List[str]:
        """Return a list of slide names in this dataset."""
        if self.annotations is None:
            raise errors.AnnotationsError(
                "No annotations loaded; is the annotations file empty?"
            )
        if 'slide' not in self.annotations.columns:
            raise errors.AnnotationsError(
                f"{'slide'} not found in annotations file."
            )
        ann = self.filtered_annotations
        ann = ann.loc[~ann.slide.isin(sf.util.EMPTY)]
        slides = ann.slide.unique().tolist()
        return slides

    def split(
        self,
        model_type: Optional[str] = None,
        labels: Optional[Union[Dict, str]] = None,
        val_strategy: str = 'fixed',
        splits: Optional[str] = None,
        val_fraction: Optional[float] = None,
        val_k_fold: Optional[int] = None,
        k_fold_iter: Optional[int] = None,
        site_labels: Optional[Union[str, Dict[str, str]]] = 'site',
        read_only: bool = False,
        from_wsi: bool = False,
    ) -> Tuple["Dataset", "Dataset"]:
        """Split this dataset into a training and validation dataset.

        If a validation split has already been prepared (e.g. K-fold iterations
        were already determined), the previously generated split will be used.
        Otherwise, create a new split and log the result in the TFRecord
        directory so future models may use the same split for consistency.

        Args:
            model_type (str): Either 'categorical' or 'linear'. Defaults
                to 'categorical' if ``labels`` is provided.
            labels (dict or str):  Either a dictionary of slides: labels,
                or an outcome label (``str``). Used for balancing outcome
                labels in training and validation cohorts. Defaults to None.
            val_strategy (str): Either 'k-fold', 'k-fold-preserved-site',
                'bootstrap', or 'fixed'. Defaults to 'fixed'.
            splits (str, optional): Path to JSON file containing validation
                splits. Defaults to None.
            outcome_key (str, optional): Key indicating outcome label in
                slide_labels_dict. Defaults to 'outcome_label'.
            val_fraction (float, optional): Proportion of data for validation.
                Not used if strategy is k-fold. Defaults to None.
            val_k_fold (int): K, required if using K-fold validation.
                Defaults to None.
            k_fold_iter (int, optional): Which K-fold iteration to generate
                starting at 1. Fequired if using K-fold validation.
                Defaults to None.
            site_labels (dict, optional): Dict mapping patients to site labels,
                or an outcome column with site labels. Only used for site
                preserved cross validation. Defaults to 'site'.
            read_only (bool): Prevents writing validation splits to file.
                Defaults to False.

        Returns:
            A tuple containing

                :class:`slideflow.Dataset`: Training dataset.

                :class:`slideflow.Dataset`: Validation dataset.
        """
        if (not k_fold_iter and 'k-fold' in val_strategy):
            raise errors.DatasetSplitError(
                "If strategy is 'k-fold', must supply k_fold_iter "
                "(int starting at 1)"
            )
        if (not val_k_fold and 'k-fold' in val_strategy):
            raise errors.DatasetSplitError(
                "If strategy is 'k-fold', must supply val_k_fold (K)"
            )
        if val_strategy == 'k-fold-preserved-site' and not site_labels:
            raise errors.DatasetSplitError(
                "k-fold-preserved-site requires site_labels (dict of "
                "patients:sites, or name of annotation column header"
            )
        if (val_strategy == 'k-fold-preserved-site'
           and isinstance(site_labels, str)):
            site_labels, _ = self.labels(site_labels, format='name')
        if val_strategy == 'k-fold-preserved-site' and site_labels is None:
            raise errors.DatasetSplitError(
                f"Must supply site_labels for strategy {val_strategy}"
            )
        if val_strategy in ('bootstrap', 'fixed') and val_fraction is None:
            raise errors.DatasetSplitError(
                f"Must supply val_fraction for strategy {val_strategy}"
            )
        if isinstance(labels, str):
            labels = self.labels(labels)[0]
        if labels is None and model_type is None:
            labels = self.patients()
            model_type = 'linear'
        elif model_type is None:
            model_type = 'categorical'

        # Prepare dataset
        patients = self.patients()
        splits_file = splits
        training_tfr = []
        val_tfr = []
        accepted_split = None
        slide_list = list(labels.keys())

        # Assemble dict of patients linking to list of slides & outcome labels
        # dataset.labels() ensures no duplicate labels for a single patient
        tfr_dir_list = self.tfrecords() if not from_wsi else self.slide_paths()
        skip_tfr_verification = False
        if not len(tfr_dir_list) and not from_wsi:
            log.warning("No tfrecords found; splitting from annotations only.")
            tfr_dir_list = tfr_dir_list_names = self.slides()
            skip_tfr_verification = True
        elif not len(tfr_dir_list):
            log.warning("No slides found; splitting from annotations only.")
            tfr_dir_list = tfr_dir_list_names = self.slides()
            skip_tfr_verification = True
        else:
            tfr_dir_list_names = [
                sf.util.path_to_name(tfr) for tfr in tfr_dir_list
            ]
        patients_dict = {}
        num_warned = 0
        for slide in slide_list:
            patient = slide if not patients else patients[slide]
            # Skip slides not found in directory
            if slide not in tfr_dir_list_names:
                log.debug(f"Slide {slide} missing tfrecord, skipping")
                num_warned += 1
                continue
            if patient not in patients_dict:
                patients_dict[patient] = {
                    'outcome_label': labels[slide],
                    'slides': [slide]
                }
            elif patients_dict[patient]['outcome_label'] != labels[slide]:
                ol = patients_dict[patient]['outcome_label']
                ok = labels[slide]
                raise errors.DatasetSplitError(
                    f"Multiple labels found for {patient} ({ol}, {ok})"
                )
            else:
                patients_dict[patient]['slides'] += [slide]

        # Add site labels to the patients dict if doing
        # preserved-site cross-validation
        if val_strategy == 'k-fold-preserved-site':
            assert site_labels is not None
            site_slide_list = list(site_labels.keys())
            for slide in site_slide_list:
                patient = slide if not patients else patients[slide]
                # Skip slides not found in directory
                if slide not in tfr_dir_list_names:
                    continue
                if 'site' not in patients_dict[patient]:
                    patients_dict[patient]['site'] = site_labels[slide]
                elif patients_dict[patient]['site'] != site_labels[slide]:
                    ol = patients_dict[patient]['slide']
                    ok = site_labels[slide]
                    _tail = f"{patient} ({ol}, {ok})"
                    raise errors.DatasetSplitError(
                        f"Multiple site labels found for {_tail}"
                    )
        if num_warned:
            log.warning(f"{num_warned} slides missing tfrecords, skipping")
        patients_list = list(patients_dict.keys())
        sorted_patients = [p for p in patients_list]
        sorted_patients.sort()
        shuffle(patients_list)

        # Create and log a validation subset
        if val_strategy == 'none':
            log.info("val_strategy is None; skipping validation")
            train_slides = np.concatenate([
                patients_dict[patient]['slides']
                for patient in patients_dict.keys()
            ]).tolist()
            val_slides = []
        elif val_strategy == 'bootstrap':
            assert val_fraction is not None
            num_val = int(val_fraction * len(patients_list))
            log.info(
                f"Boostrap validation: selecting {num_val} "
                "patients at random for validation testing"
            )
            val_patients = patients_list[0:num_val]
            train_patients = patients_list[num_val:]
            if not len(val_patients) or not len(train_patients):
                raise errors.InsufficientDataForSplitError
            val_slides = np.concatenate([
                patients_dict[patient]['slides']
                for patient in val_patients
            ]).tolist()
            train_slides = np.concatenate([
                patients_dict[patient]['slides']
                for patient in train_patients
            ]).tolist()
        else:
            # Try to load validation split
            if (not splits_file or not exists(splits_file)):
                loaded_splits = []
            else:
                loaded_splits = sf.util.load_json(splits_file)
            for split_id, split in enumerate(loaded_splits):
                # First, see if strategy is the same
                if split['strategy'] != val_strategy:
                    continue
                # If k-fold, check that k-fold length is the same
                if (val_strategy in ('k-fold', 'k-fold-preserved-site')
                   and len(list(split['tfrecords'].keys())) != val_k_fold):
                    continue

                # Then, check if patient lists are the same
                sp_pts = list(split['patients'].keys())
                sp_pts.sort()
                if sp_pts == sorted_patients:
                    # Finally, check if outcome variables are the same
                    c1 = [patients_dict[p]['outcome_label'] for p in sp_pts]
                    c2 = [split['patients'][p]['outcome_label']for p in sp_pts]
                    if c1 == c2:
                        log.info(
                            f"Using {val_strategy} validation split detected"
                            f" at [green]{splits_file}[/] (ID: {split_id})"
                        )
                        accepted_split = split
                        break

            # If no split found, create a new one
            if not accepted_split:
                if splits_file:
                    log.info("No compatible train/val split found.")
                    log.info(f"Logging new split at [green]{splits_file}")
                else:
                    log.info("No training/validation splits file provided.")
                    log.info("Unable to save or load validation splits.")
                new_split = {
                    'strategy': val_strategy,
                    'patients': patients_dict,
                    'tfrecords': {}
                }  # type: Any
                if val_strategy == 'fixed':
                    assert val_fraction is not None
                    num_val = int(val_fraction * len(patients_list))
                    val_patients = patients_list[0:num_val]
                    train_patients = patients_list[num_val:]
                    if not len(val_patients) or not len(train_patients):
                        raise errors.InsufficientDataForSplitError
                    val_slides = np.concatenate([
                        patients_dict[patient]['slides']
                        for patient in val_patients
                    ]).tolist()
                    train_slides = np.concatenate([
                        patients_dict[patient]['slides']
                        for patient in train_patients
                    ]).tolist()
                    new_split['tfrecords']['validation'] = val_slides
                    new_split['tfrecords']['training'] = train_slides

                elif val_strategy in ('k-fold', 'k-fold-preserved-site'):
                    assert val_k_fold is not None
                    if (val_strategy == 'k-fold-preserved-site'):
                        k_fold_patients = split_patients_preserved_site(
                            patients_dict,
                            val_k_fold,
                            balance=('outcome_label'
                                     if model_type == 'categorical'
                                     else None)
                        )
                    elif model_type == 'categorical':
                        k_fold_patients = split_patients_balanced(
                            patients_dict,
                            val_k_fold,
                            balance='outcome_label'
                        )
                    else:
                        k_fold_patients = split_patients(
                            patients_dict, val_k_fold
                        )
                    # Verify at least one patient is in each k_fold group
                    if (len(k_fold_patients) != val_k_fold
                       or not min([len(pl) for pl in k_fold_patients])):
                        raise errors.InsufficientDataForSplitError
                    train_patients = []
                    for k in range(1, val_k_fold+1):
                        new_split['tfrecords'][f'k-fold-{k}'] = np.concatenate(
                            [patients_dict[patient]['slides']
                             for patient in k_fold_patients[k-1]]
                        ).tolist()
                        if k == k_fold_iter:
                            val_patients = k_fold_patients[k-1]
                        else:
                            train_patients += k_fold_patients[k-1]
                    val_slides = np.concatenate([
                        patients_dict[patient]['slides']
                        for patient in val_patients
                    ]).tolist()
                    train_slides = np.concatenate([
                        patients_dict[patient]['slides']
                        for patient in train_patients
                    ]).tolist()
                else:
                    raise errors.DatasetSplitError(
                        f"Unknown validation strategy {val_strategy}."
                    )
                # Write the new split to log
                loaded_splits += [new_split]
                if not read_only and splits_file:
                    sf.util.write_json(loaded_splits, splits_file)
            else:
                # Use existing split
                if val_strategy == 'fixed':
                    val_slides = accepted_split['tfrecords']['validation']
                    train_slides = accepted_split['tfrecords']['training']
                elif val_strategy in ('k-fold', 'k-fold-preserved-site'):
                    assert val_k_fold is not None
                    k_id = f'k-fold-{k_fold_iter}'
                    val_slides = accepted_split['tfrecords'][k_id]
                    train_slides = np.concatenate([
                        accepted_split['tfrecords'][f'k-fold-{ki}']
                        for ki in range(1, val_k_fold+1)
                        if ki != k_fold_iter
                    ]).tolist()
                else:
                    raise errors.DatasetSplitError(
                        f"Unknown val_strategy {val_strategy} requested."
                    )

            # Perform final integrity check to ensure no patients
            # are in both training and validation slides
            if patients:
                validation_pt = list(set([patients[s] for s in val_slides]))
                training_pt = list(set([patients[s] for s in train_slides]))
            else:
                validation_pt, training_pt = val_slides, train_slides
            if sum([pt in training_pt for pt in validation_pt]):
                raise errors.DatasetSplitError(
                    "At least one patient is in both val and training sets."
                )

        # Assemble list of tfrecords
        if val_strategy != 'none':
            val_tfr = [
                tfr for tfr in tfr_dir_list
                if path_to_name(tfr) in val_slides or tfr in val_slides
            ]
            training_tfr = [
                tfr for tfr in tfr_dir_list
                if path_to_name(tfr) in train_slides or tfr in train_slides
            ]
        if not len(val_tfr) == len(val_slides):
            raise errors.DatasetError(
                f"Number of validation tfrecords ({len(val_tfr)}) does "
                f"not match number of validation slides ({len(val_slides)}). "
                "This may happen if multiple tfrecords were found for a slide."
            )
        if not len(training_tfr) == len(train_slides):
            raise errors.DatasetError(
                f"Number of training tfrecords ({len(training_tfr)}) does "
                f"not match number of training slides ({len(train_slides)}). "
                "This may happen if multiple tfrecords were found for a slide."
            )
        training_dts = copy.deepcopy(self)
        training_dts = training_dts.filter(filters={'slide': train_slides})
        val_dts = copy.deepcopy(self)
        val_dts = val_dts.filter(filters={'slide': val_slides})
        if not skip_tfr_verification and not from_wsi:
            assert sorted(training_dts.tfrecords()) == sorted(training_tfr)
            assert sorted(val_dts.tfrecords()) == sorted(val_tfr)
        elif not skip_tfr_verification:
            assert sorted(training_dts.slide_paths()) == sorted(training_tfr)
            assert sorted(val_dts.slide_paths()) == sorted(val_tfr)
        return training_dts, val_dts

    def split_tfrecords_by_roi(
        self,
        destination: str,
        roi_filter_method: Union[str, float] = 'center'
    ) -> None:
        """Split dataset tfrecords into separate tfrecords according to ROI.

        Will generate two sets of tfrecords, with identical names: one with
        tiles inside the ROIs, one with tiles outside the ROIs. Will skip any
        tfrecords that are missing ROIs. Requires slides to be available.

        Args:
            destination (str): Destination path.
            roi_filter_method (str or float): Method of filtering tiles with
                ROIs. Either 'center' or float (0-1). If 'center', tiles are
                filtered with ROIs based on the center of the tile. If float,
                tiles are filtered based on the proportion of the tile inside
                the ROI, and ``roi_filter_method`` is interpreted as a
                threshold. If the proportion of a tile inside the ROI is
                greater than this number, the tile is included. For example,
                if ``roi_filter_method=0.7``, a tile that is 80% inside of an
                ROI will be included, and a tile that is 50% inside of an ROI
                will be excluded. Defaults to 'center'.

        Returns:
            None
        """
        tfrecords = self.tfrecords()
        slides = {path_to_name(s): s for s in self.slide_paths()}
        rois = self.rois()
        manifest = self.manifest()

        if self.tile_px is None or self.tile_um is None:
            raise errors.DatasetError(
                "tile_px and tile_um must be non-zero to process TFRecords."
            )

        for tfr in tfrecords:
            slidename = path_to_name(tfr)
            if slidename not in slides:
                continue
            try:
                slide = WSI(
                    slides[slidename],
                    self.tile_px,
                    self.tile_um,
                    rois=rois,
                    roi_method='inside',
                    roi_filter_method=roi_filter_method
                )
            except errors.SlideLoadError as e:
                log.error(e)
                continue
            parser = sf.io.get_tfrecord_parser(
                tfr,
                decode_images=False,
                to_numpy=True
            )
            if parser is None:
                log.error(f"Could not read TFRecord {tfr}; skipping")
                continue
            reader = sf.io.TFRecordDataset(tfr)
            if not exists(join(destination, 'inside')):
                os.makedirs(join(destination, 'inside'))
            if not exists(join(destination, 'outside')):
                os.makedirs(join(destination, 'outside'))
            in_path = join(destination, 'inside', f'{slidename}.tfrecords')
            out_path = join(destination, 'outside', f'{slidename}.tfrecords')
            inside_roi_writer = sf.io.TFRecordWriter(in_path)
            outside_roi_writer = sf.io.TFRecordWriter(out_path)
            for record in track(reader, total=manifest[tfr]['total']):
                parsed = parser(record)
                loc_x, loc_y = parsed['loc_x'], parsed['loc_y']
                tile_in_roi = any([
                    roi.poly.contains(sg.Point(loc_x, loc_y))
                    for roi in slide.rois
                ])
                # Convert from a Tensor -> Numpy array
                if hasattr(record, 'numpy'):
                    record = record.numpy()
                if tile_in_roi:
                    inside_roi_writer.write(record)
                else:
                    outside_roi_writer.write(record)
            inside_roi_writer.close()
            outside_roi_writer.close()

    def summary(self) -> None:
        """Print a summary of this dataset."""
        # Get ROI information.
        patients = self.patients()
        has_rois = defaultdict(bool)
        slides_with_roi = {}
        patients_with_roi = defaultdict(bool)
        for r in self.rois():
            s = sf.util.path_to_name(r)
            with open(r, 'r') as f:
                has_rois[s] = len(f.read().split('\n')) > 2
        for sp in self.slide_paths():
            s = sf.util.path_to_name(sp)
            slides_with_roi[s] = has_rois[s]
        for s in self.slides():
            p = patients[s]
            if s in slides_with_roi and slides_with_roi[s]:
                patients_with_roi[p] = True

        # Print summary.
        if self.annotations is not None:
            num_patients = len(self.annotations.patient.unique())
        else:
            num_patients = 0
        print("Overview:")
        table = [("Configuration file:", self._config),
                 ("Tile size (px):",     self.tile_px),
                 ("Tile size (um):",     self.tile_um),
                 ("Slides:",             len(self.slides())),
                 ("Patients:",           num_patients),
                 ("Slides with ROIs:",   len([s for s in slides_with_roi
                                              if slides_with_roi[s]])),
                 ("Patients with ROIs:", len([p for p in patients_with_roi
                                              if patients_with_roi[p]]))]
        print(tabulate(table, tablefmt='fancy_outline'))
        print("\nFilters:")
        table = [("Filters:",           pformat(self.filters)),
                 ("Filter Blank:",       pformat(self.filter_blank)),
                 ("Min Tiles:",          pformat(self.min_tiles))]
        print(tabulate(table, tablefmt='fancy_grid'))
        print("\nSources:")
        if not self.sources:
            print("<None>")
        else:
            for source in self.sources:
                print(f"\n{source}")
                d = self.sources[source]
                print(tabulate(zip(d.keys(), d.values()),
                               tablefmt="fancy_outline"))

        print("\nNumber of tiles in TFRecords:", self.num_tiles)
        print("Annotation columns:")
        print("<NA>" if self.annotations is None else self.annotations.columns)

    def tensorflow(
        self,
        labels: Labels = None,
        batch_size: Optional[int] = None,
        from_wsi: bool = False,
        **kwargs: Any
    ) -> "tf.data.Dataset":
        """Return a Tensorflow Dataset object that interleaves tfrecords.

        The returned dataset yields a batch of (image, label) for each tile.
        Labels may be specified either via a dict mapping slide names to
        outcomes, or a parsing function which accept and image and slide name,
        returning a dict {'image_raw': image(tensor)} and label (int or float).

        Args:
            labels (dict or str, optional): Dict or function. If dict, must
                map slide names to outcome labels. If function, function must
                accept an image (tensor) and slide name (str), and return a
                dict {'image_raw': image (tensor)} and label (int or float).
                If not provided, all labels will be None.
            batch_size (int): Batch size.

        Keyword Args:
            augment (str or bool): Image augmentations to perform. Augmentations include:

                * ``'x'``: Random horizontal flip
                * ``'y'``: Random vertical flip
                * ``'r'``: Random 90-degree rotation
                * ``'j'``: Random JPEG compression (50% chance to compress with quality between 50-100)
                * ``'b'``: Random Gaussian blur (10% chance to blur with sigma between 0.5-2.0)
                * ``'n'``: Random :ref:`stain_augmentation` (requires stain normalizer)

                Combine letters to define augmentations, such as ``'xyrjn'``.
                A value of True will use ``'xyrjb'``.
            deterministic (bool, optional): When num_parallel_calls is specified,
                if this boolean is specified, it controls the order in which the
                transformation produces elements. If set to False, the
                transformation is allowed to yield elements out of order to trade
                determinism for performance. Defaults to False.
            drop_last (bool, optional): Drop the last non-full batch.
                Defaults to False.
            from_wsi (bool): Generate predictions from tiles dynamically
                extracted from whole-slide images, rather than TFRecords.
                Defaults to False (use TFRecords).
            incl_loc (str, optional): 'coord', 'grid', or None. Return (x,y)
                origin coordinates ('coord') for each tile center along with tile
                images, or the (x,y) grid coordinates for each tile.
                Defaults to 'coord'.
            incl_slidenames (bool, optional): Include slidenames as third returned
                variable. Defaults to False.
            infinite (bool, optional): Create an finite dataset. WARNING: If
                infinite is False && balancing is used, some tiles will be skipped.
                Defaults to True.
            img_size (int): Image width in pixels.
            normalizer (:class:`slideflow.norm.StainNormalizer`, optional):
                Normalizer to use on images. Defaults to None.
            num_parallel_reads (int, optional): Number of parallel reads for each
                TFRecordDataset. Defaults to 4.
            num_shards (int, optional): Shard the tfrecord datasets, used for
                multiprocessing datasets. Defaults to None.
            pool (multiprocessing.Pool): Shared multiprocessing pool. Useful
                if ``from_wsi=True``, for sharing a unified processing pool between
                dataloaders. Defaults to None.
            rois (list(str), optional): List of ROI paths. Only used if
                from_wsi=True.  Defaults to None.
            roi_method (str, optional): Method for extracting ROIs. Only used if
                from_wsi=True. Defaults to 'auto'.
            shard_idx (int, optional): Index of the tfrecord shard to use.
                Defaults to None.
            standardize (bool, optional): Standardize images to (0,1).
                Defaults to True.
            tile_um (int, optional): Size of tiles to extract from WSI, in
                microns. Only used if from_wsi=True. Defaults to None.
            tfrecord_parser (Callable, optional): Custom parser for TFRecords.
                Defaults to None.
            transform (Callable, optional): Arbitrary transform function.
                Performs transformation after augmentations but before
                standardization. Defaults to None.
            **decode_kwargs (dict): Keyword arguments to pass to
                :func:`slideflow.io.tensorflow.decode_image`.

        Returns:
            tf.data.Dataset

        """
        from slideflow.io.tensorflow import interleave

        if self.tile_px is None:
            raise errors.DatasetError("tile_px and tile_um must be non-zero"
                                      "to create dataloaders.")
        if self.prob_weights is not None and from_wsi:
            log.warning("Dataset balancing is disabled when `from_wsi=True`")
        if self._clip not in (None, {}) and from_wsi:
            log.warning("Dataset clipping is disabled when `from_wsi=True`")

        if from_wsi:
            tfrecords = self.slide_paths()
            kwargs['rois'] = self.rois()
            kwargs['tile_um'] = self.tile_um
            kwargs['from_wsi'] = True
            prob_weights = None
            clip = None
        else:
            tfrecords = self.tfrecords()
            prob_weights = self.prob_weights
            clip = self._clip
            if not tfrecords:
                raise errors.TFRecordsNotFoundError
            self.verify_img_format(progress=False)

        return interleave(paths=tfrecords,
                          labels=labels,
                          img_size=self.tile_px,
                          batch_size=batch_size,
                          prob_weights=prob_weights,
                          clip=clip,
                          **kwargs)

    def tfrecord_report(
        self,
        dest: str,
        normalizer: Optional["StainNormalizer"] = None
    ) -> None:
        """Create a PDF report of TFRecords.

        Reports include 10 example tiles per TFRecord. Report is saved
        in the target destination.

        Args:
            dest (str): Directory in which to save the PDF report.
            normalizer (`slideflow.norm.StainNormalizer`, optional):
                Normalizer to use on image tiles. Defaults to None.

        """
        if normalizer is not None:
            log.info(f'Using realtime {normalizer.method} normalization')

        tfrecord_list = self.tfrecords()
        reports = []
        log.info('Generating TFRecords report...')
        # Get images for report
        for tfr in track(tfrecord_list, description='Generating report...'):
            dataset = sf.io.TFRecordDataset(tfr)
            parser = sf.io.get_tfrecord_parser(
                tfr,
                ('image_raw',),
                to_numpy=True,
                decode_images=False
            )
            if not parser:
                continue
            sample_tiles = []
            for i, record in enumerate(dataset):
                if i > 9:
                    break
                image_raw_data = parser(record)[0]
                if normalizer:
                    image_raw_data = normalizer.jpeg_to_jpeg(image_raw_data)
                sample_tiles += [image_raw_data]
            reports += [SlideReport(sample_tiles,
                                    tfr,
                                    tile_px=self.tile_px,
                                    tile_um=self.tile_um,
                                    ignore_thumb_errors=True)]

        # Generate and save PDF
        log.info('Generating PDF (this may take some time)...')
        pdf_report = ExtractionReport(reports, title='TFRecord Report')
        timestring = datetime.now().strftime('%Y%m%d-%H%M%S')
        if exists(dest) and isdir(dest):
            filename = join(dest, f'tfrecord_report-{timestring}.pdf')
        elif sf.util.path_to_ext(dest) == 'pdf':
            filename = join(dest)
        else:
            raise ValueError(f"Could not find destination directory {dest}.")
        pdf_report.save(filename)
        log.info(f'TFRecord report saved to [green]{filename}')

    def tfrecord_heatmap(
        self,
        tfrecord: Union[str, List[str]],
        tile_dict: Dict[int, float],
        outdir: str,
        **kwargs
    ) -> None:
        """Create a tfrecord-based WSI heatmap.

        Uses a dictionary of tile values for heatmap display, and saves to
        the specified directory.

        Args:
            tfrecord (str or list(str)): Path(s) to tfrecord(s).
            tile_dict (dict): Dictionary mapping tfrecord indices to a
                tile-level value for display in heatmap format
            outdir (str): Path to destination directory.

        """
        slide_paths = {
            sf.util.path_to_name(sp): sp for sp in self.slide_paths()
        }
        if not self.tile_px or not self.tile_um:
            raise errors.DatasetError(
                "Dataset tile_px & tile_um must be set to create TFRecords."
            )
        for tfr in sf.util.as_list(tfrecord):
            name = sf.util.path_to_name(tfr)
            if name not in slide_paths:
                raise errors.SlideNotFoundError(f'Unable to find slide {name}')
            sf.util.tfrecord_heatmap(
                tfrecord=tfr,
                slide=slide_paths[name],
                tile_px=self.tile_px,
                tile_um=self.tile_um,
                tile_dict=tile_dict,
                outdir=outdir,
                **kwargs
            )

    def tfrecords(self, source: Optional[str] = None) -> List[str]:
        """Return a list of all tfrecords.

        Args:
            source (str, optional): Only return tfrecords from this dataset
                source. Defaults to None (return all tfrecords in dataset).

        Returns:
            List of tfrecords paths.

        """
        if source and source not in self.sources.keys():
            log.error(f"Dataset {source} not found.")
            return []
        if source is None:
            sources_to_search = list(self.sources.keys())  # type: List[str]
        else:
            sources_to_search = [source]

        tfrecords_list = []
        folders_to_search = []
        for source in sources_to_search:
            if not self._tfrecords_set(source):
                log.warning(f"tfrecords path not set for source {source}")
                continue
            tfrecords = self.sources[source]['tfrecords']
            label = self.sources[source]['label']
            if label is None:
                continue
            tfrecord_path = join(tfrecords, label)
            if not exists(tfrecord_path):
                log.debug(
                    f"TFRecords path not found: {tfrecord_path}"
                )
                continue
            folders_to_search += [tfrecord_path]
        for folder in folders_to_search:
            tfrecords_list += glob(join(folder, "*.tfrecords"))
        tfrecords_list = list(set(tfrecords_list))

        # Filter the list by filters
        if self.annotations is not None:
            slides = self.slides()
            filtered_tfrecords_list = [
                tfrecord for tfrecord in tfrecords_list
                if path_to_name(tfrecord) in slides
            ]
            filtered = filtered_tfrecords_list
        else:
            log.warning("Error filtering TFRecords, are annotations empty?")
            filtered = tfrecords_list

        # Filter by min_tiles
        manifest = self.manifest(filter=False)
        if not all([f in manifest for f in filtered]):
            self.update_manifest()
            manifest = self.manifest(filter=False)
        if self.min_tiles:
            return [
                f for f in filtered
                if f in manifest and manifest[f]['total'] >= self.min_tiles
            ]
        else:
            return [f for f in filtered
                    if f in manifest and manifest[f]['total'] > 0]

    def tfrecords_by_subfolder(self, subfolder: str) -> List[str]:
        """Return a list of all tfrecords in a specific subfolder.

        Ignores any dataset filters.

        Args:
            subfolder (str): Path to subfolder to check for tfrecords.

        Returns:
            List of tfrecords paths.
        """
        tfrecords_list = []
        folders_to_search = []
        for source in self.sources:
            if self.sources[source]['label'] is None:
                continue
            if not self._tfrecords_set(source):
                log.warning(f"tfrecords path not set for source {source}")
                continue
            base_dir = join(
                self.sources[source]['tfrecords'],
                self.sources[source]['label']
            )
            tfrecord_path = join(base_dir, subfolder)
            if not exists(tfrecord_path):
                raise errors.DatasetError(
                    f"Unable to find subfolder [bold]{subfolder}[/] in "
                    f"source [bold]{source}[/], tfrecord directory: "
                    f"[green]{base_dir}"
                )
            folders_to_search += [tfrecord_path]
        for folder in folders_to_search:
            tfrecords_list += glob(join(folder, "*.tfrecords"))
        return tfrecords_list

    def tfrecords_folders(self) -> List[str]:
        """Return folders containing tfrecords."""
        folders = []
        for source in self.sources:
            if self.sources[source]['label'] is None:
                continue
            if not self._tfrecords_set(source):
                log.warning(f"tfrecords path not set for source {source}")
                continue
            folders += [join(
                self.sources[source]['tfrecords'],
                self.sources[source]['label']
            )]
        return folders

    def tfrecords_from_tiles(self, delete_tiles: bool = False) -> None:
        """Create tfrecord files from a collection of raw images.

        Images must be stored in the dataset source(s) tiles directory.

        Args:
            delete_tiles (bool): Remove tiles after storing in tfrecords.

        Returns:
            None
        """
        if not self.tile_px or not self.tile_um:
            raise errors.DatasetError(
                "Dataset tile_px & tile_um must be set to create TFRecords."
            )
        for source in self.sources:
            log.info(f'Working on dataset source {source}')
            config = self.sources[source]
            if not (self._tiles_set(source) and self._tfrecords_set(source)):
                log.error("tiles and/or tfrecords paths not set for "
                          f"source {source}")
                continue
            tfrecord_dir = join(config['tfrecords'], config['label'])
            tiles_dir = join(config['tiles'], config['label'])
            if not exists(tiles_dir):
                log.warn(f'No tiles found for source [bold]{source}')
                continue
            sf.io.write_tfrecords_multi(tiles_dir, tfrecord_dir)
            self.update_manifest()
            if delete_tiles:
                shutil.rmtree(tiles_dir)

    def tfrecords_have_locations(self) -> bool:
        """Check if TFRecords have associated tile location information."""
        for tfr in self.tfrecords():
            try:
                tfr_has_loc = sf.io.tfrecord_has_locations(tfr)
            except errors.TFRecordsError:
                # Encountered when the TFRecord is empty.
                continue
            if not tfr_has_loc:
                log.info(f"{tfr}: Tile location information missing.")
                return False
        return True

    def thumbnails(
        self,
        outdir: str,
        size: int = 512,
        roi: bool = False,
        enable_downsample: bool = True
    ) -> None:
        """Generate square slide thumbnails with black borders of fixed size.

        Saves thumbnails to the specified directory.

        Args:
            size (int, optional): Width/height of thumbnail in pixels.
                Defaults to 512.
            dataset (:class:`slideflow.Dataset`, optional): Dataset
                from which to generate activations. If not supplied, will
                calculate activations for all tfrecords at the tile_px/tile_um
                matching the supplied model, optionally using provided filters
                and filter_blank.
            filters (dict, optional): Dataset filters to use for
                selecting slides. See :meth:`slideflow.Dataset.filter` for
                more information. Defaults to None.
            filter_blank (list(str) or str, optional): Skip slides that have
                blank values in these patient annotation columns.
                Defaults to None.
            roi (bool, optional): Include ROI in the thumbnail images.
                Defaults to False.
            enable_downsample (bool, optional): If True and a thumbnail is not
                embedded in the slide file, downsampling is permitted to
                accelerate thumbnail calculation.
        """
        slide_list = self.slide_paths()
        rois = self.rois()
        log.info(f'Saving thumbnails to [green]{outdir}')
        for slide_path in tqdm(slide_list, desc="Generating thumbnails..."):
            log.debug(f'Working on [green]{path_to_name(slide_path)}[/]...')
            try:
                whole_slide = WSI(slide_path,
                                  tile_px=1000,
                                  tile_um=1000,
                                  stride_div=1,
                                  enable_downsample=enable_downsample,
                                  rois=rois,
                                  verbose=False)
            except errors.MissingROIError:
                log.info(f"Skipping {slide_path}; missing ROI")
                continue
            except Exception as e:
                log.error(
                    f"Error generating thumbnail for {slide_path}: {e}"
                )
                continue
            if roi:
                thumb = whole_slide.thumb(rois=True)
            else:
                thumb = whole_slide.square_thumb(size)
            thumb.save(join(outdir, f'{whole_slide.name}.png'))
        log.info('Thumbnail generation complete.')

    def train_val_split(
        self,
        *args: Any,
        **kwargs: Any
    ) -> Tuple["Dataset", "Dataset"]:
        """Deprecated function."""  # noqa: D401
        warnings.warn(
            "Dataset.train_val_split() is deprecated and will be "
            "removed in a future version. Please use Dataset.split()",
            DeprecationWarning
        )
        return self.split(*args, **kwargs)

    def transform_tfrecords(self, dest: str, **kwargs) -> None:
        """Transform TFRecords, saving to a target path.

        Tfrecords will be saved in the output directory nested by source name.

        Args:
            dest (str): Destination.

        """
        if not exists(dest):
            os.makedirs(dest)
        total = len(self.tfrecords())
        pb = tqdm(total=total)
        for source in self.sources:
            log.debug(f"Working on source {source}")
            tfr_dest = join(dest, source)
            if not exists(tfr_dest):
                os.makedirs(tfr_dest)
            for tfr in self.tfrecords(source=source):
                sf.io.tensorflow.transform_tfrecord(
                    tfr,
                    join(tfr_dest, basename(tfr)),
                    **kwargs
                )
                pb.update(1)
        log.info(f"Saved {total} transformed tfrecords to {dest}.")

    def torch(
        self,
        labels: Optional[Union[Dict[str, Any], str, pd.DataFrame]] = None,
        batch_size: Optional[int] = None,
        rebuild_index: bool = False,
        from_wsi: bool = False,
        **kwargs: Any
    ) -> "DataLoader":
        """Return a PyTorch DataLoader object that interleaves tfrecords.

        The returned dataloader yields a batch of (image, label) for each tile.

        Args:
            labels (dict, str, or pd.DataFrame, optional): If a dict is provided,
                expect a dict mapping slide names to outcome labels. If a str,
                will intepret as categorical annotation header. For linear
                outcomes, or outcomes with manually assigned labels, pass the
                first result of dataset.labels(...). If None, returns slide
                instead of label.
            batch_size (int): Batch size.
            rebuild_index (bool): Re-build index files even if already present.
                Defaults to True.

        Keyword Args:
            augment (str or bool): Image augmentations to perform. Augmentations include:

                * ``'x'``: Random horizontal flip
                * ``'y'``: Random vertical flip
                * ``'r'``: Random 90-degree rotation
                * ``'j'``: Random JPEG compression (50% chance to compress with quality between 50-100)
                * ``'b'``: Random Gaussian blur (10% chance to blur with sigma between 0.5-2.0)
                * ``'n'``: Random :ref:`stain_augmentation` (requires stain normalizer)

                Combine letters to define augmentations, such as ``'xyrjn'``.
                A value of True will use ``'xyrjb'``.
            chunk_size (int, optional): Chunk size for image decoding.
                Defaults to 1.
            drop_last (bool, optional): Drop the last non-full batch.
                Defaults to False.
            from_wsi (bool): Generate predictions from tiles dynamically
                extracted from whole-slide images, rather than TFRecords.
                Defaults to False (use TFRecords).
            incl_loc (bool, optional): Include loc_x and loc_y (image tile
                center coordinates, in base / level=0 dimension) as additional
                returned variables. Defaults to False.
            incl_slidenames (bool, optional): Include slidenames as third returned
                variable. Defaults to False.
            infinite (bool, optional): Infinitely repeat data. Defaults to True.
            max_size (bool, optional): Unused argument present for legacy
                compatibility; will be removed.
            model_type (str, optional): Used to generate random labels
                (for StyleGAN2). Not required. Defaults to 'categorical'.
            num_replicas (int, optional): Number of GPUs or unique instances which
                will have their own DataLoader. Used to interleave results among
                workers without duplications. Defaults to 1.
            num_workers (int, optional): Number of DataLoader workers.
                Defaults to 2.
            normalizer (:class:`slideflow.norm.StainNormalizer`, optional):
                Normalizer. Defaults to None.
            onehot (bool, optional): Onehot encode labels. Defaults to False.
            persistent_workers (bool, optional): Sets the DataLoader
                persistent_workers flag. Defaults toNone (4 if not using a SPAMS
                normalizer, 1 if using SPAMS).
            pin_memory (bool, optional): Pin memory to GPU. Defaults to True.
            pool (multiprocessing.Pool): Shared multiprocessing pool. Useful
                if from_wsi=True, for sharing a unified processing pool between
                dataloaders. Defaults to None.
            prefetch_factor (int, optional): Number of batches to prefetch in each
                SlideflowIterator. Defaults to 1.
            rank (int, optional): Worker ID to identify this worker.
                Used to interleave results.
                among workers without duplications. Defaults to 0 (first worker).
            rois (list(str), optional): List of ROI paths. Only used if
                from_wsi=True.  Defaults to None.
            roi_method (str, optional): Method for extracting ROIs. Only used if
                from_wsi=True. Defaults to 'auto'.
            standardize (bool, optional): Standardize images to mean 0 and
                variance of 1. Defaults to True.
            tile_um (int, optional): Size of tiles to extract from WSI, in
                microns. Only used if from_wsi=True. Defaults to None.
            transform (Callable, optional): Arbitrary torchvision transform
                function. Performs transformation after augmentations but
                before standardization. Defaults to None.
            tfrecord_parser (Callable, optional): Custom parser for TFRecords.
                Defaults to None.

        """
        from slideflow.io.torch import interleave_dataloader

        if isinstance(labels, str) and not exists(labels):
            labels = self.labels(labels)[0]
        if self.tile_px is None:
            raise errors.DatasetError("tile_px and tile_um must be non-zero"
                                      "to create dataloaders.")
        if self._clip not in (None, {}) and from_wsi:
            log.warning("Dataset clipping is disabled when `from_wsi=True`")

        if from_wsi:
            tfrecords = self.slide_paths()
            kwargs['rois'] = self.rois()
            kwargs['tile_um'] = self.tile_um
            kwargs['img_size'] = self.tile_px
            indices = None
            clip = None
        else:
            self.build_index(rebuild_index)
            tfrecords = self.tfrecords()
            if not tfrecords:
                raise errors.TFRecordsNotFoundError
            self.verify_img_format(progress=False)
            _idx_dict = self.load_indices()
            indices = [_idx_dict[path_to_name(tfr)] for tfr in tfrecords]
            clip = self._clip

        if self.prob_weights:
            prob_weights = [self.prob_weights[tfr] for tfr in tfrecords]
        else:
            prob_weights = None

        return interleave_dataloader(tfrecords=tfrecords,
                                     batch_size=batch_size,
                                     labels=labels,
                                     num_tiles=self.num_tiles,
                                     prob_weights=prob_weights,
                                     clip=clip,
                                     indices=indices,
                                     from_wsi=from_wsi,
                                     **kwargs)

    def unclip(self) -> "Dataset":
        """Return a dataset object with all clips removed.

        Returns:
            :class:`slideflow.Dataset`: Dataset with clips removed.

        """
        ret = copy.deepcopy(self)
        ret._clip = {}
        return ret

    def update_manifest(self, force_update: bool = False) -> None:
        """Update tfrecord manifests.

        Args:
            forced_update (bool, optional): Force regeneration of the
                manifests from scratch.

        """
        tfrecords_folders = self.tfrecords_folders()
        for tfr_folder in tfrecords_folders:
            sf.io.update_manifest_at_dir(
                directory=tfr_folder,
                force_update=force_update
            )

    def update_annotations_with_slidenames(
        self,
        annotations_file: str
    ) -> None:
        """Automatically associated slide names and paths in the annotations.

        Attempts to automatically associate slide names from a directory
        with patients in a given annotations file, skipping any slide names
        that are already present in the annotations file.

        Args:
            annotations_file (str): Path to annotations file.

        """
        header, _ = sf.util.read_annotations(annotations_file)
        slide_list = self.slide_paths(apply_filters=False)

        # First, load all patient names from the annotations file
        try:
            patient_index = header.index('patient')
        except ValueError:
            raise errors.AnnotationsError(
                f"Patient header {'patient'} not found in annotations."
            )
        patients = []
        pt_to_slide = {}
        with open(annotations_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader)
            for row in csv_reader:
                patients.extend([row[patient_index]])
        patients = list(set(patients))
        log.debug(f"Number of patients in annotations: {len(patients)}")
        log.debug(f"Slides found: {len(slide_list)}")

        # Then, check for sets of slides that would match to the same patient;
        # due to ambiguity, these will be skipped.
        n_occur = {}
        for slide in slide_list:
            if _shortname(slide) not in n_occur:
                n_occur[_shortname(slide)] = 1
            else:
                n_occur[_shortname(slide)] += 1
        slides_to_skip = [s for s in slide_list if n_occur[_shortname(s)] > 1]

        # Next, search through the slides folder for all valid slide files
        for file in slide_list:
            slide = path_to_name(file)
            # First, skip this slide due to ambiguity if needed
            if slide in slides_to_skip:
                log.warning(f"Skipping slide {slide} due to ambiguity")
            # Then, make sure the shortname and long name
            # aren't both in the annotation file
            if ((slide != _shortname(slide))
               and (slide in patients)
               and (_shortname(slide) in patients)):
                log.warning(f"Skipping slide {slide} due to ambiguity")
            # Check if either the slide name or the shortened version
            # are in the annotation file
            if any(x in patients for x in [slide, _shortname(slide)]):
                slide = slide if slide in patients else _shortname(slide)
                pt_to_slide.update({slide: slide})

        # Now, write the assocations
        n_updated = 0
        n_missing = 0
        with open(annotations_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader)
            with open('temp.csv', 'w') as csv_outfile:
                csv_writer = csv.writer(csv_outfile, delimiter=',')

                # Write to existing "slide" column in the annotations file,
                # otherwise create new column
                try:
                    slide_index = header.index('slide')
                except ValueError:
                    header.extend(['slide'])
                    csv_writer.writerow(header)
                    for row in csv_reader:
                        patient = row[patient_index]
                        if patient in pt_to_slide:
                            row.extend([pt_to_slide[patient]])
                            n_updated += 1
                        else:
                            row.extend([""])
                            n_missing += 1
                        csv_writer.writerow(row)
                else:
                    csv_writer.writerow(header)
                    for row in csv_reader:
                        pt = row[patient_index]
                        # Only write column if no slide is in the annotation
                        if (pt in pt_to_slide) and (row[slide_index] == ''):
                            row[slide_index] = pt_to_slide[pt]
                            n_updated += 1
                        elif ((pt not in pt_to_slide)
                              and (row[slide_index] == '')):
                            n_missing += 1
                        csv_writer.writerow(row)
        if n_updated:
            log.info(f"Done; associated slides with {n_updated} annotations.")
            if n_missing:
                log.info(f"Slides not found for {n_missing} annotations.")
        elif n_missing:
            log.debug(f"Slides missing for {n_missing} annotations.")
        else:
            log.debug("Annotations up-to-date, no changes made.")

        # Finally, backup the old annotation file and overwrite
        # existing with the new data
        backup_file = f"{annotations_file}.backup"
        if exists(backup_file):
            os.remove(backup_file)
        assert isinstance(annotations_file, str)
        shutil.move(annotations_file, backup_file)
        shutil.move('temp.csv', annotations_file)

    def verify_annotations_slides(self) -> None:
        """Verify that annotations are correctly loaded."""
        if self.annotations is None:
            log.warn("Annotations not loaded.")
            return

        # Verify no duplicate slide names are found
        ann = self.annotations.loc[self.annotations.slide.isin(self.slides())]
        if not ann.slide.is_unique:
            raise errors.AnnotationsError(
                "Duplicate slide names detected in the annotation file."
            )

        # Verify that there are no tfrecords with the same name.
        # This is a problem because the tfrecord name is used to
        # identify the slide.
        tfrecords = self.tfrecords()
        if len(tfrecords):
            tfrecord_names = [sf.util.path_to_name(tfr) for tfr in tfrecords]
            if not len(set(tfrecord_names)) == len(tfrecord_names):
                duplicate_tfrs = [
                    tfr for tfr in tfrecords
                    if tfrecord_names.count(sf.util.path_to_name(tfr)) > 1
                ]
                raise errors.AnnotationsError(
                    "Multiple TFRecords with the same names detected: {}".format(
                        ', '.join(duplicate_tfrs)
                    )
                )

        # Verify all slides in the annotation column are valid
        n_missing = len(self.annotations.loc[
            (self.annotations.slide.isin(['', ' '])
             | self.annotations.slide.isna())
        ])
        if n_missing == 1:
            log.warn("1 patient does not have a slide assigned.")
        if n_missing > 1:
            log.warn(f"{n_missing} patients do not have a slide assigned.")

    def verify_img_format(self, *, progress: bool = True) -> Optional[str]:
        """Verify that all tfrecords have the same image format (PNG/JPG).

        Returns:
            str: image format (png or jpeg)

        """
        tfrecords = self.tfrecords()
        if len(tfrecords):
            with mp.Pool(sf.util.num_cpu(),
                         initializer=sf.util.set_ignore_sigint) as pool:
                img_formats = []
                mapped = pool.imap_unordered(
                    sf.io.detect_tfrecord_format,
                    tfrecords
                )
                if progress:
                    mapped = track(
                        mapped,
                        description="Verifying tfrecord formats...",
                        transient=True
                    )
                for *_, fmt in mapped:
                    if fmt is not None:
                        img_formats += [fmt]
                if len(set(img_formats)) > 1:
                    log_msg = "Mismatched TFRecord image formats:\n"
                    for tfr, fmt in zip(tfrecords, img_formats):
                        log_msg += f"{tfr}: {fmt}\n"
                    log.error(log_msg)
                    raise errors.MismatchedImageFormatsError(
                        "Mismatched TFRecord image formats detected"
                    )
                if len(img_formats):
                    return img_formats[0]
                else:
                    return None
        else:
            return None

    def verify_slide_names(self, allow_errors: bool = False) -> bool:
        """Verify that slide names inside TFRecords match the file names.

        Args:
            allow_errors (bool): Do not raise an error if there is a mismatch.
                Defaults to False.

        Returns:
            bool: If all slide names inside TFRecords match the TFRecord
                file names.

        Raises:
            sf.errors.MismatchedSlideNamesError: If any slide names inside
                TFRecords do not match the TFRecord file names,
                and allow_errors=False.

        """
        tfrecords = self.tfrecords()
        if len(tfrecords):
            pb = track(
                tfrecords,
                description="Verifying tfrecord slide names...",
                transient=True
            )
            for tfr in pb:
                first_record = sf.io.get_tfrecord_by_index(tfr, 0)
                if first_record['slide'] == sf.util.path_to_name(tfr):
                    continue
                elif allow_errors:
                    return False
                else:
                    raise errors.MismatchedSlideNamesError(
                        "Mismatched slide name in TFRecord {}: expected slide "
                        "name {} based on filename, but found {}. ".format(
                            tfr,
                            sf.util.path_to_name(tfr),
                            first_record['slide']
                        )
                )
        return True
