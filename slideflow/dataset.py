import copy
import queue
import threading
import time
import os
import csv
import shutil
import types
import multiprocessing
import shapely.geometry as sg
import numpy as np
import pandas as pd
from random import shuffle
from glob import glob
from datetime import datetime
from tqdm import tqdm
from os.path import isdir, join, exists, dirname, basename
from multiprocessing.dummy import Pool as DPool
from collections import defaultdict

import slideflow as sf
from slideflow import errors
from slideflow.slide import WSI, ExtractionReport, SlideReport
from slideflow.util import log, TCGA, _shortname, path_to_name
from slideflow.util import colors as col


def _tile_extractor(path, tfrecord_dir, tiles_dir, reports, tma, qc,
                    wsi_kwargs, generator_kwargs, qc_kwargs):
    """Internal function to extract tiles. Slide processing needs to be
    process-isolated when num_workers > 1 .

    Args:
        tfrecord_dir (str): Path to TFRecord directory.
        tiles_dir (str): Path to tiles directory (loose format).
        reports (dict): Multiprocessing-enabled dict.
        tma (bool): Slides are in TMA format.
        qc (bool): Quality control method.
        wsi_kwargs (dict): Keyword arguments for sf.WSI.
        generator_kwargs (dict): Keyword arguments for WSI.extract_tiles()
        qc_kwargs(dict): Keyword arguments for quality control.
    """
    # Record function arguments in case we need to re-call the function
    # (for corrupt tiles)
    local_args = locals()
    log.handlers[0].flush_line = True
    try:
        log.debug(f'Extracting tiles for {path_to_name(path)}')
        if tma:
            slide = sf.slide.TMA(
                path=path,
                tile_px=wsi_kwargs['tile_px'],
                tile_um=wsi_kwargs['tile_um'],
                stride_div=wsi_kwargs['stride_div'],
                enable_downsample=wsi_kwargs['enable_downsample'],
                report_dir=tfrecord_dir
            )
        else:
            slide = sf.slide.WSI(path, **wsi_kwargs)
        # Apply quality control (blur filtering)
        if qc:
            slide.qc(method=qc, **qc_kwargs)
        if not slide.loaded_correctly():
            return
        try:
            report = slide.extract_tiles(
                tfrecord_dir=tfrecord_dir,
                tiles_dir=tiles_dir,
                **generator_kwargs
            )
        except errors.TileCorruptionError:
            if wsi_kwargs['enable_downsample']:
                log.warning(f'{path} corrupt; disabling downsampling')
                report = _tile_extractor(**local_args)
            else:
                log.error(f'{path} corrupt; skipping')
                return
        del slide
        reports.update({path: report})
    except (KeyboardInterrupt, SystemExit):
        print('Exiting...')
        return


def _fill_queue(slide_list, q, q_size, buffer=None):
    for path in slide_list:
        warned = False
        if buffer:
            while True:
                if q.qsize() < q_size:
                    try:
                        buffered = join(buffer, basename(path))
                        shutil.copy(path, buffered)
                        q.put(buffered)
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


def split_patients_list(patients_dict, n, balance=None, preserved_site=False):
    """Splits a dictionary of patients into n groups,
    balancing according to key "balance" if provided.

    Args:
        patients_dict (dict): Nested citionary mapping patient names to
            dict of outcomes: labels
        n (int): Number of splits to generate.
        balance (str, optional): Annotation header to balance splits across.
        preserved_site (bool, optional): Use site-preserved cross-validation,
            assuming site information is under the nested dictionary key
            'site' in patients_dict.

    Returns:
        List of patient splits
    """

    if preserved_site and not sf.util.CPLEX_AVAILABLE:
        raise errors.CPLEXNotFoundError
    patient_list = list(patients_dict.keys())
    shuffle(patient_list)

    def flatten(arr):
        '''Flattens an array'''
        return [y for x in arr for y in x]

    def split(a, n):
        '''Function to split a list into n components'''
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
                for i in range(n))

    if balance:
        # Get patient outcome labels
        patient_outcome_labels = [
            patients_dict[p][balance] for p in patient_list
        ]
        # Get unique outcomes
        unique_labels = list(set(patient_outcome_labels))
        n_unique = len(set(unique_labels))
        if preserved_site:
            # Delayed import in case CPLEX not installed
            import slideflow.io.preservedsite.crossfolds as cv

            site_list = [patients_dict[p]['site'] for p in patient_list]
            df = pd.DataFrame(
                list(zip(patient_list, patient_outcome_labels, site_list)),
                columns=['patient', 'outcome_label', 'site']
            )
            df = cv.generate(
                df,
                'outcome_label',
                unique_labels,
                crossfolds=n,
                target_column='CV',
                patient_column='patient',
                site_column='site'
            )
            log.info(col.bold("Train/val split with Preserved-Site Cross-Val"))
            log.info(col.bold(
                "Category\t" + "\t".join([str(cat) for cat in range(n_unique)])
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
        else:
            # Now, split patient_list according to outcomes
            pt_by_outcome = [
                [p for p in patient_list if patients_dict[p][balance] == uo]
                for uo in unique_labels
            ]
            # Then, for each sublist, split into n components
            pt_by_outcome_by_n = [
                list(split(sub_l, n)) for sub_l in pt_by_outcome
            ]
            # Print splitting as a table
            log.info(col.bold(
                "Category\t" + "\t".join([str(cat) for cat in range(n_unique)])
            ))
            for k in range(n):
                matching = [str(len(clist[k])) for clist in pt_by_outcome_by_n]
                log.info(f"K-fold-{k}\t" + "\t".join(matching))
            # Join sublists
            splits = [
                flatten([
                    item[ni] for item in pt_by_outcome_by_n
                ]) for ni in range(n)
            ]
    else:
        splits = list(split(patient_list, n))
    return splits


class Dataset:
    """Object to supervise organization of slides, tfrecords, and tiles
    across one or more sources in a stored configuration file."""

    def __init__(self, config, sources, tile_px, tile_um, annotations=None,
                 filters=None, filter_blank=None, min_tiles=0):
        self.tile_px = tile_px
        self.tile_um = tile_um
        self._filters = filters if filters else {}
        self._filter_blank = filter_blank if filter_blank is not None else []
        if not isinstance(self._filter_blank, list):
            self._filter_blank = [self.filter_blank]
        self._min_tiles = min_tiles
        self._clip = {}
        self.prob_weights = None
        self._config = config
        self.annotations = None
        self.annotations_file = None
        loaded_config = sf.util.load_json(config)
        sources = sources if isinstance(sources, list) else [sources]
        try:
            self.sources = {
                k: v for k, v in loaded_config.items() if k in sources
            }
            self.sources_names = list(self.sources.keys())
        except KeyError:
            sources_list = ', '.join(sources)
            raise errors.SourceNotFoundError(sources_list, config)
        if (tile_px is not None) and (tile_um is not None):
            label = f"{tile_px}px_{tile_um}um"
        else:
            label = None
        for source in self.sources:
            self.sources[source]['label'] = label
        if annotations is not None:
            self.load_annotations(annotations)

    def __repr__(self):
        _b = "Dataset(config={!r}, sources={!r}, tile_px={!r}, tile_um={!r})"
        return _b.format(
            self._config,
            self.sources_names,
            self.tile_px,
            self.tile_um
        )

    @property
    def num_tiles(self):
        """Returns the total number of tiles in the tfrecords in this dataset,
        after filtering/clipping.
        """
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
    def filters(self):
        """Returns the active filters, if any."""
        return self._filters

    @property
    def filter_blank(self):
        """Returns the active filter_blank filter, if any."""
        return self._filter_blank

    @property
    def min_tiles(self):
        """Returns the active min_tiles filter, if any (defaults to 0)."""
        return self._min_tiles

    @property
    def filtered_annotations(self):
        filtered_idx = self.annotations[TCGA.slide].isin(self.slides())
        return self.annotations[filtered_idx]

    @property
    def img_format(self):
        return self.verify_img_format()

    def _assert_size_matches_hp(self, hp):
        """Checks if dataset tile size (px/um) matches the given parameters."""
        if isinstance(hp, dict):
            hp_px = hp['tile_px']
            hp_um = hp['tile_um']
        elif isinstance(hp, sf.model.ModelParams):
            hp_px = hp.tile_px
            hp_um = hp.tile_um
        else:
            raise ValueError(f"Unrecognized hyperparameter type {type(hp)}")
        if self.tile_px != hp_px or self.tile_um != hp_um:
            d_sz = f'({self.tile_px}px, {self.tile_um}um)'
            m_sz = f'({hp_px}px, {hp_um}um)'
            msg = f"Dataset tile size {d_sz} does not match model {m_sz}"
            raise ValueError(msg)

    def load_annotations(self, annotations):
        # Load annotations
        if isinstance(annotations, str):
            if not exists(annotations):
                msg = f'Unable to find annotations file {annotations}'
                raise errors.AnnotationsError(msg)
            try:
                self.annotations = pd.read_csv(annotations, dtype=str)
                self.annotations.fillna('', inplace=True)
                self.annotations_file = annotations
            except pd.errors.EmptyDataError:
                log.error(f"Unable to load empty annotations {annotations}")
        elif isinstance(annotations, pd.core.frame.DataFrame):
            self.annotations = annotations
            self.annotations.fillna('', inplace=True)
        else:
            msg = 'Invalid annotations format; expected path or DataFrame'
            raise errors.AnnotationsError(msg)

        # Check annotations
        if len(self.annotations.columns) == 1:
            msg = "Only one annotations column detected (is it in CSV format?)"
            raise errors.AnnotationsError(msg)
        if len(self.annotations.columns) != len(set(self.annotations.columns)):
            msg = "Annotations file has duplicate headers; all must be unique"
            raise errors.AnnotationsError(msg)
        if TCGA.patient not in self.annotations.columns:
            msg = f'Patient identifier {TCGA.patient} missing in annotations.'
            raise errors.AnnotationsError(msg)
        if TCGA.slide not in self.annotations.columns:
            log.info(f"Column '{TCGA.slide}' missing in annotations.")
            log.info("Attempting to associate patients with slides...")
            self.update_annotations_with_slidenames(annotations)
            self.load_annotations(annotations)

    def balance(self, headers=None, strategy='category', force=False):
        """Returns a dataset with prob_weights reflecting balancing per tile,
        slide, patient, or category.

        Saves balancing information to the dataset variable prob_weights, which
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
            balanced :class:`slideflow.dataset.Dataset` object.
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
            msg = "Unable to balance; no tfrecords found."
            raise errors.DatasetBalanceError(msg)

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
            # Ensure that header is not type 'float'
            if not isinstance(headers, list):
                headers = [headers]
            if any(ret.is_float(h) for h in headers) and not force:
                msg = f"Headers {','.join(headers)} appear to be `float`."
                msg += "Categorical outcomes required for balancing. "
                msg += "To force balancing with these outcomes, pass "
                msg += "`force=True` to Dataset.balance()"
                raise errors.DatasetBalanceError(msg)

            labels, _ = ret.labels(headers, use_float=False)
            cats, cat_prob, tfr_cats = {}, {}, {}
            for tfrecord in tfrecords:
                slide = path_to_name(tfrecord)
                balance_cat = labels[slide]
                if not isinstance(balance_cat, list):
                    balance_cat = [balance_cat]
                balance_cat = '-'.join(map(str, balance_cat))
                tfr_cats[tfrecord] = balance_cat
                tiles = totals[tfrecord]
                if balance_cat not in cats:
                    cats.update({balance_cat: {
                        'num_slides': 1,
                        'num_tiles': tiles
                    }})
                else:
                    cats[balance_cat]['num_slides'] += 1
                    cats[balance_cat]['num_tiles'] += tiles
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

    def build_index(self, force=True):
        """Builds index files for TFRecords, required for PyTorch."""

        from slideflow.util import tfrecord2idx

        def create_index(filename):
            nonlocal force
            index_name = join(
                dirname(filename),
                path_to_name(filename)+'.index'
            )
            if not exists(index_name) or force:
                tfrecord2idx.create_index(filename, index_name)

        pool = DPool(16)
        for _ in tqdm(pool.imap(create_index, self.tfrecords()),
                      desc='Creating index files...',
                      ncols=80,
                      total=len(self.tfrecords()),
                      leave=False):
            pass

    def clear_filters(self):
        """Returns a dataset with all filters cleared.

        Returns:
            :class:`slideflow.dataset.Dataset` object.
        """

        ret = copy.deepcopy(self)
        ret._filters = {}
        ret._filter_blank = []
        ret._min_tiles = 0
        return ret

    def clip(self, max_tiles=0, strategy=None, headers=None):
        '''Returns a dataset clipped to either a fixed maximum number of tiles
        per tfrecord, or to the min number of tiles per patient or category.

        Args:
            max_tiles (int, optional): Clip the maximum number of tiles per
                tfrecord to this number.
            strategy (str, optional): 'slide', 'patient', or 'category'.
                Clip the maximum number of tiles to the minimum tiles seen
                across slides, patients, or categories. If 'category', headers
                must be provided. Defaults to None.
            headers (list of str, optional): List of annotation headers to use
                if clipping by minimum category count (strategy='category').
                Defaults to None.

        Returns:
            clipped :class:`slideflow.dataset.Dataset` object.
        '''

        if strategy == 'category' and not headers:
            msg = "headers must be provided if clip strategy is 'category'."
            raise errors.DatasetClipError(msg)
        if strategy is None and headers is not None:
            strategy = 'category'
        if strategy is None and headers is None and not max_tiles:
            return self

        ret = copy.deepcopy(self)
        manifest = ret.manifest()
        tfrecords = ret.tfrecords()
        slides = [path_to_name(tfr) for tfr in tfrecords]
        totals = {tfr: manifest[tfr]['total'] for tfr in tfrecords}

        if not tfrecords:
            msg = "Unable to clip; no tfrecords found."
            raise errors.DatasetClipError(msg)
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
            labels, _ = ret.labels(headers, use_float=False)
            categories = {}
            cat_fraction = {}
            tfr_cats = {}
            for tfrecord in tfrecords:
                slide = path_to_name(tfrecord)
                balance_category = labels[slide]
                if not isinstance(balance_category, list):
                    balance_category = [balance_category]
                balance_category = '-'.join(map(str, balance_category))
                tfr_cats[tfrecord] = balance_category
                tiles = totals[tfrecord]
                if balance_category not in categories:
                    categories[balance_category] = tiles
                else:
                    categories[balance_category] += tiles

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

    def extract_tiles(self, save_tiles=False, save_tfrecords=True,
                      source=None, stride_div=1, enable_downsample=True,
                      roi_method='inside', skip_missing_roi=False,
                      skip_extracted=True, tma=False, randomize_origin=False,
                      buffer=None, num_workers=1, q_size=4, qc=None,
                      report=True, **kwargs):

        """Extract tiles from a group of slides, saving extracted tiles to
        either loose image or in TFRecord binary format.

        Args:
            save_tiles (bool, optional): Save images of extracted tiles to
                project tile directory. Defaults to False.
            save_tfrecords (bool, optional): Save compressed image data from
                extracted tiles into TFRecords in the corresponding TFRecord
                directory. Defaults to True.
            source (str, optional): Name of dataset source from which to select
                slides for extraction. Defaults to None. If not provided, will
                default to all sources in project.
            stride_div (int, optional): Stride divisor for tile extraction.
                A stride of 1 will extract non-overlapping tiles.
                A stride_div of 2 will extract overlapping tiles, with a stride
                equal to 50% of the tile width. Defaults to 1.
            enable_downsample (bool, optional): Enable downsampling for slides.
                This may result in corrupted image tiles if downsampled slide
                layers are corrupted or incomplete. Defaults to True.
            roi_method (str, optional): Either 'inside', 'outside' or 'ignore'.
                Indicates whether tiles are extracted inside or outside ROIs,
                or if ROIs are ignored entirely. Defaults to 'inside'.
            skip_missing_roi (bool, optional): Skip slides missing ROIs.
                Defaults to False.
            skip_extracted (bool, optional): Skip slides that have already
                been extracted. Defaults to True.
            tma (bool, optional): Reads slides as Tumor Micro-Arrays (TMAs),
                detecting and extracting tumor cores. Defaults to False.
                Experimental function with limited testing.
            randomize_origin (bool, optional): Randomize pixel starting
                position during extraction. Defaults to False.
            buffer (str, optional): Slides will be copied to this directory
                before extraction. Defaults to None. Using an SSD or ramdisk
                buffer vastly improves tile extraction speed.
            num_workers (int, optional): Extract tiles from this many slides
                simultaneously. Defaults to 1.
            q_size (int, optional): Size of queue when using a buffer.
                Defaults to 4.
            qc (str, optional): 'otsu', 'blur', 'both', or None. Perform blur
                detection quality control - discarding tiles with detected
                out-of-focus regions or artifact - and/or otsu's method.
                Increases tile extraction time. Defaults to None.
            report (bool, optional): Save a PDF report of tile extraction.
                Defaults to True.

        Keyword Args:
            normalizer (str, optional): Normalization strategy.
                Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image.
                If None, will use slideflow.slide.norm_tile.jpg.
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
            full_core (bool, optional): Only used if extracting from TMA.
                If True, will save entire TMA core as image.
                Otherwise, will extract sub-images from each core using the
                given tile micron size. Defaults to False.
            shuffle (bool, optional): Shuffle tiles prior to storage in
                tfrecords. Defaults to True.
            num_threads (int, optional): Number of workers threads for each
                tile extractor.
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
        """

        if not save_tiles and not save_tfrecords:
            msg = 'Either save_tiles or save_tfrecords must be true.'
            raise errors.DatasetError(msg)
        if q_size < num_workers:
            msg = f"q_size ({q_size}) < num_workers {num_workers}; "
            msg += "some workers will not be used"
            log.warn(msg)
        if not self.tile_px or not self.tile_um:
            msg = "Dataset tile_px and tile_um must be != 0 to extract tiles"
            raise errors.DatasetError(msg)

        if source:
            sources = [source] if not isinstance(source, list) else source
        else:
            sources = self.sources
        pdf_report = None
        self.verify_annotations_slides()

        # Set up kwargs for tile extraction generator and quality control
        qc_kwargs = {k[3:]: v for k, v in kwargs.items() if k[:3] == 'qc_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:3] != 'qc_'}
        sf.slide.log_extraction_params(**kwargs)

        for source in sources:
            log.info(f'Working on dataset source {col.bold(source)}...')
            roi_dir = self.sources[source]['roi']
            src_conf = self.sources[source]
            if 'dry_run' not in kwargs or not kwargs['dry_run']:
                if save_tfrecords:
                    tfrecord_dir = join(
                        src_conf['tfrecords'],
                        src_conf['label']
                    )
                else:
                    tfrecord_dir = None
                if save_tiles:
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
                    log.info(f'Re-extracting {len(interrupted)} interrupted')
                    for interrupted_slide in interrupted:
                        log.info(interrupted_slide)
                        if interrupted_slide in done:
                            del done[done.index(interrupted_slide)]

                slide_list = [
                    s for s in slide_list if path_to_name(s) not in done
                ]
                if len(done):
                    log.info(f'Skipping {len(done)} slides; already done.')
            _tail = f"({self.tile_um} um, {self.tile_px} px)"
            log.info(f'Extracting tiles from {len(slide_list)} slides{_tail}')

            # Verify slides and estimate total number of tiles
            log.info('Verifying slides...')
            total_tiles = 0
            for slide_path in tqdm(slide_list,
                                   leave=False,
                                   desc="Verifying slides..."):
                if tma:
                    slide = sf.slide.TMA(
                        slide_path,
                        self.tile_px,
                        self.tile_um,
                        stride_div,
                        silent=True
                    )
                else:
                    slide = sf.slide.WSI(
                        slide_path,
                        self.tile_px,
                        self.tile_um,
                        stride_div,
                        roi_dir=roi_dir,
                        roi_method=roi_method,
                        skip_missing_roi=skip_missing_roi,
                        silent=True
                    )
                est = slide.estimated_num_tiles
                log.debug(f"Estimated tiles for slide {slide.name}: {est}")
                total_tiles += est
                del slide
            log.info(f'Total estimated tiles to extract: {total_tiles}')

            # Use multithreading if specified, extracting tiles
            # from all slides in the filtered list
            if len(slide_list):
                q = queue.Queue()
                task_finished = False
                manager = multiprocessing.Manager()
                ctx = multiprocessing.get_context('fork')
                reports = manager.dict()
                counter = manager.Value('i', 0)
                counter_lock = manager.Lock()

                if total_tiles:
                    pb = sf.util.ProgressBar(
                        total_tiles,
                        counter_text='tiles',
                        leadtext='Extracting tiles... ',
                        show_counter=True,
                        show_eta=True,
                        mp_counter=counter,
                        mp_lock=counter_lock
                    )
                    pb.auto_refresh(0.1)
                else:
                    pb = None

                # If only one worker, use a single shared multiprocessing pool
                if num_workers == 1:
                    # Detect CPU cores if num_threads not specified
                    if 'num_threads' not in kwargs:
                        num_threads = os.cpu_count()
                        if num_threads is None:
                            num_threads = 8
                    else:
                        num_threads = kwargs['num_threads']
                    log.info(f'Extracting tiles with {num_threads} threads')
                    kwargs['pool'] = multiprocessing.Pool(num_threads)
                wsi_kwargs = {
                    'tile_px': self.tile_px,
                    'tile_um': self.tile_um,
                    'stride_div': stride_div,
                    'enable_downsample': enable_downsample,
                    'roi_dir': roi_dir,
                    'roi_method': roi_method,
                    'randomize_origin': randomize_origin,
                    'skip_missing_roi': skip_missing_roi,
                    'pb_counter': counter,
                    'counter_lock': counter_lock
                }
                extraction_kwargs = {
                    'tfrecord_dir': tfrecord_dir,
                    'tiles_dir': tiles_dir,
                    'reports': reports,
                    'tma': tma,
                    'qc': qc,
                    'generator_kwargs': kwargs,
                    'qc_kwargs': qc_kwargs,
                    'wsi_kwargs': wsi_kwargs
                }

                # Worker to grab slide path from queue and start extraction
                def worker():
                    while not task_finished:
                        path = q.get()
                        if path is None:
                            q.task_done()
                            break
                        if num_workers > 1:
                            process = ctx.Process(target=_tile_extractor,
                                                  args=(path,),
                                                  kwargs=extraction_kwargs)
                            process.start()
                            process.join()
                        else:
                            _tile_extractor(path, **extraction_kwargs)
                        if buffer:
                            os.remove(path)
                        q.task_done()

                # Start the worker threads
                threads = [
                    threading.Thread(target=worker) for _ in range(num_workers)
                ]
                for thread in threads:
                    thread.start()

                # Put each slide path into queue
                _fill_queue(slide_list, q, q_size, buffer=buffer)
                task_finished = True
                for thread in threads:
                    thread.join()
                if pb:
                    pb.end()
                if report:
                    log.info('Generating PDF (this may take some time)...', )
                    rep_vals = reports.values()
                    num_slides = len(slide_list)
                    img_kwargs = defaultdict(lambda: None)
                    img_kwargs.update(kwargs)
                    img_kwargs = sf.slide._update_kw_with_defaults(img_kwargs)
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
                    pdf_report = sf.slide.ExtractionReport(
                        rep_vals,
                        meta=report_meta
                    )
                    _time = datetime.now().strftime('%Y%m%d-%H%M%S')
                    pdf_dir = tfrecord_dir if tfrecord_dir else ''
                    pdf_report.save(
                        join(pdf_dir, f'tile_extraction_report-{_time}.pdf')
                    )
                    warn_path = join(pdf_dir, f'warn_report-{_time}.txt')
                    with open(warn_path, 'w') as warn_f:
                        warn_f.write(pdf_report.warn_txt)

        # Update manifest & rebuild indices
        self.update_manifest(force_update=True)
        self.build_index(True)
        return pdf_report

    def extract_tiles_from_tfrecords(self, dest):
        """Extracts tiles from a set of TFRecords.

        Args:
            dest (str): Path to directory in which to save tile images.
                If None, uses dataset default. Defaults to None.
        """
        for source in self.sources:
            to_extract_tfrecords = self.tfrecords(source=source)
            if dest:
                tiles_dir = dest
            else:
                tiles_dir = join(self.sources[source]['tiles'],
                                 self.sources[source]['label'])
                if not exists(tiles_dir):
                    os.makedirs(tiles_dir)
            for tfr in to_extract_tfrecords:
                sf.io.extract_tiles(tfr, tiles_dir)

    def filter(self, *args, **kwargs):
        """Return a filtered dataset.

        Keyword Args:
            filters (dict): Filters dict to use when selecting tfrecords.
                See :meth:`get_dataset` documentation for more information
                on filtering. Defaults to None.
            filter_blank (list): Exclude slides blank in these columns.
                Defaults to None.
            min_tiles (int): Filter out tfrecords that have less than this
                minimum number of tiles.

        Returns:
            :class:`slideflow.dataset.Dataset` object.
        """
        if len(args) == 1 and 'filters' not in kwargs:
            kwargs['filters'] = args[0]
        elif len(args):
            msg = "filter() accepts either one argument (filters), or any "
            msg += "combination of keywords (filters, filter_blank, min_tiles)"
            raise ValueError(msg)
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

    def harmonize_labels(self, *args, header=None):
        '''Returns categorical label assignments to int, harmonized with
        another dataset to ensure consistency between datasets.

        Args:
            *args (:class:`slideflow.Dataset`): Any number of Datasets.
            header (str): Categorical annotation header.

        Returns:
            Dict mapping slide names to categories.
        '''

        if header is None:
            raise ValueError("Must supply kwarg 'header'")
        if not isinstance(header, str):
            raise ValueError('Harmonized labels require a single header.')

        _, my_unique = self.labels(header, use_float=False)
        other_uniques = [
            np.array(dts.labels(header, use_float=False)[1]) for dts in args
        ]
        other_uniques = np.concatenate(other_uniques + [np.array(my_unique)])
        all_unique = sorted(list(set(other_uniques)))
        labels_to_int = dict(zip(all_unique, range(len(all_unique))))
        return labels_to_int

    def is_float(self, header):
        """Returns True if labels in the given header can all be converted to
        `float`, else False."""

        filtered_labels = self.filtered_annotations[header]
        try:
            filtered_labels = [float(o) for o in filtered_labels]
            return True
        except ValueError:
            return False

    def labels(self, headers, use_float=False, assign=None, format='index'):
        """Returns a dict of slide names mapped to patient id and label(s).

        Args:
            headers (list(str)) Annotation header(s) that specifies label.
                May be a list or string.
            use_float (bool, optional) Either bool, dict, or 'auto'.
                If true, convert data into float; if unable, raise TypeError.
                If false, interpret all data as categorical.
                If a dict is provided, look up each header to determine type.
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
            1) Dictionary mapping slides to outcome labels in numerical format
                (float for linear outcomes, int of outcome label id for
                categorical outcomes).
            2) List of unique labels. For categorical outcomes, this will be a
                list of str; indices correspond with the outcome label id.
        """
        slides = self.slides()
        results = {}
        headers = [headers] if not isinstance(headers, list) else headers
        headers_done = {}
        unique_labels = {}
        for header in headers:
            if assign and (len(headers) > 1 or header in assign):
                assigned_for_header = assign[header]
            elif assign:
                assigned_for_header = assign
            else:
                assigned_for_header = None
            unique_labels_for_this_header = []
            headers_done[header] = {}
            try:
                filtered_labels = self.filtered_annotations[header]
            except KeyError:
                raise errors.AnnotationsError(f"Missing column {header}.")

            # Determine whether values should be converted into float
            if type(use_float) == dict and header not in use_float:
                msg = f"use_float is dict, but header {header} is missing."
                raise ValueError(msg)
            elif type(use_float) == dict:
                header_is_float = use_float[header]
            elif type(use_float) == bool:
                header_is_float = use_float
            elif use_float == 'auto':
                header_is_float = self.is_float(header)
            else:
                raise ValueError(f"Invalid use_float option {use_float}")

            # Ensure labels can be converted to desired type,
            # then assign values
            if header_is_float and not self.is_float(header):
                msg = f"Unable to convert all labels of {header} into 'float' "
                msg += f"({','.join(filtered_labels)})."
                raise TypeError(msg)
            elif not header_is_float:
                log.debug(f'Interpreting column "{header}" as continuous')
                unique_labels_for_this_header = list(set(filtered_labels))
                unique_labels_for_this_header.sort()
                for i, ul in enumerate(unique_labels_for_this_header):
                    n_matching_filtered = sum(f == ul for f in filtered_labels)
                    if assigned_for_header and ul not in assigned_for_header:
                        msg = f"assign was provided, but label {ul} missing"
                        raise KeyError(msg)
                    elif assigned_for_header:
                        val_msg = assigned_for_header[ul]
                        n_s = str(n_matching_filtered)
                        log.debug(" ".join([
                            header,
                            ul,
                            'assigned to',
                            str(val_msg),
                            f'[{n_s} slides]'
                        ]))
                    else:
                        n_s = str(n_matching_filtered)
                        log.debug(" ".join([
                            header,
                            ul,
                            'assigned to',
                            str(i),
                            f'[{n_s} slides]'
                        ]))

            def _process_label(o):
                if header_is_float:
                    return float(o)
                elif assigned_for_header:
                    return assigned_for_header[o]
                elif format == 'name':
                    return o
                else:
                    return unique_labels_for_this_header.index(o)

            # Assemble results dictionary
            patient_labels = {}
            num_warned = 0
            warn_threshold = 3
            for ann in self.filtered_annotations.to_dict(orient="records"):
                slide = ann[TCGA.slide]
                patient = ann[TCGA.patient]
                ann_label = _process_label(ann[header])

                # Ensure patients do not have multiple labels
                if patient not in patient_labels:
                    patient_labels[patient] = ann_label
                elif patient_labels[patient] != ann_label:
                    msg = f"{patient} has multiple labels (header {header}): "
                    msg += f"{patient_labels[patient]}, {ann_label}"
                    log.error(msg)
                    num_warned += 1
                elif ((slide in slides)
                      and (slide in results)
                      and (slide in headers_done[header])):
                    continue
                # Mark this slide as having been already assigned a label
                # with his header
                headers_done[header][slide] = True
                if slide in slides:
                    if slide in results:
                        if not isinstance(results[slide], list):
                            results[slide] = [results[slide]]
                        results[slide] += [ann_label]
                    elif header_is_float:
                        results[slide] = [ann_label]
                    else:
                        results[slide] = ann_label
            if num_warned >= warn_threshold:
                log.warning(f"...{num_warned} warnings, see log for details")
            unique_labels[header] = unique_labels_for_this_header
        if len(headers) == 1:
            unique_labels = unique_labels[headers[0]]
        return results, unique_labels

    def load_indices(self):
        """Reads TFRecord indices. Needed for PyTorch."""

        pool = DPool(16)
        tfrecords = self.tfrecords()
        indices = {}

        def load_index(tfr):
            index_name = join(dirname(tfr), path_to_name(tfr)+'.index')
            tfr_name = path_to_name(tfr)
            if not exists(index_name):
                raise OSError(f"Could not find index path for TFRecord {tfr}")
            if os.stat(index_name).st_size == 0:
                index = None
            else:
                index = np.loadtxt(index_name, dtype=np.int64)
            return tfr_name, index

        for tfr_name, index in tqdm(pool.imap(load_index, tfrecords),
                                    desc="Loading indices...",
                                    total=len(tfrecords),
                                    leave=False):
            indices[tfr_name] = index
        return indices

    def manifest(self, key='path', filter=True):
        """Generates a manifest of all tfrecords.

        Args:
            key (str): Either 'path' (default) or 'name'. Determines key format
                in the manifest dictionary.

        Returns:
            dict: Dict mapping key (path or slide name) to number of tiles.
        """
        if key not in ('path', 'name'):
            raise ValueError("'key' must be in ['path, 'name']")

        all_manifest = {}
        for source in self.sources:
            if self.sources[source]['label'] is None:
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
                    del(all_manifest[tfr])
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

    def patients(self):
        """Returns a list of patient IDs from this dataset."""
        result = {}
        pairs = list(zip(
            self.filtered_annotations[TCGA.slide],
            self.filtered_annotations[TCGA.patient]
        ))
        for slide, patient in pairs:
            if slide in result and result[slide] != patient:
                msg = f"Slide {slide} assigned to multiple patients: "
                msg += f"({patient}, {result[slide]})"
                raise errors.AnnotationsError(msg)
            else:
                result[slide] = patient
        return result

    def remove_filter(self, **kwargs):
        """Removes a specific filter from the active filters.

        Keyword Args:
            filters (list of str): Filter keys. Will remove filters with
                these keys.
            filter_blank (list of str): Will remove these headers stored in
                filter_blank.

        Returns:
            :class:`slideflow.dataset.Dataset` object.
        """

        for kwarg in kwargs:
            if kwarg not in ('filters', 'filter_blank'):
                raise ValueError(f'Unknown filtering argument {kwarg}')
        ret = copy.deepcopy(self)
        if 'filters' in kwargs:
            if not isinstance(kwargs['filters'], list):
                raise TypeError("'filters' must be a list.")
            for f in kwargs['filters']:
                if f not in ret._filters:
                    msg = f"Filter {f} not found in dataset (active filters:"
                    msg += f"{','.join(list(ret._filters.keys()))})"
                    raise errors.DatasetFilterError(msg)
                else:
                    del ret._filters[f]
        if 'filter_blank' in kwargs:
            if not isinstance(kwargs['filter_blank'], list):
                kwargs['filter_blank'] = [kwargs['filter_blank']]
            for f in kwargs['filter_blank']:
                if f not in ret._filter_blank:
                    msg = f"Filter_blank {f} not found in dataset (active "
                    msg += f"filter_blank: {','.join(ret._filter_blank)})"
                    raise errors.DatasetFilterError(msg)
                else:
                    del ret._filter_blank[ret._filter_blank.index(f)]
        return ret

    def resize_tfrecords(self, tile_px):
        """Resizes images in a set of TFRecords to a given pixel size.

        Args:
            tile_px (int): Target pixel size for resizing TFRecord images.
        """

        if sf.backend() == 'torch':
            raise NotImplementedError("Not implemented for PyTorch backend.")

        log.info(f'Resizing TFRecord tiles to ({tile_px}, {tile_px})')
        tfrecords_list = self.tfrecords()
        log.info(f'Resizing {len(tfrecords_list)} tfrecords')
        for tfr in tfrecords_list:
            sf.io.tensorflow.transform_tfrecord(
                tfr,
                tfr+'.transformed',
                resize=tile_px
            )

    def rois(self):
        """Returns a list of all ROIs."""
        rois_list = []
        for source in self.sources:
            rois_list += glob(join(self.sources[source]['roi'], "*.csv"))
        slides = self.slides()
        return [r for r in list(set(rois_list)) if path_to_name(r) in slides]

    def slide_paths(self, source=None, apply_filters=True):
        """Returns a list of paths to either all slides, or slides matching
        dataset filters.

        Args:
            source (str, optional): Dataset source name.
                Defaults to None (using all sources).
            filter (bool, optional): Return only slide paths meeting filter
                criteria. If False, return all slides. Defaults to True.
        """
        if source and source not in self.sources.keys():
            raise errors.DatasetError(f"Dataset {source} not found.")
        # Get unfiltered paths
        if source:
            paths = sf.util.get_slide_paths(self.sources[source]['slides'])
        else:
            paths = []
            for src in self.sources:
                paths += sf.util.get_slide_paths(self.sources[src]['slides'])

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

    def slides(self):
        """Returns a list of slide names in this dataset."""

        # Begin filtering slides with annotations
        slides = []
        slide_patient_dict = {}
        if self.annotations is None:
            log.warn("No annotations loaded; is the annotations file empty?")
        for ann in self.annotations.to_dict(orient="records"):
            skip_annotation = False
            if TCGA.slide not in ann.keys():
                msg = f"{TCGA.slide} not found in annotations file."
                raise errors.AnnotationsError(msg)

            # Skip missing or blank slides
            if ann[TCGA.slide] in sf.util.SLIDE_ANNOTATIONS_TO_IGNORE:
                continue

            # Ensure slides are only assigned to a single patient
            if ann[TCGA.slide] not in slide_patient_dict:
                slide_patient_dict.update({ann[TCGA.slide]: ann[TCGA.patient]})
            elif slide_patient_dict[ann[TCGA.slide]] != ann[TCGA.patient]:
                msg = f"Multiple patients assigned to slide {ann[TCGA.slide]}."
                raise errors.DatasetError(msg)

            # Only return slides with annotation values specified in "filters"
            if self.filters:
                for filter_key in self.filters.keys():
                    if filter_key not in ann.keys():
                        msg = f"Filter header {filter_key} not in annotations."
                        raise IndexError(msg)

                    ann_val = ann[filter_key]
                    filter_vals = self.filters[filter_key]
                    if not isinstance(filter_vals, list):
                        filter_vals = [filter_vals]

                    # Allow filtering based on shortnames if the key
                    # is a TCGA patient ID
                    if filter_key == TCGA.patient:
                        if ((ann_val not in filter_vals)
                            and (_shortname(ann_val) not in filter_vals)
                            and (ann_val not in
                                 [_shortname(fv) for fv in filter_vals])
                            and (_shortname(ann_val) not in
                                 [_shortname(fv) for fv in filter_vals])):
                            skip_annotation = True
                            break
                    else:
                        if ann_val not in filter_vals:
                            skip_annotation = True
                            break

            # Filter out slides that are blank in a given annotation
            # column ("filter_blank")
            if self.filter_blank and self.filter_blank != [None]:
                for fb in self.filter_blank:
                    if fb not in ann.keys():
                        msg = f"Filter blank header {fb} not in annotations."
                        raise errors.DatasetFilterError(msg)
                    if ann[fb] is None or ann[fb] == '':
                        skip_annotation = True
                        break
            if skip_annotation:
                continue
            slides += [ann[TCGA.slide]]
        return slides

    def split_tfrecords_by_roi(self, destination):
        """Split dataset tfrecords into separate tfrecords according to ROI.

        Will generate two sets of tfrecords, with identical names: one with
        tiles inside the ROIs, one with tiles outside the ROIs. Will skip any
        tfrecords that are missing ROIs. Requires slides to be available.
        """
        tfrecords = self.tfrecords()
        slides = {path_to_name(s): s for s in self.slide_paths()}
        rois = self.rois()
        manifest = self.manifest()

        for tfr in tfrecords:
            slidename = path_to_name(tfr)
            if slidename not in slides:
                continue
            slide = WSI(
                slides[slidename],
                self.tile_px,
                self.tile_um,
                rois=rois,
                skip_missing_roi=True
            )
            if slide.load_error:
                continue
            parser = sf.io.get_tfrecord_parser(
                tfr,
                decode_images=False,
                to_numpy=True
            )
            reader = sf.io.TFRecordDataset(tfr)
            if not exists(join(destination, 'inside')):
                os.makedirs(join(destination, 'inside'))
            if not exists(join(destination, 'outside')):
                os.makedirs(join(destination, 'outside'))
            in_path = join(destination, 'inside', f'{slidename}.tfrecords')
            out_path = join(destination, 'outside', f'{slidename}.tfrecords')
            inside_roi_writer = sf.io.TFRecordWriter(in_path)
            outside_roi_writer = sf.io.TFRecordWriter(out_path)
            for record in tqdm(reader, total=manifest[tfr]['total']):
                parsed = parser(record)
                loc_x, loc_y = parsed['loc_x'], parsed['loc_y']
                tile_in_roi = any([
                    annPoly.contains(sg.Point(loc_x, loc_y))
                    for annPoly in slide.annPolys
                ])
                record_bytes = sf.io.read_and_return_record(record, parser)
                if tile_in_roi:
                    inside_roi_writer.write(record_bytes)
                else:
                    outside_roi_writer.write(record_bytes)
            inside_roi_writer.close()
            outside_roi_writer.close()

    def tensorflow(self, labels=None, batch_size=None, **kwargs):
        """Returns a Tensorflow Dataset object that interleaves tfrecords
        from this dataset.

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
            onehot (bool, optional): Onehot encode labels. Defaults to False.
            incl_slidenames (bool, optional): Include slidenames as third
                returned variable. Defaults to False.
            infinite (bool, optional): Infinitely repeat data.
                Defaults to True.
            rank (int, optional): Worker ID to identify which worker this
                represents. Used to interleave results among workers without
                duplications. Defaults to 0 (first worker).
            num_replicas (int, optional): Number of GPUs or unique instances
                which will have their own DataLoader. Used to interleave
                results among workers without duplications. Defaults to 1.
            normalizer (:class:`slideflow.norm.StainNormalizer`, optional):
                Normalizer to use on images.
            seed (int, optional): Use the following seed when randomly
                interleaving. Necessary for synchronized multiprocessing
                distributed reading.
            chunk_size (int, optional): Chunk size for image decoding.
                Defaults to 16.
            preload_factor (int, optional): Number of batches to preload.
                Defaults to 1.
            augment (str, optional): Image augmentations to perform. String
                    containing characters designating augmentations.
                    'x' indicates random x-flipping, 'y' y-flipping,
                    'r' rotating, and 'j' JPEG compression/decompression at
                    random quality levels. Passing either 'xyrj' or True will
                    use all augmentations.
            standardize (bool, optional): Standardize images to (0,1).
                Defaults to True.
            num_workers (int, optional): Number of DataLoader workers.
                Defaults to 2.
            deterministic (bool, optional): When num_parallel_calls is
                specified, if this boolean is specified (True or False), it
                controls the order in which the transformation produces
                elements. If set to False, the transformation is allowed to
                yield elements out of order to trade determinism for
                performance. Defaults to False.
            drop_last (bool, optional): Drop the last non-full batch.
                Defaults to False.
        """

        from slideflow.io.tensorflow import interleave

        tfrecords = self.tfrecords()
        if not tfrecords:
            raise errors.TFRecordsNotFoundError
        self.verify_img_format()

        return interleave(tfrecords=tfrecords,
                          labels=labels,
                          img_size=self.tile_px,
                          batch_size=batch_size,
                          prob_weights=self.prob_weights,
                          clip=self._clip,
                          **kwargs)

    def tfrecord_report(self, dest, normalizer=None):

        """Creates a PDF report of TFRecords, including 10 example tiles
        per TFRecord.

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
        for tfr in tfrecord_list:
            print(f'\r\033[KWorking on {col.green(path_to_name(tfr))}', end='')
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
            reports += [SlideReport(sample_tiles, tfr)]

        # Generate and save PDF
        print('\r\033[K', end='')
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
        log.info(f'TFRecord report saved to {col.green(filename)}')

    def tfrecords(self, source=None):
        """Returns a list of all tfrecords.

        Args:
            source (str, optional): Only return tfrecords from this dataset
                source. Defaults to None (return all tfrecords in dataset).

        Returns:
            List of tfrecords paths
        """

        if source and source not in self.sources.keys():
            log.error(f"Dataset {source} not found.")
            return None
        if source is None:
            sources_to_search = list(self.sources.keys())
        else:
            sources_to_search = [source]

        tfrecords_list = []
        folders_to_search = []
        for source in sources_to_search:
            tfrecords = self.sources[source]['tfrecords']
            label = self.sources[source]['label']
            if label is None:
                continue
            tfrecord_path = join(tfrecords, label)
            if not exists(tfrecord_path):
                msg = f"TFRecords path not found: {col.green(tfrecord_path)}"
                log.debug(msg)
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
                if manifest[f]['total'] >= self.min_tiles
            ]
        else:
            return [f for f in filtered if manifest[f]['total'] > 0]

    def tfrecords_by_subfolder(self, subfolder):
        """Returns a list of all tfrecords in a specific subfolder,
        ignoring filters.
        """
        tfrecords_list = []
        folders_to_search = []
        for source in self.sources:
            if self.sources[source]['label'] is None:
                continue
            base_dir = join(
                self.sources[source]['tfrecords'],
                self.sources[source]['label']
            )
            tfrecord_path = join(base_dir, subfolder)
            if not exists(tfrecord_path):
                msg = f"Unable to find subfolder {col.bold(subfolder)} in "
                msg += f"source {col.bold(source)}, tfrecord directory: "
                msg += f"{col.green(base_dir)}"
                raise errors.DatasetError(msg)
            folders_to_search += [tfrecord_path]
        for folder in folders_to_search:
            tfrecords_list += glob(join(folder, "*.tfrecords"))
        return tfrecords_list

    def tfrecords_folders(self):
        """Returns folders containing tfrecords."""
        folders = []
        for source in self.sources:
            if self.sources[source]['label'] is None:
                continue
            folders += [join(
                self.sources[source]['tfrecords'],
                self.sources[source]['label']
            )]
        return folders

    def tfrecords_from_tiles(self, delete_tiles=False):
        """Create tfrecord files from a collection of raw images,
        as stored in project tiles directory"""

        if not self.tile_px or not self.tile_um:
            msg = "Dataset tile_px & tile_um must be set to create TFRecords."
            raise errors.DatasetError(msg)

        for source in self.sources:
            log.info(f'Working on dataset source {source}')
            config = self.sources[source]
            tfrecord_dir = join(config['tfrecords'], config['label'])
            tiles_dir = join(config['tiles'], config['label'])
            if not exists(tiles_dir):
                log.warn(f'No tiles found for source {col.bold(source)}')
                continue
            sf.io.write_tfrecords_multi(tiles_dir, tfrecord_dir)
            self.update_manifest()
            if delete_tiles:
                shutil.rmtree(tiles_dir)

    def training_validation_split(self, *args, **kwargs):
        msg = "Dataset.training_validation_split() moved to train_val_split()"
        msg += ", please update."
        log.warn(msg)
        self.train_val_split(*args, **kwargs)

    def train_val_split(self, model_type, labels, val_strategy, splits=None,
                        val_fraction=None, val_k_fold=None, k_fold_iter=None,
                        site_labels=None, read_only=False):

        """From a specified subfolder in the project's main TFRecord folder,
        prepare a training set and validation set.

        If a validation split has already been prepared (e.g. K-fold iterations
        were already determined), the previously generated split will be used.
        Otherwise, create a new split and log the result in the TFRecord
        directory so future models may use the same split for consistency.

        Args:
            model_type (str): Either 'categorical' or 'linear'.
            labels (dict):  Dictionary mapping slides to labels. Used for
                balancing outcome labels in training and validation cohorts.
            val_strategy (str): Either 'k-fold', 'k-fold-preserved-site',
                'bootstrap', or 'fixed'.
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
            site_labels (dict, optional): Dict mapping patients to site labels.
                Used for site preserved cross validation.
            read_only (bool): Prevents writing validation splits to file.
                Defaults to False.

        Returns:
            slideflow.Dataset: training dataset,
            slideflow.Dataset: validation dataset
        """
        if (not k_fold_iter and val_strategy == 'k-fold'):
            msg = "If strategy is 'k-fold', must supply k_fold_iter "
            msg += "(int starting at 1)"
            raise errors.DatasetSplitError(msg)
        if (not val_k_fold and val_strategy == 'k-fold'):
            msg = "If strategy is 'k-fold', must supply val_k_fold (K)"
            raise errors.DatasetSplitError(msg)
        if val_strategy == 'k-fold-preserved-site' and not site_labels:
            msg = "k-fold-preserved-site requires site_labels (dict of "
            msg += "patients:sites, or name of annotation column header"
            raise errors.DatasetSplitError(msg)
        if isinstance(site_labels, str):
            site_labels, _ = self.labels(site_labels, format='name')

        # Prepare dataset
        patients = self.patients()
        splits_file = splits
        k_fold = val_k_fold
        training_tfrecords = []
        val_tfrecords = []
        accepted_split = None
        slide_list = list(labels.keys())

        # Assemble dict of patients linking to list of slides & outcome labels
        # dataset.labels() ensures no duplicate labels for a single patient
        tfrecord_dir_list = self.tfrecords()
        if not len(tfrecord_dir_list):
            raise errors.TFRecordsNotFoundError
        tfrecord_dir_list_names = [
            tfr.split('/')[-1][:-10] for tfr in tfrecord_dir_list
        ]
        patients_dict = {}
        num_warned = 0
        for slide in slide_list:
            patient = slide if not patients else patients[slide]
            # Skip slides not found in directory
            if slide not in tfrecord_dir_list_names:
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
                msg = f"Multiple labels found for {patient} ({ol}, {ok})"
                raise errors.DatasetSplitError(msg)
            else:
                patients_dict[patient]['slides'] += [slide]

        # Add site labels to the patients dict if doing
        # preserved-site cross-validation
        if val_strategy == 'k-fold-preserved-site':
            site_slide_list = list(site_labels.keys())
            for slide in site_slide_list:
                patient = slide if not patients else patients[slide]
                # Skip slides not found in directory
                if slide not in tfrecord_dir_list_names:
                    continue
                if 'site' not in patients_dict[patient]:
                    patients_dict[patient]['site'] = site_labels[slide]
                elif patients_dict[patient]['site'] != site_labels[slide]:
                    ol = patients_dict[patient]['slide']
                    ok = site_labels[slide]
                    _tail = f"{patient} ({ol}, {ok})"
                    msg = f"Multiple site labels found for {_tail}"
                    raise errors.DatasetSplitError(msg)
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
            num_val = int(val_fraction * len(patients_list))
            msg = f"Boostrap validation: selecting {col.bold(num_val)} "
            msg += "patients at random for validation testing"
            log.info(msg)
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
                   and len(list(split['tfrecords'].keys())) != k_fold):
                    continue

                # Then, check if patient lists are the same
                sp_pts = list(split['patients'].keys())
                sp_pts.sort()
                if sp_pts == sorted_patients:
                    # Finally, check if outcome variables are the same
                    c1 = [patients_dict[p]['outcome_label'] for p in sp_pts]
                    c2 = [split['patients'][p]['outcome_label']for p in sp_pts]
                    if c1 == c2:
                        msg = f"Using {val_strategy} validation split detected"
                        msg += f" at {col.green(splits_file)} (ID: {split_id})"
                        log.info(msg)
                        accepted_split = split
                        break

            # If no split found, create a new one
            if not accepted_split:
                if splits_file:
                    log.info("No compatible train/val split found.")
                    log.info(f"Logging new split at {col.green(splits_file)}")
                else:
                    log.info("No training/validation splits file provided.")
                    log.info("Unable to save or load validation splits.")
                new_split = {
                    'strategy': val_strategy,
                    'patients': patients_dict,
                    'tfrecords': {}
                }
                if val_strategy == 'fixed':
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
                    if model_type == 'categorical':
                        balance = 'outcome_label'
                    else:
                        balance = None
                    pres_site = (val_strategy == 'k-fold-preserved-site')
                    k_fold_patients = split_patients_list(
                        patients_dict,
                        k_fold,
                        balance=balance,
                        preserved_site=pres_site
                    )
                    # Verify at least one patient is in each k_fold group
                    if (len(k_fold_patients) != k_fold
                       or not min([len(pl) for pl in k_fold_patients])):
                        raise errors.InsufficientDataForSplitError
                    train_patients = []
                    for k in range(1, k_fold+1):
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
                    msg = f"Unknown validation strategy {val_strategy}."
                    raise errors.DatasetSplitError(msg)
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
                    k_id = f'k-fold-{k_fold_iter}'
                    val_slides = accepted_split['tfrecords'][k_id]
                    train_slides = np.concatenate([
                        accepted_split['tfrecords'][f'k-fold-{ki}']
                        for ki in range(1, k_fold+1)
                        if ki != k_fold_iter
                    ]).tolist()
                else:
                    msg = f"Unknown val_strategy {val_strategy} requested."
                    raise errors.DatasetSplitError()

            # Perform final integrity check to ensure no patients
            # are in both training and validation slides
            if patients:
                validation_pt = list(set([patients[s] for s in val_slides]))
                training_pt = list(set([patients[s] for s in train_slides]))
            else:
                validation_pt, training_pt = val_slides, train_slides
            if sum([pt in training_pt for pt in validation_pt]):
                msg = "At least one patient is in both val and training sets."
                raise errors.DatasetSplitError(msg)

            # Return list of tfrecords
            val_tfrecords = [
                tfr for tfr in tfrecord_dir_list
                if path_to_name(tfr) in val_slides
            ]
            training_tfrecords = [
                tfr for tfr in tfrecord_dir_list
                if path_to_name(tfr) in train_slides
            ]
        assert(len(val_tfrecords) == len(val_slides))
        assert(len(training_tfrecords) == len(train_slides))
        training_dts = copy.deepcopy(self)
        training_dts = training_dts.filter(filters={'slide': train_slides})
        val_dts = copy.deepcopy(self)
        val_dts = val_dts.filter(filters={'slide': val_slides})
        assert(sorted(training_dts.tfrecords()) == sorted(training_tfrecords))
        assert(sorted(val_dts.tfrecords()) == sorted(val_tfrecords))
        return training_dts, val_dts

    def torch(self, labels, batch_size=None, rebuild_index=False, **kwargs):
        """Returns a PyTorch DataLoader object that interleaves tfrecords.

        The returned dataloader yields a batch of (image, label) for each tile.

        Args:
            labels (dict or str): If a dict is provided, expect a dict mapping
                slide names to outcome labels. If a str, will intepret as
                categorical annotation header. For linear outcomes, or outcomes
                with manually assigned labels, pass the first result of
                dataset.labels(...). If None, returns slide instead of label.
            batch_size (int): Batch size.
            rebuild_index (bool): Re-build index files even if already present.
                Defaults to True.

        Keyword Args:
            onehot (bool, optional): Onehot encode labels. Defaults to False.
            incl_slidenames (bool, optional): Include slidenames as third
                returned variable. Defaults to False.
            infinite (bool, optional): Infinitely repeat data.
                Defaults to True.
            rank (int, optional): Worker ID to identify which worker this
                represents. Used to interleave results among workers without
                duplications. Defaults to 0 (first worker).
            num_replicas (int, optional): Number of GPUs or unique instances
                which will have their own DataLoader. Used to interleave
                results among workers without duplications. Defaults to 1.
            normalizer (:class:`slideflow.norm.StainNormalizer`, optional):
                Normalizer to use on images. Defaults to None.
            seed (int, optional): Use the following seed when randomly
                interleaving. Necessary for synchronized multiprocessing.
            chunk_size (int, optional): Chunk size for image decoding.
                Defaults to 16.
            preload_factor (int, optional): Number of batches to preload.
                Defaults to 1.
            augment (str, optional): Image augmentations to perform. Str
                containing characters designating augmentations. 'x' indicates
                random x-flipping, 'y' y-flipping, 'r' rotating, 'j' JPEG
                compression/decompression at random quality levels, and 'b'
                random gaussian blur. Passing either 'xyrjb' or True will use
                all augmentations. Defaults to 'xyrjb'.
            standardize (bool, optional): Standardize images to (0,1).
                Defaults to True.
            num_workers (int, optional): Number of DataLoader workers.
                Defaults to 2.
            pin_memory (bool, optional): Pin memory to GPU.
                Defaults to True.
            drop_last (bool, optional): Drop the last non-full batch.
                Defaults to False.
        """

        from slideflow.io.torch import interleave_dataloader

        if isinstance(labels, str):
            labels = self.labels(labels)[0]

        self.build_index(rebuild_index)
        tfrecords = self.tfrecords()
        if not tfrecords:
            raise errors.TFRecordsNotFoundError
        self.verify_img_format()

        if self.prob_weights:
            prob_weights = [self.prob_weights[tfr] for tfr in tfrecords]
        else:
            prob_weights = None
        indices = self.load_indices()
        indices = [indices[path_to_name(tfr)] for tfr in tfrecords]
        return interleave_dataloader(tfrecords=tfrecords,
                                     img_size=self.tile_px,
                                     batch_size=batch_size,
                                     labels=labels,
                                     num_tiles=self.num_tiles,
                                     prob_weights=prob_weights,
                                     clip=self._clip,
                                     indices=indices,
                                     **kwargs)

    def unclip(self):
        """Returns a dataset object with all clips removed.

        Returns:
            :class:`slideflow.dataset.Dataset` object.
        """
        ret = copy.deepcopy(self)
        ret._clip = {}
        return ret

    def update_manifest(self, force_update=False):
        """Updates tfrecord manifest.

        Args:
            forced_update (bool, optional): Force regeneration of the
                manifest from scratch.
        """
        tfrecords_folders = self.tfrecords_folders()
        for tfr_folder in tfrecords_folders:
            sf.io.update_manifest_at_dir(
                directory=tfr_folder,
                force_update=force_update
            )

    def update_annotations_with_slidenames(self, annotations_file):
        """Attempts to automatically associate slide names from a directory
        with patients in a given annotations file, skipping any slide names
        that are already present in the annotations file.
        """

        header, _ = sf.util.read_annotations(annotations_file)
        slide_list = self.slide_paths(apply_filters=False)

        # First, load all patient names from the annotations file
        try:
            patient_index = header.index(TCGA.patient)
        except ValueError:
            msg = f"Patient header {TCGA.patient} not found in annotations."
            raise errors.AnnotationsError(msg)
        patients = []
        pt_to_slide = {}
        with open(annotations_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader, None)
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
            header = next(csv_reader, None)
            with open('temp.csv', 'w') as csv_outfile:
                csv_writer = csv.writer(csv_outfile, delimiter=',')

                # Write to existing "slide" column in the annotations file,
                # otherwise create new column
                try:
                    slide_index = header.index(TCGA.slide)
                except ValueError:
                    header.extend([TCGA.slide])
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
        shutil.move(annotations_file, backup_file)
        shutil.move('temp.csv', annotations_file)

    def verify_annotations_slides(self):
        """Verify that annotations are correctly loaded."""

        # Verify no duplicate slide names are found
        slides_from_anns = self.slides()
        if len(slides_from_anns) != len(list(set(slides_from_anns))):
            msg = "Duplicate slide names detected in the annotation file."
            raise errors.AnnotationsError(msg)

        # Verify all slides in the annotation column are valid
        sf.util.multi_warn(
            self.annotations.to_dict(orient="records"),
            lambda x: x[TCGA.slide] == '',
            lambda x: f'Patient {x[TCGA.patient]} has no slide assigned.'
        )

    def verify_img_format(self):
        """Verify that all tfrecords have the same image format (PNG/JPG)."""

        tfrecords = self.tfrecords()
        if len(tfrecords):
            img_formats = []
            pb = tqdm(
                tfrecords,
                desc="Verifying tfrecord formats...",
                leave=False
            )
            for tfr in pb:
                fmt = sf.io.detect_tfrecord_format(tfr)[-1]
                img_formats += [fmt]
            if len(set(img_formats)) > 1:
                log_msg = "Mismatched TFRecord image formats:\n"
                for tfr, fmt in zip(tfrecords, img_formats):
                    log_msg += f"{tfr}: {fmt}\n"
                log.error(log_msg)
                err_msg = "Mismatched TFRecord image formats detected"
                raise errors.MismatchedImageFormatsError(err_msg)
            return img_formats[0]
        else:
            return None
