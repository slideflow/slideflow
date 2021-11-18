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
import slideflow as sf
import numpy as np
import pandas as pd

from random import shuffle
from glob import glob
from os import listdir
from datetime import datetime
from tqdm import tqdm
from os.path import isdir, join, exists, dirname
from multiprocessing.dummy import Pool as DPool
from slideflow.util import log, TCGA, _shortname, ProgressBar

def _tile_extractor(slide_path, tfrecord_dir, tiles_dir, roi_dir, roi_method, skip_missing_roi, randomize_origin, tma,
                    tile_px, tile_um, stride_div, downsample, buffer, pb_counter, counter_lock, reports,
                    generator_kwargs, qc, qc_kwargs):
    """Internal function to extract tiles. Slide processing needs to be process-isolated when num_workers > 1 ."""

    # Record function arguments in case we need to re-call the function (for corrupt tiles)
    local_args = locals()

    log.handlers[0].flush_line = True
    try:
        log.debug(f'Extracting tiles for slide {sf.util.path_to_name(slide_path)}')

        if tma:
            slide = sf.slide.TMA(slide_path,
                                 tile_px,
                                 tile_um,
                                 stride_div,
                                 enable_downsample=downsample,
                                 report_dir=tfrecord_dir,
                                 buffer=buffer)
        else:
            slide = sf.slide.WSI(slide_path,
                                 tile_px,
                                 tile_um,
                                 stride_div,
                                 enable_downsample=downsample,
                                 roi_dir=roi_dir,
                                 roi_method=roi_method,
                                 randomize_origin=randomize_origin,
                                 skip_missing_roi=skip_missing_roi,
                                 buffer=buffer,
                                 pb_counter=pb_counter,
                                 counter_lock=counter_lock)

        # Apply quality control (blur filtering)
        if qc:
            slide.qc(method=qc, **qc_kwargs)

        if not slide.loaded_correctly():
            return

        try:
            report = slide.extract_tiles(tfrecord_dir=tfrecord_dir, tiles_dir=tiles_dir, **generator_kwargs)

        except sf.slide.TileCorruptionError:
            if downsample:
                log.warning(f'Corrupt tile in {sf.util.path_to_name(slide_path)}; will try disabling downsampling')
                report = _tile_extractor(**local_args)
            else:
                log.error(f'Corrupt tile in {sf.util.path_to_name(slide_path)}; skipping slide')
                return
        del slide
        reports.update({slide_path: report})
    except (KeyboardInterrupt, SystemExit):
        print('Exiting...')
        return

def split_patients_list(patients_dict, n, balance=None, randomize=True, preserved_site=False):
    '''Splits a dictionary of patients into n groups, balancing according to key "balance" if provided.'''

    if preserved_site and not sf.util.CPLEX_AVAILABLE:
        log.error("CPLEX not detected; unable to perform preserved-site validation.")
        raise sf.util.CPLEXError("CPLEX not detected; unable to perform preserved-site validation.")

    patient_list = list(patients_dict.keys())
    shuffle(patient_list)

    def flatten(l):
        '''Flattens a list'''
        return [y for x in l for y in x]

    def split(a, n):
        '''Function to split a list into n components'''
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    if balance:
        # Get patient outcome labels
        patient_outcome_labels = [patients_dict[p][balance] for p in patient_list]

        # Get unique outcomes
        unique_labels = list(set(patient_outcome_labels))
        if preserved_site:
            site_list = [patients_dict[p]['site'] for p in patient_list]
            import slideflow.io.preservedsite.crossfolds as cv
            df = pd.DataFrame(list(zip(patient_list, patient_outcome_labels, site_list)),
                              columns = ['patient', 'outcome_label', 'site'])
            df = cv.generate(df,
                             'outcome_label',
                             unique_labels,
                             crossfolds = n,
                             target_column = 'CV',
                             patient_column = 'patient',
                             site_column = 'site')

            log.info(sf.util.bold("Generating Split with Preserved Site Cross Validation"))
            log.info(sf.util.bold("Category\t" + "\t".join([str(cat) for cat in range(len(set(unique_labels)))])))
            for k in range(n):
                log.info(f"K-fold-{k}\t" + "\t".join([str(len(df[(df.CV == str(k+1)) & (df.outcome_label == o)].index))
                                                       for o in unique_labels]))

            return [df.loc[df.CV == str(ni+1), "patient"].tolist() for ni in range(n)]

        else:
            # Now, split patient_list according to outcomes
            patients_split_by_outcomes = [[p for p in patient_list if patients_dict[p][balance] == uo] for uo in unique_labels]

            # Then, for each sublist, split into n components
            patients_split_by_outcomes_split_by_n = [list(split(sub_l, n)) for sub_l in patients_split_by_outcomes]

            # Print splitting as a table
            log.info(sf.util.bold("Category\t" + "\t".join([str(cat) for cat in range(len(set(unique_labels)))])))
            for k in range(n):
                log.info(f"K-fold-{k}\t" + "\t".join([str(len(clist[k])) for clist in patients_split_by_outcomes_split_by_n]))

            # Join sublists
            return [flatten([item[ni] for item in patients_split_by_outcomes_split_by_n]) for ni in range(n)]
    else:
        return list(split(patient_list, n))

class DatasetError(Exception):
    pass

class Dataset:
    """Object to supervise organization of slides, tfrecords, and tiles
    across one or more sources in a stored configuration file."""

    def __init__(self, config, sources, tile_px, tile_um, annotations=None, filters=None, filter_blank=None, min_tiles=0):

        self.tile_px = tile_px
        self.tile_um = tile_um
        self._filters = filters if filters else {}
        self._filter_blank = filter_blank if filter_blank else []
        self._filter_blank = [self.filter_blank] if not isinstance(self._filter_blank, list) else self._filter_blank
        self._min_tiles = min_tiles
        self._clip = {}
        self.prob_weights = None
        self._config = config
        self.annotations = None
        self.annotations_file = None

        loaded_config = sf.util.load_json(config)
        sources = sources if isinstance(sources, list) else [sources]

        try:
            self.sources = {k:v for (k,v) in loaded_config.items() if k in sources}
            self.sources_names = list(self.sources.keys())
        except KeyError:
            sources_list = ", ".join(sources)
            err_msg = f"Unable to find source '{sf.util.bold(sources_list)}' in config file {sf.util.green(config)}"
            log.error(err_msg)
            raise DatasetError(err_msg)

        if (tile_px is not None) and (tile_um is not None):
            label = f"{tile_px}px_{tile_um}um"
        else:
            label = None

        for source in self.sources:
            self.sources[source]['label'] = label

        if annotations is not None:
            self.load_annotations(annotations)

    def __repr__(self):
        return "Dataset(config={!r}, sources={!r}, tile_px={!r}, tile_um={!r})".format(self._config, self.sources_names, self.tile_px, self.tile_um)

    @property
    def num_tiles(self):
        """Returns the total number of tiles in the tfrecords in this dataset, after filtering/clipping."""
        tfrecords = self.tfrecords()
        manifest = self.manifest()
        if not all([tfr in manifest for tfr in tfrecords]):
            self.update_manifest()
        return sum(manifest[tfr]['total'] if 'clipped' not in manifest[tfr] else manifest[tfr]['clipped'] for tfr in tfrecords)

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
        return self.annotations[self.annotations[TCGA.slide].isin(self.slides())]

    def load_annotations(self, annotations):
        # Load annotations
        if isinstance(annotations, str):
            if not os.path.exists(annotations):
                raise DatasetError(f"Could not find annotations file {annotations}")
            try:
                self.annotations = pd.read_csv(annotations)
                self.annotations.fillna('', inplace=True)
                self.annotations_file = annotations
            except pd.errors.EmptyDataError:
                log.error(f"Unable to load annotations from {annotations}; empty file.")
        elif isinstance(annotations, pd.core.frame.DataFrame):
            self.annotations = annotations
            self.annotations.fillna('', inplace=True)
        else:
            raise DatasetError(f"Unrecognized annotations format {type(annotations)}; expected path (str) or DataFrame")

        # Check annotations
        if len(self.annotations.columns) == 1:
            log.error("Only one column detected - please confirm that the annotations file is in comma separated format.")
        if len(self.annotations.columns) != len(set(self.annotations.columns)):
            raise DatasetError("Annotations file containers at least one duplicate header; all headers must be unique")
        if TCGA.patient not in self.annotations.columns:
            raise DatasetError(f"Patient identifier {TCGA.patient} not found in annotations.")
        if TCGA.slide not in self.annotations.columns:
            log.info(f"Column '{TCGA.slide}' not found in annotations. Attempting to associate patients with slides...")
            self.update_annotations_with_slidenames(annotations)
            self.load_annotations(annotations)

    def balance(self, headers=None, strategy='category', force=False):
        """Returns a dataset with prob_weights reflecting balancing per tile, slide, patient, or category.

        Saves balancing information to the dataset variable prob_weights, which is used by the interleaving dataloaders
        when sampling from tfrecords to create a batch.

        Tile level balancing will create prob_weights reflective of the number of tiles per slide, thus causing the
        batch sampling to mirror random sampling from the entire population of tiles (rather than randomly sampling
        from slides).

        Slide level balancing is the default behavior, where batches are assembled by randomly sampling from each
        slide/tfrecord with equal probability. This balancing behavior would be the same as not balancing.

        Patient level balancing is used to randomly sample from individual patients with equal probability. This is
        distinct from slide level balancing, as some patients may have multiple slides per patient.

        Category level balancing takes a list of annotation header(s) and generates prob_weights such that each category
        is sampled equally. This requires categorical outcomes.

        Args:
            headers (list of str, optional): List of annotation headers if balancing by category. Defaults to None.
            strategy (str, optional): 'tile', 'slide', 'patient', or 'category'. Create prob_weights used to balance
                dataset batches to evenly distribute slides, patients, or categories in a given batch. Tile-level
                balancing generates prob_weights reflective of the total number of tiles in a slide.
                Defaults to 'category.'
            force (bool, optional): If using category-level balancing, interpret all headers as categorical variables,
                even if the header appears to be a float.

        Returns:
            balanced :class:`slideflow.dataset.Dataset` object.
        """

        ret = copy.deepcopy(self)
        manifest = ret.manifest()
        tfrecords = ret.tfrecords()
        slides = [sf.util.path_to_name(tfr) for tfr in tfrecords]
        totals = {tfr: (manifest[tfr]['total'] if 'clipped' not in manifest[tfr] else manifest[tfr]['clipped']) for tfr in tfrecords}

        if strategy == 'none' or strategy is None:
            return self

        if strategy == 'tile':
            ret.prob_weights = {tfr: totals[tfr] / sum(totals.values()) for tfr in tfrecords}

        if strategy == 'slide':
            ret.prob_weights = {tfr: 1/len(tfrecords) for tfr in tfrecords}

        if strategy == 'patient':
            patients = ret.patients() # Maps tfrecords to patients
            rev_patients = {}         # Will map patients to list of tfrecords
            for slide in patients:
                if slide not in slides: continue
                if patients[slide] not in rev_patients:
                    rev_patients[patients[slide]] = [slide]
                else:
                    rev_patients[patients[slide]] += [slide]
            ret.prob_weights = {tfr: 1/(len(rev_patients) * len(rev_patients[patients[sf.util.path_to_name(tfr)]])) for tfr in tfrecords}

        if strategy == 'category':
            # Ensure that header is not type 'float'
            if not isinstance(headers, list): headers = [headers]
            if any(ret.is_float(h) for h in headers) and not force:
                raise DatasetError(f"Headers {','.join(headers)} appear to be `float`. Categorical outcomes required " + \
                                    "for balancing. To force balancing with these outcomes, pass `force=True` to Dataset.balance()")

            labels, _ = ret.labels(headers, use_float=False, verbose=False)
            categories, categories_prob, tfrecord_categories = {}, {}, {}
            for tfrecord in tfrecords:
                slide = sf.util.path_to_name(tfrecord)
                balance_category = labels[slide]
                if not isinstance(balance_category, list): balance_category = [balance_category]
                balance_category = '-'.join(map(str, balance_category))
                tfrecord_categories[tfrecord] = balance_category
                tiles = totals[tfrecord]
                if balance_category not in categories:
                    categories.update({balance_category: {
                        'num_slides': 1,
                        'num_tiles': tiles
                    }})
                else:
                    categories[balance_category]['num_slides'] += 1
                    categories[balance_category]['num_tiles'] += tiles

            for category in categories:
                lowest_category_slide_count = min([categories[i]['num_slides'] for i in categories])
                categories_prob[category] = lowest_category_slide_count / categories[category]['num_slides']

            total_prob = sum([categories_prob[tfrecord_categories[tfr]] for tfr in tfrecords])
            ret.prob_weights = {tfr: categories_prob[tfrecord_categories[tfr]]/total_prob for tfr in tfrecords}

        return ret

    def build_index(self, force=True):
        """Builds index files for TFRecords, required for PyTorch dataloader."""

        from slideflow.util import tfrecord2idx

        def create_index(filename):
            nonlocal force
            index_name = join(dirname(filename), sf.util.path_to_name(filename)+'.index')
            if not exists(index_name) or force:
                tfrecord2idx.create_index(filename, index_name)

        pool = DPool(16)
        for res in tqdm(pool.imap(create_index, self.tfrecords()), desc='Creating index files...', ncols=80, total=len(self.tfrecords()), leave=False):
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
        '''Returns a dataset clipped to either a fixed maximum number of tiles per tfrecord, or to the minimum number
        of tiles per patient or category.

        Args:
            max_tiles (int, optional): Clip the maximum number of tiles per tfrecord to this number.
            strategy (str, optional): 'slide', 'patient', or 'category'. Clip the maximum number of tiles to the
                minimum tiles seen across slides, patients, or categories. If 'category', headers must be provided.
                Defaults to None.
            headers (list of str, optional): List of annotation headers to use if clipping by minimum category count
                (strategy='category'). Defaults to None.

        Returns:
            clipped :class:`slideflow.dataset.Dataset` object.
        '''

        if strategy == 'category' and not headers:
            raise DatasetError("headers must be provided if clip strategy is 'category'.")
        if strategy is None and headers is not None:
            strategy = 'category'
        if strategy is None and headers is None and not max_tiles:
            return self

        ret = copy.deepcopy(self)
        manifest = ret.manifest()
        tfrecords = ret.tfrecords()
        slides = [sf.util.path_to_name(tfr) for tfr in tfrecords]
        totals = {tfr: manifest[tfr]['total'] for tfr in tfrecords}

        if strategy == 'slide':
            clip = min(min(totals.values()), max_tiles) if max_tiles else min(totals.values())
            ret._clip = {tfr: (clip if totals[tfr] > clip else totals[tfr]) for tfr in manifest}

        elif strategy == 'patient':
            patients = ret.patients() # Maps slide name to patient
            rev_patients = {}         # Will map patients to list of slide names
            slide_totals = {sf.util.path_to_name(tfr): t for tfr,t in totals.items()}
            for slide in patients:
                if slide not in slides: continue
                if patients[slide] not in rev_patients:
                    rev_patients[patients[slide]] = [slide]
                else:
                    rev_patients[patients[slide]] += [slide]
            tiles_per_patient = {pt: sum([slide_totals[slide] for slide in slide_list]) for pt, slide_list in rev_patients.items()}
            clip = min(min(tiles_per_patient.values()), max_tiles) if max_tiles else min(tiles_per_patient.values())
            ret._clip = {tfr: (clip if slide_totals[sf.util.path_to_name(tfr)] > clip else totals[tfr]) for tfr in manifest}

        elif strategy == 'category':
            labels, _ = ret.labels(headers, use_float=False, verbose=False)
            categories, categories_tile_fraction, tfrecord_categories = {}, {}, {}
            for tfrecord in tfrecords:
                slide = sf.util.path_to_name(tfrecord)
                balance_category = labels[slide]
                if not isinstance(balance_category, list): balance_category = [balance_category]
                balance_category = '-'.join(map(str, balance_category))
                tfrecord_categories[tfrecord] = balance_category
                tiles = totals[tfrecord]
                if balance_category not in categories:
                    categories[balance_category] = tiles
                else:
                    categories[balance_category] += tiles

            for category in categories:
                lowest_category_tile_count = min([categories[i] for i in categories])
                categories_tile_fraction[category] = lowest_category_tile_count / categories[category]

            ret._clip = {tfr: int(totals[tfr] * categories_tile_fraction[tfrecord_categories[tfr]]) for tfr in manifest}

        elif max_tiles:
            ret._clip = {tfr: (max_tiles if totals[tfr] > max_tiles else totals[tfr]) for tfr in manifest}

        return ret

    def extract_tiles(self, save_tiles=False, save_tfrecords=True, source=None, stride_div=1, enable_downsample=True,
                      roi_method='inside', skip_missing_roi=False, skip_extracted=True, tma=False, randomize_origin=False,
                      buffer=None, num_workers=1, q_size=4, qc=None, report=True, **kwargs):

        """Extract tiles from a group of slides, saving extracted tiles to either loose image or in
        TFRecord binary format.

        Args:
            save_tiles (bool, optional): Save images of extracted tiles to project tile directory. Defaults to False.
            save_tfrecords (bool, optional): Save compressed image data from extracted tiles into TFRecords
                in the corresponding TFRecord directory. Defaults to True.
            source (str, optional): Name of dataset source from which to select slides for extraction. Defaults to None.
                If not provided, will default to all sources in project.
            stride_div (int, optional): Stride divisor to use when extracting tiles. Defaults to 1.
                A stride of 1 will extract non-overlapping tiles.
                A stride_div of 2 will extract overlapping tiles, with a stride equal to 50% of the tile width.
            enable_downsample (bool, optional): Enable downsampling when reading slide images. Defaults to True.
                This may result in corrupted image tiles if downsampled slide layers are corrupted or incomplete.
                Recommend manual confirmation of tile integrity.
            roi_method (str, optional): Either 'inside', 'outside', or 'ignore'. Defaults to 'inside'.
                Indicates whether tiles are extracted inside or outside ROIs, or if ROIs are ignored entirely.
            skip_missing_roi (bool, optional): Skip slides that are missing ROIs. Defaults to False.
            skip_extracted (bool, optional): Skip slides that have already been extracted. Defaults to True.
            tma (bool, optional): Reads slides as Tumor Micro-Arrays (TMAs), detecting and extracting tumor cores.
                Defaults to False. Experimental function with limited testing.
            randomize_origin (bool, optional): Randomize pixel starting position during extraction. Defaults to False.
            buffer (str, optional): Slides will be copied to this directory before extraction. Defaults to None.
                Using an SSD or ramdisk buffer vastly improves tile extraction speed.
            num_workers (int, optional): Extract tiles from this many slides simultaneously. Defaults to 4.
            q_size (int, optional): Size of queue when using a buffer. Defaults to 4.
            qc (str, optional): 'otsu', 'blur', 'both', or None. Perform blur detection quality control - discarding
                tiles with detected out-of-focus regions or artifact - and/or otsu's method. Increases tile extraction
                time. Defaults to None.
            report (bool, optional): Save a PDF report of tile extraction. Defaults to True.

        Keyword Args:
            normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image. Defaults to None.
                If None but using a normalizer, will use an internal tile for normalization.
                Internal default tile can be found at slideflow.slide.norm_tile.jpg
            whitespace_fraction (float, optional): Range 0-1. Defaults to 1.
                Discard tiles with this fraction of whitespace. If 1, will not perform whitespace filtering.
            whitespace_threshold (int, optional): Range 0-255. Defaults to 230.
                Threshold above which a pixel (RGB average) is considered whitespace.
            grayspace_fraction (float, optional): Range 0-1. Defaults to 0.6.
                Discard tiles with this fraction of grayspace. If 1, will not perform grayspace filtering.
            grayspace_threshold (float, optional): Range 0-1. Defaults to 0.05.
                Pixels in HSV format with saturation below this threshold are considered grayspace.
            img_format (str, optional): 'png' or 'jpg'. Defaults to 'jpg'. Image format to use in tfrecords.
                PNG (lossless) format recommended for fidelity, JPG (lossy) for efficiency.
            full_core (bool, optional): Only used if extracting from TMA. If True, will save entire TMA core as image.
                Otherwise, will extract sub-images from each core using the given tile micron size. Defaults to False.
            shuffle (bool, optional): Shuffle tiles prior to storage in tfrecords. Defaults to True.
            num_threads (int, optional): Number of workers threads for each tile extractor. Defaults to 4.
            qc_blur_radius (int, optional): Quality control blur radius for out-of-focus area detection. Only used if
                qc=True. Defaults to 3.
            qc_blur_threshold (float, optional): Quality control blur threshold for detecting out-of-focus areas.
                Only used if qc=True. Defaults to 0.1
            qc_filter_threshold (float, optional): Float between 0-1. Tiles with more than this proportion of blur
                will be discarded. Only used if qc=True. Defaults to 0.6.
            qc_mpp (float, optional): Microns-per-pixel indicating image magnification level at which quality control
                is performed. Defaults to mpp=4 (effective magnification 2.5 X)
            dry_run (bool, optional): Determine tiles that would be extracted, but do not export any images.
                Defaults to None.
        """

        if not save_tiles and not save_tfrecords:
            log.error('Either save_tiles or save_tfrecords must be true to extract tiles.')
            return
        if q_size < num_workers:
            log.warn(f"q_size ({q_size}) less than num_workers {num_workers}; some workers will not be used")

        if source:  sources = [source] if not isinstance(source, list) else source
        else:       sources = self.sources

        self.verify_annotations_slides()

        # Set up kwargs for tile extraction generator and quality control
        qc_kwargs = {k[3:]:v for k,v in kwargs.items() if k[:3] == 'qc_'}
        kwargs = {k:v for k,v in kwargs.items() if k[:3] != 'qc_'}
        sf.slide.log_extraction_params(**kwargs)

        for source in sources:
            log.info(f'Working on dataset source {sf.util.bold(source)}...')

            roi_dir = self.sources[source]['roi']
            source_config = self.sources[source]
            if 'dry_run' not in kwargs or not kwargs['dry_run']:
                tfrecord_dir = join(source_config['tfrecords'], source_config['label']) if save_tfrecords else None
                tiles_dir = join(source_config['tiles'], source_config['label']) if save_tiles else None
                if save_tfrecords and not exists(tfrecord_dir):
                    os.makedirs(tfrecord_dir)
                if save_tiles and not os.path.exists(tiles_dir):
                    os.makedirs(tiles_dir)
            else:
                save_tfrecords, save_tiles = False, False
                tfrecord_dir, tiles_dir = None, None

            # Prepare list of slides for extraction
            slide_list = self.slide_paths(source=source)

            # Check for interrupted or already-extracted tfrecords
            if skip_extracted and save_tfrecords:
                already_done = [sf.util.path_to_name(tfr) for tfr in self.tfrecords(source=source)]
                interrupted = [sf.util.path_to_name(marker) for marker in glob(join((tfrecord_dir
                                                           if tfrecord_dir else tiles_dir), '*.unfinished'))]
                if len(interrupted):
                    log.info(f'Interrupted tile extraction in {len(interrupted)} tfrecords, will re-extract slides')
                    for interrupted_slide in interrupted:
                        log.info(interrupted_slide)
                        if interrupted_slide in already_done:
                            del already_done[already_done.index(interrupted_slide)]

                slide_list = [slide for slide in slide_list if sf.util.path_to_name(slide) not in already_done]
                if len(already_done):
                    log.info(f'Skipping {len(already_done)} slides; TFRecords already generated.')
            log.info(f'Extracting tiles from {len(slide_list)} slides ({self.tile_um} um, {self.tile_px} px)')

            # Verify slides and estimate total number of tiles
            log.info('Verifying slides...')
            total_tiles = 0
            for slide_path in tqdm(slide_list, leave=False, desc="Verifying slides..."):
                if tma:
                    slide = sf.slide.TMA(slide_path, self.tile_px, self.tile_um, stride_div, silent=True)
                else:
                    slide = sf.slide.WSI(slide_path,
                                         self.tile_px,
                                         self.tile_um,
                                         stride_div,
                                         roi_dir=roi_dir,
                                         roi_method=roi_method,
                                         skip_missing_roi=skip_missing_roi,
                                         silent=True)
                log.debug(f"Estimated tiles for slide {slide.name}: {slide.estimated_num_tiles}")
                total_tiles += slide.estimated_num_tiles
                del slide
            log.info(f'Total estimated tiles to extract: {total_tiles}')

            # Use multithreading if specified, extracting tiles from all slides in the filtered list
            if len(slide_list):
                q = queue.Queue()
                task_finished = False
                manager = multiprocessing.Manager()
                ctx = multiprocessing.get_context('fork')
                reports = manager.dict()
                counter = manager.Value('i', 0)
                counter_lock = manager.Lock()

                if total_tiles:
                    pb = ProgressBar(total_tiles,
                                     counter_text='tiles',
                                     leadtext='Extracting tiles... ',
                                     show_counter=True,
                                     show_eta=True,
                                     mp_counter=counter,
                                     mp_lock=counter_lock)
                    pb.auto_refresh(0.1)
                else:
                    pb = None

                # If only one worker, use a single shared multiprocessing pool
                if num_workers == 1:
                    num_threads = 8 if 'num_threads' not in kwargs else kwargs['num_threads']
                    kwargs['pool'] = multiprocessing.Pool(num_threads)

                extraction_kwargs = {
                    'tfrecord_dir': tfrecord_dir,
                    'tiles_dir': tiles_dir,
                    'roi_dir': roi_dir,
                    'roi_method': roi_method,
                    'skip_missing_roi': skip_missing_roi,
                    'randomize_origin': randomize_origin,
                    'tma': tma,
                    'tile_px': self.tile_px,
                    'tile_um': self.tile_um,
                    'stride_div': stride_div,
                    'downsample': enable_downsample,
                    'buffer': buffer,
                    'pb_counter': counter,
                    'counter_lock': counter_lock,
                    'generator_kwargs': kwargs,
                    'qc': qc,
                    'qc_kwargs': qc_kwargs,
                    'reports': reports
                }

                # Worker to grab slide path from queue and start tile extraction
                def worker():
                    while True:
                        try:
                            path = q.get()
                            if num_workers > 1:
                                process = ctx.Process(target=_tile_extractor, args=(path,), kwargs=extraction_kwargs)
                                process.start()
                                process.join()
                            else:
                                _tile_extractor(path, **extraction_kwargs)
                            if buffer and buffer != 'vmtouch':
                                os.remove(path)
                            q.task_done()
                        except queue.Empty:
                            if task_finished:
                                return

                # Start the worker threads
                threads = [threading.Thread(target=worker, daemon=True) for t in range(num_workers)]
                for thread in threads:
                    thread.start()

                # Put each slide path into queue
                for slide_path in slide_list:
                    warned = False
                    if buffer and buffer != 'vmtouch':
                        while True:
                            if q.qsize() < q_size:
                                try:
                                    buffered_path = join(buffer, os.path.basename(slide_path))
                                    shutil.copy(slide_path, buffered_path)
                                    q.put(buffered_path)
                                    break
                                except OSError as e:
                                    if not warned:
                                        formatted_slide = sf.util._shortname(sf.util.path_to_name(slide_path))
                                        log.debug(f'OSError encountered for slide {formatted_slide}: buffer likely full')
                                        log.debug(f'Queue size: {q.qsize()}')
                                        warned = True
                                    time.sleep(1)
                            else:
                                time.sleep(1)
                    else:
                        q.put(slide_path)
                q.join()
                task_finished = True
                if pb: pb.end()
                if report:
                    log.info('Generating PDF (this may take some time)...', )
                    reports_vals = reports.values()
                    num_slides=len(slide_list)
                    report_meta = types.SimpleNamespace(
                        tile_px=self.tile_px,
                        tile_um=self.tile_um,
                        qc=qc,
                        total_slides=num_slides,
                        slides_skipped=len([r for r in reports_vals if r is None]),
                        roi_method=roi_method,
                        stride=stride_div,
                        gs_frac=(None if 'grayspace_fraction' not in kwargs else kwargs['grayspace_fraction']),
                        gs_thresh=(None if 'grayspace_threshold' not in kwargs else kwargs['grayspace_threshold']),
                        ws_frac=(None if 'whitespace_fraction' not in kwargs else kwargs['whitespace_fraction']),
                        ws_thresh=(None if 'whitespace_threshold' not in kwargs else kwargs['whitespace_threshold']),
                        normalizer=(None if 'normalizer' not in kwargs else kwargs['normalizer']),
                        img_format=(None if 'img_format' not in kwargs else kwargs['img_format'])
                    )
                    pdf_report = sf.slide.ExtractionReport(reports_vals, meta=report_meta)
                    timestring = datetime.now().strftime('%Y%m%d-%H%M%S')
                    pdf_dir = tfrecord_dir if tfrecord_dir else ''
                    pdf_report.save(join(pdf_dir, f'tile_extraction_report-{timestring}.pdf'))
                    with open(join(pdf_dir, f'warn_report-{timestring}.txt'), 'w') as warn_f:
                        warn_f.write(pdf_report.warn_txt)

            # Update manifest & rebuild indices
            self.update_manifest()
            self.build_index(True)

    def extract_tiles_from_tfrecords(self, dest):
        """Extracts tiles from a set of TFRecords.

        Args:
            dest (str): Path to directory in which to save tile images. Defaults to None. If None, uses dataset default.
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
            filters (dict): Filters dict to use when selecting tfrecords. Defaults to None.
                See :meth:`get_dataset` documentation for more information on filtering.
            filter_blank (list): Slides blank in these columns will be excluded. Defaults to None.
            min_tiles (int): Filter out tfrecords that have less than this minimum number of tiles.

        Returns:
            :class:`slideflow.dataset.Dataset` object.
        """
        if len(args) == 1 and 'filters' not in kwargs:
            kwargs['filters'] = args[0]
        elif len(args):
            raise DatasetError("filter() accepts either only one argument (filters), or any combination of keyword" + \
                               "arguments (filters, filter_blank, min_tiles)")
        for kwarg in kwargs:
            if kwarg not in ('filters', 'filter_blank', 'min_tiles'):
                raise sf.util.UserError(f'Unknown filtering argument {kwarg}')
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

    def is_float(self, header):
        """Returns True if labels in the given header can all be converted to `float`, else False."""

        filtered_labels = self.filtered_annotations[header]
        try:
            filtered_labels = [float(o) for o in filtered_labels]
            return True
        except ValueError:
            return False

    def labels(self, headers, use_float=False, assigned_labels=None, verbose=True, format='index'):
        """Returns a dictionary of slide names mapping to patient id and [an] label(s).

        Args:
            headers (list(str)) Annotation header(s) that specifies label variable. May be a list or string.
            use_float (bool, optional) Either bool, dict, or 'auto'.
                If true, will try to convert all data into float. If unable, will raise TypeError.
                If false, will interpret all data as categorical.
                If a dict is provided, will look up each header to determine whether float is used.
                If 'auto', will try to convert all data into float. For each header in which this fails, will
                interpret as categorical instead.
            assigned_labels (dict, optional):  Dictionary mapping label ids to label names. If not provided, will map
                ids to names by sorting alphabetically.
            verbose (bool, optional): Verbose output.
            format (str, optional): Either 'index' or 'name.' Indicates which format should be used for categorical
                outcomes when returning the label dictionary. If 'name', uses the string label name. If 'index',
                returns an int (index corresponding with the returned list of unique outcome names as str).
                Defaults to 'index'.

        Returns:
            1) Dictionary mapping slides to outcome labels in numerical format (float for linear outcomes,
                int of outcome label id for categorical outcomes).
            2) List of unique labels. For categorical outcomes, this will be a list of str, whose indices correspond
                with the outcome label id.
        """

        slides = self.slides()
        results = {}
        headers = [headers] if not isinstance(headers, list) else headers
        assigned_headers = {}
        unique_labels = {}
        for header in headers:
            if assigned_labels and (len(headers) > 1 or header in assigned_labels):
                assigned_labels_for_this_header = assigned_labels[header]
            elif assigned_labels:
                assigned_labels_for_this_header = assigned_labels
            else:
                assigned_labels_for_this_header = None

            unique_labels_for_this_header = []
            assigned_headers[header] = {}
            try:
                filtered_labels = self.filtered_annotations[header]
            except KeyError:
                log.error(f"Unable to find column {header} in annotation file.")
                raise DatasetError(f"Unable to find column {header} in annotation file.")

            # Determine whether values should be converted into float
            if type(use_float) == dict and header not in use_float:
                raise DatasetError(f"Dict was provided to use_float, but header {header} is missing.")
            elif type(use_float) == dict:
                use_float_for_this_header = use_float[header]
            elif type(use_float) == bool:
                use_float_for_this_header = use_float
            elif use_float == 'auto':
                use_float_for_this_header = self.is_float(header)
            else:
                raise DatasetError(f"Invalid use_float option {use_float}")

            # Ensure labels can be converted to desired type, then assign values
            if use_float_for_this_header and not self.is_float(header):
                raise TypeError(f"Unable to convert all labels of {header} into type 'float' ({','.join(filtered_labels)}).")
            elif not use_float_for_this_header:
                if verbose: log.debug(f'Assigning label descriptors in column "{header}" to numerical values')
                unique_labels_for_this_header = list(set(filtered_labels))
                unique_labels_for_this_header.sort()
                for i, ul in enumerate(unique_labels_for_this_header):
                    num_matching_slides_filtered = sum(l == ul for l in filtered_labels)
                    if assigned_labels_for_this_header and ul not in assigned_labels_for_this_header:
                        raise KeyError(f"assigned_labels was provided, but label {ul} not found in this dict")
                    elif assigned_labels_for_this_header:
                        if verbose:
                            val_msg = assigned_labels_for_this_header[ul]
                            n_s = sf.util.bold(str(num_matching_slides_filtered))
                            log.info(f"{header} '{sf.util.blue(ul)}' assigned to value '{val_msg}' [{n_s} slides]")
                    else:
                        if verbose:
                            n_s = sf.util.bold(str(num_matching_slides_filtered))
                            log.info(f"{header} '{sf.util.blue(ul)}' assigned to value '{i}' [{n_s} slides]")

            # Create function to process/convert label
            def _process_label(o):
                if use_float_for_this_header:
                    return float(o)
                elif assigned_labels_for_this_header:
                    return assigned_labels_for_this_header[o]
                elif format == 'name':
                    return o
                else:
                    return unique_labels_for_this_header.index(o)

            # Assemble results dictionary
            patient_labels = {}
            num_warned = 0
            warn_threshold = 3
            for annotation in self.filtered_annotations.to_dict(orient="records"):
                slide = annotation[TCGA.slide]
                patient = annotation[TCGA.patient]
                annotation_label = _process_label(annotation[header])

                # Mark this slide as having been already assigned a label with his header
                assigned_headers[header][slide] = True

                # Ensure patients do not have multiple labels
                if patient not in patient_labels:
                    patient_labels[patient] = annotation_label
                elif patient_labels[patient] != annotation_label:
                    log.error(f"Multiple labels in header {header} found for patient {patient}:")
                    log.error(f"{patient_labels[patient]}")
                    log.error(f"{annotation_label}")
                    num_warned += 1
                elif (slide in slides) and (slide in results) and (slide in assigned_headers[header]):
                    continue

                if slide in slides:
                    if slide in results:
                        so = results[slide]
                        results[slide] = [so] if not isinstance(so, list) else so
                        results[slide] += [annotation_label]
                    else:
                        results[slide] = (annotation_label if not use_float_for_this_header else [annotation_label])
            if num_warned >= warn_threshold:
                log.warning(f"...{num_warned} total warnings, see project log for details")
            unique_labels[header] = unique_labels_for_this_header
        if len(headers) == 1:
            unique_labels = unique_labels[headers[0]]
        return results, unique_labels

    def load_indices(self):
        indices = {}

        def load_index(tfr):
            index_name = join(dirname(tfr), sf.util.path_to_name(tfr)+'.index')
            tfr_name = sf.util.path_to_name(tfr)
            if not exists(index_name):
                raise OSError(f"Could not find index path for TFRecord {tfr}")
            if os.stat(index_name).st_size == 0:
                index = None
            else:
                index = np.loadtxt(index_name, dtype=np.int64)
            return tfr_name, index

        pool = DPool(16)
        tfrecords = self.tfrecords()
        for tfr_name, index in tqdm(pool.imap(load_index, tfrecords), desc="Loading indices...", total=len(tfrecords), leave=False):
            indices[tfr_name] = index

        return indices

    def manifest(self, key='path', filter=True):
        """Generates a manifest of all tfrecords.

        Args:
            key (str): Either 'path' (default) or 'name'. Determines key format in the manifest dictionary.

        Returns:
            dict: Dictionary mapping key (path or slide name) to number of total tiles.
        """
        if key not in ('path', 'name'):
            raise DatasetError("'key' must be in ['path, 'name']")

        combined_manifest = {}
        for source in self.sources:
            if self.sources[source]['label'] is None: continue
            tfrecord_dir = join(self.sources[source]['tfrecords'], self.sources[source]['label'])
            manifest_path = join(tfrecord_dir, "manifest.json")
            if not exists(manifest_path):
                log.debug(f"No manifest file detected in {tfrecord_dir}; will create now")

                # Import delayed until here in order to avoid importing backend until necessary,
                # as tensorflow/torch claims a GPU once imported
                import slideflow.io
                slideflow.io.update_manifest_at_dir(tfrecord_dir)

            relative_manifest = {} if not exists(manifest_path) else sf.util.load_json(manifest_path)
            global_manifest = {}
            for record in relative_manifest:
                k = join(tfrecord_dir, record)
                global_manifest.update({k: relative_manifest[record]})
            combined_manifest.update(global_manifest)

        # Now filter out any tfrecords that would be excluded by filters
        if filter:
            filtered_tfrecords = self.tfrecords()
            manifest_tfrecords = list(combined_manifest.keys())
            for tfr in manifest_tfrecords:
                if tfr not in filtered_tfrecords:
                    del(combined_manifest[tfr])

        # Log clipped tile totals if applicable
        for tfr in combined_manifest:
            if tfr in self._clip:
                combined_manifest[tfr]['clipped'] = min(self._clip[tfr], combined_manifest[tfr]['total'])
            else:
                combined_manifest[tfr]['clipped'] = combined_manifest[tfr]['total']

        if key == 'path':
            return combined_manifest
        else:
            return {sf.util.path_to_name(t):v for t,v in combined_manifest.items()}

    def patients(self):
        """Returns a list of patient IDs from this dataset."""

        result = {}
        pairs = list(zip(self.filtered_annotations[TCGA.slide], self.filtered_annotations[TCGA.patient]))
        for slide, patient in pairs:
            if slide in result and result[slide] != patient:
                raise DatasetError(f"Slide {slide} assigned to multiple patients in annotations file ({patient}, {result[slide]})")
            else:
                result[slide] = patient
        return result

    def remove_filter(self, **kwargs):
        """Removes a specific filter from the active filters.

        Keyword Args:
            filters (list of str): Filter keys. Will remove filters with these keys.
            filter_blank (list of str): Will remove these headers stored in filter_blank.

        Returns:
            :class:`slideflow.dataset.Dataset` object.
        """

        for kwarg in kwargs:
            if kwarg not in ('filters', 'filter_blank'):
                raise sf.util.UserError(f'Unknown filtering argument {kwarg}')
        ret = copy.deepcopy(self)
        if 'filters' in kwargs:
            if not isinstance(kwargs['filters'], list):
                raise TypeError("'filters' must be a list.")
            for f in kwargs['filters']:
                if f not in ret._filters:
                    raise DatasetError(f"Filter {f} not found in dataset (active filters: {','.join(list(ret._filters.keys()))})")
                else:
                    del ret._filters[f]
        if 'filter_blank' in kwargs:
            if not isinstance(kwargs['filter_blank'], list):
                kwargs['filter_blank'] = [kwargs['filter_blank']]
            for f in kwargs['filter_blank']:
                if f not in ret._filter_blank:
                    raise DatasetError(f"Filter_blank {f} not found in dataset (active filter_blank: {','.join(ret._filter_blank)})")
                else:
                    del ret._filter_blank[ret._filter_blank.index(f)]
        return ret

    def resize_tfrecords(self, tile_px):
        """Resizes images in a set of TFRecords to a given pixel size.

        Args:
            tile_px (int): Target pixel size for resizing TFRecord images.
        """

        if sf.backend() == 'torch':
            raise NotImplementedError("This function has not yet been implemented for the PyTorch backend.")

        log.info(f'Resizing TFRecord tiles to ({tile_px}, {tile_px})')
        tfrecords_list = self.tfrecords()
        log.info(f'Resizing {len(tfrecords_list)} tfrecords')
        for tfr in tfrecords_list:
            sf.io.tensorflow.transform_tfrecord(tfr, tfr+'.transformed', resize=tile_px)

    def rois(self):
        """Returns a list of all ROIs."""
        rois_list = []
        for source in self.sources:
            rois_list += glob(join(self.sources[source]['roi'], "*.csv"))
        slides = self.slides()
        return [l for l in list(set(rois_list)) if sf.util.path_to_name(l) in slides]

    def slide_paths(self, source=None, apply_filters=True):
        """Returns a list of paths to either all slides, or slides matching dataset filters.

        Args:
            source (str, optional): Dataset source name. Defaults to None (using all sources).
            filter (bool, optional): Return only slide paths meeting filter criteria. If False, return all slides.
                Defaults to True.
        """

        if source and source not in self.sources.keys():
            log.error(f"Dataset {source} not found.")
            return None

        # Get unfiltered paths
        if source:
            paths = sf.util.get_slide_paths(self.sources[source]['slides'])
        else:
            paths = []
            for source in self.sources:
                paths += sf.util.get_slide_paths(self.sources[source]['slides'])

        # Remove any duplicates from shared dataset paths
        paths = list(set(paths))

        # Filter paths
        if apply_filters:
            filtered_slides = self.slides()
            filtered_paths = [path for path in paths if sf.util.path_to_name(path) in filtered_slides]
            return filtered_paths
        else:
            return paths

    def slide_report(self, stride_div=1, destination='auto', tma=False, enable_downsample=True, roi_method='inside',
                     skip_missing_roi=False, normalizer=None, normalizer_source=None):

        """Creates a PDF report of slides, including images of 10 example extracted tiles.

        Args:
            stride_div (int, optional): Stride divisor for tile extraction. Defaults to 1.
            destination (str, optional): Either 'auto' or explicit filename at which to save the PDF report.
                Defaults to 'auto'.
            tma (bool, optional): Interpret slides as TMA (tumor microarrays). Defaults to False.
            enable_downsample (bool, optional): Enable downsampling during tile extraction. Defaults to True.
            roi_method (str, optional): Either 'inside', 'outside', or 'ignore'. Defaults to 'inside'.
                Determines how ROIs will guide tile extraction
            skip_missing_roi (bool, optional): Skip tiles that are missing ROIs. Defaults to False.
            normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image. Defaults to None.
                If None but using a normalizer, will use an internal tile for normalization.
                Internal default tile can be found at slideflow.slide.norm_tile.jpg
        """

        from slideflow.slide import TMA, WSI, ExtractionReport

        log.info('Generating slide report...')
        reports = []
        for source in self.sources:
            roi_dir = self.sources[source]['roi']
            slide_list = self.slide_paths(source=source)

            # Function to extract tiles from a slide
            def get_slide_report(slide_path):
                print(f'\r\033[KGenerating report for slide {sf.util.green(sf.util.path_to_name(slide_path))}...', end='')

                if tma:
                    whole_slide = TMA(slide_path,
                                      self.tile_px,
                                      self.tile_um,
                                      stride_div,
                                      enable_downsample=enable_downsample)
                else:
                    whole_slide = WSI(slide_path,
                                      self.tile_px,
                                      self.tile_um,
                                      stride_div,
                                      enable_downsample=enable_downsample,
                                      roi_dir=roi_dir,
                                      roi_method=roi_method,
                                      skip_missing_roi=skip_missing_roi)

                if not whole_slide.loaded_correctly():
                    return

                report = whole_slide.extract_tiles(normalizer=normalizer, normalizer_source=normalizer_source)
                return report

            for slide_path in slide_list:
                report = get_slide_report(slide_path)
                reports += [report]
        print('\r\033[K', end='')
        log.info('Generating PDF (this may take some time)...', )
        pdf_report = ExtractionReport(reports, tile_px=self.tile_px, tile_um=self.tile_um)
        timestring = datetime.now().strftime('%Y%m%d-%H%M%S')
        filename = destination if destination != 'auto' else join(self.root, f'tile_extraction_report-{timestring}.pdf')
        pdf_report.save(filename)
        log.info(f'Slide report saved to {sf.util.green(filename)}')

    def slides(self):
        """Returns a list of slide names in this dataset."""

        # Begin filtering slides with annotations
        slides = []
        slide_patient_dict = {}
        if self.annotations is None:
            log.error("No annotations loaded; is the annotations file empty?")
        for ann in self.annotations.to_dict(orient="records"):
            skip_annotation = False
            if TCGA.slide not in ann.keys():
                err_msg = f"{TCGA.slide} not found in annotations file."
                log.error(err_msg)
                raise DatasetError(err_msg)

            # Skip missing or blank slides
            if ann[TCGA.slide] in sf.util.SLIDE_ANNOTATIONS_TO_IGNORE:
                continue

            # Ensure slides are only assigned to a single patient
            if ann[TCGA.slide] not in slide_patient_dict:
                slide_patient_dict.update({ann[TCGA.slide]: ann[TCGA.patient]})
            elif slide_patient_dict[ann[TCGA.slide]] != ann[TCGA.patient]:
                log.error(f"Multiple patients assigned to slide {sf.util.green(ann[TCGA.slide])}.")
                return None

            # Only return slides with annotation values specified in "filters"
            if self.filters:
                for filter_key in self.filters.keys():
                    if filter_key not in ann.keys():
                        log.error(f"Filter header {sf.util.bold(filter_key)} not found in annotations file.")
                        raise IndexError(f"Filter header {filter_key} not found in annotations file.")

                    ann_val = ann[filter_key]
                    filter_vals = self.filters[filter_key]
                    filter_vals = [filter_vals] if not isinstance(filter_vals, list) else filter_vals

                    # Allow filtering based on shortnames if the key is a TCGA patient ID
                    if filter_key == TCGA.patient:
                        if ((ann_val not in filter_vals) and
                            (sf.util._shortname(ann_val) not in filter_vals) and
                            (ann_val not in [sf.util._shortname(fv) for fv in filter_vals]) and
                            (sf.util._shortname(ann_val) not in [sf.util._shortname(fv) for fv in filter_vals])):

                            skip_annotation = True
                            break
                    else:
                        if ann_val not in filter_vals:
                            skip_annotation = True
                            break

            # Filter out slides that are blank in a given annotation column ("filter_blank")
            if self.filter_blank and self.filter_blank != [None]:
                for fb in self.filter_blank:
                    if fb not in ann.keys():
                        err_msg = f"Unable to filter blank slides from header {fb}; header was not found in annotations."
                        log.error(err_msg)
                        raise DatasetError(err_msg)

                    if not ann[fb] or ann[fb] == '':
                        skip_annotation = True
                        break
            if skip_annotation: continue
            slides += [ann[TCGA.slide]]
        return slides

    def split_tfrecords_by_roi(self, destination):
        """Split dataset tfrecords into separate tfrecords according to ROI.

        Will generate two sets of tfrecords, with identical names: one with tiles inside the ROIs, one with tiles
        outside the ROIs. Will skip any tfrecords that are missing ROIs. Requires slides to be available.
        """

        import slideflow.io
        from slideflow.slide import WSI

        tfrecords = self.tfrecords()
        slides = {sf.util.path_to_name(s):s for s in self.slide_paths()}
        rois = self.rois()
        manifest = self.manifest()

        for tfr in tfrecords:
            slidename = sf.util.path_to_name(tfr)
            if slidename not in slides:
                continue
            slide = WSI(slides[slidename], self.tile_px, self.tile_um, rois=rois, skip_missing_roi=True)
            if slide.load_error:
                continue
            parser = sf.io.get_tfrecord_parser(tfr, decode_images=False, to_numpy=True)
            reader = sf.io.TFRecordDataset(tfr)
            if not exists(join(destination, 'inside')):
                os.makedirs(join(destination, 'inside'))
            if not exists(join(destination, 'outside')):
                os.makedirs(join(destination, 'outside'))
            inside_roi_writer = sf.io.TFRecordWriter(join(destination, 'inside', f'{slidename}.tfrecords'))
            outside_roi_writer = sf.io.TFRecordWriter(join(destination, 'outside', f'{slidename}.tfrecords'))
            for record in tqdm(reader, total=manifest[tfr]['total']):
                parsed = parser(record)
                loc_x, loc_y = parsed['loc_x'], parsed['loc_y']
                tile_in_roi = any([annPoly.contains(sg.Point(loc_x, loc_y)) for annPoly in slide.annPolys])
                record_bytes = sf.io.read_and_return_record(record, parser)
                if tile_in_roi:
                    inside_roi_writer.write(record_bytes)
                else:
                    outside_roi_writer.write(record_bytes)
            inside_roi_writer.close()
            outside_roi_writer.close()

    def tensorflow(self, labels=None, batch_size=None, **kwargs):
        """Returns a Tensorflow Dataset object that interleaves tfrecords from this dataset.

        The returned dataset yields a batch of (image, label) for each tile. Labels may be specified either via
        a dict mapping slide names to outcomes, or a parsing function which accept and image and slide name and returns
        a dict {'image_raw': image (tensor)} and a label (int or float).

        Args:
            labels (dict or str, optional): Dict or function. If dict, must map slide names to outcome labels.
                If function, function must accept an image (tensor) and slide name (str), and return a dict
                {'image_raw': image (tensor)} and label (int or float). If not provided,  all labels will be None.
            batch_size (int): Batch size.

        Keyword Args:
            onehot (bool, optional): Onehot encode labels. Defaults to False.
            incl_slidenames (bool, optional): Include slidenames as third returned variable. Defaults to False.
            infinite (bool, optional): Infinitely repeat data. Defaults to False.
            rank (int, optional): Worker ID to identify which worker this represents. Used to interleave results
                among workers without duplications. Defaults to 0 (first worker).
            num_replicas (int, optional): Number of GPUs or unique instances which will have their own DataLoader. Used to
                interleave results among workers without duplications. Defaults to 1.
            normalizer (:class:`slideflow.slide.StainNormalizer` or str, optional): Normalizer to use on images.
                Defaults to None.
            seed (int, optional): Use the following seed when randomly interleaving. Necessary for synchronized
                multiprocessing distributed reading.
            chunk_size (int, optional): Chunk size for image decoding. Defaults to 16.
            preload_factor (int, optional): Number of batches to preload. Defaults to 1.
            augment (str, optional): Image augmentations to perform. String containing characters designating augmentations.
                    'x' indicates random x-flipping, 'y' y-flipping, 'r' rotating, and 'j' JPEG compression/decompression
                    at random quality levels. Passing either 'xyrj' or True will use all augmentations.
            standardize (bool, optional): Standardize images to (0,1). Defaults to True.
            num_workers (int, optional): Number of DataLoader workers. Defaults to 2.
        """

        from slideflow.io.tensorflow import interleave

        if 'normalizer' in kwargs and isinstance(kwargs['normalizer'], str):
            kwargs['normalizer'] = sf.slide.StainNormalizer(kwargs['normalizer'])

        return interleave(tfrecords=self.tfrecords(),
                          labels=labels,
                          img_size=self.tile_px,
                          batch_size=batch_size,
                          prob_weights=self.prob_weights,
                          clip=self._clip,
                          **kwargs)

    def tfrecord_report(self, destination, normalizer=None, normalizer_source=None):

        """Creates a PDF report of TFRecords, including 10 example tiles per TFRecord.

        Args:
            destination (str): Path to directory in which to save the PDF report
            normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image. Defaults to None.
                If None but using a normalizer, will use an internal tile for normalization.
                Internal default tile can be found at slideflow.slide.norm_tile.jpg
        """

        from slideflow.slide import ExtractionReport, SlideReport

        if normalizer: log.info(f'Using realtime {normalizer} normalization')
        normalizer = None if not normalizer else sf.slide.StainNormalizer(method=normalizer, source=normalizer_source)

        tfrecord_list = self.tfrecords()
        reports = []
        log.info('Generating TFRecords report...')
        for tfr in tfrecord_list:
            print(f'\r\033[KGenerating report for tfrecord {sf.util.green(sf.util.path_to_name(tfr))}...', end='')
            dataset = sf.io.TFRecordDataset(tfr)
            parser = sf.io.get_tfrecord_parser(tfr, ('image_raw',), to_numpy=True, decode_images=False)
            if not parser: continue
            sample_tiles = []
            for i, record in enumerate(dataset):
                if i > 9: break
                image_raw_data = parser(record)[0]
                if normalizer:
                    image_raw_data = normalizer.jpeg_to_jpeg(image_raw_data)
                sample_tiles += [image_raw_data]
            reports += [SlideReport(sample_tiles, tfr)]

        print('\r\033[K', end='')
        log.info('Generating PDF (this may take some time)...')
        pdf_report = ExtractionReport(reports, tile_px=self.tile_px, tile_um=self.tile_um)
        timestring = datetime.now().strftime('%Y%m%d-%H%M%S')
        if os.path.exists(destination) and os.path.isdir(destination):
            filename = join(destination, f'tfrecord_report-{timestring}.pdf')
        elif sf.util.path_to_ext(destination) == 'pdf':
            filename = join(destination)
        else:
            raise ValueError(f"Could not find destination directory {destination}.")
        pdf_report.save(filename)
        log.info(f'TFRecord report saved to {sf.util.green(filename)}')

    def tfrecords(self, source=None):
        """Returns a list of all tfrecords."""
        if source and source not in self.sources.keys():
            log.error(f"Dataset {source} not found.")
            return None

        sources_to_search = list(self.sources.keys()) if not source else [source]

        tfrecords_list = []
        folders_to_search = []
        for source in sources_to_search:
            tfrecords = self.sources[source]['tfrecords']
            label = self.sources[source]['label']
            if label is None: continue
            tfrecord_path = join(tfrecords, label)
            if not exists(tfrecord_path):
                log.debug(f"TFRecords path not found: {sf.util.green(tfrecord_path)}")
                continue
            folders_to_search += [tfrecord_path]
        for folder in folders_to_search:
            tfrecords_list += glob(join(folder, "*.tfrecords"))

        # Filter the list by filters
        if self.annotations is not None:
            slides = self.slides()
            filtered_tfrecords_list = [tfrecord for tfrecord in tfrecords_list if sf.util.path_to_name(tfrecord) in slides]
            filtered = filtered_tfrecords_list
        else:
            log.warning("No annotations loaded; unable to filter TFRecords list. Is the annotations file empty?")
            filtered = tfrecords_list

        # Filter by min_tiles
        manifest = self.manifest(filter=False)
        if not all ([f in manifest for f in filtered]):
            self.update_manifest()
            manifest = self.manifest(filter=False)
        if self.min_tiles:
            return [f for f in filtered if manifest[f]['total'] >= self.min_tiles]
        else:
            return [f for f in filtered if manifest[f]['total'] > 0]

    def tfrecords_by_subfolder(self, subfolder):
        """Returns a list of all tfrecords in a specific subfolder, ignoring filters."""
        tfrecords_list = []
        folders_to_search = []
        for source in self.sources:
            if self.sources[source]['label'] is None: continue
            base_dir = join(self.sources[source]['tfrecords'], self.sources[source]['label'])
            tfrecord_path = join(base_dir, subfolder)
            if not exists(tfrecord_path):
                err_msg = f"Unable to find subfolder {sf.util.bold(subfolder)} in source {sf.util.bold(source)}, " + \
                            f"tfrecord directory: {sf.util.green(base_dir)}"
                log.error(err_msg)
                raise DatasetError(err_msg)
            folders_to_search += [tfrecord_path]
        for folder in folders_to_search:
            tfrecords_list += glob(join(folder, "*.tfrecords"))
        return tfrecords_list

    def tfrecords_folders(self):
        """Returns folders containing tfrecords."""
        folders = []
        for source in self.sources:
            if self.sources[source]['label'] is None: continue
            folders += [join(self.sources[source]['tfrecords'], self.sources[source]['label'])]
        return folders

    def tfrecords_from_tiles(self, delete_tiles=False):
        """Create tfrecord files from a collection of raw images, as stored in project tiles directory"""
        for source in self.sources:
            log.info(f'Working on dataset source {source}')
            config = self.sources[source]
            tfrecord_dir = join(config['tfrecords'], config['label'])
            tiles_dir = join(config['tiles'], config['label'])
            if not exists(tiles_dir):
                log.warn(f'No tiles found for dataset source {sf.util.bold(source)}')
                continue
            sf.io.write_tfrecords_multi(tiles_dir, tfrecord_dir)
            self.update_manifest()
            if delete_tiles:
                shutil.rmtree(tiles_dir)

    def training_validation_split(self, model_type, labels, val_strategy, splits=None, val_fraction=None,
                                  val_k_fold=None, k_fold_iter=None, site_labels=None, read_only=False):

        """From a specified subfolder within the project's main TFRecord folder, prepare a training set and validation set.
        If a validation split has already been prepared (e.g. K-fold iterations were already determined),
        the previously generated split will be used. Otherwise, create a new split and log the result in the
        TFRecord directory so future models may use the same split for consistency.

        Args:
            model_type (str): Either 'categorical' or 'linear'.
            labels (dict):  Dictionary mapping slides to labels. Used for balancing outcome labels in
                training and validation cohorts.
            val_strategy (str): Either 'k-fold', 'k-fold-preserved-site', 'bootstrap', or 'fixed'.
            splits (str, optional): Path to JSON file containing validation splits. Defaults to None.
            outcome_key (str, optional): Key indicating outcome label in slide_labels_dict. Defaults to 'outcome_label'.
            val_fraction (float, optional): Proportion of data for validation. Not used if strategy is k-fold.
                Defaults to None
            val_k_fold (int): K, required if using K-fold validation. Defaults to None.
            k_fold_iter (int, optional): Which K-fold iteration to generate starting at 1. Fequired if using K-fold
                validation. Defaults to None.
            site_labels (dict, optional): Dict mapping patients to site labels. Used for site preserved cross validation.
            read_only (bool): Prevents writing validation splits to file. Defaults to False.

        Returns:
            (slideflow.dataset.Dataset, slideflow.dataset.Dataset): training dataset, validation dataset
        """

        if (not k_fold_iter and val_strategy=='k-fold'):
            raise DatasetError("If strategy is 'k-fold', must supply k_fold_iter (int starting at 1)")
        if (not val_k_fold and val_strategy=='k-fold'):
            raise DatasetError("If strategy is 'k-fold', must supply val_k_fold (K)")
        if val_strategy == 'k-fold-preserved-site' and not site_labels:
            raise DatasetError("k-fold-preserved-site requires site_labels (dict mapping patients to sites, or " + \
                                "name of annotation column header")
        if isinstance(site_labels, str):
            site_labels, _ = self.labels(site_labels, verbose=False, format='name')

        # Prepare dataset
        patients = self.patients()
        splits_file = splits
        k_fold = val_k_fold
        training_tfrecords = []
        val_tfrecords = []
        accepted_split = None
        slide_list = list(labels.keys())

        # Assemble dictionary of patients linking to list of slides and outcome labels
        # dataset.labels() ensures no duplicate outcome labels are found in a single patient
        tfrecord_dir_list = self.tfrecords()
        if not len(tfrecord_dir_list):
            raise ValueError("Unable to generate training/val split; no tfrecords found")
        tfrecord_dir_list_names = [tfr.split('/')[-1][:-10] for tfr in tfrecord_dir_list]
        patients_dict = {}
        num_warned = 0
        for slide in slide_list:
            patient = slide if not patients else patients[slide]
            # Skip slides not found in directory
            if slide not in tfrecord_dir_list_names:
                log.debug(f"Slide {slide} not found in tfrecord directory, skipping")
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
                err_msg = f"Multiple outcome labels found for patient {patient} ({ol}, {ok})"
                log.error(err_msg)
                raise DatasetError(err_msg)
            else:
                patients_dict[patient]['slides'] += [slide]

        # Add site labels to the patients dict if doing preserved-site cross-validation
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
                    err_msg = f"Multiple site labels found for patient {patient} ({ol}, {ok})"
                    log.error(err_msg)
                    raise DatasetError(err_msg)

        if num_warned:
            log.warning(f"Total of {num_warned} slides not found in tfrecord directory, skipping")
        patients_list = list(patients_dict.keys())
        sorted_patients = [p for p in patients_list]
        sorted_patients.sort()
        shuffle(patients_list)

        # Create and log a validation subset
        if val_strategy == 'none':
            log.info(f"Validation strategy set to 'none'; selecting no tfrecords for validation.")
            training_slides = np.concatenate([patients_dict[patient]['slides']
                                                for patient in patients_dict.keys()]).tolist()
            validation_slides = []
        elif val_strategy == 'bootstrap':
            num_val = int(val_fraction * len(patients_list))
            log.info(f"Boostrap validation: selecting {sf.util.bold(num_val)} pts at random for validation testing")
            validation_patients = patients_list[0:num_val]
            training_patients = patients_list[num_val:]
            if not len(validation_patients) or not len(training_patients):
                err_msg = "Insufficient number of patients to generate validation dataset."
                log.error(err_msg)
                raise DatasetError(err_msg)
            validation_slides = np.concatenate([patients_dict[patient]['slides']
                                                for patient in validation_patients]).tolist()
            training_slides = np.concatenate([patients_dict[patient]['slides']
                                                for patient in training_patients]).tolist()
        else:
            # Try to load validation split
            loaded_splits = [] if (not splits_file or not exists(splits_file)) else sf.util.load_json(splits_file)
            for split in loaded_splits:
                # First, see if strategy is the same
                if split['strategy'] != val_strategy:
                    continue
                # If k-fold, check that k-fold length is the same
                if (val_strategy == 'k-fold' or val_strategy == 'k-fold-preserved-site') \
                    and len(list(split['tfrecords'].keys())) != k_fold:

                    continue

                # Then, check if patient lists are the same
                split_patients = list(split['patients'].keys())
                split_patients.sort()
                if split_patients == sorted_patients:
                    # Finally, check if outcome variables are the same
                    if [patients_dict[p]['outcome_label'] for p in split_patients] == \
                        [split['patients'][p]['outcome_label']for p in split_patients]:

                        log.info(f"Using {val_strategy} validation split detected at {sf.util.green(splits_file)}")
                        accepted_split = split
                        break

            # If no split found, create a new one
            if not accepted_split:
                if splits_file:
                    log.info(f"No compatible training/validation split found; will log new split at {sf.util.green(splits_file)}")
                else:
                    log.info(f"No training/validation splits file provided; unable to save or load validation splits.")
                new_split = {
                    'strategy':        val_strategy,
                    'patients':        patients_dict,
                    'tfrecords':    {}
                }
                if val_strategy == 'fixed':
                    num_val = int(val_fraction * len(patients_list))
                    validation_patients = patients_list[0:num_val]
                    training_patients = patients_list[num_val:]
                    if not len(validation_patients) or not len(training_patients):
                        err_msg = "Insufficient number of patients to generate validation dataset."
                        log.error(err_msg)
                        raise DatasetError(err_msg)
                    validation_slides = np.concatenate([patients_dict[patient]['slides']
                                                        for patient in validation_patients]).tolist()
                    training_slides = np.concatenate([patients_dict[patient]['slides']
                                                        for patient in training_patients]).tolist()
                    new_split['tfrecords']['validation'] = validation_slides
                    new_split['tfrecords']['training'] = training_slides
                elif val_strategy == 'k-fold' or val_strategy == 'k-fold-preserved-site':
                    balance = 'outcome_label' if model_type == 'categorical' else None
                    k_fold_patients = split_patients_list(patients_dict,
                                                          k_fold,
                                                          balance=balance,
                                                          randomize=True,
                                                          preserved_site=(val_strategy == 'k-fold-preserved-site'))
                    # Verify at least one patient is in each k_fold group
                    if len(k_fold_patients) != k_fold or not min([len(pl) for pl in k_fold_patients]):
                        err_msg = "Insufficient number of patients to generate validation dataset."
                        log.error(err_msg)
                        raise DatasetError(err_msg)
                    training_patients = []
                    for k in range(1, k_fold+1):
                        new_split['tfrecords'][f'k-fold-{k}'] = np.concatenate([patients_dict[patient]['slides']
                                                                                    for patient in k_fold_patients[k-1]]).tolist()
                        if k == k_fold_iter:
                            validation_patients = k_fold_patients[k-1]
                        else:
                            training_patients += k_fold_patients[k-1]
                    validation_slides = np.concatenate([patients_dict[patient]['slides']
                                                        for patient in validation_patients]).tolist()
                    training_slides = np.concatenate([patients_dict[patient]['slides']
                                                        for patient in training_patients]).tolist()
                else:
                    err_msg = f"Unknown validation strategy {val_strategy} requested."
                    log.error(err_msg)
                    raise DatasetError(err_msg)
                # Write the new split to log
                loaded_splits += [new_split]
                if not read_only and splits_file:
                    sf.util.write_json(loaded_splits, splits_file)
            else:
                # Use existing split
                if val_strategy == 'fixed':
                    validation_slides = accepted_split['tfrecords']['validation']
                    training_slides = accepted_split['tfrecords']['training']
                elif val_strategy == 'k-fold' or val_strategy == 'k-fold-preserved-site':
                    validation_slides = accepted_split['tfrecords'][f'k-fold-{k_fold_iter}']
                    training_slides = np.concatenate([accepted_split['tfrecords'][f'k-fold-{ki}']
                                                        for ki in range(1, k_fold+1)
                                                        if ki != k_fold_iter]).tolist()
                else:
                    err_msg = f"Unknown validation strategy {val_strategy} requested."
                    log.error(err_msg)
                    raise DatasetError(err_msg)

            # Perform final integrity check to ensure no patients are in both training and validation slides
            if patients:
                validation_pt = list(set([patients[slide] for slide in validation_slides]))
                training_pt = list(set([patients[slide] for slide in training_slides]))
            else:
                validation_pt, training_pt = validation_slides, training_slides
            if sum([pt in training_pt for pt in validation_pt]):
                err_msg = "At least one patient is in both validation and training sets."
                log.error(err_msg)
                raise DatasetError(err_msg)

            # Return list of tfrecords
            val_tfrecords = [tfr for tfr in tfrecord_dir_list if sf.util.path_to_name(tfr) in validation_slides]
            training_tfrecords = [tfr for tfr in tfrecord_dir_list if sf.util.path_to_name(tfr) in training_slides]

        assert(len(val_tfrecords) == len(validation_slides))
        assert(len(training_tfrecords) == len(training_slides))

        training_dts = copy.deepcopy(self).filter(filters={'slide': training_slides})
        val_dts = copy.deepcopy(self).filter(filters={'slide': validation_slides})

        assert(sorted(training_dts.tfrecords()) == sorted(training_tfrecords))
        assert(sorted(val_dts.tfrecords()) == sorted(val_tfrecords))

        return training_dts, val_dts

    def torch(self, labels, batch_size=None, rebuild_index=False, **kwargs):
        """Returns a PyTorch DataLoader object that interleaves tfrecords from this dataset.

        The returned data loader yields a batch of (image, label) for each tile.

        Args:
            labels (dict or str): If a dict is provided, expect a dict mapping slide names to outcome labels. If a str,
                will intepret as categorical annotation header. For linear outcomes, or outcomes with manually
                assigned labels, pass the first result of dataset.labels(...).
                If None, will return slide name instead of label.
            batch_size (int): Batch size.
            rebuild_index (bool): Re-build index files even if already present. Defaults to True.

        Keyword Args:
            onehot (bool, optional): Onehot encode labels. Defaults to False.
            incl_slidenames (bool, optional): Include slidenames as third returned variable. Defaults to False.
            infinite (bool, optional): Infinitely repeat data. Defaults to False.
            rank (int, optional): Worker ID to identify which worker this represents. Used to interleave results
                among workers without duplications. Defaults to 0 (first worker).
            num_replicas (int, optional): Number of GPUs or unique instances which will have their own DataLoader. Used to
                interleave results among workers without duplications. Defaults to 1.
            normalizer (:class:`slideflow.slide.StainNormalizer` or str, optional): Normalizer to use on images.
                Defaults to None.
            seed (int, optional): Use the following seed when randomly interleaving. Necessary for synchronized
                multiprocessing distributed reading.
            chunk_size (int, optional): Chunk size for image decoding. Defaults to 16.
            preload_factor (int, optional): Number of batches to preload. Defaults to 1.
            augment (str, optional): Image augmentations to perform. String containing characters designating augmentations.
                    'x' indicates random x-flipping, 'y' y-flipping, 'r' rotating, and 'j' JPEG compression/decompression
                    at random quality levels. Passing either 'xyrj' or True will use all augmentations.
            standardize (bool, optional): Standardize images to (0,1). Defaults to True.
            num_workers (int, optional): Number of DataLoader workers. Defaults to 2.
            pin_memory (bool, optional): Pin memory to GPU. Defaults to True.
        """

        from slideflow.io.torch import interleave_dataloader

        if isinstance(labels, str):
            labels = self.labels(labels)[0]

        if 'normalizer' in kwargs and isinstance(kwargs['normalizer'], str):
            kwargs['normalizer'] = sf.slide.StainNormalizer(kwargs['normalizer'])

        self.build_index(rebuild_index)
        tfrecords = self.tfrecords()
        prob_weights = [self.prob_weights[tfr] for tfr in tfrecords] if self.prob_weights else None
        indices = self.load_indices()
        indices = [indices[sf.util.path_to_name(tfr)] for tfr in tfrecords]
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
            forced_update (bool, optional): Force regeneration of the manifest from scratch.
        """

        # Import delayed until here in order to avoid importing tensorflow/torch until necessary,
        # as tensorflow/torch claims a GPU once imported

        import slideflow.io

        tfrecords_folders = self.tfrecords_folders()
        for tfr_folder in tfrecords_folders:
            slideflow.io.update_manifest_at_dir(directory=tfr_folder,
                                                force_update=force_update)

    def update_annotations_with_slidenames(self, annotations_file):
        """Attempts to automatically associate slide names from a directory with patients in a given annotations file,
            skipping any slide names that are already present in the annotations file."""

        header, _ = sf.util.read_annotations(annotations_file)
        slide_list = self.slide_paths(apply_filters=False)

        # First, load all patient names from the annotations file
        try:
            patient_index = header.index(TCGA.patient)
        except:
            err_msg = f"Patient header {TCGA.patient} not found in annotations file."
            log.error(err_msg)
            raise DatasetError(f"Patient header {TCGA.patient} not found in annotations file.")
        patients = []
        patient_slide_dict = {}
        with open(annotations_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader, None)
            for row in csv_reader:
                patients.extend([row[patient_index]])
        patients = list(set(patients))
        log.debug(f"Number of patients in annotations: {len(patients)}")
        log.debug(f"Slides found: {len(slide_list)}")

        # Then, check for sets of slides that would match to the same patient; due to ambiguity, these will be skipped.
        num_occurrences = {}
        for slide in slide_list:
            if _shortname(slide) not in num_occurrences:
                num_occurrences[_shortname(slide)] = 1
            else:
                num_occurrences[_shortname(slide)] += 1
        slides_to_skip = [slide for slide in slide_list if num_occurrences[_shortname(slide)] > 1]

        # Next, search through the slides folder for all valid slide files
        num_warned = 0
        warn_threshold = 1
        for slide_filename in slide_list:
            slide_name = sf.util.path_to_name(slide_filename)
            print_func = print if num_warned < warn_threshold else None
            # First, skip this slide due to ambiguity if needed
            if slide_name in slides_to_skip:
                lead_msg = f"Unable to associate slide {slide_name} due to ambiguity"
                log.warning(f"{lead_msg}; multiple slides match to patient {_shortname(slide_name)}; skipping.")
                num_warned += 1
            # Then, make sure the shortname and long name aren't both in the annotation file
            if (slide_name != _shortname(slide_name)) and (slide_name in patients) and (_shortname(slide_name) in patients):
                lead_msg = f"Unable to associate slide {slide_name} due to ambiguity"
                log.warning(f"{lead_msg}; both {slide_name} and {_shortname(slide_name)} are patients; skipping.")
                num_warned += 1

            # Check if either the slide name or the shortened version are in the annotation file
            if any(x in patients for x in [slide_name, _shortname(slide_name)]):
                slide = slide_name if slide_name in patients else _shortname(slide_name)
                patient_slide_dict.update({slide: slide_name})
            else:
                #log.warning(f"Slide '{slide_name}' not found in annotations file, skipping.")
                #num_warned += 1
                pass
        if num_warned >= warn_threshold:
            log.warning(f"...{num_warned} total warnings, see project log for details")

        # Now, write the assocations
        num_updated_annotations = 0
        num_missing = 0
        with open(annotations_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader, None)
            with open('temp.csv', 'w') as csv_outfile:
                csv_writer = csv.writer(csv_outfile, delimiter=',')

                # Write to existing "slide" column in the annotations file if it exists,
                # otherwise create new column
                try:
                    slide_index = header.index(TCGA.slide)
                    csv_writer.writerow(header)
                    for row in csv_reader:
                        patient = row[patient_index]
                        # Only write column if no slide is documented in the annotation
                        if (patient in patient_slide_dict) and (row[slide_index] == ''):
                            row[slide_index] = patient_slide_dict[patient]
                            num_updated_annotations += 1
                        elif (patient not in patient_slide_dict) and (row[slide_index] == ''):
                            num_missing += 1
                        csv_writer.writerow(row)
                except:
                    header.extend([TCGA.slide])
                    csv_writer.writerow(header)
                    for row in csv_reader:
                        patient = row[patient_index]
                        if patient in patient_slide_dict:
                            row.extend([patient_slide_dict[patient]])
                            num_updated_annotations += 1
                        else:
                            row.extend([""])
                            num_missing += 1
                        csv_writer.writerow(row)
        if num_updated_annotations:
            log.info(f"Successfully associated slides with {num_updated_annotations} annotation entries.")
            if num_missing:
                log.info(f"Slides not found for {num_missing} annotations.")
        elif num_missing:
            log.debug(f"No annotation updates performed. Slides not found for {num_missing} annotations.")
        else:
            log.debug(f"Annotations up-to-date, no changes made.")

        # Finally, backup the old annotation file and overwrite existing with the new data
        backup_file = f"{annotations_file}.backup"
        if exists(backup_file):
            os.remove(backup_file)
        shutil.move(annotations_file, backup_file)
        shutil.move('temp.csv', annotations_file)

    def verify_annotations_slides(self):
        """Verify that annotations are correctly loaded."""

        # Verify no duplicate slide names are found
        slide_list_from_annotations = self.slides()
        if len(slide_list_from_annotations) != len(list(set(slide_list_from_annotations))):
            log.error("Duplicate slide names detected in the annotation file.")
            raise DatasetError("Duplicate slide names detected in the annotation file.")

        # Verify all slides in the annotation column are valid
        num_warned = 0
        warn_threshold = 3
        for annotation in self.annotations.to_dict(orient="records"):
            slide = annotation[TCGA.slide]
            if slide == '':
                log.warning(f"Patient {sf.util.green(annotation[TCGA.patient])} has no slide assigned.")
                num_warned += 1
        if num_warned >= warn_threshold:
            log.warning(f"...{num_warned} total warnings, see project log for details")
