
import queue
import threading
import time
import os
import csv
import shutil
import multiprocessing
import shapely.geometry as sg
import slideflow as sf

from glob import glob
from os import listdir
from datetime import datetime
from tqdm import tqdm
from os.path import isdir, join, exists, dirname
from slideflow.util import log, TCGA, _shortname, ProgressBar

class DatasetError(Exception):
    pass

def _tile_extractor(slide_path, tfrecord_dir, tiles_dir, roi_dir, roi_method, skip_missing_roi, randomize_origin,
                    tma, tile_px, tile_um, stride_div, downsample, buffer, pb_counter, counter_lock, generator_kwargs):
    """Internal function to execute tile extraction. Slide processing needs to be process-isolated."""

    # Record function arguments in case we need to re-call the function (for corrupt tiles)
    local_args = locals()

    from slideflow.slide import TMA, WSI, TileCorruptionError
    log.handlers[0].flush_line = True
    try:
        log.debug(f'Extracting tiles for slide {sf.util.path_to_name(slide_path)}')

        if tma:
            whole_slide = TMA(slide_path,
                              tile_px,
                              tile_um,
                              stride_div,
                              enable_downsample=downsample,
                              report_dir=tfrecord_dir,
                              buffer=buffer)
        else:
            whole_slide = WSI(slide_path,
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

        if not whole_slide.loaded_correctly():
            return

        try:
            report = whole_slide.extract_tiles(tfrecord_dir=tfrecord_dir, tiles_dir=tiles_dir, **generator_kwargs)

        except TileCorruptionError:
            if downsample:
                log.warning(f'Corrupt tile in {sf.util.path_to_name(slide_path)}; will try disabling downsampling')
                report = _tile_extractor(**local_args)
            else:
                log.error(f'Corrupt tile in {sf.util.path_to_name(slide_path)}; skipping slide')
                return
        del whole_slide
        return report
    except (KeyboardInterrupt, SystemExit):
        print('Exiting...')
        return

class Dataset:
    """Object to supervise organization of slides, tfrecords, and tiles
    across a one or more sources in a stored configuration file."""

    def __init__(self, config_file, sources, tile_px, tile_um, annotations=None, filters=None, filter_blank=None):
        self.tile_px = tile_px
        self.tile_um = tile_um
        self.annotations = []
        self.filters = filters
        self.filter_blank = filter_blank

        config = sf.util.load_json(config_file)
        sources = sources if isinstance(sources, list) else [sources]

        try:
            self.sources = {k:v for (k,v) in config.items() if k in sources}
        except KeyError:
            sources_list = ", ".join(sources)
            err_msg = f"Unable to find source '{sf.util.bold(sources_list)}' in config file {sf.util.green(config_file)}"
            log.error(err_msg)
            raise DatasetError(err_msg)

        if (tile_px is not None) and (tile_um is not None):
            label = f"{tile_px}px_{tile_um}um"
        else:
            label = None

        for source in self.sources:
            self.sources[source]['label'] = label

        if annotations:
            self.load_annotations(annotations)

    def apply_filters(self, **kwargs):
        for kwarg in kwargs:
            if kwarg not in ('filters', 'filter_blank'):
                raise sf.util.UserError(f'Unknown filtering argument {kwarg}')
        if 'filters' in kwargs:
            self.filters = kwargs['filters']
        if 'filter_blank' in kwargs:
            self.filter_blank = kwargs['filter_blank']

    def filtered(self, **kwargs):
        """Temporarily filters dataset with the provided filters or filter_blank.

        Use by calling this function through 'with'. After exiting the 'with' block,
        filters will be reverted to their original values.

        .. code-block:: python

            with dataset.filtered(filters={...}):
                ...

        Args:
            None

        Keyword Args:
            filters (dict, optional): Filters dict to use when selecting tfrecords. Defaults to None.
                Ignored if dataset is supplied.
                See :meth:`get_dataset` documentation for more information on filtering.
            filter_blank (list, optional): Slides blank in these columns will be excluded. Defaults to None.
                Ignored if dataset is supplied.

        Returns:
            Object that, when entered, temporarily filters this dataset using the supplied filters or filter_blank,
            and when exited, reverts filters back to their previous state.
        """

        for kwarg in kwargs:
            if kwarg not in ('filters', 'filter_blank'):
                raise sf.util.UserError(f'Unknown filtering argument {kwarg}')
        parent = self

        class TempFilter:
            def __init__(self): pass
            def __enter__(self):
                self.old_filters = parent.filters
                self.old_filter_blank = parent.filter_blank
                parent.apply_filters(**kwargs)
            def __exit__(self, exc_type, exc_value, traceback):
                parent.apply_filters(self.old_filters, self.old_filter_blank)

        return TempFilter()

    def extract_tiles(self, save_tiles=False, save_tfrecord=True, source=None, stride_div=1, enable_downsample=False,
                      roi_method='inside', skip_missing_roi=True, skip_extracted=True, tma=False,
                      randomize_origin=False, buffer=None, num_workers=4, **kwargs):

        """Extract tiles from a group of slides, saving extracted tiles to either loose image or in
        TFRecord binary format.

        Args:
            save_tiles (bool, optional): Save images of extracted tiles to project tile directory. Defaults to False.
            save_tfrecord (bool, optional): Save compressed image data from extracted tiles into TFRecords
                in the corresponding TFRecord directory. Defaults to True.
            source (str, optional): Name of dataset source from which to select slides for extraction. Defaults to None.
                If not provided, will default to all sources in project.
            stride_div (int, optional): Stride divisor to use when extracting tiles. Defaults to 1.
                A stride of 1 will extract non-overlapping tiles.
                A stride_div of 2 will extract overlapping tiles, with a stride equal to 50% of the tile width.
            enable_downsample (bool, optional): Enable downsampling when reading slide images. Defaults to False.
                This may result in corrupted image tiles if downsampled slide layers are corrupted or incomplete.
                Recommend manual confirmation of tile integrity.
            roi_method (str, optional): Either 'inside', 'outside', or 'ignore'. Defaults to 'inside'.
                Indicates whether tiles are extracted inside or outside ROIs, or if ROIs are ignored entirely.
            skip_missing_roi (bool, optional): Skip slides that are missing ROIs. Defaults to True.
            skip_extracted (bool, optional): Skip slides that have already been extracted. Defaults to True.
            tma (bool, optional): Reads slides as Tumor Micro-Arrays (TMAs), detecting and extracting tumor cores.
                Defaults to False. Experimental function with limited testing.
            randomize_origin (bool, optional): Randomize pixel starting position during extraction. Defaults to False.
            buffer (str, optional): Slides will be copied to this directory before extraction. Defaults to None.
                Using an SSD or ramdisk buffer vastly improves tile extraction speed.
            num_workers (int, optional): Extract tiles from this many slides simultaneously. Defaults to 4.

        Keyword Args:
            normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image. Defaults to None.
                If None but using a normalizer, will use an internal tile for normalization.
                Internal default tile can be found at slideflow.util.norm_tile.jpg
            whitespace_fraction (float, optional): Range 0-1. Defaults to 1.
                Discard tiles with this fraction of whitespace. If 1, will not perform whitespace filtering.
            whitespace_threshold (int, optional): Range 0-255. Defaults to 230.
                Threshold above which a pixel (RGB average) is considered whitespace.
            grayspace_fraction (float, optional): Range 0-1. Defaults to 0.6.
                Discard tiles with this fraction of grayspace. If 1, will not perform grayspace filtering.
            grayspace_threshold (float, optional): Range 0-1. Defaults to 0.05.
                Pixels in HSV format with saturation below this threshold are considered grayspace.
            img_format (str, optional): 'png' or 'jpg'. Defaults to 'png'. Image format to use in tfrecords.
                PNG (lossless) format recommended for fidelity, JPG (lossy) for efficiency.
            full_core (bool, optional): Only used if extracting from TMA. If True, will save entire TMA core as image.
                Otherwise, will extract sub-images from each core using the given tile micron size. Defaults to False.
            shuffle (bool, optional): Shuffle tiles prior to storage in tfrecords. Defaults to True.
            num_threads (int, optional): Number of workers threads for each tile extractor. Defaults to 4.
        """

        import slideflow.slide

        if not save_tiles and not save_tfrecord:
            log.error('Either save_tiles or save_tfrecord must be true to extract tiles.')
            return

        if source:  sources = [source] if not isinstance(source, list) else source
        else:       sources = self.sources

        self.verify_annotations_slides()
        sf.slide.log_extraction_params(**kwargs)

        for source in sources:
            log.info(f'Working on dataset source {sf.util.bold(source)}...')

            roi_dir = self.sources[source]['roi']
            source_config = self.sources[source]
            tfrecord_dir = join(source_config['tfrecords'], source_config['label'])
            if save_tfrecord and not exists(tfrecord_dir):
                tfrecord_dir = join(source_config['tfrecords'], source_config['label'])
                tiles_dir = None
                os.makedirs(tfrecord_dir)
            if save_tiles and not os.path.exists(tiles_dir):
                tiles_dir = join(source_config['tiles'], source_config['label'])
                tfrecord_dir = None
                os.makedirs(tiles_dir)

            # Prepare list of slides for extraction
            slide_list = self.get_slide_paths(source=source)

            # Check for interrupted or already-extracted tfrecords
            if skip_extracted and save_tfrecord:
                already_done = [sf.util.path_to_name(tfr) for tfr in self.get_tfrecords(source=source)]
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
            for slide_path in tqdm(slide_list, leave=False):
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
                ctx = multiprocessing.get_context('spawn')
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
                    'generator_kwargs': kwargs
                }

                # Worker to grab slide path from queue and start tile extraction
                def worker():
                    while True:
                        try:
                            path = q.get()
                            process = ctx.Process(target=_tile_extractor, args=(path,), kwargs=extraction_kwargs)
                            process.start()
                            process.join()
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
                            if q.qsize() < num_workers:
                                try:
                                    buffered_path = join(buffer, os.path.basename(slide_path))
                                    shutil.copy(slide_path, buffered_path)
                                    q.put(buffered_path)
                                    break
                                except OSError as e:
                                    if not warned:
                                        formatted_slide = sf.util._shortname(sf.util.path_to_name(slide_path))
                                        log.warn(f'OSError encountered for slide {formatted_slide}: buffer likely full')
                                        log.info(f'Q size: {q.qsize()}')
                                        warned = True
                                    time.sleep(1)
                            else:
                                time.sleep(1)
                    else:
                        q.put(slide_path)
                q.join()
                task_finished = True
                if pb: pb.end()
                log.info('Generating PDF (this may take some time)...', )
                pdf_report = sf.slide.ExtractionReport(reports.values(), tile_px=self.tile_px, tile_um=self.tile_um)
                timestring = datetime.now().strftime('%Y%m%d-%H%M%S')
                pdf_report.save(join(tfrecord_dir, f'tile_extraction_report-{timestring}.pdf'))

            # Update manifest
            self.update_manifest()

    def extract_tiles_from_tfrecords(self, dest):
        """Extracts tiles from a set of TFRecords.

        Args:
            dest (str): Path to directory in which to save tile images. Defaults to None. If None, uses dataset default.
        """
        for source in self.sources:
            to_extract_tfrecords = self.get_tfrecords(source=source)
            if dest:
                tiles_dir = dest
            else:
                tiles_dir = join(self.sources[source]['tiles'],
                                 self.sources[source]['label'])
                if not exists(tiles_dir):
                    os.makedirs(tiles_dir)
            for tfr in to_extract_tfrecords:
                sf.io.tfrecords.extract_tiles(tfr, tiles_dir)

    def generate_tfrecords_from_tiles(self, delete_tiles=True):
        """Create tfrecord files from a collection of raw images, as stored in project tiles directory"""
        for source in self.sources:
            log.info(f'Working on dataset source {source}')
            config = self.sources[source]
            tfrecord_dir = join(config['tfrecords'], config['label'])
            tiles_dir = join(config['tiles'], config['label'])
            if not exists(tiles_dir):
                log.warn(f'No tiles found for dataset source {sf.util.bold(source)}')
                continue

            # Check to see if subdirectories in the target folders are slide directories (contain images)
            #  or are further subdirectories (e.g. validation and training)
            log.info('Scanning tile directory structure...')
            if sf.util.contains_nested_subdirs(tiles_dir):
                subdirs = [_dir for _dir in os.listdir(tiles_dir) if isdir(join(tiles_dir, _dir))]
                for subdir in subdirs:
                    tfrecord_subdir = join(tfrecord_dir, subdir)
                    sf.io.tfrecords.write_tfrecords_multi(join(tiles_dir, subdir), tfrecord_subdir)
            else:
                sf.io.tfrecords.write_tfrecords_multi(tiles_dir, tfrecord_dir)

            self.update_manifest()

            if delete_tiles:
                shutil.rmtree(tiles_dir)

    def get_manifest(self, key='path'):
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
                log.info(f"No manifest file detected in {tfrecord_dir}; will create now")

                # Import delayed until here in order to avoid importing tensorflow until necessary,
                # as tensorflow claims a GPU once imported
                import slideflow.io.tfrecords
                slideflow.io.tfrecords.update_manifest_at_dir(tfrecord_dir)

            relative_manifest = sf.util.load_json(manifest_path)
            global_manifest = {}
            for record in relative_manifest:
                k = join(tfrecord_dir, record) if key == 'path' else sf.util.path_to_name(record)
                global_manifest.update({k: relative_manifest[record]})
            combined_manifest.update(global_manifest)

        # Now filter out any tfrecords that would be excluded by filters
        if key == 'path':
            filtered_tfrecords = self.get_tfrecords()
        else:
            filtered_tfrecords = [sf.util.path_to_name(tfr) for tfr in self.get_tfrecords()]
        manifest_tfrecords = list(combined_manifest.keys())
        for tfr in manifest_tfrecords:
            if tfr not in filtered_tfrecords:
                del(combined_manifest[tfr])

        return combined_manifest

    def get_rois(self):
        """Returns a list of all ROIs."""
        rois_list = []
        for source in self.sources:
            rois_list += glob(join(self.sources[source]['roi'], "*.csv"))
        rois_list = list(set(rois_list))
        return rois_list

    def get_slides(self):
        """Returns a list of slide names in this dataset."""

        # Begin filtering slides with annotations
        slides = []
        self.filter_blank = [self.filter_blank] if not isinstance(self.filter_blank, list) else self.filter_blank
        slide_patient_dict = {}
        if not len(self.annotations):
            log.error("No annotations loaded; is the annotations file empty?")
        for ann in self.annotations:
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

    def get_slide_paths(self, source=None, filter=True):
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
        if filter:
            filtered_slides = self.get_slides()
            filtered_paths = [path for path in paths if sf.util.path_to_name(path) in filtered_slides]
            return filtered_paths
        else:
            return paths

    def get_tfrecords(self, source=None):
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
                log.warning(f"TFRecords path not found: {sf.util.green(tfrecord_path)}")
                return []
            subdirs = [sd for sd in listdir(tfrecord_path) if isdir(join(tfrecord_path, sd))]
            formatted_subdirs = ', '.join([sf.util.green(s) for s in subdirs])
            folders_to_search += [tfrecord_path]
        for folder in folders_to_search:
            tfrecords_list += glob(join(folder, "*.tfrecords"))

        # Now filter the list
        if self.annotations:
            slides = self.get_slides()
            filtered_tfrecords_list = [tfrecord for tfrecord in tfrecords_list if tfrecord.split('/')[-1][:-10] in slides]
            return filtered_tfrecords_list
        else:
            log.warning("No annotations loaded; unable to filter TFRecords list. Is the annotations file empty?")
            return tfrecords_list

    def get_tfrecords_by_subfolder(self, subfolder):
        """Returns a list of tfrecords in a specific subfolder."""
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

    def get_tfrecords_folders(self):
        """Returns folders containing tfrecords."""
        folders = []
        for source in self.sources:
            if self.sources[source]['label'] is None: continue
            folders += [join(self.sources[source]['tfrecords'], self.sources[source]['label'])]
        return folders

    def get_labels_from_annotations(self, headers, use_float=False, assigned_labels=None, key='label', verbose=True):
        """Returns a dictionary of slide names mapping to patient id and [an] label(s).

        Args:
            headers (list) Annotation header(s) that specifies label variable. May be a list or string.
            use_float (bool, optional) Either bool, dict, or 'auto'.
                If true, will try to convert all data into float. If unable, will raise TypeError.
                If false, will interpret all data as categorical.
                If a dict is provided, will look up each header to determine whether float is used.
                If 'auto', will try to convert all data into float. For each header in which this fails, will
                interpret as categorical instead.
            assigned_labels (dict, optional):  Dictionary mapping label ids to label names. If not provided, will map
                ids to names by sorting alphabetically.
            key (str, optional): Key name to use for the returned dictionary. Defaults to 'label'.
            verbose (bool, optional): Verbose output.

        Returns:
            1) Dictionary with slides as keys and dictionaries as values.
                The value dictionaries contain both "TCGA.patient" and "label" (or manually specified) keys.
            2) list of unique labels
        """

        slides = self.get_slides()
        filtered_annotations = [a for a in self.annotations if a[TCGA.slide] in slides]
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
                filtered_labels = [a[header] for a in filtered_annotations]
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
                try:
                    filtered_labels = [float(o) for o in filtered_labels]
                    use_float_for_this_header = True
                except ValueError:
                    use_float_for_this_header = False
            else:
                raise DatasetError(f"Invalid use_float option {use_float}")

            # Ensure labels can be converted to desired type, then assign values
            if use_float_for_this_header:
                try:
                    filtered_labels = [float(o) for o in filtered_labels]
                except ValueError:
                    raise TypeError(f"Unable to convert label {header} into type 'float'.")
            else:
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
                else:
                    return unique_labels_for_this_header.index(o)

            # Assemble results dictionary
            patient_labels = {}
            num_warned = 0
            warn_threshold = 3
            for annotation in filtered_annotations:
                slide = annotation[TCGA.slide]
                patient = annotation[TCGA.patient]
                annotation_label = _process_label(annotation[header])
                print_func = print if num_warned < warn_threshold else None

                # Mark this slide as having been already assigned a label with his header
                assigned_headers[header][slide] = True

                # Ensure patients do not have multiple labels
                if patient not in patient_labels:
                    patient_labels[patient] = annotation_label
                elif patient_labels[patient] != annotation_label:
                    log.error(f"Multiple different labels in header {header} found for patient {patient}:")
                    log.error(f"{patient_labels[patient]}")
                    log.error(f"{annotation_label}")
                    num_warned += 1
                elif (slide in slides) and (slide in results) and (slide in assigned_headers[header]):
                    continue

                if slide in slides:
                    if slide in results:
                        so = results[slide][key]
                        results[slide][key] = [so] if not isinstance(so, list) else so
                        results[slide][key] += [annotation_label]
                    else:
                        results[slide] = {key: annotation_label if not use_float_for_this_header else [annotation_label]}
                        results[slide][TCGA.patient] = patient
            if num_warned >= warn_threshold:
                log.warning(f"...{num_warned} total warnings, see project log for details")
            unique_labels[header] = unique_labels_for_this_header
        if len(headers) == 1:
            unique_labels = unique_labels[headers[0]]
        return results, unique_labels

    def resize_tfrecords(self, tile_px):
        """Resizes images in a set of TFRecords to a given pixel size.

        Args:
            tile_px (int): Target pixel size for resizing TFRecord images.
        """

        log.info(f'Resizing TFRecord tiles to ({tile_px}, {tile_px})')
        tfrecords_list = self.get_tfrecords()
        log.info(f'Resizing {len(tfrecords_list)} tfrecords')
        for tfr in tfrecords_list:
            sf.io.tfrecords.transform_tfrecord(tfr, tfr+'.transformed', resize=tile_px)

    def slide_report(self, stride_div=1, destination='auto', tma=False, enable_downsample=False,
                        roi_method='inside', skip_missing_roi=False, normalizer=None, normalizer_source=None):

        """Creates a PDF report of slides, including images of 10 example extracted tiles.

        Args:
            stride_div (int, optional): Stride divisor for tile extraction. Defaults to 1.
            destination (str, optional): Either 'auto' or explicit filename at which to save the PDF report.
                Defaults to 'auto'.
            tma (bool, optional): Interpret slides as TMA (tumor microarrays). Defaults to False.
            enable_downsample (bool, optional): Enable downsampling during tile extraction. Defaults to False.
            roi_method (str, optional): Either 'inside', 'outside', or 'ignore'. Defaults to 'inside'.
                Determines how ROIs will guide tile extraction
            skip_missing_roi (bool, optional): Skip tiles that are missing ROIs. Defaults to False.
            normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image. Defaults to None.
                If None but using a normalizer, will use an internal tile for normalization.
                Internal default tile can be found at slideflow.util.norm_tile.jpg
        """

        from slideflow.slide import TMA, WSI, ExtractionReport

        log.info('Generating slide report...')
        reports = []
        for source in self.sources:
            roi_dir = self.sources[source]['roi']
            slide_list = self.get_slide_paths(source=source)

            # Function to extract tiles from a slide
            def get_slide_report(slide_path):
                print(f'\r\033[KGenerating report for slide {sf.util.green(sf.util.path_to_name(slide_path))}...', end='')

                if tma:
                    whole_slide = TMA(slide_path,
                                      self.tile_px,
                                      self.tile_um,
                                      stride_div,
                                      enable_downsample=enable_downsample,
                                      silent=True)
                else:
                    whole_slide = WSI(slide_path,
                                      self.tile_px,
                                      self.tile_um,
                                      stride_div,
                                      enable_downsample=enable_downsample,
                                      roi_dir=roi_dir,
                                      roi_method=roi_method,
                                      silent=True,
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

    def slide_to_label(self, headers, use_float=False, return_unique=False, verbose=True):
        """Returns dictionary mapping slide names to labels.

        Args:
            headers (str): Header column from which to read labels
            use_float (bool, optional): Interpret labels as float. Defaults to False.
            return_unique (bool, optional): Return a list of all unique labels in addition to the mapping dict.
                Defaults to False.
            verbose (bool, optional): Verbose output. Defaults to True.

        Raises:
            DatasetError: If no labels were found for the given header.

        Returns:
            dict: Dict mapping slide names to labels
        """
        labels, unique_labels = self.get_labels_from_annotations(headers=headers, use_float=use_float, verbose=verbose)
        if not use_float and not unique_labels:
            raise DatasetError(f"No labels were detected for header {headers} in this dataset")
        elif not use_float:
            return_dict = {k:unique_labels[v['label']] for k, v in labels.items()}
        else:
            return_dict = {k:labels[k]['label'] for k,v in labels.items()}
        if return_unique:
            return return_dict, unique_labels
        else:
            return return_dict

    def split_tfrecords_by_roi(self, destination):
        """Split dataset tfrecords into separate tfrecords according to ROI.

        Will generate two sets of tfrecords, with identical names: one with tiles inside the ROIs, one with tiles
        outside the ROIs. Will skip any tfrecords that are missing ROIs. Requires slides to be available.
        """

        from slideflow.slide import WSI
        import slideflow.io.tfrecords
        import tensorflow as tf

        tfrecords = self.get_tfrecords()
        slides = {sf.util.path_to_name(s):s for s in self.get_slide_paths()}
        rois = self.get_rois()
        manifest = self.get_manifest()

        for tfr in tfrecords:
            slidename = sf.util.path_to_name(tfr)
            if slidename not in slides:
                continue
            slide = WSI(slides[slidename], self.tile_px, self.tile_um, roi_list=rois, skip_missing_roi=True)
            if slide.load_error:
                continue
            feature_description, _ = sf.io.tfrecords.detect_tfrecord_format(tfr)
            parser = sf.io.tfrecords.get_tfrecord_parser(tfr, ('loc_x', 'loc_y'), to_numpy=True)
            reader = tf.data.TFRecordDataset(tfr)
            if not exists(join(destination, 'inside')):
                os.makedirs(join(destination, 'inside'))
            if not exists(join(destination, 'outside')):
                os.makedirs(join(destination, 'outside'))
            inside_roi_writer = tf.io.TFRecordWriter(join(destination, 'inside', f'{slidename}.tfrecords'))
            outside_roi_writer = tf.io.TFRecordWriter(join(destination, 'outside', f'{slidename}.tfrecords'))
            for record in tqdm(reader, total=manifest[tfr]['total']):
                loc_x, loc_y = parser(record)
                tile_in_roi = any([annPoly.contains(sg.Point(loc_x, loc_y)) for annPoly in slide.annPolys])
                record_bytes = sf.io.tfrecords._read_and_return_record(record, feature_description)
                if tile_in_roi:
                    inside_roi_writer.write(record_bytes)
                else:
                    outside_roi_writer.write(record_bytes)
            inside_roi_writer.close()
            outside_roi_writer.close()

    def tfrecord_report(self, destination, normalizer=None, normalizer_source=None):

        """Creates a PDF report of TFRecords, including 10 example tiles per TFRecord.

        Args:
            destination (str): Path to directory in which to save the PDF report
            normalizer (str, optional): Normalization strategy to use on image tiles. Defaults to None.
            normalizer_source (str, optional): Path to normalizer source image. Defaults to None.
                If None but using a normalizer, will use an internal tile for normalization.
                Internal default tile can be found at slideflow.util.norm_tile.jpg
        """

        from slideflow.slide import ExtractionReport, SlideReport
        import tensorflow as tf

        if normalizer: log.info(f'Using realtime {normalizer} normalization')
        normalizer = None if not normalizer else sf.util.StainNormalizer(method=normalizer, source=normalizer_source)

        tfrecord_list = self.get_tfrecords()
        reports = []
        log.info('Generating TFRecords report...')
        for tfr in tfrecord_list:
            print(f'\r\033[KGenerating report for tfrecord {sf.util.green(sf.util.path_to_name(tfr))}...', end='')
            dataset = tf.data.TFRecordDataset(tfr)
            parser = sf.io.tfrecords.get_tfrecord_parser(tfr, ('image_raw',), to_numpy=True, decode_images=False)
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
        filename = join(destination, f'tfrecord_report-{timestring}.pdf')
        pdf_report.save(filename)
        log.info(f'TFRecord report saved to {sf.util.green(filename)}')

    def load_annotations(self, annotations_file):
        """Load annotations from a given CSV file."""
        # Verify annotations file exists
        if not os.path.exists(annotations_file):
            raise DatasetError(f"Annotations file {sf.util.green(annotations_file)} does not exist, unable to load")

        header, current_annotations = sf.util.read_annotations(annotations_file)

        # Check for duplicate headers in annotations file
        if len(header) != len(set(header)):
            err_msg = "Annotations file containers at least one duplicate header; all headers must be unique"
            log.error(err_msg)
            raise DatasetError(err_msg)

        # Verify there is a patient header
        try:
            patient_index = header.index(TCGA.patient)
        except:
            print(header)
            err_msg = f"Check that annotations file is formatted correctly and contains header '{TCGA.patient}'."
            log.error(err_msg)
            raise DatasetError(err_msg)

        # Verify that a slide header exists; if not, offer to make one and
        # automatically associate slide names with patients
        try:
            slide_index = header.index(TCGA.slide)
        except:
            log.info(f"Header column '{TCGA.slide}' not found. Attempting to associate patients with slides...")
            self.update_annotations_with_slidenames(annotations_file)
            header, current_annotations = sf.util.read_annotations(annotations_file)
        self.annotations = current_annotations

    def verify_annotations_slides(self):
        """Verify that annotations are correctly loaded."""

        # Verify no duplicate slide names are found
        slide_list_from_annotations = self.get_slides()
        if len(slide_list_from_annotations) != len(list(set(slide_list_from_annotations))):
            log.error("Duplicate slide names detected in the annotation file.")
            raise DatasetError("Duplicate slide names detected in the annotation file.")

        # Verify all slides in the annotation column are valid
        num_warned = 0
        warn_threshold = 3
        for annotation in self.annotations:
            print_func = print if num_warned < warn_threshold else None
            slide = annotation[TCGA.slide]
            if slide == '':
                log.warning(f"Patient {sf.util.green(annotation[TCGA.patient])} has no slide assigned.")
                num_warned += 1
        if num_warned >= warn_threshold:
            log.warning(f"...{num_warned} total warnings, see project log for details")

    def update_manifest(self, force_update=False):
        """Updates tfrecord manifest.

        Args:
            forced_update (bool, optional): Force regeneration of the manifest from scratch.
        """

        # Import delayed until here in order to avoid importing tensorflow until necessary,
        # as tensorflow claims a GPU once imported
        import slideflow.io.tfrecords

        tfrecords_folders = self.get_tfrecords_folders()
        for tfr_folder in tfrecords_folders:
            slideflow.io.tfrecords.update_manifest_at_dir(directory=tfr_folder,
                                                          force_update=force_update)

    def update_annotations_with_slidenames(self, annotations_file):
        """Attempts to automatically associate slide names from a directory with patients in a given annotations file,
            skipping any slide names that are already present in the annotations file."""
        header, _ = sf.util.read_annotations(annotations_file)
        slide_list = self.get_slide_paths(filter=False)

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