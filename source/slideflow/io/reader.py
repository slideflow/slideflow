import io
import imghdr
import numpy as np
import random
import pyspng
import slideflow as sf
import dareblopy as db

from os import listdir
from os.path import isfile, join
from slideflow.util import log
from PIL import Image

BALANCE_BY_CATEGORY = 'BALANCE_BY_CATEGORY'
BALANCE_BY_PATIENT = 'BALANCE_BY_PATIENT'
NO_BALANCE = 'NO_BALANCE'

FEATURE_DESCRIPTION = {'slide':        db.FixedLenFeature([], db.string),
                       'image_raw':    db.FixedLenFeature([], db.string),
                       'loc_x':        db.FixedLenFeature([], db.int64),
                       'loc_y':        db.FixedLenFeature([], db.int64)}

class TFRecordsError(Exception):
    pass

def _get_images_by_dir(directory):
    files = [f for f in listdir(directory) if (isfile(join(directory, f))) and
                (sf.util.path_to_ext(f) in ("jpg", "png"))]
    return files

def _decode_image(img_string, img_type, standardize=False, normalizer=None, augment=False):
    tf_decoders = {
        'png': pyspng.load,
        'jpeg': lambda x: np.array(Image.open(io.BytesIO(x))),
        'jpg': lambda x: np.array(Image.open(io.BytesIO(x)))
    }
    decoder = tf_decoders[img_type.lower()]
    image = decoder(img_string)

    if normalizer:
        image = normalizer.rgb_to_rgb(image)
    if augment is True or (isinstance(augment, str) and 'j' in augment):
        # Not implemented outside of tensorflow's tf.image.adjust_jpeg_quality()
        pass
    if augment is True or (isinstance(augment, str) and 'r' in augment):
        # Rotate randomly 0, 90, 180, 270 degrees
        image = np.rot90(image, np.random.choice(range(5)))
    if augment is True or (isinstance(augment, str) and 'x' in augment):
        if np.random.rand() < 0.5:
            image = np.fliplr(image)
    if augment is True or (isinstance(augment, str) and 'y' in augment):
        if np.random.random() < 0.5:
            image = np.flipud(image)
    if standardize:
        # Not the same as tensorflow's per_image_standardization
        image = (image + 1) * (255/2)
    return image

def detect_tfrecord_format(tfr):
    it = db.ParsedTFRecordsDatasetIterator(filenames=[tfr], batch_size=1, features=FEATURE_DESCRIPTION)
    slide, img, loc_x, loc_y = next(it)
    return imghdr.what('', img[0])

def get_tfrecord_parser(tfrecord_path,
                        features_to_return=None,
                        decode_images=True,
                        standardize=False,
                        normalizer=None,
                        augment=False):

    img_type = detect_tfrecord_format(tfrecord_path)
    if features_to_return is None:
        features_to_return = list(FEATURE_DESCRIPTION.keys())

    def parser(slide, img, loc_x, loc_y):#slide, img, loc_x, loc_y):
        slide = slide[0]
        img = img[0]
        #loc_x = loc_x[0]#np.squeeze(loc_x)
        #loc_y = loc_y[0]#np.squeeze(loc_y)
        if decode_images:
            img = _decode_image(img, img_type, standardize, normalizer, augment)
        slide = slide.decode('utf-8')
        features = {
            'slide': slide,
            'image_raw': img,
            'loc_x': loc_x,
            'loc_y': loc_y
        }
        if type(features_to_return) == dict:
            return {label: features[f] for label, f in features_to_return.items()}
        else:
            return [features[f] for f in features_to_return]

    return parser

def interleave_tfrecords(tfrecords,
                         label_parser=None,
                         model_type='categorical',
                         balance=None,
                         finite=False,
                         annotations=None,  # Maps slide to outcome category directly
                         max_tiles=0,
                         min_tiles=0,
                         augment=True,
                         standardize=True,
                         normalizer=None,
                         manifest=None,
                         slides=None,
                         buffer_size=8,
                         seed=None):

    '''Generates an interleaved dataset from a collection of tfrecord files,
    sampling from tfrecord files randomly according to balancing if provided.
    Requires manifest for balancing. Assumes TFRecord files are named by slide.

    Args:
        tfrecords:      Array of paths to TFRecord files
        batch_size:     Batch size
        balance:        Whether to use balancing for batches. Options are BALANCE_BY_CATEGORY,
                            BALANCE_BY_PATIENT, and NO_BALANCE. If finite option is used, will drop
                            tiles in order to maintain proportions across the interleaved dataset.
        augment:        Whether to use data augmentation (random flip/rotate)
        finite:         Whether create finite or infinite datasets. WARNING: If finite option is
                            used with balancing, some tiles will be skipped.
        max_tiles:      Maximum number of tiles to use per slide.
        min_tiles:      Minimum number of tiles that each slide must have to be included.
    '''
    log.debug(f'Interleaving {len(tfrecords)} tfrecords: finite={finite}, max_tiles={max_tiles}, min={min_tiles}')
    datasets, datasets_categories, dataset_filenames, num_tiles = [], [], [], []
    global_num_tiles, num_tfrecords_empty, num_tfrecords_less_than_min = 0, 0, 0
    prob_weights, base_parser = None, None
    categories, categories_prob, categories_tile_fraction = {}, {}, {}
    if seed is not None:
        log.debug(f"Initializing with random seed {seed}")
        random.seed(seed)

    if label_parser is None:
        label_parser = default_label_parser

    if slides is None:
        slides = [sf.util.path_to_name(t) for t in tfrecords]

    if tfrecords == []:
        raise TFRecordsError('No TFRecords found.')

    # Get the base TFRecord parser, based on the first tfrecord
    base_parser = get_tfrecord_parser(tfrecords[0],
                                      ('slide', 'image_raw'),
                                      standardize=standardize,
                                      normalizer=normalizer,
                                      augment=augment)

    #  -------  Get Dataset Readers & Prepare Balancing -----------------------

    if manifest:
        pb = sf.util.ProgressBar(len(tfrecords), counter_text='files', leadtext='Interleaving tfrecords... ')
        for filename in tfrecords:
            slide_name = sf.util.path_to_name(filename)

            if slide_name not in slides:
                continue

            # Determine total number of tiles available in TFRecord
            try:
                tiles = manifest[filename]['total']
            except KeyError:
                log.error(f'Manifest not finished, unable to find {sf.util.green(filename)}')
                raise TFRecordsError(f'Manifest not finished, unable to find {filename}')

            # Ensure TFRecord has minimum number of tiles; otherwise, skip
            if not min_tiles and tiles == 0:
                num_tfrecords_empty += 1
                continue
            elif tiles < min_tiles:
                num_tfrecords_less_than_min += 1
                continue

            # Assign category by outcome if this is a categorical model,
            #    Merging category names if there are multiple outcomes
            #   (balancing across all combinations of outcome categories equally)
            # Otherwise, consider all slides from the same category (effectively skipping balancing).
            #   Appropriate for linear models.
            if model_type == 'categorical' and annotations is not None:
                category = annotations[slide_name]
                category = [category] if not isinstance(category, list) else category
                category = '-'.join(map(str, category))
            elif model_type == 'categorical' and balance == BALANCE_BY_CATEGORY:
                raise TFRecordsError('No annotations provided; unable to perform category-level balancing')
            else:
                category = 1

            datasets += [db.TFRecordsDatasetIterator(filenames=[filename], batch_size=1, buffer_size=buffer_size)]
            dataset_filenames += [filename]
            datasets_categories += [category]

            # Cap number of tiles to take from TFRecord at maximum specified
            if max_tiles and tiles > max_tiles:
                log.info(f'Only taking maximum of {max_tiles} (of {tiles}) tiles from {sf.util.green(filename)}')
                tiles = max_tiles

            if category not in categories.keys():
                categories.update({category: {'num_slides': 1,
                                            'num_tiles': tiles}})
            else:
                categories[category]['num_slides'] += 1
                categories[category]['num_tiles'] += tiles
            num_tiles += [tiles]
            pb.increase_bar_value()
        pb.end()

        if num_tfrecords_empty:
            log.info(f'Skipped {num_tfrecords_empty} empty tfrecords')
        if num_tfrecords_less_than_min:
            log.info(f'Skipped {num_tfrecords_less_than_min} tfrecords with less than {min_tiles} tiles')

        for category in categories:
            lowest_category_slide_count = min([categories[i]['num_slides'] for i in categories])
            lowest_category_tile_count = min([categories[i]['num_tiles'] for i in categories])
            categories_prob[category] = lowest_category_slide_count / categories[category]['num_slides']
            categories_tile_fraction[category] = lowest_category_tile_count / categories[category]['num_tiles']

        # Balancing
        if not balance or balance == NO_BALANCE:
            log.debug(f'Not balancing input')
            prob_weights = [i/sum(num_tiles) for i in num_tiles]
        if balance == BALANCE_BY_PATIENT:
            log.debug(f'Balancing input across patients')
            prob_weights = [1.0] * len(datasets)
            if finite:
                # Only take as many tiles as the number of tiles in the smallest dataset
                minimum_tiles = min(num_tiles)
                for i in range(len(datasets)):
                    num_tiles[i] = minimum_tiles
        if balance == BALANCE_BY_CATEGORY:
            log.debug(f'Balancing input across categories')
            prob_weights = [categories_prob[datasets_categories[i]] for i in range(len(datasets))]
            if finite:
                # Only take as many tiles as the number of tiles in the smallest category
                for i in range(len(datasets)):
                    num_tiles[i] = int(num_tiles[i] * categories_tile_fraction[datasets_categories[i]])
                    fraction = categories_tile_fraction[datasets_categories[i]]
                    log.debug(f'Tile fraction (dataset {i+1}/{len(datasets)}): {fraction}, taking {num_tiles[i]}')
                log.debug(f'Global num tiles: {global_num_tiles}')

        # Take the calculcated number of tiles from each dataset and calculate global number of tiles
        #for i in range(len(datasets)):
        #    datasets[i] = datasets[i].take(num_tiles[i])
        #    if not finite:
        #        datasets[i] = datasets[i].repeat()
        #global_num_tiles = sum(num_tiles)

    else:
        manifest_msg = 'No manifest detected! Unable to perform balancing or any tile-level selection operations'
        if (balance and balance != NO_BALANCE) or max_tiles or min_tiles:
            log.error(manifest_msg)
        else:
            log.warning(manifest_msg)
        pb = sf.util.ProgressBar(len(tfrecords), counter_text='files', leadtext='Interleaving tfrecords... ')
        for filename in tfrecords:
            slide_name = sf.util.path_to_name(filename)

            if slide_name not in slides:
                continue

            datasets += [db.TFRecordsDatasetIterator(filenames=[filename], batch_size=1, buffer_size=buffer_size)]
            dataset_filenames += [filename]
            pb.increase_bar_value()
        pb.end()

    #  -------  Interleave and batch datasets ---------------------------------
    if not len(datasets):
        raise TFRecordsError('No TFRecords found after filter criteria; please verify TFRecords exist')

    db_parser = db.RecordParser(FEATURE_DESCRIPTION, True)
    def process_record(record, include_slidenames=True):
        db_parsed = db_parser.parse_single_example(record)
        slide, img = base_parser(*db_parsed)
        parsed_img, label = label_parser(img, slide)
        if include_slidenames:
            return parsed_img, label, slide
        else:
            return parsed_img, label

    def interleaver(include_slidenames=True):
        while len(datasets):
            #idx = np.random.choice(range(len(datasets)), 1, p=prob_weights)[0]
            idx = random.choices(range(len(datasets)), prob_weights, k=1)[0]
            try:
                record = next(datasets[idx])[0]
                yield process_record(record, include_slidenames)
            except StopIteration:
                if finite:
                    log.debug(f"TFRecord iterator exhausted: {dataset_filenames[idx]}")
                    del datasets[idx]
                    del prob_weights[idx]
                    del dataset_filenames[idx]
                    continue
                else:
                    log.debug(f"Re-creating iterator for {dataset_filenames[idx]}")
                    datasets[idx] = db.TFRecordsDatasetIterator(filenames=[dataset_filenames[idx]], batch_size=1, buffer_size=buffer_size)

    #pool = DPool(1)
    #pool2 = DPool(1)
    #dataset = pool.imap(partial(process_record, include_slidenames=False), interleaver())
    #dataset_with_slidenames = pool2.imap(partial(process_record, include_slidenames=True), interleaver())

    dataset = interleaver(False)
    dataset_with_slidenames = interleaver(True)

    return dataset, dataset_with_slidenames, global_num_tiles

def default_label_parser(image, slide):
        '''Parses raw entry read from TFRecord.'''
        return image, None