import time
import imghdr
import numpy as np
import random
import torchvision
import torch
import slideflow as sf
from os import listdir
from tqdm import tqdm
from os.path import isfile, join, dirname, exists
from slideflow.tfrecord.torch.dataset import MultiTFRecordDataset, TFRecordDataset
from slideflow.statistics import to_onehot
from slideflow.util import log
import multiprocessing as mp
import threading
from collections import deque
from queue import Queue

FEATURE_DESCRIPTION = {'image_raw':    'byte',
                       'slide':        'byte',
                       'loc_x':        'int',
                       'loc_y':        'int'}

#IMAGE_BYTES     = b'\n\timage_raw\x12' #starts with '\n' + 3 bytes
#SLIDE_BYTES     = b'\nI\n\x05slide\x12' # sometimes has @\n>\n< before slidename starts
#LOC_X_BYTES     = b'\n\x05loc_x\x12'
#LOC_Y_BYTES     = b'\n\x05loc_y\x12'

class TFRecordsError(Exception):
    pass

class EmptyTFRecordsError(Exception):
    pass

class InterleaveIterator(torch.utils.data.IterableDataset):
    """Pytorch Iterable Dataset that interleaves tfrecords with the interleave() function below, serving as a bridge
    between the python generator returned by interleave() and the pytorch DataLoader class."""

    def __init__(self,
        tfrecords,                                  # Path to tfrecord files to interleave
        img_size,                                   # Image width in pixels
        labels                  = None,             # Dict mapping slide names to labels
        incl_slidenames         = False,            # Include slide names (returns image, label, slide)
        rank                    = 0,                # Which GPU replica this dataset is being used for
        num_replicas            = 1,                # Total number of GPU replicas
        seed                    = None,             # Tensorflow seed for random sampling
        augment                 = True,             # Slideflow augmentations to perform
        standardize             = True,             # Standardize images to mean 0 and variance of 1
        num_tiles               = None,             # Manifest mapping tfrecord names to number of total tiles
        infinite                = True,             # Inifitely loop through dataset
        max_size                = None,             # Artificially limit dataset size, useful for metrics
        prob_weights            = None,
        normalizer              = None,
        clip                    = None,
        chunk_size              = 16,               # Chunk size for interleaving
        preload                 = 8,                # Preload this many samples for parallelization
        use_labels              = True,             # Enable use of labels (disabled for non-class-conditional GANs)
        model_type              = 'categorical',    # Used to generate random labels (for StyleGAN2). Not required.
        onehot                  = False,
        **kwargs                                    # Kwargs for Dataset base class
    ):
        self.tfrecords = tfrecords
        self.img_size = img_size
        self.rank = rank
        self.num_replicas = num_replicas
        self.augment = augment
        self.standardize = standardize
        self.infinite = infinite
        self.max_size = max_size
        self.prob_weights = prob_weights
        self.clip = clip
        self.use_labels = use_labels
        self.seed = seed
        self.chunk_size = chunk_size
        self.preload = preload
        self.normalizer = normalizer
        self.onehot = onehot
        self.chunk_size = chunk_size
        self.incl_slidenames = incl_slidenames
        self.num_tiles = num_tiles
        self.model_type = model_type

        # Values for random label generation, for GAN
        if labels is not None:
            if self.onehot:
                _all_labels_raw = np.array(list(labels.values()))
                _unique_raw = np.unique(_all_labels_raw)
                max_label = np.max(_unique_raw)
                labels = {k:to_onehot(v, max_label+1) for k,v in labels.items()}

            _all_labels = np.array(list(labels.values()))
            self.unique_labels = np.unique(_all_labels, axis=0)
            self.label_prob = np.array([np.sum(_all_labels == i) for i in self.unique_labels]) / len(_all_labels)
        else:
            self.unique_labels = None
            self.label_prob = None

        self.labels = labels

    @property
    def name(self):
        return 'slideflow-test'#self._name

    @property
    def resolution(self):
        return self.img_size

    @property
    def image_shape(self):
        return (3, self.resolution, self.resolution)

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def label_shape(self):
        '''For use with StyleGAN2'''
        if self.use_labels and self.unique_labels is not None:
            return self.unique_labels[0].shape
        else:
            return 0

    @property
    def label_dim(self):
        if self.use_labels:
            assert len(self.label_shape) == 1
            return self.label_shape[0]
        else:
            return 0

    @property
    def has_labels(self):
        return self.use_labels and any(x != 0 for x in self.label_shape)

    def _parser(self, image, slide):
        if self.labels is not None:
            label = self.labels[slide]
        else:
            label = 0
        image = image.permute(2, 0, 1) # HWC => CHW
        label = torch.tensor(label)
        if self.incl_slidenames:
            return image, label, slide
        else:
            return image, label

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if not worker_info else worker_info.id
        num_workers = 1 if not worker_info else worker_info.num_workers

        preload_q = Queue(self.preload)

        dataset = interleave(self.tfrecords,
                            standardize=self.standardize,
                            augment=self.augment,
                            prob_weights=self.prob_weights,
                            clip=self.clip,
                            infinite=self.infinite,
                            normalizer=self.normalizer,
                            num_replicas=self.num_replicas * num_workers,
                            rank=self.rank + worker_id,
                            chunk_size=self.chunk_size,
                            seed=self.seed)
        def chunk_yielder(dts):
            nonlocal preload_q
            for image, slide in dts:
                preload_q.put(self._parser(image, slide))
            preload_q.put(None)

        test_thread = threading.Thread(target=chunk_yielder, args=(dataset,))
        test_thread.start()

        while True:
            record = preload_q.get()
            if record is None: return
            else:
                yield record

    def close(self):
        pass

    def get_details(self, idx):
        raise NotImplementedError

    def get_label(self, idx):
        '''#Randomly returns a label, for use with StyleGAN2
        if self.use_labels and self.model_type == 'categorical':
            label = random.choices(self.unique_labels, weights=self.label_prob, k=1)[0]
            return label.copy() if not self.onehot else to_onehot(label, self.max_label+1).copy()
        elif self.use_labels:
            return [np.random.rand()]
        else:
            return np.zeros((1,))'''
        if self.use_labels and self.model_type == 'categorical':
            return random.choices(self.unique_labels, weights=self.label_prob, k=1)[0]
        elif self.use_labels:
            return [np.random.rand()]
        else:
            return np.zeros((1,))

def _get_images_by_dir(directory):
    files = [f for f in listdir(directory) if (isfile(join(directory, f))) and
                (sf.util.path_to_ext(f) in ("jpg", "png"))]
    return files

def _decode_image(img_string, img_type, standardize=False, normalizer=None, augment=False):
    '''Decodes image. Torch implementation; different than sf.io.tensorflow'''

    np_data = torch.from_numpy(np.fromstring(img_string, np.uint8))
    image = torchvision.io.decode_image(np_data).permute(1, 2, 0) # CWH => WHC

    def random_jpeg_compression(img):
        q = random.randrange(50, 100)
        img = torchvision.io.encode_jpeg(img.permute(2, 0, 1), quality=q) # WHC => CWH
        return torchvision.io.decode_image(img).permute(1, 2, 0) # CWH => WHC

    if augment is True or (isinstance(augment, str) and 'j' in augment):
        if np.random.rand() < 0.5:
            image = random_jpeg_compression(image)
    if augment is True or (isinstance(augment, str) and 'r' in augment):
        # Rotate randomly 0, 90, 180, 270 degrees
        image = torch.rot90(image, np.random.choice(range(5)))
    if augment is True or (isinstance(augment, str) and 'x' in augment):
        if np.random.rand() < 0.5:
            image = torch.fliplr(image)
    if augment is True or (isinstance(augment, str) and 'y' in augment):
        if np.random.random() < 0.5:
            image = torch.flipud(image)
    if normalizer:
        image = torch.from_numpy(normalizer.rgb_to_rgb(image.numpy()))
    if standardize:
        # Not the same as tensorflow's per_image_standardization
        # Convert back: image = (image + 1) * (255/2)
        image = image / 127.5 - 1
    return image

def detect_tfrecord_format(tfr):
    '''Detects tfrecord format. Torch implementation; different than sf.io.tensorflow

    Returns:
        str: Image file type (png/jpeg)
        dict: Feature description dictionary (including or excluding location data as supported)
    '''

    try:
        it = iter(TFRecordDataset(tfr, None, FEATURE_DESCRIPTION, autoshard=False))
        record = next(it)
        feature_description = FEATURE_DESCRIPTION
    except KeyError:
        feature_description = {k:v for k,v in FEATURE_DESCRIPTION.items() if k in ('slide', 'image_raw')}
        try:
            it = iter(TFRecordDataset(tfr, None, feature_description, autoshard=False))
            record = next(it)
        except KeyError:
            raise TFRecordsError(f"Unable to detect TFRecord format for record: {tfr}")

    img = bytes(next(it)['image_raw'])
    img_type = imghdr.what('', img)
    return img_type, feature_description

def get_tfrecord_parser(tfrecord_path, features_to_return=None, decode_images=True, standardize=False, normalizer=None,
                        augment=False):

    '''Gets tfrecord parser using dareblopy reader. Torch implementation; different than sf.io.tensorflow

    Args:
        tfrecord_path (str): Path to tfrecord to parse.
        features_to_return (list or dict, optional): Designates format for how features should be returned from parser.
            If a list of feature names is provided, the parsing function will return tfrecord features as a list
            in the order provided. If a dictionary of labels (keys) mapping to feature names (values) is provided,
            features will be returned from the parser as a dictionary matching the same format. If None, will
            return all features as a list.
        decode_images (bool, optional): Decode raw image strings into image arrays. Defaults to True.
        standardize (bool, optional): Standardize images into the range (0,1). Defaults to False.
        normalizer (:class:`slideflow.util.StainNormalizer`): Stain normalizer to use on images. Defaults to None.
        augment (str): Image augmentations to perform. String containing characters designating augmentations.
            'x' indicates random x-flipping, 'y' y-flipping, 'r' rotating, and 'j' JPEG compression/decompression
            at random quality levels. Passing either 'xyrj' or True will use all augmentations.

    Returns:
        func: Parsing function
        dict: Detected feature description for the tfrecord
    '''
    img_type, detected_features = detect_tfrecord_format(tfrecord_path)
    if features_to_return is None:
        features_to_return = detected_features
    elif not all(f in detected_features for f in features_to_return):
        raise TFRecordsError(f'Not all designated features {",".join(list(features_to_return.keys()))} were found ' + \
                             f'in the tfrecord {",".join(list(detected_features.keys()))}')

    def parser(record):
        '''Each item in args is an array with one item, as the dareblopy reader returns items in batches
        and we have set our batch_size = 1 for interleaving.'''

        slide = bytes(record['slide']).decode('utf-8')
        img = bytes(record['image_raw'])

        features = {
            'slide': slide,
            'image_raw': _decode_image(img, img_type, standardize, normalizer, augment) if decode_images else img
        }
        if ('loc_x' in features_to_return) and ('loc_y' in features_to_return):
            features['loc_x'] = record['loc_x'][0]
            features['loc_y'] = record['loc_y'][0]
        if type(features_to_return) == dict:
            return {label: features[f] for label, f in features_to_return.items()}
        else:
            return [features[f] for f in features_to_return]

    return parser, detected_features

def interleave(tfrecords, prob_weights=None, clip=None, infinite=False, augment=True, standardize=True,
               normalizer=None, seed=None, num_threads=16, chunk_size=16, num_replicas=1, rank=0):

    """Returns a generator that interleaves records from a collection of tfrecord files, sampling from tfrecord files
    randomly according to balancing if provided (requires manifest). Assumes TFRecord files are named by slide.

    Different than tensorflow backend implementation (sf.io.tensorflow). Supports Pytorch.
    Use interleave_dataloader for the torch DataLoader class; use this function directly to get images from a generator
    with no PyTorch data processing.

    Args:
        tfrecords (list(str)): List of paths to TFRecord files.
        infinite (bool, optional): Create an finite dataset. WARNING: If infinite is False && balancing is used,
            some tiles will be skipped. Defaults to True.
        labels (dict, optional): Dict mapping slide names to outcome labels, used for balancing. Defaults to None.
        augment (str, optional): Image augmentations to perform. String containing characters designating augmentations.
                'x' indicates random x-flipping, 'y' y-flipping, 'r' rotating, and 'j' JPEG compression/decompression
                at random quality levels. Passing either 'xyrj' or True will use all augmentations.
        standardize (bool, optional): Standardize images to (0,1). Defaults to True.
        normalizer (:class:`slideflow.util.StainNormalizer`, optional): Normalizer to use on images. Defaults to None.
        manifest (dict, optional): Dataset manifest containing number of tiles per tfrecord.
        seed (int, optional): Use the following seed when randomly interleaving. Necessary for synchronized
            multiprocessing distributed reading.
        num_threads (int, optional): Number of threads to use decoding images. Defaults to 8.
        chunk_size (int, optional): Chunk size for image decoding. Defaults to 16.
        num_replicas (int, optional): Number of total workers reading the dataset with this interleave function,
            defined as number of gpus * number of torch DataLoader workers. Used to interleave results among workers
            without duplications. Defaults to 1.
        rank (int, optional): Worker ID to identify which worker this represents. Used to interleave results
            among workers without duplications. Defaults to 0 (first worker).
    """

    base_parser = None
    index_paths = {}

    if seed is not None:
        if rank == 0:
            log.debug(f"Initializing with random seed {seed}")
        random.seed(seed)
    if rank == 0:
        log.debug(f'Interleaving {len(tfrecords)} tfrecords: infinite={infinite}, num_replicas={num_replicas}')

    # -------- Get the base TFRecord parser, based on the first tfrecord ------
    img_type, _ = detect_tfrecord_format(tfrecords[0])
    base_parser, _ = get_tfrecord_parser(tfrecords[0], ('image_raw', 'slide'), decode_images=False)

    # -------- Set up TFRecord indexes for sharding ---------------------------
    # Index files not created in this interleave function, as there may be multiple instances of this function
    # running across processes, and having each create index files would result in conflicts / corruption.
    for filename in tfrecords:
        index_name = join(dirname(filename), sf.util.path_to_name(filename)+'.index')
        index_paths[sf.util.path_to_name(filename)] = index_name
        if not exists(index_name):
            raise TFRecordsError(f"Could not find index path for TFRecord {filename}")

    #  -------  Interleave and batch datasets ---------------------------------
    if prob_weights:
        assert len(prob_weights) == len(tfrecords)
        weights = {sf.util.path_to_name(tfr):v for tfr,v in prob_weights.items()}
    else:
        weights = None
    multi_loader = iter(MultiTFRecordDataset(tfrecords, index_paths, weights, shard=(rank, num_replicas), infinite=infinite))

    # Randomly interleaves datasets according to weights, reading parsed records to a buffer
    # And sending parsed results to a queue after reaching a set buffer size
    raw_q = Queue(2)
    proc_q = Queue(2)

    def interleaver():
        msg = []
        global_idx = -1
        while True:
            global_idx += 1
            try:
                record = next(multi_loader)
                #parsed =
                msg += [base_parser(record)]
                if len(msg) < chunk_size:
                    continue
                else:
                    raw_q.put(msg)
                    msg = []
            except (StopIteration):
                break

        raw_q.put(msg)
        raw_q.put(None)

    # Reads a buffer batch of images/labels, processing images using thread pools.
    def decoder():
        def threading_worker(r):
                i, s = r
                i = _decode_image(i, img_type=img_type, standardize=standardize, normalizer=normalizer, augment=augment)
                return i, s

        while True:
            records = raw_q.get()
            if records is None:
                proc_q.put(None)
                return
            pool = mp.dummy.Pool(num_threads)
            decoded_images = pool.map(threading_worker, records)
            proc_q.put(decoded_images)

    # Parallelize the tfrecord reading interleaver, and the image processing decoder
    thread = threading.Thread(target=interleaver)
    thread.start()
    procs = threading.Thread(target=decoder)
    procs.start()

    def retriever():
        total_yielded = 0
        got = 0
        while True:
            record = proc_q.get()
            got += 1
            if record is None: return
            for (img, slide) in record:
                yield img, slide
            total_yielded += len(record)

    return retriever()

def interleave_dataloader(tfrecords, img_size, batch_size, prob_weights=None, onehot=False, num_tiles=None, incl_slidenames=False,
                          infinite=False, rank=0, num_replicas=1, labels=None, normalizer=None, seed=0, chunk_size=16,
                          preload_factor=1, prefetch_factor=1, augment=True, standardize=True, num_workers=2, pin_memory=True):

    """Prepares a PyTorch DataLoader with a new InterleaveIterator instance, interleaving tfrecords and processing
    labels and tiles, with support for scaling the dataset across GPUs and dataset workers.

    Args:
        tfrecords (list(str)): List of paths to TFRecord files.
        img_size (int): Tile size in pixels.
        batch_size (int): Batch size.
        onehot (bool, optional): Onehot encode labels. Defaults to False.
        incl_slidenames (bool, optional): Include slidenames as third returned variable. Defaults to False.
        infinite (bool, optional): Infinitely repeat data. Defaults to False.
        rank (int, optional): Worker ID to identify which worker this represents. Used to interleave results
            among workers without duplications. Defaults to 0 (first worker).
        num_replicas (int, optional): Number of GPUs or unique instances which will have their own DataLoader. Used to
            interleave results among workers without duplications. Defaults to 1.
        labels (dict, optional): Dict mapping slide names to outcome labels, used for balancing. Defaults to None.
        normalizer (:class:`slideflow.util.StainNormalizer`, optional): Normalizer to use on images. Defaults to None.
        seed (int, optional): Use the following seed when randomly interleaving. Necessary for synchronized
            multiprocessing distributed reading.
        chunk_size (int, optional): Chunk size for image decoding. Defaults to 16.
        preload_factor (int, optional): Number of batches to preload in each SlideflowIterator. Defaults to 1.
        prefetch_factor (int, optional): Number of batches to prefecth in the DataLoader class. Defaults to 1.
        manifest (dict, optional): Dataset manifest containing number of tiles per tfrecord.
        balance (str, optional): Batch-level balancing. Options: category, patient, and None.
            If infinite is not True, will drop tiles in order to maintain proportions across the interleaved dataset.
        augment (str, optional): Image augmentations to perform. String containing characters designating augmentations.
                'x' indicates random x-flipping, 'y' y-flipping, 'r' rotating, and 'j' JPEG compression/decompression
                at random quality levels. Passing either 'xyrj' or True will use all augmentations.
        standardize (bool, optional): Standardize images to (0,1). Defaults to True.
        num_workers (int, optional): Number of DataLoader workers. Defaults to 2.
        pin_memory (bool, optional): Pin memory to GPU. Defaults to True.
    """

    kwargs = {var:val for var,val in locals().items() if var not in ('batch_size', 'num_workers', 'pin_memory', 'preload_factor', 'prefetch_factor')}
    torch.multiprocessing.set_sharing_strategy('file_system')
    iterator = InterleaveIterator(use_labels=(labels is not None), preload=(batch_size//num_replicas)*preload_factor, **kwargs)
    dataloader = torch.utils.data.DataLoader(iterator,
                                             batch_size=batch_size//num_replicas,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory,
                                             prefetch_factor=prefetch_factor)
    dataloader.num_tiles = iterator.num_tiles
    return dataloader