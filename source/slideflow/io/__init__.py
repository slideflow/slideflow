"""TFRecord reading/writing utilities."""

import os
import copy
import slideflow as sf

from slideflow.util import log
from os.path import join, exists

# Backend-specific imports and configuration
if os.environ['SF_BACKEND'] == 'tensorflow':
    import tensorflow as tf
    from slideflow.io.tensorflow import get_tfrecord_parser, TFRecordsError
    from tensorflow.data import TFRecordDataset
    dataloss_errors = (tf.errors.DataLossError, TFRecordsError)
    parser_kwargs = {'to_numpy': True}

elif os.environ['SF_BACKEND'] == 'torch':
    from slideflow.io.torch import get_tfrecord_parser, TFRecordsError
    from slideflow.tfrecord.torch.dataset import TFRecordDataset
    dataloss_errors = (TFRecordsError,)
    parser_kwargs = {}

else:
    raise ValueError(f"Unknown backend {os.environ['SF_BACKEND']}")

# -----------------------------------------------------------------------------

def update_manifest_at_dir(directory, force_update=False):
    '''Log number of tiles in each TFRecord file present in the given directory and all subdirectories,
    saving manifest to file within the parent directory.'''

    manifest_path = join(directory, "manifest.json")
    manifest = {} if not exists(manifest_path) else sf.util.load_json(manifest_path)
    prior_manifest = copy.deepcopy(manifest)
    try:
        relative_tfrecord_paths = sf.util.get_relative_tfrecord_paths(directory)
    except FileNotFoundError:
        log.debug(f"Unable to update manifest at {directory}; TFRecords not found")
        return

    # Verify all tfrecords in manifest exist
    for rel_tfr in prior_manifest.keys():
        tfr = join(directory, rel_tfr)
        if not exists(tfr):
            log.warning(f"TFRecord in manifest was not found at {tfr}; removing")
            del(manifest[rel_tfr])

    for rel_tfr in relative_tfrecord_paths:
        tfr = join(directory, rel_tfr)

        if (not force_update) and (rel_tfr in manifest) and ('total' in manifest[rel_tfr]):
            continue

        manifest.update({rel_tfr: {}})
        try:
            raw_dataset = TFRecordDataset(tfr, None, None)
            parser = get_tfrecord_parser(tfr, ('slide',), **parser_kwargs)
        except Exception as e:
            log.error(f"Unable to open TFRecords file with {os.environ['SF_BACKEND']}: {str(e)}")
            return
        if log.getEffectiveLevel() <= 20: print(f"\r\033[K + Verifying tiles in {sf.util.green(rel_tfr)}...", end="")
        total = 0
        try:
            for raw_record in raw_dataset:
                slide = parser(raw_record)[0]
                if hasattr(slide, 'decode'):
                    slide = slide.decode('utf-8')
                if slide not in manifest[rel_tfr]:
                    manifest[rel_tfr][slide] = 1
                else:
                    manifest[rel_tfr][slide] += 1
                total += 1
        except dataloss_errors:
            print('\r\033[K', end="")
            log.error(f"Corrupt or incomplete TFRecord at {tfr}")
            log.info(f"Deleting and removing corrupt TFRecord from manifest...")
            del(raw_dataset)
            os.remove(tfr)
            del(manifest[rel_tfr])
            continue
        manifest[rel_tfr]['total'] = total
        print('\r\033[K', end="")

    # Write manifest file
    if (manifest != prior_manifest) or (manifest == {}):
        sf.util.write_json(manifest, manifest_path)

    return manifest