import os
import cv2
import multiprocessing as mp
import numpy as np
from functools import partial
from zarr.convenience import (_might_close, normalize_store_arg,
                              _create_group, _check_and_update_path, StoreLike,
                              BaseStore)

def save_zarr_compressed(store: StoreLike, *args, zarr_version=None, path=None, compressor=None, **kwargs):
    if len(args) == 0 and len(kwargs) == 0:
        raise ValueError('at least one array must be provided')
    # handle polymorphic store arg
    may_need_closing = _might_close(store)
    _store: BaseStore = normalize_store_arg(store, mode="w", zarr_version=zarr_version)
    path = _check_and_update_path(_store, path)
    try:
        grp = _create_group(_store, path=path, overwrite=True, zarr_version=zarr_version)
        for i, arr in enumerate(args):
            k = 'arr_{}'.format(i)
            grp.create_dataset(k, data=arr, overwrite=True, zarr_version=zarr_version, compressor=compressor)
        for k, arr in kwargs.items():
            grp.create_dataset(k, data=arr, overwrite=True, zarr_version=zarr_version, compressor=compressor)
    finally:
        if may_need_closing:
            # needed to ensure zip file records are written
            _store.close()

def fast_outlines_list(masks, num_threads=None):
    """Get outlines of masks as a list to loop over for plotting. Accelerated
    by multithreading for large images.
    """
    if num_threads is None:
        num_threads = os.cpu_count()

    def get_mask_outline(mask_id):
        mn = (masks == mask_id)
        if mn.sum() > 0:
            contours = cv2.findContours(
                mn.astype(np.uint8),
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_NONE)
            contours = contours[-2]
            cmax = np.argmax([c.shape[0] for c in contours])
            pix = contours[cmax].astype(int).squeeze()
            if len(pix) > 4:
                return pix
            else:
                return np.zeros((0,2))

    with mp.dummy.Pool(num_threads) as pool:
        return pool.map(get_mask_outline, np.unique(masks)[1:])


def sparse_split_indices(shape, splits):
    ar = np.arange(1, shape, int(shape/splits))
    ar[-1] = shape-1
    return list(zip(ar[0:-1], ar[1:]))


def get_sparse_chunk_centroid(sparse_mask, shape):
    return np.array([np.mean(np.unravel_index(row.data, shape), 1).astype(np.int32)
                     for row in sparse_mask])


def get_sparse_centroid(mask, sparse_mask):
    n_proc = os.cpu_count() if os.cpu_count() else 8
    with mp.Pool(n_proc) as pool:
        return np.concatenate(
            pool.map(
                partial(get_sparse_chunk_centroid, shape=mask.shape),
                [sparse_mask[i:j] for (i, j) in sparse_split_indices(sparse_mask.shape[0], n_proc)]
            ))

def sparse_mask(mask):
    from scipy.sparse import csr_matrix
    raveled = np.ravel(mask)
    nonzero = raveled > 0
    cols = np.where(nonzero)[0]
    return csr_matrix((cols, (raveled[nonzero], cols)),
                      shape=(mask.max() + 1, cols[-1]+1),
                      dtype=(np.uint32 if (cols[-1]+1) < 4294967295 else np.uint64))
