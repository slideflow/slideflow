import os
import cv2
import numpy as np
import multiprocessing as mp
import slideflow as sf
from os.path import dirname, join, exists, isfile
from typing import Optional
from functools import partial
from zarr.convenience import (_might_close, normalize_store_arg,
                              _create_group, _check_and_update_path, StoreLike,
                              BaseStore)

# --- Utility functions -------------------------------------------------------

def save_zarr_compressed(
    store: StoreLike,
    *args,
    zarr_version=None,
    path=None,
    compressor=None,
    **kwargs
) -> None:
    """Save compressed array using zarr."""
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
        num_threads = sf.util.num_cpu()

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
                     if row.getnnz()
                     else (0, 0)
                     for row in sparse_mask])


def get_sparse_centroid(mask, sparse_mask):
    n_proc = sf.util.num_cpu(default=8)
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

# --- WSI Cell Segmentation transformation utility ----------------------------

class ApplySegmentation:

    def __init__(self, source: Optional[str] = None) -> None:
        """QC function which loads a saved numpy mask to a WSI.

        Examples
            Apply saved segmentation mask to a slide.

                >>> wsi = sf.WSI(...)
                >>> segment = ApplySegmentation('.../masks.zip')
                >>> wsi.qc(segment)

            Search for masks in a folder, and apply if matching mask found.

                >>> wsi = sf.WSI(...)
                >>> segment = ApplySegmentation('.../masks_folder/')
                >>> wsi.qc(segment)

        Args:
            source (str, optional): Path to search for qc mask.
                Searches for a *.zip file matching "[slidename]-masks.zip".
                If None, will search in the same directory as the slide.
                Defaults to None.
        """
        self.source = source

    def __repr__(self):
        return "CellSegment(source={!r})".format(
            self.source
        )

    def __call__(self, wsi: "sf.WSI") -> None:
        """Applies a segmentation mask to a given slide from a saved npz file.

        Args:
            wsi (sf.WSI): Whole-slide image.

        Returns:
            None
        """
        from slideflow.cellseg import Segmentation

        # If source is not specified, look for masks in the same directory
        # as the slide.
        source = self.source if self.source is not None else dirname(wsi.path)

        if exists(source) and isfile(source):
            mask_path = source
        elif exists(join(source, wsi.name+'-masks.zip')):
            mask_path = join(source, wsi.name+'-masks.zip')
        else:
            raise sf.errors.QCError("Segmentation mask not found.")
        seg = Segmentation.load(mask_path)
        wsi.apply_segmentation(seg)
        return None