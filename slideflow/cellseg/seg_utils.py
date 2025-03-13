import os
import cv2
import numpy as np
import multiprocessing as mp
import slideflow as sf
from os.path import dirname, join, exists, isfile
from typing import Optional
from functools import partial
import zarr

# --- Utility functions -------------------------------------------------------

def save_zarr_compressed(
    store,
    *args,
    path: Optional[str] = None,
    compressor=None,
    **kwargs
) -> None:
    """
    Save one or more arrays into a new Zarr group.
    
    Parameters
    ----------
    store : Store or str
        A Zarr store or a path (string or os.PathLike) where the Zarr data will be written.
    *args : arrays
        Positional arrays to be stored. They will be saved with keys "arr_0", "arr_1", etc.
    path : str, optional
        Subpath within the store.
    compressor : numcodecs.Codec, optional
        A compressor instance to use.
    **kwargs : arrays
        Keyword arrays to store (with the key names as provided).
    
    Raises
    ------
    ValueError
        If no array is provided.
    """
    if len(args) == 0 and len(kwargs) == 0:
        raise ValueError('At least one array must be provided')
    
    # Open or create the group using the public API.
    grp = zarr.open_group(store, mode='w', path=path)
    
    # Save positional arguments as datasets with keys "arr_0", "arr_1", etc.
    for i, arr in enumerate(args):
        key = f'arr_{i}'
        grp.create_dataset(key, data=arr, overwrite=True, compressor=compressor, **kwargs)
    
    # Save keyword arguments as datasets.
    for key, arr in kwargs.items():
        grp.create_dataset(key, data=arr, overwrite=True, compressor=compressor, **kwargs)
    
    # If the underlying store has a close() method (e.g. ZipStore), call it.
    if hasattr(grp.store, 'close'):
        grp.store.close()


def fast_outlines_list(masks, num_threads: Optional[int] = None):
    """
    Get outlines of mask regions as a list for plotting.
    Uses multithreading to process large images.
    """
    if num_threads is None:
        num_threads = sf.util.num_cpu()

    def get_mask_outline(mask_id):
        mn = (masks == mask_id)
        if mn.sum() > 0:
            contours = cv2.findContours(
                mn.astype(np.uint8),
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_NONE
            )[-2]
            cmax = np.argmax([c.shape[0] for c in contours])
            pix = contours[cmax].astype(int).squeeze()
            if len(pix) > 4:
                return pix
            else:
                return np.zeros((0, 2), dtype=int)

    with mp.dummy.Pool(num_threads) as pool:
        return pool.map(get_mask_outline, np.unique(masks)[1:])


def sparse_split_indices(shape, splits):
    ar = np.arange(1, shape, int(shape / splits))
    ar[-1] = shape - 1
    return list(zip(ar[:-1], ar[1:]))


def get_sparse_chunk_centroid(sparse_mask, shape):
    return np.array([
        np.mean(np.unravel_index(row.data, shape), axis=1).astype(np.int32)
        if row.getnnz() else (0, 0)
        for row in sparse_mask
    ])


def get_sparse_centroid(mask, sparse_mask):
    n_proc = sf.util.num_cpu(default=8)
    with mp.Pool(n_proc) as pool:
        chunks = [sparse_mask[i:j] for (i, j) in sparse_split_indices(sparse_mask.shape[0], n_proc)]
        return np.concatenate(pool.map(partial(get_sparse_chunk_centroid, shape=mask.shape), chunks))


def sparse_mask(mask):
    from scipy.sparse import csr_matrix
    raveled = np.ravel(mask)
    nonzero = raveled > 0
    cols = np.where(nonzero)[0]
    return csr_matrix(
        (cols, (raveled[nonzero], cols)),
        shape=(mask.max() + 1, cols[-1] + 1),
        dtype=(np.uint32 if (cols[-1] + 1) < 4294967295 else np.uint64)
    )


# --- WSI Cell Segmentation transformation utility ----------------------------

class ApplySegmentation:
    def __init__(self, source: Optional[str] = None) -> None:
        """
        QC function which loads a saved numpy mask to a WSI.
        
        If source is not specified, searches in the same directory as the slide.
        
        Parameters
        ----------
        source : str, optional
            Path to search for the QC mask. This may be a *.zip file or a directory.
        """
        self.source = source

    def __repr__(self):
        return f"CellSegment(source={self.source!r})"

    def __call__(self, wsi: "sf.WSI") -> None:
        """
        Applies a segmentation mask to a given slide.
        
        Parameters
        ----------
        wsi : sf.WSI
            Whole-slide image.
        
        Raises
        ------
        sf.errors.QCError
            If a segmentation mask is not found.
        """
        from slideflow.cellseg import Segmentation

        # Determine the source path (if not provided, use the slide's directory)
        source = self.source if self.source is not None else dirname(wsi.path)

        if exists(source) and isfile(source):
            mask_path = source
        elif exists(join(source, wsi.name + '-masks.zip')):
            mask_path = join(source, wsi.name + '-masks.zip')
        else:
            raise sf.errors.QCError("Segmentation mask not found.")
        seg = Segmentation.load(mask_path)
        wsi.apply_segmentation(seg)
        return None
