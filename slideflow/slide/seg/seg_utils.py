import os
import cv2
import multiprocessing as mp
import numpy as np


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


def get_sparse_centroid(mask, sparse_mask):
    return np.array([np.mean(np.unravel_index(row.data, mask.shape), 1).astype(np.int32)
                     for (R, row) in enumerate(sparse_mask) if R>0])


def sparse_mask(mask):
    from scipy.sparse import csr_matrix
    cols = np.arange(mask.size)
    return csr_matrix((cols, (np.ravel(mask), cols)),
                      shape=(mask.max() + 1, mask.size),
                      dtype=np.int64)

