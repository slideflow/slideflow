import numpy as np

# -----------------------------------------------------------------------------

def center_square_pad(img, size, padval=0):
    if img.shape[0] < size:
        pad_x = int(np.round((size - img.shape[0]) / 2))+1
    else:
        pad_x = 0
    if img.shape[1] < size:
        pad_y = int(np.round((size - img.shape[1]) / 2))+1
    else:
        pad_y = 0
    return np.pad(img, ((pad_x, pad_x), (pad_y, pad_y), (0, 0)), mode='constant', constant_values=padval)

def topleft_pad(img, size, padval=0):
    if img.shape[0] < size:
        pad_x = (size - img.shape[0])
    else:
        pad_x = 0
    if img.shape[1] < size:
        pad_y = (size - img.shape[1])
    else:
        pad_y = 0
    return np.pad(img, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant', constant_values=padval)
