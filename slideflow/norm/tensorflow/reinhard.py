"""
Reinhard normalization based on method of:

E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.
"""

from __future__ import division

import tensorflow as tf
import tensorflow_probability as tfp

from slideflow.norm.tensorflow.reinhard_fast import transform as transform_fast
from slideflow.norm.tensorflow.reinhard_fast import fit as fit_fast

@tf.function
def standardize_brightness(I):
    """

    :param I:
    :return:
    """
    p = tfp.stats.percentile(I, 90) #p = np.percentile(I, 90)
    p = tf.cast(p, tf.float32)
    return tf.cast(tf.experimental.numpy.clip(tf.cast(I, tf.float32) * tf.constant(255.0, dtype=tf.float32) / p, 0, 255), tf.uint8) #np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)

@tf.function
def transform(I, tgt_mean, tgt_std):
    I = standardize_brightness(I)
    return transform_fast(I, tgt_mean, tgt_std)

@tf.function
def fit(target, reduce=False):
    target = standardize_brightness(target)
    return fit_fast(target, reduce=reduce)
