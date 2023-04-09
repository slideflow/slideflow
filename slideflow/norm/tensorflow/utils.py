import tensorflow as tf
import tensorflow_probability as tfp
from typing import Optional

# -----------------------------------------------------------------------------

def clip_size(I: tf.Tensor, max_size: int = 2048) -> tf.Tensor:
    """Crop an image to a maximum height/width."""
    if len(I.shape) == 3:
        w, h = I.shape[0], I.shape[1]
    else:
        w, h = I.shape[1], I.shape[2]
    if w > max_size or h > max_size:
        if w > h:
            h = int((h / w) * max_size)
            w = max_size
        else:
            w = int((w / h) * max_size)
            h = max_size
        I = tf.image.resize(I, (w, h))
    return I


@tf.function
def brightness_percentile(I: tf.Tensor) -> tf.Tensor:
    """Calculate the brightness percentile for an image."""
    p = tfp.stats.percentile(I, 90)  # p = np.percentile(I, 90)
    return tf.cast(p, tf.float32)


@tf.function
def standardize_brightness(I: tf.Tensor, mask: bool = False) -> tf.Tensor:
    """Standardize image brightness to the 90th percentile.

    Args:
        I (tf.Tensor): Image, uint8.

    Returns:
        tf.Tensor: Brightness-standardized image (uint8)
    """
    if mask:
        ones = tf.math.reduce_all(I == 255, axis=len(I.shape)-1)
    bI = I if not mask else I[~ ones]
    p = brightness_percentile(bI)
    scaled = tf.cast(I, tf.float32) * tf.constant(255.0, dtype=tf.float32) / p
    scaled = tf.experimental.numpy.clip(scaled, 0, 255)
    scaled = tf.cast(scaled, tf.uint8)
    if mask:
        scaled = tf.where(
            tf.repeat(tf.expand_dims(ones, axis=-1), 3, axis=-1),
            tf.ones_like(scaled, dtype=tf.uint8) * 255,
            scaled
        )
    return scaled