'''From https://github.com/tensorflow/addons/blob/v0.15.0/tensorflow_addons/image/filters.py#L218-L302'''

import tensorflow as tf


def opt_kernel(sigma: float) -> int:
    """Calculate an appropriate Gaussian kernel size for a given sigma."""
    return int((sigma * 4) + 1)


def auto_gaussian(image: tf.Tensor, sigma: float) -> tf.Tensor:
    """Perform Gaussian blur on an image with a given sigma, automatically
    calculating the appropriate Gaussian kernel size.

    Args:
        image (tf.Tensor): Image or batch of images.
        sigma (float): Gaussian sigma.

    Returns:
        tf.Tensor: Image(s) with Gaussian blur applied.
    """
    return gaussian_filter2d(image, opt_kernel(sigma), sigma=sigma)


def get_ndims(image):
    """Get dimensions from an image."""
    return image.get_shape().ndims or tf.rank(image)


def to_4D_image(image):
    """Convert 2/3/4D image to 4D image.
    Args:
      image: 2/3/4D `Tensor`.
    Returns:
      4D `Tensor` with the same type.
    """
    with tf.control_dependencies(
        [
            tf.debugging.assert_rank_in(
                image, [2, 3, 4], message="`image` must be 2/3/4D tensor"
            )
        ]
    ):
        ndims = image.get_shape().ndims
        if ndims is None:
            return _dynamic_to_4D_image(image)
        elif ndims == 2:
            return image[None, :, :, None]
        elif ndims == 3:
            return image[None, :, :, :]
        else:
            return image


def normalize_tuple(value, n, name):
    """Transforms an integer or iterable of integers into an integer tuple.
    A copy of tensorflow.python.keras.util.
    Args:
      value: The value to validate and convert. Could an int, or any iterable
        of ints.
      n: The size of the tuple to be returned.
      name: The name of the argument being validated, e.g. "strides" or
        "kernel_size". This is only used to format error messages.
    Returns:
      A tuple of n integers.
    Raises:
      ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise TypeError(
                "The `"
                + name
                + "` argument must be a tuple of "
                + str(n)
                + " integers. Received: "
                + str(value)
            )
        if len(value_tuple) != n:
            raise ValueError(
                "The `"
                + name
                + "` argument must be a tuple of "
                + str(n)
                + " integers. Received: "
                + str(value)
            )
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError(
                    "The `"
                    + name
                    + "` argument must be a tuple of "
                    + str(n)
                    + " integers. Received: "
                    + str(value)
                    + " "
                    "including element "
                    + str(single_value)
                    + " of type"
                    + " "
                    + str(type(single_value))
                )
        return value_tuple


def _dynamic_to_4D_image(image):
    shape = tf.shape(image)
    original_rank = tf.rank(image)
    # 4D image => [N, H, W, C] or [N, C, H, W]
    # 3D image => [1, H, W, C] or [1, C, H, W]
    # 2D image => [1, H, W, 1]
    left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = tf.concat(
        [
            tf.ones(shape=left_pad, dtype=tf.int32),
            shape,
            tf.ones(shape=right_pad, dtype=tf.int32),
        ],
        axis=0,
    )
    return tf.reshape(image, new_shape)


def from_4D_image(image, ndims):
    """Convert back to an image with `ndims` rank.
    Args:
      image: 4D `Tensor`.
      ndims: The original rank of the image.
    Returns:
      `ndims`-D `Tensor` with the same type.
    """
    with tf.control_dependencies(
        [tf.debugging.assert_rank(image, 4, message="`image` must be 4D tensor")]
    ):
        if isinstance(ndims, tf.Tensor):
            return _dynamic_from_4D_image(image, ndims)
        elif ndims == 2:
            return tf.squeeze(image, [0, 3])
        elif ndims == 3:
            return tf.squeeze(image, [0])
        else:
            return image


def _dynamic_from_4D_image(image, original_rank):
    shape = tf.shape(image)
    # 4D image <= [N, H, W, C] or [N, C, H, W]
    # 3D image <= [1, H, W, C] or [1, C, H, W]
    # 2D image <= [1, H, W, 1]
    begin = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    end = 4 - tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = shape[begin:end]
    return tf.reshape(image, new_shape)


def _get_gaussian_kernel(sigma, filter_shape):
    """Compute 1D Gaussian kernel."""
    sigma = tf.convert_to_tensor(sigma)
    x = tf.range(-filter_shape // 2 + 1, filter_shape // 2 + 1)
    x = tf.cast(x ** 2, sigma.dtype)
    x = tf.nn.softmax(-x / (2.0 * (sigma ** 2)))
    return x


def _get_gaussian_kernel_2d(gaussian_filter_x, gaussian_filter_y):
    """Compute 2D Gaussian kernel given 1D kernels."""
    gaussian_kernel = tf.matmul(gaussian_filter_x, gaussian_filter_y)
    return gaussian_kernel


def _pad(image, filter_shape, mode="CONSTANT", constant_values=0):

    """Explicitly pad a 4-D image.
    Equivalent to the implicit padding method offered in `tf.nn.conv2d` and
    `tf.nn.depthwise_conv2d`, but supports non-zero, reflect and symmetric
    padding mode. For the even-sized filter, it pads one more value to the
    right or the bottom side.
    Args:
      image: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      filter_shape: A `tuple`/`list` of 2 integers, specifying the height
        and width of the 2-D filter.
      mode: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC".
        The type of padding algorithm to use, which is compatible with
        `mode` argument in `tf.pad`. For more details, please refer to
        https://www.tensorflow.org/api_docs/python/tf/pad.
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode.
    """
    if mode.upper() not in {"REFLECT", "CONSTANT", "SYMMETRIC"}:
        raise ValueError(
            'padding should be one of "REFLECT", "CONSTANT", or "SYMMETRIC".'
        )
    constant_values = tf.convert_to_tensor(constant_values, image.dtype)
    filter_height, filter_width = filter_shape
    pad_top = (filter_height - 1) // 2
    pad_bottom = filter_height - 1 - pad_top
    pad_left = (filter_width - 1) // 2
    pad_right = filter_width - 1 - pad_left
    paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    return tf.pad(image, paddings, mode=mode, constant_values=constant_values)


@tf.function
def gaussian_filter2d(image, filter_shape=(3, 3), sigma=1.0, padding="REFLECT", constant_values=0, name=None):

    """Perform Gaussian blur on image(s).
    Args:
      image: Either a 2-D `Tensor` of shape `[height, width]`,
        a 3-D `Tensor` of shape `[height, width, channels]`,
        or a 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      filter_shape: An `integer` or `tuple`/`list` of 2 integers, specifying
        the height and width of the 2-D gaussian filter. Can be a single
        integer to specify the same value for all spatial dimensions.
      sigma: A `float` or `tuple`/`list` of 2 floats, specifying
        the standard deviation in x and y direction the 2-D gaussian filter.
        Can be a single float to specify the same value for all spatial
        dimensions.
      padding: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC".
        The type of padding algorithm to use, which is compatible with
        `mode` argument in `tf.pad`. For more details, please refer to
        https://www.tensorflow.org/api_docs/python/tf/pad.
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode.
      name: A name for this operation (optional).
    Returns:
      2-D, 3-D or 4-D `Tensor` of the same dtype as input.
    Raises:
      ValueError: If `image` is not 2, 3 or 4-dimensional,
        if `padding` is other than "REFLECT", "CONSTANT" or "SYMMETRIC",
        if `filter_shape` is invalid,
        or if `sigma` is invalid.
    """
    with tf.name_scope(name or "gaussian_filter2d"):
        if isinstance(sigma, (list, tuple)):
            if len(sigma) != 2:
                raise ValueError("sigma should be a float or a tuple/list of 2 floats")
        else:
            sigma = (sigma,) * 2

        if not isinstance(sigma[0], (tf.Tensor)) and any(s < 0 for s in sigma):
            raise ValueError("sigma should be greater than or equal to 0.")

        image = tf.convert_to_tensor(image, name="image")
        sigma = tf.convert_to_tensor(sigma, name="sigma")

        original_ndims = get_ndims(image)
        image = to_4D_image(image)

        # Keep the precision if it's float;
        # otherwise, convert to float32 for computing.
        orig_dtype = image.dtype
        if not image.dtype.is_floating:
            image = tf.cast(image, tf.float32)

        channels = tf.shape(image)[3]
        filter_shape = normalize_tuple(filter_shape, 2, "filter_shape")

        sigma = tf.cast(sigma, image.dtype)
        gaussian_kernel_x = _get_gaussian_kernel(sigma[1], filter_shape[1])
        gaussian_kernel_x = gaussian_kernel_x[tf.newaxis, :]

        gaussian_kernel_y = _get_gaussian_kernel(sigma[0], filter_shape[0])
        gaussian_kernel_y = gaussian_kernel_y[:, tf.newaxis]

        gaussian_kernel_2d = _get_gaussian_kernel_2d(
            gaussian_kernel_y, gaussian_kernel_x
        )
        gaussian_kernel_2d = gaussian_kernel_2d[:, :, tf.newaxis, tf.newaxis]
        gaussian_kernel_2d = tf.tile(gaussian_kernel_2d, [1, 1, channels, 1])

        image = _pad(image, filter_shape, mode=padding, constant_values=constant_values)

        output = tf.nn.depthwise_conv2d(
            input=image,
            filter=gaussian_kernel_2d,
            strides=(1, 1, 1, 1),
            padding="VALID",
        )
        output = from_4D_image(output, original_ndims)
        return tf.cast(output, orig_dtype)
