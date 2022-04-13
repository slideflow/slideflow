'''From https://github.com/tensorflow/probability/blob/v0.16.0/tensorflow_probability/python/stats/quantiles.py#L428-L642'''

import numpy as np
import tensorflow as tf

assert_rank_in = tf.debugging.assert_rank_in
assert_greater_equal = tf.debugging.assert_greater_equal
assert_less_equal = tf.debugging.assert_less_equal
assert_integer = tf.debugging.assert_integer


def as_numpy_dtype(dtype):
  """Returns a `np.dtype` based on this `dtype`."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'as_numpy_dtype'):
    return dtype.as_numpy_dtype
  return dtype


def is_integer(dtype):
  """Returns whether this is a (non-quantized) integer type."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'is_integer') and not callable(dtype.is_integer):
    return dtype.is_integer
  return np.issubdtype(np.dtype(dtype), np.integer)


def rotate_transpose(x, shift, name='rotate_transpose'):
  with tf.name_scope(name):
    x = tf.convert_to_tensor(x, name='x')
    shift = tf.convert_to_tensor(shift, name='shift')
    shift_value_static = tf.get_static_value(shift)
    ndims = tf.TensorShape(x.shape).rank
    if ndims is not None and shift_value_static is not None:
      if ndims < 2:
        return x
      shift_value_static = np.sign(shift_value_static) * (
          abs(shift_value_static) % ndims)
      if shift_value_static == 0:
        return x
      perm = np.roll(np.arange(ndims), shift_value_static)
      return tf.transpose(a=x, perm=perm)
    else:
      ndims = tf.rank(x)
      shift = tf.where(
          tf.less(shift, 0), -shift % ndims, ndims - shift % ndims)
      first = tf.range(0, shift)
      last = tf.range(shift, ndims)
      perm = tf.concat([last, first], 0)
      return tf.transpose(a=x, perm=perm)


def with_dependencies(dependencies, output_tensor, name=None):
  if tf.executing_eagerly():
    return output_tensor
  with tf.name_scope(name or 'control_dependency') as name:
    with tf.control_dependencies(d for d in dependencies if d is not None):
      output_tensor = tf.convert_to_tensor(output_tensor)
      if isinstance(output_tensor, tf.Tensor):
        return tf.identity(output_tensor, name=name)
      else:
        return tf.IndexedSlices(
            tf.identity(output_tensor.values, name=name),
            output_tensor.indices,
            output_tensor.dense_shape)


def _get_best_effort_ndims(x,
                           expect_ndims=None,
                           expect_ndims_at_least=None,
                           expect_ndims_no_more_than=None):
  """Get static ndims if possible.  Fallback on `tf.rank(x)`."""
  ndims_static = _get_static_ndims(
      x,
      expect_ndims=expect_ndims,
      expect_ndims_at_least=expect_ndims_at_least,
      expect_ndims_no_more_than=expect_ndims_no_more_than)
  if ndims_static is not None:
    return ndims_static
  return tf.rank(x)


def _insert_back_keepdims(x, axis):
  for i in sorted(axis):
    x = tf.expand_dims(x, axis=i)
  return x


def _move_dims_to_flat_end(x, axis, x_ndims, right_end=True):

  if not axis:
    return x

  other_dims = sorted(set(range(x_ndims)).difference(axis))
  perm = other_dims + list(axis) if right_end else list(axis) + other_dims
  x_permed = tf.transpose(a=x, perm=perm)

  if tf.TensorShape(x.shape).is_fully_defined():
    x_shape = tf.TensorShape(x.shape).as_list()
    other_shape = [x_shape[i] for i in other_dims]
    end_shape = [np.prod([x_shape[i] for i in axis])]
    full_shape = (
        other_shape + end_shape if right_end else end_shape + other_shape)
  else:
    other_shape = tf.gather(tf.shape(x), tf.cast(other_dims, tf.int64))
    full_shape = tf.concat(
        [other_shape, [-1]] if right_end else [[-1], other_shape], axis=0)
  return tf.reshape(x_permed, shape=full_shape)


def _make_static_axis_non_negative_list(axis, ndims):
  axis = tf.non_negative_axis(axis, ndims)

  axis_const = tf.get_static_value(axis)
  if axis_const is None:
    raise ValueError(
        'Expected argument `axis` to be statically available. '
        'Found: {}.'.format(axis))

  # Make at least 1-D.
  axis = axis_const + np.zeros([1], dtype=axis_const.dtype)

  return list(int(dim) for dim in axis)


def _get_static_ndims(x,
                      expect_static=False,
                      expect_ndims=None,
                      expect_ndims_no_more_than=None,
                      expect_ndims_at_least=None):
  ndims = tf.TensorShape(x.shape).rank

  if ndims is None:
    if expect_static:
      raise ValueError(
          'Expected argument `x` to have statically defined `ndims`. '
          'Found: {}.'.format(x))
    return

  if expect_ndims is not None:
    ndims_message = (
        'Expected argument `x` to have ndims {}. Found tensor {}.'.format(
            expect_ndims, x))
    if ndims != expect_ndims:
      raise ValueError(ndims_message)

  if expect_ndims_at_least is not None:
    ndims_at_least_message = (
        'Expected argument `x` to have ndims >= {}. Found tensor {}.'.format(
            expect_ndims_at_least, x))
    if ndims < expect_ndims_at_least:
      raise ValueError(ndims_at_least_message)

  if expect_ndims_no_more_than is not None:
    ndims_no_more_than_message = (
        'Expected argument `x` to have ndims <= {}. Found tensor {}.'.format(
            expect_ndims_no_more_than, x))
    if ndims > expect_ndims_no_more_than:
      raise ValueError(ndims_no_more_than_message)

  return ndims


def percentile(x,
               q,
               axis=None,
               interpolation=None,
               keepdims=False,
               validate_args=False,
               preserve_gradients=True,
               name=None):
  name = name or 'percentile'
  allowed_interpolations = {'linear', 'lower', 'higher', 'nearest', 'midpoint'}

  if interpolation is None:
    interpolation = 'nearest'
  else:
    if interpolation not in allowed_interpolations:
      raise ValueError(
          'Argument `interpolation` must be in {}. Found {}.'.format(
              allowed_interpolations, interpolation))

  with tf.name_scope(name):
    x = tf.convert_to_tensor(x, name='x')

    if (interpolation in {'linear', 'midpoint'} and
        is_integer(x.dtype)):
      raise TypeError('{} interpolation not allowed with dtype {}'.format(
          interpolation, x.dtype))

    # Double is needed here and below, else we get the wrong index if the array
    # is huge along axis.
    q = tf.cast(q, tf.float64)
    _get_static_ndims(q, expect_ndims_no_more_than=1)

    if validate_args:
      q = with_dependencies([
          assert_rank_in(q, [0, 1]),
          assert_greater_equal(q, tf.cast(0., tf.float64)),
          assert_less_equal(q, tf.cast(100., tf.float64))
      ], q)

    # Move `axis` dims of `x` to the rightmost, call it `y`.
    if axis is None:
      y = tf.reshape(x, [-1])
    else:
      x_ndims = _get_static_ndims(
          x, expect_static=True, expect_ndims_at_least=1)
      axis = _make_static_axis_non_negative_list(axis, x_ndims)
      y = _move_dims_to_flat_end(x, axis, x_ndims, right_end=True)

    frac_at_q_or_below = q / 100.

    sorted_y = tf.sort(y, axis=-1, direction='ASCENDING')

    d = tf.cast(tf.shape(y)[-1], tf.float64)

    def _get_indices(interp_type):
      """Get values of y at the indices implied by interp_type."""
      if interp_type == 'lower':
        indices = tf.math.floor((d - 1) * frac_at_q_or_below)
      elif interp_type == 'higher':
        indices = tf.math.ceil((d - 1) * frac_at_q_or_below)
      elif interp_type == 'nearest':
        indices = tf.round((d - 1) * frac_at_q_or_below)
      return tf.clip_by_value(
          tf.cast(indices, tf.int32), 0,
          tf.shape(y)[-1] - 1)

    if interpolation in ['nearest', 'lower', 'higher']:
      gathered_y = tf.gather(sorted_y, _get_indices(interpolation), axis=-1)
    elif interpolation == 'midpoint':
      gathered_y = 0.5 * (
          tf.gather(sorted_y, _get_indices('lower'), axis=-1) +
          tf.gather(sorted_y, _get_indices('higher'), axis=-1))
    elif interpolation == 'linear':
      larger_y_idx = _get_indices('higher')
      exact_idx = (d - 1) * frac_at_q_or_below
      if preserve_gradients:
        smaller_y_idx = tf.maximum(larger_y_idx - 1, 0)
        larger_y_idx = tf.minimum(smaller_y_idx + 1, tf.shape(y)[-1] - 1)
        fraction = tf.cast(larger_y_idx, tf.float64) - exact_idx
      else:
        smaller_y_idx = _get_indices('lower')
        fraction = tf.math.ceil((d - 1) * frac_at_q_or_below) - exact_idx

      fraction = tf.cast(fraction, y.dtype)
      gathered_y = (
          tf.gather(sorted_y, larger_y_idx, axis=-1) * (1 - fraction) +
          tf.gather(sorted_y, smaller_y_idx, axis=-1) * fraction)

    # Propagate NaNs
    if x.dtype in (tf.bfloat16, tf.float16, tf.float32, tf.float64):
      # Apparently tf.is_nan doesn't like other dtypes
      nan_batch_members = tf.reduce_any(tf.math.is_nan(x), axis=axis)
      right_rank_matched_shape = tf.pad(
          tf.shape(nan_batch_members),
          paddings=[[0, tf.rank(q)]],
          constant_values=1)
      nan_batch_members = tf.reshape(
          nan_batch_members, shape=right_rank_matched_shape)
      nan = np.array(np.nan, as_numpy_dtype(gathered_y.dtype))
      gathered_y = tf.where(nan_batch_members, nan, gathered_y)

    # Expand dimensions if requested
    if keepdims:
      if axis is None:
        ones_vec = tf.ones(
            shape=[_get_best_effort_ndims(x) + _get_best_effort_ndims(q)],
            dtype=tf.int32)
        gathered_y *= tf.ones(ones_vec, dtype=x.dtype)
      else:
        gathered_y = _insert_back_keepdims(gathered_y, axis)
    return rotate_transpose(gathered_y, tf.rank(q))