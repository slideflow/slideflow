"""Tensorflow model utility functions."""

import tensorflow as tf
import os
import tempfile
from typing import Tuple, Dict, List, Union, Any, TYPE_CHECKING
from slideflow.util import log

if TYPE_CHECKING:
    import neptune.new as neptune


def log_summary(
    model: tf.keras.Model,
    neptune_run: "neptune.Run" = None
) -> None:
    """Log the model summary.

    Args:
        model (tf.keras.Model): Tensorflow/Keras model.
        neptune_run (neptune.Run, optional): Neptune run. Defaults to None.
    """
    if log.getEffectiveLevel() <= 20:
        print()
        model.summary()
    if neptune_run:
        summary_string = []
        model.summary(print_fn=lambda x: summary_string.append(x))
        neptune_run['summary'] = "\n".join(summary_string)


def get_layer_index_by_name(model: tf.keras.Model, name: str) -> int:
    for i, layer in enumerate(model.layers):
        if layer.name == name:
            return i
    raise IndexError(f"Layer {name} not found.")


def batch_loss_crossentropy(
    features: tf.Tensor,
    diff: float = 0.5,
    eps: float = 1e-5
) -> tf.Tensor:
    split = tf.split(features, 8, axis=0)

    def tstat(first, rest):
        first_mean = tf.math.reduce_mean(first, axis=0)
        rest_mean = tf.math.reduce_mean(rest, axis=0)

        # Variance
        A = tf.math.reduce_sum(tf.math.square(first - first_mean), axis=0) / (first_mean.shape[0] - 1)
        B = tf.math.reduce_sum(tf.math.square(rest - rest_mean), axis=0) / (rest_mean.shape[0] - 1)

        # Not performing square root of SE for computational reasons
        se = tf.math.sqrt((A / first_mean.shape[0]) + (B / rest_mean.shape[0]))
        t_square = tf.math.square((first_mean - rest_mean - diff) / se)
        return tf.math.reduce_mean(t_square)

    stats = [
        tstat(
            split[n],
            tf.concat([
                sp for i, sp in enumerate(split)
                if i != n
            ], axis=0))
        for n in range(len(split))
    ]
    return tf.math.reduce_mean(tf.stack(stats)) * eps


def negative_log_likelihood(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Negative log likelihood loss.

    Implemented by Fred Howard, adapted from
    https://github.com/havakv/pycox/blob/master/pycox/models/loss.py

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predictions.

    Returns:
        tf.Tensor: Loss.
    """
    events = tf.reshape(y_pred[:, -1], [-1])  # E
    pred_hr = tf.reshape(y_pred[:, 0], [-1])  # y_pred
    time = tf.reshape(y_true, [-1])           # y_true

    order = tf.argsort(time)  # direction='DESCENDING'
    sorted_events = tf.gather(events, order)            # pylint: disable=no-value-for-parameter
    sorted_predictions = tf.gather(pred_hr, order)      # pylint: disable=no-value-for-parameter

    # Finds maximum HR in predictions
    gamma = tf.math.reduce_max(sorted_predictions)

    # Small constant value
    eps = tf.constant(1e-7, dtype=tf.float32)

    log_cumsum_h = tf.math.add(
                    tf.math.log(
                        tf.math.add(
                            tf.math.cumsum(             # pylint: disable=no-value-for-parameter
                                tf.math.exp(
                                    tf.math.subtract(sorted_predictions, gamma))),
                            eps)),
                    gamma)

    neg_likelihood = -tf.math.divide(
                        tf.reduce_sum(
                            tf.math.multiply(
                                tf.subtract(sorted_predictions, log_cumsum_h),
                                sorted_events)),
                        tf.reduce_sum(sorted_events))

    return neg_likelihood


def negative_log_likelihood_breslow(
    y_true: tf.Tensor,
    y_pred: tf.Tensor
) -> tf.Tensor:
    """Negative log likelihood loss, Breslow approximation.

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predictions.

    Returns:
        tf.Tensor: Breslow loss.
    """
    events = tf.reshape(y_pred[:, -1], [-1])
    pred = tf.reshape(y_pred[:, 0], [-1])
    time = tf.reshape(y_true, [-1])

    order = tf.argsort(time, direction='DESCENDING')
    sorted_time = tf.gather(time, order)                # pylint: disable=no-value-for-parameter
    sorted_events = tf.gather(events, order)            # pylint: disable=no-value-for-parameter
    sorted_pred = tf.gather(pred, order)                # pylint: disable=no-value-for-parameter

    Y_hat_c = sorted_pred
    Y_label_T = sorted_time
    Y_label_E = sorted_events
    Obs = tf.reduce_sum(Y_label_E)

    # numerical stability
    amax = tf.reduce_max(Y_hat_c)
    Y_hat_c_shift = tf.subtract(Y_hat_c, amax)
    # Y_hat_c_shift = tf.debugging.check_numerics(Y_hat_c_shift, message="checking y_hat_c_shift")
    Y_hat_hr = tf.exp(Y_hat_c_shift)
    Y_hat_cumsum = tf.math.log(tf.cumsum(Y_hat_hr)) + amax  # pylint: disable=no-value-for-parameter

    unique_values, segment_ids = tf.unique(Y_label_T)
    loss_s2_v = tf.math.segment_max(Y_hat_cumsum, segment_ids)
    loss_s2_count = tf.math.segment_sum(Y_label_E, segment_ids)

    loss_s2 = tf.reduce_sum(tf.multiply(loss_s2_v, loss_s2_count))
    loss_s1 = tf.reduce_sum(tf.multiply(Y_hat_c, Y_label_E))
    loss_breslow = tf.divide(tf.subtract(loss_s2, loss_s1), Obs)
    return loss_breslow


def concordance_index(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Calculate concordance index (C-index).

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predictions.

    Returns:
        tf.Tensor: Concordance index.
    """
    E = y_pred[:, -1]
    y_pred = y_pred[:, :-1]
    E = tf.reshape(E, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    y_pred = -y_pred  # negative of log hazard ratio to have correct relationship with survival
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)
    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    event = tf.multiply(tf.transpose(E), E)
    f = tf.multiply(tf.cast(f, tf.float32), event)
    f = tf.compat.v1.matrix_band_part(tf.cast(f, tf.float32), -1, 0)
    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)
    return tf.where(tf.equal(f, 0), 0.0, g/f)


def add_regularization(
    model: tf.keras.Model,
    regularizer: tf.keras.layers.Layer
) -> tf.keras.Model:
    '''Adds regularization (e.g. L2) to all eligible layers of a model.
    This function is from "https://sthalles.github.io/keras-regularizer/" '''

    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print('Regularizer must be a subclass of tf.keras.regularizers.Regularizer')
        return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


def get_uq_predictions(
    img: tf.Tensor,
    pred_fn: tf.keras.Model,
    num_outcomes: int,
    uq_n: int = 30
) -> Tuple[tf.Tensor, tf.Tensor, int]:
    if not num_outcomes:
        yp_drop = {}  # type: Union[List[Any], Dict[int, List]]
    else:
        yp_drop = {n: [] for n in range(num_outcomes)}
    for _ in range(uq_n):
        yp = pred_fn(img, training=False)
        if not num_outcomes:
            num_outcomes = 1 if not isinstance(yp, list) else len(yp)
            yp_drop = {n: [] for n in range(num_outcomes)}
        if num_outcomes > 1:
            for o in range(num_outcomes):
                yp_drop[o] += [yp[o]]
        else:
            yp_drop[0] += [yp]
    if num_outcomes > 1:
        yp_drop = [tf.stack(yp_drop[n], axis=0) for n in range(num_outcomes)]
        yp_mean = [tf.math.reduce_mean(yp_drop[n], axis=0) for n in range(num_outcomes)]
        yp_std = [tf.math.reduce_std(yp_drop[n], axis=0) for n in range(num_outcomes)]
    else:
        yp_drop = tf.stack(yp_drop[0], axis=0)
        yp_mean = tf.math.reduce_mean(yp_drop, axis=0)
        yp_std = tf.math.reduce_std(yp_drop, axis=0)
    return yp_mean, yp_std, num_outcomes
