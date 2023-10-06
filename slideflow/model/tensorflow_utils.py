"""Tensorflow model utility functions."""

import os
import tempfile
from typing import (TYPE_CHECKING, Any, Dict, List, Tuple, Union, Optional,
                    Callable)

import numpy as np
import slideflow as sf
from pandas.core.frame import DataFrame
from slideflow.stats import df_from_pred
from slideflow.util import log, ImgBatchSpeedColumn
from rich.progress import Progress, TimeElapsedColumn, SpinnerColumn

import tensorflow as tf

if TYPE_CHECKING:
    import neptune.new as neptune

# -----------------------------------------------------------------------------

def log_summary(
    model: tf.keras.Model,
    neptune_run: "neptune.Run" = None
) -> None:
    """Log the model summary.

    Args:
        model (tf.keras.Model): Tensorflow/Keras model.
        neptune_run (neptune.Run, optional): Neptune run. Defaults to None.
    """
    if sf.getLoggingLevel() <= 20:
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
    num_outcomes: Optional[int] = None,
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


def unwrap(
    model: tf.keras.models.Model
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Unwraps a Tensorflow model built in Slideflow, returning the
    input tensor, post-convolutional output tensor, and final model output
    tensor.

    Args:
        model (tf.keras.models.Model): Model built with Slideflow.

    Returns:
        A tuple containing

            tf.Tensor:  Input tensor.

            tf.Tensor:  Post-convolutional layer output tensor.

            tf.Tensor:  Final model output tensor.
    """
    submodel = model.layers[1]
    x = submodel.outputs[0]
    postconv = x
    for layer_index in range(2, len(model.layers)):
        extracted_layer = model.layers[layer_index]
        x = extracted_layer(x)

    return submodel.inputs, postconv, x


def flatten(
    model: tf.keras.models.Model
) -> tf.keras.models.Model:
    """Unwrapped then flattens a Tensorflow model."""

    inputs, _, outputs = unwrap(model)
    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


def eval_from_model(
    model: "tf.keras.Model",
    dataset: "tf.data.Dataset",
    model_type: Optional[str],
    loss: Optional[Callable],
    num_tiles: int = 0,
    uq: bool = False,
    uq_n: int = 30,
    steps: Optional[int] = None,
    predict_only: bool = False,
    pb_label: str = "Evaluating...",
    verbosity: str = 'full',
) -> Tuple[DataFrame, float, float]:
    """Evaluates predictions (y_true, y_pred, tile_to_slide) from a given
    Tensorflow model and dataset, generating predictions, accuracy, and loss.

    Args:
        model (str): Path to Tensorflow model.
        dataset (tf.data.Dataset): Tensorflow dataset.
        model_type (str, optional): 'categorical', 'linear', or 'cph'.
            Will not attempt to calculate accuracy for non-categorical models.
            Defaults to 'categorical'.
        loss (Callable, optional): Loss function which accepts (y_true, y_pred).

    Keyword args:
        num_tiles (int, optional): Used for progress bar. Defaults to 0.
        uq (bool, optional): Perform uncertainty quantification with dropout.
            Defaults to False.
        uq_n (int, optional): Number of per-tile inferences to perform is
            calculating uncertainty via dropout.
        steps (int, optional): Number of steps (batches) of evaluation to
            perform. If None, uses the full dataset. Defaults to None.
        predict_only (bool, optional): Only generate predictions without
            comparisons to y_true. Defaults to False.
        pb_label (str, optional): Progress bar label.
            Defaults to "Evaluating..."
        verbosity (str, optional): Either 'full', 'quiet', or 'silent'.
            Verbosity for progress bars.

    Returns:
        pd.DataFrame, accuracy, loss
    """

    if verbosity not in ('silent', 'quiet', 'full'):
        raise ValueError(f"Invalid value '{verbosity}' for argument 'verbosity'")

    @tf.function
    def get_predictions(img, training=False):
        return model(img, training=training)

    y_true, y_pred, tile_to_slides, locations, y_std = [], [], [], [], []
    num_vals, num_batches, num_outcomes, running_loss = 0, 0, 0, 0
    batch_size = 0
    loc_missing = False

    is_cat = (model_type == 'categorical')
    if not is_cat:
        acc = None

    if verbosity != 'silent':
        pb = Progress(SpinnerColumn(), transient=True)
        pb.add_task(pb_label, total=None)
        pb.start()
    else:
        pb = None
    try:
        for step, batch in enumerate(dataset):
            if steps is not None and step >= steps:
                break

            # --- Detect data structure, if this is the first batch ---------------
            if not batch_size:
                if len(batch) not in (3, 5):
                    raise IndexError(
                        "Unexpected number of items returned from dataset batch. "
                        f"Expected either '3' or '5', got: {len(batch)}")

                incl_loc = (len(batch) == 5)
                batch_size = batch[2].shape[0]
                if verbosity != 'silent':
                    pb.stop()
                    pb = Progress(
                        SpinnerColumn(),
                        *Progress.get_default_columns(),
                        TimeElapsedColumn(),
                        ImgBatchSpeedColumn(),
                        transient=sf.getLoggingLevel()>20 or verbosity == 'quiet')
                    task = pb.add_task(
                        pb_label,
                        total=num_tiles if not steps else steps*batch_size)
                    pb.start()
            # ---------------------------------------------------------------------

            if incl_loc:
                img, yt, slide, loc_x, loc_y = batch
                if not loc_missing and loc_x is None:
                    log.warning("TFrecord location information not found.")
                    loc_missing = True
                elif not loc_missing:
                    locations += [tf.stack([loc_x, loc_y], axis=-1).numpy()]  # type: ignore
            else:
                img, yt, slide = batch

            if verbosity != 'silent':
                pb.advance(task, slide.shape[0])
            tile_to_slides += [_byte.decode('utf-8') for _byte in slide.numpy()]
            num_vals += slide.shape[0]
            num_batches += 1

            if uq:
                yp, yp_std, num_outcomes = get_uq_predictions(
                    img, get_predictions, num_outcomes, uq_n
                )
                y_pred += [yp]
                y_std += [yp_std]  # type: ignore
            else:
                yp = get_predictions(img, training=False)
                y_pred += [yp]

            if not predict_only:
                if isinstance(yt, dict):
                    y_true += [[yt[f'out-{o}'].numpy() for o in range(len(yt))]]
                    yt = [yt[f'out-{o}'] for o in range(len(yt))]
                    if loss is not None:
                        loss_val = [loss(yt[i], yp[i]) for i in range(len(yt))]
                        loss_val = [tf.boolean_mask(l, tf.math.is_finite(l)) for l in loss_val]
                        batch_loss = tf.math.reduce_sum(loss_val).numpy()
                        running_loss = (((num_vals - slide.shape[0]) * running_loss) + batch_loss) / num_vals
                else:
                    y_true += [yt.numpy()]
                    if loss is not None:
                        loss_val = loss(yt, yp)
                        if tf.rank(loss_val):
                            # Loss is a vector
                            is_finite = tf.math.is_finite(loss_val)
                            batch_loss = tf.math.reduce_sum(tf.boolean_mask(loss_val, is_finite)).numpy()
                        else:
                            # Loss is a scalar
                            batch_loss = loss_val.numpy()  # type: ignore
                        running_loss = (((num_vals - slide.shape[0]) * running_loss) + batch_loss) / num_vals
    except KeyboardInterrupt:
        if pb is not None:
            pb.stop()
        raise

    if verbosity != 'silent':
        pb.stop()

    if y_pred == []:
        raise ValueError("Insufficient data for evaluation.")

    if isinstance(y_pred[0], list):
        # Concatenate predictions for each outcome
        y_pred = [np.concatenate(yp) for yp in zip(*y_pred)]
        if uq:
            y_std = [np.concatenate(ys) for ys in zip(*y_std)]  # type: ignore
    else:
        y_pred = [np.concatenate(y_pred)]
        if uq:
            y_std = [np.concatenate(y_std)]

    if not predict_only and isinstance(y_true[0], list):
        # Concatenate y_true for each outcome
        y_true = [np.concatenate(yt) for yt in zip(*y_true)]
        if is_cat:
            acc = [
                np.sum(y_true[i] == np.argmax(y_pred[i], axis=1)) / num_vals
                for i in range(len(y_true))
            ]
    elif not predict_only:
        y_true = [np.concatenate(y_true)]
        if is_cat:
            acc = np.sum(y_true[0] == np.argmax(y_pred[0], axis=1)) / num_vals
    else:
        y_true = None  # type: ignore

    if locations != []:
        locations = np.concatenate(locations)
    else:
        locations = None  # type: ignore
    if not uq:
        y_std = None  # type: ignore

    # Create pandas DataFrame from arrays
    df = df_from_pred(y_true, y_pred, y_std, tile_to_slides, locations)

    # Note that Keras loss during training includes regularization losses,
    # so this loss will not match validation loss calculated during training
    log.debug("Evaluation complete.")
    return df, acc, running_loss  # type: ignore


def predict_from_model(
    model: "tf.keras.Model",
    dataset: "tf.data.Dataset",
    pb_label: str = "Predicting...",
    **kwargs
) -> DataFrame:
    """Generate a DataFrame of predictions from a model.

    Args:
        model (str): Path to Tensorflow model.
        dataset (tf.data.Dataset): Tensorflow dataset.

    Keyword args:
        num_tiles (int, optional): Used for progress bar. Defaults to 0.
        uq (bool, optional): Perform uncertainty quantification with dropout.
            Defaults to False.
        uq_n (int, optional): Number of per-tile inferences to perform is
            calculating uncertainty via dropout.
        steps (int, optional): Number of steps (batches) of evaluation to
            perform. If None, uses the full dataset. Defaults to None.
        pb_label (str, optional): Progress bar label.
            Defaults to "Predicting..."
        verbosity (str, optional): Either 'full', 'quiet', or 'silent'.
            Verbosity for progress bars.

    Returns:
        pd.DataFrame
    """
    df, _, _ = eval_from_model(
        model,
        dataset,
        model_type=None,
        loss=None,
        predict_only=True,
        pb_label=pb_label,
        **kwargs
    )
    return df

# -----------------------------------------------------------------------------

class CosineAnnealer:

    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0

    def step(self):
        self.n += 1
        cos = np.cos(np.pi * (self.n / self.steps)) + 1
        return self.end + (self.start - self.end) / 2. * cos


class OneCycleScheduler(tf.keras.callbacks.Callback):
    """ `Callback` that schedules the learning rate on a 1cycle policy as per Leslie Smith's paper(https://arxiv.org/pdf/1803.09820.pdf).
    If the model supports a momentum parameter, it will also be adapted by the schedule.
    The implementation adopts additional improvements as per the fastai library: https://docs.fast.ai/callbacks.one_cycle.html, where
    only two phases are used and the adaptation is done using cosine annealing.
    In phase 1 the LR increases from `lr_max / div_factor` to `lr_max` and momentum decreases from `mom_max` to `mom_min`.
    In the second phase the LR decreases from `lr_max` to `lr_max / (div_factor * 1e4)` and momemtum from `mom_max` to `mom_min`.
    By default the phases are not of equal length, with the phase 1 percentage controlled by the parameter `phase_1_pct`.
    """

    def __init__(self, lr_max, steps, mom_min=0.85, mom_max=0.95, phase_1_pct=0.3, div_factor=25.):
        super(OneCycleScheduler, self).__init__()
        lr_min = lr_max / div_factor
        final_lr = lr_max / (div_factor * 1e4)
        phase_1_steps = steps * phase_1_pct
        phase_2_steps = steps - phase_1_steps

        self.phase_1_steps = phase_1_steps
        self.phase_2_steps = phase_2_steps
        self.phase = 0
        self.step = 0

        self.phases = [[CosineAnnealer(lr_min, lr_max, phase_1_steps), CosineAnnealer(mom_max, mom_min, phase_1_steps)],
                 [CosineAnnealer(lr_max, final_lr, phase_2_steps), CosineAnnealer(mom_min, mom_max, phase_2_steps)]]

        self.lrs = []
        self.moms = []

    def on_train_begin(self, logs=None):
        self.phase = 0
        self.step = 0

        self.set_lr(self.lr_schedule().start)
        self.set_momentum(self.mom_schedule().start)

    def on_train_batch_begin(self, batch, logs=None):
        self.lrs.append(self.get_lr())
        self.moms.append(self.get_momentum())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step >= self.phase_1_steps:
            self.phase = 1

        self.set_lr(self.lr_schedule().step())
        self.set_momentum(self.mom_schedule().step())

    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None

    def get_momentum(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.momentum)
        except AttributeError:
            return None

    def set_lr(self, lr):
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        except AttributeError:
            pass # ignore

    def set_momentum(self, mom):
        try:
            tf.keras.backend.set_value(self.model.optimizer.momentum, mom)
        except AttributeError:
            pass # ignore

    def lr_schedule(self):
        return self.phases[self.phase][0]

    def mom_schedule(self):
        return self.phases[self.phase][1]

    def plot(self):
        import matplotlib.pyplot as plt
        ax = plt.subplot(1, 2, 1)
        ax.plot(self.lrs)
        ax.set_title('Learning Rate')
        ax = plt.subplot(1, 2, 2)
        ax.plot(self.moms)
        ax.set_title('Momentum')

# -----------------------------------------------------------------------------

def build_uq_model(model, n_repeat=30):
    """Rebuild a dropout-based UQ model to return predictions and uncertainties."""
    layers = [l for l in model.layers]
    n_dim = model.layers[2].output.shape[1]
    n_out = model.output.shape[1]
    log.info("Building UQ model with n_repeat={} (n_dim={}, n_out={})".format(
        n_repeat, n_dim, n_out
    ))
    new_layers = (layers[0:3]
                  + [tf.keras.layers.RepeatVector(n_repeat),
                     tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, n_dim)))]
                  + layers[3:]
                  + [tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, n_repeat, n_out)))])
    new_core = tf.keras.models.Sequential(new_layers)
    yp_mean = tf.math.reduce_mean(new_core.output, axis=1)
    yp_std = tf.math.reduce_std(new_core.output, axis=1)
    uq_model = tf.keras.models.Model(inputs=new_core.input, outputs=[yp_mean, yp_std])
    return uq_model