"""PyTorch model utility functions."""

import types
from types import SimpleNamespace
from typing import Dict, Generator, Iterable, List, Tuple, Union, Optional

import torch
import numpy as np
import slideflow as sf
from pandas.core.frame import DataFrame
from scipy.special import softmax
from slideflow.stats import df_from_pred
from slideflow.errors import DatasetError
from slideflow.util import log, ImgBatchSpeedColumn
from rich.progress import Progress, TimeElapsedColumn, SpinnerColumn
from functools import reduce

# -----------------------------------------------------------------------------

def cycle(iterable: Iterable) -> Generator:
    while True:
        for i in iterable:
            yield i


def get_module_by_name(module: Union[torch.Tensor, torch.nn.Module],
                       access_string: str):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)


def print_module_summary(
    module: torch.nn.Module,
    inputs: List[torch.Tensor],
    max_nesting: int = 3,
    skip_redundant: bool = True
) -> str:
    """Prints and returns summary of a torch module.

    Args:
        module (torch.nn.Module): PyTorch module.
        inputs (torch.Tensor): Input tensors, for calculating layer sizes.
        max_nesting (int, optional): Module depth. Defaults to 3.
        skip_redundant (bool, optional): Skip redundant entries.
            Defaults to True.

    Returns:
        str: Summary of the module.
    """
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]

    def pre_hook(_mod, _inputs):
        nesting[0] += 1

    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(types.SimpleNamespace(mod=mod, outputs=outputs))

    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(e.outputs[0].shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    summary_rows = []
    print()
    for row in rows:
        str_row = '  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths))
        summary_rows += [str_row]
        print(str_row)
    print()
    return '\n'.join(summary_rows)


def enable_dropout(m: torch.nn.Module) -> None:
    for module in m.modules():
        if module.__class__.__name__ == 'LinearBlock':
            for submodule in module.modules():
                if submodule.__class__.__name__.startswith('Dropout'):
                    submodule.train()


def get_uq_predictions(
    img: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    model: torch.nn.Module,
    num_outcomes: Optional[int] = None,
    uq_n: int = 30
) -> Tuple[Union[torch.Tensor, List[torch.Tensor]],
           Union[torch.Tensor, List[torch.Tensor]],
           int]:
    """Performs UQ inference (mean and stdev/uncertainty), calculated
    using a set number of forward passes.

    Args:
        img (torch.Tensor): Batch of input images.
        model (torch.nn.Module): Model to use for inference.
        num_outcomes (int): Number of expected outcomes.
        uq_n (int, optional): Number of forward passes. Defaults to 30.

    Returns:
        A tuple containing

            torch.Tensor: Mean of forward passes.

            torch.Tensor: Standard deviation of forward passes.

            int: Number of detected outcomes.
    """
    enable_dropout(model)
    if not num_outcomes:
        yp_drop = {}  # type: Dict[int, List]
    else:
        yp_drop = {n: [] for n in range(num_outcomes)}
    for _ in range(uq_n):
        yp = model(*img)
        if not num_outcomes:
            num_outcomes = 1 if not isinstance(yp, (list, tuple)) else len(yp)
            yp_drop = {n: [] for n in range(num_outcomes)}
        if num_outcomes > 1:
            for o in range(num_outcomes):
                yp_drop[o] += [yp[o]]
        else:
            yp_drop[0] += [yp]
    if num_outcomes > 1:
        stacked = [torch.stack(yp_drop[n], dim=0) for n in range(num_outcomes)]
        yp_mean = [torch.mean(stacked[n], dim=0) for n in range(num_outcomes)]
        yp_std = [torch.std(stacked[n], dim=0) for n in range(num_outcomes)]
    else:
        stacked = torch.stack(yp_drop[0], dim=0)  # type: ignore
        yp_mean = torch.mean(stacked, dim=0)  # type: ignore
        yp_std = torch.std(stacked, dim=0)  # type: ignore
    return yp_mean, yp_std, num_outcomes


def eval_from_model(
    model: "torch.nn.Module",
    dataset: "torch.utils.data.DataLoader",
    model_type: str,
    torch_args: Optional[SimpleNamespace],
    uq: bool = False,
    uq_n: int = 30,
    steps: Optional[int] = None,
    pb_label: str = "Evaluating...",
    verbosity: str = 'full',
    predict_only: bool = False,
) -> Tuple[DataFrame, float, float]:
    """Evaluates a model from a dataset of (y_true, y_pred, tile_to_slide),
    returning predictions, accuracy, and loss.

    Args:
        model (str): Path to PyTorch model.
        dataset (tf.data.Dataset): PyTorch dataloader.
        model_type (str, optional): 'categorical', 'linear', or 'cph'. If
            multiple linear outcomes are present, y_true is stacked into a
            single vector for each image. Defaults to 'categorical'.
        torch_args (namespace): Namespace containing num_slide_features,
            slide_input, update_corrects, and update_loss functions.

    Keyword args:
        uq (bool, optional): Perform uncertainty quantification with dropout.
            Defaults to False.
        uq_n (int, optional): Number of forward passes to perform
            when calculating MC Dropout uncertainty. Defaults to 30.
        steps (int, optional): Number of steps (batches) of evaluation to
            perform. If None, uses the full dataset. Defaults to None.
        pb_label (str, optional): Progress bar label.
            Defaults to "Predicting..."
        verbosity (str, optional): Either 'full', 'quiet', or 'silent'.
            Verbosity for progress bars.
        predict_only (bool, optional): Only generate predictions without
            comparisons to y_true. Defaults to False.

    Returns:
        pd.DataFrame, accuracy, loss
    """
    if verbosity not in ('silent', 'quiet', 'full'):
        raise ValueError(f"Invalid value '{verbosity}' for argument 'verbosity'")
    if not predict_only and torch_args is None:
        raise ValueError("Argument `torch_args` must be supplied if evaluating.")

    y_true, y_pred, tile_to_slides, locations, y_std = [], [], [], [], []
    losses, total, num_outcomes, batch_size = 0, 0, 0, 0
    corrects, acc, loss = None, None, None
    model.eval()
    device = torch.device('cuda:0')

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
                batch_size = batch[0].shape[0]
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
                        total=dataset.num_tiles if not steps else steps*batch_size)  # type: ignore
                    pb.start()
            # ---------------------------------------------------------------------

            if incl_loc:
                img, yt, slide, loc_x, loc_y = batch
                locations += [torch.stack([loc_x, loc_y], dim=-1).cpu().numpy()]
            else:
                img, yt, slide = batch

            if verbosity != 'silent':
                pb.advance(task, img.shape[0])

            img = img.to(device, non_blocking=True)
            img = img.to(memory_format=torch.channels_last)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    # GPU normalization
                    if torch_args is not None and torch_args.normalizer:
                        img = torch_args.normalizer.preprocess(img)

                    # Slide-level features
                    if torch_args is not None and torch_args.num_slide_features:
                        slide_inp = torch.tensor([
                            torch_args.slide_input[s] for s in slide
                        ])
                        inp = (img, slide_inp.to(device))
                    else:
                        inp = (img,)  # type: ignore

                    if uq:
                        res, yp_std, num_outcomes = get_uq_predictions(
                            inp, model, num_outcomes, uq_n
                        )
                        if isinstance(yp_std, list):
                            yp_std = [y.cpu().numpy().copy() for y in yp_std]
                        else:
                            yp_std = yp_std.cpu().numpy().copy()
                        y_std += [yp_std]  # type: ignore
                    else:
                        res = model(*inp)

                    if not predict_only:
                        assert torch_args is not None
                        corrects = torch_args.update_corrects(res, yt, corrects)
                        losses = torch_args.update_loss(res, yt, losses, img.size(0))

                    if isinstance(res, list):
                        res = [r.cpu().numpy().copy() for r in res]
                    else:
                        res = res.cpu().numpy().copy()

                    y_pred += [res]

            if not predict_only and type(yt) == dict:
                y_true += [[yt[f'out-{o}'] for o in range(len(yt))]]
            elif not predict_only:
                yt = yt.detach().numpy().copy()
                y_true += [yt]
            tile_to_slides += slide
            total += img.shape[0]
    except KeyboardInterrupt:
        if pb is not None:
            pb.stop()
        raise

    if not total:
        raise DatasetError("Empty dataset, unable to predict/evaluate.")
    if verbosity != 'silent':
        pb.stop()

    if y_pred == []:
        raise ValueError("Insufficient data for evaluation.")

    # Concatenate predictions for each outcome.
    if type(y_pred[0]) == list:
        y_pred = [np.concatenate(yp) for yp in zip(*y_pred)]
        if uq:
            y_std = [np.concatenate(ys) for ys in zip(*y_std)]  # type: ignore
    else:
        y_pred = [np.concatenate(y_pred)]
        if uq:
            y_std = [np.concatenate(y_std)]

    # Concatenate y_true for each outcome
    if not predict_only and type(y_true[0]) == list:
        y_true = [np.concatenate(yt) for yt in zip(*y_true)]

        # Merge multiple linear outcomes into a single vector
        if model_type == 'linear':
            y_true = [np.stack(y_true, axis=1)]  # type: ignore
    elif not predict_only:
        y_true = [np.concatenate(y_true)]
    else:
        y_true = None  # type: ignore

    # We will need to enforce softmax encoding for tile-level statistics.
    if model_type == 'categorical':
        y_pred = [softmax(yp, axis=1) for yp in y_pred]

    # Calculate final accuracy and loss
    if not predict_only:
        loss = losses / total
        if isinstance(corrects, dict):
            acc = {k: v.cpu().numpy()/total for k, v in corrects.items()}
        elif isinstance(corrects, (int, float)):
            acc = corrects / total  # type: ignore
        else:
            assert corrects is not None, "Empty dataset to evaluate/predict."
            acc = corrects.cpu().numpy() / total

    if locations != []:
        locations = np.concatenate(locations)
    else:
        locations = None  # type: ignore
    if not uq:
        y_std = None  # type: ignore

    # Create pandas DataFrame from arrays
    df = df_from_pred(y_true, y_pred, y_std, tile_to_slides, locations)

    log.debug("Evaluation complete.")
    return df, acc, loss  # type: ignore


def predict_from_model(
    model: "torch.nn.Module",
    dataset: "torch.utils.data.DataLoader",
    model_type: str,
    torch_args: Optional[SimpleNamespace],
    pb_label: str = "Predicting...",
    **kwargs,
) -> DataFrame:
    """Generates predictions (y_true, y_pred, tile_to_slide) from
    a given PyTorch model and dataset.

    Args:
        model (str): Path to PyTorch model.
        dataset (tf.data.Dataset): PyTorch dataloader.
        model_type (str, optional): 'categorical', 'linear', or 'cph'. If
            multiple linear outcomes are present, y_true is stacked into a
            single vector for each image. Defaults to 'categorical'.
        torch_args (namespace): Namespace containing num_slide_features
            and slide_input.

    Keyword args:
        uq (bool, optional): Perform uncertainty quantification with dropout.
            Defaults to False.
        uq_n (int, optional): Number of forward passes to perform
            when calculating MC Dropout uncertainty. Defaults to 30.
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
        model_type=model_type,
        torch_args=torch_args,
        pb_label=pb_label,
        predict_only=True,
        **kwargs
    )
    return df

# -----------------------------------------------------------------------------

def xception(*args, **kwargs):
    import pretrainedmodels
    return pretrainedmodels.xception(*args, **kwargs)


def nasnetalarge(*args, **kwargs):
    import pretrainedmodels
    return pretrainedmodels.nasnetalarge(*args, **kwargs)