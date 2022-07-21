"""PyTorch model utility functions."""

import types
from types import SimpleNamespace
from typing import Dict, Generator, Iterable, List, Tuple, Union

import numpy as np
import slideflow as sf
from pandas.core.frame import DataFrame
from scipy.special import softmax
from slideflow.stats import df_from_pred
from slideflow.util import log
from tqdm import tqdm

import torch


def cycle(iterable: Iterable) -> Generator:
    while True:
        for i in iterable:
            yield i


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
    num_outcomes: int,
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



def _predict_from_model(
    model: "torch.nn.Module",
    dataset: "torch.utils.data.DataLoader",
    model_type: str,
    pred_args: SimpleNamespace,
    uq_n: int = 30,
    num_tiles: int = 0,
    incl_loc: bool = False
) -> DataFrame:
    """Generates predictions (y_true, y_pred, tile_to_slide) from
    a given PyTorch model and dataset.

    Args:
        model (torch.nn.Module): PyTorch model.
        dataset (torch.utils.data.DatatLoader): PyTorch dataloader.
        pred_args (namespace): Namespace containing slide_input,
            update_corrects, and update_loss functions.
        model_type (str, optional): 'categorical', 'linear', or 'cph'.
            If multiple linear outcomes are present, y_true is stacked into
            a single vector for each image. Defaults to 'categorical'.
        uq_n (int, optional): Number of forward passes to perform
            when calculating MC Dropout uncertainty. Defaults to 30.

    Returns:
        pd.DataFrame
    """

    # Get predictions and performance metrics
    log.debug("Generating predictions from torch model")
    y_pred, tile_to_slides = [], []
    locations = [] if incl_loc else None
    y_std = [] if pred_args.uq else None  # type: ignore
    num_outcomes = 0
    model.eval()
    device = torch.device('cuda:0')
    pb = tqdm(
        desc='Predicting...',
        total=dataset.num_tiles,  # type: ignore
        ncols=80,
        unit='img',
        leave=False
    )
    for batch in dataset:  # TODO: support not needing to supply yt

        # Parse batch
        if incl_loc:
            img, yt, slide, loc_x, loc_y = batch
            locations += [torch.stack([loc_x, loc_y], dim=-1).cpu().numpy()]
        else:
            img, yt, slide = batch

        img = img.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # Slide-level features
                if pred_args.num_slide_features:
                    slide_inp = torch.tensor([
                        pred_args.slide_input[s] for s in slide
                    ])
                    inp = (img, slide_inp.to(device))
                else:
                    inp = (img,)  # type: ignore
                if pred_args.uq:
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
                if isinstance(res, list):
                    res = [r.cpu().numpy().copy() for r in res]
                else:
                    res = res.cpu().numpy().copy()
                y_pred += [res]
        tile_to_slides += slide
        pb.update(img.shape[0])

    # Concatenate predictions for each outcome
    if type(y_pred[0]) == list:
        y_pred = [np.concatenate(yp) for yp in zip(*y_pred)]
        if pred_args.uq:
            y_std = [np.concatenate(ys) for ys in zip(*y_std)]  # type: ignore
    else:
        y_pred = [np.concatenate(y_pred)]
        if pred_args.uq:
            y_std = [np.concatenate(y_std)]

    # We will need to enforce softmax encoding for tile-level statistics.
    if model_type == 'categorical':
        y_pred = [softmax(yp, axis=1) for yp in y_pred]

    if incl_loc:
        locations = np.concatenate(locations)

    # Create pandas DataFrame from arrays
    df = df_from_pred(None, y_pred, y_std, tile_to_slides, locations)

    log.debug("Prediction complete.")
    return df


def _eval_from_model(
    model: "torch.nn.Module",
    dataset: "torch.utils.data.DataLoader",
    model_type: str,
    pred_args: SimpleNamespace,
    uq_n: int = 30,
    num_tiles: int = 0,
    incl_loc: bool = False
) -> Tuple[DataFrame, float, float]:
    """Generates predictions (y_true, y_pred, tile_to_slide) from
    a given PyTorch model and dataset.

    Args:
        model (str): Path to PyTorch model.
        dataset (tf.data.Dataset): PyTorch dataloader.
        pred_args (namespace): Namespace containing slide_input,
            update_corrects, and update_loss functions.
        model_type (str, optional): 'categorical', 'linear', or 'cph'. If
            multiple linear outcomes are present, y_true is stacked into a
            single vector for each image. Defaults to 'categorical'.
        uq_n (int, optional): Number of forward passes to perform
            when calculating MC Dropout uncertainty. Defaults to 30.

    Returns:
        pd.DataFrame, accuracy, loss
    """

    y_true, y_pred, tile_to_slides = [], [], []
    locations = [] if incl_loc else None
    y_std = [] if pred_args.uq else None  # type: ignore
    corrects = pred_args.running_corrects
    losses = 0
    total = 0
    num_outcomes = 0

    log.debug("Evaluating torch model")

    model.eval()
    device = torch.device('cuda:0')
    pb = tqdm(
        desc='Evaluating...',
        total=dataset.num_tiles,  # type: ignore
        ncols=80,
        unit='img',
        leave=False
    )
    for batch in dataset:

        # Parse batch
        if incl_loc:
            img, yt, slide, loc_x, loc_y = batch
            locations += [torch.stack([loc_x, loc_y], dim=-1).cpu().numpy()]
        else:
            img, yt, slide = batch

        img = img.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # Slide-level features
                if pred_args.num_slide_features:
                    slide_inp = torch.tensor([
                        pred_args.slide_input[s] for s in slide
                    ])
                    inp = (img, slide_inp.to(device))
                else:
                    inp = (img,)  # type: ignore
                if pred_args.uq:
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
                corrects = pred_args.update_corrects(res, yt, corrects)
                losses = pred_args.update_loss(res, yt, losses, img.size(0))
                if isinstance(res, list):
                    res = [r.cpu().numpy().copy() for r in res]
                else:
                    res = res.cpu().numpy().copy()
                y_pred += [res]
        if type(yt) == dict:
            y_true += [[yt[f'out-{o}'] for o in range(len(yt))]]
        else:
            yt = yt.detach().numpy().copy()
            y_true += [yt]
        tile_to_slides += slide
        total += img.shape[0]
        pb.update(img.shape[0])

    # Concatenate predictions for each outcome.
    if type(y_pred[0]) == list:
        y_pred = [np.concatenate(yp) for yp in zip(*y_pred)]
        if pred_args.uq:
            y_std = [np.concatenate(ys) for ys in zip(*y_std)]  # type: ignore
    else:
        y_pred = [np.concatenate(y_pred)]
        if pred_args.uq:
            y_std = [np.concatenate(y_std)]

    # Concatenate y_true for each outcome
    if type(y_true[0]) == list:
        y_true = [np.concatenate(yt) for yt in zip(*y_true)]

        # Merge multiple linear outcomes into a single vector
        if model_type == 'linear':
            y_true = [np.stack(y_true, axis=1)]  # type: ignore
    else:
        y_true = [np.concatenate(y_true)]

    # We will need to enforce softmax encoding for tile-level statistics.
    if model_type == 'categorical':
        y_pred = [softmax(yp, axis=1) for yp in y_pred]

    # Calculate final accuracy and loss
    loss = losses / total
    if isinstance(corrects, dict):
        acc = {k: v.cpu().numpy()/total for k, v in corrects.items()}
    elif isinstance(corrects, (int, float)):
        acc = corrects / total  # type: ignore
    else:
        acc = corrects.cpu().numpy() / total
    if sf.getLoggingLevel() <= 20:
        sf.util.clear_console()

    if incl_loc:
        locations = np.concatenate(locations)

    # Create pandas DataFrame from arrays
    df = df_from_pred(y_true, y_pred, y_std, tile_to_slides, locations)

    log.debug("Evaluation complete.")
    return df, acc, loss  # type: ignore
