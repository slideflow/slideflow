import os
import pandas as pd
import slideflow as sf
import numpy as np
from os.path import join, exists, isdir
from typing import Union, List, Optional, Callable
from slideflow import Dataset, log, errors
from slideflow.util import path_to_name
from slideflow.stats.metrics import ClassifierMetrics
from .._params import (
    TrainerConfig, ModelConfigCLAM, TrainerConfigCLAM
)

# -----------------------------------------------------------------------------

def eval_mil(
    weights: str,
    dataset: Dataset,
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]],
    config: Optional[TrainerConfig] = None,
    *,
    outdir: str = 'mil',
) -> pd.DataFrame:
    """Evaluate a multi-instance learning model."""
    import torch

    if isinstance(config, TrainerConfigCLAM):
        raise NotImplementedError

    # Read configuration from saved model, if available
    if config is None:
        if not exists(join(weights, 'mil_params.json')):
            raise errors.ModelError(
                f"Could not find `mil_params.json` at {weights}. Check the "
                "provided model/weights path, or provide a configuration "
                "with 'config'."
            )
        else:
            p = sf.util.load_json(join(weights, 'mil_params.json'))
            config = sf.mil.mil_config(trainer=p['trainer'], **p['params'])

    # Prepare ground-truth labels
    labels, unique = dataset.labels(outcomes, format='id')

    # Prepare bags and targets
    if isinstance(bags, str):
        bags = dataset.pt_files(bags)
    else:
        bags = np.array(bags)
    slides = [path_to_name(f) for f in bags]
    targets = np.array([labels[s] for s in slides])

    # Detect feature size from bags
    n_features = torch.load(bags[0]).shape[-1]
    n_out = len(unique)

    log.info(f"Building model {config.model_fn.__name__}")
    if isinstance(config, TrainerConfigCLAM):
        config_size = config.model_fn.sizes[config.model_config.model_size]
        model = config.model_fn(size=[n_features] + config_size[1:])
        transformer = False
    elif isinstance(config.model_config, ModelConfigCLAM):
        model = config.model_fn(size=[n_features, 256, 128])
        transformer = False
    else:
        model = config.model_fn(n_features, n_out)
        transformer = True
    if isdir(weights):
        if exists(join(weights, 'models', 'best_valid.pth')):
            weights = join(weights, 'models', 'best_valid.pth')
        elif exists(join(weights, 'results', 's_0_checkpoint.pt')):
            weights = join(weights, 'results', 's_0_checkpoint.pt')
        else:
            raise errors.ModelError(
                f"Could not find model weights at path {weights}"
            )
    log.info(f"Loading model weights from [green]{weights}[/]")
    model.load_state_dict(torch.load(weights))

    # Prepare device.
    device = torch.device('cuda')
    model.relocate()  # type: ignore
    model.eval()

    # Inference.
    y_true = []
    y_pred = []
    for i, bag in enumerate(bags):
        loaded = torch.load(bag).to(device)
        y_true.append(targets[i])
        with torch.no_grad():
            if transformer:
                loaded = torch.unsqueeze(loaded, dim=0)
                lens = torch.from_numpy(np.array([loaded.shape[0]])).to(device)
                model_out = model(loaded, lens)
                y_pred.append(torch.nn.functional.softmax(model_out, dim=1).cpu().numpy())
            else:
                y_pred.append(torch.nn.functional.softmax(model(loaded)[0], dim=1).cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.concatenate(y_pred, axis=0)

    # Generate metrics
    for idx in range(y_pred.shape[-1]):
        m = ClassifierMetrics(y_true=(y_true == idx).astype(int), y_pred=y_pred[:, idx])
        log.info(f"AUC (cat #{idx+1}): {m.auroc:.3f}")
        log.info(f"AP  (cat #{idx+1}): {m.ap:.3f}")

    # Save results
    if not exists(outdir):
        os.makedirs(outdir)
    model_dir = sf.util.get_new_model_dir(outdir, config.model_config.model)
    df_dict = dict(slide=slides, y_true=y_true)
    for i in range(y_pred.shape[-1]):
        df_dict[f'y_pred{i}'] = y_pred[:, i]
    df = pd.DataFrame(df_dict)
    pred_out = join(model_dir, 'predictions.parquet')
    df.to_parquet(pred_out)
    log.info(f"Predictions saved to [green]{pred_out}[/]")

    return df

# -----------------------------------------------------------------------------

def predict_from_model(
    model: Callable,
    config: TrainerConfig,
    dataset: "sf.Dataset",
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]]
):
    import torch

    # Prepare labels.
    labels, unique = dataset.labels(outcomes, format='id')

    # Prepare bags and targets.
    if isinstance(bags, str):
        bags = dataset.pt_files(bags)
    else:
        bags = np.array(bags)
    slides = [path_to_name(f) for f in bags]
    targets = np.array([labels[s] for s in slides])

    # Inference.
    transformer = not (isinstance(config, TrainerConfigCLAM)
                       or isinstance(config.model_config, ModelConfigCLAM))
    device = torch.device('cuda')
    y_true = []
    y_pred = []
    log.info("Generating predictions...")
    for i, bag in enumerate(bags):
        loaded = torch.load(bag).to(device)
        y_true.append(targets[i])
        with torch.no_grad():
            if transformer:
                loaded = torch.unsqueeze(loaded, dim=0)
                lens = torch.from_numpy(np.array([loaded.shape[0]])).to(device)
                model_out = model(loaded, lens)
                y_pred.append(torch.nn.functional.softmax(model_out, dim=1).cpu().numpy())
            else:
                y_pred.append(torch.nn.functional.softmax(model(loaded)[0], dim=1).cpu().numpy())

    # Create dataframe.
    yp = np.concatenate(y_pred, axis=0)
    df_dict = dict(slide=slides, y_true=np.array(y_true))
    for i in range(yp.shape[-1]):
        df_dict[f'y_pred{i}'] = yp[:, i]

    return pd.DataFrame(df_dict)