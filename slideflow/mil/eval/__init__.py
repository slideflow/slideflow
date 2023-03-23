import os
import pandas as pd
import slideflow as sf
import numpy as np
from rich.progress import Progress
from os.path import join, exists, isdir, dirname
from typing import Union, List, Optional, Callable, Tuple
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
    attention_heatmaps: bool = False,
    **heatmap_kwargs
) -> pd.DataFrame:
    """Evaluate a multi-instance learning model.

    Saves results for the evaluation in the target folder, including
    predictions (parquet format), attention (Numpy format for each slide),
    and attention heatmaps (if ``attention_heatmaps=True``).

    Logs classifier metrics (AUROC and AP) to the console.

    Args:
        weights (str): Path to model weights to load.
        dataset (sf.Dataset): Dataset to evaluation.
        outcomes (str, list(str)): Outcomes.
        bags (str, list(str)): Path to bags, or list of bag file paths.
            Each bag should contain PyTorch array of features from all tiles in
            a slide, with the shape ``(n_tiles, n_features)``.
        config (TrainerConfig): Configuration for building model.
            If ``weights`` is a path to a model directory, will attempt to
            read ``mil_params.json`` from this location and auto-load
            saved configuration. Defaults to None.

    Keyword arguments:
        outdir (str): Path at which to save results.
        attention_heatmaps (bool): Generate attention heatmaps for slides.
            Defaults to False.

    """
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
    y_true = np.array([labels[s] for s in slides])

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
    if (isinstance(config, TrainerConfigCLAM)
       or isinstance(config.model_config, ModelConfigCLAM)):
        y_pred, y_att = _predict_clam(model, bags, attention=True)
    else:
        y_pred, y_att = _predict_transformer(model, bags, attention=True)

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

    # Export attention
    att_path = join(model_dir, 'attention')
    if not exists(att_path):
        os.makedirs(att_path)
    for slide, att in zip(slides, y_att):
        np.savez(join(att_path, f'{slide}_att.npz'), att)
    log.info(f"Attention scores exported to [green]{att_path}[/]")

    # Attention heatmaps
    if attention_heatmaps:
        generate_attention_heatmaps(
            outdir=join(model_dir, 'heatmaps'),
            dataset=dataset,
            bags=bags,
            attention=y_att,
            **heatmap_kwargs
        )

    return df

# -----------------------------------------------------------------------------


def predict_from_model(
    model: Callable,
    config: TrainerConfig,
    dataset: "sf.Dataset",
    outcomes: Union[str, List[str]],
    bags: Union[str, np.ndarray, List[str]],
    *,
    attention: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[np.ndarray]]]:
    """Generate predictions from a model.

    Args:
        model (torch.nn.Module): Model from which to generate predictions.
        config (TrainerConfig): Configuration for the MIL model.
        dataset (sf.Dataset): Dataset from which to generation predictions.
        outcomes (str, list(str)): Outcomes.
        bags (str, list(str)): Path to bags, or list of bag file paths.
            Each bag should contain PyTorch array of features from all tiles in
            a slide, with the shape ``(n_tiles, n_features)``.

    Returns:
        pd.DataFrame: Dataframe of predictions.

        list(np.ndarray): Attention scores (if ``attention=True``)
    """

    # Prepare labels.
    labels, unique = dataset.labels(outcomes, format='id')

    # Prepare bags and targets.
    if isinstance(bags, str):
        bags = dataset.pt_files(bags)
    else:
        bags = np.array(bags)
    slides = [path_to_name(f) for f in bags]
    y_true = np.array([labels[s] for s in slides])

    # Inference.
    if (isinstance(config, TrainerConfigCLAM)
       or isinstance(config.model_config, ModelConfigCLAM)):
        y_pred, y_att = _predict_clam(model, bags, attention=attention)
    else:
        y_pred, y_att = _predict_transformer(model, bags, attention=attention)

    # Create dataframe.
    df_dict = dict(slide=slides, y_true=y_true)
    for i in range(y_pred.shape[-1]):
        df_dict[f'y_pred{i}'] = y_pred[:, i]
    df = pd.DataFrame(df_dict)

    if attention:
        return df, y_att
    else:
        return df


def generate_attention_heatmaps(
    outdir: str,
    dataset: "sf.Dataset",
    bags: Union[List[str], np.ndarray],
    attention: Union[np.ndarray, List[np.ndarray]],
    **kwargs
) -> None:
    """Generate and save attention heatmaps.

    Args:
        outdir (str): Path at which to save heatmap images.
        dataset (sf.Dataset): Dataset.
        bags (str, list(str)): List of bag file paths.
            Each bag should contain PyTorch array of features from all tiles in
            a slide, with the shape ``(n_tiles, n_features)``.
        attention (list(np.ndarray)): Attention scores for each slide.
            Length of ``attention`` should equal the length of ``bags``.

    """
    assert len(bags) == len(attention)
    if not exists(outdir):
        os.makedirs(outdir)
    pb = Progress(transient=True)
    task = pb.add_task('Generating heatmaps...', total=len(bags))
    pb.start()
    with sf.util.cleanup_progress(pb):
        for i, bag in enumerate(bags):
            pb.advance(task)
            slidename = sf.util.path_to_name(bag)
            slide_path = dataset.find_slide(slide=slidename)
            locations_file = join(dirname(bag), f'{slidename}.index.npz')
            if slide_path is None:
                log.info(f"Unable to find slide {slidename}")
                continue
            if not exists(locations_file):
                log.info(
                    f"Unable to find locations index file for {slidename}"
                )
                continue
            sf.util.location_heatmap(
                locations=np.load(locations_file)['arr_0'],
                values=attention[i],
                slide=slide_path,
                tile_px=dataset.tile_px,
                tile_um=dataset.tile_um,
                outdir=outdir,
                **kwargs
            )
    log.info(f"Attention heatmaps saved to [green]{outdir}[/]")

# -----------------------------------------------------------------------------

def _predict_clam(
    model: Callable,
    bags: Union[np.ndarray, List[str]],
    attention: bool = False
) -> Tuple[np.ndarray, List[np.ndarray]]:
    import torch
    device = torch.device('cuda')
    y_pred = []
    y_att  = []
    log.info("Generating predictions...")
    for bag in bags:
        loaded = torch.load(bag).to(device)
        with torch.no_grad():
            logits, att, _ = model(loaded, return_attention=True)
            if attention:
                y_att.append(np.squeeze(att.cpu().numpy()))
            y_pred.append(torch.nn.functional.softmax(logits, dim=1).cpu().numpy())
    yp = np.concatenate(y_pred, axis=0)
    return yp, y_att


def _predict_transformer(
    model: Callable,
    bags: Union[np.ndarray, List[str]],
    attention: bool = False
) -> Tuple[np.ndarray, List[np.ndarray]]:
    import torch
    device = torch.device('cuda')
    y_pred = []
    y_att  = []
    log.info("Generating predictions...")
    for bag in bags:
        loaded = torch.load(bag).to(device)
        with torch.no_grad():
            lens = torch.from_numpy(np.array([loaded.shape[0]])).to(device)
            loaded = torch.unsqueeze(loaded, dim=0)
            model_out = model(loaded, lens)
            if attention:
                att = model.calculate_attention(loaded, lens).cpu().numpy()
                y_att.append(np.squeeze(att))
            y_pred.append(torch.nn.functional.softmax(model_out, dim=1).cpu().numpy())
    yp = np.concatenate(y_pred, axis=0)
    return yp, y_att
