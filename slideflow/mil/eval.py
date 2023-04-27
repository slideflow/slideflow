"""Tools for evaluation MIL models."""

import os
import pandas as pd
import slideflow as sf
import numpy as np
from rich.progress import Progress
from os.path import join, exists, isdir, dirname
from typing import Union, List, Optional, Callable, Tuple, Any
from slideflow import Dataset, log, errors
from slideflow.util import path_to_name
from slideflow.stats.metrics import ClassifierMetrics
from ._params import (
    _TrainerConfig, ModelConfigCLAM, TrainerConfigCLAM
)

# -----------------------------------------------------------------------------

def eval_mil(
    weights: str,
    dataset: Dataset,
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]],
    config: Optional[_TrainerConfig] = None,
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
        config (:class:`slideflow.mil.TrainerConfigFastAI` or :class:`slideflow.mil.TrainerConfigCLAM`):
            Configuration for building model. If ``weights`` is a path to a
            model directory, will attempt to read ``mil_params.json`` from this
            location and load saved configuration. Defaults to None.

    Keyword arguments:
        outdir (str): Path at which to save results.
        attention_heatmaps (bool): Generate attention heatmaps for slides.
            Defaults to False.
        interpolation (str, optional): Interpolation strategy for smoothing
            attention heatmaps. Defaults to 'bicubic'.
        cmap (str, optional): Matplotlib colormap for heatmap. Can be any
            valid matplotlib colormap. Defaults to 'inferno'.
        norm (str, optional): Normalization strategy for assigning heatmap
            values to colors. Either 'two_slope', or any other valid value
            for the ``norm`` argument of ``matplotlib.pyplot.imshow``.
            If 'two_slope', normalizes values less than 0 and greater than 0
            separately. Defaults to None.

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
    slides = list(labels.keys())
    if isinstance(bags, str):
        bags = dataset.pt_files(bags)
    else:
        bags = np.array([b for b in bags if path_to_name(b) in slides])

    # Handle the case where some bags are missing.
    if len(bags) != len(slides):
        slides = [path_to_name(b) for b in bags]

    y_true = np.array([labels[s] for s in slides])

    # Detect feature size from bags
    n_features = torch.load(bags[0]).shape[-1]
    n_out = len(unique)

    log.info(f"Building model {config.model_fn.__name__}")
    if isinstance(config, TrainerConfigCLAM):
        config_size = config.model_fn.sizes[config.model_config.model_size]
        model = config.model_fn(size=[n_features] + config_size[1:])
    elif isinstance(config.model_config, ModelConfigCLAM):
        model = config.model_fn(size=[n_features, 256, 128])
    else:
        model = config.model_fn(n_features, n_out)
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
    if hasattr(model, 'relocate'):
        model.relocate()  # type: ignore
    model.eval()

    # Inference.
    if (isinstance(config, TrainerConfigCLAM)
       or isinstance(config.model_config, ModelConfigCLAM)):
        y_pred, y_att = _predict_clam(model, bags, attention=True)
    else:
        y_pred, y_att = _predict_mil(
            model, bags, attention=True, use_lens=config.model_config.use_lens
        )

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
    if y_att:
        att_path = join(model_dir, 'attention')
        if not exists(att_path):
            os.makedirs(att_path)
        for slide, att in zip(slides, y_att):
            if 'SF_ALLOW_ZIP' in os.environ and os.environ['SF_ALLOW_ZIP'] == '0':
                out_path = join(att_path, f'{slide}_att.npy')
                np.save(out_path, att)
            else:
                out_path = join(att_path, f'{slide}_att.npz')
                np.savez(out_path, att)
        log.info(f"Attention scores exported to [green]{out_path}[/]")

    # Attention heatmaps
    if y_att and attention_heatmaps:
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
    config: _TrainerConfig,
    dataset: "sf.Dataset",
    outcomes: Union[str, List[str]],
    bags: Union[str, np.ndarray, List[str]],
    *,
    attention: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[np.ndarray]]]:
    """Generate predictions from a model.

    Args:
        model (torch.nn.Module): Model from which to generate predictions.
        config (:class:`slideflow.mil.TrainerConfigFastAI` or :class:`slideflow.mil.TrainerConfigCLAM`):
            Configuration for the MIL model.
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
    slides = list(labels.keys())
    if isinstance(bags, str):
        bags = dataset.pt_files(bags)
    else:
        bags = np.array([b for b in bags if path_to_name(b) in slides])

    # Handle the case where some bags are missing.
    if len(bags) != len(slides):
        slides = [path_to_name(b) for b in bags]

    y_true = np.array([labels[s] for s in slides])

    # Inference.
    if (isinstance(config, TrainerConfigCLAM)
       or isinstance(config.model_config, ModelConfigCLAM)):
        y_pred, y_att = _predict_clam(model, bags, attention=attention)
    else:
        y_pred, y_att = _predict_mil(
            model, bags, attention=attention, use_lens=config.model_config.use_lens
        )

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

    Keyword args:
        interpolation (str, optional): Interpolation strategy for smoothing
            heatmap. Defaults to 'bicubic'.
        cmap (str, optional): Matplotlib colormap for heatmap. Can be any
            valid matplotlib colormap. Defaults to 'inferno'.
        norm (str, optional): Normalization strategy for assigning heatmap
            values to colors. Either 'two_slope', or any other valid value
            for the ``norm`` argument of ``matplotlib.pyplot.imshow``.
            If 'two_slope', normalizes values less than 0 and greater than 0
            separately. Defaults to None.


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
            npy_loc_file = locations_file[:-1] + 'y'
            if slide_path is None:
                log.info(f"Unable to find slide {slidename}")
                continue
            if exists(locations_file):
                locations = np.load(locations_file)['arr_0']
            elif exists(npy_loc_file):
                locations = np.load(npy_loc_file)
            else:
                log.info(
                    f"Unable to find locations index file for {slidename}"
                )
                continue
            sf.util.location_heatmap(
                locations=locations,
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
    attention: bool = False,
    device: Optional[Any] = None
) -> Tuple[np.ndarray, List[np.ndarray]]:

    import torch
    from slideflow.mil.models import CLAM_MB, CLAM_SB

    if isinstance(model, (CLAM_MB, CLAM_SB)):
        clam_kw = dict(return_attention=True)
    else:
        clam_kw = {}
        attention = False

    # Auto-detect device.
    if device is None:
        if next(model.parameters()).is_cuda:
            log.debug("Auto device detection: using CUDA")
            device = torch.device('cuda')
        else:
            log.debug("Auto device detection: using CPU")
            device = torch.device('cpu')
    elif isinstance(device, str):
        log.debug(f"Using {device}")
        device = torch.device(device)

    y_pred = []
    y_att  = []
    log.info("Generating predictions...")
    for bag in bags:
        loaded = torch.load(bag).to(device)
        with torch.no_grad():
            if clam_kw:
                logits, att, _ = model(loaded, **clam_kw)
            else:
                logits, att = model(loaded, **clam_kw)
            if attention:
                y_att.append(np.squeeze(att.cpu().numpy()))
            y_pred.append(torch.nn.functional.softmax(logits, dim=1).cpu().numpy())
    yp = np.concatenate(y_pred, axis=0)
    return yp, y_att


def _predict_mil(
    model: Callable,
    bags: Union[np.ndarray, List[str]],
    attention: bool = False,
    attention_pooling: str = 'avg',
    use_lens: bool = False,
    device: Optional[Any] = None
) -> Tuple[np.ndarray, List[np.ndarray]]:

    import torch

    # Auto-detect device.
    if device is None:
        if next(model.parameters()).is_cuda:
            log.debug("Auto device detection: using CUDA")
            device = torch.device('cuda')
        else:
            log.debug("Auto device detection: using CPU")
            device = torch.device('cpu')
    elif isinstance(device, str):
        log.debug(f"Using {device}")
        device = torch.device(device)

    y_pred = []
    y_att  = []
    log.info("Generating predictions...")
    if attention and not hasattr(model, 'calculate_attention'):
        log.warning(
            "Model '{}' does not have a method 'calculate_attention'. "
            "Unable to calculate or display attention heatmaps.".format(
                model.__class__.__name__
            )
        )
        attention = False
    for bag in bags:
        loaded = torch.load(bag).to(device)
        loaded = torch.unsqueeze(loaded, dim=0)
        with torch.no_grad():
            if use_lens:
                lens = torch.from_numpy(np.array([loaded.shape[1]])).to(device)
                model_args = (loaded, lens)
            else:
                model_args = (loaded,)
            model_out = model(*model_args)
            if attention:
                att = torch.squeeze(model.calculate_attention(*model_args))
                if len(att.shape) == 2:
                    # Attention needs to be pooled
                    if attention_pooling == 'avg':
                        att = torch.mean(att, dim=-1)
                    elif attention_pooling == 'max':
                        att = torch.amax(att, dim=-1)
                    else:
                        raise ValueError(
                            "Unrecognized attention pooling strategy '{}'".format(
                                attention_pooling
                            )
                        )
                y_att.append(att.cpu().numpy())
            y_pred.append(torch.nn.functional.softmax(model_out, dim=1).cpu().numpy())
    yp = np.concatenate(y_pred, axis=0)
    return yp, y_att
