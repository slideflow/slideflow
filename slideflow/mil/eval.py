"""Tools for evaluation MIL models."""

import os
import pandas as pd
import slideflow as sf
import numpy as np
from rich.progress import Progress
from os.path import join, exists, isdir, dirname
from typing import Union, List, Optional, Callable, Tuple, Any, TYPE_CHECKING
from slideflow import Dataset, log, errors
from slideflow.util import path_to_name
from slideflow.stats.metrics import ClassifierMetrics
from .features import MILFeatures
from ._params import (
    _TrainerConfig, ModelConfigCLAM, TrainerConfigCLAM
)
from .utils import build_bag_encoder

if TYPE_CHECKING:
    import torch
    from slideflow.norm import StainNormalizer
    from slideflow.model.base import BaseFeatureExtractor

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
        config (:class:`slideflow.mil.TrainerConfigFastAI` or 
        :class:`slideflow.mil.TrainerConfigCLAM`):
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

    # Prepare ground-truth labels
    labels, unique = dataset.labels(outcomes, format='id')

    # Prepare bags and targets
    slides = list(labels.keys())
    if isinstance(bags, str):
        bags = dataset.pt_files(bags)
    else:
        bags = np.array([b for b in bags if path_to_name(b) in slides])

    # Ensure slide names are sorted according to the bags.
    slides = [path_to_name(b) for b in bags]

    y_true = np.array([labels[s] for s in slides])

    # Detect feature size from bags
    n_features = torch.load(bags[0]).shape[-1]
    n_out = len(unique)

    # Load model
    model, config = _load_model_weights(
        weights, config, n_features=n_features, n_out=n_out
    )

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
        m = ClassifierMetrics(
            y_true=(y_true == idx).astype(int), y_pred=y_pred[:, idx])
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

    # Print categorical metrics, including per-category accuracy
    outcome_name = outcomes if isinstance(
        outcomes, str) else '-'.join(outcomes)
    metrics_df = df.rename(
        columns={c: f"{outcome_name}-{c}" for c in df.columns if c != 'slide'}
    )
    sf.stats.metrics.categorical_metrics(metrics_df, level='slide')

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


def predict_slide(
    model: str,
    slide: sf.WSI,
    encoder: Optional["BaseFeatureExtractor"] = None,
    *,
    normalizer: Optional["StainNormalizer"] = None,
    config: Optional[_TrainerConfig] = None,
    attention: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Generate predictions (and attention) for a single slide.

    Args:
        model (str): Path to MIL model.
        slide (str): Path to slide.
        encoder (:class:`slideflow.mil.BaseFeatureExtractor`, optional):
            Feature encoder. If not provided, will attempt to auto-detect
            encoder from model.

            .. note::
                If the encoder has a stain normalizer, this will be used to
                normalize the slide before extracting features.

    Keyword Args:
        normalizer (:class:`slideflow.stain.StainNormalizer`, optional):
            Stain normalizer. If not provided, will attempt to use stain
            normalizer from encoder.
        config (:class:`slideflow.mil.TrainerConfigFastAI` or :class:`slideflow.mil.TrainerConfigCLAM`):
            Configuration for building model. If None, will attempt to read
            ``mil_params.json`` from the model directory and load saved
            configuration. Defaults to None.
        attention (bool): Whether to return attention scores. Defaults to
            False.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: Predictions and attention scores.
    """
    # Try to auto-determine the encoder
    if encoder is None:
        encoder, detected_normalizer = build_bag_encoder(
            model, allow_errors=True)
        if encoder is None:
            raise ValueError(
                "Unable to auto-detect feature encoder used for model {}. "
                "Please specify an encoder.".format(model)
            )

    # Determine stain normalization
    if detected_normalizer is not None and normalizer is not None:
        log.warning(
            "Bags were generated with a stain normalizer, but a different stain "
            "normalizer was provided to this function. Overriding with provided "
            "stain normalizer."
        )
    elif detected_normalizer is not None:
        normalizer = detected_normalizer

    # Convert slide to bags
    masked_bags = encoder(slide, normalizer=normalizer)
    bags = np.ma.getdata(masked_bags[~masked_bags.mask.any(axis=2)])
    bags = np.expand_dims(bags, axis=0).astype(np.float32)

    # Load model
    model_fn, config = _load_model_weights(model, config, bags.shape[2], 2)

    # Generate predictions.
    if (isinstance(config, TrainerConfigCLAM)
       or isinstance(config.model_config, ModelConfigCLAM)):
        y_pred, y_att = _predict_clam(model_fn, bags, attention=attention)
    else:
        y_pred, y_att = _predict_mil(
            model_fn, bags, attention=attention, use_lens=config.model_config.use_lens
        )

    return y_pred, y_att


def predict_from_model(
    model: Callable,
    config: _TrainerConfig,
    dataset: "sf.Dataset",
    outcomes: Union[str, List[str]],
    bags: Union[str, np.ndarray, List[str]],
    *,
    attention: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[np.ndarray]]]:
    """Generate predictions for a dataset from a saved MIL model.

    Args:
        model (torch.nn.Module): Model from which to generate predictions.
        config (:class:`slideflow.mil.TrainerConfigFastAI` or 
        :class:`slideflow.mil.TrainerConfigCLAM`):
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

    # Ensure slide names are sorted according to the bags.
    slides = [path_to_name(b) for b in bags]

    y_true = np.array([labels[s] for s in slides])

    # Inference.
    if (isinstance(config, TrainerConfigCLAM)
       or isinstance(config.model_config, ModelConfigCLAM)):
        y_pred, y_att = _predict_clam(model, bags, attention=attention)
    else:
        y_pred, y_att = _predict_mil(
            model, bags, attention=attention, 
            use_lens=config.model_config.use_lens
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


def generate_mil_features(
    weights: str,
    config: _TrainerConfig,
    dataset: "sf.Dataset",
    outcomes: Union[str, List[str]],
    bags: Union[str, np.ndarray, List[str]]
):
    """Generate activations weights from the last layer of an MIL model.

    Returns MILFeatures object.

    Args:
        weights (str): Path to model weights to load.
        dataset (:class:`slideflow.Dataset`): Dataset.
        outcomes (str, list(str)): Outcomes.
        bags (str, list(str)): fPath to bags, or list of bag file paths.
            Each bag should contain PyTorch array of features from all tiles in
            a slide, with the shape ``(n_tiles, n_features)``.
        config (:class:`slideflow.mil.TrainerConfigFastAI` or 
        :class:`slideflow.mil.TrainerConfigCLAM`):
            Configuration for building model. If ``weights`` is a path to a
            model directory, will attempt to read ``mil_params.json`` from this
            location and load saved configuration. Defaults to None.
    """

    import torch

    if isinstance(config, TrainerConfigCLAM):
        raise NotImplementedError
    # Check for correct model
    acceptable_models = ['transmil', 'attention_mil', 'clam_sb', 'clam_mb']
    if config.model_config.model.lower() not in acceptable_models:
        raise errors.ModelError(
            f"Model {config.model_config.model} is not supported.")

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

    # Ensure slide names are sorted according to the bags.
    slides = [path_to_name(b) for b in bags]

    # Detect feature size from bags
    n_features = torch.load(bags[0]).shape[-1]
    n_out = len(unique)

    # Build the model
    if isinstance(config, TrainerConfigCLAM):
        config_size = config.model_fn.sizes[config.model_config.model_size]
        _size = [n_features] + config_size[1:]
        model = config.model_fn(size=_size)
        log.info(
            f"Building model {config.model_fn.__name__} (size={_size})")
    elif isinstance(config.model_config, ModelConfigCLAM):
        config_size = config.model_fn.sizes[config.model_config.model_size]
        _size = [n_features] + config_size[1:]
        model = config.model_fn(size=_size)
        log.info(
            f"Building model {config.model_fn.__name__} (size={_size})")
    else:
        model = config.model_fn(n_features, n_out)
        log.info(f"Building model {config.model_fn.__name__} "
                 f"(in={n_features}, out={n_out})")
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

    annotations = dataset.annotations
    return MILFeatures(model, bags, slides, annotations)


def generate_attention_heatmaps(
    outdir: str,
    dataset: "sf.Dataset",
    bags: Union[List[str], np.ndarray],
    attention: Union[np.ndarray, List[np.ndarray]],
    **kwargs
) -> None:
    """Generate and save attention heatmaps for a dataset.

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


def _load_model_weights(
    weights: str,
    config: Optional[_TrainerConfig],
    n_features: int,
    n_out: int
) -> Tuple["torch.nn.Module", _TrainerConfig]:
    """Load weights and build model.

    Args:
        weights (str): Path to model weights.
        config (:class:`slideflow.mil.TrainerConfigFastAI` or :class:`slideflow.mil.TrainerConfigCLAM`):
            Configuration for building model. If ``weights`` is a path to a
            model directory, will attempt to read ``mil_params.json`` from this
            location and load saved configuration. Defaults to None.
        n_features (int): Number of features in the input data.
        n_out (int): Number of output classes.

    Returns:
        :class:`torch.nn.Module`: Loaded model.
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

    # Build the model
    if isinstance(config, TrainerConfigCLAM):
        config_size = config.model_fn.sizes[config.model_config.model_size]
        _size = [n_features] + config_size[1:]
        model = config.model_fn(size=_size)
        log.info(f"Building model {config.model_fn.__name__} (size={_size})")
    elif isinstance(config.model_config, ModelConfigCLAM):
        config_size = config.model_fn.sizes[config.model_config.model_size]
        _size = [n_features] + config_size[1:]
        model = config.model_fn(size=_size)
        log.info(f"Building model {config.model_fn.__name__} (size={_size})")
    else:
        model = config.model_fn(n_features, n_out)
        log.info(f"Building model {config.model_fn.__name__} "
                 f"(in={n_features}, out={n_out})")
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
    return model, config


def _load_bag(bag: Union[str, np.ndarray, "torch.Tensor"]) -> "torch.Tensor":
    """Load bag from file or convert to torch.Tensor."""
    import torch

    if isinstance(bag, str):
        return torch.load(bag)
    elif isinstance(bag, np.ndarray):
        return torch.from_numpy(bag)
    elif isinstance(bag, torch.Tensor):
        return bag
    else:
        raise ValueError(
            "Unrecognized bag type '{}'".format(type(bag))
        )


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
    y_att = []
    log.info("Generating predictions...")
    for bag in bags:
        loaded = _load_bag(bag).to(device)
        with torch.no_grad():
            if clam_kw:
                logits, att, _ = model(loaded, **clam_kw)
            else:
                logits, att = model(loaded, **clam_kw)
            if attention:
                y_att.append(np.squeeze(att.cpu().numpy()))
            y_pred.append(torch.nn.functional.softmax(
                logits, dim=1).cpu().numpy())
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
    y_att = []
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
        loaded = torch.unsqueeze(_load_bag(bag).to(device), dim=0)
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
            y_pred.append(torch.nn.functional.softmax(
                model_out, dim=1).cpu().numpy())
    yp = np.concatenate(y_pred, axis=0)
    return yp, y_att
