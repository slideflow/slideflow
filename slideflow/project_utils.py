"""Utility functions for slideflow.Project."""

import re
import os
import requests
import tempfile
import logging
import pandas as pd
from collections import defaultdict
from functools import wraps
from os.path import dirname, exists, join, realpath, isdir
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import slideflow as sf
from slideflow import errors
from slideflow.util import log, relative_path

# Set the tensorflow logger
if sf.getLoggingLevel() == logging.DEBUG:
    logging.getLogger('tensorflow').setLevel(logging.DEBUG)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
else:
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def auto_dataset(method: Callable):
    """Wrapper to convert filter arguments into a dataset."""
    @wraps(method)
    def _impl(obj, model=None, *args, **kwargs):
        return _filters_to_dataset(obj, method, model, *args, **kwargs)
    return _impl


def auto_dataset_allow_none(method: Callable):
    """Wrapper function to convert filter arguments to a dataset, allowing
    errors."""
    @wraps(method)
    def _impl(obj, model=None, dataset=None, *args, **kwargs):
        try:
            return _filters_to_dataset(obj, method, model, *args, dataset=dataset, **kwargs)
        except errors.ModelParamsNotFoundError:
            if 'dataset' not in kwargs:
                return method(obj, model, dataset=None, *args, **kwargs)
            else:
                raise
    return _impl


def _filters_to_dataset(obj, method, model, *args, **kwargs):
    filter_keys = ['filters', 'filter_blank', 'min_tiles']
    has_filters = any([k in kwargs for k in filter_keys])
    if model is not None:
        try:
            config = sf.util.get_model_config(model)
        except (errors.ModelParamsNotFoundError, TypeError):
            if 'dataset' in kwargs:
                config = None
            else:
                raise
    if has_filters and 'dataset' in kwargs:
        k_s = ', '.join(filter_keys)
        raise errors.ProjectError(
            f"Cannot supply both `dataset` and filter arguments ({k_s})."
            " Instead, supply a filtered dataset (Dataset.filter(...))"
        )
    if 'dataset' in kwargs and kwargs['dataset']:
        if model is not None:
            if config is not None:
                kwargs['dataset']._assert_size_matches_hp(config['hp'])
            return method(obj, model, *args, **kwargs)
        else:
            return method(obj, *args, **kwargs)
    else:
        if model is None:
            raise errors.ProjectError("Missing argument 'model'.")
        dataset = obj.dataset(
            tile_px=config['hp']['tile_px'],
            tile_um=config['hp']['tile_um'],
            **{k: v for k, v in kwargs.items() if k in filter_keys},
            verification='slides'
        )
        kwargs['dataset'] = dataset
        return method(obj, model, *args, **kwargs)

# -----------------------------------------------------------------------------


def _project_config(
    name: str = 'MyProject',
    annotations: str = './annotations.csv',
    dataset_config: str = './datasets.json',
    sources: Optional[Union[str, List[str]]] = None,
    models_dir: str = './models',
    eval_dir: str = './eval'
) -> Dict:
    args = locals()
    args['slideflow_version'] = sf.__version__
    if sources is None:
        args['sources'] = []
    elif isinstance(sources, str):
        args['sources'] = [sources]
    return args


def _heatmap_worker(
    slide: str,
    args: SimpleNamespace,
    kwargs: Any
) -> None:
    """Heatmap worker for :meth:`slideflow.Project.generate_heatmaps.`

    Any function loading a slide must be kept in an isolated process, as
    loading more than one slide in a single process causes instability / hangs.
    I suspect this is a libvips or openslide issue but I haven't been able to
    identify the root cause. Isolating processes when multiple slides are to be
    processed sequentially is a workaround, hence the process-isolated worker.

    Args:
        slide (str): Path to slide.
        args (SimpleNamespace): Namespace of heatmap arguments.
        kwargs (dict): kwargs for heatmap.save()
    """
    sf.setLoggingLevel(args.verbosity)
    heatmap = sf.Heatmap(slide,
                         model=args.model,
                         stride_div=args.stride_div,
                         rois=args.rois,
                         roi_method=args.roi_method,
                         batch_size=args.batch_size,
                         img_format=args.img_format,
                         num_threads=args.num_threads)
    heatmap.save(args.outdir, **kwargs)


def _train_worker(
    datasets: Tuple[sf.Dataset, sf.Dataset],
    model_kw: Dict,
    training_kw: Dict,
    results_dict: Dict,
    verbosity: int
) -> None:
    """Internal function to execute model training in an isolated process."""
    sf.setLoggingLevel(verbosity)
    train_dts, val_dts = datasets
    trainer = sf.model.build_trainer(**model_kw)
    results = trainer.train(train_dts, val_dts, **training_kw)
    results_dict.update({model_kw['name']: results})


def _setup_input_labels(
    dts: sf.Dataset,
    inpt_headers: List[str],
    val_dts: Optional[sf.Dataset] = None
) -> Tuple[Dict, List[int], Dict]:
    '''
    Args:
        dts (:class:`slideflow.Dataset`): Training dataset.
        inpt_headers (list(str)): Annotation headers for slide-level input.
        val_dts (:class:`slideflow.Dataset`, optional): Validation dataset.
            Used for harmonizing categorical labels to ensure consistency.
    '''
    # Dict mapping input headers to # of labels
    feature_len = {}
    # Nested dict mapping input vars to either category ID dicts or 'float'
    inpt_classes = {}  # type: Dict
    # Dict mapping slide names to slide-level model input
    model_inputs = defaultdict(list)  # type: Dict
    for inpt in inpt_headers:
        if val_dts is not None:
            is_float = dts.is_float(inpt) and val_dts.is_float(inpt)
        else:
            is_float = dts.is_float(inpt)
        kind = 'float' if is_float else 'categorical'
        log.info(f"Adding input variable [blue]{inpt}[/] as {kind}")

        labels, unique = dts.labels(inpt, use_float=is_float)
        slides = list(labels.keys())

        if is_float:
            feature_len[inpt] = 1
            inpt_classes[inpt] = 'float'
            for slide in slides:
                _label = labels[slide]
                if isinstance(_label, list) and len(_label) == 1:
                    _label = _label[0]
                model_inputs[slide] += [_label]
        else:
            feature_len[inpt] = len(unique)
            inpt_classes[inpt] = dict(zip(range(len(unique)), unique))
            for slide in slides:
                onehot_label = sf.util.to_onehot(
                    labels[slide], len(unique)  # type: ignore
                )
                # Concatenate onehot labels together
                model_inputs[slide] += list(onehot_label)

    feature_sizes = [feature_len[i] for i in inpt_headers]
    return inpt_classes, feature_sizes, model_inputs


def get_validation_settings(**kwargs: Any) -> SimpleNamespace:
    """Returns a namespace of validation settings.

    Args:
        strategy (str): Validation dataset selection strategy.
            Options include bootstrap, k-fold, k-fold-manual,
            k-fold-preserved-site, fixed, and none. Defaults to 'k-fold'.
        k_fold (int): Total number of K if using K-fold validation.
            Defaults to 3.
        k (int): Iteration of K-fold to train, starting at 1. Defaults to None
            (training all k-folds).
        k_fold_header (str): Annotations file header column for manually
            specifying k-fold. Only used if validation strategy is
            'k-fold-manual'. Defaults to None.
        k_fold_header (str): Annotations file header column for manually
            specifying k-fold or for preserved-site cross validation. Only used
            if validation strategy is 'k-fold-manual' or
            'k-fold-preserved-site'. Defaults to None for k-fold-manual and
            'site' for k-fold-preserved-site.
        fraction (float): Fraction of dataset to use for validation testing,
            if strategy is 'fixed'.
        source (str): Dataset source to use for validation.
            Defaults to None (same as training).
        annotations (str): Path to annotations file for validation dataset.
            Defaults to None (same as training).
        filters (dict): Filters dictionary to use for validation dataset.
            See :meth:`slideflow.Dataset.filter` for more information.
            Defaults to None (same as training).

    """
    if 'strategy' in kwargs and kwargs['strategy'] == 'k-fold-preserved-site':
        default_header = 'site'
    else:
        default_header = None
    args_dict = {
        'strategy': 'k-fold',
        'k_fold': 3,
        'k': None,
        'k_fold_header': default_header,
        'fraction': None,
        'source': None,
        'annotations': None,
        'filters': None,
        'dataset': None,
    }
    if 'dataset' in kwargs and len(kwargs) > 1:
        raise ValueError(
            "Cannot supply validation dataset settings if val_dataset "
            "is supplied. Got: {}".format(
                ', '.join(['val_'+k for k in kwargs.keys() if k != 'dataset'])
            )
        )
    if 'dataset' in kwargs:
        args_dict['strategy'] = 'fixed'
    for k in kwargs:
        if k not in args_dict:
            raise ValueError(f"Unrecognized validation setting {k}")
        args_dict[k] = kwargs[k]
    args = SimpleNamespace(**args_dict)

    if args.strategy is None:
        args.strategy = 'none'
    if (args.k_fold_header is None and args.strategy == 'k-fold-manual'):
        raise ValueError(
            "val_strategy 'k-fold-manual' requires 'k_fold_header'"
        )
    return args


def add_source(
    name: str,
    path: str,
    slides: Optional[str] = None,
    roi: Optional[str] = None,
    tiles: Optional[str] = None,
    tfrecords: Optional[str] = None,
) -> None:
    """Adds a dataset source to a dataset configuration file.

    Args:
        name (str): Source name.
        slides (str): Path to directory containing slides.
        roi (str): Path to directory containing CSV ROIs.
        tiles (str): Path to directory in which to store extracted tiles.
        tfrecords (str): Directory to store TFRecords of extracted tiles.
        path (str): Path to dataset configuration file.
    """

    try:
        datasets_data = sf.util.load_json(path)
    except FileNotFoundError:
        datasets_data = {}
    datasets_data.update({name: {
        'slides': slides,
        'roi': roi,
        'tiles': tiles,
        'tfrecords': tfrecords,
    }})
    sf.util.write_json(datasets_data, path)
    log.info(f'Saved dataset source {name} to {path}')


def load_sources(path: str) -> Tuple[Dict, List]:
    """Loads datasets configuration dictionaries from a datasets.json file."""
    try:
        sources_data = sf.util.load_json(path)
        sources = list(sources_data.keys())
        sources.sort()
    except FileNotFoundError:
        sources_data = {}
        sources = []
    return sources_data, sources


# --- Ensembling utility functions. -------------------------------------------

def create_ensemble_dataframe(
    ensemble_path: str,
    member_id: int,
    kfold_path: str,
    level: str,
    kfold_int: Optional[int] = None,
    epoch: Optional[int] = None
) -> pd.DataFrame:
    """Create an ensemble prediction dataframe from a given ensemble member.

    From a given ensemble member (and specified k-fold and epoch), initiates a
    dataframe for storing predictions for all ensemble members.

    Args:
        ensemble_path (str): Path to root ensemble directory.
        member_id (int): ID of the ensemble member.
        kfold_path (str): Path to target k-fold model for a specific member
            of the prediction ensemble.
        level (str): Prediction level, either 'slide', 'patient', or 'tile'.

    Keyword Args:
        kfold_int (int, optional): K-fold ID. Defaults to None
        epoch (int, optional): Epoch. Defaults to None

    Returns:
        DataFrame of ensemble predictions.

    """
    # Find the specified predictions file for the ensemble member.
    if kfold_int is None or epoch is None:
        pred_file = find_matching_file(kfold_path, f"{level}_predictions")
        save_format = predict_file_type(kfold_path)
    else:
        pred_file = find_predictions(kfold_path, level=level, epoch=epoch)
        save_format = detect_predictions_format(pred_file)

    # Load the predictions into a dataframe.
    df = sf.util.load_predictions(pred_file)

    # Drop empty column ??
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # Determine sorting headers.
    sort_headers = [level] if level != 'tile' else ["slide", "loc_x", "loc_y"]

    # Create ensemble dataframe
    df = df.sort_values(by=sort_headers)
    headers = df.columns.tolist()
    new_headers = [s + f"_ens{member_id+1}" for s in headers]
    header_change = dict(zip(headers, new_headers))
    df.rename(columns=header_change, inplace=True)

    # Feather format request resetting index at creation.
    if save_format == 'feather':
        df = df.reset_index()

    # Save ensemble dataframe
    if kfold_int is None or epoch is None:
        out_path = join(
            ensemble_path,
            f'ensemble_{level}_predictions'
        )
    else:
        out_path = join(
            ensemble_path,
            f'ensemble_{level}_predictions_kfold{kfold_int}_epoch{epoch}'
        )
    save_dataframe(df, out_path, format=save_format)
    return df


def add_to_ensemble_dataframe(
    ensemble_path: str,
    member_id: int,
    kfold_path: str,
    level: str,
    kfold_int: Optional[int] = None,
    epoch: Optional[int] = None
) -> pd.DataFrame:
    """Add predictions from a given member to the ensemble predictions dataframe.

    From a given ensemble member (and specified k-fold and epoch), loads model
    predictions and adds to the dataframe storing predictions for all ensemble
    members.

    Args:
        ensemble_path (str): Path to root ensemble directory.
        member_id (int): ID of the ensemble member.
        kfold_path (str): Path to target k-fold model for a specific member
            of the prediction ensemble.
        level (str): Prediction level, either 'slide', 'patient', or 'tile'.

    Keyword Args:
        kfold_int (int, optional): K-fold ID. Defaults to None
        epoch (int, optional): Epoch. Defaults to None

    Returns:
        DataFrame of ensemble predictions.

    """
    # Find and load the ensemble predictions.
    if kfold_int is None or epoch is None:
        pred_file = find_matching_file(
            ensemble_path,
            f"ensemble_{level}_predictions",
            allow_missing=True
        )
    else:
        pred_file = find_matching_file(
            ensemble_path,
            f"ensemble_{level}_predictions_kfold{kfold_int}_epoch{epoch}",
            allow_missing=True)

    # If the dataframe has not yet been created, create it.
    if not pred_file:
        if kfold_int is None or epoch is None:
            return create_ensemble_dataframe(
                ensemble_path=ensemble_path,
                member_id=member_id,
                kfold_path=kfold_path,
                level=level
            )
        else:
            return create_ensemble_dataframe(
                ensemble_path=ensemble_path,
                member_id=member_id,
                kfold_path=kfold_path,
                kfold_int=kfold_int,
                epoch=epoch,
                level=level
            )

    # Find and load the member predictions.
    if kfold_int is None or epoch is None:
        member_file = find_matching_file(
            kfold_path,
            f"{level}_predictions"
        )
    else:
        member_file = find_predictions(kfold_path, level=level, epoch=epoch)  # slide_file_name

    member_df = sf.util.load_predictions(member_file)
    save_format = detect_predictions_format(pred_file)
    df = sf.util.load_predictions(pred_file)

    # Drop empty column ??
    if 'Unnamed: 0' in member_df.columns:
        member_df.drop(columns=['Unnamed: 0'], inplace=True)

    # Determine sorting headers.
    if level == 'tile':
        left_headers =[f'{h}_ens1' for h in ('slide', 'loc_x', 'loc_y')]
        right_headers =[f'{h}_ens{member_id+1}' for h in ('slide', 'loc_x', 'loc_y')]
    else:
        left_headers = [f'{level}_ens1']
        right_headers = [f'{level}_ens{member_id+1}']

    # Merge dataframes.
    member_headers = member_df.columns.tolist()
    ensemble_headers = [s + f"_ens{member_id+1}" for s in member_headers]
    header_change = dict(zip(member_headers, ensemble_headers))
    member_df.rename(columns=header_change, inplace=True)
    df = pd.merge(
        df,
        member_df,
        how="inner",
        left_on=left_headers,
        right_on=right_headers
    )
    df.drop(columns=right_headers, inplace=True)

    if "patient" in df.columns:
        df.drop(columns=[f'patient_ens{member_id+1}'], inplace=True)

    # Save dataframe.
    if kfold_int is None or epoch is None:
        out_path = join(
            ensemble_path,
            f'ensemble_{level}_predictions'
        )
    else:
        out_path = join(
            ensemble_path,
            f'ensemble_{level}_predictions_kfold{kfold_int}_epoch{epoch}'
        )
    save_dataframe(df, out_path, format=save_format)
    return df


def update_ensemble_dataframe_headers(
    ensemble_path: str,
    level: str,
    kfold_int: Optional[int] = None,
    epoch: Optional[int] = None
) -> pd.DataFrame:
    """Updates headers in the specified ensemble dataframe.

    Args:
        ensemble_path (str): Path to root ensemble directory.
        level (str, optional): Prediction level, either 'slide', 'patient',
            or'tile'.

    Keyword Args:
        epoch (int, optional): Epoch. Defaults to None
        kfold_int (int, optional): K-fold ID. Defaults to None

    Returns:
        DataFrame of ensemble predictions, with renamed headers.

    """
    # Find and load the ensemble predictions.
    if kfold_int is None or epoch is None:
        pred_file = find_matching_file(
            ensemble_path,
            f"ensemble_{level}_predictions"
        )
    else:
        pred_file = find_matching_file(
            ensemble_path,
            f"ensemble_{level}_predictions_kfold{kfold_int}_epoch{epoch}"
        )
    save_format = detect_predictions_format(pred_file)
    df = sf.util.load_predictions(pred_file)

    # Rename main tile/patient/slide headers.
    for colname in ["slide", "loc_x", "loc_y", "patient"]:
        if f"{colname}_ens1" in df.columns:
            df.rename(columns={f"{colname}_ens1": colname}, inplace=True)

    # Update the remaining headers, including an ensemble average.
    ensemble_headers = df.columns.tolist()
    member_headers = [h[:-5] for h in ensemble_headers if h.endswith('_ens1')]
    for header in member_headers:
        matching_ensemble_headers = [h for h in ensemble_headers if header in h]
        df[header] = df.loc[:, matching_ensemble_headers].mean(axis=1)

    # Move the patient column to the beginning.
    if "patient" in df.columns:
        patient_col = df.pop("patient")
        df.insert(0, "patient", patient_col)

    # Save dataframe.
    if kfold_int is None or epoch is None:
        out_path = join(
            ensemble_path,
            f'ensemble_{level}_predictions'
        )
    else:
        out_path = join(
            ensemble_path,
            f'ensemble_{level}_predictions_kfold{kfold_int}_epoch{epoch}'
        )
    save_dataframe(df, out_path, format=save_format)
    return df


def ensemble_train_predictions(ensemble_path: str) -> None:
    """Merge predictions for a given ensemble of models.

    Args:
        ensemble_path (str): Path to directory containing ensemble members,
            as generated by :meth:`slideflow.Project.train_ensemble()`.
    """
    if not exists(join(ensemble_path, 'ensemble_params.json')):
        raise OSError("Could not find ensemble_params.json.")

    # Path to each ensemble member.
    member_paths = sorted([
        join(ensemble_path, x) for x in os.listdir(ensemble_path)
        if isdir(join(ensemble_path, x))
    ])

    # Model directory names for each k-fold.
    # Each ensemble member will have these same folders.
    # For example, "00001-outcome-HP0-kfold1"
    kfold_dirs = sorted([
        x for x in os.listdir(member_paths[0])
        if isdir(join(member_paths[0], x))
    ])

    # Read the expected epochs from params.json.
    params = sf.util.load_json(join(ensemble_path, 'ensemble_params.json'))
    epochs = params['ensemble_epochs']

    for kfold_dir in kfold_dirs:
        for epoch in epochs:
            for member_id, member_path in enumerate(member_paths):
                kfold_path = join(member_path, kfold_dir)
                kfold_int = int(re.findall(r'\d', kfold_dir)[-1])

                # Create (or add to) the ensemble dataframe.
                for level in ('tile', 'slide', 'patient'):
                    add_to_ensemble_dataframe(
                        ensemble_path=ensemble_path,
                        member_id=member_id,
                        kfold_path=kfold_path,
                        kfold_int=kfold_int,
                        epoch=epoch,
                        level=level
                    )

            for level in ('tile', 'slide', 'patient'):
                update_ensemble_dataframe_headers(
                    ensemble_path=ensemble_path,
                    kfold_int=kfold_int,
                    epoch=epoch,
                    level=level
                )


def find_predictions(
    path: str,
    level: str,
    epoch: int,
    allow_missing: bool = False,
):
    """Find a predictions file at the given path.

    Args:
        path (str): Directory to search.
        level (str): 'patient', 'slide', or 'tile'.
        epoch (int): Epoch number.

    Keyword Args:
        allow_missing (bool): Do not raise an error if a match is not
            found. Defaults to False.

    Returns:
        Filename of predictions file

    Raises:
        OSError: If a valid file is not found, and ``allow_missing=False``.

        ValueError: If multiple files are found.

    """
    assert level in ('patient', 'slide', 'tile')
    return find_matching_file(
        path,
        f'{level}_predictions_val_epoch{epoch}',
        allow_missing=allow_missing
    )


def save_dataframe(df: pd.DataFrame, filename: str, format: str):
    """Saves a given dataframe to a path in the specified format.

    Args:
        df (pd.DataFrame): Dataframe of predictions.
        filename (str): Path to destination filename, without extension.
        format (str): Format in which to save the dataframe, either 'csv',
            'parquet', or 'feather'.

    Returns:
        None

    """
    if format == "csv":
        df.to_csv(f"{filename}.csv", index=False)
    elif format == "parquet":
        df.to_parquet(
            f"{filename}.parquet.gzip",
            index=False,
            compression='gzip')
    elif format == "feather":
        df.to_feather(f"{filename}.feather")
    else:
        raise ValueError(f"Unrecognized save format: {format}")


def detect_predictions_format(path: str):
    """Detect format of a given predictions dataframe.

    Args:
        path (str): Path to predictions file.

    Returns:
        str: format of predictions file (e.g. 'csv', 'parquet', 'feather')

    """
    if path.endswith("csv"):
        return 'csv'
    elif path.endswith("parquet") or path.endswith("gzip"):
        return 'parquet'
    elif path.endswith("feather"):
        return 'feather'
    else:
        return sf.util.path_to_ext(path)


def predict_file_type(path: str) -> str:
    """ \To return the format of a given predictions dataframe.

    Args:
        path (str): Path to predictions file.

    Returns:
        str: format of predictions file (e.g. 'csv', 'parquet', 'feather')
    """
    filenames = [x for x in os.listdir(path)
        if "predictions" in x]
    if len(filenames) == 0:
        raise FileNotFoundError
    else:
        filename = filenames[0]

    return detect_predictions_format(filename)


def find_matching_file(path: str, filename: str, allow_missing: bool = False):
    """Find a file at the given directory which startswith the given filename.

    Args:
        path (str): Directory to search.
        filename (str): Search for files that start with this string.
        allow_missing (bool): Allow missing files. If True and no file is found,
            returns None. If False and no matching file is found, will raise
            an OSError.

    Returns:
        str: Path to matching file.

    """
    results = [
        join(path, f) for f in os.listdir(path)
        if (f.startswith(filename) and os.path.isfile(join(path, f)))
    ]
    if not len(results):
        if allow_missing:
            return None
        else:
            raise OSError(
                f'Could not find file matching "{filename}" at {path}'
            )
    if len(results) > 1:
        raise ValueError(
            f'Multiple files matching "{filename}" found at {path}'
        )
    return results[0]


def get_matching_directory(curr_path: str, label: str) -> str:
    """Finding the path to the directory that has the term 'lable' in
        its name and is present in curr_path

    Args:
        curr_path (str): The path to the directory to seach in.
        lable (str): The string that should be present in the returned
            directory path

    Returns:
        str: Path to matching file.

    """
    list_of_dirs = _sorted_subdirectories(curr_path)
    try:
        curr_dir = [x for x in list_of_dirs
            if f'{label}' in str(x).split("/")[-1]]
        if len(curr_dir) > 1:
            raise ValueError(
                f'Multiple files matching "{label}" found at {curr_path}'
            )
        if len(curr_dir) == 0:
            raise IndexError(f'Directory matching "{label}" does not exist')
    except IndexError:
        raise IndexError(f'Directory matching "{label}" does not exist')

    return curr_dir[0]


def get_first_nested_directory(path):
    """To return the first element of a sorted list of paths to all the
    directories in 'path'

    Args:
        path (str): The path to the root directory

    Returns:
        Path to a directory

    """
    return _sorted_subdirectories(path)[0]


def _sorted_subdirectories(path):
    """To return a sorted list of paths to all the directories in 'path'

    Args:
        path (str): The path to the root directory

    Returns:
        List of paths

    """
    return sorted([
        join(path, x) for x in os.listdir(path)
        if isdir(join(path, x))
    ])


# -----------------------------------------------------------------------------

def interactive_project_setup(project_folder: str) -> Dict:
    """Guides user through project creation at the given folder,
    saving configuration to "settings.json".
    """
    if not exists(project_folder):
        os.makedirs(project_folder)
    project = {}  # type: Dict[str, Any]
    project['name'] = input('What is the project name? ')
    project['annotations'] = sf.util.path_input(
        'Annotations file location [./annotations.csv] ',
        root=project_folder,
        default='./annotations.csv',
        filetype='csv',
        verify=False
    )
    # Dataset configuration
    project['dataset_config'] = sf.util.path_input(
        'Dataset configuration file location [./datasets.json] ',
        root=project_folder,
        default='./datasets.json',
        filetype='json',
        verify=False
    )
    project['sources'] = []
    while not project['sources']:
        path = relative_path(project['dataset_config'], project_folder)
        datasets_data, sources = load_sources(path)
        print('[bold]Detected dataset sources:')
        if not len(sources):
            print(' [None]')
        else:
            for i, name in enumerate(sources):
                print(f' {i+1}. {name}')
            print(f' {len(sources)+1}. ADD NEW')
            valid_source_choices = [str(v) for v in range(1, len(sources)+2)]
            selection = sf.util.choice_input(
                'Which datasets should be used? ',
                valid_choices=valid_source_choices,
                multi_choice=True
            )
        if not len(sources) or str(len(sources)+1) in selection:
            # Create new dataset
            print(f"{'[bold]Creating new dataset source'}")
            source_name = input('What is the dataset source name? ')
            source_slides = sf.util.path_input(
                'Where are the slides stored? [./slides] ',
                root=project_folder,
                default='./slides',
                create_on_invalid=True
            )
            source_roi = sf.util.path_input(
                'Where are the ROI files (CSV) stored? [./slides] ',
                root=project_folder,
                default='./slides',
                create_on_invalid=True
            )
            source_tiles = sf.util.path_input(
                'Image tile storage location [./tiles] ',
                root=project_folder,
                default='./tiles',
                create_on_invalid=True
            )
            source_tfrecords = sf.util.path_input(
                'TFRecord storage location [./tfrecords] ',
                root=project_folder,
                default='./tfrecords',
                create_on_invalid=True
            )
            add_source(
                name=source_name,
                slides=relative_path(source_slides, project_folder),
                roi=relative_path(source_roi, project_folder),
                tiles=relative_path(source_tiles, project_folder),
                tfrecords=relative_path(source_tfrecords, project_folder),
                path=relative_path(project['dataset_config'], project_folder)
            )
            print('Updated dataset configuration file.')
        else:
            try:
                project['sources'] = [sources[int(j)-1] for j in selection]
            except TypeError:
                print(f'Invalid selection: {selection}')
                continue

    project['models_dir'] = sf.util.path_input(
        'Where should the saved models be stored? [./models] ',
        root=project_folder,
        default='./models',
        create_on_invalid=True
    )
    project['eval_dir'] = sf.util.path_input(
        'Where should model evaluations be stored? [./eval] ',
        root=project_folder,
        default='./eval',
        create_on_invalid=True
    )
    # Save settings as relative paths
    settings = _project_config(**project)
    sf.util.write_json(settings, join(project_folder, 'settings.json'))

    # Write a sample actions.py file
    sample_path = join(dirname(realpath(__file__)), 'sample_actions.py')
    with open(sample_path, 'r') as sample_file:
        sample_actions = sample_file.read()
        with open(join(project_folder, 'actions.py'), 'w') as actions_file:
            actions_file.write(sample_actions)
    log.info('Project configuration saved.')
    return settings

# -----------------------------------------------------------------------------

class _ProjectConfig:
    def __init__(self):
        pass

    @classmethod
    def to_dict(cls):
        with tempfile.TemporaryDirectory() as temp_dir:
            log.info(f"Downloading {cls.config_url}")
            r = requests.get(cls.config_url, allow_redirects=True)
            config_dest = join(temp_dir, 'config.json')
            open(config_dest, 'wb').write(r.content)
            if sf.util.md5(config_dest) != cls.config_md5:
                raise errors.ChecksumError("Remote config URL failed MD5 checksum.")
            config = sf.util.load_json(config_dest)
        config['annotations'] = cls.labels_url
        config['annotations_md5'] = cls.labels_md5
        return config

class BreastER(_ProjectConfig):
    config_url = 'https://raw.githubusercontent.com/jamesdolezal/slideflow/1.4.3/datasets/breast_er/breast_er.json'
    config_md5 = '6732f7e2473e2d58bc88a7aca1f0e770'
    labels_url = 'https://raw.githubusercontent.com/jamesdolezal/slideflow/1.4.3/datasets/breast_er/breast_labels.csv'
    labels_md5 = 'e25028e87760749973ceea691e6d63d7'

class ThyroidBRS(_ProjectConfig):
    config_url = 'https://raw.githubusercontent.com/jamesdolezal/slideflow/master/datasets/thyroid_brs/thyroid_brs.json'
    config_md5 = 'c4fbe83766db8f637780f7881cb1045e'
    labels_url = 'https://raw.githubusercontent.com/jamesdolezal/slideflow/master/datasets/thyroid_brs/thyroid_labels.csv'
    labels_md5 = 'c04f2569dc3a914241fae0d0b644a327'

class LungAdenoSquam(_ProjectConfig):
    config_url = 'https://raw.githubusercontent.com/jamesdolezal/slideflow/master/datasets/lung_adeno_squam/lung_adeno_squam.json'
    config_md5 = '9239d18b66e054132700c08831560669'
    labels_url = 'https://raw.githubusercontent.com/jamesdolezal/slideflow/master/datasets/lung_adeno_squam/lung_labels.csv'
    labels_md5 = '6619d520d707e211b22b477996bcfdcd'