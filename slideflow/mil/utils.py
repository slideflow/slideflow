"""Utility functions for MIL."""

import slideflow as sf
import numpy as np
import logging

from os.path import exists, join, isdir
from typing import Optional, Tuple, Union, Dict, List, TYPE_CHECKING
from slideflow import errors, log
from slideflow.util import path_to_name
from slideflow.model.torch_utils import get_device
from ._params import (
    _TrainerConfig
)

if TYPE_CHECKING:
    import torch
    from slideflow.model.base import BaseFeatureExtractor
    from slideflow.norm import StainNormalizer

# -----------------------------------------------------------------------------

def load_model_weights(
    weights: str,
    config: Optional[_TrainerConfig] = None,
    *,
    input_shape: Optional[int] = None,
    output_shape: Optional[int] = None,
    strict: bool = False,
    **kwargs
) -> Tuple["torch.nn.Module", _TrainerConfig]:
    """Load weights and build model.

    Args:
        weights (str): Path to model weights.
        config (:class:`slideflow.mil.TrainerConfig`):
            Configuration for building model. If ``weights`` is a path to a
            model directory, will attempt to read ``mil_params.json`` from this
            location and load saved configuration. Defaults to None.

    Keyword Args:
        input_shape (int): Number of features in the input data.
        output_shape (int): Number of output classes.
        strict (bool): Whether to strictly enforce that all hyperparameters
            are recognized. Defaults to False.

    Returns:
        :class:`torch.nn.Module`: Loaded model.
    """
    import torch

    #Log kwargs
    logging.debug(f"kwargs in load_model_weights: {kwargs}")

    if exists(join(weights, 'mil_params.json')):
        logging.info(f"Loading mil_params.json from {join(weights, 'mil_params.json')}")
        mil_params = sf.util.load_json(join(weights, 'mil_params.json'))
        logging.info(f"Loaded mil_params.json: {mil_params}")
    else:
        mil_params = None



    # Read configuration from saved model, if available
    if config is None:
        if mil_params is None:
            raise errors.ModelError(
                f"Could not find `mil_params.json` at {weights}. Check the "
                "provided model/weights path, or provide a configuration "
                "with 'config'."
            )
        else:
            config = sf.mil.mil_config(
                trainer=mil_params['trainer'],
                **mil_params['params'],
                validate=strict
            )

    logging.info(f"Loaded config: {config}")

    # Determine the input and output shapes, reading from the model
    # configuration, if necessary.
    if input_shape is None or output_shape is None:
        if mil_params is None:
            raise errors.ModelError(
                f"Could not find `mil_params.json` at {weights}. Check the "
                "provided model/weights path, or provide the input and output "
                "shape via input_shape and output_shape."
            )
        else:
            if input_shape is None and 'input_shape' in mil_params:
                input_shape = mil_params['input_shape']
            elif input_shape is None:
                raise errors.ModelError(
                    'Could not find input_shape in `mil_params.json`.'
                )
            if output_shape is None and 'output_shape' in mil_params:
                output_shape = mil_params['output_shape']
            elif output_shape is None:
                raise errors.ModelError(
                    'Could not find output_shape in `mil_params.json`.'
                )

    # Build the model
    config_dict = config.to_dict()

    z_dim = config_dict.get('z_dim', None)
    encoder_layers = config_dict.get('encoder_layers', None)
    dropout_p = config_dict.get('dropout_p', None)
    activation_function = config_dict.get('activation_function', None)
    goal = config_dict.get('task', None)

    logging.info(f"Building model with input_shape={input_shape}, output_shape={output_shape}, z_dim={z_dim}, encoder_layers={encoder_layers}, dropout_p={dropout_p}, activation_function={activation_function}, goal={goal}")  

    model_kwargs = {
        'z_dim': z_dim,
        'encoder_layers': encoder_layers,
        'dropout_p': dropout_p,
        'activation_function': activation_function,
        'goal': goal
    }

    logging.info(f"Model config: {model_kwargs}")
    model = config.build_model(input_shape, output_shape,
                               **model_kwargs)
    if isdir(weights):
        weights = _find_weights_path(weights, mil_params)
    log.info(f"Loading model weights from [green]{weights}[/]")

    try:
        model.load_state_dict(
            torch.load(weights, map_location=get_device()),
            strict=strict
        )
    except:
        #Log the differences between the model and the weights
        model_state_dict = model.state_dict()
        weights_state_dict = torch.load(weights, map_location=get_device())
        missing_keys = set(model_state_dict.keys()) - set(weights_state_dict.keys())
        unexpected_keys = set(weights_state_dict.keys()) - set(model_state_dict.keys())
        if len(missing_keys) > 0:
            log.warning(f"Missing keys in model: {missing_keys}")
        if len(unexpected_keys) > 0:
            log.warning(f"Unexpected keys in model: {unexpected_keys}")
        raise errors.ModelError(
            f"Could not load model weights from {weights}. "
            "Check the provided model/weights path, or provide a configuration "
            "with 'config'."
        )

    # Prepare device.
    if hasattr(model, 'relocate'):
        model.relocate()  # type: ignore
    model.eval()
    return model, config


def _find_weights_path(path: str, mil_params: Dict) -> str:
    """Determine location of model weights from a given model directory."""
    if exists(join(path, 'models', 'best_valid.pth')):
        weights = join(path, 'models', 'best_valid.pth')
    elif exists(join(path, "best-epoch.ckpt")):
        weights = join(path, "best-epoch.ckpt")
    elif exists(join(path, 'results', 's_0_checkpoint.pt')):
        weights = join(path, 'results', 's_0_checkpoint.pt')
    elif 'weights' in mil_params and mil_params['weights']:
        if mil_params['weights'].startswith('/'):
            weights = mil_params['weights']
        elif mil_params['weights'].startswith('./'):
            weights = join(path, mil_params['weights'][2:])
        else:
            weights = join(path, mil_params['weights'])
    else:
        raise errors.ModelError(
            f"Could not find model weights at path {weights}"
        )
    return weights


def _load_bag(bag: Union[str, np.ndarray, "torch.Tensor"]) -> "torch.Tensor":
    """Load bag from file or convert to torch.Tensor."""
    import torch

    if isinstance(bag, str):
        return torch.load(bag).to(torch.float32)
    elif isinstance(bag, np.ndarray):
        return torch.from_numpy(bag).to(torch.float32)
    elif isinstance(bag, torch.Tensor):
        return bag
    else:
        raise ValueError(
            "Unrecognized bag type '{}'".format(type(bag))
        )


def _detect_device(
    model: "torch.nn.Module",
    device: Optional[str] = None,
    verbose: bool = False
) -> "torch.device":
    """Auto-detect device from the given model."""
    import torch

    if device is None:
        device = next(model.parameters()).device
        if verbose:
            log.debug(f"Auto device detection: using {device}")
    elif isinstance(device, str):
        if verbose:
            log.debug(f"Using {device}")
        device = torch.device(device)
    return device


def _get_nested_bags(dataset, bag_directories):
    # This is a nested list of bag paths, where each nested list contains
    # the paths to bags at one magnification level.
    _matching_bag_paths = [dataset.pt_files(b) for b in bag_directories]

    # Convert the above to a nested list of slide names.
    _nested_slides = [[path_to_name(b) for b in _bag] for _bag in _matching_bag_paths]

    # Identify the subset of slide names present in all of the outer lists.
    # These are the slides that have bags at all magnification levels.
    slides = list(set.intersection(*[set(s) for s in _nested_slides]))

    # Filter the bags to only those that have all magnification levels.
    nested_bag_paths = [
        [b for b in _bag if path_to_name(b) in slides]
        for _bag in _matching_bag_paths
    ]
    assert(all([len(b) == len(slides) for b in nested_bag_paths]))
    nested_bags = np.array(nested_bag_paths)  # shape: (num_modes, num_train_slides)
    # Transpose the above, so that each row is a slide, and each column is a
    # magnification level.
    nested_bags = nested_bags.T  # shape: (num_train_slides, num_modes)

    # Sort the slides by the bag order.
    slides = np.array(slides)
    slides = slides[np.argsort(slides)]
    bag_to_slide = np.array([path_to_name(b) for b in nested_bags[:, 0]])
    bag_order = np.argsort(bag_to_slide)
    nested_bags = nested_bags[bag_order]
    assert(np.all(bag_to_slide[bag_order] == slides))

    return nested_bags, list(slides)


def aggregate_bags_by_slide(
    bags: np.ndarray,
    labels: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate bags by slide.

    Args:
        bags (np.ndarray): Array of bag paths.
        labels (dict): Dictionary mapping slide names to labels.

    Returns:
        tuple: (bags, targets)

    """
    # Convert bags to nested list, where each sublist contains bags for a slide.
    slides = np.unique([path_to_name(bag) for bag in bags])
    bags = np.array([
        [bag for bag in bags if path_to_name(bag) == slide]
        for slide in slides
    ])

    # Prepare targets, mapping each bag sublist to the label of the first bag.
    targets = np.array([labels[path_to_name(sublist[0])] for sublist in bags])

    return bags, targets


def aggregate_bags_by_patient(
    bags: np.ndarray,
    labels: Dict[str, int],
    slide_to_patient: Dict[str, str],
    task: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate bags by patient.

    Args:
        bags (np.ndarray): Array of bag paths.
        labels (dict): Dictionary mapping slide names to labels.

    Returns:
        tuple: (bags, targets)

    """
    # Create a reverse dictionary, mapping patient codes to a list of bags.
    patient_to_bags = {}  # type: Dict[str, List[str]]
    for bag in bags:
        patient = slide_to_patient[path_to_name(bag)]
        if patient not in patient_to_bags:
            patient_to_bags[patient] = []
        patient_to_bags[patient].append(bag)

    # Create array where each element contains the list of slides for a patient.
    bags = np.array([lst for lst in patient_to_bags.values()], dtype=object)

    # Create a dictionary mapping patients to their labels.
    patients_labels = {}
    for patient, patient_bags in patient_to_bags.items():
        # Confirm that all slides for a patient have the same label.
        if len(np.unique([labels[path_to_name(b)] for b in patient_bags], axis=0)) != 1:
            logging.warning(
                f"Patient {patient} has slides/bags with different labels."
            )
        patients_labels[patient] = labels[path_to_name(patient_bags[0])]

    # Prepare targets, mapping each bag sublist to the label of the first bag.
    targets = np.array([patients_labels[slide_to_patient[path_to_name(sublist[0])]]
                        for sublist in bags])

    return bags, targets


def aggregate_trainval_bags_by_slide(
    bags: np.ndarray,
    labels: Dict[str, int],
    train_slides: List[str],
    val_slides: List[str],
    *,
    log_manifest: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate training/validation bags by slide.

    Args:
        bags (np.ndarray): Array of bag paths.
        labels (dict): Dictionary mapping slide names to labels.
        train_slides (list): List of training slide names.
        val_slides (list): List of validation slide names.

    Keyword Args:
        log_manifest (str): Path to manifest file to write.
            Defaults to None.

    Returns:
        tuple: (bags, targets, train_idx, val_idx)

    """
    # Prepare targets
    targets = np.array([labels[path_to_name(f)] for f in bags])

    # Prepare training/validation indices
    train_idx = np.array([i for i, bag in enumerate(bags)
                        if path_to_name(bag) in train_slides])
    val_idx = np.array([i for i, bag in enumerate(bags)
                        if path_to_name(bag) in val_slides])

    # Write slide/bag manifest
    if log_manifest is not None:
        sf.util.log_manifest(
            [bag for bag in bags if path_to_name(bag) in train_slides],
            [bag for bag in bags if path_to_name(bag) in val_slides],
            labels=labels,
            filename=log_manifest
        )

    return bags, targets, train_idx, val_idx


def aggregate_trainval_bags_by_patient(
    bags: np.ndarray,
    labels: Dict[str, int],
    train_slides: List[str],
    val_slides: List[str],
    slide_to_patient: Dict[str, str],  # slide -> patient
    *,
    log_manifest: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate training/validation bags by patient.

    Args:
        bags (np.ndarray): Array of bag paths.
        labels (dict): Dictionary mapping slide names to labels.
        train_slides (list): List of training slide names.
        val_slides (list): List of validation slide names.

    Keyword Args:
        log_manifest (str): Path to manifest file to write.
            Defaults to None.

    Returns:
        tuple: (bags, targets, train_idx, val_idx)
    """
    # Create a reverse dictionary, mapping patient codes to a list of bags.
    patient_to_bags = {}  # type: Dict[str, List[str]]
    for bag in bags:
        patient = slide_to_patient[path_to_name(bag)]
        if patient not in patient_to_bags:
            patient_to_bags[patient] = []
        patient_to_bags[patient].append(bag)

    # Create array where each element is a list of bag paths for one patient.
    bags = np.array([lst for lst in patient_to_bags.values()], dtype=object)

    patients_labels = {}
    # For each patient, confirm that all slides have the same label or consistent survival labels.
    for patient, patient_bags in patient_to_bags.items():
        # Retrieve labels for all bags/slides belonging to this patient.
        bag_labels = [labels[path_to_name(b)] for b in patient_bags]

        # If the set of labels is not unique, we need to check if these are survival labels.
        # 'bag_labels' might look like [(duration, event), (duration, event), ...].
        # Or it could be integers/classes, e.g. [0, 0, 1].
        unique_labels = np.unique(bag_labels, axis=0)

        if len(unique_labels) > 1:
            # More than one unique label. Check if they are survival labels
            # (duration, event) or [duration, event].
            # We will assume they are all the same type if they come from the same dictionary.
            all_survival_format = all(
                isinstance(lbl, (list, tuple))
                and len(lbl) == 2
                for lbl in bag_labels
            )

            if all_survival_format:
                # If they are all 2-element lists/tuples, check durations and events separately.
                durations = [lbl[0] for lbl in bag_labels]
                events = [lbl[1] for lbl in bag_labels]
                # If either durations or events differ across slides, it's inconsistent.
                if not (len(set(durations)) == 1 and len(set(events)) == 1):
                    logging.warning(
                        f"Patient {patient} has slides/bags with inconsistent survival labels."
                    )
            else:
                # Not survival or not consistent => log a warning.
                logging.warning(
                    f"Patient {patient} has slides/bags with different labels."
                )

        # Assign the label for the patient: just take the first one (assuming consistency).
        patients_labels[patient] = bag_labels[0]

    # Prepare targets, mapping each bag sublist to the label of the first bag in that sublist.
    targets = np.array(
        [
            patients_labels[slide_to_patient[path_to_name(sublist[0])]]
            for sublist in bags
        ]
    )

    # Identify the bag indices of the training and validation patients.
    train_idx = np.array([
        i for i, sublist in enumerate(bags)
        if path_to_name(sublist[0]) in train_slides
    ])
    val_idx = np.array([
        i for i, sublist in enumerate(bags)
        if path_to_name(sublist[0]) in val_slides
    ])

    # Optional: Write patient/bag manifest if log_manifest is provided
    if log_manifest is not None:
        # For demonstration, assume a helper function sf.util.log_manifest exists.
        train_patients = list(np.unique([
            slide_to_patient[path_to_name(bags[i][0])] for i in train_idx
        ]))
        val_patients = list(np.unique([
            slide_to_patient[path_to_name(bags[i][0])] for i in val_idx
        ]))
        # Example call. Adjust to match your real logging function's signature.
        sf.util.log_manifest(
            train_patients,
            val_patients,
            labels=patients_labels,
            filename=log_manifest,
            remove_extension=False
        )

    return bags, targets, train_idx, val_idx

def _is_list_of_paths(bag):
    return ((isinstance(bag, list) or (isinstance(bag, np.ndarray))
             and isinstance(bag[0], str)))
