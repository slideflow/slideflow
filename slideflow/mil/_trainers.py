"""Training functions for various multi-instance learning (MIL) models."""

import os
import csv
import numpy as np
import slideflow as sf
from os.path import join, exists
from typing import Union, List, Optional, Dict, TYPE_CHECKING
from slideflow import Dataset, log
from slideflow.util import path_to_name

if TYPE_CHECKING:
    from .clam import CLAM_Args

# -----------------------------------------------------------------------------

def train_mil(
    model: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]],
    *,
    outdir: str = 'mil',
    exp_label: Optional[str] = None,
    **kwargs
):
    """Train a multi-instance learning model.

    Args:
        train_dataset (:class:`slideflow.Dataset`): Training dataset.
        val_dataset (:class:`slideflow.Dataset`): Validation dataset.
        outcomes (str): Outcome column (annotation header) from which to
            derive category labels.
        bags (str): Either a path to directory with \*.pt files, or a list
            of paths to individual \*.pt files. Each file should contain
            exported feature vectors, with each file containing all tile
            features for one patient.

    Keyword args:
        outdir (str): Directory in which to save model and results.
        exp_label (str): Experiment label, used for naming the subdirectory
            in the ``{project root}/mil`` folder, where training history
            and the model will be saved.
        **kwargs (Any): Any additional keyword arguments will be passed
            as parameters to the MIL model trainer.
    """
    if exp_label is None:
        exp_label = model
    if model == 'marugoto':
        return train_marugoto(
            train_dataset,
            val_dataset,
            outcomes,
            bags,
            outdir=outdir,
            exp_label=exp_label,
            **kwargs
        )
    elif model == 'clam':
        if 'clam_args' in kwargs and len(kwargs) > 1:
            raise ValueError(f"Unrecognized keyword arguments: {list(kwargs.keys())}")
        elif 'clam_args' in kwargs:
            clam_args = kwargs['clam_args']
        elif len(kwargs):
            from .clam import CLAM_Args
            clam_args = CLAM_Args(**kwargs)
        else:
            clam_args = None
        return train_clam(
            train_dataset,
            val_dataset,
            outcomes,
            bags,
            outdir=outdir,
            exp_label=exp_label,
            clam_args=clam_args
        )
    else:
        raise ValueError(f"Unrecognized MIL model: {model}")


# -----------------------------------------------------------------------------

def train_clam(
    train_dataset: Dataset,
    val_dataset: Dataset,
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]],
    *,
    outdir: str = 'mil',
    exp_label: str = "CLAM",
    clam_args: Optional["CLAM_Args"] = None,
) -> None:
    """Train a CLAM model from layer activations exported with
    :meth:`slideflow.project.generate_features_for_clam`.

    See :ref:`clam_mil` for more information.

    Args:
        train_dataset (:class:`slideflow.Dataset`): Training dataset.
        val_dataset (:class:`slideflow.Dataset`): Validation dataset.
        outcomes (str): Outcome column (annotation header) from which to
            derive category labels.
        bags (str): Either a path to directory with \*.pt files, or a list
            of paths to individual \*.pt files. Each file should contain
            exported feature vectors, with each file containing all tile
            features for one patient.

    Keyword args:
        outdir (str): Directory in which to save model and results.
        exp_label (str): Experiment label, used for naming the subdirectory
            in the ``{project root}/mil`` folder, where training history
            and the model will be saved.
        clam_args (optional): Namespace with clam arguments, as provided
            by :func:`slideflow.clam.get_args`.

    Returns:
        None

    """
    import slideflow.clam as clam
    from slideflow.clam import CLAM_Dataset, CLAM_Args

    # Set up output directory in project root
    if not exists(outdir):
        os.makedirs(outdir)
    outdir = sf.util.create_new_model_dir(outdir, exp_label)
    results_dir = join(outdir, 'results')
    if not exists(results_dir):
        os.makedirs(results_dir)

    # Set up labels.
    labels, unique_train = train_dataset.labels(outcomes, format='name', use_float=False)
    val_labels, unique_val = val_dataset.labels(outcomes, format='name', use_float=False)
    labels.update(val_labels)
    unique_labels = np.unique(unique_train + unique_val)
    label_dict = dict(zip(unique_labels, range(len(unique_labels))))

    # Set up bags.
    if isinstance(bags, str):
        train_bags = train_dataset.pt_files(bags)
        val_bags = val_dataset.pt_files(bags)
    else:
        train_bags = val_bags = bags

    # Set up datasets.
    train_mil_dataset = CLAM_Dataset(
        train_bags,
        annotations=train_dataset.filtered_annotations,
        label_col=outcomes,
        label_dict=label_dict
    )
    val_mil_dataset = CLAM_Dataset(
        val_bags,
        annotations=val_dataset.filtered_annotations,
        label_col=outcomes,
        label_dict=label_dict
    )

    # Get base CLAM args/settings if not provided.
    if not clam_args:
        num_features = train_mil_dataset.detect_num_features()
        clam_args = clam.get_args(model_size=[num_features, 256, 128])
    assert isinstance(clam_args, CLAM_Args)
    clam_args.results_dir = results_dir
    clam_args.n_classes = len(unique_labels)

    # Save clam settings
    sf.util.write_json(clam_args.to_dict(), join(outdir, 'experiment.json'))

    # Run CLAM
    datasets = (train_mil_dataset, val_mil_dataset, val_mil_dataset)
    results, test_auc, val_auc, test_acc, val_acc = clam.train(
        datasets, 0, clam_args
    )
    return results


def legacy_train_clam(
    exp_name: str,
    pt_files: str,
    outcomes: Union[str, List[str]],
    dataset: Dataset,
    train_slides: Union[str, List[str]] = 'auto',
    val_slides: Union[str, List[str]] = 'auto',
    splits: str = 'splits.json',
    clam_args: Optional["CLAM_Args"] = None,
    attention_heatmaps: bool = True,
    outdir: str = None,
) -> None:
    """Deprecated function. Train a CLAM model from layer activations
    exported with :meth:`slideflow.Project.generate_features_for_clam`.

    Preferred API is :meth:`slideflow.Project.train_mil()`.

    See :ref:`clam_mil` for more information.

    Args:
        exp_name (str): Name of experiment. Makes clam/{exp_name} folder.
        pt_files (str): Path to pt_files containing tile-level features.
        outcomes (str): Annotation column which specifies the outcome.
        dataset (:class:`slideflow.Dataset`): Dataset object from
            which to generate activations.
        train_slides (str, optional): List of slide names for training.
            If 'auto' (default), will auto-generate training/val split.
        validation_slides (str, optional): List of slides for validation.
            If 'auto' (default), will auto-generate training/val split.
        splits (str, optional): Filename of JSON file in which to log
            training/val splits. Looks for filename in project root
            directory. Defaults to "splits.json".
        clam_args (optional): Namespace with clam arguments, as provided
            by :func:`slideflow.clam.get_args`.
        attention_heatmaps (bool, optional): Save attention heatmaps of
            validation dataset.

    Returns:
        None
    """
    import slideflow.clam as clam
    from slideflow.clam import export_attention
    from slideflow.clam import Generic_MIL_Dataset

    # Set up CLAM experiment data directory
    clam_dir = join(outdir, exp_name)
    results_dir = join(clam_dir, 'results')
    if not exists(results_dir):
        os.makedirs(results_dir)

    # Detect number of features automatically from saved pt_files
    pt_file_paths = [
        join(pt_files, p) for p in os.listdir(pt_files)
        if sf.util.path_to_ext(join(pt_files, p)) == 'pt'
    ]
    num_features = clam.detect_num_features(pt_file_paths[0])

    # Get base CLAM args/settings if not provided.
    if not clam_args:
        clam_args = clam.get_args(model_size=[num_features, 256, 128])
    assert isinstance(clam_args, CLAM_Args)

    # Note: CLAM only supports categorical outcomes
    labels, unique_labels = dataset.labels(outcomes, use_float=False)

    if train_slides == val_slides == 'auto':
        k_train_slides = {}  # type: Dict
        k_val_slides = {}  # type: Dict
        for k in range(clam_args.k):
            train_dts, val_dts = dataset.split(
                'categorical',
                labels,
                val_strategy='k-fold',
                splits=splits,
                val_k_fold=clam_args.k,
                k_fold_iter=k+1
            )
            k_train_slides[k] = [
                path_to_name(t) for t in train_dts.tfrecords()
            ]
            k_val_slides[k] = [
                path_to_name(t) for t in val_dts.tfrecords()
            ]
    else:
        k_train_slides = {0: train_slides}
        k_val_slides = {0: val_slides}

    # Remove slides without associated .pt files
    num_skipped = 0
    for k in k_train_slides:
        n_supplied = len(k_train_slides[k]) + len(k_val_slides[k])
        k_train_slides[k] = [
            s for s in k_train_slides[k] if exists(join(pt_files, s+'.pt'))
        ]
        k_val_slides[k] = [
            s for s in k_val_slides[k] if exists(join(pt_files, s+'.pt'))
        ]
        n_train = len(k_train_slides[k])
        n_val = len(k_val_slides[k])
        num_skipped += n_supplied - (n_train + n_val)
    if num_skipped:
        log.warn(f'Skipping {num_skipped} slides missing .pt files.')

    # Set up training/validation splits (mirror base model)
    split_dir = join(clam_dir, 'splits')
    if not exists(split_dir):
        os.makedirs(split_dir)
    header = ['', 'train', 'val', 'test']
    for k in range(clam_args.k):
        with open(join(split_dir, f'splits_{k}.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            # Currently, the below sets the val & test sets to be the same
            for i in range(max(len(k_train_slides[k]),
                                len(k_val_slides[k]))):
                row = [i]  # type: List
                if i < len(k_train_slides[k]):
                    row += [k_train_slides[k][i]]
                else:
                    row += ['']
                if i < len(k_val_slides[k]):
                    row += [k_val_slides[k][i], k_val_slides[k][i]]
                else:
                    row += ['', '']
                writer.writerow(row)

    # Assign CLAM settings based on this project
    clam_args.results_dir = results_dir
    clam_args.n_classes = len(unique_labels)
    clam_args.split_dir = split_dir
    clam_args.data_root_dir = pt_files

    # Save clam settings
    sf.util.write_json(clam_args.to_dict(), join(clam_dir, 'experiment.json'))

    # Create CLAM dataset
    clam_dataset = Generic_MIL_Dataset(
        annotations=dataset.filtered_annotations,
        data_dir=pt_files,
        shuffle=False,
        seed=clam_args.seed,
        print_info=True,
        label_col=outcomes,
        label_dict=dict(zip(unique_labels, range(len(unique_labels)))),
        patient_strat=False,
        ignore=[]
    )
    # Run CLAM
    clam.main(clam_args, clam_dataset)

    # Get attention from trained model on validation set(s)
    for k in k_val_slides:
        tfr = dataset.tfrecords()
        attention_tfrecords = [
            t for t in tfr if path_to_name(t) in k_val_slides[k]
        ]
        attention_dir = join(clam_dir, 'attention', str(k))
        if not exists(attention_dir):
            os.makedirs(attention_dir)
        rev_labels = dict(zip(range(len(unique_labels)), unique_labels))
        export_attention(
            vars(clam_args),
            ckpt_path=join(results_dir, f's_{k}_checkpoint.pt'),
            outdir=attention_dir,
            pt_files=pt_files,
            slides=k_val_slides[k],
            reverse_labels=rev_labels,
            labels=labels
        )
        if attention_heatmaps:
            heatmaps_dir = join(clam_dir, 'attention_heatmaps', str(k))
            if not exists(heatmaps_dir):
                os.makedirs(heatmaps_dir)

            for att_tfr in attention_tfrecords:
                attention_dict = {}
                slide = path_to_name(att_tfr)
                try:
                    with open(join(attention_dir, slide+'.csv'), 'r') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            attention_dict.update({
                                int(row[0]): float(row[1])
                            })
                except FileNotFoundError:
                    print(f'Attention scores for slide {slide} not found')
                    continue
                dataset.tfrecord_heatmap(
                    att_tfr,
                    tile_dict=attention_dict,
                    outdir=heatmaps_dir
                )

# -----------------------------------------------------------------------------

def train_marugoto(
    train_dataset: Dataset,
    val_dataset: Dataset,
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]],
    *,
    outdir: str = 'mil',
    exp_label: str = 'marugoto',
    **kwargs
) -> None:
    """Train a Marugoto aMIL model.

    Args:
        train_dataset (:class:`slideflow.Dataset`): Training dataset.
        val_dataset (:class:`slideflow.Dataset`): Validation dataset.
        outcomes (str): Outcome column (annotation header) from which to
            derive category labels.
        bags (str): Either a path to directory with \*.pt files, or a list
            of paths to individual \*.pt files. Each file should contain
            exported feature vectors, with each file containing all tile
            features for one patient.

    Keyword args:
        outdir (str): Directory in which to save model and results.
        exp_label (str): Experiment label, used for naming the subdirectory
            in the ``{project root}/mil`` folder, where training history
            and the model will be saved.
        lr_max (float): Maximum learning rate.
        epochs (int): Maximum epochs.
    """
    from slideflow.mil import marugoto

    # Set up output directory in project root
    if not exists(outdir):
        os.makedirs(outdir)
    outdir = sf.util.create_new_model_dir(outdir, exp_label)

    # Prepare labels and slides
    labels, unique_train = train_dataset.labels(outcomes, format='name')
    val_labels, unique_val = val_dataset.labels(outcomes, format='name')
    labels.update(val_labels)
    unique_categories = np.unique(unique_train + unique_val)

    # Prepare bags
    if isinstance(bags, str):
        train_bags = train_dataset.pt_files(bags)
        val_bags = val_dataset.pt_files(bags)
        bags = np.concatenate((train_bags, val_bags))
    else:
        bags = np.array(bags)
    targets = np.array([labels[path_to_name(f)] for f in bags])

    # Prepare training/validation indices
    train_slides = train_dataset.slides()
    train_idx = np.array([i for i, bag in enumerate(bags)
                            if path_to_name(bag) in train_slides])
    val_slides = val_dataset.slides()
    val_idx = np.array([i for i, bag in enumerate(bags)
                            if path_to_name(bag) in val_slides])

    # Train
    return marugoto.train_mil(
        bags=bags,
        targets=targets,
        train_idx=train_idx,
        val_idx=val_idx,
        unique_categories=unique_categories,
        outdir=outdir,
        **kwargs
    )