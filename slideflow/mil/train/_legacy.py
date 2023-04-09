"""Legacy CLAM trainer."""

import os
import csv
import slideflow as sf
from os.path import join, exists
from typing import Union, List, Optional, Dict, TYPE_CHECKING
from slideflow import Dataset, log, errors
from slideflow.util import path_to_name

if TYPE_CHECKING:
    from ..clam import CLAM_Args

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
    assert isinstance(clam_args, clam.CLAM_Args)

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
    clam_dataset = clam.Generic_MIL_Dataset(
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
        clam.export_attention(
            clam_args.to_dict(),
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
                try:
                    dataset.tfrecord_heatmap(
                        att_tfr,
                        tile_dict=attention_dict,
                        outdir=heatmaps_dir,
                        cmap='coolwarm'
                    )
                except errors.SlideNotFoundError:
                    log.warning(f"Unable to find slide {slide}; skipping heatmap")

