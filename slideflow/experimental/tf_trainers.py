import os
import json
import numpy as np
import tensorflow as tf
import multiprocessing
from types import SimpleNamespace
from typing import Tuple, Dict, List, Any, Union, Optional, TYPE_CHECKING

import slideflow as sf
from slideflow import project_utils, Project
from slideflow.util import errors, log
from slideflow.util import colors as col
from slideflow.model.tensorflow import (
    _PredictionAndEvaluationCallback, LinearTrainer, ModelParams
)

if TYPE_CHECKING:
    import torch
    import neptune.new as neptune


def linear_to_cat_metrics_from_dataset(
    model: Union["tf.keras.Model", "torch.nn.Module"],
    model_type: str,
    labels: Dict[str, Any],
    patients: Dict[str, str],
    dataset: Union["tf.data.Dataset", "torch.utils.data.DataLoader"],
    pred_args: SimpleNamespace,
    outcome_names: Optional[List[str]] = None,
    label: str = '',
    data_dir: str = '',
    num_tiles: int = 0,
    histogram: bool = False,
    save_predictions: bool = True,
    neptune_run: Optional["neptune.Run"] = None,
) -> Tuple[Dict, float, float]:
    """Modified to support conversion of linear predictions to categorical."""
    yt, yp, y_std, t_s, acc, loss = sf.stats.eval_from_dataset(
        model,
        dataset,
        model_type,
        pred_args,
        num_tiles=num_tiles
    )
    yp = np.array([[x[0], x[0]] for x in yp])
    metrics = sf.stats.metrics_from_pred(
        y_true=yt,
        y_pred=yp,
        y_std=y_std,
        tile_to_slides=t_s,
        labels=labels,
        patients=patients,
        model_type=model_type,
        outcome_names=outcome_names,
        label=label,
        data_dir=data_dir,
        save_predictions=save_predictions,
        histogram=histogram,
        plot=True,
        neptune_run=neptune_run
    )
    return metrics, acc, loss


def _train_val_worker(
    datasets: Tuple[sf.Dataset, sf.Dataset],
    model_kw: Dict,
    training_kw: Dict,
    results_dict: Dict,
    verbosity: int
) -> None:
    """Internal function to execute model training in an isolated process."""
    log.setLevel(verbosity)
    train_dts, val_dts = datasets
    trainer = CatValLinearTrainer(**model_kw)
    results = trainer.train(train_dts, val_dts, **training_kw)
    results_dict.update({model_kw['name']: results})

class _ValLabelEvalCallback(_PredictionAndEvaluationCallback):
    """Modified to support conversion of linear predictions to categorical."""

    def __init__(self, parent: "CatValLinearTrainer", cb_args: SimpleNamespace) -> None:
        super().__init__(parent, cb_args)
        self.val_labels = parent.val_labels

    def _metrics_from_dataset(
        self,
        epoch_label: str,
        pred_args: SimpleNamespace
    ) -> Tuple[Dict, float, float]:
        """Modified to use validation-specific labels if specified."""

        if self.val_labels is None:
            model_type = self.hp.model_type()
            labels = self.parent.labels
        else:
            model_type = 'categorical'
            labels = self.val_labels

        return linear_to_cat_metrics_from_dataset(
            self.model,
            model_type=model_type,
            labels=labels,
            patients=self.parent.patients,
            dataset=self.cb_args.validation_data_with_slidenames,
            outcome_names=self.parent.outcome_names,
            label=epoch_label,
            data_dir=self.parent.outdir,
            num_tiles=self.cb_args.num_val_tiles,
            histogram=False,
            save_predictions=self.cb_args.save_predictions,
            pred_args=pred_args
        )


class CatValLinearTrainer(LinearTrainer):
    """Modified to support conversion of linear predictions to categorical."""

    def __init__(
        self,
        *args,
        val_labels: Dict[str, Any],
        val_outcome_names: List[str],
        **kwargs
    ) -> None:

        super().__init__(*args, **kwargs)
        # Set the eval callback to our new version defined above
        self.eval_callback = _ValLabelEvalCallback

        # Set up the validation labels
        self.val_labels = val_labels
        self.val_annotations_tables = []
        val_outcome_labels = np.array(list(val_labels.values()))
        if len(val_outcome_labels.shape) == 1:
            val_outcome_labels = np.expand_dims(val_outcome_labels, axis=1)
        if not val_outcome_names:
            val_outcome_names = [
                f'Outcome {i}'
                for i in range(val_outcome_labels.shape[1])
            ]
        if not isinstance(val_outcome_names, list):
            val_outcome_names = [val_outcome_names]
        if len(val_outcome_names) != val_outcome_labels.shape[1]:
            num_names = len(val_outcome_names)
            num_outcomes = val_outcome_labels.shape[1]
            raise errors.ModelError(
                f'Size of val_outcome_names ({num_names}) != number of '
                f'outcomes {num_outcomes}'
            )
        self.val_outcome_names = val_outcome_names
        self.val_num_classes = {
            i: np.unique(val_outcome_labels[:,i]).shape[0]
            for i in range(val_outcome_labels.shape[1])
        }
        with tf.device('/cpu'):
            for oi in range(val_outcome_labels.shape[1]):
                self.val_annotations_tables += [tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(
                        self.slides,
                        val_outcome_labels[:,oi]
                    ), -1
                )]

    def _parse_val_tfrecord_labels(self, image, slide):
        '''Parses raw entry read from TFRecord.'''
        image_dict = { 'tile_image': image }
        if self.val_num_classes is None:
            label = None
        elif len(self.val_num_classes) > 1:
            label = {
                f'out-{oi}': self.val_annotations_tables[oi].lookup(slide)
                for oi in range(len(self.val_num_classes))
            }
        else:
            label = self.val_annotations_tables[0].lookup(slide)
        return image_dict, label

    def _interleave_kwargs_val(self, **kwargs):
        args = self._interleave_kwargs(**kwargs)
        args['labels'] = self._parse_val_tfrecord_labels
        return args


class ExperimentalProject(Project):
    def __init__(*args, **kwargs):
        super().__init__()


    def _train_hp(
        self,
        hp_name: str,
        hp: ModelParams,
        outcomes: List[str],
        val_settings: SimpleNamespace,
        ctx: multiprocessing.context.BaseContext,
        filters: Optional[Dict],
        filter_blank: Optional[Union[str, List[str]]],
        input_header: Optional[Union[str, List[str]]],
        min_tiles: int,
        max_tiles: int,
        mixed_precision: bool,
        splits: str,
        results_dict: Dict,
        training_kwargs: Dict,
        balance_headers: Optional[Union[str, List[str]]]
    ) -> None:
        # --- Prepare dataset ---------------------------------------------
        # Filter out slides that are blank in the outcome label,
        # or blank in any of the input_header categories
        if filter_blank is not None and not isinstance(filter_blank, list):
            filter_blank = [filter_blank]
        if filter_blank:
            filter_blank += [o for o in outcomes]
        else:
            filter_blank = [o for o in outcomes]
        if input_header is not None and not isinstance(input_header, list):
            input_header = [input_header]
        if input_header is not None:
            filter_blank += input_header
        dataset = self.dataset(hp.tile_px, hp.tile_um)
        dataset = dataset.filter(
            filters=filters,
            filter_blank=filter_blank,
            min_tiles=min_tiles
        )
        # --- Load labels -------------------------------------------------
        use_float = (hp.model_type() in ['linear', 'cph'])
        labels, unique = dataset.labels(outcomes, use_float=use_float)
        if hp.model_type() == 'categorical' and len(outcomes) == 1:
            outcome_labels = dict(zip(range(len(unique)), unique))
        elif hp.model_type() == 'categorical':
            assert isinstance(unique, dict)
            outcome_labels = {
                k: dict(zip(range(len(ul)), ul))  # type: ignore
                for k, ul in unique.items()
            }
        else:
            outcome_labels = dict(zip(range(len(outcomes)), outcomes))
        if hp.model_type() != 'linear' and len(outcomes) > 1:
            log.info('Using multi-outcome approach for categorical outcome')
        # If multiple categorical outcomes are used,
        # create a merged variable for k-fold splitting
        if hp.model_type() == 'categorical' and len(outcomes) > 1:
            split_labels = {
                k: '-'.join(map(str, v))  # type: ignore
                for k, v in labels.items()
            }
        else:
            split_labels = labels  # type: ignore

        # --- Load validation labels ------------------------------------------------------------------------------

        if val_settings.outcomes:
            if not isinstance(val_settings.outcomes, list):
                val_settings.outcomes = [val_settings.outcomes]
            if len(val_settings.outcomes) > 1:
                raise errors.ModelParamsError(f"Can only use single validation header")

        if val_settings.outcomes:
            val_labels, val_unique_labels = dataset.labels(val_settings.outcomes, use_float=False)
            val_outcome_labels = dict(zip(range(len(val_unique_labels)), val_unique_labels))
        else:
            val_labels = None

        # --- Prepare k-fold validation configuration ---------------------
        results_log_path = os.path.join(self.root, 'results_log.csv')
        k_header = val_settings.k_fold_header
        if val_settings.k is not None and not isinstance(val_settings.k, list):
            val_settings.k = [val_settings.k]
        if val_settings.strategy == 'k-fold-manual':
            _, unique_k = dataset.labels(k_header, format='name')
            valid_k = [int(kf) for kf in unique_k]
            k_fold = len(valid_k)
            log.info(f"Manual folds: {', '.join([str(ks) for ks in valid_k])}")
            if val_settings.k:
                valid_k = [kf for kf in valid_k if kf in val_settings.k]
        elif val_settings.strategy in ('k-fold',
                                        'k-fold-preserved-site',
                                        'bootstrap'):
            k_fold = val_settings.k_fold
            if val_settings.k is None:
                valid_k = list(range(1, k_fold+1))
            else:
                valid_k = [
                    kf for kf in range(1, k_fold+1) if kf in val_settings.k
                ]
        else:
            k_fold = 0
            valid_k = [None]  # type: ignore
        # Create model labels
        label_string = '-'.join(outcomes)
        model_name = f'{label_string}-{hp_name}'
        if k_fold is None:
            model_iterations = [model_name]
        else:
            model_iterations = [f'{model_name}-kfold{k}' for k in valid_k]

        s_args = SimpleNamespace(
            model_name=model_name,
            outcomes=outcomes,
            val_outcomes=val_settings.outcomes,
            k_header=k_header,
            valid_k=valid_k,
            split_labels=split_labels,
            splits=splits,
            labels=labels,
            val_labels=val_labels,
            min_tiles=min_tiles,
            max_tiles=max_tiles,
            outcome_labels=outcome_labels,
            val_outcome_labels=val_outcome_labels,
            filters=filters,
            training_kwargs=training_kwargs,
            mixed_precision=mixed_precision,
            ctx=ctx,
            results_dict=results_dict,
            bal_headers=balance_headers,
            input_header=input_header
        )
        # --- Train on a specific K-fold --------------------------------------
        for k in valid_k:
            s_args.k = k
            self._train_split(dataset, hp, val_settings, s_args)
        # --- Record results --------------------------------------------------
        if (not val_settings.source
            and (val_settings.strategy is None
                    or val_settings.strategy == 'none')):
            log.info(f'No validation performed.')
        else:
            for mi in model_iterations:
                if mi not in results_dict or 'epochs' not in results_dict[mi]:
                    log.error(f'Training failed for model {model_name}')
                else:
                    sf.util.update_results_log(
                        results_log_path,
                        mi,
                        results_dict[mi]['epochs']
                    )
            log.info(f'Training results saved: {col.green(results_log_path)}')


    def _train_split(
            self,
            dataset: sf.Dataset,
            hp: ModelParams,
            val_settings: SimpleNamespace,
            s_args: SimpleNamespace
        ) -> None:
            '''Trains a model for a given training/validation split.
            Args:
                dataset (:class:`slideflow.Dataset`): Dataset to split into
                    training and validation.
                hp (:class:`slideflow.ModelParams`): Model parameters.
                val_settings (:class:`types.SimpleNamspace`): Validation settings.
                s_args (:class:`types.SimpleNamspace`): Training settings.
            '''
            # Log current model name and k-fold iteration, if applicable
            k_msg = ''
            if s_args.k is not None:
                k_msg = f' ({val_settings.strategy} #{s_args.k})'
            if log.getEffectiveLevel() <= 20:
                print()
            log.info(f'Training model {col.bold(s_args.model_name)}{k_msg}...')
            log.info(f'Hyperparameters: {hp}')
            log.info(f'Val settings: {json.dumps(vars(val_settings), indent=2)}')
            # --- Set up validation data ------------------------------------------
            manifest = dataset.manifest()
            # Use an external validation dataset if supplied
            if val_settings.source:
                train_dts = dataset
                val_dts = sf.Dataset(
                    tile_px=hp.tile_px,
                    tile_um=hp.tile_um,
                    config=self.dataset_config,
                    sources=val_settings.source,
                    annotations=val_settings.annotations,
                    filters=val_settings.filters,
                    filter_blank=val_settings.filter_blank
                )
                is_float = (hp.model_type() in ['linear', 'cph'])
                val_labels, _ = val_dts.labels(s_args.outcomes, use_float=is_float)
                s_args.labels.update(val_labels)
            # Use manual k-fold assignments if indicated
            elif val_settings.strategy == 'k-fold-manual':
                t_filters = {
                    s_args.k_header: [j for j in s_args.valid_k if j != s_args.k]
                }
                train_dts = dataset.filter(t_filters)
                val_dts = dataset.filter(filters={s_args.k_header: [s_args.k]})
            # No validation
            elif val_settings.strategy == 'none':
                train_dts = dataset
                val_dts = None
            # Otherwise, calculate k-fold splits
            else:
                if val_settings.strategy == 'k-fold-preserved-site':
                    site_labels = dataset.labels(
                        s_args.k_header,
                        format='name'
                    )[0]  # type: Any
                else:
                    site_labels = None
                train_dts, val_dts = dataset.train_val_split(
                    hp.model_type(),
                    s_args.split_labels,
                    val_strategy=val_settings.strategy,
                    splits=os.path.join(self.root, s_args.splits),
                    val_fraction=val_settings.fraction,
                    val_k_fold=val_settings.k_fold,
                    k_fold_iter=s_args.k,
                    site_labels=site_labels
                )
            # ---- Balance and clip datasets --------------------------------------
            if s_args.bal_headers is None:
                s_args.bal_headers = s_args.outcomes
            print(s_args.outcomes)
            train_dts = train_dts.balance(s_args.bal_headers, hp.training_balance)
            train_dts = train_dts.clip(s_args.max_tiles)
            if val_dts:
                val_dts = val_dts.balance(
                    s_args.bal_headers,
                    hp.validation_balance
                )
                val_dts = val_dts.clip(s_args.max_tiles)
            num_train = len(train_dts.tfrecords())
            num_val = 0 if not val_dts else len(val_dts.tfrecords())
            log.info(f'Using {num_train} training TFRecords, {num_val} validation')
            # --- Prepare additional slide-level input ----------------------------
            if s_args.input_header:
                _res = project_utils._setup_input_labels(
                    dataset,
                    s_args.input_header,
                    val_dts=val_dts
                )
                inpt_labels, feature_sizes, slide_inp = _res
            else:
                inpt_labels = None
                feature_sizes = None
                slide_inp = None
            # --- Initialize model ------------------------------------------------
            # Using the project annotation file, assemble slides for training,
            # as well as the slide annotations dictionary (output labels)
            full_name = s_args.model_name
            if s_args.k is not None:
                full_name += f'-kfold{s_args.k}'
            git_commit = sf.util.detect_git_commit()
            model_dir = sf.util.get_new_model_dir(self.models_dir, full_name)
            # Log model settings and hyperparameters
            config = {
                'slideflow_version': sf.__version__,
                'project': self.name,
                'backend': sf.backend(),
                'git_commit': git_commit,
                'model_name': s_args.model_name,
                'full_model_name': full_name,
                'stage': 'training',
                'img_format': train_dts.img_format,
                'tile_px': hp.tile_px,
                'tile_um': hp.tile_um,
                'max_tiles': s_args.max_tiles,
                'min_tiles': s_args.min_tiles,
                'model_type': hp.model_type(),
                'outcomes': s_args.outcomes,
                'val_outcome_label_headers': s_args.val_outcomes,
                'input_features': s_args.input_header,
                'input_feature_sizes': feature_sizes,
                'input_feature_labels': inpt_labels,
                'outcome_labels': s_args.outcome_labels,
                'dataset_config': self.dataset_config,
                'sources': self.sources,
                'annotations': self.annotations,
                'validation_strategy': val_settings.strategy,
                'validation_fraction': val_settings.fraction,
                'validation_k_fold': val_settings.k_fold,
                'k_fold_i': s_args.k,
                'filters': s_args.filters,
                'hp': hp.get_dict(),
                'training_kwargs': s_args.training_kwargs,
            }
            model_kwargs = {
                'hp': hp,
                'name': full_name,
                'manifest': manifest,
                'feature_names': s_args.input_header,
                'feature_sizes': feature_sizes,
                'outcome_names': s_args.outcomes,
                'val_outcome_names': s_args.val_outcomes,
                'outdir': model_dir,
                'config': config,
                'patients': dataset.patients(),
                'slide_input': slide_inp,
                'labels': s_args.labels,
                'val_labels': s_args.val_labels,
                'mixed_precision': s_args.mixed_precision,
                'use_neptune': self.use_neptune,
                'neptune_api': self.neptune_api,
                'neptune_workspace': self.neptune_workspace,
            }
            process = s_args.ctx.Process(target=_train_val_worker,
                                        args=((train_dts, val_dts),
                                            model_kwargs,
                                            s_args.training_kwargs,
                                            s_args.results_dict,
                                            self.verbosity))
            process.start()
            log.debug(f'Spawning training process (PID: {process.pid})')
            process.join()
