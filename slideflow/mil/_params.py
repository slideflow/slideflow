"""Model and trainer configuration for MIL models."""

import numpy as np
import os
import torch
import slideflow as sf
import pandas as pd
from torch import nn
from typing import Optional, Union, Callable, List, Tuple, Any, TYPE_CHECKING
from slideflow import log, errors, Dataset

from ._registry import get_trainer, build_model_config

if TYPE_CHECKING:
    from fastai.learner import Learner

# -----------------------------------------------------------------------------

def mil_config(model: Union[str, Callable], trainer: str = 'fastai', **kwargs):
    """Create a multiple-instance learning (MIL) training configuration.

    All models by default are trained with the FastAI trainer. Additional
    trainers and additional models can be installed with ``slideflow-extras``.

    Args:
        model (str, Callable): Either the name of a model, or a custom torch
            module. Valid model names include ``"attention_mil"``,
            ``"transmil"``, and ``"bistro.transformer"``.
        trainer (str): Type of MIL trainer to use. Only 'fastai' is available,
            unless additional trainers are installed.
        **kwargs: All additional keyword arguments are passed to
            :class:`slideflow.mil.TrainerConfig`

    """
    return get_trainer(trainer)(model=model, **kwargs)

# -----------------------------------------------------------------------------

class TrainerConfig:

    tag = 'fastai'

    def __init__(
        self,
        model: Union[str, Callable] = 'attention_mil',
        *,
        aggregation_level: str = 'slide',
        lr: Optional[float] = None,
        wd: float = 1e-5,
        bag_size: int = 512,
        max_val_bag_size: Optional[int] = None,
        fit_one_cycle: bool = True,
        epochs: int = 32,
        batch_size: int = 64,
        drop_last: bool = True,
        save_monitor: str = 'valid_loss',
        weighted_loss: bool = True,
        **kwargs
    ):
        r"""Training configuration for FastAI MIL models.

        This configuration should not be created directly, but rather should
        be created through :func:`slideflow.mil.mil_config`, which will create
        and prepare an appropriate trainer configuration.

        Args:
            model (str, Callable): Either the name of a model, or a custom torch
                module. Valid model names include ``"attention_mil"``,
                ``"transmil"``, and ``"bistro.transformer"``.

        Keyword args:
            aggregation_level (str): When equal to ``'slide'`` each bag
                contains tiles from a single slide. When equal to ``'patient'``
                tiles from all slides of a patient are grouped together.
            lr (float, optional): Learning rate. If ``fit_one_cycle=True``,
                this is the maximum learning rate. If None, uses the Leslie
                Smith `LR Range test <https://arxiv.org/abs/1506.01186>`_ to
                find an optimal learning rate. Defaults to None.
            wd (float): Weight decay. Only used if ``fit_one_cycle=False``.
                Defaults to 1e-5.
            bag_size (int): Bag size. Defaults to 512.
            max_val_bag_size (int, optional): Maximum validation bag size. If
                None, all validation bags will be unclipped and unpadded (full size).
                Defaults to None.
            fit_one_cycle (bool): Use `1cycle <https://sgugger.github.io/the-1cycle-policy.html>`_
                learning rate schedule. Defaults to True.
            epochs (int): Maximum number of epochs. Defaults to 32.
            batch_size (int): Batch size. Defaults to 64.
            **kwargs: All additional keyword arguments are passed to
                :class:`slideflow.mil.MILModelConfig`.

        """
        self._aggregation_level = aggregation_level
        self.lr = lr
        self.wd = wd
        self.bag_size = bag_size
        self.max_val_bag_size = max_val_bag_size
        self.fit_one_cycle = fit_one_cycle
        self.epochs = epochs
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.save_monitor = save_monitor
        self.weighted_loss = weighted_loss
        if isinstance(model, str):
            self.model_config = build_model_config(model, **kwargs)
        else:
            sf.log.info("Attempting to load custom model class for MIL training.")
            from slideflow.mil import MILModelConfig
            self.model_config = MILModelConfig(model, **kwargs)
        self.model_config.verify_trainer(self)

    def __str__(self):
        out = f"{self.__class__.__name__}("
        for p, val in self.to_dict().items():
            if p != 'model_config':
                out += '\n  {}={!r}'.format(p, val)
        out += '\n)'
        return out

    @property
    def model_fn(self):
        """MIL model architecture (class/module)."""
        return self.model_config.model_fn

    @property
    def loss_fn(self):
        """MIL loss function."""
        return self.model_config.loss_fn

    @property
    def is_multimodal(self):
        """Whether the model is multimodal."""
        return self.model_config.is_multimodal

    @property
    def model_type(self):
        """Type of model (classification or regression)."""
        return self.model_config.model_type

    @property
    def aggregation_level(self):
        """Aggregation level for MIL training."""
        if hasattr(self, '_aggregation_level'):
            return self._aggregation_level
        else:
            return 'slide'

    @aggregation_level.setter
    def aggregation_level(self, value):
        if value not in ('slide', 'patient'):
            raise ValueError("Aggregation level must be either 'slide' or 'patient'.")
        self._aggregation_level = value

    def _verify_eval_params(self, **kwargs):
        pass

    def is_classification(self):
        """Whether the model is a classification model."""
        return self.model_config.is_classification()

    def get_metrics(self):
        """Get model metrics.

        Returns:
            List[Callable]: List of metrics to use for model evaluation.
            Defaults to RocAuc for classification models, and mse and Pearson
            correlation coefficient for regression models.

        """
        from fastai.vision.all import RocAuc, mse, PearsonCorrCoef

        model_metrics = self.model_config.get_metrics()

        if self.is_classification():
            fallback = [RocAuc()]
        else:
            fallback = [mse, PearsonCorrCoef()]
        return model_metrics or fallback

    def prepare_training(
        self,
        outcomes: Union[str, List[str]],
        exp_label: Optional[str],
        outdir: Optional[str]
    ) -> str:
        """Prepare for training.

        Sets up the output directory for the model.

        Args:
            outcomes (str, list(str)): Outcomes.
            exp_label (str): Experiment label.
            outdir (str): Output directory.

        Returns:
            str: Output directory.

        """
        log.info("Training FastAI MIL model with config:")
        log.info(f"{str(self)}")
        # Set up experiment label
        if exp_label is None:
            try:
                if isinstance(self.model_config.model, str):
                    model_name = self.model_config.model
                else:
                    model_name = self.model_config.model.__name__
                exp_label = '{}-{}'.format(
                    model_name,
                    "-".join(outcomes if isinstance(outcomes, list) else [outcomes])
                )
            except Exception:
                exp_label = 'no_label'
        # Set up output model directory
        if outdir:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outdir = sf.util.create_new_model_dir(outdir, exp_label)
        return outdir

    def build_model(self, n_in: int, n_out: int, **kwargs):
        """Build the model.

        Args:
            n_in (int): Number of input features.
            n_out (int): Number of output features.

        Keyword args:
            **kwargs: Additional keyword arguments to pass to the model constructor.

        Returns:
            torch.nn.Module: PyTorch model.

        """
        if self.model_config.model_kwargs:
            model_kw = self.model_config.model_kwargs
        else:
            model_kw = dict()
        return self.model_config.build_model(n_in, n_out, **model_kw, **kwargs)

    def to_dict(self):
        """Converts this training configuration to a dictionary."""
        d = {k:v for k,v in vars(self).items()
                if k not in (
                    'self',
                    'model_fn',
                    'loss_fn',
                    'build_model',
                    'is_multimodal'
                ) and not k.startswith('_')}
        if self.model_config is None:
            return d
        else:
            d.update(self.model_config.to_dict())
            del d['model_config']
            return d

    def json_dump(self):
        """Converts this training configuration to a JSON-compatible dict."""
        return dict(
            trainer=self.tag,
            params=self.to_dict()
        )

    def predict(self, model, bags, attention=False, **kwargs):
        """Generate model prediction from bags.

        Args:
            model (torch.nn.Module): Loaded PyTorch MIL model.
            bags (torch.Tensor): Bags, with shape ``(n_bags, n_tiles, n_features)``.

        Keyword args:
            attention (bool): Whether to return attention maps.

        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: Predictions and attention.

        """
        self._verify_eval_params(**kwargs)
        return self.model_config.predict(model, bags, attention=attention, **kwargs)

    def batched_predict(
        self,
        model: "torch.nn.Module",
        loaded_bags: torch.Tensor,
        **kwargs
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate predictions from a batch of bags.

        Args:
            model (torch.nn.Module): Loaded PyTorch MIL model.
            loaded_bags (torch.Tensor): Loaded bags, with shape ``(n_bags, n_tiles, n_features)``.

        Keyword args:
            device (torch.device, optional): Device on which to run the model.
                If None, uses the default device.
            forward_kwargs (dict, optional): Additional keyword arguments to
                pass to the model's forward function.
            attention (bool): Whether to return attention maps.
            attention_pooling (str): Attention pooling strategy. Either 'avg'
                or 'max'. Defaults to 'avg'.
            uq (bool): Whether to return uncertainty quantification.

        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: Predictions and attention.

        """
        return self.model_config.batched_predict(model, loaded_bags, **kwargs)

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        outcomes: Union[str, List[str]],
        bags: Union[str, List[str]],
        *,
        outdir: str = 'mil',
        exp_label: Optional[str] = None,
        **kwargs
    ) -> "Learner":
        """Train a multiple-instance learning (MIL) model.

        Args:
            config (:class:`slideflow.mil.TrainerConfig`):
                Trainer and model configuration.
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
            attention_heatmaps (bool): Generate attention heatmaps for slides.
                Not available for multi-modal MIL models. Defaults to False.
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
        from slideflow.mil.train import _train_mil, _train_multimodal_mil

        # Prepare output directory
        outdir = self.prepare_training(outcomes, exp_label, outdir)

        # Use training data as validation if no validation set is provided
        if val_dataset is None:
            sf.log.info(
                "Training without validation; metrics will be calculated on training data."
            )
            val_dataset = train_dataset

        # Check if multimodal training
        if self.is_multimodal:
            train_fn = _train_multimodal_mil
        else:
            train_fn = _train_mil

        # Execute training
        return train_fn(
            self,
            train_dataset,
            val_dataset,
            outcomes,
            bags,
            outdir=outdir,
            **kwargs
        )

    def eval(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        outcomes: Union[str, List[str]],
        bags: Union[str, List[str]],
        *,
        outdir: str = 'mil',
        attention_heatmaps: bool = False,
        uq: bool = False,
        aggregation_level: Optional[str] = None,
        params: Optional[dict] = None,
        **heatmap_kwargs
    ) -> pd.DataFrame:
        """Evaluate a multiple-instance learning model.

        Saves results for the evaluation in the target folder, including
        predictions (parquet format), attention (Numpy format for each slide),
        and attention heatmaps (if ``attention_heatmaps=True``).

        Logs classifier metrics (AUROC and AP) to the console.

        Args:
            model (torch.nn.Module): Loaded PyTorch MIL model.
            dataset (sf.Dataset): Dataset to evaluation.
            outcomes (str, list(str)): Outcomes.
            bags (str, list(str)): Path to bags, or list of bag file paths.
                Each bag should contain PyTorch array of features from all tiles in
                a slide, with the shape ``(n_tiles, n_features)``.

        Keyword arguments:
            outdir (str): Path at which to save results.
            attention_heatmaps (bool): Generate attention heatmaps for slides.
                Not available for multi-modal MIL models. Defaults to False.
            interpolation (str, optional): Interpolation strategy for smoothing
                attention heatmaps. Defaults to 'bicubic'.
            cmap (str, optional): Matplotlib colormap for heatmap. Can be any
                valid matplotlib colormap. Defaults to 'inferno'.
            norm (str, optional): Normalization strategy for assigning heatmap
                values to colors. Either 'two_slope', or any other valid value
                for the ``norm`` argument of ``matplotlib.pyplot.imshow``.
                If 'two_slope', normalizes values less than 0 and greater than 0
                separately. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe of predictions.
        """
        from slideflow.mil.eval import run_eval

        params_to_verify = dict(
            attention_heatmaps=attention_heatmaps,
            heatmap_kwargs=heatmap_kwargs,
            uq=uq,
            aggregation_level=aggregation_level
        )

        self._verify_eval_params(**params_to_verify)
        self.model_config._verify_eval_params(**params_to_verify)

        eval_kwargs = dict(
            dataset=dataset,
            outcomes=outcomes,
            bags=bags,
            config=self,
            outdir=outdir,
            params=params,
            aggregation_level=(aggregation_level or self.aggregation_level)
        )

        return run_eval(
            model,
            attention_heatmaps=attention_heatmaps,
            uq=uq,
            **heatmap_kwargs,
            **eval_kwargs
        )

    def _build_dataloader(
        self,
        bags,
        targets,
        encoder,
        dataset_kwargs,
        dataloader_kwargs,
    ) -> "torch.utils.DataLoader":

        return self.model_config._build_dataloader(
            bags,
            targets,
            encoder,
            dataset_kwargs=dataset_kwargs,
            dataloader_kwargs=dataloader_kwargs
        )

    def build_train_dataloader(
        self,
        bags,
        targets,
        encoder,
        *,
        dataset_kwargs = None,
        dataloader_kwargs = None
    ) -> "torch.utils.DataLoader":
        """Build a training dataloader.

        Args:
            bags (list): List of bags.
            targets (list): List of targets.
            encoder (torch.nn.Module): Encoder for bags.

        Keyword args:
            dataset_kwargs (dict): Keyword arguments for the dataset.
            dataloader_kwargs (dict): Keyword arguments for the dataloader.

        Returns:
            torch.utils.DataLoader: Training dataloader.

        """
        dataset_kwargs = dataset_kwargs or dict()
        dataloader_kwargs = dataloader_kwargs or dict()

        # Dataset kwargs
        if 'bag_size' not in dataset_kwargs:
            dataset_kwargs['bag_size'] = self.bag_size

        # Dataloader kwargs
        if 'drop_last' not in dataloader_kwargs:
            dataloader_kwargs['drop_last'] = self.drop_last
        if 'batch_size' not in dataloader_kwargs:
            dataloader_kwargs['batch_size'] = self.batch_size
        if 'shuffle' not in dataloader_kwargs:
            dataloader_kwargs['shuffle'] = True

        return self._build_dataloader(
            bags,
            targets,
            encoder,
            dataset_kwargs=dataset_kwargs,
            dataloader_kwargs=dataloader_kwargs
        )

    def build_val_dataloader(
        self,
        bags,
        targets,
        encoder,
        *,
        dataset_kwargs = None,
        dataloader_kwargs = None
    ) -> "torch.utils.DataLoader":
        """Build a validation dataloader.

        Args:
            bags (list): List of bags.
            targets (list): List of targets.
            encoder (torch.nn.Module): Encoder for bags.

        Keyword args:
            dataset_kwargs (dict): Keyword arguments for the dataset.
            dataloader_kwargs (dict): Keyword arguments for the dataloader.

        Returns:
            torch.utils.DataLoader: Validation dataloader.

        """
        dataset_kwargs = dataset_kwargs or dict()
        dataloader_kwargs = dataloader_kwargs or dict()

        # Dataset kwargs
        if 'bag_size' not in dataset_kwargs:
            dataset_kwargs['bag_size'] = None
        if 'max_bag_size' not in dataset_kwargs:
            dataset_kwargs['max_bag_size'] = self.max_val_bag_size

        # Dataloader kwargs
        if 'batch_size' not in dataloader_kwargs:
            dataloader_kwargs['batch_size'] = 1

        return self._build_dataloader(
            bags,
            targets,
            encoder,
            dataset_kwargs=dataset_kwargs,
            dataloader_kwargs=dataloader_kwargs
        )

    def inspect_batch(self, batch) -> Tuple[int, int]:
        """Inspect a batch of data.

        Args:
            batch: One batch of data.

        Returns:
            Tuple[int, int]: Number of input and output features.

        """
        return self.model_config.inspect_batch(batch)

    def run_metrics(self, df, level='slide', outdir=None):
        """Run metrics and save plots to disk.

        Args:
            df (pd.DataFrame): Dataframe with predictions and outcomes.
            level (str): Level at which to calculate metrics. Either 'slide' or 'patient'.
            outdir (str): Output directory for saving metrics.

        """
        self.model_config.run_metrics(df, level=level, outdir=outdir)


# -----------------------------------------------------------------------------

class MILModelConfig:

    losses = {
        'cross_entropy': nn.CrossEntropyLoss,
        'mse': nn.MSELoss,
        'mae': nn.L1Loss,
        'huber': nn.SmoothL1Loss
    }

    def __init__(
        self,
        model: Union[str, Callable] = 'attention_mil',
        *,
        use_lens: Optional[bool] = None,
        apply_softmax: bool = True,
        model_kwargs: Optional[dict] = None,
        validate: bool = True,
        loss: Union[str, Callable] = 'cross_entropy',
        **kwargs
    ) -> None:
        """Model configuration for an MIL model.

        Args:
            model (str, Callable): Either the name of a model, or a custom torch
                module. Valid model names include ``"attention_mil"`` and
                ``"transmil"``. Defaults to 'attention_mil'.

        Keyword args:
            use_lens (bool, optional): Whether the model expects a second
                argument to its ``.forward()`` function, an array with the
                bag size for each slide. If None, will default to True for
                ``'attention_mil'`` models and False otherwise.
                Defaults to None.
            apply_softmax (bool): Whether to apply softmax to model outputs.
                Defaults to True. Ignored if the model is not a classification
                model.
            model_kwargs (dict, optional): Additional keyword arguments to pass
                to the model constructor. Defaults to None.
            validate (bool): Whether to validate the keyword arguments. If True,
                will raise an error if any unrecognized keyword arguments are
                passed. Defaults to True.
            loss (str, Callable): Loss function. Defaults to 'cross_entropy'.

        """
        self.model = model
        self._apply_softmax = apply_softmax
        self.model_kwargs = model_kwargs
        self.loss = loss
        if use_lens is None and (hasattr(self.model_fn, 'use_lens')
                                 and self.model_fn.use_lens):
            self.use_lens = True
        elif use_lens is None:
            self.use_lens = False
        else:
            self.use_lens = use_lens
        if kwargs and validate:
            raise errors.UnrecognizedHyperparameterError("Unrecognized parameters: {}".format(
                ', '.join(list(kwargs.keys()))
            ))
        elif kwargs:
            log.warning("Ignoring unrecognized parameters: {}".format(
                ', '.join(list(kwargs.keys()))
            ))

    @property
    def apply_softmax(self):
        """Whether softmax will be applied to model outputs."""
        return self.is_classification() and self._apply_softmax

    @property
    def model_fn(self):
        """MIL model architecture (class/module)."""
        if not isinstance(self.model, str):
            return self.model
        return sf.mil.get_model(self.model)

    @property
    def loss_fn(self):
        """MIL loss function."""
        return self.losses[self.loss]

    @property
    def is_multimodal(self):
        """Whether the model is multimodal."""
        return ((isinstance(self.model, str) and self.model.lower() == 'mm_attention_mil')
                or (hasattr(self.model_fn, 'is_multimodal')
                    and self.model_fn.is_multimodal))

    @property
    def rich_name(self):
        return f"[bold]{self.model_fn.__name__}[/]"

    @property
    def model_type(self):
        """Type of model (classification or regression)."""
        if self.loss == 'cross_entropy':
            return 'classification'
        else:
            return 'regression'

    def is_classification(self):
        """Whether the model is a classification model."""
        return self.model_type == 'classification'

    def verify_trainer(self, trainer):
        pass

    def get_metrics(self):
        return None

    def to_dict(self):
        """Converts this model configuration to a dictionary."""
        d = {k:v for k,v in vars(self).items()
                if k not in (
                    'self',
                    'model_fn',
                    'loss_fn',
                    'build_model',
                    'is_multimodal'
                ) and not k.startswith('_')}
        if not isinstance(d['model'], str):
            d['model'] = d['model'].__name__
        return d

    def _verify_eval_params(self, **kwargs):
        """Verify evaluation parameters for the model."""

        if self.is_multimodal:
            if kwargs.get('attention_heatmaps'):
                raise ValueError(
                    "Attention heatmaps cannot yet be exported for multi-modal "
                    "models. Please use Slideflow Studio for visualization of "
                    "multi-modal attention."
                )
            if kwargs.get('heatmap_kwargs'):
                kwarg_names = ', '.join(list(kwargs['heatmap_kwargs'].keys()))
                raise ValueError(
                    f"Unrecognized keyword arguments: '{kwarg_names}'. Attention "
                    "heatmap keyword arguments are not currently supported for "
                    "multi-modal models."
                )

    def inspect_batch(self, batch) -> Tuple[int, int]:
        """Inspect a batch of data.

        Args:
            batch: One batch of data.

        Returns:
            Tuple[int, int]: Number of input and output features.

        """
        if self.is_multimodal:
            if self.use_lens:
                n_in = [b[0].shape[-1] for b in batch[:-1]]
            else:
                n_in = [b.shape[-1] for b in batch[:-1][0]]
        else:
            n_in = batch[0].shape[-1]
        targets = batch[-1]
        if len(targets.shape) == 1:
            n_out = 1
        else:
            n_out = targets.shape[-1]
        return n_in, n_out

    def build_model(self, n_in: int, n_out: int, **kwargs):
        """Build the model.

        Args:
            n_in (int): Number of input features.
            n_out (int): Number of output features.

        Keyword args:
            **kwargs: Additional keyword arguments to pass to the model constructor.

        Returns:
            torch.nn.Module: PyTorch model.

        """
        log.info(f"Building model {self.rich_name} (n_in={n_in}, n_out={n_out})")
        return self.model_fn(n_in, n_out, **kwargs)

    def _build_dataloader(
        self,
        bags,
        targets,
        encoder,
        *,
        dataset_kwargs = None,
        dataloader_kwargs = None
    ) -> "torch.utils.DataLoader":
        from fastai.vision.all import DataLoader
        from slideflow.mil import data as data_utils

        dataset_kwargs = dataset_kwargs or dict()
        dataloader_kwargs = dataloader_kwargs or dict()

        if 'use_lens' not in dataset_kwargs:
            dataset_kwargs['use_lens'] = self.use_lens

        if self.is_multimodal:
            dts_fn = data_utils.build_multibag_dataset
        else:
            dts_fn = data_utils.build_dataset

        dataset = dts_fn(bags, targets, encoder=encoder, **dataset_kwargs)
        dataloader = DataLoader(dataset, **dataloader_kwargs)
        return dataloader

    def predict(self, model, bags, attention=False, apply_softmax=None, **kwargs):
        """Generate model prediction from bags.

        Args:
            model (torch.nn.Module): Loaded PyTorch MIL model.
            bags (torch.Tensor): Bags, with shape ``(n_bags, n_tiles, n_features)``.

        Keyword args:
            attention (bool): Whether to return attention maps.
            apply_softmax (bool): Whether to apply softmax to model outputs.
            attention_pooling (bool): Whether to pool attention maps with
                average pooling. Defaults to None.

        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: Predictions and attention.

        """
        self._verify_eval_params(**kwargs)

        from slideflow.mil.eval import predict_from_bags, predict_from_multimodal_bags

        if apply_softmax is None:
            apply_softmax = self.apply_softmax

        pred_fn = predict_from_multimodal_bags if self.is_multimodal else predict_from_bags
        return pred_fn(
            model,
            bags,
            attention=attention,
            use_lens=self.use_lens,
            apply_softmax=apply_softmax,
            **kwargs
        )

    def batched_predict(
        self,
        model: "torch.nn.Module",
        loaded_bags: torch.Tensor,
        *,
        device: Optional[Any] = None,
        forward_kwargs: Optional[dict] = None,
        attention: bool = False,
        attention_pooling: Optional[str] = None,
        uq: bool = False,
        apply_softmax: Optional[bool] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate predictions from a batch of bags.

        More efficient than calling :meth:`predict` multiple times.

        Args:
            model (torch.nn.Module): Loaded PyTorch MIL model.
            loaded_bags (torch.Tensor): Loaded bags, with shape ``(n_bags, n_tiles, n_features)``.

        Keyword args:
            device (torch.device, optional): Device on which to run the model.
                If None, uses the default device.
            forward_kwargs (dict, optional): Additional keyword arguments to
                pass to the model's forward function.
            attention (bool): Whether to return attention maps.
            attention_pooling (str): Attention pooling strategy. Either 'avg'
                or 'max'. Defaults to None.
            uq (bool): Whether to return uncertainty quantification.

        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: Predictions and attention.

        """
        from slideflow.mil.eval import run_inference

        if apply_softmax is None:
            apply_softmax = self.apply_softmax

        return run_inference(
            model,
            loaded_bags,
            attention=attention,
            attention_pooling=attention_pooling,
            forward_kwargs=forward_kwargs,
            apply_softmax=apply_softmax,
            use_lens=self.use_lens,
            device=device,
            uq=uq,
        )

    def run_metrics(self, df, level='slide', outdir=None) -> None:
        """Run metrics and save plots to disk.

        Args:
            df (pd.DataFrame): Dataframe with predictions and outcomes.
            level (str): Level at which to calculate metrics. Either 'slide' or 'patient'.
            outdir (str): Output directory for saving metrics.

        """
        if self.is_classification():
            sf.stats.metrics.classification_metrics(df, level=level, data_dir=outdir)
        else:
            sf.stats.metrics.regression_metrics(df, level=level, data_dir=outdir)

# -----------------------------------------------------------------------------

