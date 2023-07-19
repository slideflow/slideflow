import csv
import os
import multiprocessing as mp
from collections import defaultdict
from typing import Union, List, Optional, Callable, Tuple, Any
from os.path import join, exists, isdir, dirname
import numpy as np
import pandas as pd
from ._params import (
    _TrainerConfig, ModelConfigCLAM, TrainerConfigCLAM
)
from slideflow import Dataset, log, errors
from slideflow.util import log, path_to_name
import slideflow as sf
import torch
import tensorflow as tf


class MILFeatures():
    """Loads annotations, saved layer activations / features, predictions, 
    and prepares output pandas DataFrame."""

    def __init__(
        self,
        model: Union[str, "tf.keras.models.Model", "torch.nn.Module"],
        bags: Union[np.ndarray, List[str], str],
        slides: list,
        annotations: Union[str, "pd.core.frame.DataFrame"],
        use_lens: bool,
        config: Optional[_TrainerConfig] = None,
        dataset: Optional["sf.Dataset"] = None,
        outcomes: Optional[Union[str, List[str]]] = None,
        attention_pooling: Optional[str] = 'avg',
        device: Optional[Any] = None
    ):
        """Loads in model from Callable or path to model weights and config.
        Saves activations from hidden layer weights, predictions, and attention weight, 
        storing to internal parameters ``self.activations``, ``self.predictions``, ``self.attentions`` dictionaries 
        mapping slides to arrays of weights.
        Converts annotations to DataFrame if necessary.

        Args:
            model (str): Path to model from which to calculate activations.
            bags (str, list): Path or list of feature bags,
            slides (list): List of slides, 
            annotations: Union[str, "pd.core.frame.DataFrame"],
            use_lens (bool): Spec used for generating model args in _get_mil_activations,
        Keyword Args:
            config (:class: `TrainerConfig`]: Trainer for MIL model,
            dataset (:class:`slideflow.Dataset`): Dataset from which to
                generate activations.
            outcomes (str, List[str]): Outcome(s) of the model, 
            attention_pooling (str): pooling strategy for MIL model layers,
            device (Any): device backend for torch tensors
        """
        if type(model) is not str:
            self.model = model
            self.slides = slides
            self.annotations = self._get_annotations(annotations)
        elif (config is not None) and (outcomes is not None) and (dataset is not None):
            log.info(f"Building model {config.model_fn.__name__} from path")
            self.slides, self.model = self.generate_model(
                model, config, dataset, outcomes, bags)
            self.annotations = dataset.annotations
            if isinstance(bags, str):
                bags = dataset.pt_files(bags)
            else:
                bags = np.array([b for b in bags if path_to_name(b) in slides])
        else:
            raise RuntimeError(
                'Model path detected without config, dataset, bags, or outcomes')

        self.num_features, self.predictions, self.attentions, self.activations = self._get_mil_activations(
            self.model, bags, attention_pooling, use_lens, device)

    def generate_model(
        self,
        weights: str,
        config: _TrainerConfig,
        dataset: "sf.Dataset",
        outcomes: Union[str, List[str]],
        bags: Union[str, np.ndarray, List[str]]
    ):
        """Generate model from model path and config.

        Returns callable model object.

        Args:
            weights (str): Path to model weights to load.
            dataset (sf.Dataset): Dataset to evaluation.
            outcomes (str, list(str)): Outcomes.
            bags (str, list(str)): fPath to bags, or list of bag file paths.
                Each bag should contain PyTorch array of features from all tiles in
                a slide, with the shape ``(n_tiles, n_features)``.
            config (:class:`slideflow.mil.TrainerConfigFastAI` or :class:`slideflow.mil.TrainerConfigCLAM`):
                Configuration for building model. If ``weights`` is a path to a
                model directory, will attempt to read ``mil_params.json`` from this
                location and load saved configuration. Defaults to None.
        """

        import torch

        if isinstance(config, TrainerConfigCLAM):
            raise NotImplementedError
        # Check for correct model
        if config.model_config.model.lower() != 'transmil' and config.model_config.model.lower() != 'attention_mil':
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
        return slides, model

    def _get_mil_activations(
        self,
        model: Callable,
        bags: Union[np.ndarray, List[str]],
        attention_pooling: str,
        use_lens: bool,
        device: Optional[Any]
    ):
        """Loads in model from Callable and calculates activations, predictions, and attention weights.

        Args:
            model (Callable): Model from which to calculate activations.
            bags (list): List of feature bags,
            use_lens (bool): Spec used for generating model args in _get_mil_activations,
                generate activations.
            attention_pooling (str): pooling strategy for MIL model layers,
            device (Any): device backend for torch tensors
        """

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
        hs = []
        log.info("Generating predictions...")

        if not hasattr(model, 'calculate_attention'):
            log.warning(
                "Model '{}' does not have a method 'calculate_attention'. "
                "Unable to calculate or display attention heatmaps.".format(
                    model.__class__.__name__
                )
            )

        for bag in bags:
            loaded = torch.load(bag).to(device)
            loaded = torch.unsqueeze(loaded, dim=0)
            with torch.no_grad():
                if use_lens:
                    lens = torch.from_numpy(
                        np.array([loaded.shape[1]])).to(device)
                    model_args = (loaded, lens)
                else:
                    model_args = (loaded,)
                model_out = model(*model_args)
                h = model.get_last_layer_activations(*model_args)
                if device == torch.device('cuda'):
                    h = h.to(torch.device("cpu"))
                hs.append(h)

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

        num_features, acts = self._get_activations(hs)
        atts = self._get_attentions(y_att)
        preds = self._get_predictions(yp)

        return num_features, preds, atts, acts

    def _get_activations(self, hlw):
        """Formats list of activation weights into dictionary of lists, matched by slide.
        Returns the dictionary."""
        if hlw is None or self.slides is None:
            return None, {}
        activations = {}
        for slide, h in zip(self.slides, hlw):
            activations[slide] = h.numpy()[0]

        num_features = hlw[0].shape[1]
        return num_features, activations

    def _get_annotations(self, annotations):
        """Converts passed annotation str to DataFrame if necessary.
        Returns the DataFrame."""
        if annotations is None:
            return None
        elif type(annotations) is str:
            return pd.read_csv(annotations)
        else:
            return annotations

    def _get_attentions(self, atts):
        """Formats list of attention weights into dictionary of lists, matched by slide.
        Returns the dictionary."""
        if atts is None or self.slides is None:
            return None
        attentions = {}
        for slide, att in zip(self.slides, atts):
            attentions[slide] = att
        return attentions

    def _get_predictions(self, preds):
        """Formats list of prediction weights into dictionary of lists, matched by slide.
        Returns the dictionary."""
        if preds is None or self.slides is None:
            return None
        predictions = {}
        for slide, pred in zip(self.slides, preds):
            predictions[slide] = pred
        return predictions

    @classmethod
    def from_df(cls, df: "pd.core.frame.DataFrame", *, annotations=Union[str, "pd.core.frame.DataFrame"]):
        """Load MILFeatures of activations, as exported by :meth:`MILFeatures.to_df()`"""
        obj = cls(None, None, None, None, None)
        if 'slide' in df.columns:
            obj.slides = df['slide'].values
        elif df.index.name == 'slide':
            obj.slides = df.index.values
            df['slide'] = df.index
        else:
            raise ValueError("No slides in DataFrame columns")

        if 'activations' in df.columns:
            obj.activations = {
                s: df.loc[df.slide == s].activations.values.tolist()[0]
                for s in obj.slides
            }
            # obj.num_classes = next(df.iterrows())[1].predictions.shape[0]
        if 'predictions' in df.columns:
            obj.predictions = {
                s: np.stack(df.loc[df.slide == s].predictions.values[0])
                for s in obj.slides
            }
            # obj.num_classes = next(df.iterrows())[1].predictions.shape[0]
        if 'attentions' in df.columns:
            obj.attentions = {
                s: np.stack(df.loc[df.slide == s].attentions.values[0])
                for s in obj.slides
            }
        if annotations:
            obj.annotations = obj._get_annotations(annotations)

        return obj

    def to_df(self) -> pd.core.frame.DataFrame:
        """Export activations to
        a pandas DataFrame.

        Returns:
            pd.core.frame.DataFrame: Dataframe with columns 'slide', 
            'patient', 'activations', 'predictions', and 'attentions'.
        """
        assert self.activations is not None
        assert self.slides is not None

        index = [s for s in self.slides]
        df_dict = {}
        df_dict.update({
            'activations': pd.Series([
                self.activations[s]
                for s in self.slides], index=index)
        })
        if self.predictions:
            df_dict.update({
                'predictions': pd.Series([
                    self.predictions[s]
                    for s in self.slides], index=index)
            })
        if self.attentions:
            df_dict.update({
                'attentions': pd.Series([
                    self.attentions[s]
                    for s in self.slides], index=index)
            })

        df = pd.DataFrame.from_dict(df_dict)
        df['slide'] = df.index

        if self.annotations is None:
            log.warning(
                "No annotation file was given, so patients will not be extracted."
            )
            return df
        patients = self.annotations[['slide', 'patient']]
        df2 = df.set_index('slide').join(
            patients.set_index('slide'), how='inner')
        if self.predictions:
            if self.attentions:
                df2 = df2[['patient', 'activations',
                           'predictions', 'activations']]
            else:
                df2 = df2[['patient', 'activations', 'predictions']]
        elif self.attentions:
            df2 = df2[['patient', 'activations', 'activations']]
        else:
            df2 = df2[['patient', 'activations']]

        return df2
