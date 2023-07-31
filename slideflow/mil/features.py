import csv
import os
from typing import Union, List, Optional, Callable, Tuple, Any
from os.path import join, exists, isdir, dirname
import numpy as np
import pandas as pd
import re
from ._params import (
    _TrainerConfig, ModelConfigCLAM, TrainerConfigCLAM
)
from .models.clam import CLAM_SB, CLAM_MB
from .models.mil_fc import MIL_fc, MIL_fc_mc
from .models.att_mil import Attention_MIL
from .models.transmil import TransMIL
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
        config: Optional[_TrainerConfig] = None,
        dataset: Optional["sf.Dataset"] = None,
        outcomes: Optional[Union[str, List[str]]] = None,
        attention_pooling: Optional[str] = 'avg',
        device: Optional[Any] = None
    ):
        """Loads in model from Callable or path to model weights and config.
        Saves activations from last hidden layer weights, predictions, and 
        attention weight, storing to internal parameters ``self.activations``,
         ``self.predictions``, ``self.attentions`` dictionaries 
        mapping slides to arrays of weights.
        Converts annotations to DataFrame if necessary.

        Args:
            model (str): Path to model from which to calculate activations.
            bags (str, list): Path or list of feature bags,
            slides (list): List of slides, 
            annotations: Union[str, "pd.core.frame.DataFrame"],
        Keyword Args:
            config (:class:`TrainerConfig`]: Trainer for MIL model,
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
            if type(model) is Attention_MIL:
                use_lens = True
            else:
                use_lens = False
        elif (config is not None) and (outcomes is not None) and \
                (dataset is not None):
            log.info(f"Building model {config.model_fn.__name__} from path")
            self.slides, self.model, use_lens = self._generate_model(
                model, config, dataset, outcomes, bags)
            self.annotations = dataset.annotations
            if isinstance(bags, str):
                bags = dataset.pt_files(bags)
            else:
                bags = np.array([b for b in bags if path_to_name(b) in slides])
        else:
            raise RuntimeError(
                'Model path detected without config, dataset, bags, or \
                    outcomes')
        if self.model:
            self.num_features, self.predictions, self.attentions, \
                self.activations = self._get_mil_activations(
                    self.model, bags, attention_pooling, use_lens, device)

    def _generate_model(
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
            config (:class:`slideflow.mil.TrainerConfigFastAI` or 
            :class:`slideflow.mil.TrainerConfigCLAM`):
                Configuration for building model. If ``weights`` is a path to a
                model directory, will attempt to read ``mil_params.json`` from 
                this location and load saved configuration. Defaults to None.
            dataset (:class:`slideflow.Dataset`): Dataset to evaluation.
            outcomes (str, list(str)): Outcomes.
            bags (str, list(str)): Path to bags, or list of bag file paths.
                Each bag should contain PyTorch array of features from all tiles
                in a slide, with the shape ``(n_tiles, n_features)``.
        """

        import torch

        if isinstance(config, TrainerConfigCLAM):
            raise NotImplementedError
        # Check for correct model
        acceptable_models = ['transmil', 'attention_mil', 'clam_sb', 'clam_mb']
        if config.model_config.model.lower() not in acceptable_models:
            raise errors.ModelErrors(
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

        try:
            use_lens = config.model_config.use_lens
        except AttributeError:
            use_lens = False

        return slides, model, use_lens

    def _get_mil_activations(
        self,
        model: Callable,
        bags: Union[np.ndarray, List[str]],
        attention_pooling: str,
        use_lens: bool,
        device: Optional[Any]
    ):
        """Loads in model from Callable and calculates predictions,
        attentions, and activations weights.

        Args:
            model (Callable): Model from which to calculate activations.
            bags (list): List of feature bags,
            attention_pooling (str): pooling strategy for MIL model layers,
            use_lens (bool): Spec used for generating model args in 
                _get_mil_activations, generate activations.
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

                if type(model) is Attention_MIL or type(model) is TransMIL:
                    model_out = model(*model_args)
                    h = model.get_last_layer_activations(*model_args)
                    att = torch.squeeze(model.calculate_attention(*model_args))
                    if len(att.shape) == 2:
                        # Attention needs to be pooled
                        if attention_pooling == 'avg':
                            att = torch.mean(att, dim=-1)
                        elif attention_pooling == 'max':
                            att = torch.amax(att, dim=-1)
                        else:
                            raise ValueError(
                                "Unrecognized attention pooling strategy \
                                    '{}'".format(
                                    attention_pooling
                                )
                            )
                    y_att.append(att.cpu().numpy())
                elif type(model) is MIL_fc or type(model) is MIL_fc_mc:
                    model_out = model(*model_args)
                    h = model.get_last_layer_activations(*model_args)
                    y_att = None
                else:
                    model_out = model(*model_args)[0]
                    h, A = model.get_last_layer_activations(*model_args)

                    if A.shape[0] == 1:
                        y_att.append(A.cpu().numpy()[0])
                    else:
                        y_att.append(A.cpu().numpy())

                if device == torch.device('cuda'):
                    h = h.to(torch.device("cpu"))
                hs.append(h)

                y_pred.append(torch.nn.functional.softmax(
                    model_out, dim=1).cpu().numpy())
        yp = np.concatenate(y_pred, axis=0)

        num_features, acts = self._get_activations(hs)
        atts = self._get_attentions(y_att)
        preds = self._get_predictions(yp)

        return num_features, preds, atts, acts

    def _get_activations(self, hlw):
        """Formats list of activation weights into dictionary of lists, 
        matched by slide."""
        if hlw is None or self.slides is None:
            return None, {}
        activations = {}
        for slide, h in zip(self.slides, hlw):
            activations[slide] = h.numpy()

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
        """Formats list of attention weights into dictionary of lists, 
        matched by slide."""
        if atts is None or self.slides is None:
            return None
        attentions = {}
        for slide, att in zip(self.slides, atts):
            attentions[slide] = att
        return attentions

    def _get_predictions(self, preds):
        """Formats list of prediction weights into dictionary of lists, 
        matched by slide."""
        if preds is None or self.slides is None:
            return None
        predictions = {}
        for slide, pred in zip(self.slides, preds):
            predictions[slide] = pred
        return predictions

    def _format(self, column):
        """Formats dataframe columns to numpy arrays of floats"""
        numbers = re.findall(r'-?\d+\.\d+', column)
        # Convert numbers to floats
        return np.array([float(num) for num in numbers])

    @classmethod
    def from_df(cls, df: "pd.core.frame.DataFrame", *,
                annotations: Union[str, "pd.core.frame.DataFrame"] = None):
        """Load MILFeatures of activations, as exported by 
        :meth:`MILFeatures.to_df()`"""
        obj = cls(None, None, None, None)
        if 'slide' in df.columns:
            obj.slides = df['slide'].values
        elif df.index.name == 'slide':
            obj.slides = df.index.values
            df['slide'] = df.index
        else:
            raise ValueError("No slides in DataFrame columns")

        if 'activations' in df.columns:
            df['activations'] = df['activations'].apply(obj._format)
            obj.activations = {
                s: np.stack(df.loc[df.slide == s].activations.values)
                for s in obj.slides
            }
        else:
            act_cols = [col for col in df.columns if 'activations_' in col]
            if act_cols:
                obj.activations = {}
                for c in act_cols:
                    df[c] = df[c].apply(obj._format)
                for s in obj.slides:
                    r = [df.loc[df.slide == s][act_cols].values.tolist()[0]]
                    if len(r[0]) > 2:
                        raise NotImplementedError(
                            "More than 1 attention branches not implemented")
                    obj.activations[s] = np.vstack((r[0][0], r[0][1]))
            else:
                raise ValueError("No activations in DataFrame columns")

        if 'predictions' in df.columns:
            df['predictions'] = df['predictions'].apply(obj._format)
            obj.predictions = {
                s: np.stack(df.loc[df.slide == s].predictions.values[0])
                for s in obj.slides
            }

        if 'attentions' in df.columns:
            df['attentions'] = df['attentions'].apply(obj._format)
            obj.attentions = {
                s: np.stack(df.loc[df.slide == s].attentions.values[0])
                for s in obj.slides
            }
        else:
            att_cols = [col for col in df.columns if 'attentions_' in col]
            if att_cols:
                obj.attentions = {}
                for c in att_cols:
                    df[c] = df[c].apply(obj._format)
                for s in obj.slides:
                    r = [df.loc[df.slide == s][att_cols].values.tolist()[0]]
                    if len(r[0]) > 2:
                        raise NotImplementedError(
                            "More than 1 attention branches not implemented")
                    obj.attentions[s] = np.vstack((r[0][0], r[0][1]))

        if annotations:
            obj.annotations = obj._get_annotations(annotations)

        return obj

    def to_df(self, predictions=True, attentions=True
              ) -> pd.core.frame.DataFrame:
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

        branches = list(self.activations.values())[0].shape
        if len(branches) == 1:
            branches = 1
        else:
            branches = branches[0]

        if branches == 1:
            df_dict.update({
                'activations': pd.Series([
                    self.activations[s][0]
                    for s in self.slides], index=index)
            })
        else:
            for b in range(branches):
                name = 'activations_{}'.format(b)
                df_dict.update({
                    name: pd.Series([
                        self.activations[s][b]
                        for s in self.slides], index=index)
                })

        if predictions and self.predictions:
            df_dict.update({
                'predictions': pd.Series([
                    self.predictions[s]
                    for s in self.slides], index=index)
            })
        if attentions and self.attentions:
            if branches == 1:
                df_dict.update({
                    'attentions': pd.Series([
                        list(self.attentions[s])
                        if len(self.attentions[s]) == 1
                        else self.attentions[s]
                        for s in self.slides], index=index)
                })
            else:
                for b in range(branches):
                    name = 'attentions_{}'.format(b)
                    df_dict.update({
                        name: pd.Series([
                            self.attentions[s][b]
                            for s in self.slides], index=index)
                    })

        df = pd.DataFrame.from_dict(df_dict)
        df['slide'] = df.index

        if self.annotations is None:
            log.warning(
                "No annotation file was given. Patients will not be extracted."
            )
            return df

        patients = self.annotations[['slide', 'patient']]
        df2 = df.set_index('slide').join(
            patients.set_index('slide'), how='inner')
        p = df2.pop('patient')
        df2.insert(0, 'patient', p)
        return df2
