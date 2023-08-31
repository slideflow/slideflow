import re
import numpy as np
import pandas as pd
import torch
import slideflow as sf
import os
from typing import Union, List, Optional, Callable, Any, TYPE_CHECKING
from os.path import join
from slideflow import log, errors
from slideflow.util import log, path_to_name

from ._params import (
    _TrainerConfig
)
from .models.mil_fc import MIL_fc, MIL_fc_mc
from .models.att_mil import Attention_MIL
from .models.transmil import TransMIL
from .utils import load_model_weights

if TYPE_CHECKING:
    try:
        import tensorflow as tf
    except ImportError:
        pass

# -----------------------------------------------------------------------------

class MILFeatures:
    """Loads annotations, saved layer activations / features, predictions, 
    and prepares output pandas DataFrame."""

    def __init__(
        self,
        model: Union[str, "tf.keras.models.Model", "torch.nn.Module"],
        bags: Union[np.ndarray, List[str], str],
        *,
        slides: Optional[list],
        config: Optional[_TrainerConfig] = None,
        dataset: Optional["sf.Dataset"] = None,
        attention_pooling: Optional[str] = 'avg',
        device: Optional[Any] = None
    ) -> None:
        """Loads in model from Callable or path to model weights and config.
        Saves activations from last hidden layer weights, predictions, and 
        attention weight, storing to internal parameters ``self.activations``,
         ``self.predictions``, ``self.attentions`` dictionaries 
        mapping slides to arrays of weights.
        Converts annotations to DataFrame if necessary.

        Args:
            model (str): Path to model from which to calculate activations.
            bags (str, list): Path or list of feature bags.
            slides (list): List of slides. 
        Keyword Args:
            config (:class:`TrainerConfig`]: Trainer for MIL model,
            dataset (:class:`slideflow.Dataset`): Dataset from which to
                generate activations.
            outcomes (str, List[str]): Outcome(s) of the model, 
            attention_pooling (str): pooling strategy for MIL model layers,
            device (Any): device backend for torch tensors
        """

        # --- Prepare data ----------------------------------------------------
        # Find bags.
        if isinstance(bags, str) and dataset is not None:
            bags = dataset.pt_files(bags)
        elif isinstance(bags, str) and slides:
            bags = np.array([
                join(bags, f) for f in os.listdir(bags)
                if f.endswith('.pt') and path_to_name(f) in slides
            ])
        elif isinstance(bags, str):
            bags = np.array([
                join(bags, f) for f in os.listdir(bags)
                if f.endswith('.pt')
            ])
        elif slides:
            bags = np.array([b for b in bags if path_to_name(b) in slides])
        elif dataset:
            bags = np.array([b for b in bags 
                             if path_to_name(b) in dataset.slides()])

        # Determine slides.
        self.slides = np.array([path_to_name(b) for b in bags])

        # --- Prepare model ---------------------------------------------------
        # Load or build the model.
        if isinstance(model, str):
            self.model, config = load_model_weights(model, config)
            use_lens = config.model_config.use_lens
        else:
            self.model = model
            if isinstance(model, Attention_MIL):
                use_lens = True
            else:
                use_lens = False

        # Ensure model is compatible.
        acceptable_models = ['transmil', 'attention_mil', 'clam_sb', 'clam_mb']
        if config.model_config.model.lower() not in acceptable_models:
            raise errors.ModelErrors(
                f"Model {config.model_config.model} is not supported.")
        
        # --- Generate activations --------------------------------------------
        n_feat, preds, attention, act = self._get_mil_activations(
            self.model, bags, attention_pooling, use_lens, device
        )
        self.num_features = n_feat
        self.predictions = preds
        self.attentions = attention
        self.activations = act

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

                if isinstance(model, (Attention_MIL, TransMIL)):
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
                elif isinstance(model, (MIL_fc, MIL_fc_mc)):
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
        elif isinstance(annotations, str):
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
    def from_df(
        cls, 
        df: "pd.core.frame.DataFrame", 
        *,
        annotations: Union[str, "pd.core.frame.DataFrame"] = None
    ) -> None:
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

    def to_df(
        self, 
        predictions: bool = True, 
        attentions: bool = True
    ) -> pd.core.frame.DataFrame:
        """Export activations to a pandas DataFrame.

        Returns:
            pd.core.frame.DataFrame: Dataframe with columns 'slide', 
                'activations', 'predictions', and 'attentions'.

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
        return df
