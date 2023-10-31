import re
import numpy as np
import pandas as pd
import torch
import slideflow as sf
import os
from typing import Union, List, Optional, Callable, Tuple, Any, Dict
from os.path import join, dirname, exists
from slideflow import log, errors
from slideflow.util import log, path_to_name

from ._params import (
    _TrainerConfig
)
from .models.mil_fc import MIL_fc, MIL_fc_mc
from .models.att_mil import Attention_MIL
from .models.transmil import TransMIL
from .utils import load_model_weights

# -----------------------------------------------------------------------------

class MILFeatures:
    """Loads annotations, saved layer activations / features, predictions,
    and prepares output pandas DataFrame."""

    uq = None

    def __init__(
        self,
        model: Optional[Union[str, "torch.nn.Module"]],
        bags: Union[np.ndarray, List[str], str],
        *,
        slides: Optional[list] = None,
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
        if attention_pooling is not None and attention_pooling not in ('avg', 'max'):
            raise ValueError(
                "Unrecognized attention pooling strategy '{}'".format(
                    attention_pooling))
        self.attention_pooling = attention_pooling

        # Find bags.
        bags = self._find_bags(bags, dataset, slides)

        # Determine slides.
        self.slides = np.array([path_to_name(b) for b in bags])

        # --- Prepare model ---------------------------------------------------
        if model is not None:
            # Load or build the model.
            self.model, self.use_lens = self._load_model(model, config)
            self.set_device(device)
            self.model.to(self.device)  # type: ignore

            # Ensure model is compatible.
            if not hasattr(self.model, 'get_last_layer_activations'):
                raise errors.ModelError(
                    f"Model {model.__class__.__name__} is not supported; could not "
                    "find method 'get_last_layer_activations'")

            # Generate activations.
            (self.num_features,
             self.predictions,
             self.attentions,
             self.activations) = self._get_mil_activations(bags)

            # Find tile locations from bags.
            self.locations = self._get_bag_locations(bags)  # type: ignore
        else:
            self.model = None  # type: ignore
            self.use_lens = None  # type: ignore
            self.device = None
            self.num_features = None
            self.predictions = None
            self.attentions = None
            self.activations = None
            self.locations = None

    def _find_bags(
        self,
        bags: Union[np.ndarray, List[str], str],
        dataset: Optional["sf.Dataset"],
        slides: Optional[List[str]]
    ) -> np.ndarray:
        """Find bags from path, dataset, or slides.

        Args:
            bags (str): Path to bags.
            dataset (:class:`slideflow.Dataset`): Dataset from which to
                generate activations.
            slides (list): List of slides.

        Returns:
            np.ndarray: Array of bag paths.

        """
        if isinstance(bags, str) and dataset is not None:
            return dataset.pt_files(bags)
        elif isinstance(bags, str) and slides:
            return np.array([
                join(bags, f) for f in os.listdir(bags)
                if f.endswith('.pt') and path_to_name(f) in slides
            ])
        elif isinstance(bags, str):
            return np.array([
                join(bags, f) for f in os.listdir(bags)
                if f.endswith('.pt')
            ])
        elif slides:
            return np.array([b for b in bags if path_to_name(b) in slides])
        elif dataset:
            return np.array([b for b in bags
                             if path_to_name(b) in dataset.slides()])
        else:
            return np.array(bags)

    def _get_bag_locations(self, bags: List[str]) -> Optional[Dict[str, np.ndarray]]:
        """Get tile locations from bags.

        Args:
            bags (list): List of feature bags.

        Returns:
            dict: Dictionary mapping slide names to tile locations.

        """
        if bags is None:
            return None
        locations = {}
        for bag in bags:
            slide = path_to_name(bag)
            bag_index = join(dirname(bag), f'{slide}.index.npz')
            if not exists(bag_index):
                log.warning(
                    f"Could not find index file for bag {bag}. Unable to determine "
                    "tile location information.")
                return None
            locations[slide] = np.load(bag_index)['arr_0']
        return locations

    def _load_model(
        self,
        model: Union[str, "torch.nn.Module"],
        config: Optional[_TrainerConfig]
    ) -> Tuple[Callable, bool]:
        """Loads in model from Callable or path to model weights and config.

        Returns model and use_lens spec for generating model args in
        _get_mil_activations.

        Args:
            model (str): Path to model from which to calculate activations.
            config (:class:`TrainerConfig`): Trainer for MIL model,

        Returns:
            model (Callable): Model from which to calculate activations.
            use_lens (bool): Spec used for generating model args in
                _get_mil_activations.

        """
        if isinstance(model, str):
            model, config = load_model_weights(model, config)
            if isinstance(model, Attention_MIL) or isinstance(model, TransMIL):
                use_lens = config.model_config.use_lens
            else:
                use_lens = False
        else:
            use_lens = isinstance(model, Attention_MIL)
        return model, use_lens  # type: ignore

    def _get_mil_activations(self, bags: Union[np.ndarray, List[str]]):
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

        # If initialized using classmethod ``MILFeatures.from_df()``:
        if not self.model:
            return None, None, None, None

        y_pred = []
        y_att = []
        hs = []
        log.info("Calculating layer activations...")

        for bag in bags:
            # Load the bag.
            loaded = torch.load(bag).to(self.device)
            loaded = torch.unsqueeze(loaded, dim=0)

            with torch.no_grad():

                # Apply lens to model input.
                if self.use_lens:
                    lens = torch.from_numpy(
                        np.array([loaded.shape[1]])).to(self.device)
                    model_args = (loaded, lens)
                else:
                    model_args = (loaded,)

                # Attention MIL and TransMIL.
                if isinstance(self.model, (Attention_MIL, TransMIL)):
                    model_out = self.model(*model_args)
                    h = self.model.get_last_layer_activations(*model_args)  # type: ignore
                    att = torch.squeeze(self.model.calculate_attention(*model_args))  # type: ignore
                    if len(att.shape) == 2 and not self.attention_pooling:
                        raise ValueError("Attention pooling required for 2D attention")
                    elif len(att.shape) == 2:
                        att = self._attention_pool(att)
                    y_att.append(att.cpu().numpy())

                # FC MIL (CLAM implementation)
                elif isinstance(self.model, (MIL_fc, MIL_fc_mc)):
                    model_out = self.model(*model_args)
                    h = self.model.get_last_layer_activations(*model_args)  # type: ignore
                    y_att = None

                # CLAM models.
                else:
                    model_out = self.model(*model_args)[0]
                    h, A = self.model.get_last_layer_activations(*model_args)  # type: ignore
                    if A.shape[0] == 1:
                        y_att.append(A.cpu().numpy()[0])
                    else:
                        y_att.append(A.cpu().numpy())

                hs.append(h.cpu())
                yp = torch.nn.functional.softmax(model_out, dim=1).cpu().numpy()
                y_pred.append(yp)
        yp = np.concatenate(y_pred, axis=0)

        num_features, acts = self._get_activations(hs)
        atts = self._get_attentions(y_att)
        preds = self._get_predictions(yp)

        return num_features, preds, atts, acts

    def _attention_pool(self, att):
        """Pools attention weights according to attention_pooling strategy"""
        assert len(att.shape) == 2
        if self.attention_pooling == 'avg':
            return torch.mean(att, dim=-1)
        elif self.attention_pooling == 'max':
            return torch.amax(att, dim=-1)
        else:
            raise ValueError(f"Unknown attention pooling strategy {self.attention_pooling}")

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
        if type(column) != 'str' or column.dtype == 'float32':
            return column
        numbers = re.findall(r'-?\d+\.\d+', column)
        # Convert numbers to floats
        return np.array([float(num) for num in numbers])

    def set_device(self, device: Any) -> None:
        """Auto-detect device."""
        if device is not None:
            self.device = device
        elif self.model is None:
            self.device = None
        else:
            self.device = next(self.model.parameters()).device  # type: ignore
        log.debug(f"Using device {self.device}")

    @classmethod
    def from_df(
        cls,
        df: "pd.core.frame.DataFrame",
        *,
        annotations: Union[str, "pd.core.frame.DataFrame"] = None
    ) -> None:
        """Load MILFeatures of activations, as exported by
        :meth:`MILFeatures.to_df()`"""

        obj = cls(None, None)
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

    def map_activations(self, **kwargs) -> "sf.SlideMap":
        """Map activations with UMAP.

        Keyword args:
            ...

        Returns:
            sf.SlideMap

        """
        return sf.SlideMap.from_features(self, **kwargs)