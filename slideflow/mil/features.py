import csv
import os
import multiprocessing as mp
from collections import defaultdict
from typing import Union, List, Optional, Callable, Tuple, Any

import numpy as np
import pandas as pd
from ._params import (
    _TrainerConfig, ModelConfigCLAM, TrainerConfigCLAM
)
import slideflow as sf
import torch
import tensorflow as tf

class MILActivations():

    def __init__(self, 
                 hlw: list, 
                 y_pred: list,
                 y_att,
                 slides: list, 
                 annotations: Union[str, "pd.core.frame.DataFrame"]):
        self.slides = slides
        self.num_features, self.activations = self._get_activations(hlw)
        self.annotations = self._get_annotations(annotations)
        self.attentions= self._get_attentions(y_att)
        self.predictions= self._get_predictions(y_pred)
        """Saves activations from hidden layer weights, storing to
        internal parameter ``self.activations`` dictionary mapping slides to arrays of activations.
        Converts annotations to DataFrame if necessary.

        Args:
            hlw (list): List of last layer activation weights from the slides.
            y_pred (list): List of prediction weights from the slides.
            y_att (list): List of attention weights from the slides.
            slides (list): List of slide names.
            annotations (str, DataFrame): annotations for each slide/patient.
        """
    def _get_activations(self, hlw):
        """Formats list of activation weights into dictionary of lists, matched by slide.
        Returns the dictionary."""
        if hlw is None or self.slides is None:
            return None, {}
        activations = {}
        for slide, h in zip(self.slides, hlw):
            activations[slide] = h.numpy()[0]
        
        num_features= len(hlw[0])

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
    
    def _get_attentions(self, attentions):
        """Formats list of attention weights into dictionary of lists, matched by slide.
        Returns the dictionary."""
        if attentions is None or self.slides is None:
            return None
        attentions = {}
        for slide, att in zip(self.slides, attentions):
            attentions[slide]= att
        return attentions

    def _get_predictions(self, predictions):
        """Formats list of prediction weights into dictionary of lists, matched by slide.
        Returns the dictionary."""
        if predictions is None or self.slides is None:
            return None
        predictions = {}
        for slide, pred in zip(self.slides, predictions):
            predictions[slide]= pred
        return predictions

    @classmethod
    def from_df(cls, df: "pd.core.frame.DataFrame", *, annotations= Union[str, "pd.core.frame.DataFrame"]):
        """Load MILActivations of activations, as exported by :meth:`MILActivations.to_df()`"""
        obj= cls(None, None, None)
        if 'slide' in df.columns:
            obj.slides = df['slide'].values
        elif df.index.name=='slide':
            obj.slides= df.index.values
            df['slide']= df.index
        else:
            #some error TODO: find correct error to raise here. bad input format or something
            raise RuntimeError
        if 'activations' in df.columns:
            obj.activations = {
                s: df.loc[df.slide==s].activations.values.tolist()[0]
                for s in obj.slides
            }
            #obj.num_classes = next(df.iterrows())[1].predictions.shape[0]
        if 'predictions' in df.columns:
            obj.predictions = {
                s: np.stack(df.loc[df.slide==s].predictions.values)
                for s in obj.slides
            }
            #obj.num_classes = next(df.iterrows())[1].predictions.shape[0]
        if 'attentions' in df.columns:
            obj.attentions= {

            }
        if annotations:
            obj.annotations= obj._get_annotations(annotations)
        
        return obj

    def to_df(self) -> pd.core.frame.DataFrame:
        """Export activations to
        a pandas DataFrame.

        Returns:
            pd.core.frame.DataFrame: Dataframe with columns 'slide', 
            'patient', and 'activations'.
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

        df = pd.DataFrame(df_dict)
        df['slide'] = df.index

        if not self.annotations:
            #TODO Raise a warning about the patients being unable to be extracted
            return df
        patients = self.annotations[['slide', 'patient']]
        df2 = df.set_index('slide').join(
            patients.set_index('slide'), how='inner')
        df2 = df2[['patient', 'activations']]

        return df2
