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

class MILActivations():

    def __init__(self, model: Callable, config: _TrainerConfig, hlw: list, slides: list, annotations: pd.DataFrame):
        self.model= model
        self.config= config
        self.slides= slides
        self.n_in= hlw[0].shape[1]
        self.annotations= self.get_annotations(annotations)
        self.activations= self.get_activations(hlw, slides)

    def get_activations(self, hlw, slides):
        activations= {}
        for slide, h in zip(slides, hlw):
            activations[slide]=h.numpy()[0]
        return activations

    def get_annotations(self, annotations):
        if type(annotations) is str:
            return pd.read_csv(annotations)
        else:
            return annotations

    def to_df(self):
        patients= self.annotations[['slide', 'patient']]
        index= [s for s in self.slides]
        df_dict= {}
        df_dict.update({
            'activations': pd.Series([
                self.activations[s]
                for s in self.slides], index=index)
        })

        df = pd.DataFrame(df_dict)
        df['slide']= df.index
        df2=df.set_index('slide').join(patients.set_index('slide'), how='inner')
        df2= df2[['patient', 'activations']]

        return df2
