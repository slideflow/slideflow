import os
os.environ['SF_BACKEND'] = 'tensorflow'
import multiprocessing
import slideflow as sf
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
import neural_structured_learning as nsl
#from rich.progress import track
from tqdm import tqdm

logging.getLogger('slideflow').setLevel(logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'


def main():
    P = sf.Project(root='/home/prajval/DATA/PROJECTS/TCGA_LUNG')

    hp = sf.model.ModelParams(
            model='xception',
            tile_px=299,
            tile_um=302,
            batch_size=128,
            epochs=[1,3],
            early_stop=True,
            early_stop_method='accuracy',
            dropout=0.1,
            #uq=True,
            hidden_layer_width=1024,
            optimizer='Adam',
            learning_rate=0.0001,
            learning_rate_decay_steps=512,
            learning_rate_decay=0.98,
            loss='sparse_categorical_crossentropy',
            normalizer='reinhard_fast',
            include_top=False,
            hidden_layers=2,
            pooling='avg',
            augment='xyrjb'
        )

    P.ensemble_train_predictions(
        "/home/prajval/DATA/PROJECTS/TCGA_LUNG/models/00023-cohort-ensemble",
        save_format = "feather")

    # P.predict_ensemble('/home/prajval/DATA/PROJECTS/TCGA_LUNG/models/00015-cohort-ensemble',epoch_number=1, format = "feather")

    # P.train_ensemble(
    #     outcomes='cohort',
    #     number_of_ensembles = 5,
    #     pretrain=None, # used for resnet50 and vgg19 
    #     steps_per_epoch_override =10,
    #     val_k = 3,
    #     # val_strategy = None,
    #     save_predictions = "feather",
    #     params=hp
    # )

    # P.train(
    #     outcomes='cohort',
    #     pretrain=None, # used for resnet50 and vgg19 
    #     steps_per_epoch_override =10,
    #     val_k = 1,
    #     params=hp
    # )

    

if __name__=='__main__':
    multiprocessing.freeze_support()
    main()
