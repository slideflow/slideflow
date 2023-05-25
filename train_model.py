import requests
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['SF_BACKEND'] = 'tensorflow'
import click
import multiprocessing
from multiprocessing import freeze_support
sys.path.insert(0, '/mnt/data/PROJECTS/SALIVARY_GLAND/slideflow/')
import slideflow as sf
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
sys.stderr = sys.__stdout__

def main():
    sf.about()
    params = sf.ModelParams(
        augment='xyrjn',
        training_balance='category',
        validation_balance='none',
        batch_size=256,
        drop_images=False,
        dropout=0.5,
        early_stop=True,
        early_stop_method='loss',
        early_stop_patience=15,
        epochs=[1,2,3,4,5],
        hidden_layer_width=256,
        hidden_layers=3,
        learning_rate=0.0001,
        learning_rate_decay=0.97,
        learning_rate_decay_steps=100000,
        loss='sparse_categorical_crossentropy',
        model='xception',
        optimizer='Adam',
        pooling='avg',
        tile_px=299,
        tile_um=302,
        toplayer_epochs=0,
        trainable_layers=0)
    
    requests.post("https://ntfy.sh/rhubarb_pearsonlab", data="Started training! 🫣".encode(encoding='utf-8'))

    P = sf.Project('/mnt/data/PROJECTS/SALIVARY_GLAND/ACC_COMBINED')
    
    P.train('cancer_type',
            params=params,
            val_strategy=None,
            multi_gpu=True,
            pretrain='/mnt/data/models/pancan_trained_non_normalized-kfold1',
            load_method='full')

    requests.post("https://ntfy.sh/rhubarb_pearsonlab", data="Done training! 😀".encode(encoding='utf-8'))

if __name__=='__main__':
    multiprocessing.freeze_support()
    main()

