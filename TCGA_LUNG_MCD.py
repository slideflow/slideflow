import os
os.environ['SF_BACKEND'] = 'tensorflow'
import multiprocessing
import slideflow as sf
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
import neural_structured_learning as nsl

logging.getLogger('slideflow').setLevel(logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'


def main():
    P = sf.Project(root='/home/prajval/DATA/PROJECTS/TCGA_LUNG')

    ### ================== Training on 'primary_diagnosis' ================== ###
    # Running on 299px and basic hyper parameters to run deep ensembles. hp from Biscuit

    # hp = sf.model.ModelParams(
    #     model='xception',
    #     tile_px=299,
    #     tile_um=302,
    #     batch_size=128,
    #     epochs=[1],
    #     early_stop=True,
    #     early_stop_method='accuracy',
    #     dropout=0.1,
    #     uq=True,
    #     hidden_layer_width=1024,
    #     optimizer='Adam',
    #     learning_rate=0.0001,
    #     learning_rate_decay_steps=512,
    #     learning_rate_decay=0.98,
    #     loss='sparse_categorical_crossentropy',
    #     normalizer='reinhard_fast',
    #     include_top=False,
    #     hidden_layers=2,
    #     pooling='avg',
    #     augment='xyrjb'
    # )

    # P.train(
    #     outcomes='cohort',
    #     pretrain=None, 
    #     val_k = 1,
    #     params=hp
    # )

    ### ================== Runnign prediction ================== ###
    # Predicting xception - MC Dropout method
    P.predict('/home/prajval/DATA/PROJECTS/TCGA_LUNG/models/00000-cohort-HP0-kfold1/cohort-HP0-kfold1_epoch1', batch_size = 64)


if __name__=='__main__':
    multiprocessing.freeze_support()
    main()