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
#sys.path.insert(0, '/mnt/data/PROJECTS/SALIVARY_GLAND/slideflow/')
import slideflow as sf
from slideflow import simclr
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def main():
    
    ##### ACC Project #####
    #P = sf.Project('/mnt/data/PROJECTS/SALIVARY_GLAND/ACC_COMBINED')
    # acc_data = sf.Dataset(
    #     config='/mnt/data/PROJECTS/SALIVARY_GLAND/ACC_COMBINED/datasets.json',
    #     sources=['UMICH_ACC', 'UCSF_ACC', 'DFCI_ACC', 'NHS_ACC_1',
    #                'NHS_ACC_2', 'MD_ANDERSON_1', 'MD_ANDERSON_2', 'UCH_ACC'],
    #     annotations='/mnt/data/PROJECTS/SALIVARY_GLAND/ACC_COMBINED/salivary_gland_database.csv',
    #     tile_px=299, tile_um=302)
    
    
    ##### Fanconi Anemia Project #####
    P = sf.Project('/mnt/data/PROJECTS/ROCK_FA_HNSC')
    fa_hnsc_data = sf.Dataset(
        config='/mnt/data/PROJECTS/ROCK_FA_HNSC/datasets.json',
        sources=['ROCK_FA_HNSC'],
        annotations='/mnt/data/PROJECTS/ROCK_FA_HNSC/rockefeller_clinical_annotations.csv',
        tile_px=299, tile_um=302)

    
    requests.post("https://ntfy.sh/rhubarb_pearsonlab", data="Beginning to train SimCLR model.".encode(encoding='utf-8'))
    simclr_args = simclr.get_args(
    mode='train',
    train_mode='pretrain',
    train_batch_size=256,
    temperature=0.1,
    learning_rate=0.075,
    learning_rate_scaling='sqrt',
    weight_decay=1e-4,
    train_epochs=100,
    image_size=299,
    checkpoint_epochs=20)
    
    P.train_simclr(simclr_args, fa_hnsc_data)
    #P.generate_features('/media/ss4tbSSD/SID/PROJECTS/HNSC_PREMAL/simclr/nih_full/ckpt-435710.ckpt', dataset=full_dataset, exp_label='training_uiowa_NIH_DF_RH')
    requests.post("https://ntfy.sh/rhubarb_pearsonlab", data="Completed training SimCLR model.".encode(encoding='utf-8'))

if __name__=='__main__':
    multiprocessing.freeze_support()
    main()