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
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'


sys.stderr = sys.__stdout__

def main():
    
    ##### ACC Project #####
    # P = sf.Project('/mnt/data/PROJECTS/SALIVARY_GLAND/ACC_COMBINED')
    # acc_data = sf.Dataset(
    #     config='/mnt/data/PROJECTS/SALIVARY_GLAND/ACC_COMBINED/datasets.json',
    #     sources=['UMICH_ACC', 'UCSF_ACC', 'DFCI_ACC', 'NHS_ACC_1',
    #             'NHS_ACC_2', 'MD_ANDERSON_1', 'MD_ANDERSON_2', 'UCH_ACC'],
    #     annotations='/mnt/data/PROJECTS/SALIVARY_GLAND/ACC_COMBINED/salivary_gland_database.csv',
    #     tile_px=299, tile_um=302)

    ##### Fanconi Anemia Project #####
    P = sf.Project('/mnt/data/PROJECTS/ROCK_FA_HNSC')
    fa_hnsc_data = sf.Dataset(
        config='/mnt/data/PROJECTS/ROCK_FA_HNSC/datasets.json',
        sources=['ROCK_FA_HNSC', 'TCGA_HNSC_new'],
        annotations='/mnt/data/PROJECTS/ROCK_FA_HNSC/annotations/hpvneg_rockefeller_clinical_annotations.csv',
        tile_px=299, tile_um=302)

    
    features = sf.DatasetFeatures(sf.model.build_feature_extractor('ctranspath', tile_px=299), fa_hnsc_data, batch_size=512, cache='/mnt/data/PROJECTS/ROCK_FA_HNSC/feature_extraction/fa_ctranspath_activations.pkl')
    features.save_cache('/mnt/data/PROJECTS/ROCK_FA_HNSC/feature_extraction/fa_ctranspath_activations.pkl')
    # Note: default for min_dist=0.1 (range 0.0-0.99), and default for n_neighbors=15
    umap = features.map_activations(n_neighbors=30, min_dist=0.1)
    labels, _ = fa_hnsc_data.labels('fa_comp_group', format='name')
    umap.label_by_slide(labels)
    
    # Note: adding subsample=n to the .save() will plot the number of tiles you want to include
    umap.save('/mnt/data/PROJECTS/ROCK_FA_HNSC/feature_extraction/visualizations/ctranspath_fa_hnsc',
            xlabel='UMAP1',
            ylabel='UMAP2',
            title='CTransPath Features on Fanconi Anemia HNSCC Samples',
            s=5)

if __name__=='__main__':
    multiprocessing.freeze_support()
    main()