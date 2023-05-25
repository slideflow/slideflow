import requests
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

import os
os.environ['SF_BACKEND'] = 'tensorflow'
import click
import multiprocessing
from multiprocessing import freeze_support
sys.path.append(os.path.dirname('/mnt/home/ecdyer/slideflow_ecdyer/slideflow/slideflow'))
import slideflow as sf
from slideflow import simclr
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
sys.stderr = sys.__stdout__


def main():
    ### Salivary Gland Project
    P = sf.Project('/mnt/data/PROJECTS/SALIVARY_GLAND/ACC_COMBINED')
    
    ### Fanconi Anemia Project
    # P = sf.Project('/mnt/data/PROJECTS/ROCK_FA_HNSC')
    # fa_hnsc_data = sf.Dataset(
    #     config='/mnt/data/PROJECTS/ROCK_FA_HNSC/datasets.json',
    #     sources=['ROCK_FA_HNSC', 'TCGA_HNSC_new', 'UCH_HNSC_HPV'],
    #     annotations='/mnt/data/PROJECTS/ROCK_FA_HNSC/annotations/all_hnsc_clinical_annotations.csv',
    #     tile_px=299, tile_um=302)
    
###### STEP 1: TILE EXTRACTION (if needed) #######  
    requests.post("https://ntfy.sh/rhubarb_pearsonlab", data="Beginning tile extraction.".encode(encoding='utf-8'))
    P.extract_tiles(tile_px=299,
                    tile_um=302,
                    source = ['MD_ANDERSON_2'],
                    qc=None,
                    grayspace_fraction=0.7,
                    roi_method='inside',
                    img_format='jpg')
    requests.post("https://ntfy.sh/rhubarb_pearsonlab", data="Tile extraction complete.".encode(encoding='utf-8'))

##### STEP 2: PRE-TRAINED FEATURE EXTRACTION ######
    # requests.post("https://ntfy.sh/rhubarb_pearsonlab", data="Beginning pre-trained feature extraction.".encode(encoding='utf-8'))
    # extractors = ['ctranspath', 'retccl']
    # for e in extractors:
    #     features = sf.DatasetFeatures(sf.model.build_feature_extractor(e, tile_px=299), fa_hnsc_data, batch_size=512)
    #     cache_name = '/mnt/data/PROJECTS/ROCK_FA_HNSC/feature_extraction/all_hnsc_' + e + '.pkl'
    #     features.save_cache(cache_name)
    #     # Note: default for min_dist=0.1 (range 0.0-0.99), and default for n_neighbors=15
    #     umap = features.map_activations(n_neighbors=30, min_dist=0.2)
    #     labels, _ = fa_hnsc_data.labels('fa_comp_group', format='name')
    #     umap.label_by_slide(labels)
        
    #     # Note: adding subsample=n to the .save() will plot the number of tiles you want to include
    #     # Note: parameter `s=5` is the size of the markers on the plot
    #     # Note: Create a new axis to be able to modify the legend 

    #     umap_path = '/mnt/data/PROJECTS/ROCK_FA_HNSC/feature_extraction/visualizations/all_hnsc_' + e
    #     umap.save(umap_path,
    #             xlabel='UMAP1',
    #             ylabel='UMAP2',
    #             title= e + ' Features on HNSCC Samples',
    #             legend='FA Group/HPV Status',
    #             subsample=6000,
    #             s=5)
    # requests.post("https://ntfy.sh/rhubarb_pearsonlab", data="Completed pre-trained feature extraction.".encode(encoding='utf-8'))
    
    #### STEP 2.5: Make necessary modifications to the UMAP plot visualizations ######
    # umap = sf.SlideMap.load('/mnt/data/PROJECTS/ROCK_FA_HNSC/feature_extraction/visualizations/all_hnsc_ctranspath/')
    # labels, _ = fa_hnsc_data.labels('fa_comp_group', format='name')
    # umap.label_by_slide(labels)
    # umap.save_plot('/mnt/data/PROJECTS/ROCK_FA_HNSC/feature_extraction/visualizations/visualization_revisions/all_hnsc_ctranspath.png',
    #             xlabel='UMAP1',
    #             ylabel='UMAP2',
    #             title= 'CTransPath Features on Fanconi Anemia HNSCC Samples',
    #             legend='FA Comp/HPV',
    #             subsample=12000,
    #             loc='lower left',
    #             ncol=2,
    #             legend_kwargs={'fontsize':3,'markerscale':0.4,'frameon':False,'title_fontsize':7},
    #             s=1,
    #             cmap="tab20")
  
##### STEP 3: SELF-SUPERVISED FEATURE EXTRACTION #####
    # requests.post("https://ntfy.sh/rhubarb_pearsonlab", data="Beginning to train SimCLR model.".encode(encoding='utf-8'))
    # simclr_args = simclr.get_args(
    # mode='train',
    # train_mode='pretrain',
    # train_batch_size=64,
    # temperature=0.1,
    # learning_rate=0.075,
    # learning_rate_scaling='sqrt',
    # weight_decay=1e-4,
    # train_epochs=100,
    # image_size=299,
    # checkpoint_epochs=20)
    
    # P.train_simclr(simclr_args, fa_hnsc_data)
    # # #P.generate_features('/media/ss4tbSSD/SID/PROJECTS/HNSC_PREMAL/simclr/nih_full/ckpt-435710.ckpt', dataset=full_dataset, exp_label='training_uiowa_NIH_DF_RH')
    # requests.post("https://ntfy.sh/rhubarb_pearsonlab", data="Completed training SimCLR model.".encode(encoding='utf-8'))
    
if __name__=='__main__':
    multiprocessing.freeze_support()
    main()
