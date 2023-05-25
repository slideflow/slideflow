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
sys.path.append(os.path.dirname('/mnt/home/ecdyer/slideflow_ecdyer/slideflow/slideflow'))
import slideflow as sf
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
#print(sf.__version__)

sys.stderr = sys.__stdout__
sf.about()

def main():

    dfci_acc = sf.Dataset(config = "/mnt/data/PROJECTS/SALIVARY_GLAND/ACC_COMBINED/datasets.json",
                          annotations="/mnt/data/PROJECTS/SALIVARY_GLAND/ACC_COMBINED/salivary_gland_database.csv",
                          sources=['DFCI_ACC'],
                          tile_px=299,
                          tile_um=302)

    #dfci_acc.convert_xml_rois()

    ##### ACC Project #####
    #P = sf.Project('/mnt/data/PROJECTS/SALIVARY_GLAND/ACC_COMBINED')
    
    ##### Fanconi Anemia Project #####
    P = sf.Project('/mnt/data/PROJECTS/ROCK_FA_HNSC')
    
    # image = sf.WSI('/mnt/labshare/SLIDES/MD_ANDERSON_new/box_3_scans/2M09.mrxs', tile_px=299, tile_um=302, rois=['/mnt/labshare/QUPATH/MD_ANDERSON_ACC/ROI/2M09.csv']).preview()
    # image = image.convert("RGB")
    # image.save('/home/ecdyer/2M09.jpg', format='JPEG')
    requests.post("https://ntfy.sh/rhubarb_pearsonlab", data="Beginning tile extraction.".encode(encoding='utf-8'))
    P.extract_tiles(tile_px=299,
                    tile_um=302,
                    qc=None,
                    grayspace_fraction=0.7,
                    roi_method='inside',
                    img_format='png',
                    source='ROCK_FA_HNSC')
    requests.post("https://ntfy.sh/rhubarb_pearsonlab", data="Tile extraction complete.".encode(encoding='utf-8'))


if __name__=='__main__':
    multiprocessing.freeze_support()
    main()
