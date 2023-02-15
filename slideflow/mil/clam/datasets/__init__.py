"""Updated datasets for CLAM."""

import os
import torch
from os.path import join
from slideflow.util import path_to_name

from .dataset_generic import Generic_WSI_Classification_Dataset


# -----------------------------------------------------------------------------

class CLAM_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self, pt_files, **kwargs):
        super().__init__(**kwargs)
        if isinstance(pt_files, str):
            self.pt_files = {
                path_to_name(filename): join(pt_files, filename)
                for filename in os.listdir(pt_files)
            }
        else:
            self.pt_files = {path_to_name(path): path for path in pt_files}

    def detect_num_features(self):
        features = torch.load(list(self.pt_files.values())[0])
        return features.size()[1]

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide'][idx]
        label = self.slide_data['label'][idx]
        features = torch.load(self.pt_files[slide_id])
        if self.lasthalf:
            features = torch.split(features, 1024, dim = 1)[1]
        return features, label
