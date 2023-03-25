import pandas as pd
import numpy as np
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import ResNet as ResNet
from torch.utils.data import Dataset
from ..base import BaseFeatureExtractor

class RetCCLFeatures(BaseFeatureExtractor):
    """RetCCl pretrained feature extractor.

    Feature dimensions: 2048

    GitHub: https://github.com/Xiyue-Wang/RetCCL
    """

    tag = 'retccl'

    def __init__(self, device='cuda', center_crop=False):
        super().__init__(backend='torch')

        self.model = ResNet.resnet50(
            num_classes=128,
            mlp=False, 
            two_branch=False, 
            normlinear=True
        )

        self.model.fc = torch.nn.Identity().to(device)

        checkpoint_path = "/home/prajval/slideflow/slideflow/model/extractors/best_ckpt.pth"

        td = torch.load(checkpoint_path)
        self.model.load_state_dict(td, strict=True)
        self.model = self.model.to(device)
        self.model.eval()

        # ---------------------------------------------------------------------
        self.num_features = 2048
        self.num_classes = 0
        all_transforms = [transforms.CenterCrop(256)] if center_crop else []
        all_transforms += [
            transforms.Lambda(lambda x: x.permute(2, 0, 1)),
            transforms.Lambda(lambda x: x / 255.),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
            transforms.Lambda(lambda x: x.unsqueeze(0))
        ]
        self.transform = transforms.Compose(all_transforms)
        self.preprocess_kwargs = dict(standardize=False)
        # ---------------------------------------------------------------------

    def __call__(self, batch_images):
        assert batch_images.dtype == torch.uint8
        batch_images = self.transform(batch_images)
        batch_images = batch_images.to(device='cuda')
        return self.model(batch_images)