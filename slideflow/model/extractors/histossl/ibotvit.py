# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the same directory as this source file.


"""ViT architecture with pre-loaded weights from iBOT training."""

import torch

from .encoders import vit_base
from .core import Extractor
from slideflow.util import log as logger


class iBOTViT(Extractor):  # pylint: disable=abstract-method
    """Vision transform model trained with iBOT (See [1]_).

    Parameters
    ----------
    architecture: str = 'vit_base_pancan'
        Model architecture. Must only be "vit_base_pancan" as of now.
    encoder: str = 'student'
        Whether to load the weights from the student or teacher encoder.
    mean: Tuple[float, float, float] = IMAGENET_MEAN
        Mean values used for mean/std normalization of image channels.
    std: Tuple[float, float, float] = IMAGENET_STD:
        Std values used for mean/std normalization of image channels.

    References
    ----------
    .. [1] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong.
    Image BERT pre-training with online tokenizer. In International Conference on Learning
    Representations, 2022
    """

    def __init__(
        self,
        weights_path: str,
        architecture="vit_base_pancan",
        encoder="student",
    ):
        super(iBOTViT, self).__init__()

        self.encoder = encoder

        # Load weights for iBOT[ViT-B]PanCancer.
        assert (
            architecture == "vit_base_pancan"
        ), "Weights are released for `vit_base_pancan` architecture only."
        self.feature_extractor = vit_base(
            patch_size=16, num_classes=0, use_mean_pooling=False
        )
        self._weights_path = weights_path

        # Load state_dict.
        state_dict = torch.load(self._weights_path, map_location="cpu")
        state_dict = state_dict[self.encoder]
        state_dict = {
            k.replace("module.", ""): v for k, v in state_dict.items()
        }
        state_dict = {
            k.replace("backbone.", ""): v for k, v in state_dict.items()
        }
        # Set weights with state_dict.
        msg = self.feature_extractor.load_state_dict(state_dict, strict=False)
        logger.debug(
            f"Pretrained weights found at {self._weights_path} and loaded with msg: {msg}"
        )

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute and return features.

        Parameters
        ----------
        images: torch.Tensor
            input of size (n_tiles, n_channels, dim_x, dim_y)

        Returns
        -------
        features : torch.Tensor
            tensor of size (n_tiles, features_dim)
        """
        # intermediate output
        # see https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/eval_linear.py#L208
        features = self.feature_extractor.get_intermediate_layers(images, 1)
        features = torch.cat([x[:, 0] for x in features], dim=-1)

        return features