import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F
import numpy as np
from .transmil import TransMIL
from .att_mil import Attention_MIL, Attention

class TransMILActivation(TransMIL):

    def __init__(self, TransMIL):
        super().__init__(TransMIL.n_feats, TransMIL.n_classes)

    def calculate_attention(self, h):
        h = self._fc1(h) #[B, n, 1024] -> [B, n, 512]

        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]], dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]

        #---->Translayer x2
        return self.layer2.calculate_attention(h) #[B, N, 512]

    def forward(self, h):
        print("forward activation")
        h = self._fc1(h) #[B, n, 1024] -> [B, n, 512]

        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]], dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]

        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]
    
        return h

class AttentionMILActivation(Attention_MIL):

    def __init__(self, Attention_MIL):
        super().__init__(Attention_MIL.n_feats, Attentin_MIL.n_classes)
