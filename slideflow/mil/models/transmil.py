import torch
import torch.nn as nn
import numpy as np

from slideflow.model.torch_utils import get_device

# -----------------------------------------------------------------------------

class TransMIL(nn.Module):
    def __init__(self, n_feats: int, n_out: int,):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(n_feats, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_out
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

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
        att = self.layer2.calculate_attention(h) #[B, N, 512]

        # Remove padding
        return att[:,:H,:]

    def relocate(self):
        self.to(get_device())

    def forward(self, h, return_attention=False):
        if return_attention:
            # FIXME: compute attention
            B, n, _ = h.shape
            h = self.get_last_layer_activations(h)  # [B, n, 1024] -> [B, 512]
            logits = self._fc2(h)  # [B, n_classes]
            temp_attention = torch.ones(B, n, 1) / n
            return logits, temp_attention # WARNING this is a placeholder to fix a bug
        else:
            h = self.get_last_layer_activations(h)  # [B, n, 1024] -> [B, 512]
            logits = self._fc2(h)  # [B, n_classes]
            return logits

    def get_last_layer_activations(self, h):
        h = self._fc1(h)  # [B, n, 1024] -> [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        return h

# -----------------------------------------------------------------------------

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()

        try:
            from nystrom_attention import NystromAttention
        except ImportError:
            raise ImportError(
                "The package 'nystrom_attention' has not been installed, but "
                "is required for TransMIL. Install with 'pip install "
                "nystrom_attention'"
            )

        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2, # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,        # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def calculate_attention(self, x):
        return self.attn(self.norm(x))

    def forward(self, x):
        x = x + self.calculate_attention(x)
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x
