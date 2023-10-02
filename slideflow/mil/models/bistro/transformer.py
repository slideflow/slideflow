"""MIL models from https://github.com/peng-lab/HistoBistro/"""

import torch
import torch.nn as nn
from einops import repeat

from .model_utils import Attention, FeedForward, PreNorm

from slideflow.model.torch_utils import get_device

# -----------------------------------------------------------------------------

class TransformerBlocks(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                    ]
                )
            )

    def forward(self, x, register_hook=False):
        for attn, ff in self.layers:
            x = attn(x, register_hook=register_hook) + x
            x = ff(x) + x
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        *,
        dim=512,
        depth=2,
        heads=8,
        mlp_dim=512,
        pool='cls',
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
        pos_enc=None,
    ):
        super().__init__()
        assert pool in {
            'cls', 'mean'
        }, 'pool type must be either cls (class token) or mean (mean pooling)'

        self.projection = nn.Sequential(nn.Linear(input_dim, heads*dim_head, bias=True), nn.ReLU())
        self.mlp_head = nn.Sequential(nn.LayerNorm(mlp_dim), nn.Linear(mlp_dim, num_classes))
        self.transformer = TransformerBlocks(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.pos_enc = pos_enc

    def forward(self, x, coords=None, register_hook=False):
        b, _, _ = x.shape

        x = self.projection(x)

        if self.pos_enc:
            x = x + self.pos_enc(coords)

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.dropout(x)
        x = self.transformer(x, register_hook=register_hook)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(self.norm(x))

    def relocate(self):
        self.to(get_device())
