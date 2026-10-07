"""Query/value LoRA adapters for timm vision transformers."""

import math
from os import PathLike

import torch
from torch import nn


class LoRAQKV(nn.Module):
    """Add low-rank updates to a packed projection's query and value slices."""

    def __init__(self, base, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        if not isinstance(base, nn.Linear) or base.out_features != 3 * base.in_features:
            raise ValueError('LoRA requires a linear packed Q/K/V projection')
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError('invalid rank, alpha or dropout')
        self.base = base
        d = base.in_features
        self.scale = alpha / rank
        opts = {'device': base.weight.device, 'dtype': base.weight.dtype}
        self.Aq = nn.Parameter(torch.empty(rank, d, **opts))
        self.Av = nn.Parameter(torch.empty(rank, d, **opts))
        self.Bq = nn.Parameter(torch.zeros(d, rank, **opts))
        self.Bv = nn.Parameter(torch.zeros(d, rank, **opts))
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.Aq, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Av, a=math.sqrt(5))
        self.train(base.training)

    def forward(self, x):
        out = self.base(x)
        x = self.drop(x)
        dq = (x @ self.Aq.t()) @ self.Bq.t() * self.scale
        dv = (x @ self.Av.t()) @ self.Bv.t() * self.scale
        return out + torch.cat([dq, torch.zeros_like(dq), dv], -1)


def _projections(model, first_block, rank, alpha, dropout):
    if not hasattr(model, 'blocks') or not 0 <= first_block < len(model.blocks):
        raise ValueError('first_block must select a transformer block')
    if rank < 1 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError('invalid rank, alpha or dropout')
    if any(isinstance(getattr(getattr(b, 'attn', None), 'qkv', None), LoRAQKV)
           for b in model.blocks):
        raise ValueError('encoder already has adapters')
    projections = []
    for block in model.blocks[first_block:]:
        qkv = getattr(getattr(block, 'attn', None), 'qkv', None)
        if not isinstance(qkv, nn.Linear) or qkv.out_features != 3 * qkv.in_features:
            raise ValueError('LoRA requires linear packed Q/K/V projections')
        projections.append(qkv)
    return projections


def init_lora(model, *, first_block=32, rank=8, alpha=16, dropout=0.05):
    """Freeze a ViT and add trainable adapters to the selected suffix."""
    projections = _projections(model, first_block, rank, alpha, dropout)
    for param in model.parameters():
        param.requires_grad_(False)
    for block, qkv in zip(model.blocks[first_block:], projections):
        block.attn.qkv = LoRAQKV(qkv, rank, alpha, dropout)
    model.first_adapted = first_block
    return model


def adapter_state_dict(model, *, first_block=None):
    """Return adapter tensors with block indices relative to the adapted suffix."""
    first_block = getattr(model, 'first_adapted', 32) if first_block is None else first_block
    if not 0 <= first_block < len(model.blocks):
        raise ValueError('invalid first_block')
    state = {}
    for i, block in enumerate(model.blocks[first_block:]):
        qkv = block.attn.qkv
        if not isinstance(qkv, LoRAQKV):
            raise ValueError(f'block {first_block + i} has no adapter')
        for name in ('Aq', 'Av', 'Bq', 'Bv'):
            state[f'{i}.attn.qkv.{name}'] = getattr(qkv, name).detach().cpu().clone()
    return state


def apply_lora(model, adapters, *, first_block=32, rank=8, alpha=16, dropout=0.05):
    """Validate and load adapter tensors into a frozen base ViT, in place."""
    if isinstance(adapters, (str, PathLike)):
        adapters = torch.load(adapters, map_location='cpu', weights_only=True)
    projections = _projections(model, first_block, rank, alpha, dropout)
    expected = {
        f'{i}.attn.qkv.{name}': shape
        for i, qkv in enumerate(projections)
        for name, shape in [('Aq', (rank, qkv.in_features)), ('Av', (rank, qkv.in_features)),
                            ('Bq', (qkv.in_features, rank)), ('Bv', (qkv.in_features, rank))]
    }
    if set(adapters) != set(expected):
        raise ValueError('adapter keys do not match the selected transformer blocks')
    for key, shape in expected.items():
        value = adapters[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f'adapter {key} must have shape {shape}')
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f'adapter {key} must contain finite floating-point values')
    init_lora(model, first_block=first_block, rank=rank, alpha=alpha, dropout=dropout)
    with torch.no_grad():
        for key, value in adapters.items():
            i, name = key.split('.', 1)
            param = model.blocks[first_block + int(i)].get_parameter(name)
            param.copy_(value)
    return model
