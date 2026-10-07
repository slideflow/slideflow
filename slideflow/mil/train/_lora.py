"""Train a ViT LoRA adapter and nnMIL head from patient tile bags."""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from slideflow.model.extractors._lora import adapter_state_dict, init_lora


def encode_tiles(model, images, *, checkpoint_blocks=False):
    """Encode normalized tiles through the frozen ViT prefix and adapted suffix."""
    first = getattr(model, 'first_adapted', len(model.blocks))
    with torch.no_grad():
        x = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(images))))
        for block in model.blocks[:first]:
            x = block(x)
    x = x.detach()
    for block in model.blocks[first:]:
        x = checkpoint(block, x, use_reentrant=False) if checkpoint_blocks else block(x)
    return model.forward_head(model.norm(x), pre_logits=True)


def prepare_tiles(extractor, tiles, device):
    """Convert (patient, tile, height, width, channel) uint8 bags to model input."""
    x = torch.as_tensor(tiles)
    if x.ndim != 5 or x.shape[-1] != 3:
        raise ValueError('tiles must have shape (patients, tiles, height, width, 3)')
    n, k = x.shape[:2]
    x = x.reshape(n * k, *x.shape[2:]).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    return extractor.transform(x), n, k


def predict_lora(extractor, head, batches, *, objective='binary', device=None):
    """Return patient labels and scores from tile-bag batches."""
    model = extractor.model
    device = torch.device(device or next(model.parameters()).device)
    model.eval(); head.eval()
    labels, logits, rs_scores = [], [], []
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        for batch in batches:
            tiles, target = batch[:2]
            x, n, k = prepare_tiles(extractor, tiles, device)
            features = encode_tiles(model, x).reshape(n, k, -1).float()
            output = head(features).float()
            score = output[:, 1] - output[:, 0] if objective == 'joint' else output.reshape(-1)
            logits.extend(score.float().cpu().tolist())
            labels.extend(torch.as_tensor(target).reshape(-1).tolist())
            if objective == 'joint':
                rs_scores.extend((output[:, 2] * 10 + 18).cpu().tolist())
    return {'labels': labels, 'logits': logits, 'rs_pred': rs_scores}


def train_lora(extractor, head, train_batches, *, val_batches=None, epochs=8,
               first_block=32, rank=8, alpha=16, lr_adapter=5e-5,
               lr_head=2e-4, weight_decay=1e-5, adapt=True, device=None,
               objective='binary', positive_weight=None, seed=None, outdir=None):
    """Fit LoRA and an nnMIL binary or joint RS head; save weights and history."""
    model = extractor.model
    device = torch.device(device or next(model.parameters()).device)
    expected = {'binary': 1, 'joint': 3}
    if epochs < 1 or objective not in expected or head.head.out_features != expected[objective]:
        raise ValueError('epochs, objective, and head outputs do not agree')
    if objective == 'joint' and (positive_weight is None or positive_weight <= 0):
        raise ValueError('joint training needs the training-fold positive class weight')
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
    if adapt:
        init_lora(model, first_block=first_block, rank=rank, alpha=alpha)
    else:
        for param in model.parameters():
            param.requires_grad_(False)
        model.first_adapted = len(model.blocks)
    model.to(device); head.to(device)
    groups = [{'params': head.parameters(), 'lr': lr_head}]
    if adapt:
        groups.append({'params': [p for p in model.parameters() if p.requires_grad], 'lr': lr_adapter})
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        model.train(); head.train()
        loss_sum, patients = 0.0, 0
        for batch in train_batches:
            tiles, target = batch[:2]
            x, n, k = prepare_tiles(extractor, tiles, device)
            y = torch.as_tensor(target, dtype=torch.float32, device=device).reshape(-1)
            if y.numel() != n or not torch.all((y == 0) | (y == 1)):
                raise ValueError('each patient needs a binary label')
            if objective == 'joint':
                if len(batch) != 3:
                    raise ValueError('joint training needs measured RS in each batch')
                rs = torch.as_tensor(batch[2], dtype=torch.float32, device=device).reshape(-1)
                if rs.numel() != n or not torch.isfinite(rs).all():
                    raise ValueError('measured RS is missing or invalid')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                features = encode_tiles(model, x, checkpoint_blocks=adapt).reshape(n, k, -1).float()
                output = head(features).float()
                if objective == 'joint':
                    weight = torch.tensor([1., positive_weight], device=device, dtype=torch.float32)
                    loss = F.cross_entropy(output[:, :2], y.long(), weight=weight)
                    loss = loss + F.mse_loss(output[:, 2], (rs - 18.) / 10.)
                else:
                    loss = F.binary_cross_entropy_with_logits(output.reshape(-1), y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * n
            patients += n
        if not patients:
            raise ValueError('training batches are empty')
        schedule.step()
        row = {'epoch': epoch, 'train_loss': loss_sum / patients,
               'train_patients': patients, 'epoch_seconds': time.monotonic() - started}
        if val_batches is not None:
            predictions = predict_lora(extractor, head, val_batches, objective=objective, device=device)
            if predictions['labels'] and objective == 'binary':
                row['val_loss'] = float(F.binary_cross_entropy_with_logits(
                    torch.tensor(predictions['logits']), torch.tensor(predictions['labels'], dtype=torch.float32)))
                row['val_patients'] = len(predictions['labels'])
        history.append(row)
    if outdir is not None:
        root = Path(outdir)
        root.mkdir(parents=True, exist_ok=False)
        if adapt:
            torch.save(adapter_state_dict(model, first_block=first_block), root/'adapters.pt')
        torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, root/'head.pt')
        (root/'history.json').write_text(json.dumps({
            'encoder': extractor.tag, 'adapt': adapt, 'objective': objective,
            'positive_weight': positive_weight, 'seed': seed,
            'first_block': first_block if adapt else None,
            'rank': rank if adapt else None, 'alpha': alpha if adapt else None,
            'epochs': epochs, 'history': history
        }, indent=2) + '\n')
    return history
