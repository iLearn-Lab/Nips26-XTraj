"""GAT training using masked interior-road reconstruction."""

import torch
import torch.nn.functional as F

from utils.training import finite_loss


def mask_positions(attention_mask, ratio):
    order = attention_mask.long().cumsum(dim=1)
    length = order[:, -1:]
    interior = attention_mask & (order > 1) & (order < length)
    count = (length - 2).clamp_min(0)
    want = (count * ratio).round().long().clamp_min(1).minimum(count)
    scores = torch.rand(attention_mask.shape, device=attention_mask.device)
    ranks = scores.masked_fill(~interior, float("inf")).argsort(dim=1).argsort(dim=1)
    return interior & (ranks < want)


def train_epoch(encoder, reconstructor, loader, optimizer, *, x_features, x_heat, edge_index,
                device=None, mask_ratio=0.3):
    """Update both models over one caller-supplied loader; write no artifacts.

    Loader batches are ``(roads[B,T], valid_mask[B,T])``. The graph and node
    features are already prepared and aligned by road ID. The caller owns the
    training schedule, optimizer and any distributed wrappers.
    """
    if not 0 < mask_ratio < 1:
        raise ValueError("mask_ratio must be in (0, 1)")
    device = torch.device(device) if device is not None else next(encoder.parameters()).device
    x_features, x_heat = x_features.to(device), x_heat.to(device)
    edge_index = edge_index.to(device)
    encoder.train()
    reconstructor.train()
    for roads, valid_mask in loader:
        roads, valid_mask = roads.to(device), valid_mask.to(device).bool()
        masked = mask_positions(valid_mask, mask_ratio)
        if not bool(masked.any()):
            continue
        z = encoder(x_geo=None, x_poi=x_features, x_heat=x_heat, edge_index=edge_index, edge_attr=None)
        hidden = reconstructor(token_emb=z[roads], attn_mask=valid_mask, masked_positions=masked)
        loss = F.cross_entropy(hidden[masked] @ z.T, roads[masked])
        finite_loss(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
