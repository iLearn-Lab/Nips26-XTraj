"""Compute city road embeddings from an existing encoder and prepared graph."""

import torch


@torch.no_grad()
def encode_roads(encoder, *, x_features, x_heat, edge_index):
    """Return embeddings in memory, suitable for frozen downstream training."""
    encoder.eval()
    device = next(encoder.parameters()).device
    return encoder(
        x_geo=None, x_poi=x_features.to(device), x_heat=x_heat.to(device),
        edge_index=edge_index.to(device), edge_attr=None,
    ).detach()
