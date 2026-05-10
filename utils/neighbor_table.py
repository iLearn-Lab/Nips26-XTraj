from __future__ import annotations

from typing import Tuple

import torch


def build_padded_neighbors(
    *,
    edge_index: torch.Tensor,
    num_nodes: int,
    pad_value: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a padded neighbor table for graph-constrained decoding.

    Args:
        edge_index: LongTensor [2, E] with directed edges (src -> dst).
        num_nodes: number of nodes N.
        pad_value: value used to pad neighbor ids.

    Returns:
        neighbors: LongTensor [N, Kmax]
        mask: BoolTensor [N, Kmax] (True means valid neighbor)

    Notes:
        - This assumes node ids are within [0, N-1].
        - Kmax is the max out-degree in edge_index.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be [2,E], got shape={tuple(edge_index.shape)}")

    if edge_index.numel() == 0:
        neighbors = torch.full((num_nodes, 0), pad_value, dtype=torch.long)
        mask = torch.zeros((num_nodes, 0), dtype=torch.bool)
        return neighbors, mask

    if edge_index.dtype != torch.long:
        edge_index = edge_index.long()

    src = edge_index[0]
    dst = edge_index[1]

    if src.min().item() < 0 or dst.min().item() < 0:
        raise ValueError("edge_index contains negative node id")
    if src.max().item() >= num_nodes or dst.max().item() >= num_nodes:
        raise ValueError("edge_index contains node id >= num_nodes")

    # Sort by src to pack neighbors.
    order = torch.argsort(src)
    src_s = src[order]
    dst_s = dst[order]

    deg = torch.bincount(src_s, minlength=num_nodes)  # [N]
    kmax = int(deg.max().item())

    neighbors = torch.full((num_nodes, kmax), pad_value, dtype=torch.long)
    mask = torch.zeros((num_nodes, kmax), dtype=torch.bool)

    # Prefix sums to locate each node's slice.
    offsets = torch.zeros((num_nodes + 1,), dtype=torch.long)
    offsets[1:] = torch.cumsum(deg, dim=0)

    for i in range(num_nodes):
        d = int(deg[i].item())
        if d == 0:
            continue
        s = int(offsets[i].item())
        e = int(offsets[i + 1].item())
        neighbors[i, :d] = dst_s[s:e]
        mask[i, :d] = True

    return neighbors, mask
