"""Shared point decoding for end-to-end generation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from models.road_progress_decoder import RoadProgressDecoder
from utils.time_conditions import fixed_time_conditions


@torch.inference_mode()
def decode_points(
    model: RoadProgressDecoder, z: torch.Tensor, lengths: torch.Tensor,
    routes: Sequence[np.ndarray], start: Sequence[float], end: Sequence[float],
    point_counts: Sequence[int], depart_bins: Sequence[int],
) -> list[np.ndarray]:
    if not len(routes) == len(start) == len(end) == len(point_counts) == len(depart_bins):
        raise ValueError("Point batch route and condition counts differ")
    if not len(routes):
        return []
    if min(point_counts) < 2 or max(point_counts) > model.cfg.max_steps:
        raise ValueError("Point count outside decoder range")
    if min(map(len, routes)) < 1 or max(map(len, routes)) > model.cfg.max_roads:
        raise ValueError("Route length outside decoder range")

    model.eval()
    device = z.device
    roads = np.zeros((len(routes), max(map(len, routes))), dtype=np.int64)
    mask = np.zeros_like(roads, dtype=bool)
    for i, route in enumerate(routes):
        roads[i, :len(route)] = route
        mask[i, :len(route)] = True
    counts = torch.as_tensor(point_counts, device=device, dtype=torch.long)
    duration, remaining = fixed_time_conditions(counts, max(point_counts))
    out = model.generate_batch(
        roads_unique=torch.from_numpy(roads).to(device),
        road_mask=torch.from_numpy(mask).to(device),
        z=z,
        s_start=torch.as_tensor(start, device=device, dtype=torch.float32),
        s_end=torch.as_tensor(end, device=device, dtype=torch.float32),
        decode_steps=counts - 1,
        remaining_time_seq=remaining,
        depart_time=torch.as_tensor(depart_bins, device=device, dtype=torch.long),
        total_duration=duration,
        fixed_road_lengths=lengths,
    )
    result = [s.cpu().numpy() for s in out]
    for s, n, a, b in zip(result, point_counts, start, end):
        if (
            len(s) != n or not np.isfinite(s).all() or np.any(np.diff(s) < 0)
            or s[0] != np.float32(a) or s[-1] != np.float32(b)
        ):
            raise RuntimeError("Point decoder violated the 3-second or endpoint contract")
    return result
