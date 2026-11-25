"""Generate road routes and three-second GPS points without evaluation."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from generation.points import decode_points
from generation.requests import GenerationRequest
from generation.routes import RoadGenerator
from utils.point_road_network import PointRoadNetwork
from utils.time_conditions import SAMPLE_SECONDS


@dataclass
class GeneratedTrajectory:
    roads: list[int]
    status: str
    progress: np.ndarray | None = None
    gps: np.ndarray | None = None
    time_seconds: np.ndarray | None = None


@torch.inference_mode()
def generate_trajectories(
    route_generator: RoadGenerator, point_model, network: PointRoadNetwork,
    requests: Sequence[GenerationRequest], *, hard: bool = False,
) -> list[GeneratedTrajectory]:
    """Return one generation result per request, retaining route-failure status.

    GPS columns are longitude, latitude and road ID. No reference trajectories,
    accuracy metrics, spatial comparisons or file writes are used.
    """
    if any(r.point_count > point_model.cfg.max_steps for r in requests):
        raise ValueError("Requested point count exceeds the point decoder limit")
    device = route_generator.z.device
    if next(point_model.parameters()).device != device:
        raise ValueError("Road and point models must be on the same device")
    if len(network.lengths) != len(route_generator.z) or point_model.cfg.emb_dim != route_generator.z.shape[1]:
        raise ValueError("Point model, road embeddings and geometry have different dimensions")
    routes = route_generator.generate(requests, max_roads=point_model.cfg.max_roads, hard=hard)
    results = [GeneratedTrajectory(route.roads, route.status) for route in routes]
    ready = []
    for i, (request, result) in enumerate(zip(requests, results)):
        if result.status != "reached_destination":
            continue
        roads = np.asarray(result.roads, dtype=np.int64)
        network.validate_route(roads)
        end = len(roads) - 1 + request.end_alpha
        if end < request.start_alpha:
            result.status = "reversed_same_road_endpoints"
            continue
        ready.append((i, roads, end))
    decoded = decode_points(
        point_model, route_generator.z, torch.as_tensor(network.lengths, device=device),
        [roads for _, roads, _ in ready], [requests[i].start_alpha for i, _, _ in ready],
        [end for _, _, end in ready], [requests[i].point_count for i, _, _ in ready],
        [requests[i].depart_bin for i, _, _ in ready],
    )
    for (i, roads, _), progress in zip(ready, decoded):
        request, result = requests[i], results[i]
        gps = network.gps(roads, progress)
        # Preserve endpoint fractions before adding a float32 road index.
        gps[0, :2] = network.gps(roads[:1], np.asarray([request.start_alpha]))[0, :2]
        gps[-1, :2] = network.gps(roads[-1:], np.asarray([request.end_alpha]))[0, :2]
        result.status = "generated"
        result.progress = progress
        result.gps = gps
        result.time_seconds = np.arange(request.point_count) * SAMPLE_SECONDS
    return results
