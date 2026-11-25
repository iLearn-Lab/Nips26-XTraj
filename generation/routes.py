"""Autoregressive route generation with optional distance constraints."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from generation.requests import GenerationRequest
from utils.candidate_features import (
    MAX_CANDIDATES, RoadFeatureStore, departure_period,
    hard_candidate_indices, reachable_candidates, valid_distance,
)


@dataclass
class RouteResult:
    roads: list[int]
    status: str = "pending"


class RoadGenerator:
    def __init__(self, model, z, store: RoadFeatureStore):
        device = next(model.parameters()).device
        cfg = model.cfg
        if tuple(z.shape) != (store.num_roads, cfg.emb_dim) or cfg.geo_dim != store.feature_dim:
            raise ValueError("Road model, map and embeddings have different dimensions")
        if store.feature_mode != "levels":
            raise ValueError("Use the level-encoded candidate feature store")
        self.model = model.eval()
        self.z = z.detach().to(device)
        self.store = store

    @torch.inference_mode()
    def generate(self, samples: Sequence[GenerationRequest], *, max_roads: int, hard: bool) -> list[RouteResult]:
        if max_roads < 1:
            raise ValueError("max_roads must be positive")
        if any(max(s.start_road, s.destination_road) >= self.store.num_roads for s in samples):
            raise ValueError("Request contains an unknown road ID")
        self.model.eval()
        results = [RouteResult([int(s.start_road)]) for s in samples]
        destinations = [int(s.destination_road) for s in samples]
        periods = [departure_period(s.depart_bin) for s in samples]
        for i, record in enumerate(results):
            if record.roads[-1] == destinations[i]:
                record.status = "reached_destination"
            elif valid_distance(self.store.dist, record.roads[-1], destinations[i]) is None:
                record.status = "unreachable"

        cap = MAX_CANDIDATES
        device = self.z.device
        hist_cap = self.model.cfg.max_history_len
        for _ in range(max_roads - 1):
            choices = []
            for i, record in enumerate(results):
                if record.status != "pending":
                    continue
                current, dst = record.roads[-1], destinations[i]
                visited = set(record.roads)
                candidates = reachable_candidates(self.store, current, dst, visited)
                rids = [rid for rid, _ in candidates]
                if not rids:
                    record.status = "no_eligible_successor"
                    continue
                eligible = rids
                if hard:
                    current_dist = valid_distance(self.store.dist, current, dst)
                    eligible = [rids[j] for j in hard_candidate_indices([d for _, d in candidates], current_dist)]
                if len(eligible) == 1:
                    record.roads.append(eligible[0])
                else:
                    choices.append((i, rids, eligible))
            if choices:
                rows = len(choices)
                h = max(min(hist_cap, len(results[i].roads)) for i, _, _ in choices)
                history = np.zeros((rows, h), dtype=np.int64)
                hmask = np.zeros((rows, h), dtype=bool)
                cids = np.zeros((rows, cap), dtype=np.int64)
                cmask = np.zeros((rows, cap), dtype=bool)
                geo = np.zeros((rows, cap, self.store.feature_dim), dtype=np.float32)
                for j, (i, rids, eligible) in enumerate(choices):
                    visible = results[i].roads[-hist_cap:]
                    history[j, :len(visible)] = visible
                    hmask[j, :len(visible)] = True
                    cids[j, :len(rids)] = rids
                    cmask[j, :len(rids)] = [rid in eligible for rid in rids]
                    geo[j, :len(rids)] = self.store.encode_candidates(
                        rids, destinations[i], periods[i], speed_mask_prob=0.0,
                    )
                logits = self.model(
                    history_road_ids=torch.from_numpy(history).to(device),
                    history_mask=torch.from_numpy(hmask).to(device),
                    candidate_road_ids=torch.from_numpy(cids).to(device),
                    candidate_geo=torch.from_numpy(geo).to(device),
                    candidate_mask=torch.from_numpy(cmask).to(device),
                    period_index=torch.tensor([periods[i] for i, _, _ in choices], device=device),
                    z=self.z,
                )
                for (i, rids, eligible), j in zip(choices, logits.argmax(dim=-1).cpu().tolist()):
                    if j >= len(rids) or rids[j] not in eligible:
                        raise RuntimeError("Road model selected an invalid candidate")
                    results[i].roads.append(rids[j])
            for i, record in enumerate(results):
                if record.status == "pending" and record.roads[-1] == destinations[i]:
                    record.status = "reached_destination"
            if all(record.status != "pending" for record in results):
                break
        for record in results:
            if record.status == "pending":
                record.status = "route_limit"
        return results
