"""Numerical level-encoded candidate-road features.

The candidate construction uses a small, destination-aware
candidate set and describes each candidate with local road attributes and a
greedy shortest-path suffix.  This module keeps that construction but returns
fixed-size numerical tensors for a downstream neural predictor.

The distance lookup deliberately treats zero as a valid distance.  This is
important for a candidate that is already the destination road.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


PERIOD_KEYS = ("late_night", "morning_peak", "daytime", "evening_peak", "night")
PERIOD_TO_INDEX = {name: i for i, name in enumerate(PERIOD_KEYS)}

# Feature layout after the road embedding has been concatenated by the model.
# The 29 numerical columns are:
# local: length, destination distance, 8 POI counts, total POI, density,
#        speed, speed-known mask
# future: hop count, total length, 8 POI counts, total POI, mean density,
#         mean speed, known-speed fraction, terminal distance
GEO_FEATURE_DIM = 29
POI_LEVELS = ("无", "很少", "少", "中等", "多", "很多")
DENSITY_LEVELS = ("低", "中", "高", "极高")

_DENSITY_LEVEL_VALUES = {
    **{label: float(i) / float(len(DENSITY_LEVELS) - 1) for i, label in enumerate(DENSITY_LEVELS)},
    "low": 0.0,
    "medium": 1.0 / 3.0,
    "high": 2.0 / 3.0,
    "very_high": 1.0,
    "extreme": 1.0,
}


def _density_level_value(label: Any) -> float:
    """Map Chinese or common English density labels to [0, 1]."""

    key = str(label).strip()
    return float(_DENSITY_LEVEL_VALUES.get(key, _DENSITY_LEVEL_VALUES.get(key.lower(), np.nan)))


def valid_distance(dist: np.ndarray, src: int, dst: int) -> float | None:
    """Return a finite non-negative distance, preserving a valid zero."""

    src = int(src)
    dst = int(dst)
    if src < 0 or dst < 0 or src >= dist.shape[0] or dst >= dist.shape[1]:
        return None
    value = float(dist[src, dst])
    if not np.isfinite(value) or value < 0.0:
        return None
    return value


def _load_json_rows(path: str | Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(row["fid"]): row for row in payload.get("roads", [])}


class RoadFeatureStore:
    """Load road metadata and encode candidate features for one city."""

    def __init__(
        self,
        *,
        edges_path: str | Path,
        poi_path: str | Path,
        speed_path: str | Path,
        distance_path: str | Path,
        density_path: str | Path | None = None,
        max_future_hops: int = 128,
        feature_mode: str = "numeric",
    ) -> None:
        self.edges_path = str(edges_path)
        self.poi_path = str(poi_path)
        self.speed_path = str(speed_path)
        self.distance_path = str(distance_path)
        self.density_path = str(density_path) if density_path is not None else ""
        self.max_future_hops = int(max_future_hops)
        self.feature_mode = str(feature_mode)
        if self.feature_mode not in {"numeric", "levels"}:
            raise ValueError(f"feature_mode must be numeric or levels, got {feature_mode!r}")

        self.dist = np.load(self.distance_path, mmap_mode="r")
        if self.dist.ndim != 2 or self.dist.shape[0] != self.dist.shape[1]:
            raise ValueError(f"distance table must be square, got {self.dist.shape}")
        self.num_roads = int(self.dist.shape[0])

        self.lengths = np.zeros(self.num_roads, dtype=np.float32)
        self.next_by_road: list[list[int]] = [[] for _ in range(self.num_roads)]
        seen_fids: set[int] = set()
        with Path(self.edges_path).open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                fid = int(row["fid"])
                if fid < 0 or fid >= self.num_roads:
                    continue
                if fid in seen_fids:
                    raise ValueError(f"duplicate fid {fid} in {self.edges_path}")
                seen_fids.add(fid)
                length = float(row["length"])
                if not np.isfinite(length) or length < 0.0:
                    raise ValueError(f"invalid length {length!r} for fid {fid} in {self.edges_path}")
                self.lengths[fid] = length
                # Build the line-graph successor table from edge end/start nodes.
                # First collect starts, then connect each edge to starts at its v.
        starts: dict[int, list[int]] = {}
        ends: dict[int, int] = {}
        with Path(self.edges_path).open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                fid = int(row["fid"])
                if fid < 0 or fid >= self.num_roads:
                    continue
                u = int(row["u"])
                v = int(row["v"])
                starts.setdefault(u, []).append(fid)
                ends[fid] = v
        for fid, v in ends.items():
            self.next_by_road[fid] = sorted(starts.get(v, []))
        if len(seen_fids) != self.num_roads:
            missing = sorted(set(range(self.num_roads)).difference(seen_fids))
            raise ValueError(
                f"edges.csv does not cover all distance-table road ids; "
                f"missing {len(missing)} fids (first={missing[:5]})"
            )

        poi_payload = json.loads(Path(self.poi_path).read_text(encoding="utf-8"))
        self.poi_categories = [str(x) for x in poi_payload.get("categories", [])]
        if len(self.poi_categories) != 8:
            # Keep a stable layout even if a future city has a different file.
            self.poi_categories = self.poi_categories[:8]
            while len(self.poi_categories) < 8:
                self.poi_categories.append(f"poi_{len(self.poi_categories)}")
        self.poi = np.zeros((self.num_roads, 8), dtype=np.float32)
        self.poi_total = np.zeros(self.num_roads, dtype=np.float32)
        self.poi_levels = np.zeros((self.num_roads, 8), dtype=np.float32)
        self.poi_total_level = np.zeros(self.num_roads, dtype=np.float32)
        poi_level_definition = poi_payload.get("poi_level_definition", {}) or {}
        poi_level_map = {str(name): i / 5.0 for i, name in enumerate(POI_LEVELS)}
        poi_cat_thresholds = poi_level_definition.get("category_thresholds", {}) or {}
        self.poi_cat_thresholds = {
            str(cat): np.asarray(values, dtype=np.float32) for cat, values in poi_cat_thresholds.items()
        }
        total_thresholds = poi_level_definition.get("total_poi_thresholds", []) or []
        self.poi_total_thresholds = np.asarray(total_thresholds, dtype=np.float32)
        for row in poi_payload.get("roads", []):
            fid = int(row["fid"])
            if not 0 <= fid < self.num_roads:
                continue
            counts = row.get("poi_counts_by_category", {}) or {}
            vals = [float(counts.get(cat, 0) or 0) for cat in self.poi_categories]
            self.poi[fid] = np.asarray(vals, dtype=np.float32)
            self.poi_total[fid] = float(sum(vals))
            level_by_category = row.get("poi_level_by_category", {}) or {}
            for j, cat in enumerate(self.poi_categories):
                self.poi_levels[fid, j] = float(poi_level_map.get(str(level_by_category.get(cat, "无")), 0.0))
            self.poi_total_level[fid] = float(poi_level_map.get(str(row.get("total_poi_level", "无")), 0.0))

        density_path = Path(self.density_path) if self.density_path else Path(self.edges_path).with_name(Path(self.edges_path).stem + "_density_levels.json")
        if density_path.exists():
            density_payload = json.loads(density_path.read_text(encoding="utf-8"))
            density_scores = density_payload.get("density_score_by_fid", {})
            self.density = np.asarray(
                [float(density_scores.get(str(i), 0.0)) for i in range(self.num_roads)],
                dtype=np.float32,
            )
            density_labels = density_payload.get("density_level_by_fid", {}) or {}
            self.density_level = np.asarray(
                [_density_level_value(density_labels.get(str(i), "")) for i in range(self.num_roads)],
                dtype=np.float32,
            )
            density_cuts = density_payload.get("thresholds", []) or []
            self.density_thresholds = np.asarray(density_cuts, dtype=np.float32)
        else:
            # The attribute cache is normally beside data/<city>_edges.csv.
            stem = Path(self.edges_path).stem
            fallback = Path(self.edges_path).with_name(stem.replace("_road_network", "") + "_density_levels.json")
            if fallback.exists():
                density_payload = json.loads(fallback.read_text(encoding="utf-8"))
                density_scores = density_payload.get("density_score_by_fid", {})
                self.density = np.asarray(
                    [float(density_scores.get(str(i), 0.0)) for i in range(self.num_roads)],
                    dtype=np.float32,
                )
                density_labels = density_payload.get("density_level_by_fid", {}) or {}
                self.density_level = np.asarray(
                    [_density_level_value(density_labels.get(str(i), "")) for i in range(self.num_roads)],
                    dtype=np.float32,
                )
                self.density_thresholds = np.asarray(density_payload.get("thresholds", []) or [], dtype=np.float32)
            else:
                self.density = np.zeros(self.num_roads, dtype=np.float32)
                self.density_level = np.full(self.num_roads, np.nan, dtype=np.float32)
                self.density_thresholds = np.zeros(2, dtype=np.float32)
        if np.isnan(self.density_level).any():
            valid = self.density[np.isfinite(self.density)]
            cuts = self.density_thresholds if self.density_thresholds.size >= 3 else np.quantile(valid, [0.25, 0.5, 0.75]) if valid.size else np.zeros(3)
            self.density_level[np.isnan(self.density_level)] = np.asarray(
                [
                    min(len(DENSITY_LEVELS) - 1, int(np.searchsorted(cuts, float(v), side="left")))
                    / float(len(DENSITY_LEVELS) - 1)
                    for v in self.density[np.isnan(self.density_level)]
                ],
                dtype=np.float32,
            )

        speed_payload = json.loads(Path(self.speed_path).read_text(encoding="utf-8"))
        self.speed_level_definition = speed_payload.get("speed_level_definition", {}) or {}
        self.speed = np.full((self.num_roads, len(PERIOD_KEYS)), np.nan, dtype=np.float32)
        for row in speed_payload.get("roads", []):
            fid = int(row["fid"])
            if not 0 <= fid < self.num_roads:
                continue
            for period, item in (row.get("speeds_by_period", {}) or {}).items():
                j = PERIOD_TO_INDEX.get(str(period))
                if j is not None and item and item.get("speed_kmh") is not None:
                    self.speed[fid, j] = float(item["speed_kmh"])

        length_values = self.lengths[np.isfinite(self.lengths)]
        self.length_level_thresholds = np.quantile(length_values, [0.2, 0.4, 0.6, 0.8]).astype(np.float32) if length_values.size else np.zeros(4, dtype=np.float32)
        # A sampled distance quantile table avoids materializing the full
        # NxN shortest-path matrix while keeping level semantics deterministic.
        rng = np.random.default_rng(0)
        sample_n = min(200_000, max(10_000, self.num_roads * 8))
        rr = rng.integers(0, self.num_roads, size=sample_n)
        cc = rng.integers(0, self.num_roads, size=sample_n)
        sampled_dist = np.asarray(self.dist[rr, cc], dtype=np.float32)
        sampled_dist = sampled_dist[np.isfinite(sampled_dist) & (sampled_dist >= 0)]
        self.distance_level_thresholds = np.quantile(sampled_dist, [0.2, 0.4, 0.6, 0.8]).astype(np.float32) if sampled_dist.size else np.zeros(4, dtype=np.float32)
        finite_speed = self.speed[np.isfinite(self.speed) & (self.speed > 0)]
        self.speed_global_thresholds = np.quantile(finite_speed, [0.2, 0.4, 0.6, 0.8]).astype(np.float32) if finite_speed.size else np.zeros(4, dtype=np.float32)

        # Caches make the future summary cheap when many decisions share a
        # destination.  The cache stores the raw summary values before scaling.
        self._future_cache: dict[tuple[int, int, int], tuple[np.ndarray, float]] = {}

    @property
    def feature_dim(self) -> int:
        return GEO_FEATURE_DIM

    @staticmethod
    def _quantile_level(value: float, thresholds: np.ndarray, max_level: int = 4) -> float:
        if not np.isfinite(value):
            return 0.0
        return min(max_level, int(np.searchsorted(thresholds, value, side="left"))) / float(max_level)

    @staticmethod
    def _count_level(value: float, thresholds: np.ndarray) -> float:
        """Encode an attribute count using its zero/positive level rule."""

        if not np.isfinite(value) or float(value) <= 0.0:
            return 0.0
        cuts = np.asarray(thresholds, dtype=np.float64).reshape(-1)
        if cuts.size == 0:
            return 1.0
        # The level encoding uses <= threshold boundaries and reserves level 0 for
        # exactly zero counts.  ``side='left'`` keeps a value equal to a cut in
        # the lower positive level.
        level = 1 + int(np.searchsorted(cuts, float(value), side="left"))
        return float(min(5, level)) / 5.0

    def _speed_level_value(self, value: float, period: str | int) -> float:
        if not np.isfinite(value) or float(value) <= 0.0:
            return 0.0
        period_name = PERIOD_KEYS[int(period) % len(PERIOD_KEYS)] if not isinstance(period, str) else period
        thresholds = (self.speed_level_definition.get("thresholds_by_period", {}) or {}).get(period_name)
        if not thresholds:
            return self._quantile_level(value, self.speed_global_thresholds)
        cuts = np.asarray(
            [thresholds.get(k, -np.inf) for k in ("很慢_max", "慢_max", "中等_max", "快_max")],
            dtype=np.float64,
        )
        return self._quantile_level(value, cuts)

    def _future_suffix(self, candidate: int, destination: int) -> list[int]:
        candidate = int(candidate)
        destination = int(destination)
        if candidate == destination:
            return []
        path: list[int] = []
        visited = {candidate}
        current = candidate
        for _ in range(self.max_future_hops):
            best: int | None = None
            best_dist = float("inf")
            for nxt in self.next_by_road[current] if 0 <= current < self.num_roads else []:
                nxt = int(nxt)
                if nxt in visited:
                    continue
                d = valid_distance(self.dist, nxt, destination)
                if d is not None and d < best_dist:
                    best = nxt
                    best_dist = d
            if best is None:
                break
            path.append(best)
            if best == destination:
                break
            visited.add(best)
            current = best
        return path

    def _future_summary(self, candidate: int, destination: int, period_idx: int) -> tuple[np.ndarray, float]:
        key = (int(candidate), int(destination), int(period_idx) % len(PERIOD_KEYS))
        cached = self._future_cache.get(key)
        if cached is not None:
            return cached
        suffix = self._future_suffix(key[0], key[1])
        if suffix:
            lengths = self.lengths[np.asarray(suffix, dtype=np.int64)]
            poi = self.poi[np.asarray(suffix, dtype=np.int64)].sum(axis=0)
            dens = self.density[np.asarray(suffix, dtype=np.int64)]
            speeds = self.speed[np.asarray(suffix, dtype=np.int64), key[2]]
            known = np.isfinite(speeds) & (speeds > 0.0)
            speed_values = speeds[known]
            avg_speed = float(np.mean(speed_values)) if speed_values.size else 0.0
            speed_fraction = float(np.mean(known)) if known.size else 0.0
            terminal = valid_distance(self.dist, suffix[-1], destination)
            terminal = float(terminal) if terminal is not None else float(valid_distance(self.dist, key[0], destination) or 0.0)
            summary = np.concatenate(
                [
                    np.asarray([len(suffix), float(lengths.sum())], dtype=np.float32),
                    poi.astype(np.float32),
                    np.asarray([float(poi.sum()), float(dens.mean()), avg_speed, speed_fraction], dtype=np.float32),
                ]
            )
        else:
            d = valid_distance(self.dist, key[0], destination)
            summary = np.zeros(14, dtype=np.float32)
            terminal = float(d) if d is not None else 0.0
        out = (summary, float(terminal))
        self._future_cache[key] = out
        return out

    def encode_one(
        self,
        candidate: int,
        destination: int,
        period: str | int,
        *,
        speed_mask_prob: float = 0.0,
        rng: random.Random | None = None,
    ) -> np.ndarray:
        candidate = int(candidate)
        destination = int(destination)
        if not 0 <= candidate < self.num_roads:
            raise IndexError(f"candidate road {candidate} outside [0,{self.num_roads})")
        if isinstance(period, str):
            period_idx = PERIOD_TO_INDEX.get(period, 2)
        else:
            period_idx = int(period) % len(PERIOD_KEYS)

        speed_mask_prob = float(speed_mask_prob)
        if not 0.0 <= speed_mask_prob <= 1.0:
            raise ValueError(f"speed_mask_prob must be in [0,1], got {speed_mask_prob}")

        d = valid_distance(self.dist, candidate, destination)
        d_value = float(d) if d is not None else 0.0
        local_speed = float(self.speed[candidate, period_idx])
        speed_known = 1.0 if np.isfinite(local_speed) and local_speed > 0.0 else 0.0
        if speed_known > 0.0 and speed_mask_prob > 0.0:
            masked = speed_mask_prob >= 1.0 or (rng is not None and rng.random() < speed_mask_prob)
            if masked:
                local_speed = np.nan
                speed_known = 0.0
        if not np.isfinite(local_speed):
            local_speed = 0.0
        future, terminal = self._future_summary(candidate, destination, period_idx)
        future_speed_masked = False
        if future[0] > 0.0 and speed_mask_prob > 0.0:
            future_speed_masked = speed_mask_prob >= 1.0 or (
                rng is not None and rng.random() < speed_mask_prob
            )

        # Log and scale heavy-tailed counts/distances.  These constants are
        # deliberately fixed so a cached dataset has the same semantics across
        # training runs and a zero distance remains exactly zero.
        out = np.zeros(GEO_FEATURE_DIM, dtype=np.float32)
        out[0] = np.log1p(max(float(self.lengths[candidate]), 0.0)) / 10.0
        out[1] = np.log1p(max(d_value, 0.0)) / 12.0
        out[2:10] = np.log1p(np.maximum(self.poi[candidate], 0.0)) / 8.0
        out[10] = np.log1p(max(float(self.poi_total[candidate]), 0.0)) / 10.0
        out[11] = np.log1p(max(float(self.density[candidate]), 0.0)) / 5.0
        out[12] = local_speed / 80.0
        out[13] = speed_known
        out[14] = np.log1p(float(future[0])) / 5.0
        out[15] = np.log1p(max(float(future[1]), 0.0)) / 10.0
        out[16:24] = np.log1p(np.maximum(future[2:10], 0.0)) / 8.0
        out[24] = np.log1p(max(float(future[10]), 0.0)) / 10.0
        out[25] = np.log1p(max(float(future[11]), 0.0)) / 5.0
        out[26] = 0.0 if future_speed_masked else float(future[12]) / 80.0
        out[27] = 0.0 if future_speed_masked else float(future[13])
        out[28] = np.log1p(max(float(terminal), 0.0)) / 12.0
        if self.feature_mode == "levels":
            out[0] = self._quantile_level(float(self.lengths[candidate]), self.length_level_thresholds)
            # Keep the exact shortest-path distance as a model input.  The
            # hard constraint and the shortest-path baseline use this same
            # value; only the logarithmic scale is applied for optimization.
            out[1] = np.log1p(max(d_value, 0.0)) / 12.0
            out[2:10] = self.poi_levels[candidate]
            out[10] = self.poi_total_level[candidate]
            out[11] = float(self.density_level[candidate])
            out[12] = self._speed_level_value(local_speed, period)
            out[14] = min(4, int(future[0])) / 4.0
            out[15] = self._quantile_level(float(future[1]), self.length_level_thresholds)
            # Use the same level vocabulary as the candidate conditions for the
            # future POI summary.  Raw aggregate counts are city-scale and
            # would otherwise reintroduce a transfer-sensitive feature even
            # though the local POI inputs are level encoded.
            if self.poi_cat_thresholds:
                out[16:24] = np.asarray(
                    [
                        self._count_level(
                            float(value),
                            self.poi_cat_thresholds.get(cat, np.asarray([], dtype=np.float32)),
                        )
                        for value, cat in zip(future[2:10], self.poi_categories)
                    ],
                    dtype=np.float32,
                )
            else:
                out[16:24] = np.log1p(np.maximum(future[2:10], 0.0)) / 8.0
            if self.poi_total_thresholds.size:
                out[24] = self._count_level(float(future[10]), self.poi_total_thresholds)
            else:
                out[24] = np.log1p(max(float(future[10]), 0.0)) / 10.0
            out[25] = (
                float(self._quantile_level(float(future[11]), self.density_thresholds, max_level=len(DENSITY_LEVELS) - 1))
                if self.density_thresholds.size >= 3
                else out[25]
            )
            out[26] = 0.0 if future_speed_masked else self._speed_level_value(float(future[12]), period_idx)
            out[27] = 0.0 if future_speed_masked else float(future[13])
            out[28] = self._quantile_level(float(terminal), self.distance_level_thresholds)
        return out

    def encode_candidates(
        self,
        candidates: list[int],
        destination: int,
        period: str | int,
        *,
        speed_mask_prob: float = 0.0,
        rng: random.Random | None = None,
    ) -> np.ndarray:
        return np.stack(
            [
                self.encode_one(
                    rid,
                    destination,
                    period,
                    speed_mask_prob=speed_mask_prob,
                    rng=rng,
                )
                for rid in candidates
            ],
            axis=0,
        ).astype(np.float32)



MAX_CANDIDATES = 5


def reachable_candidates(store: RoadFeatureStore, current: int, destination: int, visited: set[int]):
    """Same unvisited, reachable, ID-sorted successors in training and rollout."""
    candidates = []
    for nxt in store.next_by_road[current]:
        if nxt not in visited and (distance := valid_distance(store.dist, nxt, destination)) is not None:
            candidates.append((nxt, distance))
            if len(candidates) == MAX_CANDIDATES:
                break
    return candidates


def hard_candidate_indices(distances, current_distance):
    """Apply the original non-increasing distance constraint and its fallback."""
    if current_distance is not None:
        eligible = [i for i, d in enumerate(distances) if d <= current_distance]
        if eligible:
            return eligible
    return list(range(len(distances)))


def departure_period(depart_bin: int) -> int:
    return int(np.searchsorted([72, 108, 204, 240], depart_bin, side="right"))
