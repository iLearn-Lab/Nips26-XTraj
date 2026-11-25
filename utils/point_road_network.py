"""Directed road geometry and conversions between progress and metres."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from utils.wkt_geometry import build_polyline_index, interpolate_polyline, parse_linestring


class PointRoadNetwork:
    def __init__(self, path: Path, n_roads: int):
        self.lengths = np.zeros(n_roads, dtype=np.float32)
        self.starts = np.full(n_roads, -1, dtype=np.int64)
        self.ends = np.full(n_roads, -1, dtype=np.int64)
        self.polylines = {}
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                fid = int(row["fid"])
                if not 0 <= fid < n_roads or fid in self.polylines:
                    raise ValueError(f"Unexpected or duplicate road ID {fid} in {path}")
                polyline = build_polyline_index(parse_linestring(row["wkt"]))
                length = float(row["length"])
                if not np.isfinite(length) or length <= 0 or not np.isfinite(polyline.pts).all() or polyline.total_len <= 0:
                    raise ValueError(f"Invalid road length or polyline for road {fid}")
                self.lengths[fid] = length
                self.starts[fid] = int(row["u"])
                self.ends[fid] = int(row["v"])
                self.polylines[fid] = polyline
        if len(self.polylines) != n_roads:
            raise ValueError(f"{path} and the embedding do not cover the same road IDs")

    def validate_route(self, roads: np.ndarray) -> None:
        if np.min(roads) < 0 or np.max(roads) >= len(self.lengths):
            raise ValueError("unknown_road_id")
        if np.any(self.ends[roads[:-1]] != self.starts[roads[1:]]):
            raise ValueError("disconnected_route")

    def position(self, roads: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        indices = np.minimum(s.astype(np.int64), len(roads) - 1)
        return roads[indices], s - indices

    def metres(self, roads: np.ndarray, s: np.ndarray) -> np.ndarray:
        lengths = self.lengths[roads].astype(np.float64)
        prefix = np.r_[0.0, np.cumsum(lengths)]
        indices = np.minimum(s.astype(np.int64), len(roads) - 1)
        return prefix[indices] + (s - indices) * lengths[indices]

    def gps(self, roads: np.ndarray, s: np.ndarray) -> np.ndarray:
        ids, alphas = self.position(roads, s)
        coords = [interpolate_polyline(alpha, self.polylines[int(fid)]) for fid, alpha in zip(ids, alphas)]
        return np.column_stack((np.asarray(coords, dtype=np.float32), ids.astype(np.float32)))
