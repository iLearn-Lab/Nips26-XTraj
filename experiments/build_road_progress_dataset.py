"""Build Road-Progress 1D dataset: s = i + alpha.

This converts a raw trajectory .npy file (dataset convention) into a new dataset
where each sample stores:
  [meta, roads_unique, s_sequence, t_sequence]

- roads_unique: consecutive-dedup road_id sequence (int32)
- s_sequence: float32 array after optional uniform-time resampling
  s_k = i_k + alpha_k
- t_sequence: float32 array aligned with s_sequence, storing per-step time deltas

Where i_k is the segment index in roads_unique that the k-th GPS point belongs to.

Notes:
- road_id == 0 is treated as valid.
- We rely on edges.csv WKT geometry to compute alpha.
- Recommended workflow: first build 1s data, then subsample the 1s dataset to 3s.

Usage:
  python experiments/build_road_progress_dataset.py \
    --input data/xian/20161001_order_gps.npy \
    --edges data/xian_road_network/edges.csv \
    --output data/xian_road_progress/road_progress_1s/20161001_order_gps_road_progress_1s.npy \
    --sample-seconds 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from utils.sampling import get_unique_segments
from utils.wkt_geometry import PolylineIndex, build_fid_to_polyline_index, project_point_to_polyline_alpha


def _resample_by_seconds(s_seq: np.ndarray, t_seq: np.ndarray, sample_seconds: int) -> Tuple[np.ndarray, np.ndarray]:
    """Resample road-progress sequence to a coarser fixed interval.

    Assumes:
      - t_seq[0] is the first-step marker (typically 0)
      - for k>=1, t_seq[k] is the elapsed seconds from point k-1 -> k

    For sample_seconds=1, this behaves like dense 1s interpolation.
    For sample_seconds=3, it keeps points every 3 seconds while preserving the end point.
    """
    s = np.asarray(s_seq, dtype=np.float32).reshape(-1)
    t = np.asarray(t_seq, dtype=np.float32).reshape(-1)
    if len(s) == 0:
        return s.astype(np.float32), t.astype(np.float32)
    if len(t) != len(s):
        return s.astype(np.float32), np.zeros_like(s, dtype=np.float32)
    if sample_seconds <= 1:
        dense_s = [float(s[0])]
        dense_t = [0.0]
        for k in range(1, len(s)):
            a = float(s[k - 1])
            b = float(s[k])
            dt = int(round(float(t[k])))
            if dt <= 0:
                dense_s.append(b)
                dense_t.append(0.0)
                continue
            step = (b - a) / float(dt)
            for j in range(1, dt + 1):
                dense_s.append(a + step * j)
                dense_t.append(1.0)
        return np.asarray(dense_s, dtype=np.float32), np.asarray(dense_t, dtype=np.float32)

    dense_s, _ = _resample_by_seconds(s, t, 1)
    keep = list(range(0, len(dense_s), sample_seconds))
    if keep[-1] != len(dense_s) - 1:
        keep.append(len(dense_s) - 1)

    out_s = dense_s[keep].astype(np.float32)
    out_t = np.full((len(out_s),), float(sample_seconds), dtype=np.float32)
    if len(out_t) > 0:
        out_t[0] = 0.0
    if len(out_t) > 1 and keep[-1] == len(dense_s) - 1:
        prev_idx = keep[-2] if len(keep) >= 2 else 0
        out_t[-1] = float(keep[-1] - prev_idx)
    return out_s, out_t


def _extract_lonlat_road_ids_and_time(gps_points: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (lon, lat, road_id, time) arrays after filtering invalid road ids.

    We treat road_id >= 0 as valid.

    Assumes gps_points is array-like and each point p supports p[0], p[1], p[2].
    If p[3] is missing, time defaults to 0.
    """
    lon_list: List[float] = []
    lat_list: List[float] = []
    rid_list: List[int] = []
    t_list: List[float] = []

    for p in gps_points:
        try:
            rid = int(p[2])
        except Exception:
            continue
        if rid < 0:
            continue
        try:
            lon = float(p[0])
            lat = float(p[1])
        except Exception:
            continue

        try:
            t = float(p[3])
        except Exception:
            t = 0.0

        lon_list.append(lon)
        lat_list.append(lat)
        rid_list.append(rid)
        t_list.append(t)

    return (
        np.asarray(lon_list, dtype=np.float32),
        np.asarray(lat_list, dtype=np.float32),
        np.asarray(rid_list, dtype=np.int64),
        np.asarray(t_list, dtype=np.float32),
    )


def _segment_indices_from_road_ids(road_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given per-point road_ids, return (roads_unique, seg_idx_per_point).

    seg_idx increments when road_id changes.
    """
    if road_ids.ndim != 1:
        raise ValueError("road_ids must be 1D")
    if len(road_ids) == 0:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32)

    seg_idx = np.zeros((len(road_ids),), dtype=np.int32)
    cur = int(road_ids[0])
    i = 0
    for k in range(1, len(road_ids)):
        r = int(road_ids[k])
        if r != cur:
            i += 1
            cur = r
        seg_idx[k] = i

    roads_unique = get_unique_segments([int(x) for x in road_ids.tolist()])
    roads_unique = np.asarray(roads_unique, dtype=np.int32)

    # Sanity: last seg index should be len(roads_unique)-1.
    if len(roads_unique) > 0 and int(seg_idx.max()) != len(roads_unique) - 1:
        # This should not happen if get_unique_segments matches the seg index definition.
        raise RuntimeError(
            f"Segment index mismatch: seg_idx.max()={int(seg_idx.max())} but len(roads_unique)={len(roads_unique)}"
        )

    return roads_unique, seg_idx


def _maybe_reverse_polyline_for_segment(
    poly: PolylineIndex,
    lon: np.ndarray,
    lat: np.ndarray,
) -> PolylineIndex:
    """Heuristic: decide whether to reverse polyline direction for this segment.

    Compares start/end GPS points to polyline endpoints.
    """
    if len(lon) == 0:
        return poly

    p0 = np.asarray([lon[0], lat[0]], dtype=np.float32)
    p1 = np.asarray([lon[-1], lat[-1]], dtype=np.float32)

    a = poly.pts[0]
    b = poly.pts[-1]

    # forward: p0 near a, p1 near b
    # backward: p0 near b, p1 near a
    fwd = float(np.sum((p0 - a) ** 2) + np.sum((p1 - b) ** 2))
    bwd = float(np.sum((p0 - b) ** 2) + np.sum((p1 - a) ** 2))

    if bwd < fwd:
        rev_pts = poly.pts[::-1].copy()
        from utils.wkt_geometry import build_polyline_index

        return build_polyline_index(rev_pts)

    return poly


def build_road_progress_sample(
    *,
    meta: Any,
    gps_points: Any,
    fid2poly: dict[int, PolylineIndex],
    min_unique_len: int,
    reverse_polyline: bool,
    sample_seconds: int = 1,
) -> Any | None:
    lon, lat, road_ids, t_seq = _extract_lonlat_road_ids_and_time(gps_points)
    if len(road_ids) < 2:
        return None

    roads_unique, seg_idx = _segment_indices_from_road_ids(road_ids)
    if len(roads_unique) < int(min_unique_len):
        return None

    # Pre-compute per-segment polyline direction (optional heuristic).
    poly_per_seg: List[PolylineIndex] = []
    for i in range(len(roads_unique)):
        rid = int(roads_unique[i])
        poly = fid2poly.get(rid)
        if poly is None:
            poly = PolylineIndex(pts=np.empty((0, 2), dtype=np.float32), cumlen=np.zeros((0,), dtype=np.float32), total_len=0.0)

        if reverse_polyline:
            mask = seg_idx == i
            poly = _maybe_reverse_polyline_for_segment(poly, lon[mask], lat[mask])

        poly_per_seg.append(poly)

    s = np.zeros((len(road_ids),), dtype=np.float32)
    for k in range(len(road_ids)):
        i = int(seg_idx[k])
        rid = int(roads_unique[i])
        poly = poly_per_seg[i]

        alpha = project_point_to_polyline_alpha(np.asarray([lon[k], lat[k]], dtype=np.float32), poly)
        s[k] = float(i) + float(alpha)

    # Keep time aligned with filtered points.
    if len(t_seq) != len(s):
        t_seq = np.zeros_like(s, dtype=np.float32)

    s, t_seq = _resample_by_seconds(s, t_seq, int(sample_seconds))

    return [meta, roads_unique.astype(np.int32), s.astype(np.float32), t_seq.astype(np.float32)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build road-progress dataset (s=i+alpha) from raw trajectories")
    parser.add_argument("--input", type=str, required=True, help="Input raw .npy file")
    parser.add_argument("--edges", type=str, required=True, help="edges.csv with fid,wkt")
    parser.add_argument("--output", type=str, required=True, help="Output .npy path")

    parser.add_argument("--min-unique-len", type=int, default=5)
    parser.add_argument("--max-trajs", type=int, default=0, help="0 = all")
    parser.add_argument("--reverse-polyline", action="store_true", help="Apply per-segment polyline direction heuristic")
    parser.add_argument("--sample-seconds", type=int, default=1, help="Resample trajectory points to this many seconds")

    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading edges from {args.edges} ...")
    fid2poly = build_fid_to_polyline_index(args.edges)
    print(f"Loaded {len(fid2poly)} road polylines")

    print(f"Loading trajectories from {inp} ...")
    arr = np.load(str(inp), allow_pickle=True)

    samples: List[Any] = []
    n_in = len(arr)
    for idx, traj in enumerate(arr):
        if args.max_trajs > 0 and len(samples) >= int(args.max_trajs):
            break

        try:
            meta = traj[0]
            gps_points = traj[1]
        except Exception:
            continue

        sample = build_road_progress_sample(
            meta=meta,
            gps_points=gps_points,
            fid2poly=fid2poly,
            min_unique_len=int(args.min_unique_len),
            reverse_polyline=bool(args.reverse_polyline),
            sample_seconds=int(args.sample_seconds),
        )
        if sample is None:
            continue
        samples.append(sample)

        if (idx + 1) % 2000 == 0:
            print(f"processed {idx+1}/{n_in}, kept {len(samples)}")

    np.save(str(out), np.asarray(samples, dtype=object))
    print(f"Saved {len(samples)} samples to: {out}")


if __name__ == "__main__":
    main()
