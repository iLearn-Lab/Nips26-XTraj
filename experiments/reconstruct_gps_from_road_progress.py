"""Reconstruct lon/lat GPS points from road-progress representation.

Input road-progress dataset format (per sample):
  [meta, roads_unique, s_sequence, t_sequence]

- roads_unique: int array [M]
- s_sequence: float array [T], with s_k = i + alpha
- t_sequence: float array [T], copied from original gps_points[:,3] (the 4th value)

Output dataset format (per sample):
  [meta, gps_points]
Where gps_points is float32 [T,4] and stores:
  gps_points[:,0] = lon
  gps_points[:,1] = lat
  gps_points[:,2] = road_id
  gps_points[:,3] = time (from t_sequence, or 0 if missing)

Usage:
  python experiments/reconstruct_gps_from_road_progress.py \
    --input data/xian_road_progress/20161001_road_progress.npy \
    --edges data/xian_road_network/edges.csv \
    --output data/xian_road_progress/20161001_reconstructed_gps.npy
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

from utils.wkt_geometry import PolylineIndex, build_fid_to_polyline_index, interpolate_polyline


def _reconstruct_one(
    *,
    meta: Any,
    roads_unique: np.ndarray,
    s: np.ndarray,
    t_seq: np.ndarray | None,
    fid2poly: dict[int, PolylineIndex],
) -> Any:
    roads_unique = np.asarray(roads_unique, dtype=np.int64).reshape(-1)
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    if t_seq is None:
        t_seq = np.zeros_like(s, dtype=np.float32)
    else:
        t_seq = np.asarray(t_seq, dtype=np.float32).reshape(-1)
        if len(t_seq) != len(s):
            t_seq = np.zeros_like(s, dtype=np.float32)

    t = int(s.shape[0])
    gps = np.zeros((t, 4), dtype=np.float32)

    m = int(roads_unique.shape[0])
    if m == 0:
        gps[:, 3] = t_seq.astype(np.float32)
        return [meta, gps]

    for k in range(t):
        sk = float(s[k])
        i = int(np.floor(sk))
        alpha = sk - float(i)
        i = int(np.clip(i, 0, m - 1))
        alpha = float(np.clip(alpha, 0.0, 1.0))

        rid = int(roads_unique[i])
        poly = fid2poly.get(rid)
        if poly is None:
            lon, lat = 0.0, 0.0
        else:
            lon, lat = interpolate_polyline(alpha, poly)

        gps[k, 0] = lon
        gps[k, 1] = lat
        gps[k, 2] = float(rid)
        gps[k, 3] = float(t_seq[k])

    return [meta, gps]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct lon/lat from road-progress dataset")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--edges", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-trajs", type=int, default=0, help="0 = all")

    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading edges from {args.edges} ...")
    fid2poly = build_fid_to_polyline_index(args.edges)
    print(f"Loaded {len(fid2poly)} road polylines")

    print(f"Loading road-progress dataset from {inp} ...")
    arr = np.load(str(inp), allow_pickle=True)

    samples: List[Any] = []
    for idx, item in enumerate(arr):
        if args.max_trajs > 0 and len(samples) >= int(args.max_trajs):
            break

        try:
            meta = item[0]
            roads_unique = item[1]
            s = item[2]
            t_seq = item[3] if len(item) > 3 else None
        except Exception:
            continue

        samples.append(_reconstruct_one(meta=meta, roads_unique=roads_unique, s=s, t_seq=t_seq, fid2poly=fid2poly))

        if (idx + 1) % 2000 == 0:
            print(f"processed {idx+1}/{len(arr)}, wrote {len(samples)}")

    np.save(str(out), np.asarray(samples, dtype=object))
    print(f"Saved {len(samples)} reconstructed samples to: {out}")


if __name__ == "__main__":
    main()
