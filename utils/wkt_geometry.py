from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


def parse_linestring(wkt: str) -> np.ndarray:
    """Parse WKT LINESTRING into an (M,2) float array [[lon,lat], ...].

    Notes:
      - This repo's edges.csv currently uses LINESTRING.
      - MULTILINESTRING is not supported here (raise ValueError) so we fail fast.
    """
    if wkt is None:
        return np.empty((0, 2), dtype=np.float32)

    s = str(wkt).strip()
    if not s:
        return np.empty((0, 2), dtype=np.float32)

    if s.startswith("MULTILINESTRING"):
        raise ValueError("MULTILINESTRING is not supported in parse_linestring")

    if s.startswith("LINESTRING"):
        s = s.replace("LINESTRING (", "").replace("LINESTRING(", "").rstrip(")")

    s = s.strip().strip("(").strip(")")
    if not s:
        return np.empty((0, 2), dtype=np.float32)

    pts: List[Tuple[float, float]] = []
    for part in s.split(","):
        xy = part.strip().split()
        if len(xy) < 2:
            continue
        pts.append((float(xy[0]), float(xy[1])))

    return np.asarray(pts, dtype=np.float32)


def polyline_cumlen(pts: np.ndarray) -> np.ndarray:
    """Return cumulative length array of shape [L].

    cumlen[0] = 0
    cumlen[i] = length from pts[0] to pts[i] along the polyline.
    """
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"pts must be [L,2], got shape={pts.shape}")

    l = int(pts.shape[0])
    if l == 0:
        return np.zeros((0,), dtype=np.float32)
    if l == 1:
        return np.zeros((1,), dtype=np.float32)

    seg = pts[1:] - pts[:-1]
    seg_len = np.sqrt(np.sum(seg * seg, axis=1))  # [L-1]
    cum = np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(seg_len, dtype=np.float32)])
    return cum.astype(np.float32)


@dataclass(frozen=True)
class PolylineIndex:
    pts: np.ndarray  # [L,2] float32
    cumlen: np.ndarray  # [L] float32
    total_len: float


def build_polyline_index(pts: np.ndarray) -> PolylineIndex:
    pts = np.asarray(pts, dtype=np.float32)
    cum = polyline_cumlen(pts)
    total = float(cum[-1]) if len(cum) > 0 else 0.0
    return PolylineIndex(pts=pts, cumlen=cum, total_len=total)


def project_point_to_polyline_alpha(point: np.ndarray, poly: PolylineIndex) -> float:
    """Project a point onto a polyline and return alpha in [0,1].

    Uses Euclidean distance in lon/lat space (consistent with repo's current plotting).
    """
    pts = poly.pts
    l = int(pts.shape[0])
    if l < 2 or poly.total_len <= 0:
        return 0.0

    p = np.asarray(point, dtype=np.float32).reshape(2)

    a = pts[:-1]  # [L-1,2]
    b = pts[1:]  # [L-1,2]
    v = b - a  # [L-1,2]
    vv = np.sum(v * v, axis=1)  # [L-1]
    # Avoid division by zero for degenerate segments.
    vv = np.maximum(vv, 1e-12)

    pa = p - a  # [L-1,2]
    t = np.sum(pa * v, axis=1) / vv  # [L-1]
    t = np.clip(t, 0.0, 1.0)

    q = a + t[:, None] * v  # [L-1,2]
    d2 = np.sum((p - q) ** 2, axis=1)  # [L-1]

    j = int(np.argmin(d2))
    seg_len = float(np.sqrt(vv[j]))
    proj_len = float(poly.cumlen[j] + t[j] * seg_len)
    alpha = proj_len / float(poly.total_len)
    return float(np.clip(alpha, 0.0, 1.0))


def interpolate_polyline(point_alpha: float, poly: PolylineIndex) -> Tuple[float, float]:
    """Interpolate (lon,lat) at alpha along polyline."""
    pts = poly.pts
    l = int(pts.shape[0])
    if l == 0:
        return 0.0, 0.0
    if l == 1 or poly.total_len <= 0:
        return float(pts[0, 0]), float(pts[0, 1])

    alpha = float(np.clip(point_alpha, 0.0, 1.0))
    d = alpha * float(poly.total_len)

    # Find last index where cumlen <= d.
    # cumlen is sorted ascending.
    i = int(np.searchsorted(poly.cumlen, d, side="right") - 1)
    i = int(np.clip(i, 0, l - 2))

    d0 = float(poly.cumlen[i])
    d1 = float(poly.cumlen[i + 1])
    denom = (d1 - d0) if (d1 - d0) != 0 else 1.0
    t = (d - d0) / denom

    x = pts[i] + t * (pts[i + 1] - pts[i])
    return float(x[0]), float(x[1])


def build_fid_to_polyline_index(edges_csv_path: str) -> Dict[int, PolylineIndex]:
    """Load edges.csv and build fid -> PolylineIndex mapping.

    Expects columns: fid, wkt.
    """
    import pandas as pd

    df = pd.read_csv(edges_csv_path)
    if "fid" not in df.columns or "wkt" not in df.columns:
        raise ValueError(f"edges.csv must contain columns fid and wkt, got {list(df.columns)}")

    out: Dict[int, PolylineIndex] = {}
    for _, row in df.iterrows():
        fid = int(row["fid"])
        pts = parse_linestring(row["wkt"])
        out[fid] = build_polyline_index(pts)

    return out
