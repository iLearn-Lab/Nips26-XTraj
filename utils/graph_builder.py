import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def _minmax_normalize_columns(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array for min-max normalization, got shape={x.shape}")
    col_min = np.min(x, axis=0, keepdims=True)
    col_max = np.max(x, axis=0, keepdims=True)
    denom = col_max - col_min
    denom[denom == 0] = 1.0
    return ((x - col_min) / denom).astype(np.float32)


def parse_linestring_endpoint(wkt: str) -> Tuple[float, float]:
    """Return (lon, lat) of the end point of a WKT LINESTRING."""
    # Expected format: "LINESTRING (lon lat, lon lat, ...)"
    coords_str = wkt.replace("LINESTRING (", "").replace(")", "")
    coords = [tuple(map(float, c.split())) for c in coords_str.split(", ")]
    lon, lat = coords[-1]
    return lon, lat


@dataclass
class RoadGeoNormalizer:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    length_mean: float
    length_std: float

    def transform(self, lon: np.ndarray, lat: np.ndarray, length: np.ndarray) -> np.ndarray:
        lon_den = (self.lon_max - self.lon_min) if (self.lon_max - self.lon_min) != 0 else 1.0
        lat_den = (self.lat_max - self.lat_min) if (self.lat_max - self.lat_min) != 0 else 1.0

        lon_norm = (lon - self.lon_min) / lon_den
        lat_norm = (lat - self.lat_min) / lat_den

        length_log = np.log1p(np.maximum(length, 0.0))
        length_den = self.length_std if self.length_std != 0 else 1.0
        length_norm = (length_log - self.length_mean) / length_den

        return np.stack([lon_norm, lat_norm, length_norm], axis=-1).astype(np.float32)


def compute_road_geo_normalizer(
    edges_df: pd.DataFrame,
    *,
    boundary_json_path: Optional[Path] = None,
) -> Tuple[RoadGeoNormalizer, np.ndarray]:
    """Compute normalization stats and return (normalizer, raw_geo_features).

    raw_geo_features is [N,3] = [end_lon, end_lat, length].
    """
    endpoints = edges_df["wkt"].apply(parse_linestring_endpoint)
    end_lon = np.array([p[0] for p in endpoints], dtype=np.float32)
    end_lat = np.array([p[1] for p in endpoints], dtype=np.float32)
    length = edges_df["length"].astype(np.float32).to_numpy()

    if boundary_json_path is not None and boundary_json_path.exists():
        boundary = json.loads(boundary_json_path.read_text(encoding="utf-8"))
        lat_min = float(boundary["lati_min"])
        lat_max = float(boundary["lati_max"])
        lon_min = float(boundary["long_min"])
        lon_max = float(boundary["long_max"])
    else:
        lon_min = float(np.min(end_lon))
        lon_max = float(np.max(end_lon))
        lat_min = float(np.min(end_lat))
        lat_max = float(np.max(end_lat))

    length_log = np.log1p(np.maximum(length, 0.0)).astype(np.float32)
    length_mean = float(np.mean(length_log))
    length_std = float(np.std(length_log))

    normalizer = RoadGeoNormalizer(
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        length_mean=length_mean,
        length_std=length_std,
    )
    raw = np.stack([end_lon, end_lat, length], axis=-1).astype(np.float32)
    return normalizer, raw


def load_road_geo_features(
    road_network_dir: str | Path,
    *,
    use_boundary_json_if_exists: bool = True,
) -> Tuple[torch.Tensor, RoadGeoNormalizer, np.ndarray, pd.DataFrame]:
    """Load edges.csv and return:

    - geo_x: torch.FloatTensor [N,3] normalized geo features
    - normalizer: RoadGeoNormalizer
    - fids: np.ndarray [N] the fid values aligned to rows
    - edges_df: pd.DataFrame the raw edges table

    Assumes fids are already 0..N-1 (as produced by the data preprocessing notebook, Step 5).
    """
    road_network_dir = Path(road_network_dir)
    edges_path = road_network_dir / "edges.csv"
    edges = pd.read_csv(edges_path)

    fids = edges["fid"].astype(int).to_numpy()

    boundary_path = road_network_dir / "boundary.json"
    normalizer, raw = compute_road_geo_normalizer(
        edges,
        boundary_json_path=(boundary_path if use_boundary_json_if_exists else None),
    )

    geo_norm = normalizer.transform(raw[:, 0], raw[:, 1], raw[:, 2])
    return torch.from_numpy(geo_norm), normalizer, fids, edges


def build_poi_vocab_from_stats(roads: Sequence[dict]) -> List[str]:
    cats = set()
    for r in roads:
        poi_counts = r.get("poi_counts", {}) or {}
        cats.update(poi_counts.keys())
    return sorted(cats)


def load_poi_features_from_stats_json(
    road_poi_stats_path: str | Path,
    *,
    vocab: Optional[Sequence[str]] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """Load POI count vectors.

    Returns:
    - poi_x: torch.FloatTensor [N, C]
    - vocab: list[str] categories used as columns

    Notes:
    - If vocab is provided, unseen categories are ignored to keep feature dim stable.
    - Assumes each entry has 'fid' matching node id.
    """
    road_poi_stats_path = Path(road_poi_stats_path)
    data = json.loads(road_poi_stats_path.read_text(encoding="utf-8"))
    roads = data["roads"]

    if vocab is None:
        vocab = build_poi_vocab_from_stats(roads)

    vocab = list(vocab)
    cat2idx = {c: i for i, c in enumerate(vocab)}

    # Determine N by max fid (assuming contiguous 0..N-1)
    max_fid = max(int(r["fid"]) for r in roads) if roads else -1
    n = max_fid + 1

    poi = np.zeros((n, len(vocab)), dtype=np.float32)
    for r in roads:
        fid = int(r["fid"])
        counts: Dict[str, int] = r.get("poi_counts", {}) or {}
        for cat, cnt in counts.items():
            idx = cat2idx.get(cat)
            if idx is None:
                continue
            poi[fid, idx] = float(cnt)

    poi = np.log1p(poi)
    poi = _minmax_normalize_columns(poi)

    return torch.from_numpy(poi), vocab


def _file_signature(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {
            "path": str(p),
            "exists": False,
            "mtime_ns": None,
            "size": None,
        }
    stat = p.stat()
    return {
        "path": str(p.resolve()),
        "exists": True,
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }


def _json_stable_dumps(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _graph_cache_dir(repo_root: Path) -> Path:
    return repo_root / "checkpoints" / "graph_cache"


def _graph_cache_payload(
    *,
    city: str,
    data_root: Path,
    poi_vocab: Optional[Sequence[str]],
    min_unique_len: int,
    poi_stats_filename: str,
    include_node_heat: bool,
    node_heat_mode: str,
    include_edge_heat: bool,
    edge_heat_norm: str,
    topology_make_undirected: bool,
    road_network_dir: Path,
    traj_dir: Path,
) -> dict:
    traj_files = _iter_city_npy_paths(traj_dir)
    return {
        "city": str(city),
        "data_root": str(data_root.resolve()),
        "poi_vocab": list(poi_vocab) if poi_vocab is not None else None,
        "min_unique_len": int(min_unique_len),
        "poi_stats_filename": str(poi_stats_filename),
        "include_node_heat": bool(include_node_heat),
        "node_heat_mode": str(node_heat_mode),
        "include_edge_heat": bool(include_edge_heat),
        "edge_heat_norm": str(edge_heat_norm),
        "topology_make_undirected": bool(topology_make_undirected),
        "files": {
            "edges_csv": _file_signature(road_network_dir / "edges.csv"),
            "boundary_json": _file_signature(road_network_dir / "boundary.json"),
            "poi_stats_json": _file_signature(road_network_dir / poi_stats_filename),
            "trajectory_npys": [_file_signature(p) for p in traj_files],
        },
    }


def _graph_cache_path(cache_payload: dict, *, repo_root: Path) -> Path:
    digest = hashlib.sha256(_json_stable_dumps(cache_payload).encode("utf-8")).hexdigest()
    return _graph_cache_dir(repo_root) / f"{digest}.pt"


def _trajectory_total_duration_seconds(meta) -> float:
    if meta is None:
        raise ValueError("Trajectory metadata is missing; cannot read total_duration from meta[4]")
    if len(meta) <= 4:
        raise ValueError("Trajectory metadata missing total_duration at meta[4]")
    try:
        total_duration = float(meta[4])
    except Exception as exc:
        raise ValueError(f"Failed to parse total_duration from meta[4]: {meta[4]!r}") from exc
    if total_duration < 0:
        raise ValueError(f"Trajectory total_duration must be non-negative, got {total_duration}")
    return total_duration


def _iter_city_npy_paths(city_traj_dir: str | Path) -> list[Path]:
    city_traj_dir = Path(city_traj_dir)
    return sorted(city_traj_dir.glob("*.npy"))


def iter_city_trajectories(
    city_traj_dir: str | Path,
    *,
    min_unique_len: int = 5,
    npy_paths: Optional[Sequence[str | Path]] = None,
) -> Iterable[List[int]]:
    """Yield road-id sequences from city .npy files.

    Applies:
    - filter p[2] >= 0
    - consecutive dedup
    - length >= min_unique_len
    """
    from utils.sampling import get_unique_segments

    city_traj_dir = Path(city_traj_dir)
    paths = [Path(p) for p in npy_paths] if npy_paths is not None else _iter_city_npy_paths(city_traj_dir)
    for npy_path in paths:
        arr = np.load(npy_path, allow_pickle=True)
        for traj in arr:
            gps_points = traj[1]
            road_ids = [int(rid) for rid in gps_points if rid >= 0]
            uniq = get_unique_segments(road_ids)
            if len(uniq) >= min_unique_len:
                yield uniq


def iter_city_trajectories_with_total_duration(
    city_traj_dir: str | Path,
    *,
    min_unique_len: int = 5,
    npy_paths: Optional[Sequence[str | Path]] = None,
) -> Iterable[Tuple[List[int], float]]:
    """Yield (road_ids, total_duration_seconds) from city .npy files."""
    from utils.sampling import get_unique_segments

    city_traj_dir = Path(city_traj_dir)
    paths = [Path(p) for p in npy_paths] if npy_paths is not None else _iter_city_npy_paths(city_traj_dir)
    for npy_path in paths:
        arr = np.load(npy_path, allow_pickle=True)
        for traj in arr:
            try:
                meta = traj[0]
                gps_points = traj[1]
            except Exception:
                continue

            try:
                total_duration = _trajectory_total_duration_seconds(meta)
            except ValueError as exc:
                raise ValueError(f"{npy_path}: invalid trajectory total_duration metadata") from exc

            road_ids = [int(rid) for rid in gps_points if rid >= 0]
            uniq = get_unique_segments(road_ids)
            if len(uniq) >= min_unique_len:
                yield uniq, float(total_duration)


def iter_city_trajectories_with_depart_time(
    city_traj_dir: str | Path,
    *,
    min_unique_len: int = 5,
    meta_time_index: int = 3,
    npy_paths: Optional[Sequence[str | Path]] = None,
) -> Iterable[Tuple[List[int], int]]:
    """Yield (road_ids, depart_time) from city .npy files.

    depart_time is extracted from traj[0][meta_time_index] (default meta[3]).

    Applies the same road-id preprocessing as iter_city_trajectories:
    - filter p[2] >= 0
    - consecutive dedup
    - length >= min_unique_len

    Notes:
        If meta is missing or depart_time cannot be parsed as int, the trajectory
        is skipped.
    """
    from utils.sampling import get_unique_segments

    city_traj_dir = Path(city_traj_dir)
    paths = [Path(p) for p in npy_paths] if npy_paths is not None else _iter_city_npy_paths(city_traj_dir)
    for npy_path in paths:
        arr = np.load(npy_path, allow_pickle=True)
        for traj in arr:
            meta = traj[0]
            if meta is None or len(meta) <= int(meta_time_index):
                continue
            try:
                depart_time = int(meta[int(meta_time_index)])
            except Exception:
                continue

            gps_points = traj[1]
            road_ids = [int(rid) for rid in gps_points if rid >= 0]
            uniq = get_unique_segments(road_ids)
            if len(uniq) >= min_unique_len:
                yield uniq, depart_time


def iter_city_trajectories_with_depart_time_and_total_duration(
    city_traj_dir: str | Path,
    *,
    min_unique_len: int = 5,
    meta_time_index: int = 3,
    npy_paths: Optional[Sequence[str | Path]] = None,
) -> Iterable[Tuple[List[int], int, float]]:
    """Yield (road_ids, depart_time, total_duration_seconds) from city .npy files."""
    from utils.sampling import get_unique_segments

    city_traj_dir = Path(city_traj_dir)
    paths = [Path(p) for p in npy_paths] if npy_paths is not None else _iter_city_npy_paths(city_traj_dir)
    for npy_path in paths:
        arr = np.load(npy_path, allow_pickle=True)
        for traj in arr:
            try:
                meta = traj[0]
                gps_points = traj[1]
            except Exception:
                continue
            if meta is None or len(meta) <= int(meta_time_index):
                continue
            try:
                depart_time = int(meta[int(meta_time_index)])
            except Exception:
                continue

            try:
                total_duration = _trajectory_total_duration_seconds(meta)
            except ValueError as exc:
                raise ValueError(f"{npy_path}: invalid trajectory total_duration metadata") from exc

            road_ids = [int(rid) for rid in gps_points if rid >= 0]
            uniq = get_unique_segments(road_ids)
            if len(uniq) >= min_unique_len:
                yield uniq, depart_time, float(total_duration)


def build_topology_edges_from_edges_df(
    edges_df: pd.DataFrame,
    *,
    fid_col: str = "fid",
    u_col: str = "u",
    v_col: str = "v",
    make_undirected: bool = False,
) -> torch.Tensor:
    """Build directed topology edges between road segments.

    We treat each row (fid) as a directed road segment from intersection ``u`` to
    intersection ``v``. A legal transition ``a -> b`` exists iff the end node of
    ``a`` matches the start node of ``b``; i.e. ``a.v == b.u``.

    Returns:
      edge_index: LongTensor [2, E_topo]

    Notes:
    - We exclude self-loops.
    - make_undirected=True additionally adds the reverse edge for every directed
      transition. This is only for explicit compatibility needs; the default is
      strict directed semantics.
    """
    if edges_df.empty:
        return torch.empty((2, 0), dtype=torch.long)

    fids = edges_df[fid_col].astype(int).to_numpy()
    u = edges_df[u_col].astype(np.int64).to_numpy()
    v = edges_df[v_col].astype(np.int64).to_numpy()

    outgoing_by_start: Dict[int, List[int]] = {}
    incoming_by_end: Dict[int, List[int]] = {}
    for fid, uu, vv in zip(fids, u, v):
        outgoing_by_start.setdefault(int(uu), []).append(int(fid))
        incoming_by_end.setdefault(int(vv), []).append(int(fid))

    edge_pairs: set[Tuple[int, int]] = set()
    shared_nodes = set(incoming_by_end.keys()) & set(outgoing_by_start.keys())
    for node_id in shared_nodes:
        src_segs = incoming_by_end[node_id]
        dst_segs = outgoing_by_start[node_id]
        for a in src_segs:
            for b in dst_segs:
                if a == b:
                    continue
                edge_pairs.add((a, b))
                if make_undirected:
                    edge_pairs.add((b, a))

    if not edge_pairs:
        return torch.empty((2, 0), dtype=torch.long)

    edges = np.asarray(sorted(edge_pairs), dtype=np.int64)
    return torch.from_numpy(edges.T).long()



def align_edge_attr_to_edge_index(
    *,
    heat_counter: Counter[Tuple[int, int]],
    edge_index: torch.Tensor,
    normalize: str = "log1p_minmax",
) -> torch.Tensor:
    """Create edge_attr aligned to a given edge_index using heat transition counts.

    For each edge (a->b) in edge_index, we set edge_attr to a transformed value of
    count(a->b). If no transition observed, count=0.

    normalize:
      - "log1p_minmax": min-max normalize log1p(count)
      - "log1p_div_max": log1p(count) / max_log1p

    Returns:
      edge_attr: FloatTensor [E,1]
    """
    if edge_index.numel() == 0:
        return torch.empty((0, 1), dtype=torch.float32)

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    counts = np.array([float(heat_counter.get((int(a), int(b)), 0)) for a, b in zip(src, dst)], dtype=np.float32)
    vals = np.log1p(counts).astype(np.float32)

    if normalize == "log1p_minmax":
        vmin = float(np.min(vals)) if vals.size else 0.0
        vmax = float(np.max(vals)) if vals.size else 1.0
        denom = vmax - vmin
        if denom == 0.0:
            vals = np.zeros_like(vals, dtype=np.float32)
        else:
            vals = ((vals - vmin) / denom).astype(np.float32)
    elif normalize == "log1p_div_max":
        mx = float(np.max(vals)) if vals.size else 1.0
        if mx == 0:
            mx = 1.0
        vals = (vals / mx).astype(np.float32)
    else:
        raise ValueError(f"Unsupported normalize mode: {normalize}")

    return torch.from_numpy(vals).view(-1, 1)


def build_node_heat_features_from_edge_counter(
    counter: Counter[Tuple[int, int]],
    *,
    num_nodes: int,
    mode: str = "log1p_in_out",
) -> np.ndarray:
    """Build node-level heat features from transition counter.

    Returns x_heat: [N, D] depending on mode.

    Supported modes:
    - "log1p_in_out": D=2, [log1p(out_strength), log1p(in_strength)]

    Note: Heat is retained only as node attributes; edge-level heat weights are
    removed from the repo-wide graph representation.
    """
    if mode != "log1p_in_out":
        raise ValueError(f"Unsupported node heat mode: {mode}")

    out_strength = np.zeros((num_nodes,), dtype=np.float32)
    in_strength = np.zeros((num_nodes,), dtype=np.float32)

    for (a, b), c in counter.items():
        if 0 <= a < num_nodes:
            out_strength[a] += float(c)
        if 0 <= b < num_nodes:
            in_strength[b] += float(c)

    x0 = np.log1p(out_strength)
    x1 = np.log1p(in_strength)
    x = np.stack([x0, x1], axis=-1).astype(np.float32)
    return _minmax_normalize_columns(x)


def build_heat_edge_counter_from_trajectories(
    trajectories: Iterable[Sequence[int]],
    *,
    num_nodes: int,
) -> Counter[Tuple[int, int]]:
    counter: Counter[Tuple[int, int]] = Counter()
    for traj in trajectories:
        for i in range(len(traj) - 1):
            a = int(traj[i])
            b = int(traj[i + 1])
            if 0 <= a < num_nodes and 0 <= b < num_nodes:
                counter[(a, b)] += 1
    return counter


def build_heat_edges_from_trajectories(
    trajectories: Iterable[Sequence[int]],
    *,
    num_nodes: int,
    return_node_heat: bool = False,
    node_heat_mode: str = "log1p_in_out",
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build trajectory-derived sparse edges plus optional node heat features.

    This helper is kept for analysis utilities. The main city graph construction
    path no longer uses trajectory heat as edge weights.

    If return_node_heat is True, also returns:
    - x_heat: torch.FloatTensor [N,2] (log1p+minmax out/in strengths)
    """
    counter = build_heat_edge_counter_from_trajectories(trajectories, num_nodes=num_nodes)

    if not counter:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        if return_node_heat:
            x_heat = torch.zeros((num_nodes, 2), dtype=torch.float32)
            return edge_index, edge_attr, x_heat
        return edge_index, edge_attr

    edges = np.array(list(counter.keys()), dtype=np.int64)
    counts = np.array(list(counter.values()), dtype=np.float32)

    edge_index = torch.from_numpy(edges.T).long()  # [2,E]

    edge_attr = np.log1p(counts)
    mx = float(np.max(edge_attr)) if edge_attr.size else 1.0
    if mx == 0:
        mx = 1.0
    edge_attr = (edge_attr / mx).astype(np.float32)
    edge_attr = torch.from_numpy(edge_attr).view(-1, 1)

    if return_node_heat:
        x_heat_np = build_node_heat_features_from_edge_counter(
            counter, num_nodes=num_nodes, mode=node_heat_mode
        )
        x_heat = torch.from_numpy(x_heat_np)
        return edge_index, edge_attr, x_heat

    return edge_index, edge_attr


def build_city_pyg_graph(
    *,
    city: str,
    data_root: str | Path = "data",
    poi_vocab: Optional[Sequence[str]] = None,
    min_unique_len: int = 5,
    poi_stats_filename: str = "road_poi_stats.json",
    include_node_heat: bool = True,
    node_heat_mode: str = "log1p_in_out",
    topology_make_undirected: bool = False,
    include_edge_heat: bool = True,
    edge_heat_norm: str = "log1p_minmax",
    traj_dir: str | Path | None = None,
) -> Tuple[Data, dict]:
    """Build a full-city PyG graph for training.

    Returns:
    - data: PyG Data(x_geo, x_poi, x_heat, edge_index, edge_attr)
    - meta: dict with normalizer and vocab

    POI stats path is expected at: data/{city}_road_network/road_poi_stats.json

    Args:
        include_node_heat: whether to add node-level heat features x_heat.
        node_heat_mode: aggregation mode for node-level heat.
        include_edge_heat: whether to add edge-level heat features edge_attr.
        edge_heat_norm: normalization mode for edge-level heat.
    """
    data_root = Path(data_root)
    repo_root = Path(__file__).resolve().parents[1]

    road_network_dir = data_root / f"{city}_road_network"
    traj_dir_path = Path(traj_dir) if traj_dir is not None else (data_root / city)
    cache_payload = _graph_cache_payload(
        city=city,
        data_root=data_root,
        poi_vocab=poi_vocab,
        min_unique_len=min_unique_len,
        poi_stats_filename=poi_stats_filename,
        include_node_heat=include_node_heat,
        node_heat_mode=node_heat_mode,
        include_edge_heat=include_edge_heat,
        edge_heat_norm=edge_heat_norm,
        topology_make_undirected=topology_make_undirected,
        road_network_dir=road_network_dir,
        traj_dir=traj_dir_path,
    )
    cache_path = _graph_cache_path(cache_payload, repo_root=repo_root)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        data = Data(
            x_geo=cached["x_geo"],
            x_poi=cached["x_poi"],
            x_heat=cached["x_heat"],
            edge_index=cached["edge_index"],
            edge_attr=cached["edge_attr"],
            num_nodes=int(cached["num_nodes"]),
        )
        return data, cached["meta"]

    geo_x, geo_norm, fids, edges_df = load_road_geo_features(road_network_dir)

    num_nodes = int(geo_x.shape[0])
    # Basic sanity: if fids are not contiguous, downstream will break.
    if len(fids) and (np.min(fids) != 0 or np.max(fids) != num_nodes - 1):
        raise ValueError(
            f"Expected fid to be contiguous 0..N-1, got min={np.min(fids)} max={np.max(fids)} N={num_nodes}"
        )

    # Directed transition graph between road segments under one-way semantics.
    topo_edge_index = build_topology_edges_from_edges_df(
        edges_df,
        make_undirected=bool(topology_make_undirected),
    )

    poi_stats_path = road_network_dir / poi_stats_filename
    if poi_stats_path.exists():
        poi_x, vocab = load_poi_features_from_stats_json(poi_stats_path, vocab=poi_vocab)
        # If stats JSON contains fewer/more roads, align by trunc/pad
        if poi_x.shape[0] < num_nodes:
            pad = torch.zeros((num_nodes - poi_x.shape[0], poi_x.shape[1]), dtype=poi_x.dtype)
            poi_x = torch.cat([poi_x, pad], dim=0)
        elif poi_x.shape[0] > num_nodes:
            poi_x = poi_x[:num_nodes]
    else:
        vocab = list(poi_vocab) if poi_vocab is not None else []
        poi_x = torch.zeros((num_nodes, len(vocab)), dtype=torch.float32)

    edge_index = topo_edge_index
    heat_counter: Counter[Tuple[int, int]] | None = None
    if include_node_heat or include_edge_heat:
        trajs = iter_city_trajectories(traj_dir_path, min_unique_len=min_unique_len)
        heat_counter = build_heat_edge_counter_from_trajectories(trajs, num_nodes=num_nodes)

    if include_edge_heat:
        if heat_counter is None:
            heat_counter = Counter()
        edge_attr = align_edge_attr_to_edge_index(
            heat_counter=heat_counter,
            edge_index=edge_index,
            normalize=edge_heat_norm,
        )
    else:
        edge_attr = torch.empty((0, 1), dtype=torch.float32)

    if include_node_heat:
        if heat_counter is None:
            heat_counter = Counter()
        x_heat_np = build_node_heat_features_from_edge_counter(
            heat_counter, num_nodes=num_nodes, mode=node_heat_mode
        )
        x_heat = torch.from_numpy(x_heat_np)
    else:
        x_heat = None

    data = Data(
        x_geo=geo_x,
        x_poi=poi_x,
        x_heat=x_heat,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes,
    )

    meta = {
        "city": city,
        "geo_normalizer": geo_norm,
        "poi_vocab": vocab,
        "include_node_heat": include_node_heat,
        "node_heat_normalization": ("log1p_minmax_per_dim" if include_node_heat else None),
        "include_edge_heat": include_edge_heat,
        "edge_heat_normalization": (edge_heat_norm if include_edge_heat else None),
        "edge_heat_removed": (not include_edge_heat),
        "poi_normalization": "log1p_minmax_per_dim",
        "topology_make_undirected": bool(topology_make_undirected),
        "graph_cache_path": str(cache_path),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "x_geo": geo_x.cpu(),
            "x_poi": poi_x.cpu(),
            "x_heat": (x_heat.cpu() if x_heat is not None else None),
            "edge_index": edge_index.cpu(),
            "edge_attr": edge_attr.cpu(),
            "num_nodes": int(num_nodes),
            "meta": meta,
            "cache_payload": cache_payload,
        },
        cache_path,
    )

    return data, meta


