"""Batch generate trajectories conditioned on real trajectory metadata.

This script extracts per-trajectory conditions from either raw GPS `.npy` files
or road-progress `.npy` files and generates new trajectories.

Supported generation modes:
- model (default): use a trained graph-constrained AR Transformer.
- shortest_path: ignore the model and reconstruct a graph shortest path from
  each sample's start road to destination road.

Supported output formats:
- road_ids (default): each generated sample is a `list[int]` road_id sequence.
- dataset: each generated sample is `[meta, points]`, where `points[:, 0]`
  stores endpoint percentages and `points[:, 1]` stores road_id. Only the
  first row stores the start percentage, only the last row stores the end
  percentage, and intermediate percentages are 0.

Generation conditions per input trajectory:
- raw GPS input `[meta, gps_points]`
  - start road = first unique road id in the trajectory
  - destination road = last unique road id in the trajectory
  - depart time = `meta[3]` by default
  - total duration = sum of `gps_points[:,3]`
  - endpoint percentages are projected from the first/last GPS point when
    `--output-format=dataset`
- road-progress input `[meta, roads_unique, s_seq, t_seq]`
  - start road = first value in `roads_unique`
  - destination road = last value in `roads_unique`
  - depart time = `meta[3]` by default
  - total duration = sum of `t_seq`
  - endpoint percentages come from `s_seq[0]` and `s_seq[-1]`
- model mode decode budget = fixed `--max-steps` (default: 100)

The script also reports generation success rate, where success means the
generated trajectory's final road id equals the target destination road.

By default, each GPU worker writes a shard file:
  <output_dir>/generated_worker_<id>.npy

Optionally, you can merge shards into a single output file via --output-file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

from experiments.build_road_progress_dataset import _extract_lonlat_road_ids_and_time, _segment_indices_from_road_ids
from models.traj_ar_transformer import GraphConstrainedARTransformer, TrajARTransformerConfig
from utils.graph_builder import build_city_pyg_graph
from utils.neighbor_table import build_padded_neighbors
from utils.reachability import load_reachability_table, shortest_path_between_nodes
from utils.sampling import get_unique_segments
from utils.wkt_geometry import PolylineIndex, build_fid_to_polyline_index, project_point_to_polyline_alpha


def _pick_device(device_id: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def _load_z(emb_path: str) -> torch.Tensor:
    obj = torch.load(emb_path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "z" in obj:
        z = obj["z"]
    else:
        z = obj
    if not isinstance(z, torch.Tensor):
        raise TypeError(f"Expected embeddings to be a torch.Tensor, got {type(z)}")
    return z.detach().float().cpu()


def _load_shortest_dist_table(path: str) -> torch.Tensor:
    return torch.from_numpy(load_reachability_table(path))


MetaItem = Any
MAX_ENDPOINT_PERCENTAGE = 1.0 - 1e-4


def _resolve_visible_history_len(cli_value: int, ckpt_args: object) -> tuple[int | None, str]:
    if int(cli_value) > 0:
        return int(cli_value), "cli"
    if isinstance(ckpt_args, dict):
        val = ckpt_args.get("causal_window")
        if val is not None and int(val) > 0:
            return int(val), "checkpoint_causal_window"
        for key in ("max_traj_len", "segment_len"):
            val = ckpt_args.get(key)
            if val is not None and int(val) > 0:
                return int(val), f"checkpoint_{key}"
    return None, "full_prefix"


class MetadataItem(NamedTuple):
    meta: MetaItem
    start_road: int
    destination_road: int
    length: int
    depart_time: int | None
    total_duration: float | None
    start_percentage: float
    end_percentage: float


def _clip_endpoint_percentage(value: float, *, fallback: float) -> float:
    try:
        return float(np.clip(float(value), 0.0, MAX_ENDPOINT_PERCENTAGE))
    except Exception:
        return float(fallback)


def _default_endpoint_percentages() -> tuple[float, float]:
    return 0.0, MAX_ENDPOINT_PERCENTAGE


def _total_duration_from_points(gps_points: Any) -> float | None:
    pts = np.asarray(gps_points)
    if pts.ndim != 2 or pts.shape[1] < 4:
        return None
    try:
        vals = np.asarray(pts[:, 3], dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if len(vals) == 0:
        return None
    return float(vals.sum())


def _endpoint_percentages_from_points(
    gps_points: Any,
    fid2poly: dict[int, PolylineIndex] | None,
) -> tuple[float, float]:
    start_percentage, end_percentage = _default_endpoint_percentages()
    if not fid2poly:
        return start_percentage, end_percentage

    try:
        lon, lat, road_ids, _ = _extract_lonlat_road_ids_and_time(gps_points)
        if len(road_ids) == 0 or len(lon) == 0 or len(lat) == 0:
            return start_percentage, end_percentage

        roads_unique, _ = _segment_indices_from_road_ids(road_ids)
        if len(roads_unique) == 0:
            return start_percentage, end_percentage

        start_poly = fid2poly.get(int(roads_unique[0]))
        end_poly = fid2poly.get(int(roads_unique[-1]))

        if start_poly is not None:
            start_percentage = _clip_endpoint_percentage(
                project_point_to_polyline_alpha(np.asarray([lon[0], lat[0]], dtype=np.float32), start_poly),
                fallback=0.0,
            )
        if end_poly is not None:
            end_percentage = _clip_endpoint_percentage(
                project_point_to_polyline_alpha(np.asarray([lon[-1], lat[-1]], dtype=np.float32), end_poly),
                fallback=MAX_ENDPOINT_PERCENTAGE,
            )
    except Exception:
        return _default_endpoint_percentages()

    return start_percentage, end_percentage


def _total_duration_from_meta(meta: Any) -> float | None:
    if meta is None:
        return None
    candidates: list[float | None] = []
    try:
        candidates.append(float(meta[4]) if len(meta) > 4 else None)
    except Exception:
        candidates.append(None)
    try:
        candidates.append(float(meta[2]) if len(meta) > 2 else None)
    except Exception:
        candidates.append(None)
    for value in candidates:
        if value is None:
            continue
        if np.isfinite(value) and value > 0:
            return float(value)
    return None


def _metadata_from_road_progress_item(
    traj: Any,
    meta_time_index: int,
) -> MetadataItem | None:
    try:
        meta = traj[0]
        roads_unique = np.asarray(traj[1], dtype=np.int32).reshape(-1)
        s_seq = np.asarray(traj[2], dtype=np.float32).reshape(-1)
        t_seq = np.asarray(traj[3], dtype=np.float32).reshape(-1)
    except Exception:
        return None

    if len(roads_unique) < 2 or len(s_seq) == 0:
        return None

    depart_time = None
    if meta is not None and len(meta) > meta_time_index:
        try:
            depart_time = int(meta[meta_time_index])
        except Exception:
            depart_time = None

    total_duration = None
    if len(t_seq) > 0:
        total_duration = float(t_seq.sum())
    if total_duration is None or not np.isfinite(total_duration) or total_duration <= 0:
        total_duration = _total_duration_from_meta(meta)

    start_percentage = _clip_endpoint_percentage(float(s_seq[0]), fallback=0.0)
    end_percentage = _clip_endpoint_percentage(
        float(s_seq[-1]) - float(max(len(roads_unique) - 1, 0)),
        fallback=MAX_ENDPOINT_PERCENTAGE,
    )

    return MetadataItem(
        meta=meta,
        start_road=int(roads_unique[0]),
        destination_road=int(roads_unique[-1]),
        length=int(len(roads_unique)),
        depart_time=depart_time,
        total_duration=total_duration,
        start_percentage=start_percentage,
        end_percentage=end_percentage,
    )


def _metadata_from_gps_item(
    traj: Any,
    meta_time_index: int,
    fid2poly: dict[int, PolylineIndex] | None,
) -> MetadataItem | None:
    try:
        meta = traj[0]
        depart_time = None
        if meta is not None and len(meta) > meta_time_index:
            try:
                depart_time = int(meta[meta_time_index])
            except Exception:
                depart_time = None

        gps_points = traj[1]
        road_ids = [int(p[2]) for p in gps_points if p[2] >= 0]

        if len(road_ids) < 2:
            return None

        unique_ids = get_unique_segments(road_ids)
        if len(unique_ids) < 2:
            return None

        start_road = int(unique_ids[0])
        destination_road = int(unique_ids[-1])
        length = int(len(unique_ids))
        total_duration = _total_duration_from_points(gps_points)
        start_percentage, end_percentage = _endpoint_percentages_from_points(gps_points, fid2poly)

        return MetadataItem(
            meta=meta,
            start_road=start_road,
            destination_road=destination_road,
            length=length,
            depart_time=depart_time,
            total_duration=total_duration,
            start_percentage=start_percentage,
            end_percentage=end_percentage,
        )
    except Exception:
        return None


def extract_trajectory_metadata(
    npy_path: str,
    meta_time_index: int = 3,
    fid2poly: dict[int, PolylineIndex] | None = None,
) -> List[MetadataItem]:
    """Extract trajectory-level generation conditions from a .npy file."""
    arr = np.load(npy_path, allow_pickle=True)
    metadata: List[MetadataItem] = []

    for traj in arr:
        item = _metadata_from_road_progress_item(traj, meta_time_index=meta_time_index)
        if item is None:
            item = _metadata_from_gps_item(traj, meta_time_index=meta_time_index, fid2poly=fid2poly)
        if item is not None:
            metadata.append(item)

    return metadata


class TrajectoryMetadataDataset(Dataset):
    """Dataset that yields metadata tuples for conditioned trajectory generation."""

    def __init__(self, metadata: List[MetadataItem]):
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> MetadataItem:
        return self.metadata[idx]


def collate_metadata(batch: List[MetadataItem]):
    metas = [x.meta for x in batch]
    start_roads = torch.tensor([x.start_road for x in batch], dtype=torch.long)
    destination_roads = torch.tensor([x.destination_road for x in batch], dtype=torch.long)
    depart_times = torch.tensor([0 if x.depart_time is None else int(x.depart_time) for x in batch], dtype=torch.long)
    total_durations = torch.tensor([0.0 if x.total_duration is None else float(x.total_duration) for x in batch], dtype=torch.float32)
    start_percentages = torch.tensor([float(x.start_percentage) for x in batch], dtype=torch.float32)
    end_percentages = torch.tensor([float(x.end_percentage) for x in batch], dtype=torch.float32)
    return metas, start_roads, destination_roads, depart_times, total_durations, start_percentages, end_percentages


def _wrap_dataset_like(meta: Any, traj_road_ids: List[int], start_percentage: float, end_percentage: float) -> Any:
    """Wrap a generated road-id trajectory into [meta, points].

    points[:,0] stores s = i + alpha (segment index + percentage).
    points[:,1] stores road_id.
    """
    from utils.sampling import get_unique_segments

    t = int(len(traj_road_ids))
    points = np.zeros((t, 2), dtype=np.float32)
    if t > 0:
        points[:, 1] = np.asarray(traj_road_ids, dtype=np.float32)
        roads_unique = get_unique_segments(traj_road_ids)
        num_segments = len(roads_unique)
        # s_start = 0 + start_alpha
        points[0, 0] = 0.0 + _clip_endpoint_percentage(start_percentage, fallback=0.0)
        # s_end = (num_segments - 1) + end_alpha
        points[-1, 0] = float(max(num_segments - 1, 0)) + _clip_endpoint_percentage(end_percentage, fallback=MAX_ENDPOINT_PERCENTAGE)
    return [meta, points]


def generate_batch(
    model: GraphConstrainedARTransformer,
    metas: list,
    start_roads: torch.Tensor,
    destination_roads: torch.Tensor,
    depart_times: torch.Tensor,
    total_durations: torch.Tensor,
    start_percentages: torch.Tensor,
    end_percentages: torch.Tensor,
    z: torch.Tensor,
    neighbors: torch.Tensor,
    neighbor_mask: torch.Tensor,
    geo_x: torch.Tensor | None,
    shortest_dist_table: torch.Tensor | None,
    top_k: int,
    temperature: float,
    greedy: bool,
    max_steps: int,
    device: torch.device,
    output_format: str,
    stop_at_destination: bool,
    visible_history_len: int | None,
) -> tuple[List[Any], int, int, int]:
    start_roads = start_roads.to(device)
    destination_roads = destination_roads.to(device)
    depart_times = depart_times.to(device)
    total_durations = total_durations.to(device)
    start_percentages = start_percentages.to(device)
    end_percentages = end_percentages.to(device)

    steps = max(int(max_steps), 1)
    use_override_counting = shortest_dist_table is not None
    _start_road_cond = (start_roads if bool(getattr(model.cfg, "use_start_road_embedding", False)) else None)
    _end_road_cond = (destination_roads if bool(getattr(model.cfg, "use_end_road_embedding", False)) else None)
    if use_override_counting:
        trajs, override_counts = model.generate_batch(
            start_roads=start_roads,
            steps=steps,
            z=z,
            neighbors=neighbors,
            neighbor_mask=neighbor_mask,
            geo_x=geo_x,
            start_road=_start_road_cond,
            end_road=_end_road_cond,
            shortest_dist_table=shortest_dist_table,
            destination_roads=destination_roads,
            top_k=top_k,
            temperature=temperature,
            greedy=greedy,
            depart_time=(depart_times if bool(getattr(model.cfg, "use_depart_time", False)) else None),
            total_duration=(total_durations if bool(getattr(model.cfg, "use_total_duration", False)) else None),
            device=device,
            stop_at_destination=stop_at_destination,
            visible_history_len=visible_history_len,
            return_constraint_override_count=True,
        )
    else:
        trajs = model.generate_batch(
            start_roads=start_roads,
            steps=steps,
            z=z,
            neighbors=neighbors,
            neighbor_mask=neighbor_mask,
            geo_x=geo_x,
            start_road=_start_road_cond,
            end_road=_end_road_cond,
            shortest_dist_table=shortest_dist_table,
            destination_roads=destination_roads,
            top_k=top_k,
            temperature=temperature,
            greedy=greedy,
            depart_time=(depart_times if bool(getattr(model.cfg, "use_depart_time", False)) else None),
            total_duration=(total_durations if bool(getattr(model.cfg, "use_total_duration", False)) else None),
            device=device,
            stop_at_destination=stop_at_destination,
            visible_history_len=visible_history_len,
        )
        import torch as _torch
        override_counts = _torch.zeros(len(trajs), dtype=_torch.long)

    results: List[Any] = []
    success_count = 0
    total_override_count = 0
    total_steps = 0
    for i, traj in enumerate(trajs):
        destination_road = int(destination_roads[i].item())
        traj_ids = [int(x) for x in traj.detach().cpu().tolist()]
        if len(traj_ids) > 0 and traj_ids[-1] == destination_road:
            success_count += 1
        if use_override_counting:
            total_override_count += int(override_counts[i].item())
            total_steps += max(len(traj_ids) - 1, 0)
        if output_format == "dataset":
            results.append(
                _wrap_dataset_like(
                    metas[i],
                    traj_ids,
                    float(start_percentages[i].item()),
                    float(end_percentages[i].item()),
                )
            )
        else:
            results.append(traj_ids)

    return results, success_count, total_override_count, total_steps


def generate_shortest_path_batch(
    metas: list,
    start_roads: torch.Tensor,
    destination_roads: torch.Tensor,
    start_percentages: torch.Tensor,
    end_percentages: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None,
    output_format: str,
) -> tuple[List[Any], int]:
    results: List[Any] = []
    success_count = 0

    start_roads_cpu = start_roads.detach().cpu()
    destination_roads_cpu = destination_roads.detach().cpu()
    start_percentages_cpu = start_percentages.detach().cpu()
    end_percentages_cpu = end_percentages.detach().cpu()

    for i in range(len(metas)):
        start_road = int(start_roads_cpu[i].item())
        destination_road = int(destination_roads_cpu[i].item())
        path_ids, _ = shortest_path_between_nodes(
            edge_index=edge_index,
            num_nodes=num_nodes,
            src=start_road,
            dst=destination_road,
            edge_weight=edge_weight,
        )
        traj_ids = [int(x) for x in path_ids]
        if len(traj_ids) > 0 and traj_ids[-1] == destination_road:
            success_count += 1
        if output_format == "dataset":
            if len(traj_ids) == 0:
                traj_ids = [start_road]
            results.append(
                _wrap_dataset_like(
                    metas[i],
                    traj_ids,
                    float(start_percentages_cpu[i].item()),
                    float(end_percentages_cpu[i].item()),
                )
            )
        else:
            results.append(traj_ids)

    return results, success_count


def _build_edge_weight_from_edges_csv(edge_index: torch.Tensor, edges_csv_path: str, mode: str) -> torch.Tensor | None:
    if str(mode) != "length":
        return None

    import pandas as pd

    edges_df = pd.read_csv(edges_csv_path)
    fid_to_length = dict(zip(edges_df["fid"].astype(int), edges_df["length"].astype(float)))
    src_fids = edge_index[0].detach().cpu().tolist()
    weights = np.array([fid_to_length.get(int(fid), 1.0) for fid in src_fids], dtype=np.float32)
    weights = np.maximum(weights, 1e-3)
    return torch.from_numpy(weights)


def worker_process(
    gpu_id: int,
    metadata_chunk: List[MetadataItem],
    ckpt_path: str,
    emb_path: str,
    city: str,
    data_root: str,
    distance_table_path: str,
    batch_size: int,
    top_k: int,
    temperature: float,
    greedy: bool,
    max_steps: int,
    visible_history_len_cli: int,
    output_dir: Path,
    worker_id: int,
    output_format: str,
    stop_at_destination: bool,
    generation_mode: str,
    shortest_path_mode: str,
    edges_path: str,
):
    device = _pick_device(gpu_id)
    print(f"[Worker {worker_id}] Using device: {device}, processing {len(metadata_chunk)} trajectories")

    data, _ = build_city_pyg_graph(city=city, data_root=data_root, include_node_heat=False, include_edge_heat=False)
    num_nodes = int(data.num_nodes)
    edge_index_cpu = data.edge_index.cpu()
    edge_weight_cpu = _build_edge_weight_from_edges_csv(edge_index_cpu, edges_path, shortest_path_mode) if edges_path else None

    dataset = TrajectoryMetadataDataset(metadata_chunk)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_metadata,
    )

    all_generated: List[Any] = []
    total_success = 0

    if generation_mode == "shortest_path":
        with torch.no_grad():
            for batch_idx, (metas, start_roads, destination_roads, _depart_times, _total_durations, start_percentages, end_percentages) in enumerate(dataloader):
                batch_trajs, batch_success = generate_shortest_path_batch(
                    metas=metas,
                    start_roads=start_roads,
                    destination_roads=destination_roads,
                    start_percentages=start_percentages,
                    end_percentages=end_percentages,
                    edge_index=edge_index_cpu,
                    num_nodes=num_nodes,
                    edge_weight=edge_weight_cpu,
                    output_format=output_format,
                )
                all_generated.extend(batch_trajs)
                total_success += int(batch_success)

                if (batch_idx + 1) % 10 == 0:
                    rate = total_success / max(len(all_generated), 1)
                    print(
                        f"[Worker {worker_id}] Generated {len(all_generated)}/{len(metadata_chunk)} trajectories | "
                        f"success_rate={rate:.4f}"
                    )
    else:
        geo_x = data.x_geo.float().to(device)

        z = _load_z(emb_path).to(device)
        if int(z.shape[0]) != num_nodes:
            raise ValueError(f"Embeddings N={int(z.shape[0])} != graph num_nodes={num_nodes}")

        neighbors, nb_mask = build_padded_neighbors(
            edge_index=data.edge_index,
            num_nodes=num_nodes,
            pad_value=0,
        )
        neighbors = neighbors.to(device)
        nb_mask = nb_mask.to(device)
        shortest_dist_table = _load_shortest_dist_table(distance_table_path).to(device) if distance_table_path else None

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        cfg_dict = ckpt.get("cfg", {})
        cfg = TrajARTransformerConfig(**cfg_dict)
        visible_history_len, visible_history_source = _resolve_visible_history_len(
            int(visible_history_len_cli),
            ckpt.get("args", {}),
        )

        model = GraphConstrainedARTransformer(cfg, pad_id=-1)
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device)
        model.eval()
        if visible_history_len is None:
            print(f"[Worker {worker_id}] visible_history_len: full_prefix")
        else:
            print(f"[Worker {worker_id}] visible_history_len: {visible_history_len} ({visible_history_source})")

        if bool(getattr(cfg, "use_depart_time", False)):
            missing_depart = sum(1 for item in metadata_chunk if item.depart_time is None)
            if missing_depart:
                raise ValueError(f"Checkpoint requires depart_time, but {missing_depart} trajectories in this shard lack usable meta[{3}] timing")
        if bool(getattr(cfg, "use_total_duration", False)):
            missing_duration = sum(1 for item in metadata_chunk if item.total_duration is None)
            if missing_duration:
                raise ValueError(f"Checkpoint requires total_duration, but {missing_duration} trajectories in this shard lack usable gps_points[:,3] timing")

        total_override_count = 0
        total_steps = 0
        with torch.no_grad():
            for batch_idx, (metas, start_roads, destination_roads, depart_times, total_durations, start_percentages, end_percentages) in enumerate(dataloader):
                batch_trajs, batch_success, batch_override_count, batch_steps = generate_batch(
                    model=model,
                    metas=metas,
                    start_roads=start_roads,
                    destination_roads=destination_roads,
                    depart_times=depart_times,
                    total_durations=total_durations,
                    start_percentages=start_percentages,
                    end_percentages=end_percentages,
                    z=z,
                    neighbors=neighbors,
                    neighbor_mask=nb_mask,
                    geo_x=geo_x,
                    shortest_dist_table=shortest_dist_table,
                    top_k=top_k,
                    temperature=temperature,
                    greedy=greedy,
                    max_steps=max_steps,
                    device=device,
                    output_format=output_format,
                    stop_at_destination=stop_at_destination,
                    visible_history_len=visible_history_len,
                )
                all_generated.extend(batch_trajs)
                total_success += int(batch_success)
                total_override_count += int(batch_override_count)
                total_steps += int(batch_steps)

                if (batch_idx + 1) % 10 == 0:
                    rate = total_success / max(len(all_generated), 1)
                    override_rate = total_override_count / max(total_steps, 1)
                    print(
                        f"[Worker {worker_id}] Generated {len(all_generated)}/{len(metadata_chunk)} trajectories | "
                        f"success_rate={rate:.4f} | constraint_override_rate={override_rate:.4f}"
                    )

    output_file = _worker_output_path(output_dir, worker_id)
    np.save(output_file, np.array(all_generated, dtype=object))
    success_rate = total_success / max(len(all_generated), 1)
    print(
        f"[Worker {worker_id}] Saved {len(all_generated)} trajectories to {output_file} | "
        f"success={total_success}/{len(all_generated)} ({success_rate:.4f})"
    )
    if total_steps > 0:
        override_rate = total_override_count / total_steps
        print(
            f"[Worker {worker_id}] Distance-table constraint statistics: "
            f"overrides={total_override_count}/{total_steps} ({override_rate:.4f})"
        )


def _worker_output_path(output_dir: Path, worker_id: int) -> Path:
    return output_dir / f"generated_worker_{worker_id}.npy"


def _merge_worker_outputs(worker_files: List[Path], merged_path: Path) -> int:
    all_items: List[Any] = []
    for worker_file in worker_files:
        items = np.load(worker_file, allow_pickle=True)
        all_items.extend(list(items))
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(merged_path, np.array(all_items, dtype=object))
    return len(all_items)


def _run_generation_for_paths(
    *,
    input_paths: List[Path],
    output_root: Path,
    output_file_arg: str,
    args: argparse.Namespace,
) -> None:
    if len(input_paths) == 0:
        print("No input files found. Exiting.")
        return

    for input_idx, input_path in enumerate(input_paths, start=1):
        if len(input_paths) > 1:
            print(f"\n[{input_idx}/{len(input_paths)}] Processing {input_path.name}")

        if len(input_paths) == 1 and input_path == output_root:
            output_dir = output_root
        elif output_root.suffix == ".npy":
            output_dir = output_root.parent / input_path.stem
        else:
            output_dir = output_root / input_path.stem

        if output_file_arg:
            output_file_path = Path(output_file_arg)
            if len(input_paths) == 1:
                merged_output_file = output_file_path
            else:
                if output_file_path.suffix == ".npy":
                    merged_output_file = output_file_path.parent / input_path.name.replace("_road_progress_3s", "")
                else:
                    merged_output_file = output_file_path / input_path.name.replace("_road_progress_3s", "")
            merged_output_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            merged_output_file = None

        fid2poly = None
        if args.output_format == "dataset":
            if not args.edges:
                raise ValueError("--edges is required when --output-format=dataset")
            print(f"Loading edges from {args.edges}...")
            fid2poly = build_fid_to_polyline_index(args.edges)
            print(f"Loaded {len(fid2poly)} road polylines")

        print(f"Extracting metadata from {input_path}...")
        metadata = extract_trajectory_metadata(str(input_path), meta_time_index=args.meta_time_index, fid2poly=fid2poly)

        if args.max_trajs > 0:
            metadata = metadata[: args.max_trajs]

        print(f"Extracted {len(metadata)} trajectories")
        if len(metadata) == 0:
            print(f"No valid trajectories found in {input_path}. Skipping.")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
        if len(gpu_ids) == 0:
            raise ValueError("No valid GPU ids parsed from --gpus")

        print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")

        chunk_size = (len(metadata) + len(gpu_ids) - 1) // len(gpu_ids)
        metadata_chunks = [metadata[i * chunk_size : (i + 1) * chunk_size] for i in range(len(gpu_ids))]
        metadata_chunks = [chunk for chunk in metadata_chunks if len(chunk) > 0]

        print(f"Split into {len(metadata_chunks)} chunks")
        for i, chunk in enumerate(metadata_chunks):
            print(f"  Worker {i}: {len(chunk)} trajectories")

        if len(gpu_ids) == 1:
            worker_process(
                gpu_id=gpu_ids[0],
                metadata_chunk=metadata_chunks[0],
                ckpt_path=args.ckpt,
                emb_path=args.emb,
                city=args.city,
                data_root=args.data_root,
                distance_table_path=args.distance_table,
                batch_size=args.batch_size,
                top_k=args.top_k,
                temperature=args.temperature,
                greedy=bool(args.greedy),
                max_steps=args.max_steps,
                visible_history_len_cli=int(args.visible_history_len),
                output_dir=output_dir,
                worker_id=0,
                output_format=args.output_format,
                stop_at_destination=bool(args.stop_at_destination),
                generation_mode=args.generation_mode,
                shortest_path_mode=args.shortest_path_mode,
                edges_path=args.edges,
            )
        else:
            mp.set_start_method("spawn", force=True)
            processes = []

            for i, (gpu_id, chunk) in enumerate(zip(gpu_ids, metadata_chunks)):
                p = mp.Process(
                    target=worker_process,
                    args=(
                        gpu_id,
                        chunk,
                        args.ckpt,
                        args.emb,
                        args.city,
                        args.data_root,
                        args.distance_table,
                        args.batch_size,
                        args.top_k,
                        args.temperature,
                        bool(args.greedy),
                        args.max_steps,
                        int(args.visible_history_len),
                        output_dir,
                        i,
                        args.output_format,
                        bool(args.stop_at_destination),
                        args.generation_mode,
                        args.shortest_path_mode,
                        args.edges,
                    ),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

        print(f"\nAll workers finished. Generated trajectories saved to {output_dir}/")
        print("Output files: generated_worker_*.npy")

        expected_worker_files = [_worker_output_path(output_dir, worker_id) for worker_id in range(len(metadata_chunks))]
        missing_worker_files = [path for path in expected_worker_files if not path.exists()]
        if missing_worker_files:
            missing = ", ".join(str(path) for path in missing_worker_files)
            raise FileNotFoundError(f"Missing worker output files: {missing}")

        total_generated = 0
        total_success = 0
        for chunk, worker_file in zip(metadata_chunks, expected_worker_files):
            items = np.load(worker_file, allow_pickle=True)
            total_generated += len(items)
            for item, meta_item in zip(items, chunk):
                target_destination = int(meta_item.destination_road)
                if args.output_format == "dataset":
                    points = item[1]
                    if len(points) == 0:
                        continue
                    final_road = int(points[-1][1])
                else:
                    if len(item) == 0:
                        continue
                    final_road = int(item[-1])
                if final_road == target_destination:
                    total_success += 1

        if total_generated > 0:
            print(
                f"Overall success rate: {total_success}/{total_generated} "
                f"({(total_success / total_generated):.4f})"
            )

        if merged_output_file is not None:
            n = _merge_worker_outputs(worker_files=expected_worker_files, merged_path=merged_output_file)
            print(f"Merged {n} trajectories to: {merged_output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch generate trajectories from real trajectory metadata.")

    parser.add_argument("--generation-mode", type=str, default="model", choices=["model", "shortest_path"], help="Generation backend: trained model or exact graph shortest path")
    parser.add_argument("--ckpt", type=str, default="", help="Downstream checkpoint (required for --generation-mode=model)")
    parser.add_argument("--emb", type=str, default="", help="Exported embedding file (.pt) (required for --generation-mode=model)")
    parser.add_argument("--city", type=str, default="xian", choices=["xian", "chengdu"])
    parser.add_argument("--data-root", type=str, default="data")

    parser.add_argument("--input", type=str, required=True, help="Input .npy file or directory. Supports raw GPS samples [meta, gps_points] and road-progress samples [meta, roads_unique, s_seq, t_seq]")
    parser.add_argument("--edges", type=str, default="", help="edges.csv used to extract start/end percentages for dataset output and length-weighted shortest paths")
    parser.add_argument("--output", type=str, required=True, help="Output directory for generated trajectories")

    parser.add_argument("--batch-size", type=int, default=128, help="Per-worker decode batch size")
    parser.add_argument("--distance-table", type=str, default="", help="Optional .npy all-pairs shortest-distance table used for destination-aware model selection")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k sampling (0 = no filter)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--greedy", action="store_true", help="Greedy decoding: pick highest-scoring road segment (no graph constraint)")
    parser.add_argument("--max-steps", type=int, default=100, help="Fixed maximum decode steps for every trajectory in model mode")
    parser.add_argument("--visible-history-len", type=int, default=-1, help="How many recent road segments are visible when predicting the next road; <=0 means auto from checkpoint segment_len, otherwise full prefix fallback")
    parser.add_argument("--stop-at-destination", action="store_true", help="Stop decoding once the generated path reaches the target destination")
    parser.add_argument("--shortest-path-mode", type=str, default="length", choices=["length", "hop"], help="Edge weighting used in shortest_path mode")

    parser.add_argument("--meta-time-index", type=int, default=3, help="Index in traj[0] for depart_time")

    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs (e.g., '0,1,2,3,4')")
    parser.add_argument("--max-trajs", type=int, default=0, help="Max trajectories to generate (0 = all)")

    parser.add_argument(
        "--output-format",
        type=str,
        default="road_ids",
        choices=["road_ids", "dataset"],
        help="Output format: 'road_ids' saves list[int]; 'dataset' saves [meta, points] with endpoint percentage in col 0 and road_id in col 1.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help="Optional merged output .npy file path (if set, shards will be merged).",
    )

    args = parser.parse_args()

    if args.generation_mode == "model":
        if not args.ckpt:
            raise ValueError("--ckpt is required when --generation-mode=model")
        if not args.emb:
            raise ValueError("--emb is required when --generation-mode=model")
    else:
        if args.shortest_path_mode == "length" and not args.edges:
            raise ValueError("--edges is required when --generation-mode=shortest_path and --shortest-path-mode=length")

    input_path = Path(args.input)
    if input_path.is_dir():
        input_paths = sorted(input_path.glob("*.npy"))
    else:
        input_paths = [input_path]

    if len(input_paths) == 0:
        raise ValueError(f"No .npy files found in: {input_path}")

    requested_gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if len(requested_gpu_ids) == 0:
        raise ValueError("--gpus must specify at least 1 GPU id")

    _run_generation_for_paths(
        input_paths=input_paths,
        output_root=Path(args.output),
        output_file_arg=args.output_file,
        args=args,
    )


if __name__ == "__main__":
    main()
