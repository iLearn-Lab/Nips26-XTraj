from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.multiprocessing as mp

from experiments.reconstruct_gps_from_road_progress import _reconstruct_one
from models.road_progress_decoder import RoadProgressDecoder, RoadProgressDecoderConfig
from utils.sampling import get_unique_segments
from utils.wkt_geometry import build_fid_to_polyline_index


MAX_ENDPOINT_PERCENTAGE = 1.0 - 1e-4
SAMPLE_FORMAT_ROAD_PROGRESS = "road_progress"
SAMPLE_FORMAT_ROADS_ENDPOINT_DATASET = "roads_endpoint_dataset"
SAMPLE_FORMAT_ROADS_DATASET = "roads_dataset"


@dataclass
class ParsedSample:
    meta: Any
    roads_unique: np.ndarray
    s_start: float
    s_end: float
    ref_t: np.ndarray | None
    depart_time: int | None
    total_duration: float | None
    decode_steps: int

def _load_z(emb_path: str) -> torch.Tensor:
    obj = torch.load(emb_path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "z" in obj:
        z = obj["z"]
    else:
        z = obj
    if not isinstance(z, torch.Tensor):
        raise TypeError(f"Expected embeddings to be a torch.Tensor, got {type(z)}")
    return z.detach().float().cpu()


def _gps4_to_gps5(gps4: np.ndarray) -> np.ndarray:
    """Convert GPS4 [lon, lat, road_id, time] to GPS5 [lon, lat, road_id, heading, speed].

    Heading is computed from consecutive GPS points (in degrees, 0-360).
    Speed is computed from distance and time intervals (in m/s).
    """
    gps4 = np.asarray(gps4, dtype=np.float32)
    if gps4.ndim != 2 or gps4.shape[1] != 4:
        raise ValueError(f"Expected gps array with shape [T,4], got {gps4.shape}")

    n = gps4.shape[0]
    gps5 = np.zeros((n, 5), dtype=np.float32)
    gps5[:, :3] = gps4[:, :3]  # Copy lon, lat, road_id

    # Compute heading and speed from consecutive points
    for i in range(n):
        if i == 0:
            # First point: use heading from first to second point
            if n > 1:
                dx = gps4[1, 0] - gps4[0, 0]  # lon difference
                dy = gps4[1, 1] - gps4[0, 1]  # lat difference
                heading = np.arctan2(dx, dy) * 180.0 / np.pi  # Convert to degrees
                heading = (heading + 360.0) % 360.0  # Normalize to [0, 360)
                gps5[0, 3] = heading

                # Speed: distance / time
                dt = gps4[1, 3]  # time interval
                if dt > 0:
                    # Approximate distance in meters (rough estimate for small distances)
                    dist = np.sqrt((dx * 111000 * np.cos(gps4[0, 1] * np.pi / 180.0))**2 + (dy * 111000)**2)
                    speed = dist / dt
                    gps5[0, 4] = speed
                else:
                    gps5[0, 4] = 0.0
            else:
                gps5[0, 3] = 0.0  # heading
                gps5[0, 4] = 0.0  # speed
        else:
            # Use heading from previous to current point
            dx = gps4[i, 0] - gps4[i-1, 0]
            dy = gps4[i, 1] - gps4[i-1, 1]
            heading = np.arctan2(dx, dy) * 180.0 / np.pi
            heading = (heading + 360.0) % 360.0
            gps5[i, 3] = heading

            # Speed: distance / time
            dt = gps4[i, 3]
            if dt > 0:
                dist = np.sqrt((dx * 111000 * np.cos(gps4[i-1, 1] * np.pi / 180.0))**2 + (dy * 111000)**2)
                speed = dist / dt
                gps5[i, 4] = speed
            else:
                gps5[i, 4] = 0.0

    return gps5


def _infer_sample_format(item: Any) -> str | None:
    try:
        if len(item) >= 4:
            roads_unique = np.asarray(item[1]).reshape(-1)
            s_seq = np.asarray(item[2], dtype=np.float32).reshape(-1)
            t_seq = np.asarray(item[3], dtype=np.float32).reshape(-1)
            if roads_unique.ndim == 1 and len(roads_unique) > 0 and s_seq.ndim == 1 and len(s_seq) > 0 and t_seq.ndim == 1:
                return SAMPLE_FORMAT_ROAD_PROGRESS
        if len(item) >= 2:
            pts = np.asarray(item[1])
            if pts.ndim == 2 and pts.shape[1] == 2:
                return SAMPLE_FORMAT_ROADS_ENDPOINT_DATASET
            if pts.ndim == 2 and pts.shape[1] >= 3:
                return SAMPLE_FORMAT_ROADS_DATASET
    except Exception:
        return None
    return None


def _road_ids_from_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Expected points array with shape [T,>=3], got {pts.shape}")
    road_ids = [int(p[2]) for p in pts if int(p[2]) >= 0]
    return np.asarray(get_unique_segments(road_ids), dtype=np.int32)


def _time_seq_from_points(points: np.ndarray) -> np.ndarray | None:
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] < 4:
        return None
    t_seq = np.asarray(pts[:, 3], dtype=np.float32)
    if t_seq.ndim != 1 or len(t_seq) == 0:
        return None
    t_seq = t_seq.copy()
    t_seq[0] = 0.0
    return t_seq


def _endpoint_info_from_points(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Parse endpoint dataset where points[:,0] stores s = i + alpha format.

    points[0,0] = 0 + start_alpha
    points[-1,0] = (num_segments - 1) + end_alpha
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected endpoint dataset array with shape [T,2], got {pts.shape}")
    road_ids = [int(p[1]) for p in pts if int(p[1]) >= 0]
    roads_unique = np.asarray(get_unique_segments(road_ids), dtype=np.int32)
    if len(roads_unique) == 0:
        raise ValueError("Endpoint dataset has no valid road ids")

    s_start = float(pts[0, 0]) if len(pts) > 0 else 0.0
    s_end = float(pts[-1, 0]) if len(pts) > 0 else float(max(len(roads_unique) - 1, 0)) + MAX_ENDPOINT_PERCENTAGE
    start_pct = float(np.clip(s_start, 0.0, MAX_ENDPOINT_PERCENTAGE))
    end_pct = float(np.clip(s_end - float(max(len(roads_unique) - 1, 0)), 0.0, MAX_ENDPOINT_PERCENTAGE))

    return roads_unique, start_pct, end_pct


def _total_duration_from_meta(meta: Any) -> float | None:
    """Infer total duration from metadata.

    Raw trajectory metadata uses meta[2] for duration, while some generated samples
    store an explicit duration fallback in meta[4]. Prefer meta[2] when available.
    """
    if meta is None:
        return None

    candidates: list[float | None] = []
    try:
        candidates.append(float(meta[2]) if len(meta) > 2 else None)
    except Exception:
        candidates.append(None)
    try:
        candidates.append(float(meta[4]) if len(meta) > 4 else None)
    except Exception:
        candidates.append(None)

    for value in candidates:
        if value is None:
            continue
        if np.isfinite(value) and value > 0:
            return float(value)
    return None


def _total_duration_from_t_seq(t_seq: np.ndarray | None) -> float | None:
    if t_seq is None:
        return None
    arr = np.asarray(t_seq, dtype=np.float32).reshape(-1)
    if len(arr) == 0:
        return None
    return float(arr.sum())


def _remaining_time_from_t_seq(t_seq: np.ndarray, total_duration: float | None = None) -> np.ndarray:
    arr = np.asarray(t_seq, dtype=np.float32).reshape(-1)
    if len(arr) == 0:
        return np.zeros((0,), dtype=np.float32)
    if total_duration is None:
        total_duration = float(arr.sum())
    denom = max(float(total_duration), 1e-6)
    elapsed = np.cumsum(arr, dtype=np.float32)
    remaining = (float(total_duration) - elapsed) / denom
    return np.clip(remaining, 0.0, 1.0).astype(np.float32)


def _constant_step_remaining_time(length: int, sample_seconds: float, total_duration: float) -> np.ndarray:
    if length <= 0:
        return np.zeros((0,), dtype=np.float32)
    t_seq = np.full((length,), float(sample_seconds), dtype=np.float32)
    t_seq[0] = 0.0
    return _remaining_time_from_t_seq(t_seq, total_duration=total_duration)


def _resolve_emb_path(ckpt: dict[str, Any], emb_arg: str) -> str:
    emb_path = emb_arg or ckpt.get("args", {}).get("emb", "")
    if not emb_path:
        raise ValueError("Embedding path not provided and not found in checkpoint args")
    if not Path(emb_path).is_absolute():
        emb_path = str((REPO_ROOT / emb_path).resolve())
    return emb_path


def _parse_sample(item: Any, cfg: RoadProgressDecoderConfig) -> ParsedSample | None:
    sample_format = _infer_sample_format(item)
    if sample_format is None:
        return None

    try:
        meta = item[0]
        if sample_format == SAMPLE_FORMAT_ROAD_PROGRESS:
            roads_unique = np.asarray(item[1], dtype=np.int32)
            ref_s = np.asarray(item[2], dtype=np.float32)
            ref_t = np.asarray(item[3], dtype=np.float32) if len(item) > 3 else None
            s_start = float(ref_s[0]) if len(ref_s) > 0 else 0.0
            s_end = float(ref_s[-1]) if len(ref_s) > 0 else float(max(len(roads_unique) - 1, 0)) + MAX_ENDPOINT_PERCENTAGE
        else:
            gps_points = np.asarray(item[1])
            if sample_format == SAMPLE_FORMAT_ROADS_ENDPOINT_DATASET:
                roads_unique, start_pct, end_pct = _endpoint_info_from_points(gps_points)
                s_start = start_pct
                s_end = float(max(len(roads_unique) - 1, 0)) + end_pct
                ref_t = None
            else:
                roads_unique = _road_ids_from_points(gps_points)
                s_start = 0.0
                s_end = float(max(len(roads_unique) - 1, 0)) + MAX_ENDPOINT_PERCENTAGE
                ref_t = _time_seq_from_points(gps_points)
    except Exception:
        return None

    if roads_unique.ndim != 1 or len(roads_unique) == 0:
        return None

    depart_time = None
    if cfg.use_depart_time:
        try:
            depart_time = int(meta[3])
        except Exception:
            depart_time = 0

    total_duration = _total_duration_from_t_seq(ref_t)
    if total_duration is None:
        total_duration = _total_duration_from_meta(meta)
    if (cfg.use_total_duration or cfg.use_remaining_time) and total_duration is None:
        raise ValueError("Input sample lacks usable total_duration; expected t_seq sum, meta[2], or generated-sample meta[4]")

    decode_steps = int(cfg.max_steps)
    if ref_t is not None and len(ref_t) > 0:
        decode_steps = min(decode_steps, max(1, int(len(ref_t) - 1)))
    else:
        # Try to get GPS count from meta[4] if available
        try:
            if len(meta) > 4:
                gps_count = int(meta[4])
                if gps_count > 0:
                    decode_steps = min(decode_steps, gps_count)
        except Exception:
            pass

    return ParsedSample(
        meta=meta,
        roads_unique=roads_unique,
        s_start=float(s_start),
        s_end=float(s_end),
        ref_t=(np.asarray(ref_t, dtype=np.float32) if ref_t is not None else None),
        depart_time=depart_time,
        total_duration=(float(total_duration) if total_duration is not None else None),
        decode_steps=int(decode_steps),
    )


def _collate_parsed_samples(
    samples: List[ParsedSample],
    cfg: RoadProgressDecoderConfig,
    device: torch.device,
    sample_seconds: float,
) -> dict[str, Any]:
    batch_size = len(samples)
    max_roads = max(len(sample.roads_unique) for sample in samples)

    roads_unique = torch.zeros((batch_size, max_roads), dtype=torch.long, device=device)
    road_mask = torch.zeros((batch_size, max_roads), dtype=torch.bool, device=device)
    s_start = torch.zeros((batch_size,), dtype=torch.float32, device=device)
    s_end = torch.zeros((batch_size,), dtype=torch.float32, device=device)
    decode_steps = torch.zeros((batch_size,), dtype=torch.long, device=device)

    depart_time = torch.zeros((batch_size,), dtype=torch.long, device=device) if cfg.use_depart_time else None
    total_duration = torch.zeros((batch_size,), dtype=torch.float32, device=device) if cfg.use_total_duration else None
    remaining_time_seq = None
    if cfg.use_remaining_time:
        remaining_time_seq = torch.zeros((batch_size, int(cfg.max_steps) + 1), dtype=torch.float32, device=device)

    for i, sample in enumerate(samples):
        road_count = len(sample.roads_unique)
        roads_unique[i, :road_count] = torch.as_tensor(sample.roads_unique, dtype=torch.long, device=device)
        road_mask[i, :road_count] = True
        s_start[i] = float(sample.s_start)
        s_end[i] = float(sample.s_end)
        decode_steps[i] = int(sample.decode_steps)
        if depart_time is not None:
            depart_time[i] = int(sample.depart_time or 0)
        if total_duration is not None:
            total_duration[i] = float(sample.total_duration if sample.total_duration is not None else 0.0)
        if remaining_time_seq is not None:
            if sample.total_duration is None:
                raise ValueError("remaining_time generation requires total_duration for every sample")
            if sample.ref_t is not None and len(sample.ref_t) > 0:
                rem = _remaining_time_from_t_seq(sample.ref_t, total_duration=float(sample.total_duration))
            else:
                rem = _constant_step_remaining_time(int(cfg.max_steps) + 1, sample_seconds, float(sample.total_duration))
            rem_t = torch.as_tensor(rem, dtype=torch.float32, device=device)
            take = min(int(rem_t.numel()), int(cfg.max_steps) + 1)
            remaining_time_seq[i, :take] = rem_t[:take]

    return {
        "roads_unique": roads_unique,
        "road_mask": road_mask,
        "s_start": s_start,
        "s_end": s_end,
        "decode_steps": decode_steps,
        "depart_time": depart_time,
        "total_duration": total_duration,
        "remaining_time_seq": remaining_time_seq,
    }


def _postprocess_generated(
    sample: ParsedSample,
    s_gen: torch.Tensor,
    fid2poly: dict,
    force_length: bool,
    sample_seconds: float,
) -> Any:
    s_out = s_gen.detach().cpu().numpy().astype(np.float32)
    ref_t = sample.ref_t

    if ref_t is not None and len(ref_t) > 0:
        if force_length:
            target_len = int(len(ref_t))
            if len(s_out) >= target_len:
                s_out = s_out[:target_len]
            else:
                pad_val = float(s_out[-1]) if len(s_out) > 0 else 0.0
                s_out = np.pad(s_out, (0, target_len - len(s_out)), constant_values=pad_val)
            t_seq = ref_t.astype(np.float32)
        else:
            t_seq = ref_t[: len(s_out)].astype(np.float32)
    else:
        # Generate uniform time intervals based on total_duration
        if sample.total_duration is not None and sample.total_duration > 0 and len(s_out) > 1:
            # Distribute total_duration uniformly across all points
            interval = float(sample.total_duration) / max(1, len(s_out) - 1)
            t_seq = np.full(len(s_out), interval, dtype=np.float32)
            t_seq[0] = 0.0  # First point has 0 time
        else:
            # Fallback to sample_seconds if no total_duration available
            t_seq = np.full_like(s_out, float(sample_seconds), dtype=np.float32)
            if len(t_seq) > 0:
                t_seq[0] = 0.0

    rec = _reconstruct_one(
        meta=sample.meta,
        roads_unique=sample.roads_unique,
        s=s_out,
        t_seq=t_seq,
        fid2poly=fid2poly,
    )
    gps5 = _gps4_to_gps5(np.asarray(rec[1], dtype=np.float32))
    return [sample.meta, gps5]


@torch.no_grad()
def _generate_batch(
    *,
    model: RoadProgressDecoder,
    cfg: RoadProgressDecoderConfig,
    z: torch.Tensor,
    fid2poly: dict,
    samples: List[ParsedSample],
    device: torch.device,
    force_length: bool,
    sample_seconds: float,
) -> List[Any]:
    if not samples:
        return []

    batch = _collate_parsed_samples(samples, cfg, device, sample_seconds)
    s_list = model.generate_batch(
        roads_unique=batch["roads_unique"],
        road_mask=batch["road_mask"],
        z=z,
        s_start=batch["s_start"],
        s_end=batch["s_end"],
        decode_steps=batch["decode_steps"],
        remaining_time_seq=batch["remaining_time_seq"],
        depart_time=batch["depart_time"],
        total_duration=batch["total_duration"],
        device=device,
        stop_at_end=not force_length,
    )
    return [_postprocess_generated(sample, s_gen, fid2poly, force_length, sample_seconds) for sample, s_gen in zip(samples, s_list)]


def _output_path(output_dir: Path, input_path: Path) -> Path:
    name = input_path.name
    for suffix in ("_road_progress_3s", "_road_progress_1s", "_road_progress"):
        if suffix in name:
            return output_dir / name.replace(suffix, "")
    return output_dir / name


def _save_atomic(path: Path, samples: List[Any]) -> None:
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(str(tmp), np.asarray(samples, dtype=object))
    tmp.replace(path)


def _process_files(
    *,
    worker_id: int,
    gpu_id: int | None,
    file_paths: List[Path],
    output_dir: Path,
    ckpt_path: str,
    emb_path: str,
    edges_path: str,
    force_length: bool,
    skip_existing: bool,
    log_every: int,
    batch_size: int,
    sample_seconds: float,
) -> None:
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = RoadProgressDecoderConfig(**ckpt["cfg"])
    model = RoadProgressDecoder(cfg)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model = model.to(device)
    model.eval()

    z = _load_z(emb_path).to(device)
    fid2poly = build_fid_to_polyline_index(edges_path)

    print(f"[Worker {worker_id}] device={device} files={len(file_paths)}")

    for file_idx, path in enumerate(file_paths, start=1):
        out_path = _output_path(output_dir, path)
        if skip_existing and out_path.exists():
            print(f"[Worker {worker_id}] skip existing {out_path.name}")
            continue

        arr = np.load(str(path), allow_pickle=True)
        total = len(arr)
        samples: List[Any] = []
        skipped = 0
        parsed_samples: List[ParsedSample] = []

        print(f"[Worker {worker_id}] [{file_idx}/{len(file_paths)}] start {path.name} trajectories={total}")
        for traj_idx, item in enumerate(arr, start=1):
            parsed = _parse_sample(item, cfg)
            if parsed is None:
                skipped += 1
            else:
                parsed_samples.append(parsed)

            if traj_idx % log_every == 0 or traj_idx == total:
                print(
                    f"[Worker {worker_id}] {path.name} parsed {traj_idx}/{total} done "
                    f"valid_so_far={len(parsed_samples)} skipped={skipped}"
                )

        for batch_start in range(0, len(parsed_samples), batch_size):
            batch = parsed_samples[batch_start: batch_start + batch_size]
            samples.extend(
                _generate_batch(
                    model=model,
                    cfg=cfg,
                    z=z,
                    fid2poly=fid2poly,
                    samples=batch,
                    device=device,
                    force_length=force_length,
                    sample_seconds=sample_seconds,
                )
            )
            done = min(batch_start + len(batch), len(parsed_samples))
            if done % log_every == 0 or done == len(parsed_samples):
                print(
                    f"[Worker {worker_id}] {path.name} generated {done}/{len(parsed_samples)} valid trajectories "
                    f"saved_so_far={len(samples)} skipped={skipped}"
                )

        _save_atomic(out_path, samples)
        print(
            f"[Worker {worker_id}] finished {path.name} -> {out_path} "
            f"samples={len(samples)} skipped={skipped}"
        )


def worker_process(
    worker_id: int,
    gpu_id: int | None,
    file_paths: List[str],
    output_dir: str,
    ckpt_path: str,
    emb_path: str,
    edges_path: str,
    force_length: bool,
    skip_existing: bool,
    log_every: int,
    batch_size: int,
    sample_seconds: float,
) -> None:
    _process_files(
        worker_id=worker_id,
        gpu_id=gpu_id,
        file_paths=[Path(p) for p in file_paths],
        output_dir=Path(output_dir),
        ckpt_path=ckpt_path,
        emb_path=emb_path,
        edges_path=edges_path,
        force_length=force_length,
        skip_existing=skip_existing,
        log_every=log_every,
        batch_size=batch_size,
        sample_seconds=sample_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch generate GPS trajectories from road-progress or road-segment files")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing .npy files in road-progress format [meta, roads_unique, s, t], old road-segment format [meta, gps_points[:,2]=road_id], or endpoint format [meta, points[:,0]=percentage, points[:,1]=road_id]")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--emb", type=str, default="")
    parser.add_argument("--edges", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", help="Single-device fallback: auto/cpu/cuda:0")
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=1, help="Number of worker processes per GPU")
    parser.add_argument("--batch-size", type=int, default=128, help="Per-worker GPU generation batch size")
    parser.add_argument("--sample-seconds", type=float, default=3.0, help="Fallback step seconds for remaining-time generation when input lacks ref_t")
    parser.add_argument("--force-length", action="store_true", help="Force generated trajectory length to match input t_sequence length")
    parser.add_argument("--max-files", type=int, default=0, help="0 = all files")
    parser.add_argument("--log-every", type=int, default=1000, help="Print progress every N trajectories")
    parser.add_argument("--skip-existing", action="store_true", help="Skip output files that already exist")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob("*.npy"))
    if not paths:
        raise ValueError(f"No .npy files found in: {input_dir}")
    if args.max_files > 0:
        paths = paths[: int(args.max_files)]

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    emb_path = _resolve_emb_path(ckpt, args.emb)

    gpu_ids: List[int] = []
    if args.gpus.strip():
        gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip()]
    elif args.device.startswith("cuda:"):
        gpu_ids = [int(args.device.split(":", 1)[1])]
    elif args.device == "auto" and torch.cuda.is_available():
        gpu_ids = [0]

    workers_per_gpu = max(int(args.workers_per_gpu), 1)

    if gpu_ids:
        assignments: List[List[Path]] = [[] for _ in range(len(gpu_ids) * workers_per_gpu)]
        worker_gpu_ids: List[int] = []
        for gpu_id in gpu_ids:
            for _ in range(workers_per_gpu):
                worker_gpu_ids.append(gpu_id)
        for i, path in enumerate(paths):
            assignments[i % len(assignments)].append(path)

        print(f"Using GPUs {gpu_ids} with workers_per_gpu={workers_per_gpu}; total_workers={len(assignments)}")
        for i, (gpu_id, file_list) in enumerate(zip(worker_gpu_ids, assignments)):
            print(f"  Worker {i}: cuda:{gpu_id} files={len(file_list)}")

        mp.set_start_method("spawn", force=True)
        processes: List[mp.Process] = []
        for worker_id, (gpu_id, file_list) in enumerate(zip(worker_gpu_ids, assignments)):
            p = mp.Process(
                target=worker_process,
                args=(
                    worker_id,
                    gpu_id,
                    [str(pth) for pth in file_list],
                    str(output_dir),
                    args.ckpt,
                    emb_path,
                    args.edges,
                    bool(args.force_length),
                    bool(args.skip_existing),
                    int(args.log_every),
                    max(int(args.batch_size), 1),
                    float(args.sample_seconds),
                ),
            )
            p.start()
            processes.append(p)

        exit_code = 0
        for p in processes:
            p.join()
            if p.exitcode != 0:
                exit_code = p.exitcode or 1
        if exit_code != 0:
            raise SystemExit(exit_code)
    else:
        if args.device == "cpu" or not torch.cuda.is_available():
            gpu_id = None
            device_desc = "cpu"
        else:
            gpu_id = 0
            device_desc = "cuda:0"
        print(f"Using single worker on {device_desc}")
        _process_files(
            worker_id=0,
            gpu_id=gpu_id,
            file_paths=paths,
            output_dir=output_dir,
            ckpt_path=args.ckpt,
            emb_path=emb_path,
            edges_path=args.edges,
            force_length=bool(args.force_length),
            skip_existing=bool(args.skip_existing),
            log_every=int(args.log_every),
            batch_size=max(int(args.batch_size), 1),
            sample_seconds=float(args.sample_seconds),
        )


if __name__ == "__main__":
    main()
