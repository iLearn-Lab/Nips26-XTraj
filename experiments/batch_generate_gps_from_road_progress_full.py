"""Generate GPS trajectories from road progress using full trajectory model.

This script is updated to work with models trained using train_road_progress_decoder_full.py,
which uses absolute remaining time (in minutes) instead of normalized [0,1] remaining time.

Key differences from the original batch_generate_gps_from_road_progress.py:
1. Uses absolute remaining time in minutes (not normalized [0,1])
2. Compatible with models trained on full trajectories (no segment_len)
3. Supports decoder_local_window for efficient generation

Usage:
    python experiments/batch_generate_gps_from_road_progress_full.py \
        --input-dir "data/road_progress" \
        --output-dir "generated_output" \
        --ckpt "checkpoints/xian_decoder_full_epoch0010.pt" \
        --emb "checkpoints/embedding/xian_embedding.pt" \
        --edges "/path/to/edges.csv" \
        --gpus 0 \
        --batch-size 64
"""

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
    gps4 = np.asarray(gps4, dtype=np.float32)
    if gps4.ndim != 2 or gps4.shape[1] != 4:
        raise ValueError(f"Expected gps array with shape [T,4], got {gps4.shape}")
    gps5 = np.zeros((gps4.shape[0], 5), dtype=np.float32)
    gps5[:, :4] = gps4
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
    """Parse endpoint dataset where points[:,0] stores s = i + alpha format."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"Expected points array with shape [T,>=2], got {pts.shape}")
    s_values = pts[:, 0]
    road_indices = np.floor(s_values).astype(np.int32)
    unique_roads = []
    for r in road_indices:
        if len(unique_roads) == 0 or r != unique_roads[-1]:
            unique_roads.append(r)
    roads_unique = np.asarray(unique_roads, dtype=np.int32)
    s_start = float(s_values[0])
    s_end = float(s_values[-1])
    return roads_unique, s_start, s_end


def _depart_time_from_meta(meta: Any) -> int | None:
    if not isinstance(meta, (list, tuple, np.ndarray)) or len(meta) < 4:
        return None
    try:
        return int(meta[3])
    except (ValueError, TypeError):
        return None


def _total_duration_from_t_seq(t_seq: np.ndarray | None) -> float | None:
    if t_seq is None:
        return None
    arr = np.asarray(t_seq, dtype=np.float32).reshape(-1)
    if len(arr) == 0:
        return None
    return float(arr.sum())


def _absolute_remaining_time_from_t_seq(t_seq: np.ndarray, total_duration: float | None = None) -> np.ndarray:
    """Convert elapsed time sequence to absolute remaining time in minutes.

    This matches the training data preprocessing in RoadProgressDatasetFull.

    Args:
        t_seq: [T] elapsed time in seconds at each step
        total_duration: total trajectory duration in seconds (if None, computed from t_seq)

    Returns:
        remaining_time: [T] remaining time in minutes
    """
    arr = np.asarray(t_seq, dtype=np.float32).reshape(-1)
    if len(arr) == 0:
        return np.zeros((0,), dtype=np.float32)

    if total_duration is None:
        total_duration = float(arr.sum())

    elapsed = np.cumsum(arr, dtype=np.float32)
    remaining_seconds = float(total_duration) - elapsed
    remaining_minutes = remaining_seconds / 60.0

    # Clip to non-negative (avoid numerical issues)
    return np.clip(remaining_minutes, 0.0, None).astype(np.float32)


def _constant_step_remaining_time(length: int, sample_seconds: float, total_duration: float) -> np.ndarray:
    """Generate remaining time sequence for constant sampling interval.

    Args:
        length: number of steps
        sample_seconds: time interval between steps in seconds
        total_duration: total trajectory duration in seconds

    Returns:
        remaining_time: [length] remaining time in minutes
    """
    if length <= 0:
        return np.zeros((0,), dtype=np.float32)
    t_seq = np.full((length,), float(sample_seconds), dtype=np.float32)
    t_seq[0] = 0.0
    return _absolute_remaining_time_from_t_seq(t_seq, total_duration=total_duration)


def _resolve_emb_path(ckpt: dict[str, Any], emb_arg: str) -> str:
    emb_path = emb_arg or ckpt.get("args", {}).get("emb", "")
    if not emb_path:
        raise ValueError("Embedding path not provided and not found in checkpoint args")
    if not Path(emb_path).is_absolute():
        emb_path = str((REPO_ROOT / emb_path).resolve())
    return emb_path


def _parse_sample(item: Any, cfg: RoadProgressDecoderConfig) -> ParsedSample | None:
    """Parse a sample from the input dataset."""
    try:
        fmt = _infer_sample_format(item)
        if fmt is None:
            return None

        meta = item[0] if len(item) > 0 else None
        depart_time = _depart_time_from_meta(meta)

        if fmt == SAMPLE_FORMAT_ROAD_PROGRESS:
            roads_unique = np.asarray(item[1], dtype=np.int32).reshape(-1)
            s_seq = np.asarray(item[2], dtype=np.float32).reshape(-1)
            t_seq = np.asarray(item[3], dtype=np.float32).reshape(-1)
            s_start = float(s_seq[0])
            s_end = float(s_seq[-1])
            ref_t = t_seq
            total_duration = _total_duration_from_t_seq(t_seq)
            decode_steps = len(s_seq) - 1

        elif fmt == SAMPLE_FORMAT_ROADS_ENDPOINT_DATASET:
            points = np.asarray(item[1])
            roads_unique, s_start, s_end = _endpoint_info_from_points(points)
            ref_t = None
            total_duration = None
            decode_steps = int(cfg.max_steps)

        elif fmt == SAMPLE_FORMAT_ROADS_DATASET:
            points = np.asarray(item[1])
            roads_unique = _road_ids_from_points(points)

            # Check if points has enhanced info (columns 3-6)
            if points.ndim == 2 and points.shape[1] >= 6:
                # Enhanced format: [s, road_id, orig_gps_count, orig_duration, orig_s_start, orig_s_end]
                orig_gps_count = int(points[0, 2])
                orig_duration = float(points[0, 3])
                s_start = float(points[0, 4])
                s_end = float(points[0, 5])
                decode_steps = orig_gps_count - 1
                total_duration = orig_duration
                # Generate uniform t_seq based on original GPS count
                ref_t = np.full((orig_gps_count,), orig_duration / orig_gps_count, dtype=np.float32)
                ref_t[0] = 0.0
            elif points.ndim == 2 and points.shape[1] >= 4:
                # Old enhanced format: [s, road_id, orig_gps_count, orig_duration]
                orig_gps_count = int(points[0, 2])
                orig_duration = float(points[0, 3])
                s_start = 0.0
                s_end = float(len(roads_unique)) * MAX_ENDPOINT_PERCENTAGE
                decode_steps = orig_gps_count - 1
                total_duration = orig_duration
                ref_t = np.full((orig_gps_count,), orig_duration / orig_gps_count, dtype=np.float32)
                ref_t[0] = 0.0
            else:
                # Original format: [s, road_id] only
                s_start = 0.0
                s_end = float(len(roads_unique)) * MAX_ENDPOINT_PERCENTAGE
                ref_t = _time_seq_from_points(points)
                total_duration = _total_duration_from_t_seq(ref_t)
                decode_steps = len(points) - 1 if len(points) > 0 else int(cfg.max_steps)

        else:
            return None

        if len(roads_unique) == 0:
            return None

        return ParsedSample(
            meta=meta,
            roads_unique=roads_unique,
            s_start=s_start,
            s_end=s_end,
            ref_t=ref_t,
            depart_time=depart_time,
            total_duration=total_duration,
            decode_steps=decode_steps,
        )

    except Exception as e:
        print(f"Warning: Failed to parse sample: {e}")
        return None


def _validate_sample(sample: ParsedSample, cfg: RoadProgressDecoderConfig, use_default_duration: bool = True) -> bool:
    """Validate that sample has required conditional inputs.

    Args:
        sample: Parsed sample to validate
        cfg: Model configuration
        use_default_duration: If True, allow missing total_duration (will use default)
    """
    if (cfg.use_total_duration or cfg.use_remaining_time) and sample.total_duration is None:
        if not use_default_duration:
            return False
        # Will use default duration, so it's valid
    if cfg.use_depart_time and sample.depart_time is None:
        # Missing depart_time is acceptable, will use default (0)
        pass
    return True


def _build_batch(
    samples: List[ParsedSample],
    cfg: RoadProgressDecoderConfig,
    device: torch.device,
    sample_seconds: float = 3.0,
    default_duration_seconds: float = 600.0,  # Default 10 minutes
) -> dict[str, Any]:
    """Build a batch tensor dict from parsed samples.

    Args:
        samples: List of parsed samples
        cfg: Model configuration
        device: Target device
        sample_seconds: Time interval between samples (for constant step generation)
        default_duration_seconds: Default duration when total_duration is missing
    """
    batch_size = len(samples)
    max_roads = max(len(s.roads_unique) for s in samples)

    roads_unique = torch.zeros((batch_size, max_roads), dtype=torch.long, device=device)
    road_mask = torch.zeros((batch_size, max_roads), dtype=torch.bool, device=device)
    s_start = torch.zeros((batch_size,), dtype=torch.float32, device=device)
    s_end = torch.zeros((batch_size,), dtype=torch.float32, device=device)
    decode_steps = torch.zeros((batch_size,), dtype=torch.long, device=device)

    depart_time = None
    if cfg.use_depart_time:
        depart_time = torch.zeros((batch_size,), dtype=torch.long, device=device)

    total_duration = None
    if cfg.use_total_duration:
        total_duration = torch.zeros((batch_size,), dtype=torch.float32, device=device)

    remaining_time_seq = None
    if cfg.use_remaining_time:
        remaining_time_seq = torch.zeros((batch_size, int(cfg.max_steps) + 1), dtype=torch.float32, device=device)

    for i, sample in enumerate(samples):
        m = len(sample.roads_unique)
        roads_unique[i, :m] = torch.from_numpy(sample.roads_unique).long()
        road_mask[i, :m] = True
        s_start[i] = sample.s_start
        s_end[i] = sample.s_end
        decode_steps[i] = sample.decode_steps

        if depart_time is not None:
            depart_time[i] = sample.depart_time if sample.depart_time is not None else 0

        # Use default duration if missing
        duration = sample.total_duration if sample.total_duration is not None else default_duration_seconds

        if total_duration is not None:
            total_duration[i] = duration

        if remaining_time_seq is not None:
            if sample.ref_t is not None:
                rem = _absolute_remaining_time_from_t_seq(sample.ref_t, total_duration=float(duration))
            else:
                rem = _constant_step_remaining_time(int(cfg.max_steps) + 1, sample_seconds, float(duration))

            rem_t = torch.from_numpy(rem).float()
            take = min(int(rem_t.shape[0]), int(remaining_time_seq.shape[1]))
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


def _generate_batch(
    model: RoadProgressDecoder,
    z: torch.Tensor,
    batch: dict[str, Any],
    device: torch.device,
) -> List[torch.Tensor]:
    """Generate s sequences for a batch."""
    z_device = z.to(device)

    s_seqs = model.generate_batch(
        roads_unique=batch["roads_unique"],
        z=z_device,
        road_mask=batch["road_mask"],
        s_start=batch["s_start"],
        s_end=batch["s_end"],
        decode_steps=batch["decode_steps"],
        depart_time=batch["depart_time"],
        total_duration=batch["total_duration"],
        remaining_time_seq=batch["remaining_time_seq"],
        device=device,
        stop_at_end=True,
    )

    return s_seqs


def _worker_process(
    rank: int,
    world_size: int,
    input_files: List[Path],
    output_dir: Path,
    ckpt_path: str,
    emb_path: str,
    edges_path: str,
    batch_size: int,
    log_every: int,
    gpu_id: int = None,
):
    """Worker process for parallel generation."""
    # Use gpu_id if provided, otherwise use rank
    actual_gpu = gpu_id if gpu_id is not None else rank
    device = torch.device(f"cuda:{actual_gpu}" if torch.cuda.is_available() else "cpu")

    # Load checkpoint and model
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("cfg", {})
    cfg = RoadProgressDecoderConfig(**cfg_dict)

    model = RoadProgressDecoder(cfg)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    # Load embeddings
    z = _load_z(emb_path)

    # Load edges for GPS reconstruction
    fid_to_polyline = build_fid_to_polyline_index(edges_path)

    # Process assigned files
    for file_idx, input_file in enumerate(input_files):
        if file_idx % world_size != rank:
            continue

        print(f"[GPU {actual_gpu}] Processing {input_file.name}...")

        # Load input data
        data = np.load(input_file, allow_pickle=True)

        # Parse samples
        samples = []
        for item in data:
            sample = _parse_sample(item, cfg)
            if sample is not None and _validate_sample(sample, cfg):
                samples.append(sample)

        if len(samples) == 0:
            print(f"[GPU {actual_gpu}] No valid samples in {input_file.name}, skipping")
            continue

        print(f"[GPU {actual_gpu}] Generating {len(samples)} trajectories from {input_file.name}...")

        # Generate in batches
        all_results = []
        num_batches = (len(samples) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(samples))
            batch_samples = samples[start_idx:end_idx]

            # Build batch
            batch = _build_batch(batch_samples, cfg, device)

            # Generate
            with torch.no_grad():
                s_seqs = _generate_batch(model, z, batch, device)

            # Reconstruct GPS
            for i, (sample, s_seq) in enumerate(zip(batch_samples, s_seqs)):
                s_np = s_seq.cpu().numpy()
                gps = _reconstruct_one(
                    meta=sample.meta,
                    roads_unique=sample.roads_unique,
                    s=s_np,
                    t_seq=None,
                    fid2poly=fid_to_polyline,
                )

                # _reconstruct_one returns [meta, gps_array]
                if isinstance(gps, list) and len(gps) >= 2:
                    gps_array = gps[1]
                else:
                    gps_array = gps

                # Convert to numpy array if needed
                if not isinstance(gps_array, np.ndarray):
                    gps_array = np.asarray(gps_array, dtype=np.float32)

                # Convert to 5-column format if needed
                if gps_array.ndim == 2 and gps_array.shape[1] == 4:
                    gps_array = _gps4_to_gps5(gps_array)

                result = [sample.meta, gps_array]
                all_results.append(result)

            if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == num_batches:
                print(f"[GPU {actual_gpu}] {input_file.name}: {end_idx}/{len(samples)} trajectories generated")

        # Save results
        output_file = output_dir / f"{input_file.stem}_generated_gpu{actual_gpu}.npy"
        np.save(output_file, np.array(all_results, dtype=object))
        print(f"[GPU {actual_gpu}] Saved {len(all_results)} trajectories to {output_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GPS trajectories from road progress using full trajectory model"
    )

    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing input .npy files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for generated trajectories")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--emb", type=str, default="",
                        help="Path to road embeddings (if not in checkpoint)")
    parser.add_argument("--edges", type=str, required=True,
                        help="Path to edges.csv for GPS reconstruction")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs (e.g., '0,1,2,3')")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size per GPU")
    parser.add_argument("--workers-per-gpu", type=int, default=1,
                        help="Number of worker processes per GPU (default: 1)")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log progress every N batches")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve embedding path
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    emb_path = _resolve_emb_path(ckpt, args.emb)

    # Get input files
    input_files = sorted(input_dir.glob("*.npy"))
    if len(input_files) == 0:
        raise ValueError(f"No .npy files found in {input_dir}")

    print(f"Found {len(input_files)} input files")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Embeddings: {emb_path}")
    print(f"Edges: {args.edges}")
    print(f"Output directory: {output_dir}")
    print()

    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    world_size = len(gpu_ids) * args.workers_per_gpu

    if world_size == 1:
        # Single process
        _worker_process(
            rank=0,
            world_size=1,
            input_files=input_files,
            output_dir=output_dir,
            ckpt_path=args.ckpt,
            emb_path=emb_path,
            edges_path=args.edges,
            batch_size=args.batch_size,
            log_every=args.log_every,
            gpu_id=gpu_ids[0],  # Use the specified GPU ID
        )
    else:
        # Multi-process
        mp.set_start_method("spawn", force=True)
        processes = []

        for gpu_idx, gpu_id in enumerate(gpu_ids):
            for worker_idx in range(args.workers_per_gpu):
                rank = gpu_idx * args.workers_per_gpu + worker_idx
                p = mp.Process(
                    target=_worker_process,
                    args=(
                        rank,
                        world_size,
                        input_files,
                        output_dir,
                        args.ckpt,
                        emb_path,
                        args.edges,
                        args.batch_size,
                        args.log_every,
                        gpu_id,  # Pass the actual GPU ID
                    ),
                )
                p.start()
                processes.append(p)

        for p in processes:
            p.join()

    print("\nGeneration completed!")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
