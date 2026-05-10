"""Sample n trajectories from original dataset and generate road segments + GPS points.

This script:
1. Randomly samples n trajectories from the original dataset
2. Generates road segments using the segment-level generator
3. Generates GPS points using the point-level generator
4. Saves sampled original trajectories and generated trajectories as separate .npy files

Usage:
    python experiments/sample_and_generate_trajectories.py \
        --input-dir "/path/to/datasets/chengdu_7days" \
        --output-dir "sampled_generation/chengdu" \
        --num-samples 100 \
        --segment-ckpt "checkpoints/segment_level_generator/chengdu_ar_heat_road_gps_epoch0780.pt" \
        --point-ckpt "checkpoints/point_level_generator/chengdu_point_decoder_epoch0020.pt" \
        --emb "checkpoints/embedding/chengdu_embedding.pt" \
        --city chengdu \
        --data-root "/path/to/datasets" \
        --edges "/home/ubuntu/workplace/didi/chengdu_road_network/edges.csv" \
        --distance-table "checkpoints/chengdu_length_dist.npy" \
        --gpu 0
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def sample_trajectories(input_dir: Path, num_samples: int, seed: int = 42) -> np.ndarray:
    """Randomly sample n trajectories from all .npy files in input_dir."""
    print(f"Sampling {num_samples} trajectories from {input_dir}...")

    npy_files = sorted(input_dir.glob("*.npy"))
    if len(npy_files) == 0:
        raise ValueError(f"No .npy files found in {input_dir}")

    print(f"Found {len(npy_files)} .npy files")

    # Load all trajectories
    all_trajs = []
    for npy_file in npy_files:
        arr = np.load(npy_file, allow_pickle=True)
        all_trajs.extend(list(arr))

    print(f"Total trajectories: {len(all_trajs)}")

    if num_samples > len(all_trajs):
        print(f"Warning: Requested {num_samples} samples but only {len(all_trajs)} available. Using all.")
        num_samples = len(all_trajs)

    # Random sampling
    random.seed(seed)
    sampled_indices = random.sample(range(len(all_trajs)), num_samples)
    sampled_trajs = [all_trajs[i] for i in sorted(sampled_indices)]

    print(f"Sampled {len(sampled_trajs)} trajectories")
    return np.array(sampled_trajs, dtype=object)


def save_sampled_to_temp(sampled_trajs: np.ndarray, temp_dir: Path) -> Path:
    """Save sampled trajectories to a temporary .npy file."""
    temp_file = temp_dir / "sampled_input.npy"
    np.save(temp_file, sampled_trajs)
    print(f"Saved sampled trajectories to {temp_file}")
    return temp_file


def run_segment_generation(
    input_file: Path,
    output_dir: Path,
    segment_ckpt: str,
    emb: str,
    city: str,
    data_root: str,
    edges: str,
    distance_table: str | None,
    greedy: bool,
    gpu: int,
    batch_size: int,
    max_steps: int,
    stop_at_destination: bool,
) -> tuple[Path, Path]:
    """Run segment-level generation (batch_generate_trajectories.py).

    Returns:
        (merged_output, length_info_file): paths to generated segments and length info
    """
    print("\n" + "=" * 80)
    print("Step 1: Generating road segments...")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_output = output_dir / "generated_segments.npy"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "batch_generate_trajectories.py"),
        "--generation-mode", "model",
        "--ckpt", segment_ckpt,
        "--emb", emb,
        "--city", city,
        "--data-root", data_root,
        "--input", str(input_file),
        "--output", str(output_dir / "segment_generation"),
        "--gpus", str(gpu),
        "--batch-size", str(batch_size),
        "--max-steps", str(max_steps),
        "--output-format", "dataset",
        "--edges", edges,
        "--output-file", str(merged_output),
    ]

    if distance_table:
        cmd += ["--distance-table", distance_table]
    if greedy:
        cmd.append("--greedy")

    if stop_at_destination:
        cmd.append("--stop-at-destination")

    print(f"Running command:\n{' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)

    if not merged_output.exists():
        raise RuntimeError(f"Segment generation failed: {merged_output} not found")

    # Inject original trajectory time and GPS count info into meta
    print(f"Injecting original trajectory info into segment metadata...")
    orig_data = np.load(input_file, allow_pickle=True)
    seg_data = np.load(merged_output, allow_pickle=True)

    if len(orig_data) != len(seg_data):
        raise RuntimeError(f"Length mismatch: orig={len(orig_data)}, seg={len(seg_data)}")

    # Modify segment metadata to include original trajectory duration and GPS count
    enhanced_seg_data = []
    for orig_item, seg_item in zip(orig_data, seg_data):
        # orig_item: [meta, roads_unique, s_seq, t_seq]
        # seg_item: [meta, points] where points is [N, 2]: [s, road_id]
        orig_meta = orig_item[0]
        seg_meta = seg_item[0]
        points = seg_item[1]

        # Get original trajectory info
        orig_t_seq = np.asarray(orig_item[3], dtype=np.float32)
        orig_s_seq = np.asarray(orig_item[2], dtype=np.float32)
        orig_duration = float(orig_t_seq.sum())
        orig_gps_count = len(orig_s_seq)

        # Create enhanced meta with duration info
        # Format: [id1, id2, duration, depart_time, gps_count]
        # batch_generate_gps_from_road_progress.py will read:
        #   - meta[2] for total_duration
        #   - meta[3] for depart_time
        #   - meta[4] for gps_count (used as decode_steps)
        enhanced_meta = np.array([
            seg_meta[0] if len(seg_meta) > 0 else '',
            seg_meta[1] if len(seg_meta) > 1 else '',
            orig_duration,  # meta[2]: total duration
            seg_meta[3] if len(seg_meta) > 3 else 0,  # meta[3]: depart_time
            orig_gps_count,  # meta[4]: original GPS count
        ], dtype=object)

        # Keep points in [N, 2] format for SAMPLE_FORMAT_ROADS_ENDPOINT_DATASET
        enhanced_seg_data.append([enhanced_meta, points])

    # Overwrite original segment data with enhanced version
    np.save(merged_output, np.array(enhanced_seg_data, dtype=object))
    print(f"Saved enhanced segment data to {merged_output}")

    print(f"\nSegment generation completed: {merged_output}")
    return merged_output, None


def run_point_generation(
    input_dir: Path,
    output_dir: Path,
    point_ckpt: str,
    emb: str,
    edges: str,
    gpu: int,
    batch_size: int,
    workers_per_gpu: int,
    use_full_model: bool = True,
) -> Path:
    """Run point-level generation.

    Args:
        use_full_model: If True, use batch_generate_gps_from_road_progress_full.py (for models
                        trained with train_road_progress_decoder_full.py using absolute remaining time).
                        If False (default), use batch_generate_gps_from_road_progress.py (for models
                        trained with train_road_progress_decoder_ddp.py using normalized remaining time [0,1]).
    """
    print("\n" + "=" * 80)
    print("Step 2: Generating GPS points...")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Choose the appropriate generation script
    if use_full_model:
        script_name = "batch_generate_gps_from_road_progress_full.py"
        print("Using full trajectory model generation script (absolute remaining time)")
    else:
        script_name = "batch_generate_gps_from_road_progress.py"
        print("Using original generation script (normalized remaining time)")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / script_name),
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--ckpt", point_ckpt,
        "--emb", emb,
        "--edges", edges,
        "--gpus", str(gpu),
        "--workers-per-gpu", str(workers_per_gpu),
        "--batch-size", str(batch_size),
        "--log-every", "10",
    ]

    print(f"Running command:\n{' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)

    # Find generated output file
    generated_files = list(output_dir.glob("*.npy"))
    if len(generated_files) == 0:
        raise RuntimeError(f"Point generation failed: no .npy files found in {output_dir}")

    # Merge all generated files if multiple
    if len(generated_files) == 1:
        final_output = generated_files[0]
    else:
        print(f"Merging {len(generated_files)} generated files...")
        all_trajs = []
        for f in generated_files:
            arr = np.load(f, allow_pickle=True)
            all_trajs.extend(list(arr))
        final_output = output_dir / "generated_gps_merged.npy"
        np.save(final_output, np.array(all_trajs, dtype=object))

    print(f"\nPoint generation completed: {final_output}")
    return final_output


def main():
    parser = argparse.ArgumentParser(
        description="Sample trajectories and generate road segments + GPS points"
    )

    # Input/output
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing original trajectory .npy files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for sampled and generated trajectories")
    parser.add_argument("--num-samples", type=int, required=True,
                        help="Number of trajectories to sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")

    # Model checkpoints
    parser.add_argument("--segment-ckpt", type=str, required=True,
                        help="Path to segment-level generator checkpoint")
    parser.add_argument("--point-ckpt", type=str, required=True,
                        help="Path to point-level generator checkpoint")
    parser.add_argument("--emb", type=str, required=True,
                        help="Path to road embeddings")

    # City and data
    parser.add_argument("--city", type=str, required=True,
                        help="City name (e.g., chengdu, xian)")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory for city data")
    parser.add_argument("--edges", type=str, required=True,
                        help="Path to edges.csv")
    parser.add_argument("--distance-table", type=str, default=None,
                        help="Path to distance table .npy (optional; mutually exclusive with --greedy)")
    parser.add_argument("--greedy", action="store_true",
                        help="Greedy decoding: pick highest-scoring road segment (no graph constraint)")

    # Generation parameters
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--segment-batch-size", type=int, default=2048,
                        help="Batch size for segment generation")
    parser.add_argument("--point-batch-size", type=int, default=2048,
                        help="Batch size for point generation")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Max steps for segment generation")
    parser.add_argument("--workers-per-gpu", type=int, default=1,
                        help="Workers per GPU for point generation")
    parser.add_argument("--stop-at-destination", action="store_true", default=True,
                        help="Stop generation when reaching destination (default: True)")
    parser.add_argument("--no-stop-at-destination", action="store_false", dest="stop_at_destination",
                        help="Do not stop at destination")
    parser.add_argument("--use-full-model", action="store_true", default=False,
                        help="Use full trajectory model with absolute remaining time (for train_road_progress_decoder_full.py)")
    parser.add_argument("--use-normalized-time", action="store_false", dest="use_full_model",
                        help="Use normalized remaining time [0,1] (for train_road_progress_decoder_ddp.py, default)")
    # GPS point generation always uses stop_at_end=True and pads short sequences with the
    # endpoint value, so no --force-length flag is needed.

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: Sample trajectories
    print("=" * 80)
    print("Step 0: Sampling trajectories from original dataset")
    print("=" * 80)
    sampled_trajs = sample_trajectories(input_dir, args.num_samples, args.seed)

    # Save sampled original trajectories
    sampled_output = output_dir / "sampled_original.npy"
    np.save(sampled_output, sampled_trajs)
    print(f"Saved sampled original trajectories to: {sampled_output}")

    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Save sampled trajectories to temp file for generation
        temp_input = save_sampled_to_temp(sampled_trajs, temp_dir)

        # Step 1: Generate road segments
        segment_output, length_info_file = run_segment_generation(
            input_file=temp_input,
            output_dir=temp_dir / "segment_output",
            segment_ckpt=args.segment_ckpt,
            emb=args.emb,
            city=args.city,
            data_root=args.data_root,
            edges=args.edges,
            distance_table=args.distance_table,
            greedy=args.greedy,
            gpu=args.gpu,
            batch_size=args.segment_batch_size,
            max_steps=args.max_steps,
            stop_at_destination=args.stop_at_destination,
        )

        # Step 2: Generate GPS points
        point_output = run_point_generation(
            input_dir=temp_dir / "segment_output",
            output_dir=temp_dir / "point_output",
            point_ckpt=args.point_ckpt,
            emb=args.emb,
            edges=args.edges,
            gpu=args.gpu,
            batch_size=args.point_batch_size,
            workers_per_gpu=args.workers_per_gpu,
            use_full_model=args.use_full_model,
        )

        # Copy final generated trajectories to output directory
        final_generated = output_dir / "generated_trajectories.npy"
        generated_trajs = np.load(point_output, allow_pickle=True)
        np.save(final_generated, generated_trajs)
        print(f"\nCopied final generated trajectories to: {final_generated}")

    # Summary
    print("\n" + "=" * 80)
    print("Generation completed successfully!")
    print("=" * 80)
    print(f"Sampled original trajectories: {sampled_output} ({len(sampled_trajs)} trajectories)")
    print(f"Generated trajectories:        {final_generated} ({len(generated_trajs)} trajectories)")
    print(f"Output directory:              {output_dir}")


if __name__ == "__main__":
    main()
