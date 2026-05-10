"""Offline script to build and save a reachability artifact.

Usage:
    python experiments/build_reachability_table.py \
        --city xian \
        --data-root data \
        --output checkpoints/xian_C.npy \
        [--use-length]   # weight edges by road length (default: hop count)

By default this script saves a float32 shortest-path distance table [N, N],
where unreachable pairs are stored as -1.

With --binary, it instead saves a uint8 binary reachability matrix [N, N],
where C[i][j] = 1 means road segment i can reach road segment j, else 0.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from utils.graph_builder import build_city_pyg_graph
from utils.reachability import build_binary_reachability_matrix, build_reachability_table, save_reachability_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and save a shortest-path or binary reachability matrix for a city."
    )
    parser.add_argument("--city", type=str, default="xian", choices=["xian", "chengdu"])
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--min-unique-len", type=int, default=5, help="Minimum unique-road trajectory length used when rebuilding the city graph")
    parser.add_argument(
        "--output", type=str, default="",
        help="Output .npy path. Default: checkpoints/{city}_reachable.npy when --binary else checkpoints/{city}_C.npy"
    )
    parser.add_argument(
        "--binary", action="store_true",
        help="Save a binary reachability matrix where 1 means reachable and 0 means unreachable."
    )
    parser.add_argument(
        "--use-length", action="store_true",
        help="Weight edges by road length instead of hop count."
    )
    parser.add_argument(
        "--show-progress", action="store_true",
        help="Print a source-by-source progress bar while building the shortest-path table."
    )
    args = parser.parse_args()

    default_name = f"{args.city}_reachable.npy" if args.binary else f"{args.city}_C.npy"
    output = args.output or str(Path(args.data_root).parent / "checkpoints" / default_name)

    print(f"Building {'binary reachability matrix' if args.binary else 'reachability table'} for city={args.city}")
    if not args.binary:
        print("  unreachable_sentinel: -1")
    print(f"  data_root      : {args.data_root}")
    print(f"  min_unique_len : {args.min_unique_len}")
    print(f"  use_length     : {args.use_length}")
    print(f"  binary         : {args.binary}")
    print(f"  output         : {output}")
    print(f"  progress       : {args.show_progress}")

    data, _ = build_city_pyg_graph(
        city=args.city,
        data_root=args.data_root,
        min_unique_len=args.min_unique_len,
        include_node_heat=False,
    )
    num_nodes = int(data.num_nodes)
    edge_index = data.edge_index  # [2, E]
    print(f"  num_nodes      : {num_nodes}")
    print(f"  num_edges      : {int(edge_index.shape[1])}")

    edge_weight = None
    if args.use_length:
        import pandas as pd

        edges_path = Path(args.data_root) / f"{args.city}_road_network" / "edges.csv"
        edges_df = pd.read_csv(edges_path)
        fid_to_length = dict(zip(edges_df["fid"].astype(int), edges_df["length"].astype(float)))
        src_fids = edge_index[0].tolist()
        weights = np.array([fid_to_length.get(int(f), 1.0) for f in src_fids], dtype=np.float32)
        weights = np.maximum(weights, 1e-3)
        edge_weight = torch.from_numpy(weights)
        print(f"  edge_weight    : length-based, mean={float(weights.mean()):.2f}")

    print("Computing all-pairs shortest paths (Dijkstra)...")
    t0 = time.time()
    C_dist = build_reachability_table(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weight=edge_weight,
        sentinel=-1.0,
        show_progress=bool(args.show_progress),
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")

    reachable = C_dist >= 0
    reachable_frac = float(reachable.sum() - num_nodes) / max(1, num_nodes * (num_nodes - 1))
    print(f"  reachable pairs (excl. diagonal): {reachable_frac*100:.1f}%")

    if args.binary:
        C = build_binary_reachability_matrix(C_dist)
        print(f"  binary stats: diag_sum={int(np.trace(C))} unique={np.unique(C).tolist()}")
    else:
        C = C_dist
        unreachable_count = int((C < 0).sum())
        print(f"  C[0,0]={C[0,0]:.4f}  C[0,1]={C[0,1]:.4f}  C[1,0]={C[1,0]:.4f}  unreachable={unreachable_count}")

    save_reachability_table(C, output)
    print(f"Saved to: {output}  shape={C.shape}  dtype={C.dtype}")


if __name__ == "__main__":
    main()
