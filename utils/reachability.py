"""Reachability table C[i][j]: shortest-path distance from road i to road j.

Build once offline with build_reachability_table.py, then load here for RL.

Storage format: float32 numpy array [N, N], saved as .npy.
Unreachable pairs may use either a negative sentinel (for generation, typically -1)
or a non-finite sentinel such as np.inf.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

import numpy as np
import torch


def _build_csr_graph(
    *,
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
):
    from scipy.sparse import csr_matrix

    src = edge_index[0].cpu().numpy().astype(np.int32)
    dst = edge_index[1].cpu().numpy().astype(np.int32)

    if edge_weight is not None:
        w = edge_weight.cpu().numpy().reshape(-1).astype(np.float64)
        w = np.maximum(w, 1e-9)
    else:
        w = np.ones(len(src), dtype=np.float64)

    return csr_matrix((w, (src, dst)), shape=(num_nodes, num_nodes))


def reconstruct_shortest_path(
    predecessors: np.ndarray,
    *,
    src: int,
    dst: int,
) -> list[int]:
    """Reconstruct one shortest path from src to dst using scipy predecessors."""
    src = int(src)
    dst = int(dst)

    if predecessors.ndim == 1:
        predecessors_row = predecessors
    elif predecessors.ndim == 2:
        predecessors_row = predecessors[src]
    else:
        raise ValueError(f"Expected predecessors to be 1D or 2D, got shape {predecessors.shape}")

    if src == dst:
        return [src]

    prev = int(predecessors_row[dst])
    if prev < 0:
        return []

    path = [dst]
    cur = dst
    while cur != src:
        cur = int(predecessors_row[cur])
        if cur < 0:
            return []
        path.append(cur)

    path.reverse()
    return path


def shortest_path_between_nodes(
    *,
    edge_index: torch.Tensor,
    num_nodes: int,
    src: int,
    dst: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> tuple[list[int], float]:
    """Return one shortest path and its distance between two road-segment nodes."""
    from scipy.sparse.csgraph import shortest_path

    src = int(src)
    dst = int(dst)
    if src < 0 or src >= int(num_nodes) or dst < 0 or dst >= int(num_nodes):
        raise ValueError(f"src/dst out of range for num_nodes={num_nodes}: src={src}, dst={dst}")

    graph = _build_csr_graph(edge_index=edge_index, num_nodes=num_nodes, edge_weight=edge_weight)
    dist, predecessors = shortest_path(
        graph,
        method="D",
        directed=True,
        unweighted=False,
        indices=src,
        return_predecessors=True,
    )
    distance = float(dist[dst])
    if not np.isfinite(distance):
        return [], float("inf")

    path = reconstruct_shortest_path(predecessors, src=src, dst=dst)
    if not path:
        return [], float("inf")
    return path, distance


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_reachability_table(
    *,
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
    sentinel: float = float("inf"),
    show_progress: bool = False,
) -> np.ndarray:
    """Compute all-pairs shortest-path distance matrix C [N, N].

    Args:
        edge_index: LongTensor [2, E] directed edges (src, dst).
        num_nodes:  N, number of road-segment nodes.
        edge_weight: optional FloatTensor [E] or [E, 1] with positive weights.
                     If None, hop-count (weight=1) is used.
        sentinel:   value for unreachable pairs (default inf).
        show_progress: whether to print a source-by-source progress bar.

    Returns:
        C: float32 ndarray [N, N].  C[i, i] == 0.  C[i, j] == sentinel if
           j is unreachable from i.

    Notes:
        Uses Dijkstra from every source via scipy.sparse.csgraph.
        For N=4198 this takes ~10-30 s on CPU.
    """
    from scipy.sparse.csgraph import shortest_path

    graph = _build_csr_graph(edge_index=edge_index, num_nodes=num_nodes, edge_weight=edge_weight)

    if not show_progress:
        C = shortest_path(graph, method="D", directed=True, unweighted=False)
        C = C.astype(np.float32)
        C[~np.isfinite(C)] = float(sentinel)
        return C

    C = np.empty((num_nodes, num_nodes), dtype=np.float32)
    bar_width = 30
    t0 = time.time()
    for src_idx in range(num_nodes):
        dist = shortest_path(
            graph,
            method="D",
            directed=True,
            unweighted=False,
            indices=src_idx,
            return_predecessors=False,
        )
        row = np.asarray(dist, dtype=np.float32).reshape(-1)
        row[~np.isfinite(row)] = float(sentinel)
        C[src_idx] = row

        completed = src_idx + 1
        filled = int(bar_width * completed / max(1, num_nodes))
        bar = "#" * filled + "-" * (bar_width - filled)
        elapsed = time.time() - t0
        rate = completed / elapsed if elapsed > 0 else 0.0
        remaining = (num_nodes - completed) / rate if rate > 0 else 0.0
        print(
            f"\rprogress [{bar}] {completed}/{num_nodes} "
            f"elapsed={elapsed:.1f}s eta={remaining:.1f}s",
            end="",
            flush=True,
        )

    print()
    return C


def build_binary_reachability_matrix(
    C: np.ndarray,
    *,
    include_self: bool = True,
) -> np.ndarray:
    """Convert a distance reachability table into a binary reachability matrix."""
    R = (np.isfinite(C) & (C >= 0)).astype(np.uint8)
    if not include_self:
        np.fill_diagonal(R, 0)
    return R


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_reachability_table(C: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), C)


def load_reachability_table(path: str | Path) -> np.ndarray:
    """Load C table saved by save_reachability_table.

    Returns float32 ndarray [N, N].
    """
    C = np.load(str(path)).astype(np.float32)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {C.shape}")
    return C


# ---------------------------------------------------------------------------
# Torch wrapper for RL use
# ---------------------------------------------------------------------------

class ReachabilityTable:
    """Thin wrapper around C ndarray for fast batched RL queries.

    Usage:
        rt = ReachabilityTable.from_file("checkpoints/xian_C.npy", device)
        dist = rt.dist(cur_roads, dest)   # [B] float tensor
        delta = rt.progress(cur, nxt, dest)  # [B, K] float tensor
    """

    def __init__(self, C: np.ndarray, device: torch.device) -> None:
        self._C = torch.from_numpy(C).to(device)  # [N, N] float32
        self.num_nodes = int(C.shape[0])
        self.device = device

    @classmethod
    def from_file(cls, path: str | Path, device: torch.device) -> "ReachabilityTable":
        C = load_reachability_table(path)
        return cls(C, device)

    def dist(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Return C[src, dst].

        src: LongTensor [...] road ids
        dst: LongTensor [...] road ids (broadcastable with src)
        Returns: FloatTensor [...] distances (sentinel preserved for unreachable).
        """
        src = src.to(self.device).long()
        dst = dst.to(self.device).long()
        return self._C[src, dst]

    def progress(
        self,
        cur: torch.Tensor,
        nxt: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        """ΔC = C[cur, dst] - C[nxt, dst]: positive means nxt is closer to dst.

        cur: LongTensor [B]
        nxt: LongTensor [B, K]  (K neighbor candidates)
        dst: LongTensor [B]

        Returns: FloatTensor [B, K]
        """
        b = int(cur.shape[0])
        k = int(nxt.shape[1])
        cur_d = self._C[cur.to(self.device), dst.to(self.device)]          # [B]
        nxt_d = self._C[nxt.to(self.device).reshape(-1),
                        dst.to(self.device).unsqueeze(1).expand(b, k)
                        .reshape(-1)].reshape(b, k)                         # [B, K]
        return cur_d.unsqueeze(1) - nxt_d                                   # [B, K]

    def is_reachable(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Return BoolTensor: True where C[src, dst] is a valid non-negative distance."""
        dist = self.dist(src, dst)
        return torch.isfinite(dist) & (dist >= 0)
