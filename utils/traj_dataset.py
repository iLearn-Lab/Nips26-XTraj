from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from utils.graph_builder import (
    _iter_city_npy_paths,
    iter_city_trajectories,
    iter_city_trajectories_with_depart_time,
    iter_city_trajectories_with_total_duration,
    iter_city_trajectories_with_depart_time_and_total_duration,
)

StartMode = Literal["fixed", "random", "sliding"]


@dataclass
class TrajSegmentSpec:
    segment_len: int = 64
    start_mode: StartMode = "random"
    start_idx: int = 0
    stride: int = 1
    seed: int = 0


def pad_collate_1d(seqs: Sequence[torch.Tensor], pad_value: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of 1D LongTensors into [B,T] plus attention mask."""
    if len(seqs) == 0:
        return torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.bool)

    lens = [int(s.numel()) for s in seqs]
    tmax = max(lens)
    b = len(seqs)

    x = torch.full((b, tmax), pad_value, dtype=torch.long)
    m = torch.zeros((b, tmax), dtype=torch.bool)

    for i, s in enumerate(seqs):
        l = int(s.numel())
        if l == 0:
            continue
        x[i, :l] = s
        m[i, :l] = True

    return x, m


def pad_collate_1d_with_depart_time(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    pad_value: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of (road_ids, depart_time) into [B,T] plus attention mask."""
    if len(batch) == 0:
        return (
            torch.empty((0, 0), dtype=torch.long),
            torch.empty((0, 0), dtype=torch.bool),
            torch.empty((0,), dtype=torch.long),
        )

    seqs = [s for (s, _) in batch]
    depart = torch.stack([t for (_, t) in batch], dim=0).view(-1)
    x, m = pad_collate_1d(seqs, pad_value=pad_value)
    return x, m, depart


def pad_collate_1d_with_total_duration(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    pad_value: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of (road_ids, total_duration)."""
    if len(batch) == 0:
        return (
            torch.empty((0, 0), dtype=torch.long),
            torch.empty((0, 0), dtype=torch.bool),
            torch.empty((0,), dtype=torch.float32),
        )

    seqs = [s for (s, _) in batch]
    total_duration = torch.stack([d for (_, d) in batch], dim=0).view(-1)
    x, m = pad_collate_1d(seqs, pad_value=pad_value)
    return x, m, total_duration


def pad_collate_1d_with_destination(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    pad_value: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of (road_ids, destination_road) into [B,T] plus mask."""
    if len(batch) == 0:
        return (
            torch.empty((0, 0), dtype=torch.long),
            torch.empty((0, 0), dtype=torch.bool),
            torch.empty((0,), dtype=torch.long),
        )

    seqs = [s for (s, _) in batch]
    destination = torch.stack([d for (_, d) in batch], dim=0).view(-1)
    x, m = pad_collate_1d(seqs, pad_value=pad_value)
    return x, m, destination


def pad_collate_1d_with_start_and_destination(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    pad_value: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of (road_ids, start_road, destination_road)."""
    if len(batch) == 0:
        return (
            torch.empty((0, 0), dtype=torch.long),
            torch.empty((0, 0), dtype=torch.bool),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
        )

    seqs = [s for (s, _, _) in batch]
    start = torch.stack([u for (_, u, _) in batch], dim=0).view(-1)
    destination = torch.stack([d for (_, _, d) in batch], dim=0).view(-1)
    x, m = pad_collate_1d(seqs, pad_value=pad_value)
    return x, m, start, destination


def pad_collate_1d_with_start_and_destination_and_total_duration(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    pad_value: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of (road_ids, start_road, destination_road, total_duration)."""
    if len(batch) == 0:
        return (
            torch.empty((0, 0), dtype=torch.long),
            torch.empty((0, 0), dtype=torch.bool),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    seqs = [s for (s, _, _, _) in batch]
    start = torch.stack([u for (_, u, _, _) in batch], dim=0).view(-1)
    destination = torch.stack([d for (_, _, d, _) in batch], dim=0).view(-1)
    total_duration = torch.stack([t for (_, _, _, t) in batch], dim=0).view(-1)
    x, m = pad_collate_1d(seqs, pad_value=pad_value)
    return x, m, start, destination, total_duration


def pad_collate_1d_with_start_and_destination_and_depart_time(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    pad_value: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of (road_ids, start_road, destination_road, depart_time)."""
    if len(batch) == 0:
        return (
            torch.empty((0, 0), dtype=torch.long),
            torch.empty((0, 0), dtype=torch.bool),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
        )

    seqs = [s for (s, _, _, _) in batch]
    start = torch.stack([u for (_, u, _, _) in batch], dim=0).view(-1)
    destination = torch.stack([d for (_, _, d, _) in batch], dim=0).view(-1)
    depart = torch.stack([t for (_, _, _, t) in batch], dim=0).view(-1)
    x, m = pad_collate_1d(seqs, pad_value=pad_value)
    return x, m, start, destination, depart


def pad_collate_1d_with_start_and_destination_and_depart_time_and_total_duration(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    pad_value: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of (road_ids, start_road, destination_road, depart_time, total_duration)."""
    if len(batch) == 0:
        return (
            torch.empty((0, 0), dtype=torch.long),
            torch.empty((0, 0), dtype=torch.bool),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    seqs = [s for (s, _, _, _, _) in batch]
    start = torch.stack([u for (_, u, _, _, _) in batch], dim=0).view(-1)
    destination = torch.stack([d for (_, _, d, _, _) in batch], dim=0).view(-1)
    depart = torch.stack([t for (_, _, _, t, _) in batch], dim=0).view(-1)
    total_duration = torch.stack([v for (_, _, _, _, v) in batch], dim=0).view(-1)
    x, m = pad_collate_1d(seqs, pad_value=pad_value)
    return x, m, start, destination, depart, total_duration


def pad_collate_1d_with_destination_and_total_duration(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    pad_value: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of (road_ids, destination_road, total_duration)."""
    if len(batch) == 0:
        return (
            torch.empty((0, 0), dtype=torch.long),
            torch.empty((0, 0), dtype=torch.bool),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    seqs = [s for (s, _, _) in batch]
    destination = torch.stack([d for (_, d, _) in batch], dim=0).view(-1)
    total_duration = torch.stack([t for (_, _, t) in batch], dim=0).view(-1)
    x, m = pad_collate_1d(seqs, pad_value=pad_value)
    return x, m, destination, total_duration


def pad_collate_1d_with_destination_and_depart_time(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    pad_value: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of (road_ids, destination_road, depart_time)."""
    if len(batch) == 0:
        return (
            torch.empty((0, 0), dtype=torch.long),
            torch.empty((0, 0), dtype=torch.bool),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
        )

    seqs = [s for (s, _, _) in batch]
    destination = torch.stack([d for (_, d, _) in batch], dim=0).view(-1)
    depart = torch.stack([t for (_, _, t) in batch], dim=0).view(-1)
    x, m = pad_collate_1d(seqs, pad_value=pad_value)
    return x, m, destination, depart


def pad_collate_1d_with_depart_time_and_total_duration(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    pad_value: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of (road_ids, depart_time, total_duration)."""
    if len(batch) == 0:
        return (
            torch.empty((0, 0), dtype=torch.long),
            torch.empty((0, 0), dtype=torch.bool),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    seqs = [s for (s, _, _) in batch]
    depart = torch.stack([d for (_, d, _) in batch], dim=0).view(-1)
    total_duration = torch.stack([t for (_, _, t) in batch], dim=0).view(-1)
    x, m = pad_collate_1d(seqs, pad_value=pad_value)
    return x, m, depart, total_duration


def pad_collate_1d_with_destination_and_depart_time_and_total_duration(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    pad_value: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of (road_ids, destination_road, depart_time, total_duration)."""
    if len(batch) == 0:
        return (
            torch.empty((0, 0), dtype=torch.long),
            torch.empty((0, 0), dtype=torch.bool),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    seqs = [s for (s, _, _, _) in batch]
    destination = torch.stack([d for (_, d, _, _) in batch], dim=0).view(-1)
    depart = torch.stack([t for (_, _, t, _) in batch], dim=0).view(-1)
    total_duration = torch.stack([u for (_, _, _, u) in batch], dim=0).view(-1)
    x, m = pad_collate_1d(seqs, pad_value=pad_value)
    return x, m, destination, depart, total_duration


class TrajectorySegmentDataset(Dataset):
    """Trajectory dataset that returns subsequences of road ids.

    Optional returned conditions:
    - destination road = last road id of the full trajectory
    - depart_time from traj meta
    """

    def __init__(
        self,
        *,
        traj_dir: str,
        spec: TrajSegmentSpec,
        min_unique_len: int = 5,
        max_trajs: int = 0,
        return_depart_time: bool = False,
        return_start: bool = False,
        return_destination: bool = False,
        return_total_duration: bool = False,
        meta_time_index: int = 3,
    ) -> None:
        self.spec = spec
        self.return_depart_time = bool(return_depart_time)
        self.return_start = bool(return_start)
        self.return_destination = bool(return_destination)
        self.return_total_duration = bool(return_total_duration)
        self.file_paths = [str(Path(p)) for p in _iter_city_npy_paths(traj_dir)]
        self.file_to_day = {str(Path(p)): Path(p).stem for p in self.file_paths}

        self.trajs: List[List[int]] = []
        self.depart_times: List[int] = []
        self.total_durations: List[float] = []
        self.traj_file_ids: List[int] = []
        self.traj_starts: List[int] = []
        self.traj_destinations: List[int] = []

        if self.return_depart_time and self.return_total_duration:
            for file_id, npy_path in enumerate(self.file_paths):
                it = iter_city_trajectories_with_depart_time_and_total_duration(
                    traj_dir,
                    min_unique_len=min_unique_len,
                    meta_time_index=int(meta_time_index),
                    npy_paths=[npy_path],
                )
                for road_ids, depart_time, total_duration in it:
                    self.trajs.append(road_ids)
                    self.depart_times.append(int(depart_time))
                    self.total_durations.append(float(total_duration))
                    self.traj_file_ids.append(int(file_id))
                    self.traj_starts.append(int(road_ids[0]))
                    self.traj_destinations.append(int(road_ids[-1]))
                    if max_trajs and len(self.trajs) >= max_trajs:
                        break
                if max_trajs and len(self.trajs) >= max_trajs:
                    break
        elif self.return_depart_time:
            for file_id, npy_path in enumerate(self.file_paths):
                it = iter_city_trajectories_with_depart_time(
                    traj_dir,
                    min_unique_len=min_unique_len,
                    meta_time_index=int(meta_time_index),
                    npy_paths=[npy_path],
                )
                for road_ids, depart_time in it:
                    self.trajs.append(road_ids)
                    self.depart_times.append(int(depart_time))
                    self.traj_file_ids.append(int(file_id))
                    self.traj_starts.append(int(road_ids[0]))
                    self.traj_destinations.append(int(road_ids[-1]))
                    if max_trajs and len(self.trajs) >= max_trajs:
                        break
                if max_trajs and len(self.trajs) >= max_trajs:
                    break
        elif self.return_total_duration:
            for file_id, npy_path in enumerate(self.file_paths):
                it = iter_city_trajectories_with_total_duration(
                    traj_dir,
                    min_unique_len=min_unique_len,
                    npy_paths=[npy_path],
                )
                for road_ids, total_duration in it:
                    self.trajs.append(road_ids)
                    self.total_durations.append(float(total_duration))
                    self.traj_file_ids.append(int(file_id))
                    self.traj_starts.append(int(road_ids[0]))
                    self.traj_destinations.append(int(road_ids[-1]))
                    if max_trajs and len(self.trajs) >= max_trajs:
                        break
                if max_trajs and len(self.trajs) >= max_trajs:
                    break
        else:
            for file_id, npy_path in enumerate(self.file_paths):
                for t in iter_city_trajectories(traj_dir, min_unique_len=min_unique_len, npy_paths=[npy_path]):
                    self.trajs.append(t)
                    self.traj_file_ids.append(int(file_id))
                    self.traj_starts.append(int(t[0]))
                    self.traj_destinations.append(int(t[-1]))
                    if max_trajs and len(self.trajs) >= max_trajs:
                        break
                if max_trajs and len(self.trajs) >= max_trajs:
                    break

        if len(self.trajs) == 0:
            raise ValueError(f"No trajectories found under: {traj_dir}")

        if self.return_depart_time and len(self.depart_times) != len(self.trajs):
            raise RuntimeError("depart_times length mismatch")
        if self.return_total_duration and len(self.total_durations) != len(self.trajs):
            raise RuntimeError("total_durations length mismatch")
        if len(self.traj_starts) != len(self.trajs):
            raise RuntimeError("traj_starts length mismatch")
        if len(self.traj_destinations) != len(self.trajs):
            raise RuntimeError("traj_destinations length mismatch")

        self._rng = random.Random(spec.seed)

        self._sliding_index: List[Tuple[int, int]] = []
        if self.spec.start_mode == "sliding":
            seg_len = int(self.spec.segment_len)
            stride = max(int(self.spec.stride), 1)
            for ti, traj in enumerate(self.trajs):
                if len(traj) < 2:
                    continue
                max_start = max(0, len(traj) - seg_len)
                for s in range(0, max_start + 1, stride):
                    self._sliding_index.append((ti, s))
            if len(self._sliding_index) == 0:
                raise ValueError("Sliding mode produced 0 segments; reduce segment_len or stride")

    def __len__(self) -> int:
        if self.spec.start_mode == "sliding":
            return len(self._sliding_index)
        return len(self.trajs)

    def _slice(self, traj: List[int], start: int) -> List[int]:
        seg_len = int(self.spec.segment_len)
        start = int(start)
        if start < 0:
            start = 0
        end = min(len(traj), start + seg_len)
        return traj[start:end]

    def _pack_item(self, seg: List[int], traj_index: int):
        x = torch.tensor(seg, dtype=torch.long)
        out = [x]

        if self.return_start:
            out.append(torch.tensor(int(self.traj_starts[traj_index]), dtype=torch.long))

        if self.return_destination:
            out.append(torch.tensor(int(self.traj_destinations[traj_index]), dtype=torch.long))

        if self.return_depart_time:
            out.append(torch.tensor(int(self.depart_times[traj_index]), dtype=torch.long))

        if self.return_total_duration:
            out.append(torch.tensor(float(self.total_durations[traj_index]), dtype=torch.float32))

        if len(out) == 1:
            return out[0]
        return tuple(out)

    def __getitem__(self, idx: int):
        if self.spec.start_mode == "sliding":
            ti, start = self._sliding_index[idx]
            traj = self.trajs[ti]
            seg = self._slice(traj, start)
            return self._pack_item(seg, ti)

        traj = self.trajs[idx]
        if self.spec.start_mode == "fixed":
            start = int(self.spec.start_idx)
        elif self.spec.start_mode == "random":
            seg_len = int(self.spec.segment_len)
            max_start = max(0, len(traj) - max(seg_len, 2))
            start = self._rng.randint(0, max_start)
        else:
            raise ValueError(f"Unknown start_mode: {self.spec.start_mode}")

        seg = self._slice(traj, start)
        return self._pack_item(seg, idx)
