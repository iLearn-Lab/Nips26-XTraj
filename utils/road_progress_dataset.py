"""Dataset for Road-Progress AR Decoder training.

Loads samples from the resampled road-progress directory:
  sample = [meta, roads_unique, s_seq, t_seq]

Each __getitem__ returns a sliding-window segment of the s sequence,
along with the corresponding roads_unique, depart_time, and normalized
full-trajectory remaining-time sequence aligned with the sampled s values.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


_EPS = 1e-6


def _load_road_progress_dir(
    data_dir: str,
    min_roads: int = 3,
    min_gps: int = 10,
    max_trajs: int = 0,
    meta_time_index: int = 3,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int], List[float], List[float], List[float]]:
    """Load all .npy files from data_dir.

    Returns:
        roads_list:    list of int32 arrays [M]
        s_list:        list of float32 arrays [T]
        t_list:        list of float32 arrays [T]
        depart_list:   list of int depart_time bins
        s_start_list:  list of float full-trajectory s_seq[0]
        s_end_list:    list of float full-trajectory s_seq[-1]
        duration_list: list of float total trajectory duration in seconds
    """
    roads_list: List[np.ndarray] = []
    s_list: List[np.ndarray] = []
    t_list: List[np.ndarray] = []
    depart_list: List[int] = []
    s_start_list: List[float] = []
    s_end_list: List[float] = []
    duration_list: List[float] = []

    paths = sorted(Path(data_dir).glob("*.npy"))
    if not paths:
        raise ValueError(f"No .npy files found in: {data_dir}")

    for p in paths:
        arr = np.load(str(p), allow_pickle=True)
        for sample in arr:
            try:
                meta = sample[0]
                roads = np.asarray(sample[1], dtype=np.int32).reshape(-1)
                s_seq = np.asarray(sample[2], dtype=np.float32).reshape(-1)
                t_seq = np.asarray(sample[3], dtype=np.float32).reshape(-1)
            except Exception:
                continue

            if len(roads) < min_roads or len(s_seq) < min_gps or len(t_seq) != len(s_seq):
                continue

            try:
                depart = int(meta[meta_time_index])
            except Exception:
                depart = 0

            roads_list.append(roads)
            s_list.append(s_seq)
            t_list.append(t_seq)
            depart_list.append(depart)
            s_start_list.append(float(s_seq[0]))
            s_end_list.append(float(s_seq[-1]))
            duration_list.append(float(t_seq.sum()))

            if max_trajs > 0 and len(roads_list) >= max_trajs:
                return roads_list, s_list, t_list, depart_list, s_start_list, s_end_list, duration_list

    return roads_list, s_list, t_list, depart_list, s_start_list, s_end_list, duration_list


def _elapsed_time_from_t_seq(t_seq: np.ndarray) -> np.ndarray:
    t = np.asarray(t_seq, dtype=np.float32).reshape(-1)
    if len(t) == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.cumsum(t, dtype=np.float32)


def _normalized_remaining_time(elapsed_seq: np.ndarray, total_duration: float) -> np.ndarray:
    elapsed = np.asarray(elapsed_seq, dtype=np.float32).reshape(-1)
    if len(elapsed) == 0:
        return np.zeros((0,), dtype=np.float32)
    denom = max(float(total_duration), _EPS)
    remaining = (float(total_duration) - elapsed) / denom
    return np.clip(remaining, 0.0, 1.0).astype(np.float32)


def pad_roads(
    roads_list: List[torch.Tensor], pad_value: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad road sequences to [B, M_max].

    Returns:
        roads: LongTensor [B, M_max]
        mask:  BoolTensor [B, M_max] True=valid
    """
    m_max = max(r.numel() for r in roads_list)
    b = len(roads_list)
    roads = torch.full((b, m_max), pad_value, dtype=torch.long)
    mask = torch.zeros((b, m_max), dtype=torch.bool)
    for i, r in enumerate(roads_list):
        l = r.numel()
        roads[i, :l] = r
        mask[i, :l] = True
    return roads, mask


def pad_s_seq(
    s_list: List[torch.Tensor], pad_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad 1D float sequences to [B, T_max].

    Returns:
        s:    FloatTensor [B, T_max]
        mask: BoolTensor [B, T_max] True=valid
    """
    t_max = max(s.numel() for s in s_list)
    b = len(s_list)
    s_out = torch.full((b, t_max), pad_value, dtype=torch.float32)
    mask = torch.zeros((b, t_max), dtype=torch.bool)
    for i, s in enumerate(s_list):
        l = s.numel()
        s_out[i, :l] = s
        mask[i, :l] = True
    return s_out, mask


def road_progress_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, float, float, float]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for RoadProgressDataset.

    Each item:
        (roads [M], s_seg [T], remaining_time_seg [T], road_mask [M],
         depart_time int, s_start float, s_end float, total_duration float)

    Returns:
        roads:          LongTensor  [B, M_max]
        road_mask:      BoolTensor  [B, M_max]
        s_seq:          FloatTensor [B, T_max]
        s_mask:         BoolTensor  [B, T_max]
        remaining_time: FloatTensor [B, T_max]
        depart_time:    LongTensor  [B]
        s_start:        FloatTensor [B]
        s_end:          FloatTensor [B]
        total_duration: FloatTensor [B]
    """
    roads_tensors = [item[0] for item in batch]
    s_tensors = [item[1] for item in batch]
    remaining_time_tensors = [item[2] for item in batch]
    departs = torch.tensor([item[4] for item in batch], dtype=torch.long)
    s_starts = torch.tensor([item[5] for item in batch], dtype=torch.float32)
    s_ends = torch.tensor([item[6] for item in batch], dtype=torch.float32)
    total_durations = torch.tensor([item[7] for item in batch], dtype=torch.float32)

    roads, road_mask = pad_roads(roads_tensors)
    s_seq, s_mask = pad_s_seq(s_tensors)
    remaining_time_seq, remaining_time_mask = pad_s_seq(remaining_time_tensors)
    if not torch.equal(s_mask, remaining_time_mask):
        raise RuntimeError("remaining_time_seq mask must match s_seq mask")

    return roads, road_mask, s_seq, s_mask, remaining_time_seq, departs, s_starts, s_ends, total_durations


class RoadProgressDataset(Dataset):
    """Dataset that yields full trajectories for training.

    Each sample is a complete trajectory with all GPS points.
    """

    def __init__(
        self,
        *,
        data_dir: str,
        min_roads: int = 3,
        min_gps: int = 10,
        max_trajs: int = 0,
        meta_time_index: int = 3,
    ) -> None:
        roads_list, s_list, t_list, depart_list, s_start_list, s_end_list, duration_list = _load_road_progress_dir(
            data_dir,
            min_roads=min_roads,
            min_gps=min_gps,
            max_trajs=max_trajs,
            meta_time_index=meta_time_index,
        )

        if len(roads_list) == 0:
            raise ValueError(f"No valid samples found in: {data_dir}")

        self.roads_list = roads_list
        self.s_list = s_list
        self.t_list = t_list
        self.depart_list = depart_list
        self.s_start_list = s_start_list
        self.s_end_list = s_end_list
        self.duration_list = duration_list
        self.elapsed_list = [_elapsed_time_from_t_seq(t_seq) for t_seq in t_list]
        self.remaining_time_list = [
            _normalized_remaining_time(elapsed_seq, total_duration)
            for elapsed_seq, total_duration in zip(self.elapsed_list, self.duration_list)
        ]

    def __len__(self) -> int:
        return len(self.roads_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, float, float, float]:
        # Return full trajectory (no segmentation)
        s_seq = self.s_list[idx]
        remaining_time_seq = self.remaining_time_list[idx]
        s_seg = torch.tensor(s_seq, dtype=torch.float32)
        remaining_time_seg = torch.tensor(remaining_time_seq, dtype=torch.float32)

        roads = torch.tensor(self.roads_list[idx], dtype=torch.long)
        depart = self.depart_list[idx]
        s_start = float(self.s_start_list[idx])
        s_end = float(self.s_end_list[idx])

        # road_mask: all valid (roads_unique has no padding at item level)
        road_mask = torch.ones(len(roads), dtype=torch.bool)
        total_duration = float(self.duration_list[idx])

        return roads, s_seg, remaining_time_seg, road_mask, depart, s_start, s_end, total_duration
