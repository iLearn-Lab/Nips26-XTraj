"""V7 autoregressive point-position training; no speed loss."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.road_progress_decoder import progress_to_metres, route_metres
from utils.training import finite_loss

POSITION_LOSS_SCALE_M = 10.0


@dataclass
class PointTrainingConfig:
    rollout_steps: int = 32
    origin_rollout_steps: int = 64
    origin_rollout_every: int = 4
    grad_accum_steps: int = 1
    checkpoint_rollout: bool = True

    def check(self):
        if min(self.rollout_steps, self.origin_rollout_steps, self.grad_accum_steps) < 1:
            raise ValueError("Rollout lengths and accumulation steps must be positive")
        if self.origin_rollout_every < 0:
            raise ValueError("Origin rollout frequency must be nonnegative")


def position_loss(pred_m, truth_m, interior):
    error = (pred_m.float() - truth_m.float()).masked_fill(~interior, 0)
    return F.smooth_l1_loss(
        error / POSITION_LOSS_SCALE_M, torch.zeros_like(error), reduction="sum",
    ) / interior.sum().clamp_min(1)


def train_epoch(model, loader, optimizer, z, road_lengths, *, epoch=1, config=None, scaler=None, device=None):
    """Run one position-supervised epoch over caller-prepared batches.

    Supply a persistent GradScaler for mixed-precision training across calls.
    The epoch index only rotates origin-rollout batches; no epoch count or
    checkpoint selection is imposed by this function.
    """
    config = config or PointTrainingConfig()
    config.check()
    device = torch.device(device) if device is not None else next(model.parameters()).device
    z = z.detach().to(device)
    road_lengths = road_lengths.to(device)
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(loader):
        roads, road_mask, s, s_mask, remaining, depart, start, end, duration = (
            x.to(device, non_blocking=True) for x in batch
        )
        n_steps = s_mask.sum(dim=1) - 1
        from_origin = config.origin_rollout_every > 0 and (
            (batch_idx + epoch - 1) % config.origin_rollout_every == 0
        )
        horizon = min(config.origin_rollout_steps if from_origin else config.rollout_steps, s.shape[1] - 2)
        offsets = torch.arange(1, horizon + 1, device=device)
        if from_origin:
            rollout_start = None
            point_idx = offsets[None, :].expand(s.shape[0], -1)
        else:
            max_start = (n_steps - 1 - horizon).clamp_min(0)
            rollout_start = (torch.rand(n_steps.shape, device=device) * (max_start + 1)).long()
            point_idx = rollout_start[:, None] + offsets
        interior = point_idx < n_steps[:, None]
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            pred = model(
                roads_unique=roads, s_seq=s, z=z, road_mask=road_mask, s_mask=s_mask,
                remaining_time_seq=remaining, depart_time=depart, total_duration=duration,
                s_start=start, s_end=end, road_lengths_m=road_lengths,
                rollout_start=rollout_start, rollout_steps=horizon, checkpoint_rollout=config.checkpoint_rollout,
            )
        lengths, prefix, counts = route_metres(roads, road_mask, road_lengths)
        truth = progress_to_metres(s.gather(1, point_idx.minimum(n_steps[:, None])), prefix, lengths, counts)
        loss = position_loss(pred, truth, interior)
        finite_loss(loss)
        group_start = (batch_idx // config.grad_accum_steps) * config.grad_accum_steps
        group_size = min(config.grad_accum_steps, len(loader) - group_start)
        scaler.scale(loss / group_size).backward()
        if batch_idx + 1 == group_start + group_size:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
