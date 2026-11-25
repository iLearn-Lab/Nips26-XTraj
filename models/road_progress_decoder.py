"""Road-Progress Autoregressive Decoder.

Given a known road segment sequence (roads_unique) and departure time,
autoregressively generates resampled s values (s = i + alpha).

Architecture:
  - Path Encoder: TransformerEncoder over road embeddings z[roads_unique]
  - AR Decoder: Causal TransformerDecoder conditioned on path encoding
  - Output head: predicts nonnegative along-road displacement each step.

Input data format (from build_road_progress_dataset.py):
  sample = [meta, roads_unique, s_seq, t_seq]
  - roads_unique: int32 [M]  -- consecutive-dedup road ids
  - s_seq:        float32 [T] -- s_k = i + alpha, uniformly resampled
  - t_seq:        float32 [T] -- time deltas aligned with s_seq

Departure time is a learned embedding of the five-minute time bin.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _causal_mask(t: int, device: torch.device, window: int = 0) -> torch.Tensor:
    future_mask = torch.triu(torch.ones((t, t), dtype=torch.bool, device=device), diagonal=1)
    if window <= 0:
        return future_mask
    past_mask = torch.tril(torch.ones((t, t), dtype=torch.bool, device=device), diagonal=-int(window))
    return future_mask | past_mask


def route_metres(
    roads: torch.Tensor, road_mask: Optional[torch.Tensor], road_lengths_m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = road_lengths_m[roads.clamp(min=0)].float()
    if road_mask is not None:
        lengths = lengths.masked_fill(~road_mask, 0)
        counts = road_mask.sum(1)
    else:
        counts = torch.full_like(roads[:, 0], roads.shape[1])
    prefix = nn.functional.pad(lengths.cumsum(dim=1), (1, 0))
    return lengths, prefix, counts


def progress_to_metres(s: torch.Tensor, prefix: torch.Tensor, lengths: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    scalar = s.ndim == 1
    if scalar:
        s = s[:, None]
    idx = torch.minimum(s.long().clamp(min=0), counts[:, None] - 1)
    out = prefix.gather(1, idx) + (s - idx) * lengths.gather(1, idx)
    return out.squeeze(1) if scalar else out


def _metres_to_progress(m: torch.Tensor, prefix: torch.Tensor, lengths: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    seg = torch.searchsorted(prefix.contiguous(), m[:, None].contiguous(), right=True).squeeze(1) - 1
    seg = seg.clamp(min=0).minimum(counts - 1)
    idx = seg[:, None]
    alpha = (m - prefix.gather(1, idx).squeeze(1)) / lengths.gather(1, idx).squeeze(1)
    return seg + alpha.clamp(0, 1)


def _position_step(
    prev_s, prev_m, raw, pace, end_s, end_m, final_step, prefix, lengths_m, road_counts,
    *, boundary_grad=False,
):
    """Advance without backtracking or overshooting; the last point is fixed."""
    proposed_m = prev_m + raw.clamp_min(0) * pace
    bounded_m = proposed_m.minimum(end_m)
    if boundary_grad:
        # Keep the projected position in the forward pass. Where a clamp
        # blocks the gradient, pass a local position gradient to the head.
        saturated = (raw < 0) | (proposed_m > end_m)
        recovery = (raw - raw.detach()) * pace.detach()
        bounded_m = bounded_m + recovery.masked_fill(~saturated, 0)
    next_m = torch.where(final_step, end_m, bounded_m)
    next_s = _metres_to_progress(next_m, prefix, lengths_m, road_counts).maximum(prev_s).minimum(end_s)
    next_s = torch.where(next_m >= end_m, end_s, next_s)
    return next_s, next_m


def _metric_context(
    current: torch.Tensor, start: torch.Tensor, end: torch.Tensor, n_steps: torch.Tensor, positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_m = (end - start).clamp_min(0)
    remaining_m = (end[:, None] - current).clamp_min(0)
    pace = remaining_m / (n_steps[:, None] - positions).clamp_min(1)
    covered = ((current - start[:, None]) / total_m[:, None].clamp_min(1e-4)).clamp(0, 1)
    nominal_pace = total_m / n_steps.clamp_min(1)
    context = torch.stack(
        (covered, torch.log1p(pace) / 4, (torch.log1p(nominal_pace) / 4)[:, None].expand_as(pace)),
        dim=-1,
    )
    return context, pace


@dataclass
class RoadProgressDecoderConfig:
    emb_dim: int = 64           # road embedding dimension (from encoder)
    d_model: int = 256          # transformer hidden dim
    n_heads: int = 8
    n_enc_layers: int = 2       # path encoder layers
    n_dec_layers: int = 4       # AR decoder layers
    dropout: float = 0.1
    max_roads: int = 128        # max road segments per trajectory
    max_steps: int = 1024       # max AR generation steps

    # Departure time conditioning (discrete bin, e.g. 0..287 for 5-min intervals)
    use_depart_time: bool = False
    num_time_bins: int = 288

    # Optional global condition: full-trajectory total duration in seconds.
    use_total_duration: bool = False

    # Optional per-step condition: normalized full-trajectory remaining time.
    use_remaining_time: bool = False

    # Decoder self-attention local history window; <=0 means full history.
    decoder_local_window: int = 128

    # Delta-s safety clamp during generation
    delta_s_max: float = 2.0

    # Endpoint conditioning.
    # use_start_point: condition on s_start
    # use_end_point:   condition on s_end (re-enables endpoint-aware generation)
    # use_endpoints:   legacy alias -> sets both use_start_point and use_end_point
    use_start_point: bool = False
    use_end_point: bool = False
    use_endpoints: bool = False
    # Legacy checkpoints predict increments in segment-index units.
    predict_metric_steps: bool = False
    # v5+ share local attention and position projection in training and generation.
    position_rollout: bool = False
    # v6 can learn from position errors at the two hard projection boundaries.
    position_boundary_grad: bool = False


class RoadProgressDecoder(nn.Module):
    """Conditional AR decoder: roads_unique + depart_time -> s sequence.

    Position models train by rolling out metric positions. Older models
    predict per-step increments with teacher forcing.

    Generation:
        Autoregressively accumulates s until s >= M (end of last segment)
        or max_steps is reached.
    """

    def __init__(self, cfg: RoadProgressDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.position_rollout and not cfg.predict_metric_steps:
            raise ValueError("Position rollout requires metric decoding")
        if cfg.position_boundary_grad and not cfg.position_rollout:
            raise ValueError("Boundary gradients require position rollout")
        if self.cfg.use_endpoints:
            self.cfg.use_start_point = True
            self.cfg.use_end_point = True

        # --- Path Encoder ---
        self.road_in_proj = nn.Linear(cfg.emb_dim, cfg.d_model)
        self.road_pos_emb = nn.Embedding(cfg.max_roads, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.path_encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_enc_layers)
        self.path_norm = nn.LayerNorm(cfg.d_model)

        # --- AR Decoder ---
        self.s_in_proj = nn.Linear(1, cfg.d_model)
        self.s_pos_emb = nn.Embedding(cfg.max_steps, cfg.d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.n_dec_layers)
        self.dec_norm = nn.LayerNorm(cfg.d_model)

        # Output head: scalar multiplier for metric steps, or legacy delta_s.
        self.out_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )
        self.road_length_proj = nn.Linear(1, cfg.d_model) if cfg.predict_metric_steps else None
        self.metric_context_proj = nn.Linear(3, cfg.d_model) if cfg.predict_metric_steps else None
        if cfg.predict_metric_steps:
            # At initialization the head follows the steady pace needed to
            # reach the requested endpoint in the remaining time.
            nn.init.normal_(self.out_head[-1].weight, std=1e-3)
            nn.init.constant_(self.out_head[-1].bias, 1.0 if cfg.position_rollout else math.log(math.expm1(1.0)))

        # Departure time: discrete bin embedding
        if cfg.use_depart_time:
            self.time_emb = nn.Embedding(cfg.num_time_bins, cfg.d_model)
            nn.init.normal_(self.time_emb.weight, std=0.02)
        else:
            self.time_emb = None

        if cfg.use_total_duration:
            self.duration_proj = nn.Linear(1, cfg.d_model)
        else:
            self.duration_proj = None

        if cfg.use_remaining_time:
            self.remaining_time_proj = nn.Linear(1, cfg.d_model)
        else:
            self.remaining_time_proj = None

        # Start/end-point s conditioning
        if cfg.use_start_point or cfg.use_endpoints:
            self.s_start_proj = nn.Linear(1, cfg.d_model)
        else:
            self.s_start_proj = None
        if cfg.use_end_point or cfg.use_endpoints:
            self.s_end_proj = nn.Linear(1, cfg.d_model)
        else:
            self.s_end_proj = None

        nn.init.normal_(self.road_pos_emb.weight, std=0.02)
        nn.init.normal_(self.s_pos_emb.weight, std=0.02)

    def _decoder_window(self) -> int:
        return max(int(getattr(self.cfg, "decoder_local_window", 0)), 0)

    def _decoder_receptive_field(self) -> int:
        local = self._decoder_window()
        # The most recent token can see local - 1 extra tokens at each layer.
        return 1 + len(self.decoder.layers) * (local - 1) if local else 0

    def _prepare_remaining_time_seq(
        self,
        remaining_time_seq: Optional[torch.Tensor],
        *,
        batch_size: int,
        min_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.remaining_time_proj is None:
            return None
        if remaining_time_seq is None:
            raise ValueError("remaining_time_seq required when cfg.use_remaining_time=True")
        if remaining_time_seq.ndim == 1:
            remaining_time_seq = remaining_time_seq.unsqueeze(0)
        if remaining_time_seq.ndim != 2 or int(remaining_time_seq.shape[0]) != batch_size:
            raise ValueError(
                f"remaining_time_seq must be [B,T], got shape={tuple(remaining_time_seq.shape)} for batch_size={batch_size}"
            )
        if int(remaining_time_seq.shape[1]) < int(min_len):
            raise ValueError(
                f"remaining_time_seq length {int(remaining_time_seq.shape[1])} is smaller than required {int(min_len)}"
            )
        return remaining_time_seq.to(device=device).float()

    def _build_decoder_input(
        self,
        *,
        s_tokens: torch.Tensor,
        pos_ids: torch.Tensor,
        remaining_time_tokens: Optional[torch.Tensor] = None,
        metric_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        s_features = s_tokens.float().clamp_min(0).log1p() if self.cfg.predict_metric_steps else s_tokens.float()
        h = self.s_in_proj(s_features.unsqueeze(-1))
        h = h + self.s_pos_emb(pos_ids.long())
        if self.metric_context_proj is not None:
            if metric_context is None:
                raise ValueError("Metric decoding requires current route position and time budget")
            h = h + self.metric_context_proj(metric_context)
        if self.remaining_time_proj is not None:
            if remaining_time_tokens is None:
                raise ValueError("remaining_time_tokens required when cfg.use_remaining_time=True")
            h = h + self.remaining_time_proj(remaining_time_tokens.unsqueeze(-1).float())
        return h

    # ------------------------------------------------------------------
    def encode_path(
        self,
        roads_unique: torch.Tensor,                    # [B, M]
        z: torch.Tensor,                               # [N, emb_dim]
        road_mask: Optional[torch.Tensor] = None,      # [B, M] bool True=valid
        depart_time: Optional[torch.Tensor] = None,    # [B] LongTensor (discrete bin)
        total_duration: Optional[torch.Tensor] = None, # [B] float seconds
        s_start: Optional[torch.Tensor] = None,        # [B] float
        s_end: Optional[torch.Tensor] = None,          # [B] float
        route_lengths_m: Optional[torch.Tensor] = None, # [B, M]
    ) -> torch.Tensor:
        """Encode road sequence -> memory [B, M, d_model]."""
        b, m = roads_unique.shape
        device = roads_unique.device

        rid_safe = roads_unique.clamp(min=0)
        h = self.road_in_proj(z[rid_safe])             # [B, M, d_model]
        if self.road_length_proj is not None:
            if route_lengths_m is None:
                raise ValueError("Metric decoding requires the length of each road")
            h = h + self.road_length_proj((route_lengths_m.log1p() / 5).unsqueeze(-1))

        pos = torch.arange(m, device=device).unsqueeze(0).expand(b, m)
        h = h + self.road_pos_emb(pos)

        if self.time_emb is not None and depart_time is not None:
            dt = depart_time.to(device).long().clamp(0, self.cfg.num_time_bins - 1)
            h = h + self.time_emb(dt).unsqueeze(1)    # [B, d_model] -> broadcast over M

        if self.duration_proj is not None and total_duration is not None:
            duration = torch.log1p(total_duration.to(device).float().clamp(min=0.0)).view(b, 1)
            h = h + self.duration_proj(duration).unsqueeze(1)

        if self.s_start_proj is not None and s_start is not None:
            value = s_start.to(device).float()
            if self.cfg.predict_metric_steps:
                value = value.log1p()
            h = h + self.s_start_proj(value.view(b, 1, 1)).expand(b, m, -1)

        if self.s_end_proj is not None and s_end is not None:
            value = s_end.to(device).float()
            if self.cfg.predict_metric_steps:
                value = value.log1p()
            h = h + self.s_end_proj(value.view(b, 1, 1)).expand(b, m, -1)

        kpm = (~road_mask.to(torch.bool)) if road_mask is not None else None
        mem = self.path_encoder(h, src_key_padding_mask=kpm)
        return self.path_norm(mem)                     # [B, M, d_model]

    def _position_window_step(
        self, s_tokens, pos, remaining, mem, mem_kpm, prefix, lengths, counts,
        start_m, end_m, n_steps, last_index=None,
    ):
        context, pace = _metric_context(
            progress_to_metres(s_tokens, prefix, lengths, counts), start_m, end_m, n_steps, pos,
        )
        h = self._build_decoder_input(
            s_tokens=s_tokens, pos_ids=pos, remaining_time_tokens=remaining, metric_context=context,
        )
        out = self.decoder(
            h, mem, tgt_mask=_causal_mask(h.shape[1], h.device, self._decoder_window()),
            memory_key_padding_mask=mem_kpm,
        )
        if last_index is None:
            out, pace = out[:, -1], pace[:, -1]
        else:
            rows = torch.arange(h.shape[0], device=h.device)
            out, pace = out[rows, last_index], pace[rows, last_index]
        return self.out_head(self.dec_norm(out)).squeeze(-1).float(), pace

    def _rollout_positions(
        self, s_seq, remaining, mem, mem_kpm, prefix, lengths, road_counts,
        s_start, s_end, n_steps, start_index, rollout_steps, checkpoint_rollout,
    ):
        """Start at an observed prefix and predict the next positions recursively."""
        start_m = progress_to_metres(s_start, prefix, lengths, road_counts)
        end_m = progress_to_metres(s_end, prefix, lengths, road_counts)
        from_origin = start_index is None
        if from_origin:
            start_index = torch.zeros_like(n_steps)
            prev_s, prev_m = s_start, start_m
        else:
            prev_s = s_seq.gather(1, start_index[:, None]).squeeze(1)
            prev_m = progress_to_metres(prev_s, prefix, lengths, road_counts)
        history = [prev_s]
        predictions = []
        field = self._decoder_receptive_field()
        width = min(s_seq.shape[1] - 1, field or s_seq.shape[1] - 1)
        if from_origin:
            width = min(width, rollout_steps)
        offsets = torch.arange(width, device=s_seq.device)[None, :]
        for k in range(rollout_steps):
            # Right slots are causal future slots. Repeating the current index
            # there avoids reading observations beyond the rollout's seed.
            current = (start_index + k).minimum(n_steps - 1)
            left = (current - width + 1).clamp_min(0)
            window_offsets = offsets[:, :k + 1] if from_origin else offsets
            pos = (left[:, None] + window_offsets).minimum(current[:, None])
            rel = (pos - start_index[:, None]).clamp(min=0, max=k)
            own_s = torch.stack(history, dim=1).gather(1, rel)
            tokens = own_s if from_origin else torch.where(pos > start_index[:, None], own_s, s_seq.gather(1, pos))
            decoder_args = (
                tokens, pos, remaining.gather(1, pos) if remaining is not None else None,
                mem, mem_kpm, prefix, lengths, road_counts, start_m, end_m, n_steps, current - left,
            )
            if checkpoint_rollout and self.training and torch.is_grad_enabled():
                # Recompute decoder activations on backward while retaining the
                # full gradient through the generated positions and history.
                raw, pace = checkpoint(self._position_window_step, *decoder_args, use_reentrant=False)
            else:
                raw, pace = self._position_window_step(*decoder_args)
            prev_s, prev_m = _position_step(
                prev_s, prev_m, raw, pace, s_end, end_m, current + 1 == n_steps,
                prefix, lengths, road_counts, boundary_grad=self.training and self.cfg.position_boundary_grad,
            )
            history.append(prev_s)
            predictions.append(prev_m)
        return torch.stack(predictions, dim=1)

    # ------------------------------------------------------------------
    def forward(
        self,
        *,
        roads_unique: torch.Tensor,                    # [B, M]
        s_seq: torch.Tensor,                           # [B, T] float
        z: torch.Tensor,                               # [N, emb_dim]
        road_mask: Optional[torch.Tensor] = None,      # [B, M] bool
        s_mask: Optional[torch.Tensor] = None,         # [B, T] bool True=valid
        remaining_time_seq: Optional[torch.Tensor] = None,  # [B, T] float in [0,1]
        depart_time: Optional[torch.Tensor] = None,    # [B] LongTensor (discrete bin)
        total_duration: Optional[torch.Tensor] = None, # [B] float seconds
        s_start: Optional[torch.Tensor] = None,        # [B] float
        s_end: Optional[torch.Tensor] = None,          # [B] float
        road_lengths_m: Optional[torch.Tensor] = None, # [N]
        rollout_start: Optional[torch.Tensor] = None,  # [B] seed indices; None starts at the origin
        rollout_steps: int = 0,
        checkpoint_rollout: bool = False,
    ) -> torch.Tensor:
        """Return rolled-out positions in metres, or legacy teacher-forced increments."""
        b, t = s_seq.shape
        device = s_seq.device

        if t < 2:
            raise ValueError("s_seq must have at least 2 time steps")
        if t > self.cfg.max_steps:
            raise ValueError(f"T={t} exceeds cfg.max_steps={self.cfg.max_steps}")

        remaining_time_seq = self._prepare_remaining_time_seq(
            remaining_time_seq,
            batch_size=b,
            min_len=t,
            device=device,
        )

        route_lengths = prefix = counts = None
        if self.cfg.predict_metric_steps:
            if road_lengths_m is None or s_mask is None or s_start is None or s_end is None:
                raise ValueError("Metric training requires road lengths, point mask and endpoints")
            route_lengths, prefix, counts = route_metres(roads_unique, road_mask, road_lengths_m)
        mem = self.encode_path(
            roads_unique,
            z,
            road_mask=road_mask,
            depart_time=depart_time,
            total_duration=total_duration,
            s_start=s_start,
            s_end=s_end,
            route_lengths_m=route_lengths,
        )
        mem_kpm = (~road_mask.to(torch.bool)) if road_mask is not None else None
        if self.cfg.position_rollout:
            if rollout_steps < 1:
                raise ValueError("Position training requires a positive rollout horizon")
            return self._rollout_positions(
                s_seq, remaining_time_seq, mem, mem_kpm, prefix, route_lengths, counts,
                s_start, s_end, s_mask.sum(1) - 1, rollout_start, rollout_steps, checkpoint_rollout,
            )

        # Decoder input: s[0..T-2], predict delta at positions 1..T-1
        s_in = s_seq[:, :-1].float()                   # [B, T-1]
        t_dec = t - 1
        pos = torch.arange(t_dec, device=device).unsqueeze(0).expand(b, t_dec)
        remaining_in = remaining_time_seq[:, :-1] if remaining_time_seq is not None else None
        context = pace = None
        if self.cfg.predict_metric_steps:
            context, pace = _metric_context(
                progress_to_metres(s_in, prefix, route_lengths, counts),
                progress_to_metres(s_start, prefix, route_lengths, counts),
                progress_to_metres(s_end, prefix, route_lengths, counts),
                s_mask.sum(1) - 1, pos,
            )
        h = self._build_decoder_input(
            s_tokens=s_in, pos_ids=pos, remaining_time_tokens=remaining_in, metric_context=context,
        )

        causal = _causal_mask(t_dec, device, window=self._decoder_window())
        tgt_kpm = (~s_mask[:, :-1].to(torch.bool)) if s_mask is not None else None

        out = self.decoder(
            h, mem,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=mem_kpm,
        )
        out = self.dec_norm(out)                       # [B, T-1, d_model]
        raw = self.out_head(out).squeeze(-1)           # [B, T-1]
        step = nn.functional.softplus(raw.float())
        return step * pace if pace is not None else step

    @torch.no_grad()
    def generate_batch(
        self,
        *,
        roads_unique: torch.Tensor,                    # [B, M] or [M]
        z: torch.Tensor,                               # [N, emb_dim]
        road_mask: Optional[torch.Tensor] = None,      # [B, M] bool
        s_start: torch.Tensor | float = 0.0,           # [B] or scalar
        s_end: Optional[torch.Tensor | float] = None,  # [B] or scalar
        decode_steps: Optional[torch.Tensor | int] = None,
        remaining_time_seq: Optional[torch.Tensor] = None,  # [B, L] or [L]
        depart_time: Optional[torch.Tensor] = None,    # [B] LongTensor
        total_duration: Optional[torch.Tensor] = None, # [B] float seconds
        device: Optional[torch.device] = None,
        stop_at_end: bool = True,
        fixed_road_lengths: Optional[torch.Tensor] = None,  # [N], enable fixed-budget decoding
    ) -> list[torch.Tensor]:
        """Autoregressively generate a batch of s sequences.

        Passing fixed_road_lengths makes decode_steps an exact time budget.

        Returns:
            list[FloatTensor[L_i]] where each sequence includes s_start.
        """
        if device is None:
            device = next(self.parameters()).device

        if roads_unique.ndim == 1:
            roads_unique = roads_unique.unsqueeze(0)
        if roads_unique.ndim != 2:
            raise ValueError(f"roads_unique must be [B,M] or [M], got shape={tuple(roads_unique.shape)}")
        roads_unique = roads_unique.to(device).long()
        b, m = roads_unique.shape

        if road_mask is None:
            road_mask = torch.ones((b, m), dtype=torch.bool, device=device)
        else:
            if road_mask.ndim == 1:
                road_mask = road_mask.unsqueeze(0)
            if road_mask.shape != roads_unique.shape:
                raise ValueError(f"road_mask shape {tuple(road_mask.shape)} != roads_unique shape {tuple(roads_unique.shape)}")
            road_mask = road_mask.to(device=device, dtype=torch.bool)

        if isinstance(s_start, (int, float)):
            s_start = torch.full((b,), float(s_start), dtype=torch.float32, device=device)
        else:
            if s_start.ndim == 0:
                s_start = s_start.view(1)
            if s_start.ndim != 1 or int(s_start.shape[0]) != b:
                raise ValueError(f"s_start must be scalar or [B], got shape={tuple(s_start.shape)}")
            s_start = s_start.to(device=device).float()

        valid_road_counts = road_mask.long().sum(dim=1).clamp(min=1)
        default_s_end = valid_road_counts.float() - 1e-4
        if s_end is None:
            s_end = default_s_end
        elif isinstance(s_end, (int, float)):
            s_end = torch.full((b,), float(s_end), dtype=torch.float32, device=device)
        else:
            if s_end.ndim == 0:
                s_end = s_end.view(1)
            if s_end.ndim != 1 or int(s_end.shape[0]) != b:
                raise ValueError(f"s_end must be scalar or [B], got shape={tuple(s_end.shape)}")
            s_end = s_end.to(device=device).float()

        if self.time_emb is not None:
            if depart_time is None:
                raise ValueError("depart_time required when cfg.use_depart_time=True")
            if depart_time.ndim == 0:
                depart_time = depart_time.view(1)
            if depart_time.ndim != 1 or int(depart_time.shape[0]) != b:
                raise ValueError(f"depart_time must be [B], got shape={tuple(depart_time.shape)}")
            depart_time = depart_time.to(device).long()

        if self.duration_proj is not None:
            if total_duration is None:
                raise ValueError("total_duration required when cfg.use_total_duration=True")
            if total_duration.ndim == 0:
                total_duration = total_duration.view(1)
            if total_duration.ndim != 1 or int(total_duration.shape[0]) != b:
                raise ValueError(f"total_duration must be [B], got shape={tuple(total_duration.shape)}")
            total_duration = total_duration.to(device).float()

        if decode_steps is None:
            decode_steps = torch.full((b,), int(self.cfg.max_steps), dtype=torch.long, device=device)
        elif isinstance(decode_steps, int):
            decode_steps = torch.full((b,), int(decode_steps), dtype=torch.long, device=device)
        else:
            if decode_steps.ndim == 0:
                decode_steps = decode_steps.view(1)
            if decode_steps.ndim != 1 or int(decode_steps.shape[0]) != b:
                raise ValueError(f"decode_steps must be int or [B], got shape={tuple(decode_steps.shape)}")
            decode_steps = decode_steps.to(device=device).long()
        decode_steps = decode_steps.clamp(min=0, max=int(self.cfg.max_steps))

        fixed = fixed_road_lengths is not None
        if self.cfg.predict_metric_steps and not fixed:
            raise ValueError("Metric checkpoints must use fixed-budget generation with road lengths")
        if fixed:
            if torch.any(decode_steps < 1) or torch.any((s_start < 0) | (s_end < s_start) | (s_end > valid_road_counts)):
                raise ValueError("Fixed decoding requires at least one step and valid ordered endpoints")
            route_lengths, prefix_m, valid_road_counts = route_metres(roads_unique, road_mask, fixed_road_lengths.to(device))
            end_m = progress_to_metres(s_end, prefix_m, route_lengths, valid_road_counts)
            start_m = progress_to_metres(s_start, prefix_m, route_lengths, valid_road_counts)
            if not self.cfg.position_rollout:
                # Preserve the fixed-budget projection of older checkpoints.
                avg_m = (end_m - start_m) / decode_steps
                max_step_m = 3.0 * avg_m
                reserve_m = 0.05 * avg_m
            current_m = start_m.clone()

        max_decode_steps = int(decode_steps.max().item()) if b > 0 else 0
        t_cap = max_decode_steps + 1
        required_remaining_len = max(1, max_decode_steps)
        remaining_time_seq = self._prepare_remaining_time_seq(
            remaining_time_seq,
            batch_size=b,
            min_len=required_remaining_len,
            device=device,
        )

        s_start_cond = s_start if self.s_start_proj is not None else None
        s_end_cond = s_end if self.s_end_proj is not None else None
        mem = self.encode_path(
            roads_unique,
            z,
            road_mask=road_mask,
            depart_time=depart_time,
            total_duration=total_duration,
            s_start=s_start_cond,
            s_end=s_end_cond,
            route_lengths_m=route_lengths if fixed else None,
        )
        mem_kpm = ~road_mask

        s_out = torch.zeros((b, t_cap), dtype=torch.float32, device=device)
        lengths = torch.ones((b,), dtype=torch.long, device=device)
        finished = decode_steps <= 0
        s_limit = s_end

        if t_cap > 0:
            s_out[:, 0] = s_start
        if stop_at_end and not fixed:
            finished = finished | (s_start >= s_limit)

        history_size = self._decoder_receptive_field() if self.cfg.position_rollout else self._decoder_window()
        for step in range(max_decode_steps):
            active_idx = torch.nonzero(~finished, as_tuple=False).squeeze(-1)
            if int(active_idx.numel()) == 0:
                break

            active_lengths = lengths[active_idx]
            # Every still-active row has generated exactly 'step + 1' points.
            left = max(0, step + 1 - history_size) if history_size else 0
            window_s = s_out[active_idx, left:step + 1]
            window_pos = torch.arange(left, step + 1, device=device).expand(active_idx.numel(), -1)
            window_remaining = (
                remaining_time_seq[active_idx, left:step + 1] if remaining_time_seq is not None else None
            )
            if self.cfg.position_rollout:
                prefix = prefix_m[active_idx]
                lengths_m = route_lengths[active_idx]
                counts = valid_road_counts[active_idx]
                raw, pace = self._position_window_step(
                    window_s, window_pos, window_remaining, mem[active_idx], mem_kpm[active_idx],
                    prefix, lengths_m, counts, start_m[active_idx], end_m[active_idx], decode_steps[active_idx],
                )
                s_new, next_m = _position_step(
                    window_s[:, -1], current_m[active_idx], raw, pace, s_end[active_idx], end_m[active_idx],
                    step + 1 == decode_steps[active_idx], prefix, lengths_m, counts,
                )
                current_m[active_idx] = next_m
                s_out[active_idx, step + 1] = s_new
                lengths[active_idx] = step + 2
                finished[active_idx] = step + 1 == decode_steps[active_idx]
                continue

            context = pace = None
            if self.cfg.predict_metric_steps:
                context, pace = _metric_context(
                    progress_to_metres(window_s, prefix_m[active_idx], route_lengths[active_idx], valid_road_counts[active_idx]),
                    start_m[active_idx], end_m[active_idx], decode_steps[active_idx], window_pos,
                )
            h = self._build_decoder_input(
                s_tokens=window_s,
                pos_ids=window_pos,
                remaining_time_tokens=window_remaining,
                metric_context=context,
            )

            window_len = int(window_s.shape[1])
            causal = _causal_mask(window_len, device)
            out = self.decoder(
                h,
                mem[active_idx],
                tgt_mask=causal,
                memory_key_padding_mask=mem_kpm[active_idx],
            )
            out = self.dec_norm(out)
            raw = self.out_head(out).squeeze(-1)
            step_pred = nn.functional.softplus(raw[:, -1].float())

            prev_s = s_out[active_idx, active_lengths - 1]
            if not fixed:
                s_new = torch.minimum(prev_s + step_pred.clamp(max=float(self.cfg.delta_s_max)), s_limit[active_idx])
            else:
                prefix = prefix_m[active_idx].contiguous()
                lengths_m = route_lengths[active_idx]
                prev_m = current_m[active_idx]
                dest_m = end_m[active_idx]
                left_steps = decode_steps[active_idx] - step - 1
                lo = torch.maximum(prev_m, dest_m - left_steps * max_step_m[active_idx])
                hi = torch.minimum(prev_m + max_step_m[active_idx], dest_m - left_steps * reserve_m[active_idx])
                if pace is not None:
                    proposed_m = prev_m + step_pred * pace[:, -1]
                else:
                    proposed_s = torch.minimum(prev_s + step_pred, s_limit[active_idx])
                    proposed_m = progress_to_metres(proposed_s, prefix, lengths_m, valid_road_counts[active_idx])
                next_m = torch.where(left_steps == 0, dest_m, torch.minimum(proposed_m.maximum(lo), hi.maximum(lo)))
                # Invert cumulative road length. A padded boundary and an
                # endpoint at alpha=1 both belong to the last valid road.
                seg = torch.searchsorted(prefix, next_m[:, None], right=True).squeeze(1) - 1
                seg = seg.clamp(min=0).minimum(valid_road_counts[active_idx] - 1)
                alpha = (next_m - prefix.gather(1, seg[:, None]).squeeze(1)) / (
                    lengths_m.gather(1, seg[:, None]).squeeze(1)
                )
                s_new = torch.where(
                    left_steps == 0, s_limit[active_idx],
                    (seg + alpha.clamp(0, 1)).maximum(prev_s).minimum(s_limit[active_idx]),
                )
                current_m[active_idx] = next_m
            next_pos = lengths[active_idx]
            valid_write = next_pos < t_cap
            if bool(valid_write.any()):
                write_idx = active_idx[valid_write]
                write_pos = next_pos[valid_write]
                s_out[write_idx, write_pos] = s_new[valid_write]
                lengths[write_idx] = write_pos + 1

            finished_by_steps = lengths[active_idx] - 1 >= decode_steps[active_idx]
            finished_by_end = s_new >= s_limit[active_idx]
            finished[active_idx] = finished_by_steps | (finished_by_end if stop_at_end and not fixed else False)

        return [s_out[i, : int(lengths[i].item())].clone() for i in range(b)]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        *,
        roads_unique: torch.Tensor,                    # [M] or [1, M]
        z: torch.Tensor,                               # [N, emb_dim]
        s_start: float = 0.0,
        s_end: Optional[float] = None,                 # if None, use end of last segment
        max_steps: Optional[int] = None,
        remaining_time_seq: Optional[torch.Tensor] = None,  # [L] or [1, L]
        depart_time: Optional[torch.Tensor] = None,    # scalar or [1] LongTensor (discrete bin)
        total_duration: Optional[torch.Tensor] = None, # scalar or [1] float seconds
        device: Optional[torch.device] = None,
        stop_at_end: bool = True,
    ) -> torch.Tensor:
        """Autoregressively generate s sequence.

        Returns:
            s_out: FloatTensor [L] (includes s_start)
        """
        if device is None:
            device = next(self.parameters()).device

        if roads_unique.ndim == 1:
            roads_unique = roads_unique.unsqueeze(0)
        roads_unique = roads_unique.to(device)

        if self.time_emb is not None:
            if depart_time is None:
                raise ValueError("depart_time required when cfg.use_depart_time=True")
            if depart_time.ndim == 0:
                depart_time = depart_time.view(1)
            depart_time = depart_time.to(device).long()

        if self.duration_proj is not None:
            if total_duration is None:
                raise ValueError("total_duration required when cfg.use_total_duration=True")
            if total_duration.ndim == 0:
                total_duration = total_duration.view(1)
            total_duration = total_duration.to(device).float()

        decode_steps = int(max_steps) if max_steps is not None else int(self.cfg.max_steps)
        trajs = self.generate_batch(
            roads_unique=roads_unique,
            z=z,
            road_mask=torch.ones_like(roads_unique, dtype=torch.bool, device=device),
            s_start=torch.tensor([float(s_start)], dtype=torch.float32, device=device),
            s_end=(torch.tensor([float(s_end)], dtype=torch.float32, device=device) if s_end is not None else None),
            decode_steps=torch.tensor([decode_steps], dtype=torch.long, device=device),
            remaining_time_seq=remaining_time_seq,
            depart_time=depart_time,
            total_duration=total_duration,
            device=device,
            stop_at_end=stop_at_end,
        )
        return trajs[0]
