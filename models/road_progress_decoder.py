"""Road-Progress Autoregressive Decoder.

Given a known road segment sequence (roads_unique) and departure time,
autoregressively generates resampled s values (s = i + alpha).

Architecture:
  - Path Encoder: TransformerEncoder over road embeddings z[roads_unique]
  - AR Decoder: Causal TransformerDecoder conditioned on path encoding
  - Output head: predicts delta_s >= 0 at each step (SmoothL1 loss)

Input data format (from build_road_progress_dataset.py):
  sample = [meta, roads_unique, s_seq, t_seq]
  - roads_unique: int32 [M]  -- consecutive-dedup road ids
  - s_seq:        float32 [T] -- s_k = i + alpha, uniformly resampled
  - t_seq:        float32 [T] -- time deltas aligned with s_seq

Departure time:
  Treated as a continuous float (e.g. minutes-of-day: 0..1440).
  Encoded via sinusoidal + linear projection to capture daily periodicity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def _causal_mask(t: int, device: torch.device, window: int = 0) -> torch.Tensor:
    future_mask = torch.triu(torch.ones((t, t), dtype=torch.bool, device=device), diagonal=1)
    if window <= 0:
        return future_mask
    past_mask = torch.tril(torch.ones((t, t), dtype=torch.bool, device=device), diagonal=-int(window))
    return future_mask | past_mask


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


class RoadProgressDecoder(nn.Module):
    """Conditional AR decoder: roads_unique + depart_time -> s sequence.

    Training (teacher forcing):
        Predicts delta_s_t = s_t - s_{t-1} >= 0 for t = 1..T-1.
        Loss: SmoothL1 on valid steps.

    Generation:
        Autoregressively accumulates s until s >= M (end of last segment)
        or max_steps is reached.
    """

    def __init__(self, cfg: RoadProgressDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
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

        # Output head: scalar delta_s (softplus ensures >= 0)
        self.out_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )

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

    def _build_generation_window(
        self,
        *,
        s_seq: torch.Tensor,
        s_mask: torch.Tensor,
        lengths: torch.Tensor,
        remaining_time_seq: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size = int(s_seq.shape[0])
        device = s_seq.device
        max_visible = int(lengths.max().item()) if batch_size > 0 else 0
        window_len = max_visible
        local_window = self._decoder_window()
        if local_window > 0:
            window_len = min(window_len, local_window)

        window_s = torch.zeros((batch_size, window_len), dtype=s_seq.dtype, device=device)
        window_mask = torch.zeros((batch_size, window_len), dtype=torch.bool, device=device)
        window_pos = torch.zeros((batch_size, window_len), dtype=torch.long, device=device)
        window_remaining = None
        if remaining_time_seq is not None:
            window_remaining = torch.zeros((batch_size, window_len), dtype=torch.float32, device=device)

        for row in range(batch_size):
            row_len = int(lengths[row].item())
            take = min(row_len, window_len)
            src_start = row_len - take
            dst_start = window_len - take
            window_s[row, dst_start:] = s_seq[row, src_start:row_len]
            window_mask[row, dst_start:] = s_mask[row, src_start:row_len]
            window_pos[row, dst_start:] = torch.arange(src_start, row_len, device=device, dtype=torch.long)
            if window_remaining is not None:
                window_remaining[row, dst_start:] = remaining_time_seq[row, src_start:row_len]

        return window_s, window_mask, window_pos, window_remaining

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
    ) -> torch.Tensor:
        h = self.s_in_proj(s_tokens.unsqueeze(-1).float())
        h = h + self.s_pos_emb(pos_ids.long())
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
    ) -> torch.Tensor:
        """Encode road sequence -> memory [B, M, d_model]."""
        b, m = roads_unique.shape
        device = roads_unique.device

        rid_safe = roads_unique.clamp(min=0)
        h = self.road_in_proj(z[rid_safe])             # [B, M, d_model]

        pos = torch.arange(m, device=device).unsqueeze(0).expand(b, m)
        h = h + self.road_pos_emb(pos)

        if self.time_emb is not None and depart_time is not None:
            dt = depart_time.to(device).long().clamp(0, self.cfg.num_time_bins - 1)
            h = h + self.time_emb(dt).unsqueeze(1)    # [B, d_model] -> broadcast over M

        if self.duration_proj is not None and total_duration is not None:
            duration = torch.log1p(total_duration.to(device).float().clamp(min=0.0)).view(b, 1)
            h = h + self.duration_proj(duration).unsqueeze(1)

        if self.s_start_proj is not None and s_start is not None:
            h = h + self.s_start_proj(s_start.to(device).float().view(b, 1, 1)).expand(b, m, -1)

        if self.s_end_proj is not None and s_end is not None:
            h = h + self.s_end_proj(s_end.to(device).float().view(b, 1, 1)).expand(b, m, -1)

        kpm = (~road_mask.to(torch.bool)) if road_mask is not None else None
        mem = self.path_encoder(h, src_key_padding_mask=kpm)
        return self.path_norm(mem)                     # [B, M, d_model]

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
    ) -> torch.Tensor:
        """Teacher-forcing forward.

        Returns:
            delta_s_pred: FloatTensor [B, T-1], softplus-activated (>= 0)
        """
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

        mem = self.encode_path(
            roads_unique,
            z,
            road_mask=road_mask,
            depart_time=depart_time,
            total_duration=total_duration,
            s_start=s_start,
            s_end=s_end,
        )

        # Decoder input: s[0..T-2], predict delta at positions 1..T-1
        s_in = s_seq[:, :-1].float()                   # [B, T-1]
        t_dec = t - 1
        pos = torch.arange(t_dec, device=device).unsqueeze(0).expand(b, t_dec)
        remaining_in = remaining_time_seq[:, :-1] if remaining_time_seq is not None else None
        h = self._build_decoder_input(s_tokens=s_in, pos_ids=pos, remaining_time_tokens=remaining_in)

        causal = _causal_mask(t_dec, device, window=self._decoder_window())
        mem_kpm = (~road_mask.to(torch.bool)) if road_mask is not None else None
        tgt_kpm = (~s_mask[:, :-1].to(torch.bool)) if s_mask is not None else None

        out = self.decoder(
            h, mem,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=mem_kpm,
        )
        out = self.dec_norm(out)                       # [B, T-1, d_model]
        raw = self.out_head(out).squeeze(-1)           # [B, T-1]
        return nn.functional.softplus(raw)

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
    ) -> list[torch.Tensor]:
        """Autoregressively generate a batch of s sequences.

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
        )
        mem_kpm = ~road_mask

        s_out = torch.zeros((b, t_cap), dtype=torch.float32, device=device)
        s_mask = torch.zeros((b, t_cap), dtype=torch.bool, device=device)
        lengths = torch.ones((b,), dtype=torch.long, device=device)
        finished = decode_steps <= 0
        s_limit = s_end

        if t_cap > 0:
            s_out[:, 0] = s_start
            s_mask[:, 0] = True
        if stop_at_end:
            finished = finished | (s_start >= s_limit)

        for _ in range(max_decode_steps):
            active_idx = torch.nonzero(~finished, as_tuple=False).squeeze(-1)
            if int(active_idx.numel()) == 0:
                break

            active_lengths = lengths[active_idx]
            window_s, window_mask, window_pos, window_remaining = self._build_generation_window(
                s_seq=s_out[active_idx],
                s_mask=s_mask[active_idx],
                lengths=active_lengths,
                remaining_time_seq=(remaining_time_seq[active_idx] if remaining_time_seq is not None else None),
            )
            h = self._build_decoder_input(
                s_tokens=window_s,
                pos_ids=window_pos,
                remaining_time_tokens=window_remaining,
            )

            window_len = int(window_s.shape[1])
            causal = _causal_mask(window_len, device)
            out = self.decoder(
                h,
                mem[active_idx],
                tgt_mask=causal,
                tgt_key_padding_mask=(~window_mask),
                memory_key_padding_mask=mem_kpm[active_idx],
            )
            out = self.dec_norm(out)
            raw = self.out_head(out).squeeze(-1)
            delta_s = nn.functional.softplus(raw[:, -1])
            delta_s = delta_s.clamp(max=float(self.cfg.delta_s_max))

            prev_s = s_out[active_idx, active_lengths - 1]
            s_new = torch.minimum(prev_s + delta_s, s_limit[active_idx])
            next_pos = lengths[active_idx]
            valid_write = next_pos < t_cap
            if bool(valid_write.any()):
                write_idx = active_idx[valid_write]
                write_pos = next_pos[valid_write]
                s_out[write_idx, write_pos] = s_new[valid_write]
                s_mask[write_idx, write_pos] = True
                lengths[write_idx] = write_pos + 1

            finished_by_steps = lengths[active_idx] - 1 >= decode_steps[active_idx]
            finished_by_end = s_new >= s_limit[active_idx]
            finished[active_idx] = finished_by_steps | (finished_by_end if stop_at_end else False)

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
