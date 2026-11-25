"""Candidate-index predictor for the level-encoded road variant."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CandidateRoadTransformerConfig:
    emb_dim: int = 64
    geo_dim: int = 29
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    max_history_len: int = 50
    num_periods: int = 5


class CandidateRoadTransformer(nn.Module):
    """Score a small, destination-aware candidate set.

    ``history_road_ids`` is the sequence before the decision and
    ``candidate_road_ids`` is the destination-aware candidate list.  The candidate
    target is an index in that list, which avoids a full-city softmax and keeps
    the same candidate semantics as the original construction.
    """

    def __init__(self, cfg: CandidateRoadTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d_model = int(cfg.d_model)
        self.history_proj = nn.Linear(int(cfg.emb_dim), d_model)
        self.candidate_proj = nn.Sequential(
            nn.Linear(int(cfg.emb_dim) + int(cfg.geo_dim), d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.position = nn.Embedding(int(cfg.max_history_len), d_model)
        self.period_emb = nn.Embedding(int(cfg.num_periods), d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(cfg.n_heads),
            dim_feedforward=4 * d_model,
            dropout=float(cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=int(cfg.n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.query = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        nn.init.normal_(self.position.weight, std=0.02)
        nn.init.normal_(self.period_emb.weight, std=0.02)

    def forward(
        self,
        *,
        history_road_ids: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_road_ids: torch.Tensor,
        candidate_geo: torch.Tensor,
        candidate_mask: torch.Tensor,
        period_index: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        if history_road_ids.ndim != 2:
            raise ValueError("history_road_ids must be [B,H]")
        if candidate_road_ids.ndim != 2:
            raise ValueError("candidate_road_ids must be [B,K]")
        if candidate_geo.shape[:2] != candidate_road_ids.shape:
            raise ValueError("candidate_geo must have shape [B,K,G]")
        if history_mask.shape != history_road_ids.shape:
            raise ValueError("history_mask shape mismatch")
        if candidate_mask.shape != candidate_road_ids.shape:
            raise ValueError("candidate_mask shape mismatch")
        b, h = history_road_ids.shape
        if h > int(self.cfg.max_history_len):
            raise ValueError(f"history length {h} exceeds {self.cfg.max_history_len}")

        hist = self.history_proj(z[history_road_ids.clamp(min=0)])
        pos = torch.arange(h, device=hist.device).unsqueeze(0)
        hist = hist + self.position(pos)
        period = self.period_emb(period_index.long().clamp(min=0, max=int(self.cfg.num_periods) - 1))
        hist = hist + period.unsqueeze(1)
        valid = history_mask.to(torch.bool)
        causal = torch.triu(torch.ones((h, h), dtype=torch.bool, device=hist.device), diagonal=1)
        hist = self.backbone(hist, mask=causal, src_key_padding_mask=~valid)
        hist = self.norm(hist)
        last_idx = valid.long().sum(dim=1).clamp(min=1) - 1
        context = hist[torch.arange(b, device=hist.device), last_idx]
        q = self.query(context)

        cand_emb = z[candidate_road_ids.clamp(min=0)]
        cand_input = torch.cat([cand_emb, candidate_geo.float()], dim=-1)
        c = self.candidate_proj(cand_input)
        logits = torch.einsum("bd,bkd->bk", q, c) / float(self.cfg.d_model) ** 0.5
        return logits.masked_fill(~candidate_mask.to(torch.bool), float("-inf"))

