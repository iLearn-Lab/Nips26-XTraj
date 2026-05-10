from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def _causal_mask(t: int, device: torch.device, window_size: int = 0) -> torch.Tensor:
    """Return a [T,T] boolean attention mask with True above diagonal.

    Args:
        t: sequence length
        device: torch device
        window_size: if > 0, limit attention to last window_size tokens (0 = unlimited)

    Returns:
        mask: [T,T] bool tensor, True means masked (cannot attend)
    """
    # Standard causal mask: cannot attend to future
    mask = torch.triu(torch.ones((t, t), dtype=torch.bool, device=device), diagonal=1)

    # Add window constraint: cannot attend to tokens too far in the past
    if window_size > 0:
        for i in range(t):
            # Position i can only attend to [max(0, i-window_size+1), i]
            if i >= window_size:
                mask[i, :i-window_size+1] = True

    return mask


@dataclass
class TrajARTransformerConfig:
    emb_dim: int = 64
    geo_dim: int = 0
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    max_len: int = 256

    # Optional condition: start/end road embedding conditioning.
    use_start_road_embedding: bool = False
    use_end_road_embedding: bool = False

    # Optional condition: departure time (discrete bins).
    use_depart_time: bool = False
    num_time_bins: int = 288

    # Optional condition: full-trajectory total duration in seconds.
    use_total_duration: bool = False

    # Attention window size: limit causal attention to last N tokens (0 = unlimited)
    causal_window: int = 0


@dataclass
class MaskedTrajectoryReconstructionConfig:
    emb_dim: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    max_len: int = 256


class MaskedTrajectoryReconstructionTransformer(nn.Module):
    """Lightweight bidirectional Transformer for masked trajectory reconstruction.

    Inputs:
        token_emb: FloatTensor [B,T,D] road-token embeddings before masking
        attn_mask: BoolTensor [B,T] valid-token mask
        masked_positions: BoolTensor [B,T] positions to replace with learned [MASK]

    Output:
        hidden: FloatTensor [B,T,D] contextualized token states
    """

    def __init__(self, cfg: MaskedTrajectoryReconstructionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(int(cfg.emb_dim), int(cfg.d_model))
        self.pos_emb = nn.Embedding(int(cfg.max_len), int(cfg.d_model))
        self.mask_token = nn.Parameter(torch.zeros(int(cfg.d_model)))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(cfg.d_model),
            nhead=int(cfg.n_heads),
            dim_feedforward=4 * int(cfg.d_model),
            dropout=float(cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(enc_layer, num_layers=int(cfg.n_layers))
        self.out_norm = nn.LayerNorm(int(cfg.d_model))
        self.out_proj = nn.Identity() if int(cfg.d_model) == int(cfg.emb_dim) else nn.Linear(int(cfg.d_model), int(cfg.emb_dim))

        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        *,
        token_emb: torch.Tensor,
        attn_mask: torch.Tensor,
        masked_positions: torch.Tensor,
    ) -> torch.Tensor:
        if token_emb.ndim != 3:
            raise ValueError(f"token_emb must be [B,T,D], got shape={tuple(token_emb.shape)}")
        if attn_mask.shape != token_emb.shape[:2]:
            raise ValueError("attn_mask shape must match token_emb[:2]")
        if masked_positions.shape != token_emb.shape[:2]:
            raise ValueError("masked_positions shape must match token_emb[:2]")
        if int(token_emb.shape[1]) > int(self.cfg.max_len):
            raise ValueError(f"T={int(token_emb.shape[1])} exceeds cfg.max_len={int(self.cfg.max_len)}")

        b, t, _ = token_emb.shape
        device = token_emb.device
        valid = attn_mask.to(torch.bool)
        masked = masked_positions.to(torch.bool) & valid

        h = self.in_proj(token_emb)
        if bool(masked.any()):
            h = h.clone()
            h[masked] = self.mask_token.to(device=device)

        pos = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
        h = h + self.pos_emb(pos)
        key_padding = ~valid
        h = self.backbone(h, src_key_padding_mask=key_padding)
        h = self.out_norm(h)
        h = self.out_proj(h)
        h = h * valid.unsqueeze(-1)
        return h


class GraphConstrainedARTransformer(nn.Module):
    """Causal Transformer for next-hop prediction with neighbor-constrained softmax.

    Inputs:
        road_ids: LongTensor [B,T] (pad tokens indicated by pad_id)
        z: FloatTensor [N, emb_dim] fixed road embeddings
        neighbors: LongTensor [N, K]
        neighbor_mask: BoolTensor [N, K]
        start_road (optional): LongTensor [B] start road id for embedding conditioning
        end_road (optional): LongTensor [B] end road id for embedding conditioning
        depart_time (optional): LongTensor [B] discrete time bins
        total_duration (optional): FloatTensor [B] full-trajectory duration in seconds

    Scoring:
        For each position t, scores candidates in the neighbor set of road_ids[:, t]
        using dot(query_t, proj(z[neighbor])).

    Returns:
        logits: FloatTensor [B,T,K] (masked invalid positions will be -inf)
    """

    def __init__(
        self,
        cfg: TrajARTransformerConfig,
        *,
        pad_id: int = -1,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad_id = int(pad_id)

        road_feat_dim = int(cfg.emb_dim) + int(cfg.geo_dim)

        self.in_proj = nn.Linear(road_feat_dim, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.out_norm = nn.LayerNorm(cfg.d_model)

        self.cand_proj = nn.Linear(road_feat_dim, cfg.d_model, bias=False)

        if bool(cfg.use_depart_time):
            self.time_emb = nn.Embedding(int(cfg.num_time_bins), cfg.d_model)
            nn.init.normal_(self.time_emb.weight, std=0.02)
        else:
            self.time_emb = None

        if bool(cfg.use_total_duration):
            self.duration_proj = nn.Linear(1, cfg.d_model)
        else:
            self.duration_proj = None

        self.start_emb_proj = nn.Linear(int(cfg.emb_dim), cfg.d_model) if bool(cfg.use_start_road_embedding) else None
        self.end_emb_proj = nn.Linear(int(cfg.emb_dim), cfg.d_model) if bool(cfg.use_end_road_embedding) else None

        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def _check_geo(self, z: torch.Tensor, geo_x: Optional[torch.Tensor]) -> None:
        if int(self.cfg.geo_dim) <= 0:
            return
        if geo_x is None:
            raise ValueError("geo_x must be provided when cfg.geo_dim > 0")
        if geo_x.ndim != 2:
            raise ValueError(f"geo_x must be [N,G], got shape={tuple(geo_x.shape)}")
        if int(geo_x.shape[0]) != int(z.shape[0]):
            raise ValueError(f"geo_x N={int(geo_x.shape[0])} != z N={int(z.shape[0])}")
        if int(geo_x.shape[1]) != int(self.cfg.geo_dim):
            raise ValueError(f"geo_x G={int(geo_x.shape[1])} != cfg.geo_dim={int(self.cfg.geo_dim)}")

    def _lookup_road_feat(
        self,
        road_ids: torch.Tensor,
        z: torch.Tensor,
        geo_x: Optional[torch.Tensor],
    ) -> torch.Tensor:
        feat = z[road_ids]
        if int(self.cfg.geo_dim) > 0:
            assert geo_x is not None
            feat = torch.cat([feat, geo_x[road_ids]], dim=-1)
        return feat

    def forward(
        self,
        *,
        road_ids: torch.Tensor,
        z: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
        geo_x: Optional[torch.Tensor] = None,
        start_road: Optional[torch.Tensor] = None,
        end_road: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        depart_time: Optional[torch.Tensor] = None,
        total_duration: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if road_ids.ndim != 2:
            raise ValueError(f"road_ids must be [B,T], got shape={tuple(road_ids.shape)}")
        b, t = road_ids.shape
        device = road_ids.device

        if t > self.cfg.max_len:
            raise ValueError(f"T={t} exceeds cfg.max_len={self.cfg.max_len}")

        self._check_geo(z, geo_x)

        if self.time_emb is not None:
            if depart_time is None:
                raise ValueError("depart_time must be provided when cfg.use_depart_time=True")
            if depart_time.ndim != 1 or int(depart_time.shape[0]) != int(b):
                raise ValueError(f"depart_time must be [B], got shape={tuple(depart_time.shape)}")

        if self.duration_proj is not None:
            if total_duration is None:
                raise ValueError("total_duration must be provided when cfg.use_total_duration=True")
            if total_duration.ndim != 1 or int(total_duration.shape[0]) != int(b):
                raise ValueError(f"total_duration must be [B], got shape={tuple(total_duration.shape)}")

        if self.start_emb_proj is not None:
            if start_road is None:
                raise ValueError("start_road must be provided when cfg.use_start_road_embedding=True")
            if start_road.ndim != 1 or int(start_road.shape[0]) != int(b):
                raise ValueError(f"start_road must be [B], got shape={tuple(start_road.shape)}")
        if self.end_emb_proj is not None:
            if end_road is None:
                raise ValueError("end_road must be provided when cfg.use_end_road_embedding=True")
            if end_road.ndim != 1 or int(end_road.shape[0]) != int(b):
                raise ValueError(f"end_road must be [B], got shape={tuple(end_road.shape)}")

        rid = road_ids
        valid_tok = rid != self.pad_id
        rid_safe = rid.clamp(min=0)
        x = self._lookup_road_feat(rid_safe, z, geo_x)
        x = x * valid_tok.unsqueeze(-1)

        h = self.in_proj(x)
        pos = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
        h = h + self.pos_emb(pos)

        if self.time_emb is not None and depart_time is not None:
            dt = depart_time.to(device=device).long().clamp(min=0, max=int(self.cfg.num_time_bins) - 1)
            h = h + self.time_emb(dt).unsqueeze(1)

        if self.duration_proj is not None and total_duration is not None:
            duration = torch.log1p(total_duration.to(device=device).float().clamp(min=0.0)).view(b, 1)
            h = h + self.duration_proj(duration).unsqueeze(1)

        if self.start_emb_proj is not None and start_road is not None:
            src = start_road.to(device=device).long().clamp(min=0)
            h = h + self.start_emb_proj(z[src]).unsqueeze(1)

        if self.end_emb_proj is not None and end_road is not None:
            dst = end_road.to(device=device).long().clamp(min=0)
            h = h + self.end_emb_proj(z[dst]).unsqueeze(1)

        causal = _causal_mask(t, device=device, window_size=self.cfg.causal_window)

        if attn_mask is None:
            key_padding = ~valid_tok
        else:
            key_padding = ~attn_mask.to(torch.bool)

        h = self.backbone(h, mask=causal, src_key_padding_mask=key_padding)
        h = self.out_norm(h)

        nb = neighbors[rid_safe]
        nb_mask = neighbor_mask[rid_safe]

        cand = self._lookup_road_feat(nb, z, geo_x)
        cand_h = self.cand_proj(cand)
        logits = (h.unsqueeze(2) * cand_h).sum(dim=-1) / math.sqrt(self.cfg.d_model)

        logits = logits.masked_fill(~nb_mask, float("-inf"))
        logits = logits.masked_fill(~valid_tok.unsqueeze(-1), float("-inf"))
        return logits

    @torch.no_grad()
    def generate_batch(
        self,
        *,
        start_roads: torch.Tensor,
        steps: int,
        z: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
        geo_x: Optional[torch.Tensor] = None,
        start_road: Optional[torch.Tensor] = None,
        end_road: Optional[torch.Tensor] = None,
        shortest_dist_table: Optional[torch.Tensor] = None,
        destination_roads: Optional[torch.Tensor] = None,
        top_k: int = 0,
        temperature: float = 1.0,
        greedy: bool = False,
        depart_time: Optional[torch.Tensor] = None,
        total_duration: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        stop_at_destination: bool = False,
        forbid_revisit: bool = False,
        visible_history_len: Optional[int] = None,
        return_top_candidate_probs: int = 0,
        return_constraint_override_count: bool = False,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[list[list[tuple[int, float]]]]] | tuple[list[torch.Tensor], torch.Tensor] | tuple[list[torch.Tensor], list[list[list[tuple[int, float]]]], torch.Tensor]:
        """Generate a batch of trajectories with constrained sampling.

        The full decoded trajectory is retained internally, but when
        ``visible_history_len`` is set to a positive integer, each decode step
        only exposes the most recent trailing window to the Transformer.

        When ``return_top_candidate_probs`` is positive, the method also
        returns per-step top-N candidate road ids and probabilities for each
        generated trajectory.
        """
        if device is None:
            device = next(self.parameters()).device

        self._check_geo(z, geo_x)

        if start_roads.ndim != 1:
            raise ValueError(f"start_roads must be [B], got shape={tuple(start_roads.shape)}")

        if visible_history_len is not None:
            visible_history_len = int(visible_history_len)
            if visible_history_len <= 0:
                visible_history_len = None

        return_top_candidate_probs = max(int(return_top_candidate_probs), 0)
        use_shortest_distance_rule = destination_roads is not None and shortest_dist_table is not None

        b = int(start_roads.shape[0])
        start_roads = start_roads.to(device=device).long()

        destination_tensor = None
        if destination_roads is not None:
            if destination_roads.ndim != 1 or int(destination_roads.shape[0]) != b:
                raise ValueError(f"destination_roads must be [B], got shape={tuple(destination_roads.shape)}")
            destination_tensor = destination_roads.to(device=device).long()

        if start_road is not None:
            if start_road.ndim != 1 or int(start_road.shape[0]) != b:
                raise ValueError(f"start_road must be [B], got shape={tuple(start_road.shape)}")
            start_road = start_road.to(device=device).long()

        if end_road is not None:
            if end_road.ndim != 1 or int(end_road.shape[0]) != b:
                raise ValueError(f"end_road must be [B], got shape={tuple(end_road.shape)}")
            end_road = end_road.to(device=device).long()

        if shortest_dist_table is not None:
            if shortest_dist_table.ndim != 2 or int(shortest_dist_table.shape[0]) != int(shortest_dist_table.shape[1]):
                raise ValueError(f"shortest_dist_table must be square [N,N], got shape={tuple(shortest_dist_table.shape)}")
            if int(shortest_dist_table.shape[0]) != int(neighbors.shape[0]):
                raise ValueError(
                    f"shortest_dist_table N={int(shortest_dist_table.shape[0])} != neighbors N={int(neighbors.shape[0])}"
                )
            shortest_dist_table = shortest_dist_table.to(device=device).float()

        if self.time_emb is not None:
            if depart_time is None:
                raise ValueError("depart_time must be provided when cfg.use_depart_time=True")
            if depart_time.ndim != 1 or int(depart_time.shape[0]) != b:
                raise ValueError(f"depart_time must be [B], got shape={tuple(depart_time.shape)}")
            depart_time = depart_time.to(device=device).long()

        if self.duration_proj is not None:
            if total_duration is None:
                raise ValueError("total_duration must be provided when cfg.use_total_duration=True")
            if total_duration.ndim != 1 or int(total_duration.shape[0]) != b:
                raise ValueError(f"total_duration must be [B], got shape={tuple(total_duration.shape)}")
            total_duration = total_duration.to(device=device).float()

        num_steps = max(int(steps), 0)
        road_ids = torch.full((b, num_steps + 1), fill_value=self.pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros((b, num_steps + 1), dtype=torch.bool, device=device)
        lengths = torch.ones((b,), dtype=torch.long, device=device)
        finished = torch.zeros((b,), dtype=torch.bool, device=device)

        road_ids[:, 0] = start_roads
        attn_mask[:, 0] = True
        visited = [set([int(start_roads[i].item())]) for i in range(b)] if forbid_revisit else None
        step_top_candidates: list[list[list[tuple[int, float]]]] | None = None
        if return_top_candidate_probs > 0:
            step_top_candidates = [[] for _ in range(b)]
        constraint_override_count = torch.zeros((b,), dtype=torch.long, device=device)

        if destination_tensor is not None and stop_at_destination:
            finished = start_roads == destination_tensor

        for _ in range(num_steps):
            active_idx = torch.nonzero(~finished, as_tuple=False).squeeze(-1)
            if int(active_idx.numel()) == 0:
                break

            active_lengths = lengths[active_idx]
            cur_len = int(active_lengths.max().item())
            if visible_history_len is None:
                window_start = 0
            else:
                window_start = max(0, cur_len - visible_history_len)
            window_road_ids = road_ids[active_idx, window_start:cur_len]
            window_attn_mask = attn_mask[active_idx, window_start:cur_len]
            logits = self.forward(
                road_ids=window_road_ids,
                z=z,
                neighbors=neighbors,
                neighbor_mask=neighbor_mask,
                geo_x=geo_x,
                start_road=(start_road[active_idx] if start_road is not None else None),
                end_road=(end_road[active_idx] if end_road is not None else None),
                attn_mask=window_attn_mask,
                depart_time=(depart_time[active_idx] if depart_time is not None else None),
                total_duration=(total_duration[active_idx] if total_duration is not None else None),
            )

            last_pos = (active_lengths - 1) - window_start
            last = logits[torch.arange(int(active_idx.numel()), device=device), last_pos]
            if visited is not None:
                cur = road_ids[active_idx, active_lengths - 1]
                candidate_ids = neighbors[cur]
                revisit_mask = torch.zeros_like(last, dtype=torch.bool)
                for row_idx, global_idx in enumerate(active_idx.detach().cpu().tolist()):
                    seen = visited[int(global_idx)]
                    if seen:
                        revisit_mask[row_idx] = torch.tensor(
                            [int(candidate) in seen for candidate in candidate_ids[row_idx].detach().cpu().tolist()],
                            device=device,
                            dtype=torch.bool,
                        )
                last = last.masked_fill(revisit_mask, float("-inf"))
            finite_rows = torch.isfinite(last).any(dim=-1)

            if not bool(finite_rows.any()):
                finished[active_idx] = True
                continue

            stalled_idx = active_idx[~finite_rows]
            if int(stalled_idx.numel()) > 0:
                finished[stalled_idx] = True

            sample_idx = active_idx[finite_rows]
            sample_logits = last[finite_rows]

            if temperature != 1.0:
                sample_logits = sample_logits / max(float(temperature), 1e-6)

            probs = torch.softmax(sample_logits, dim=-1)
            report_candidate_probs = probs
            report_candidate_slots = None
            valid_candidate_mask = torch.isfinite(sample_logits)
            top_prob_slots = torch.argmax(probs.masked_fill(~valid_candidate_mask, -1.0), dim=-1)
            if use_shortest_distance_rule and destination_tensor is not None:
                cur = road_ids[sample_idx, lengths[sample_idx] - 1]
                candidate_ids = neighbors[cur]
                dest = destination_tensor[sample_idx]
                cur_dist = shortest_dist_table[cur, dest]
                cand_dist = shortest_dist_table[
                    candidate_ids.reshape(-1),
                    dest.unsqueeze(1).expand_as(candidate_ids).reshape(-1),
                ].reshape_as(candidate_ids)

                cur_reachable = cur_dist >= 0
                cand_reachable = cand_dist >= 0
                non_worsening_mask = cand_reachable & (cand_dist <= cur_dist.unsqueeze(1))
                eligible_mask = torch.where(cur_reachable.unsqueeze(1), non_worsening_mask, cand_reachable)
                eligible_mask = eligible_mask & valid_candidate_mask

                ranked_scores = probs.masked_fill(~valid_candidate_mask, -1.0)
                sorted_slots = torch.argsort(ranked_scores, dim=-1, descending=True)
                ranked_eligible = eligible_mask.gather(dim=-1, index=sorted_slots)
                has_eligible = ranked_eligible.any(dim=-1)
                first_eligible_rank = ranked_eligible.to(torch.int64).argmax(dim=-1)
                chosen_rank = torch.where(has_eligible, first_eligible_rank, torch.zeros_like(first_eligible_rank))
                picked_slots = sorted_slots.gather(dim=-1, index=chosen_rank.unsqueeze(-1)).squeeze(-1)
                if return_constraint_override_count:
                    override_mask = has_eligible & (picked_slots != top_prob_slots)
                    constraint_override_count[sample_idx] += override_mask.to(torch.long)
            elif greedy:
                picked_slots = top_prob_slots
            elif top_k and top_k > 0:
                k = min(int(top_k), int(probs.shape[-1]))
                vals, idx = torch.topk(probs, k=k, dim=-1)
                vals = vals / vals.sum(dim=-1, keepdim=True)
                report_candidate_probs = vals
                report_candidate_slots = idx
                sampled_in_topk = torch.multinomial(vals, num_samples=1)
                picked_slots = idx.gather(dim=-1, index=sampled_in_topk).squeeze(-1)
            else:
                picked_slots = torch.multinomial(probs, num_samples=1).squeeze(-1)

            cur_pos = lengths[sample_idx] - 1
            cur = road_ids[sample_idx, cur_pos]
            if step_top_candidates is not None:
                top_n = min(int(return_top_candidate_probs), int(report_candidate_probs.shape[-1]))
                if top_n > 0:
                    top_vals, top_local_idx = torch.topk(report_candidate_probs, k=top_n, dim=-1)
                    if report_candidate_slots is None:
                        top_slots = top_local_idx
                    else:
                        top_slots = report_candidate_slots.gather(dim=-1, index=top_local_idx)
                    top_candidate_ids = neighbors[cur].gather(dim=-1, index=top_slots)
                    for row_idx, global_idx in enumerate(sample_idx.detach().cpu().tolist()):
                        entries = [
                            (int(candidate_id), float(probability))
                            for candidate_id, probability in zip(
                                top_candidate_ids[row_idx].detach().cpu().tolist(),
                                top_vals[row_idx].detach().cpu().tolist(),
                            )
                        ]
                        step_top_candidates[int(global_idx)].append(entries)
            nxt = neighbors[cur, picked_slots]
            next_pos = lengths[sample_idx]
            road_ids[sample_idx, next_pos] = nxt
            attn_mask[sample_idx, next_pos] = True
            lengths[sample_idx] = next_pos + 1
            if visited is not None:
                for global_idx, nxt_val in zip(sample_idx.detach().cpu().tolist(), nxt.detach().cpu().tolist()):
                    visited[int(global_idx)].add(int(nxt_val))

            if destination_tensor is not None and stop_at_destination:
                finished[sample_idx] = nxt == destination_tensor[sample_idx]

        trajs = [road_ids[i, : int(lengths[i].item())].clone() for i in range(b)]
        if step_top_candidates is not None and return_constraint_override_count:
            return trajs, step_top_candidates, constraint_override_count.clone()
        if step_top_candidates is not None:
            return trajs, step_top_candidates
        if return_constraint_override_count:
            return trajs, constraint_override_count.clone()
        return trajs

    @torch.no_grad()
    def generate(
        self,
        *,
        start_road: int,
        steps: int,
        z: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
        geo_x: Optional[torch.Tensor] = None,
        start_road_id: Optional[int] = None,
        end_road_id: Optional[int] = None,
        shortest_dist_table: Optional[torch.Tensor] = None,
        destination_road: Optional[torch.Tensor | int] = None,
        top_k: int = 0,
        temperature: float = 1.0,
        depart_time: Optional[torch.Tensor] = None,
        total_duration: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        stop_at_destination: bool = False,
        forbid_revisit: bool = False,
        visible_history_len: Optional[int] = None,
        return_top_candidate_probs: int = 0,
        return_constraint_override_count: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[list[tuple[int, float]]]] | tuple[torch.Tensor, int] | tuple[torch.Tensor, list[list[tuple[int, float]]], int]:
        if device is None:
            device = next(self.parameters()).device

        self._check_geo(z, geo_x)

        destination_tensor = None
        if destination_road is not None:
            if isinstance(destination_road, int):
                destination_tensor = torch.tensor([int(destination_road)], dtype=torch.long, device=device)
            else:
                if destination_road.ndim == 0:
                    destination_tensor = destination_road.view(1).to(device=device).long()
                elif destination_road.ndim == 1 and int(destination_road.shape[0]) == 1:
                    destination_tensor = destination_road.to(device=device).long()
                else:
                    raise ValueError("generate expects destination_road as int, scalar tensor, or shape [1]")

        if self.time_emb is not None:
            if depart_time is None:
                raise ValueError("depart_time must be provided when cfg.use_depart_time=True")
            if depart_time.ndim == 0:
                depart_time = depart_time.view(1)
            if depart_time.ndim != 1 or int(depart_time.shape[0]) != 1:
                raise ValueError("generate expects depart_time as scalar or shape [1]")
            depart_time = depart_time.to(device=device).long()

        if self.duration_proj is not None:
            if total_duration is None:
                raise ValueError("total_duration must be provided when cfg.use_total_duration=True")
            if total_duration.ndim == 0:
                total_duration = total_duration.view(1)
            if total_duration.ndim != 1 or int(total_duration.shape[0]) != 1:
                raise ValueError("generate expects total_duration as scalar or shape [1]")
            total_duration = total_duration.to(device=device).float()

        trajs = self.generate_batch(
            start_roads=torch.tensor([int(start_road)], dtype=torch.long, device=device),
            steps=steps,
            z=z,
            neighbors=neighbors,
            neighbor_mask=neighbor_mask,
            geo_x=geo_x,
            start_road=(torch.tensor([int(start_road_id)], dtype=torch.long, device=device) if start_road_id is not None else None),
            end_road=(torch.tensor([int(end_road_id)], dtype=torch.long, device=device) if end_road_id is not None else None),
            shortest_dist_table=shortest_dist_table,
            destination_roads=destination_tensor,
            top_k=top_k,
            temperature=temperature,
            depart_time=depart_time,
            total_duration=total_duration,
            device=device,
            stop_at_destination=stop_at_destination,
            forbid_revisit=forbid_revisit,
            visible_history_len=visible_history_len,
            return_top_candidate_probs=return_top_candidate_probs,
            return_constraint_override_count=return_constraint_override_count,
        )
        if return_top_candidate_probs > 0 and return_constraint_override_count:
            traj_list, top_candidate_probs, override_counts = trajs
            return traj_list[0], top_candidate_probs[0], int(override_counts[0].item())
        if return_top_candidate_probs > 0:
            traj_list, top_candidate_probs = trajs
            return traj_list[0], top_candidate_probs[0]
        if return_constraint_override_count:
            traj_list, override_counts = trajs
            return traj_list[0], int(override_counts[0].item())
        return trajs[0]
