"""Bidirectional masked-road reconstruction used to train the GAT."""

from dataclasses import dataclass
import torch
import torch.nn as nn


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
