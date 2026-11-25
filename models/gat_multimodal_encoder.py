from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class EncoderConfig:
    geo_in_dim: int = 0
    poi_in_dim: int = 21
    heat_in_dim: int = 2
    edge_dim: int = 0

    d_geo: int = 32
    d_poi: int = 32
    d_heat: int = 16

    hidden_dim: int = 128
    out_dim: int = 64

    num_layers: int = 4
    heads: int = 4
    dropout: float = 0.1


class MultiModalGATv2Encoder(nn.Module):
    """Multimodal road-segment encoder.

    Inputs:
    - x_geo: optional [N,3] (endpoint_lon_norm, endpoint_lat_norm, length_norm);
      set ``EncoderConfig.geo_in_dim=0`` to omit this branch for transfer.
    - x_poi: [N,21] POI, density and observed-speed level features
    - x_heat: [N,2] node out/in transition-count levels
    - edge_index: [2,E]
    - edge_attr: [E,De] optional edge-level heat features aligned with edge_index

    Output:
    - z: [N,out_dim]

    Notes:
    - No road_id embedding (for transferability).
    - Uses GATv2Conv over topology edges, optionally with edge heat features.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        # Set encoder hyperparameters
        if int(cfg.geo_in_dim) > 0:
            self.geo_encoder: Optional[nn.Module] = MLP(
                cfg.geo_in_dim, cfg.hidden_dim, cfg.d_geo, dropout=cfg.dropout
            )
            geo_out = cfg.d_geo
        else:
            # A transfer-oriented variant can omit absolute coordinates and
            # raw road length entirely.  The graph topology and ordinal
            # level attributes then carry the node information.
            self.geo_encoder = None
            geo_out = 0
        
        # If POI data is unavailable, fall back to a uni-modal setup
        if cfg.poi_in_dim > 0:
            self.poi_encoder: Optional[nn.Module] = MLP(
                cfg.poi_in_dim, cfg.hidden_dim, cfg.d_poi, dropout=cfg.dropout
            )
            poi_out = cfg.d_poi
        else:
            self.poi_encoder = None
            poi_out = 0

        # When node-level heat features are provided perform tri-modal fusion; otherwise fall back to bi-modal fusion
        if cfg.heat_in_dim > 0:
            self.heat_encoder: Optional[nn.Module] = MLP(
                cfg.heat_in_dim, cfg.hidden_dim, cfg.d_heat, dropout=cfg.dropout
            )
            heat_out = cfg.d_heat
        else:
            self.heat_encoder = None
            heat_out = 0

        # Tri-modal feature fusion
        fused_in = geo_out + poi_out + heat_out
        if fused_in <= 0:
            raise ValueError("At least one encoder input branch must be enabled")

        # Fuse the input modalities
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # Configure convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(cfg.num_layers):
            conv = GATv2Conv(
                in_channels=cfg.hidden_dim,
                out_channels=cfg.hidden_dim // cfg.heads,
                heads=cfg.heads,
                dropout=cfg.dropout,
                concat=True,
                edge_dim=(cfg.edge_dim if cfg.edge_dim > 0 else None),
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(cfg.hidden_dim))

        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.out_dim)

    def forward(
        self,
        *,
        x_geo: Optional[torch.Tensor],
        x_poi: Optional[torch.Tensor],
        x_heat: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = []

        if self.geo_encoder is not None:
            if x_geo is None:
                raise ValueError("cfg.geo_in_dim>0 but x_geo is None")
            parts.append(self.geo_encoder(x_geo))

        if self.poi_encoder is not None:
            if x_poi is None:
                raise ValueError("cfg.poi_in_dim>0 but x_poi is None")
            h_poi = self.poi_encoder(x_poi)
            parts.append(h_poi)

        if self.heat_encoder is not None:
            if x_heat is None:
                raise ValueError("cfg.heat_in_dim>0 but x_heat is None")
            h_heat = self.heat_encoder(x_heat)
            parts.append(h_heat)

        h = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        h = self.fuse(h)

        if self.cfg.edge_dim > 0:
            if edge_attr is None:
                raise ValueError("cfg.edge_dim>0 but edge_attr is None")
            if edge_attr.ndim != 2 or int(edge_attr.shape[1]) != int(self.cfg.edge_dim):
                raise ValueError(
                    f"Expected edge_attr shape [E,{self.cfg.edge_dim}], got {tuple(edge_attr.shape)}"
                )
        elif edge_attr is not None and edge_attr.numel() > 0:
            raise ValueError("edge_attr was provided but cfg.edge_dim==0")

        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            if self.cfg.edge_dim > 0:
                h = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h = conv(h, edge_index)
            h = nn.functional.dropout(h, p=self.cfg.dropout, training=self.training)
            h = norm(h + h_in)

        z = self.out_proj(h)
        return z
