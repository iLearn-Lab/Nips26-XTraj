from __future__ import annotations
import torch

import argparse
import sys
from dataclasses import asdict
from pathlib import Path


# Allow running as a script from repo root or elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from utils.graph_builder import RoadGeoNormalizer  # ensure this class is importable
# Allowlist custom class for safe globals
torch.serialization.add_safe_globals([RoadGeoNormalizer])
from models.gat_multimodal_encoder import EncoderConfig, MultiModalGATv2Encoder  # noqa: E402
from utils.graph_builder import build_city_pyg_graph  # noqa: E402


def _pick_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export road-segment embeddings z[N,64] from a trained encoder checkpoint.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint saved by train_encoder_linkpred.py")
    parser.add_argument("--out", type=str, required=True, help="Output path for exported embeddings (.pt)")

    parser.add_argument("--city", type=str, default="xian", choices=["xian", "chengdu"])
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--traj-dir", type=str, default="", help="Optional trajectory directory for rebuilding heat features; defaults to data-root/city")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)
    traj_dir = str(Path(args.traj_dir)) if args.traj_dir else str(Path(args.data_root) / args.city)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load ckpt (trusted local file).
    # Use weights_only=True to avoid unpickling arbitrary objects (safer; we only need tensors).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg_dict = ckpt.get("cfg", {})
    ckpt_args = ckpt.get("args", {}) or {}
    ckpt_meta = ckpt.get("meta", {}) or {}

    include_node_heat = bool(ckpt_meta.get("include_node_heat", True))
    include_edge_heat = bool(ckpt_meta.get("include_edge_heat", not bool(ckpt_meta.get("edge_heat_removed", False))))
    edge_heat_norm = str(ckpt_meta.get("edge_heat_normalization", "log1p_minmax"))
    min_unique_len = int(ckpt_args.get("min_unique_len", 5))
    poi_vocab = ckpt_meta.get("poi_vocab", None)

    # Rebuild the exact same graph feature dimensions (esp. POI vocab).
    data, meta = build_city_pyg_graph(
        city=args.city,
        data_root=args.data_root,
        poi_vocab=poi_vocab,
        min_unique_len=min_unique_len,
        include_node_heat=include_node_heat,
        include_edge_heat=include_edge_heat,
        edge_heat_norm=edge_heat_norm,
        traj_dir=traj_dir,
    )

    cfg = EncoderConfig(**cfg_dict)
    model = MultiModalGATv2Encoder(cfg)
    model.load_state_dict(ckpt["model_state"])

    device = _pick_device(args.device)
    model = model.to(device)
    model.eval()

    x_geo = data.x_geo.to(device)
    x_poi = data.x_poi.to(device) if cfg.poi_in_dim > 0 else None
    x_heat = data.x_heat.to(device) if getattr(data, "x_heat", None) is not None and cfg.heat_in_dim > 0 else None
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device) if getattr(data, "edge_attr", None) is not None and cfg.edge_dim > 0 else None

    with torch.no_grad():
        z = model(
            x_geo=x_geo,
            x_poi=x_poi,
            x_heat=x_heat,
            edge_index=edge_index,
            edge_attr=edge_attr,
        ).detach().cpu()

    # Avoid pickling custom classes in meta to keep the export easy to load.
    geo_norm = meta.get("geo_normalizer", None)
    if hasattr(geo_norm, "__dataclass_fields__"):
        geo_norm = asdict(geo_norm)

    export = {
        "city": args.city,
        "z": z,  # [N, out_dim]
        "cfg": asdict(cfg),
        "graph_meta": {
            "poi_vocab": meta.get("poi_vocab", None),
            "include_node_heat": meta.get("include_node_heat", None),
            "include_edge_heat": meta.get("include_edge_heat", None),
            "edge_heat_normalization": meta.get("edge_heat_normalization", None),
            "geo_normalizer": geo_norm,
        },
        "source_ckpt": str(ckpt_path),
        "build_args": {
            "data_root": str(args.data_root),
            "traj_dir": traj_dir,
            "min_unique_len": min_unique_len,
        },
    }

    torch.save(export, out_path)
    print("saved embeddings to:", out_path)
    print("z shape:", tuple(z.shape))


if __name__ == "__main__":
    main()
