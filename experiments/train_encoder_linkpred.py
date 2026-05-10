from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

# Allow running as a script from repo root or elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from models.gat_multimodal_encoder import EncoderConfig, MultiModalGATv2Encoder, lookup_trajectory_embeddings
from models.traj_ar_transformer import MaskedTrajectoryReconstructionConfig, MaskedTrajectoryReconstructionTransformer
from utils.graph_builder import build_city_pyg_graph
from utils.traj_dataset import TrajSegmentSpec, TrajectorySegmentDataset, pad_collate_1d


def _set_random_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_worker_init_fn(base_seed: int):
    base_seed = int(base_seed)

    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = base_seed + int(worker_id)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _worker_init_fn


def _pick_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _split_day_indices(num_days: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if num_days < 2:
        raise ValueError(f"Need at least 2 day files for train/validation split, got {num_days}")

    indices = list(range(num_days))
    rng = random.Random(int(seed))
    rng.shuffle(indices)

    train_count = int(math.floor(num_days * float(train_ratio)))
    train_count = max(1, min(train_count, num_days - 1))
    train_idx = sorted(indices[:train_count])
    val_idx = sorted(indices[train_count:])
    if not val_idx:
        raise RuntimeError("Validation split is empty")
    return train_idx, val_idx


def _split_item_indices(num_items: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if num_items < 2:
        raise ValueError(f"Need at least 2 items for train/validation split, got {num_items}")

    indices = list(range(num_items))
    rng = random.Random(int(seed))
    rng.shuffle(indices)

    train_count = int(math.floor(num_items * float(train_ratio)))
    train_count = max(1, min(train_count, num_items - 1))
    train_idx = sorted(indices[:train_count])
    val_idx = sorted(indices[train_count:])
    if not val_idx:
        raise RuntimeError("Validation split is empty")
    return train_idx, val_idx


def _collect_subset_indices(ds: TrajectorySegmentDataset, allowed_file_ids: set[int]) -> list[int]:
    if ds.spec.start_mode == "sliding":
        out: list[int] = []
        for idx, (traj_idx, _) in enumerate(ds._sliding_index):
            if int(ds.traj_file_ids[traj_idx]) in allowed_file_ids:
                out.append(idx)
        return out
    return [idx for idx, file_id in enumerate(ds.traj_file_ids) if int(file_id) in allowed_file_ids]


def _sample_mask_positions(attn_mask: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    if attn_mask.ndim != 2:
        raise ValueError(f"attn_mask must be [B,T], got shape={tuple(attn_mask.shape)}")

    b, t = attn_mask.shape
    out = torch.zeros((b, t), dtype=torch.bool, device=attn_mask.device)
    for i in range(b):
        valid_idx = torch.nonzero(attn_mask[i], as_tuple=False).squeeze(-1)
        valid_len = int(valid_idx.numel())
        if valid_len < 3:
            continue

        interior_idx = valid_idx[1:-1]
        interior_len = int(interior_idx.numel())
        if interior_len <= 0:
            continue

        num_mask = int(round(float(mask_ratio) * interior_len))
        num_mask = min(max(1, num_mask), interior_len)
        perm = torch.randperm(interior_len, device=attn_mask.device)
        pick = interior_idx[perm[:num_mask]]
        out[i, pick] = True
    return out


def _topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 5)) -> dict[str, float]:
    if logits.numel() == 0:
        return {f"acc@{k}": 0.0 for k in ks}

    max_k = min(max(ks), int(logits.shape[-1]))
    _, topi = torch.topk(logits, k=max_k, dim=-1)
    out: dict[str, float] = {}
    for kk in ks:
        eval_k = min(int(kk), int(topi.shape[-1]))
        hit = (topi[:, :eval_k] == targets.unsqueeze(-1)).any(dim=-1)
        out[f"acc@{kk}"] = float(hit.float().mean().item())
    return out


def _run_epoch(
    *,
    encoder: MultiModalGATv2Encoder,
    decoder: MaskedTrajectoryReconstructionTransformer,
    dl: DataLoader,
    device: torch.device,
    x_geo_d: torch.Tensor,
    x_poi_d: torch.Tensor | None,
    x_heat_d: torch.Tensor | None,
    edge_index_d: torch.Tensor,
    edge_attr_d: torch.Tensor | None,
    mask_ratio: float,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    is_train = optimizer is not None
    if is_train:
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()

    total_loss = 0.0
    total_batches = 0
    total_masked = 0
    total_acc1_weighted = 0.0
    total_acc5_weighted = 0.0
    last_debug = None

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for road_ids, attn_mask in dl:
            road_ids = road_ids.to(device)
            attn_mask = attn_mask.to(device)
            if road_ids.numel() == 0:
                continue

            masked_pos = _sample_mask_positions(attn_mask, mask_ratio=float(mask_ratio))
            if not bool(masked_pos.any()):
                continue

            z = encoder(
                x_geo=x_geo_d,
                x_poi=x_poi_d,
                x_heat=x_heat_d,
                edge_index=edge_index_d,
                edge_attr=edge_attr_d,
            )

            token_emb = lookup_trajectory_embeddings(
                node_embeddings=z,
                road_ids=road_ids.clamp(min=0),
            )
            hidden = decoder(
                token_emb=token_emb,
                attn_mask=attn_mask,
                masked_positions=masked_pos,
            )

            h_masked = hidden[masked_pos]
            targets = road_ids[masked_pos]
            logits = h_masked @ z.transpose(0, 1)
            loss = F.cross_entropy(logits, targets, reduction="mean")

            if is_train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            metrics = _topk_accuracy(logits, targets, ks=(1, 5))
            n_masked = int(targets.numel())
            total_loss += float(loss.item())
            total_batches += 1
            total_masked += n_masked
            total_acc1_weighted += metrics["acc@1"] * n_masked
            total_acc5_weighted += metrics["acc@5"] * n_masked

            valid_lens = attn_mask.sum(dim=1)
            masked_lens = masked_pos.sum(dim=1)
            last_debug = {
                "road_ids_shape": tuple(road_ids.shape),
                "attn_mask_shape": tuple(attn_mask.shape),
                "valid_lens": [int(v) for v in valid_lens.detach().cpu().tolist()],
                "masked_lens": [int(v) for v in masked_lens.detach().cpu().tolist()],
                "masked_logits_shape": tuple(logits.shape),
            }

    if total_batches == 0 or total_masked == 0:
        return {
            "has_valid_batches": False,
            "loss": 0.0,
            "acc@1": 0.0,
            "acc@5": 0.0,
            "masked_tokens": 0,
            "debug": last_debug,
        }

    return {
        "has_valid_batches": True,
        "loss": total_loss / total_batches,
        "acc@1": total_acc1_weighted / total_masked,
        "acc@5": total_acc5_weighted / total_masked,
        "masked_tokens": total_masked,
        "debug": last_debug,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multimodal road encoder via masked trajectory reconstruction.")
    parser.add_argument("--city", type=str, default="xian", choices=["xian", "chengdu"])
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--traj-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out-dim", type=int, default=64)

    parser.add_argument("--decoder-d-model", type=int, default=64)
    parser.add_argument("--decoder-heads", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--decoder-dropout", type=float, default=0.1)

    parser.add_argument("--segment-len", type=int, default=64)
    parser.add_argument("--start-mode", type=str, default="random", choices=["fixed", "random", "sliding"])
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-trajs", type=int, default=0)
    parser.add_argument("--min-unique-len", type=int, default=6)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=23)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible training")
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--include-node-heat",
        action="store_true",
        help="include node-level heat features x_heat (default: enabled)",
    )
    parser.add_argument(
        "--disable-node-heat",
        action="store_true",
        help="disable node-level heat features x_heat",
    )

    parser.add_argument("--save", type=str, default="./checkpoints", help="optional encoder checkpoint path")
    args = parser.parse_args()

    if args.hidden_dim % args.heads != 0:
        raise ValueError(f"hidden_dim must be divisible by heads, got {args.hidden_dim} % {args.heads} != 0")
    if args.decoder_d_model % args.decoder_heads != 0:
        raise ValueError(
            f"decoder_d_model must be divisible by decoder_heads, got {args.decoder_d_model} % {args.decoder_heads} != 0"
        )
    if not (0.0 < float(args.mask_ratio) < 1.0):
        raise ValueError(f"mask_ratio must be in (0, 1), got {args.mask_ratio}")

    include_node_heat = True
    if args.disable_node_heat:
        include_node_heat = False
    if args.include_node_heat:
        include_node_heat = True

    device = _pick_device(args.device)
    _set_random_seed(args.seed)


    data, meta = build_city_pyg_graph(
        city=args.city,
        data_root=args.data_root,
        min_unique_len=args.min_unique_len,
        include_node_heat=include_node_heat,
        traj_dir=args.traj_dir,
    )

    num_nodes = int(data.num_nodes)
    edge_index = data.edge_index


    cfg = EncoderConfig(
        geo_in_dim=int(data.x_geo.shape[1]),
        poi_in_dim=int(data.x_poi.shape[1]),
        heat_in_dim=(int(data.x_heat.shape[1]) if getattr(data, "x_heat", None) is not None else 0),
        edge_dim=(int(data.edge_attr.shape[1]) if getattr(data, "edge_attr", None) is not None and data.edge_attr.numel() > 0 else 0),
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        out_dim=args.out_dim,
    )

    dec_cfg = MaskedTrajectoryReconstructionConfig(
        emb_dim=args.out_dim,
        d_model=args.decoder_d_model,
        n_heads=args.decoder_heads,
        n_layers=args.decoder_layers,
        dropout=args.decoder_dropout,
        max_len=args.segment_len,
    )


    encoder = MultiModalGATv2Encoder(cfg).to(device)
    decoder = MaskedTrajectoryReconstructionTransformer(dec_cfg).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    x_geo_d = data.x_geo.to(device)
    x_poi_d = data.x_poi.to(device) if cfg.poi_in_dim > 0 else None
    x_heat_d = data.x_heat.to(device) if cfg.heat_in_dim > 0 else None
    edge_index_d = edge_index.to(device)
    edge_attr_d = data.edge_attr.to(device) if cfg.edge_dim > 0 else None

    spec = TrajSegmentSpec(
        segment_len=args.segment_len,
        start_mode=args.start_mode,
        start_idx=args.start_idx,
        stride=args.stride,
        seed=args.seed,
    )
    ds = TrajectorySegmentDataset(
        traj_dir=args.traj_dir,
        spec=spec,
        min_unique_len=args.min_unique_len,
        max_trajs=args.max_trajs,
    )

    available_file_ids = sorted({int(file_id) for file_id in ds.traj_file_ids})
    if len(available_file_ids) >= 2:
        split_train_idx, split_val_idx = _split_day_indices(len(available_file_ids), args.train_ratio, args.split_seed)
        train_days = [available_file_ids[i] for i in split_train_idx]
        val_days = [available_file_ids[i] for i in split_val_idx]
        train_idx = _collect_subset_indices(ds, set(train_days))
        val_idx = _collect_subset_indices(ds, set(val_days))
        split_mode = "day"
    else:
        train_idx, val_idx = _split_item_indices(len(ds), args.train_ratio, args.split_seed)
        train_days = available_file_ids
        val_days = available_file_ids
        split_mode = "sample_fallback"
    if not train_idx or not val_idx:
        raise RuntimeError("Train/validation split produced an empty subset")

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(int(args.seed))
    worker_init_fn = _make_worker_init_fn(args.seed) if int(args.num_workers) > 0 else None

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pad_collate_1d,
        generator=data_loader_generator,
        worker_init_fn=worker_init_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate_1d,
        generator=data_loader_generator,
        worker_init_fn=worker_init_fn,
    )


    best_val_loss = float("inf")
    best_joint_encoder_state = None
    save_path = Path(args.save) if args.save else None
    epoch_width = max(3, len(str(args.epochs)))

    for epoch in range(1, args.epochs + 1):
        train_stats = _run_epoch(
            encoder=encoder,
            decoder=decoder,
            dl=train_dl,
            device=device,
            x_geo_d=x_geo_d,
            x_poi_d=x_poi_d,
            x_heat_d=x_heat_d,
            edge_index_d=edge_index_d,
            edge_attr_d=edge_attr_d,
            mask_ratio=args.mask_ratio,
            optimizer=opt,
        )
        val_stats = _run_epoch(
            encoder=encoder,
            decoder=decoder,
            dl=val_dl,
            device=device,
            x_geo_d=x_geo_d,
            x_poi_d=x_poi_d,
            x_heat_d=x_heat_d,
            edge_index_d=edge_index_d,
            edge_attr_d=edge_attr_d,
            mask_ratio=args.mask_ratio,
            optimizer=None,
        )

        joint_improved = False
        if val_stats["has_valid_batches"] and val_stats["loss"] < best_val_loss:
            best_val_loss = float(val_stats["loss"])
            best_joint_encoder_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            joint_improved = True

        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_stats['loss']:.6f} | "
            f"val_loss={val_stats['loss']:.6f}"
        )

        if save_path is not None:
            current_state = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
            epoch_path = save_path.with_name(f"{save_path.stem}_epoch{epoch:0{epoch_width}d}{save_path.suffix}")
            epoch_ckpt = {
                "model_state": current_state,
                "cfg": asdict(cfg),
                "meta": meta,
                "args": {
                    **vars(args),
                    "pretrain_task": "masked_traj_reconstruction",
                    "selection_metric": "epoch_snapshot",
                    "epoch": epoch,
                },
            }
            torch.save(epoch_ckpt, epoch_path)
            print("saved epoch encoder checkpoint to:", epoch_path)

            if joint_improved and best_joint_encoder_state is not None:
                joint_ckpt = {
                    "model_state": best_joint_encoder_state,
                    "cfg": asdict(cfg),
                    "meta": meta,
                    "args": {
                        **vars(args),
                        "pretrain_task": "masked_traj_reconstruction",
                        "selection_metric": "joint_val_loss",
                        "epoch": epoch,
                    },
                }
                torch.save(joint_ckpt, save_path)
                print("updated joint-best encoder checkpoint to:", save_path)

    if save_path is not None:
        default_state = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}

        if not save_path.exists():
            joint_state = best_joint_encoder_state if best_joint_encoder_state is not None else default_state
            joint_ckpt = {
                "model_state": joint_state,
                "cfg": asdict(cfg),
                "meta": meta,
                "args": {**vars(args), "pretrain_task": "masked_traj_reconstruction", "selection_metric": "joint_val_loss", "epoch": args.epochs},
            }
            torch.save(joint_ckpt, save_path)
            print("saved joint-best encoder checkpoint to:", save_path)


if __name__ == "__main__":
    main()
