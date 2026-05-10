from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

# Allow running as a script from repo root or elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from models.road_progress_decoder import RoadProgressDecoder, RoadProgressDecoderConfig
from utils.road_progress_dataset import RoadProgressDataset, road_progress_collate


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may reduce performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def setup_ddp(rank: int, world_size: int, master_port: str = '12355'):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up distributed training."""
    dist.destroy_process_group()


def _pick_device(device: str, rank: int = 0) -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")


def _load_z(emb_path: str) -> torch.Tensor:
    obj = torch.load(emb_path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "z" in obj:
        z = obj["z"]
    else:
        z = obj
    if not isinstance(z, torch.Tensor):
        raise TypeError(f"Expected embeddings to be a torch.Tensor, got {type(z)}")
    return z.detach().float().cpu()


def train_one_epoch(
    model,
    dl,
    opt,
    scaler,
    z_d,
    cfg,
    args,
    device,
    epoch,
    rank,
    world_size,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_end_loss = 0.0
    total_valid_steps = 0
    total_mae = 0.0
    total_neg_steps = 0
    total_all_steps = 0

    for batch_idx, (roads, road_mask, s_seq, s_mask, remaining_time_seq, depart_time, s_starts, s_ends, total_duration) in enumerate(dl):
        roads = roads.to(device, non_blocking=True)
        road_mask = road_mask.to(device, non_blocking=True)
        s_seq = s_seq.to(device, non_blocking=True)
        s_mask = s_mask.to(device, non_blocking=True)
        remaining_time_seq = remaining_time_seq.to(device, non_blocking=True) if args.use_remaining_time else None
        depart_time = depart_time.to(device, non_blocking=True).long() if args.use_depart_time else None
        total_duration = total_duration.to(device, non_blocking=True) if args.use_total_duration else None
        s_starts = s_starts.to(device, non_blocking=True) if args.use_start_point else None
        s_ends = s_ends.to(device, non_blocking=True) if args.use_end_point else None

        if s_seq.shape[1] < 2:
            continue
        if roads.shape[1] > cfg.max_roads:
            raise ValueError(f"Batch road length M={roads.shape[1]} exceeds cfg.max_roads={cfg.max_roads}")
        if s_seq.shape[1] > cfg.max_steps:
            raise ValueError(f"Batch s length T={s_seq.shape[1]} exceeds cfg.max_steps={cfg.max_steps}")

        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            pred_delta = model(
                roads_unique=roads,
                s_seq=s_seq,
                z=z_d,
                road_mask=road_mask,
                s_mask=s_mask,
                remaining_time_seq=remaining_time_seq,
                depart_time=depart_time,
                total_duration=total_duration,
                s_start=s_starts,
                s_end=s_ends,
            )

            target_delta_raw = s_seq[:, 1:] - s_seq[:, :-1]
            target_delta = target_delta_raw.clamp(min=0.0)
            valid = s_mask[:, 1:] & s_mask[:, :-1]

            if valid.sum().item() == 0:
                continue

            delta_loss = F.smooth_l1_loss(pred_delta[valid], target_delta[valid], reduction="mean")

            loss = delta_loss
            if args.use_end_point and args.end_loss_weight > 0.0:
                pred_s_seq = s_seq[:, :1] + torch.cumsum(pred_delta, dim=1)
                last_idx = s_mask.long().sum(dim=1) - 1
                last_idx_dec = (last_idx - 1).clamp(min=0)
                pred_s_last = pred_s_seq[torch.arange(s_seq.shape[0], device=device), last_idx_dec]
                true_s_last = s_seq[torch.arange(s_seq.shape[0], device=device), last_idx]
                end_loss = F.smooth_l1_loss(pred_s_last, true_s_last, reduction="mean")
                loss = delta_loss + args.end_loss_weight * end_loss

            # Gradient accumulation
            loss = loss / args.grad_accum_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Update weights every grad_accum_steps
        if (batch_idx + 1) % args.grad_accum_steps == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        total_loss += float(loss.item()) * args.grad_accum_steps
        if args.use_end_point and args.end_loss_weight > 0.0:
            total_end_loss += float(end_loss.item())
        n_valid = int(valid.sum().item())
        total_valid_steps += n_valid
        total_all_steps += int(valid.numel())
        total_neg_steps += int((target_delta_raw[valid] < 0).sum().item())

        with torch.no_grad():
            mae = (pred_delta[valid] - target_delta[valid]).abs().mean()
            total_mae += float(mae.item()) * n_valid

        # Print progress every 50 batches (only on rank 0)
        if rank == 0 and (batch_idx + 1) % 50 == 0:
            current_loss = total_loss / max(1, batch_idx + 1)
            print(f"  epoch {epoch} | batch {batch_idx + 1}/{len(dl)} | loss={current_loss:.4f}", flush=True)

    # Handle remaining gradients if batch count is not divisible by grad_accum_steps
    if len(dl) % args.grad_accum_steps != 0:
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    # Synchronize metrics across all GPUs
    if world_size > 1:
        metrics = torch.tensor(
            [total_loss, total_end_loss, total_valid_steps, total_mae, total_neg_steps, total_all_steps],
            device=device
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, total_end_loss, total_valid_steps, total_mae, total_neg_steps, total_all_steps = metrics.tolist()

    return {
        "total_loss": total_loss,
        "total_end_loss": total_end_loss,
        "total_valid_steps": total_valid_steps,
        "total_mae": total_mae,
        "total_neg_steps": total_neg_steps,
        "total_all_steps": total_all_steps,
        "num_batches": len(dl),
    }


def main_worker(rank: int, world_size: int, args):
    """Main training function for each process."""
    # Set random seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed + rank)  # Different seed for each rank

    # Setup DDP
    if world_size > 1:
        setup_ddp(rank, world_size, args.master_port)

    device = _pick_device(args.device, rank)
    is_main = (rank == 0)

    # Dataset and DataLoader
    ds = RoadProgressDataset(
        data_dir=args.data_dir,
        min_roads=int(args.min_roads),
        min_gps=int(args.min_gps),
        max_trajs=int(args.max_trajs),
        meta_time_index=int(args.meta_time_index),
    )

    # Use DistributedSampler for multi-GPU
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=road_progress_collate,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # Load embeddings
    z = _load_z(args.emb)
    max_road_id = max(int(r.max()) for r in ds.roads_list if len(r) > 0)
    if max_road_id >= int(z.shape[0]):
        raise ValueError(f"Dataset max road_id={max_road_id} exceeds embedding table size N={int(z.shape[0])}")

    # Model configuration
    cfg = RoadProgressDecoderConfig(
        emb_dim=int(z.shape[1]),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_enc_layers=int(args.n_enc_layers),
        n_dec_layers=int(args.n_dec_layers),
        dropout=float(args.dropout),
        max_roads=int(args.max_roads),
        max_steps=int(args.max_steps),
        use_depart_time=bool(args.use_depart_time),
        num_time_bins=int(args.num_time_bins),
        use_total_duration=bool(args.use_total_duration),
        use_remaining_time=bool(args.use_remaining_time),
        decoder_local_window=int(args.decoder_local_window),
        delta_s_max=float(args.delta_s_max),
        use_start_point=bool(args.use_start_point),
        use_end_point=bool(args.use_end_point),
        use_endpoints=bool(args.use_endpoints),
    )

    model = RoadProgressDecoder(cfg).to(device)

    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        if is_main:
            print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(model, DDP):
            model.module.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt and args.use_amp:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        if is_main:
            print(f"Resumed from epoch {start_epoch - 1}, starting at epoch {start_epoch}")

    z_d = z.to(device)

    if is_main:
        print("== Config ==")
        for k, v in sorted(vars(args).items()):
            print(f"{k}: {v}")
        print("device:", device)
        print("world_size:", world_size)
        print("num_embeddings:", int(z.shape[0]), "emb_dim:", int(z.shape[1]))
        print("num_samples:", len(ds), "num_raw_trajs:", len(ds.roads_list))
        print("model cfg:", asdict(cfg))
        print(f"Effective batch size: {args.batch_size * world_size * args.grad_accum_steps}")
        print(f"Starting from epoch: {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        metrics = train_one_epoch(
            model, dl, opt, scaler, z_d, cfg, args, device, epoch, rank, world_size
        )

        if is_main:
            if metrics["total_valid_steps"] == 0:
                print(f"epoch {epoch:03d}/{args.epochs} | no valid steps")
                continue

            avg_loss = metrics["total_loss"] / max(1, metrics["num_batches"])
            avg_end_loss = metrics["total_end_loss"] / max(1, metrics["num_batches"])
            avg_mae = metrics["total_mae"] / metrics["total_valid_steps"]
            neg_ratio = metrics["total_neg_steps"] / max(1, metrics["total_valid_steps"])
            valid_ratio = metrics["total_valid_steps"] / max(1, metrics["total_all_steps"])

            print(
                f"epoch {epoch:03d}/{args.epochs} | "
                f"loss={avg_loss:.6f} | end_loss={avg_end_loss:.6f} | valid_ratio={valid_ratio:.3f} | "
                f"delta_mae={avg_mae:.6f} | neg_target_ratio={neg_ratio:.6f}",
                flush=True
            )

            # Save checkpoint
            if args.save and args.save_interval > 0 and epoch % args.save_interval == 0:
                model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                ckpt = {
                    "model_state": model_state,
                    "optimizer_state": opt.state_dict(),
                    "scaler_state": scaler.state_dict() if args.use_amp else None,
                    "cfg": asdict(cfg),
                    "args": vars(args),
                    "epoch": epoch,
                    "metrics": {
                        "loss": avg_loss,
                        "end_loss": avg_end_loss,
                        "valid_ratio": valid_ratio,
                        "delta_mae": avg_mae,
                        "neg_target_ratio": neg_ratio,
                    }
                }
                Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                save_path = args.save.replace(".pt", f"_epoch{epoch:04d}.pt")
                torch.save(ckpt, save_path)
                print(f"  -> saved checkpoint to: {save_path}", flush=True)

    # Final save
    if is_main and args.save:
        model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        ckpt = {
            "model_state": model_state,
            "cfg": asdict(cfg),
            "args": vars(args),
        }
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, args.save)
        print("saved final checkpoint to:", args.save)

    if world_size > 1:
        cleanup_ddp()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Road-Progress decoder with multi-GPU support")

    parser.add_argument("--data-dir", type=str, default="data/xian_road_progress/road_progress_3s")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--emb", type=str, default="checkpoints/xian_z128.pt")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")

    parser.add_argument("--min-roads", type=int, default=3)
    parser.add_argument("--min-gps", type=int, default=10)
    parser.add_argument("--max-trajs", type=int, default=0)
    parser.add_argument("--meta-time-index", type=int, default=3)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-enc-layers", type=int, default=2)
    parser.add_argument("--n-dec-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-roads", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--delta-s-max", type=float, default=2.0)

    parser.add_argument("--use-depart-time", action="store_true")
    parser.add_argument("--num-time-bins", type=int, default=288)
    parser.add_argument("--use-total-duration", action="store_true")
    parser.add_argument("--use-remaining-time", action="store_true")
    parser.add_argument("--decoder-local-window", type=int, default=128)
    parser.add_argument("--use-start-point", action="store_true")
    parser.add_argument("--use-end-point", action="store_true")
    parser.add_argument("--use-endpoints", action="store_true")
    parser.add_argument("--end-loss-weight", type=float, default=0.5)

    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--save-interval", type=int, default=10)

    # Performance optimization arguments
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (0=main process only)")
    parser.add_argument("--use-amp", action="store_true", default=True, help="Use automatic mixed precision")
    parser.add_argument("--master-port", type=str, default="12355", help="DDP master port (change if running multiple trainings)")
    parser.add_argument("--no-amp", action="store_false", dest="use_amp")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs (e.g., '0,1,2,3')")

    args = parser.parse_args()
    if args.use_endpoints:
        args.use_start_point = True
        args.use_end_point = True

    # Parse GPU IDs
    gpu_ids = [int(x) for x in args.gpus.split(",")]
    world_size = len(gpu_ids)

    if world_size == 1:
        # Single GPU training
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        main_worker(0, 1, args)
    else:
        # Multi-GPU training with DDP
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        import torch.multiprocessing as mp
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

