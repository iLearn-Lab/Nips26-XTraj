from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

# Allow running as a script from repo root or elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.traj_ar_transformer import GraphConstrainedARTransformer, TrajARTransformerConfig
from utils.graph_builder import build_city_pyg_graph
from utils.neighbor_table import build_padded_neighbors
from utils.traj_dataset import (
    pad_collate_1d,
    pad_collate_1d_with_depart_time,
    pad_collate_1d_with_depart_time_and_total_duration,
    pad_collate_1d_with_destination,
    pad_collate_1d_with_destination_and_depart_time,
    pad_collate_1d_with_destination_and_depart_time_and_total_duration,
    pad_collate_1d_with_destination_and_total_duration,
    pad_collate_1d_with_start_and_destination,
    pad_collate_1d_with_start_and_destination_and_depart_time,
    pad_collate_1d_with_start_and_destination_and_depart_time_and_total_duration,
    pad_collate_1d_with_start_and_destination_and_total_duration,
    pad_collate_1d_with_total_duration,
)


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


def _load_z(emb_path: str) -> torch.Tensor:
    obj = torch.load(emb_path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "z" in obj:
        z = obj["z"]
    else:
        z = obj
    if not isinstance(z, torch.Tensor):
        raise TypeError(f"Expected embeddings to be a torch.Tensor, got {type(z)}")
    return z.detach().float().cpu()


def _metrics_from_logits(
    *,
    logits: torch.Tensor,  # [B,T,K]
    targets: torch.Tensor,  # [B,T]
    neighbors: torch.Tensor,  # [B,T,K]
    valid_step: torch.Tensor,  # [B,T] bool
    ks=(1, 5),
) -> Tuple[dict, int]:
    """Compute Recall@K and MRR on valid steps only."""
    b, t, kmax = logits.shape

    if valid_step.sum().item() == 0:
        return {f"recall@{k}": 0.0 for k in ks} | {"mrr": 0.0}, 0

    # TopK indices over candidate slots.
    max_k = max(ks)
    topv, topi = torch.topk(logits, k=min(max_k, kmax), dim=-1)  # [B,T,K']
    top_nb = torch.gather(neighbors, dim=-1, index=topi)  # [B,T,K']

    # Hits for each K.
    out = {}
    for kk in ks:
        eval_k = min(int(kk), top_nb.shape[-1])
        hit = (top_nb[..., :eval_k] == targets.unsqueeze(-1)).any(dim=-1)  # [B,T]
        hit = hit & valid_step
        out[f"recall@{kk}"] = float(hit.sum().item() / valid_step.sum().item())

    # MRR.
    # rank = first position where equals target.
    eq = top_nb == targets.unsqueeze(-1)  # [B,T,K']
    # set non-valid steps to all-false
    eq = eq & valid_step.unsqueeze(-1)

    # For each position, find first match.
    # If no match, reciprocal rank 0.
    idx = torch.arange(eq.shape[-1], device=eq.device).view(1, 1, -1)
    masked = torch.where(eq, idx, torch.full_like(idx, eq.shape[-1]))
    first = masked.min(dim=-1).values  # [B,T]
    rr = torch.where(first < eq.shape[-1], 1.0 / (first.float() + 1.0), torch.zeros_like(first, dtype=torch.float))
    rr = rr * valid_step.float()
    out["mrr"] = float(rr.sum().item() / valid_step.sum().item())

    return out, int(valid_step.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train graph-constrained autoregressive Transformer for trajectory next-hop prediction.")

    parser.add_argument("--city", type=str, default="xian", choices=["xian", "chengdu"])
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--traj-dir", type=str, default="", help="Custom trajectory directory (overrides data-root/city)")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--emb", type=str, default="checkpoints/xian_z64.pt", help="Path to exported embeddings (.pt) containing key 'z'")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--max-traj-len", type=int, default=64, help="Fixed trajectory length (pad or truncate)")
    parser.add_argument("--min-unique-len", type=int, default=5)
    parser.add_argument("--causal-window", type=int, default=0, help="Limit causal attention window size (0=unlimited, -1=no limit)")
    parser.add_argument("--max-trajs", type=int, default=0, help="0 = no limit")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible training")

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)


    # Optional departure-time conditioning.
    parser.add_argument("--use-depart-time", action="store_true", help="Condition next-hop prediction on traj meta depart_time (meta[3])")
    parser.add_argument("--num-time-bins", type=int, default=288, help="Number of discrete depart_time bins")
    parser.add_argument("--meta-time-index", type=int, default=3, help="Index in traj[0] meta for depart_time")

    # Optional start/end road embedding conditioning.
    parser.add_argument("--use-start-road-embedding", action="store_true", help="Condition next-hop prediction on start road embedding")
    parser.add_argument("--use-end-road-embedding", action="store_true", help="Condition next-hop prediction on end road embedding")

    # Optional total-duration conditioning.
    parser.add_argument("--use-total-duration", action="store_true", help="Condition next-hop prediction on full-trajectory total duration in seconds")

    parser.add_argument("--save", type=str, default="", help="Optional checkpoint path for downstream model")
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint every N epochs (0 = only save at end)")

    args = parser.parse_args()

    device = _pick_device(args.device)
    _set_random_seed(args.seed)

    # Load graph topology. We only need edge_index.
    data, _ = build_city_pyg_graph(
        city=args.city,
        data_root=args.data_root,
        min_unique_len=args.min_unique_len,
        include_node_heat=False,
        include_edge_heat=False,
        traj_dir=args.traj_dir if args.traj_dir else None,
    )
    num_nodes = int(data.num_nodes)

    # Load road embeddings z.
    z = _load_z(args.emb)
    if int(z.shape[0]) != num_nodes:
        raise ValueError(f"Embeddings N={int(z.shape[0])} != graph num_nodes={num_nodes}. Make sure you exported z for the same city.")

    # Neighbor table.
    neighbors, nb_mask = build_padded_neighbors(
        edge_index=data.edge_index,
        num_nodes=num_nodes,
        pad_value=0,
    )

    # Dataset - simplified: just load full trajectories and pad to fixed length
    traj_dir = args.traj_dir if args.traj_dir else str(Path(args.data_root) / args.city)

    from utils.graph_builder import (
        iter_city_trajectories,
        iter_city_trajectories_with_depart_time,
        iter_city_trajectories_with_total_duration,
        iter_city_trajectories_with_depart_time_and_total_duration,
    )

    # Load trajectories
    trajs = []
    depart_times = []
    total_durations = []

    if args.use_depart_time and args.use_total_duration:
        for road_ids, depart_time, total_duration in iter_city_trajectories_with_depart_time_and_total_duration(
            traj_dir, min_unique_len=args.min_unique_len, meta_time_index=args.meta_time_index
        ):
            if len(road_ids) > args.max_traj_len:
                road_ids = road_ids[:args.max_traj_len]
            trajs.append(road_ids)
            depart_times.append(depart_time)
            total_durations.append(total_duration)
            if args.max_trajs > 0 and len(trajs) >= args.max_trajs:
                break
    elif args.use_depart_time:
        for road_ids, depart_time in iter_city_trajectories_with_depart_time(
            traj_dir, min_unique_len=args.min_unique_len, meta_time_index=args.meta_time_index
        ):
            if len(road_ids) > args.max_traj_len:
                road_ids = road_ids[:args.max_traj_len]
            trajs.append(road_ids)
            depart_times.append(depart_time)
            if args.max_trajs > 0 and len(trajs) >= args.max_trajs:
                break
    elif args.use_total_duration:
        for road_ids, total_duration in iter_city_trajectories_with_total_duration(
            traj_dir, min_unique_len=args.min_unique_len
        ):
            if len(road_ids) > args.max_traj_len:
                road_ids = road_ids[:args.max_traj_len]
            trajs.append(road_ids)
            total_durations.append(total_duration)
            if args.max_trajs > 0 and len(trajs) >= args.max_trajs:
                break
    else:
        for road_ids in iter_city_trajectories(traj_dir, min_unique_len=args.min_unique_len):
            if len(road_ids) > args.max_traj_len:
                road_ids = road_ids[:args.max_traj_len]
            trajs.append(road_ids)
            if args.max_trajs > 0 and len(trajs) >= args.max_trajs:
                break

    # Create simple dataset
    class FixedLengthTrajDataset(torch.utils.data.Dataset):
        def __init__(self, trajs, depart_times, total_durations, use_start, use_end, use_depart, use_duration):
            self.trajs = trajs
            self.depart_times = depart_times
            self.total_durations = total_durations
            self.use_start = use_start
            self.use_end = use_end
            self.use_depart = use_depart
            self.use_duration = use_duration

        def __len__(self):
            return len(self.trajs)

        def __getitem__(self, idx):
            road_ids = torch.tensor(self.trajs[idx], dtype=torch.long)
            out = [road_ids]

            if self.use_start:
                out.append(torch.tensor(self.trajs[idx][0], dtype=torch.long))
            if self.use_end:
                out.append(torch.tensor(self.trajs[idx][-1], dtype=torch.long))
            if self.use_depart:
                out.append(torch.tensor(self.depart_times[idx], dtype=torch.long))
            if self.use_duration:
                out.append(torch.tensor(self.total_durations[idx], dtype=torch.float32))

            return tuple(out) if len(out) > 1 else out[0]

    ds = FixedLengthTrajDataset(
        trajs, depart_times, total_durations,
        use_start=args.use_start_road_embedding,
        use_end=args.use_end_road_embedding,
        use_depart=args.use_depart_time,
        use_duration=args.use_total_duration,
    )

    def _collate(batch):
        if args.use_end_road_embedding and args.use_start_road_embedding and args.use_depart_time and args.use_total_duration:
            return pad_collate_1d_with_start_and_destination_and_depart_time_and_total_duration(batch, pad_value=-1)
        if args.use_end_road_embedding and args.use_start_road_embedding and args.use_depart_time:
            return pad_collate_1d_with_start_and_destination_and_depart_time(batch, pad_value=-1)
        if args.use_end_road_embedding and args.use_start_road_embedding and args.use_total_duration:
            return pad_collate_1d_with_start_and_destination_and_total_duration(batch, pad_value=-1)
        if args.use_end_road_embedding and args.use_start_road_embedding:
            return pad_collate_1d_with_start_and_destination(batch, pad_value=-1)
        if args.use_end_road_embedding and args.use_depart_time and args.use_total_duration:
            return pad_collate_1d_with_destination_and_depart_time_and_total_duration(batch, pad_value=-1)
        if args.use_end_road_embedding and args.use_depart_time:
            return pad_collate_1d_with_destination_and_depart_time(batch, pad_value=-1)
        if args.use_end_road_embedding and args.use_total_duration:
            return pad_collate_1d_with_destination_and_total_duration(batch, pad_value=-1)
        if args.use_end_road_embedding:
            return pad_collate_1d_with_destination(batch, pad_value=-1)
        if args.use_depart_time and args.use_total_duration:
            return pad_collate_1d_with_depart_time_and_total_duration(batch, pad_value=-1)
        if args.use_depart_time:
            return pad_collate_1d_with_depart_time(batch, pad_value=-1)
        if args.use_total_duration:
            return pad_collate_1d_with_total_duration(batch, pad_value=-1)
        x, m = pad_collate_1d(batch, pad_value=-1)
        return x, m

    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(int(args.seed))
    worker_init_fn = _make_worker_init_fn(args.seed) if int(args.num_workers) > 0 else None

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
        generator=data_loader_generator,
        worker_init_fn=worker_init_fn,
    )

    # Model.
    cfg = TrajARTransformerConfig(
        emb_dim=int(z.shape[1]),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        dropout=float(args.dropout),
        max_len=int(args.max_traj_len),
        use_start_road_embedding=bool(args.use_start_road_embedding),
        use_end_road_embedding=bool(args.use_end_road_embedding),
        use_depart_time=bool(args.use_depart_time),
        num_time_bins=int(args.num_time_bins),
        use_total_duration=bool(args.use_total_duration),
        causal_window=int(args.causal_window),
    )

    model = GraphConstrainedARTransformer(cfg, pad_id=-1).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    z_d = z.to(device)
    neighbors_d = neighbors.to(device)
    nb_mask_d = nb_mask.to(device)

    print("== Config ==")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")
    print("device:", device)
    print("num_nodes:", num_nodes, "Kmax:", int(neighbors.shape[1]))
    print("model cfg:", asdict(cfg))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_steps = 0
        total_valid_steps = 0
        sum_recall1 = 0.0
        sum_recall5 = 0.0
        sum_mrr = 0.0

        for batch in dl:
            start_road = None
            end_road = None
            depart_time = None
            total_duration = None

            if args.use_end_road_embedding and args.use_start_road_embedding and args.use_depart_time and args.use_total_duration:
                road_ids, attn_mask, start_road, end_road, depart_time, total_duration = batch
            elif args.use_end_road_embedding and args.use_start_road_embedding and args.use_depart_time:
                road_ids, attn_mask, start_road, end_road, depart_time = batch
            elif args.use_end_road_embedding and args.use_start_road_embedding and args.use_total_duration:
                road_ids, attn_mask, start_road, end_road, total_duration = batch
            elif args.use_end_road_embedding and args.use_start_road_embedding:
                road_ids, attn_mask, start_road, end_road = batch
            elif args.use_end_road_embedding and args.use_depart_time and args.use_total_duration:
                road_ids, attn_mask, end_road, depart_time, total_duration = batch
            elif args.use_end_road_embedding and args.use_depart_time:
                road_ids, attn_mask, end_road, depart_time = batch
            elif args.use_end_road_embedding and args.use_total_duration:
                road_ids, attn_mask, end_road, total_duration = batch
            elif args.use_end_road_embedding:
                road_ids, attn_mask, end_road = batch
            elif args.use_depart_time and args.use_total_duration:
                road_ids, attn_mask, depart_time, total_duration = batch
            elif args.use_depart_time:
                road_ids, attn_mask, depart_time = batch
            elif args.use_total_duration:
                road_ids, attn_mask, total_duration = batch
            else:
                road_ids, attn_mask = batch

            road_ids = road_ids.to(device)
            attn_mask = attn_mask.to(device)
            if start_road is not None:
                start_road = start_road.to(device)
            if end_road is not None:
                end_road = end_road.to(device)
            if depart_time is not None:
                depart_time = depart_time.to(device)
            if total_duration is not None:
                total_duration = total_duration.to(device)

            # Need at least 2 tokens to have a target.
            if road_ids.shape[1] < 2:
                continue

            # Inputs/targets.
            x = road_ids[:, :-1]
            y = road_ids[:, 1:]
            x_mask = attn_mask[:, :-1]
            y_mask = attn_mask[:, 1:]

            logits = model(
                road_ids=x,
                z=z_d,
                neighbors=neighbors_d,
                neighbor_mask=nb_mask_d,
                start_road=start_road,
                end_road=end_road,
                attn_mask=x_mask,
                depart_time=depart_time,
                total_duration=total_duration,
            )  # [B,T-1,K]

            # Valid steps: both tokens valid and target must be inside neighbor set of current road.
            # gather neighbor ids for current roads
            cur = x.clamp(min=0)
            nb = neighbors_d[cur]  # [B,T-1,K]
            nbm = nb_mask_d[cur]

            # Is target in neighbor set?
            y_in = (nb == y.unsqueeze(-1)) & nbm
            y_in = y_in.any(dim=-1)  # [B,T-1]

            valid_step = x_mask & y_mask & y_in

            if valid_step.sum().item() == 0:
                continue

            # Convert target road id to candidate slot index.
            # Each valid step should have exactly one matching candidate.
            match = (nb == y.unsqueeze(-1)) & nbm  # [B,T-1,K]
            slot = match.float().argmax(dim=-1)  # [B,T-1]

            # Compute CE only on valid positions to avoid NaNs from fully-masked steps.
            logits_v = logits[valid_step]  # [M,K]
            slot_v = slot[valid_step]  # [M]
            loss = F.cross_entropy(logits_v, slot_v, reduction="mean")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            total_valid_steps += int(valid_step.sum().item())
            total_steps += int(valid_step.numel())

            with torch.no_grad():
                m, n_valid = _metrics_from_logits(
                    logits=logits,
                    targets=y,
                    neighbors=nb,
                    valid_step=valid_step,
                    ks=(1, 5),
                )
                if n_valid > 0:
                    sum_recall1 += m["recall@1"] * n_valid
                    sum_recall5 += m["recall@5"] * n_valid
                    sum_mrr += m["mrr"] * n_valid

        if total_valid_steps == 0:
            print(f"epoch {epoch:03d}/{args.epochs} | no valid steps (check neighbor table / trajectories)")
            continue

        avg_loss = total_loss / max(1, len(dl))
        valid_ratio = total_valid_steps / max(1, total_steps)
        r1 = sum_recall1 / total_valid_steps
        r5 = sum_recall5 / total_valid_steps
        mrr = sum_mrr / total_valid_steps

        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"loss={avg_loss:.6f} | valid_step_ratio={valid_ratio:.3f} | "
            f"R@1={r1:.4f} R@5={r5:.4f} MRR={mrr:.4f}"
        )

        # Auto-save every N epochs (non-overwriting: embed epoch number in filename)
        if args.save and args.save_interval > 0 and epoch % args.save_interval == 0:
            save_path = Path(args.save)
            epoch_path = save_path.parent / f"{save_path.stem}_epoch{epoch:04d}{save_path.suffix}"
            ckpt = {
                "model_state": model.state_dict(),
                "cfg": asdict(cfg),
                "args": vars(args),
                "epoch": epoch,
            }
            epoch_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, epoch_path)
            print(f"  -> saved checkpoint to: {epoch_path}")

    if args.save:
        save_path = Path(args.save)
        epoch_path = save_path.parent / f"{save_path.stem}_epoch{args.epochs:04d}{save_path.suffix}"
        ckpt = {
            "model_state": model.state_dict(),
            "cfg": asdict(cfg),
            "args": vars(args),
            "epoch": args.epochs,
        }
        epoch_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, epoch_path)
        print("saved downstream checkpoint to:", epoch_path)


if __name__ == "__main__":
    main()
