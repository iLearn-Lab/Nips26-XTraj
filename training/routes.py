"""Candidate-road cross-entropy training with frozen GAT embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.candidate_road_transformer import CandidateRoadTransformer
from utils.training import finite_loss


class FixedEmbeddingCandidateModel(nn.Module):
    """Keep the full road embedding replicated under DataParallel.

    Passing ``z`` as a normal DataParallel argument would split the road
    dimension across devices.  Registering it as a buffer makes each replica
    receive the complete embedding table while batch tensors are split.
    """

    def __init__(self, base: CandidateRoadTransformer, z: torch.Tensor) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("z", z.detach().float(), persistent=False)

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.base(**kwargs, z=self.z)


def train_epoch(model, loader, optimizer, *, device=None):
    """Train on prepared candidate tensors without logging or saving weights."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.train()
    for batch in loader:
        width = int(batch["history_mask"].sum(dim=1).max())
        target = batch["target"].to(device)
        logits = model(
            history_road_ids=batch["history"][:, :width].to(device),
            history_mask=batch["history_mask"][:, :width].to(device),
            candidate_road_ids=batch["candidates"].to(device),
            candidate_geo=batch["geo"].to(device),
            candidate_mask=batch["candidate_mask"].to(device),
            period_index=batch["period"].to(device),
        )
        loss = F.cross_entropy(logits, target)
        finite_loss(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
