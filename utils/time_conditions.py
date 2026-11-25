"""Fixed three-second time conditions shared by training and generation."""

import torch

SAMPLE_SECONDS = 3


def fixed_time_conditions(n_points: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    steps = n_points - 1
    positions = torch.arange(width, device=steps.device)
    remaining = (1.0 - positions[None, :] / steps[:, None]).clamp(0, 1)
    return SAMPLE_SECONDS * steps.float(), remaining
