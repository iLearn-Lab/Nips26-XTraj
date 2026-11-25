"""Small helpers shared by training steps."""

import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_loss(loss):
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite training loss")
