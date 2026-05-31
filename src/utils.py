"""Shared utilities: reproducibility helpers."""

import random

import numpy as np
import torch

from src.config import SEED


def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs.

    Note: full determinism on GPU also requires deterministic cuDNN
    algorithms, which can slow training. We enable the cheap, safe
    seeding here and leave strict determinism opt-in.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
