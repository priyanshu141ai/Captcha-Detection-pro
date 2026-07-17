"""Reproducible random-state helpers for training and tests."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ReproducibilityState:
    seed: int
    deterministic_algorithms: bool
    cuda_seeded: bool


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1.")
    return seed


def seed_everything(seed: int, *, deterministic: bool = False) -> ReproducibilityState:
    """Seed Python, NumPy, and PyTorch without changing default model behavior."""
    validated_seed = _validate_seed(seed)
    random.seed(validated_seed)
    np.random.seed(validated_seed)
    torch.manual_seed(validated_seed)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed_all(validated_seed)
    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        if deterministic:
            torch.backends.cudnn.benchmark = False
    return ReproducibilityState(
        seed=validated_seed,
        deterministic_algorithms=deterministic,
        cuda_seeded=cuda_available,
    )


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy inside a PyTorch DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    """Create an independently seeded PyTorch generator."""
    generator = torch.Generator()
    generator.manual_seed(_validate_seed(seed))
    return generator
