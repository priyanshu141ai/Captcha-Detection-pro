"""Validation-only scalar temperature scaling for position-wise logits."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def negative_log_likelihood(logits: Tensor, targets: Tensor, temperature: float = 1.0) -> float:
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("Logits and targets must have aligned [N, C] and [N] shapes.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be a positive finite value.")
    return float(nn.functional.cross_entropy(logits / temperature, targets))


def fit_temperature(logits: Tensor, targets: Tensor, *, max_iterations: int = 50) -> float:
    """Fit one positive temperature by minimizing validation-set cross entropy."""
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("Logits and targets must have aligned [N, C] and [N] shapes.")
    if logits.shape[0] < 1 or logits.shape[1] < 2 or max_iterations < 1:
        raise ValueError("Temperature fitting requires samples, classes, and iterations.")
    values = logits.detach().to(dtype=torch.float64, device="cpu")
    labels = targets.detach().to(dtype=torch.long, device="cpu")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=max_iterations,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = nn.functional.cross_entropy(values / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0))
    if not math.isfinite(temperature):
        raise RuntimeError("Temperature optimization produced a non-finite result.")
    return temperature


__all__ = ["fit_temperature", "negative_log_likelihood"]
