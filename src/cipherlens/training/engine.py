"""Model V1 optimization, evaluation, and early stopping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from cipherlens.models import CaptchaCodec, CaptchaCRNN, levenshtein_distance


@dataclass(frozen=True)
class OptimizationComponents:
    loss: nn.CrossEntropyLoss
    optimizer: Optimizer
    scheduler: ReduceLROnPlateau


@dataclass
class EarlyStopping:
    patience: int
    best_score: tuple[float, float, float] = (-1.0, -1.0, float("-inf"))
    best_epoch: int = 0
    stale_epochs: int = 0
    best_metrics: dict[str, float] = field(default_factory=dict)

    def update(self, epoch: int, metrics: dict[str, float]) -> bool:
        score = (
            metrics["character_accuracy"],
            metrics["exact_accuracy"],
            -metrics["loss"],
        )
        if score <= self.best_score:
            self.stale_epochs += 1
            return False
        self.best_score = score
        self.best_epoch = epoch
        self.stale_epochs = 0
        self.best_metrics = dict(metrics)
        return True

    @property
    def should_stop(self) -> bool:
        return self.stale_epochs >= self.patience

    def state_dict(self) -> dict[str, object]:
        return {
            "patience": self.patience,
            "best_score": list(self.best_score),
            "best_epoch": self.best_epoch,
            "stale_epochs": self.stale_epochs,
            "best_metrics": self.best_metrics,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        patience = int(state.get("patience", 0))
        if patience < 1:
            raise ValueError("Resume checkpoint early-stopping patience is invalid.")
        score = state.get("best_score")
        if not isinstance(score, list) or len(score) != 3:
            raise ValueError("Resume checkpoint early-stopping score is invalid.")
        values = tuple(float(value) for value in score)
        self.patience = patience
        self.best_score = (values[0], values[1], values[2])
        self.best_epoch = int(state.get("best_epoch", 0))
        self.stale_epochs = int(state.get("stale_epochs", 0))
        metrics = state.get("best_metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError("Resume checkpoint best metrics are invalid.")
        self.best_metrics = {str(key): float(value) for key, value in metrics.items()}


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def build_optimization(
    model: CaptchaCRNN,
    class_weights: torch.Tensor,
    *,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    scheduler_factor: float,
    scheduler_patience: int,
) -> OptimizationComponents:
    loss = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
    )
    return OptimizationComponents(loss, optimizer, scheduler)


def train_one_epoch(
    model: CaptchaCRNN,
    loader: DataLoader[Any],
    loss_fn: nn.CrossEntropyLoss,
    optimizer: Optimizer,
    device: torch.device,
    *,
    gradient_clip_norm: float,
) -> float:
    model.train()
    running_loss = 0.0
    seen = 0
    for images, targets, _labels in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits.permute(1, 2, 0), targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()
        running_loss += float(loss.detach()) * images.shape[0]
        seen += images.shape[0]
    return running_loss / max(seen, 1)


def evaluate(
    model: CaptchaCRNN,
    loader: DataLoader[Any],
    codec: CaptchaCodec,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    exact_matches = 0
    edit_distance = 0
    character_count = 0
    sample_count = 0
    with torch.inference_mode():
        for images, targets, labels in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = loss_fn(logits.permute(1, 2, 0), targets)
            total_loss += float(loss) * images.shape[0]
            for (prediction, _confidence), label in zip(
                codec.greedy_decode(logits), labels, strict=True
            ):
                exact_matches += int(prediction == label)
                edit_distance += levenshtein_distance(prediction, label)
                character_count += len(label)
                sample_count += 1
    return {
        "loss": total_loss / max(sample_count, 1),
        "exact_accuracy": exact_matches / max(sample_count, 1),
        "character_accuracy": max(0.0, 1.0 - edit_distance / max(character_count, 1)),
    }


__all__ = [
    "EarlyStopping",
    "OptimizationComponents",
    "build_optimization",
    "choose_device",
    "evaluate",
    "train_one_epoch",
]
