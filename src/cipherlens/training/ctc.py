"""Experimental Model V2 CTC optimization and candidate artifacts."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from cipherlens.data import NORMALIZATION_MEAN, NORMALIZATION_STD, PREPROCESSING_VERSION
from cipherlens.models import levenshtein_distance
from cipherlens.models.ctc import (
    CTC_ARCHITECTURE_NAME,
    CTC_BLANK_INDEX,
    CTC_MODEL_VERSION,
    CaptchaCTCCRNN,
    CTCCodec,
    CTCModelConfig,
)
from cipherlens.training.data import TrainingSplit


def _lengths(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    batch_size = targets.shape[0]
    return (
        torch.full((batch_size,), logits.shape[0], dtype=torch.long),
        torch.full((batch_size,), targets.shape[1], dtype=torch.long),
    )


def ctc_loss(logits: Tensor, targets: Tensor, loss_fn: nn.CTCLoss) -> Tensor:
    input_lengths, target_lengths = _lengths(logits, targets)
    value: Tensor = loss_fn(logits.log_softmax(dim=2), targets, input_lengths, target_lengths)
    return value


def build_ctc_optimization(
    model: CaptchaCTCCRNN,
    *,
    learning_rate: float,
    weight_decay: float,
    scheduler_factor: float,
    scheduler_patience: int,
) -> tuple[nn.CTCLoss, Optimizer, ReduceLROnPlateau]:
    loss = nn.CTCLoss(blank=CTC_BLANK_INDEX, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
    )
    return loss, optimizer, scheduler


def train_ctc_epoch(
    model: CaptchaCTCCRNN,
    loader: DataLoader[Any],
    loss_fn: nn.CTCLoss,
    optimizer: Optimizer,
    device: torch.device,
    *,
    gradient_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0
    for images, targets, _labels in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = ctc_loss(model(images), targets, loss_fn)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        total_loss += float(loss.detach()) * images.shape[0]
        sample_count += images.shape[0]
    return total_loss / max(sample_count, 1)


def evaluate_ctc(
    model: CaptchaCTCCRNN,
    loader: DataLoader[Any],
    codec: CTCCodec,
    loss_fn: nn.CTCLoss,
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
            total_loss += float(ctc_loss(logits, targets, loss_fn)) * images.shape[0]
            for (prediction, _confidence), label in zip(
                codec.greedy_decode(logits), labels, strict=True
            ):
                exact_matches += int(prediction == label)
                edit_distance += levenshtein_distance(prediction, label)
                character_count += len(label)
                sample_count += 1
    character_error_rate = edit_distance / max(character_count, 1)
    return {
        "loss": total_loss / max(sample_count, 1),
        "exact_accuracy": exact_matches / max(sample_count, 1),
        "character_accuracy": max(0.0, 1.0 - character_error_rate),
        "character_error_rate": character_error_rate,
    }


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def build_ctc_metadata(
    *,
    project_root: Path,
    config: CTCModelConfig,
    split: TrainingSplit,
    run_config: dict[str, object],
    dataset_sources: list[dict[str, str]],
    device: torch.device,
) -> dict[str, object]:
    return {
        "architecture": {
            "name": CTC_ARCHITECTURE_NAME,
            "version": CTC_MODEL_VERSION,
            "status": "experimental",
            "config": asdict(config),
        },
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "input_width": config.image_width,
            "input_height": config.image_height,
            "color_mode": "RGB",
            "normalization_mean": [NORMALIZATION_MEAN] * 3,
            "normalization_std": [NORMALIZATION_STD] * 3,
        },
        "dataset": {
            "version": split.dataset_version,
            "split_version": split.split_version,
            "selection_hash": split.selection_hash,
            "manifest_path": split.manifest_path,
            "train_samples": len(split.training),
            "validation_samples": len(split.validation),
            "sources": dataset_sources,
            "external_test_used": False,
        },
        "training_config": run_config,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": str(device),
        },
        "git_commit": _git_commit(project_root),
        "created_at": datetime.now(UTC).isoformat(),
    }


def save_ctc_candidate(
    path: Path,
    *,
    model: CaptchaCTCCRNN,
    codec: CTCCodec,
    config: CTCModelConfig,
    metrics: dict[str, float],
    epoch: int,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(
        {
            "checkpoint_version": 2,
            "checkpoint_kind": "experimental_ctc_candidate",
            "architecture_name": CTC_ARCHITECTURE_NAME,
            "model_version": CTC_MODEL_VERSION,
            "model_state": {
                name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
            },
            "charset": codec.charset,
            "blank_index": codec.blank_index,
            "model_config": asdict(config),
            "metrics": dict(metrics),
            "epoch": epoch,
            "metadata": {
                **metadata,
                "selected_epoch": epoch,
                "validation_metrics": dict(metrics),
            },
            "created_at": datetime.now(UTC).isoformat(),
        },
        temporary,
    )
    temporary.replace(path)


__all__ = [
    "build_ctc_metadata",
    "build_ctc_optimization",
    "ctc_loss",
    "evaluate_ctc",
    "save_ctc_candidate",
    "train_ctc_epoch",
]
