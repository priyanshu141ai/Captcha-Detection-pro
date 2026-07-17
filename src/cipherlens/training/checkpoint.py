"""Candidate checkpoint metadata, atomic writes, warm starts, and resume state."""

from __future__ import annotations

import json
import platform
import random
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cipherlens.data import NORMALIZATION_MEAN, NORMALIZATION_STD, PREPROCESSING_VERSION
from cipherlens.models import (
    MODEL_ARCHITECTURE_NAME,
    MODEL_VERSION,
    CaptchaCodec,
    CaptchaCRNN,
    ModelConfig,
)
from cipherlens.training.data import TrainingSplit
from cipherlens.training.engine import EarlyStopping

CHECKPOINT_VERSION = 2


@dataclass(frozen=True)
class ResumeState:
    epoch: int
    history: list[dict[str, float | int]]
    best_model_state: dict[str, Tensor]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


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


def build_run_metadata(
    *,
    project_root: Path,
    model_config: ModelConfig,
    split: TrainingSplit,
    run_config: dict[str, object],
    dataset_sources: list[dict[str, str]],
    device: torch.device,
) -> dict[str, object]:
    return {
        "run_id": str(uuid.uuid4()),
        "run_started_at": _utc_now(),
        "git_commit": _git_commit(project_root),
        "architecture": {
            "name": MODEL_ARCHITECTURE_NAME,
            "version": MODEL_VERSION,
            "config": asdict(model_config),
        },
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "input_width": model_config.image_width,
            "input_height": model_config.image_height,
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
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "device": str(device),
        },
    }


def ensure_safe_artifact_paths(
    candidate_path: Path, resume_path: Path, approved_checkpoint: Path
) -> None:
    candidate = candidate_path.resolve()
    resume = resume_path.resolve()
    approved = approved_checkpoint.resolve()
    if candidate == approved or resume == approved:
        raise ValueError("Training artifacts must not overwrite the approved checkpoint.")
    if candidate == resume:
        raise ValueError("Candidate and resume checkpoint paths must be different.")


def _cpu_state(state: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in state.items()}


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_candidate_checkpoint(
    path: Path,
    *,
    model_state: dict[str, Tensor],
    codec: CaptchaCodec,
    model_config: ModelConfig,
    metrics: dict[str, float],
    epoch: int,
    metadata: dict[str, object],
) -> None:
    candidate_metadata = {
        **metadata,
        "selected_epoch": epoch,
        "validation_metrics": dict(metrics),
    }
    _atomic_torch_save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "checkpoint_kind": "candidate",
            "model_state": _cpu_state(model_state),
            "charset": codec.charset,
            "model_config": asdict(model_config),
            "metrics": dict(metrics),
            "epoch": epoch,
            "metadata": candidate_metadata,
            "created_at": _utc_now(),
        },
        path,
    )


def _rng_state(generator: torch.Generator) -> dict[str, object]:
    numpy_state = cast(tuple[Any, ...], np.random.get_state())
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generator": generator.get_state(),
    }


def save_resume_checkpoint(
    path: Path,
    *,
    model: CaptchaCRNN,
    best_model_state: dict[str, Tensor],
    codec: CaptchaCodec,
    model_config: ModelConfig,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    early_stopping: EarlyStopping,
    epoch: int,
    history: list[dict[str, float | int]],
    metadata: dict[str, object],
    generator: torch.Generator,
) -> None:
    _atomic_torch_save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "checkpoint_kind": "training_resume",
            "model_state": _cpu_state(model.state_dict()),
            "best_model_state": _cpu_state(best_model_state),
            "charset": codec.charset,
            "model_config": asdict(model_config),
            "metrics": dict(early_stopping.best_metrics),
            "epoch": epoch,
            "metadata": metadata,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "early_stopping": early_stopping.state_dict(),
            "history": history,
            "rng_state": _rng_state(generator),
            "created_at": _utc_now(),
        },
        path,
    )


def _restore_rng(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(cast(tuple[Any, ...], state["python"]))
    numpy_state = cast(dict[str, Any], state["numpy"])
    np.random.set_state(
        (
            str(numpy_state["name"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(cast(Tensor, state["torch"]))
    cuda_state = cast(list[Tensor], state.get("cuda", []))
    if torch.cuda.is_available() and cuda_state:
        torch.cuda.set_rng_state_all(cuda_state)
    generator.set_state(cast(Tensor, state["loader_generator"]))


def _optimizer_to_device(optimizer: Optimizer, device: torch.device) -> None:
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if isinstance(value, Tensor):
                optimizer_state[key] = value.to(device)


def load_resume_checkpoint(
    path: Path,
    *,
    model: CaptchaCRNN,
    codec: CaptchaCodec,
    model_config: ModelConfig,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    early_stopping: EarlyStopping,
    expected_dataset_version: str,
    generator: torch.Generator,
    device: torch.device,
) -> ResumeState:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION
        or checkpoint.get("checkpoint_kind") != "training_resume"
    ):
        raise ValueError("Resume checkpoint schema is invalid.")
    if checkpoint.get("charset") != codec.charset or checkpoint.get("model_config") != asdict(
        model_config
    ):
        raise ValueError("Resume checkpoint model or vocabulary is incompatible.")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Resume checkpoint metadata is missing.")
    dataset = metadata.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("version") != expected_dataset_version:
        raise ValueError("Resume checkpoint dataset version is incompatible.")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    _optimizer_to_device(optimizer, device)
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    early_stopping.load_state_dict(checkpoint["early_stopping"])
    _restore_rng(checkpoint["rng_state"], generator)
    history = checkpoint.get("history")
    best_state = checkpoint.get("best_model_state")
    if not isinstance(history, list) or not isinstance(best_state, dict) or not best_state:
        raise ValueError("Resume checkpoint history or best model state is invalid.")
    return ResumeState(
        int(checkpoint["epoch"]),
        cast(list[dict[str, float | int]], history),
        cast(dict[str, Tensor], best_state),
    )


def warm_start_model(
    model: CaptchaCRNN,
    codec: CaptchaCodec,
    checkpoint_path: Path,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Initialization checkpoint schema is invalid.")
    checkpoint_config = ModelConfig(**checkpoint.get("model_config", {}))
    if checkpoint_config != model.config:
        raise ValueError(
            "The initialization checkpoint model configuration does not match the current model."
        )
    checkpoint_codec = CaptchaCodec(checkpoint["charset"])
    checkpoint_state = cast(dict[str, Tensor], checkpoint["model_state"])
    model_state = model.state_dict()
    for name, tensor in checkpoint_state.items():
        if (
            not name.startswith("classifier.")
            and name in model_state
            and model_state[name].shape == tensor.shape
        ):
            model_state[name] = tensor
    shared_characters = sorted(set(codec.charset) & set(checkpoint_codec.charset))
    for character in shared_characters:
        old_index = checkpoint_codec.char_to_index[character]
        new_index = codec.char_to_index[character]
        model_state["classifier.weight"][new_index] = checkpoint_state["classifier.weight"][
            old_index
        ]
        model_state["classifier.bias"][new_index] = checkpoint_state["classifier.bias"][old_index]
    model.load_state_dict(model_state)
    return len(shared_characters)


__all__ = [
    "CHECKPOINT_VERSION",
    "ResumeState",
    "build_run_metadata",
    "ensure_safe_artifact_paths",
    "load_resume_checkpoint",
    "save_candidate_checkpoint",
    "save_resume_checkpoint",
    "warm_start_model",
    "write_history",
]
