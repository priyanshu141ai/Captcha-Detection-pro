"""Evaluation summary for optional experimental CTC checkpoints."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor

from cipherlens.data import prepare_image
from cipherlens.evaluation.metrics import expected_calibration_error, reliability_bins
from cipherlens.evaluation.runner import benchmark_latency, load_manifest_selection
from cipherlens.models import levenshtein_distance
from cipherlens.models.ctc import (
    CTC_ARCHITECTURE_NAME,
    CTC_BLANK_INDEX,
    CaptchaCTCCRNN,
    CTCCodec,
    CTCModelConfig,
)
from cipherlens.training.ctc import ctc_loss


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_ctc_checkpoint(
    path: Path, device: torch.device
) -> tuple[CaptchaCTCCRNN, CTCCodec, dict[str, Any], dict[str, Any]]:
    checkpoint: Any = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("CTC checkpoint must contain a dictionary.")
    required = {"model_state", "charset", "model_config", "blank_index"}
    if not required <= set(checkpoint):
        raise ValueError("CTC checkpoint is missing required fields.")
    if checkpoint.get("architecture_name") != CTC_ARCHITECTURE_NAME:
        raise ValueError("Checkpoint is not a CipherLens CTC model.")
    if checkpoint.get("blank_index") != CTC_BLANK_INDEX:
        raise ValueError("CTC checkpoint blank index is incompatible.")
    try:
        config = CTCModelConfig(**checkpoint["model_config"])
        codec = CTCCodec(checkpoint["charset"])
        model = CaptchaCTCCRNN(codec.num_classes, config).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("CTC checkpoint is incompatible with this application version.") from error
    model.eval()
    metadata = checkpoint.get("metadata")
    return model, codec, checkpoint, metadata if isinstance(metadata, dict) else {}


def evaluate_ctc_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    *,
    project_root: Path,
    split: str = "validation",
    dataset_report_path: Path | None = None,
    batch_size: int = 32,
    device: str = "cpu",
    torch_threads: int = 2,
    ece_bins: int = 10,
    latency_warmup: int = 5,
    latency_runs: int = 50,
) -> dict[str, Any]:
    if batch_size < 1 or torch_threads < 1:
        raise ValueError("CTC evaluation batch size and torch threads must be positive.")
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable.")
    torch.set_num_threads(torch_threads)
    selection = load_manifest_selection(
        manifest_path,
        project_root=project_root,
        split=split,
        dataset_report_path=dataset_report_path,
    )
    model, codec, checkpoint, metadata = _load_ctc_checkpoint(checkpoint_path, selected_device)
    charset = set(codec.charset)
    if any(not set(sample.label) <= charset for sample in selection.samples):
        raise ValueError("Evaluation labels are incompatible with the CTC vocabulary.")
    loss_fn = torch.nn.CTCLoss(blank=CTC_BLANK_INDEX, zero_infinity=True)
    predictions: list[tuple[str, str, float]] = []
    total_loss = 0.0
    first_tensor: Tensor | None = None
    for offset in range(0, len(selection.samples), batch_size):
        samples = selection.samples[offset : offset + batch_size]
        tensors = []
        for sample in samples:
            with Image.open(sample.path) as image:
                tensors.append(prepare_image(image, model.config, augment=False))
        images = torch.stack(tensors).to(selected_device)
        if first_tensor is None:
            first_tensor = images[:1]
        targets = torch.stack([codec.encode(sample.label) for sample in samples]).to(
            selected_device
        )
        with torch.inference_mode():
            logits = model(images)
            batch_loss = ctc_loss(logits, targets, loss_fn)
        total_loss += float(batch_loss) * len(samples)
        predictions.extend(
            (sample.label, prediction, confidence)
            for sample, (prediction, confidence) in zip(
                samples, codec.greedy_decode(logits), strict=True
            )
        )
    if first_tensor is None:
        raise ValueError("CTC evaluation has no selected samples.")
    sample_count = len(predictions)
    character_count = sum(len(target) for target, _prediction, _confidence in predictions)
    distances = [
        levenshtein_distance(target, prediction) for target, prediction, _confidence in predictions
    ]
    exact = [target == prediction for target, prediction, _confidence in predictions]
    confidences = [confidence for _target, _prediction, confidence in predictions]
    bins = reliability_bins(confidences, exact, ece_bins)
    normalized_distances = [
        distance / max(len(target), len(prediction), 1)
        for distance, (target, prediction, _confidence) in zip(distances, predictions, strict=True)
    ]
    character_error_rate = sum(distances) / max(character_count, 1)
    latency = benchmark_latency(
        model,
        first_tensor,
        device=selected_device,
        warmup_runs=latency_warmup,
        measured_runs=latency_runs,
    )
    dataset_metadata = metadata.get("dataset")
    evidence_status = (
        "versioned_checkpoint_and_manifest_match"
        if isinstance(dataset_metadata, dict)
        and dataset_metadata.get("version") == selection.dataset_version
        and dataset_metadata.get("split_version") == selection.split_version
        else "provisional_checkpoint_training_split_unverified"
    )
    model_version = checkpoint.get("model_version", "2.0-experimental")
    return {
        "schema_version": "1.0",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "model": {
            "architecture_name": CTC_ARCHITECTURE_NAME,
            "model_version": str(model_version),
            "checkpoint_version": str(checkpoint.get("checkpoint_version", "unknown")),
            "checkpoint_path": _project_path(checkpoint_path, project_root),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "cpu_model_tensor_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in (*model.parameters(), *model.buffers())
            ),
        },
        "evidence": {
            "dataset_version": selection.dataset_version,
            "split_version": selection.split_version,
            "manifest_sha256": selection.manifest_sha256,
            "split": split,
            "status": evidence_status,
            "external_test_status": selection.external_test_status,
        },
        "metrics": {
            "sample_count": sample_count,
            "character_count": character_count,
            "character_accuracy": max(0.0, 1.0 - character_error_rate),
            "exact_accuracy": sum(exact) / sample_count,
            "character_error_rate": character_error_rate,
            "normalized_edit_distance": statistics.fmean(normalized_distances),
            "mean_confidence": statistics.fmean(confidences),
            "median_confidence": statistics.median(confidences),
            "sequence_ece": expected_calibration_error(bins),
            "loss": total_loss / sample_count,
        },
        "calibration": {
            "temperature": 1.0,
            "note": "CTC temperature scaling is not configured.",
        },
        "latency": asdict(latency),
    }


def write_ctc_evaluation_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


__all__ = ["evaluate_ctc_checkpoint", "write_ctc_evaluation_summary"]
