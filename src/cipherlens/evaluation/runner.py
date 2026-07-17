"""Manifest-bound checkpoint evaluation and latency benchmarking."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor

from cipherlens.data import prepare_image
from cipherlens.evaluation.calibration import fit_temperature, negative_log_likelihood
from cipherlens.evaluation.metrics import EvaluationMetrics, EvaluationRecord, calculate_metrics
from cipherlens.inference import CaptchaRecognizer
from cipherlens.models import MODEL_ARCHITECTURE_NAME, MODEL_VERSION


class EvaluationPendingError(RuntimeError):
    """Raised when a requested evidence split has no configured samples."""


@dataclass(frozen=True)
class EvaluationSample:
    path: Path
    display_path: str
    source: str
    label: str


@dataclass(frozen=True)
class ManifestSelection:
    samples: tuple[EvaluationSample, ...]
    split: str
    dataset_version: str
    split_version: str
    manifest_sha256: str
    external_test_status: str


@dataclass(frozen=True)
class LatencyMetrics:
    scope: str
    device: str
    warmup_runs: int
    measured_runs: int
    mean_ms: float
    median_ms: float
    p95_ms: float


@dataclass(frozen=True)
class EvaluationResult:
    records: tuple[EvaluationRecord, ...]
    metrics: EvaluationMetrics
    calibrated_metrics: EvaluationMetrics | None
    temperature: float
    raw_nll: float
    calibrated_nll: float | None
    latency: LatencyMetrics
    charset: str
    architecture_name: str
    model_version: str
    checkpoint_version: str
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    parameter_count: int
    dataset_version: str
    split_version: str
    manifest_sha256: str
    split: str
    evidence_status: str
    external_test_status: str
    evaluated_at: str


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


def load_manifest_selection(
    manifest_path: Path,
    *,
    project_root: Path,
    split: str,
    dataset_report_path: Path | None = None,
) -> ManifestSelection:
    if split not in {"validation", "external_test"}:
        raise ValueError("Evaluation split must be 'validation' or 'external_test'.")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"dataset_version", "source", "path", "label", "split", "valid", "sha256"}
    if not rows or not required <= set(rows[0]):
        raise ValueError("Split manifest is empty or missing required columns.")
    versions = {row["dataset_version"] for row in rows}
    if len(versions) != 1:
        raise ValueError("Split manifest contains multiple dataset versions.")
    dataset_version = versions.pop()
    split_version = _sha256(manifest_path)
    external_test_status = (
        "configured"
        if any(
            row["valid"].strip().lower() == "true" and row["split"] == "external_test"
            for row in rows
        )
        else "pending"
    )
    if dataset_report_path is not None:
        try:
            report = json.loads(dataset_report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Dataset report could not be read: {dataset_report_path}") from error
        if not isinstance(report, dict):
            raise ValueError("Dataset report must contain a JSON object.")
        if report.get("dataset_version") != dataset_version:
            raise ValueError("Dataset report and evaluation manifest versions do not match.")
        split_version = str(report.get("split_version", split_version))
        report_splits = report.get("splits")
        if isinstance(report_splits, dict):
            external_test_status = str(report_splits.get("external_test_status", "pending"))

    root = project_root.resolve()
    samples: list[EvaluationSample] = []
    seen_paths: set[Path] = set()
    for row in rows:
        if row["valid"].strip().lower() != "true" or row["split"] != split:
            continue
        path = (root / row["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Manifest path escapes the project root: {row['path']}") from error
        if path in seen_paths:
            raise ValueError(f"Evaluation manifest contains a duplicate path: {row['path']}")
        seen_paths.add(path)
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation image not found: {path}")
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"Evaluation image hash differs from the manifest: {row['path']}")
        samples.append(EvaluationSample(path, row["path"], row["source"], row["label"]))
    if not samples:
        raise EvaluationPendingError(
            f"No valid {split} samples are configured; evaluation pending."
        )
    return ManifestSelection(
        tuple(samples),
        split,
        dataset_version,
        split_version,
        _sha256(manifest_path),
        external_test_status,
    )


def _decode_records(
    logits: Tensor,
    samples: tuple[EvaluationSample, ...],
    recognizer: CaptchaRecognizer,
    *,
    temperature: float,
) -> list[EvaluationRecord]:
    probabilities = (logits / temperature).softmax(dim=2)
    max_probabilities, indices = probabilities.max(dim=2)
    records: list[EvaluationRecord] = []
    for batch_index, sample in enumerate(samples):
        characters = [
            recognizer.codec.index_to_char[int(indices[position, batch_index])]
            for position in range(indices.shape[0])
        ]
        position_confidences = tuple(
            float(max_probabilities[position, batch_index])
            for position in range(max_probabilities.shape[0])
        )
        confidence = math.exp(
            sum(math.log(max(value, 1e-8)) for value in position_confidences)
            / len(position_confidences)
        )
        records.append(
            EvaluationRecord(
                sample.display_path,
                sample.source,
                sample.label,
                "".join(characters),
                confidence,
                position_confidences,
            )
        )
    return records


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_latency(
    model: torch.nn.Module,
    tensor: Tensor,
    *,
    device: torch.device,
    warmup_runs: int,
    measured_runs: int,
) -> LatencyMetrics:
    if warmup_runs < 0 or measured_runs < 1:
        raise ValueError("Latency warmup cannot be negative and measured runs must be positive.")
    model.eval()
    sample = tensor.to(device)
    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(sample)
        _synchronize(device)
        durations = []
        for _ in range(measured_runs):
            started = time.perf_counter_ns()
            model(sample)
            _synchronize(device)
            durations.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(durations)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return LatencyMetrics(
        "preprocessed single-sample model forward",
        str(device),
        warmup_runs,
        measured_runs,
        statistics.fmean(durations),
        statistics.median(durations),
        ordered[p95_index],
    )


def _checkpoint_identity(recognizer: CaptchaRecognizer) -> tuple[str, str, str]:
    metadata = recognizer.metadata
    architecture = metadata.get("architecture") if isinstance(metadata, dict) else None
    if isinstance(architecture, dict):
        name = str(architecture.get("name", MODEL_ARCHITECTURE_NAME))
        version = str(architecture.get("version", MODEL_VERSION))
    else:
        name, version = MODEL_ARCHITECTURE_NAME, MODEL_VERSION
    raw_checkpoint_version = recognizer.checkpoint_version
    checkpoint_version = (
        str(raw_checkpoint_version) if raw_checkpoint_version is not None else "legacy-unversioned"
    )
    return name, version, checkpoint_version


def _evidence_status(recognizer: CaptchaRecognizer, selection: ManifestSelection) -> str:
    metadata = recognizer.metadata
    dataset = metadata.get("dataset") if isinstance(metadata, dict) else None
    if (
        isinstance(dataset, dict)
        and dataset.get("version") == selection.dataset_version
        and dataset.get("split_version") == selection.split_version
    ):
        return "versioned_checkpoint_and_manifest_match"
    return "provisional_checkpoint_training_split_unverified"


def evaluate_checkpoint(
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
    temperature_scaling: bool = False,
) -> EvaluationResult:
    if batch_size < 1:
        raise ValueError("Evaluation batch size must be positive.")
    if temperature_scaling and split != "validation":
        raise ValueError("Temperature fitting is allowed on validation data only.")
    selection = load_manifest_selection(
        manifest_path,
        project_root=project_root,
        split=split,
        dataset_report_path=dataset_report_path,
    )
    recognizer = CaptchaRecognizer(checkpoint_path, device=device, torch_threads=torch_threads)
    checkpoint_charset = set(recognizer.codec.charset)
    if any(
        len(sample.label) != recognizer.config.sequence_length
        or not set(sample.label) <= checkpoint_charset
        for sample in selection.samples
    ):
        raise ValueError("Evaluation labels are incompatible with the checkpoint vocabulary.")

    all_logits: list[Tensor] = []
    all_targets: list[Tensor] = []
    first_tensor: Tensor | None = None
    for offset in range(0, len(selection.samples), batch_size):
        batch = selection.samples[offset : offset + batch_size]
        tensors = []
        for sample in batch:
            with Image.open(sample.path) as image:
                tensors.append(prepare_image(image, recognizer.config, augment=False))
        images = torch.stack(tensors).to(recognizer.device)
        if first_tensor is None:
            first_tensor = images[:1]
        targets = torch.stack([recognizer.codec.encode(sample.label) for sample in batch])
        with torch.inference_mode():
            logits = recognizer.model(images)
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets)

    if first_tensor is None:
        raise EvaluationPendingError(
            f"No valid {split} samples are configured; evaluation pending."
        )
    logits = torch.cat(all_logits, dim=1)
    targets = torch.cat(all_targets, dim=0)
    flat_logits = logits.permute(1, 0, 2).reshape(-1, logits.shape[2])
    flat_targets = targets.reshape(-1)
    raw_records = _decode_records(logits, selection.samples, recognizer, temperature=1.0)
    raw_metrics = calculate_metrics(raw_records, recognizer.codec.charset, ece_bins=ece_bins)
    raw_nll = negative_log_likelihood(flat_logits, flat_targets)

    temperature = 1.0
    calibrated_metrics: EvaluationMetrics | None = None
    calibrated_nll: float | None = None
    if temperature_scaling:
        temperature = fit_temperature(flat_logits, flat_targets)
        calibrated_records = _decode_records(
            logits, selection.samples, recognizer, temperature=temperature
        )
        calibrated_metrics = calculate_metrics(
            calibrated_records, recognizer.codec.charset, ece_bins=ece_bins
        )
        calibrated_nll = negative_log_likelihood(flat_logits, flat_targets, temperature)

    latency = benchmark_latency(
        recognizer.model,
        first_tensor,
        device=recognizer.device,
        warmup_runs=latency_warmup,
        measured_runs=latency_runs,
    )
    architecture_name, model_version, checkpoint_version = _checkpoint_identity(recognizer)
    return EvaluationResult(
        tuple(raw_records),
        raw_metrics,
        calibrated_metrics,
        temperature,
        raw_nll,
        calibrated_nll,
        latency,
        recognizer.codec.charset,
        architecture_name,
        model_version,
        checkpoint_version,
        _project_path(checkpoint_path, project_root),
        _sha256(checkpoint_path),
        checkpoint_path.stat().st_size,
        sum(parameter.numel() for parameter in recognizer.model.parameters()),
        selection.dataset_version,
        selection.split_version,
        selection.manifest_sha256,
        split,
        _evidence_status(recognizer, selection),
        selection.external_test_status,
        datetime.now(UTC).isoformat(),
    )


__all__ = [
    "EvaluationPendingError",
    "EvaluationResult",
    "EvaluationSample",
    "LatencyMetrics",
    "ManifestSelection",
    "benchmark_latency",
    "evaluate_checkpoint",
    "load_manifest_selection",
]
