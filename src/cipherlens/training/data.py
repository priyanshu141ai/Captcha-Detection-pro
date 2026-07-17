"""Versioned training splits, reproducible loaders, and loss weights."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from cipherlens.data import (
    CaptchaDataset,
    CaptchaSample,
    collate_captchas,
    coverage_aware_split,
)
from cipherlens.models import CaptchaCodec, ModelConfig
from cipherlens.utils import make_torch_generator, seed_worker


@dataclass(frozen=True)
class TrainingSplit:
    training: tuple[CaptchaSample, ...]
    validation: tuple[CaptchaSample, ...]
    dataset_version: str
    split_version: str
    selection_hash: str
    manifest_path: str | None


@dataclass(frozen=True)
class TrainingLoaders:
    training: DataLoader[Any]
    validation: DataLoader[Any]
    generator: torch.Generator


def _hash_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fallback_split(
    samples: list[CaptchaSample],
    validation_fraction: float,
    seed: int,
    project_root: Path,
) -> TrainingSplit:
    training, validation = coverage_aware_split(samples, validation_fraction, seed)
    assignments = {sample.path.resolve(): "train" for sample in training} | {
        sample.path.resolve(): "validation" for sample in validation
    }
    records: list[dict[str, str]] = []
    selected: list[dict[str, str]] = []
    for sample in sorted(samples, key=lambda item: item.path.resolve().as_posix()):
        resolved = sample.path.resolve()
        try:
            display_path = resolved.relative_to(project_root.resolve()).as_posix()
        except ValueError:
            display_path = f"external/{sample.path.parent.name}/{sample.path.name}"
        record = {
            "path": display_path,
            "label": sample.label,
            "sha256": hashlib.sha256(sample.path.read_bytes()).hexdigest(),
        }
        records.append(record)
        selected.append({**record, "split": assignments[resolved]})
    dataset_version = _hash_json(records)
    selection_hash = _hash_json(selected)
    return TrainingSplit(
        tuple(training),
        tuple(validation),
        dataset_version,
        selection_hash,
        selection_hash,
        None,
    )


def load_training_split(
    samples: list[CaptchaSample],
    *,
    project_root: Path,
    manifest_path: Path | None,
    dataset_report_path: Path | None,
    validation_fraction: float,
    seed: int,
) -> TrainingSplit:
    """Load exact train/validation rows and reject stale or external samples."""
    if manifest_path is None:
        return _fallback_split(samples, validation_fraction, seed, project_root)
    resolved_manifest = (
        manifest_path if manifest_path.is_absolute() else project_root / manifest_path
    )
    if not resolved_manifest.is_file():
        raise FileNotFoundError(f"Split manifest not found: {resolved_manifest}")
    with resolved_manifest.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"dataset_version", "path", "label", "split", "valid", "sha256"}
    if not rows or not required <= set(rows[0]):
        raise ValueError("Split manifest is empty or missing required columns.")
    versions = {row["dataset_version"] for row in rows}
    if len(versions) != 1:
        raise ValueError("Split manifest contains multiple dataset versions.")
    dataset_version = versions.pop()
    rows_by_path = {
        (project_root / row["path"]).resolve(): row
        for row in rows
        if row["valid"].strip().lower() == "true"
    }
    training: list[CaptchaSample] = []
    validation: list[CaptchaSample] = []
    selected_rows: list[dict[str, str]] = []
    for sample in samples:
        resolved_path = sample.path.resolve()
        row = rows_by_path.get(resolved_path)
        if row is None:
            raise ValueError(f"Sample is absent or invalid in the split manifest: {sample.path}")
        if row["label"] != sample.label:
            raise ValueError(f"Label differs from the split manifest for: {sample.path}")
        if hashlib.sha256(sample.path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Image hash differs from the split manifest for: {sample.path}")
        split = row["split"]
        if split == "train":
            training.append(sample)
        elif split == "validation":
            validation.append(sample)
        else:
            raise ValueError(f"Training cannot consume a {split!r} manifest row: {sample.path}")
        selected_rows.append(
            {
                "path": row["path"],
                "label": row["label"],
                "sha256": row["sha256"],
                "split": split,
            }
        )
    if not training or not validation:
        raise ValueError(
            "The selected manifest rows require non-empty train and validation splits."
        )

    split_version = hashlib.sha256(resolved_manifest.read_bytes()).hexdigest()
    if dataset_report_path is not None:
        report_path = (
            dataset_report_path
            if dataset_report_path.is_absolute()
            else project_root / dataset_report_path
        )
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Dataset report could not be read: {report_path}") from error
        if report.get("dataset_version") != dataset_version:
            raise ValueError("Dataset report and split manifest versions do not match.")
        split_version = str(report.get("split_version", split_version))

    selected_rows.sort(key=lambda row: row["path"])
    return TrainingSplit(
        tuple(training),
        tuple(validation),
        dataset_version,
        split_version,
        _hash_json(selected_rows),
        resolved_manifest.resolve().as_posix(),
    )


def build_loaders(
    split: TrainingSplit,
    codec: CaptchaCodec,
    model_config: ModelConfig,
    *,
    batch_size: int,
    num_workers: int,
    cache_images: bool,
    device: torch.device,
    seed: int,
) -> TrainingLoaders:
    generator = make_torch_generator(seed)
    training = DataLoader(
        CaptchaDataset(list(split.training), codec, model_config, True, cache_images),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate_captchas,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
    )
    validation = DataLoader(
        CaptchaDataset(list(split.validation), codec, model_config, False, cache_images),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_captchas,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
    )
    return TrainingLoaders(training, validation, generator)


def build_class_weights(samples: tuple[CaptchaSample, ...], codec: CaptchaCodec) -> torch.Tensor:
    counts = Counter(character for sample in samples for character in sample.label)
    values = torch.tensor([counts[character] for character in codec.charset], dtype=torch.float32)
    weights = torch.sqrt(values.max() / values.clamp_min(1.0)).clamp_max(4.0)
    return weights / weights.mean()


__all__ = [
    "TrainingLoaders",
    "TrainingSplit",
    "build_class_weights",
    "build_loaders",
    "load_training_split",
]
