"""Evidence-aligned model registry comparison without fabricated values."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_id: str
    architecture: str
    version: str
    lifecycle_status: str
    checkpoint_display: str
    checkpoint_path: Path | None
    evaluation_summary_path: Path | None
    training_history_paths: tuple[Path, ...]
    notes: str


def _optional_path(value: object, root: Path, name: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a path string or null.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_registry(path: Path, *, project_root: Path) -> tuple[ModelRegistryEntry, ...]:
    try:
        document: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"Model registry could not be read: {path}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Model registry schema version is missing or unsupported.")
    raw_models = document.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("Model registry must contain a non-empty models list.")
    entries: list[ModelRegistryEntry] = []
    for index, raw_entry in enumerate(raw_models):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Model registry entry {index} must be a mapping.")
        required = {"id", "architecture", "version", "lifecycle_status", "notes"}
        if not required <= set(raw_entry):
            raise ValueError(f"Model registry entry {index} is missing required fields.")
        history_values = raw_entry.get("training_histories", [])
        if not isinstance(history_values, list):
            raise ValueError(f"Model registry entry {index} histories must be a list.")
        entries.append(
            ModelRegistryEntry(
                str(raw_entry["id"]),
                str(raw_entry["architecture"]),
                str(raw_entry["version"]),
                str(raw_entry["lifecycle_status"]),
                str(raw_entry.get("checkpoint") or ""),
                _optional_path(
                    raw_entry.get("checkpoint"), project_root, f"models[{index}].checkpoint"
                ),
                _optional_path(
                    raw_entry.get("evaluation_summary"),
                    project_root,
                    f"models[{index}].evaluation_summary",
                ),
                tuple(
                    resolved
                    for history_index, value in enumerate(history_values)
                    if (
                        resolved := _optional_path(
                            value,
                            project_root,
                            f"models[{index}].training_histories[{history_index}]",
                        )
                    )
                    is not None
                ),
                str(raw_entry["notes"]),
            )
        )
    identifiers = [entry.model_id for entry in entries]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Model registry identifiers must be unique.")
    return tuple(entries)


def _read_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Evaluation summary could not be read: {path}") from error
    if not isinstance(value, dict) or value.get("schema_version") != "1.0":
        raise ValueError(f"Evaluation summary schema is unsupported: {path}")
    return value


def _training_stability(paths: tuple[Path, ...]) -> tuple[int, float | None, float | None, str]:
    best_exact: list[float] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            history: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Training history could not be read: {path}") from error
        if not isinstance(history, list) or not history:
            raise ValueError(f"Training history is empty or invalid: {path}")
        values = [
            float(record["exact_accuracy"])
            for record in history
            if isinstance(record, dict) and "exact_accuracy" in record
        ]
        if not values:
            raise ValueError(f"Training history has no exact-accuracy values: {path}")
        best_exact.append(max(values))
    if not best_exact:
        return 0, None, None, "not_available"
    if len(best_exact) == 1:
        return 1, best_exact[0], None, "insufficient_runs"
    return (
        len(best_exact),
        statistics.fmean(best_exact),
        statistics.stdev(best_exact),
        "measured",
    )


def _nested(mapping: dict[str, Any], section: str, key: str) -> object | None:
    values = mapping.get(section)
    return values.get(key) if isinstance(values, dict) else None


def _reported(summary: dict[str, Any] | None, section: str, key: str) -> object:
    if summary is None:
        return ""
    value = _nested(summary, section, key)
    return "" if value is None else value


def _number(value: object) -> float:
    if isinstance(value, (int, float, str)) and value != "":
        return float(value)
    raise ValueError("A required comparison metric is missing or invalid.")


def build_comparison_rows(entries: tuple[ModelRegistryEntry, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    evidence_keys: set[tuple[object, object, object, object]] = set()
    for entry in entries:
        summary = _read_summary(entry.evaluation_summary_path)
        training_runs, training_mean, training_std, stability_status = _training_stability(
            entry.training_history_paths
        )
        checkpoint_available = bool(entry.checkpoint_path and entry.checkpoint_path.is_file())
        summary_checkpoint_sha = _nested(summary, "model", "checkpoint_sha256") if summary else None
        checkpoint_identity_verified = False
        if checkpoint_available and isinstance(summary_checkpoint_sha, str):
            if (
                entry.checkpoint_path is None
                or _sha256(entry.checkpoint_path) != summary_checkpoint_sha
            ):
                raise ValueError(
                    f"Evaluation summary checkpoint hash differs for {entry.model_id}."
                )
            checkpoint_identity_verified = True
        if summary is not None:
            evidence_key = (
                _nested(summary, "evidence", "dataset_version"),
                _nested(summary, "evidence", "split_version"),
                _nested(summary, "evidence", "split"),
                _nested(summary, "metrics", "sample_count"),
            )
            evidence_keys.add(evidence_key)
        evidence_status = _nested(summary, "evidence", "status") if summary else None
        external_status = _nested(summary, "evidence", "external_test_status") if summary else None
        if entry.lifecycle_status == "production_baseline":
            decision = "retain_default_no_validated_challenger"
        elif entry.lifecycle_status == "deferred":
            decision = "deferred"
        else:
            decision = "not_eligible_missing_evidence"
        rows.append(
            {
                "model_id": entry.model_id,
                "architecture": entry.architecture,
                "version": entry.version,
                "lifecycle_status": entry.lifecycle_status,
                "checkpoint_path": entry.checkpoint_display,
                "checkpoint_available": checkpoint_available,
                "checkpoint_identity_verified": checkpoint_identity_verified,
                "evaluation_status": "measured" if summary else "not_available",
                "evidence_status": evidence_status or "",
                "dataset_version": _reported(summary, "evidence", "dataset_version"),
                "split_version": _reported(summary, "evidence", "split_version"),
                "split": _reported(summary, "evidence", "split"),
                "external_test_status": external_status or "",
                "sample_count": _reported(summary, "metrics", "sample_count"),
                "character_accuracy": _reported(summary, "metrics", "character_accuracy"),
                "exact_accuracy": _reported(summary, "metrics", "exact_accuracy"),
                "character_error_rate": _reported(summary, "metrics", "character_error_rate"),
                "latency_median_ms": _reported(summary, "latency", "median_ms"),
                "latency_p95_ms": _reported(summary, "latency", "p95_ms"),
                "checkpoint_size_bytes": _reported(summary, "model", "checkpoint_size_bytes"),
                "cpu_model_tensor_bytes": _reported(summary, "model", "cpu_model_tensor_bytes"),
                "sequence_ece": _reported(summary, "metrics", "sequence_ece"),
                "training_runs": training_runs or "",
                "training_exact_mean": training_mean if training_mean is not None else "",
                "training_exact_std": training_std if training_std is not None else "",
                "stability_status": stability_status,
                "promotion_eligible": False,
                "decision": decision,
                "notes": entry.notes,
            }
        )
    if len(evidence_keys) > 1:
        raise ValueError("Measured model summaries do not use identical evaluation evidence.")
    baselines = [row for row in rows if row["lifecycle_status"] == "production_baseline"]
    if len(baselines) != 1:
        raise ValueError("Model registry must contain exactly one production baseline.")
    baseline = baselines[0]
    required_metrics = (
        "exact_accuracy",
        "character_error_rate",
        "latency_median_ms",
        "cpu_model_tensor_bytes",
        "sequence_ece",
    )
    for row in rows:
        if row is baseline or row["lifecycle_status"] == "deferred":
            continue
        comparable = all(row[key] != "" and baseline[key] != "" for key in required_metrics)
        eligible = bool(
            comparable
            and row["checkpoint_available"]
            and baseline["checkpoint_available"]
            and row["checkpoint_identity_verified"]
            and baseline["checkpoint_identity_verified"]
            and row["evaluation_status"] == "measured"
            and row["evidence_status"] == "versioned_checkpoint_and_manifest_match"
            and baseline["evidence_status"] == "versioned_checkpoint_and_manifest_match"
            and row["split"] == "external_test"
            and row["stability_status"] == "measured"
            and _number(row["exact_accuracy"]) >= _number(baseline["exact_accuracy"])
            and _number(row["character_error_rate"]) <= _number(baseline["character_error_rate"])
            and _number(row["latency_median_ms"]) <= _number(baseline["latency_median_ms"]) * 1.10
            and _number(row["cpu_model_tensor_bytes"])
            <= _number(baseline["cpu_model_tensor_bytes"]) * 1.25
            and _number(row["sequence_ece"]) <= _number(baseline["sequence_ece"])
        )
        row["promotion_eligible"] = eligible
        if eligible:
            row["decision"] = "eligible_for_human_promotion_review"
    return rows


def _format(value: object, *, percentage: bool = False) -> str:
    if value == "" or value is None:
        return "Not measured"
    if isinstance(value, (int, float)) and percentage:
        return f"{float(value):.2%}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _mib(value: object) -> object:
    return (
        float(value) / (1024 * 1024) if isinstance(value, (int, float, str)) and value != "" else ""
    )


def write_comparison(rows: list[dict[str, object]], *, csv_path: Path, document_path: Path) -> None:
    if not rows:
        raise ValueError("At least one model comparison row is required.")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_path.write_text(stream.getvalue(), encoding="utf-8")

    table = [
        "| Model | Status | Exact | CER | Median ms | Model MiB | CPU tensors MiB | ECE | Stability | Decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        checkpoint_size = row["checkpoint_size_bytes"]
        tensor_size = row["cpu_model_tensor_bytes"]
        table.append(
            "| "
            + " | ".join(
                (
                    str(row["model_id"]),
                    str(row["lifecycle_status"]),
                    _format(row["exact_accuracy"], percentage=True),
                    _format(row["character_error_rate"], percentage=True),
                    _format(row["latency_median_ms"]),
                    _format(_mib(checkpoint_size)),
                    _format(_mib(tensor_size)),
                    _format(row["sequence_ece"]),
                    str(row["stability_status"]),
                    str(row["decision"]),
                )
            )
            + " |"
        )
    document = """# CipherLens Model Comparison

## Decision

Model V1 remains the production default. It is the only model with measured
repository validation evidence, and that evidence is provisional because the
legacy checkpoint lacks versioned training-split provenance. No model is eligible
for promotion without aligned external-test evidence and repeated-run stability.

## Evidence table

""" + "\n".join(table)
    document += """

CSV blanks and table cells marked `Not measured` mean the evidence does not exist;
they are not zero values. Latency is warmed-up single-sample model-forward time.
CPU memory is resident model tensor
storage only, excluding framework and workspace overhead. Comparisons require the
same dataset version, split version, split role, and sample count; checkpoint
SHA-256 must match its evaluation summary.

Promotion review requires aligned versioned external-test evidence, at least two
training runs, exact accuracy no worse than V1, CER and ECE no worse than V1,
median latency within 10% of V1, CPU model tensors within 25% of V1, and explicit
human approval. Passing these gates permits review; it does not auto-promote.

## Architecture decisions

- **V1 position-wise CRNN:** retained as the rollback-safe production baseline.
- **V2 CRNN-CTC:** implemented as an optional experiment, but no candidate was
  trained or evaluated during this milestone.
- **V3 transformer OCR:** deferred. The current 1,000-image dataset, incomplete
  provenance, and missing external-test set do not justify the added dependency,
  compute, and overfitting risk.
"""
    document_path.parent.mkdir(parents=True, exist_ok=True)
    document_path.write_text(document, encoding="utf-8")


__all__ = [
    "ModelRegistryEntry",
    "build_comparison_rows",
    "load_model_registry",
    "write_comparison",
]
