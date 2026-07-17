"""CSV, JSON, PNG, and model-card evaluation exports."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from cipherlens.evaluation.metrics import EvaluationMetrics
from cipherlens.evaluation.runner import EvaluationResult
from cipherlens.models import levenshtein_distance


@dataclass(frozen=True)
class EvaluationReportPaths:
    model_comparison: Path
    failed_predictions: Path
    evaluation_summary: Path
    per_character_metrics: Path
    per_position_metrics: Path
    confidence_distribution: Path
    confusion_matrix: Path
    reliability_diagram: Path
    model_card: Path

    @classmethod
    def from_directories(
        cls, evaluation_path: Path, figures_path: Path, model_card_path: Path
    ) -> EvaluationReportPaths:
        return cls(
            evaluation_path / "model_comparison.csv",
            evaluation_path / "failed_predictions.csv",
            evaluation_path / "evaluation_summary.json",
            evaluation_path / "per_character_metrics.csv",
            evaluation_path / "per_position_metrics.csv",
            evaluation_path / "confidence_distribution.csv",
            figures_path / "confusion_matrix.png",
            figures_path / "reliability_diagram.png",
            model_card_path,
        )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, stream.getvalue())


def _rounded(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    image.save(temporary, format="PNG", optimize=True)
    temporary.replace(path)


def _confusion_color(value: int, maximum: int) -> tuple[int, int, int]:
    ratio = math.sqrt(value / maximum) if maximum else 0.0
    return (
        round(247 - 216 * ratio),
        round(250 - 131 * ratio),
        round(252 - 72 * ratio),
    )


def write_confusion_matrix(
    path: Path, metrics: EvaluationMetrics, charset: str, *, split: str
) -> None:
    cell = 19
    left, top, right, bottom = 130, 150, 90, 125
    matrix_size = cell * len(charset)
    image = Image.new("RGB", (left + matrix_size + right, top + matrix_size + bottom), "white")
    draw = ImageDraw.Draw(image)
    title_font, body_font, label_font = _font(24), _font(14), _font(12)
    draw.text((left, 30), f"Character confusion matrix - {split}", fill="#111827", font=title_font)
    draw.text(
        (left, 68),
        "Rows: true labels. Columns: predictions. Color: raw count (square-root scale).",
        fill="#4B5563",
        font=body_font,
    )
    maximum = max(max(row) for row in metrics.confusion_matrix)
    for row_index, row in enumerate(metrics.confusion_matrix):
        y = top + row_index * cell
        draw.text((left - 24, y + 2), charset[row_index], fill="#111827", font=label_font)
        for column_index, value in enumerate(row):
            x = left + column_index * cell
            draw.rectangle(
                (x, y, x + cell - 1, y + cell - 1),
                fill=_confusion_color(value, maximum),
            )
    for column_index, character in enumerate(charset):
        draw.text(
            (left + column_index * cell + 5, top - 22),
            character,
            fill="#111827",
            font=label_font,
        )
    draw.text((left - 82, top - 22), "true", fill="#4B5563", font=body_font)
    draw.text((left, top - 48), "predictions", fill="#4B5563", font=body_font)
    legend_y = top + matrix_size + 42
    draw.text((left, legend_y - 2), "count", fill="#4B5563", font=body_font)
    for index in range(101):
        draw.line(
            (left + 65 + index * 2, legend_y, left + 65 + index * 2, legend_y + 18),
            fill=_confusion_color(round(maximum * (index / 100) ** 2), maximum),
        )
    draw.text((left + 60, legend_y + 24), "0", fill="#4B5563", font=label_font)
    draw.text((left + 245, legend_y + 24), str(maximum), fill="#4B5563", font=label_font)
    _save_png(image, path)


def write_reliability_diagram(
    path: Path,
    metrics: EvaluationMetrics,
    *,
    split: str,
    temperature: float,
    calibrated: bool,
) -> None:
    image = Image.new("RGB", (1000, 760), "white")
    draw = ImageDraw.Draw(image)
    title_font, body_font, label_font = _font(25), _font(15), _font(12)
    qualifier = "temperature-scaled" if calibrated else "uncalibrated"
    draw.text(
        (95, 28),
        f"Sequence confidence reliability - {split}",
        fill="#111827",
        font=title_font,
    )
    draw.text(
        (95, 70),
        f"{qualifier}; ECE={metrics.sequence_ece:.4f}; n={metrics.sample_count}; T={temperature:.4f}",
        fill="#4B5563",
        font=body_font,
    )
    left, right, top, bottom = 95, 930, 130, 505
    draw.line((left, bottom, right, bottom), fill="#111827", width=2)
    draw.line((left, top, left, bottom), fill="#111827", width=2)
    draw.line((left, bottom, right, top), fill="#6B7280", width=2)
    bin_width = (right - left) / len(metrics.reliability_bins)
    for index, item in enumerate(metrics.reliability_bins):
        x0 = round(left + index * bin_width + 4)
        x1 = round(left + (index + 1) * bin_width - 4)
        if item.accuracy is not None:
            y = round(bottom - item.accuracy * (bottom - top))
            draw.rectangle((x0, y, x1, bottom), fill="#3B82F6", outline="#1D4ED8")
        if item.mean_confidence is not None:
            marker_x = round(left + item.mean_confidence * (right - left))
            marker_y = round(bottom - (item.accuracy or 0.0) * (bottom - top))
            draw.ellipse((marker_x - 4, marker_y - 4, marker_x + 4, marker_y + 4), fill="#111827")
    for tick in range(6):
        value = tick / 5
        x = round(left + value * (right - left))
        y = round(bottom - value * (bottom - top))
        draw.text((x - 12, bottom + 10), f"{value:.1f}", fill="#4B5563", font=label_font)
        draw.text((left - 42, y - 7), f"{value:.1f}", fill="#4B5563", font=label_font)
    draw.text((410, bottom + 43), "mean sequence confidence", fill="#111827", font=body_font)
    draw.text((13, 300), "exact accuracy", fill="#111827", font=body_font)
    draw.text((640, 112), "ideal calibration", fill="#6B7280", font=label_font)

    histogram_top, histogram_bottom = 585, 700
    maximum_count = max(item.count for item in metrics.reliability_bins)
    draw.text((left, 548), "Samples per confidence bin", fill="#111827", font=body_font)
    for index, item in enumerate(metrics.reliability_bins):
        x0 = round(left + index * bin_width + 4)
        x1 = round(left + (index + 1) * bin_width - 4)
        height = (item.count / maximum_count) * (histogram_bottom - histogram_top)
        draw.rectangle((x0, round(histogram_bottom - height), x1, histogram_bottom), fill="#9CA3AF")
        if item.count:
            draw.text(
                (x0 + 2, round(histogram_bottom - height) - 17),
                str(item.count),
                fill="#374151",
                font=label_font,
            )
    draw.line((left, histogram_bottom, right, histogram_bottom), fill="#111827", width=2)
    _save_png(image, path)


def _summary_payload(result: EvaluationResult) -> dict[str, object]:
    def aggregate(metrics: EvaluationMetrics) -> dict[str, object]:
        return {
            "sample_count": metrics.sample_count,
            "character_count": metrics.character_count,
            "character_accuracy": metrics.character_accuracy,
            "exact_accuracy": metrics.exact_accuracy,
            "character_error_rate": metrics.character_error_rate,
            "normalized_edit_distance": metrics.normalized_edit_distance,
            "mean_confidence": metrics.mean_confidence,
            "median_confidence": metrics.median_confidence,
            "sequence_ece": metrics.sequence_ece,
            "per_position_accuracy": metrics.per_position_accuracy,
            "per_position_support": metrics.per_position_support,
        }

    return {
        "schema_version": "1.0",
        "evaluated_at": result.evaluated_at,
        "model": {
            "architecture_name": result.architecture_name,
            "model_version": result.model_version,
            "checkpoint_version": result.checkpoint_version,
            "checkpoint_path": result.checkpoint_path,
            "checkpoint_sha256": result.checkpoint_sha256,
            "checkpoint_size_bytes": result.checkpoint_size_bytes,
            "parameter_count": result.parameter_count,
            "cpu_model_tensor_bytes": result.cpu_model_tensor_bytes,
        },
        "evidence": {
            "dataset_version": result.dataset_version,
            "split_version": result.split_version,
            "manifest_sha256": result.manifest_sha256,
            "split": result.split,
            "status": result.evidence_status,
            "external_test_status": result.external_test_status,
        },
        "metrics": aggregate(result.metrics),
        "calibration": {
            "temperature": result.temperature,
            "raw_nll": result.raw_nll,
            "calibrated_nll": result.calibrated_nll,
            "calibrated_metrics": (
                aggregate(result.calibrated_metrics)
                if result.calibrated_metrics is not None
                else None
            ),
            "note": (
                "Temperature was fit and evaluated on validation data; an independent calibration "
                "split is not configured."
                if result.calibrated_metrics is not None
                else "Temperature scaling was not applied."
            ),
        },
        "latency": asdict(result.latency),
    }


def _model_card(result: EvaluationResult) -> str:
    metrics = result.metrics
    calibrated = result.calibrated_metrics
    evidence_text = (
        "The checkpoint dataset and split versions match the evaluation manifest. These are still "
        "validation-only, dataset-specific results—not external-generalization evidence."
        if result.evidence_status == "versioned_checkpoint_and_manifest_match"
        else "The evaluated checkpoint does not record enough training-split provenance to prove "
        "that it was trained without overlap with the newer versioned manifest. These values are "
        "therefore dataset-specific, provisional diagnostics—not promotion or external-"
        "generalization evidence."
    )
    calibration_text = (
        f"Validation-fitted temperature `{result.temperature:.4f}` changed sequence ECE from "
        f"`{metrics.sequence_ece:.4f}` to `{calibrated.sequence_ece:.4f}`. This is not independent "
        "calibration evidence."
        if calibrated is not None
        else "Temperature scaling was not applied because no independent calibration split exists."
    )
    return f"""# CipherLens Model Card

## Model summary

CipherLens Model V1 is a compact position-wise CRNN for six-character synthetic,
owned, or explicitly authorized CAPTCHA-style images. It must not be integrated
with third-party sites or used to bypass access controls.

- Architecture: `{result.architecture_name}` version `{result.model_version}`
- Checkpoint schema: `{result.checkpoint_version}`
- Vocabulary size: `{len(result.charset)}`
- Parameters: `{result.parameter_count:,}`
- Checkpoint size: `{result.checkpoint_size_bytes / (1024 * 1024):.2f} MiB`
- CPU model tensors: `{result.cpu_model_tensor_bytes / (1024 * 1024):.2f} MiB`

## Evaluation evidence

| Metric | {result.split.replace("_", " ").title()} |
|---|---:|
| Samples | {metrics.sample_count} |
| Character accuracy | {metrics.character_accuracy:.4%} |
| Exact-string accuracy | {metrics.exact_accuracy:.4%} |
| Character error rate | {metrics.character_error_rate:.4%} |
| Mean normalized edit distance | {metrics.normalized_edit_distance:.6f} |
| Sequence expected calibration error | {metrics.sequence_ece:.6f} |

Evidence status: `{result.evidence_status}`. {evidence_text}

External-test status: **{result.external_test_status}**. No external-test score is
reported when that split is unavailable.

## Calibration and confidence

{calibration_text}

The reliability diagram uses sequence confidence against exact-string correctness;
bars show observed accuracy and the lower panel shows sample count per bin.

## Runtime

Single-sample `{result.latency.scope}` latency on `{result.latency.device}`:
mean `{result.latency.mean_ms:.3f} ms`, median `{result.latency.median_ms:.3f} ms`,
p95 `{result.latency.p95_ms:.3f} ms` across `{result.latency.measured_runs}` measured runs
after `{result.latency.warmup_runs}` warmups. Image decode and preprocessing are excluded.

## Known limitations

- Fixed six-character output and checkpoint vocabulary.
- Dataset generation provenance and independent license evidence remain incomplete.
- No external-test or independent calibration split is currently configured.
- Confidence should not be interpreted as a universal probability of correctness.
- The current validation evidence may overlap with historical training data.

## Reproduce

```powershell
python -m scripts.evaluate_model
```

Artifacts: [comparison](../reports/evaluation/model_comparison.csv),
[failures](../reports/evaluation/failed_predictions.csv),
[confusion matrix](../reports/figures/confusion_matrix.png), and
[reliability diagram](../reports/figures/reliability_diagram.png). See also the
[model comparison](model-comparison.md).

Generated from dataset version `{result.dataset_version}` and split version
`{result.split_version}` at `{result.evaluated_at}`.
"""


def write_evaluation_reports(
    result: EvaluationResult, paths: EvaluationReportPaths
) -> EvaluationReportPaths:
    metrics = result.metrics
    calibrated_ece = result.calibrated_metrics.sequence_ece if result.calibrated_metrics else None
    comparison = {
        "model_name": result.architecture_name,
        "model_version": result.model_version,
        "checkpoint_version": result.checkpoint_version,
        "checkpoint_path": result.checkpoint_path,
        "checkpoint_sha256": result.checkpoint_sha256,
        "dataset_version": result.dataset_version,
        "split_version": result.split_version,
        "split": result.split,
        "sample_count": metrics.sample_count,
        "character_accuracy": _rounded(metrics.character_accuracy),
        "exact_accuracy": _rounded(metrics.exact_accuracy),
        "character_error_rate": _rounded(metrics.character_error_rate),
        "normalized_edit_distance": _rounded(metrics.normalized_edit_distance),
        "sequence_ece": _rounded(metrics.sequence_ece),
        "temperature": _rounded(result.temperature),
        "calibrated_sequence_ece": _rounded(calibrated_ece),
        "latency_mean_ms": _rounded(result.latency.mean_ms),
        "latency_median_ms": _rounded(result.latency.median_ms),
        "latency_p95_ms": _rounded(result.latency.p95_ms),
        "checkpoint_size_bytes": result.checkpoint_size_bytes,
        "parameter_count": result.parameter_count,
        "cpu_model_tensor_bytes": result.cpu_model_tensor_bytes,
        "evidence_status": result.evidence_status,
        "external_test_status": result.external_test_status,
    }
    _write_csv(paths.model_comparison, list(comparison), [comparison])
    failure_fields = [
        "path",
        "source",
        "target",
        "prediction",
        "confidence",
        "edit_distance",
        "wrong_positions",
    ]
    failures = []
    for record in result.records:
        if record.target == record.prediction:
            continue
        failures.append(
            {
                "path": record.path,
                "source": record.source,
                "target": record.target,
                "prediction": record.prediction,
                "confidence": _rounded(record.confidence),
                "edit_distance": levenshtein_distance(record.target, record.prediction),
                "wrong_positions": "|".join(
                    str(index + 1)
                    for index, (target, prediction) in enumerate(
                        zip(record.target, record.prediction, strict=True)
                    )
                    if target != prediction
                ),
            }
        )
    _write_csv(paths.failed_predictions, failure_fields, failures)
    _write_csv(
        paths.per_character_metrics,
        ["character", "support", "predicted", "true_positives", "precision", "recall", "f1"],
        [
            {
                **asdict(item),
                "precision": _rounded(item.precision),
                "recall": _rounded(item.recall),
                "f1": _rounded(item.f1),
            }
            for item in metrics.per_character
        ],
    )
    _write_csv(
        paths.per_position_metrics,
        ["position", "support", "accuracy"],
        [
            {"position": index + 1, "support": support, "accuracy": _rounded(accuracy)}
            for index, (support, accuracy) in enumerate(
                zip(metrics.per_position_support, metrics.per_position_accuracy, strict=True)
            )
        ],
    )
    diagram_metrics = result.calibrated_metrics or metrics
    _write_csv(
        paths.confidence_distribution,
        ["lower", "upper", "count", "mean_confidence", "exact_accuracy", "gap"],
        [
            {
                "lower": _rounded(item.lower),
                "upper": _rounded(item.upper),
                "count": item.count,
                "mean_confidence": _rounded(item.mean_confidence),
                "exact_accuracy": _rounded(item.accuracy),
                "gap": _rounded(item.gap),
            }
            for item in diagram_metrics.reliability_bins
        ],
    )
    _atomic_text(
        paths.evaluation_summary,
        json.dumps(_summary_payload(result), indent=2, sort_keys=True) + "\n",
    )
    write_confusion_matrix(paths.confusion_matrix, metrics, result.charset, split=result.split)
    write_reliability_diagram(
        paths.reliability_diagram,
        diagram_metrics,
        split=result.split,
        temperature=result.temperature,
        calibrated=result.calibrated_metrics is not None,
    )
    _atomic_text(paths.model_card, _model_card(result))
    return paths


__all__ = [
    "EvaluationReportPaths",
    "write_confusion_matrix",
    "write_evaluation_reports",
    "write_reliability_diagram",
]
