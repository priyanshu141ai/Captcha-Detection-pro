from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from cipherlens.evaluation import (
    EvaluationPendingError,
    EvaluationRecord,
    EvaluationReportPaths,
    calculate_metrics,
    evaluate_checkpoint,
    fit_temperature,
    load_manifest_selection,
    negative_log_likelihood,
    write_evaluation_reports,
)
from cipherlens.models import MODEL_ARCHITECTURE_NAME, MODEL_VERSION, CaptchaCRNN, ModelConfig


class EvaluationMetricTests(unittest.TestCase):
    def test_fixed_length_metrics_and_reliability_are_exact(self) -> None:
        records = [
            EvaluationRecord("one.png", "fixture", "AAA", "AAA", 0.9, (0.9, 0.9, 0.9)),
            EvaluationRecord("two.png", "fixture", "ABA", "ABB", 0.6, (0.8, 0.7, 0.6)),
        ]

        metrics = calculate_metrics(records, "AB", ece_bins=5)

        self.assertEqual(metrics.sample_count, 2)
        self.assertEqual(metrics.character_count, 6)
        self.assertAlmostEqual(metrics.character_accuracy, 5 / 6)
        self.assertEqual(metrics.exact_accuracy, 0.5)
        self.assertAlmostEqual(metrics.character_error_rate, 1 / 6)
        self.assertAlmostEqual(metrics.normalized_edit_distance, 1 / 6)
        self.assertEqual(metrics.per_position_accuracy, (1.0, 1.0, 0.5))
        self.assertEqual(sum(item.count for item in metrics.reliability_bins), 2)
        self.assertAlmostEqual(metrics.sequence_ece, 0.35)
        self.assertEqual(metrics.confusion_matrix, ((4, 1), (0, 1)))
        missing = calculate_metrics(records, "ABC", ece_bins=5).per_character[2]
        self.assertIsNone(missing.precision)
        self.assertIsNone(missing.recall)
        self.assertIsNone(missing.f1)

    def test_temperature_scaling_reduces_validation_nll(self) -> None:
        logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0]])
        targets = torch.tensor([0, 1, 0, 0])

        temperature = fit_temperature(logits, targets)

        self.assertGreaterEqual(temperature, 0.05)
        self.assertLessEqual(temperature, 20.0)
        self.assertLessEqual(
            negative_log_likelihood(logits, targets, temperature),
            negative_log_likelihood(logits, targets),
        )


class EvaluationPipelineTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        torch.manual_seed(7)
        images = root / "images"
        images.mkdir()
        rows = []
        for index, label in enumerate(("AAAAAA", "BBBBBB")):
            path = images / f"{index}.png"
            Image.new("RGB", (16, 16), (index * 255, 255, 255)).save(path)
            rows.append(
                {
                    "dataset_version": "dataset-v1",
                    "source": "fixture",
                    "path": path.relative_to(root).as_posix(),
                    "label": label,
                    "split": "validation",
                    "valid": "True",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        manifest = root / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report = root / "report.json"
        report.write_text(
            json.dumps(
                {
                    "dataset_version": "dataset-v1",
                    "split_version": "split-v1",
                    "splits": {"external_test_status": "pending"},
                }
            ),
            encoding="utf-8",
        )
        config = ModelConfig(
            image_height=16,
            image_width=16,
            hidden_size=4,
            lstm_layers=1,
            sequence_length=6,
        )
        model = CaptchaCRNN(2, config)
        checkpoint = root / "model.pt"
        torch.save(
            {
                "checkpoint_version": 2,
                "model_state": model.state_dict(),
                "charset": "AB",
                "model_config": vars(config),
                "metadata": {
                    "architecture": {
                        "name": MODEL_ARCHITECTURE_NAME,
                        "version": MODEL_VERSION,
                    },
                    "dataset": {"version": "dataset-v1", "split_version": "split-v1"},
                },
            },
            checkpoint,
        )
        return checkpoint, manifest, report

    def test_checkpoint_evaluation_and_reports_use_generated_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, manifest, report = self._fixture(root)

            result = evaluate_checkpoint(
                checkpoint,
                manifest,
                project_root=root,
                dataset_report_path=report,
                batch_size=2,
                torch_threads=1,
                ece_bins=5,
                latency_warmup=0,
                latency_runs=2,
                temperature_scaling=True,
            )
            paths = EvaluationReportPaths.from_directories(
                root / "reports/evaluation", root / "reports/figures", root / "docs/model-card.md"
            )
            write_evaluation_reports(result, paths)

            self.assertEqual(result.metrics.sample_count, 2)
            self.assertEqual(result.metrics.character_count, 12)
            self.assertEqual(result.evidence_status, "versioned_checkpoint_and_manifest_match")
            self.assertEqual(result.external_test_status, "pending")
            self.assertIsNotNone(result.calibrated_metrics)
            self.assertGreater(result.temperature, 0.0)
            self.assertGreater(result.parameter_count, 0)
            for path in vars(paths).values():
                self.assertTrue(path.is_file(), path)
            with Image.open(paths.confusion_matrix) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, 100)
            summary = json.loads(paths.evaluation_summary.read_text(encoding="utf-8"))
            self.assertEqual(summary["evidence"]["external_test_status"], "pending")
            self.assertIn("External-test status", paths.model_card.read_text(encoding="utf-8"))

    def test_missing_external_split_is_reported_as_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _checkpoint, manifest, report = self._fixture(root)

            with self.assertRaisesRegex(EvaluationPendingError, "pending"):
                load_manifest_selection(
                    manifest,
                    project_root=root,
                    split="external_test",
                    dataset_report_path=report,
                )


if __name__ == "__main__":
    unittest.main()
