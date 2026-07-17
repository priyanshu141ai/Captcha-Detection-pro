from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cipherlens.evaluation.comparison import (
    build_comparison_rows,
    load_model_registry,
    write_comparison,
)


class ModelComparisonTests(unittest.TestCase):
    def _summary(
        self,
        path: Path,
        *,
        dataset_version: str = "dataset-v1",
        split: str = "validation",
        evidence_status: str = "provisional_checkpoint_training_split_unverified",
        external_status: str = "pending",
        exact_accuracy: float = 0.0,
        character_error_rate: float = 1.0,
        median_ms: float = 2.0,
        tensor_bytes: int = 2048,
        sequence_ece: float = 0.0,
        checkpoint_sha256: str | None = None,
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "model": {
                        "checkpoint_size_bytes": 1024,
                        "cpu_model_tensor_bytes": tensor_bytes,
                        "checkpoint_sha256": checkpoint_sha256,
                    },
                    "evidence": {
                        "dataset_version": dataset_version,
                        "split_version": "split-v1",
                        "split": split,
                        "status": evidence_status,
                        "external_test_status": external_status,
                    },
                    "metrics": {
                        "sample_count": 2,
                        "character_accuracy": 0.0,
                        "exact_accuracy": exact_accuracy,
                        "character_error_rate": character_error_rate,
                        "sequence_ece": sequence_ece,
                    },
                    "latency": {"median_ms": median_ms, "p95_ms": median_ms * 1.5},
                }
            ),
            encoding="utf-8",
        )

    def test_missing_models_remain_blank_and_zero_metrics_remain_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "v1.json"
            self._summary(summary)
            registry = root / "registry.yaml"
            registry.write_text(
                "schema_version: 1\n"
                "models:\n"
                "  - id: v1\n"
                "    architecture: positionwise\n"
                "    version: '1'\n"
                "    lifecycle_status: production_baseline\n"
                "    checkpoint: missing-v1.pt\n"
                "    evaluation_summary: v1.json\n"
                "    training_histories: []\n"
                "    notes: baseline\n"
                "  - id: v2\n"
                "    architecture: ctc\n"
                "    version: '2'\n"
                "    lifecycle_status: experimental_not_trained\n"
                "    checkpoint: missing-v2.pt\n"
                "    evaluation_summary: missing-v2.json\n"
                "    training_histories: []\n"
                "    notes: pending\n",
                encoding="utf-8",
            )

            rows = build_comparison_rows(load_model_registry(registry, project_root=root))
            csv_path, document = root / "comparison.csv", root / "comparison.md"
            write_comparison(rows, csv_path=csv_path, document_path=document)

            self.assertEqual(rows[0]["exact_accuracy"], 0.0)
            self.assertEqual(rows[0]["sequence_ece"], 0.0)
            self.assertEqual(rows[1]["exact_accuracy"], "")
            self.assertEqual(rows[1]["evaluation_status"], "not_available")
            self.assertIn("CSV blanks", document.read_text(encoding="utf-8"))
            self.assertEqual(len(csv_path.read_text(encoding="utf-8").splitlines()), 3)

    def test_candidate_must_pass_every_promotion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "baseline.pt").write_bytes(b"baseline")
            (root / "candidate.pt").write_bytes(b"candidate")
            common = {
                "split": "external_test",
                "evidence_status": "versioned_checkpoint_and_manifest_match",
                "external_status": "configured",
            }
            self._summary(
                root / "baseline.json",
                exact_accuracy=0.90,
                character_error_rate=0.05,
                median_ms=2.0,
                tensor_bytes=1000,
                sequence_ece=0.04,
                checkpoint_sha256=hashlib.sha256(b"baseline").hexdigest(),
                **common,
            )
            self._summary(
                root / "candidate.json",
                exact_accuracy=0.91,
                character_error_rate=0.04,
                median_ms=2.1,
                tensor_bytes=1200,
                sequence_ece=0.03,
                checkpoint_sha256=hashlib.sha256(b"candidate").hexdigest(),
                **common,
            )
            for index, accuracy in enumerate((0.90, 0.92), start=1):
                (root / f"history-{index}.json").write_text(
                    json.dumps([{"exact_accuracy": accuracy}]), encoding="utf-8"
                )
            registry = root / "registry.yaml"
            registry.write_text(
                "schema_version: 1\nmodels:\n"
                "  - {id: v1, architecture: a, version: '1', lifecycle_status: production_baseline, "
                "checkpoint: baseline.pt, evaluation_summary: baseline.json, training_histories: [], notes: baseline}\n"
                "  - id: v2\n"
                "    architecture: b\n"
                "    version: '2'\n"
                "    lifecycle_status: candidate\n"
                "    checkpoint: candidate.pt\n"
                "    evaluation_summary: candidate.json\n"
                "    training_histories: [history-1.json, history-2.json]\n"
                "    notes: candidate\n",
                encoding="utf-8",
            )

            rows = build_comparison_rows(load_model_registry(registry, project_root=root))

            self.assertTrue(rows[1]["promotion_eligible"])
            self.assertEqual(rows[1]["decision"], "eligible_for_human_promotion_review")

    def test_mismatched_measured_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._summary(root / "one.json", dataset_version="one")
            self._summary(root / "two.json", dataset_version="two")
            registry = root / "registry.yaml"
            registry.write_text(
                "schema_version: 1\nmodels:\n"
                "  - {id: one, architecture: a, version: '1', lifecycle_status: candidate, "
                "evaluation_summary: one.json, training_histories: [], notes: one}\n"
                "  - {id: two, architecture: b, version: '2', lifecycle_status: candidate, "
                "evaluation_summary: two.json, training_histories: [], notes: two}\n",
                encoding="utf-8",
            )

            entries = load_model_registry(registry, project_root=root)
            with self.assertRaisesRegex(ValueError, "identical evaluation evidence"):
                build_comparison_rows(entries)


if __name__ == "__main__":
    unittest.main()
