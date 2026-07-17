from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from cipherlens.data import CaptchaSample
from cipherlens.inference import CaptchaRecognizer
from cipherlens.models import CaptchaCodec, CaptchaCRNN, ModelConfig
from cipherlens.training import (
    EarlyStopping,
    TrainingSplit,
    build_class_weights,
    build_optimization,
    build_run_metadata,
    create_tracker,
    ensure_safe_artifact_paths,
    load_resume_checkpoint,
    load_training_split,
    save_candidate_checkpoint,
    save_resume_checkpoint,
)
from cipherlens.utils import make_torch_generator


class TrainingDataTests(unittest.TestCase):
    def test_versioned_manifest_controls_split_and_rejects_stale_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            samples: list[CaptchaSample] = []
            rows: list[dict[str, str]] = []
            for index, split in enumerate(("train", "train", "validation")):
                path = images / f"{index}.png"
                path.write_bytes(f"image-{index}".encode())
                label = ("A" if index < 2 else "B") * 6
                samples.append(CaptchaSample(path, label))
                rows.append(
                    {
                        "dataset_version": "dataset-v1",
                        "path": path.relative_to(root).as_posix(),
                        "label": label,
                        "split": split,
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
                json.dumps({"dataset_version": "dataset-v1", "split_version": "split-v1"}),
                encoding="utf-8",
            )

            result = load_training_split(
                samples,
                project_root=root,
                manifest_path=manifest,
                dataset_report_path=report,
                validation_fraction=0.2,
                seed=42,
            )

            self.assertEqual(len(result.training), 2)
            self.assertEqual(len(result.validation), 1)
            self.assertEqual(result.dataset_version, "dataset-v1")
            self.assertEqual(result.split_version, "split-v1")
            rows[0]["split"] = "external_test"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "cannot consume"):
                load_training_split(
                    samples,
                    project_root=root,
                    manifest_path=manifest,
                    dataset_report_path=report,
                    validation_fraction=0.2,
                    seed=42,
                )
            rows[0]["split"] = "train"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            samples[0].path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                load_training_split(
                    samples,
                    project_root=root,
                    manifest_path=manifest,
                    dataset_report_path=report,
                    validation_fraction=0.2,
                    seed=42,
                )

    def test_class_weights_favor_rare_characters_and_remain_normalized(self) -> None:
        codec = CaptchaCodec("AB")
        weights = build_class_weights(
            (
                CaptchaSample(Path("one.png"), "AAAAAA"),
                CaptchaSample(Path("two.png"), "AAAAAB"),
            ),
            codec,
        )
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertGreater(float(weights[1]), float(weights[0]))


class TrainingCheckpointTests(unittest.TestCase):
    def test_candidate_metadata_and_resume_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.pt"
            resume_path = root / "resume.pt"
            codec = CaptchaCodec("AB")
            config = ModelConfig()
            model = CaptchaCRNN(codec.num_classes, config)
            split = TrainingSplit((), (), "dataset-v1", "split-v1", "selection-v1", None)
            metadata = build_run_metadata(
                project_root=root,
                model_config=config,
                split=split,
                run_config={"seed": 42},
                dataset_sources=[],
                device=torch.device("cpu"),
            )
            components = build_optimization(
                model,
                torch.ones(codec.num_classes),
                device=torch.device("cpu"),
                learning_rate=1e-3,
                weight_decay=1e-4,
                scheduler_factor=0.5,
                scheduler_patience=1,
            )
            stopping = EarlyStopping(2)
            metrics = {"loss": 1.0, "character_accuracy": 0.5, "exact_accuracy": 0.25}
            self.assertTrue(stopping.update(1, metrics))
            best_state = {
                name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
            }
            save_candidate_checkpoint(
                candidate,
                model_state=best_state,
                codec=codec,
                model_config=config,
                metrics=metrics,
                epoch=1,
                metadata=metadata,
            )
            generator = make_torch_generator(42)
            history: list[dict[str, float | int]] = [{"epoch": 1, "loss": 1.0}]
            save_resume_checkpoint(
                resume_path,
                model=model,
                best_model_state=best_state,
                codec=codec,
                model_config=config,
                optimizer=components.optimizer,
                scheduler=components.scheduler,
                early_stopping=stopping,
                epoch=1,
                history=history,
                metadata=metadata,
                generator=generator,
            )

            candidate_payload = torch.load(candidate, map_location="cpu", weights_only=True)
            self.assertEqual(candidate_payload["checkpoint_version"], 2)
            self.assertEqual(candidate_payload["metadata"]["dataset"]["version"], "dataset-v1")
            self.assertEqual(candidate_payload["metadata"]["architecture"]["version"], "1.0")
            self.assertEqual(candidate_payload["metadata"]["preprocessing"]["version"], "1.0")
            self.assertEqual(candidate_payload["metadata"]["validation_metrics"], metrics)
            self.assertEqual(CaptchaRecognizer(candidate).codec.charset, "AB")

            restored_model = CaptchaCRNN(codec.num_classes, config)
            restored_components = build_optimization(
                restored_model,
                torch.ones(codec.num_classes),
                device=torch.device("cpu"),
                learning_rate=1e-3,
                weight_decay=1e-4,
                scheduler_factor=0.5,
                scheduler_patience=1,
            )
            restored_stopping = EarlyStopping(2)
            restored = load_resume_checkpoint(
                resume_path,
                model=restored_model,
                codec=codec,
                model_config=config,
                optimizer=restored_components.optimizer,
                scheduler=restored_components.scheduler,
                early_stopping=restored_stopping,
                expected_dataset_version="dataset-v1",
                generator=make_torch_generator(99),
                device=torch.device("cpu"),
            )
            self.assertEqual(restored.epoch, 1)
            self.assertEqual(restored.history, history)
            self.assertEqual(restored_stopping.best_epoch, 1)
            self.assertTrue(
                all(
                    torch.equal(value, restored_model.state_dict()[name])
                    for name, value in model.state_dict().items()
                )
            )

    def test_artifact_safety_and_disabled_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            approved = root / "production.pt"
            with self.assertRaisesRegex(ValueError, "approved checkpoint"):
                ensure_safe_artifact_paths(approved, root / "resume.pt", approved)
            with self.assertRaisesRegex(ValueError, "must be different"):
                ensure_safe_artifact_paths(root / "same.pt", root / "same.pt", approved)

        with patch(
            "cipherlens.training.tracking.importlib.import_module",
            side_effect=AssertionError("disabled tracking imported MLflow"),
        ):
            tracker = create_tracker(False)
            tracker.log_parameters({"seed": 42})
            tracker.close()
        with (
            patch(
                "cipherlens.training.tracking.importlib.import_module",
                side_effect=ImportError,
            ),
            self.assertRaisesRegex(RuntimeError, "not installed"),
        ):
            create_tracker(True)

    def test_early_stopping_state_is_resumable(self) -> None:
        stopping = EarlyStopping(2)
        metrics = {"loss": 1.0, "character_accuracy": 0.5, "exact_accuracy": 0.25}
        self.assertTrue(stopping.update(1, metrics))
        self.assertFalse(stopping.update(2, metrics))
        self.assertFalse(stopping.update(3, metrics))
        self.assertTrue(stopping.should_stop)
        restored = EarlyStopping(99)
        restored.load_state_dict(stopping.state_dict())
        self.assertEqual(restored.state_dict(), stopping.state_dict())


if __name__ == "__main__":
    unittest.main()
