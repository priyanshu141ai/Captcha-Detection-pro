from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from cipherlens.evaluation.ctc import evaluate_ctc_checkpoint, write_ctc_evaluation_summary
from cipherlens.models.ctc import CaptchaCTCCRNN, CTCCodec, CTCModelConfig
from cipherlens.training.ctc import build_ctc_optimization, ctc_loss
from scripts.train_ctc_experiment import main as train_ctc


class CTCModelTests(unittest.TestCase):
    def test_codec_collapses_repeats_but_preserves_blank_separated_characters(self) -> None:
        codec = CTCCodec("AB")
        logits = torch.full((5, 1, codec.num_classes), -10.0)
        for time_index, class_index in enumerate((1, 1, 0, 1, 2)):
            logits[time_index, 0, class_index] = 10.0

        prediction, confidence = codec.greedy_decode(logits)[0]

        self.assertEqual(prediction, "AAB")
        self.assertGreater(confidence, 0.99)
        self.assertEqual(codec.encode("AB").tolist(), [1, 2])

    def test_model_shape_and_ctc_loss_are_valid(self) -> None:
        codec = CTCCodec("AB")
        config = CTCModelConfig(hidden_size=4, lstm_layers=1)
        model = CaptchaCTCCRNN(codec.num_classes, config)
        images = torch.randn(2, 3, config.image_height, config.image_width)
        targets = torch.stack((codec.encode("AAAAAA"), codec.encode("ABABAB")))
        loss_fn, _optimizer, _scheduler = build_ctc_optimization(
            model,
            learning_rate=1e-3,
            weight_decay=1e-4,
            scheduler_factor=0.5,
            scheduler_patience=1,
        )

        logits = model(images)
        loss = ctc_loss(logits, targets, loss_fn)

        self.assertEqual(tuple(logits.shape), (44, 2, 3))
        self.assertTrue(torch.isfinite(loss))

    def test_safe_experimental_cli_trains_tiny_generated_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            labels = root / "labels.txt"
            rows = []
            for index, label in enumerate(("AAAAAA", "ABABAB", "BBBBBB", "BABABA")):
                filename = f"{index}.png"
                Image.new("RGB", (24, 16), (index * 50, 255, 255)).save(images / filename)
                rows.append(f"{filename} {label}")
            labels.write_text("\n".join(rows) + "\n", encoding="utf-8")
            candidate = root / "ctc-candidate.pt"
            history = root / "history.json"

            result = train_ctc(
                [
                    "--dataset",
                    str(labels),
                    str(images),
                    "--no-split-manifest",
                    "--output",
                    str(candidate),
                    "--history-output",
                    str(history),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--hidden-size",
                    "4",
                    "--lstm-layers",
                    "1",
                    "--torch-threads",
                    "1",
                    "--device",
                    "cpu",
                ]
            )

            self.assertEqual(result, 0)
            self.assertTrue(history.is_file())
            payload = torch.load(candidate, map_location="cpu", weights_only=True)
            self.assertEqual(payload["checkpoint_kind"], "experimental_ctc_candidate")
            self.assertEqual(payload["blank_index"], 0)
            self.assertEqual(payload["metadata"]["architecture"]["status"], "experimental")

            manifest = root / "manifest.csv"
            manifest_rows = []
            for index, label in enumerate(("AAAAAA", "ABABAB", "BBBBBB", "BABABA")):
                path = images / f"{index}.png"
                manifest_rows.append(
                    {
                        "dataset_version": "fixture-v1",
                        "source": "fixture",
                        "path": path.relative_to(root).as_posix(),
                        "label": label,
                        "split": "validation",
                        "valid": "True",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
                writer.writeheader()
                writer.writerows(manifest_rows)
            summary = evaluate_ctc_checkpoint(
                candidate,
                manifest,
                project_root=root,
                batch_size=2,
                torch_threads=1,
                ece_bins=5,
                latency_warmup=0,
                latency_runs=1,
            )
            summary_path = root / "ctc-summary.json"
            write_ctc_evaluation_summary(summary, summary_path)
            self.assertEqual(summary["schema_version"], "1.0")
            self.assertEqual(summary["metrics"]["sample_count"], 4)
            self.assertTrue(summary_path.is_file())

    def test_experimental_cli_refuses_production_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot overwrite"):
            train_ctc(["--output", "models/captcha_crnn.pt", "--epochs", "1"])


if __name__ == "__main__":
    unittest.main()
