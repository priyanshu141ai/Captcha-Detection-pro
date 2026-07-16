from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.data import CaptchaDataset, CaptchaSample
from src.model import CaptchaCodec, CaptchaCRNN, ModelConfig
from train import parse_args, warm_start_model


class TrainingPipelineTests(unittest.TestCase):
    def test_yaml_defaults_and_cli_overrides_are_applied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "training.yaml"
            config.write_text(
                "training:\n  epochs: 3\n  batch_size: 8\n  seed: 7\n", encoding="utf-8"
            )

            configured = parse_args(["--config", str(config)])
            overridden = parse_args(["--config", str(config), "--epochs", "5"])

        self.assertEqual(configured.epochs, 3)
        self.assertEqual(configured.batch_size, 8)
        self.assertEqual(configured.seed, 7)
        self.assertEqual(overridden.epochs, 5)

    def test_warm_start_preserves_shared_classifier_rows(self) -> None:
        config = ModelConfig()
        old_codec = CaptchaCodec("AB")
        old_model = CaptchaCRNN(old_codec.num_classes, config)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "warm-start.pt"
            torch.save(
                {
                    "model_state": old_model.state_dict(),
                    "charset": old_codec.charset,
                    "model_config": {
                        "image_height": config.image_height,
                        "image_width": config.image_width,
                        "hidden_size": config.hidden_size,
                        "lstm_layers": config.lstm_layers,
                        "sequence_length": config.sequence_length,
                    },
                },
                checkpoint,
            )

            new_codec = CaptchaCodec("ABC")
            new_model = CaptchaCRNN(new_codec.num_classes, config)
            shared = warm_start_model(new_model, new_codec, checkpoint)

        self.assertEqual(shared, 2)
        self.assertTrue(
            torch.equal(
                new_model.classifier.weight[new_codec.char_to_index["A"]],
                old_model.classifier.weight[old_codec.char_to_index["A"]],
            )
        )

    def test_cached_dataset_survives_source_removal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "captcha.png"
            Image.new("RGB", (151, 41), "white").save(image_path)
            dataset = CaptchaDataset(
                [CaptchaSample(image_path, "AAAAAA")],
                CaptchaCodec("A"),
                ModelConfig(),
                augment=False,
                cache_images=True,
            )
            image_path.unlink()
            tensor, target, label = dataset[0]

        self.assertEqual(tuple(tensor.shape), (3, 48, 176))
        self.assertEqual(tuple(target.shape), (6,))
        self.assertEqual(label, "AAAAAA")


if __name__ == "__main__":
    unittest.main()
