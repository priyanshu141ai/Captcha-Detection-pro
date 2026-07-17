from __future__ import annotations

import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

from cipherlens.data import (
    CaptchaSample,
    coverage_aware_split,
    load_samples,
    observed_charset,
    prepare_image,
)
from cipherlens.inference import CaptchaRecognizer
from cipherlens.models import CaptchaCodec, CaptchaCRNN, ModelConfig, levenshtein_distance


def save_tiny_checkpoint(path: Path) -> ModelConfig:
    config = ModelConfig(image_height=16, image_width=24, hidden_size=4, lstm_layers=1)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = CaptchaCRNN(2, config)
    torch.save(
        {
            "checkpoint_version": "test-v1",
            "charset": "AB",
            "model_config": asdict(config),
            "model_state": model.state_dict(),
            "metadata": {"architecture": {"name": "test-crnn", "version": "test-v1"}},
        },
        path,
    )
    return config


class RuntimeContractTests(unittest.TestCase):
    def test_label_rows_require_existing_images_and_six_characters(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "sample.png"
            labels_path = root / "labels.txt"
            Image.new("RGB", (12, 8), "white").save(image_path)
            labels_path.write_text("sample.png ABABAB\n", encoding="utf-8")

            samples = load_samples(labels_path, root)
            self.assertEqual(
                [(sample.path, sample.label) for sample in samples], [(image_path, "ABABAB")]
            )

            invalid_rows = ("sample.png SHORT\n", "malformed-row\n", "missing.png ABABAB\n", "")
            for row in invalid_rows:
                with self.subTest(row=row):
                    labels_path.write_text(row, encoding="utf-8")
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        load_samples(labels_path, root)

    def test_codec_round_trip_and_unknown_character_validation(self) -> None:
        codec = CaptchaCodec("AB")
        encoded = codec.encode("ABBAAB")
        logits = torch.full((6, 1, codec.num_classes), -20.0)
        logits[torch.arange(6), 0, encoded] = 20.0

        self.assertEqual(codec.greedy_decode(logits)[0][0], "ABBAAB")
        with self.assertRaises(ValueError):
            codec.encode("ABC")
        self.assertEqual(levenshtein_distance("ABBAAB", "ABAAB"), 1)

    def test_split_is_deterministic_and_retains_training_vocabulary(self) -> None:
        samples = [
            CaptchaSample(Path(f"sample-{index}.png"), "ABABAB" if index % 2 else "BABABA")
            for index in range(12)
        ]

        first = coverage_aware_split(samples, validation_fraction=0.25, seed=42)
        second = coverage_aware_split(samples, validation_fraction=0.25, seed=42)

        self.assertEqual(first, second)
        training, validation = first
        self.assertEqual((len(training), len(validation)), (9, 3))
        self.assertEqual(observed_charset(training), observed_charset(samples))

    def test_preprocessing_and_model_output_are_deterministic(self) -> None:
        config = ModelConfig(image_height=16, image_width=24, hidden_size=4, lstm_layers=1)
        image = Image.new("RGB", (31, 9), (0, 127, 255))
        first = prepare_image(image, config)
        second = prepare_image(image, config)

        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (3, 16, 24))
        self.assertGreaterEqual(float(first.min()), -1.0)
        self.assertLessEqual(float(first.max()), 1.0)
        output = CaptchaCRNN(2, config).eval()(first.unsqueeze(0))
        self.assertEqual(tuple(output.shape), (6, 1, 2))

    def test_generated_checkpoint_inference_is_compatible_and_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tiny.pt"
            config = save_tiny_checkpoint(checkpoint)
            recognizer = CaptchaRecognizer(checkpoint, torch_threads=1)
            image = Image.new("RGB", (config.image_width, config.image_height), "white")

            first = recognizer.predict(image)
            second = recognizer.predict(image)

        self.assertEqual(first, second)
        self.assertEqual(len(first.text), 6)
        self.assertEqual(len(first.per_character_confidence), 6)
        self.assertEqual(recognizer.model_version, "test-v1")


if __name__ == "__main__":
    unittest.main()
