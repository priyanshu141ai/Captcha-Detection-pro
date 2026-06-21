from __future__ import annotations

import unittest
from pathlib import Path

import torch
from PIL import Image

from src.data import coverage_aware_split, load_samples, observed_charset, prepare_image
from src.inference import CaptchaRecognizer
from src.model import CaptchaCodec, CaptchaCRNN, ModelConfig, levenshtein_distance


ROOT = Path(__file__).resolve().parents[1]


class CipherLensCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = load_samples(ROOT / "labels.txt", ROOT / "data" / "batch_0")
        cls.charset = observed_charset(cls.samples)

    def test_dataset_contract_and_split_coverage(self) -> None:
        training, validation = coverage_aware_split(self.samples, seed=42)
        self.assertEqual(len(self.samples), 500)
        self.assertEqual(len(training), 400)
        self.assertEqual(len(validation), 100)
        self.assertEqual({len(sample.label) for sample in self.samples}, {6})
        self.assertTrue(set(self.charset) <= set("".join(sample.label for sample in training)))

    def test_model_output_has_six_character_positions(self) -> None:
        config = ModelConfig()
        codec = CaptchaCodec(self.charset)
        with Image.open(self.samples[0].path) as image:
            tensor = prepare_image(image, config)
        output = CaptchaCRNN(codec.num_classes, config)(tensor.unsqueeze(0))
        self.assertEqual(tuple(output.shape), (6, 1, len(self.charset)))

    def test_codec_preserves_repeated_characters(self) -> None:
        codec = CaptchaCodec("AB")
        logits = torch.full((6, 1, 2), -8.0)
        logits[:, 0, codec.char_to_index["A"]] = 8.0
        text, confidence = codec.greedy_decode(logits)[0]
        self.assertEqual(text, "AAAAAA")
        self.assertGreater(confidence, 0.99)

    def test_levenshtein_distance(self) -> None:
        self.assertEqual(levenshtein_distance("TAGbCN", "TAGbCN"), 0)
        self.assertEqual(levenshtein_distance("TAGbCN", "TAGCN"), 1)

    def test_checkpoint_inference(self) -> None:
        checkpoint = ROOT / "models" / "captcha_crnn.pt"
        if not checkpoint.is_file():
            self.skipTest("Trained checkpoint is not available.")
        recognizer = CaptchaRecognizer(checkpoint)
        with Image.open(self.samples[0].path) as image:
            prediction = recognizer.predict(image)
        self.assertEqual(prediction.text, self.samples[0].label)
        self.assertGreater(prediction.confidence, 0.8)


if __name__ == "__main__":
    unittest.main()
