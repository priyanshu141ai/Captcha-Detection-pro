from __future__ import annotations

import unittest
from pathlib import Path

from PIL import Image

from cipherlens.data import load_samples
from cipherlens.inference import CaptchaRecognizer

ROOT = Path(__file__).resolve().parents[2]
BATCH_0_LABELS = ROOT / "labels.txt"
BATCH_0_IMAGES = ROOT / "data" / "batch_0"
BATCH_1_LABELS = ROOT / "requirements2.txt"
BATCH_1_IMAGES = ROOT / "data" / "batch_1"
CHECKPOINT = ROOT / "models" / "captcha_crnn.pt"
ARTIFACTS_AVAILABLE = all(
    path.exists()
    for path in (BATCH_0_LABELS, BATCH_0_IMAGES, BATCH_1_LABELS, BATCH_1_IMAGES, CHECKPOINT)
)


@unittest.skipUnless(ARTIFACTS_AVAILABLE, "Approved compatibility artifacts are unavailable.")
class ProductionArtifactCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.batch_0 = load_samples(BATCH_0_LABELS, BATCH_0_IMAGES)
        cls.batch_1 = load_samples(BATCH_1_LABELS, BATCH_1_IMAGES)

    def test_supplied_dataset_contract(self) -> None:
        self.assertEqual((len(self.batch_0), len(self.batch_1)), (500, 500))
        self.assertEqual({len(sample.label) for sample in self.batch_0 + self.batch_1}, {6})
        self.assertFalse(
            {sample.path for sample in self.batch_0} & {sample.path for sample in self.batch_1}
        )

    def test_approved_checkpoint_known_predictions(self) -> None:
        recognizer = CaptchaRecognizer(CHECKPOINT)
        for sample, minimum_confidence in ((self.batch_0[0], 0.8), (self.batch_1[0], 0.7)):
            with self.subTest(image=sample.path.name), Image.open(sample.path) as image:
                prediction = recognizer.predict(image)
                self.assertEqual(prediction.text, sample.label)
                self.assertGreater(prediction.confidence, minimum_confidence)
                self.assertEqual(len(prediction.per_character_confidence), 6)


if __name__ == "__main__":
    unittest.main()
