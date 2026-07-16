from __future__ import annotations

import unittest

from cipherlens.data import prepare_image
from cipherlens.inference import CaptchaRecognizer, load_uploaded_image
from cipherlens.models import CaptchaCRNN, ModelConfig
from src.data import prepare_image as legacy_prepare_image
from src.inference import CaptchaRecognizer as LegacyCaptchaRecognizer
from src.model import CaptchaCRNN as LegacyCaptchaCRNN
from src.model import ModelConfig as LegacyModelConfig
from src.validation import load_uploaded_image as legacy_load_uploaded_image


class ImportCompatibilityTests(unittest.TestCase):
    def test_legacy_imports_resolve_to_new_package_objects(self) -> None:
        self.assertIs(legacy_prepare_image, prepare_image)
        self.assertIs(LegacyCaptchaRecognizer, CaptchaRecognizer)
        self.assertIs(LegacyCaptchaCRNN, CaptchaCRNN)
        self.assertIs(LegacyModelConfig, ModelConfig)
        self.assertIs(legacy_load_uploaded_image, load_uploaded_image)


if __name__ == "__main__":
    unittest.main()
