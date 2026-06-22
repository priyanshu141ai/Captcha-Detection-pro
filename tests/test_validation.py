from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

from src.inference import CaptchaRecognizer, CheckpointValidationError
from src.validation import UploadLimits, UploadValidationError, load_uploaded_image


def encoded_image(image_format: str = "PNG", size: tuple[int, int] = (151, 41)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format=image_format)
    return output.getvalue()


class UploadValidationTests(unittest.TestCase):
    def test_accepts_valid_png_and_jpeg(self) -> None:
        for image_format in ("PNG", "JPEG"):
            with self.subTest(image_format=image_format):
                image = load_uploaded_image(encoded_image(image_format))
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (151, 41))

    def test_rejects_empty_invalid_and_unsupported_files(self) -> None:
        cases = (b"", b"not-an-image", encoded_image("GIF"))
        for data in cases:
            with self.subTest(size=len(data)):
                with self.assertRaises(UploadValidationError):
                    load_uploaded_image(data)

    def test_enforces_byte_and_pixel_limits(self) -> None:
        with self.assertRaises(UploadValidationError):
            load_uploaded_image(b"1234", UploadLimits(max_bytes=3))
        with self.assertRaises(UploadValidationError):
            load_uploaded_image(
                encoded_image(size=(11, 10)), UploadLimits(max_pixels=100)
            )


class CheckpointValidationTests(unittest.TestCase):
    def test_rejects_missing_checkpoint(self) -> None:
        with self.assertRaises(CheckpointValidationError):
            CaptchaRecognizer("models/does-not-exist.pt")

    def test_rejects_checkpoint_missing_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "invalid.pt"
            torch.save({"charset": "ABC"}, checkpoint)
            with self.assertRaises(CheckpointValidationError):
                CaptchaRecognizer(checkpoint)


if __name__ == "__main__":
    unittest.main()
