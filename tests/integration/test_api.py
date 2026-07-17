from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from PIL import Image

from cipherlens.api import create_app
from cipherlens.config import CipherLensSettings, load_project_settings
from cipherlens.inference import Prediction
from cipherlens.models import CaptchaCodec, ModelConfig

ROOT = Path(__file__).resolve().parents[2]


def encoded_image(image_format: str = "PNG", size: tuple[int, int] = (151, 41)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format=image_format)
    return output.getvalue()


class StubRecognizer:
    architecture_name = "captcha_crnn_positionwise"
    model_version = "1.0-test"
    checkpoint_version = "test-checkpoint"
    device = torch.device("cpu")
    config = ModelConfig()
    codec = CaptchaCodec("ABC123")
    parameter_count = 123

    def predict(self, image: Image.Image) -> Prediction:
        del image
        return Prediction("ABC123", 0.9, (0.91, 0.92, 0.93, 0.94, 0.95, 0.96))


class StubFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, checkpoint_path: str | Path, *, torch_threads: int) -> StubRecognizer:
        del checkpoint_path, torch_threads
        self.calls += 1
        return StubRecognizer()


class FailingRecognizer(StubRecognizer):
    def predict(self, image: Image.Image) -> Prediction:
        del image
        raise RuntimeError("private failure detail")


class FailingFactory(StubFactory):
    def __call__(self, checkpoint_path: str | Path, *, torch_threads: int) -> StubRecognizer:
        del checkpoint_path, torch_threads
        self.calls += 1
        return FailingRecognizer()


def api_settings(
    *,
    max_bytes: int = 10 * 1024 * 1024,
    max_pixels: int = 4_000_000,
    max_batch_size: int = 8,
) -> CipherLensSettings:
    settings = load_project_settings(ROOT, environ={})
    return replace(
        settings,
        runtime=replace(
            settings.runtime,
            max_upload_bytes=max_bytes,
            max_upload_pixels=max_pixels,
        ),
        api=replace(settings.api, max_batch_size=max_batch_size),
    )


class FastAPIIntegrationTests(unittest.TestCase):
    def test_health_readiness_model_info_openapi_and_single_startup_load(self) -> None:
        factory = StubFactory()
        application = create_app(api_settings(), recognizer_factory=factory)
        with TestClient(application) as client:
            health = client.get("/health")
            ready = client.get("/ready")
            model = client.get("/model-info")
            schema = client.get("/openapi.json")
            client.get("/health")

        self.assertEqual(factory.calls, 1)
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(health.headers["x-request-id"], health.json()["request_id"])
        self.assertEqual(ready.json()["status"], "ready")
        self.assertEqual(model.json()["model_version"], "1.0-test")
        self.assertEqual(model.json()["parameter_count"], 123)
        self.assertTrue(
            {"/health", "/ready", "/model-info", "/predict", "/predict/batch", "/metrics"}
            <= set(schema.json()["paths"])
        )

    def test_single_and_batch_prediction_contracts_and_metrics(self) -> None:
        application = create_app(api_settings(), recognizer_factory=StubFactory())
        image = encoded_image()
        with TestClient(application) as client:
            single = client.post("/predict", files={"file": ("captcha.png", image, "image/png")})
            batch = client.post(
                "/predict/batch",
                files=[
                    ("files", ("one.png", image, "image/png")),
                    ("files", ("two.png", image, "image/png")),
                ],
            )
            metrics = client.get("/metrics")

        self.assertEqual(single.status_code, 200)
        payload = single.json()
        self.assertEqual(payload["predicted_text"], "ABC123")
        self.assertEqual(payload["confidence"], 0.9)
        self.assertEqual(len(payload["per_character_confidence"]), 6)
        self.assertEqual(payload["model_version"], "1.0-test")
        self.assertGreaterEqual(payload["inference_time_ms"], 0)
        self.assertEqual(len(batch.json()["predictions"]), 2)
        self.assertIn("cipherlens_inference_images_total 3", metrics.text)
        self.assertIn('outcome="success"} 2', metrics.text)

    def test_rejects_corrupt_unsupported_and_mismatched_uploads(self) -> None:
        application = create_app(api_settings(), recognizer_factory=StubFactory())
        image = encoded_image()
        cases = (
            (("captcha.png", b"not-an-image", "image/png"), 400, "invalid_image"),
            (("captcha.png", image, "image/gif"), 415, "unsupported_mime_type"),
            (("captcha.gif", image, "image/gif"), 415, "unsupported_extension"),
            (("captcha.jpg", image, "image/jpeg"), 415, "image_type_mismatch"),
        )
        with TestClient(application) as client:
            for upload, status, code in cases:
                with self.subTest(code=code):
                    response = client.post("/predict", files={"file": upload})
                    self.assertEqual(response.status_code, status)
                    self.assertEqual(response.json()["error"]["code"], code)
                    self.assertEqual(
                        response.headers["x-request-id"], response.json()["error"]["request_id"]
                    )

    def test_rejects_byte_pixel_batch_and_request_body_limits(self) -> None:
        image = encoded_image()
        cases = (
            (
                api_settings(max_bytes=32),
                "/predict",
                {"files": {"file": ("captcha.png", image, "image/png")}},
                "file_too_large",
            ),
            (
                api_settings(max_pixels=100),
                "/predict",
                {"files": {"file": ("captcha.png", encoded_image(size=(11, 10)), "image/png")}},
                "image_too_large",
            ),
            (
                api_settings(max_batch_size=1),
                "/predict/batch",
                {
                    "files": [
                        ("files", ("one.png", image, "image/png")),
                        ("files", ("two.png", image, "image/png")),
                    ]
                },
                "batch_too_large",
            ),
        )
        for settings, path, kwargs, code in cases:
            with (
                self.subTest(code=code),
                TestClient(create_app(settings, recognizer_factory=StubFactory())) as client,
            ):
                response = client.post(path, **kwargs)
                self.assertEqual(response.status_code, 413)
                self.assertEqual(response.json()["error"]["code"], code)

        tiny = api_settings(max_bytes=1)
        with TestClient(create_app(tiny, recognizer_factory=StubFactory())) as client:
            response = client.post(
                "/predict",
                content=b"x" * (1024 * 1024 + 2),
                headers={"content-type": "application/octet-stream"},
            )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "request_too_large")

    def test_missing_file_uses_structured_validation_error(self) -> None:
        with TestClient(create_app(api_settings(), recognizer_factory=StubFactory())) as client:
            response = client.post("/predict")

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "invalid_request")

    def test_inference_failure_is_structured_without_private_detail(self) -> None:
        image = encoded_image()
        with TestClient(create_app(api_settings(), recognizer_factory=FailingFactory())) as client:
            response = client.post("/predict", files={"file": ("captcha.png", image, "image/png")})
            metrics = client.get("/metrics")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "inference_failed")
        self.assertNotIn("private", response.text)
        self.assertIn('outcome="failure"} 1', metrics.text)

    def test_missing_checkpoint_keeps_health_up_and_readiness_down(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = api_settings()
            settings = replace(
                settings,
                runtime=replace(settings.runtime, checkpoint_path=Path(directory) / "missing.pt"),
            )
            application = create_app(settings)
            with TestClient(application) as client:
                health = client.get("/health")
                ready = client.get("/ready")
                model = client.get("/model-info")
                metrics = client.get("/metrics")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(ready.status_code, 503)
        self.assertEqual(ready.json()["status"], "not_ready")
        self.assertEqual(model.status_code, 503)
        self.assertEqual(model.json()["error"]["code"], "model_not_ready")
        self.assertIn("cipherlens_model_ready 0", metrics.text)
        self.assertIn("cipherlens_model_load_failures_total 1", metrics.text)


if __name__ == "__main__":
    unittest.main()
