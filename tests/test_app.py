from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from cipherlens.inference import InferenceAPIError, Prediction, ServedPrediction


def generated_image() -> bytes:
    output = BytesIO()
    Image.new("RGB", (151, 41), "white").save(output, format="PNG")
    return output.getvalue()


IMAGE = generated_image()


class StreamlitSmokeTests(unittest.TestCase):
    def test_initial_screen_renders_without_exception(self) -> None:
        app = AppTest.from_file("app.py", default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertEqual([button.label for button in app.button], ["Recognize text"])
        self.assertTrue(app.button[0].disabled)

    def test_invalid_environment_configuration_is_reported_without_traceback(self) -> None:
        with patch.dict("os.environ", {"CIPHERLENS_TORCH_THREADS": "invalid"}):
            app = AppTest.from_file("app.py", default_timeout=20).run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.error), 1)
        self.assertIn("configuration is invalid", app.error[0].value)

    def test_api_prediction_displays_model_latency_and_serving_source(self) -> None:
        prediction = ServedPrediction(
            "ABC123", 0.9, (0.9,) * 6, "1.0-test", 4.2, "request-1", "api"
        )
        with patch(
            "cipherlens.inference.client.InferenceAPIClient.predict", return_value=prediction
        ):
            app = AppTest.from_file("app.py", default_timeout=20).run()
            app.file_uploader[0].upload("captcha.png", IMAGE, "image/png").run()
            app.button[0].click().run()

        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.prediction.text, "ABC123")
        result_markup = "\n".join(item.value for item in app.markdown)
        self.assertIn("1.0-test", result_markup)
        self.assertIn("4.20 ms", result_markup)
        self.assertIn("FastAPI", result_markup)

    def test_retryable_api_failure_uses_local_fallback(self) -> None:
        failure = InferenceAPIError("backend_unavailable", "API unavailable")
        recognizer = Mock()
        recognizer.model_version = "1.0-test"
        recognizer.predict.return_value = Prediction("ABC123", 0.9, (0.9,) * 6)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.touch()
            with (
                patch.dict("os.environ", {"CIPHERLENS_CHECKPOINT": str(checkpoint)}),
                patch(
                    "cipherlens.inference.client.InferenceAPIClient.predict", side_effect=failure
                ),
                patch("cipherlens.inference.CaptchaRecognizer", return_value=recognizer),
            ):
                app = AppTest.from_file("app.py", default_timeout=20).run()
                app.file_uploader[0].upload("captcha.png", IMAGE, "image/png").run()
                app.button[0].click().run()

        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.prediction.source, "local")
        self.assertEqual(app.session_state.prediction.text, "ABC123")
        self.assertIn("local model was used", app.info[0].value)

    def test_non_retryable_api_error_is_shown_without_fallback(self) -> None:
        failure = InferenceAPIError(
            "unsupported_mime_type",
            "The uploaded MIME type is unsupported.",
            status_code=415,
            request_id="request-1",
        )
        with patch("cipherlens.inference.client.InferenceAPIClient.predict", side_effect=failure):
            app = AppTest.from_file("app.py", default_timeout=20).run()
            app.file_uploader[0].upload("captcha.png", IMAGE, "image/png").run()
            app.button[0].click().run()

        self.assertFalse(app.exception)
        self.assertIsNone(app.session_state.prediction)
        self.assertIn("unsupported", app.error[0].value)
        self.assertIn("request-1", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
