from __future__ import annotations

import unittest
from typing import cast
from unittest.mock import Mock

import requests

from cipherlens.inference import InferenceAPIClient, InferenceAPIError


def response(payload: object, *, status_code: int = 200) -> requests.Response:
    result = Mock(spec=requests.Response)
    result.ok = 200 <= status_code < 300
    result.status_code = status_code
    result.headers = {"x-request-id": "request-1"}
    result.json.return_value = payload
    return cast(requests.Response, result)


class InferenceAPIClientTests(unittest.TestCase):
    def test_validates_success_response_and_sanitizes_filename(self) -> None:
        session = Mock(spec=requests.Session)
        session.post.return_value = response(
            {
                "predicted_text": "ABC123",
                "confidence": 0.9,
                "per_character_confidence": [0.9] * 6,
                "model_version": "1.0",
                "inference_time_ms": 4.2,
                "request_id": "request-1",
            }
        )
        client = InferenceAPIClient(
            "http://127.0.0.1:8000/", session=cast(requests.Session, session)
        )

        prediction = client.predict(
            b"image", filename=r"C:\private/folder\captcha.png", content_type="image/png"
        )

        self.assertEqual(prediction.text, "ABC123")
        self.assertEqual(prediction.source, "api")
        self.assertEqual(prediction.request_id, "request-1")
        call = session.post.call_args
        self.assertEqual(call.kwargs["files"]["file"][0], "captcha.png")
        self.assertEqual(call.kwargs["timeout"], 15.0)

    def test_structured_server_error_preserves_safe_reference(self) -> None:
        session = Mock(spec=requests.Session)
        session.post.return_value = response(
            {
                "error": {
                    "code": "model_not_ready",
                    "message": "The inference model is not ready.",
                    "request_id": "request-1",
                }
            },
            status_code=503,
        )
        client = InferenceAPIClient(
            "http://127.0.0.1:8000", session=cast(requests.Session, session)
        )

        with self.assertRaises(InferenceAPIError) as caught:
            client.predict(b"image", filename="captcha.png", content_type="image/png")

        self.assertEqual(caught.exception.code, "model_not_ready")
        self.assertEqual(caught.exception.request_id, "request-1")
        self.assertTrue(caught.exception.retryable)

    def test_network_and_invalid_payload_errors_are_safe_and_retryable(self) -> None:
        session = Mock(spec=requests.Session)
        session.post.side_effect = requests.ConnectionError("private network detail")
        client = InferenceAPIClient(
            "http://127.0.0.1:8000", session=cast(requests.Session, session)
        )
        with self.assertRaises(InferenceAPIError) as unavailable:
            client.predict(b"image", filename="captcha.png", content_type="image/png")
        self.assertNotIn("private", unavailable.exception.message)
        self.assertTrue(unavailable.exception.retryable)

        session.post.side_effect = None
        session.post.return_value = response({"unexpected": True})
        with self.assertRaises(InferenceAPIError) as invalid:
            client.predict(b"image", filename="captcha.png", content_type="image/png")
        self.assertEqual(invalid.exception.status_code, 502)
        self.assertTrue(invalid.exception.retryable)


if __name__ == "__main__":
    unittest.main()
