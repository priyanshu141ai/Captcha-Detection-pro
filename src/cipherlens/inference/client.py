"""Typed HTTP client for the separate CipherLens inference service."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import requests
from pydantic import ValidationError

from cipherlens.api.schemas import ErrorResponse, PredictionResponse

_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


@dataclass(frozen=True)
class ServedPrediction:
    text: str
    confidence: float
    per_character_confidence: tuple[float, ...]
    model_version: str
    inference_time_ms: float
    request_id: str | None
    source: str


class InferenceAPIError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.request_id = request_id

    @property
    def retryable(self) -> bool:
        return self.status_code is None or self.status_code >= 500


def _request_id(value: object) -> str | None:
    return value if isinstance(value, str) and _SAFE_REQUEST_ID.fullmatch(value) else None


class InferenceAPIClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 15.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def predict(
        self,
        data: bytes,
        *,
        filename: str,
        content_type: str,
    ) -> ServedPrediction:
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                files={"file": (Path(filename.replace("\\", "/")).name, data, content_type)},
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as error:
            raise InferenceAPIError(
                "backend_unavailable", "The inference API could not be reached."
            ) from error

        header_request_id = _request_id(response.headers.get("x-request-id"))
        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError as error:
            raise InferenceAPIError(
                "invalid_backend_response",
                "The inference API returned an invalid response.",
                status_code=502,
                request_id=header_request_id,
            ) from error

        if not response.ok:
            try:
                detail = ErrorResponse.model_validate(payload).error
            except ValidationError:
                raise InferenceAPIError(
                    "backend_error",
                    "The inference API rejected the request.",
                    status_code=response.status_code,
                    request_id=header_request_id,
                ) from None
            raise InferenceAPIError(
                detail.code,
                detail.message,
                status_code=response.status_code,
                request_id=_request_id(detail.request_id) or header_request_id,
            )

        try:
            prediction = PredictionResponse.model_validate(payload)
        except ValidationError as error:
            raise InferenceAPIError(
                "invalid_backend_response",
                "The inference API returned an invalid response.",
                status_code=502,
                request_id=header_request_id,
            ) from error
        return ServedPrediction(
            text=prediction.predicted_text,
            confidence=prediction.confidence,
            per_character_confidence=tuple(prediction.per_character_confidence),
            model_version=prediction.model_version,
            inference_time_ms=prediction.inference_time_ms,
            request_id=_request_id(prediction.request_id) or header_request_id,
            source="api",
        )


__all__ = ["InferenceAPIClient", "InferenceAPIError", "ServedPrediction"]
