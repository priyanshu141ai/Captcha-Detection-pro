"""FastAPI application factory for local, authorized CipherLens inference."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol, cast

import torch
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from cipherlens.api.schemas import (
    BatchPredictionItem,
    BatchPredictionResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadyResponse,
)
from cipherlens.api.validation import APIError, validate_upload
from cipherlens.config import CipherLensSettings, load_project_settings
from cipherlens.inference import (
    CaptchaRecognizer,
    CheckpointValidationError,
    Prediction,
    UploadLimits,
)
from cipherlens.logging import configure_logging
from cipherlens.models import CaptchaCodec, ModelConfig
from cipherlens.monitoring import MetricsRegistry

ROOT = Path.cwd().resolve()
LOGGER = logging.getLogger("cipherlens.api")
_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
_KNOWN_PATHS = {
    "/health",
    "/ready",
    "/model-info",
    "/predict",
    "/predict/batch",
    "/metrics",
    "/openapi.json",
    "/docs",
    "/redoc",
}


class Recognizer(Protocol):
    architecture_name: str
    model_version: str
    checkpoint_version: str
    device: torch.device
    config: ModelConfig
    codec: CaptchaCodec
    parameter_count: int

    def predict(self, image: Image.Image) -> Prediction: ...


class RecognizerFactory(Protocol):
    def __call__(
        self,
        checkpoint_path: str | Path,
        *,
        torch_threads: int,
    ) -> Recognizer: ...


def _load_recognizer(checkpoint_path: str | Path, *, torch_threads: int) -> Recognizer:
    return CaptchaRecognizer(checkpoint_path, torch_threads=torch_threads)


@dataclass(frozen=True)
class TimedPrediction:
    prediction: Prediction
    elapsed_ms: float


class RequestBodyTooLarge(Exception):
    """Raised before multipart parsing can consume an oversized request body."""


class RequestContextMiddleware:
    """Apply request IDs, body limits, structured 500s, and HTTP counters."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        route_limits: dict[str, int],
        metrics: MetricsRegistry,
    ) -> None:
        self.app = app
        self.route_limits = route_limits
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        path = str(scope.get("path", ""))
        method = str(scope.get("method", "GET")).upper()
        limit = self.route_limits.get(path) if method == "POST" else None
        received_bytes = 0
        response_started = False
        status_code = 500

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if limit is not None and message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > limit:
                    raise RequestBodyTooLarge
            return message

        async def contextual_send(message: Message) -> None:
            nonlocal response_started, status_code
            if message["type"] == "http.response.start":
                response_started = True
                status_code = int(message["status"])
                MutableHeaders(scope=message).append("x-request-id", request_id)
            await send(message)

        try:
            raw_content_length = dict(scope.get("headers", [])).get(b"content-length")
            if limit is not None and raw_content_length is not None:
                try:
                    if int(raw_content_length) > limit:
                        raise RequestBodyTooLarge
                except ValueError:
                    pass
            await self.app(scope, limited_receive, contextual_send)
        except RequestBodyTooLarge:
            if response_started:
                raise
            status_code = 413
            await _send_error(
                send, status_code, "request_too_large", "Request body is too large.", request_id
            )
        except Exception:
            if response_started:
                raise
            status_code = 500
            LOGGER.exception(
                "Unhandled API request failure",
                extra={"event": "request_failed", "request_id": request_id},
            )
            await _send_error(
                send,
                status_code,
                "internal_error",
                "The request could not be completed.",
                request_id,
            )
        finally:
            metric_path = path if path in _KNOWN_PATHS else "unmatched"
            self.metrics.observe_http(method, metric_path, status_code)


async def _send_error(
    send: Send, status_code: int, code: str, message: str, request_id: str
) -> None:
    body = (
        ErrorResponse(error=ErrorDetail(code=code, message=message, request_id=request_id))
        .model_dump_json()
        .encode("utf-8")
    )
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"x-request-id", request_id.encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _request_id(request: Request) -> str:
    return cast(str, request.state.request_id)


def _recognizer(request: Request) -> Recognizer:
    recognizer = cast(Recognizer | None, request.app.state.recognizer)
    if recognizer is None:
        raise APIError(503, "model_not_ready", "The inference model is not ready.")
    return recognizer


def _predict_sync(recognizer: Recognizer, images: list[Image.Image]) -> list[TimedPrediction]:
    results: list[TimedPrediction] = []
    for image in images:
        started = time.perf_counter()
        prediction = recognizer.predict(image)
        results.append(TimedPrediction(prediction, (time.perf_counter() - started) * 1000))
    return results


async def _predict(request: Request, images: list[Image.Image]) -> list[TimedPrediction]:
    recognizer = _recognizer(request)
    semaphore = cast(asyncio.Semaphore, request.app.state.inference_semaphore)
    metrics = cast(MetricsRegistry, request.app.state.metrics)
    try:
        async with semaphore:
            results = await asyncio.to_thread(_predict_sync, recognizer, images)
    except Exception as error:
        metrics.observe_inference((), outcome="failure")
        LOGGER.exception(
            "Model inference failed",
            extra={"event": "inference_failed", "request_id": _request_id(request)},
        )
        raise APIError(500, "inference_failed", "Model inference failed.") from error
    metrics.observe_inference((item.elapsed_ms / 1000 for item in results), outcome="success")
    LOGGER.info(
        "Inference completed",
        extra={
            "event": "inference_completed",
            "request_id": _request_id(request),
            "model_version": recognizer.model_version,
            "sample_count": len(results),
            "inference_ms": sum(item.elapsed_ms for item in results),
        },
    )
    return results


def create_app(
    settings: CipherLensSettings | None = None,
    *,
    recognizer_factory: RecognizerFactory = _load_recognizer,
) -> FastAPI:
    configured = settings or load_project_settings(ROOT)
    configure_logging(configured.runtime.log_level, configured.runtime.log_format)
    metrics = MetricsRegistry()
    limits = UploadLimits(
        max_bytes=configured.runtime.max_upload_bytes,
        max_pixels=configured.runtime.max_upload_pixels,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.recognizer = None
        application.state.inference_semaphore = asyncio.Semaphore(
            configured.api.max_inference_concurrency
        )
        try:
            application.state.recognizer = recognizer_factory(
                configured.runtime.checkpoint_path,
                torch_threads=configured.runtime.torch_threads,
            )
            LOGGER.info(
                "Inference model loaded",
                extra={
                    "event": "model_loaded",
                    "model_version": application.state.recognizer.model_version,
                },
            )
        except CheckpointValidationError:
            metrics.record_model_load_failure()
            LOGGER.error("Inference model is unavailable", extra={"event": "model_load_failed"})
        yield
        application.state.recognizer = None

    application = FastAPI(
        title="CipherLens Inference API",
        version="1.0.0",
        description=(
            "CPU inference for synthetic, owned, or explicitly authorized CAPTCHA-style images. "
            "This service does not submit images to third-party systems."
        ),
        lifespan=lifespan,
    )
    application.state.metrics = metrics
    application.add_middleware(
        RequestContextMiddleware,
        route_limits={
            "/predict": limits.max_bytes + _MULTIPART_OVERHEAD_BYTES,
            "/predict/batch": (
                limits.max_bytes * configured.api.max_batch_size + _MULTIPART_OVERHEAD_BYTES
            ),
        },
        metrics=metrics,
    )

    @application.exception_handler(APIError)
    async def api_error_handler(request: Request, error: APIError) -> JSONResponse:
        request_id = _request_id(request)
        LOGGER.warning(
            "API request rejected",
            extra={"event": "request_rejected", "request_id": request_id},
        )
        payload = ErrorResponse(
            error=ErrorDetail(code=error.code, message=error.message, request_id=request_id)
        )
        return JSONResponse(status_code=error.status_code, content=payload.model_dump())

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        del error
        request_id = _request_id(request)
        payload = ErrorResponse(
            error=ErrorDetail(
                code="invalid_request",
                message="The request does not match the documented API schema.",
                request_id=request_id,
            )
        )
        return JSONResponse(status_code=422, content=payload.model_dump())

    @application.exception_handler(HTTPException)
    async def http_error_handler(request: Request, error: HTTPException) -> JSONResponse:
        request_id = _request_id(request)
        payload = ErrorResponse(
            error=ErrorDetail(
                code="http_error",
                message=str(error.detail),
                request_id=request_id,
            )
        )
        return JSONResponse(status_code=error.status_code, content=payload.model_dump())

    @application.get("/health", response_model=HealthResponse, tags=["service"])
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(request_id=_request_id(request))

    @application.get(
        "/ready",
        response_model=ReadyResponse,
        responses={503: {"model": ReadyResponse}},
        tags=["service"],
    )
    async def ready(request: Request) -> ReadyResponse | JSONResponse:
        request_id = _request_id(request)
        if request.app.state.recognizer is None:
            payload = ReadyResponse(status="not_ready", request_id=request_id)
            return JSONResponse(status_code=503, content=payload.model_dump())
        return ReadyResponse(status="ready", request_id=request_id)

    @application.get(
        "/model-info",
        response_model=ModelInfoResponse,
        responses={503: {"model": ErrorResponse}},
        tags=["model"],
    )
    async def model_info(request: Request) -> ModelInfoResponse:
        recognizer = _recognizer(request)
        return ModelInfoResponse(
            architecture=recognizer.architecture_name,
            model_version=recognizer.model_version,
            checkpoint_version=recognizer.checkpoint_version,
            device=str(recognizer.device),
            vocabulary_size=recognizer.codec.num_classes,
            input_width=recognizer.config.image_width,
            input_height=recognizer.config.image_height,
            sequence_length=recognizer.config.sequence_length,
            parameter_count=recognizer.parameter_count,
            request_id=_request_id(request),
        )

    @application.post(
        "/predict",
        response_model=PredictionResponse,
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["inference"],
    )
    async def predict(
        request: Request,
        file: Annotated[UploadFile, File(description="One PNG or JPEG image.")],
    ) -> PredictionResponse:
        _recognizer(request)
        validated = await validate_upload(file, limits)
        try:
            result = (await _predict(request, [validated.image]))[0]
        finally:
            validated.image.close()
        recognizer = _recognizer(request)
        return PredictionResponse(
            predicted_text=result.prediction.text,
            confidence=result.prediction.confidence,
            per_character_confidence=list(result.prediction.per_character_confidence),
            model_version=recognizer.model_version,
            inference_time_ms=result.elapsed_ms,
            request_id=_request_id(request),
        )

    @application.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["inference"],
    )
    async def predict_batch(
        request: Request,
        files: Annotated[list[UploadFile], File(description="Repeated PNG or JPEG files.")],
    ) -> BatchPredictionResponse:
        recognizer = _recognizer(request)
        if len(files) > configured.api.max_batch_size:
            for upload in files:
                await upload.close()
            raise APIError(413, "batch_too_large", "The batch contains too many files.")
        images: list[Image.Image] = []
        try:
            for upload in files:
                images.append((await validate_upload(upload, limits)).image)
            results = await _predict(request, images)
        finally:
            for upload in files:
                await upload.close()
            for image in images:
                image.close()
        predictions = [
            BatchPredictionItem(
                index=index,
                predicted_text=item.prediction.text,
                confidence=item.prediction.confidence,
                per_character_confidence=list(item.prediction.per_character_confidence),
                model_version=recognizer.model_version,
                inference_time_ms=item.elapsed_ms,
            )
            for index, item in enumerate(results)
        ]
        return BatchPredictionResponse(
            predictions=predictions,
            request_id=_request_id(request),
        )

    @application.get("/metrics", response_class=PlainTextResponse, tags=["service"])
    async def prometheus_metrics(request: Request) -> PlainTextResponse:
        content = metrics.render(ready=request.app.state.recognizer is not None)
        return PlainTextResponse(content, media_type="text/plain; version=0.0.4")

    return application


app = create_app()

__all__ = ["Recognizer", "RecognizerFactory", "app", "create_app"]
