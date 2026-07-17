"""Pydantic contracts for the CipherLens inference API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: Literal["cipherlens-api"] = "cipherlens-api"
    request_id: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    request_id: str


class ModelInfoResponse(BaseModel):
    architecture: str
    model_version: str
    checkpoint_version: str
    device: str
    vocabulary_size: int = Field(ge=1)
    input_width: int = Field(ge=1)
    input_height: int = Field(ge=1)
    sequence_length: int = Field(ge=1)
    parameter_count: int = Field(ge=1)
    request_id: str


class PredictionPayload(BaseModel):
    predicted_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    per_character_confidence: list[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(min_length=1)
    model_version: str
    inference_time_ms: float = Field(ge=0.0)


class PredictionResponse(PredictionPayload):
    request_id: str


class BatchPredictionItem(PredictionPayload):
    index: int = Field(ge=0)


class BatchPredictionResponse(BaseModel):
    predictions: list[BatchPredictionItem]
    request_id: str


__all__ = [
    "BatchPredictionItem",
    "BatchPredictionResponse",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "PredictionPayload",
    "PredictionResponse",
    "ReadyResponse",
]
