"""Backward-compatible inference imports; prefer :mod:`cipherlens.inference`."""

from cipherlens.inference import (
    CaptchaRecognizer,
    CheckpointValidationError,
    InferenceAPIClient,
    InferenceAPIError,
    Prediction,
    ServedPrediction,
)

__all__ = [
    "CaptchaRecognizer",
    "CheckpointValidationError",
    "InferenceAPIClient",
    "InferenceAPIError",
    "Prediction",
    "ServedPrediction",
]
