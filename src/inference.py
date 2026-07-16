"""Backward-compatible inference imports; prefer :mod:`cipherlens.inference`."""

from cipherlens.inference import (
    CaptchaRecognizer,
    CheckpointValidationError,
    Prediction,
)

__all__ = ["CaptchaRecognizer", "CheckpointValidationError", "Prediction"]
