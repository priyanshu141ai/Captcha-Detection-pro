"""Checkpoint inference and upload validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from cipherlens.config import ConfigurationError, validate_torch_threads
from cipherlens.data import prepare_image
from cipherlens.models import CaptchaCodec, CaptchaCRNN, ModelConfig


@dataclass(frozen=True)
class Prediction:
    text: str
    confidence: float


class CheckpointValidationError(RuntimeError):
    """Raised when a model checkpoint is missing, malformed, or incompatible."""


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise CheckpointValidationError(f"Model checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as error:
        raise CheckpointValidationError(
            "The model checkpoint could not be loaded safely."
        ) from error
    if not isinstance(checkpoint, dict):
        raise CheckpointValidationError("The model checkpoint must contain a dictionary.")
    required = {"model_state", "charset"}
    missing = required - checkpoint.keys()
    if missing:
        raise CheckpointValidationError(
            f"The model checkpoint is missing required fields: {', '.join(sorted(missing))}."
        )
    if not isinstance(checkpoint["charset"], str) or not checkpoint["charset"]:
        raise CheckpointValidationError("The model checkpoint character set is invalid.")
    if not isinstance(checkpoint["model_state"], dict):
        raise CheckpointValidationError("The model checkpoint state is invalid.")
    return checkpoint


class CaptchaRecognizer:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        torch_threads: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise CheckpointValidationError("CUDA inference was requested but CUDA is unavailable.")
        raw_threads: object = (
            torch_threads
            if torch_threads is not None
            else os.getenv("CIPHERLENS_TORCH_THREADS", "2")
        )
        try:
            configured_threads = validate_torch_threads(raw_threads, "CIPHERLENS_TORCH_THREADS")
        except ConfigurationError as error:
            raise CheckpointValidationError(str(error)) from error
        torch.set_num_threads(configured_threads)

        checkpoint = _load_checkpoint(Path(checkpoint_path), self.device)
        try:
            self.config = ModelConfig(**checkpoint.get("model_config", {}))
            self.codec = CaptchaCodec(checkpoint["charset"])
            self.model = CaptchaCRNN(self.codec.num_classes, self.config).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"], strict=True)
        except (TypeError, ValueError, RuntimeError) as error:
            raise CheckpointValidationError(
                "The model checkpoint is incompatible with this application version."
            ) from error
        self.model.eval()
        self.metrics = (
            checkpoint.get("metrics", {}) if isinstance(checkpoint.get("metrics"), dict) else {}
        )

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Prediction:
        tensor = prepare_image(image, self.config, augment=False).unsqueeze(0).to(self.device)
        text, confidence = self.codec.greedy_decode(self.model(tensor))[0]
        return Prediction(text=text, confidence=confidence)


class UploadValidationError(ValueError):
    """Raised when an uploaded file is unsafe or unsupported."""


@dataclass(frozen=True)
class UploadLimits:
    max_bytes: int = 10 * 1024 * 1024
    max_pixels: int = 4_000_000
    allowed_formats: tuple[str, ...] = ("PNG", "JPEG")


def load_uploaded_image(data: bytes, limits: UploadLimits | None = None) -> Image.Image:
    limits = limits or UploadLimits()
    if not data:
        raise UploadValidationError("The uploaded file is empty.")
    if len(data) > limits.max_bytes:
        raise UploadValidationError(
            f"The uploaded file exceeds the {limits.max_bytes // (1024 * 1024)} MB limit."
        )

    try:
        with Image.open(BytesIO(data)) as probe:
            image_format = (probe.format or "").upper()
            if image_format not in limits.allowed_formats:
                allowed = ", ".join(limits.allowed_formats)
                raise UploadValidationError(f"Unsupported image format. Use {allowed}.")
            width, height = probe.size
            if width < 1 or height < 1:
                raise UploadValidationError("The uploaded image has invalid dimensions.")
            if width * height > limits.max_pixels:
                raise UploadValidationError(
                    f"The image exceeds the {limits.max_pixels:,}-pixel safety limit."
                )
            probe.verify()

        with Image.open(BytesIO(data)) as image:
            image.load()
            return image.convert("RGB")
    except UploadValidationError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as error:
        raise UploadValidationError("The file is not a valid PNG or JPEG image.") from error


__all__ = [
    "CaptchaRecognizer",
    "CheckpointValidationError",
    "Prediction",
    "UploadLimits",
    "UploadValidationError",
    "load_uploaded_image",
]
