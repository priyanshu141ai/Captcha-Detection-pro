from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.data import prepare_image
from src.model import CaptchaCodec, CaptchaCRNN, ModelConfig


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
        raise CheckpointValidationError("The model checkpoint could not be loaded safely.") from error
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
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise CheckpointValidationError("CUDA inference was requested but CUDA is unavailable.")
        configured_threads = int(os.getenv("CIPHERLENS_TORCH_THREADS", "2"))
        torch.set_num_threads(max(1, configured_threads))

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
        self.metrics = checkpoint.get("metrics", {}) if isinstance(checkpoint.get("metrics"), dict) else {}

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Prediction:
        tensor = prepare_image(image, self.config, augment=False).unsqueeze(0).to(self.device)
        text, confidence = self.codec.greedy_decode(self.model(tensor))[0]
        return Prediction(text=text, confidence=confidence)
