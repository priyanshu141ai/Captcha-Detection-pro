from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from src.data import prepare_image
from src.model import CaptchaCodec, CaptchaCRNN, ModelConfig


@dataclass(frozen=True)
class Prediction:
    text: str
    confidence: float


class CaptchaRecognizer:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=False)
        self.config = ModelConfig(**checkpoint.get("model_config", {}))
        self.codec = CaptchaCodec(checkpoint["charset"])
        self.model = CaptchaCRNN(self.codec.num_classes, self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.metrics = checkpoint.get("metrics", {})

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Prediction:
        tensor = prepare_image(image, self.config, augment=False).unsqueeze(0).to(self.device)
        text, confidence = self.codec.greedy_decode(self.model(tensor))[0]
        return Prediction(text=text, confidence=confidence)

