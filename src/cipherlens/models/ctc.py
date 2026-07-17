"""Experimental Model V2 CRNN with CTC decoding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from cipherlens.models import ConvBlock

CTC_ARCHITECTURE_NAME = "captcha_crnn_ctc"
CTC_MODEL_VERSION = "2.0-experimental"
CTC_BLANK_INDEX = 0


@dataclass(frozen=True)
class CTCModelConfig:
    image_height: int = 48
    image_width: int = 176
    hidden_size: int = 128
    lstm_layers: int = 2

    def __post_init__(self) -> None:
        if min(self.image_height, self.image_width, self.hidden_size, self.lstm_layers) < 1:
            raise ValueError("CTC model dimensions and layer count must be positive.")


class CTCCodec:
    """One-based vocabulary with zero reserved for the CTC blank token."""

    def __init__(self, charset: str) -> None:
        if not charset:
            raise ValueError("The CTC character set cannot be empty.")
        if len(set(charset)) != len(charset):
            raise ValueError("The CTC character set contains duplicates.")
        self.charset = charset
        self.blank_index = CTC_BLANK_INDEX
        self.char_to_index = {
            character: index for index, character in enumerate(charset, start=1)
        }
        self.index_to_char = {index: character for character, index in self.char_to_index.items()}

    @property
    def num_classes(self) -> int:
        return len(self.charset) + 1

    def encode(self, text: str) -> Tensor:
        try:
            values = [self.char_to_index[character] for character in text]
        except KeyError as error:
            raise ValueError(f"Unknown CTC character in label: {error.args[0]!r}") from error
        return torch.tensor(values, dtype=torch.long)

    def greedy_decode(self, logits: Tensor) -> list[tuple[str, float]]:
        if logits.ndim != 3 or logits.shape[2] != self.num_classes:
            raise ValueError("CTC logits must have shape [time, batch, classes].")
        probabilities = logits.detach().softmax(dim=2)
        max_probabilities, indices = probabilities.max(dim=2)
        decoded: list[tuple[str, float]] = []
        for batch_index in range(indices.shape[1]):
            characters: list[str] = []
            confidences: list[float] = []
            previous = self.blank_index
            for time_index in range(indices.shape[0]):
                index = int(indices[time_index, batch_index])
                if index != self.blank_index and index != previous:
                    characters.append(self.index_to_char[index])
                    confidences.append(float(max_probabilities[time_index, batch_index]))
                previous = index
            confidence = (
                math.exp(
                    sum(math.log(max(value, 1e-8)) for value in confidences) / len(confidences)
                )
                if confidences
                else 0.0
            )
            decoded.append(("".join(characters), confidence))
        return decoded


class CaptchaCTCCRNN(nn.Module):
    """CNN + BiLSTM that emits a variable-length CTC time sequence."""

    def __init__(self, num_classes: int, config: CTCModelConfig | None = None) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("CTC requires a blank token and at least one character class.")
        self.config = config or CTCModelConfig()
        self.features = nn.Sequential(
            ConvBlock(3, 32, (2, 2)),
            ConvBlock(32, 64, (2, 2)),
            ConvBlock(64, 128, (2, 1)),
            ConvBlock(128, 256, (2, 1)),
        )
        self.sequence = nn.LSTM(
            input_size=256,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.lstm_layers,
            dropout=0.2 if self.config.lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Linear(self.config.hidden_size * 2, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images).mean(dim=2)
        sequence, _ = self.sequence(features.permute(2, 0, 1))
        return cast(Tensor, self.classifier(sequence))


__all__ = [
    "CTC_ARCHITECTURE_NAME",
    "CTC_BLANK_INDEX",
    "CTC_MODEL_VERSION",
    "CTCCodec",
    "CTCModelConfig",
    "CaptchaCTCCRNN",
]
