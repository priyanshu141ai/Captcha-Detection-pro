"""Model V1 architecture and text codec."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    image_height: int = 48
    image_width: int = 176
    hidden_size: int = 128
    lstm_layers: int = 2
    sequence_length: int = 6


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, pool: tuple[int, int]) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )


class CaptchaCRNN(nn.Module):
    """Small CNN + BiLSTM with a shared classifier across six fixed positions."""

    def __init__(self, num_classes: int, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
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
        features = self.features(images)
        features = features.mean(dim=2)
        features = F.adaptive_avg_pool1d(features, self.config.sequence_length)
        sequence = features.permute(2, 0, 1)
        sequence, _ = self.sequence(sequence)
        return cast(Tensor, self.classifier(sequence))


class CaptchaCodec:
    def __init__(self, charset: str) -> None:
        if not charset:
            raise ValueError("The character set cannot be empty.")
        if len(set(charset)) != len(charset):
            raise ValueError("The character set contains duplicates.")
        self.charset = charset
        self.char_to_index = {char: index for index, char in enumerate(charset)}
        self.index_to_char = {index: char for char, index in self.char_to_index.items()}

    @property
    def num_classes(self) -> int:
        return len(self.charset)

    def encode(self, text: str) -> Tensor:
        try:
            values = [self.char_to_index[char] for char in text]
        except KeyError as error:
            raise ValueError(f"Unknown character in label: {error.args[0]!r}") from error
        return torch.tensor(values, dtype=torch.long)

    def greedy_decode(self, logits: Tensor) -> list[tuple[str, float]]:
        probabilities = logits.detach().softmax(dim=2)
        max_probabilities, indices = probabilities.max(dim=2)
        decoded: list[tuple[str, float]] = []

        for batch_index in range(indices.shape[1]):
            characters: list[str] = []
            confidences: list[float] = []
            for time_index in range(indices.shape[0]):
                index = int(indices[time_index, batch_index])
                characters.append(self.index_to_char[index])
                confidences.append(float(max_probabilities[time_index, batch_index]))

            if confidences:
                confidence = math.exp(
                    sum(math.log(max(value, 1e-8)) for value in confidences) / len(confidences)
                )
            else:
                confidence = 0.0
            decoded.append(("".join(characters), confidence))
        return decoded


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


__all__ = [
    "CaptchaCRNN",
    "CaptchaCodec",
    "ConvBlock",
    "ModelConfig",
    "levenshtein_distance",
]
