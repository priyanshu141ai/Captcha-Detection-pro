"""Dataset loading, splitting, and shared image preprocessing."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import Tensor
from torch.utils.data import Dataset

from cipherlens.data.audit import (
    AuditedSample,
    AuditIssue,
    DatasetAuditResult,
    DuplicateFinding,
    ManifestEntry,
    audit_dataset,
    write_dataset_audit,
)

PREPROCESSING_VERSION = "1.0"
NORMALIZATION_MEAN = 0.5
NORMALIZATION_STD = 0.5


@dataclass(frozen=True)
class CaptchaSample:
    path: Path
    label: str


class ImagePreprocessingConfig(Protocol):
    @property
    def image_height(self) -> int: ...

    @property
    def image_width(self) -> int: ...


class TextEncoder(Protocol):
    def encode(self, text: str) -> Tensor: ...


def load_samples(labels_path: Path, images_dir: Path) -> list[CaptchaSample]:
    samples: list[CaptchaSample] = []
    for line_number, raw_line in enumerate(
        labels_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid label row at line {line_number}: {raw_line!r}")
        filename, label = parts
        if len(label) != 6:
            raise ValueError(f"Invalid label length at line {line_number}: expected 6 characters.")
        image_path = images_dir / filename
        if not image_path.is_file():
            raise FileNotFoundError(f"Image listed in labels file does not exist: {image_path}")
        samples.append(CaptchaSample(image_path, label))
    if not samples:
        raise ValueError("No labeled images were found.")
    return samples


def observed_charset(samples: list[CaptchaSample]) -> str:
    return "".join(sorted({character for sample in samples for character in sample.label}))


def coverage_aware_split(
    samples: list[CaptchaSample], validation_fraction: float = 0.2, seed: int = 42
) -> tuple[list[CaptchaSample], list[CaptchaSample]]:
    """Keep at least one occurrence of every observed character in training."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    remaining = Counter(character for sample in shuffled for character in sample.label)
    validation: list[CaptchaSample] = []
    target_size = max(1, round(len(shuffled) * validation_fraction))

    for sample in shuffled:
        sample_counts = Counter(sample.label)
        if len(validation) < target_size and all(
            remaining[character] - count >= 1 for character, count in sample_counts.items()
        ):
            validation.append(sample)
            remaining.subtract(sample_counts)

    validation_paths = {sample.path for sample in validation}
    training = [sample for sample in shuffled if sample.path not in validation_paths]
    return training, validation


def prepare_image(
    image: Image.Image,
    config: ImagePreprocessingConfig,
    augment: bool = False,
) -> Tensor:
    image = image.convert("RGB")
    if augment:
        image = image.rotate(
            random.uniform(-3.0, 3.0),
            resample=Image.Resampling.BILINEAR,
            translate=(random.randint(-3, 3), random.randint(-2, 2)),
            fillcolor=(255, 255, 255),
        )
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.82, 1.2))
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.08))
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.45)))

    image = image.resize((config.image_width, config.image_height), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    if augment and random.random() < 0.35:
        tensor = (tensor + torch.randn_like(tensor) * random.uniform(0.005, 0.025)).clamp(0.0, 1.0)
    return (tensor - NORMALIZATION_MEAN) / NORMALIZATION_STD


class CaptchaDataset(Dataset[tuple[Tensor, Tensor, str]]):
    def __init__(
        self,
        samples: list[CaptchaSample],
        codec: TextEncoder,
        config: ImagePreprocessingConfig,
        augment: bool,
        cache_images: bool = True,
    ) -> None:
        self.samples = samples
        self.codec = codec
        self.config = config
        self.augment = augment
        self.image_bytes = (
            [sample.path.read_bytes() for sample in samples] if cache_images else None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        sample = self.samples[index]
        source = BytesIO(self.image_bytes[index]) if self.image_bytes is not None else sample.path
        with Image.open(source) as image:
            tensor = prepare_image(image, self.config, augment=self.augment)
        return tensor, self.codec.encode(sample.label), sample.label


def collate_captchas(
    batch: list[tuple[Tensor, Tensor, str]],
) -> tuple[Tensor, Tensor, list[str]]:
    images, encoded, labels = zip(*batch, strict=True)
    return torch.stack(images), torch.stack(encoded), list(labels)


__all__ = [
    "NORMALIZATION_MEAN",
    "NORMALIZATION_STD",
    "PREPROCESSING_VERSION",
    "AuditIssue",
    "AuditedSample",
    "CaptchaDataset",
    "CaptchaSample",
    "DatasetAuditResult",
    "DuplicateFinding",
    "ImagePreprocessingConfig",
    "ManifestEntry",
    "TextEncoder",
    "audit_dataset",
    "collate_captchas",
    "coverage_aware_split",
    "load_samples",
    "observed_charset",
    "prepare_image",
    "write_dataset_audit",
]
