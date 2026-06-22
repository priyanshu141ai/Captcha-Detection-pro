from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError


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

