"""Strict multipart image validation for API boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from PIL import Image

from cipherlens.inference import UploadLimits, UploadValidationError, load_uploaded_image

_UPLOAD_TYPES = {
    ".png": ("image/png", "PNG"),
    ".jpg": ("image/jpeg", "JPEG"),
    ".jpeg": ("image/jpeg", "JPEG"),
}


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ValidatedUpload:
    image: Image.Image


async def validate_upload(upload: UploadFile, limits: UploadLimits) -> ValidatedUpload:
    suffix = Path(upload.filename or "").suffix.lower()
    expected = _UPLOAD_TYPES.get(suffix)
    if expected is None:
        await upload.close()
        raise APIError(415, "unsupported_extension", "Use a PNG, JPG, or JPEG filename.")
    mime_type = (upload.content_type or "").split(";", maxsplit=1)[0].strip().lower()
    if mime_type != expected[0]:
        await upload.close()
        raise APIError(415, "unsupported_mime_type", "The uploaded MIME type is unsupported.")
    try:
        data = await upload.read(limits.max_bytes + 1)
    finally:
        await upload.close()
    if len(data) > limits.max_bytes:
        raise APIError(413, "file_too_large", "The uploaded file exceeds the byte limit.")
    try:
        image = load_uploaded_image(data, limits)
    except UploadValidationError as error:
        message = str(error)
        if "pixel safety limit" in message:
            raise APIError(413, "image_too_large", message) from error
        if "Unsupported image format" in message:
            raise APIError(415, "unsupported_image_format", message) from error
        raise APIError(400, "invalid_image", message) from error
    with Image.open(BytesIO(data)) as probe:
        decoded_format = (probe.format or "").upper()
    if decoded_format != expected[1]:
        image.close()
        raise APIError(
            415,
            "image_type_mismatch",
            "The extension and MIME type do not match the decoded image.",
        )
    return ValidatedUpload(image)


__all__ = ["APIError", "ValidatedUpload", "validate_upload"]
