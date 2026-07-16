"""Privacy-safe structured logging configuration."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

_STANDARD_RECORD_FIELDS = set(logging.makeLogRecord({}).__dict__)
_SAFE_CONTEXT_FIELDS = {
    "event",
    "request_id",
    "model_version",
    "device",
    "epoch",
    "sample_count",
    "inference_ms",
}
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class JsonFormatter(logging.Formatter):
    """Emit one bounded JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _SAFE_CONTEXT_FIELDS:
            if field not in _STANDARD_RECORD_FIELDS and hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(level: str = "INFO", log_format: str = "console") -> logging.Logger:
    """Configure the CipherLens logger idempotently and return it."""
    normalized_level = level.upper()
    normalized_format = log_format.lower()
    if normalized_level not in _LOG_LEVELS:
        raise ValueError(f"Unsupported log level: {level!r}.")
    if normalized_format not in {"console", "json"}:
        raise ValueError(f"Unsupported log format: {log_format!r}.")

    logger = logging.getLogger("cipherlens")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    if normalized_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
    logger.addHandler(handler)
    logger.setLevel(normalized_level)
    logger.propagate = False
    return logger
