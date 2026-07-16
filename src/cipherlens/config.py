"""Typed configuration loading and environment validation for CipherLens."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when configuration cannot be loaded or validated."""


@dataclass(frozen=True)
class RuntimeSettings:
    checkpoint_path: Path = Path("models/captcha_crnn.pt")
    torch_threads: int = 2
    confidence_threshold: float = 0.75
    max_upload_bytes: int = 10 * 1024 * 1024
    max_upload_pixels: int = 4_000_000
    log_level: str = "INFO"
    log_format: str = "console"


@dataclass(frozen=True)
class TrainingSettings:
    labels_path: Path = Path("labels.txt")
    images_path: Path = Path("data/batch_0")
    output_path: Path = Path("models/captcha_crnn.pt")
    history_output_path: Path = Path("training_history.json")
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    validation_fraction: float = 0.2
    patience: int = 15
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    torch_threads: int = min(4, os.cpu_count() or 1)
    cache_images: bool = True
    deterministic: bool = False


@dataclass(frozen=True)
class CipherLensSettings:
    runtime: RuntimeSettings
    training: TrainingSettings
    source: Path | None = None


_RUNTIME_KEYS = {
    "checkpoint_path",
    "torch_threads",
    "confidence_threshold",
    "max_upload_bytes",
    "max_upload_pixels",
    "log_level",
    "log_format",
}
_TRAINING_KEYS = {
    "labels_path",
    "images_path",
    "output_path",
    "history_output_path",
    "epochs",
    "batch_size",
    "learning_rate",
    "validation_fraction",
    "patience",
    "seed",
    "device",
    "num_workers",
    "torch_threads",
    "cache_images",
    "deterministic",
}
_ENVIRONMENT_KEYS = {
    "checkpoint_path": "CIPHERLENS_CHECKPOINT",
    "torch_threads": "CIPHERLENS_TORCH_THREADS",
    "confidence_threshold": "CIPHERLENS_CONFIDENCE_THRESHOLD",
    "max_upload_bytes": "CIPHERLENS_MAX_UPLOAD_BYTES",
    "max_upload_pixels": "CIPHERLENS_MAX_UPLOAD_PIXELS",
    "log_level": "CIPHERLENS_LOG_LEVEL",
    "log_format": "CIPHERLENS_LOG_FORMAT",
}
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Configuration section {name!r} must be a mapping.")
    if not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"Configuration section {name!r} has a non-string key.")
    return value


def _reject_unknown_keys(values: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigurationError(f"Unknown {name} configuration field(s): {', '.join(unknown)}.")


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, (str, Path)):
        raise ConfigurationError(f"{name} must be a non-empty string or path.")
    text = str(value).strip()
    if not text:
        raise ConfigurationError(f"{name} must not be empty.")
    return text


def _integer(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ConfigurationError(f"{name} must be an integer.")
    try:
        parsed = int(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer.") from error
    if parsed < minimum or (maximum is not None and parsed > maximum):
        expected = f"at least {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise ConfigurationError(f"{name} must be {expected}.")
    return parsed


def _floating(
    value: object,
    name: str,
    *,
    minimum: float,
    maximum: float,
    inclusive_minimum: bool = True,
    inclusive_maximum: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigurationError(f"{name} must be a number.")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number.") from error
    below_minimum = parsed < minimum if inclusive_minimum else parsed <= minimum
    above_maximum = parsed > maximum if inclusive_maximum else parsed >= maximum
    if below_minimum or above_maximum:
        left_bracket = "[" if inclusive_minimum else "("
        right_bracket = "]" if inclusive_maximum else ")"
        raise ConfigurationError(
            f"{name} must be in the range {left_bracket}{minimum}, {maximum}{right_bracket}."
        )
    return parsed


def _boolean(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ConfigurationError(f"{name} must be a boolean.")


def validate_torch_threads(value: object, name: str = "torch_threads") -> int:
    """Validate a bounded process-wide PyTorch CPU thread count."""
    return _integer(value, name, minimum=1, maximum=256)


def _runtime_settings(
    values: Mapping[str, object],
    environment: Mapping[str, str],
    project_root: Path,
) -> RuntimeSettings:
    defaults = RuntimeSettings()

    def configured(name: str, default: object) -> object:
        environment_name = _ENVIRONMENT_KEYS[name]
        return environment.get(environment_name, values.get(name, default))

    raw_checkpoint = _non_empty_string(
        configured("checkpoint_path", defaults.checkpoint_path),
        "runtime.checkpoint_path",
    )
    checkpoint_path = Path(raw_checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    log_level = _non_empty_string(
        configured("log_level", defaults.log_level), "runtime.log_level"
    ).upper()
    if log_level not in _LOG_LEVELS:
        raise ConfigurationError(
            f"runtime.log_level must be one of {', '.join(sorted(_LOG_LEVELS))}."
        )
    log_format = _non_empty_string(
        configured("log_format", defaults.log_format), "runtime.log_format"
    ).lower()
    if log_format not in {"console", "json"}:
        raise ConfigurationError("runtime.log_format must be 'console' or 'json'.")

    return RuntimeSettings(
        checkpoint_path=checkpoint_path.resolve(),
        torch_threads=validate_torch_threads(
            configured("torch_threads", defaults.torch_threads), "runtime.torch_threads"
        ),
        confidence_threshold=_floating(
            configured("confidence_threshold", defaults.confidence_threshold),
            "runtime.confidence_threshold",
            minimum=0.0,
            maximum=1.0,
        ),
        max_upload_bytes=_integer(
            configured("max_upload_bytes", defaults.max_upload_bytes),
            "runtime.max_upload_bytes",
            minimum=1,
        ),
        max_upload_pixels=_integer(
            configured("max_upload_pixels", defaults.max_upload_pixels),
            "runtime.max_upload_pixels",
            minimum=1,
        ),
        log_level=log_level,
        log_format=log_format,
    )


def _training_settings(values: Mapping[str, object]) -> TrainingSettings:
    defaults = TrainingSettings()
    device = _non_empty_string(values.get("device", defaults.device), "training.device").lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("training.device must be 'auto', 'cpu', or 'cuda'.")

    return TrainingSettings(
        labels_path=Path(
            _non_empty_string(
                values.get("labels_path", defaults.labels_path), "training.labels_path"
            )
        ),
        images_path=Path(
            _non_empty_string(
                values.get("images_path", defaults.images_path), "training.images_path"
            )
        ),
        output_path=Path(
            _non_empty_string(
                values.get("output_path", defaults.output_path), "training.output_path"
            )
        ),
        history_output_path=Path(
            _non_empty_string(
                values.get("history_output_path", defaults.history_output_path),
                "training.history_output_path",
            )
        ),
        epochs=_integer(values.get("epochs", defaults.epochs), "training.epochs", minimum=1),
        batch_size=_integer(
            values.get("batch_size", defaults.batch_size), "training.batch_size", minimum=1
        ),
        learning_rate=_floating(
            values.get("learning_rate", defaults.learning_rate),
            "training.learning_rate",
            minimum=0.0,
            maximum=1.0,
            inclusive_minimum=False,
        ),
        validation_fraction=_floating(
            values.get("validation_fraction", defaults.validation_fraction),
            "training.validation_fraction",
            minimum=0.0,
            maximum=1.0,
            inclusive_minimum=False,
            inclusive_maximum=False,
        ),
        patience=_integer(
            values.get("patience", defaults.patience), "training.patience", minimum=1
        ),
        seed=_integer(
            values.get("seed", defaults.seed),
            "training.seed",
            minimum=0,
            maximum=2**32 - 1,
        ),
        device=device,
        num_workers=_integer(
            values.get("num_workers", defaults.num_workers),
            "training.num_workers",
            minimum=0,
        ),
        torch_threads=validate_torch_threads(
            values.get("torch_threads", defaults.torch_threads), "training.torch_threads"
        ),
        cache_images=_boolean(
            values.get("cache_images", defaults.cache_images), "training.cache_images"
        ),
        deterministic=_boolean(
            values.get("deterministic", defaults.deterministic), "training.deterministic"
        ),
    )


def load_settings(
    config_path: Path | None = None,
    *,
    project_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> CipherLensSettings:
    """Load defaults, optional YAML, and validated runtime environment overrides."""
    root = (project_root or Path.cwd()).resolve()
    environment = os.environ if environ is None else environ
    document: Mapping[str, object] = {}
    source: Path | None = None

    if config_path is not None:
        source = config_path.expanduser()
        if not source.is_absolute():
            source = root / source
        source = source.resolve()
        if not source.is_file():
            raise ConfigurationError(f"Configuration file not found: {source}")
        try:
            loaded: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ConfigurationError(f"Could not read configuration file: {source}") from error
        document = _as_mapping(loaded, "root")

    _reject_unknown_keys(document, {"runtime", "training"}, "top-level")
    runtime_values = _as_mapping(document.get("runtime"), "runtime")
    training_values = _as_mapping(document.get("training"), "training")
    _reject_unknown_keys(runtime_values, _RUNTIME_KEYS, "runtime")
    _reject_unknown_keys(training_values, _TRAINING_KEYS, "training")

    return CipherLensSettings(
        runtime=_runtime_settings(runtime_values, environment, root),
        training=_training_settings(training_values),
        source=source,
    )


def load_project_settings(
    project_root: Path,
    *,
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> CipherLensSettings:
    """Load the project YAML selected explicitly, by environment, or by default."""
    environment = os.environ if environ is None else environ
    selected = config_path
    if selected is None:
        configured_path = environment.get("CIPHERLENS_CONFIG")
        if configured_path is not None:
            normalized_path = configured_path.strip()
            if not normalized_path:
                raise ConfigurationError("CIPHERLENS_CONFIG must not be empty.")
            selected = Path(normalized_path)
        else:
            selected = Path("configs/default.yaml")
    return load_settings(selected, project_root=project_root, environ=environment)
