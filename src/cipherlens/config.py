"""Typed configuration loading and environment validation for CipherLens."""

from __future__ import annotations

import os
import string
from collections.abc import Mapping, Sequence
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
class APISettings:
    max_batch_size: int = 8
    max_inference_concurrency: int = 1


@dataclass(frozen=True)
class TrainingSettings:
    labels_path: Path = Path("labels.txt")
    images_path: Path = Path("data/batch_0")
    output_path: Path = Path("models/captcha_crnn_candidate.pt")
    history_output_path: Path = Path("artifacts/candidate-training-history.json")
    resume_output_path: Path = Path("artifacts/candidate-training-resume.pt")
    split_manifest_path: Path | None = Path("artifacts/split_manifest.csv")
    dataset_report_path: Path | None = Path("artifacts/dataset_report.json")
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
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "CipherLens"
    mlflow_run_name: str | None = None


@dataclass(frozen=True)
class EvaluationSettings:
    split_manifest_path: Path = Path("artifacts/split_manifest.csv")
    output_path: Path = Path("reports/evaluation")
    figures_path: Path = Path("reports/figures")
    model_card_path: Path = Path("docs/model-card.md")
    split: str = "validation"
    batch_size: int = 32
    device: str = "cpu"
    torch_threads: int = 2
    ece_bins: int = 10
    latency_warmup: int = 5
    latency_runs: int = 50
    temperature_scaling: bool = False


@dataclass(frozen=True)
class DatasetSourceSettings:
    name: str
    labels_path: Path
    images_path: Path
    role: str = "development"
    provenance: str = "Not documented"
    authorization: str = "Maintainer confirmation pending"


@dataclass(frozen=True)
class DatasetSettings:
    name: str = "cipherlens-repository-dataset"
    sources: tuple[DatasetSourceSettings, ...] = ()
    expected_width: int = 151
    expected_height: int = 41
    label_length: int = 6
    expected_charset: str = string.digits + string.ascii_uppercase + string.ascii_lowercase
    rare_character_threshold: int = 5
    perceptual_hash_distance: int = 4
    validation_fraction: float = 0.2
    split_seed: int = 42
    artifacts_path: Path = Path("artifacts")
    dataset_card_path: Path = Path("docs/dataset-card.md")


@dataclass(frozen=True)
class CipherLensSettings:
    runtime: RuntimeSettings
    api: APISettings
    training: TrainingSettings
    evaluation: EvaluationSettings
    dataset: DatasetSettings
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
_API_KEYS = {"max_batch_size", "max_inference_concurrency"}
_TRAINING_KEYS = {
    "labels_path",
    "images_path",
    "output_path",
    "history_output_path",
    "resume_output_path",
    "split_manifest_path",
    "dataset_report_path",
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
    "weight_decay",
    "gradient_clip_norm",
    "scheduler_factor",
    "scheduler_patience",
    "mlflow_enabled",
    "mlflow_tracking_uri",
    "mlflow_experiment",
    "mlflow_run_name",
}
_EVALUATION_KEYS = {
    "split_manifest_path",
    "output_path",
    "figures_path",
    "model_card_path",
    "split",
    "batch_size",
    "device",
    "torch_threads",
    "ece_bins",
    "latency_warmup",
    "latency_runs",
    "temperature_scaling",
}
_DATASET_KEYS = {
    "name",
    "sources",
    "expected_width",
    "expected_height",
    "label_length",
    "expected_charset",
    "rare_character_threshold",
    "perceptual_hash_distance",
    "validation_fraction",
    "split_seed",
    "artifacts_path",
    "dataset_card_path",
}
_DATASET_SOURCE_KEYS = {
    "name",
    "labels_path",
    "images_path",
    "role",
    "provenance",
    "authorization",
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
_API_ENVIRONMENT_KEYS = {
    "max_batch_size": "CIPHERLENS_API_MAX_BATCH_SIZE",
    "max_inference_concurrency": "CIPHERLENS_API_MAX_CONCURRENCY",
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


def _resolved_path(value: object, name: str, project_root: Path) -> Path:
    path = Path(_non_empty_string(value, name)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _optional_path(value: object, name: str) -> Path | None:
    return None if value is None else Path(_non_empty_string(value, name))


def _optional_string(value: object, name: str) -> str | None:
    return None if value is None else _non_empty_string(value, name)


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


def _api_settings(values: Mapping[str, object], environment: Mapping[str, str]) -> APISettings:
    defaults = APISettings()

    def configured(name: str, default: object) -> object:
        return environment.get(_API_ENVIRONMENT_KEYS[name], values.get(name, default))

    return APISettings(
        max_batch_size=_integer(
            configured("max_batch_size", defaults.max_batch_size),
            "api.max_batch_size",
            minimum=1,
            maximum=100,
        ),
        max_inference_concurrency=_integer(
            configured("max_inference_concurrency", defaults.max_inference_concurrency),
            "api.max_inference_concurrency",
            minimum=1,
            maximum=32,
        ),
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
        resume_output_path=Path(
            _non_empty_string(
                values.get("resume_output_path", defaults.resume_output_path),
                "training.resume_output_path",
            )
        ),
        split_manifest_path=_optional_path(
            values.get("split_manifest_path", defaults.split_manifest_path),
            "training.split_manifest_path",
        ),
        dataset_report_path=_optional_path(
            values.get("dataset_report_path", defaults.dataset_report_path),
            "training.dataset_report_path",
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
        weight_decay=_floating(
            values.get("weight_decay", defaults.weight_decay),
            "training.weight_decay",
            minimum=0.0,
            maximum=1.0,
        ),
        gradient_clip_norm=_floating(
            values.get("gradient_clip_norm", defaults.gradient_clip_norm),
            "training.gradient_clip_norm",
            minimum=0.0,
            maximum=1_000.0,
            inclusive_minimum=False,
        ),
        scheduler_factor=_floating(
            values.get("scheduler_factor", defaults.scheduler_factor),
            "training.scheduler_factor",
            minimum=0.0,
            maximum=1.0,
            inclusive_minimum=False,
            inclusive_maximum=False,
        ),
        scheduler_patience=_integer(
            values.get("scheduler_patience", defaults.scheduler_patience),
            "training.scheduler_patience",
            minimum=0,
        ),
        mlflow_enabled=_boolean(
            values.get("mlflow_enabled", defaults.mlflow_enabled),
            "training.mlflow_enabled",
        ),
        mlflow_tracking_uri=_optional_string(
            values.get("mlflow_tracking_uri", defaults.mlflow_tracking_uri),
            "training.mlflow_tracking_uri",
        ),
        mlflow_experiment=_non_empty_string(
            values.get("mlflow_experiment", defaults.mlflow_experiment),
            "training.mlflow_experiment",
        ),
        mlflow_run_name=_optional_string(
            values.get("mlflow_run_name", defaults.mlflow_run_name),
            "training.mlflow_run_name",
        ),
    )


def _evaluation_settings(values: Mapping[str, object]) -> EvaluationSettings:
    defaults = EvaluationSettings()
    split = _non_empty_string(values.get("split", defaults.split), "evaluation.split").lower()
    if split not in {"validation", "external_test"}:
        raise ConfigurationError("evaluation.split must be 'validation' or 'external_test'.")
    device = _non_empty_string(values.get("device", defaults.device), "evaluation.device").lower()
    if device not in {"cpu", "cuda"}:
        raise ConfigurationError("evaluation.device must be 'cpu' or 'cuda'.")
    return EvaluationSettings(
        split_manifest_path=Path(
            _non_empty_string(
                values.get("split_manifest_path", defaults.split_manifest_path),
                "evaluation.split_manifest_path",
            )
        ),
        output_path=Path(
            _non_empty_string(
                values.get("output_path", defaults.output_path), "evaluation.output_path"
            )
        ),
        figures_path=Path(
            _non_empty_string(
                values.get("figures_path", defaults.figures_path), "evaluation.figures_path"
            )
        ),
        model_card_path=Path(
            _non_empty_string(
                values.get("model_card_path", defaults.model_card_path),
                "evaluation.model_card_path",
            )
        ),
        split=split,
        batch_size=_integer(
            values.get("batch_size", defaults.batch_size), "evaluation.batch_size", minimum=1
        ),
        device=device,
        torch_threads=validate_torch_threads(
            values.get("torch_threads", defaults.torch_threads), "evaluation.torch_threads"
        ),
        ece_bins=_integer(
            values.get("ece_bins", defaults.ece_bins),
            "evaluation.ece_bins",
            minimum=2,
            maximum=100,
        ),
        latency_warmup=_integer(
            values.get("latency_warmup", defaults.latency_warmup),
            "evaluation.latency_warmup",
            minimum=0,
        ),
        latency_runs=_integer(
            values.get("latency_runs", defaults.latency_runs),
            "evaluation.latency_runs",
            minimum=1,
        ),
        temperature_scaling=_boolean(
            values.get("temperature_scaling", defaults.temperature_scaling),
            "evaluation.temperature_scaling",
        ),
    )


def _default_dataset_sources(project_root: Path) -> tuple[DatasetSourceSettings, ...]:
    common = {
        "provenance": (
            "Repository-tracked CAPTCHA-style images; exact generation process is not documented."
        ),
        "authorization": (
            "Supplied by the repository owner for authorized educational use; "
            "independent license evidence is pending."
        ),
    }
    return (
        DatasetSourceSettings(
            name="batch_0",
            labels_path=(project_root / "labels.txt").resolve(),
            images_path=(project_root / "data/batch_0").resolve(),
            **common,
        ),
        DatasetSourceSettings(
            name="batch_1",
            labels_path=(project_root / "requirements2.txt").resolve(),
            images_path=(project_root / "data/batch_1").resolve(),
            **common,
        ),
    )


def _dataset_settings(values: Mapping[str, object], project_root: Path) -> DatasetSettings:
    defaults = DatasetSettings()
    raw_sources = values.get("sources")
    if raw_sources is None:
        sources = _default_dataset_sources(project_root)
    else:
        if isinstance(raw_sources, (str, bytes)) or not isinstance(raw_sources, Sequence):
            raise ConfigurationError("dataset.sources must be a list of mappings.")
        parsed_sources: list[DatasetSourceSettings] = []
        for index, raw_source in enumerate(raw_sources):
            source_name = f"dataset.sources[{index}]"
            source = _as_mapping(raw_source, source_name)
            _reject_unknown_keys(source, _DATASET_SOURCE_KEYS, source_name)
            role = _non_empty_string(source.get("role", "development"), f"{source_name}.role")
            if role not in {"development", "external_test"}:
                raise ConfigurationError(
                    f"{source_name}.role must be 'development' or 'external_test'."
                )
            parsed_sources.append(
                DatasetSourceSettings(
                    name=_non_empty_string(source.get("name"), f"{source_name}.name"),
                    labels_path=_resolved_path(
                        source.get("labels_path"), f"{source_name}.labels_path", project_root
                    ),
                    images_path=_resolved_path(
                        source.get("images_path"), f"{source_name}.images_path", project_root
                    ),
                    role=role,
                    provenance=_non_empty_string(
                        source.get("provenance", "Not documented"),
                        f"{source_name}.provenance",
                    ),
                    authorization=_non_empty_string(
                        source.get("authorization", "Maintainer confirmation pending"),
                        f"{source_name}.authorization",
                    ),
                )
            )
        sources = tuple(parsed_sources)

    if not sources or not any(source.role == "development" for source in sources):
        raise ConfigurationError("dataset.sources must include at least one development source.")
    names = [source.name for source in sources]
    if len(names) != len(set(names)):
        raise ConfigurationError("dataset source names must be unique.")

    charset = _non_empty_string(
        values.get("expected_charset", defaults.expected_charset), "dataset.expected_charset"
    )
    if len(charset) != len(set(charset)):
        raise ConfigurationError("dataset.expected_charset must not contain duplicates.")

    return DatasetSettings(
        name=_non_empty_string(values.get("name", defaults.name), "dataset.name"),
        sources=sources,
        expected_width=_integer(
            values.get("expected_width", defaults.expected_width),
            "dataset.expected_width",
            minimum=1,
        ),
        expected_height=_integer(
            values.get("expected_height", defaults.expected_height),
            "dataset.expected_height",
            minimum=1,
        ),
        label_length=_integer(
            values.get("label_length", defaults.label_length),
            "dataset.label_length",
            minimum=1,
        ),
        expected_charset=charset,
        rare_character_threshold=_integer(
            values.get("rare_character_threshold", defaults.rare_character_threshold),
            "dataset.rare_character_threshold",
            minimum=1,
        ),
        perceptual_hash_distance=_integer(
            values.get("perceptual_hash_distance", defaults.perceptual_hash_distance),
            "dataset.perceptual_hash_distance",
            minimum=0,
            maximum=64,
        ),
        validation_fraction=_floating(
            values.get("validation_fraction", defaults.validation_fraction),
            "dataset.validation_fraction",
            minimum=0.0,
            maximum=1.0,
            inclusive_minimum=False,
            inclusive_maximum=False,
        ),
        split_seed=_integer(
            values.get("split_seed", defaults.split_seed),
            "dataset.split_seed",
            minimum=0,
            maximum=2**32 - 1,
        ),
        artifacts_path=_resolved_path(
            values.get("artifacts_path", defaults.artifacts_path),
            "dataset.artifacts_path",
            project_root,
        ),
        dataset_card_path=_resolved_path(
            values.get("dataset_card_path", defaults.dataset_card_path),
            "dataset.dataset_card_path",
            project_root,
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

    _reject_unknown_keys(
        document, {"runtime", "api", "training", "evaluation", "dataset"}, "top-level"
    )
    runtime_values = _as_mapping(document.get("runtime"), "runtime")
    api_values = _as_mapping(document.get("api"), "api")
    training_values = _as_mapping(document.get("training"), "training")
    evaluation_values = _as_mapping(document.get("evaluation"), "evaluation")
    dataset_values = _as_mapping(document.get("dataset"), "dataset")
    _reject_unknown_keys(runtime_values, _RUNTIME_KEYS, "runtime")
    _reject_unknown_keys(api_values, _API_KEYS, "api")
    _reject_unknown_keys(training_values, _TRAINING_KEYS, "training")
    _reject_unknown_keys(evaluation_values, _EVALUATION_KEYS, "evaluation")
    _reject_unknown_keys(dataset_values, _DATASET_KEYS, "dataset")

    return CipherLensSettings(
        runtime=_runtime_settings(runtime_values, environment, root),
        api=_api_settings(api_values, environment),
        training=_training_settings(training_values),
        evaluation=_evaluation_settings(evaluation_values),
        dataset=_dataset_settings(dataset_values, root),
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
