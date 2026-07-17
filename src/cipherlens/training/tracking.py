"""Optional MLflow tracking with a dependency-free disabled path."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Protocol


class ExperimentTracker(Protocol):
    def log_parameters(self, values: dict[str, object]) -> None: ...

    def log_metrics(self, values: dict[str, float], step: int) -> None: ...

    def log_artifact(self, path: Path) -> None: ...

    def close(self, *, failed: bool = False) -> None: ...


class NullTracker:
    def log_parameters(self, values: dict[str, object]) -> None:
        del values

    def log_metrics(self, values: dict[str, float], step: int) -> None:
        del values, step

    def log_artifact(self, path: Path) -> None:
        del path

    def close(self, *, failed: bool = False) -> None:
        del failed


class MlflowTracker:
    def __init__(
        self,
        *,
        tracking_uri: str | None,
        experiment: str,
        run_name: str | None,
    ) -> None:
        try:
            self._mlflow: Any = importlib.import_module("mlflow")
        except ImportError as error:
            raise RuntimeError(
                "MLflow tracking was enabled but MLflow is not installed. "
                "Install the project with '.[tracking]' or disable tracking."
            ) from error
        if tracking_uri is not None:
            self._mlflow.set_tracking_uri(tracking_uri)
        self._mlflow.set_experiment(experiment)
        self._mlflow.start_run(run_name=run_name)

    def log_parameters(self, values: dict[str, object]) -> None:
        parameters = {
            key: "null"
            if value is None
            else value
            if isinstance(value, (str, int, float, bool))
            else json.dumps(value, sort_keys=True)
            for key, value in values.items()
        }
        self._mlflow.log_params(parameters)

    def log_metrics(self, values: dict[str, float], step: int) -> None:
        self._mlflow.log_metrics(values, step=step)

    def log_artifact(self, path: Path) -> None:
        if path.is_file():
            self._mlflow.log_artifact(str(path))

    def close(self, *, failed: bool = False) -> None:
        self._mlflow.end_run(status="FAILED" if failed else "FINISHED")


def create_tracker(
    enabled: bool,
    *,
    tracking_uri: str | None = None,
    experiment: str = "CipherLens",
    run_name: str | None = None,
) -> ExperimentTracker:
    if not enabled:
        return NullTracker()
    return MlflowTracker(
        tracking_uri=tracking_uri,
        experiment=experiment,
        run_name=run_name,
    )


__all__ = ["ExperimentTracker", "MlflowTracker", "NullTracker", "create_tracker"]
