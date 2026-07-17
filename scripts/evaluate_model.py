"""Evaluate one checkpoint against a versioned manifest split."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from cipherlens.config import ConfigurationError, load_project_settings
from cipherlens.evaluation import (
    EvaluationPendingError,
    EvaluationReportPaths,
    evaluate_checkpoint,
    write_evaluation_reports,
)
from cipherlens.logging import configure_logging

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("cipherlens.evaluation")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args(argv)
    try:
        settings = load_project_settings(ROOT, config_path=config_args.config)
    except ConfigurationError as error:
        config_parser.error(str(error))
    defaults = settings.evaluation
    parser = argparse.ArgumentParser(description="Evaluate CipherLens on a manifest split.")
    parser.add_argument("--config", type=Path, help="YAML defaults file.")
    parser.add_argument("--checkpoint", type=Path, default=settings.runtime.checkpoint_path)
    parser.add_argument("--split-manifest", type=Path, default=defaults.split_manifest_path)
    parser.add_argument(
        "--dataset-report", type=Path, default=settings.training.dataset_report_path
    )
    parser.add_argument("--output-dir", type=Path, default=defaults.output_path)
    parser.add_argument("--figures-dir", type=Path, default=defaults.figures_path)
    parser.add_argument("--model-card", type=Path, default=defaults.model_card_path)
    parser.add_argument("--split", choices=("validation", "external_test"), default=defaults.split)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=defaults.device)
    parser.add_argument("--torch-threads", type=int, default=defaults.torch_threads)
    parser.add_argument("--ece-bins", type=int, default=defaults.ece_bins)
    parser.add_argument("--latency-warmup", type=int, default=defaults.latency_warmup)
    parser.add_argument("--latency-runs", type=int, default=defaults.latency_runs)
    scaling = parser.add_mutually_exclusive_group()
    scaling.add_argument("--temperature-scale", action="store_true", dest="temperature_scaling")
    scaling.add_argument("--no-temperature-scale", action="store_false", dest="temperature_scaling")
    parser.set_defaults(temperature_scaling=defaults.temperature_scaling)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=settings.runtime.log_level,
    )
    parser.add_argument(
        "--log-format", choices=("console", "json"), default=settings.runtime.log_format
    )
    return parser.parse_args(argv)


def _path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level, args.log_format)
    checkpoint = _path(args.checkpoint)
    manifest = _path(args.split_manifest)
    output = _path(args.output_dir)
    figures = _path(args.figures_dir)
    model_card = _path(args.model_card)
    dataset_report = _path(args.dataset_report)
    if (
        checkpoint is None
        or manifest is None
        or output is None
        or figures is None
        or model_card is None
    ):
        raise ValueError("Evaluation paths must not be empty.")
    try:
        result = evaluate_checkpoint(
            checkpoint,
            manifest,
            project_root=ROOT,
            split=args.split,
            dataset_report_path=dataset_report,
            batch_size=args.batch_size,
            device=args.device,
            torch_threads=args.torch_threads,
            ece_bins=args.ece_bins,
            latency_warmup=args.latency_warmup,
            latency_runs=args.latency_runs,
            temperature_scaling=args.temperature_scaling,
        )
    except EvaluationPendingError as error:
        LOGGER.warning("%s", error, extra={"event": "evaluation_pending", "split": args.split})
        return 0
    paths = EvaluationReportPaths.from_directories(output, figures, model_card)
    write_evaluation_reports(result, paths)
    print(
        json.dumps(
            {
                "split": result.split,
                "samples": result.metrics.sample_count,
                "character_accuracy": round(result.metrics.character_accuracy, 6),
                "exact_accuracy": round(result.metrics.exact_accuracy, 6),
                "character_error_rate": round(result.metrics.character_error_rate, 6),
                "sequence_ece": round(result.metrics.sequence_ece, 6),
                "external_test_status": result.external_test_status,
                "outputs": [str(path) for path in as_paths(paths)],
            },
            indent=2,
        )
    )
    return 0


def as_paths(paths: EvaluationReportPaths) -> tuple[Path, ...]:
    return (
        paths.model_comparison,
        paths.failed_predictions,
        paths.evaluation_summary,
        paths.per_character_metrics,
        paths.per_position_metrics,
        paths.confidence_distribution,
        paths.confusion_matrix,
        paths.reliability_diagram,
        paths.model_card,
    )


if __name__ == "__main__":
    raise SystemExit(main())
