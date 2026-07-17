"""Evaluate an experimental CTC checkpoint when a candidate exists."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from cipherlens.config import ConfigurationError, load_project_settings
from cipherlens.evaluation.ctc import evaluate_ctc_checkpoint, write_ctc_evaluation_summary
from cipherlens.evaluation.runner import EvaluationPendingError
from cipherlens.logging import configure_logging

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("cipherlens.evaluation.ctc")


def _path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args(argv)
    try:
        settings = load_project_settings(ROOT, config_path=config_args.config)
    except ConfigurationError as error:
        config_parser.error(str(error))
    defaults = settings.evaluation
    parser = argparse.ArgumentParser(description="Evaluate experimental CipherLens CTC Model V2.")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/captcha_crnn_ctc_candidate.pt")
    )
    parser.add_argument("--split-manifest", type=Path, default=defaults.split_manifest_path)
    parser.add_argument(
        "--dataset-report", type=Path, default=settings.training.dataset_report_path
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evaluation/v2/evaluation_summary.json"),
    )
    parser.add_argument("--split", choices=("validation", "external_test"), default=defaults.split)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=defaults.device)
    parser.add_argument("--torch-threads", type=int, default=defaults.torch_threads)
    parser.add_argument("--ece-bins", type=int, default=defaults.ece_bins)
    parser.add_argument("--latency-warmup", type=int, default=defaults.latency_warmup)
    parser.add_argument("--latency-runs", type=int, default=defaults.latency_runs)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=settings.runtime.log_level,
    )
    parser.add_argument(
        "--log-format", choices=("console", "json"), default=settings.runtime.log_format
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level, args.log_format)
    checkpoint = _path(args.checkpoint)
    manifest = _path(args.split_manifest)
    report = _path(args.dataset_report)
    output = _path(args.output)
    if checkpoint is None or manifest is None or output is None:
        raise ValueError("CTC evaluation paths must not be empty.")
    if not checkpoint.is_file():
        LOGGER.warning(
            "CTC candidate not found; evaluation pending: %s",
            checkpoint,
            extra={"event": "ctc_evaluation_pending"},
        )
        return 0
    try:
        summary = evaluate_ctc_checkpoint(
            checkpoint,
            manifest,
            project_root=ROOT,
            split=args.split,
            dataset_report_path=report,
            batch_size=args.batch_size,
            device=args.device,
            torch_threads=args.torch_threads,
            ece_bins=args.ece_bins,
            latency_warmup=args.latency_warmup,
            latency_runs=args.latency_runs,
        )
    except EvaluationPendingError as error:
        LOGGER.warning("%s", error, extra={"event": "ctc_evaluation_pending"})
        return 0
    write_ctc_evaluation_summary(summary, output)
    metrics = summary["metrics"]
    print(
        json.dumps(
            {
                "output": str(output),
                "sample_count": metrics["sample_count"],
                "exact_accuracy": metrics["exact_accuracy"],
                "character_error_rate": metrics["character_error_rate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
