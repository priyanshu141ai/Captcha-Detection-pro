"""Backward-compatible CLI for modular CipherLens Model V1 training."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from cipherlens.config import ConfigurationError, load_project_settings
from cipherlens.data import load_samples, observed_charset
from cipherlens.logging import configure_logging
from cipherlens.models import CaptchaCodec, CaptchaCRNN, ModelConfig
from cipherlens.training import (
    EarlyStopping,
    build_class_weights,
    build_loaders,
    build_optimization,
    build_run_metadata,
    choose_device,
    create_tracker,
    ensure_safe_artifact_paths,
    evaluate,
    load_resume_checkpoint,
    load_training_split,
    save_candidate_checkpoint,
    save_resume_checkpoint,
    train_one_epoch,
    warm_start_model,
    write_history,
)
from cipherlens.utils import seed_everything

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("cipherlens.training")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args(argv)
    try:
        settings = load_project_settings(ROOT, config_path=config_args.config)
    except ConfigurationError as error:
        config_parser.error(str(error))

    defaults = settings.training
    parser = argparse.ArgumentParser(description="Train the CipherLens CRNN CAPTCHA recognizer.")
    parser.add_argument("--config", type=Path, help="YAML defaults file.")
    parser.add_argument("--labels", type=Path, default=defaults.labels_path)
    parser.add_argument("--images", type=Path, default=defaults.images_path)
    parser.add_argument(
        "--extra-dataset",
        nargs=2,
        action="append",
        type=Path,
        default=[],
        metavar=("LABELS", "IMAGES"),
    )
    parser.add_argument("--output", type=Path, default=defaults.output_path)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--init-checkpoint", type=Path)
    initialization.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-output", type=Path, default=defaults.resume_output_path)
    manifest = parser.add_mutually_exclusive_group()
    manifest.add_argument("--split-manifest", type=Path, default=defaults.split_manifest_path)
    manifest.add_argument(
        "--no-split-manifest", action="store_const", const=None, dest="split_manifest"
    )
    report = parser.add_mutually_exclusive_group()
    report.add_argument("--dataset-report", type=Path, default=defaults.dataset_report_path)
    report.add_argument(
        "--no-dataset-report", action="store_const", const=None, dest="dataset_report"
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--validation-fraction", type=float, default=defaults.validation_fraction)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--scheduler-factor", type=float, default=defaults.scheduler_factor)
    parser.add_argument("--scheduler-patience", type=int, default=defaults.scheduler_patience)
    parser.add_argument("--gradient-clip-norm", type=float, default=defaults.gradient_clip_norm)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=defaults.device)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--torch-threads", type=int, default=defaults.torch_threads)
    parser.add_argument(
        "--no-cache-images",
        action="store_false",
        dest="cache_images",
        default=defaults.cache_images,
    )
    parser.add_argument("--history-output", type=Path, default=defaults.history_output_path)
    parser.add_argument("--deterministic", action="store_true", default=defaults.deterministic)
    tracking = parser.add_mutually_exclusive_group()
    tracking.add_argument("--mlflow", action="store_true", dest="mlflow_enabled")
    tracking.add_argument("--no-mlflow", action="store_false", dest="mlflow_enabled")
    parser.set_defaults(
        mlflow_enabled=defaults.mlflow_enabled,
        approved_checkpoint=settings.runtime.checkpoint_path,
    )
    parser.add_argument("--mlflow-uri", default=defaults.mlflow_tracking_uri)
    parser.add_argument("--mlflow-experiment", default=defaults.mlflow_experiment)
    parser.add_argument("--mlflow-run-name", default=defaults.mlflow_run_name)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=settings.runtime.log_level,
    )
    parser.add_argument(
        "--log-format", choices=("console", "json"), default=settings.runtime.log_format
    )
    return parser.parse_args(argv)


def _project_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _path_text(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size, and patience must be positive.")
    if args.learning_rate <= 0 or not 0 <= args.weight_decay <= 1:
        raise ValueError("learning-rate must be positive and weight-decay must be in [0, 1].")
    if not 0 < args.validation_fraction < 1 or not 0 < args.scheduler_factor < 1:
        raise ValueError("validation-fraction and scheduler-factor must be between 0 and 1.")
    if args.scheduler_patience < 0 or args.gradient_clip_norm <= 0:
        raise ValueError("scheduler-patience and gradient-clip-norm are invalid.")
    if args.num_workers < 0 or args.torch_threads < 1:
        raise ValueError("num-workers cannot be negative and torch-threads must be positive.")


def _run_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "validation_fraction": args.validation_fraction,
        "patience": args.patience,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_patience": args.scheduler_patience,
        "gradient_clip_norm": args.gradient_clip_norm,
        "seed": args.seed,
        "device": args.device,
        "num_workers": args.num_workers,
        "torch_threads": args.torch_threads,
        "cache_images": args.cache_images,
        "deterministic": args.deterministic,
        "split_manifest": _path_text(args.split_manifest),
        "dataset_report": _path_text(args.dataset_report),
        "output": _path_text(args.output),
        "resume_output": _path_text(args.resume_output),
        "history_output": _path_text(args.history_output),
        "init_checkpoint": _path_text(args.init_checkpoint),
        "resume_checkpoint": _path_text(args.resume_checkpoint),
        "mlflow_enabled": args.mlflow_enabled,
        "mlflow_experiment": args.mlflow_experiment,
        "mlflow_run_name": args.mlflow_run_name,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level.upper(), args.log_format)
    _validate_args(args)
    for name in (
        "labels",
        "images",
        "output",
        "resume_output",
        "history_output",
        "split_manifest",
        "dataset_report",
        "init_checkpoint",
        "resume_checkpoint",
    ):
        setattr(args, name, _project_path(getattr(args, name)))
    args.extra_dataset = [
        (_project_path(labels), _project_path(images)) for labels, images in args.extra_dataset
    ]
    approved_checkpoint = Path(args.approved_checkpoint)
    ensure_safe_artifact_paths(args.output, args.resume_output, approved_checkpoint)

    seed_everything(args.seed, deterministic=args.deterministic)
    torch.set_num_threads(args.torch_threads)
    device = choose_device(args.device)
    dataset_sources = [(args.labels, args.images), *args.extra_dataset]
    samples = [
        sample
        for labels_path, images_path in dataset_sources
        for sample in load_samples(labels_path, images_path)
    ]
    resolved_paths = [sample.path.resolve() for sample in samples]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("The configured datasets contain duplicate image paths.")
    split = load_training_split(
        samples,
        project_root=ROOT,
        manifest_path=args.split_manifest,
        dataset_report_path=args.dataset_report,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    codec = CaptchaCodec(observed_charset(samples))
    model_config = ModelConfig()
    loaders = build_loaders(
        split,
        codec,
        model_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_images=args.cache_images,
        device=device,
        seed=args.seed,
    )
    model = CaptchaCRNN(codec.num_classes, model_config).to(device)
    if args.init_checkpoint is not None:
        shared_count = warm_start_model(model, codec, args.init_checkpoint)
        LOGGER.info(
            "initialized_from=%s shared_character_weights=%d",
            args.init_checkpoint,
            shared_count,
            extra={"event": "training_warm_start"},
        )
    components = build_optimization(
        model,
        build_class_weights(split.training, codec),
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
    )
    early_stopping = EarlyStopping(args.patience)
    source_metadata = [
        {"labels": _path_text(labels) or "", "images": _path_text(images) or ""}
        for labels, images in dataset_sources
    ]
    run_config = _run_config(args)
    metadata = build_run_metadata(
        project_root=ROOT,
        model_config=model_config,
        split=split,
        run_config=run_config,
        dataset_sources=source_metadata,
        device=device,
    )
    history: list[dict[str, float | int]] = []
    best_model_state: dict[str, torch.Tensor] = {}
    start_epoch = 1
    if args.resume_checkpoint is not None:
        resume = load_resume_checkpoint(
            args.resume_checkpoint,
            model=model,
            codec=codec,
            model_config=model_config,
            optimizer=components.optimizer,
            scheduler=components.scheduler,
            early_stopping=early_stopping,
            expected_dataset_version=split.dataset_version,
            generator=loaders.generator,
            device=device,
        )
        start_epoch = resume.epoch + 1
        history = resume.history
        best_model_state = resume.best_model_state
        metadata["resumed_from"] = _path_text(args.resume_checkpoint)
        save_candidate_checkpoint(
            args.output,
            model_state=best_model_state,
            codec=codec,
            model_config=model_config,
            metrics=early_stopping.best_metrics,
            epoch=early_stopping.best_epoch,
            metadata=metadata,
        )

    tracker = create_tracker(
        args.mlflow_enabled,
        tracking_uri=args.mlflow_uri,
        experiment=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
    )
    started = time.perf_counter()
    failed = True
    try:
        tracker.log_parameters({**run_config, "dataset_version": split.dataset_version})
        LOGGER.info(
            "device=%s samples=%d train=%d validation=%d dataset_version=%s",
            device,
            len(samples),
            len(split.training),
            len(split.validation),
            split.dataset_version,
            extra={"event": "training_started", "device": str(device)},
        )
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                loaders.training,
                components.loss,
                components.optimizer,
                device,
                gradient_clip_norm=args.gradient_clip_norm,
            )
            metrics = evaluate(model, loaders.validation, codec, components.loss, device)
            components.scheduler.step(metrics["loss"])
            record: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_loss,
                **metrics,
                "learning_rate": float(components.optimizer.param_groups[0]["lr"]),
            }
            history.append(record)
            improved = early_stopping.update(epoch, metrics)
            if improved:
                best_model_state = {
                    name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
                }
                save_candidate_checkpoint(
                    args.output,
                    model_state=best_model_state,
                    codec=codec,
                    model_config=model_config,
                    metrics=metrics,
                    epoch=epoch,
                    metadata=metadata,
                )
            write_history(args.history_output, history)
            save_resume_checkpoint(
                args.resume_output,
                model=model,
                best_model_state=best_model_state,
                codec=codec,
                model_config=model_config,
                optimizer=components.optimizer,
                scheduler=components.scheduler,
                early_stopping=early_stopping,
                epoch=epoch,
                history=history,
                metadata=metadata,
                generator=loaders.generator,
            )
            tracker.log_metrics(
                {key: float(value) for key, value in record.items() if key != "epoch"}, epoch
            )
            LOGGER.info(
                "epoch=%03d train_loss=%.4f val_loss=%.4f char_acc=%.3f exact_acc=%.3f",
                epoch,
                train_loss,
                metrics["loss"],
                metrics["character_accuracy"],
                metrics["exact_accuracy"],
                extra={"event": "training_epoch", "epoch": epoch},
            )
            if early_stopping.should_stop:
                LOGGER.info(
                    "Early stopping after %d epochs.",
                    epoch,
                    extra={"event": "training_early_stop", "epoch": epoch},
                )
                break
        tracker.log_artifact(args.output)
        tracker.log_artifact(args.history_output)
        LOGGER.info(
            "saved=%s best_epoch=%d elapsed_seconds=%.1f",
            args.output,
            early_stopping.best_epoch,
            time.perf_counter() - started,
            extra={"event": "training_completed"},
        )
        failed = False
        return 0
    finally:
        tracker.close(failed=failed)


if __name__ == "__main__":
    raise SystemExit(main())
