"""Train an optional experimental CRNN-CTC candidate without touching Model V1."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from cipherlens.config import ConfigurationError, load_project_settings
from cipherlens.data import load_samples, observed_charset
from cipherlens.logging import configure_logging
from cipherlens.models.ctc import CaptchaCTCCRNN, CTCCodec, CTCModelConfig
from cipherlens.training import EarlyStopping, build_loaders, choose_device, load_training_split
from cipherlens.training.checkpoint import write_history
from cipherlens.training.ctc import (
    build_ctc_metadata,
    build_ctc_optimization,
    evaluate_ctc,
    save_ctc_candidate,
    train_ctc_epoch,
)
from cipherlens.utils import seed_everything

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("cipherlens.training.ctc")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args(argv)
    try:
        settings = load_project_settings(ROOT, config_path=config_args.config)
    except ConfigurationError as error:
        config_parser.error(str(error))
    training = settings.training
    parser = argparse.ArgumentParser(description="Train experimental CipherLens Model V2 (CTC).")
    parser.add_argument("--config", type=Path, help="Project YAML configuration.")
    parser.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        type=Path,
        metavar=("LABELS", "IMAGES"),
        help="Override development datasets; repeat for multiple sources.",
    )
    parser.add_argument("--split-manifest", type=Path, default=training.split_manifest_path)
    parser.add_argument(
        "--no-split-manifest", action="store_const", const=None, dest="split_manifest"
    )
    parser.add_argument("--dataset-report", type=Path, default=training.dataset_report_path)
    parser.add_argument("--output", type=Path, default=Path("models/captcha_crnn_ctc_candidate.pt"))
    parser.add_argument(
        "--history-output", type=Path, default=Path("artifacts/ctc-training-history.json")
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=training.batch_size)
    parser.add_argument("--learning-rate", type=float, default=training.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=training.weight_decay)
    parser.add_argument("--patience", type=int, default=training.patience)
    parser.add_argument("--scheduler-factor", type=float, default=training.scheduler_factor)
    parser.add_argument("--scheduler-patience", type=int, default=training.scheduler_patience)
    parser.add_argument("--gradient-clip-norm", type=float, default=training.gradient_clip_norm)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=training.seed)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=training.device)
    parser.add_argument("--num-workers", type=int, default=training.num_workers)
    parser.add_argument("--torch-threads", type=int, default=training.torch_threads)
    parser.add_argument(
        "--no-cache-images",
        action="store_false",
        dest="cache_images",
        default=training.cache_images,
    )
    parser.add_argument("--deterministic", action="store_true", default=training.deterministic)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=settings.runtime.log_level,
    )
    parser.add_argument(
        "--log-format", choices=("console", "json"), default=settings.runtime.log_format
    )
    parser.set_defaults(
        approved_checkpoint=settings.runtime.checkpoint_path,
        configured_sources=tuple(
            (source.labels_path, source.images_path)
            for source in settings.dataset.sources
            if source.role == "development"
        ),
    )
    return parser.parse_args(argv)


def _path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _validate(args: argparse.Namespace) -> None:
    positive = (
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.patience,
        args.gradient_clip_norm,
        args.hidden_size,
        args.lstm_layers,
        args.torch_threads,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Positive CTC experiment values must be greater than zero.")
    if not 0 <= args.weight_decay <= 1 or not 0 < args.scheduler_factor < 1:
        raise ValueError("CTC weight decay or scheduler factor is invalid.")
    if args.scheduler_patience < 0 or args.num_workers < 0:
        raise ValueError("CTC scheduler patience and worker count cannot be negative.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level, args.log_format)
    _validate(args)
    output = _path(args.output)
    history_output = _path(args.history_output)
    manifest = _path(args.split_manifest)
    dataset_report = _path(args.dataset_report)
    if output is None or history_output is None:
        raise ValueError("CTC output paths must not be empty.")
    if output == Path(args.approved_checkpoint).resolve():
        raise ValueError("The experimental CTC candidate cannot overwrite Model V1.")
    raw_sources = args.dataset or args.configured_sources
    sources = [(_path(labels), _path(images)) for labels, images in raw_sources]
    if not sources or any(labels is None or images is None for labels, images in sources):
        raise ValueError("At least one valid CTC development dataset is required.")
    typed_sources = [(labels, images) for labels, images in sources if labels and images]

    seed_everything(args.seed, deterministic=args.deterministic)
    torch.set_num_threads(args.torch_threads)
    device = choose_device(args.device)
    samples = [
        sample
        for labels_path, images_path in typed_sources
        for sample in load_samples(labels_path, images_path)
    ]
    split = load_training_split(
        samples,
        project_root=ROOT,
        manifest_path=manifest,
        dataset_report_path=dataset_report,
        validation_fraction=0.2,
        seed=args.seed,
    )
    codec = CTCCodec(observed_charset(samples))
    config = CTCModelConfig(hidden_size=args.hidden_size, lstm_layers=args.lstm_layers)
    loaders = build_loaders(
        split,
        codec,
        config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_images=args.cache_images,
        device=device,
        seed=args.seed,
    )
    model = CaptchaCTCCRNN(codec.num_classes, config).to(device)
    loss_fn, optimizer, scheduler = build_ctc_optimization(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
    )
    stopping = EarlyStopping(args.patience)
    run_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_patience": args.scheduler_patience,
        "gradient_clip_norm": args.gradient_clip_norm,
        "hidden_size": args.hidden_size,
        "lstm_layers": args.lstm_layers,
        "seed": args.seed,
        "device": args.device,
        "num_workers": args.num_workers,
        "deterministic": args.deterministic,
    }
    metadata = build_ctc_metadata(
        project_root=ROOT,
        config=config,
        split=split,
        run_config=run_config,
        dataset_sources=[
            {"labels": _display(labels), "images": _display(images)}
            for labels, images in typed_sources
        ],
        device=device,
    )
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    LOGGER.info(
        "experimental_model=ctc device=%s train=%d validation=%d",
        device,
        len(split.training),
        len(split.validation),
        extra={"event": "ctc_training_started"},
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = train_ctc_epoch(
            model,
            loaders.training,
            loss_fn,
            optimizer,
            device,
            gradient_clip_norm=args.gradient_clip_norm,
        )
        metrics = evaluate_ctc(model, loaders.validation, codec, loss_fn, device)
        scheduler.step(metrics["loss"])
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        if stopping.update(epoch, metrics):
            save_ctc_candidate(
                output,
                model=model,
                codec=codec,
                config=config,
                metrics=metrics,
                epoch=epoch,
                metadata=metadata,
            )
        write_history(history_output, history)
        LOGGER.info(
            "epoch=%03d train_loss=%.4f val_loss=%.4f cer=%.4f exact=%.4f",
            epoch,
            train_loss,
            metrics["loss"],
            metrics["character_error_rate"],
            metrics["exact_accuracy"],
            extra={"event": "ctc_training_epoch", "epoch": epoch},
        )
        if stopping.should_stop:
            break
    LOGGER.info(
        "experimental_candidate=%s best_epoch=%d elapsed_seconds=%.1f",
        output,
        stopping.best_epoch,
        time.perf_counter() - started,
        extra={"event": "ctc_training_completed"},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
