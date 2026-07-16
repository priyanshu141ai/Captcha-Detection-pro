from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from cipherlens.config import ConfigurationError, load_project_settings
from cipherlens.data import (
    CaptchaDataset,
    collate_captchas,
    coverage_aware_split,
    load_samples,
    observed_charset,
)
from cipherlens.logging import configure_logging
from cipherlens.models import CaptchaCodec, CaptchaCRNN, ModelConfig, levenshtein_distance
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
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML defaults file. CIPHERLENS_CONFIG is used when this is omitted.",
    )
    parser.add_argument("--labels", type=Path, default=defaults.labels_path)
    parser.add_argument("--images", type=Path, default=defaults.images_path)
    parser.add_argument(
        "--extra-dataset",
        nargs=2,
        action="append",
        type=Path,
        default=[],
        metavar=("LABELS", "IMAGES"),
        help="Additional label file and image directory. May be supplied multiple times.",
    )
    parser.add_argument("--output", type=Path, default=defaults.output_path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Warm-start from a checkpoint, preserving weights for shared characters.",
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--validation-fraction", type=float, default=defaults.validation_fraction)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=defaults.device)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=defaults.num_workers,
        help="DataLoader worker processes. Keep at 0 on Windows unless benchmarked.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=defaults.torch_threads,
        help="Maximum CPU threads used by PyTorch.",
    )
    parser.add_argument(
        "--no-cache-images",
        action="store_false",
        dest="cache_images",
        default=defaults.cache_images,
        help="Read images from disk every epoch instead of caching compressed bytes in memory.",
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        default=defaults.history_output_path,
        help="JSON training-history output path.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=defaults.deterministic,
        help="Request deterministic PyTorch algorithms; may reduce performance.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=settings.runtime.log_level,
    )
    parser.add_argument(
        "--log-format", choices=("console", "json"), default=settings.runtime.log_format
    )
    return parser.parse_args(argv)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def warm_start_model(
    model: CaptchaCRNN,
    codec: CaptchaCodec,
    checkpoint_path: Path,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_config = ModelConfig(**checkpoint.get("model_config", {}))
    if checkpoint_config != model.config:
        raise ValueError(
            "The initialization checkpoint model configuration does not match the current model."
        )

    checkpoint_codec = CaptchaCodec(checkpoint["charset"])
    checkpoint_state = checkpoint["model_state"]
    model_state = model.state_dict()

    for name, tensor in checkpoint_state.items():
        if name.startswith("classifier."):
            continue
        if name in model_state and model_state[name].shape == tensor.shape:
            model_state[name] = tensor

    shared_characters = sorted(set(codec.charset) & set(checkpoint_codec.charset))
    for character in shared_characters:
        old_index = checkpoint_codec.char_to_index[character]
        new_index = codec.char_to_index[character]
        model_state["classifier.weight"][new_index] = checkpoint_state["classifier.weight"][
            old_index
        ]
        model_state["classifier.bias"][new_index] = checkpoint_state["classifier.bias"][old_index]

    model.load_state_dict(model_state)
    return len(shared_characters)


def evaluate(
    model: CaptchaCRNN,
    loader: DataLoader[Any],
    codec: CaptchaCodec,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    exact_matches = 0
    edit_distance = 0
    character_count = 0
    sample_count = 0

    with torch.inference_mode():
        for images, targets, labels in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = loss_fn(logits.permute(1, 2, 0), targets)
            predictions = codec.greedy_decode(logits)

            total_loss += float(loss) * images.shape[0]
            for (prediction, _), label in zip(predictions, labels, strict=True):
                exact_matches += int(prediction == label)
                edit_distance += levenshtein_distance(prediction, label)
                character_count += len(label)
                sample_count += 1

    return {
        "loss": total_loss / max(sample_count, 1),
        "exact_accuracy": exact_matches / max(sample_count, 1),
        "character_accuracy": max(0.0, 1.0 - edit_distance / max(character_count, 1)),
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level.upper(), args.log_format)
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size, and patience must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive.")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be between 0 and 1.")
    if args.num_workers < 0 or args.torch_threads < 1:
        raise ValueError("num-workers cannot be negative and torch-threads must be positive.")

    seed_everything(args.seed, deterministic=args.deterministic)
    torch.set_num_threads(args.torch_threads)
    device = choose_device(args.device)
    config = ModelConfig()
    dataset_sources = [(args.labels, args.images), *args.extra_dataset]
    samples = [
        sample
        for labels_path, images_dir in dataset_sources
        for sample in load_samples(labels_path, images_dir)
    ]
    sample_paths = [sample.path.resolve() for sample in samples]
    if len(sample_paths) != len(set(sample_paths)):
        raise ValueError("The configured datasets contain duplicate image paths.")
    charset = observed_charset(samples)
    codec = CaptchaCodec(charset)
    training_samples, validation_samples = coverage_aware_split(
        samples, args.validation_fraction, args.seed
    )

    loader_options = {
        "num_workers": args.num_workers,
        "collate_fn": collate_captchas,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    training_loader = DataLoader(
        CaptchaDataset(
            training_samples,
            codec,
            config,
            augment=True,
            cache_images=args.cache_images,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        CaptchaDataset(
            validation_samples,
            codec,
            config,
            augment=False,
            cache_images=args.cache_images,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )

    model = CaptchaCRNN(codec.num_classes, config).to(device)
    if args.init_checkpoint is not None:
        shared_count = warm_start_model(model, codec, args.init_checkpoint)
        LOGGER.info(
            "initialized_from=%s shared_character_weights=%d",
            args.init_checkpoint,
            shared_count,
            extra={"event": "training_warm_start"},
        )
    character_counts = torch.zeros(codec.num_classes, dtype=torch.float32)
    for sample in training_samples:
        for character in sample.label:
            character_counts[codec.char_to_index[character]] += 1
    class_weights = torch.sqrt(character_counts.max() / character_counts.clamp_min(1.0)).clamp_max(
        4.0
    )
    class_weights /= class_weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_score = (-1.0, -1.0, float("-inf"))
    stale_epochs = 0
    started = time.perf_counter()

    LOGGER.info(
        "device=%s datasets=%d samples=%d train=%d validation=%d charset=%r",
        device,
        len(dataset_sources),
        len(samples),
        len(training_samples),
        len(validation_samples),
        charset,
        extra={"event": "training_started", "device": str(device), "sample_count": len(samples)},
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, targets, _ in training_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits.permute(1, 2, 0), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.detach()) * images.shape[0]
            seen += images.shape[0]

        metrics = evaluate(model, validation_loader, codec, loss_fn, device)
        scheduler.step(metrics["loss"])
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            **metrics,
        }
        history.append(record)
        LOGGER.info(
            "epoch=%03d train_loss=%.4f val_loss=%.4f char_acc=%.3f exact_acc=%.3f",
            epoch,
            record["train_loss"],
            metrics["loss"],
            metrics["character_accuracy"],
            metrics["exact_accuracy"],
            extra={"event": "training_epoch", "epoch": epoch},
        )

        score = (
            metrics["character_accuracy"],
            metrics["exact_accuracy"],
            -metrics["loss"],
        )
        if score > best_score:
            best_score = score
            stale_epochs = 0
            checkpoint = {
                "checkpoint_version": 1,
                "model_state": model.state_dict(),
                "charset": charset,
                "model_config": asdict(config),
                "metrics": metrics,
                "epoch": epoch,
                "training": {
                    "seed": args.seed,
                    "train_samples": len(training_samples),
                    "validation_samples": len(validation_samples),
                    "dataset_sources": [
                        {"labels": str(labels), "images": str(images)}
                        for labels, images in dataset_sources
                    ],
                },
            }
            temporary_output = args.output.with_suffix(f"{args.output.suffix}.tmp")
            torch.save(
                {
                    **checkpoint,
                },
                temporary_output,
            )
            temporary_output.replace(args.output)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                LOGGER.info(
                    "Early stopping after %d epochs.",
                    epoch,
                    extra={"event": "training_early_stop", "epoch": epoch},
                )
                break

    args.history_output.parent.mkdir(parents=True, exist_ok=True)
    args.history_output.write_text(json.dumps(history, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - started
    LOGGER.info(
        "saved=%s best_character_accuracy=%.3f elapsed_seconds=%.1f",
        args.output,
        best_score[0],
        elapsed,
        extra={"event": "training_completed"},
    )


if __name__ == "__main__":
    main()
