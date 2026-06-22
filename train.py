from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data import (
    CaptchaDataset,
    collate_captchas,
    coverage_aware_split,
    load_samples,
    observed_charset,
)
from src.model import CaptchaCodec, CaptchaCRNN, ModelConfig, levenshtein_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CipherLens CRNN CAPTCHA recognizer.")
    parser.add_argument("--labels", type=Path, default=Path("labels.txt"))
    parser.add_argument("--images", type=Path, default=Path("data/batch_0"))
    parser.add_argument(
        "--extra-dataset",
        nargs=2,
        action="append",
        type=Path,
        default=[],
        metavar=("LABELS", "IMAGES"),
        help="Additional label file and image directory. May be supplied multiple times.",
    )
    parser.add_argument("--output", type=Path, default=Path("models/captcha_crnn.pt"))
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Warm-start from a checkpoint, preserving weights for shared characters.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes. Keep at 0 on Windows unless benchmarked.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Maximum CPU threads used by PyTorch.",
    )
    parser.add_argument(
        "--no-cache-images",
        action="store_true",
        help="Read images from disk every epoch instead of caching compressed bytes in memory.",
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        default=Path("training_history.json"),
        help="JSON training-history output path.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        model_state["classifier.weight"][new_index] = checkpoint_state["classifier.weight"][old_index]
        model_state["classifier.bias"][new_index] = checkpoint_state["classifier.bias"][old_index]

    model.load_state_dict(model_state)
    return len(shared_characters)


def evaluate(
    model: CaptchaCRNN,
    loader: DataLoader,
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
            for (prediction, _), label in zip(predictions, labels):
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
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size, and patience must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive.")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be between 0 and 1.")
    if args.num_workers < 0 or args.torch_threads < 1:
        raise ValueError("num-workers cannot be negative and torch-threads must be positive.")

    seed_everything(args.seed)
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
            cache_images=not args.no_cache_images,
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
            cache_images=not args.no_cache_images,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )

    model = CaptchaCRNN(codec.num_classes, config).to(device)
    if args.init_checkpoint is not None:
        shared_count = warm_start_model(model, codec, args.init_checkpoint)
        print(
            f"initialized_from={args.init_checkpoint} shared_character_weights={shared_count}"
        )
    character_counts = torch.zeros(codec.num_classes, dtype=torch.float32)
    for sample in training_samples:
        for character in sample.label:
            character_counts[codec.char_to_index[character]] += 1
    class_weights = torch.sqrt(character_counts.max() / character_counts.clamp_min(1.0)).clamp_max(4.0)
    class_weights /= class_weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_score = (-1.0, -1.0, float("-inf"))
    stale_epochs = 0
    started = time.perf_counter()

    print(
        f"device={device} datasets={len(dataset_sources)} samples={len(samples)} "
        f"train={len(training_samples)} "
        f"validation={len(validation_samples)} charset={charset!r}"
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
        print(
            f"epoch={epoch:03d} train_loss={record['train_loss']:.4f} "
            f"val_loss={metrics['loss']:.4f} char_acc={metrics['character_accuracy']:.3f} "
            f"exact_acc={metrics['exact_accuracy']:.3f}"
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
                print(f"Early stopping after {epoch} epochs.")
                break

    args.history_output.parent.mkdir(parents=True, exist_ok=True)
    args.history_output.write_text(json.dumps(history, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - started
    print(
        f"saved={args.output} best_character_accuracy={best_score[0]:.3f} "
        f"elapsed_seconds={elapsed:.1f}"
    )


if __name__ == "__main__":
    main()
