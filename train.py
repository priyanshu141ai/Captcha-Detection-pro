from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--output", type=Path, default=Path("models/captcha_crnn.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")

    seed_everything(args.seed)
    device = choose_device(args.device)
    config = ModelConfig()
    samples = load_samples(args.labels, args.images)
    charset = observed_charset(samples)
    codec = CaptchaCodec(charset)
    training_samples, validation_samples = coverage_aware_split(
        samples, args.validation_fraction, args.seed
    )

    training_loader = DataLoader(
        CaptchaDataset(training_samples, codec, config, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_captchas,
    )
    validation_loader = DataLoader(
        CaptchaDataset(validation_samples, codec, config, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_captchas,
    )

    model = CaptchaCRNN(codec.num_classes, config).to(device)
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
    best_character_accuracy = -1.0
    stale_epochs = 0
    started = time.perf_counter()

    print(
        f"device={device} samples={len(samples)} train={len(training_samples)} "
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

        if metrics["character_accuracy"] > best_character_accuracy:
            best_character_accuracy = metrics["character_accuracy"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "charset": charset,
                    "model_config": asdict(config),
                    "metrics": metrics,
                    "epoch": epoch,
                },
                args.output,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    Path("training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - started
    print(
        f"saved={args.output} best_character_accuracy={best_character_accuracy:.3f} "
        f"elapsed_seconds={elapsed:.1f}"
    )


if __name__ == "__main__":
    main()
