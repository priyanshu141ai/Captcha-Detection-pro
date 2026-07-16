from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from PIL import Image

from cipherlens.inference import CaptchaRecognizer

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "captcha_crnn.pt"
CASES = (
    (ROOT / "data" / "batch_0" / "captcha_00000.png", "TAGbCN"),
    (ROOT / "data" / "batch_1" / "captcha_00500.png", "7LBNCF"),
)


def main() -> None:
    started = time.perf_counter()
    recognizer = CaptchaRecognizer(CHECKPOINT)
    load_seconds = time.perf_counter() - started
    predictions = []

    for image_path, expected in CASES:
        inference_started = time.perf_counter()
        with Image.open(image_path) as image:
            prediction = recognizer.predict(image)
        inference_seconds = time.perf_counter() - inference_started
        if prediction.text != expected:
            raise AssertionError(
                f"Expected {expected!r} for {image_path.name}, got {prediction.text!r}."
            )
        predictions.append(
            {
                "image": image_path.name,
                "prediction": prediction.text,
                "confidence": round(prediction.confidence, 4),
                "inference_ms": round(inference_seconds * 1000, 2),
            }
        )

    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "checkpoint_load_ms": round(load_seconds * 1000, 2),
                "predictions": predictions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
