"""Generate deterministic CipherLens dataset audit artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cipherlens.config import load_project_settings
from cipherlens.data import audit_dataset, write_dataset_audit

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve()
    settings = load_project_settings(root, config_path=args.config, environ={})
    result = audit_dataset(settings.dataset, root)
    outputs = write_dataset_audit(result, settings.dataset)
    print(
        json.dumps(
            {
                "dataset_version": result.dataset_version,
                "split_version": result.split_version,
                "valid_samples": sum(sample.valid for sample in result.samples),
                "errors": sum(issue.severity == "error" for issue in result.issues),
                "outputs": [str(path) for path in outputs],
            },
            indent=2,
        )
    )
    return 1 if result.has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
