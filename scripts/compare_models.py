"""Build an evidence-aligned model comparison from the model registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from cipherlens.evaluation.comparison import (
    build_comparison_rows,
    load_model_registry,
    write_comparison,
)

ROOT = Path(__file__).resolve().parents[1]


def _path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare registered CipherLens models.")
    parser.add_argument("--registry", type=Path, default=Path("configs/model-registry.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/evaluation/model_comparison.csv")
    )
    parser.add_argument("--document", type=Path, default=Path("docs/model-comparison.md"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    entries = load_model_registry(_path(args.registry), project_root=ROOT)
    rows = build_comparison_rows(entries)
    write_comparison(rows, csv_path=_path(args.output), document_path=_path(args.document))
    print(
        f"Compared {len(rows)} registered models; measured={sum(row['evaluation_status'] == 'measured' for row in rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
