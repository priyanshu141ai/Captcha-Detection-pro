from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from PIL import Image

from cipherlens.config import DatasetSettings, DatasetSourceSettings
from cipherlens.data import audit_dataset, write_dataset_audit


def _image(path: Path, seed: int, size: tuple[int, int] = (32, 16)) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size)
    image.putdata(
        [
            (
                (x * 31 + y * 17 + seed * 13) % 256,
                (x * 11 + y * 47 + seed * 29) % 256,
                (x * 53 + y * 7 + seed * 19) % 256,
            )
            for y in range(height)
            for x in range(width)
        ]
    )
    image.save(path)
    return image


def _source(root: Path, name: str, role: str = "development") -> DatasetSourceSettings:
    images = root / name
    images.mkdir()
    return DatasetSourceSettings(
        name=name,
        labels_path=root / f"{name}.txt",
        images_path=images,
        role=role,
        provenance="Generated test fixture",
        authorization="Owned test fixture",
    )


def _settings(
    root: Path,
    sources: tuple[DatasetSourceSettings, ...],
    charset: str = "ABC",
) -> DatasetSettings:
    return DatasetSettings(
        name="fixture",
        sources=sources,
        expected_width=32,
        expected_height=16,
        expected_charset=charset,
        perceptual_hash_distance=0,
        validation_fraction=0.25,
        artifacts_path=root / "artifacts",
        dataset_card_path=root / "docs/dataset-card.md",
    )


class DatasetAuditTests(unittest.TestCase):
    def test_manifest_and_outputs_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _source(root, "development")
            labels = ["AAAAAB", "AAAABA", "AAABAA", "AABAAA"]
            for index, _label in enumerate(labels):
                _image(source.images_path / f"{index}.png", index)
            source.labels_path.write_text(
                "".join(f"{index}.png {label}\n" for index, label in enumerate(labels)),
                encoding="utf-8",
            )
            settings = _settings(root, (source,), charset="AB")

            first = audit_dataset(settings, root)
            paths = write_dataset_audit(first, settings)
            before = {path: path.read_bytes() for path in paths}
            second = audit_dataset(settings, root)
            write_dataset_audit(second, settings)

            self.assertFalse(first.has_errors)
            self.assertEqual(first.dataset_version, second.dataset_version)
            self.assertEqual(first.split_version, second.split_version)
            self.assertEqual(
                Counter(entry.split for entry in first.manifest),
                {"train": 3, "validation": 1},
            )
            self.assertEqual(before, {path: path.read_bytes() for path in paths})
            report = json.loads((settings.artifacts_path / "dataset_report.json").read_text())
            self.assertEqual(report["dataset_version"], first.dataset_version)

            changed_seed = audit_dataset(replace(settings, split_seed=7), root)
            self.assertEqual(first.dataset_version, changed_seed.dataset_version)
            self.assertNotEqual(first.split_version, changed_seed.split_version)

    def test_invalid_files_and_duplicates_are_reported_without_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _source(root, "development")
            original = _image(source.images_path / "exact-a.png", 1)
            shutil.copyfile(source.images_path / "exact-a.png", source.images_path / "exact-b.png")
            original.save(source.images_path / "near.png", compress_level=0)
            _image(source.images_path / "wrong-size.png", 2, (31, 16))
            (source.images_path / "corrupt.png").write_bytes(b"not an image")
            _image(source.images_path / "short-label.png", 3)
            _image(source.images_path / "bad-character.png", 4)
            source.labels_path.write_text(
                "exact-a.png AAAAAA\n"
                "exact-b.png BBBBBB\n"
                "near.png CCCCCC\n"
                "wrong-size.png ABCABC\n"
                "corrupt.png ABCACB\n"
                "missing.png ACBACB\n"
                "short-label.png ABC\n"
                "bad-character.png AAAAA!\n"
                "exact-a.png CCCCCC\n"
                "broken row with extras\n",
                encoding="utf-8",
            )

            result = audit_dataset(_settings(root, (source,)), root)
            codes = {issue.code for issue in result.issues}
            relations = {finding.relation for finding in result.duplicates}

            self.assertTrue(
                {
                    "corrupt_image",
                    "duplicate_path",
                    "invalid_image_dimensions",
                    "invalid_label_character",
                    "invalid_label_length",
                    "malformed_label_row",
                    "missing_image",
                }
                <= codes
            )
            self.assertTrue({"exact_hash", "near_duplicate"} <= relations)
            self.assertFalse(any(finding.cross_split for finding in result.duplicates))
            self.assertTrue((source.images_path / "corrupt.png").exists())
            self.assertTrue((source.images_path / "wrong-size.png").exists())

    def test_external_duplicates_are_excluded_from_development(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development = _source(root, "development")
            external = _source(root, "external", role="external_test")
            _image(development.images_path / "shared.png", 1)
            _image(development.images_path / "train.png", 2)
            shutil.copyfile(
                development.images_path / "shared.png", external.images_path / "shared.png"
            )
            development.labels_path.write_text(
                "shared.png AAAAAA\ntrain.png BBBBBB\n", encoding="utf-8"
            )
            external.labels_path.write_text("shared.png AAAAAA\n", encoding="utf-8")

            result = audit_dataset(_settings(root, (development, external), charset="AB"), root)
            splits = {entry.path: entry.split for entry in result.manifest}

            self.assertEqual(splits["development/shared.png"], "excluded")
            self.assertEqual(splits["external/shared.png"], "external_test")
            self.assertEqual(splits["development/train.png"], "train")
            self.assertFalse(any(finding.cross_split for finding in result.duplicates))


if __name__ == "__main__":
    unittest.main()
