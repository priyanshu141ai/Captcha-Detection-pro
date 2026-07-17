"""Deterministic, non-destructive dataset auditing and split manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO, StringIO
from itertools import combinations
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image, UnidentifiedImageError

from cipherlens.config import DatasetSettings, DatasetSourceSettings

SCHEMA_VERSION = "1.0"
MANIFEST_VERSION = "1"


@dataclass(frozen=True)
class AuditIssue:
    code: str
    severity: str
    source: str
    path: str
    line_number: int | None
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "source": self.source,
            "path": self.path,
            "line_number": self.line_number,
            "message": self.message,
        }


@dataclass(frozen=True)
class AuditedSample:
    source: str
    source_role: str
    path: str
    label: str
    line_number: int
    sha256: str
    perceptual_hash: int
    width: int
    height: int
    image_format: str
    valid: bool

    @property
    def perceptual_hash_hex(self) -> str:
        return f"{self.perceptual_hash:016x}"


@dataclass(frozen=True)
class SourceSummary:
    name: str
    role: str
    labels_path: str
    images_path: str
    labels_sha256: str
    provenance: str
    authorization: str
    label_rows: int
    decoded_images: int
    valid_samples: int

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role,
            "labels_path": self.labels_path,
            "images_path": self.images_path,
            "labels_sha256": self.labels_sha256,
            "provenance": self.provenance,
            "authorization": self.authorization,
            "label_rows": self.label_rows,
            "decoded_images": self.decoded_images,
            "valid_samples": self.valid_samples,
        }


@dataclass(frozen=True)
class ManifestEntry:
    dataset_version: str
    manifest_version: str
    source: str
    source_role: str
    path: str
    label: str
    split: str
    valid: bool
    related_group: str
    sha256: str
    perceptual_hash: str
    width: int
    height: int
    image_format: str

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_version": self.dataset_version,
            "manifest_version": self.manifest_version,
            "source": self.source,
            "source_role": self.source_role,
            "path": self.path,
            "label": self.label,
            "split": self.split,
            "valid": self.valid,
            "related_group": self.related_group,
            "sha256": self.sha256,
            "perceptual_hash": self.perceptual_hash,
            "width": self.width,
            "height": self.height,
            "image_format": self.image_format,
        }


@dataclass(frozen=True)
class DuplicateFinding:
    relation: str
    group_id: str
    path_a: str
    path_b: str
    label_a: str
    label_b: str
    sha256_a: str
    sha256_b: str
    perceptual_distance: int | None
    split_a: str
    split_b: str
    cross_split: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "relation": self.relation,
            "group_id": self.group_id,
            "path_a": self.path_a,
            "path_b": self.path_b,
            "label_a": self.label_a,
            "label_b": self.label_b,
            "sha256_a": self.sha256_a,
            "sha256_b": self.sha256_b,
            "perceptual_distance": self.perceptual_distance,
            "split_a": self.split_a,
            "split_b": self.split_b,
            "cross_split": self.cross_split,
        }


@dataclass(frozen=True)
class DatasetAuditResult:
    dataset_name: str
    dataset_version: str
    split_version: str
    source_summaries: tuple[SourceSummary, ...]
    samples: tuple[AuditedSample, ...]
    issues: tuple[AuditIssue, ...]
    manifest: tuple[ManifestEntry, ...]
    duplicates: tuple[DuplicateFinding, ...]
    character_frequency: tuple[dict[str, object], ...]
    report: dict[str, object]

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)


@dataclass(frozen=True)
class _PairFinding:
    relation: str
    index_a: int
    index_b: int
    perceptual_distance: int | None


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, index: int) -> int:
        parent = self._parent[index]
        if parent != index:
            self._parent[index] = self.find(parent)
        return self._parent[index]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self._rank[root_left] < self._rank[root_right]:
            root_left, root_right = root_right, root_left
        self._parent[root_right] = root_left
        if self._rank[root_left] == self._rank[root_right]:
            self._rank[root_left] += 1


def _portable_path(path: Path, project_root: Path, source_name: str) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return f"external/{source_name}/{path.name}"


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=1)
def _dct_matrix() -> np.ndarray:
    size = 32
    indices = np.arange(size, dtype=np.float64)
    matrix = np.cos(np.pi * (2 * indices + 1) * indices[:, None] / (2 * size))
    matrix[0, :] *= np.sqrt(1 / size)
    matrix[1:, :] *= np.sqrt(2 / size)
    return matrix


def _perceptual_hash(image: Image.Image) -> int:
    pixels = np.asarray(
        image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64
    )
    matrix = _dct_matrix()
    low_frequency = (matrix @ pixels @ matrix.T)[:8, :8].ravel()
    median = float(np.median(low_frequency[1:]))
    value = 0
    for bit in low_frequency > median:
        value = (value << 1) | int(bit)
    return value


def _validate_source(
    source: DatasetSourceSettings,
    settings: DatasetSettings,
    project_root: Path,
    seen_paths: dict[Path, tuple[str, int]],
) -> tuple[list[AuditedSample], list[AuditIssue], SourceSummary]:
    samples: list[AuditedSample] = []
    issues: list[AuditIssue] = []
    labels_display = _portable_path(source.labels_path, project_root, source.name)
    images_display = _portable_path(source.images_path, project_root, source.name)
    try:
        labels_bytes = source.labels_path.read_bytes()
        labels_text = labels_bytes.decode("utf-8")
    except (OSError, UnicodeError) as error:
        issues.append(
            AuditIssue(
                "label_manifest_unreadable",
                "error",
                source.name,
                labels_display,
                None,
                f"Could not read the UTF-8 label manifest: {type(error).__name__}.",
            )
        )
        return (
            samples,
            issues,
            SourceSummary(
                source.name,
                source.role,
                labels_display,
                images_display,
                "",
                source.provenance,
                source.authorization,
                0,
                0,
                0,
            ),
        )

    label_rows = 0
    image_root = source.images_path.resolve()
    allowed = set(settings.expected_charset)
    for line_number, raw_line in enumerate(labels_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        label_rows += 1
        parts = line.split()
        if len(parts) != 2:
            issues.append(
                AuditIssue(
                    "malformed_label_row",
                    "error",
                    source.name,
                    labels_display,
                    line_number,
                    "Expected '<filename> <label>'.",
                )
            )
            continue
        filename, label = parts
        image_path = (source.images_path / filename).resolve()
        try:
            image_path.relative_to(image_root)
        except ValueError:
            issues.append(
                AuditIssue(
                    "path_outside_source",
                    "error",
                    source.name,
                    filename,
                    line_number,
                    "Referenced image escapes the configured images directory.",
                )
            )
            continue
        path_display = _portable_path(image_path, project_root, source.name)
        if image_path in seen_paths:
            first_source, first_line = seen_paths[image_path]
            issues.append(
                AuditIssue(
                    "duplicate_path",
                    "error",
                    source.name,
                    path_display,
                    line_number,
                    f"Path was already listed by {first_source} at line {first_line}.",
                )
            )
            continue
        seen_paths[image_path] = (source.name, line_number)

        valid = True
        if len(label) != settings.label_length:
            valid = False
            issues.append(
                AuditIssue(
                    "invalid_label_length",
                    "error",
                    source.name,
                    path_display,
                    line_number,
                    f"Label length is {len(label)}; expected {settings.label_length}.",
                )
            )
        unexpected = sorted(set(label) - allowed)
        if unexpected:
            valid = False
            issues.append(
                AuditIssue(
                    "invalid_label_character",
                    "error",
                    source.name,
                    path_display,
                    line_number,
                    f"Label contains characters outside the configured vocabulary: {unexpected!r}.",
                )
            )
        if not image_path.is_file():
            issues.append(
                AuditIssue(
                    "missing_image",
                    "error",
                    source.name,
                    path_display,
                    line_number,
                    "Referenced image does not exist.",
                )
            )
            continue
        try:
            image_bytes = image_path.read_bytes()
            with Image.open(BytesIO(image_bytes)) as image:
                image.verify()
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
                width, height = image.size
                image_format = image.format or "UNKNOWN"
                perceptual_hash = _perceptual_hash(image)
        except (OSError, SyntaxError, ValueError, UnidentifiedImageError) as error:
            issues.append(
                AuditIssue(
                    "corrupt_image",
                    "error",
                    source.name,
                    path_display,
                    line_number,
                    f"Image could not be decoded and verified: {type(error).__name__}.",
                )
            )
            continue
        if (width, height) != (settings.expected_width, settings.expected_height):
            valid = False
            issues.append(
                AuditIssue(
                    "invalid_image_dimensions",
                    "error",
                    source.name,
                    path_display,
                    line_number,
                    (
                        f"Image dimensions are {width}x{height}; expected "
                        f"{settings.expected_width}x{settings.expected_height}."
                    ),
                )
            )
        samples.append(
            AuditedSample(
                source.name,
                source.role,
                path_display,
                label,
                line_number,
                hashlib.sha256(image_bytes).hexdigest(),
                perceptual_hash,
                width,
                height,
                image_format,
                valid,
            )
        )

    return (
        samples,
        issues,
        SourceSummary(
            source.name,
            source.role,
            labels_display,
            images_display,
            hashlib.sha256(labels_bytes).hexdigest(),
            source.provenance,
            source.authorization,
            label_rows,
            len(samples),
            sum(sample.valid for sample in samples),
        ),
    )


def _related_groups(
    samples: list[AuditedSample], threshold: int
) -> tuple[_UnionFind, list[_PairFinding], int | None]:
    union_find = _UnionFind(len(samples))
    pairs: list[_PairFinding] = []
    minimum_distance: int | None = None
    by_hash: dict[str, list[int]] = defaultdict(list)
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_hash[sample.sha256].append(index)
        by_label[sample.label].append(index)

    for indices in by_hash.values():
        for left, right in combinations(indices, 2):
            union_find.union(left, right)
            pairs.append(_PairFinding("exact_hash", left, right, 0))
    for indices in by_label.values():
        for left, right in combinations(indices, 2):
            union_find.union(left, right)
            if samples[left].sha256 != samples[right].sha256:
                pairs.append(_PairFinding("duplicate_label", left, right, None))
    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            if samples[left].sha256 == samples[right].sha256:
                continue
            distance = (samples[left].perceptual_hash ^ samples[right].perceptual_hash).bit_count()
            minimum_distance = (
                distance if minimum_distance is None else min(minimum_distance, distance)
            )
            if distance <= threshold:
                union_find.union(left, right)
                pairs.append(_PairFinding("near_duplicate", left, right, distance))
    return union_find, pairs, minimum_distance


def _group_ids(samples: list[AuditedSample], union_find: _UnionFind) -> dict[int, str]:
    paths_by_root: dict[int, list[str]] = defaultdict(list)
    for index, sample in enumerate(samples):
        paths_by_root[union_find.find(index)].append(sample.path)
    return {
        root: f"group-{_canonical_hash(sorted(paths))[:16]}"
        for root, paths in paths_by_root.items()
    }


def _assign_splits(
    samples: list[AuditedSample],
    union_find: _UnionFind,
    settings: DatasetSettings,
) -> dict[int, str]:
    assignments: dict[int, str] = {}
    external_roots = {
        union_find.find(index)
        for index, sample in enumerate(samples)
        if sample.valid and sample.source_role == "external_test"
    }
    eligible = [
        index
        for index, sample in enumerate(samples)
        if sample.valid
        and sample.source_role == "development"
        and union_find.find(index) not in external_roots
    ]
    shuffled = list(eligible)
    random.Random(settings.split_seed).shuffle(shuffled)
    remaining = Counter(character for index in eligible for character in samples[index].label)
    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in eligible:
        members_by_root[union_find.find(index)].append(index)
    target_size = max(1, round(len(eligible) * settings.validation_fraction))
    validation_count = 0
    processed_roots: set[int] = set()
    for index in shuffled:
        root = union_find.find(index)
        if root in processed_roots:
            continue
        processed_roots.add(root)
        members = members_by_root[root]
        counts = Counter(character for member in members for character in samples[member].label)
        can_validate = validation_count + len(members) <= target_size and all(
            remaining[character] - count >= 1 for character, count in counts.items()
        )
        split = "validation" if can_validate else "train"
        for member in members:
            assignments[member] = split
        if can_validate:
            validation_count += len(members)
            remaining.subtract(counts)

    for index, sample in enumerate(samples):
        if index in assignments:
            continue
        if not sample.valid:
            assignments[index] = "excluded"
        elif sample.source_role == "external_test":
            assignments[index] = "external_test"
        else:
            assignments[index] = "excluded"
    return assignments


def _character_frequency(
    manifest: list[ManifestEntry], settings: DatasetSettings
) -> tuple[dict[str, object], ...]:
    counters: dict[str, Counter[str]] = {
        split: Counter(
            character
            for entry in manifest
            if entry.valid and entry.split == split
            for character in entry.label
        )
        for split in ("train", "validation", "external_test")
    }
    total = counters["train"] + counters["validation"] + counters["external_test"]
    characters = list(settings.expected_charset)
    characters.extend(sorted(set(total) - set(characters)))
    return tuple(
        {
            "character": character,
            "codepoint": f"U+{ord(character):04X}",
            "total_count": total[character],
            "train_count": counters["train"][character],
            "validation_count": counters["validation"][character],
            "external_test_count": counters["external_test"][character],
            "is_rare": 0 < total[character] <= settings.rare_character_threshold,
            "is_unseen_overall": total[character] == 0,
            "is_unseen_in_validation": total[character] > 0
            and counters["validation"][character] == 0,
        }
        for character in characters
    )


def _split_summary(manifest: list[ManifestEntry], split: str) -> dict[str, object]:
    entries = [entry for entry in manifest if entry.split == split and entry.valid]
    return {
        "samples": len(entries),
        "sources": dict(sorted(Counter(entry.source for entry in entries).items())),
        "observed_characters": len({character for entry in entries for character in entry.label}),
    }


def audit_dataset(settings: DatasetSettings, project_root: Path) -> DatasetAuditResult:
    """Audit configured sources without modifying or deleting source data."""
    samples: list[AuditedSample] = []
    issues: list[AuditIssue] = []
    summaries: list[SourceSummary] = []
    seen_paths: dict[Path, tuple[str, int]] = {}
    for source in settings.sources:
        source_samples, source_issues, summary = _validate_source(
            source, settings, project_root, seen_paths
        )
        samples.extend(source_samples)
        issues.extend(source_issues)
        summaries.append(summary)

    dataset_version = _canonical_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "sources": [
                {
                    "name": summary.name,
                    "role": summary.role,
                    "labels_sha256": summary.labels_sha256,
                }
                for summary in summaries
            ],
            "samples": [
                {
                    "source": sample.source,
                    "path": sample.path,
                    "label": sample.label,
                    "sha256": sample.sha256,
                    "width": sample.width,
                    "height": sample.height,
                    "valid": sample.valid,
                }
                for sample in sorted(samples, key=lambda item: (item.source, item.path))
            ],
            "issues": [issue.as_dict() for issue in issues],
        }
    )
    union_find, pair_findings, minimum_distance = _related_groups(
        samples, settings.perceptual_hash_distance
    )
    groups = _group_ids(samples, union_find)
    assignments = _assign_splits(samples, union_find, settings)
    manifest = [
        ManifestEntry(
            dataset_version,
            MANIFEST_VERSION,
            sample.source,
            sample.source_role,
            sample.path,
            sample.label,
            assignments[index],
            sample.valid,
            groups[union_find.find(index)],
            sample.sha256,
            sample.perceptual_hash_hex,
            sample.width,
            sample.height,
            sample.image_format,
        )
        for index, sample in enumerate(samples)
    ]
    manifest.sort(key=lambda entry: (entry.source, entry.path))
    split_version = _canonical_hash(
        {
            "manifest_version": MANIFEST_VERSION,
            "dataset_version": dataset_version,
            "seed": settings.split_seed,
            "validation_fraction": settings.validation_fraction,
            "entries": [entry.as_dict() for entry in manifest],
        }
    )
    duplicates = tuple(
        DuplicateFinding(
            finding.relation,
            groups[union_find.find(finding.index_a)],
            samples[finding.index_a].path,
            samples[finding.index_b].path,
            samples[finding.index_a].label,
            samples[finding.index_b].label,
            samples[finding.index_a].sha256,
            samples[finding.index_b].sha256,
            finding.perceptual_distance,
            assignments[finding.index_a],
            assignments[finding.index_b],
            assignments[finding.index_a] != assignments[finding.index_b]
            and assignments[finding.index_a] != "excluded"
            and assignments[finding.index_b] != "excluded",
        )
        for finding in sorted(
            pair_findings,
            key=lambda item: (
                item.relation,
                samples[item.index_a].path,
                samples[item.index_b].path,
            ),
        )
    )
    frequency = _character_frequency(manifest, settings)
    issue_counts = dict(sorted(Counter(issue.code for issue in issues).items()))
    counted_relations = Counter(item.relation for item in duplicates)
    relation_counts = {
        relation: counted_relations[relation]
        for relation in ("exact_hash", "near_duplicate", "duplicate_label")
    }
    rare = [str(row["character"]) for row in frequency if row["is_rare"]]
    unseen = [str(row["character"]) for row in frequency if row["is_unseen_overall"]]
    unseen_validation = [
        str(row["character"]) for row in frequency if row["is_unseen_in_validation"]
    ]
    split_summaries = {
        split: _split_summary(manifest, split)
        for split in ("train", "validation", "external_test", "excluded")
    }
    external_samples = cast(int, split_summaries["external_test"]["samples"])
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": settings.name,
        "dataset_version": dataset_version,
        "split_version": split_version,
        "policy": {
            "expected_dimensions": [settings.expected_width, settings.expected_height],
            "label_length": settings.label_length,
            "expected_charset": settings.expected_charset,
            "rare_character_threshold": settings.rare_character_threshold,
            "perceptual_hash": "64-bit DCT pHash",
            "perceptual_hash_distance": settings.perceptual_hash_distance,
            "validation_fraction": settings.validation_fraction,
            "split_seed": settings.split_seed,
            "related_grouping": ["exact_hash", "near_duplicate", "duplicate_label"],
        },
        "sources": [summary.as_dict() for summary in summaries],
        "summary": {
            "label_rows": sum(summary.label_rows for summary in summaries),
            "decoded_images": len(samples),
            "valid_samples": sum(sample.valid for sample in samples),
            "invalid_or_missing_samples": sum(summary.label_rows for summary in summaries)
            - sum(sample.valid for sample in samples),
            "character_occurrences": sum(cast(int, row["total_count"]) for row in frequency),
            "observed_characters": sum(not bool(row["is_unseen_overall"]) for row in frequency),
            "rare_characters": rare,
            "unseen_characters": unseen,
            "unseen_in_validation": unseen_validation,
        },
        "splits": {
            **split_summaries,
            "calibration": {
                "samples": 0,
                "status": "not_configured",
                "note": "No calibration split is created from the current validation evidence.",
            },
            "external_test_status": "available" if external_samples else "pending",
        },
        "duplicates": {
            "relation_counts": relation_counts,
            "cross_split_leakage": sum(item.cross_split for item in duplicates),
            "minimum_non_exact_perceptual_distance": minimum_distance,
            "note": "Perceptual hashing is heuristic and cannot prove samples are unrelated.",
        },
        "validation": {
            "error_count": sum(issue.severity == "error" for issue in issues),
            "issue_counts": issue_counts,
            "issues": [issue.as_dict() for issue in issues],
        },
        "artifacts": {
            "dataset_report": _portable_path(
                settings.artifacts_path / "dataset_report.json", project_root, "artifacts"
            ),
            "character_frequency": _portable_path(
                settings.artifacts_path / "character_frequency.csv", project_root, "artifacts"
            ),
            "duplicate_report": _portable_path(
                settings.artifacts_path / "duplicate_report.csv", project_root, "artifacts"
            ),
            "split_manifest": _portable_path(
                settings.artifacts_path / "split_manifest.csv", project_root, "artifacts"
            ),
            "dataset_card": _portable_path(
                settings.dataset_card_path, project_root, "dataset-card"
            ),
        },
    }
    return DatasetAuditResult(
        settings.name,
        dataset_version,
        split_version,
        tuple(summaries),
        tuple(samples),
        tuple(issues),
        tuple(manifest),
        duplicates,
        frequency,
        report,
    )


def _csv_text(rows: list[dict[str, object]], fieldnames: list[str]) -> str:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _dataset_card(result: DatasetAuditResult, settings: DatasetSettings) -> str:
    split_counts = Counter(entry.split for entry in result.manifest if entry.valid)
    rare = [str(row["character"]) for row in result.character_frequency if row["is_rare"]]
    unseen = [
        str(row["character"]) for row in result.character_frequency if row["is_unseen_overall"]
    ]
    unseen_validation = [
        str(row["character"])
        for row in result.character_frequency
        if row["is_unseen_in_validation"]
    ]
    lines = [
        "# CipherLens Dataset Card",
        "",
        "> Generated deterministically by `python -m scripts.audit_dataset`. Source images are",
        "> audited non-destructively; suspicious files are reported and never deleted.",
        "",
        "## Identity",
        "",
        f"- Dataset: `{result.dataset_name}`",
        f"- Dataset version: `{result.dataset_version}`",
        f"- Split version: `{result.split_version}`",
        f"- Valid samples: {sum(sample.valid for sample in result.samples)}",
        f"- Contract: {settings.label_length} characters; "
        f"{settings.expected_width}x{settings.expected_height} pixels",
        "",
        "## Intended use",
        "",
        "Educational recognition of synthetic, owned, or explicitly authorized CAPTCHA-style",
        "images. The dataset must not be used to automate third-party access-control bypass.",
        "",
        "## Sources and provenance",
        "",
        "| Source | Role | Samples | Provenance | Authorization |",
        "|---|---|---:|---|---|",
    ]
    for source in result.source_summaries:
        lines.append(
            f"| {source.name} | {source.role} | {source.valid_samples} | "
            f"{source.provenance} | {source.authorization} |"
        )
    lines.extend(
        [
            "",
            "The exact generator, collection date, license, and retention history are not recorded",
            "in the repository. Maintainers must resolve those provenance gaps before treating the",
            "dataset as independently redistributable evidence.",
            "",
            "## Deterministic splits",
            "",
            "| Split | Samples | Role |",
            "|---|---:|---|",
            f"| Train | {split_counts['train']} | Model fitting only |",
            f"| Validation | {split_counts['validation']} | Development evaluation |",
            "| Calibration | 0 | Not configured; validation is not silently reused |",
            f"| External test | {split_counts['external_test']} | Independent evaluation only |",
            "",
            (
                "External evaluation is **pending** because no separately sourced authorized "
                "external-test dataset is configured."
                if not split_counts["external_test"]
                else "An external-test source is configured and remains isolated from development."
            ),
            "",
            "Exact hashes, perceptual near-duplicates, and repeated labels are assigned one related",
            "group before splitting. Development samples overlapping an external group are excluded.",
            "",
            "## Character coverage",
            "",
            f"- Expected vocabulary: `{settings.expected_charset}`",
            f"- Rare (1-{settings.rare_character_threshold} occurrences): "
            f"`{''.join(rare) or 'none'}`",
            f"- Unseen overall: `{''.join(unseen) or 'none'}`",
            f"- Observed overall but unseen in validation: "
            f"`{''.join(unseen_validation) or 'none'}`",
            "",
            "Full per-character counts are stored in `artifacts/character_frequency.csv`.",
            "",
            "## Quality and duplicate audit",
            "",
            f"- Validation errors: {sum(issue.severity == 'error' for issue in result.issues)}",
            f"- Exact/label/near-duplicate findings: {len(result.duplicates)}",
            f"- Cross-split duplicate leakage: "
            f"{sum(finding.cross_split for finding in result.duplicates)}",
            f"- Near-duplicate threshold: 64-bit DCT pHash Hamming distance <= "
            f"{settings.perceptual_hash_distance}",
            "",
            "Perceptual hashing is heuristic: geometric or generator-level relationships may remain",
            "undetected. No generator-family metadata exists, so grouping currently relies on paths,",
            "labels, exact hashes, and perceptual similarity.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "python -m scripts.audit_dataset",
            "```",
            "",
            "Review `artifacts/dataset_report.json` and `artifacts/duplicate_report.csv` before",
            "using a new dataset version for training or evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_dataset_audit(result: DatasetAuditResult, settings: DatasetSettings) -> tuple[Path, ...]:
    """Write deterministic reports and a dataset card, replacing only known outputs."""
    report_path = settings.artifacts_path / "dataset_report.json"
    frequency_path = settings.artifacts_path / "character_frequency.csv"
    duplicates_path = settings.artifacts_path / "duplicate_report.csv"
    manifest_path = settings.artifacts_path / "split_manifest.csv"
    _atomic_write(
        report_path,
        json.dumps(result.report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    frequency_rows = [dict(row) for row in result.character_frequency]
    _atomic_write(
        frequency_path,
        _csv_text(
            frequency_rows,
            [
                "character",
                "codepoint",
                "total_count",
                "train_count",
                "validation_count",
                "external_test_count",
                "is_rare",
                "is_unseen_overall",
                "is_unseen_in_validation",
            ],
        ),
    )
    duplicate_rows = [item.as_dict() for item in result.duplicates]
    _atomic_write(
        duplicates_path,
        _csv_text(
            duplicate_rows,
            [
                "relation",
                "group_id",
                "path_a",
                "path_b",
                "label_a",
                "label_b",
                "sha256_a",
                "sha256_b",
                "perceptual_distance",
                "split_a",
                "split_b",
                "cross_split",
            ],
        ),
    )
    manifest_rows = [entry.as_dict() for entry in result.manifest]
    _atomic_write(
        manifest_path,
        _csv_text(
            manifest_rows,
            [
                "dataset_version",
                "manifest_version",
                "source",
                "source_role",
                "path",
                "label",
                "split",
                "valid",
                "related_group",
                "sha256",
                "perceptual_hash",
                "width",
                "height",
                "image_format",
            ],
        ),
    )
    _atomic_write(settings.dataset_card_path, _dataset_card(result, settings))
    return report_path, frequency_path, duplicates_path, manifest_path, settings.dataset_card_path


__all__ = [
    "AuditIssue",
    "AuditedSample",
    "DatasetAuditResult",
    "DuplicateFinding",
    "ManifestEntry",
    "audit_dataset",
    "write_dataset_audit",
]
