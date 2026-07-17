"""Fixed-length OCR metrics and confidence calibration summaries."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass

from cipherlens.models import levenshtein_distance


@dataclass(frozen=True)
class EvaluationRecord:
    path: str
    source: str
    target: str
    prediction: str
    confidence: float
    position_confidences: tuple[float, ...]


@dataclass(frozen=True)
class CharacterMetrics:
    character: str
    support: int
    predicted: int
    true_positives: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float | None
    accuracy: float | None
    gap: float | None


@dataclass(frozen=True)
class EvaluationMetrics:
    sample_count: int
    character_count: int
    character_accuracy: float
    exact_accuracy: float
    character_error_rate: float
    normalized_edit_distance: float
    mean_confidence: float
    median_confidence: float
    sequence_ece: float
    per_position_accuracy: tuple[float, ...]
    per_position_support: tuple[int, ...]
    per_character: tuple[CharacterMetrics, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    reliability_bins: tuple[ReliabilityBin, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def reliability_bins(
    confidences: list[float], outcomes: list[bool], bin_count: int
) -> tuple[ReliabilityBin, ...]:
    if len(confidences) != len(outcomes) or not confidences:
        raise ValueError("Confidence and outcome values must be non-empty and aligned.")
    if bin_count < 2:
        raise ValueError("At least two reliability bins are required.")
    grouped: list[list[tuple[float, bool]]] = [[] for _ in range(bin_count)]
    for confidence, outcome in zip(confidences, outcomes, strict=True):
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence values must be finite and between zero and one.")
        index = min(int(confidence * bin_count), bin_count - 1)
        grouped[index].append((confidence, outcome))

    bins: list[ReliabilityBin] = []
    for index, values in enumerate(grouped):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        if not values:
            bins.append(ReliabilityBin(lower, upper, 0, None, None, None))
            continue
        mean_confidence = sum(value[0] for value in values) / len(values)
        accuracy = sum(value[1] for value in values) / len(values)
        bins.append(
            ReliabilityBin(
                lower,
                upper,
                len(values),
                mean_confidence,
                accuracy,
                abs(accuracy - mean_confidence),
            )
        )
    return tuple(bins)


def expected_calibration_error(bins: tuple[ReliabilityBin, ...]) -> float:
    total = sum(item.count for item in bins)
    if total < 1:
        raise ValueError("Expected calibration error requires populated bins.")
    return sum((item.count / total) * (item.gap or 0.0) for item in bins)


def calculate_metrics(
    records: list[EvaluationRecord], charset: str, *, ece_bins: int = 10
) -> EvaluationMetrics:
    if not records:
        raise ValueError("Evaluation requires at least one prediction record.")
    if not charset or len(set(charset)) != len(charset):
        raise ValueError("The evaluation character set is empty or contains duplicates.")
    sequence_length = len(records[0].target)
    if sequence_length < 1:
        raise ValueError("Evaluation targets must not be empty.")
    charset_set = set(charset)
    for record in records:
        if (
            len(record.target) != sequence_length
            or len(record.prediction) != sequence_length
            or len(record.position_confidences) != sequence_length
        ):
            raise ValueError("Fixed-length targets, predictions, and confidences must align.")
        if not set(record.target + record.prediction) <= charset_set:
            raise ValueError("Evaluation records contain a character outside the vocabulary.")

    index_by_character = {character: index for index, character in enumerate(charset)}
    confusion = [[0 for _ in charset] for _ in charset]
    position_correct = [0] * sequence_length
    position_support = [0] * sequence_length
    support = {character: 0 for character in charset}
    predicted = {character: 0 for character in charset}
    true_positives = {character: 0 for character in charset}
    exact_matches = 0
    character_matches = 0
    edit_distance = 0
    normalized_distances: list[float] = []

    for record in records:
        exact_matches += int(record.target == record.prediction)
        distance = levenshtein_distance(record.target, record.prediction)
        edit_distance += distance
        normalized_distances.append(distance / max(len(record.target), len(record.prediction), 1))
        for position, (target, prediction) in enumerate(
            zip(record.target, record.prediction, strict=True)
        ):
            is_correct = target == prediction
            character_matches += int(is_correct)
            position_correct[position] += int(is_correct)
            position_support[position] += 1
            support[target] += 1
            predicted[prediction] += 1
            true_positives[target] += int(is_correct)
            confusion[index_by_character[target]][index_by_character[prediction]] += 1

    per_character = []
    for character in charset:
        tp = true_positives[character]
        precision = tp / predicted[character] if predicted[character] else None
        recall = tp / support[character] if support[character] else None
        if support[character] == 0:
            f1 = None
        elif tp == 0:
            f1 = 0.0
        elif precision is not None and recall is not None:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None
        per_character.append(
            CharacterMetrics(
                character,
                support[character],
                predicted[character],
                tp,
                precision,
                recall,
                f1,
            )
        )

    confidence_bins = reliability_bins(
        [record.confidence for record in records],
        [record.target == record.prediction for record in records],
        ece_bins,
    )
    character_count = len(records) * sequence_length
    return EvaluationMetrics(
        sample_count=len(records),
        character_count=character_count,
        character_accuracy=character_matches / character_count,
        exact_accuracy=exact_matches / len(records),
        character_error_rate=edit_distance / character_count,
        normalized_edit_distance=sum(normalized_distances) / len(normalized_distances),
        mean_confidence=sum(record.confidence for record in records) / len(records),
        median_confidence=statistics.median(record.confidence for record in records),
        sequence_ece=expected_calibration_error(confidence_bins),
        per_position_accuracy=tuple(
            correct / count
            for correct, count in zip(position_correct, position_support, strict=True)
        ),
        per_position_support=tuple(position_support),
        per_character=tuple(per_character),
        confusion_matrix=tuple(tuple(row) for row in confusion),
        reliability_bins=confidence_bins,
    )


__all__ = [
    "CharacterMetrics",
    "EvaluationMetrics",
    "EvaluationRecord",
    "ReliabilityBin",
    "calculate_metrics",
    "expected_calibration_error",
    "reliability_bins",
]
