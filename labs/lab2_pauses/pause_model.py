"""Shared feature extraction and inference for text-based pause prediction."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from pymorphy3 import MorphAnalyzer

from align_words import WORD_RE
from extract_pauses import normalize_punctuation


VOWELS = frozenset("аеёиоуыэюя")


def text_boundary_rows(text: str) -> list[dict[str, str | float]]:
    """Convert new Russian text into the same boundary rows used for training."""

    matches = list(WORD_RE.finditer(text))
    boundary_count = max(0, len(matches) - 1)
    rows: list[dict[str, str | float]] = []
    for index, (left, right) in enumerate(zip(matches[:-1], matches[1:], strict=True)):
        rows.append(
            {
                "boundary_index": str(index),
                "left_word": left.group(0),
                "right_word": right.group(0),
                "punctuation": normalize_punctuation(text[left.end() : right.start()].strip()),
                "position_ratio": (index + 1) / (boundary_count + 1),
            }
        )
    return rows


class BoundaryFeatureBuilder:
    """Build interpretable lexical, punctuation and morphological features."""

    def __init__(self) -> None:
        self.morph = MorphAnalyzer()

    @lru_cache(maxsize=100_000)
    def _morphology(self, word: str) -> tuple[str, str, str, str]:
        word = word.replace("\u0301", "")
        parse = self.morph.parse(word)[0]
        tag = parse.tag
        return (
            str(tag.POS or "NONE"),
            str(tag.case or "NONE"),
            str(tag.number or "NONE"),
            str(tag.gender or "NONE"),
        )

    @staticmethod
    def _word_features(prefix: str, word: str) -> dict[str, object]:
        lower = word.lower().replace("ё", "е").replace("\u0301", "")
        return {
            f"{prefix}_word": lower,
            f"{prefix}_length": len(lower),
            f"{prefix}_starts_upper": bool(word[:1].isupper()),
            f"{prefix}_vowel_count": sum(char in VOWELS for char in lower),
            f"{prefix}_prefix1": lower[:1],
            f"{prefix}_prefix2": lower[:2],
            f"{prefix}_suffix1": lower[-1:],
            f"{prefix}_suffix2": lower[-2:],
            f"{prefix}_suffix3": lower[-3:],
        }

    def transform_row(self, row: dict[str, str | float]) -> dict[str, object]:
        left = str(row["left_word"])
        right = str(row["right_word"])
        left_pos, left_case, left_number, left_gender = self._morphology(left.lower())
        right_pos, right_case, right_number, right_gender = self._morphology(right.lower())
        features: dict[str, object] = {
            "punctuation": str(row["punctuation"]),
            "has_written_punctuation": str(row["punctuation"]) != "NONE",
            "position_ratio": float(row["position_ratio"]),
            "left_pos": left_pos,
            "left_case": left_case,
            "left_number": left_number,
            "left_gender": left_gender,
            "right_pos": right_pos,
            "right_case": right_case,
            "right_number": right_number,
            "right_gender": right_gender,
            "pos_pair": f"{left_pos}>{right_pos}",
        }
        features.update(self._word_features("left", left))
        features.update(self._word_features("right", right))
        return features

    def transform(self, rows: Iterable[dict[str, str | float]]) -> list[dict[str, object]]:
        return [self.transform_row(row) for row in rows]


class PausePredictor:
    """Load a trained lab-2 model and predict pauses from plain text."""

    def __init__(self, model_path: Path | str) -> None:
        self.bundle = joblib.load(model_path)
        self.features = BoundaryFeatureBuilder()

    def predict_rows(self, rows: list[dict[str, str | float]]) -> list[dict[str, object]]:
        if not rows:
            return []
        matrix = self.bundle["vectorizer"].transform(self.features.transform(rows))
        probabilities = self.bundle["classifier"].predict_proba(matrix)[:, 1]
        duration_ms = np.expm1(self.bundle["regressor"].predict(matrix))
        duration_ms = np.maximum(duration_ms, 0.0)
        threshold = float(self.bundle["classification_threshold"])
        result: list[dict[str, object]] = []
        for row, probability, predicted_duration in zip(
            rows, probabilities, duration_ms, strict=True
        ):
            has_pause = bool(probability >= threshold)
            result.append(
                dict(row)
                | {
                    "pause_probability": float(probability),
                    "has_pause": has_pause,
                    "duration_ms": float(predicted_duration) if has_pause else 0.0,
                }
            )
        return result

    def predict_text(self, text: str) -> list[dict[str, object]]:
        return self.predict_rows(text_boundary_rows(text))
