"""Shared text and pronunciation helpers for laboratory work 3."""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


WORD_PART_RE = r"[А-Яа-яЁё][А-Яа-яЁё\u0301]*"
WORD_RE = re.compile(rf"{WORD_PART_RE}(?:[-‐‑–—]{WORD_PART_RE})*")
STRESS_MARKS = {"\u0300", "\u0301"}


def extract_words(text: str) -> list[str]:
    """Extract the Cyrillic word tokens used by the lab-2 aligner."""

    return [match.group(0) for match in WORD_RE.finditer(unicodedata.normalize("NFC", text))]


def normalize_word(word: str) -> str:
    """Convert a corpus word to the orthography expected by the MFA lexicon."""

    decomposed = unicodedata.normalize("NFD", word.casefold())
    without_stress = "".join(char for char in decomposed if char not in STRESS_MARKS)
    return unicodedata.normalize("NFC", without_stress).translate(
        str.maketrans({"‐": "-", "‑": "-", "–": "-", "—": "-"})
    )


def iter_metadata(path: Path) -> Iterable[tuple[str, str]]:
    """Yield ``(utterance_id, normalized_text)`` from lab-1 metadata."""

    with path.open("r", encoding="utf-8", newline="") as source:
        for line_number, row in enumerate(csv.reader(source, delimiter="|"), start=1):
            if len(row) != 3:
                raise ValueError(f"{path}:{line_number}: expected 3 fields, got {len(row)}")
            utterance_id, _, normalized_text = row
            yield utterance_id, unicodedata.normalize("NFC", normalized_text)


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def parse_dictionary_line(line: str) -> tuple[str, tuple[str, ...]] | None:
    """Parse MFA dictionaries with or without pronunciation-probability columns."""

    fields = line.strip().split()
    if len(fields) < 2 or fields[0].startswith("#"):
        return None
    phone_start = 1
    while phone_start < len(fields) and _is_number(fields[phone_start]):
        phone_start += 1
    if phone_start == len(fields):
        return None
    return normalize_word(fields[0]), tuple(fields[phone_start:])


def read_dictionary(path: Path) -> dict[str, list[tuple[str, ...]]]:
    """Load all pronunciation variants from an MFA dictionary."""

    pronunciations: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            parsed = parse_dictionary_line(line)
            if parsed is None:
                continue
            word, phones = parsed
            if phones not in pronunciations[word]:
                pronunciations[word].append(phones)
    return dict(pronunciations)


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Compute Levenshtein distance for two phone sequences."""

    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_phone in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_phone in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_phone != hyp_phone),
                )
            )
        previous = current
    return previous[-1]


def best_reference_distance(
    references: Sequence[Sequence[str]], hypothesis: Sequence[str]
) -> tuple[int, Sequence[str]]:
    """Return distance to the closest acceptable pronunciation variant."""

    if not references:
        raise ValueError("at least one reference pronunciation is required")
    return min((edit_distance(reference, hypothesis), reference) for reference in references)

