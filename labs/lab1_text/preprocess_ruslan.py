"""Build filtered three-column RUSLAN metadata for laboratory work 1."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import unicodedata

from text_filter import TextFilter
from text_normalizer import TextNormalizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPOSITORY_ROOT / "data" / "RUSLAN" / "metadata_RUSLAN_22200.csv"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "metadata_RUSLAN_22200_normalized.csv"

# These corpus rows were listened to manually. The general classifier remains
# conservative for unseen input, while preprocessing may retain confirmed aligned
# utterances or remove confirmed silent markup.
REVIEWED_INPUT_TEXT = {
    "000880_RUSLAN": "Я на букву О /библиография к Окуджаве/.",
    "001024_RUSLAN": "А Лев Уфлянд* еще больше подливает желчи, плюет на русский народ.",
    "001028_RUSLAN": "Уфлянда зовут Владимир прим. автора.",
    "006729_RUSLAN": (
        "…Контора размещалась тогда на улице Пикк. Строго напротив здания "
        "Госбезопасности (ул. Пагари, один)."
    ),
}
REVIEWED_SAFE_TEXT = {
    "000880_RUSLAN": "Я на букву О библиография к Окуджаве.",
    "001024_RUSLAN": "А Лев Уфлянд еще больше подливает желчи, плюет на русский народ.",
    "001028_RUSLAN": REVIEWED_INPUT_TEXT["001028_RUSLAN"],
    "006729_RUSLAN": REVIEWED_INPUT_TEXT["006729_RUSLAN"],
}
REVIEWED_SAFE_IDS = frozenset(REVIEWED_SAFE_TEXT)


def apply_reviewed_corpus_fixes(
    utterance_id: str, text: str, normalizer: TextNormalizer
) -> str:
    """Apply only transformations confirmed by listening to the named recording."""
    if text != REVIEWED_INPUT_TEXT.get(utterance_id):
        return text
    if utterance_id == "000880_RUSLAN":
        return normalizer.normalize(text.replace("/", ""))
    if utterance_id == "001024_RUSLAN":
        return normalizer.normalize(text.replace("*", ""))
    return text


def is_reviewed_corpus_safe(utterance_id: str, normalized: str) -> bool:
    """Return true only for the exact ID/text pair confirmed by listening."""
    return REVIEWED_SAFE_TEXT.get(utterance_id) == normalized


def preprocess(input_path: Path, output_path: Path) -> dict[str, int]:
    """Normalize and filter metadata, returning processing statistics."""
    normalizer = TextNormalizer()
    text_filter = TextFilter()
    stats = {
        "total": 0,
        "kept": 0,
        "changed": 0,
        "rejected": 0,
        "reviewed_overrides": 0,
    }

    if not input_path.is_file():
        raise FileNotFoundError(
            f"RUSLAN metadata not found: {input_path}. "
            "Download and unpack the corpus into data/RUSLAN first."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as source, \
            output_path.open("w", encoding="utf-8", newline="") as destination:
        reader = csv.reader(source, delimiter="|", quoting=csv.QUOTE_NONE)
        writer = csv.writer(
            destination,
            delimiter="|",
            quoting=csv.QUOTE_NONE,
            quotechar=None,
            lineterminator="\n",
            escapechar="\\",
        )
        for line_number, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValueError(
                    f"Expected 2 columns in {input_path} at line {line_number}, "
                    f"got {len(row)}"
                )
            utterance_id, raw_text = row
            stats["total"] += 1
            normalized = normalizer.normalize(raw_text)
            normalized = apply_reviewed_corpus_fixes(
                utterance_id, normalized, normalizer
            )
            if normalized != raw_text:
                stats["changed"] += 1
            is_reviewed_safe = is_reviewed_corpus_safe(utterance_id, normalized)
            if is_reviewed_safe:
                stats["reviewed_overrides"] += 1
            if text_filter.filter(normalized) != 1 and not is_reviewed_safe:
                stats["rejected"] += 1
                continue
            if normalized != unicodedata.normalize("NFC", normalized):
                raise AssertionError(f"Normalizer produced non-NFC text at line {line_number}")
            writer.writerow((utterance_id, raw_text, normalized))
            stats["kept"] += 1

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = preprocess(args.input, args.output)
    print(
        "Rows: {total}; kept: {kept}; changed: {changed}; rejected: {rejected}; "
        "reviewed overrides: {reviewed_overrides}".format(**result)
    )
