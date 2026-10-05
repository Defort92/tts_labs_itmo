"""Count potentially non-normalized phenomena in RUSLAN metadata."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import re
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPOSITORY_ROOT / "data" / "RUSLAN" / "metadata_RUSLAN_22200.csv"

PATTERNS = {
    "digits": re.compile(r"\d"),
    "latin_letters": re.compile(r"[A-Za-z\u00c0-\u024f]"),
    "technical_symbols": re.compile(r"[@&*+=/\\<>\[\]{}_\|~^#]"),
    "named_symbols": re.compile(r"[%№€£¥₽$°]"),
    "likely_abbreviations": re.compile(
        r"(?<![А-Яа-яЁё])(?:[А-ЯЁ]\.|ул\.|г\.|г-н|г-ж[аи]|т\.\s*д\.|т\.\s*п\.)",
        re.IGNORECASE,
    ),
    "repeated_punctuation": re.compile(r"!{2,}|\?{2,}|(?<!\.)\.\.(?!\.)"),
    "unusual_quotes": re.compile(r"[“”„‟]"),
    "invisible_characters": re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]"),
    "emoticons": re.compile(r"(?:[:;=8xX][-^']?[()DPpРр]|[()]{2,}|<3)"),
}


def analyze(path: Path, example_limit: int = 3) -> tuple[int, dict[str, int], dict[str, list[str]]]:
    counts = defaultdict(int)
    examples: dict[str, list[str]] = defaultdict(list)
    total = 0
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for line_number, row in enumerate(csv.reader(source, delimiter="|"), start=1):
            if len(row) != 2:
                raise ValueError(f"Expected 2 columns at line {line_number}, got {len(row)}")
            total += 1
            text = row[1]
            for name, pattern in PATTERNS.items():
                if pattern.search(text):
                    counts[name] += 1
                    if len(examples[name]) < example_limit:
                        examples[name].append(text)
    return total, dict(counts), dict(examples)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"metadata file not found: {args.input}")

    total, counts, examples = analyze(args.input, args.examples)
    print(f"Total utterances: {total}")
    for name in PATTERNS:
        count = counts.get(name, 0)
        share = count / total * 100 if total else 0
        print(f"\n{name}: {count} ({share:.2f}%)")
        for example in examples.get(name, []):
            print(f"  - {example}")
