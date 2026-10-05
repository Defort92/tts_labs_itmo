"""Audit normalization changes and filter decisions on the RUSLAN corpus."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import csv
from pathlib import Path
import random
import re
import sys
import unicodedata

from text_filter import TextFilter
from text_normalizer import TextNormalizer
from preprocess_ruslan import (
    apply_reviewed_corpus_fixes,
    is_reviewed_corpus_safe,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPOSITORY_ROOT / "data" / "RUSLAN" / "metadata_RUSLAN_22200.csv"
OUTPUT_PATH = REPOSITORY_ROOT / "data" / "metadata_RUSLAN_22200_normalized.csv"


def describe_changes(raw: str, normalized: str) -> list[str]:
    """Return non-exclusive descriptions of transformations made to one row."""
    if raw == normalized:
        return []
    reasons: list[str] = []
    if raw != raw.strip():
        reasons.append("outer_whitespace")
    if re.search(r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]{2,}", raw):
        reasons.append("repeated_or_special_whitespace")
    if re.search(r"\s+[,.!?;:…]", raw) or re.search(r"[,;:](?=[А-Яа-яЁё])", raw):
        reasons.append("punctuation_spacing")
    if any(mark in raw for mark in "‑‐−–―"):
        reasons.append("dash_or_hyphen")
    if any(mark in raw for mark in "“”„‟"):
        reasons.append("quotation_marks")
    if re.search(r"[\u200b-\u200f\u2060\ufeff]", raw):
        reasons.append("invisible_characters")
    if re.search(r"\?{2,}|!{2,}|!\.+(?=\s|$)|(?<!\.)\.\.(?!\.)", raw):
        reasons.append("repeated_punctuation")
    if raw != unicodedata.normalize("NFC", raw):
        reasons.append("unicode_nfc")
    if not reasons:
        reasons.append("combined_typography")
    return reasons


def choose_examples(records: list[dict[str, str]], limit: int = 5) -> list[dict[str, str]]:
    """Choose deterministic examples spread through a category."""
    if len(records) <= limit:
        return records
    generator = random.Random(2026 + len(records))
    return [records[index] for index in sorted(generator.sample(range(len(records)), limit))]


def main(all_rejections: bool = False) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    normalizer = TextNormalizer()
    text_filter = TextFilter()
    kept_rows: dict[str, str] = {}
    with OUTPUT_PATH.open(encoding="utf-8", newline="") as output_file:
        for row in csv.reader(output_file, delimiter="|", quoting=csv.QUOTE_NONE):
            kept_rows[row[0]] = row[2]

    change_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    change_examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    rejection_examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    reviewed_examples: list[dict[str, str]] = []
    total = changed = rejected = 0

    with INPUT_PATH.open(encoding="utf-8-sig", newline="") as input_file:
        for utterance_id, raw in csv.reader(
            input_file, delimiter="|", quoting=csv.QUOTE_NONE
        ):
            total += 1
            normalized = normalizer.normalize(raw)
            normalized = apply_reviewed_corpus_fixes(
                utterance_id, normalized, normalizer
            )
            record = {"id": utterance_id, "raw": raw, "normalized": normalized}
            changes = describe_changes(raw, normalized)
            if changes:
                changed += 1
                for reason in changes:
                    change_counts[reason] += 1
                    change_examples[reason].append(record)

            reasons = text_filter.rejection_reasons(normalized)
            reviewed_safe = is_reviewed_corpus_safe(utterance_id, normalized)
            rejected_by_rules = bool(reasons) and not reviewed_safe
            rejected_by_output = utterance_id not in kept_rows
            if rejected_by_rules != rejected_by_output:
                raise AssertionError(f"Decision mismatch for {utterance_id}")
            if not rejected_by_output and kept_rows[utterance_id] != normalized:
                raise AssertionError(f"Normalized text mismatch for {utterance_id}")
            if reviewed_safe:
                reviewed_examples.append(record | {"reasons": ",".join(reasons) or "silent_markup"})
            if rejected_by_rules:
                rejected += 1
                for reason in reasons:
                    rejection_counts[reason] += 1
                    rejection_examples[reason].append(record)

    print(
        f"SUMMARY total={total} changed={changed} rejected={rejected} "
        f"kept={total-rejected} reviewed_overrides={len(reviewed_examples)}"
    )
    print("\nCHANGE_COUNTS")
    for reason, count in change_counts.most_common():
        print(f"{reason}|{count}|{count / total:.4%}")
    print("\nREJECTION_COUNTS")
    for reason, count in rejection_counts.most_common():
        print(f"{reason}|{count}|{count / total:.4%}")

    print("\nREVIEWED_OVERRIDES")
    for record in reviewed_examples:
        print(f"{record['id']}|{record['reasons']}|{record['normalized']}")

    print("\nCHANGE_EXAMPLES")
    for reason, _ in change_counts.most_common():
        print(f"\n[{reason}]")
        for record in choose_examples(change_examples[reason]):
            print(f"{record['id']}|{record['raw']}|{record['normalized']}")

    print("\nREJECTION_EXAMPLES")
    for reason, _ in rejection_counts.most_common():
        print(f"\n[{reason}]")
        records = rejection_examples[reason] if all_rejections else choose_examples(
            rejection_examples[reason]
        )
        for record in records:
            print(f"{record['id']}|{record['normalized']}")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--all-rejections", action="store_true")
    arguments = argument_parser.parse_args()
    main(all_rejections=arguments.all_rejections)
