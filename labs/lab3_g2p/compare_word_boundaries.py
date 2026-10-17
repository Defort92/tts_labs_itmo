"""Compare MFA word intervals with the independent CTC alignment from lab 2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from g2p_utils import normalize_word


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lab2", type=Path, default=project_root / "data" / "lab2_pauses" / "word_alignments.csv"
    )
    parser.add_argument(
        "--lab3", type=Path, default=project_root / "data" / "lab3_g2p" / "word_pronunciations.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=project_root / "data" / "lab3_g2p" / "word_boundary_comparison.csv"
    )
    args = parser.parse_args()

    lab2: dict[tuple[str, int], dict[str, str]] = {}
    with args.lab2.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            lab2[(row["utterance_id"], int(row["word_index"]))] = row

    start_differences: list[float] = []
    end_differences: list[float] = []
    mismatched_words = 0
    missing_lab2 = 0
    rows_written = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.lab3.open("r", encoding="utf-8", newline="") as source,
        args.output.open("w", encoding="utf-8", newline="") as target,
    ):
        reader = csv.DictReader(source)
        writer = csv.writer(target)
        writer.writerow(
            [
                "utterance_id", "word_index", "word", "same_word",
                "lab2_start_s", "mfa_start_s", "start_abs_error_ms",
                "lab2_end_s", "mfa_end_s", "end_abs_error_ms",
            ]
        )
        for row in reader:
            key = (row["utterance_id"], int(row["word_index"]))
            reference = lab2.get(key)
            if reference is None:
                missing_lab2 += 1
                continue
            same_word = normalize_word(reference["word"]) == normalize_word(row["word"])
            mismatched_words += not same_word
            start_error = abs(float(reference["start_s"]) - float(row["start_s"])) * 1000
            end_error = abs(float(reference["end_s"]) - float(row["end_s"])) * 1000
            start_differences.append(start_error)
            end_differences.append(end_error)
            writer.writerow(
                [
                    key[0], key[1], row["word"], int(same_word),
                    reference["start_s"], row["start_s"], f"{start_error:.3f}",
                    reference["end_s"], row["end_s"], f"{end_error:.3f}",
                ]
            )
            rows_written += 1

    metrics = {
        "compared_words": rows_written,
        "missing_lab2_rows": missing_lab2,
        "word_label_mismatches": mismatched_words,
        "start_absolute_error": summarize(start_differences),
        "end_absolute_error": summarize(end_differences),
        "note": (
            "This is agreement between two automatic aligners, not error against manual gold. "
            "Lab 2 CTC intervals tend to be narrower than MFA word intervals."
        ),
    }
    (args.output.parent / "word_boundary_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

