"""Audit coverage and structural consistency of lab-3 phoneme artifacts."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from g2p_utils import iter_metadata


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "lab3_g2p"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", type=Path, default=project_root / "data" / "metadata_RUSLAN_22200_normalized.csv"
    )
    parser.add_argument("--alignments", type=Path, default=data_dir / "phoneme_alignments.csv")
    parser.add_argument("--utterances", type=Path, default=data_dir / "utterance_phonemes.csv")
    parser.add_argument("--textgrid-dir", type=Path, default=data_dir / "textgrids")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    expected = {utterance_id for utterance_id, _ in iter_metadata(args.metadata)}
    with args.utterances.open("r", encoding="utf-8", newline="") as source:
        utterance_rows = list(csv.DictReader(source))
    utterance_counts = Counter(row["utterance_id"] for row in utterance_rows)
    duplicates = [key for key, count in utterance_counts.items() if count != 1]

    rows_by_utterance: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.alignments.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            rows_by_utterance[row["utterance_id"]].append(row)
    bad_timings: list[str] = []
    empty_speech: list[str] = []
    for utterance_id, rows in rows_by_utterance.items():
        rows.sort(key=lambda row: int(row["phone_index"]))
        starts = [float(row["start_s"]) for row in rows]
        ends = [float(row["end_s"]) for row in rows]
        if any(start < -0.001 or end <= start for start, end in zip(starts, ends, strict=True)):
            bad_timings.append(utterance_id)
        elif any(left_end > right_start + 0.001 for left_end, right_start in zip(ends[:-1], starts[1:], strict=True)):
            bad_timings.append(utterance_id)
        if not any(row["is_silence"] == "0" for row in rows):
            empty_speech.append(utterance_id)

    actual = set(utterance_counts)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    textgrid_ids = {path.stem for path in args.textgrid_dir.rglob("*.TextGrid")}
    missing_textgrids = sorted(actual - textgrid_ids)
    print(f"Expected utterances: {len(expected)}")
    print(f"Parsed utterances: {len(actual)}")
    print(f"Phone intervals: {sum(len(rows) for rows in rows_by_utterance.values())}")
    print(f"Missing utterances: {len(missing)}")
    print(f"Unexpected utterances: {len(unexpected)}")
    print(f"Duplicate utterance rows: {len(duplicates)}")
    print(f"Bad timing groups: {len(bad_timings)}")
    print(f"Empty speech groups: {len(empty_speech)}")
    print(f"Missing TextGrids for parsed rows: {len(missing_textgrids)}")

    structural_failure = duplicates or unexpected or bad_timings or empty_speech or missing_textgrids
    if structural_failure:
        raise SystemExit("structural audit failed")
    if args.require_complete and missing:
        raise SystemExit("full-corpus audit failed")


if __name__ == "__main__":
    main()

