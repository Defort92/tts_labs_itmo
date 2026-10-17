"""Audit coverage and structural consistency of lab-2 generated artifacts."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from align_words import iter_metadata


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=project_root / "data" / "metadata_RUSLAN_22200_normalized.csv",
    )
    parser.add_argument(
        "--alignments",
        type=Path,
        default=project_root / "data" / "lab2_pauses" / "word_alignments.csv",
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=project_root / "data" / "lab2_pauses" / "alignment_status.csv",
    )
    parser.add_argument(
        "--pauses",
        type=Path,
        default=project_root / "data" / "lab2_pauses" / "pause_boundaries.csv",
    )
    parser.add_argument(
        "--textgrid-dir",
        type=Path,
        default=project_root / "data" / "lab2_pauses" / "textgrids",
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    expected_ids = [utterance_id for utterance_id, _ in iter_metadata(args.metadata)]
    with args.status.open("r", encoding="utf-8", newline="") as source:
        statuses = list(csv.DictReader(source))
    status_counts = Counter(row["utterance_id"] for row in statuses)
    duplicate_status_ids = sorted(key for key, count in status_counts.items() if count != 1)
    ok_statuses = {row["utterance_id"]: row for row in statuses if row["status"] == "ok"}
    ok_counts = {
        utterance_id: int(row["word_count"])
        for utterance_id, row in ok_statuses.items()
    }

    word_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.alignments.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            word_rows[row["utterance_id"]].append(row)
    bad_word_groups: list[str] = []
    bad_word_timings: list[str] = []
    low_confidence = 0
    for utterance_id, expected_count in ok_counts.items():
        rows = word_rows.get(utterance_id, [])
        rows.sort(key=lambda row: int(row["word_index"]))
        if len(rows) != expected_count or [int(row["word_index"]) for row in rows] != list(range(expected_count)):
            bad_word_groups.append(utterance_id)
        else:
            duration_s = float(ok_statuses[utterance_id]["duration_s"])
            starts = [float(row["start_s"]) for row in rows]
            ends = [float(row["end_s"]) for row in rows]
            valid_intervals = all(
                -0.001 <= start <= end <= duration_s + 0.05
                for start, end in zip(starts, ends, strict=True)
            )
            monotonic = all(
                left_end <= right_start + 0.001
                for left_end, right_start in zip(ends[:-1], starts[1:], strict=True)
            )
            if not valid_intervals or not monotonic:
                bad_word_timings.append(utterance_id)
        low_confidence += sum(float(row["confidence"]) < 0.5 for row in rows)

    missing_textgrids = sorted(
        utterance_id
        for utterance_id in ok_counts
        if not (args.textgrid_dir / f"{utterance_id}.TextGrid").is_file()
    )

    pause_count = None
    duplicate_pause_keys: list[tuple[str, int]] = []
    expected_pause_count = sum(max(0, word_count - 1) for word_count in ok_counts.values())
    if args.pauses.is_file():
        with args.pauses.open("r", encoding="utf-8", newline="") as source:
            pause_rows = list(csv.DictReader(source))
        pause_count = len(pause_rows)
        pause_keys = Counter(
            (row["utterance_id"], int(row["boundary_index"])) for row in pause_rows
        )
        duplicate_pause_keys = sorted(key for key, count in pause_keys.items() if count != 1)

    missing = sorted(set(expected_ids) - set(status_counts))
    unexpected = sorted(set(status_counts) - set(expected_ids))
    errors = [row for row in statuses if row["status"] != "ok"]
    print(f"Expected utterances: {len(expected_ids)}")
    print(f"Recorded statuses: {len(statuses)}")
    print(f"Aligned: {len(ok_counts)}")
    print(f"Errors: {len(errors)}")
    print(f"Missing statuses: {len(missing)}")
    print(f"Unexpected statuses: {len(unexpected)}")
    print(f"Duplicate status IDs: {len(duplicate_status_ids)}")
    print(f"Bad word groups: {len(bad_word_groups)}")
    print(f"Bad word timings: {len(bad_word_timings)}")
    print(f"Missing TextGrids: {len(missing_textgrids)}")
    print(f"Words with confidence < 0.5: {low_confidence}")
    print(f"Pause boundaries: {pause_count if pause_count is not None else 'not generated'}")
    print(f"Expected pause boundaries: {expected_pause_count}")
    print(f"Duplicate pause keys: {len(duplicate_pause_keys)}")

    pause_structure_failed = (
        pause_count is not None
        and (pause_count != expected_pause_count or bool(duplicate_pause_keys))
    )
    if (
        duplicate_status_ids
        or unexpected
        or bad_word_groups
        or bad_word_timings
        or missing_textgrids
        or pause_structure_failed
    ):
        raise SystemExit("structural audit failed")
    if args.require_complete and (missing or errors):
        raise SystemExit("full-corpus audit failed")


if __name__ == "__main__":
    main()
