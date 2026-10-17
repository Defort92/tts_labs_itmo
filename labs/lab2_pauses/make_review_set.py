"""Create short audio clips for manual validation of pause measurements."""

from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path

from align_words import iter_metadata


def choose_examples(rows: list[dict[str, str]], threshold_ms: float, per_group: int) -> list[tuple[str, dict[str, str]]]:
    groups: list[tuple[str, list[dict[str, str]]]] = [
        (
            "comma_with_pause",
            sorted(
                [row for row in rows if row["punctuation"] == "," and float(row["gap_ms"]) >= threshold_ms],
                key=lambda row: float(row["gap_ms"]),
                reverse=True,
            ),
        ),
        (
            "comma_without_pause",
            sorted(
                [row for row in rows if row["punctuation"] == "," and float(row["gap_ms"]) < threshold_ms],
                key=lambda row: float(row["gap_ms"]),
            ),
        ),
        (
            "no_punctuation_long_pause",
            sorted(
                [row for row in rows if row["punctuation"] == "NONE" and float(row["gap_ms"]) >= threshold_ms],
                key=lambda row: float(row["gap_ms"]),
                reverse=True,
            ),
        ),
        (
            "near_threshold",
            sorted(rows, key=lambda row: abs(float(row["gap_ms"]) - threshold_ms)),
        ),
    ]
    selected: list[tuple[str, dict[str, str]]] = []
    used: set[tuple[str, str]] = set()
    for category, candidates in groups:
        added = 0
        for row in candidates:
            key = (row["utterance_id"], row["boundary_index"])
            if key in used:
                continue
            selected.append((category, row))
            used.add(key)
            added += 1
            if added == per_group:
                break
    return selected


def copy_wav_fragment(source: Path, target: Path, start_s: float, end_s: float) -> None:
    with wave.open(str(source), "rb") as wav_in:
        params = wav_in.getparams()
        start_frame = max(0, round(start_s * params.framerate))
        end_frame = min(params.nframes, round(end_s * params.framerate))
        wav_in.setpos(start_frame)
        frames = wav_in.readframes(max(0, end_frame - start_frame))
    with wave.open(str(target), "wb") as wav_out:
        wav_out.setparams(params)
        wav_out.writeframes(frames)


def build_argument_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=project_root / "data" / "metadata_RUSLAN_22200_normalized.csv",
    )
    parser.add_argument(
        "--pauses",
        type=Path,
        default=project_root / "data" / "lab2_pauses" / "pause_boundaries.csv",
    )
    parser.add_argument(
        "--alignments",
        type=Path,
        default=project_root / "data" / "lab2_pauses" / "word_alignments.csv",
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=project_root / "data" / "RUSLAN" / "RUSLAN",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "lab2_pauses" / "review_clips",
    )
    parser.add_argument("--threshold-ms", type=float, default=80.0)
    parser.add_argument("--per-group", type=int, default=4)
    parser.add_argument("--context-s", type=float, default=0.75)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    texts = dict(iter_metadata(args.metadata))
    with args.pauses.open("r", encoding="utf-8", newline="") as source:
        pause_rows = list(csv.DictReader(source))
    alignment_by_boundary: dict[tuple[str, str], float] = {}
    with args.alignments.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["boundary_after_s"]:
                alignment_by_boundary[(row["utterance_id"], row["word_index"])] = float(
                    row["boundary_after_s"]
                )

    examples = choose_examples(pause_rows, args.threshold_ms, args.per_group)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The set is regenerated from the current full-corpus labels. Remove only
    # clips previously created by this script so stale pilot files cannot be
    # mistaken for items from the new checklist.
    for old_clip in args.output_dir.glob("*.wav"):
        old_clip.unlink()
    checklist_rows: list[dict[str, object]] = []
    for review_index, (category, row) in enumerate(examples, start=1):
        key = (row["utterance_id"], row["boundary_index"])
        boundary_s = alignment_by_boundary[key]
        clip_start_s = max(0.0, boundary_s - args.context_s)
        clip_end_s = boundary_s + args.context_s
        filename = f"{review_index:02d}_{row['utterance_id']}_b{int(row['boundary_index']):02d}.wav"
        copy_wav_fragment(
            args.wav_dir / f"{row['utterance_id']}.wav",
            args.output_dir / filename,
            clip_start_s,
            clip_end_s,
        )
        checklist_rows.append(
            {
                "review_index": review_index,
                "category": category,
                "clip": filename,
                "utterance_id": row["utterance_id"],
                "boundary": f"{row['left_word']} | {row['right_word']}",
                "punctuation": row["punctuation"],
                "measured_pause_ms": row["gap_ms"],
                "phrase": texts[row["utterance_id"]],
                "human_has_pause": "",
                "human_comment": "",
            }
        )

    checklist_path = args.output_dir / "review_checklist.csv"
    with checklist_path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(checklist_rows[0]))
        writer.writeheader()
        writer.writerows(checklist_rows)
    print(f"Review clips: {len(checklist_rows)}")
    print(f"Checklist: {checklist_path}")


if __name__ == "__main__":
    main()
